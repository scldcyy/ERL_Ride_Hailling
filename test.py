import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from datetime import datetime

# ==========================================
# 1. 全局配置 (Configuration)
# ==========================================
CONFIG = {
    # 仿真参数
    'N_DRIVERS': 50,  # 司机数量 (为了演示速度调小，论文可用 500+)
    'N_ZONES': 20,  # 区域数量
    'TIME_STEPS_PER_DAY': 48,  # 一天的时间步 (例如每30分钟一步，共48步)

    # 经济学参数 (论文核心修改点)
    'BASE_FARE': 3.0,
    'PRICE_PER_MINUTE': 0.5,
    'OPPORTUNITY_COST_PER_STEP': 0.5,  # 司机的油耗/时间成本
    'DRIVER_ACCEPT_THRESHOLD': 1.5,  # [修复缺陷2] 司机拒单阈值：预期净收入低于此值则拒单

    # EA & RL 参数
    'POP_SIZE': 8,  # 种群大小
    'N_GEN': 10,  # 进化代数
    'HIDDEN_DIM': 64,
    'LR_ACTOR': 0.001,
}

# 确保图片保存目录存在
os.makedirs('paper_results', exist_ok=True)
sns.set_theme(style="whitegrid")


# ==========================================
# 2. 模拟器与环境 (Environment with Refusal Logic)
# ==========================================

class MockSimulator:
    """[修复缺陷1] 当缺少真实数据时，生成模拟的订单流和拓扑结构"""

    def __init__(self, n_zones):
        self.n_zones = n_zones
        self.adjacency = {i: [max(0, i - 1), min(n_zones - 1, i + 1), i] for i in range(n_zones)}

    def generate_orders(self, time_step):
        """生成符合正态分布的高峰期订单"""
        orders = []
        # 模拟早晚高峰 (Time step 10-15 and 35-40)
        base_demand = 5
        if (10 <= time_step <= 15) or (35 <= time_step <= 40):
            base_demand = 15

        for _ in range(np.random.poisson(base_demand)):
            origin = np.random.randint(0, self.n_zones)
            # 目的地倾向于在附近
            dest = np.clip(origin + np.random.choice([-2, -1, 1, 2]), 0, self.n_zones - 1)
            dist = abs(origin - dest) + 1
            duration = dist  # 简单假设
            orders.append({
                'origin_idx': origin,
                'dest_idx': dest,
                'duration': duration,
                'distance': dist,
                'matched': False,
                'wait_time': 0
            })
        return orders


class RideHailingEnv:
    def __init__(self):
        self.n_zones = CONFIG['N_ZONES']
        self.simulator = MockSimulator(self.n_zones)
        self.action_dim = self.n_zones + 1  # 0: Serve/Stay, 1..N: Reposition

    def reset(self):
        self.time = 0
        self.driver_locs = np.random.randint(0, self.n_zones, size=CONFIG['N_DRIVERS'])
        self.driver_status = np.zeros(CONFIG['N_DRIVERS'], dtype=int)  # 0: Idle, 1: Busy
        self.driver_free_time = np.zeros(CONFIG['N_DRIVERS'], dtype=int)
        self.pending_orders = []
        self.metrics = {'revenue': 0, 'served': 0, 'rejected': 0, 'wait_time': 0,
                        'driver_incomes': np.zeros(CONFIG['N_DRIVERS'])}
        return self._get_state()

    def _get_state(self):
        # 简单状态定义: [Loc, Time, Local_Demand, Local_Supply]
        states = np.zeros((CONFIG['N_DRIVERS'], 4))

        # 统计各区域供需
        demand_map = np.zeros(self.n_zones)
        for o in self.pending_orders:
            if not o['matched']: demand_map[o['origin_idx']] += 1

        supply_map = np.zeros(self.n_zones)
        idle_drivers = self.driver_locs[self.driver_status == 0]
        for loc in idle_drivers: supply_map[loc] += 1

        for i in range(CONFIG['N_DRIVERS']):
            loc = self.driver_locs[i]
            states[i] = [loc / self.n_zones, self.time / CONFIG['TIME_STEPS_PER_DAY'],
                         demand_map[loc] / 10.0, supply_map[loc] / 10.0]
        return states

    def get_valid_mask(self):
        # 简化版 Mask: 总是允许接单(0)或去邻居节点
        mask = np.zeros((CONFIG['N_DRIVERS'], self.action_dim), dtype=bool)
        mask[:, 0] = True
        for i in range(CONFIG['N_DRIVERS']):
            loc = self.driver_locs[i]
            neighbors = self.simulator.adjacency[loc]
            for n in neighbors: mask[i, n + 1] = True
        return mask

    def step(self, actions, platform_params):
        # 1. 更新司机状态 (Busy -> Idle)
        self.driver_free_time[self.driver_free_time > 0] -= 1
        freed = (self.driver_status == 1) & (self.driver_free_time == 0)
        self.driver_status[freed] = 0

        # 2. 生成新订单
        new_orders = self.simulator.generate_orders(self.time)
        self.pending_orders.extend(new_orders)

        # 移除超时订单 (>3 steps)
        self.pending_orders = [o for o in self.pending_orders if not o['matched'] and o['wait_time'] < 3]
        for o in self.pending_orders:
            o['wait_time'] += 1
            self.metrics['wait_time'] += 1

        rewards = np.zeros(CONFIG['N_DRIVERS'])

        # 获取当前时刻的定价参数矩阵 (Time x Zone)
        # platform_params['lambda'] 是完整矩阵
        current_surge = platform_params['lambda'][self.time]
        current_subsidy = platform_params['subsidy'][self.time]
        commission = platform_params['commission']

        # 3. 处理司机动作
        idle_indices = np.where(self.driver_status == 0)[0]
        np.random.shuffle(idle_indices)  # 随机顺序防止死锁

        for i in idle_indices:
            act = actions[i]
            loc = self.driver_locs[i]

            if act == 0:  # 意图：接单 (Serve)
                # 寻找本地订单
                local_orders = [o for o in self.pending_orders if o['origin_idx'] == loc and not o['matched']]

                if local_orders:
                    order = local_orders[0]
                    duration = max(1, order['duration'])

                    # --- [修复缺陷2] 核心逻辑：收益计算与拒单 ---
                    fare_base = CONFIG['BASE_FARE'] + duration * CONFIG['PRICE_PER_MINUTE']
                    total_fare = fare_base * current_surge[loc]  # 动态定价

                    # 司机预期净收入 = (车费 * (1-抽成)) + 补贴 - (时间成本 * 时长)
                    expected_net_income = (total_fare * (1 - commission)) + current_subsidy[loc] - (
                                duration * CONFIG['OPPORTUNITY_COST_PER_STEP'])

                    if expected_net_income >= CONFIG['DRIVER_ACCEPT_THRESHOLD']:
                        # 接单成功
                        order['matched'] = True
                        self.driver_status[i] = 1
                        self.driver_free_time[i] = duration
                        self.driver_locs[i] = order['dest_idx']

                        rewards[i] = expected_net_income
                        # 记录平台收入
                        self.metrics['revenue'] += (total_fare * commission) - current_subsidy[loc]
                        self.metrics['served'] += 1
                        self.metrics['driver_incomes'][i] += expected_net_income
                    else:
                        # 拒单：虽有单但钱太少，司机选择空闲
                        self.metrics['rejected'] += 1
                        rewards[i] = -0.1  # 小惩罚
                else:
                    # 无单可接
                    rewards[i] = -0.1
            else:
                # 调度 (Reposition)
                target = act - 1
                rewards[i] = -CONFIG['OPPORTUNITY_COST_PER_STEP']  # 移动成本
                self.driver_locs[i] = target

        self.time += 1
        done = (self.time >= CONFIG['TIME_STEPS_PER_DAY'])
        return self._get_state(), rewards, done, self.metrics


# ==========================================
# 3. 强化学习 Agent (Simplified PPO)
# ==========================================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, CONFIG['HIDDEN_DIM']), nn.Tanh(),
            nn.Linear(CONFIG['HIDDEN_DIM'], action_dim)
        )

    def forward(self, state, mask):
        logits = self.net(state)
        logits = logits.masked_fill(~mask, -1e9)
        return Categorical(logits=logits)


class PPOAgent:
    def __init__(self, env):
        self.actor = Actor(4, env.action_dim)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=CONFIG['LR_ACTOR'])

    def select_action(self, state, mask):
        state_t = torch.FloatTensor(state)
        mask_t = torch.BoolTensor(mask)
        with torch.no_grad():
            dist = self.actor(state_t, mask_t)
            action = dist.sample()
        return action.numpy()

    def update(self, rollouts):
        # 简化版更新，仅为了让代码可运行，实际需完整 PPO Loss
        pass

    # ==========================================


# 4. 进化算法与策略编码 (Strategy Encoder)
# ==========================================

class StrategyEncoder:
    """[修复缺陷3] 将低维基因映射为时空动态定价矩阵"""

    def __init__(self):
        # 基因定义: [Commission, Surge_Base, Surge_Peak, Peak_Center, Peak_Width, Sub_Base, Sub_Peak]
        self.bounds = np.array([
            [0.1, 0.3],  # Commission
            [1.0, 1.2],  # Surge Base
            [1.2, 2.5],  # Surge Peak
            [0.2, 0.8],  # Peak Time Center (0-1)
            [0.05, 0.2],  # Peak Width
            [0.0, 0.5],  # Subsidy Base
            [0.5, 3.0]  # Subsidy Peak
        ])
        self.dim = len(self.bounds)

    def decode(self, gene):
        comm, s_base, s_peak, p_center, p_width, sub_base, sub_peak = gene

        # 生成时间维度的高斯曲线
        t = np.linspace(0, 1, CONFIG['TIME_STEPS_PER_DAY'])
        # Gaussian Kernel
        temporal_profile = np.exp(-0.5 * ((t - p_center) / p_width) ** 2)

        # 映射到 Surge 矩阵 (Time x Zone) - 假设中心区域高峰更明显
        surge_t = s_base + (s_peak - s_base) * temporal_profile
        lambda_matrix = np.tile(surge_t[:, np.newaxis], (1, CONFIG['N_ZONES']))

        # 映射到 Subsidy 矩阵
        sub_t = sub_base + (sub_peak - sub_base) * temporal_profile
        subsidy_matrix = np.tile(sub_t[:, np.newaxis], (1, CONFIG['N_ZONES']))

        return {
            'commission': comm,
            'lambda': lambda_matrix,
            'subsidy': subsidy_matrix
        }


def calculate_gini(incomes):
    """计算基尼系数"""
    if np.sum(incomes) == 0: return 0.0
    incomes = np.sort(incomes)
    n = len(incomes)
    index = np.arange(1, n + 1)
    return ((2 * index - n - 1) * incomes).sum() / (n * np.sum(incomes))


def evaluate_genome(gene, env, agent):
    """评估单个策略基因"""
    encoder = StrategyEncoder()
    params = encoder.decode(gene)

    state = env.reset()
    done = False

    while not done:
        mask = env.get_valid_mask()
        actions = agent.select_action(state, mask)
        state, rewards, done, metrics = env.step(actions, params)

    # 提取目标函数值
    profit = metrics['revenue']
    wait_time = metrics['wait_time']
    gini = calculate_gini(metrics['driver_incomes'])

    return [profit, wait_time, gini]


# ==========================================
# 5. 实验主循环 (Experiments & Plotting)
# ==========================================

def run_experiments():
    print(">>> 初始化环境与智能体...")
    env = RideHailingEnv()
    agent = PPOAgent(env)
    encoder = StrategyEncoder()

    # 初始化随机种群
    population = np.random.rand(CONFIG['POP_SIZE'], encoder.dim)
    population = population * (encoder.bounds[:, 1] - encoder.bounds[:, 0]) + encoder.bounds[:, 0]

    history = {'gen': [], 'profit': [], 'wait': [], 'gini': []}

    print(f">>> 开始进化过程 (共 {CONFIG['N_GEN']} 代)...")

    for gen in range(CONFIG['N_GEN']):
        gen_profits = []
        gen_waits = []
        gen_ginis = []

        # 评估当前种群
        fitness_scores = []
        for i in range(CONFIG['POP_SIZE']):
            # 每个个体运行一次仿真
            objs = evaluate_genome(population[i], env, agent)
            fitness_scores.append(objs)
            gen_profits.append(objs[0])
            gen_waits.append(objs[1])
            gen_ginis.append(objs[2])

        # 记录历史
        avg_p = np.mean(gen_profits)
        avg_w = np.mean(gen_waits)
        avg_g = np.mean(gen_ginis)
        history['gen'].append(gen)
        history['profit'].append(gen_profits)
        history['wait'].append(gen_waits)
        history['gini'].append(gen_ginis)

        print(f"Generation {gen + 1}: Avg Profit={avg_p:.2f}, Avg Wait={avg_w:.2f}, Avg Gini={avg_g:.4f}")

        # 简单进化逻辑 (Selection & Mutation)
        # 这里的 Fitness 简化为：Profit - 2 * WaitTime (多目标加权)
        fitness_vals = np.array(gen_profits) - 2.0 * np.array(gen_waits)
        best_indices = np.argsort(fitness_vals)[-CONFIG['POP_SIZE'] // 2:]  # 选前 50%
        parents = population[best_indices]

        # 变异产生下一代
        offspring = []
        for p in parents:
            child = p + np.random.normal(0, 0.05, size=encoder.dim)
            child = np.clip(child, encoder.bounds[:, 0], encoder.bounds[:, 1])
            offspring.append(p)
            offspring.append(child)
        population = np.array(offspring)[:CONFIG['POP_SIZE']]

    print(">>> 实验结束，正在绘图...")
    plot_results(history)


def plot_results(history):
    # [修复缺陷4] 完善的绘图逻辑

    # 1. 收敛曲线 (Convergence)
    plt.figure(figsize=(10, 6))
    means = [np.mean(p) for p in history['profit']]
    maxs = [np.max(p) for p in history['profit']]
    plt.plot(history['gen'], means, label='Avg Profit', marker='o')
    plt.plot(history['gen'], maxs, label='Max Profit', linestyle='--')
    plt.xlabel('Generation')
    plt.ylabel('Platform Profit')
    plt.title('Optimization Convergence')
    plt.legend()
    plt.savefig('paper_results/1_convergence.png')
    plt.close()

    # 2. 帕累托前沿 (Pareto Front)
    # 取最后一代数据
    last_gen_profit = history['profit'][-1]
    last_gen_wait = history['wait'][-1]
    last_gen_gini = history['gini'][-1]

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(last_gen_wait, last_gen_profit, c=last_gen_gini, cmap='viridis', s=100)
    plt.colorbar(sc, label='Gini Coefficient (Fairness)')
    plt.xlabel('Avg Wait Time (Lower is Better)')
    plt.ylabel('Platform Profit (Higher is Better)')
    plt.title('Pareto Front: Profit vs Service Quality')
    plt.grid(True, alpha=0.3)
    plt.savefig('paper_results/2_pareto_front.png')
    plt.close()

    # 3. 公平性分布 (Gini Distribution)
    # 对比第一代和最后一代
    data = []
    for g in [0, len(history['gen']) - 1]:
        label = 'Initial' if g == 0 else 'Optimized'
        for val in history['gini'][g]:
            data.append({'Stage': label, 'Gini': val})

    df = pd.DataFrame(data)
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x='Stage', y='Gini', palette="muted")
    plt.title('Driver Income Fairness Improvement')
    plt.savefig('paper_results/3_fairness_gini.png')
    plt.close()

    print(f"图表已保存至: {os.path.abspath('paper_results')}")


if __name__ == '__main__':
    run_experiments()