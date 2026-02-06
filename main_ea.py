import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats.qmc import LatinHypercube
from tqdm import tqdm

from shared_ppo import RideHailingEnv, SharedPPOAgent, CONFIG, Trainer


# --- 扩展 Trainer 以支持 EA 统计需求 ---
class StatsTrainer(Trainer):
    def train_and_evaluate(self, platform_params, num_episodes=5):
        # 记录每轮的指标
        ep_profits = []
        ep_driver_incomes = []
        ep_service_quality = []

        for _ in range(num_episodes):
            state = self.env.reset()
            ep_reward = 0

            # 记录该Episode的累计补贴成本
            ep_implicit_cost = 0  # 隐性补贴（低保填坑）
            ep_explicit_cost = 0  # 显性补贴（策略发放）

            while True:
                mask = self.env.get_valid_actions_mask()
                actions = self.agent.select_actions(state, mask)
                next_state, rewards, done, info = self.env.step(actions, platform_params)

                # PPO Buffer 存储
                self.agent.buffer.rewards.append(rewards)
                self.agent.buffer.is_terminals.append(done)

                # 累加成本
                if 'step_implicit_subsidy' in info:
                    ep_implicit_cost += info['step_implicit_subsidy']
                if 'step_explicit_subsidy' in info:
                    ep_explicit_cost += info['step_explicit_subsidy']

                state = next_state
                ep_reward += np.sum(rewards)

                if done:
                    # 提取本回合的统计信息
                    # 1. 平台净利润 = (总流水 * 抽成) - 显性补贴 - 隐性补贴
                    commission = platform_params['commission']
                    gross_revenue = info['total_revenue']

                    profit = (gross_revenue * commission) - ep_explicit_cost - ep_implicit_cost

                    # 2. 司机平均收入
                    avg_income = ep_reward / CONFIG['N_DRIVERS']

                    # 3. 服务质量 (使用 -TotalWaitTime)
                    quality = -info['total_wait_time']

                    ep_profits.append(profit)
                    ep_driver_incomes.append(avg_income)
                    ep_service_quality.append(quality)
                    break

            # 每回合结束更新一次 Agent (Short-term learning)
            self.agent.update()

        return np.array([np.mean(ep_profits), np.mean(ep_driver_incomes), np.mean(ep_service_quality)])


# --- Strategy Encoder (Time-Varying) ---
class StrategyEncoder:
    def __init__(self):
        # 扩展基因维度以支持分时段定价，利用时空不平衡特性
        # 0: Commission [0.1, 0.35]
        # 1: Surge Base (平峰基础溢价) [1.0, 1.5]
        # 2: Surge Peak Multiplier (高峰倍率) [1.0, 2.0] -> Peak Surge = Base * Multiplier
        # 3: Subsidy Base (平峰补贴) [0.0, 2.0]
        # 4: Subsidy Peak (高峰补贴) [0.0, 5.0]
        # 5: Peak Hour Center (高峰中心时间点，归一化 0-1) [0.2, 0.8]
        # 6: Peak Hour Width (高峰持续宽度) [0.05, 0.2]
        self.bounds = np.array([
            (0.1, 0.35),
            (1.0, 1.5),
            (1.0, 2.0),
            (0.0, 2.0),
            (0.0, 5.0),
            (0.2, 0.8),
            (0.05, 0.2)
        ])
        self.dim = 7

    def decode(self, gene):
        comm, s_base, s_peak_mul, sub_base, sub_peak, peak_center, peak_width = gene

        n_zones = CONFIG.get('N_ZONES', 10)
        time_steps = CONFIG.get('TIME_STEPS_PER_DAY', 144)

        # 构建时间向量 [0, 1]
        t_vec = np.linspace(0, 1, time_steps)

        # 使用高斯函数模拟高峰时段
        # peak_mask 接近 1 时为高峰，接近 0 时为平峰
        peak_mask = np.exp(-0.5 * ((t_vec - peak_center) / peak_width) ** 2)

        # 1. 计算 Lambda (Surge)
        # 形状: (Time, )
        surge_profile = s_base + (s_base * s_peak_mul - s_base) * peak_mask
        # 广播到所有区域 (Time, Zone) -> 未来可在此处扩展空间异质性
        lambda_matrix = np.tile(surge_profile[:, np.newaxis], (1, n_zones))

        # 2. 计算 Subsidy
        subsidy_profile = sub_base + (sub_peak - sub_base) * peak_mask
        subsidy_matrix = np.tile(subsidy_profile[:, np.newaxis], (1, n_zones))

        return {
            'commission': comm,
            'lambda': lambda_matrix,
            'subsidy': subsidy_matrix
        }


def evaluate_strategy_real(gene, trainer: StatsTrainer):
    encoder = StrategyEncoder()
    platform_params = encoder.decode(gene)
    objectives = trainer.train_and_evaluate(platform_params, num_episodes=5)  # 实际评估5个回合
    agent_weights = trainer.agent.get_weights()
    return objectives, agent_weights


# --- NSGA-II Utilities (Full Implementation) ---
def fast_non_dominated_sort(objectives):
    pop_size = objectives.shape[0]
    S = [[] for _ in range(pop_size)]
    n = np.zeros(pop_size)
    rank = np.zeros(pop_size)
    fronts = [[]]
    for p in range(pop_size):
        for q in range(pop_size):
            # Check if p dominates q (Maximization)
            if np.all(objectives[p] >= objectives[q]) and np.any(objectives[p] > objectives[q]):
                S[p].append(q)
            elif np.all(objectives[q] >= objectives[p]) and np.any(objectives[q] > objectives[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)
    return fronts[:-1], rank


def calculate_crowding_distance(objectives, fronts):
    distances = np.zeros(objectives.shape[0])
    for front in fronts:
        if len(front) == 0: continue
        l = len(front)
        for m in range(objectives.shape[1]):
            sorted_indices = sorted(front, key=lambda x: objectives[x, m])
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            f_min = objectives[sorted_indices[0], m]
            f_max = objectives[sorted_indices[-1], m]
            if f_max == f_min: continue
            for i in range(1, l - 1):
                distances[sorted_indices[i]] += (objectives[sorted_indices[i + 1], m] - objectives[
                    sorted_indices[i - 1], m]) / (f_max - f_min)
    return distances


def tournament_selection(pop_indices, ranks, distances):
    a, b = np.random.choice(pop_indices, 2, replace=False)
    if ranks[a] < ranks[b]:
        return a
    elif ranks[b] < ranks[a]:
        return b
    else:
        return a if distances[a] > distances[b] else b


def sbx_crossover(p1, p2, bounds, eta=15):
    u = np.random.rand()
    if u <= 0.5:
        beta = (2 * u) ** (1.0 / (eta + 1))
    else:
        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return np.clip(c1, bounds[:, 0], bounds[:, 1]), np.clip(c2, bounds[:, 0], bounds[:, 1])


def polynomial_mutation(p, bounds, eta=20, prob=0.1):
    if np.random.rand() > prob: return p
    mutant = np.copy(p)
    for i in range(len(p)):
        u = np.random.rand()
        if u < 0.5:
            delta = (2 * u) ** (1 / (eta + 1)) - 1
        else:
            delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
        mutant[i] += delta * (bounds[i, 1] - bounds[i, 0])
    return np.clip(mutant, bounds[:, 0], bounds[:, 1])


# --- ERL_Solver Class ---
class ERL_Solver:
    def __init__(self, simulator_path, pop_size=20, max_gens=30, use_surrogate=True, use_transfer=True):
        self.sim_path = simulator_path
        self.pop_size = pop_size
        self.max_gens = max_gens
        self.use_surrogate = use_surrogate
        self.use_transfer = use_transfer
        self.trainer = StatsTrainer(simulator_path=self.sim_path)
        self.encoder = StrategyEncoder()
        self.history = {'gen': [], 'best_profit': [], 'avg_driver_inc': [], 'best_quality': []}

    def solve(self):
        # 初始化种群
        sampler = LatinHypercube(d=self.encoder.dim)
        pop_genes = sampler.random(n=self.pop_size) * (
                self.encoder.bounds[:, 1] - self.encoder.bounds[:, 0]) + self.encoder.bounds[:, 0]

        archive_X = []
        archive_Y = []
        archive_W = {}

        init_w = self.trainer.agent.get_weights()

        print(f"--- ERL Start (Surrogate={self.use_surrogate}, Transfer={self.use_transfer}) ---")

        # 初始种群评估
        for i in range(self.pop_size):
            self.trainer.agent.load_by_weights(init_w)
            fit, w = evaluate_strategy_real(pop_genes[i], self.trainer)
            archive_X.append(pop_genes[i])
            archive_Y.append(fit)
            if self.use_transfer:
                archive_W[tuple(pop_genes[i])] = w

        for gen in range(self.max_gens):
            # 1. 代理模型训练和虚拟演化
            if self.use_surrogate:
                surrogates = [GaussianProcessRegressor(kernel=C(1.0) * RBF(1.0), n_restarts_optimizer=0) for _ in
                              range(3)]
                X_train = np.array(archive_X)
                Y_train = np.array(archive_Y)

                # 训练三个 GP 模型
                for k in range(3):
                    std = Y_train[:, k].std() + 1e-6
                    surrogates[k].fit(X_train, (Y_train[:, k] - Y_train[:, k].mean()) / std)

                # 虚拟演化 (Virtual Evolution)
                virtual_pop = np.copy(pop_genes)
                for _ in range(10):  # 运行10代虚拟演化
                    v_fitness = np.zeros((self.pop_size, 3))
                    for k in range(3):
                        v_fitness[:, k] = surrogates[k].predict(virtual_pop)

                    fronts, ranks = fast_non_dominated_sort(v_fitness)
                    dists = calculate_crowding_distance(v_fitness, fronts)

                    offspring = []
                    while len(offspring) < self.pop_size:
                        p1 = tournament_selection(range(self.pop_size), ranks, dists)
                        p2 = tournament_selection(range(self.pop_size), ranks, dists)
                        c1, c2 = sbx_crossover(virtual_pop[p1], virtual_pop[p2], self.encoder.bounds)
                        offspring.extend([polynomial_mutation(c1, self.encoder.bounds),
                                          polynomial_mutation(c2, self.encoder.bounds)])
                    virtual_pop = np.array(offspring[:self.pop_size])

                # 从虚拟种群中选择最佳个体
                v_fitness = np.zeros((self.pop_size, 3))
                for k in range(3):
                    v_fitness[:, k] = surrogates[k].predict(virtual_pop)
                fronts, _ = fast_non_dominated_sort(v_fitness)
                # 简单策略：随机选一个帕累托前沿的个体
                candidate_gene = virtual_pop[np.random.choice(fronts[0])]
            else:
                # 不使用代理模型时，直接变异生成候选
                idx = np.random.randint(0, self.pop_size)
                candidate_gene = polynomial_mutation(pop_genes[idx], self.encoder.bounds, prob=1.0)

            # 2. 真实评估 (Real Evaluation)
            if self.use_transfer and len(archive_X) > 0:
                # 找到最近邻并加载权重
                dists = np.linalg.norm(np.array(archive_X) - candidate_gene, axis=1)
                nearest_idx = np.argmin(dists)
                nearest_key = tuple(archive_X[nearest_idx])
                if nearest_key in archive_W:
                    self.trainer.agent.load_by_weights(archive_W[nearest_key])
                else:
                    self.trainer.agent.load_by_weights(init_w)
            else:
                self.trainer.agent.load_by_weights(init_w)

            real_fit, real_w = evaluate_strategy_real(candidate_gene, self.trainer)

            # 3. 更新存档
            archive_X.append(candidate_gene)
            archive_Y.append(real_fit)
            if self.use_transfer:
                archive_W[tuple(candidate_gene)] = real_w

            # 4. 更新种群 (简单替换)
            replace_idx = np.random.randint(0, self.pop_size)
            pop_genes[replace_idx] = candidate_gene

            # 5. 记录日志
            best_profit = np.max(np.array(archive_Y)[:, 0])
            self.history['gen'].append(gen)
            self.history['best_profit'].append(best_profit)
            self.history['avg_driver_inc'].append(np.mean(np.array(archive_Y)[:, 1]))
            print(f"Gen {gen} | Best Profit: {best_profit:.2f}")

        return self.history, np.array(archive_Y)


if __name__ == '__main__':
    # 简单的运行检查
    # 注意：运行此文件需要先运行 generate_split_simulators.py 生成 .pkl 文件
    if os.path.exists('model/generators/simulator_hex_weekday.pkl'):
        print("Simulator found, starting test run...")
        solver = ERL_Solver(simulator_path='model/generators/simulator_hex_weekday.pkl', max_gens=2, pop_size=4)
        solver.solve()
    else:
        # 为了防止报错，寻找任何可能的 simulator
        gen_dir = 'model/generators'
        if os.path.exists(gen_dir):
            files = [f for f in os.listdir(gen_dir) if f.endswith('.pkl')]
            if files:
                sim_path = os.path.join(gen_dir, files[0])
                print(f"Using found simulator: {sim_path}")
                solver = ERL_Solver(simulator_path=sim_path, max_gens=2, pop_size=4)
                solver.solve()
            else:
                print("Error: No simulator files found. Please run generate_split_simulators.py.")
        else:
            print("Error: model/generators directory not found.")