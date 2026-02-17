import numpy as np
import torch
import copy
from collections import deque

# 引入之前的模块
from env_core import HeterogeneousRideHailingEnv, CONFIG
from agent_core import SharedPPOAgent


class StateAdapter:
    """
    适配器：将环境的原始数据转换为 PPO Agent 需要的特征向量
    Feature Dim = 5: [Norm_Lat, Norm_Lng, Local_Surge, Local_Demand_Supply_Ratio, Time]
    """

    @staticmethod
    def extract_driver_states(env, current_prices):
        # 1. 位置归一化 (简单处理，假设 Hex ID 映射到 0-N)
        # 实际项目中建议使用 H3 的 lat/lng，这里用 idx/n_zones 代替
        loc_norm = env.driver_locations / env.n_zones

        # 2. 时间归一化
        time_norm = np.full(CONFIG['N_DRIVERS'], env.time / CONFIG['TIME_STEPS_PER_DAY'])

        # 3. 局部价格 (Surge)
        # current_prices 是全图所有区域的价格向量
        local_surges = current_prices[env.driver_locations]

        # 4. 局部供需比 (作为司机感知热度的特征)
        # 需实时计算所有区域的供需
        zone_demand = np.zeros(env.n_zones)
        for o in env.pending_orders: zone_demand[o['origin_idx']] += 1

        zone_supply = np.zeros(env.n_zones)
        idle_drivers = (env.driver_status == 0)
        np.add.at(zone_supply, env.driver_locations[idle_drivers], 1)

        ratios = zone_demand / (zone_supply + 1.0)
        local_ratios = ratios[env.driver_locations]

        # 5. 辅助特征：是否归属于繁忙区域 (简单示例)
        # 这里为了保持维度为 5，我们堆叠这些特征
        # Shape: (N_DRIVERS, 5)
        states = np.stack([
            loc_norm,  # 0: 位置
            loc_norm,  # 1: 经度 (暂以前者代替，实际需h3解析)
            local_surges,  # 2: 当前价格
            local_ratios,  # 3: 供需状况
            time_norm  # 4: 时间
        ], axis=1)

        return states


class BiLevelEvaluator:
    """
    双层博弈评估器 (The Judge)
    职责：接收上层的一个策略 -> 训练下层司机适应 -> 返回上层目标函数值
    """

    def __init__(self, simulator_path):
        print(f"Initializing Bi-level Evaluator with {simulator_path}...")
        self.env = HeterogeneousRideHailingEnv(simulator_path)

        # 初始化下层司机 (State Dim=5, Action Dim=7)
        self.driver_agent = SharedPPOAgent(state_dim=5, action_dim=7)

        # 备份初始权重，用于每次评估前重置司机（保证公平比较）
        # 在实际论文中，也可以采用 "Co-evolution" 不重置，视实验设置而定
        # 这里采用 "Reset" 模式，更严谨但计算量大
        self.initial_agent_weights = copy.deepcopy(self.driver_agent.policy.state_dict())

    def _wrap_pricing_strategy(self, strategy_input, strategy_type):
        """将 GP/DE/RL 的输出统一包装为函数 P(d, s, t) -> price_map"""

        if strategy_type == 'GP':
            # GP Input: DEAP compiled function
            func = strategy_input

            def pricing_wrapper(d_vec, s_vec, t):
                # 归一化输入
                d_norm = d_vec / (d_vec.max() + 1e-6)
                s_norm = s_vec / (s_vec.max() + 1e-6)
                t_norm = t / CONFIG['TIME_STEPS_PER_DAY']

                # 尝试向量化调用，如果 GP func 不支持则循环
                try:
                    surges = func(d_norm, s_norm, t_norm)
                except:
                    # Fallback for scalar-only functions
                    surges = np.array([func(d, s, t_norm) for d, s in zip(d_norm, s_norm)])

                return np.clip(surges, 1.0, 5.0)  # 价格上下界约束

            return pricing_wrapper

        elif strategy_type == 'DE':
            # DE Input: Parameters [w1, w2, w3, bias]
            # 假设公式: Price = bias + w1*D + w2*S + w3*Time (简单线性示例)
            w = strategy_input

            def pricing_wrapper(d_vec, s_vec, t):
                d_norm = d_vec / (d_vec.max() + 1e-6)
                s_norm = s_vec / (s_vec.max() + 1e-6)
                t_norm = t / CONFIG['TIME_STEPS_PER_DAY']

                raw = w[3] + w[0] * d_norm + w[1] * s_norm + w[2] * t_norm
                return np.clip(raw, 1.0, 5.0)

            return pricing_wrapper

        return None

    def evaluate(self, strategy_input, strategy_type='GP', adaptation_epochs=5, test_episodes=1):
        """
        核心评估流程
        Returns: [Total_Profit, Service_Rate, Gini_Index]
        """
        # 1. 包装策略
        pricing_func = self._wrap_pricing_strategy(strategy_input, strategy_type)

        # 2. 重置司机智能体 (公平起见，每次从白板或预训练模型开始)
        self.driver_agent.policy.load_state_dict(self.initial_agent_weights)
        self.driver_agent.optimizer = torch.optim.Adam(self.driver_agent.policy.parameters(), lr=3e-4)

        # 3. Adaptation Phase (司机适应期)
        # 在测试指标前，先让司机针对当前价格策略训练几轮
        for _ in range(adaptation_epochs):
            self._run_episode(pricing_func, training=True)

        # 4. Test Phase (真实评估期)
        # 司机策略固定，统计指标
        metrics_buffer = {'profit': [], 'service': [], 'gini': []}
        for _ in range(test_episodes):
            m = self._run_episode(pricing_func, training=False)
            metrics_buffer['profit'].append(m['profit'])
            metrics_buffer['service'].append(m['service_rate'])
            metrics_buffer['gini'].append(m['gini'])

        # 返回均值
        return [
            np.mean(metrics_buffer['profit']),
            np.mean(metrics_buffer['service']),
            np.mean(metrics_buffer['gini'])
        ]

    def _run_episode(self, pricing_func, training):
        """运行单日仿真"""
        # 获取初始观测
        # 注意：env.reset() 返回的是 dict，我们需要转为 tensor
        obs_dict = self.env.reset()

        # 初始定价 (用于生成初始 State)
        # 此时 d=0, s=all, t=0
        d_init = np.zeros(self.env.n_zones)
        s_init = np.zeros(self.env.n_zones)
        # 简单处理：初始 supply 分布
        drivers = obs_dict['driver_locs']
        np.add.at(s_init, drivers, 1)

        current_prices = pricing_func(d_init, s_init, 0)
        state_vec = StateAdapter.extract_driver_states(self.env, current_prices)

        # 追踪累计收益以计算 Reward
        last_incomes = copy.deepcopy(obs_dict['driver_income'])

        episode_done = False
        while not episode_done:
            # A. 司机决策
            actions = self.driver_agent.select_action(state_vec, training=training)

            # B. 环境演化
            # env.step 内部会调用 pricing_func 更新价格
            _, platform_profit, episode_done, info = self.env.step(actions, pricing_func)

            # C. 重新计算价格和状态 (用于下一时刻)
            # 这一步略显冗余，因为 step 内部算过一次，但为了获取 State 必须再算或从 env 获取
            # 为了解耦，我们在这里重新获取 env 的供需并计算价格
            # (优化点：让 env.step 返回 current_prices)

            # 获取新时刻的供需
            d_curr = np.zeros(self.env.n_zones)
            for o in self.env.pending_orders: d_curr[o['origin_idx']] += 1
            s_curr = np.zeros(self.env.n_zones)
            idle_mask = (self.env.driver_status == 0)
            np.add.at(s_curr, self.env.driver_locations[idle_mask], 1)

            new_prices = pricing_func(d_curr, s_curr, self.env.time)
            next_state_vec = StateAdapter.extract_driver_states(self.env, new_prices)

            # D. 计算司机奖励 (Step Reward)
            # Reward = (Current_Income - Last_Income) - Cost
            # Cost 在 env.step 里已经扣除到 income 里了 (油耗等)，所以直接看 income 增量
            current_incomes = self.env.accumulated_income
            step_rewards = (current_incomes - last_incomes)
            # 加上一点生存奖励或惩罚
            # step_rewards += -0.01

            last_incomes = copy.deepcopy(current_incomes)

            # E. 存储与训练
            if training:
                # 只有活跃司机才产生有效训练样本
                # 这里简化：存所有
                self.driver_agent.store_reward(step_rewards, np.full(CONFIG['N_DRIVERS'], episode_done))

            state_vec = next_state_vec

        # 一天结束，如果是训练模式，更新 Agent
        if training:
            self.driver_agent.update()

        return info


# --- 测试代码 (if __name__ == "__main__") ---
if __name__ == "__main__":
    import os

    # 1. 确保有模拟器文件 (生成一个假的用于测试)
    dummy_sim_path = "dummy_simulator.pkl"
    if not os.path.exists(dummy_sim_path):
        print("Generating dummy simulator for testing...")
        import pickle


        # 伪造一个简单的 Simulator 对象结构
        class DummySim:
            def __init__(self):
                self.adjacency = {i: {0: i} for i in range(277)}  # 简单的自环拓扑

            def generate_orders(self, t, hexes):
                # 随机生成几个订单
                import random
                return [{'origin_hex': 0, 'dest_hex': 0, 'duration': 10}] if random.random() < 0.5 else []


        with open(dummy_sim_path, 'wb') as f:
            pickle.dump(DummySim(), f)

    # 2. 初始化评估器
    evaluator = BiLevelEvaluator(dummy_sim_path)
    print("Evaluator Initialized.")

    # 3. 测试 DE 策略评估
    # 参数: [w1(D), w2(S), w3(T), bias]
    test_params = [0.5, -0.2, 0.1, 1.2]

    print("\n>>> Testing DE Strategy Evaluation...")
    print("Running Adaptation (Training drivers)... this might take time.")
    results = evaluator.evaluate(test_params, strategy_type='DE', adaptation_epochs=2, test_episodes=1)

    print(f"\nEvaluation Result:")
    print(f"Total Profit: {results[0]:.2f}")
    print(f"Service Rate: {results[1]:.2f}")
    print(f"Gini Index:   {results[2]:.4f}")

    # 4. 测试 GP 策略评估 (模拟一个简单函数)
    print("\n>>> Testing GP Strategy Evaluation...")
    # 模拟一个 lambda 函数: Price = 1 + Demand
    gp_mock_func = lambda d, s, t: 1.0 + d * 2.0

    results_gp = evaluator.evaluate(gp_mock_func, strategy_type='GP', adaptation_epochs=1, test_episodes=1)
    print(f"GP Result: {results_gp}")

    print("\nTest Finished Successfully.")