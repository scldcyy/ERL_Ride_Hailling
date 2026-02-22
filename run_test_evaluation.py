import os
import pickle
import numpy as np
import torch
from deap import gp
from tqdm import tqdm

from basic_config import CONFIG
from ride_hailing_env import RideHailingEnv
from shared_ppo import Trainer, calculate_gini
from RL_platform import PlatformPPOAgent, RLPlatformPolicy
from MODE_platform import parameterized_surge, parameterized_subsidy
# 导入 GP 必须的算子环境
from MOGP_platform import pset, toolbox


def evaluate_policy_on_test_env(sim_path, platform_params, num_test_episodes=5):
    """在固定随机种子的测试环境中评估策略"""
    env = RideHailingEnv(sim_path)
    trainer = Trainer(simulator_path=sim_path)  # 复用底层已训练好的司机 PPO (此处省略加载底层权重的代码，假设用同一套)

    test_profits, test_efficiencies, test_fairnesses = [], [], []

    for i in range(num_test_episodes):
        # 强制固定测试集的全局种子，保证每个算法遇到的订单生成、位置初始化完全一致
        np.random.seed(2026 + i)
        torch.manual_seed(2026 + i)

        state = trainer.env.reset()
        ep_total_profit = 0

        while True:
            # 使用决定性动作评估（argmax），消除底层的探索噪音
            actions = trainer.agent.select_actions(state)
            next_state, rewards, done, info = trainer.env.step(actions, platform_params)
            ep_total_profit += info['step_profit']
            state = next_state

            if done:
                wait_time = -info['total_wait_time']
                gini_index = calculate_gini(info['driver_income'])

                test_profits.append(ep_total_profit)
                test_efficiencies.append(wait_time)
                test_fairnesses.append(-gini_index)
                break

    return np.mean(test_profits), np.mean(test_efficiencies), np.mean(test_fairnesses)


def run_fair_benchmark(sim_path, save_dir="results"):
    print("=== Starting Fair Test Set Evaluation ===")
    test_results = {}

    # 1. 评估 SAMO-GP 测试集前沿
    gp_path = f"{save_dir}/samogp_results.pkl"
    if os.path.exists(gp_path):
        print("\nEvaluating SAMO-GP Pareto Front on Test Set...")
        with open(gp_path, 'rb') as f:
            gp_data = pickle.load(f)
        gp_test_front = []
        for surge_str, subsidy_str in tqdm(gp_data['formulas']):
            surge_func = toolbox.compile(expr=gp.PrimitiveTree.from_string(surge_str, pset))
            subsidy_func = toolbox.compile(expr=gp.PrimitiveTree.from_string(subsidy_str, pset))
            params = {
                'surge': lambda t, no, nd, sd: np.vectorize(surge_func)(t, no, nd, sd),
                'subsidy': lambda t, no, nd, sd: np.vectorize(subsidy_func)(t, no, nd, sd)
            }
            res = evaluate_policy_on_test_env(sim_path, params)
            gp_test_front.append(res)
        test_results['SAMO-GP'] = np.array(gp_test_front)

    # 2. 评估 SAMO-DE 测试集前沿 (逻辑类似，解析固定的 P 向量模板)
    de_path = f"{save_dir}/samode_results.pkl"
    if os.path.exists(de_path):
        print("\nEvaluating SAMO-DE Pareto Front on Test Set...")
        with open(de_path, 'rb') as f:
            de_data = pickle.load(f)

        de_test_front = []
        for p_vector in tqdm(de_data['parameters']):
            # 闭包绑定 p_vector，避免 lambda 后期求值问题
            params = {
                'surge': lambda t, no, nd, sd, p=p_vector: parameterized_surge(t, no, nd, sd, p),
                'subsidy': lambda t, no, nd, sd, p=p_vector: parameterized_subsidy(t, no, nd, sd, p)
            }
            res = evaluate_policy_on_test_env(sim_path, params)
            de_test_front.append(res)
        test_results['SAMO-DE'] = np.array(de_test_front)

        # 3. 评估 MARL-PPO 最终策略
    marl_path = f"{save_dir}/marl_platform_policy.pth"
    if os.path.exists(marl_path):
        print("\nEvaluating MARL-PPO on Test Set...")
        agent = PlatformPPOAgent()
        agent.load(marl_path)
        rl_policy = RLPlatformPolicy(agent)
        params = {'surge': rl_policy.surge, 'subsidy': rl_policy.subsidy}

        # RL 没有前沿，测 10 次作为轨迹散点
        rl_test_front = []
        for _ in tqdm(range(10)):
            rl_test_front.append(evaluate_policy_on_test_env(sim_path, params, num_test_episodes=1))
        test_results['MARL'] = np.array(rl_test_front)

    # 保存严谨的测试集结果
    with open(f"{save_dir}/test_pareto_fronts.pkl", 'wb') as f:
        pickle.dump(test_results, f)
    print("\nFair test evaluation completed. Saved to test_pareto_fronts.pkl")


if __name__ == "__main__":
    sim_path = 'model/generators/simulator_hex_scaling=0.004257843312339327_weekday.pkl'
    test_results=pickle.load(open('results/test_pareto_fronts.pkl', 'rb'))
    pass
    # run_fair_benchmark(sim_path)