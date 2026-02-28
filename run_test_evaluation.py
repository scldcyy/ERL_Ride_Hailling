import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from shared_ppo import Trainer
from MORL_platform import PlatformPPOAgent, RLPlatformPolicy, extract_pareto_front
from MODE_platform import parameterized_surge, parameterized_subsidy


# --- 1. 还原 MOGP 字符串公式的辅助函数 ---
# 必须提供与 DEAP 注册时完全一致的受保护数学算子，以便使用 eval() 执行字符串
def protected_div(left, right):
    safe_right = np.where(np.abs(right) > 1e-6, right, 1.0)
    return np.where(np.abs(right) > 1e-6, left / safe_right, 1.0)


def protected_log(x):
    safe_x = np.where(np.abs(x) > 1e-6, np.abs(x), 1.0)
    return np.where(np.abs(x) > 1e-6, np.log(safe_x), 0.0)


# 定义映射字典，将 DEAP 的算子映射为 Python 可执行代码
GP_CONTEXT = {
    'add': np.add,
    'sub': np.subtract,
    'mul': np.multiply,
    'protected_div': protected_div,
    'protected_log': protected_log,
    'sin': np.sin,
    'np': np
}


def parse_gp_formula(formula_str):
    """将 MOGP 保存的字符串公式转化为可执行的 Lambda 函数"""

    def gp_func(t, no, nd, sd):
        local_vars = {'t': t, 'no': no, 'nd': nd, 'sd': sd}
        try:
            # 动态执行公式字符串
            res = eval(formula_str, GP_CONTEXT, local_vars)
            if np.isscalar(res) or np.ndim(res) == 0:
                res = np.full_like(no, float(res))
            return res
        except Exception as e:
            print(f"GP Formula Eval Error: {e}")
            return np.ones_like(no)

    return gp_func


def reevaluate_policies(sim_path, policies, algo_name, num_episodes=15):
    """在一个干净的环境中重新评估策略集合，提取真正的测试集帕累托前沿"""
    print(f"\n>>> Re-evaluating {algo_name} test performance ({len(policies)} policies)...")
    trainer = Trainer(simulator_path=sim_path)
    test_fitnesses = []

    for policy in tqdm(policies, desc=f"Eval {algo_name}"):
        trainer.reset_to_base_weights()
        trainer.agent.optimizer.state.clear()

        # 运行测试回合，取最后几轮的平稳均值
        fitness = trainer.train_and_evaluate(policy, num_episodes=num_episodes)
        test_fitnesses.append(fitness[:3])

    test_fitnesses = np.array(test_fitnesses)
    # 提取测试集上的帕累托前沿
    test_pf, _ = extract_pareto_front(test_fitnesses, policies)
    print(f"[{algo_name}] Retained {len(test_pf)} Pareto optimal points on test set.")
    return test_pf


def load_and_prepare_mogp(filepath):
    with open(filepath, 'rb') as f: data = pickle.load(f)
    return [{'surge': parse_gp_formula(s), 'subsidy': parse_gp_formula(sub)} for s, sub in data['formulas']]


def load_and_prepare_mode(filepath):
    with open(filepath, 'rb') as f: data = pickle.load(f)
    return [{'surge': lambda t, no, nd, sd, p=p: parameterized_surge(t, no, nd, sd, p),
             'subsidy': lambda t, no, nd, sd, p=p: parameterized_subsidy(t, no, nd, sd, p)} for p in data['parameters']]


def load_and_prepare_rl(filepath):
    agent = PlatformPPOAgent()
    agent.load(filepath)
    # 生成均匀分布的测试偏好
    test_weights = [
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
        [0.33, 0.33, 0.34], [0.7, 0.15, 0.15], [0.15, 0.7, 0.15], [0.15, 0.15, 0.7]
    ]
    policies = []
    for w in test_weights:
        rl_policy = RLPlatformPolicy(agent, w, is_eval=True)  # 启用 eval 模式
        policies.append({'surge': rl_policy.surge, 'subsidy': rl_policy.subsidy})
    return policies


def plot_test_pareto_fronts(pf_dict, save_path="results/test_pareto_comparison.png"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = {'MOGP': '#4C72B0', 'MODE': '#55A868', 'RL': '#C44E52'}
    markers = {'MOGP': 'o', 'MODE': '^', 'RL': '*'}
    sizes = {'MOGP': 60, 'MODE': 60, 'RL': 200}

    for algo, pf in pf_dict.items():
        if len(pf) > 0:
            ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2],
                       c=colors[algo], marker=markers[algo], s=sizes[algo], alpha=0.8, label=algo)

    ax.set_xlabel('Profit')
    ax.set_ylabel('Efficiency / -Wait Time')
    ax.set_zlabel('Fairness / -Gini')
    ax.set_title('True Test Set 3D Pareto Front')
    ax.legend()
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# --- 2. 策略加载工厂 ---
def load_best_mogp_policy(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    # 假设我们挑选利润 (Profit，第0维) 最高的那组策略
    best_idx = np.argmax([fit[0] for fit in data['pareto_fitness']])
    best_surge_str, best_subsidy_str = data['formulas'][best_idx]

    print(f"[Loaded MOGP] Surge: {best_surge_str}")
    print(f"[Loaded MOGP] Subsidy: {best_subsidy_str}")

    return {
        'surge': parse_gp_formula(best_surge_str),
        'subsidy': parse_gp_formula(best_subsidy_str)
    }


def load_best_mode_policy(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    best_idx = np.argmax([fit[0] for fit in data['pareto_fitness']])
    best_params = data['parameters'][best_idx]

    print(f"[Loaded MODE] Params: {np.round(best_params, 2)}")

    return {
        'surge': lambda t, no, nd, sd: parameterized_surge(t, no, nd, sd, best_params),
        'subsidy': lambda t, no, nd, sd: parameterized_subsidy(t, no, nd, sd, best_params)
    }


def load_rl_policy(filepath):
    agent = PlatformPPOAgent()
    agent.load(filepath)
    pref_weights = np.random.dirichlet(np.ones(3))
    rl_policy = RLPlatformPolicy(agent,pref_weights)
    print(f"[Loaded RL] Model weights loaded from {filepath}")
    return {
        'surge': rl_policy.surge,
        'subsidy': rl_policy.subsidy
    }


# --- 3. 运行评估循环 ---
def run_evaluation(sim_path, policy_params, algo_name, num_episodes=15):
    print(f"\n>>> Evaluating {algo_name} Strategy ...")
    trainer = Trainer(simulator_path=sim_path)

    # 重置底层司机经验，确保公平起跑
    trainer.reset_to_base_weights()
    trainer.agent.optimizer.state.clear()

    # 运行多轮以达到下层司机的纳什均衡
    fitness = trainer.train_and_evaluate(policy_params, num_episodes=num_episodes)
    return fitness  # 返回 [Profit, -Wait Time, -Gini]


# --- 4. 可视化对比 ---
def plot_comparison(results_dict, save_path="results/algorithm_comparison.png"):
    algos = list(results_dict.keys())
    profits = [results_dict[alg][0] for alg in algos]
    efficiencies = [results_dict[alg][1] for alg in algos]  # Wait Time (负值)
    fairnesses = [results_dict[alg][2] for alg in algos]  # Gini (负值)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#4C72B0', '#55A868', '#C44E52']

    # Profit (越高越好)
    axs[0].bar(algos, profits, color=colors)
    axs[0].set_title('Platform Profit (Higher is better)')
    axs[0].set_ylabel('Profit')

    # Efficiency (越高越好，因为是负的等待时间)
    axs[1].bar(algos, efficiencies, color=colors)
    axs[1].set_title('Efficiency / -Wait Time (Higher is better)')

    # Fairness (越高越好，因为是负的Gini)
    axs[2].bar(algos, fairnesses, color=colors)
    axs[2].set_title('Fairness / -Gini (Higher is better)')

    for ax in axs:
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n[Done] Comparison plot saved to {save_path}")
    plt.show()


def plot_pareto_fronts(mogp_path, mode_path, rl_scores=None, save_path="results/pareto_comparison_3d.png"):
    """绘制 MOGP, MODE 的 3D 帕累托前沿，以及 RL 的单点对比"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 加载并绘制 MOGP 的帕累托前沿
    if os.path.exists(mogp_path):
        with open(mogp_path, 'rb') as f:
            mogp_data = pickle.load(f)
            mogp_pf = np.array(mogp_data['pareto_fitness'])
            ax.scatter(mogp_pf[:, 0], mogp_pf[:, 1], mogp_pf[:, 2],
                       c='#4C72B0', marker='o', s=60, alpha=0.8, label='MOGP Pareto Front')

    # 2. 加载并绘制 MODE 的帕累托前沿
    if os.path.exists(mode_path):
        with open(mode_path, 'rb') as f:
            mode_data = pickle.load(f)
            mode_pf = np.array(mode_data['pareto_fitness'])
            ax.scatter(mode_pf[:, 0], mode_pf[:, 1], mode_pf[:, 2],
                       c='#55A868', marker='^', s=60, alpha=0.8, label='MODE Pareto Front')

    # 3. 绘制 RL 的最终表现点
    if rl_scores is not None:
        ax.scatter(rl_scores[0], rl_scores[1], rl_scores[2],
                   c='#C44E52', marker='*', s=300, edgecolors='black', label='RL Policy')

    ax.set_xlabel('Profit (Higher is Better)')
    ax.set_ylabel('Efficiency / -Wait Time')
    ax.set_zlabel('Fairness / -Gini')
    ax.set_title('3D Pareto Front Comparison')
    ax.legend()

    # 调整视角以便更好地观察 Profit 轴
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n[Done] 3D Pareto Front plot saved to {save_path}")
    plt.show()


if __name__ == '__main__':
    sim_path = 'model/generators/simulator_hex_scaling=0.004257843312339327_weekday.pkl'
    EVAL_EPISODES = 15
    test_fronts = {}

    if os.path.exists("results/samogp_results.pkl"):
        policies = load_and_prepare_mogp("results/samogp_results.pkl")
        test_fronts['MOGP'] = reevaluate_policies(sim_path, policies, 'MOGP', EVAL_EPISODES)

    if os.path.exists("results/samode_results.pkl"):
        policies = load_and_prepare_mode("results/samode_results.pkl")
        test_fronts['MODE'] = reevaluate_policies(sim_path, policies, 'MODE', EVAL_EPISODES)

    if os.path.exists("results/morl_platform_policy.pth"):
        policies = load_and_prepare_rl("results/morl_platform_policy111.pth")
        test_fronts['RL'] = reevaluate_policies(sim_path, policies, 'RL', EVAL_EPISODES)

    if test_fronts:
        plot_test_pareto_fronts(test_fronts)

    with open("results/test_pareto_fronts111.pkl", "wb") as f:
        pickle.dump(test_fronts, f)