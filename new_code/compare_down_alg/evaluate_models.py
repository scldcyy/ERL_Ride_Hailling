import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from generate_simulator import PassengerSimulator
from basic_config import CONFIG
from ride_hailing_env import RideHailingEnv
from shared_ppo import SharedPPOAgent
from attention_ppo import AttentionPPOAgent


def calculate_gini(incomes):
    """计算基尼系数"""
    if len(incomes) == 0: return 0.0
    incomes = np.sort(np.clip(incomes, 0, None))
    n = len(incomes)
    index = np.arange(1, n + 1)
    total_income = np.sum(incomes)
    if total_income == 0: return 0.0
    return (2 * np.sum(index * incomes)) / (n * total_income) - (n + 1) / n


def evaluate_model(agent_name, agent, env, platform_params, num_episodes=5):
    print(f"\n>>> 正在评估模型: {agent_name}")

    # 切换至评估模式，关闭 Dropout/BatchNorm 等层（虽然当前网络主要为 Linear，但这是好习惯）
    agent.policy.eval()

    ep_profits = []
    ep_completion_rates = []
    ep_wait_times = []
    ep_ginis = []

    for ep in tqdm(range(num_episodes), desc=f"Evaluating {agent_name}"):
        # 固定每次 Episode 的 Seed，确保不同模型面对同样的初始环境和订单序列
        np.random.seed(100 + ep)
        torch.manual_seed(100 + ep)

        state, action_mask = env.reset()
        ep_total_profit = 0

        while True:
            # 推理动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                mask_tensor = torch.BoolTensor(action_mask)
                # 直接调用策略网络生成动作
                actions, _ = agent.policy.act(state_tensor, mask_tensor)
                actions = actions.numpy()

            # 动态构建 Critic 预估器 (兼容 MLP 和 Attention 两种架构)
            def critic_estimator_batch(d_ids, dest_idxs):
                if not d_ids: return np.array([])
                proxy_states = [env.get_proxy_state(d, dest) for d, dest in zip(d_ids, dest_idxs)]
                state_tensor = torch.FloatTensor(np.array(proxy_states))
                with torch.no_grad():
                    # 如果有注意力特征处理方法，说明是 Attention 网络
                    if hasattr(agent.policy, '_process_state'):
                        features = agent.policy._process_state(state_tensor)
                        return agent.policy.critic(features).squeeze(-1).numpy()
                    else:
                        return agent.policy.critic(state_tensor).squeeze(-1).numpy()

            next_state, rewards, done, info = env.step(
                actions, platform_params,
                value_estimator=critic_estimator_batch
            )

            ep_total_profit += info['step_profit']
            state = next_state
            action_mask = info['action_mask']

            if done:
                total_demand = info['total_generated'] + 1e-6
                completion_rate = info['total_served'] / total_demand
                wait_time = info['total_wait_time'] / (info['total_served'] + 1e-6)
                gini_index = calculate_gini(info['driver_income_rate'])

                ep_profits.append(ep_total_profit)
                ep_completion_rates.append(completion_rate)
                ep_wait_times.append(wait_time)
                ep_ginis.append(gini_index)
                break

    return {
        'profits': ep_profits,
        'completion_rates': ep_completion_rates,
        'wait_times': ep_wait_times,
        'ginis': ep_ginis
    }


def plot_evaluation_results(results_dict):
    """绘制各项 KPI 对比图表"""
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()

    metrics = [
        ('profits', 'Total Platform Profit ($)', 'higher'),
        ('completion_rates', 'Order Completion Rate', 'higher'),
        ('wait_times', 'Avg Passenger Wait Time', 'lower'),
        ('ginis', 'Driver Income Gini Index', 'lower')
    ]

    models = list(results_dict.keys())
    colors = ['#1f77b4', '#ff7f0e']  # 蓝色和橙色

    for idx, (metric_key, title, direction) in enumerate(metrics):
        ax = axs[idx]

        # 提取数据绘制箱线图或折线图。这里使用带误差棒的柱状图显示均值和波动
        means = [np.mean(results_dict[model][metric_key]) for model in models]
        stds = [np.std(results_dict[model][metric_key]) for model in models]

        bars = ax.bar(models, means, yerr=stds, capsize=10, color=colors, alpha=0.8)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # 标注最优方向
        best_note = "↑ (Higher is better)" if direction == 'higher' else "↓ (Lower is better)"
        ax.set_ylabel(f"Metric Value {best_note}")

        # 在柱子上标注具体数值
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.3f}',
                    ha='center', va='bottom', fontsize=12)

    plt.suptitle('Performance Comparison: Baseline MLP vs. Attention PPO (Test Env)', fontsize=18, y=1.02)
    plt.tight_layout()
    save_path = 'model_evaluation_comparison.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n对比图表已保存至: {save_path}")
    plt.show()


if __name__ == '__main__':
    # ==================== 配置区 ====================
    # 请确保这里填入你实际保存的模型权重文件名
    MLP_MODEL_PATH = 'Baseline_MLP.pth'  # MLP基线模型
    ATTN_MODEL_PATH = 'Attention_PPO.pth'  # 注意力网络模型

    SIMULATOR_PATH = '../generator/simulator_driver_nums=400_hex_scaling=0.017031373249357308_weekday.pkl'
    NUM_EVAL_EPISODES = 10  # 评估的天数
    # ================================================

    platform_params = {
        'surge': lambda t, no, nd, sd: 1.0 + 1.2 * np.maximum(0, sd - 0.8) ** 1.2,
        'subsidy': lambda t, no, nd, sd: 3.0 * (
                    np.exp(-50 * (t - 0.30) ** 2) + np.exp(-40 * (t - 0.75) ** 2)) * np.maximum(0, sd - 0.5)
    }

    if not os.path.exists(SIMULATOR_PATH):
        print(f"找不到模拟器文件: {SIMULATOR_PATH}")
    else:
        results = {}
        env = RideHailingEnv(SIMULATOR_PATH)

        # 1. 评估 Baseline MLP
        if os.path.exists(MLP_MODEL_PATH):
            agent_mlp = SharedPPOAgent(CONFIG['STATE_DIM'], CONFIG['ACTION_DIM'], **CONFIG)
            agent_mlp.policy.load_state_dict(torch.load(MLP_MODEL_PATH))
            results['Baseline MLP'] = evaluate_model('Baseline MLP', agent_mlp, env, platform_params, NUM_EVAL_EPISODES)
        else:
            print(f"未找到 MLP 模型: {MLP_MODEL_PATH}，请检查路径。")

        # 2. 评估 Attention PPO
        if os.path.exists(ATTN_MODEL_PATH):
            agent_attn = AttentionPPOAgent(CONFIG['STATE_DIM'], CONFIG['ACTION_DIM'], **CONFIG)
            agent_attn.policy.load_state_dict(torch.load(ATTN_MODEL_PATH))
            results['Attention PPO'] = evaluate_model('Attention PPO', agent_attn, env, platform_params,
                                                      NUM_EVAL_EPISODES)
        else:
            print(f"未找到 Attention 模型: {ATTN_MODEL_PATH}，请检查路径。")

        # 3. 绘制对比图
        if len(results) == 2:
            plot_evaluation_results(results)