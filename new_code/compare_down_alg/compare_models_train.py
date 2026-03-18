import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from generate_simulator import PassengerSimulator
from basic_config import CONFIG
from ride_hailing_env import RideHailingEnv

# 导入三个版本的 Agent
from shared_ppo import SharedPPOAgent as MLPAgent
from attention_ppo import AttentionPPOAgent
from gcn_ppo import GCNPPOAgent


def calculate_gini(incomes):
    if len(incomes) == 0: return 0.0
    incomes = np.sort(np.clip(incomes, 0, None))
    n = len(incomes)
    index = np.arange(1, n + 1)
    total_income = np.sum(incomes)
    if total_income == 0: return 0.0
    return (2 * np.sum(index * incomes)) / (n * total_income) - (n + 1) / n


def train_agent(agent_name, agent_class, simulator_path, platform_params, num_episodes=30):
    print(f"\n--- Training {agent_name} ---")

    # 固定随机种子以保证环境初始化的公平性
    torch.manual_seed(42)
    np.random.seed(42)

    env = RideHailingEnv(simulator_path)
    agent = agent_class(CONFIG['STATE_DIM'], CONFIG['ACTION_DIM'], **CONFIG)

    ep_profits, ep_completion_rates, ep_ginis = [], [], []

    for ep in tqdm(range(num_episodes)):
        state, action_mask = env.reset()
        ep_total_profit = 0

        while True:
            current_active_flags = (env.driver_status == 0)
            actions = agent.select_actions(state, action_mask)

            def critic_estimator_batch(d_ids, dest_idxs):
                if not d_ids: return np.array([])
                proxy_states = [env.get_proxy_state(d, dest) for d, dest in zip(d_ids, dest_idxs)]
                state_tensor = torch.FloatTensor(np.array(proxy_states))
                with torch.no_grad():
                    # 兼容 MLP, Attention 和 GCN 的特征处理逻辑
                    if hasattr(agent.policy_old, '_process_state'):
                        features = agent.policy_old._process_state(state_tensor)
                        return agent.policy_old.critic(features).squeeze(-1).numpy()
                    else:
                        return agent.policy_old.critic(state_tensor).squeeze(-1).numpy()

            next_state, rewards, done, info = env.step(actions, platform_params, value_estimator=critic_estimator_batch)

            agent.buffer.rewards.append(rewards)
            agent.buffer.active_flags.append(current_active_flags)

            ep_total_profit += info['step_profit']
            state = next_state
            action_mask = info['action_mask']

            if done:
                total_demand = info['total_generated'] + 1e-6
                completion_rate = info['total_served'] / total_demand
                gini_index = calculate_gini(info['driver_income'])

                ep_profits.append(ep_total_profit)
                ep_completion_rates.append(completion_rate)
                ep_ginis.append(gini_index)
                break

        agent.update()

    # 训练结束后保存模型权重
    save_path = f"ppo_agent_{agent_name.lower()}.pth"
    torch.save(agent.policy.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return {
        'profits': ep_profits,
        'completion_rates': ep_completion_rates,
        'ginis': ep_ginis
    }


def plot_three_way_comparison(results_mlp, results_attn, results_gcn):
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

    metrics = [
        ('profits', 'Total Platform Profit'),
        ('completion_rates', 'Completion Rate'),
        ('ginis', 'Driver Income Gini Index (Lower is fairer)')
    ]

    colors = {
        'Baseline (MLP)': '#1f77b4',  # 蓝色
        'Attention + PosEncode': '#ff7f0e',  # 橙色
        'GCN (Graph Conv)': '#2ca02c'  # 绿色
    }

    for idx, (key, title) in enumerate(metrics):
        axs[idx].plot(results_mlp[key], label='Baseline (MLP)', color=colors['Baseline (MLP)'], linestyle='--',
                      alpha=0.8)
        axs[idx].plot(results_attn[key], label='Attention + PosEncode', color=colors['Attention + PosEncode'],
                      linewidth=2, alpha=0.8)
        axs[idx].plot(results_gcn[key], label='GCN (Graph Conv)', color=colors['GCN (Graph Conv)'], linewidth=2,
                      alpha=0.8)

        axs[idx].set_title(title, fontsize=14)
        axs[idx].set_xlabel('Episodes', fontsize=12)
        axs[idx].grid(True, alpha=0.3)
        axs[idx].legend(fontsize=11)

    plt.tight_layout()
    save_file = 'architecture_comparison_3way.png'
    plt.savefig(save_file, dpi=150)
    print(f"\nThree-way comparison plot saved to '{save_file}'")
    plt.show()


if __name__ == '__main__':
    platform_params = {
        'surge': lambda t, no, nd, sd: 1.0 + 1.2 * np.maximum(0, sd - 0.8) ** 1.2,
        'subsidy': lambda t, no, nd, sd: 3.0 * (
                    np.exp(-50 * (t - 0.30) ** 2) + np.exp(-40 * (t - 0.75) ** 2)) * np.maximum(0, sd - 0.5)
    }
    sim_path = '../generator/simulator_driver_nums=400_hex_scaling=0.017031373249357308_weekday.pkl'

    if not os.path.exists(sim_path):
        print(f"Simulator path invalid: {sim_path}")
    else:
        # 依次运行三种架构的训练

        results_gcn = train_agent("GCN", GCNPPOAgent, sim_path, platform_params, num_episodes=30)
        results_attn = train_agent("Attention", AttentionPPOAgent, sim_path, platform_params, num_episodes=30)
        results_mlp = train_agent("MLP", MLPAgent, sim_path, platform_params, num_episodes=30)


        # 绘制并保存三者对比图
        plot_three_way_comparison(results_mlp, results_attn, results_gcn)
