import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import matplotlib.pyplot as plt
from tqdm import tqdm

from basic_config import CONFIG
from shared_ppo import Trainer, calculate_gini


# --- 1. 偏好条件化的多目标连续动作 PPO 代理 (MORL Leader) ---
class PlatformActorCritic(nn.Module):
    # state_dim 改为 7: [t, no, nd, sd, w_profit, w_efficiency, w_fairness]
    def __init__(self, state_dim=7, action_dim=2, hidden_dim=128):
        super(PlatformActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))

    def act(self, state):
        action_mean = self.actor(state)
        action_std = torch.exp(self.action_log_std)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_std = torch.exp(self.action_log_std)
        dist = Normal(action_mean, action_std)
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state).squeeze()
        return action_logprobs, state_values, dist_entropy


class PlatformPPOAgent:
    def __init__(self, state_dim=7, action_dim=2, hidden_dim=128):
        self.policy = PlatformActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.policy_old = PlatformActorCritic(state_dim, action_dim, hidden_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.states, self.actions, self.logprobs, self.rewards = [], [], [], []

    def select_actions(self, states):
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states)
            actions, logprobs = self.policy_old.act(states_tensor)

        self.states.append(states_tensor)
        self.actions.append(actions)
        self.logprobs.append(logprobs)

        actions_np = actions.numpy()
        surge = np.clip(actions_np[:, 0] * 2.0 + 3.0, CONFIG['MIN_SURGE'], CONFIG['MAX_SURGE'])
        subsidy = np.clip(actions_np[:, 1] * 10.0 + 10.0, CONFIG['MIN_SUBSIDY'], CONFIG['MAX_SUBSIDY'])
        return surge, subsidy

    def update(self):
        if len(self.states) == 0: return

        old_states = torch.stack(self.states).view(-1, 7)
        old_actions = torch.stack(self.actions).view(-1, 2)
        old_logprobs = torch.stack(self.logprobs).view(-1)

        discounted_rewards = []
        discounted_reward = np.zeros(self.states[0].shape[0])

        for reward_array in reversed(self.rewards):
            discounted_reward = reward_array + 0.99 * discounted_reward
            discounted_rewards.insert(0, discounted_reward.copy())

        rewards_tensor = torch.tensor(np.array(discounted_rewards), dtype=torch.float32).view(-1)

        with torch.no_grad():
            old_state_values = self.policy_old.critic(old_states).squeeze()
            advantages = rewards_tensor - old_state_values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-7)

        for _ in range(10):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * advantages

            loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.MSELoss()(state_values,
                                                                        rewards_tensor) - 0.01 * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.states, self.actions, self.logprobs, self.rewards = [], [], [], []

    def save(self, filepath):
        torch.save(self.policy.state_dict(), filepath)

    def load(self, filepath):
        self.policy.load_state_dict(torch.load(filepath))
        self.policy_old.load_state_dict(self.policy.state_dict())


# --- 2. 桥接机制：附带偏好权重的策略闭包 ---
class RLPlatformPolicy:
    def __init__(self, agent, pref_weights):
        self.agent = agent
        self.weights = pref_weights  # [w_profit, w_efficiency, w_fairness]
        self.current_surge = None
        self.current_subsidy = None
        self.last_t = -1
        self.current_no = None

    def update_actions(self, t, no, nd, sd):
        if t != self.last_t:
            n_zones = len(no)
            t_array = np.full(n_zones, t)

            w1 = np.full(n_zones, self.weights[0])
            w2 = np.full(n_zones, self.weights[1])
            w3 = np.full(n_zones, self.weights[2])

            # 状态拼接：原有 4 维 + 偏好 3 维 = 7 维
            states = np.stack([t_array, no / 50.0, nd / 50.0, sd, w1, w2, w3], axis=-1)

            self.current_surge, self.current_subsidy = self.agent.select_actions(states)
            self.last_t = t
            self.current_no = no

    def surge(self, t, no, nd, sd):
        self.update_actions(t, no, nd, sd)
        return self.current_surge

    def subsidy(self, t, no, nd, sd):
        self.update_actions(t, no, nd, sd)
        return self.current_subsidy


# --- 3. 联合训练循环 (Preference-Conditioned Bi-level MARL) ---
def train_morl(sim_path, num_episodes=200):
    trainer = Trainer(simulator_path=sim_path)
    platform_agent = PlatformPPOAgent()

    # 新增：用于记录训练过程中的真实（未加权）指标
    history_profit, history_efficiency, history_fairness = [], [], []

    print("=== Starting Multi-Objective PC-PPO MARL Training ===")
    for ep in tqdm(range(num_episodes)):
        state = trainer.env.reset()

        # 1. 动态生成偏好向量 (使用 Dirichlet 分布保证和为 1)
        pref_weights = np.random.dirichlet(np.ones(3))

        rl_policy = RLPlatformPolicy(platform_agent, pref_weights)
        platform_params = {'surge': rl_policy.surge, 'subsidy': rl_policy.subsidy}

        ep_total_profit = 0  # 记录本轮总利润

        while True:
            actions = trainer.agent.select_actions(state)
            next_state, rewards, done, info = trainer.env.step(actions, platform_params)

            trainer.agent.buffer.rewards.append(rewards)
            ep_total_profit += info['step_profit']

            zone_profits = info.get('zone_profits', np.zeros(277))
            unfulfilled_penalty = rl_policy.current_no * -0.5 if rl_policy.current_no is not None else np.zeros(277)

            # 2. 根据当前的偏好权重进行动态标量化！
            r_profit = zone_profits / 100.0
            r_efficiency = unfulfilled_penalty

            local_step_reward = (pref_weights[0] * r_profit + pref_weights[1] * r_efficiency)
            platform_agent.rewards.append(local_step_reward)

            state = next_state
            if done:
                gini_index = calculate_gini(info['driver_income'])
                wait_time = -info['total_wait_time']

                # 公平性主要在回合结束体现
                terminal_gini_penalty = -gini_index * 20.0
                platform_agent.rewards[-1] += pref_weights[2] * terminal_gini_penalty

                # 记录真实评估指标
                history_profit.append(ep_total_profit)
                history_efficiency.append(wait_time)
                history_fairness.append(-gini_index)
                break

        trainer.agent.update()
        platform_agent.update()

    convergence_history = (history_profit, history_efficiency, history_fairness)
    return platform_agent, convergence_history


# --- 新增：绘制 MORL 的收敛曲线 ---
def plot_morl_convergence(history, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    profits, efficiencies, fairnesses = history

    # 由于 MORL 每一轮的偏好权重都在随机跳跃，曲线会非常震荡
    # 我们绘制原始点的散点，并叠加一条平滑的移动平均线 (Moving Average)
    def moving_average(data, window_size=20):
        if len(data) < window_size: return data
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    plt.figure(figsize=(15, 5))

    # 1. Platform Profit
    plt.subplot(1, 3, 1)
    plt.plot(profits, color='b', alpha=0.3, label='Raw Episode')
    plt.plot(np.arange(19, len(profits)), moving_average(profits), color='darkblue', linewidth=2,
             label='Moving Avg (20)')
    plt.title('MORL Training: Platform Profit')
    plt.xlabel('Episode')
    plt.ylabel('Profit')
    plt.legend()

    # 2. Efficiency (-Wait Time)
    plt.subplot(1, 3, 2)
    plt.plot(efficiencies, color='g', alpha=0.3, label='Raw Episode')
    plt.plot(np.arange(19, len(efficiencies)), moving_average(efficiencies), color='darkgreen', linewidth=2,
             label='Moving Avg (20)')
    plt.title('MORL Training: Efficiency (-Wait Time)')
    plt.xlabel('Episode')
    plt.legend()

    # 3. Fairness (-Gini Index)
    plt.subplot(1, 3, 3)
    plt.plot(fairnesses, color='r', alpha=0.3, label='Raw Episode')
    plt.plot(np.arange(19, len(fairnesses)), moving_average(fairnesses), color='darkred', linewidth=2,
             label='Moving Avg (20)')
    plt.title('MORL Training: Fairness (-Gini Index)')
    plt.xlabel('Episode')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/morl_convergence.png")
    print(f"\n[Save] MORL Convergence plot saved to {save_dir}/morl_convergence.png")
    plt.show()


# --- 4. 生成帕累托前沿 (Pareto Sweep) ---
def extract_pareto_front(fitnesses, weights_list):
    """简单的非支配排序，找出帕累托前沿"""
    is_efficient = np.ones(fitnesses.shape[0], dtype=bool)
    for i, c in enumerate(fitnesses):
        if is_efficient[i]:
            # 判断是否有其他的点全方位支配点 c (由于都是最大化，其他点 >= c 且至少有一维 > c)
            is_efficient[is_efficient] = np.any(fitnesses[is_efficient] > c, axis=1) | np.all(
                fitnesses[is_efficient] == c, axis=1)
            is_efficient[i] = True  # 保留自己（如果没有被强支配）
            # 反向检查：如果别的点全方位强于 c，则 c 失效
            if np.any(np.all(fitnesses > c, axis=1) & np.any(fitnesses > c, axis=1)):
                is_efficient[i] = False

    return fitnesses[is_efficient], [weights_list[i] for i, valid in enumerate(is_efficient) if valid]


def evaluate_morl_pareto(sim_path, trained_agent):
    print("\n=== Sweeping Preferences to Generate Pareto Front ===")
    trainer = Trainer(simulator_path=sim_path)

    # 构建一组均匀分布的偏好权重进行测试
    test_weights = [
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],  # 极端偏好
        [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],  # 两两均衡
        [0.33, 0.33, 0.34], [0.7, 0.15, 0.15], [0.15, 0.7, 0.15], [0.15, 0.15, 0.7]  # 综合偏好
    ]

    all_fitness = []

    for w in tqdm(test_weights, desc="Evaluating Weights"):
        # 针对每一组特定权重，重置底层司机，让他们适应这种特定偏好的平台策略
        trainer.reset_to_base_weights()
        trainer.agent.optimizer.state.clear()

        rl_policy = RLPlatformPolicy(trained_agent, pref_weights=w)
        platform_params = {'surge': rl_policy.surge, 'subsidy': rl_policy.subsidy}

        # 运行 10 个 episode 让底层收敛
        fitness = trainer.train_and_evaluate(platform_params, num_episodes=10)
        all_fitness.append(fitness[:3])

    all_fitness = np.array(all_fitness)
    pareto_fitness, pareto_weights = extract_pareto_front(all_fitness, test_weights)

    print("\n[MORL Pareto Front Discovered]")
    for i, fit in enumerate(pareto_fitness):
        print(f"Weight: {pareto_weights[i]} -> Profit: {fit[0]:.2f}, Efficiency: {fit[1]:.2f}, Fairness: {fit[2]:.4f}")

    return pareto_fitness, pareto_weights


if __name__ == '__main__':
    sim_path = 'model/generators/simulator_hex_scaling=0.004257843312339327_weekday.pkl'
    if not os.path.exists(sim_path):
        print(f"Simulator file not found at {sim_path}")
    else:
        # 1. 训练具备多目标条件生成的 PC-PPO
        # 修改：接收返回的 convergence_history
        trained_agent, convergence_history = train_morl(sim_path, num_episodes=3000)

        # 2. 绘制收敛图 (由于训练轨迹震荡大，加入了移动平均线)
        plot_morl_convergence(convergence_history)

        # 3. 扫过不同的偏好参数，生成帕累托前沿
        pareto_fitness, pareto_weights = evaluate_morl_pareto(sim_path, trained_agent)

        # 4. 保存结果以供 evaluate_comparison.py 读取
        save_dir = "results_new"
        os.makedirs(save_dir, exist_ok=True)

        # 保存为与 MOGP 和 MODE 相同的格式结构
        results_data = {
            'pareto_fitness': pareto_fitness,
            'parameters': pareto_weights,  # 将偏好权重作为参数记录
            'convergence': convergence_history  # 一并保存训练历史数据
        }

        with open(f"{save_dir}/morl_results.pkl", 'wb') as f:
            pickle.dump(results_data, f)

        trained_agent.save(f"{save_dir}/morl_platform_policy.pth")

        print(f"\nMORL Pareto Front results saved to {save_dir}/morl_results.pkl")