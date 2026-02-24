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


# --- 1. 平台侧的连续动作 PPO 代理 (Leader) ---
class PlatformActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PlatformActorCritic, self).__init__()
        # Actor 输出均值 (通过 Tanh 限制在 [-1, 1])
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
        # 可学习的标准差参数
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

        # 经验回放池
        self.states, self.actions, self.logprobs, self.rewards = [], [], [], []

    def select_actions(self, states):
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states)
            actions, logprobs = self.policy_old.act(states_tensor)

        self.states.append(states_tensor)
        self.actions.append(actions)
        self.logprobs.append(logprobs)

        actions_np = actions.numpy()
        surge = np.clip(actions_np[:, 0] * 2.0 + 3.0, 1.0, 5.0)
        subsidy = np.clip(actions_np[:, 1] * 10.0 + 10.0, 0.0, 20.0)
        return surge, subsidy

    def update(self):
        if len(self.states) == 0: return

        old_states = torch.stack(self.states).view(-1, 4)
        old_actions = torch.stack(self.actions).view(-1, 2)
        old_logprobs = torch.stack(self.logprobs).view(-1)

        # === 核心修复 1：空间信用分配 (按区域独立计算折扣回报) ===
        discounted_rewards = []
        discounted_reward = np.zeros(self.states[0].shape[0])  # 形状为 (N_ZONES,)

        for reward_array in reversed(self.rewards):
            discounted_reward = reward_array + 0.99 * discounted_reward
            discounted_rewards.insert(0, discounted_reward.copy())

        # 将 (T, N_ZONES) 拉平为 T * N_ZONES 维度
        rewards_tensor = torch.tensor(np.array(discounted_rewards), dtype=torch.float32).view(-1)

        # --- FIX: Calculate and normalize advantages ONCE outside the loop ---
        with torch.no_grad():
            old_state_values = self.policy_old.critic(old_states).squeeze()
            advantages = rewards_tensor - old_state_values

        # 标准化 Advantage 和 Return 以稳定训练
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-7)

        for _ in range(10):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # 注意: 这里不再重新计算 Advantage!

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
        """保存上层平台策略网络的权重"""
        torch.save(self.policy.state_dict(), filepath)

    def load(self, filepath):
        """加载上层平台策略网络的权重"""
        self.policy.load_state_dict(torch.load(filepath))
        self.policy_old.load_state_dict(self.policy.state_dict())


# --- 2. 桥接机制：将平台 RL 包装为 lambda 函数供底层调用 ---
class RLPlatformPolicy:
    def __init__(self, agent, preference_weights):
        self.agent = agent
        self.weights = preference_weights
        self.current_surge = None
        self.current_subsidy = None
        self.last_t = -1
        self.current_no = None  # 记录当前步的订单分布

    def update_actions(self, t, no, nd, sd):
        if t != self.last_t:
            n_zones = len(no)
            t_array = np.full(n_zones, t)
            # 将 3 维权重扩展并拼接到每个区域的状态中
            w1 = np.full(n_zones, self.weights[0])
            w2 = np.full(n_zones, self.weights[1])
            w3 = np.full(n_zones, self.weights[2])

            # 现在的 state 是 7 维
            states = np.stack([t_array, no / 50.0, nd / 50.0, sd, w1, w2, w3], axis=-1)

            self.current_surge, self.current_subsidy = self.agent.select_actions(states)
            self.last_t = t
            self.current_no = no  # 暴露给 train 循环用于密集惩罚

    def surge(self, t, no, nd, sd):
        self.update_actions(t, no, nd, sd)
        return self.current_surge

    def subsidy(self, t, no, nd, sd):
        self.update_actions(t, no, nd, sd)
        return self.current_subsidy


# --- 3. 联合训练循环 (Bi-level MARL) ---
def train_marl(sim_path, num_episodes=50):
    trainer = Trainer(simulator_path=sim_path)
    platform_agent = PlatformPPOAgent()

    history_profit, history_efficiency, history_fairness = [], [], []

    print("=== Starting Rescued Bi-level MARL Training ===")
    for ep in tqdm(range(num_episodes)):
        state = trainer.env.reset()

        # 1. 使用 Dirichlet 分布随机采样和为 1 的权重向量
        # 例如: [0.8, 0.1, 0.1] (侧重利润) 或 [0.33, 0.33, 0.33] (均衡)
        pref_weights = np.random.dirichlet(np.ones(3))

        rl_policy = RLPlatformPolicy(platform_agent, pref_weights)

        platform_params = {'surge': rl_policy.surge, 'subsidy': rl_policy.subsidy}
        ep_total_profit = 0

        while True:
            actions = trainer.agent.select_actions(state)
            next_state, rewards, done, info = trainer.env.step(actions, platform_params)

            trainer.agent.buffer.rewards.append(rewards)
            ep_total_profit += info['step_profit']

            # === 核心修复 2：密集的多目标局部奖励 (Dense Local Reward) ===
            zone_profits = info.get('zone_profits', np.zeros(277))

            # 使用未匹配订单量作为效率的实时负面惩罚 (-Wait Time Proxy)
            unfulfilled_penalty = rl_policy.current_no * -0.5 if rl_policy.current_no is not None else np.zeros(277)

            # 获取原生的三个独立奖励向量 (N_ZONES,)
            r_profit = zone_profits / 100.0
            r_efficiency = unfulfilled_penalty  # 负的等待时间代理
            r_fairness = np.zeros(277)  # 过程中通常公平性为0，只有终局有

            # 2. 根据当前 Episode 的随机偏好进行加权组合！
            local_step_reward = (pref_weights[0] * r_profit +
                                 pref_weights[1] * r_efficiency +
                                 pref_weights[2] * r_fairness)

            platform_agent.rewards.append(local_step_reward)
            # ============================================================

            state = next_state
            if done:
                total_demand = info['total_generated'] + 1e-6
                completion_rate = info['total_served'] / total_demand
                wait_time = -info['total_wait_time']
                gini_index = calculate_gini(info['driver_income'])

                # 终局惩罚只保留对公平性 (Gini) 的考核，平摊到最后一个状态的各个区域
                terminal_gini_penalty = -gini_index * 20.0
                platform_agent.rewards[-1] += terminal_gini_penalty

                history_profit.append(ep_total_profit)
                history_efficiency.append(wait_time)
                history_fairness.append(-gini_index)
                break

        trainer.agent.update()
        platform_agent.update()

    return history_profit, history_efficiency, history_fairness, platform_agent


if __name__ == '__main__':
    sim_path = 'model/generators/simulator_hex_scaling=0.004257843312339327_weekday.pkl'
    if not os.path.exists(sim_path):
        print(f"Simulator file not found at {sim_path}")
    else:
        # --- 修改：接收返回的 trained_agent ---
        p_hist, e_hist, f_hist, trained_agent = train_marl(sim_path, num_episodes=1000)

        # 保存结果
        save_dir = "results"
        os.makedirs(save_dir, exist_ok=True)

        results_data = {
            'trajectory_fitness': np.column_stack((p_hist, e_hist, f_hist)),
            'convergence': (p_hist, e_hist, f_hist)
        }

        with open(f"{save_dir}/marl_results.pkl", 'wb') as f:
            pickle.dump(results_data, f)

        print(f"\nMARL Baseline results saved to {save_dir}/marl_results.pkl")

        # --- 新增：保存 PyTorch 模型权重 ---
        model_path = f"{save_dir}/marl_platform_policy.pth"
        trained_agent.save(model_path)
        print(f"Platform RL Model weights saved to {model_path}")