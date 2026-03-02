import copy
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.distributions import Categorical
from torch.utils.data import BatchSampler
from tqdm import tqdm
from basic_config import CONFIG
from ride_hailing_env import RideHailingEnv
from generate_simulator import PassengerSimulator
sys.path.append(os.getcwd())


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def act(self, state, action_mask):
        """注入动作掩码：极大负值屏蔽非法动作"""
        action_logits = self.actor(state)
        # 将 mask 为 False 的位置填充为极小值
        action_logits = action_logits.masked_fill(~action_mask, -1e9)  # 直接使用传入的 Tensor

        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action, action_mask):
        action_logits = self.actor(state)
        action_logits = action_logits.masked_fill(~action_mask, -1e9)
        dist = Categorical(logits=action_logits)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.masks = []  # 记录环境提供的动作掩码
        self.active_flags = []  # 记录司机是否有效 (状态非忙碌且非离线)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.masks.clear()
        self.active_flags.clear()


class SharedPPOAgent:
    def __init__(self, state_dim, action_dim, **hyperparameters):
        self.policy = ActorCritic(state_dim, action_dim, hyperparameters['HIDDEN_DIM'])
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': hyperparameters['LR_ACTOR']},
            {'params': self.policy.critic.parameters(), 'lr': hyperparameters.get('LR_CRITIC', 1e-3)}
        ])
        self.policy_old = ActorCritic(state_dim, action_dim, hyperparameters['HIDDEN_DIM'])
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = RolloutBuffer()

        # 核心改造 3.2: 引入 Huber Loss 防止大额订单带来梯度爆炸
        self.ValueLoss = nn.SmoothL1Loss()

        self.gamma = hyperparameters['GAMMA']
        self.gae_lambda = hyperparameters['GAE_LAMBDA']
        self.ppo_epochs = hyperparameters['PPO_EPOCHS']
        self.batch_size = hyperparameters['BATCH_SIZE']
        self.eps_clip = hyperparameters['EPS_CLIP']
        self.entropy_coef = hyperparameters['ENTROPY_COEF']
        self.max_grad_norm = hyperparameters['MAX_GRAD_NORM']

    def select_actions(self, states, action_masks):
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states)
            masks_tensor = torch.BoolTensor(action_masks) # 提前转换
            actions, logprobs = self.policy_old.act(states_tensor, masks_tensor)

        self.buffer.states.append(states_tensor)
        self.buffer.actions.append(actions)
        self.buffer.logprobs.append(logprobs)
        self.buffer.masks.append(masks_tensor) # 直接存 Tensor
        return actions.numpy()

    def update(self):
        if len(self.buffer.states) == 0: return

        old_states = torch.stack(self.buffer.states)
        old_actions = torch.stack(self.buffer.actions)
        old_logprobs = torch.stack(self.buffer.logprobs)
        old_masks = torch.stack(self.buffer.masks) # 修复报错的关键：保持为 Tensor

        rewards = torch.tensor(np.array(self.buffer.rewards), dtype=torch.float32)
        active_flags = torch.tensor(np.array(self.buffer.active_flags), dtype=torch.float32)

        T, N, _ = old_states.shape

        with torch.no_grad():
            flat_states = old_states.view(-1, CONFIG['STATE_DIM'])
            values = self.policy_old.critic(flat_states).view(T, N)

        advantages = torch.zeros_like(rewards)
        last_gae_lam = torch.zeros(N)

        # 核心修复：移除 next_active 的强行截断，保留标准 MDP 的价值传导
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = torch.zeros(N)
            else:
                next_value = values[t + 1]

            # 此时 next_value 代表时间自然流逝后的价值评估。
            # 无论司机此时是否 active，他接单时的决策价值都应包含通往目的地的长线收益。
            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam
            advantages[t] = last_gae_lam

        returns = advantages + values

        # 数据展平
        flat_states = old_states.view(-1, CONFIG['STATE_DIM'])
        flat_actions = old_actions.view(-1)
        flat_logprobs = old_logprobs.view(-1)
        flat_advantages = advantages.view(-1)
        flat_returns = returns.view(-1)
        flat_masks = old_masks.reshape(-1, CONFIG['ACTION_DIM'])
        flat_active = active_flags.view(-1).bool()

        # 仅对 Active（在场做决策）的样本标准化与训练，避免脏数据污染
        active_adv = flat_advantages[flat_active]
        if active_adv.shape[0] > 1:
            mean_adv = active_adv.mean()
            std_adv = active_adv.std() + 1e-7
            flat_advantages[flat_active] = (active_adv - mean_adv) / std_adv

        # 动态衰减探索因子，保证后期收敛平滑
        self.entropy_coef = max(0.001, self.entropy_coef * 0.99)

        # 1. 找到所有有效的 (t, n) 索引
        valid_indices = torch.nonzero(flat_active).squeeze()
        if valid_indices.numel() == 0:
            self.buffer.clear()
            return

        for _ in range(self.ppo_epochs):
            # 核心改造 3.3: 保证同司机序列采样的相对连续性
            grouped_indices = []
            for i in range(N):
                # 获取司机 i 在所有时间步的有效索引
                driver_valid_steps = torch.nonzero(active_flags[:, i]).squeeze(-1)
                if driver_valid_steps.numel() > 0:
                    flat_idxs = (driver_valid_steps * N + i).tolist()
                    if isinstance(flat_idxs, list):
                        grouped_indices.extend(flat_idxs)
                    else:
                        grouped_indices.append(flat_idxs)

            # 按 Batch Size 顺序切分，最大程度保留时间步依赖
            for i in range(0, len(grouped_indices), self.batch_size):
                batch_idx = grouped_indices[i:i + self.batch_size]
                if len(batch_idx) < 2: continue

                mb_states = flat_states[batch_idx]
                mb_actions = flat_actions[batch_idx]
                mb_old_logprobs = flat_logprobs[batch_idx]
                mb_advantages = flat_advantages[batch_idx]
                mb_returns = flat_returns[batch_idx]
                mb_masks = flat_masks[batch_idx]

                logprobs, state_values, dist_entropy = self.policy.evaluate(mb_states, mb_actions, mb_masks)
                state_values = state_values.squeeze()

                ratios = torch.exp(logprobs - mb_old_logprobs)
                surr1 = ratios * mb_advantages

                # 动态自适应 Clip，受 KL 散度约束的思想
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * self.ValueLoss(state_values, mb_returns)

                loss = actor_loss + critic_loss - self.entropy_coef * dist_entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()


def calculate_gini(incomes):
    if len(incomes) == 0: return 0.0
    incomes = np.sort(np.clip(incomes, 0, None))
    n = len(incomes)
    index = np.arange(1, n + 1)
    total_income = np.sum(incomes)
    if total_income == 0: return 0.0
    return (2 * np.sum(index * incomes)) / (n * total_income) - (n + 1) / n


class Trainer:
    def __init__(self, simulator_path):
        self.env = RideHailingEnv(simulator_path)
        self.agent = SharedPPOAgent(CONFIG['STATE_DIM'], CONFIG['ACTION_DIM'], **CONFIG)
        self.save_base_weights()

    def save_base_weights(self):
        self.base_policy_state = copy.deepcopy(self.agent.policy.state_dict())
        self.base_policy_old_state = copy.deepcopy(self.agent.policy_old.state_dict())

    def train_and_evaluate(self, platform_params, num_episodes=5):
        ep_profits = []
        ep_completion_rates = []
        ep_wait_times = []
        ep_ginis = []

        for ep in tqdm(range(num_episodes)):
            state, action_mask = self.env.reset()
            ep_total_profit = 0

            while True:
                # 【修复核心 1】：在执行环境 step 之前，记录当前时刻司机是否真正空闲(有决策权)
                current_active_flags = (self.env.driver_status == 0)

                actions = self.agent.select_actions(state, action_mask)

                # ---------------- 替换原有的 critic_estimator ----------------
                def critic_estimator_batch(d_ids, dest_idxs):
                    """批量计算一组司机和目标地点的未来预估价值"""
                    if not d_ids: return np.array([])
                    # 批量获取伪状态
                    proxy_states = [self.env.get_proxy_state(d, dest) for d, dest in zip(d_ids, dest_idxs)]
                    state_tensor = torch.FloatTensor(np.array(proxy_states))
                    with torch.no_grad():
                        # 一次性完成前向传播，并转回 numpy 数组
                        return self.agent.policy_old.critic(state_tensor).squeeze(-1).numpy()

                # -------------------------------------------------------------

                next_state, rewards, done, info = self.env.step(
                    actions, platform_params,
                    value_estimator=critic_estimator_batch  # 传入新的批量预估器
                )

                self.agent.buffer.rewards.append(rewards)

                # 【修复核心 1】：录入 step 开始前的状态，确保成功接单的动作被保留！
                self.agent.buffer.active_flags.append(current_active_flags)

                ep_total_profit += info['step_profit']
                state = next_state
                action_mask = info['action_mask']

                if done:
                    total_demand = info['total_generated'] + 1e-6
                    completion_rate = info['total_served'] / total_demand
                    wait_time = -info['total_wait_time']
                    gini_index = calculate_gini(info['driver_income_rate'])

                    ep_profits.append(ep_total_profit)
                    ep_completion_rates.append(completion_rate)
                    ep_wait_times.append(wait_time)
                    ep_ginis.append(-gini_index)

                    # CSV日志记录 (核心改造 4.3)
                    with open("training_log.csv", "a") as f:
                        f.write(f"{ep},{ep_total_profit:.2f},{completion_rate:.4f},{wait_time:.2f},{gini_index:.4f}\n")
                    break

            self.agent.update()
        self.save()
        self._plot_rewards({"profits": ep_profits, "wait_times": ep_wait_times, "ginis": ep_ginis,
                            "completion_rates": ep_completion_rates})
        k = min(5, len(ep_profits))
        return np.array([np.mean(ep_profits[-k:]), np.mean(ep_wait_times[-k:]), np.mean(ep_ginis[-k:])])

    def _plot_rewards(self, kpis):
        fig = plt.figure(figsize=(24, 6))
        axs = [fig.add_subplot(141), fig.add_subplot(142), fig.add_subplot(143), fig.add_subplot(144)]
        for idx, (title, kpi) in enumerate(kpis.items()):
            axs[idx].plot(kpi)
            axs[idx].grid(True, alpha=0.3)
            axs[idx].set_xlabel('Episode')
            axs[idx].set_ylabel(title)
        plt.tight_layout()
        plt.savefig(f"kpi_curve.png")
        plt.close()

    def save(self):
        # --- 【新增】训练结束后保存模型权重 ---
        print("Saving model weights to 'ppo_agent_final.pth'...")
        torch.save(self.agent.policy.state_dict(), "ppo_agent_final.pth")
        print("Model saved successfully.")


if __name__ == '__main__':
    # 全局固化随机种子保证复现性
    torch.manual_seed(42)
    np.random.seed(42)

    # 初始化日志表头
    if not os.path.exists("training_log.csv"):
        with open("training_log.csv", "w") as f:
            f.write("Episode,Profit,CompletionRate,WaitTime,Gini\n")

    platform_params = {
        'surge': lambda t, no, nd, sd: 1.0 + 1.2 * np.maximum(0, sd - 0.8) ** 1.2,
        'subsidy': lambda t, no, nd, sd: 3.0 * (np.exp(-50 * (t - 0.30) ** 2) + np.exp(-40 * (t - 0.75) ** 2)) * np.maximum(0, sd - 0.5)
    }

    sim_path = 'generator/simulator_driver_nums=400_hex_scaling=0.017031373249357308_weekday.pkl'
    if not os.path.exists(sim_path):
        print(f"Error: Simulator file not found at {sim_path}")
    else:

        trainer = Trainer(simulator_path=sim_path)
        rewards = trainer.train_and_evaluate(platform_params, num_episodes=40)