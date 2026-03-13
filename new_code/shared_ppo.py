import copy
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.distributions import Categorical
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
        action_logits = self.actor(state)
        action_logits = action_logits.masked_fill(~action_mask, -1e9)
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
        self.masks = []
        self.active_flags = []

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
            masks_tensor = torch.BoolTensor(action_masks)
            actions, logprobs = self.policy_old.act(states_tensor, masks_tensor)

        self.buffer.states.append(states_tensor)
        self.buffer.actions.append(actions)
        self.buffer.logprobs.append(logprobs)
        self.buffer.masks.append(masks_tensor)
        return actions.numpy()

    def update(self):
        if len(self.buffer.states) == 0: return

        old_states = torch.stack(self.buffer.states)
        old_actions = torch.stack(self.buffer.actions)
        old_logprobs = torch.stack(self.buffer.logprobs)
        old_masks = torch.stack(self.buffer.masks)

        rewards = torch.tensor(np.array(self.buffer.rewards), dtype=torch.float32)
        active_flags = torch.tensor(np.array(self.buffer.active_flags), dtype=torch.float32)

        T, N, _ = old_states.shape

        with torch.no_grad():
            flat_states = old_states.view(-1, CONFIG['STATE_DIM'])
            # 先计算所有的 Value，后续我们只提取 Active 的部分
            values = self.policy_old.critic(flat_states).view(T, N)

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # ---------------- 核心修复：基于 SMDP 的按司机轨迹重构与时间跳跃折现 ----------------
        for i in range(N):
            # 提取当前司机所有有决策权（Active）的时间步索引
            active_steps = torch.nonzero(active_flags[:, i]).squeeze(-1).tolist()
            if isinstance(active_steps, int):
                active_steps = [active_steps]
            if not active_steps:
                continue

            last_gae_lam = 0.0
            # 仅在有效的决策点之间进行反向遍历回溯
            for k in reversed(range(len(active_steps))):
                curr_t = active_steps[k]

                if k == len(active_steps) - 1:
                    next_value = 0.0
                    delta_t = 1
                else:
                    next_t = active_steps[k + 1]
                    next_value = values[next_t, i]
                    # 计算两个决策点之间跨越了多少个环境时间步
                    delta_t = next_t - curr_t

                # SMDP 的折现公式：gamma 需要根据跨越的时间步数进行指数级衰减
                gamma_smdp = self.gamma ** delta_t

                # 计算 TD 误差。此时的 rewards[curr_t, i] 正好包含了在 curr_t 做决策后拿到的所有收益（行程总费用）
                delta = rewards[curr_t, i] + gamma_smdp * next_value - values[curr_t, i]

                last_gae_lam = delta + gamma_smdp * self.gae_lambda * last_gae_lam
                advantages[curr_t, i] = last_gae_lam
                returns[curr_t, i] = advantages[curr_t, i] + values[curr_t, i]
        # -------------------------------------------------------------------------

        # 展平所有数据
        flat_active = active_flags.view(-1).bool()

        # ---------------- 过滤脏数据：Actor 和 Critic 都只看 Active 的状态 ----------------
        active_states = old_states.view(-1, CONFIG['STATE_DIM'])[flat_active]
        active_actions = old_actions.view(-1)[flat_active]
        active_logprobs = old_logprobs.view(-1)[flat_active]
        active_advantages = advantages.view(-1)[flat_active]
        active_returns = returns.view(-1)[flat_active]
        active_masks = old_masks.view(-1, CONFIG['ACTION_DIM'])[flat_active]

        if active_states.shape[0] == 0:
            self.buffer.clear()
            return

        # 优势值标准化 (仅基于有效数据)
        if active_advantages.shape[0] > 1:
            mean_adv = active_advantages.mean()
            std_adv = active_advantages.std() + 1e-7
            active_advantages = (active_advantages - mean_adv) / std_adv

        self.entropy_coef = max(0.001, self.entropy_coef * 0.99)
        dataset_size = active_states.shape[0]
        indices = np.arange(dataset_size)

        for _ in range(self.ppo_epochs):
            # 在 SMDP 重构且截断了时序污染后，我们可以安全地打乱 batch
            np.random.shuffle(indices)

            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = start_idx + self.batch_size
                mb_idx = indices[start_idx:end_idx]

                mb_states = active_states[mb_idx]
                mb_actions = active_actions[mb_idx]
                mb_old_logprobs = active_logprobs[mb_idx]
                mb_advantages = active_advantages[mb_idx]
                mb_returns = active_returns[mb_idx]
                mb_masks = active_masks[mb_idx]

                logprobs, state_values, dist_entropy = self.policy.evaluate(mb_states, mb_actions, mb_masks)
                state_values = state_values.squeeze(-1)

                ratios = torch.exp(logprobs - mb_old_logprobs)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                # Critic 现在的 Loss 也是百分百纯净的，不再拟合包含 0 奖励的忙碌期
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

    def reset_to_base_weights(self):
        """重置底层 PPO 代理的权重，并清空经验池，避免历史污染"""
        self.agent.policy.load_state_dict(self.base_policy_state)
        self.agent.policy_old.load_state_dict(self.base_policy_old_state)
        self.agent.buffer.clear()

    def train_and_evaluate(self, platform_params, num_episodes=5,show_fig=False):
        ep_profits = []
        ep_completion_rates = []
        ep_wait_times = []
        ep_ginis = []

        for ep in tqdm(range(num_episodes)):
            state, action_mask = self.env.reset()
            ep_total_profit = 0

            while True:
                current_active_flags = (self.env.driver_status == 0)
                actions = self.agent.select_actions(state, action_mask)

                def critic_estimator_batch(d_ids, dest_idxs):
                    if not d_ids: return np.array([])
                    proxy_states = [self.env.get_proxy_state(d, dest) for d, dest in zip(d_ids, dest_idxs)]
                    state_tensor = torch.FloatTensor(np.array(proxy_states))
                    with torch.no_grad():
                        return self.agent.policy_old.critic(state_tensor).squeeze(-1).numpy()

                next_state, rewards, done, info = self.env.step(
                    actions, platform_params,
                    value_estimator=critic_estimator_batch
                )

                self.agent.buffer.rewards.append(rewards)
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

                    with open("training_log.csv", "a") as f:
                        f.write(f"{ep},{ep_total_profit:.2f},{completion_rate:.4f},{wait_time:.2f},{gini_index:.4f}\n")
                    break

            self.agent.update()
        self.save()
        if show_fig:
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
        print("Saving model weights to 'ppo_agent_final.pth'...")
        torch.save(self.agent.policy.state_dict(), "compare_down_alg/ppo_agent_final.pth")
        print("Model saved successfully.")


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    if not os.path.exists("training_log.csv"):
        with open("training_log.csv", "w") as f:
            f.write("Episode,Profit,CompletionRate,WaitTime,Gini\n")

    platform_params = {
        'surge': lambda t, no, nd, sd: 1.0 + 1.2 * np.maximum(0, sd - 0.8) ** 1.2,
        'subsidy': lambda t, no, nd, sd: 3.0 * (
                    np.exp(-50 * (t - 0.30) ** 2) + np.exp(-40 * (t - 0.75) ** 2)) * np.maximum(0, sd - 0.5)
    }

    sim_path = 'generator/simulator_driver_nums=400_hex_scaling=0.017031373249357308_weekday.pkl'
    if not os.path.exists(sim_path):
        print(f"Error: Simulator file not found at {sim_path}")
    else:
        trainer = Trainer(simulator_path=sim_path)
        rewards = trainer.train_and_evaluate(platform_params, num_episodes=40, show_fig=True)