import copy
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler
from tqdm import tqdm
from basic_config import CONFIG
from ride_hailing_env import RideHailingEnv

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
        # Orthogonal Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def act(self, state):
        action_logits = self.actor(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_logits = self.actor(state)
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

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []


class SharedPPOAgent:
    def __init__(self, state_dim, action_dim, **hyperparameters):
        self.policy = ActorCritic(state_dim, action_dim, hyperparameters['HIDDEN_DIM'])
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=hyperparameters['LR_ACTOR'])
        self.policy_old = ActorCritic(state_dim, action_dim, hyperparameters['HIDDEN_DIM'])
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = RolloutBuffer()
        self.MseLoss = nn.MSELoss()

        self.gamma = hyperparameters['GAMMA']
        self.gae_lambda = hyperparameters['GAE_LAMBDA']
        self.ppo_epochs = hyperparameters['PPO_EPOCHS']
        self.batch_size = hyperparameters['BATCH_SIZE']
        self.eps_clip = hyperparameters['EPS_CLIP']
        self.entropy_coef = hyperparameters['ENTROPY_COEF']
        self.max_grad_norm = hyperparameters['MAX_GRAD_NORM']

    def select_actions(self, states):
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states)
            actions, logprobs = self.policy_old.act(states_tensor)
        self.buffer.states.append(states_tensor)
        self.buffer.actions.append(actions)
        self.buffer.logprobs.append(logprobs)
        return actions.numpy()

    def update(self):
        if len(self.buffer.states) == 0: return

        # 整理数据为张量: (Time_Steps, N_Drivers, Dim)
        old_states = torch.stack(self.buffer.states)
        old_actions = torch.stack(self.buffer.actions)
        old_logprobs = torch.stack(self.buffer.logprobs)
        rewards = torch.tensor(np.array(self.buffer.rewards), dtype=torch.float32)

        T, N, _ = old_states.shape

        with torch.no_grad():
            flat_states = old_states.view(-1, CONFIG['STATE_DIM'])
            values = self.policy_old.critic(flat_states).view(T, N)

        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 0.0
                next_value = 0.0
            else:
                next_non_terminal = 1.0
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam

        returns = advantages + values

        flat_states = old_states.view(-1, CONFIG['STATE_DIM'])
        flat_actions = old_actions.view(-1)
        flat_logprobs = old_logprobs.view(-1)
        flat_advantages = advantages.view(-1)
        flat_returns = returns.view(-1)

        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-7)

        dataset_size = flat_states.size(0)
        for _ in range(self.ppo_epochs):
            sampler = BatchSampler(SubsetRandomSampler(range(dataset_size)), self.batch_size, drop_last=False)
            for indices in sampler:
                indices = torch.tensor(indices)
                mb_states = flat_states[indices]
                mb_actions = flat_actions[indices]
                mb_old_logprobs = flat_logprobs[indices]
                mb_advantages = flat_advantages[indices]
                mb_returns = flat_returns[indices]

                logprobs, state_values, dist_entropy = self.policy.evaluate(mb_states, mb_actions)
                state_values = state_values.squeeze()

                ratios = torch.exp(logprobs - mb_old_logprobs)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values,
                                                                     mb_returns) - self.entropy_coef * dist_entropy

                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def get_weights(self):
        return self.policy.state_dict(), self.policy_old.state_dict()

    def load_by_weights(self, weights):
        self.policy.load_state_dict(weights[0])
        self.policy_old.load_state_dict(weights[1])


def calculate_gini(incomes):
    """计算司机收入的基尼系数 (Gini Index)"""
    if len(incomes) == 0:
        return 0.0
    # 确保没有负收入导致计算错误
    incomes = np.clip(np.sort(incomes))
    n = len(incomes)
    index = np.arange(1, n + 1)

    total_income = np.sum(incomes)
    if total_income == 0:
        return 0.0

    # Gini formula
    gini = (2 * np.sum(index * incomes)) / (n * total_income) - (n + 1) / n
    return gini

class Trainer:
    def __init__(self, simulator_path, fixed_scenarios=None):
        self.env = RideHailingEnv(simulator_path, fixed_scenarios=fixed_scenarios)
        self.agent = SharedPPOAgent(CONFIG['STATE_DIM'], CONFIG['ACTION_DIM'], **CONFIG)
        self.save_base_weights()

    def save_base_weights(self):
        """深拷贝保存 PPO 代理的初始状态字典"""
        self.base_policy_state = copy.deepcopy(self.agent.policy.state_dict())
        self.base_policy_old_state = copy.deepcopy(self.agent.policy_old.state_dict())

    def reset_to_base_weights(self):
        """重置底层 PPO 代理的权重，并清空经验池，避免历史污染"""
        self.agent.policy.load_state_dict(self.base_policy_state)
        self.agent.policy_old.load_state_dict(self.base_policy_old_state)
        self.agent.buffer.clear()

    def train_and_evaluate(self, platform_params, num_episodes=5):
        ep_profits = []
        # ep_completion_rates = []
        ep_wait_times = []
        ep_ginis = []  # --- 新增：记录基尼系数 ---

        for _ in tqdm(range(num_episodes)):
            state = self.env.reset()
            ep_total_profit = 0

            while True:
                actions = self.agent.select_actions(state)
                next_state, rewards, done, info = self.env.step(actions, platform_params)
                self.agent.buffer.rewards.append(rewards)
                ep_total_profit += info['step_profit']
                state = next_state
                if done:
                    # total_demand = info['total_generated'] + 1e-6
                    # completion_rate = info['total_served'] / total_demand
                    wait_time = -info['total_wait_time']
                    # --- 新增：计算并记录基尼系数 ---
                    gini_index = calculate_gini(info['driver_income'])
                    ep_profits.append(ep_total_profit)
                    # ep_completion_rates.append(completion_rate)
                    ep_wait_times.append(wait_time)
                    ep_ginis.append(-gini_index)  # 取负值以适应最大化求解器
                    break
            self.agent.update()

        self._plot_rewards(ep_profits,"ep_profits")
        self._plot_rewards(ep_wait_times,"ep_wait_times")
        self._plot_rewards(ep_ginis,"ep_ginis")

        return np.array([
            np.mean(ep_profits),
            np.mean(ep_wait_times),
            np.mean(ep_ginis)
        ])

    def train(self, platform_params, num_episodes=50):
        episode_rewards = []
        for episode in tqdm(range(num_episodes)):
            state = self.env.reset()
            ep_reward = 0
            while True:
                actions = self.agent.select_actions(state)
                next_state, rewards, done, info = self.env.step(actions, platform_params)
                self.agent.buffer.rewards.append(rewards)
                self.agent.buffer.is_terminals.append(done)
                state = next_state
                ep_reward += np.sum(rewards)
                if done: break
            self.agent.update()
            episode_rewards.append(ep_reward)
        self._plot_rewards(episode_rewards)
        return episode_rewards

    def _plot_rewards(self, rewards,title):
        plt.plot(rewards)
        plt.savefig(f"{title}_rewards.png")


if __name__ == '__main__':
    platform_params = {
        # Add 1e-6 to avoid log(0)
        'surge': lambda t, no, nd, sd: 1 + 0.5 * np.log(sd + 1e-6),
        # Use sd directly instead of no/nd, and add 1e-6
        'subsidy': lambda t, no, nd, sd: 1 + 0.5 * np.log(sd + 1e-6) * np.sin(t)
    }

    sim_path = 'model/generators/simulator_hex_scaling=0.004257843312339327_weekday.pkl'
    if not os.path.exists(sim_path):
        print(f"Error: Simulator file not found at {sim_path}")
        print("Please run generate_split_simulators.py first.")
    else:
        trainer = Trainer(simulator_path=sim_path)
        rewards = trainer.train_and_evaluate(platform_params, num_episodes=50)  # Reduced for test
