import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from tqdm import tqdm
from basic_config import CONFIG
from shared_ppo import Trainer


class PlatformActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor_mean = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, action_dim), nn.Sigmoid())
        self.actor_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 1))

        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))

    def act(self, state):
        mean = self.actor_mean(state) * 4.0 + 1.0
        std = torch.exp(self.actor_std)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action, log_prob

    def evaluate(self, state, action):
        mean = self.actor_mean(state) * 4.0 + 1.0
        std = torch.exp(self.actor_std)
        dist = Normal(mean, std)
        # action shape: (batch, 1), log_prob shape: (batch, 1)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(state)
        return log_prob, value, entropy


class PlatformPPO:
    def __init__(self):
        self.policy = PlatformActorCritic(3, 1)  # [AvgOrder, AvgDriver, Time]
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.buffer = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'dones': []}
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.eps_clip = 0.2
        self.epochs = 5
        self.mse_loss = nn.MSELoss()

    def update(self):
        if not self.buffer['states']: return

        # Stack inputs
        states = torch.stack(self.buffer['states']).detach()  # (T, 3)
        actions = torch.stack(self.buffer['actions']).detach()  # (T, 1)
        old_logprobs = torch.stack(self.buffer['logprobs']).detach()  # (T)
        rewards = torch.tensor(self.buffer['rewards'], dtype=torch.float32)
        dones = torch.tensor(self.buffer['dones'], dtype=torch.float32)

        with torch.no_grad():
            values = self.policy.critic(states).squeeze()

        # GAE Calculation
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = 0
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values

        # Normalize Advantages
        if advantages.std() > 1e-7:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # PPO Update Loop
        for _ in range(self.epochs):
            # logprobs: (T, 1), values: (T, 1)
            logprobs, curr_values, dist_entropy = self.policy.evaluate(states, actions)

            # Squeeze to match shapes (T)
            curr_values = curr_values.squeeze()
            logprobs = logprobs.squeeze()

            # 重要：由于 old_logprobs 是在 act 时生成的 scalar，stack 后是 (T)
            # 而 evaluate 生成的是 (T, 1)，squeeze 后也是 (T)
            # 确保维度对齐
            if logprobs.shape != old_logprobs.shape:
                logprobs = logprobs.view_as(old_logprobs)

            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # [修复点] policy loss 需要取 mean() 变成标量
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * self.mse_loss(curr_values, returns)
            entropy_loss = - 0.01 * dist_entropy.mean()

            loss = policy_loss + value_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear buffer
        for k in self.buffer: self.buffer[k] = []


class RL_Solver:
    def __init__(self, simulator_path, scenarios=None, episodes=20):
        self.trainer = Trainer(simulator_path, fixed_scenarios=scenarios)
        self.platform_agent = PlatformPPO()
        self.episodes = episodes
        self.history = {'score': []}

    def solve(self):
        for ep in range(self.episodes):
            state = self.trainer.env.reset()
            current_params = {
                'commission': 0.25,
                'lambda': np.ones((CONFIG['TIME_STEPS_PER_DAY'], 277)),
                'subsidy': np.zeros((CONFIG['TIME_STEPS_PER_DAY'], 277))
            }
            ep_score = 0
            acc_score = 0

            while True:
                t = self.trainer.env.time
                if t % 12 == 0 and t < CONFIG['TIME_STEPS_PER_DAY']:
                    if t > 0:
                        self.platform_agent.buffer['rewards'].append(acc_score)
                        self.platform_agent.buffer['dones'].append(False)
                        acc_score = 0

                    global_obs = self.trainer.env.get_global_observation()
                    p_state = torch.FloatTensor([global_obs[0].mean(), global_obs[1].mean(), t / 288.0])

                    action, log_prob = self.platform_agent.policy.act(p_state)
                    surge_val = action.item()
                    current_params['lambda'][t: min(t + 12, 288), :] = surge_val

                    self.platform_agent.buffer['states'].append(p_state)
                    self.platform_agent.buffer['actions'].append(action)
                    self.platform_agent.buffer['logprobs'].append(log_prob)

                actions = self.trainer.agent.select_actions(state)
                next_state, rewards, done, info = self.trainer.env.step(actions, current_params)
                self.trainer.agent.buffer.rewards.append(rewards)

                # 加权单目标
                step_score = info['step_profit'] * 1.0
                acc_score += step_score
                ep_score += step_score

                state = next_state
                if done:
                    self.platform_agent.buffer['rewards'].append(acc_score)
                    self.platform_agent.buffer['dones'].append(True)
                    break

            self.platform_agent.update()
            self.trainer.agent.update()

            self.history['score'].append(ep_score)
            print(f"RL Ep {ep} | Score: {ep_score:.2f}")

        # --- 新增：返回模型权重 ---
        model_artifacts = {
            'platform_policy': self.platform_agent.policy.state_dict(),
            'driver_agent': self.trainer.agent.get_weights()  # 获取 shared_ppo 中定义的 driver 权重
        }
        return self.history, model_artifacts