import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler
import numpy as np
import os

# 引用配置（确保与环境一致）
from env_core import CONFIG

# 检查是否有GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    """
    PPO 策略网络与价值网络
    输入: 司机状态向量 (State Dim)
    输出: 动作概率分布 (Action Dim) + 状态价值 (Value)
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()

        # 1. Actor: 决定动作概率
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # 2. Critic: 评估当前状态好坏
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """推断阶段：根据状态选择动作"""
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        """训练阶段：评估动作概率与状态价值"""
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class SharedPPOAgent:
    """
    下层博弈求解器：全职司机的共享大脑
    """

    def __init__(self, state_dim=3, action_dim=7, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        # 经验回放缓冲区 (Rollout Buffer)
        self.buffer = {
            'states': [],
            'actions': [],
            'logprobs': [],
            'rewards': [],
            'is_terminals': []
        }

    def select_action(self, state, training=True):
        """
        为所有司机批量选择动作
        :param state: np.array shape (N_drivers, State_Dim)
        :param training: bool, 是否在训练模式（影响是否采样/记录数据）
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            if training:
                action, action_logprob = self.policy_old.act(state)
                # 存入 buffer
                self.buffer['states'].append(state)
                self.buffer['actions'].append(action)
                self.buffer['logprobs'].append(action_logprob)
                return action.cpu().numpy()
            else:
                # 推理模式：直接取概率最大的动作 (Deterministic) 或者是采样 (Stochastic)
                # 为了评估稳定性，通常推理时也可以采样，或者取argmax
                # 这里保持一致性使用采样，但不需要梯度
                action_probs = self.policy_old.actor(state)
                return torch.argmax(action_probs, dim=1).cpu().numpy()

    def store_reward(self, rewards, is_terminals):
        """存储一步的奖励"""
        # rewards: list or np.array of shape (N_drivers,)
        self.buffer['rewards'].append(torch.tensor(rewards, dtype=torch.float32).to(device))
        self.buffer['is_terminals'].append(torch.tensor(is_terminals, dtype=torch.bool).to(device))

    def update(self):
        """
        PPO 核心更新逻辑：利用收集到的轨迹更新网络
        """
        # 1. 蒙特卡洛估计回报 (Monte Carlo Estimate of Returns)
        rewards = []
        discounted_reward = 0
        # 反向遍历计算 G_t
        for reward, is_terminal in zip(reversed(self.buffer['rewards']), reversed(self.buffer['is_terminals'])):
            if is_terminal.any():  # 这里简化处理，如果是一批数据的结束
                discounted_reward = 0
            # 注意：这里的 reward 是一个 vector (N_drivers,)
            # 我们对所有司机的奖励取平均作为 batch 的 update 信号?
            # 不，PPO 可以处理 batch 维度的奖励。我们需要保持维度。
            # 为了简化实现，这里假设 buffer 里的每一项都是 Tensor(N_drivers)
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # 归一化奖励 (加快收敛)
        rewards = torch.stack(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 将 buffer 堆叠成 Tensor
        # old_states shape: (Time_Steps, N_Drivers, State_Dim) -> (Batch_Size, State_Dim)
        old_states = torch.stack(self.buffer['states']).view(-1, self.buffer['states'][0].shape[-1]).detach()
        old_actions = torch.stack(self.buffer['actions']).view(-1).detach()
        old_logprobs = torch.stack(self.buffer['logprobs']).view(-1).detach()
        rewards = rewards.view(-1).detach()

        # 2. PPO K次 迭代更新
        for _ in range(self.K_epochs):
            # 评估旧动作在当前新策略下的概率
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = state_values.squeeze()

            # 计算优势函数 (Advantage)
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach()

            # Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # 总 Loss = Policy Loss + Value Loss + Entropy Loss
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # 梯度下降
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 3. 同步旧策略并清空 Buffer
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.clear_buffer()

    def clear_buffer(self):
        self.buffer = {
            'states': [],
            'actions': [],
            'logprobs': [],
            'rewards': [],
            'is_terminals': []
        }

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=device))


# --- 测试代码 ---
if __name__ == "__main__":
    print(">>> Testing Driver Agent Core...")

    # 1. 模拟参数
    N_DRIVERS = 10
    STATE_DIM = 5  # 假设状态维度为5
    ACTION_DIM = 7  # 0:Serve, 1-6:Move

    # 2. 初始化 Agent
    agent = SharedPPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    print("Agent initialized successfully.")

    # 3. 模拟一个时间步的数据
    # 状态: [Lat, Lng, OrderCount, DriverCount, Time]
    dummy_states = np.random.rand(N_DRIVERS, STATE_DIM)

    # 4. 测试 select_action (训练模式)
    actions = agent.select_action(dummy_states, training=True)
    print(f"Selected Actions (Training): {actions}")
    assert actions.shape == (N_DRIVERS,), "Action shape mismatch!"

    # 5. 模拟存储奖励
    dummy_rewards = np.random.rand(N_DRIVERS)
    dummy_dones = np.zeros(N_DRIVERS, dtype=bool)  # 没结束
    agent.store_reward(dummy_rewards, dummy_dones)
    print("Reward stored.")

    # 6. 再模拟几步以填充 buffer
    for _ in range(5):
        s = np.random.rand(N_DRIVERS, STATE_DIM)
        agent.select_action(s, training=True)
        agent.store_reward(np.random.rand(N_DRIVERS), np.zeros(N_DRIVERS, dtype=bool))

    # 7. 测试 update
    try:
        print("Starting PPO update...")
        agent.update()
        print("PPO update finished without errors.")
    except Exception as e:
        print(f"Update failed: {e}")

    # 8. 测试推理模式
    actions_eval = agent.select_action(dummy_states, training=False)
    print(f"Selected Actions (Inference): {actions_eval}")

    print(">>> All tests passed.")