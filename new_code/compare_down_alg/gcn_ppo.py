import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from basic_config import CONFIG
from shared_ppo import RolloutBuffer
from generate_simulator import PassengerSimulator
sys.path.append(os.getcwd())


class GraphConvLayer(nn.Module):
    """基础的图卷积层实现"""

    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.orthogonal_(self.linear.weight, gain=np.sqrt(2))

    def forward(self, x, adj):
        # x shape: (Batch, 7, in_features)
        # adj shape: (7, 7)
        support = self.linear(x)
        # 矩阵乘法进行邻居特征聚合
        out = torch.matmul(adj, support) + self.bias
        return torch.relu(out)


class GCNActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(GCNActorCritic, self).__init__()

        # 1. 预计算局部六边形邻接矩阵 (7个节点)
        A = np.zeros((7, 7), dtype=np.float32)
        # 中心节点 (0) 连接所有邻居 (1-6)
        A[0, 1:] = 1.0
        A[1:, 0] = 1.0
        # 邻居节点首尾相连成环
        for i in range(1, 7):
            next_i = i + 1 if i < 6 else 1
            A[i, next_i] = 1.0
            A[next_i, i] = 1.0
        # 添加自环
        A = A + np.eye(7, dtype=np.float32)
        # 计算度矩阵并归一化: D^{-1/2} A D^{-1/2}
        D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1)))
        A_hat = D_inv_sqrt @ A @ D_inv_sqrt

        # 将归一化后的邻接矩阵注册为不需要求导的 buffer
        self.register_buffer('adj', torch.FloatTensor(A_hat))

        # 2. 定义 GCN 层 (输入每个节点 6 维特征)
        self.gcn1 = GraphConvLayer(6, 32)
        self.gcn2 = GraphConvLayer(32, 32)

        # 3. 拼接层：7个节点 * 32维 + 3维个人特征 = 227维
        combined_dim = 7 * 32 + 3

        self.critic = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _process_state(self, state):
        # 前 42 维为空间特征，重塑为 (Batch, 7, 6)
        spatial_state = state[:, :42].view(-1, 7, 6)
        personal_state = state[:, 42:]

        # GCN 消息传递
        x = self.gcn1(spatial_state, self.adj)
        x = self.gcn2(x, self.adj)

        # 展平所有节点的表征
        flattened_gcn = x.view(-1, 7 * 32)
        return torch.cat([flattened_gcn, personal_state], dim=1)

    def act(self, state, action_mask):
        features = self._process_state(state)
        action_logits = self.actor(features)
        action_logits = action_logits.masked_fill(~action_mask, -1e9)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action, action_mask):
        features = self._process_state(state)
        action_logits = self.actor(features)
        action_logits = action_logits.masked_fill(~action_mask, -1e9)
        dist = Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(features)
        return action_logprobs, state_values, dist_entropy


class GCNPPOAgent:
    def __init__(self, state_dim, action_dim, **hyperparameters):
        self.policy = GCNActorCritic(state_dim, action_dim, hyperparameters['HIDDEN_DIM'])
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': hyperparameters['LR_ACTOR']},
            {'params': self.policy.critic.parameters(), 'lr': hyperparameters.get('LR_CRITIC', 1e-3)}
        ])
        self.policy_old = GCNActorCritic(state_dim, action_dim, hyperparameters['HIDDEN_DIM'])
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
            flat_states = old_states.view(-1, 45)
            features = self.policy_old._process_state(flat_states)
            values = self.policy_old.critic(features).view(T, N)

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        for i in range(N):
            active_steps = torch.nonzero(active_flags[:, i]).squeeze(-1).tolist()
            if isinstance(active_steps, int):
                active_steps = [active_steps]
            if not active_steps:
                continue

            last_gae_lam = 0.0
            for k in reversed(range(len(active_steps))):
                curr_t = active_steps[k]
                if k == len(active_steps) - 1:
                    next_value = 0.0
                    delta_t = 1
                else:
                    next_t = active_steps[k + 1]
                    next_value = values[next_t, i]
                    delta_t = next_t - curr_t

                gamma_smdp = self.gamma ** delta_t
                delta = rewards[curr_t, i] + gamma_smdp * next_value - values[curr_t, i]

                last_gae_lam = delta + gamma_smdp * self.gae_lambda * last_gae_lam
                advantages[curr_t, i] = last_gae_lam
                returns[curr_t, i] = advantages[curr_t, i] + values[curr_t, i]

        flat_active = active_flags.view(-1).bool()

        active_states = old_states.view(-1, 45)[flat_active]
        active_actions = old_actions.view(-1)[flat_active]
        active_logprobs = old_logprobs.view(-1)[flat_active]
        active_advantages = advantages.view(-1)[flat_active]
        active_returns = returns.view(-1)[flat_active]
        active_masks = old_masks.view(-1, 8)[flat_active]

        if active_states.shape[0] == 0:
            self.buffer.clear()
            return

        if active_advantages.shape[0] > 1:
            mean_adv = active_advantages.mean()
            std_adv = active_advantages.std() + 1e-7
            active_advantages = (active_advantages - mean_adv) / std_adv

        self.entropy_coef = max(0.001, self.entropy_coef * 0.99)
        dataset_size = active_states.shape[0]
        indices = np.arange(dataset_size)

        for _ in range(self.ppo_epochs):
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
                critic_loss = 0.5 * self.ValueLoss(state_values, mb_returns)

                loss = actor_loss + critic_loss - self.entropy_coef * dist_entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()