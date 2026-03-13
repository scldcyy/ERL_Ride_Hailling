import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from basic_config import CONFIG
from shared_ppo import RolloutBuffer, SharedPPOAgent
from generate_simulator import PassengerSimulator
sys.path.append(os.getcwd())


class AttentionActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(AttentionActorCritic, self).__init__()

        # 1. 空间特征提取 (7个节点，每个节点6维特征)
        self.node_embed = nn.Linear(6, 32)

        # ---------------- 引入可学习的位置编码 ----------------
        # 形状: (1, 7, 32)，对应 1个中心 + 6个方向的邻居
        self.pos_embed = nn.Parameter(torch.randn(1, 7, 32) * 0.02)
        # -------------------------------------------------------------

        # 2. 多头注意力机制
        self.attn = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)

        # 3. 将注意力输出 (7 * 32 = 224) 与个人特征 (3) 拼接，总维度 227
        combined_dim = 7 * 32 + 3

        self.critic = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)  # 修正为输出 action_dim
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def _process_state(self, state):
        # 分离空间特征与个人特征
        # 空间特征: 前 42 维 (7 个六边形 * 6 个特征)
        spatial_state = state[:, :42].view(-1, 7, 6)
        personal_state = state[:, 42:]

        # 节点特征映射: (Batch, 7, 32)
        node_embeddings = self.node_embed(spatial_state)

        # ---------------- 核心修复：将位置信息注入节点特征 ----------------
        # 广播机制会自动将 (1, 7, 32) 加到每一个 Batch 样本上
        node_embeddings = node_embeddings + self.pos_embed
        # -------------------------------------------------------------

        # 注意力计算
        attn_out, _ = self.attn(node_embeddings, node_embeddings, node_embeddings)

        # 展平并拼接个人特征
        flattened_attn = attn_out.reshape(-1, 7 * 32)
        return torch.cat([flattened_attn, personal_state], dim=1)

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


class AttentionPPOAgent:
    # 这里的实现与 shared_ppo.py 中的 SharedPPOAgent 几乎一致
    # 唯一的区别是在初始化时使用了 AttentionActorCritic
    def __init__(self, state_dim, action_dim, **hyperparameters):
        self.policy = AttentionActorCritic(state_dim, action_dim, hyperparameters['HIDDEN_DIM'])
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': hyperparameters['LR_ACTOR']},
            {'params': self.policy.critic.parameters(), 'lr': hyperparameters.get('LR_CRITIC', 1e-3)}
        ])
        self.policy_old = AttentionActorCritic(state_dim, action_dim, hyperparameters['HIDDEN_DIM'])
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
            flat_states = old_states.view(-1, self.policy.actor[0].in_features if not hasattr(self.policy, '_process_state') else 45)
            # 修复点：先将状态通过 _process_state 转换为 注意力特征 (227维)，再送入 critic
            features = self.policy_old._process_state(flat_states)
            values = self.policy_old.critic(features).view(T, N)

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # SMDP 的折现逻辑与基线完全一致
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