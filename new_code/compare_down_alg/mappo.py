import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from basic_config import CONFIG

sys.path.append(os.getcwd())


class MAPPORolloutBuffer:
    def __init__(self):
        self.states = []
        self.global_states = []  # 新增：记录全局状态
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.masks = []
        self.active_flags = []

    def clear(self):
        self.states.clear()
        self.global_states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.masks.clear()
        self.active_flags.clear()


class MAPPOActorCritic(nn.Module):
    def __init__(self, local_state_dim, global_state_dim, action_dim, hidden_dim):
        super(MAPPOActorCritic, self).__init__()

        # Critic 接收局部特征与全局特征的拼接 (例如 45 + 555 = 600维)
        critic_input_dim = local_state_dim + global_state_dim
        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dim * 2), nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Actor 仅依赖局部 45 维特征进行快速决策
        self.actor = nn.Sequential(
            nn.Linear(local_state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def act(self, local_state, action_mask):
        action_logits = self.actor(local_state)
        action_logits = action_logits.masked_fill(~action_mask, -1e9)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, local_state, global_state, action, action_mask):
        # Actor 评估概率与熵
        action_logits = self.actor(local_state)
        action_logits = action_logits.masked_fill(~action_mask, -1e9)
        dist = Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        # Critic 评估价值
        critic_input = torch.cat([local_state, global_state], dim=-1)
        state_values = self.critic(critic_input)

        return action_logprobs, state_values, dist_entropy


class MAPPOAgent:
    def __init__(self, local_state_dim, global_state_dim, action_dim, **hyperparameters):
        self.local_dim = local_state_dim
        self.global_dim = global_state_dim

        self.policy = MAPPOActorCritic(local_state_dim, global_state_dim, action_dim, hyperparameters['HIDDEN_DIM'])
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': hyperparameters['LR_ACTOR']},
            {'params': self.policy.critic.parameters(), 'lr': hyperparameters.get('LR_CRITIC', 1e-3)}
        ])
        self.policy_old = MAPPOActorCritic(local_state_dim, global_state_dim, action_dim, hyperparameters['HIDDEN_DIM'])
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = MAPPORolloutBuffer()

        self.ValueLoss = nn.SmoothL1Loss()
        self.gamma = hyperparameters['GAMMA']
        self.gae_lambda = hyperparameters['GAE_LAMBDA']
        self.ppo_epochs = hyperparameters['PPO_EPOCHS']
        self.batch_size = hyperparameters['BATCH_SIZE']
        self.eps_clip = hyperparameters['EPS_CLIP']
        self.entropy_coef = hyperparameters['ENTROPY_COEF']
        self.max_grad_norm = hyperparameters['MAX_GRAD_NORM']

    def select_actions(self, local_states, global_states_batch, action_masks):
        with torch.no_grad():
            states_tensor = torch.FloatTensor(local_states)
            global_tensor = torch.FloatTensor(global_states_batch)
            masks_tensor = torch.BoolTensor(action_masks)
            actions, logprobs = self.policy_old.act(states_tensor, masks_tensor)

        self.buffer.states.append(states_tensor)
        self.buffer.global_states.append(global_tensor)
        self.buffer.actions.append(actions)
        self.buffer.logprobs.append(logprobs)
        self.buffer.masks.append(masks_tensor)
        return actions.numpy()

    def update(self):
        if len(self.buffer.states) == 0: return

        old_states = torch.stack(self.buffer.states)
        old_global_states = torch.stack(self.buffer.global_states)
        old_actions = torch.stack(self.buffer.actions)
        old_logprobs = torch.stack(self.buffer.logprobs)
        old_masks = torch.stack(self.buffer.masks)

        rewards = torch.tensor(np.array(self.buffer.rewards), dtype=torch.float32)
        active_flags = torch.tensor(np.array(self.buffer.active_flags), dtype=torch.float32)

        T, N, _ = old_states.shape

        with torch.no_grad():
            flat_states = old_states.view(-1, self.local_dim)
            flat_globals = old_global_states.view(-1, self.global_dim)
            critic_input = torch.cat([flat_states, flat_globals], dim=-1)
            values = self.policy_old.critic(critic_input).view(T, N)

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # SMDP 折现计算
        for i in range(N):
            active_steps = torch.nonzero(active_flags[:, i]).squeeze(-1).tolist()
            if isinstance(active_steps, int): active_steps = [active_steps]
            if not active_steps: continue

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

        active_states = old_states.view(-1, self.local_dim)[flat_active]
        active_globals = old_global_states.view(-1, self.global_dim)[flat_active]
        active_actions = old_actions.view(-1)[flat_active]
        active_logprobs = old_logprobs.view(-1)[flat_active]
        active_advantages = advantages.view(-1)[flat_active]
        active_returns = returns.view(-1)[flat_active]
        active_masks = old_masks.view(-1, CONFIG['ACTION_DIM'])[flat_active]

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

                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    active_states[mb_idx],
                    active_globals[mb_idx],
                    active_actions[mb_idx],
                    active_masks[mb_idx]
                )
                state_values = state_values.squeeze(-1)

                ratios = torch.exp(logprobs - active_logprobs[mb_idx])
                surr1 = ratios * active_advantages[mb_idx]
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * active_advantages[mb_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * self.ValueLoss(state_values, active_returns[mb_idx])
                loss = actor_loss + critic_loss - self.entropy_coef * dist_entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()