import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from matplotlib import pyplot as plt
from tqdm import tqdm

# 导入你现有的基础组件
from basic_config import CONFIG
from ride_hailing_env import RideHailingEnv
from generate_simulator import PassengerSimulator
from shared_ppo import SharedPPOAgent


# ==========================================
# 1. 环境热修补：注入全局“上帝视角”观测能力
# ==========================================
def get_global_state(self):
    """提取全城宏观状态：全城订单分布 + 全城空闲司机分布 + 时间进度"""
    order_counts = np.zeros(self.n_zones)
    for o in self.pending_orders:
        if not o['matched']:
            order_counts[o['origin_idx']] += 1

    idle_mask = self.driver_status == 0
    idle_driver_counts = np.bincount(self.driver_locations[idle_mask], minlength=self.n_zones)
    time_prog = self.time / CONFIG['TIME_STEPS_PER_DAY']

    # 除以 50.0 仅为简单的归一化手段
    return np.concatenate([order_counts / 50.0, idle_driver_counts / 50.0, [time_prog]])


# 动态绑定方法到现有的 RideHailingEnv 类
RideHailingEnv.get_global_state = get_global_state


# ==========================================
# 2. MAPPO 核心类定义：上帝视角 Critic
# ==========================================
class MAPPORolloutBuffer:
    def __init__(self):
        self.states = []
        self.global_states = []  # 记录宏观状态
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.masks = []
        self.active_flags = []

    def clear(self):
        self.states.clear();
        self.global_states.clear();
        self.actions.clear()
        self.logprobs.clear();
        self.rewards.clear();
        self.dones.clear()
        self.masks.clear();
        self.active_flags.clear()


class MAPPOActorCritic(nn.Module):
    def __init__(self, local_state_dim, global_state_dim, action_dim, hidden_dim):
        super(MAPPOActorCritic, self).__init__()
        critic_input_dim = local_state_dim + global_state_dim

        # Critic：输入维度 = 45(局部) + 555(全局) = 600维
        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dim * 2), nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Actor：只看局部 45维 特征
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
        return action.detach(), dist.log_prob(action).detach()

    def evaluate(self, local_state, global_state, action, action_mask):
        action_logits = self.actor(local_state)
        action_logits = action_logits.masked_fill(~action_mask, -1e9)
        dist = Categorical(logits=action_logits)
        critic_input = torch.cat([local_state, global_state], dim=-1)
        return dist.log_prob(action), self.critic(critic_input), dist.entropy()


class MAPPOAgent:
    def __init__(self, local_state_dim, global_state_dim, action_dim, **hyperparams):
        self.local_dim = local_state_dim
        self.global_dim = global_state_dim
        self.policy = MAPPOActorCritic(local_state_dim, global_state_dim, action_dim, hyperparams['HIDDEN_DIM'])
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': hyperparams['LR_ACTOR']},
            {'params': self.policy.critic.parameters(), 'lr': hyperparams.get('LR_CRITIC', 1e-3)}
        ])
        self.policy_old = MAPPOActorCritic(local_state_dim, global_state_dim, action_dim, hyperparams['HIDDEN_DIM'])
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = MAPPORolloutBuffer()
        self.ValueLoss = nn.SmoothL1Loss()
        self.gamma = hyperparams['GAMMA'];
        self.gae_lambda = hyperparams['GAE_LAMBDA']
        self.ppo_epochs = hyperparams['PPO_EPOCHS'];
        self.batch_size = hyperparams['BATCH_SIZE']
        self.eps_clip = hyperparams['EPS_CLIP'];
        self.entropy_coef = hyperparams['ENTROPY_COEF']
        self.max_grad_norm = hyperparams['MAX_GRAD_NORM']

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

        for i in range(N):
            active_steps = torch.nonzero(active_flags[:, i]).squeeze(-1).tolist()
            if isinstance(active_steps, int): active_steps = [active_steps]
            if not active_steps: continue
            last_gae_lam = 0.0
            for k in reversed(range(len(active_steps))):
                curr_t = active_steps[k]
                if k == len(active_steps) - 1:
                    next_value = 0.0;
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
            self.buffer.clear();
            return

        if active_advantages.shape[0] > 1:
            active_advantages = (active_advantages - active_advantages.mean()) / (active_advantages.std() + 1e-7)

        self.entropy_coef = max(0.001, self.entropy_coef * 0.99)
        dataset_size = active_states.shape[0]
        indices = np.arange(dataset_size)

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = start_idx + self.batch_size
                mb_idx = indices[start_idx:end_idx]
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    active_states[mb_idx], active_globals[mb_idx], active_actions[mb_idx], active_masks[mb_idx])
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


# ==========================================
# 3. 辅助计算与通用 Episode 执行逻辑
# ==========================================
def calculate_gini(incomes):
    if len(incomes) == 0: return 0.0
    incomes = np.sort(np.clip(incomes, 0, None))
    n = len(incomes)
    index = np.arange(1, n + 1)
    total_income = np.sum(incomes)
    if total_income == 0: return 0.0
    return (2 * np.sum(index * incomes)) / (n * total_income) - (n + 1) / n


def run_episode(env, agent, platform_params, is_mappo, is_eval):
    """通用 Episode 执行函数，兼容 MLP 和 MAPPO，兼容 训练 和 测试"""
    state, action_mask = env.reset()
    ep_total_profit = 0

    while True:
        current_active_flags = (env.driver_status == 0)

        # MAPPO 逻辑分支
        if is_mappo:
            global_state = env.get_global_state()
            global_states_batch = np.tile(global_state, (CONFIG['N_DRIVERS'], 1))

            if is_eval:
                with torch.no_grad():
                    actions, _ = agent.policy.act(torch.FloatTensor(state), torch.BoolTensor(action_mask))
                    actions = actions.numpy()
            else:
                actions = agent.select_actions(state, global_states_batch, action_mask)

            def critic_estimator_batch(d_ids, dest_idxs):
                if not d_ids: return np.array([])
                proxy_states = [env.get_proxy_state(d, dest) for d, dest in zip(d_ids, dest_idxs)]
                local_tensor = torch.FloatTensor(np.array(proxy_states))
                global_tensor = torch.FloatTensor(np.tile(global_state, (len(d_ids), 1)))
                critic_input = torch.cat([local_tensor, global_tensor], dim=-1)
                with torch.no_grad():
                    # 测试时使用最新的 policy 网络，训练时沿用老网络以规避 off-policy 污染
                    network = agent.policy if is_eval else agent.policy_old
                    return network.critic(critic_input).squeeze(-1).numpy()

        # 传统 MLP 逻辑分支
        else:
            if is_eval:
                with torch.no_grad():
                    actions, _ = agent.policy.act(torch.FloatTensor(state), torch.BoolTensor(action_mask))
                    actions = actions.numpy()
            else:
                actions = agent.select_actions(state, action_mask)

            def critic_estimator_batch(d_ids, dest_idxs):
                if not d_ids: return np.array([])
                proxy_states = [env.get_proxy_state(d, dest) for d, dest in zip(d_ids, dest_idxs)]
                state_tensor = torch.FloatTensor(np.array(proxy_states))
                with torch.no_grad():
                    network = agent.policy if is_eval else agent.policy_old
                    return network.critic(state_tensor).squeeze(-1).numpy()

        # 环境步进
        next_state, rewards, done, info = env.step(actions, platform_params, value_estimator=critic_estimator_batch)

        # 如果是训练模式，收集 Replay Buffer
        if not is_eval:
            agent.buffer.rewards.append(rewards)
            agent.buffer.active_flags.append(current_active_flags)

        ep_total_profit += info['step_profit']
        state = next_state
        action_mask = info['action_mask']

        if done:
            total_demand = info['total_generated'] + 1e-6
            completion_rate = info['total_served'] / total_demand
            wait_time = info['avg_wait_time']
            gini_index = calculate_gini(info['driver_income'])
            return ep_total_profit, completion_rate, wait_time, gini_index


# ==========================================
# 4. 主流程：执行对比训练与测试绘图
# ==========================================
if __name__ == '__main__':
    # 配置区
    TRAIN_EPISODES = 100  # 训练步数
    EVAL_EPISODES = 10  # 测试评估步数

    platform_params = {
        'surge': lambda t, no, nd, sd: 1.0 + 1.2 * np.maximum(0, sd - 0.8) ** 1.2,
        'subsidy': lambda t, no, nd, sd: 3.0 * (
                    np.exp(-50 * (t - 0.30) ** 2) + np.exp(-40 * (t - 0.75) ** 2)) * np.maximum(0, sd - 0.5)
    }

    sim_path = '../generator/simulator_driver_nums=400_hex_scaling=0.017031373249357308_weekday.pkl'
    if not os.path.exists(sim_path):
        raise FileNotFoundError(f"未找到模拟器文件：{sim_path}")

    env = RideHailingEnv(sim_path)
    # 计算全局特征维度 (277区订单 + 277区司机 + 1维时间进度 = 555维)
    GLOBAL_DIM = CONFIG['N_ZONES'] * 2 + 1

    # 结果容器
    results = {
        'MLP': {'train': {'profits': [], 'rates': [], 'ginis': []},
                'eval': {'profits': [], 'rates': [], 'waits': [], 'ginis': []}},
        'MAPPO': {'train': {'profits': [], 'rates': [], 'ginis': []},
                  'eval': {'profits': [], 'rates': [], 'waits': [], 'ginis': []}}
    }

    for model_name in ['MAPPO', 'MLP']:
        print(f"\n[{model_name}] 初始化环境与随机种子...")
        torch.manual_seed(42);
        np.random.seed(42)

        if model_name == 'MAPPO':
            agent = MAPPOAgent(CONFIG['STATE_DIM'], GLOBAL_DIM, CONFIG['ACTION_DIM'], **CONFIG)
        else:
            agent = SharedPPOAgent(CONFIG['STATE_DIM'], CONFIG['ACTION_DIM'], **CONFIG)

        print(f"[{model_name}] 阶段 1/2: 开始模型训练 ({TRAIN_EPISODES} Episodes)...")
        for ep in tqdm(range(TRAIN_EPISODES), desc=f"Training {model_name}"):
            p, r, w, g = run_episode(env, agent, platform_params, is_mappo=(model_name == 'MAPPO'), is_eval=False)
            results[model_name]['train']['profits'].append(p)
            results[model_name]['train']['rates'].append(r)
            results[model_name]['train']['ginis'].append(g)
            agent.update()

        print(f"[{model_name}] 阶段 2/2: 开始冻结网络参数进行测试评估 ({EVAL_EPISODES} Episodes)...")
        agent.policy.eval()  # 关闭 Dropout 等机制，进入纯推理模式
        for ep in tqdm(range(EVAL_EPISODES), desc=f"Evaluating {model_name}"):
            np.random.seed(100 + ep);
            torch.manual_seed(100 + ep)  # 保证两个模型在测试中面临相同的每日订单
            p, r, w, g = run_episode(env, agent, platform_params, is_mappo=(model_name == 'MAPPO'), is_eval=True)
            results[model_name]['eval']['profits'].append(p)
            results[model_name]['eval']['rates'].append(r)
            results[model_name]['eval']['waits'].append(w)
            results[model_name]['eval']['ginis'].append(g)

    # ---------------- 绘图 1：训练收敛曲线图 ----------------
    print("\n[绘图] 正在生成训练对比折线图...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    train_metrics = [('profits', 'Total Platform Profit'), ('rates', 'Completion Rate'),
                     ('ginis', 'Driver Income Gini Index (Lower is fairer)')]
    for idx, (key, title) in enumerate(train_metrics):
        axs[idx].plot(results['MLP']['train'][key], label='Baseline (MLP)', linestyle='--', color='#1f77b4', alpha=0.8)
        axs[idx].plot(results['MAPPO']['train'][key], label='MAPPO (Global Critic)', linewidth=2, color='#d62728',
                      alpha=0.8)
        axs[idx].set_title(title, fontsize=13)
        axs[idx].set_xlabel('Episodes')
        axs[idx].grid(True, alpha=0.3)
        axs[idx].legend()
    plt.tight_layout()
    plt.savefig('mappo_vs_mlp_training.png', dpi=150)
    plt.close()

    # ---------------- 绘图 2：测试表现柱状图 ----------------
    print("[绘图] 正在生成测试评估柱状对比图...")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()
    eval_metrics = [
        ('profits', 'Total Platform Profit ($)', 'higher'),
        ('rates', 'Order Completion Rate', 'higher'),
        ('waits', 'Avg Passenger Wait Time', 'lower'),
        ('ginis', 'Driver Income Gini Index', 'lower')
    ]
    models = ['Baseline MLP', 'MAPPO']
    colors = ['#1f77b4', '#d62728']

    for idx, (metric_key, title, direction) in enumerate(eval_metrics):
        ax = axs[idx]
        means = [np.mean(results['MLP']['eval'][metric_key]), np.mean(results['MAPPO']['eval'][metric_key])]
        stds = [np.std(results['MLP']['eval'][metric_key]), np.std(results['MAPPO']['eval'][metric_key])]

        bars = ax.bar(models, means, yerr=stds, capsize=10, color=colors, alpha=0.8)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        best_note = "↑ (Higher is better)" if direction == 'higher' else "↓ (Lower is better)"
        ax.set_ylabel(f"Metric Value {best_note}")

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.3f}', ha='center', va='bottom', fontsize=12)

    plt.suptitle('Evaluation: Baseline MLP vs. MAPPO (Test Env)', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig('mappo_vs_mlp_evaluation.png', dpi=150)
    plt.close()

    print("\n>>> 对比实验全部完成！已在当前目录下保存两张高清图表：")
    print(" 1. mappo_vs_mlp_training.png")
    print(" 2. mappo_vs_mlp_evaluation.png")
