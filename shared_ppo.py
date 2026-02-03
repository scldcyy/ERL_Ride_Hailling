import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import sys

# 引入项目根目录以导入 dataset 模块
sys.path.append(os.getcwd())
try:
    from dataset.convert2polygon_bridge import PassengerSimulator
except ImportError:
    print("WARNING: Could not import PassengerSimulator. Ensure 'dataset' folder is in path.")

# --- Global Config ---
CONFIG = {
    'N_DRIVERS': 50,  # 仿真司机数 (需与 generate_split_simulators 中的 scaling 逻辑匹配)
    'TIME_STEP_MINUTES': 10,  # 时间步长 10分钟
    'TIME_STEPS_PER_DAY': 144,  # 24 * 60 / 10 = 144 steps

    # RL Params
    'HIDDEN_DIM': 256,  # 增大网络以适应更大的 Hex 空间
    'STATE_DIM':7,  # 增加状态维度: [Loc(1), Time(1), OrderCount(1), DriverCount(1), AvgNeighborOrders(1), AvgNeighborDrivers(1), LockingStatus(1)]
    'LR_ACTOR': 0.0003,
    'LR_CRITIC': 0.001,
    'GAMMA': 0.99,
    'GAE_LAMBDA': 0.95,
    'K_EPOCHS': 4,
    'EPS_CLIP': 0.2,

    # Economics
    'BASE_FARE': 2.5,
    'PRICE_PER_MINUTE': 0.5,  # 修改为按分钟计费更符合 10min 步长逻辑
    'OPPORTUNITY_COST_PER_STEP': 0.1,
    'REPOSITION_COST_PER_STEP': 0.2,
    'IDLE_REWARD': -0.05,
}


class RideHailingEnv:
    def __init__(self, simulator_path):
        print(f"Loading simulator from {simulator_path}...")
        with open(simulator_path, 'rb') as f:
            self.simulator = pickle.load(f)

        # 1. 建立 Hex ID <-> Integer Index 映射
        # 获取所有可能的 Hex ID (来自邻接表)
        self.all_hexes = list(self.simulator.adjacency.keys())
        self.n_zones = len(self.all_hexes)
        self.hex_to_idx = {h: i for i, h in enumerate(self.all_hexes)}
        self.idx_to_hex = {i: h for i, h in enumerate(self.all_hexes)}

        print(f"Environment initialized with {self.n_zones} Hex Zones.")

        # 更新 CONFIG
        CONFIG['N_ZONES'] = self.n_zones
        CONFIG['ACTION_DIM'] = self.n_zones + 1  # Action 0 = Stay/Serve, 1..N = Move to Zone i-1

        # 2. 预计算邻接索引 (用于加速 step 和 mask)
        # self.adjacency_indices[i] 包含索引 i 的所有邻居索引列表
        self.adjacency_indices = {}
        for h_id, neighbors in self.simulator.adjacency.items():
            if h_id in self.hex_to_idx:
                idx = self.hex_to_idx[h_id]
                n_indices = [self.hex_to_idx[n] for n in neighbors if n in self.hex_to_idx]
                self.adjacency_indices[idx] = n_indices

    def reset(self):
        self.time = 0
        # 随机初始化位置
        self.driver_locations = np.random.randint(0, self.n_zones, size=CONFIG['N_DRIVERS'])

        # 司机状态: 0=Idle, 1=Busy/Moving
        self.driver_status = np.zeros(CONFIG['N_DRIVERS'], dtype=int)

        # 锁定时间: 剩余多少个时间步才能变为空闲
        self.driver_free_time = np.zeros(CONFIG['N_DRIVERS'], dtype=int)

        # 统计
        self.driver_rewards = np.zeros(CONFIG['N_DRIVERS'])
        self.total_revenue = 0
        self.pending_orders = []  # 存储当前步的订单字典

        return self._get_state()

    def _get_state(self):
        # 1. 聚合订单信息
        order_counts = np.zeros(self.n_zones)
        for o in self.pending_orders:
            if not o['matched']:
                order_counts[o['origin_idx']] += 1

        # 2. 聚合空闲司机信息
        idle_mask = (self.driver_status == 0)
        idle_driver_counts = np.bincount(self.driver_locations[idle_mask], minlength=self.n_zones)

        states = np.zeros((CONFIG['N_DRIVERS'], CONFIG['STATE_DIM']))

        # 3. 构造每个司机的状态向量
        for i in range(CONFIG['N_DRIVERS']):
            loc = self.driver_locations[i]

            # 获取邻居信息
            neighbors = self.adjacency_indices.get(loc, [])
            if neighbors:
                avg_n_orders = order_counts[neighbors].mean()
                avg_n_drivers = idle_driver_counts[neighbors].mean()
            else:
                avg_n_orders = 0
                avg_n_drivers = 0

            states[i] = [
                loc / self.n_zones,  # Normalize Location ID (simple scaling)
                self.time / CONFIG['TIME_STEPS_PER_DAY'],  # Normalize Time
                order_counts[loc],
                idle_driver_counts[loc],
                avg_n_orders,
                avg_n_drivers,
                self.driver_free_time[i]  # 告知 Agent 自己是否被锁定
            ]
        return states

    def get_valid_actions_mask(self):
        """
        生成动作掩码。
        返回: (N_DRIVERS, ACTION_DIM) 的布尔矩阵。True表示动作有效。
        逻辑: 司机只能选择 Stay(0) 或者移动到邻接网格对应的 Index+1。
        """
        mask = np.zeros((CONFIG['N_DRIVERS'], CONFIG['ACTION_DIM']), dtype=bool)

        # 动作 0 (Stay/Serve Local) 总是有效的
        mask[:, 0] = True

        for i in range(CONFIG['N_DRIVERS']):
            loc = self.driver_locations[i]
            neighbors = self.adjacency_indices.get(loc, [])
            # Action index = target_zone_idx + 1
            valid_action_indices = [n + 1 for n in neighbors]
            mask[i, valid_action_indices] = True

        return mask

    def step(self, actions, platform_params):
        # --- 1. 时间流逝与状态解锁 ---
        # 减少所有忙碌司机的锁定时间
        self.driver_free_time[self.driver_free_time > 0] -= 1

        # 如果锁定时间归零，且之前是忙碌状态，则变为空闲
        freed_drivers = np.where((self.driver_status == 1) & (self.driver_free_time == 0))[0]
        self.driver_status[freed_drivers] = 0

        # --- 2. 生成新订单 ---
        # 模拟器生成的是 hex_id，需要转换为 idx
        raw_orders = self.simulator.generate_orders(self.time, self.all_hexes)
        new_orders = []
        for o in raw_orders:
            # 过滤掉不在地图映射中的异常点
            if o['origin_hex'] in self.hex_to_idx and o['dest_hex'] in self.hex_to_idx:
                o['origin_idx'] = self.hex_to_idx[o['origin_hex']]
                o['dest_idx'] = self.hex_to_idx[o['dest_hex']]
                o['matched'] = False
                o['wait_time'] = 0
                new_orders.append(o)

        # 将上一轮未匹配的订单保留 (可增加超时丢弃逻辑)
        self.pending_orders = [o for o in self.pending_orders if not o['matched'] and o['wait_time'] < 3]  # 最多等3个step
        for o in self.pending_orders: o['wait_time'] += 1
        self.pending_orders.extend(new_orders)

        rewards = np.zeros(CONFIG['N_DRIVERS'])

        # --- 3. 执行动作 (仅对空闲司机) ---
        # 这里的 actions 是 Agent 输出的，对于被锁定的司机，动作会被忽略
        idle_indices = np.where(self.driver_status == 0)[0]

        # 打乱顺序，避免低 ID 司机总是优先抢单
        np.random.shuffle(idle_indices)

        for i in idle_indices:
            action = actions[i]
            current_loc = self.driver_locations[i]

            if action == 0:  # 尝试接单 (Serve)
                # 查找当前位置的可用订单
                local_orders = [o for o in self.pending_orders
                                if o['origin_idx'] == current_loc and not o['matched']]

                if local_orders:
                    # 接单成功
                    order = local_orders[0]
                    order['matched'] = True

                    # 收益计算: 基础费 + 时长费 (模拟) * 抽成
                    # 注意: simulator 返回的 duration 已经是 step 数
                    trip_steps = max(1, int(order['duration']))
                    fare = CONFIG['BASE_FARE'] + trip_steps * CONFIG['TIME_STEP_MINUTES'] * CONFIG['PRICE_PER_MINUTE']

                    # 计算奖励 (纯利)
                    income = fare * (1 - platform_params['commission'])
                    cost = trip_steps * CONFIG['OPPORTUNITY_COST_PER_STEP']
                    rewards[i] = income - cost

                    # 更新状态
                    self.driver_status[i] = 1  # Set to Busy
                    self.driver_free_time[i] = trip_steps  # 锁定 N 个 step
                    self.driver_locations[i] = order['dest_idx']  # 逻辑上直接设为终点(简化)
                    self.total_revenue += income
                else:
                    # 没有订单，Idle Penalty
                    rewards[i] = CONFIG['IDLE_REWARD']

            else:  # 再定位 (Reposition)
                target_idx = action - 1

                # 检查合法性 (虽然有 Mask，但双重保险)
                neighbors = self.adjacency_indices.get(current_loc, [])

                if target_idx in neighbors:
                    # 移动成功
                    # 假设移动到邻居需要 1 个 time step (10 mins)
                    move_steps = 1
                    cost = CONFIG['REPOSITION_COST_PER_STEP']
                    rewards[i] = -cost

                    self.driver_status[i] = 1
                    self.driver_free_time[i] = move_steps
                    self.driver_locations[i] = target_idx
                else:
                    # 非法移动 (Mask 应该防止这种情况，但如果发生了...)
                    rewards[i] = -0.5  # 惩罚

        # --- 4. 推进时间 ---
        self.time += 1
        done = (self.time >= CONFIG['TIME_STEPS_PER_DAY'])

        return self._get_state(), rewards, done, {}


# --- PPO Components (Added Masking) ---

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

    def act(self, state, action_mask=None):
        """
        state: (Batch, State_Dim)
        action_mask: (Batch, Action_Dim) - True for valid actions
        """
        action_logits = self.actor(state)

        if action_mask is not None:
            # 将无效动作的 logits 设为极小的负数
            action_logits = action_logits.masked_fill(~action_mask, -1e8)

        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action, action_mask=None):
        action_logits = self.actor(state)

        if action_mask is not None:
            action_logits = action_logits.masked_fill(~action_mask, -1e8)

        dist = Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.masks = []  # Store masks for update

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.masks[:]


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
        self.K_epochs = hyperparameters['K_EPOCHS']
        self.eps_clip = hyperparameters['EPS_CLIP']

    def select_actions(self, states, action_mask):
        with torch.no_grad():
            states = torch.FloatTensor(states)
            mask = torch.BoolTensor(action_mask)
            actions, logprobs = self.policy_old.act(states, mask)

        self.buffer.states.append(states)
        self.buffer.actions.append(actions)
        self.buffer.logprobs.append(logprobs)
        self.buffer.masks.append(mask)  # 保存 Mask 用于 Update

        return actions.numpy()

    def update(self):
        # Flatten buffers
        old_states = torch.cat(self.buffer.states).detach()
        old_actions = torch.cat(self.buffer.actions).detach()
        old_logprobs = torch.cat(self.buffer.logprobs).detach()
        old_masks = torch.cat(self.buffer.masks).detach()

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        # Flatten rewards list of lists
        flat_rewards = []
        flat_terminals = []
        for step_rewards in self.buffer.rewards:
            flat_rewards.extend(step_rewards)
        for step_dones in self.buffer.is_terminals:
            # done is scalar per step usually in single env, but here we have multi-agent step
            # Assume is_terminals stores boolean scalars for the whole env?
            # Based on Trainer, done is scalar. But rewards is (N_DRIVERS,)
            # We need to structure advantages correctly.
            pass

        # Simplified GAE calculation (Batch-based)
        # Assuming buffer stores [Step1_Rewards(N), Step2_Rewards(N)...]
        # We process each agent's trajectory?
        # Since it's Shared PPO with random matching, we can treat (State, Action, Reward) as independent samples
        # or grouped by time. For simplicity in this heavy masking env, standard batch GAE:

        # Convert list of arrays to tensor: (Time, N_Drivers)
        rewards_tensor = torch.tensor(np.array(self.buffer.rewards), dtype=torch.float32)

        # Calculate State Values
        with torch.no_grad():
            values = self.policy_old.critic(old_states).detach()

        # Reshape values to (Time, N_Drivers) to match rewards
        n_steps = len(self.buffer.rewards)
        n_drivers = CONFIG['N_DRIVERS']
        values = values.view(n_steps, n_drivers)

        advantages = torch.zeros_like(rewards_tensor)
        last_gae_lam = 0

        # GAE Loop
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 0.0  # Last step assumes done
                next_values = 0
            else:
                next_non_terminal = 1.0
                next_values = values[t + 1]

            delta = rewards_tensor[t] + self.gamma * next_values * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam

        # Flatten for training
        advantages = advantages.view(-1)
        old_values = values.view(-1)

        # Normalize
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        returns = advantages + old_values

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_masks)

            state_values = state_values.squeeze()

            # Ratios
            ratios = torch.exp(logprobs - old_logprobs)

            # Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)

    def load_by_path(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path))
        self.policy_old.load_state_dict(torch.load(checkpoint_path))

    def load_by_weights(self, weights):
        self.policy.load_state_dict(weights[0])
        self.policy_old.load_state_dict(weights[1])

    def get_weights(self):
        return self.policy.state_dict(), self.policy_old.state_dict()

    def eval(self):
        self.policy.eval()
        self.policy_old.eval()


class Trainer:
    def __init__(self, simulator_path='model/generators/simulator_hex_weekday.pkl', checkpoint_path='model/agent.pth'):
        self.checkpoint_path = checkpoint_path

        # Initialize Environment (which loads simulator)
        self.env = RideHailingEnv(simulator_path)

        # Initialize Agent
        self.agent = SharedPPOAgent(CONFIG['STATE_DIM'], CONFIG['ACTION_DIM'], **CONFIG)

    def train(self, platform_params, num_episodes=50):
        episode_rewards = []
        print(f"\n--- Starting RL Training for {num_episodes} Episodes ---")

        for episode in tqdm(range(num_episodes), desc="Training"):
            state = self.env.reset()
            ep_reward = 0

            # Loop for one day (144 steps)
            while True:
                # 1. Get Action Mask (Valid moves)
                mask = self.env.get_valid_actions_mask()

                # 2. Agent Select Actions
                actions = self.agent.select_actions(state, mask)

                # 3. Environment Step
                next_state, rewards, done, _ = self.env.step(actions, platform_params)

                # 4. Store Buffer
                self.agent.buffer.rewards.append(rewards)
                self.agent.buffer.is_terminals.append(done)  # Simplified scalar

                state = next_state
                ep_reward += np.sum(rewards)

                if done:
                    break

            # Update Agent at end of episode
            self.agent.update()
            episode_rewards.append(ep_reward)
        self._plot_rewards(episode_rewards)
        return episode_rewards

    def visualize_simulation(self, platform_params, filename="img/hex_simulation.gif"):
        print(f"--- Generating Hex visualization to {filename} ---")
        fig, ax = plt.subplots(figsize=(10, 10))

        states_snapshots = []
        state = self.env.reset()
        done = False

        while not done:
            mask = self.env.get_valid_actions_mask()
            actions = self.agent.select_actions(state, mask)

            # Snapshot
            snapshot = {
                'time': self.env.time,
                'driver_locs': self.env.driver_locations.copy(),
                'driver_status': self.env.driver_status.copy(),
                # Convert pending orders indices back to coords for plotting?
                # Ideally we need centroids.
                # For simplicity, we just store indices.
            }
            states_snapshots.append(snapshot)
            state, _, done, _ = self.env.step(actions, platform_params)

        # Get Centroids for plotting
        # simulator.df contains explicit coords? No, Simulator has hex_ids.
        # We need h3 to lat/lng.
        import h3

        # Precompute centroids for all zones
        centroids_dict = {}
        for idx, h_id in self.env.idx_to_hex.items():
            lat, lng = h3.cell_to_latlng(h_id)
            centroids_dict[idx] = (lng, lat)  # x, y

        def animate(i):
            ax.clear()
            snapshot = states_snapshots[i]

            # Plot Drivers
            d_locs = snapshot['driver_locs']
            d_stats = snapshot['driver_status']

            idle_xy = [centroids_dict[loc] for loc, stat in zip(d_locs, d_stats) if stat == 0]
            busy_xy = [centroids_dict[loc] for loc, stat in zip(d_locs, d_stats) if stat == 1]

            if idle_xy:
                ix, iy = zip(*idle_xy)
                ax.scatter(ix, iy, c='blue', s=20, label='Idle', alpha=0.6)
            if busy_xy:
                bx, by = zip(*busy_xy)
                ax.scatter(bx, by, c='red', s=20, label='Busy', alpha=0.6)

            ax.set_title(f"Time Step: {snapshot['time']} / {CONFIG['TIME_STEPS_PER_DAY']}")
            ax.legend()

        ani = animation.FuncAnimation(fig, animate, frames=len(states_snapshots), interval=200)
        ani.save(filename, writer='pillow')
        print("Visualization saved.")

    def _plot_rewards(self,rewards):
        plt.plot(rewards)
        plt.show()


if __name__ == '__main__':
    # 简单的测试入口
    platform_params = {
        'commission': 0.2
    }

    # 确保有生成器文件
    sim_path = 'model/generators/simulator_hex_weekday.pkl'
    if not os.path.exists(sim_path):
        print(f"Error: Simulator file not found at {sim_path}")
        print("Please run generate_split_simulators.py first.")
    else:
        trainer = Trainer(simulator_path=sim_path)
        rewards = trainer.train(platform_params, num_episodes=5)
        trainer.visualize_simulation(platform_params)