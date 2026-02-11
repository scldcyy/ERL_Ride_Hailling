import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pickle
import sys
import h3

# 引入项目根目录以导入 dataset 模块
sys.path.append(os.getcwd())

# --- Global Config ---
CONFIG = {
    'N_DRIVERS': 100,  # 仿真司机数量，根据论文规模调整
    'TIME_STEP_MINUTES': 5,
    'TIME_STEPS_PER_DAY': 288,  # 24 * 60 / 5

    # RL Params (MAPPO / Shared PPO)
    'HIDDEN_DIM': 256,
    'STATE_DIM': 7, # 当前位置z_t, 当前时t,当前区域订单数,当前区域司机数,邻居订单数,邻居司机数,空驶时间
    'ACTION_DIM':8, # 是否接单的one-hot编码+6个方向的one-hot编码
    'LR_ACTOR': 0.0003,
    'LR_CRITIC': 0.001,
    'GAMMA': 0.99,
    'GAE_LAMBDA': 0.95,
    'K_EPOCHS': 4,
    'EPS_CLIP': 0.2,

    # Economics & Constraints
    'BASE_FARE': 2.5,
    'PRICE_PER_MINUTE': 0.5,
    'OPPORTUNITY_COST_PER_STEP': 0.1,
    'REPOSITION_COST_PER_STEP': 0.2,
    'IDLE_REWARD': -0.05,

    # [新增] 缺陷修复2: 司机最低接单价格约束 (Min-Fare Protection)
    # 如果订单价格低于此阈值，平台必须补贴至此价格，否则司机可能拒单(此处简化为平台强制补贴成本)
    'MIN_FARE_THRESHOLD': 4.0
}


class RideHailingEnv:
    def __init__(self, simulator_path):
        self.driver_free_time = None
        self.driver_status = None
        self.driver_locations = None
        self.time = None
        print(f"Loading simulator from {simulator_path}...")
        with open(simulator_path, 'rb') as f:
            self.simulator = pickle.load(f)

        self.all_hexes = list(self.simulator.adjacency.keys())
        self.n_zones = len(self.all_hexes)
        self.hex_to_idx = {h: i for i, h in enumerate(self.all_hexes)}
        self.idx_to_hex = {i: h for i, h in enumerate(self.all_hexes)}

        self.adjacency_indices = {}
        for h_id, neighbors in self.simulator.adjacency.items():
            idx = self.hex_to_idx[h_id]
            n_indices = {direct: self.hex_to_idx[n_h_id] for direct,n_h_id in neighbors.items()}
            self.adjacency_indices[idx] = n_indices

        self.total_revenue = 0
        self.total_served_orders = 0
        self.total_generated_orders = 0  # 新增：用于计算完成率
        self.total_wait_time = 0
        self.pending_orders = []

    def reset(self):
        self.time = 0
        self.driver_locations = np.random.randint(0, self.n_zones, size=CONFIG['N_DRIVERS'])
        self.driver_status = np.zeros(CONFIG['N_DRIVERS'], dtype=int)
        self.driver_free_time = np.zeros(CONFIG['N_DRIVERS'], dtype=int)

        self.total_revenue = 0
        self.total_served_orders = 0
        self.total_generated_orders = 0
        self.total_wait_time = 0
        self.pending_orders = []

        return self._get_state()

    def _get_state(self):
        # ... (保持原有的状态获取逻辑) ...
        order_counts = np.zeros(self.n_zones)
        for o in self.pending_orders:
            if not o['matched']:
                order_counts[o['origin_idx']] += 1

        idle_mask = (self.driver_status == 0)
        idle_driver_counts = np.bincount(self.driver_locations[idle_mask], minlength=self.n_zones)

        states = np.zeros((CONFIG['N_DRIVERS'], CONFIG['STATE_DIM']))

        for i in range(CONFIG['N_DRIVERS']):
            loc = self.driver_locations[i]
            neighbors = self.adjacency_indices.get(loc, {})
            if neighbors:
                avg_n_orders = np.array([order_counts[zone] for zone in neighbors.values()]).mean()
                avg_n_drivers = np.array([idle_driver_counts[zone] for zone in neighbors.values()]).mean()
            else:
                avg_n_orders = 0
                avg_n_drivers = 0

            states[i] = [
                loc / self.n_zones,
                self.time / CONFIG['TIME_STEPS_PER_DAY'],
                order_counts[loc],
                idle_driver_counts[loc],
                avg_n_orders,
                avg_n_drivers,
                1.0 if self.driver_free_time[i] > 0 else 0.0
            ]
        return states

    def step(self, actions, platform_params):
        # 1. 更新司机状态
        self.driver_free_time[self.driver_free_time > 0] -= 1
        freed_drivers = np.where((self.driver_status == 1) & (self.driver_free_time == 0))[0]
        self.driver_status[freed_drivers] = 0

        # 2. 生成订单
        raw_orders = self.simulator.generate_orders(self.time, self.all_hexes)
        self.total_generated_orders += len(raw_orders)  # 记录总单量

        new_orders = []
        for o in raw_orders:
            if o['origin_hex'] in self.hex_to_idx and o['dest_hex'] in self.hex_to_idx:
                o['origin_idx'] = self.hex_to_idx[o['origin_hex']]
                o['dest_idx'] = self.hex_to_idx[o['dest_hex']]
                o['matched'] = False
                o['wait_time'] = 0
                new_orders.append(o)

        # 订单积压与超时取消
        self.pending_orders = [o for o in self.pending_orders if not o['matched'] and o['wait_time'] < 3]
        for o in self.pending_orders:
            o['wait_time'] += 1
            self.total_wait_time += 1

        self.pending_orders.extend(new_orders)
        rewards = np.zeros(CONFIG['N_DRIVERS'])

        # 统计本Step的额外成本
        step_subsidy_cost = 0.0

        # 3. 执行动作
        idle_indices = np.where(self.driver_status == 0)[0]
        np.random.shuffle(idle_indices)

        for i in idle_indices:
            action = actions[i]
            current_loc = self.driver_locations[i]

            if action == 0:  # 接单 (Serve)
                local_orders = [o for o in self.pending_orders if o['origin_idx'] == current_loc and not o['matched']]
                if local_orders:
                    order = local_orders[0]
                    order['matched'] = True
                    self.total_served_orders += 1

                    trip_steps = max(1, int(order['duration']))

                    # 获取当前时空的定价参数
                    # 假设 platform_params['lambda'] 形状为 (Time, Zones)
                    surge = platform_params['lambda'][self.time, current_loc]
                    comm_rate = platform_params['commission']
                    explicit_subsidy = platform_params['subsidy'][self.time, current_loc]

                    # 基础票价
                    base_fare = (CONFIG['BASE_FARE'] + trip_steps * CONFIG['TIME_STEP_MINUTES'] * CONFIG[
                        'PRICE_PER_MINUTE'])
                    # 乘客支付总额
                    gross_fare = base_fare * surge

                    # 司机名义收入
                    driver_nominal_income = gross_fare * (1 - comm_rate) + explicit_subsidy

                    # [修复] 最低价格约束 (Min-Fare Protection)
                    # 如果名义收入低于阈值，平台必须补齐，视为额外成本
                    actual_driver_income = max(driver_nominal_income, CONFIG['MIN_FARE_THRESHOLD'])
                    gap_subsidy = actual_driver_income - driver_nominal_income
                    if gap_subsidy > 0:
                        step_subsidy_cost += gap_subsidy

                    # 运营成本
                    op_cost = trip_steps * CONFIG['OPPORTUNITY_COST_PER_STEP']

                    # 司机奖励 = 净收入
                    rewards[i] = actual_driver_income - op_cost
                    # TODO 不是一步，而是一段时间
                    self.driver_status[i] = 1
                    self.driver_free_time[i] = trip_steps
                    self.driver_locations[i] = order['dest_idx']
                    self.total_revenue += gross_fare  # 记录流水
                else:
                    rewards[i] = CONFIG['IDLE_REWARD']  # 空闲惩罚
            else:  # 调度 (Reposition)
                target_direct = action - 2
                neighbors = self.adjacency_indices.get(current_loc, {})
                if target_direct in neighbors:
                    rewards[i] = -CONFIG['REPOSITION_COST_PER_STEP']
                    self.driver_status[i] = 1
                    self.driver_free_time[i] = 1
                    self.driver_locations[i] = neighbors[target_direct]
                else:
                    rewards[i] = 0  # 非法动作/原地不动

        self.time += 1
        done = (self.time >= CONFIG['TIME_STEPS_PER_DAY'])

        # Info 包含平台计算利润所需的所有信息
        info = {
            'total_revenue': self.total_revenue,
            'total_served': self.total_served_orders,
            'total_generated': self.total_generated_orders,
            'total_wait_time': self.total_wait_time,
            'step_subsidy_cost': step_subsidy_cost  # 隐性补贴
        }

        return self._get_state(), rewards, done, info


# --- RL Agent (Shared PPO / MAPPO) ---
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
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.masks = []

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

    def select_actions(self, states):
        with torch.no_grad():
            states = torch.FloatTensor(states)
            actions, logprobs = self.policy_old.act(states)
        self.buffer.states.append(states)
        self.buffer.actions.append(actions)
        self.buffer.logprobs.append(logprobs)
        return actions.numpy()

    def update(self):
        old_states = torch.cat(self.buffer.states).detach()
        old_actions = torch.cat(self.buffer.actions).detach()
        old_logprobs = torch.cat(self.buffer.logprobs).detach()

        rewards_tensor = torch.tensor(np.array(self.buffer.rewards), dtype=torch.float32)

        with torch.no_grad():
            values = self.policy_old.critic(old_states).detach()

        n_steps = len(self.buffer.rewards)
        n_drivers = CONFIG['N_DRIVERS']
        values = values.view(n_steps, n_drivers)

        advantages = torch.zeros_like(rewards_tensor)
        last_gae_lam = 0

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 0.0
                next_values = 0
            else:
                next_non_terminal = 1.0
                next_values = values[t + 1]
            delta = rewards_tensor[t] + self.gamma * next_values * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam

        advantages = advantages.view(-1)
        old_values = values.view(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        returns = advantages + old_values

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = state_values.squeeze()
            ratios = torch.exp(logprobs - old_logprobs)
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

    def load_by_weights(self, weights):
        self.policy.load_state_dict(weights[0])
        self.policy_old.load_state_dict(weights[1])

    def get_weights(self):
        return self.policy.state_dict(), self.policy_old.state_dict()

    def eval(self):
        self.policy.eval()
        self.policy_old.eval()


# --- Trainer Class ---
class Trainer:
    def __init__(self, simulator_path):
        self.env = RideHailingEnv(simulator_path)
        self.agent = SharedPPOAgent(CONFIG['STATE_DIM'], CONFIG['ACTION_DIM'], **CONFIG)

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

    def visualize_simulation(self, platform_params, filename="img/hex_simulation.gif"):
        print(f"--- Generating Hex visualization to {filename} ---")
        fig, ax = plt.subplots(figsize=(10, 10))

        states_snapshots = []
        state = self.env.reset()
        done = False

        while not done:
            mask = self.env.get_valid_actions_mask()
            actions = self.agent.select_actions(state, mask)

            snapshot = {
                'time': self.env.time,
                'driver_locs': self.env.driver_locations.copy(),
                'driver_status': self.env.driver_status.copy(),
            }
            states_snapshots.append(snapshot)
            state, _, done, _ = self.env.step(actions, platform_params)

        centroids_dict = {}
        for idx, h_id in self.env.idx_to_hex.items():
            lat, lng = h3.cell_to_latlng(h_id)
            centroids_dict[idx] = (lng, lat)

        def animate(i):
            ax.clear()
            snapshot = states_snapshots[i]

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

    def _plot_rewards(self, rewards):
        plt.plot(rewards)
        plt.show()


if __name__ == '__main__':
    platform_params = {
        'commission': 0.2,
        'lambda': np.full((CONFIG['TIME_STEPS_PER_DAY'], 277), 2),
        'subsidy': np.full((CONFIG['TIME_STEPS_PER_DAY'], 277), 2.5)
    }

    sim_path = 'model/generators/simulator_hex_scaling=0.004257843312339327_weekday.pkl'
    if not os.path.exists(sim_path):
        print(f"Error: Simulator file not found at {sim_path}")
        print("Please run generate_split_simulators.py first.")
    else:
        trainer = Trainer(simulator_path=sim_path)
        rewards = trainer.train(platform_params, num_episodes=50)  # Reduced for test
        pass
        # trainer.agent.save(trainer.checkpoint_path)