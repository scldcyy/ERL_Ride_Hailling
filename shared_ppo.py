import os

import h3
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt, animation
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler
import pickle
import sys
import copy

from tqdm import tqdm

sys.path.append(os.getcwd())

# --- Global Config ---
CONFIG = {
    'N_DRIVERS': 200,
    'TIME_STEP_MINUTES': 5,
    'TIME_STEPS_PER_DAY': 288,
    'N_ZONES':277,

    # PPO Hyperparameters
    'HIDDEN_DIM': 256,
    'STATE_DIM': 9,  # [Lat, Lng, Orders, Drivers, AvgOrders, AvgDrivers, FreeTime, Surge, Subsidy]
    'ACTION_DIM': 8,
    'LR_ACTOR': 3e-4,
    'LR_CRITIC': 1e-3,
    'GAMMA': 0.99,
    'GAE_LAMBDA': 0.95,
    'PPO_EPOCHS': 10,
    'BATCH_SIZE': 512,
    'EPS_CLIP': 0.2,
    'ENTROPY_COEF': 0.01,
    'MAX_GRAD_NORM': 0.5,

    # Economics
    'BASE_FARE': 2.5,
    'PRICE_PER_MINUTE': 0.5,
    'OPPORTUNITY_COST_PER_STEP': 0.1,
    'REPOSITION_COST_PER_STEP': 0.2,
    'IDLE_REWARD': -0.05,
    'MIN_FARE_THRESHOLD': 4.0
}


class RideHailingEnv:
    def __init__(self, simulator_path, fixed_scenarios=None):
        with open(simulator_path, 'rb') as f:
            self.simulator = pickle.load(f)

        self.all_hexes = list(self.simulator.adjacency.keys())
        self.n_zones = len(self.all_hexes)
        self.hex_to_idx = {h: i for i, h in enumerate(self.all_hexes)}
        self.idx_to_hex = {i: h for i, h in enumerate(self.all_hexes)}

        self.adjacency_indices = {}
        for h_id, neighbors in self.simulator.adjacency.items():
            idx = self.hex_to_idx[h_id]
            n_indices = {direct: self.hex_to_idx[n_h_id] for direct, n_h_id in neighbors.items()}
            self.adjacency_indices[idx] = n_indices

        self.fixed_scenarios = fixed_scenarios
        self.current_scenario_idx = 0

        self.total_revenue = 0
        self.total_served_orders = 0
        self.total_generated_orders = 0
        self.total_wait_time = 0
        self.pending_orders = []

    def reset(self):
        self.time = 0
        # 固定种子以保证每个Scenario内的初始位置一致
        rng = np.random.RandomState(42 + self.current_scenario_idx)
        self.driver_locations = rng.randint(0, self.n_zones, size=CONFIG['N_DRIVERS'])
        self.driver_status = np.zeros(CONFIG['N_DRIVERS'], dtype=int)
        self.driver_free_time = np.zeros(CONFIG['N_DRIVERS'], dtype=int)

        self.total_revenue = 0
        self.total_served_orders = 0
        self.total_generated_orders = 0
        self.total_wait_time = 0
        self.pending_orders = []

        if self.fixed_scenarios is not None:
            self.current_day_orders = self.fixed_scenarios[self.current_scenario_idx % len(self.fixed_scenarios)]
            self.current_scenario_idx += 1

        return self._get_state(platform_params=None)

    def _get_state(self, platform_params=None):
        order_counts = np.zeros(self.n_zones)
        for o in self.pending_orders:
            if not o['matched']:
                order_counts[o['origin_idx']] += 1

        idle_mask = (self.driver_status == 0)
        idle_driver_counts = np.bincount(self.driver_locations[idle_mask], minlength=self.n_zones)

        if platform_params is None:
            current_surges = np.ones(self.n_zones)
            current_subsidies = np.zeros(self.n_zones)
        else:
            t = min(self.time, CONFIG['TIME_STEPS_PER_DAY'] - 1)
            current_surges = platform_params['lambda'][t]
            current_subsidies = platform_params['subsidy'][t]

        states = np.zeros((CONFIG['N_DRIVERS'], CONFIG['STATE_DIM']))

        for i in range(CONFIG['N_DRIVERS']):
            loc = self.driver_locations[i]
            neighbors = self.adjacency_indices.get(loc, {})
            if neighbors:
                n_locs = list(neighbors.values())
                avg_n_orders = order_counts[n_locs].mean()
                avg_n_drivers = idle_driver_counts[n_locs].mean()
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
                1.0 if self.driver_free_time[i] > 0 else 0.0,
                current_surges[loc],
                current_subsidies[loc]
            ]
        return states

    def get_global_observation(self):
        order_counts = np.zeros(self.n_zones)
        for o in self.pending_orders:
            if not o['matched']: order_counts[o['origin_idx']] += 1

        idle_mask = (self.driver_status == 0)
        driver_counts = np.bincount(self.driver_locations[idle_mask], minlength=self.n_zones)

        # 简单归一化
        obs = np.stack([
            order_counts / (order_counts.max() + 1e-6),
            driver_counts / (driver_counts.max() + 1e-6),
        ])
        return obs

    def step(self, actions, platform_params):
        prev_revenue = self.total_revenue

        self.driver_free_time[self.driver_free_time > 0] -= 1
        freed_drivers = np.where((self.driver_status == 1) & (self.driver_free_time == 0))[0]
        self.driver_status[freed_drivers] = 0

        if self.fixed_scenarios is not None:
            raw_orders = copy.deepcopy(self.current_day_orders[self.time])
        else:
            raw_orders = self.simulator.generate_orders(self.time, self.all_hexes)

        self.total_generated_orders += len(raw_orders)

        new_orders = []
        for o in raw_orders:
            if o['origin_hex'] in self.hex_to_idx and o['dest_hex'] in self.hex_to_idx:
                o['origin_idx'] = self.hex_to_idx[o['origin_hex']]
                o['dest_idx'] = self.hex_to_idx[o['dest_hex']]
                o['matched'] = False
                o['wait_time'] = 0
                new_orders.append(o)

        self.pending_orders = [o for o in self.pending_orders if not o['matched'] and o['wait_time'] < 3]
        for o in self.pending_orders:
            o['wait_time'] += 1
            self.total_wait_time += 1

        self.pending_orders.extend(new_orders)
        rewards = np.zeros(CONFIG['N_DRIVERS'])
        step_subsidy_cost = 0.0

        idle_indices = np.where(self.driver_status == 0)[0]
        # 在 Scenarios 模式下，这里的 shuffle 应该由全局种子控制，但为保持多样性保留 random
        np.random.shuffle(idle_indices)

        for i in idle_indices:
            action = actions[i]
            current_loc = self.driver_locations[i]

            if action == 0:  # Serve
                local_orders = [o for o in self.pending_orders if o['origin_idx'] == current_loc and not o['matched']]
                if local_orders:
                    order = local_orders[0]
                    order['matched'] = True
                    self.total_served_orders += 1
                    trip_steps = max(1, int(order['duration']))

                    surge = platform_params['lambda'][self.time, current_loc]
                    comm_rate = platform_params['commission']
                    explicit_subsidy = platform_params['subsidy'][self.time, current_loc]

                    base_fare = (CONFIG['BASE_FARE'] + trip_steps * CONFIG['TIME_STEP_MINUTES'] * CONFIG[
                        'PRICE_PER_MINUTE'])
                    gross_fare = base_fare * surge
                    driver_nominal_income = gross_fare * (1 - comm_rate) + explicit_subsidy

                    actual_driver_income = max(driver_nominal_income, CONFIG['MIN_FARE_THRESHOLD'])
                    gap_subsidy = actual_driver_income - driver_nominal_income
                    step_subsidy_cost += (gap_subsidy + explicit_subsidy)

                    op_cost = trip_steps * CONFIG['OPPORTUNITY_COST_PER_STEP']
                    rewards[i] = actual_driver_income - op_cost
                    self.driver_status[i] = 1
                    self.driver_free_time[i] = trip_steps
                    self.driver_locations[i] = order['dest_idx']
                    self.total_revenue += gross_fare
                else:
                    rewards[i] = CONFIG['IDLE_REWARD']
            else:  # Reposition
                target_direct = action - 2
                neighbors = self.adjacency_indices.get(current_loc, {})
                if target_direct in neighbors:
                    rewards[i] = -CONFIG['REPOSITION_COST_PER_STEP']
                    self.driver_status[i] = 1
                    self.driver_free_time[i] = 1
                    self.driver_locations[i] = neighbors[target_direct]
                else:
                    rewards[i] = 0

        step_revenue = self.total_revenue - prev_revenue
        step_profit = step_revenue * platform_params['commission'] - step_subsidy_cost

        self.time += 1
        done = (self.time >= CONFIG['TIME_STEPS_PER_DAY'])
        next_state = self._get_state(platform_params)

        info = {
            'step_profit': step_profit,
            'total_revenue': self.total_revenue,
            'total_served': self.total_served_orders,
            'total_generated': self.total_generated_orders,
            'total_wait_time': self.total_wait_time,
        }

        return next_state, rewards, done, info


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


class Trainer:
    def __init__(self, simulator_path, fixed_scenarios=None):
        self.env = RideHailingEnv(simulator_path, fixed_scenarios=fixed_scenarios)
        self.agent = SharedPPOAgent(CONFIG['STATE_DIM'], CONFIG['ACTION_DIM'], **CONFIG)

    def train_and_evaluate(self, platform_params, num_episodes=5):
        ep_profits = []
        ep_completion_rates = []
        ep_wait_times = []

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
                    total_demand = info['total_generated'] + 1e-6
                    completion_rate = info['total_served'] / total_demand
                    wait_time = -info['total_wait_time']
                    ep_profits.append(ep_total_profit)
                    ep_completion_rates.append(completion_rate)
                    ep_wait_times.append(wait_time)
                    break
            self.agent.update()
        # self._plot_rewards(ep_profits)
        # self._plot_rewards(ep_completion_rates)
        # self._plot_rewards(ep_wait_times)

        return np.array([np.mean(ep_profits), np.mean(ep_completion_rates), np.mean(ep_wait_times)])

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

    def _plot_rewards(self, rewards):
        plt.plot(rewards)
        plt.show()


if __name__ == '__main__':
    platform_params = {
        'commission': 0.2,
        'lambda': np.full((CONFIG['TIME_STEPS_PER_DAY'], CONFIG['N_ZONES']), 2),
        'subsidy': np.full((CONFIG['TIME_STEPS_PER_DAY'], CONFIG['N_ZONES']), 2.5)
    }

    sim_path = 'model/generators/simulator_hex_scaling=0.004257843312339327_weekday.pkl'
    if not os.path.exists(sim_path):
        print(f"Error: Simulator file not found at {sim_path}")
        print("Please run generate_split_simulators.py first.")
    else:
        trainer = Trainer(simulator_path=sim_path)
        rewards = trainer.train_and_evaluate(platform_params, num_episodes=50)  # Reduced for test
