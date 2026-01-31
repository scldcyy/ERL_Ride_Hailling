import os
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.distributions import Categorical
from tqdm import tqdm

import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
# Add new imports at the top of main_ea.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class DataSetProcesser:
    def __init__(self, n_zones=40):
        self.hourly_counts = None
        self.superzone_counts = None
        self.df_zones_info = None
        self.df_trips = None
        self.valid_gdf_zones = None
        self.zone_centroids = None
        cur_dir = os.path.join(os.getcwd(), 'dataset')
        self.trip_data_file = os.path.join(cur_dir, 'fhvhv_jan_01.parquet')
        self.zone_lookup_file = os.path.join(cur_dir, 'taxi+_zone_lookup.csv')
        self.shapefile_path = os.path.join(cur_dir, 'taxi_zones', 'taxi_zones.shp')
        self.n_zones = n_zones
        print("--- Initializing DataSetProcesser ---")
        self._dataLoading()

    def _dataLoading(self):
        try:
            print(f"Loading trip data from: {self.trip_data_file}")
            self.df_trips = pd.read_parquet(self.trip_data_file)
            self.df_zones_info = pd.read_csv(self.zone_lookup_file)
            self.df_trips['pickup_datetime'] = pd.to_datetime(self.df_trips['pickup_datetime'])
            self.df_trips['hour'] = self.df_trips['pickup_datetime'].dt.hour
            self._geospatialClustering()
            self.hourly_counts = self.df_trips['hour'].value_counts().sort_index()
            self.superzone_counts = self.df_trips['PU_SuperZone'].value_counts().sort_index()
            print("--- DataSetProcesser successfully initialized ---")
        except FileNotFoundError as e:
            print(
                f"FATAL ERROR: {e}. Please ensure data files exist in a './dataset/' directory relative to your script.")
            exit()

    def _geospatialClustering(self):
        print("--- Performing geospatial clustering ---")
        gdf_zones = gpd.read_file(self.shapefile_path)

        # Project to a planar CRS for accurate centroid calculation. EPSG:2263 for NY.
        gdf_zones_proj = gdf_zones.to_crs("EPSG:2263")
        # Get centroids in the projected CRS, then convert them back to lat/lon (EPSG:4326) for storage.
        gdf_zones['centroid'] = gdf_zones_proj['geometry'].centroid.to_crs(gdf_zones.crs)

        gdf_zones['longitude'] = gdf_zones['centroid'].x
        gdf_zones['latitude'] = gdf_zones['centroid'].y

        # Filter out invalid zones (LocationID > 263 are special zones like airports)
        self.valid_gdf_zones = gdf_zones[gdf_zones['LocationID'] <= 263].copy()
        coordinates = self.valid_gdf_zones[['latitude', 'longitude']].values

        kmeans = KMeans(n_clusters=self.n_zones, random_state=42, n_init=10)
        self.valid_gdf_zones['SuperZone'] = kmeans.fit_predict(coordinates)

        zone_to_superzone_map = self.valid_gdf_zones.set_index('LocationID')['SuperZone'].to_dict()

        self.df_trips['PU_SuperZone'] = self.df_trips['PULocationID'].map(zone_to_superzone_map)
        self.df_trips['DO_SuperZone'] = self.df_trips['DOLocationID'].map(zone_to_superzone_map)

        self.df_trips.dropna(subset=['PU_SuperZone', 'DO_SuperZone'], inplace=True)
        self.df_trips['PU_SuperZone'] = self.df_trips['PU_SuperZone'].astype(int)
        self.df_trips['DO_SuperZone'] = self.df_trips['DO_SuperZone'].astype(int)
        print("--- Clustering complete ---")
        self.zone_centroids = self.valid_gdf_zones.groupby('SuperZone')[
            ['latitude', 'longitude']].mean().sort_index().values


# --- Global Config for RL (IMPROVED) ---
CONFIG = {
    'N_DRIVERS': 50,
    'N_ZONES': 40,
    'TIME_STEPS_PER_DAY': 24,
    'TRIPS_PER_DRIVER_DAY': 25,
    'ACTION_DIM': 41,  # 1 (stay/accept) + 40 (reposition)
    # IMPROVEMENT: State dimension increased to hold more info
    'STATE_DIM': 7,  # [loc, time, local_orders, local_drivers, avg_neighbor_orders, avg_neighbor_drivers, idle_time]
    'HIDDEN_DIM': 128,  # Increased hidden layer size for more complex state
    'LR_ACTOR': 0.0003,
    'LR_CRITIC': 0.001,
    'GAMMA': 0.99,
    # IMPROVEMENT: GAE parameter
    'GAE_LAMBDA': 0.95,
    'K_EPOCHS': 4,
    'EPS_CLIP': 0.2,
    # Base Economics
    'BASE_FARE': 2.5,
    'PRICE_PER_MILE': 1.5,
    # IMPROVEMENT: Extracted "magic numbers" into named constants
    'OPPORTUNITY_COST_PER_STEP': 0.2,  # Cost of being busy (e.g., driving) per time step
    'REPOSITION_COST_PER_MILE': 0.3,  # Cost of repositioning per mile (fuel, wear, etc.)
    'IDLE_REWARD': -0.1,  # Small penalty for being idle
    # IMPROVEMENT: New parameter for environment state representation
    'N_NEIGHBORS': 5  # Number of nearest neighbors to consider in state
}


class PassengerSimulator:
    def __init__(self, df, n_zones, scaling_factor):
        self.df = df
        self.n_zones = n_zones
        self.scaling_factor = scaling_factor
        self.demand_model = {}
        self.transition_model = {}
        self.trip_props_model = {}
        self._learn_distributions()

    def _learn_distributions(self):
        num_days = self.df['pickup_datetime'].dt.date.nunique() or 1
        demand_counts = self.df.groupby(['hour', 'PU_SuperZone']).size() / num_days
        self.demand_model = demand_counts.to_dict()

        transitions = self.df.groupby(['hour', 'PU_SuperZone', 'DO_SuperZone']).size().reset_index(name='trans_count')
        for (hour, origin), group in transitions.groupby(['hour', 'PU_SuperZone']):
            total = group['trans_count'].sum()
            self.transition_model[(hour, origin)] = (group['DO_SuperZone'].values, group['trans_count'].values / total)

        avg_miles = self.df.groupby(['PU_SuperZone', 'DO_SuperZone'])['trip_miles'].mean()
        self.trip_props_model = avg_miles.to_dict()

    def generate_orders(self, time_slot, platform_params):
        all_orders = []
        surge_matrix = platform_params['lambda']

        for zone_id in range(self.n_zones):
            lambda_val = self.demand_model.get((time_slot, zone_id), 0)
            scaled_lambda = lambda_val * self.scaling_factor
            num_potential_requests = np.random.poisson(scaled_lambda)
            if num_potential_requests == 0: continue

            surge_multiplier = surge_matrix[time_slot, zone_id]

            for _ in range(num_potential_requests):
                transition_data = self.transition_model.get((time_slot, zone_id))
                if not transition_data: continue
                dest_zone = np.random.choice(transition_data[0], p=transition_data[1])
                distance = self.trip_props_model.get((zone_id, dest_zone), 2.0)

                base_price = CONFIG['BASE_FARE'] + distance * CONFIG['PRICE_PER_MILE']
                final_price = base_price * surge_multiplier

                valuation = np.random.normal(base_price * 1.2, base_price * 0.2)
                if valuation < final_price:
                    continue

                subsidy = platform_params['subsidy'][time_slot, zone_id]
                driver_income = final_price * (1 - platform_params['commission']) + subsidy

                duration = max(1, int(distance / 10))

                order = {
                    'origin_zone': zone_id, 'dest_zone': int(dest_zone),
                    'distance': round(distance, 2), 'price': round(final_price, 2),
                    'driver_income': round(driver_income, 2), 'subsidy_cost': subsidy,
                    'duration': duration, 'wait_time': 0, 'matched': False
                }
                all_orders.append(order)
        return all_orders


class RideHailingEnv:
    def __init__(self, passenger_simulator, zone_centroids):
        self.simulator = passenger_simulator
        self.n_drivers = CONFIG['N_DRIVERS']
        self.n_zones = CONFIG['N_ZONES']
        self.zone_centroids = zone_centroids

        print("--- Building realistic zone distance and duration matrices ---")
        self._build_zone_matrices(self.simulator.trip_props_model)
        self._find_neighbors()
        print("--- Environment initialization complete ---")

    def _build_zone_matrices(self, trip_props_model):
        self.zone_dist_matrix = np.zeros((self.n_zones, self.n_zones))
        self.zone_duration_matrix = np.zeros((self.n_zones, self.n_zones), dtype=int)

        for i in range(self.n_zones):
            for j in range(self.n_zones):
                if i == j: continue
                dist = trip_props_model.get((i, j))
                if dist is None:
                    # Estimate distance using centroids for pairs not in historical data.
                    # 1 degree latitude is approx 69 miles.
                    lat_dist = (self.zone_centroids[i, 0] - self.zone_centroids[j, 0]) * 69
                    # 1 degree longitude is approx 55 miles at NYC's latitude.
                    lon_dist = (self.zone_centroids[i, 1] - self.zone_centroids[j, 1]) * 55
                    dist = np.sqrt(lat_dist ** 2 + lon_dist ** 2)

                self.zone_dist_matrix[i, j] = dist
                # Duration based on avg speed of 10 miles per time step
                self.zone_duration_matrix[i, j] = max(1, int(dist / 10))

    def _find_neighbors(self):
        self.neighbors = {}
        for i in range(self.n_zones):
            distances = self.zone_dist_matrix[i, :]
            sorted_indices = np.argsort(distances)
            self.neighbors[i] = sorted_indices[1:CONFIG['N_NEIGHBORS'] + 1]

    def reset(self):
        self.time = 0
        self.driver_locations = np.random.randint(0, self.n_zones, size=self.n_drivers)
        self.driver_status = np.zeros(self.n_drivers, dtype=int)
        self.driver_free_time = np.zeros(self.n_drivers, dtype=int)
        self.driver_idle_time = np.zeros(self.n_drivers, dtype=int)

        self.driver_total_income = np.zeros(self.n_drivers)
        self.pending_orders = []
        self.completed_orders_stats = []
        self.platform_profit = 0

        return self._get_state()

    def _get_state(self):
        order_counts = np.bincount([o['origin_zone'] for o in self.pending_orders], minlength=self.n_zones)
        idle_drivers_mask = (self.driver_status == 0)
        idle_driver_counts = np.bincount(self.driver_locations[idle_drivers_mask], minlength=self.n_zones)

        states = np.zeros((self.n_drivers, CONFIG['STATE_DIM']))
        for i in range(self.n_drivers):
            loc = self.driver_locations[i]
            neighbor_zones = self.neighbors[loc]
            avg_neighbor_orders = order_counts[neighbor_zones].mean()
            avg_neighbor_drivers = idle_driver_counts[neighbor_zones].mean()

            states[i] = [
                loc, self.time, order_counts[loc], idle_driver_counts[loc],
                avg_neighbor_orders, avg_neighbor_drivers, self.driver_idle_time[i]
            ]
        return states

    def step(self, actions, platform_params):
        self.driver_free_time[self.driver_free_time > 0] -= 1
        freed_drivers = np.where((self.driver_status > 0) & (self.driver_free_time == 0))[0]
        self.driver_status[freed_drivers] = 0

        self.driver_idle_time[self.driver_status == 0] += 1

        for o in self.pending_orders: o['wait_time'] += 1
        new_orders = self.simulator.generate_orders(self.time, platform_params)
        self.pending_orders.extend(new_orders)

        rewards = np.zeros(self.n_drivers)
        idle_driver_indices = np.where(self.driver_status == 0)[0]
        np.random.shuffle(idle_driver_indices)

        for i in idle_driver_indices:
            action = actions[i]
            current_loc = self.driver_locations[i]

            if action == 0:
                local_orders = [o for o in self.pending_orders if o['origin_zone'] == current_loc and not o['matched']]
                if local_orders:
                    order = local_orders[0]
                    order['matched'] = True
                    rewards[i] = order['driver_income'] - (order['duration'] * CONFIG['OPPORTUNITY_COST_PER_STEP'])

                    self.driver_status[i] = 1
                    self.driver_free_time[i] = order['duration']
                    self.driver_locations[i] = order['dest_zone']
                    self.driver_idle_time[i] = 0
                    self.driver_total_income[i] += order['driver_income']
                    self.platform_profit += (order['price'] * platform_params['commission']) - order['subsidy_cost']
                    self.completed_orders_stats.append(order['wait_time'])
                else:
                    rewards[i] = CONFIG['IDLE_REWARD']
            else:
                target_zone = action - 1
                if 0 <= target_zone < self.n_zones and target_zone != current_loc:
                    dist = self.zone_dist_matrix[current_loc, target_zone]
                    duration = self.zone_duration_matrix[current_loc, target_zone]
                    rewards[i] = -CONFIG['REPOSITION_COST_PER_MILE'] * dist

                    self.driver_status[i] = 1
                    self.driver_free_time[i] = duration
                    self.driver_locations[i] = target_zone
                    self.driver_idle_time[i] = 0
                else:
                    rewards[i] = CONFIG['IDLE_REWARD'] - 0.5

        self.pending_orders = [o for o in self.pending_orders if not o['matched']]
        self.time = (self.time + 1) % CONFIG['TIME_STEPS_PER_DAY']
        done = (self.time == 0)

        return self._get_state(), rewards, done, {}


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh()
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = self.shared_layers(state)
        return self.actor(x), self.critic(x)

    def act(self, state):
        logits, _ = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.detach(), dist.log_prob(action).detach()

    def evaluate(self, state, action):
        logits, values = self.forward(state)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), values, dist.entropy()


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

        with torch.no_grad():
            _, old_values, _ = self.policy_old.evaluate(old_states, old_actions)
            old_values = old_values.squeeze()

        num_steps = len(self.buffer.rewards)
        num_agents = self.buffer.rewards[0].shape[0]
        rewards_reshaped = torch.tensor(np.array(self.buffer.rewards), dtype=torch.float32)
        terminals_reshaped = torch.tensor(np.array(self.buffer.is_terminals), dtype=torch.float32).unsqueeze(1).expand(
            -1, num_agents)
        old_values_reshaped = old_values.view(num_steps, num_agents)

        advantages = torch.zeros_like(rewards_reshaped)
        last_gae_lam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - terminals_reshaped[t]
                next_values = 0
            else:
                next_non_terminal = 1.0 - terminals_reshaped[t]
                next_values = old_values_reshaped[t + 1]

            delta = rewards_reshaped[t] + self.gamma * next_values * next_non_terminal - old_values_reshaped[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam

        advantages = advantages.view(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        value_targets = advantages + old_values.detach()

        for _ in range(self.K_epochs):
            logprobs, values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            values = values.squeeze()
            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(values, value_targets) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)

    def get_weights(self):
        return self.policy.state_dict(), self.policy_old.state_dict()

    def load_by_path(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path))
        self.policy_old.load_state_dict(torch.load(checkpoint_path))

    def load_by_weights(self, weights):
        self.policy.load_state_dict(weights[0])
        self.policy_old.load_state_dict(weights[1])

    def eval(self):
        self.policy.eval()
        self.policy_old.eval()

class Trainer:
    def __init__(self, checkpoint_path='model/agent.pth'):
        dataSetProcesser = DataSetProcesser(n_zones=CONFIG['N_ZONES'])
        self.df_trips = dataSetProcesser.df_trips

        num_days_in_dataset = self.df_trips['pickup_datetime'].dt.date.nunique() or 1
        scaling = (CONFIG['N_DRIVERS'] * CONFIG['TRIPS_PER_DRIVER_DAY']) / (len(self.df_trips) / num_days_in_dataset)

        self.pass_sim = PassengerSimulator(self.df_trips, CONFIG['N_ZONES'], scaling)
        self.env = RideHailingEnv(passenger_simulator=self.pass_sim, zone_centroids=dataSetProcesser.zone_centroids)
        self.agent = SharedPPOAgent(CONFIG['STATE_DIM'], CONFIG['ACTION_DIM'], **CONFIG)
        self.checkpoint_path = checkpoint_path


    def train(self,platform_params,num_episodes=100):
        episode_rewards = []
        print("\n--- Starting RL Training ---")
        for episode in tqdm(range(num_episodes), desc="Training Episodes"):
            state = self.env.reset()
            current_ep_reward = 0

            for t in range(CONFIG['TIME_STEPS_PER_DAY']):
                actions = self.agent.select_actions(state)
                state, rewards, done, _ = self.env.step(actions, platform_params)


                self.agent.buffer.rewards.append(rewards)
                self.agent.buffer.is_terminals.append(done)

                current_ep_reward += np.sum(rewards)

            self.agent.update()
            episode_rewards.append(current_ep_reward)
        return episode_rewards

    def test(self):
        pass


    def visualize_simulation(self, platform_params,filename="img/simulation.gif"):
        """
        Runs a one-day simulation and creates an animated GIF of driver movements.
        """
        print(f"--- Generating simulation visualization to {filename} ---")
        fig, ax = plt.subplots(figsize=(8, 8))
        states_over_time = []

        state = self.env.reset()
        done = False
        while not done:
            # Store a snapshot of current locations and orders
            snapshot = {
                'time': self.env.time,
                'driver_locs': self.env.driver_locations.copy(),
                'driver_status': self.env.driver_status.copy(),
                'order_locs': [o['origin_zone'] for o in self.env.pending_orders]
            }
            states_over_time.append(snapshot)

            actions = self.agent.select_actions(state)
            state, _, done, _ = self.env.step(actions, platform_params)

        # --- Animation function ---
        def animate(i):
            ax.clear()
            snapshot = states_over_time[i]
            time, driver_locs, driver_status, order_locs = snapshot.values()

            # Get zone centroids for plotting
            centroids = self.env.zone_centroids

            # Plot orders
            if order_locs:
                order_coords = centroids[order_locs]
                ax.scatter(order_coords[:, 1], order_coords[:, 0], c='orange', marker='x', s=50, label='Orders')

            # Plot drivers by status
            idle_mask = (driver_status == 0)
            busy_mask = (driver_status == 1)

            idle_coords = centroids[driver_locs[idle_mask]]
            busy_coords = centroids[driver_locs[busy_mask]]

            if len(idle_coords) > 0:
                ax.scatter(idle_coords[:, 1], idle_coords[:, 0], c='blue', marker='o', s=20, label='Idle Drivers')
            if len(busy_coords) > 0:
                ax.scatter(busy_coords[:, 1], busy_coords[:, 0], c='green', marker='^', s=30, label='Busy Drivers')

            ax.set_title(f"Simulation Time Step: {time}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.legend(loc='upper right')
            ax.set_xlim(centroids[:, 1].min() - 0.05, centroids[:, 1].max() + 0.05)
            ax.set_ylim(centroids[:, 0].min() - 0.05, centroids[:, 0].max() + 0.05)

        # Create and save the animation
        ani = animation.FuncAnimation(fig, animate, frames=len(states_over_time), interval=500)
        ani.save(filename, writer='pillow')
        plt.close()
        print("--- Visualization saved. ---")

    def save(self):
        self.agent.save(self.checkpoint_path)

    def draw(self, episode_rewards):
        print("--- Training Finished ---")
        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards)
        plt.title("Total Reward per Episode for All Drivers")
        plt.xlabel("Episode")
        plt.ylabel("Total Episode Reward")
        plt.grid(True)
        if not os.path.exists('img'):
            os.makedirs('img')
        plt.savefig('img/improved_reward_curve.png')
        print("Learning curve saved to 'img/improved_reward_curve.png'")


if __name__ == '__main__':
    platform_params = {
        'commission': 0.2,
        'lambda': np.ones((CONFIG['TIME_STEPS_PER_DAY'], CONFIG['N_ZONES'])) * 1.5,
        'subsidy': np.ones((CONFIG['TIME_STEPS_PER_DAY'], CONFIG['N_ZONES'])) * 1.0
    }
    #
    # ##训练##
    # trainer = Trainer()
    # episode_rewards= trainer.train(platform_params,num_episodes=100)
    # trainer.save()
    # trainer.draw(episode_rewards)

    # 可视化
    trainer = Trainer()
    trainer.agent.load_by_path('model/agent.pth')
    trainer.agent.eval()
    trainer.visualize_simulation(platform_params)

