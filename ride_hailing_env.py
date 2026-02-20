import copy
import pickle

import h3
import numpy as np

from basic_config import CONFIG, DRIVER_CONFIG


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

        # --- 新增：司机异质性初始化 ---
        self.driver_type = np.random.choice(
            [0, 1], size=CONFIG['N_DRIVERS'],
            p=[1 - DRIVER_CONFIG['FULL_TIME_RATIO'], DRIVER_CONFIG['FULL_TIME_RATIO']]
        )  # 0=兼职, 1=全职
        self.driver_daily_income = np.zeros(CONFIG['N_DRIVERS'])  # 司机当日累计收入
        self.driver_online = np.ones(CONFIG['N_DRIVERS'], dtype=bool)  # 司机是否在线

    def reset(self):
        self.time = 0
        # 固定种子以保证每个Scenario内的初始位置一致
        rng = np.random.RandomState(42 + self.current_scenario_idx)
        self.driver_locations = rng.randint(0, self.n_zones, size=CONFIG['N_DRIVERS'])
        self.driver_status = np.zeros(CONFIG['N_DRIVERS'], dtype=int)  # 0=空闲, 1=忙
        self.driver_free_time = np.zeros(CONFIG['N_DRIVERS'], dtype=int)

        # --- 重置司机异质性相关状态 ---
        self.driver_daily_income = np.zeros(CONFIG['N_DRIVERS'])
        self.driver_online = np.ones(CONFIG['N_DRIVERS'], dtype=bool)

        self.total_revenue = 0
        self.total_served_orders = 0
        self.total_generated_orders = 0
        self.total_wait_time = 0
        self.pending_orders = []

        if self.fixed_scenarios is not None:
            self.current_day_orders = self.fixed_scenarios[self.current_scenario_idx % len(self.fixed_scenarios)]
            self.current_scenario_idx += 1

        return self._get_state()

    def _get_state(self, surge=None,subsidy=None):
        order_counts = np.zeros(self.n_zones)
        for o in self.pending_orders:
            if not o['matched']:
                order_counts[o['origin_idx']] += 1

        # 仅统计在线且空闲的司机
        idle_mask = (self.driver_status == 0) & self.driver_online
        idle_driver_counts = np.bincount(self.driver_locations[idle_mask], minlength=self.n_zones)

        if surge is None or subsidy is None:
            surge = np.ones(self.n_zones)
            subsidy = np.zeros(self.n_zones)

        states = np.zeros((CONFIG['N_DRIVERS'], CONFIG['STATE_DIM']))
        # 获取六边形网格的经纬度映射
        hex_latlng = {h: h3.cell_to_latlng(h) for h in self.all_hexes}  # (lat, lng)

        for i in range(CONFIG['N_DRIVERS']):
            if not self.driver_online[i]:
                states[i] = np.zeros(CONFIG['STATE_DIM'])
                continue

            loc_idx = self.driver_locations[i]
            loc_hex = self.idx_to_hex[loc_idx]
            lat, lng = hex_latlng[loc_hex]
            neighbors = self.adjacency_indices.get(loc_idx, {})

            # 计算邻居区域的平均订单/司机数
            if neighbors:
                n_locs = list(neighbors.values())
                avg_n_orders = order_counts[n_locs].mean()
                avg_n_drivers = idle_driver_counts[n_locs].mean()
            else:
                avg_n_orders = 0
                avg_n_drivers = 0

            # 严格按照文档定义的10维状态填充
            states[i] = [
                lat / 90.0,  # 纬度归一化到[0,1]
                lng / 180.0,  # 经度归一化到[0,1] #时间系数
                self.time / CONFIG['TIME_STEPS_PER_DAY'],
                order_counts[loc_idx],  # 当前区域订单数
                idle_driver_counts[loc_idx],  # 当前区域司机数
                avg_n_orders,  # 邻居平均订单数
                avg_n_drivers,  # 邻居平均司机数
                self.driver_free_time[i] / CONFIG['TIME_STEPS_PER_DAY'],  # 空驶时间归一化
                surge[loc_idx],  # 溢价系数λ
                subsidy[loc_idx]  # 补贴
            ]
        return states

    def compile(self,platform_params):
        order_counts,idle_driver_counts=self.get_global_observation()
        surge_function = platform_params['surge']
        subsidy_function = platform_params['subsidy']
        surge=surge_function(self.time/CONFIG['TIME_STEPS_PER_DAY'], order_counts, idle_driver_counts, order_counts/(idle_driver_counts+1e-6))
        subsidy = subsidy_function(self.time/CONFIG['TIME_STEPS_PER_DAY'], order_counts, idle_driver_counts, order_counts/(idle_driver_counts+1e-6))

        return surge,subsidy


    def get_global_observation(self):
        order_counts = np.zeros(self.n_zones)
        for o in self.pending_orders:
            if not o['matched']: order_counts[o['origin_idx']] += 1

        idle_mask = (self.driver_status == 0) & self.driver_online
        idle_driver_counts = np.bincount(self.driver_locations[idle_mask], minlength=self.n_zones)

        # 归一化
        obs = np.stack([
            order_counts / (order_counts.max() + 1e-6),
            idle_driver_counts / (idle_driver_counts.max() + 1e-6),
        ])
        return obs

    def step(self, actions, platform_params):
        # 解析platform_params=(λ，subsidy, η)
        surge,subsidy=self.compile(platform_params)
        commission=DRIVER_CONFIG['PLATFORM_COMMISSION_RATIO']

        step_revenue = 0

        # --- 1. 处理司机上下线逻辑 ---
        self._update_driver_online_status(surge,subsidy)

        # --- 2. 更新司机忙闲状态 ---
        self.driver_free_time[self.driver_free_time > 0] -= 1
        freed_drivers = np.where((self.driver_status == 1) & (self.driver_free_time == 0))[0]
        self.driver_status[freed_drivers] = 0

        # --- 3. 生成新订单 ---
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

        # --- 4. 更新待匹配订单 ---
        self.pending_orders = [o for o in self.pending_orders if not o['matched'] and o['wait_time'] < 3]
        for o in self.pending_orders:
            o['wait_time'] += 1
            self.total_wait_time += 1
        self.pending_orders.extend(new_orders)

        # --- 5. 处理司机动作与收益 ---
        rewards = np.zeros(CONFIG['N_DRIVERS'])
        step_subsidy_cost = 0.0

        # 仅处理在线且空闲的司机
        idle_indices = np.where((self.driver_status == 0) & self.driver_online)[0]
        np.random.shuffle(idle_indices)

        for i in idle_indices:
            action = actions[i]
            current_loc = self.driver_locations[i]
            current_surge = surge[current_loc]
            current_subsidy = subsidy[current_loc]

            if action == 0:  # 接单
                local_orders = [o for o in self.pending_orders if o['origin_idx'] == current_loc and not o['matched']]
                if local_orders:
                    # 匹配订单
                    order = local_orders[0]
                    order['matched'] = True
                    self.total_served_orders += 1
                    trip_steps = max(1, int(order['duration']))

                    # 文档定义的价格公式：price=λ*(Fbase+α*duration)
                    base_fare = CONFIG['BASE_FARE'] + trip_steps * CONFIG['PRICE_PER_MINUTE']*CONFIG['TIME_STEP_MINUTES']
                    price = current_surge * base_fare  # 订单基础价格

                    # 司机收益计算（文档定义）
                    fuel_cost = trip_steps * DRIVER_CONFIG['FUEL_COST_PER_STEP']
                    driver_income = (1 - commission) * price - fuel_cost + current_subsidy

                    # 保证最低收入阈值
                    actual_income = max(driver_income, CONFIG['MIN_FARE_THRESHOLD'])
                    gap_subsidy = actual_income - driver_income
                    step_subsidy_cost += (gap_subsidy + current_subsidy)

                    # 更新司机状态
                    rewards[i] = actual_income
                    self.driver_status[i] = 1
                    self.driver_free_time[i] = trip_steps
                    self.driver_locations[i] = order['dest_idx']
                    self.driver_daily_income[i] += actual_income
                    step_revenue += price  # 平台原始收入（未扣补贴）

                else:
                    # 无订单可接：空闲奖励
                    rewards[i] = CONFIG['IDLE_REWARD']
            else:  # 重定位
                target_direct = action - 2
                neighbors = self.adjacency_indices.get(current_loc, {})
                if target_direct in neighbors:
                    # 行驶收益：-c*Δt
                    rewards[i] = -DRIVER_CONFIG['FUEL_COST_PER_STEP']
                    self.driver_status[i] = 1
                    self.driver_free_time[i] = 1
                    self.driver_locations[i] = neighbors[target_direct]
                else:
                    rewards[i] = CONFIG['IDLE_REWARD']  # 不动且不接单

        # --- 6. 计算平台利润 ---
        step_profit = step_revenue * commission - step_subsidy_cost
        self.total_revenue += step_profit

        self.time += 1
        done = (self.time >= CONFIG['TIME_STEPS_PER_DAY'])
        next_state = self._get_state(surge,subsidy)

        info = {
            'step_profit': step_profit,
            'total_revenue': self.total_revenue,
            'total_served': self.total_served_orders,
            'total_generated': self.total_generated_orders,
            'total_wait_time': self.total_wait_time,
            'online_drivers': np.sum(self.driver_online),
            'driver_income': self.driver_daily_income.copy()
        }

        return next_state, rewards, done, info

    def _update_driver_online_status(self, surge,subsidy):
        """更新司机上下线状态：兼职司机达到收入目标下线，低收益时司机不上线"""
        for i in range(CONFIG['N_DRIVERS']):
            if not self.driver_online[i]:
                # 已下线司机：判断是否重新上线（预期收益>阈值）
                loc = self.driver_locations[i]
                cur_surge = surge[loc]
                cur_subsidy = subsidy[loc]
                # 简单预期收益：当前区域订单密度*单位收益
                order_density = len([o for o in self.pending_orders if o['origin_idx'] == loc]) / self.n_zones
                expected_income = order_density * (cur_surge * CONFIG['BASE_FARE'] + cur_subsidy)
                if expected_income > DRIVER_CONFIG['ONLINE_THRESHOLD']:
                    self.driver_online[i] = True
                continue

            # 在线司机：兼职司机达到收入目标则下线
            if self.driver_type[i] == 0 and self.driver_daily_income[i] >= DRIVER_CONFIG['PART_TIME_INCOME_TARGET']:
                self.driver_online[i] = False
                self.driver_status[i] = 0  # 强制设为空闲
                self.driver_free_time[i] = 0