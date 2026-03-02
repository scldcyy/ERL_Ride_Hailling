import copy
import pickle
import h3
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from basic_config import CONFIG, DRIVER_CONFIG
from generate_simulator import PassengerSimulator


class RideHailingEnv:
    def __init__(self, simulator_path, fixed_drivers=True, fixed_sim=True):
        self.time = 0
        with open(simulator_path, 'rb') as f:
            self.simulator = pickle.load(f)
        self.fixed_drivers = fixed_drivers
        self.fixed_sim = fixed_sim

        self.all_hexes = list(self.simulator.adjacency.keys())
        self.n_zones = len(self.all_hexes)
        self.hex_to_idx = {h: i for i, h in enumerate(self.all_hexes)}
        self.idx_to_hex = {i: h for i, h in enumerate(self.all_hexes)}

        # 缓存：邻接表索引与经纬度映射 (向量化提速)
        self.adjacency_indices = {}
        self.hex_latlng_arr = np.zeros((self.n_zones, 2))
        for h_id, neighbors in self.simulator.adjacency.items():
            idx = self.hex_to_idx[h_id]
            self.adjacency_indices[idx] = {direct: self.hex_to_idx[n_h_id] for direct, n_h_id in neighbors.items()}
            self.hex_latlng_arr[idx] = h3.cell_to_latlng(h_id)

        # 预计算邻居数组以供快速查询，填充 -1 表示无邻居
        self.neighbor_matrix = np.full((self.n_zones, 6), -1, dtype=int)
        for idx, n_dict in self.adjacency_indices.items():
            for direct, n_idx in n_dict.items():
                self.neighbor_matrix[idx, direct] = n_idx

        self.total_revenue = 0
        self.total_served_orders = 0
        self.total_generated_orders = 0
        self.total_wait_time = 0
        self.pending_orders = []
        self.current_scenario_idx = 0

        # 司机状态
        self.driver_free_time = None
        self.driver_locations = None
        self.driver_daily_income = None
        self.driver_status = None  # 0=空闲, 1=忙, 2=下线
        self.driver_active_steps = None
        self.driver_type = np.random.choice(
            [0, 1], size=CONFIG['N_DRIVERS'],
            p=[1 - DRIVER_CONFIG['FULL_TIME_RATIO'], DRIVER_CONFIG['FULL_TIME_RATIO']]
        )

        # 追踪供需比EMA平滑
        self.sd_ema = np.zeros(self.n_zones)

    def reset(self):
        self.time = 0
        if not self.fixed_drivers:
            self.current_scenario_idx += 1
        if not self.fixed_sim:
            self.simulator.reset_fixed_orders(self.current_scenario_idx)

        rng = np.random.RandomState(42 + self.current_scenario_idx)
        self.driver_locations = rng.randint(0, self.n_zones, size=CONFIG['N_DRIVERS'])
        self.driver_status = np.zeros(CONFIG['N_DRIVERS'], dtype=int)
        self.driver_free_time = np.zeros(CONFIG['N_DRIVERS'], dtype=int)
        self.driver_daily_income = np.zeros(CONFIG['N_DRIVERS'])
        self.driver_active_steps = np.zeros(CONFIG['N_DRIVERS'])

        self.total_revenue = 0
        self.total_served_orders = 0
        self.total_generated_orders = 0
        self.total_wait_time = 0
        self.pending_orders = []
        self.sd_ema = np.zeros(self.n_zones)

        return self._get_state_and_mask(surge=np.ones(self.n_zones), subsidy=np.zeros(self.n_zones))

    def _get_state_and_mask(self, surge, subsidy):
        """核心改造 2.1 与 1.2: 空间感知场重构与动作掩码提取"""
        order_counts = np.zeros(self.n_zones)
        for o in self.pending_orders:
            if not o['matched']:
                order_counts[o['origin_idx']] += 1

        idle_mask = self.driver_status == 0
        idle_driver_counts = np.bincount(self.driver_locations[idle_mask], minlength=self.n_zones)

        # 动态更新 SD EMA
        current_sd = order_counts / (idle_driver_counts + 1e-6)
        self.sd_ema = 0.8 * self.sd_ema + 0.2 * current_sd

        # 状态维度预估: 中心点(6) + 6邻居(6*6) + 司机属性(3) = 45维
        # 特征: [lat, lng, orders, drivers, surge, subsidy]
        states = np.zeros((CONFIG['N_DRIVERS'], 45))
        action_masks = np.zeros((CONFIG['N_DRIVERS'], CONFIG['ACTION_DIM']), dtype=bool)

        # 提取全局网格特征以加速查询
        grid_features = np.column_stack([
            self.hex_latlng_arr[:, 0] / 90.0,
            self.hex_latlng_arr[:, 1] / 180.0,
            order_counts / 50.0,
            idle_driver_counts / 50.0,
            surge / 5.0,
            subsidy / 20.0
        ])
        self.cached_grid_features = grid_features  # <--- 新增这行，缓存供 Critic 查询
        dummy_feature = np.full(6, -1.0)

        for i in range(CONFIG['N_DRIVERS']):
            # 状态 1: 司机忙碌或下线，掩码全关，仅留动作1(原地)用于占位，屏蔽无效梯度
            if self.driver_status[i] != 0:
                action_masks[i, 1] = True
                continue

            loc_idx = self.driver_locations[i]

            # --- 构建空间感知场 (Flattened Neighbors) ---
            center_feat = grid_features[loc_idx]
            spatial_features = [center_feat]

            for direct in range(6):
                n_idx = self.neighbor_matrix[loc_idx, direct]
                if n_idx != -1:
                    spatial_features.append(grid_features[n_idx])
                    action_masks[i, direct + 2] = True  # 存在该邻居，开放移动动作
                else:
                    spatial_features.append(dummy_feature)

            flat_spatial = np.concatenate(spatial_features)

            # --- 动作掩码计算 ---
            action_masks[i, 1] = True  # 永远可以原地等待

            # 【修复核心 2】：删掉 if order_counts > 0 的判断。
            # 必须永远允许输出动作 0，以接住 step 内部刚刚刷新的新订单！
            action_masks[i, 0] = True

            # --- 个人特征 ---
            income_progress = self.driver_daily_income[i] / DRIVER_CONFIG['PART_TIME_INCOME_TARGET']
            time_prog = self.time / CONFIG['TIME_STEPS_PER_DAY']
            personal_feat = np.array([float(self.driver_type[i]), income_progress, time_prog])

            states[i] = np.concatenate([flat_spatial, personal_feat])

        return states, action_masks

    def compile(self, platform_params):
        """核心改造 1.4: 溢价/补贴截断与平滑处理"""
        order_counts = np.zeros(self.n_zones)
        for o in self.pending_orders:
            if not o['matched']: order_counts[o['origin_idx']] += 1
        idle_mask = self.driver_status == 0
        idle_driver_counts = np.bincount(self.driver_locations[idle_mask], minlength=self.n_zones)

        sd = order_counts / (idle_driver_counts + 1e-6)
        surge = platform_params['surge'](self.time / CONFIG['TIME_STEPS_PER_DAY'], order_counts, idle_driver_counts, sd)
        subsidy = platform_params['subsidy'](self.time / CONFIG['TIME_STEPS_PER_DAY'], order_counts, idle_driver_counts,
                                             sd)

        # NaN与溢出保护
        surge = np.clip(np.nan_to_num(surge, nan=1.0, posinf=5.0), CONFIG['MIN_SURGE'], CONFIG['MAX_SURGE'])
        subsidy = np.clip(np.nan_to_num(subsidy, nan=0.0, posinf=20.0), CONFIG['MIN_SUBSIDY'], CONFIG['MAX_SUBSIDY'])

        return surge, subsidy

    def step(self, actions, platform_params, value_estimator=None):
        """
        核心改造 1.1 与 2.2: 运筹学二分图匹配与遗憾奖励计算
        value_estimator: 接收状态预测未来价值的函数句柄，用于计算带权匹配
        """
        surge, subsidy = self.compile(platform_params)
        commission = DRIVER_CONFIG['PLATFORM_COMMISSION_RATIO']
        step_revenue, step_subsidy_cost = 0, 0.0
        zone_profits = np.zeros(self.n_zones)

        self._update_driver_online_status(surge, subsidy)

        self.driver_free_time[self.driver_free_time > 0] -= 1
        freed_drivers = np.where((self.driver_status == 1) & (self.driver_free_time == 0))[0]
        self.driver_status[freed_drivers] = 0
        self.driver_active_steps[self.driver_status == 1] += 1

        raw_orders = self.simulator.get_fixed_orders(self.time)
        self.total_generated_orders += len(raw_orders)

        # 订单向量化提速
        new_orders = [o for o in raw_orders if o['origin_hex'] in self.hex_to_idx and o['dest_hex'] in self.hex_to_idx]
        for o in new_orders:
            o.update({'origin_idx': self.hex_to_idx[o['origin_hex']], 'dest_idx': self.hex_to_idx[o['dest_hex']],
                      'matched': False, 'wait_time': 0})

        surviving_orders = []
        for o in self.pending_orders:
            if not o['matched'] and o['wait_time'] < 4:
                if np.random.rand() > 0.05 * surge[o['origin_idx']]:
                    o['wait_time'] += 1
                    self.total_wait_time += 1
                    surviving_orders.append(o)
        self.pending_orders = surviving_orders + new_orders

        rewards = np.zeros(CONFIG['N_DRIVERS'])

        # --- 核心改造 1.1: 全局二分图匹配 ---
        intent_match_drivers = [i for i in range(CONFIG['N_DRIVERS']) if self.driver_status[i] == 0 and actions[i] == 0]

        # 按区域分组进行匹配

        zone_intent_drivers = defaultdict(list)
        for d_id in intent_match_drivers:
            zone_intent_drivers[self.driver_locations[d_id]].append(d_id)

        for loc, d_list in zone_intent_drivers.items():
            local_orders = [o for o in self.pending_orders if o['origin_idx'] == loc and not o['matched']]
            if not local_orders:
                for d_id in d_list: rewards[d_id] = CONFIG['IDLE_REWARD']
                continue

            n_drivers = len(d_list)
            n_orders = len(local_orders)
            cost_matrix = np.zeros((n_drivers, n_orders))

            # ---------------- 核心改造：Batch 构建与推理 ----------------
            if value_estimator is not None:
                query_d_ids = []
                query_dest_idxs = []
                # 展平所有组合以构建 Batch
                for d_id in d_list:
                    for o in local_orders:
                        query_d_ids.append(d_id)
                        query_dest_idxs.append(o['dest_idx'])

                # 一次性推理出该区域所有的未来价值
                future_vals_flat = value_estimator(query_d_ids, query_dest_idxs)
                future_vals_matrix = future_vals_flat.reshape(n_drivers, n_orders)
            else:
                future_vals_matrix = np.zeros((n_drivers, n_orders))
            # ------------------------------------------------------------

            # 计算收益权重矩阵
            for i, d_id in enumerate(d_list):
                for j, o in enumerate(local_orders):
                    trip_steps = max(1, int(o['duration']))
                    base_fare = CONFIG['BASE_FARE'] + trip_steps * CONFIG['PRICE_PER_MINUTE'] * CONFIG[
                        'TIME_STEP_MINUTES']
                    price = surge[loc] * base_fare
                    fuel = trip_steps * DRIVER_CONFIG['FUEL_COST_PER_STEP']
                    immediate_reward = (1 - commission) * price - fuel + subsidy[loc]

                    # 直接从刚才批量计算好的矩阵中查表
                    future_val = future_vals_matrix[i, j]

                    # KM算法求最小成本匹配，因此权重取负
                    cost_matrix[i, j] = -(immediate_reward + 0.99 * future_val)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)


            # 结算匹配成功的订单
            for i, j in zip(row_ind, col_ind):
                d_id = d_list[i]
                order = local_orders[j]
                order['matched'] = True
                self.total_served_orders += 1

                trip_steps = max(1, int(order['duration']))
                base_fare = CONFIG['BASE_FARE'] + trip_steps * CONFIG['PRICE_PER_MINUTE'] * CONFIG['TIME_STEP_MINUTES']
                price = surge[loc] * base_fare
                driver_income = (1 - commission) * price - trip_steps * DRIVER_CONFIG['FUEL_COST_PER_STEP'] + subsidy[
                    loc]
                actual_income = max(driver_income, CONFIG['MIN_FARE_THRESHOLD'])

                gap_subsidy = actual_income - driver_income
                step_subsidy_cost += (gap_subsidy + subsidy[loc])
                zone_profits[loc] += price * commission - (gap_subsidy + subsidy[loc])

                rewards[d_id] = actual_income
                self.driver_status[d_id] = 1
                self.driver_free_time[d_id] = trip_steps
                self.driver_locations[d_id] = order['dest_idx']
                self.driver_daily_income[d_id] += actual_income
                step_revenue += price

            # 未匹配上的司机给予空闲惩罚
            unmatched = set(range(n_drivers)) - set(row_ind)
            for i in unmatched: rewards[d_list[i]] = CONFIG['IDLE_REWARD']

        # 处理非接单动作 (移动/原地)
        for i in range(CONFIG['N_DRIVERS']):
            if self.driver_status[i] == 0 and actions[i] != 0:
                action = actions[i]
                if action == 1:
                    rewards[i] = CONFIG['IDLE_REWARD']
                else:
                    target_direct = action - 2
                    n_idx = self.neighbor_matrix[self.driver_locations[i], target_direct]
                    if n_idx != -1:
                        rewards[i] = -DRIVER_CONFIG['FUEL_COST_PER_STEP']
                        self.driver_status[i] = 1
                        self.driver_free_time[i] = 1
                        self.driver_locations[i] = n_idx
                    else:
                        rewards[i] = CONFIG['IDLE_REWARD']

        # --- 核心改造 2.2: 遗憾奖励 (Regret Reward) 重塑 ---
        # 鼓励合作并引导向高价值洼地流动
        zone_income_sums = np.zeros(self.n_zones)
        zone_driver_counts = np.zeros(self.n_zones)
        for i in range(CONFIG['N_DRIVERS']):
            zone_income_sums[self.driver_locations[i]] += rewards[i]
            zone_driver_counts[self.driver_locations[i]] += 1

        zone_avg_income = np.divide(zone_income_sums, zone_driver_counts, out=np.zeros_like(zone_income_sums),
                                    where=zone_driver_counts != 0)

        # 结合时间衰减，越早完成高价值订单奖励越大
        alpha = 0.2 * (1.0 - self.time / CONFIG['TIME_STEPS_PER_DAY'])
        for i in range(CONFIG['N_DRIVERS']):
            local_avg = zone_avg_income[self.driver_locations[i]]
            rewards[i] = rewards[i] + alpha * (rewards[i] - local_avg)

        step_profit = step_revenue * commission - step_subsidy_cost
        self.total_revenue += step_profit
        self.time += 1
        done = (self.time >= CONFIG['TIME_STEPS_PER_DAY'])

        surge, subsidy = self.compile(platform_params)
        next_state, next_mask = self._get_state_and_mask(surge, subsidy)

        # --- 核心修复：清理已匹配的订单，防止历史脏数据无限堆积拖慢向量化运算 ---
        self.pending_orders = [o for o in self.pending_orders if not o['matched']]
        step_profit = step_revenue * commission - step_subsidy_cost

        info = {
            'step_profit': step_profit,
            'zone_profits': zone_profits,
            'total_revenue': self.total_revenue,
            'total_served': self.total_served_orders,
            'total_generated': self.total_generated_orders,
            'total_wait_time': self.total_wait_time,
            'online_drivers': np.sum(self.driver_status < 2),
            'driver_income': self.driver_daily_income.copy(),
            'driver_income_rate': self.driver_daily_income / (self.driver_active_steps + 1e-6),
            'action_mask': next_mask,
            'driver_status': self.driver_status.copy()  # 用于GAE截断
        }

        return next_state, rewards, done, info

    def _update_driver_online_status(self, surge, subsidy):
        """核心改造 1.3: 精细化司机异质性上下线"""
        for i in range(CONFIG['N_DRIVERS']):
            loc = self.driver_locations[i]
            if self.driver_status[i] == 2:
                if self.driver_type[i] == 0 and self.driver_daily_income[i] >= DRIVER_CONFIG['PART_TIME_INCOME_TARGET']:
                    continue
                # 结合EMA平滑的SD计算抢单概率
                local_sd = self.sd_ema[loc]
                order_probability = min(1.0, local_sd)
                expected_income = order_probability * (surge[loc] * CONFIG['BASE_FARE'] + subsidy[loc])

                if expected_income > DRIVER_CONFIG['ONLINE_THRESHOLD']:
                    self.driver_status[i] = 0
                # # 改为: 只要没赚够目标，无条件尝试上线 (或者把阈值设为 -1)
                # if expected_income > -1.0:
                #     self.driver_status[i] = 0
            else:
                # 兼职达标下线
                if self.driver_status[i] == 0 and self.driver_type[i] == 0 and self.driver_daily_income[i] >= \
                        DRIVER_CONFIG['PART_TIME_INCOME_TARGET']:
                    self.driver_status[i] = 2
                # 引入全职止损线逻辑 (需在DRIVER_CONFIG中配置 FULL_TIME_MIN_THRESHOLD)
                elif self.driver_status[i] == 0 and self.driver_type[i] == 1:
                    recent_income_rate = self.driver_daily_income[i] / (self.driver_active_steps[i] + 1e-6)
                    # 如果预期收益持续低于阈值，全职司机也会大概率下线休息
                    if recent_income_rate < DRIVER_CONFIG.get('FULL_TIME_MIN_THRESHOLD',
                                                              0.1) and np.random.rand() < 0.2:
                        self.driver_status[i] = 2

    def get_proxy_state(self, driver_idx, dest_idx):
        """为 Critic 提供目标位置的伪状态，用于预估未来的博弈价值"""
        if not hasattr(self, 'cached_grid_features'):
            return np.zeros(CONFIG['STATE_DIM'])

        center_feat = self.cached_grid_features[dest_idx]
        spatial_features = [center_feat]

        for direct in range(6):
            n_idx = self.neighbor_matrix[dest_idx, direct]
            if n_idx != -1:
                spatial_features.append(self.cached_grid_features[n_idx])
            else:
                spatial_features.append(np.full(6, -1.0))

        flat_spatial = np.concatenate(spatial_features)

        # 拼接该司机的当前个人特征
        income_progress = self.driver_daily_income[driver_idx] / DRIVER_CONFIG['PART_TIME_INCOME_TARGET']
        time_prog = self.time / CONFIG['TIME_STEPS_PER_DAY']
        personal_feat = np.array([float(self.driver_type[driver_idx]), income_progress, time_prog])

        return np.concatenate([flat_spatial, personal_feat])

    def render(self):
        # 可视化当前时间步的环境 TODO
        pass

if __name__ == '__main__':
    env = RideHailingEnv('generator/simulator_driver_nums=400_hex_scaling=0.017031373249357308_weekday.pkl')
    env.reset()
    platform_params = {
        # Add 1e-6 to avoid log(0)
        'surge': lambda t, no, nd, sd: 1 + 0.5 * np.log(sd + 1),
        # Use sd directly instead of no/nd, and add 1e-6
        'subsidy': lambda t, no, nd, sd: 1 + 0.5 * np.log(sd + 1) * np.sin(t)
    }
    for i in range(100):
        action = np.random.randint(0, 8, size=CONFIG['N_DRIVERS'])
        state, reward, done, info = env.step(action, platform_params)
        print(f"step: {i}, reward: {reward}, done: {done}, info: {info}")