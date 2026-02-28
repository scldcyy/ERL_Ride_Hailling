import copy
import pickle

import h3
import numpy as np

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

        self.adjacency_indices = {}
        for h_id, neighbors in self.simulator.adjacency.items():
            idx = self.hex_to_idx[h_id]
            n_indices = {direct: self.hex_to_idx[n_h_id] for direct, n_h_id in neighbors.items()}
            self.adjacency_indices[idx] = n_indices

        self.total_revenue = 0
        self.total_served_orders = 0
        self.total_generated_orders = 0
        self.total_wait_time = 0
        self.pending_orders = []
        self.current_scenario_idx = 0

        # --- 司机初始化 ---
        self.driver_free_time = None    # 司机累计空闲步数
        self.driver_locations = None    # 司机当前位置, 索引为h_id
        self.driver_daily_income = None  # 司机当日累计收入
        self.driver_status = None  # 0=空闲, 1=忙, 2=下线
        self.driver_active_steps = None    # 司机累计活跃步数
        self.driver_type = np.random.choice(
            [0, 1], size=CONFIG['N_DRIVERS'],
            p=[1 - DRIVER_CONFIG['FULL_TIME_RATIO'], DRIVER_CONFIG['FULL_TIME_RATIO']]
        )  # 0=兼职, 1=全职

    def reset(self):
        self.time = 0
        if not  self.fixed_drivers:
            self.current_scenario_idx += 1
        if not  self.fixed_sim:
            self.simulator.reset_fixed_orders(self.current_scenario_idx)
        # 固定种子以保证每个Scenario内的初始位置一致
        rng = np.random.RandomState(42 + self.current_scenario_idx)
        self.driver_locations = rng.randint(0, self.n_zones, size=CONFIG['N_DRIVERS'])
        self.driver_status = np.zeros(CONFIG['N_DRIVERS'], dtype=int)  # 0=空闲, 1=忙
        self.driver_free_time = np.zeros(CONFIG['N_DRIVERS'], dtype=int)

        # --- 重置司机异质性相关状态 ---
        self.driver_daily_income = np.zeros(CONFIG['N_DRIVERS'])
        self.driver_active_steps = np.zeros(CONFIG['N_DRIVERS'])

        self.total_revenue = 0
        self.total_served_orders = 0
        self.total_generated_orders = 0
        self.total_wait_time = 0
        self.pending_orders = []

        return self._get_state(surge=np.ones(self.n_zones),subsidy=np.zeros(self.n_zones))

    def _get_state(self, surge,subsidy):
        order_counts = np.zeros(self.n_zones)
        for o in self.pending_orders:
            if not o['matched']:
                order_counts[o['origin_idx']] += 1

        # 仅统计空闲的司机
        idle_mask = self.driver_status == 0
        idle_driver_counts = np.bincount(self.driver_locations[idle_mask], minlength=self.n_zones)

        states = np.zeros((CONFIG['N_DRIVERS'], CONFIG['STATE_DIM']))
        # 获取六边形网格的经纬度映射
        hex_latlng = {h: h3.cell_to_latlng(h) for h in self.all_hexes}  # (lat, lng)

        for i in range(CONFIG['N_DRIVERS']):
            if self.driver_status[i] != 0:
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

                # --- 新增：计算收入完成度 ---
                # 用兼职司机的目标收入作为统一的归一化基准
            income_progress = self.driver_daily_income[i] / DRIVER_CONFIG['PART_TIME_INCOME_TARGET']

            # 扩展为12维状态填充
            states[i] = [
                lat / 90.0,
                lng / 180.0,
                self.time / CONFIG['TIME_STEPS_PER_DAY'],
                order_counts[loc_idx]/ 50.0,
                idle_driver_counts[loc_idx]/ 50.0,
                avg_n_orders/ 50.0,
                avg_n_drivers/ 50.0,
                self.driver_free_time[i] / CONFIG['TIME_STEPS_PER_DAY'],
                surge[loc_idx],
                subsidy[loc_idx],
                float(self.driver_type[i]),  # 司机类型: 0=兼职, 1=全职
                income_progress  # 收入进度
            ]
        return states

    def compile(self, platform_params):
        order_counts, idle_driver_counts = self.get_global_observation()
        surge_function = platform_params['surge']
        subsidy_function = platform_params['subsidy']
        min_surge=CONFIG['MIN_SURGE']
        max_surge=CONFIG['MAX_SURGE']
        min_subsidy=CONFIG['MIN_SUBSIDY']
        max_subsidy=CONFIG['MAX_SUBSIDY']

        # Calculate supply-demand ratio safely
        sd = order_counts / (idle_driver_counts + 1e-6)

        surge = surge_function(self.time / CONFIG['TIME_STEPS_PER_DAY'], order_counts, idle_driver_counts, sd)
        subsidy = subsidy_function(self.time / CONFIG['TIME_STEPS_PER_DAY'], order_counts, idle_driver_counts, sd)

        # --- 新增：为 GP 算法添加安全网 ---
        # 1. 处理由非法公式产生的 nan 或 inf
        surge = np.nan_to_num(surge, nan=min_surge, posinf=max_surge, neginf=min_surge)
        subsidy = np.nan_to_num(subsidy, nan=min_subsidy, posinf=max_subsidy, neginf=min_subsidy)

        # 2. 截断极值，防止溢价和补贴过大导致环境崩溃
        surge = np.clip(surge, min_surge, max_surge)  # 溢价至少为1，最高限制为5倍
        subsidy = np.clip(subsidy, min_subsidy, max_subsidy)  # 补贴不能为负，设置一个合理上限

        return surge, subsidy

    def get_global_observation(self):
        order_counts = np.zeros(self.n_zones)
        for o in self.pending_orders:
            if not o['matched']: order_counts[o['origin_idx']] += 1

        idle_mask = self.driver_status == 0
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
        zone_profits = np.zeros(self.n_zones)  # --- 新增 1：初始化网格局部利润矩阵 ---

        # --- 1. 处理司机上下线逻辑 ---
        self._update_driver_online_status(surge,subsidy)

        # --- 2. 更新司机忙闲状态 ---
        self.driver_free_time[self.driver_free_time > 0] -= 1
        freed_drivers = np.where((self.driver_status == 1) & (self.driver_free_time == 0))[0]
        self.driver_status[freed_drivers] = 0
        self.driver_active_steps[self.driver_status == 1] += 1

        # --- 3. 生成新订单 ---
        raw_orders = self.simulator.get_fixed_orders(self.time)
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
        surviving_orders = []
        for o in self.pending_orders:
            # 等待时间超限
            if not o['matched'] and o['wait_time'] < 4:
                # the more expensive, the higher chance to drop
                if np.random.rand() > 0.05 * surge[o['origin_idx']]:
                    surviving_orders.append(o)

        self.pending_orders = surviving_orders

        for o in self.pending_orders:
            o['wait_time'] += 1
            self.total_wait_time += 1
        self.pending_orders.extend(new_orders)

        # --- 5. 处理司机动作与收益 ---
        rewards = np.zeros(CONFIG['N_DRIVERS'])
        step_subsidy_cost = 0.0

        # 仅处理在线且空闲的司机
        idle_indices = np.where(self.driver_status == 0)[0]
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

                    # --- 新增 2：计算并记录该网格的局部利润 ---
                    local_profit = price * commission - (gap_subsidy + current_subsidy)
                    zone_profits[current_loc] += local_profit
                    # ------------------------------------------

                    # 更新司机状态
                    rewards[i] = actual_income
                    self.driver_status[i] = 1
                    self.driver_free_time[i] = trip_steps
                    self.driver_locations[i] = order['dest_idx']
                    self.driver_daily_income[i] += actual_income
                    step_revenue += price  # 平台原始收入（未扣补贴）
                else:
                    rewards[i] = CONFIG['IDLE_REWARD']

            elif action == 1:  # 原地等待 (a_stay)
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
            'zone_profits': zone_profits,
            'total_revenue': self.total_revenue,
            'total_served': self.total_served_orders,
            'total_generated': self.total_generated_orders,
            'total_wait_time': self.total_wait_time,
            'online_drivers': np.sum(self.driver_status<2),
            'driver_income': self.driver_daily_income.copy(),
            'driver_income_rate': self.driver_daily_income / (self.driver_active_steps + 1e-6)
        }

        return next_state, rewards, done, info

    def _update_driver_online_status(self, surge, subsidy):
        """更新司机上下线状态：兼职司机达到收入目标下线，低收益时司机不上线"""
        for i in range(CONFIG['N_DRIVERS']):
            if self.driver_status[i]==2:
                # 已下线司机：判断是否重新上线
                if self.driver_type[i] == 0 and self.driver_daily_income[i] >= DRIVER_CONFIG['PART_TIME_INCOME_TARGET']:
                    continue
                loc = self.driver_locations[i]
                cur_surge = surge[loc]
                cur_subsidy = subsidy[loc]

                # [核心修复]：改为计算网格局部的供需比例，得出真实的抢单概率
                local_orders = len([o for o in self.pending_orders if o['origin_idx'] == loc])
                local_idle_drivers = np.sum((self.driver_locations == loc) & (self.driver_status == 0) )

                # 抢单概率最高为 1.0 (订单多于司机时，必定能抢到)
                order_probability = min(1.0, local_orders / (local_idle_drivers + 1e-6))

                # 预期收益 = 抢到单的概率 * 这一单的基础价值
                expected_income = order_probability * (cur_surge * CONFIG['BASE_FARE'] + cur_subsidy)

                if expected_income > DRIVER_CONFIG['ONLINE_THRESHOLD']:
                    self.driver_status[i] = 0
                continue

            # 在线司机：兼职司机达到收入目标则下线
            if self.driver_status[i] == 0 and self.driver_daily_income[i] >= DRIVER_CONFIG['PART_TIME_INCOME_TARGET']:
                self.driver_status[i] = 2  # 强制设为空闲
                self.driver_free_time[i] = 0

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