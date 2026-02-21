import numpy as np
import pickle

# --- 全局配置参数 (增强版) ---
CONFIG = {
    'N_DRIVERS': 200,
    'TIME_STEPS_PER_DAY': 288,  # 5分钟一步
    'N_ZONES': 277,

    # 司机参数
    'FULLTIME_RATIO': 0.7,
    'INCOME_TARGET_MEAN': 300,
    'INCOME_TARGET_STD': 50,
    'OPPORTUNITY_COST': 0.5,  # [新增] 司机每一步的空闲/机会成本

    # 乘客参数 [新增]
    'MAX_WAIT_TIME': 6,  # 最大等待步数 (6 * 5min = 20min)
    'BASE_CANCEL_PROB': 0.05,  # 每一步的基础取消概率

    # 经济参数
    'BASE_FARE': 10.0,
    'PRICE_PER_MIN': 2.0,
    'MIN_FARE': 12.0
}


class HeterogeneousRideHailingEnv:
    """
    支持异质性司机 + 弹性供给 + 乘客取消机制 的增强环境
    """

    def __init__(self, simulator_path):
        with open(simulator_path, 'rb') as f:
            self.simulator = pickle.load(f)

        self.n_zones = len(self.simulator.adjacency.keys())
        self.all_hexes = list(self.simulator.adjacency.keys())
        self.hex_to_idx = {h: i for i, h in enumerate(self.all_hexes)}

        # --- 初始化异质性司机 ---
        num_fulltime = int(CONFIG['N_DRIVERS'] * CONFIG['FULLTIME_RATIO'])
        self.driver_types = np.zeros(CONFIG['N_DRIVERS'], dtype=int)
        self.driver_types[:num_fulltime] = 1
        np.random.shuffle(self.driver_types)

        # 兼职司机目标收入
        self.income_targets = np.full(CONFIG['N_DRIVERS'], np.inf)
        part_time_indices = (self.driver_types == 0)
        self.income_targets[part_time_indices] = np.random.normal(
            CONFIG['INCOME_TARGET_MEAN'], CONFIG['INCOME_TARGET_STD'], size=np.sum(part_time_indices)
        )

        # 运行时状态
        self.time = 0
        self.driver_locations = np.zeros(CONFIG['N_DRIVERS'], dtype=int)
        self.driver_status = np.zeros(CONFIG['N_DRIVERS'], dtype=int)  # 0:Idle, 1:Serving, 2:Off-duty
        self.driver_idle_time = np.zeros(CONFIG['N_DRIVERS'], dtype=int)  # [新增] 记录连续空闲时长
        self.accumulated_income = np.zeros(CONFIG['N_DRIVERS'])
        self.pending_orders = []

        # 统计指标
        self.daily_stats = {'cancelled_orders': 0, 'total_orders': 0}

    def reset(self):
        self.time = 0
        self.driver_locations = np.random.randint(0, self.n_zones, CONFIG['N_DRIVERS'])
        self.driver_status = np.zeros(CONFIG['N_DRIVERS'], dtype=int)
        self.driver_idle_time = np.zeros(CONFIG['N_DRIVERS'], dtype=int)
        self.accumulated_income = np.zeros(CONFIG['N_DRIVERS'])
        self.pending_orders = []
        self.daily_stats = {'cancelled_orders': 0, 'total_orders': 0}

        # 刷新兼职目标
        part_time_indices = (self.driver_types == 0)
        self.income_targets[part_time_indices] = np.random.normal(
            CONFIG['INCOME_TARGET_MEAN'], CONFIG['INCOME_TARGET_STD'], size=np.sum(part_time_indices)
        )
        return self._get_state()

    def _calculate_gini(self):
        incomes = self.accumulated_income.copy()
        if np.sum(incomes) == 0: return 0.0
        incomes = np.sort(incomes)
        n = len(incomes)
        index = np.arange(1, n + 1)
        return (np.sum((2 * index - n - 1) * incomes)) / (n * np.sum(incomes))

    def step(self, driver_actions, platform_pricing_func):
        """
        环境核心推演
        """
        # 1. 状态更新与订单生成
        current_orders = self.simulator.generate_orders(self.time, self.all_hexes)
        self._process_new_orders(current_orders)
        self.daily_stats['total_orders'] += len(current_orders)

        # 计算区域供需 (State)
        zone_demand = np.zeros(self.n_zones)
        for o in self.pending_orders: zone_demand[o['origin_idx']] += 1
        zone_supply = np.zeros(self.n_zones)
        idle_drivers = (self.driver_status == 0)
        np.add.at(zone_supply, self.driver_locations[idle_drivers], 1)

        # 2. [平台决策] 获取当前定价
        current_prices = platform_pricing_func(zone_demand, zone_supply, self.time)

        # --- [新增机制 1] 司机供给弹性 (Entry/Exit) ---
        self._update_driver_participation(current_prices)

        # --- [新增机制 2] 乘客取消订单 ---
        self._check_order_cancellation()

        # 3. 订单匹配逻辑
        step_platform_profit = 0
        served_count = 0

        # 只有状态为 0 (Idle) 的司机参与匹配
        idle_indices = np.where(self.driver_status == 0)[0]
        np.random.shuffle(idle_indices)  # 随机顺序匹配

        for i in idle_indices:
            loc = self.driver_locations[i]
            # 查找该区域的有效订单
            my_orders = [o for o in self.pending_orders if o['origin_idx'] == loc]

            # 如果有单 -> 接单 (Serve)
            # 这里简化逻辑：只要有单，司机默认愿意接 (因为如果不接就是 Idle，收益为0)
            # 复杂的模型可以在这里加入 "Acceptance Rate" 判断
            if my_orders:
                order = my_orders[0]

                # 收益结算
                surge = current_prices[loc]
                fare = max(CONFIG['MIN_FARE'],
                           (CONFIG['BASE_FARE'] + order['duration'] * CONFIG['PRICE_PER_MIN']) * surge)

                driver_income = fare * 0.8
                platform_income = fare * 0.2

                self.accumulated_income[i] += driver_income
                step_platform_profit += platform_income

                # 状态流转
                self.driver_status[i] = 1  # Serving
                self.driver_locations[i] = order['dest_idx']
                self.driver_status[i] = 0  # 简化：瞬移完成
                self.driver_idle_time[i] = 0  # 重置空闲时间

                self.pending_orders.remove(order)
                served_count += 1

            else:
                # 没单 -> 执行调度动作 (Reposition) OR 原地等待
                # 只有全职司机(Type 1) 或者 还没赚够的兼职司机才会积极调度
                # 这里假设传入的 driver_actions 已经包含了决策逻辑
                target_loc = driver_actions[i]
                if target_loc != loc and target_loc < self.n_zones:
                    self.driver_locations[i] = target_loc
                    self.accumulated_income[i] -= 0.5  # 调度油耗成本

                # 增加空闲时间计数
                self.driver_idle_time[i] += 1

        # 4. 统计与返回
        gini = self._calculate_gini()
        # 服务率分母：总需求 = 新增 + 之前积压的 (不包含这步取消的，因为已经移除了)
        total_demand_pool = len(self.pending_orders) + served_count
        service_rate = served_count / (total_demand_pool + 1e-6)

        self.time += 1
        done = (self.time >= CONFIG['TIME_STEPS_PER_DAY'])

        info = {
            'profit': step_platform_profit,
            'service_rate': service_rate,
            'gini': gini,
            'cancelled': self.daily_stats['cancelled_orders'],
            'active_drivers': np.sum(self.driver_status != 2)
        }

        return self._get_state(), step_platform_profit, done, info

    def _process_new_orders(self, raw_orders):
        for o in raw_orders:
            if o['origin_hex'] in self.hex_to_idx and o['dest_hex'] in self.hex_to_idx:
                self.pending_orders.append({
                    'origin_idx': self.hex_to_idx[o['origin_hex']],
                    'dest_idx': self.hex_to_idx[o['dest_hex']],
                    'duration': o.get('duration', 15),
                    'wait_time': 0  # [新增] 初始化等待时间
                })

    def _update_driver_participation(self, current_prices):
        """
        [新增] 司机上下线逻辑 (供给弹性)
        Reference: Chen et al. (2020) - Dynamic Supply
        """
        # 1. 下线逻辑 (Exit): 空闲太久 或 赚够了
        active_indices = np.where(self.driver_status == 0)[0]
        for i in active_indices:
            # 兼职司机赚够离场
            if self.driver_types[i] == 0 and self.accumulated_income[i] >= self.income_targets[i]:
                if np.random.rand() < 0.8:  # 高概率下线
                    self.driver_status[i] = 2
                    continue

            # 通用逻辑：长期空闲导致下线 (耐心耗尽)
            # 比如连续空闲超过 12个时间步 (1小时)
            if self.driver_idle_time[i] > 12:
                # 概率下线，取决于当前区域价格。价格高可能会再等等。
                current_surge = current_prices[self.driver_locations[i]]
                prob_exit = 0.5 / current_surge  # 价格越高，离开概率越低
                if np.random.rand() < prob_exit:
                    self.driver_status[i] = 2

        # 2. 上线逻辑 (Entry): 被高价吸引回来
        offline_indices = np.where(self.driver_status == 2)[0]
        # 随机抽取一部分离线司机进行判断，模拟他们不定时查看APP
        check_indices = np.random.choice(offline_indices, size=int(len(offline_indices) * 0.1))

        for i in check_indices:
            loc = self.driver_locations[i]
            surge = current_prices[loc]

            # 上线概率与价格正相关 P_entry = Sigmoid(Surge)
            # 兼职司机对价格更敏感
            base_prob = 0.1 * surge
            if self.driver_types[i] == 0 and self.accumulated_income[i] < self.income_targets[i]:
                base_prob *= 1.5  # 还没赚够的兼职司机更容易回来

            if np.random.rand() < min(base_prob, 0.8):
                self.driver_status[i] = 0  # 回到 Idle 状态
                self.driver_idle_time[i] = 0

    def _check_order_cancellation(self):
        """
        [新增] 乘客取消逻辑
        Reference: Liu et al. (2022) - Impatience Function
        """
        surviving_orders = []
        for o in self.pending_orders:
            o['wait_time'] += 1

            # 判据 1: 超过最大容忍时间 -> 必定取消
            if o['wait_time'] > CONFIG['MAX_WAIT_TIME']:
                self.daily_stats['cancelled_orders'] += 1
                continue  # 丢弃该订单

            # 判据 2: 随机取消 (等待时间越长，概率越大)
            # Prob = Base + 0.02 * wait_steps
            cancel_prob = CONFIG['BASE_CANCEL_PROB'] + 0.02 * o['wait_time']
            if np.random.rand() < cancel_prob:
                self.daily_stats['cancelled_orders'] += 1
                continue

            surviving_orders.append(o)

        self.pending_orders = surviving_orders

    def _get_state(self):
        return {
            'driver_locs': self.driver_locations,
            'driver_income': self.accumulated_income,
            'driver_status': self.driver_status,  # 新增状态可见性
            'time': self.time
        }