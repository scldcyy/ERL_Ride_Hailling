CONFIG = {
    'N_DRIVERS': 400,
    'TIME_STEP_MINUTES': 5,
    'TIME_STEPS_PER_DAY': 288,
    'N_ZONES':277,

    # PPO Hyperparameters
    'HIDDEN_DIM': 256,
    'STATE_DIM': 10,  # [Lat, Lng,t, Orders, Drivers, AvgOrders, AvgDrivers, FreeTime, Surge, Subsidy]
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

# --- 新增：司机异质性配置 --
DRIVER_CONFIG = {
    'FULL_TIME_RATIO': 0.7,  # 全职司机比例
    'PART_TIME_INCOME_TARGET': 200,  # 兼职司机日收入目标
    'FUEL_COST_PER_STEP': 0.1,  # 单位时间油价消耗(c)
    'PLATFORM_COMMISSION_RATIO': 0.2,  # 平台抽成η
    'ONLINE_THRESHOLD': 0.05,  # 司机上线的最低预期收益阈值
}