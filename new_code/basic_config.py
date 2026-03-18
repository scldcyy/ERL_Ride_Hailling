CONFIG = {
    'N_DRIVERS': 500,
    'TIME_STEP_MINUTES': 5,
    'TIME_STEPS_PER_DAY': 288,
    'N_ZONES': 277,
    'TRIPS_PER_DRIVER_DAY': 25,

    # PPO Hyperparameters
    'HIDDEN_DIM': 256,
    'STATE_DIM': 45,
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
    'BASE_FARE': 3.0,
    'PRICE_PER_KM': 1.5,
    'DISTANCE_TO_KM': 1.60934,  # NYC trip_miles -> km
    'PRICE_PER_MINUTE': 0.5,    # kept only for backward compatibility
    'OPPORTUNITY_COST_PER_STEP': 0.1,
    'REPOSITION_COST_PER_STEP': 0.2,
    'IDLE_REWARD': -0.05,
    'MIN_FARE_THRESHOLD': 4.0,
    'MIN_SURGE': 1.0,
    'MAX_SURGE': 5.0,
    'MIN_SUBSIDY': 0.0,
    'MAX_SUBSIDY': 20.0,

    'GENETATE_SAVE_DIR': 'generator',
}

DRIVER_CONFIG = {
    'FULL_TIME_RATIO': 0.7,
    'PART_TIME_INCOME_TARGET': 200,
    'FUEL_COST_PER_STEP': 0.1,
    'PLATFORM_COMMISSION_RATIO': 0.2,
    'ONLINE_THRESHOLD': 0.05,
}
