import matplotlib.pyplot as plt
import os
import pickle
import torch
import numpy as np
from tqdm import tqdm

# 引入各个模块
from main_rbf_temporal import RBF_Solver, SpatiotemporalStrategyEncoder, get_normalized_coords
from main_gp import GP_Solver, pset  # 需要引入 pset 用于编译 GP 树
from main_rl_platform import RL_Solver, PlatformActorCritic
from shared_ppo import CONFIG, RideHailingEnv, SharedPPOAgent
from deap import gp

SIM_PATH = 'model/generators/simulator_hex_scaling=0.004257843312339327_weekday.pkl'
RESULTS_DIR = 'experiment_results'
os.makedirs(RESULTS_DIR, exist_ok=True)


# --- 1. 辅助函数：生成固定场景 ---
def generate_fixed_scenarios(sim_path, num_days, seed_start=0):
    """
    生成固定的订单流场景，用于公平对比。
    Train集和Test集应使用不同的 seed_start。
    """
    print(f"Generating {num_days} days of scenarios (Seed {seed_start})...")
    temp_env = RideHailingEnv(sim_path)
    scenarios = []
    # 使用独立的 RandomState 保证可复现性
    rng = np.random.RandomState(seed_start)

    for day in range(num_days):
        day_orders = []
        for t in range(288):
            # 这里的逻辑是模拟每一天的随机性
            # 注意：实际使用时 simulator.generate_orders 内部可能并没有完全受控于外部 seed
            # 这里我们假设 simulator 允许我们通过 np.random 控制（依赖全局 seed）
            # 为了严谨，我们在 shared_ppo.py 里最好通过 env 传递 seed，但这里用全局 seed 设置权宜之计
            np.random.seed(seed_start * 10000 + day * 1000 + t)
            orders = temp_env.simulator.generate_orders(t, temp_env.all_hexes)
            day_orders.append(orders)
        scenarios.append(day_orders)
    return scenarios


# --- 2. 核心函数：通用评估器 ---
def evaluate_policy_on_test_set(method_name, artifacts, test_scenarios):
    """
    加载模型并在测试集上运行。
    """
    print(f"\nEvaluating {method_name} on Test Set...")
    env = RideHailingEnv(SIM_PATH, fixed_scenarios=test_scenarios)

    # 初始化司机 Agent 并加载权重
    driver_agent = SharedPPOAgent(CONFIG['STATE_DIM'], CONFIG['ACTION_DIM'], **CONFIG)
    driver_agent.load_by_weights(artifacts['driver_agent'])

    # 准备具体的策略模型
    platform_model = None
    encoder = None  # 用于 RBF
    gp_func = None  # 用于 GP

    if method_name == 'RL':
        platform_model = PlatformActorCritic(3, 1)
        platform_model.load_state_dict(artifacts['platform_policy'])
    elif method_name == 'RBF':
        hex_list = list(env.simulator.adjacency.keys())
        encoder = SpatiotemporalStrategyEncoder(get_normalized_coords(hex_list))
        # 解码一次即可，RBF是静态参数策略
        decoded_params = encoder.decode(artifacts['best_gene'])
    elif method_name == 'GP':
        # 编译 DEAP 表达式树为可执行函数
        s= str(artifacts['best_expr'])
        gp_func = gp.compile(artifacts['best_expr'], pset)

    total_profits = []
    service_rates = []

    # 遍历所有测试天数
    for _ in tqdm(range(len(test_scenarios))):
        state = env.reset()
        ep_profit = 0

        # 每一天的参数初始化
        current_params = {
            'commission': 0.25,
            'lambda': np.ones((CONFIG['TIME_STEPS_PER_DAY'], 277)),
            'subsidy': np.zeros((CONFIG['TIME_STEPS_PER_DAY'], 277))
        }

        # 如果是 RBF，参数是全天固定的，直接赋值
        if method_name == 'RBF':
            current_params = decoded_params

        while True:
            t = env.time

            # --- 策略执行逻辑 ---
            if t < CONFIG['TIME_STEPS_PER_DAY']:
                # RL: 每个时间步动态决策 (粗粒度控制，每12步一次)
                if method_name == 'RL' and t % 12 == 0:
                    global_obs = env.get_global_observation()
                    # 构造输入状态 [AvgOrder, AvgDriver, Time]
                    p_state = torch.FloatTensor([global_obs[0].mean(), global_obs[1].mean(), t / 288.0])
                    with torch.no_grad():
                        action, _ = platform_model.act(p_state)
                    surge_val = action.item()
                    # 广播到未来12步
                    current_params['lambda'][t: min(t + 12, 288), :] = surge_val

                # GP: 每个时间步动态决策 (细粒度控制，针对每个区域)
                elif method_name == 'GP':
                    # 获取当前观测
                    order_counts = np.zeros(env.n_zones)
                    for o in env.pending_orders:
                        if not o['matched']: order_counts[o['origin_idx']] += 1
                    idle_mask = (env.driver_status == 0)
                    driver_counts = np.bincount(env.driver_locations[idle_mask], minlength=env.n_zones)

                    obs_orders = order_counts / (order_counts.max() + 1e-6)
                    obs_drivers = driver_counts / (driver_counts.max() + 1e-6)
                    obs_time = t / CONFIG['TIME_STEPS_PER_DAY']

                    try:
                        # 向量化计算 GP 公式
                        # 注意：GP生成的函数通常只能处理标量，需要列表推导或numpy vectorize
                        surges = np.array([gp_func(o, d, obs_time) for o, d in zip(obs_orders, obs_drivers)])
                        surges = np.clip(surges, 1.0, 5.0)
                        current_params['lambda'][t] = surges
                    except Exception as e:
                        # 兜底策略
                        pass

            # --- 环境交互 ---
            # 这里的关键是：测试时，司机不再更新 (training=False)
            # 但原始代码 SharedPPOAgent.select_actions 没有 train/eval 模式标志
            # 只要不调用 agent.update()，权重就不会变，符合测试逻辑
            actions = driver_agent.select_actions(state)

            next_state, rewards, done, info = env.step(actions, current_params)

            ep_profit += info['step_profit']
            state = next_state

            if done:
                total_demand = info['total_generated'] + 1e-6
                service_rate = info['total_served'] / total_demand
                total_profits.append(ep_profit)
                service_rates.append(service_rate)
                break

    return np.mean(total_profits), np.mean(service_rates)


# --- 3. 主流程 ---
def run_full_experiment():
    if not os.path.exists(SIM_PATH):
        print("Simulator not found.")
        return

    # A. 准备数据
    # 训练集：Seed 2024
    train_scenarios = generate_fixed_scenarios(SIM_PATH, num_days=5, seed_start=2024)
    # 测试集：Seed 9999 (完全未见过的数据)
    test_scenarios = generate_fixed_scenarios(SIM_PATH, num_days=10, seed_start=9999)

    results = {}

    # # B. 训练并保存 (如果已保存则跳过，可根据需要调整)
    #
    # # 1. Train RBF
    # print("\n>>> Training RBF...")
    # rbf = RBF_Solver(SIM_PATH, scenarios=train_scenarios, pop_size=10, max_gens=5)
    # hist_rbf, artifacts_rbf = rbf.solve()
    # # 保存结果
    # with open(os.path.join(RESULTS_DIR, 'rbf_model.pkl'), 'wb') as f:
    #     pickle.dump(artifacts_rbf, f)
    #
    # # 2. Train GP
    # print("\n>>> Training GP...")
    # gp_solver = GP_Solver(SIM_PATH, scenarios=train_scenarios, pop_size=10, max_gens=5)
    # hist_gp, artifacts_gp = gp_solver.solve()
    # with open(os.path.join(RESULTS_DIR, 'gp_model.pkl'), 'wb') as f:
    #     pickle.dump(artifacts_gp, f)
    #
    # # 3. Train RL
    # print("\n>>> Training RL...")
    # rl = RL_Solver(SIM_PATH, scenarios=train_scenarios, episodes=50)  # 示例设为50，实际建议2000+
    # hist_rl, artifacts_rl = rl.solve()
    # torch.save(artifacts_rl, os.path.join(RESULTS_DIR, 'rl_model.pth'))

    # C. 加载并对比测试
    print("\n==================================")
    print("      Starting Test Evaluation    ")
    print("==================================")

    metrics = {'Method': [], 'Profit': [], 'ServiceRate': []}

    # Load & Test GP
    with open(os.path.join(RESULTS_DIR, 'gp_model.pkl'), 'rb') as f:
        gp_arts = pickle.load(f)
    p_gp, sr_gp = evaluate_policy_on_test_set('GP', gp_arts, test_scenarios)
    metrics['Method'].append('GP')
    metrics['Profit'].append(p_gp)
    metrics['ServiceRate'].append(sr_gp)

    # Load & Test RBF
    with open(os.path.join(RESULTS_DIR, 'rbf_model.pkl'), 'rb') as f:
        rbf_arts = pickle.load(f)
    p_rbf, sr_rbf = evaluate_policy_on_test_set('RBF', rbf_arts, test_scenarios)
    metrics['Method'].append('RBF')
    metrics['Profit'].append(p_rbf)
    metrics['ServiceRate'].append(sr_rbf)

    # Load & Test RL
    rl_arts = torch.load(os.path.join(RESULTS_DIR, 'rl_model.pth'))
    p_rl, sr_rl = evaluate_policy_on_test_set('RL', rl_arts, test_scenarios)
    metrics['Method'].append('RL')
    metrics['Profit'].append(p_rl)
    metrics['ServiceRate'].append(sr_rl)

    # D. 绘图与打印表格
    print("\nFinal Test Results:")
    print(f"{'Method':<10} | {'Profit':<15} | {'Service Rate':<15}")
    print("-" * 45)
    for i in range(3):
        print(f"{metrics['Method'][i]:<10} | {metrics['Profit'][i]:<15.2f} | {metrics['ServiceRate'][i]:<15.4f}")

    # 绘制柱状图对比
    fig, ax1 = plt.subplots(figsize=(8, 6))

    x = np.arange(len(metrics['Method']))
    width = 0.35

    ax1.bar(x - width / 2, metrics['Profit'], width, label='Profit', color='skyblue')
    ax1.set_ylabel('Total Profit')
    ax1.set_title('Performance Comparison on Test Set')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics['Method'])

    ax2 = ax1.twinx()
    ax2.plot(x, metrics['ServiceRate'], color='orange', marker='o', linewidth=2, label='Service Rate')
    ax2.set_ylabel('Service Rate')
    ax2.set_ylim(0, 1.0)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig("final_comparison_test.png")
    print("\nComparison plot saved to final_comparison_test.png")


if __name__ == '__main__':
    run_full_experiment()
