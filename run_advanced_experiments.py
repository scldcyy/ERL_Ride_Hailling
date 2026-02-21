import os
import time
import pickle
import numpy as np
from deap import base, creator, tools, gp
import operator
import math
from tqdm import tqdm

# 导入你的自定义模块
from basic_config import CONFIG
from ride_hailing_env import RideHailingEnv
from shared_ppo import Trainer


# ==========================================
# 0. 准备工作：重新注册 DEAP 环境以加载公式
# ==========================================
def protected_div(left, right):
    try:
        return left / right if abs(right) > 1e-6 else 1.0
    except ZeroDivisionError:
        return 1.0


def protected_log(x):
    try:
        return math.log(abs(x)) if abs(x) > 1e-6 else 0.0
    except ValueError:
        return 0.0


# 必须与 main_SAMO-GP.py 完全一致，否则无法解析 load 进来的树
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)
pset = gp.PrimitiveSet("MAIN", 4)
pset.renameArguments(ARG0='t', ARG1='no', ARG2='nd', ARG3='sd')
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protected_div, 2)
pset.addPrimitive(protected_log, 1)
pset.addPrimitive(math.sin, 1)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
toolbox.register("compile", gp.compile, pset=pset)


# ==========================================
# 实验 2：真实的消融实验 (Ablation Study)
# ==========================================
def run_real_ablation(sim_path, save_dir):
    print("\n--- Running Real Ablation Study (Exp 2) ---")
    from MOGP_platform import SAMO_GP_Runner, \
        run_samo_gp  # 假设你的 main_SAMO-GP.py 命名为 MOGP_platform.py 或直接复制前面的 run_samo_gp 函数

    # 我们通过修改 k_real_evals 来模拟消融
    # 1. 完整版 SAMO-GP (代理模型 + UCB 过滤，每代真实评估 2 个)
    runner_full = SAMO_GP_Runner(sim_path)
    # 注意：为了控制实验时间，设定较小的 pop_size 和 n_gens。正式论文请放大。
    _, p_full, _, _ = run_samo_gp(runner_full, pop_size=10, n_gens=10, k_real_evals=2)

    # 2. 无代理模型的标准 GP (所有个体都必须在真实环境中跑)
    runner_standard = SAMO_GP_Runner(sim_path)
    _, p_standard, _, _ = run_samo_gp(runner_standard, pop_size=10, n_gens=10,
                                      k_real_evals=10)  # k 等于 pop_size，即退化为纯 GP

    # 将真实评估的累积次数映射为 X 轴
    evals_full = np.arange(1, 11) * 2 + 10  # 初始种群10次 + 之后每代2次
    evals_standard = np.arange(1, 11) * 10 + 10  # 初始种群10次 + 之后每代10次

    ablation_data = {
        'full_evals': evals_full, 'full_profit': p_full,
        'std_evals': evals_standard, 'std_profit': p_standard
    }
    with open(f"{save_dir}/real_ablation.pkl", 'wb') as f:
        pickle.dump(ablation_data, f)
    print("Ablation data saved.")


# ==========================================
# 实验 3：真实的可扩展性实验 (Scalability)
# ==========================================
def run_real_scalability(sim_path, save_dir):
    print("\n--- Running Real Scalability Test (Exp 3) ---")
    # 我们通过动态修改 CONFIG 中的司机数量和代理模型的特征来测试规模增加对单步运算时间的影响
    sizes = [100, 400, 1000]  # 测试 100个、400个、1000个司机
    times_gp = []

    for n_drivers in sizes:
        CONFIG['N_DRIVERS'] = n_drivers
        trainer = Trainer(simulator_path=sim_path)

        # 定义一个随机公式作为测试负载
        dummy_params = {
            'surge': lambda t, no, nd, sd: 1.0 + 0.5 * sd,
            'subsidy': lambda t, no, nd, sd: 2.0 * t
        }

        start_time = time.time()
        # 仅跑 1 个 episode 测算底层 RL 的耗时
        trainer.train_and_evaluate(dummy_params, num_episodes=1)
        cost_time = time.time() - start_time
        times_gp.append(cost_time)
        print(f"Size: {n_drivers} drivers -> Time cost: {cost_time:.2f} s")

    with open(f"{save_dir}/real_scalability.pkl", 'wb') as f:
        pickle.dump({'sizes': sizes, 'times': times_gp}, f)


# ==========================================
# 实验 4：真实微观空间定价热力图 (Spatial Heatmap)
# ==========================================
def run_real_spatial_heatmap(sim_path, gp_result_path, save_dir):
    print("\n--- Extracting Real Spatial Heatmap (Exp 4) ---")
    with open(gp_result_path, 'rb') as f:
        gp_results = pickle.load(f)

    # 提取 Pareto 前沿的第一个公式作为 "Strategy C"
    best_surge_str, best_subsidy_str = gp_results['formulas'][0]

    # 重新编译为 lambda
    expr = gp.PrimitiveTree.from_string(best_surge_str, pset)
    func_surge = toolbox.compile(expr=expr)
    real_surge_func = lambda t, no, nd, sd: np.vectorize(func_surge)(t, no, nd, sd)

    env = RideHailingEnv(sim_path)
    env.reset()

    # 步进到早高峰 8:00 (假设步长5分钟，8*12=96步)
    target_step = 96
    for _ in range(target_step):
        # 司机采取随机行动(或使用已训练的策略，这里仅为获取状态)
        dummy_actions = np.zeros(CONFIG['N_DRIVERS'], dtype=int)
        env.step(dummy_actions, {'surge': lambda t, no, nd, sd: 1, 'subsidy': lambda t, no, nd, sd: 0})

    # 提取真实状态
    order_counts, idle_driver_counts = env.get_global_observation()
    sd = order_counts / (idle_driver_counts + 1e-6)
    t = env.time / CONFIG['TIME_STEPS_PER_DAY']

    # 使用 GP 生成的真实公式计算当前的真实溢价
    real_surges = real_surge_func(t, order_counts, idle_driver_counts, sd)
    real_surges = np.clip(np.nan_to_num(real_surges), 1.0, 5.0)

    # 提取六边形映射
    hexes = env.all_hexes

    heatmap_data = {'hexes': hexes, 'surges': real_surges}
    with open(f"{save_dir}/real_heatmap.pkl", 'wb') as f:
        pickle.dump(heatmap_data, f)
    print("Real heatmap data saved.")


# ==========================================
# 实验 5：真实的纳什均衡扰动分析 (Nash Equilibrium)
# ==========================================
def run_real_nash_equilibrium(sim_path, gp_result_path, save_dir):
    print("\n--- Running Real Nash Equilibrium Perturbation (Exp 5) ---")
    with open(gp_result_path, 'rb') as f:
        gp_results = pickle.load(f)

    # 取最好公式
    best_surge_str, best_subsidy_str = gp_results['formulas'][0]
    func_surge = toolbox.compile(expr=gp.PrimitiveTree.from_string(best_surge_str, pset))
    func_subsidy = toolbox.compile(expr=gp.PrimitiveTree.from_string(best_subsidy_str, pset))

    trainer = Trainer(simulator_path=sim_path)

    deviations = np.linspace(-0.5, 0.5, 11)  # 11 个扰动点
    platform_profits = []

    for delta in tqdm(deviations, desc="Perturbing Strategy"):
        # Leader 偏离均衡：我们在最优定价公式上强行加减一个扰动量 delta
        perturbed_params = {
            'surge': lambda t, no, nd, sd: np.clip(np.vectorize(func_surge)(t, no, nd, sd) + delta, 1.0, 5.0),
            'subsidy': lambda t, no, nd, sd: np.vectorize(func_subsidy)(t, no, nd, sd)
        }

        # 使用底层 PPO 进行真实评估 (仅跑2个 episode 以加速，正式论文请设为 10+)
        fitness = trainer.train_and_evaluate(perturbed_params, num_episodes=2)
        profit = fitness[0]
        platform_profits.append(profit)

    nash_data = {'deviations': deviations, 'profits': platform_profits}
    with open(f"{save_dir}/real_nash.pkl", 'wb') as f:
        pickle.dump(nash_data, f)
    print("Nash Equilibrium real data saved.")


if __name__ == "__main__":
    sim_path = 'model/generators/simulator_hex_scaling=0.004257843312339327_weekday.pkl'
    save_dir = "results"
    gp_result_path = f"{save_dir}/samogp_results.pkl"
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(sim_path):
        print(f"Error: 找不到模拟器文件 {sim_path}")
    elif not os.path.exists(gp_result_path):
        print(f"Error: 找不到 {gp_result_path}。请先运行 main_SAMO-GP.py 生成 GP 最优解。")
    else:
        # 逐个运行真实的实验数据生成
        # run_real_ablation(sim_path, save_dir) # 注意：这个比较耗时
        run_real_scalability(sim_path, save_dir)
        run_real_spatial_heatmap(sim_path, gp_result_path, save_dir)
        run_real_nash_equilibrium(sim_path, gp_result_path, save_dir)
        print("\nAll real experiments executed successfully!")