import os
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, gp, algorithms
import operator
import math

# 导入底层环境和代理模型
from shared_ppo import Trainer
from surge_model import SurrogateModel


# --- 1. 定义受保护的数学操作 (防止 GP 生成非法数学公式导致崩溃) ---
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

def gen_rand101():
    return round(np.random.uniform(-1, 1), 2)

# --- 2. 配置 DEAP 遗传规划环境 ---
# 定义多目标适应度：最大化 Profit, 最小化 Wait Time (负值最大化), 最小化 Gini (负值最大化)
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, 1.0))
# Individual 是一个包含两棵树的列表 (一棵用于 surge, 一棵用于 subsidy)
creator.create("Individual", list, fitness=creator.FitnessMax)

# 定义终端和函数集 (输入特征为: t, N_o, N_d, SD)
pset = gp.PrimitiveSet("MAIN", 4)
pset.renameArguments(ARG0='t', ARG1='no', ARG2='nd', ARG3='sd')
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protected_div, 2)
pset.addPrimitive(protected_log, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", gen_rand101)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
# 初始化个体：由两棵独立的树组成
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.tree, toolbox.tree), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("compile", gp.compile, pset=pset)

# --- 新增：处理多树个体 (Multi-Tree GP) 的自定义交叉与变异 ---
def cx_multi_tree(ind1, ind2):
    """
    针对包含多棵树的个体进行交叉。
    以 50% 的概率独立交叉 Surge 树和 Subsidy 树。
    """
    for i in range(len(ind1)):
        if random.random() < 0.5:
            ind1[i], ind2[i] = gp.cxOnePoint(ind1[i], ind2[i])
    return ind1, ind2

def mut_multi_tree(individual, expr, pset):
    """
    针对包含多棵树的个体进行变异。
    以 50% 的概率独立变异 Surge 树和 Subsidy 树。
    """
    for i in range(len(individual)):
        if random.random() < 0.5:
            # gp.mutUniform 返回一个元组，需要解包
            individual[i], = gp.mutUniform(individual[i], expr=expr, pset=pset)
    return individual,

# 注册自定义的遗传操作
toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", cx_multi_tree)
toolbox.register("mutate", mut_multi_tree, expr=toolbox.expr, pset=pset)


# --- 3. 评估与代理模型逻辑 ---
class SAMO_GP_Runner:
    def __init__(self, sim_path):
        self.trainer = Trainer(simulator_path=sim_path)
        self.surrogate = SurrogateModel(n_reference_states=50)
        self.archive_params = []
        self.archive_fitness = []
        self.archive_inds = []

    def ind_to_params(self, individual):
        """将 DEAP 个体转化为 platform_params 字典"""
        func_surge = toolbox.compile(expr=individual[0])
        func_subsidy = toolbox.compile(expr=individual[1])
        # 包装一层以处理数组输入
        return {
            'surge': lambda t, no, nd, sd: np.vectorize(func_surge)(t, no, nd, sd),
            'subsidy': lambda t, no, nd, sd: np.vectorize(func_subsidy)(t, no, nd, sd)
        }

    def evaluate_real(self, individual, num_episodes=2):
        """在真实的 PPO 环境中评估"""
        self.trainer.reset_to_base_weights()
        params = self.ind_to_params(individual)
        # 真实评估，获取 [Profit, -Wait Time, -Gini]
        # 注意：在此缩减 num_episodes 以加速测试。实际运行时应增加。
        fitness = self.trainer.train_and_evaluate(params, num_episodes=num_episodes)

        # 将真实数据存入档案，用于更新代理模型
        self.archive_params.append(params)
        self.archive_fitness.append(fitness[:3])  # 仅取前三个目标
        self.archive_inds.append(individual)
        return tuple(fitness[:3])

    def evaluate_surrogate(self, individual):
        """使用高斯过程预测适应度"""
        params = self.ind_to_params(individual)
        mu, _ = self.surrogate.predict(params)
        return tuple(mu)


def save_checkpoint(gen, pop, hof, history, runner, filename="results/gp_checkpoint.pkl"):
    """保存当前进度的快照"""
    cp_data = {
        'gen': gen,
        'pop': pop,
        'hof': hof,
        'history': history,
        'archive_inds': runner.archive_inds,  # 存原始树结构，避开 lambda
        'archive_fitness': runner.archive_fitness
    }
    with open(filename, 'wb') as f:
        pickle.dump(cp_data, f)
    print(f" [Checkpoint] State saved at Generation {gen}.")


def load_checkpoint(filename, runner):
    """恢复训练进度并重建环境状态"""
    with open(filename, 'rb') as f:
        cp_data = pickle.load(f)

    # 重建代理模型所需的 archive_params (将树结构重新编译为 lambda)
    runner.archive_inds = cp_data['archive_inds']
    runner.archive_fitness = cp_data['archive_fitness']
    runner.archive_params = [runner.ind_to_params(ind) for ind in runner.archive_inds]

    print(f" [Checkpoint] Resumed from Generation {cp_data['gen']}.")
    return cp_data['gen'], cp_data['pop'], cp_data['hof'], cp_data['history']

# --- 4. 主循环与辅助函数 ---
def run_samo_gp(runner, pop_size=100, n_gens=50, k_real_evals=5):
    checkpoint_file = "results/gp_checkpoint.pkl"
    start_gen = 0

    # 1. 尝试从断点恢复
    if os.path.exists(checkpoint_file):
        start_gen, pop, hof, history = load_checkpoint(checkpoint_file, runner)
        history_max_profit, history_max_efficiency, history_max_fairness = history
        # 恢复代理模型状态
        runner.surrogate.update(runner.archive_params, runner.archive_fitness)
    else:
        # 全新启动
        pop = toolbox.population(n=pop_size)
        hof = tools.ParetoFront()
        history_max_profit, history_max_efficiency, history_max_fairness = [], [], []

        print("=== Generation 0: Initializing Surrogate ===")
        for ind in pop:
            ind.fitness.values = runner.evaluate_real(ind, num_episodes=3)

        runner.surrogate.update(runner.archive_params, runner.archive_fitness)
        hof.update(pop)

        # 初始代保存一下
        save_checkpoint(0, pop, hof, (history_max_profit, history_max_efficiency, history_max_fairness), runner,
                        checkpoint_file)

    # 2. 从断点处继续进化循环
    for gen in range(start_gen + 1, n_gens + 1):
        print(f"\n=== Generation {gen}/{n_gens} ===")
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.9, mutpb=0.1)

        ucb_scores = []
        for ind in offspring:
            params = runner.ind_to_params(ind)
            ucb = runner.surrogate.calculate_ucb(params, kappa=2.0)
            ucb_scores.append(np.sum(ucb))

        top_k_indices = np.argsort(ucb_scores)[-k_real_evals:]

        for i, ind in enumerate(offspring):
            if i in top_k_indices:
                ind.fitness.values = runner.evaluate_real(ind, num_episodes=3)
            else:
                ind.fitness.values = runner.evaluate_surrogate(ind)

        runner.surrogate.update(runner.archive_params, runner.archive_fitness)
        pop = toolbox.select(pop + offspring, k=pop_size)
        hof.update(pop)

        current_archive = np.array(runner.archive_fitness)
        history_max_profit.append(np.max(current_archive[:, 0]))
        history_max_efficiency.append(np.max(current_archive[:, 1]))
        history_max_fairness.append(np.max(current_archive[:, 2]))

        print(f"Gen {gen} Best Profit: {history_max_profit[-1]:.2f}")

        # --- 每代结束自动保存 Checkpoint ---
        save_checkpoint(gen, pop, hof, (history_max_profit, history_max_efficiency, history_max_fairness), runner,
                        checkpoint_file)

    return hof, history_max_profit, history_max_efficiency, history_max_fairness


def plot_and_save_results(hof, history, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    profits, efficiencies, fairnesses = history

    # --- 绘制收敛曲线 ---
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(profits, color='b', marker='o')
    plt.title('Convergence: Platform Profit')
    plt.xlabel('Generation')
    plt.ylabel('Max Profit')

    plt.subplot(1, 3, 2)
    plt.plot(efficiencies, color='g', marker='o')
    plt.title('Convergence: Efficiency (-Wait Time)')
    plt.xlabel('Generation')

    plt.subplot(1, 3, 3)
    plt.plot(fairnesses, color='r', marker='o')
    plt.title('Convergence: Fairness (-Gini Index)')
    plt.xlabel('Generation')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/samogp_convergence.png")
    plt.show()

    # --- 绘制 3D 帕累托前沿 ---
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    p_vals = [ind.fitness.values[0] for ind in hof]
    e_vals = [ind.fitness.values[1] for ind in hof]
    f_vals = [ind.fitness.values[2] for ind in hof]

    sc = ax.scatter(p_vals, e_vals, f_vals, c=p_vals, cmap='viridis', s=50)
    ax.set_xlabel('Profit')
    ax.set_ylabel('Efficiency (-Wait)')
    ax.set_zlabel('Fairness (-Gini)')
    ax.set_title('SAMO-GP Pareto Front')
    fig.colorbar(sc, ax=ax, label='Profit')

    plt.savefig(f"{save_dir}/samogp_pareto_3d.png")
    plt.show()

    # --- 保存结果用于 Baseline 对比 ---
    results_data = {
        'pareto_fitness': np.column_stack((p_vals, e_vals, f_vals)),
        'convergence': history,
        'formulas': [(str(ind[0]), str(ind[1])) for ind in hof]
    }

    with open(f"{save_dir}/samogp_results.pkl", 'wb') as f:
        pickle.dump(results_data, f)

    print(f"\nResults saved to {save_dir}/samogp_results.pkl")
    print("\nSample Best Formulas found by SAMO-GP:")
    for i in range(min(3, len(hof))):
        print(f"[{i + 1}] Surge: {hof[i][0]} | Subsidy: {hof[i][1]}")


if __name__ == '__main__':
    sim_path = 'model/generators/simulator_hex_scaling=0.004257843312339327_weekday.pkl'
    if not os.path.exists(sim_path):
        print(f"Simulator file not found at {sim_path}")
    else:
        # 初始化运行器并开始进化
        runner = SAMO_GP_Runner(sim_path)
        # 参数规模按需放大 (例如 pop_size=100, n_gens=50)
        hof, p_hist, e_hist, f_hist = run_samo_gp(runner, pop_size=50, n_gens=50, k_real_evals=5)

        plot_and_save_results(hof, (p_hist, e_hist, f_hist))