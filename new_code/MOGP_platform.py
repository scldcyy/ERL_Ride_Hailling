import os
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, gp, algorithms
import operator
import math
from scipy.spatial.distance import cdist
# 导入底层环境和代理模型
from shared_ppo import Trainer
from surrogate_model import SurrogateModel


# --- 1. 定义受保护的数学操作 ---
def protected_div(left, right):
    safe_right = np.where(np.abs(right) > 1e-6, right, 1.0)
    return np.where(np.abs(right) > 1e-6, left / safe_right, 1.0)

def protected_log(x):
    safe_x = np.where(np.abs(x) > 1e-6, np.abs(x), 1.0)
    return np.where(np.abs(x) > 1e-6, np.log(safe_x), 0.0)

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
pset.addPrimitive(np.sin, 1)
pset.addEphemeralConstant("rand101", gen_rand101)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
# 初始化个体：由两棵独立的树组成
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.tree, toolbox.tree), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("compile", gp.compile, pset=pset)

def check_height_limit(individual, max_height=6):
    """确保个体中的所有树都不超过指定的最大高度"""
    for tree in individual:
        if tree.height > max_height:
            return False
    return True

# --- 处理多树个体 (Multi-Tree GP) 的自定义交叉与变异 ---
def cx_multi_tree(ind1, ind2, max_height=6):
    """
    针对包含多棵树的个体进行交叉。
    保证至少有一棵树被交叉，另一棵有 50% 概率交叉。
    包含防止树过度膨胀 (Bloat) 的高度限制。
    """
    # 备份原始树以防超出高度限制
    ind1_backup = [toolbox.clone(t) for t in ind1]
    ind2_backup = [toolbox.clone(t) for t in ind2]

    for tree_idx in range(len(ind1)):
        ind1[tree_idx], ind2[tree_idx] = gp.cxOnePoint(ind1[tree_idx], ind2[tree_idx])

    # 高度限制检查，如果越界则撤销交叉
    if not (check_height_limit(ind1, max_height) and check_height_limit(ind2, max_height)):
        ind1[:] = ind1_backup
        ind2[:] = ind2_backup

    return ind1, ind2


def mut_multi_tree(individual, expr, pset, max_height=6):
    """
    小概率变异。
    """
    ind_backup = [toolbox.clone(t) for t in individual]

    for tree_idx in range(len(individual)):
        individual[tree_idx], = gp.mutUniform(individual[tree_idx], expr=expr, pset=pset)

    if not check_height_limit(individual, max_height):
        individual[:] = ind_backup

    return individual,

# 注册自定义的遗传操作
toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", cx_multi_tree, max_height=6)
toolbox.register("mutate", mut_multi_tree, expr=toolbox.expr, pset=pset, max_height=6)


# --- 3. 评估与代理模型逻辑 ---
class SAMO_GP_Runner:
    def __init__(self, sim_path):
        self.trainer = Trainer(simulator_path=sim_path)
        self.surrogate = SurrogateModel(n_reference_states=50)
        self.archive_params = []
        self.archive_fitness = []
        self.archive_inds = []
        self.archive_weights = []

    def ind_to_params(self, individual):
        """将 DEAP 个体转化为 platform_params 字典"""
        func_surge = toolbox.compile(expr=individual[0])
        func_subsidy = toolbox.compile(expr=individual[1])

        def safe_eval(func, t, no, nd, sd):
            # 尝试直接使用 numpy 原生数组运算
            res = func(t, no, nd, sd)
            # 如果生成的树是一个纯常数 (例如: 1.5)，它会返回标量，需要将其扩展为与区域数量相同的数组
            if np.isscalar(res) or np.ndim(res) == 0:
                res = np.full_like(no, float(res))
            return res

        return {
            'surge': lambda t, no, nd, sd: safe_eval(func_surge, t, no, nd, sd),
            'subsidy': lambda t, no, nd, sd: safe_eval(func_subsidy, t, no, nd, sd)
        }

    def evaluate_real(self, individual, num_episodes=10, init_mode=False):
        params = self.ind_to_params(individual)

        self.trainer.reset_to_base_weights()
        self.trainer.agent.optimizer.state.clear()

        # Bypass hot start if in init_mode
        if len(self.archive_params) > 0 and not init_mode:
            current_pheno = self.surrogate._get_phenotype(params)
            phenotypes = np.array([self.surrogate._get_phenotype(p) for p in self.archive_params])
            distances = np.linalg.norm(phenotypes - current_pheno, axis=1)
            best_idx = np.argmin(distances)
            best_weights = self.archive_weights[best_idx]
            self.trainer.agent.load_by_weights(best_weights)


        fitness = self.trainer.train_and_evaluate(params, num_episodes=num_episodes)

        self.archive_params.append(params)
        self.archive_fitness.append(fitness[:3])
        self.archive_inds.append(individual)
        self.archive_weights.append(self.trainer.agent.get_weights())
        return tuple(fitness[:3])

    def evaluate_surrogate(self, individual):
        """使用高斯过程预测适应度"""
        params = self.ind_to_params(individual)
        mu, _ = self.surrogate.predict(params)
        return tuple(mu)


def save_metrics(gen, hof, history, filename="results/gp_metrics.pkl"):
    """仅保存帕累托前沿的公式字符串和当前的收敛历史曲线"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    metrics_data = {
        'gen': gen,
        'pareto_front_formulas': [(str(ind[0]), str(ind[1])) for ind in hof],
        'pareto_fitness': [ind.fitness.values for ind in hof],
        'history': history
    }
    with open(filename, 'wb') as f:
        pickle.dump(metrics_data, f)
    print(f" [Save] Pareto front formulas and metrics saved at Generation {gen}.")

def generate_lhs_initial_population(toolbox, runner, pop_size):
    """
    通过表现型空间的最大最小距离过滤，生成具有 LHS 特性的初始 GP 种群。
    确保初始种群的公式在行为逻辑上具有最大的多样性。
    """
    print("Generating extended pool for LHS Phenotype selection...")
    # 生成 10 倍规模的候选池
    pool_size = pop_size * 10
    candidate_pool = toolbox.population(n=pool_size)

    phenotypes = []
    valid_candidates = []

    # 提取所有候选树的表现型
    for ind in candidate_pool:
        params = runner.ind_to_params(ind)
        pheno = runner.surrogate._get_phenotype(params)

        # 过滤掉全 nan 或极值溢出的无效公式
        if not np.any(np.isnan(pheno)) and np.max(np.abs(pheno)) < 100:
            phenotypes.append(pheno)
            valid_candidates.append(ind)

    # --- FIX 1: 防止合法公式不足导致的无限死循环 ---
    actual_pop_size = min(pop_size, len(valid_candidates))
    if actual_pop_size == 0:
         raise ValueError("致命错误: 未能生成任何有效的候选公式，请检查树的生成约束！")
    elif actual_pop_size < pop_size:
         print(f"警告: 有效候选公式数量 ({len(valid_candidates)}) 少于预期的种群规模 ({pop_size})。将使用截断后的数量。")
    # ---------------------------------------------

    phenotypes = np.array(phenotypes)

    # 贪心 Max-Min 距离选择法 (近似 LHS 覆盖)
    selected_indices = [random.randint(0, len(valid_candidates) - 1)]

    # 使用 actual_pop_size 作为循环终止条件
    while len(selected_indices) < actual_pop_size:
        # 计算所有候选点到已选集合的距离矩阵
        selected_phenos = phenotypes[selected_indices]
        dists = cdist(phenotypes, selected_phenos)

        # 找到距离已选集合最近距离的那个点
        min_dists = np.min(dists, axis=1)

        # 将已经被选中的点距离设为 -1 排除
        min_dists[selected_indices] = -1

        # 选出该最小距离中的最大值（离当前群体最远的点）
        next_idx = np.argmax(min_dists)
        selected_indices.append(next_idx)

    initial_pop = [valid_candidates[i] for i in selected_indices]
    print(f"LHS Initialization complete. Selected {actual_pop_size} diverse formulas.")
    return initial_pop

# --- 4. 主循环与辅助函数 ---
def run_samo_gp(runner, pop_size=100, n_gens=50, k_real_evals=5):
    metrics_file = "results/gp_metrics.pkl"

    # 全新启动
    pop = generate_lhs_initial_population(toolbox, runner, pop_size)
    hof = tools.ParetoFront()
    history_max_profit, history_max_efficiency, history_max_fairness = [], [], []

    print("=== Generation 0: Initializing Surrogate ===")
    for ind in pop:
        ind.fitness.values = runner.evaluate_real(ind, num_episodes=10, init_mode=True)
        # --- 标记初始种群为真实评估 ---
        ind.is_real_evaluated = True

    runner.surrogate.update(runner.archive_params, runner.archive_fitness)

    # 初始化时，所有的个体都是真实评估的，直接更新 hof
    hof.update(pop)

    # 初始代保存
    save_metrics(0, hof, (history_max_profit, history_max_efficiency, history_max_fairness), metrics_file)

    # 进化循环
    for gen in range(1, n_gens + 1):
        print(f"\n=== Generation {gen}/{n_gens} ===")
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.8, mutpb=0.3)

        ehvi_scores = []
        if len(hof) > 0:
            pf_fits = np.array([ind.fitness.values for ind in hof])
            ref_point = np.max(-pf_fits, axis=0) + np.abs(np.max(-pf_fits, axis=0)) * 0.1
            ref_point = np.where(ref_point == 0, 1e-6, ref_point)
        else:
            ref_point = None

        for ind in offspring:
            params = runner.ind_to_params(ind)
            score = runner.surrogate.calculate_ehvi_score(params, hof, ref_point)
            ehvi_scores.append(score)

        top_k_indices = np.argsort(ehvi_scores)[-k_real_evals:]

        for i, ind in enumerate(offspring):
            if i in top_k_indices:
                ind.fitness.values = runner.evaluate_real(ind, num_episodes=10)
                # --- FIX 2.2: 只有这部分是被真实环境评估过的 ---
                ind.is_real_evaluated = True
            else:
                ind.fitness.values = runner.evaluate_surrogate(ind)
                # --- FIX 2.2: 代理模型预测的适应度，打上伪造标签 ---
                ind.is_real_evaluated = False

        runner.surrogate.update(runner.archive_params, runner.archive_fitness)

        # 种群选择：代理评估和真实评估的个体都可以参与下一代的角逐
        pop = toolbox.select(pop + offspring, k=pop_size)

        # --- FIX 2.3: 严格守卫帕累托前沿！只允许真实评估的个体进入 ---
        real_evaluated_inds = [ind for ind in (pop + offspring) if getattr(ind, 'is_real_evaluated', False)]
        if real_evaluated_inds:
            hof.update(real_evaluated_inds)
        # --------------------------------------------------------

        current_archive = np.array(runner.archive_fitness)
        history_max_profit.append(np.max(current_archive[:, 0]))
        history_max_efficiency.append(np.max(current_archive[:, 1]))
        history_max_fairness.append(np.max(current_archive[:, 2]))

        print(f"Gen {gen} Best Profit: {history_max_profit[-1]:.2f}")

        # --- 每代结束只保存轻量级指标 ---
        save_metrics(gen, hof, (history_max_profit, history_max_efficiency, history_max_fairness), metrics_file)

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
    sim_path = 'generator/simulator_driver_nums=400_hex_scaling=0.017031373249357308_weekday.pkl'
    if not os.path.exists(sim_path):
        print(f"Simulator file not found at {sim_path}")
    else:
        # 初始化运行器并开始进化
        runner = SAMO_GP_Runner(sim_path)
        # 参数规模按需放大 (例如 pop_size=100, n_gens=50)
        hof, p_hist, e_hist, f_hist = run_samo_gp(runner, pop_size=50, n_gens=50, k_real_evals=10)

        plot_and_save_results(hof, (p_hist, e_hist, f_hist))