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


# --- 1. 定义受保护的数学操作 ---
def protected_div(left, right):
    """Safe division for both scalars and numpy arrays."""
    # Replace 0s with 1s in the denominator BEFORE dividing to prevent exceptions
    safe_right = np.where(np.abs(right) > 1e-6, right, 1.0)
    return np.where(np.abs(right) > 1e-6, left / safe_right, 1.0)

def protected_log(x):
    """Safe log for both scalars and numpy arrays."""
    # Replace 0s with 1s BEFORE logging to prevent RuntimeWarnings
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

# --- 新增：处理多树个体 (Multi-Tree GP) 的自定义交叉与变异 ---
def cx_multi_tree(ind1, ind2, max_height=6):
    """
    针对包含多棵树的个体进行交叉。
    保证至少有一棵树被交叉，另一棵有 50% 概率交叉。
    包含防止树过度膨胀 (Bloat) 的高度限制。
    """
    # 备份原始树以防超出高度限制
    ind1_backup = [toolbox.clone(t) for t in ind1]
    ind2_backup = [toolbox.clone(t) for t in ind2]

    # 保证至少有一棵树被交叉
    tree_idx = random.choice([0, 1])
    ind1[tree_idx], ind2[tree_idx] = gp.cxOnePoint(ind1[tree_idx], ind2[tree_idx])

    # 另一棵树有 50% 的概率交叉
    other_idx = 1 - tree_idx
    if random.random() < 0.5:
        ind1[other_idx], ind2[other_idx] = gp.cxOnePoint(ind1[other_idx], ind2[other_idx])

    # 高度限制检查，如果越界则撤销交叉
    if not (check_height_limit(ind1, max_height) and check_height_limit(ind2, max_height)):
        ind1[:] = ind1_backup
        ind2[:] = ind2_backup

    return ind1, ind2


def mut_multi_tree(individual, expr, pset, max_height=6):
    """
    针对包含多棵树的个体进行变异。
    保证至少有一棵树被变异，另一棵有 50% 概率变异。
    """
    ind_backup = [toolbox.clone(t) for t in individual]

    tree_idx = random.choice([0, 1])
    individual[tree_idx], = gp.mutUniform(individual[tree_idx], expr=expr, pset=pset)

    other_idx = 1 - tree_idx
    if random.random() < 0.5:
        individual[other_idx], = gp.mutUniform(individual[other_idx], expr=expr, pset=pset)

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

    def evaluate_real(self, individual, num_episodes=15):
        """在真实的 PPO 环境中评估"""
        params = self.ind_to_params(individual)
        # 真实评估，获取 [Profit, -Wait Time, -Gini]
        if len(self.archive_params) > 0:
            # Extract phenotype of current formula
            current_pheno = self.surrogate._get_phenotype(params)

            # Extract phenotypes of all historical formulas
            phenotypes = np.array([self.surrogate._get_phenotype(p) for p in self.archive_params])

            # Find nearest neighbor strategy in phenotype space
            distances = np.linalg.norm(phenotypes - current_pheno, axis=1)
            best_idx = np.argmin(distances)

            # Transfer parameters: load w_neighbor
            best_weights = self.archive_weights[best_idx]
            self.trainer.agent.load_by_weights(best_weights)
        else:
            # First generation fallback
            self.trainer.reset_to_base_weights()

        # Real evaluation (now requires fewer episodes to converge due to hot start)
        fitness = self.trainer.train_and_evaluate(params, num_episodes=num_episodes)

        # Save to archive
        self.archive_params.append(params)
        self.archive_fitness.append(fitness[:3])
        self.archive_inds.append(individual)
        self.archive_weights.append(self.trainer.agent.get_weights())  # Save new weights

        return tuple(fitness[:3])

    def evaluate_surrogate(self, individual):
        """使用高斯过程预测适应度"""
        params = self.ind_to_params(individual)
        mu, _ = self.surrogate.predict(params)
        return tuple(mu)


def save_checkpoint(gen, pop, hof, history, runner, filename="results/gp_checkpoint.pkl"):
    """保存当前进度的快照，包含热启动所需的模型权重"""
    cp_data = {
        'gen': gen,
        'pop': pop,
        'hof': hof,
        'history': history,
        'archive_inds': runner.archive_inds,
        'archive_fitness': runner.archive_fitness,
        'archive_weights': runner.archive_weights  # NEW: Save PPO weights for Hot Start
    }
    with open(filename, 'wb') as f:
        pickle.dump(cp_data, f)
    print(f" [Checkpoint] State saved at Generation {gen}.")


def load_checkpoint(filename, runner):
    """恢复训练进度并重建环境状态，包括热启动知识库"""
    with open(filename, 'rb') as f:
        cp_data = pickle.load(f)

    # 重建代理模型所需的 archive_params (将树结构重新编译为 lambda)
    runner.archive_inds = cp_data['archive_inds']
    runner.archive_fitness = cp_data['archive_fitness']
    runner.archive_params = [runner.ind_to_params(ind) for ind in runner.archive_inds]

    # NEW: Restore the Hot Start knowledge base
    # Using .get() ensures backward compatibility if you load an older checkpoint
    runner.archive_weights = cp_data.get('archive_weights', [])

    print(f" [Checkpoint] Resumed from Generation {cp_data['gen']}.")
    return cp_data['gen'], cp_data['pop'], cp_data['hof'], cp_data['history']


from scipy.spatial.distance import cdist


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

    phenotypes = np.array(phenotypes)

    # 贪心 Max-Min 距离选择法 (近似 LHS 覆盖)
    selected_indices = [random.randint(0, len(valid_candidates) - 1)]

    while len(selected_indices) < pop_size:
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
    print(f"LHS Initialization complete. Selected {pop_size} diverse formulas.")
    return initial_pop

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
        pop = generate_lhs_initial_population(toolbox, runner, pop_size)
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

        # EVHI 过滤
        ehvi_scores = []

        # 动态计算参考点 (Reference Point)
        if len(hof) > 0:
            pf_fits = np.array([ind.fitness.values for ind in hof])
            # 因为转化为最小化问题，参考点设为前沿在各维度上的最差值 (最大值)，并向外延伸 10%
            ref_point = np.max(-pf_fits, axis=0) + np.abs(np.max(-pf_fits, axis=0)) * 0.1
            # 防止参考点坐标为 0 导致计算异常
            ref_point = np.where(ref_point == 0, 1e-6, ref_point)
        else:
            ref_point = None

        for ind in offspring:
            params = runner.ind_to_params(ind)
            # 计算 EHVI 得分
            score = runner.surrogate.calculate_ehvi_score(params, hof, ref_point)
            ehvi_scores.append(score)

        # 筛选 EHVI 得分最高的 K 个个体进行真实环境评估
        top_k_indices = np.argsort(ehvi_scores)[-k_real_evals:]

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
        hof, p_hist, e_hist, f_hist = run_samo_gp(runner, pop_size=2, n_gens=2, k_real_evals=2)

        plot_and_save_results(hof, (p_hist, e_hist, f_hist))