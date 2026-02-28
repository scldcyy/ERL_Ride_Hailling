import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
import math

from scipy.spatial.distance import cdist

# 导入底层环境和代理模型
from shared_ppo import Trainer
from surrogate_model import SurrogateModel


# --- 1. 定义 DE 优化的固定公式模板 ---
# DE 将优化一个长度为 8 的连续实数向量 [p0, p1, ..., p7]
def parameterized_surge(t, no, nd, sd, p):
    # p[0] 到 p[3] 控制 surge
    return p[0] + p[1] * sd + p[2] * np.log(sd + 1e-6) + p[3] * t


def parameterized_subsidy(t, no, nd, sd, p):
    # p[4] 到 p[7] 控制 subsidy
    return p[4] + p[5] * sd + p[6] * np.log(sd + 1e-6) * np.sin(t) + p[7] * t


# --- 2. 配置 DEAP 连续优化环境 ---
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# 定义参数边界 (例如: 权重在 -3.0 到 3.0 之间)
MIN_BOUND, MAX_BOUND = -3.0, 3.0


def uniform_float():
    return random.uniform(MIN_BOUND, MAX_BOUND)


toolbox.register("attr_float", uniform_float)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=8)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 注册针对实数编码的多目标交叉与变异 (Simulated Binary Crossover & Polynomial Mutation)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=MIN_BOUND, up=MAX_BOUND, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=MIN_BOUND, up=MAX_BOUND, eta=20.0, indpb=1.0 / 8)
toolbox.register("select", tools.selNSGA2)


# --- 3. 评估与代理模型逻辑 ---
class SAMO_DE_Runner:
    def __init__(self, sim_path):
        self.trainer = Trainer(simulator_path=sim_path)
        self.surrogate = SurrogateModel(n_reference_states=50)
        self.archive_params = []
        self.archive_fitness = []
        self.archive_inds = []  # --- ALIGNMENT: Track DE individuals ---
        self.archive_weights = []  # --- ALIGNMENT: Track PPO weights ---

    def ind_to_params(self, individual):
        """将 DE 的连续向量映射到 platform_params 字典"""
        p = np.array(individual)
        return {
            'surge': lambda t, no, nd, sd: parameterized_surge(t, no, nd, sd, p),
            'subsidy': lambda t, no, nd, sd: parameterized_subsidy(t, no, nd, sd, p)
        }

    def evaluate_real(self, individual, num_episodes=10, init_mode=False):
        params = self.ind_to_params(individual)
        # FIX: Bypass hot start if in init_mode
        if len(self.archive_params) > 0 and not init_mode:
            current_pheno = self.surrogate._get_phenotype(params)
            phenotypes = np.array([self.surrogate._get_phenotype(p) for p in self.archive_params])
            distances = np.linalg.norm(phenotypes - current_pheno, axis=1)
            best_idx = np.argmin(distances)
            best_weights = self.archive_weights[best_idx]
            self.trainer.agent.load_by_weights(best_weights)
        else:
            self.trainer.reset_to_base_weights()
        fitness = self.trainer.train_and_evaluate(params, num_episodes=num_episodes)

        self.archive_params.append(params)
        self.archive_fitness.append(fitness[:3])
        self.archive_inds.append(individual)  # --- ALIGNMENT ---
        self.archive_weights.append(self.trainer.agent.get_weights())  # --- ALIGNMENT: Actually save the weights for Hot Start! ---
        return tuple(fitness[:3])

    def evaluate_surrogate(self, individual):
        params = self.ind_to_params(individual)
        mu, _ = self.surrogate.predict(params)
        return tuple(mu)

def save_metrics(gen, hof, history, filename="results/de_metrics.pkl"):
    """仅保存帕累托前沿个体的参数和当前的收敛历史曲线"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    metrics_data = {
        'gen': gen,
        'pareto_front_params': [list(ind) for ind in hof],
        'pareto_fitness': [ind.fitness.values for ind in hof],
        'history': history
    }
    with open(filename, 'wb') as f:
        pickle.dump(metrics_data, f)
    print(f" [Save] Pareto front and metrics saved at Generation {gen}.")

# --- ALIGNMENT: LHS Initialization ---
def generate_lhs_initial_population(toolbox, runner, pop_size):
    """
    通过表现型空间的最大最小距离过滤，生成具有 LHS 特性的初始 DE 种群。
    """
    print("Generating extended pool for LHS Phenotype selection...")
    pool_size = pop_size * 10
    candidate_pool = toolbox.population(n=pool_size)

    phenotypes = []
    valid_candidates = []

    for ind in candidate_pool:
        params = runner.ind_to_params(ind)
        pheno = runner.surrogate._get_phenotype(params)

        if not np.any(np.isnan(pheno)) and np.max(np.abs(pheno)) < 100:
            phenotypes.append(pheno)
            valid_candidates.append(ind)

    phenotypes = np.array(phenotypes)

    # --- 修复 1：动态调整实际挑选数量 ---
    actual_pop_size = min(pop_size, len(valid_candidates))
    if actual_pop_size == 0:
        raise ValueError("致命错误: 未能生成任何有效的 DE 初始向量！")

    selected_indices = [random.randint(0, len(valid_candidates) - 1)]

    while len(selected_indices) < pop_size:
        selected_phenos = phenotypes[selected_indices]
        dists = cdist(phenotypes, selected_phenos)
        min_dists = np.min(dists, axis=1)
        min_dists[selected_indices] = -1
        next_idx = np.argmax(min_dists)
        selected_indices.append(next_idx)

    initial_pop = [valid_candidates[i] for i in selected_indices]
    print(f"LHS Initialization complete. Selected {pop_size} diverse vectors.")
    return initial_pop

# --- 4. 主循环 (与 GP 逻辑一致，保证公平比对) ---
def run_samo_de(runner, pop_size=40, n_gens=20, k_real_evals=5):
    metrics_file = "results/de_metrics.pkl"

    # 全新启动 (使用 LHS 初始化)
    pop = generate_lhs_initial_population(toolbox, runner, pop_size)
    hof = tools.ParetoFront()
    history_max_profit, history_max_efficiency, history_max_fairness = [], [], []

    print("=== Generation 0: Initializing Surrogate with Real Evaluations ===")
    for ind in pop:
        ind.fitness.values = runner.evaluate_real(ind, num_episodes=10, init_mode=True)
        ind.is_real_evaluated = True  # <-- 打上真实评估标签

    runner.surrogate.update(runner.archive_params, runner.archive_fitness)
    hof.update(pop)

    # 初始代保存
    save_metrics(0, hof, (history_max_profit, history_max_efficiency, history_max_fairness), metrics_file)

    # 进化循环
    for gen in range(1, n_gens + 1):
        print(f"\n=== Generation {gen}/{n_gens} (MO-DE) ===")
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.9, mutpb=0.2)

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
                ind.is_real_evaluated = True  # <-- 真实评估
            else:
                ind.fitness.values = runner.evaluate_surrogate(ind)
                ind.is_real_evaluated = False  # <-- 代理预测，打上假标签

        runner.surrogate.update(runner.archive_params, runner.archive_fitness)
        pop = toolbox.select(pop + offspring, k=pop_size)
        # --- 修复 2：严格守卫 DE 的帕累托前沿 ---
        real_evaluated_inds = [ind for ind in (pop + offspring) if getattr(ind, 'is_real_evaluated', False)]
        if real_evaluated_inds:
            hof.update(real_evaluated_inds)

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
    plt.savefig(f"{save_dir}/samode_convergence.png")
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

    plt.savefig(f"{save_dir}/samode_pareto_3d.png")
    plt.show()

    # --- 保存结果用于 Baseline 对比 ---
    results_data = {
        'pareto_fitness': np.column_stack((p_vals, e_vals, f_vals)),
        'convergence': history,
        'parameters': [list(ind) for ind in hof]
    }


    with open(f"{save_dir}/samode_results.pkl", 'wb') as f:
        pickle.dump(results_data, f)

    print(f"\nDE Baseline results saved to {save_dir}/samode_results.pkl")
    print("\nSample Best Formulas found by SAMO-GP:")
    for i in range(min(3, len(hof))):
        print(f"[{i + 1}] Surge & Subsidy: {hof[i]}")


if __name__ == '__main__':
    sim_path = 'model/generators/simulator_hex_scaling=0.004257843312339327_weekday.pkl'
    if not os.path.exists(sim_path):
        print(f"Simulator file not found at {sim_path}")
    else:
        runner = SAMO_DE_Runner(sim_path)
        hof, p_hist, e_hist, f_hist = run_samo_de(runner, pop_size=50, n_gens=50, k_real_evals=10)
        plot_and_save_results(hof, (p_hist, e_hist, f_hist))
