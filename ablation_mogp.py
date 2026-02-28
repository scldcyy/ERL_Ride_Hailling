import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from deap import tools, algorithms

# 引入基础配置和现有 MOGP 环境
from MOGP_platform import toolbox, generate_lhs_initial_population, SAMO_GP_Runner


class AblationRunner(SAMO_GP_Runner):
    def __init__(self, sim_path):
        super().__init__(sim_path)
        self.real_eval_count = 0  # 全局真实评估计数器

    def evaluate_real_ablation(self, individual, use_hot_start, init_mode=False, num_episodes=10):
        params = self.ind_to_params(individual)

        # 无论如何，先重置底层权重清空经验池
        self.trainer.reset_to_base_weights()
        self.trainer.agent.optimizer.state.clear()

        # 热启动逻辑
        if use_hot_start and len(self.archive_params) > 0 and not init_mode:
            current_pheno = self.surrogate._get_phenotype(params)
            phenotypes = np.array([self.surrogate._get_phenotype(p) for p in self.archive_params])
            distances = np.linalg.norm(phenotypes - current_pheno, axis=1)
            best_idx = np.argmin(distances)
            best_weights = self.archive_weights[best_idx]
            self.trainer.agent.load_by_weights(best_weights)

        # 真实环境运行
        fitness = self.trainer.train_and_evaluate(params, num_episodes=num_episodes)

        self.archive_params.append(params)
        self.archive_fitness.append(fitness[:3])
        self.archive_inds.append(individual)
        self.archive_weights.append(self.trainer.agent.get_weights())
        self.real_eval_count += 1  # 严格计步
        return tuple(fitness[:3])


def run_ablation_variant(runner, variant_name, max_evals=300, pop_size=30,
                         use_surrogate=True, use_hot_start=True, k_real_evals=5):
    print(f"\n{'=' * 50}\nStarting {variant_name}\n{'=' * 50}")

    pop = generate_lhs_initial_population(toolbox, runner, pop_size)
    hof = tools.ParetoFront()
    history_profit = []

    # 1. 安全初始化：如果初始种群评估过程中达到预算，只保留已评估的个体
    valid_pop = []
    for ind in pop:
        if runner.real_eval_count >= max_evals:
            break
        ind.fitness.values = runner.evaluate_real_ablation(ind, use_hot_start=False, init_mode=True)
        ind.is_real_evaluated = True
        valid_pop.append(ind)
    pop = valid_pop

    if use_surrogate and len(runner.archive_params) > 0:
        runner.surrogate.update(runner.archive_params, runner.archive_fitness)
    hof.update(pop)

    gen = 1
    while runner.real_eval_count < max_evals:
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.8, mutpb=0.3)

        valid_offspring = []  # 核心修复：只收集成功赋予适应度的子代

        if use_surrogate:
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

            top_k_indices = set(np.argsort(ehvi_scores)[-k_real_evals:])

            for i, ind in enumerate(offspring):
                # 即使是被选中真实评估的个体，如果预算耗尽，也降级为代理模型评估
                if i in top_k_indices and runner.real_eval_count < max_evals:
                    ind.fitness.values = runner.evaluate_real_ablation(ind, use_hot_start)
                    ind.is_real_evaluated = True
                else:
                    ind.fitness.values = runner.evaluate_surrogate(ind)
                    ind.is_real_evaluated = False
                valid_offspring.append(ind)
        else:
            # 无代理模型时，一旦预算耗尽，直接跳过剩余子代
            for ind in offspring:
                if runner.real_eval_count >= max_evals:
                    break
                ind.fitness.values = runner.evaluate_real_ablation(ind, use_hot_start)
                ind.is_real_evaluated = True
                valid_offspring.append(ind)

        if use_surrogate and len(runner.archive_params) > 0:
            runner.surrogate.update(runner.archive_params, runner.archive_fitness)

        # 2. 筛选：只对合法的 pop 和 valid_offspring 进行选择
        pop = toolbox.select(pop + valid_offspring, k=min(pop_size, len(pop) + len(valid_offspring)))

        real_evaluated_inds = [ind for ind in (pop + valid_offspring) if getattr(ind, 'is_real_evaluated', False)]
        if real_evaluated_inds:
            hof.update(real_evaluated_inds)

        current_max_profit = np.max(np.array(runner.archive_fitness)[:, 0]) if len(runner.archive_fitness) > 0 else 0
        history_profit.append(current_max_profit)
        print(
            f"[{variant_name}] Gen {gen} | Evals: {runner.real_eval_count}/{max_evals} | Best Profit: {current_max_profit:.2f}")
        gen += 1

    return history_profit


if __name__ == "__main__":
    sim_path = 'model/generators/simulator_hex_scaling=0.004257843312339327_weekday.pkl'
    MAX_EVALS = 250  # 严格限制所有算法的计算预算

    variants = {
        "SAMO-GP (Surrogate + HotStart)": (True, True),
        "MOGP + Surrogate Only": (True, False),
        "MOGP + HotStart Only": (False, True),
        "Vanilla MOGP": (False, False)
    }

    results = {}

    for name, (use_surr, use_hot) in variants.items():
        runner = AblationRunner(sim_path)
        history = run_ablation_variant(runner, name, max_evals=MAX_EVALS,
                                       use_surrogate=use_surr, use_hot_start=use_hot)
        results[name] = history

    # --- 绘制收敛图 ---
    plt.figure(figsize=(10, 6))
    for name, history in results.items():
        # X轴映射到真实评估次数
        evals_timeline = np.linspace(30, MAX_EVALS, len(history))  # 假设初始种群30
        plt.plot(evals_timeline, history, label=name, linewidth=2)

    plt.title('Ablation Study: Max Profit over Real Evaluations Budget')
    plt.xlabel('Number of Real Environment Evaluations')
    plt.ylabel('Maximum Platform Profit Found')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/mogp_ablation.png')
    plt.show()

    with open("results/ablation_data.pkl", "wb") as f:
        pickle.dump(results, f)

    # with open("results/ablation_data.pkl", "rb") as f:
    #     data= pickle.load(f)
    #     pass