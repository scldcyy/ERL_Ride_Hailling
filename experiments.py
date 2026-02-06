import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys
from scipy.stats.qmc import LatinHypercube
from tqdm import tqdm

# 导入项目模块
from main_ea import ERL_Solver, StatsTrainer, StrategyEncoder, evaluate_strategy_real, polynomial_mutation
from shared_ppo import CONFIG, GreedyAgent

# 样式设置
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ExperimentSuite:
    def __init__(self, sim_path='model/generators/simulator_hex_scaling=0.010218823949614386_weekday.pkl'):
        self.sim_path = sim_path
        if not os.path.exists(self.sim_path):
            raise FileNotFoundError(f"Simulator not found at {self.sim_path}")
        self.img_dir = 'img_experiments'
        self.data_dir = 'data_experiments'
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    def run_sota_comparison(self, max_gens=30, pop_size=10):
        print(f"\n[Experiment 1] Running SOTA Comparison (Gens={max_gens}, Pop={pop_size})...")
        results = []

        # --- Method 1: Vanilla PPO (Deep RL Baseline) ---
        print(">>> Running Baseline 1: Vanilla PPO (Standard DRL)...")
        trainer_ppo = StatsTrainer(self.sim_path)
        # 固定参数 (Industry Standard)
        fixed_params = {
            'commission': 0.25,
            'lambda': np.full((CONFIG['TIME_STEPS_PER_DAY'], CONFIG['N_ZONES']), 1.0),
            'subsidy': np.zeros((CONFIG['TIME_STEPS_PER_DAY'], CONFIG['N_ZONES']))
        }

        # 每一代训练量 = Pop_Size * Episodes_Per_Ind (这里设为5)
        episodes_per_gen = pop_size * 5

        # 记录初始性能
        metrics = trainer_ppo.train_and_evaluate(fixed_params, num_episodes=5)
        results.append(
            {'Method': 'Vanilla PPO', 'Gen': 0, 'Profit': metrics[0], 'DriverInc': metrics[1], 'Quality': metrics[2]})

        for g in tqdm(range(1, max_gens), desc="Vanilla PPO"):
            # 训练一个 Generation 的量
            metrics = trainer_ppo.train_and_evaluate(fixed_params, num_episodes=episodes_per_gen // 5)
            results.append({'Method': 'Vanilla PPO', 'Gen': g, 'Profit': metrics[0], 'DriverInc': metrics[1],
                            'Quality': metrics[2]})

        # --- Method 2: GA-Greedy (Heuristic Baseline) ---
        print(">>> Running Baseline 2: GA-Greedy (Evolution + Heuristic)...")
        trainer_greedy = StatsTrainer(self.sim_path)
        n_zones = trainer_greedy.env.n_zones
        # 替换智能体为 GreedyAgent
        trainer_greedy.agent = GreedyAgent(trainer_greedy.env.adjacency_indices, n_zones)

        encoder = StrategyEncoder()
        sampler = LatinHypercube(d=encoder.dim)
        pop = sampler.random(n=pop_size) * (encoder.bounds[:, 1] - encoder.bounds[:, 0]) + encoder.bounds[:, 0]

        for g in tqdm(range(max_gens), desc="GA-Greedy"):
            gen_profits, gen_incomes, gen_qualities = [], [], []
            for i in range(pop_size):
                metrics, _ = evaluate_strategy_real(pop[i], trainer_greedy)
                gen_profits.append(metrics[0])
                gen_incomes.append(metrics[1])
                gen_qualities.append(metrics[2])

            best_idx = np.argmax(gen_profits)
            results.append(
                {'Method': 'GA-Greedy', 'Gen': g, 'Profit': gen_profits[best_idx], 'DriverInc': gen_incomes[best_idx],
                 'Quality': gen_qualities[best_idx]})

            # Simple Evolution
            best_gene = pop[best_idx]
            new_pop = [best_gene]
            for _ in range(pop_size - 1):
                new_pop.append(polynomial_mutation(best_gene, encoder.bounds, prob=1.0))
            pop = np.array(new_pop)

        # --- Method 3: H-ERL (Ours) ---
        print(">>> Running Ours: H-ERL (Hybrid)...")
        solver = ERL_Solver(self.sim_path, pop_size=pop_size, max_gens=max_gens, use_surrogate=True, use_transfer=True)
        hist, full_archive = solver.solve()

        for g, p, inc in zip(hist['gen'], hist['best_profit'], hist['avg_driver_inc']):
            # history 字典里没有 Quality 字段，此处为演示保持一致性填0，实际可修改 ERL_Solver 记录
            results.append({'Method': 'H-ERL (Ours)', 'Gen': g, 'Profit': p, 'DriverInc': inc, 'Quality': 0})

        # --- Plotting ---
        df = pd.DataFrame(results)
        df.to_csv(f"{self.data_dir}/sota_results.csv", index=False)

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Gen', y='Profit', hue='Method', style='Method', markers=True, linewidth=2.5)
        plt.title('Convergence Comparison: Platform Profit')
        plt.savefig(f"{self.img_dir}/exp1_sota_convergence.png", dpi=300)
        print(f"Saved {self.img_dir}/exp1_sota_convergence.png")

    def run_ablation_study(self, max_gens=20, pop_size=10):
        print(f"\n[Experiment 2] Running Ablation Study...")
        results = []

        # Config 1: w/ Transfer
        solver_w = ERL_Solver(self.sim_path, pop_size=pop_size, max_gens=max_gens, use_surrogate=True,
                              use_transfer=True)
        hist_w, _ = solver_w.solve()
        for g, p in zip(hist_w['gen'], hist_w['best_profit']):
            results.append({'Config': 'H-ERL (Full)', 'Gen': g, 'Profit': p})

        # Config 2: w/o Transfer
        solver_wo = ERL_Solver(self.sim_path, pop_size=pop_size, max_gens=max_gens, use_surrogate=True,
                               use_transfer=False)
        hist_wo, _ = solver_wo.solve()
        for g, p in zip(hist_wo['gen'], hist_wo['best_profit']):
            results.append({'Config': 'w/o Transfer', 'Gen': g, 'Profit': p})

        df = pd.DataFrame(results)
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Gen', y='Profit', hue='Config', style='Config', markers=True)
        plt.title('Ablation Study: Transfer Learning')
        plt.savefig(f"{self.img_dir}/exp2_ablation.png", dpi=300)
        print(f"Saved {self.img_dir}/exp2_ablation.png")

    def run_scalability_test(self):
        print(f"\n[Experiment 3] Scalability Test...")
        driver_counts = [2000, 4000, 6000]
        summary = []
        original_n = CONFIG['N_DRIVERS']

        try:
            for n in driver_counts:
                CONFIG['N_DRIVERS'] = n
                # 使用较小的参数快速验证
                solver = ERL_Solver(self.sim_path, pop_size=5, max_gens=5, use_surrogate=False, use_transfer=True)
                _, archive_Y = solver.solve()
                summary.append({
                    'Drivers': n,
                    'MaxProfit': np.max(archive_Y[:, 0]),
                    'AvgDriverInc': np.mean(archive_Y[:, 1])
                })
        finally:
            CONFIG['N_DRIVERS'] = original_n

        df = pd.DataFrame(summary)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('Drivers')
        ax1.set_ylabel('Profit', color='tab:blue')
        sns.lineplot(data=df, x='Drivers', y='MaxProfit', marker='o', color='tab:blue', ax=ax1)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Driver Income', color='tab:orange')
        sns.lineplot(data=df, x='Drivers', y='AvgDriverInc', marker='s', color='tab:orange', ax=ax2)
        plt.title('Scalability Analysis')
        plt.savefig(f"{self.img_dir}/exp3_scalability.png", dpi=300)
        print(f"Saved {self.img_dir}/exp3_scalability.png")


if __name__ == "__main__":
    suite = ExperimentSuite()
    # 依次运行所有实验
    suite.run_sota_comparison(max_gens=30, pop_size=10)
    suite.run_ablation_study(max_gens=20, pop_size=10)
    suite.run_scalability_test()