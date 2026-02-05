import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys

# 导入你的项目模块
from main_ea import ERL_Solver, StatsTrainer, StrategyEncoder, evaluate_strategy_real
from shared_ppo import CONFIG

# 样式设置
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class ExperimentSuite:
    def __init__(self, sim_path='model/generators/simulator_hex_weekday.pkl'):
        self.sim_path = sim_path
        if not os.path.exists(self.sim_path):
            raise FileNotFoundError(f"Simulator not found at {self.sim_path}")

    def run_baseline_comparison(self):
        """
        实验 1: 算法对比
        对比:
        1. Static Strategy (Fixed Params) + PPO Dispatch
        2. Random Search (No Surrogate) + PPO Dispatch
        3. Ours (Surrogate-Assisted ERL)
        """
        print("\n=== Experiment 1: Baseline Comparison ===")
        results = []

        # 1. Static Strategy Baseline
        # 固定参数: Commission=0.25, Surge=1.0, Subsidy=0.0
        print("Running Baseline: Static Strategy...")
        trainer = StatsTrainer(self.sim_path)
        static_params = {'commission': 0.25,
                         'lambda': np.full((CONFIG['TIME_STEPS_PER_DAY'], CONFIG['N_ZONES']), 1.0),
                         'subsidy': np.zeros((CONFIG['TIME_STEPS_PER_DAY'], CONFIG['N_ZONES']))}
        # 运行较长时间以模拟收敛
        static_obj = trainer.train_and_evaluate(static_params, num_episodes=20)
        # 将静态结果复制多次以便绘图对比
        for g in range(30):
            results.append({'Method': 'Static Baseline', 'Gen': g, 'Profit': static_obj[0]})

        # 2. Random Search (Ablation: No Surrogate, No Transfer)
        print("Running Baseline: Random Search...")
        solver_random = ERL_Solver(self.sim_path, pop_size=10, max_gens=30,
                                   use_surrogate=False, use_transfer=False)
        hist_random, _ = solver_random.solve()
        for g, p in zip(hist_random['gen'], hist_random['best_profit']):
            results.append({'Method': 'Random Search', 'Gen': g, 'Profit': p})

        # 3. Our Method (ERL)
        print("Running Ours: ERL...")
        solver_ours = ERL_Solver(self.sim_path, pop_size=10, max_gens=30,
                                 use_surrogate=True, use_transfer=True)
        hist_ours, _ = solver_ours.solve()
        for g, p in zip(hist_ours['gen'], hist_ours['best_profit']):
            results.append({'Method': 'Ours (ERL)', 'Gen': g, 'Profit': p})

        # 绘图
        df = pd.DataFrame(results)
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Gen', y='Profit', hue='Method', marker='o')
        plt.title('Convergence Comparison: Platform Profit')
        plt.ylabel('Max Profit ($)')
        plt.xlabel('Generation')
        plt.savefig('img/exp1_baseline_comparison.png')
        print("Saved img/exp1_baseline_comparison.png")

    def run_ablation_study(self):
        """
        实验 2: 消融实验
        验证 Transfer Learning 的有效性
        """
        print("\n=== Experiment 2: Ablation Study (Transfer Learning) ===")
        results = []

        # 1. With Transfer (Ours)
        # (复用上面的数据，或者重跑)
        print("Running: With Transfer...")
        solver_transfer = ERL_Solver(self.sim_path, pop_size=10, max_gens=20, use_transfer=True)
        hist_t, _ = solver_transfer.solve()

        # 2. Without Transfer
        print("Running: Without Transfer...")
        solver_no_transfer = ERL_Solver(self.sim_path, pop_size=10, max_gens=20, use_transfer=False)
        hist_nt, _ = solver_no_transfer.solve()

        # 整理数据
        for i in range(len(hist_t['gen'])):
            results.append({'Config': 'w/ Transfer (Ours)', 'Gen': i, 'Profit': hist_t['best_profit'][i]})
            results.append({'Config': 'w/o Transfer', 'Gen': i, 'Profit': hist_nt['best_profit'][i]})

        df = pd.DataFrame(results)
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Gen', y='Profit', hue='Config', style='Config', markers=True)
        plt.title('Ablation Study: Effectiveness of Transfer Learning')
        plt.savefig('img/exp2_ablation_transfer.png')
        print("Saved img/exp2_ablation_transfer.png")

    def run_scalability_test(self):
        """
        实验 3: 规模扩展性
        改变 CONFIG['N_DRIVERS'] 观察性能
        注意: 这需要重新修改全局 CONFIG，可能需要重载环境
        """
        print("\n=== Experiment 3: Scalability Test ===")
        driver_counts = [2000, 4000, 6000]
        final_profits = []
        final_incomes = []

        original_n = CONFIG['N_DRIVERS']

        try:
            for n_drivers in driver_counts:
                print(f"Testing Scale: {n_drivers} Drivers...")
                # 动态修改全局配置
                CONFIG['N_DRIVERS'] = n_drivers

                # 运行简短的 ERL
                solver = ERL_Solver(self.sim_path, pop_size=5, max_gens=10)
                _, archive_Y = solver.solve()

                # 取 Pareto 前沿的平均值或最大值
                best_profit = np.max(archive_Y[:, 0])
                avg_income = np.mean(archive_Y[:, 1])

                final_profits.append(best_profit)
                final_incomes.append(avg_income)

        finally:
            # 还原配置
            CONFIG['N_DRIVERS'] = original_n

        # 双轴绘图
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:red'
        ax1.set_xlabel('Number of Drivers')
        ax1.set_ylabel('Platform Profit ($)', color=color)
        ax1.plot(driver_counts, final_profits, color=color, marker='o', label='Profit')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Avg Driver Income ($)', color=color)
        ax2.plot(driver_counts, final_incomes, color=color, marker='s', linestyle='--', label='Driver Income')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Scalability Analysis: Performance vs Problem Size')
        fig.tight_layout()
        plt.savefig('img/exp3_scalability.png')
        print("Saved img/exp3_scalability.png")


if __name__ == "__main__":
    # 确保保存目录存在
    if not os.path.exists('img'):
        os.makedirs('img')

    suite = ExperimentSuite()

    # 建议依次运行，或者注释掉不想跑的部分
    # 注意：运行完整的实验非常耗时，可以减少 max_gens 和 pop_size 进行代码测试

    suite.run_baseline_comparison()
    suite.run_ablation_study()
    suite.run_scalability_test()