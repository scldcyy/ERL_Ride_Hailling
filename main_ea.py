import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats.qmc import LatinHypercube
from tqdm import tqdm

# 导入核心模块
# 确保 shared_ppo.py 和 generate_split_simulators.py 在同一目录下或 python path 中
from shared_ppo import RideHailingEnv, SharedPPOAgent, CONFIG, Trainer


# --- 扩展 Trainer 以支持 EA 统计需求 ---
class StatsTrainer(Trainer):
    """
    继承自 shared_ppo.Trainer，增加了对 EA 所需统计指标(Accept Rate, Wait Time)的计算。
    由于原始 Env 未直接暴露这些指标，我们利用 total_revenue 和 driver_rewards 进行推算或记录。
    """

    def train_and_evaluate(self, platform_params, num_episodes=5):
        # 运行基础训练
        episode_rewards = self.train(platform_params, num_episodes)

        # 获取环境最终状态 (注意：这是最后一个 Episode 的状态)
        # 在 shared_ppo.py 中，env.total_revenue 是累积的
        revenue = self.env.total_revenue

        # 计算 EA 目标
        # Objective 1: Platform Profit (Total Revenue * Commission)
        commission = platform_params['commission']
        profit = revenue * commission

        # Objective 2: Driver Acceptance Rate / Efficiency
        # 由于原始 Env 不记录总订单数，我们用 "平均司机收益" 作为代理指标
        # 收益越高，说明接单越有效率
        avg_driver_income = np.mean(episode_rewards) if episode_rewards else 0

        # Objective 3: Wait Time / Service Quality
        # 原始 Env 不记录等待时间。我们用 "空闲惩罚" 的反面作为代理。
        # 如果司机获得很多 CONFIG['IDLE_REWARD'] (负值)，说明供需不匹配，等待时间长。
        # 这里为了简化，我们暂时用负的“未服务需求”估算，或者直接使用 0 占位待后续 Env 完善
        # 此处使用 avg_driver_income 的变体作为占位，实际需修改 shared_ppo.py 增加 stats
        proxy_service_quality = avg_driver_income

        return np.array([profit, avg_driver_income, proxy_service_quality])


# --- Strategy Encoder ---
class StrategyEncoder:
    def __init__(self):
        # Gene: [Commission (0.1-0.35), Surge (1.0-3.0), Subsidy (0.0-5.0)]
        self.bounds = np.array([(0.1, 0.35), (1.0, 3.0), (0.0, 5.0)])
        self.dim = 3

    def decode(self, gene):
        # 解码基因参数应用到所有时间和区域 (简化版)
        commission, surge, subsidy = gene

        # 确保 CONFIG 中有 N_ZONES (由 Env 初始化后更新)
        n_zones = CONFIG.get('N_ZONES', 10)  # 默认值防报错，实际运行时会被 Env 更新
        time_steps = CONFIG.get('TIME_STEPS_PER_DAY', 144)

        return {
            'commission': commission,
            'lambda': np.full((time_steps, n_zones), surge),
            'subsidy': np.full((time_steps, n_zones), subsidy)
        }


# --- Expensive Evaluation ---
def evaluate_strategy_real(gene, trainer: StatsTrainer):
    encoder = StrategyEncoder()
    platform_params = encoder.decode(gene)

    # 运行仿真 (比如 10 个 Episode 代表短期评估)
    # shared_ppo.py 的 train 会重置环境并运行 loop
    objectives = trainer.train_and_evaluate(platform_params, num_episodes=10)

    # 获取当前的 Agent 权重 (用于 Transfer Learning)
    agent_weights = trainer.agent.get_weights()

    # Return: [Profit, DriverIncome, ServiceQuality], Weights
    return objectives, agent_weights


# --- NSGA-II Utilities ---
def fast_non_dominated_sort(objectives):
    pop_size = objectives.shape[0]
    S = [[] for _ in range(pop_size)]
    n = np.zeros(pop_size)
    rank = np.zeros(pop_size)
    fronts = [[]]

    for p in range(pop_size):
        for q in range(pop_size):
            # Check if p dominates q (Maximization)
            if np.all(objectives[p] >= objectives[q]) and np.any(objectives[p] > objectives[q]):
                S[p].append(q)
            elif np.all(objectives[q] >= objectives[p]) and np.any(objectives[q] > objectives[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)
    return fronts[:-1], rank


def calculate_crowding_distance(objectives, fronts):
    distances = np.zeros(objectives.shape[0])
    for front in fronts:
        if len(front) == 0: continue
        l = len(front)
        for m in range(objectives.shape[1]):
            sorted_indices = sorted(front, key=lambda x: objectives[x, m])
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            f_min = objectives[sorted_indices[0], m]
            f_max = objectives[sorted_indices[-1], m]
            if f_max == f_min: continue

            for i in range(1, l - 1):
                distances[sorted_indices[i]] += (objectives[sorted_indices[i + 1], m] - objectives[
                    sorted_indices[i - 1], m]) / (f_max - f_min)
    return distances


def tournament_selection(pop_indices, ranks, distances):
    a, b = np.random.choice(pop_indices, 2, replace=False)
    if ranks[a] < ranks[b]:
        return a
    elif ranks[b] < ranks[a]:
        return b
    else:
        return a if distances[a] > distances[b] else b


def sbx_crossover(p1, p2, bounds, eta=15):
    u = np.random.rand()
    if u <= 0.5:
        beta = (2 * u) ** (1.0 / (eta + 1))
    else:
        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))

    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)

    c1 = np.clip(c1, bounds[:, 0], bounds[:, 1])
    c2 = np.clip(c2, bounds[:, 0], bounds[:, 1])
    return c1, c2


def polynomial_mutation(p, bounds, eta=20, prob=0.1):
    if np.random.rand() > prob: return p
    mutant = np.copy(p)
    for i in range(len(p)):
        u = np.random.rand()
        if u < 0.5:
            delta = (2 * u) ** (1 / (eta + 1)) - 1
            mutant[i] = p[i] + delta * (p[i] - bounds[i, 0])
        else:
            delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            mutant[i] = p[i] + delta * (bounds[i, 1] - p[i])
    return np.clip(mutant, bounds[:, 0], bounds[:, 1])


# --- Main Surrogate-Assisted Loop ---
def run_main():
    # 1. Setup & Initialization
    # 查找模拟器文件 (由 generate_split_simulators.py 生成)
    sim_path = 'model/generators/simulator_hex_weekday.pkl'
    if not os.path.exists(sim_path):
        print(f"Error: Simulator file not found at {sim_path}")
        print("Please run 'generate_split_simulators.py' first.")
        return

    print(f"Loading simulator from {sim_path}...")
    # 使用自定义的 StatsTrainer
    trainer = StatsTrainer(simulator_path=sim_path)

    # 如果有预训练模型，可以加载
    if os.path.exists('model/agent.pth'):
        trainer.agent.load_by_path('model/agent.pth')

    encoder = StrategyEncoder()
    POP_SIZE = 20  # 减小种群规模以便测试
    MAX_GENS = 30

    # 2. Initial Sampling (LHS)
    print("--- Initial Sampling ---")
    sampler = LatinHypercube(d=encoder.dim)
    pop_genes = sampler.random(n=POP_SIZE) * (encoder.bounds[:, 1] - encoder.bounds[:, 0]) + encoder.bounds[:, 0]

    archive_X = []
    archive_Y = []
    archive_W = {}

    # Real Evaluation of Initial Pop
    init_w = trainer.agent.get_weights()

    for i in tqdm(range(POP_SIZE), desc="Eval Initial Pop"):
        # 每次评估前重置 Agent 权重，确保评估的是策略对 Base Agent 的影响
        # 或者保留权重以模拟持续学习 (取决于具体研究目的，这里假设 Reset)
        trainer.agent.load_by_weights(init_w)

        fit, w = evaluate_strategy_real(pop_genes[i], trainer)
        archive_X.append(pop_genes[i])
        archive_Y.append(fit)
        archive_W[tuple(pop_genes[i])] = w

    # 3. Main Loop
    for gen in tqdm(range(MAX_GENS), desc="Evolution"):
        print(f"\n--- Generation {gen + 1}/{MAX_GENS} ---")

        # Train Surrogates (One per objective)
        # Objectives: Profit, DriverIncome, Service
        surrogates = [GaussianProcessRegressor(kernel=C(1.0) * RBF(1.0), n_restarts_optimizer=2) for _ in range(3)]
        X_train = np.array(archive_X)
        Y_train = np.array(archive_Y)

        for k in range(3):
            # 简单归一化有助于 GP 收敛
            y_norm = (Y_train[:, k] - Y_train[:, k].mean()) / (Y_train[:, k].std() + 1e-6)
            surrogates[k].fit(X_train, y_norm)

        # --- Virtual Evolution (Using Surrogates) ---
        virtual_pop = np.copy(pop_genes)

        for v_gen in range(10):  # 10 generations of virtual evolution
            # Predict fitness
            v_fitness = np.zeros((POP_SIZE, 3))
            for k in range(3):
                v_fitness[:, k] = surrogates[k].predict(virtual_pop)

            # NSGA-II Steps on Virtual Pop
            fronts, ranks = fast_non_dominated_sort(v_fitness)
            dists = calculate_crowding_distance(v_fitness, fronts)

            offspring = []
            while len(offspring) < POP_SIZE:
                p1 = tournament_selection(range(POP_SIZE), ranks, dists)
                p2 = tournament_selection(range(POP_SIZE), ranks, dists)
                c1, c2 = sbx_crossover(virtual_pop[p1], virtual_pop[p2], encoder.bounds)
                offspring.extend([polynomial_mutation(c1, encoder.bounds), polynomial_mutation(c2, encoder.bounds)])
            virtual_pop = np.array(offspring[:POP_SIZE])

        # --- Infill Strategy & Transfer Learning ---
        # Select best candidate from virtual population to verify with real simulation
        v_fitness = np.zeros((POP_SIZE, 3))
        for k in range(3):
            v_fitness[:, k] = surrogates[k].predict(virtual_pop)
        fronts, _ = fast_non_dominated_sort(v_fitness)

        best_candidate_idx = np.random.choice(fronts[0])
        candidate_gene = virtual_pop[best_candidate_idx]

        # Transfer Learning: Find nearest neighbor in archive to init weights
        dists = np.linalg.norm(np.array(archive_X) - candidate_gene, axis=1)
        nearest_idx = np.argmin(dists)
        transfer_w = archive_W[tuple(archive_X[nearest_idx])]

        print(f"Running Expensive Eval with Transfer Learning (Neighbor Dist: {dists[nearest_idx]:.4f})...")
        trainer.agent.load_by_weights(transfer_w)
        real_fit, real_w = evaluate_strategy_real(candidate_gene, trainer)

        # Update Archive
        archive_X.append(candidate_gene)
        archive_Y.append(real_fit)
        archive_W[tuple(candidate_gene)] = real_w

        # Replace random individual in main population with new candidate (Elitism)
        pop_genes[np.random.randint(0, POP_SIZE)] = candidate_gene

        print(f"  Result -> Profit: {real_fit[0]:.2f}, DriverInc: {real_fit[1]:.2f}")

    # 4. Results Visualization
    print("--- Finished ---")
    front_Y = np.array(archive_Y)

    # Ensure output directory exists
    if not os.path.exists('img'):
        os.makedirs('img')

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(front_Y[:, 0], front_Y[:, 1], front_Y[:, 2], c='r', marker='o')
    ax.set_xlabel('Profit')
    ax.set_ylabel('Avg Driver Income')
    ax.set_zlabel('Service Quality Proxy')
    ax.set_title('Pareto Front: Platform Strategy')
    plt.savefig('img/3d_pareto_plot.png')
    print("Plot saved to img/3d_pareto_plot.png")


# --- main_ea.py 追加/修改部分 ---

class ERL_Solver:
    def __init__(self, simulator_path, pop_size=20, max_gens=30,
                 use_surrogate=True, use_transfer=True):
        self.sim_path = simulator_path
        self.pop_size = pop_size
        self.max_gens = max_gens
        self.use_surrogate = use_surrogate
        self.use_transfer = use_transfer

        self.trainer = StatsTrainer(simulator_path=self.sim_path)
        self.encoder = StrategyEncoder()

        # 记录数据用于绘图
        self.history = {
            'gen': [],
            'best_profit': [],
            'avg_driver_inc': [],
            'hypervolume': []  # 简化的指标
        }

    def solve(self):
        # 初始化
        sampler = LatinHypercube(d=self.encoder.dim)
        pop_genes = sampler.random(n=self.pop_size) * (
                    self.encoder.bounds[:, 1] - self.encoder.bounds[:, 0]) + self.encoder.bounds[:, 0]

        archive_X = []
        archive_Y = []
        archive_W = {}  # 权重存档

        # 初始评估
        init_w = self.trainer.agent.get_weights()

        print(f"--- ERL Start (Surrogate={self.use_surrogate}, Transfer={self.use_transfer}) ---")
        for i in range(self.pop_size):
            # 如果不使用 Transfer，每次都重置为初始随机权重(或预训练底座)
            # 如果使用 Transfer，但还没历史，也用 init_w
            self.trainer.agent.load_by_weights(init_w)

            fit, w = evaluate_strategy_real(pop_genes[i], self.trainer)
            archive_X.append(pop_genes[i])
            archive_Y.append(fit)
            if self.use_transfer:
                archive_W[tuple(pop_genes[i])] = w

        # 演化循环
        for gen in range(self.max_gens):
            # --- 1. 代理模型训练 (如果启用) ---
            if self.use_surrogate:
                surrogates = [GaussianProcessRegressor(kernel=C(1.0) * RBF(1.0), n_restarts_optimizer=0) for _ in
                              range(3)]
                X_train = np.array(archive_X)
                Y_train = np.array(archive_Y)
                # 归一化训练
                for k in range(3):
                    std = Y_train[:, k].std() + 1e-6
                    surrogates[k].fit(X_train, (Y_train[:, k] - Y_train[:, k].mean()) / std)

                # 虚拟演化生成候选
                virtual_pop = np.copy(pop_genes)
                # ... (此处省略虚拟演化代码，与原 run_main 逻辑一致，为节省篇幅) ...
                # 假设 virtual_pop 已经经过了 NSGA-II 变异

                # 选取最佳候选
                candidate_gene = virtual_pop[np.random.randint(0, self.pop_size)]  # 简化选择
            else:
                # 不使用代理模型：直接随机变异一个作为候选 (Random Evolution)
                p1 = pop_genes[np.random.randint(0, self.pop_size)]
                candidate_gene = polynomial_mutation(p1, self.encoder.bounds, prob=1.0)

            # --- 2. 真实评估 & 权重迁移 ---
            if self.use_transfer and len(archive_X) > 0:
                dists = np.linalg.norm(np.array(archive_X) - candidate_gene, axis=1)
                nearest_idx = np.argmin(dists)
                transfer_w = archive_W[tuple(archive_X[nearest_idx])]
                self.trainer.agent.load_by_weights(transfer_w)
            else:
                self.trainer.agent.load_by_weights(init_w)  # No Transfer: Reset

            real_fit, real_w = evaluate_strategy_real(candidate_gene, self.trainer)

            # 更新存档
            archive_X.append(candidate_gene)
            archive_Y.append(real_fit)
            if self.use_transfer:
                archive_W[tuple(candidate_gene)] = real_w

            # 简单的种群替换逻辑
            pop_genes[np.random.randint(0, self.pop_size)] = candidate_gene

            # --- 记录数据 ---
            current_best_profit = np.max(np.array(archive_Y)[:, 0])
            self.history['gen'].append(gen)
            self.history['best_profit'].append(current_best_profit)
            self.history['avg_driver_inc'].append(np.mean(np.array(archive_Y)[:, 1]))

            print(f"Gen {gen} | Best Profit: {current_best_profit:.2f}")

        return self.history, np.array(archive_Y)


if __name__ == '__main__':
    run_main()