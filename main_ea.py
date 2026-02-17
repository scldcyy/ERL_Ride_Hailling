import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as GP_RBF, ConstantKernel as C, WhiteKernel
from scipy.stats.qmc import LatinHypercube
from tqdm import tqdm
import h3  # 需要安装 h3-py
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from shared_ppo import RideHailingEnv, SharedPPOAgent, CONFIG, Trainer


# --- 引入 RBF 空间编码器 ---
class SpatialStrategyEncoder:
    def __init__(self, zone_coords, n_hotspots=5):
        """
        zone_coords: shape (N_zones, 2), 归一化的经纬度坐标 [0, 1]
        n_hotspots: RBF核的数量，即允许地图上有多少个独立的定价/补贴中心
        """
        self.zone_coords = zone_coords
        self.n_zones = len(zone_coords)
        self.n_hotspots = n_hotspots

        # 基因定义:
        # [0]: Global Base Commission [0.1, 0.35]
        # [1]: Global Base Surge [1.0, 1.2]
        # [2]: Global Base Subsidy [0.0, 1.0]
        # 接下来每个Hotspot有4个参数: [weight, center_x, center_y, sigma]
        # Weight (强度): [-1.0, 1.0] -> 负数代表低价区/高补贴区，正数代表高价区
        # Center X, Y: [0.0, 1.0] -> 在地图上的位置
        # Sigma (范围): [0.05, 0.3] -> 覆盖半径

        self.base_dim = 3
        self.hotspot_dim = 4
        self.dim = self.base_dim + self.n_hotspots * self.hotspot_dim

        # 构建边界
        self.bounds = []
        # Base params
        self.bounds.extend([(0.1, 0.35), (1.0, 1.2), (0.0, 1.0)])
        # Hotspot params
        for _ in range(n_hotspots):
            self.bounds.append((-1.0, 2.0))  # Weight: Surge增量倍数
            self.bounds.append((0.0, 1.0))  # Center X
            self.bounds.append((0.0, 1.0))  # Center Y
            self.bounds.append((0.05, 0.3))  # Sigma (Width)

        self.bounds = np.array(self.bounds)

    def decode(self, gene):
        base_comm, base_surge, base_sub = gene[0], gene[1], gene[2]

        # 初始化空间分布 (N_zones, )
        spatial_surge_offset = np.zeros(self.n_zones)
        spatial_subsidy_offset = np.zeros(self.n_zones)

        # 叠加每个 RBF Hotspot 的影响
        for k in range(self.n_hotspots):
            idx = self.base_dim + k * self.hotspot_dim
            w, cx, cy, sigma = gene[idx:idx + 4]

            # 计算所有区域到该 Hotspot 中心的欧氏距离平方
            d2 = (self.zone_coords[:, 0] - cx) ** 2 + (self.zone_coords[:, 1] - cy) ** 2
            # RBF 激活值
            activation = np.exp(-d2 / (2 * sigma ** 2))

            # w > 0 增加定价(Surge)， w < 0 增加补贴(Subsidy)
            # 这里做一个策略解耦：正权重贡献给Surge，负权重贡献给Subsidy
            if w >= 0:
                spatial_surge_offset += w * activation
            else:
                spatial_subsidy_offset += abs(w) * activation

        # 组合最终矩阵 (Time, Zones)
        # 这里简化处理：假设空间分布在一天内是静态的，或者你可以加入时间参数让Hotspot移动
        time_steps = CONFIG.get('TIME_STEPS_PER_DAY', 288)

        # Final Surge = Base + RBF_Offset (限制在 [1.0, 3.0])
        surge_vec = np.clip(base_surge + spatial_surge_offset, 1.0, 3.0)
        lambda_matrix = np.tile(surge_vec[np.newaxis, :], (time_steps, 1))

        # Final Subsidy = Base + RBF_Offset (限制在 [0.0, 10.0])
        subsidy_vec = np.clip(base_sub + spatial_subsidy_offset, 0.0, 10.0)
        subsidy_matrix = np.tile(subsidy_vec[np.newaxis, :], (time_steps, 1))

        return {
            'commission': base_comm,
            'lambda': lambda_matrix,
            'subsidy': subsidy_matrix
        }


# --- 辅助函数：提取坐标 ---
def get_normalized_coords(hex_list):
    coords = []
    for h in hex_list:
        lat, lng = h3.cell_to_latlng(h)
        coords.append([lat, lng])
    coords = np.array(coords)
    # Min-Max Normalization to [0, 1]
    min_c = coords.min(axis=0)
    max_c = coords.max(axis=0)
    norm_coords = (coords - min_c) / (max_c - min_c + 1e-6)
    return norm_coords


# --- 适配后的 Evaluate 函数 ---
def evaluate_strategy_spatial(gene, trainer: Trainer, encoder: SpatialStrategyEncoder):
    platform_params = encoder.decode(gene)
    objectives = trainer.train_and_evaluate(platform_params, num_episodes=5)
    agent_weights = trainer.agent.get_weights()
    return objectives, agent_weights


# --- 原有的 NSGA-II 工具函数 (保持不变) ---
def fast_non_dominated_sort(objectives):
    pop_size = objectives.shape[0]
    S = [[] for _ in range(pop_size)]
    n = np.zeros(pop_size)
    rank = np.zeros(pop_size)
    fronts = [[]]
    for p in range(pop_size):
        for q in range(pop_size):
            # 注意：Wait Time 是负值，这里默认所有目标都是求最大化
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
            f_min, f_max = objectives[sorted_indices[0], m], objectives[sorted_indices[-1], m]
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
    beta = (2 * u) ** (1.0 / (eta + 1)) if u <= 0.5 else (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return np.clip(c1, bounds[:, 0], bounds[:, 1]), np.clip(c2, bounds[:, 0], bounds[:, 1])


def polynomial_mutation(p, bounds, eta=20, prob=0.1):
    if np.random.rand() > prob: return p
    mutant = np.copy(p)
    for i in range(len(p)):
        u = np.random.rand()
        delta = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
        mutant[i] += delta * (bounds[i, 1] - bounds[i, 0])
    return np.clip(mutant, bounds[:, 0], bounds[:, 1])


# --- Modified ERL Solver ---
class ERL_Solver_Spatial:
    def __init__(self, simulator_path, pop_size=10, max_gens=20, use_surrogate=True):
        self.sim_path = simulator_path
        self.pop_size = pop_size
        self.max_gens = max_gens
        self.use_surrogate = use_surrogate

        self.trainer = Trainer(simulator_path=self.sim_path)

        hex_list = list(self.trainer.env.simulator.adjacency.keys())
        zone_coords = get_normalized_coords(hex_list)
        self.encoder = SpatialStrategyEncoder(zone_coords, n_hotspots=5)

        self.history = {'gen': [], 'best_profit': [], 'best_completion': [], 'best_waittime': []}

    def solve(self):
        # 初始化种群
        sampler = LatinHypercube(d=self.encoder.dim)
        # 生成归一化的种群 [0, 1]
        pop_normalized = sampler.random(n=self.pop_size)
        # 映射到真实物理空间 (用于评估)
        pop_genes = pop_normalized * (self.encoder.bounds[:, 1] - self.encoder.bounds[:, 0]) + self.encoder.bounds[:, 0]

        archive_X = []  # 存储真实的物理参数
        archive_Y = []  # 存储目标函数值
        archive_W = {}  # 存储权重
        init_w = self.trainer.agent.get_weights()

        print(f"--- Spatial ERL Start (Dim={self.encoder.dim}) ---")

        # Initial Population Evaluation
        for i in tqdm(range(self.pop_size), desc="Init Pop"):
            self.trainer.agent.load_by_weights(init_w)
            fit, w = evaluate_strategy_spatial(pop_genes[i], self.trainer, self.encoder)
            archive_X.append(pop_genes[i])
            archive_Y.append(fit)
            archive_W[tuple(pop_genes[i])] = w

        for gen in tqdm(range(self.max_gens), desc="Evolution"):
            # 1. Surrogate Assisted Evolution
            if self.use_surrogate:
                # --- 修改点 1: 调整核函数边界 ---
                # length_scale_bounds 下限放宽至 1e-4
                # WhiteKernel noise_level 初始值设高一点 (0.5)，允许更大的噪声
                kernel = C(1.0, (1e-3, 1e3)) * \
                         GP_RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e2)) + \
                         WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-5, 1e1))

                # n_restarts_optimizer 增加以提高拟合成功率
                surrogates = [GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=False) for _
                              in range(3)]

                X_train_raw = np.array(archive_X)
                Y_train_raw = np.array(archive_Y)

                # --- 修改点 2: 输入 X 的标准化 ---
                # 将 X 缩放到 [0, 1] 区间，这对各向同性 RBF 核至关重要
                scaler_x = MinMaxScaler()
                X_train_scaled = scaler_x.fit_transform(X_train_raw)

                # --- 输出 Y 的标准化 (保持原样，很重要) ---
                scalers_y = [StandardScaler() for _ in range(3)]
                Y_train_scaled = np.zeros_like(Y_train_raw)
                for k in range(3):
                    Y_train_scaled[:, k] = scalers_y[k].fit_transform(Y_train_raw[:, k].reshape(-1, 1)).flatten()

                # 限制训练集大小 (Subset Selection)
                if len(X_train_scaled) > 300:
                    indices = np.random.choice(len(X_train_scaled), 300, replace=False)
                    X_sub = X_train_scaled[indices]
                    Y_sub = Y_train_scaled[indices]
                else:
                    X_sub = X_train_scaled
                    Y_sub = Y_train_scaled

                # 训练代理模型
                models_ready = True
                for k in range(3):
                    try:
                        surrogates[k].fit(X_sub, Y_sub[:, k])
                    except Exception as e:
                        print(f"GP Fit failed for obj {k}: {e}")
                        models_ready = False
                        break

                if models_ready:
                    # 在代理模型上进行虚拟进化
                    virtual_pop = np.copy(pop_genes)
                    for _ in range(10):  # 虚拟进化代数
                        # 预测前需将 virtual_pop 归一化
                        v_pop_scaled = scaler_x.transform(virtual_pop)

                        v_fitness_scaled = np.zeros((self.pop_size, 3))
                        for k in range(3):
                            # 预测结果是归一化的 Y，但这不影响排序
                            v_fitness_scaled[:, k] = surrogates[k].predict(v_pop_scaled)

                        # 基于预测值排序
                        fronts, ranks = fast_non_dominated_sort(v_fitness_scaled)
                        dists = calculate_crowding_distance(v_fitness_scaled, fronts)

                        offspring = []
                        while len(offspring) < self.pop_size:
                            p1 = tournament_selection(range(self.pop_size), ranks, dists)
                            p2 = tournament_selection(range(self.pop_size), ranks, dists)
                            # 交叉变异在物理空间进行
                            c1, c2 = sbx_crossover(virtual_pop[p1], virtual_pop[p2], self.encoder.bounds)
                            offspring.extend([polynomial_mutation(c1, self.encoder.bounds),
                                              polynomial_mutation(c2, self.encoder.bounds)])
                        virtual_pop = np.array(offspring[:self.pop_size])

                    # 选择最佳候选 (Exploitation)
                    v_pop_scaled = scaler_x.transform(virtual_pop)
                    v_fitness_scaled = np.zeros((self.pop_size, 3))
                    for k in range(3):
                        v_fitness_scaled[:, k] = surrogates[k].predict(v_pop_scaled)

                    fronts, _ = fast_non_dominated_sort(v_fitness_scaled)
                    candidate_gene = virtual_pop[np.random.choice(fronts[0])]
                else:
                    # Fallback
                    idx = np.random.randint(0, self.pop_size)
                    candidate_gene = polynomial_mutation(pop_genes[idx], self.encoder.bounds, prob=1.0)
            else:
                idx = np.random.randint(0, self.pop_size)
                candidate_gene = polynomial_mutation(pop_genes[idx], self.encoder.bounds, prob=1.0)

            # 2. Transfer Learning Hot-Start
            if len(archive_X) > 0:
                dists = np.linalg.norm(np.array(archive_X) - candidate_gene, axis=1)
                nearest_idx = np.argmin(dists)
                nearest_key = tuple(archive_X[nearest_idx])
                if nearest_key in archive_W:
                    self.trainer.agent.load_by_weights(archive_W[nearest_key])
                else:
                    self.trainer.agent.load_by_weights(init_w)

            # 3. Real Eval (昂贵的仿真评估)
            real_fit, real_w = evaluate_strategy_spatial(candidate_gene, self.trainer, self.encoder)

            archive_X.append(candidate_gene)
            archive_Y.append(real_fit)
            archive_W[tuple(candidate_gene)] = real_w

            # Replacement (更新种群)
            pop_genes[np.random.randint(0, self.pop_size)] = candidate_gene

            # Logging
            best_profit = np.max(np.array(archive_Y)[:, 0])
            self.history['gen'].append(gen)
            self.history['best_profit'].append(best_profit)
            self.history['best_completion'].append(np.max(np.array(archive_Y)[:, 1]))
            self.history['best_waittime'].append(np.max(np.array(archive_Y)[:, 2]))

            # 格式化输出，Wait Time 取反转回正值显示
            print(f"Gen {gen} | Profit: {best_profit:.2f} | Comp: {real_fit[1]:.2f} | Wait: {-real_fit[2]:.2f}")

        return self.history

    def plot_history(self):
        plt.plot(self.history['gen'], self.history['best_profit'], label='Avg Profit', marker='o')
        plt.legend()
        plt.savefig('paper_results/best_profit.png')
        plt.close()
        plt.plot(self.history['gen'], self.history['best_completion'], label='Max Completion', linestyle='--')
        plt.legend()
        plt.savefig('paper_results/best_completion.png')
        plt.close()
        plt.plot(self.history['gen'], self.history['best_waittime'], label='Max Wait', linestyle='--')
        plt.legend()
        plt.savefig('paper_results/best_waittime.png')
        plt.close()


if __name__ == '__main__':
    # 路径需根据实际情况修改
    sim_path = 'model/generators/simulator_hex_scaling=0.004257843312339327_weekday.pkl'
    if os.path.exists(sim_path):
        solver = ERL_Solver_Spatial(simulator_path=sim_path, max_gens=30, pop_size=20)
        solver.solve()
        solver.plot_history()