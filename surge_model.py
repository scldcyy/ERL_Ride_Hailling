import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import warnings
from deap.tools import hypervolume


class SurrogateModel:
    def __init__(self, n_reference_states=50):
        """
        初始化高斯过程代理模型。

        Args:
            n_reference_states: 用于提取公式"表型特征"的参考状态数量。数量越多表征越准，但GP推断会稍微变慢。
        """
        self.n_reference_states = n_reference_states
        self.reference_states = self._generate_reference_states()

        # 为三个目标（Profit, Efficiency/Wait Time, Fairness/Gini）分别建立高斯过程
        # 使用 Matern 内核处理非平滑的适应度地形，加入 WhiteKernel 处理强化学习带来的随机噪声
        kernel = ConstantKernel(1.0) * Matern(nu=1.5) + WhiteKernel(noise_level=0.1)

        self.gp_profit = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
        self.gp_efficiency = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
        self.gp_fairness = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)

        self.is_trained = False

    def _generate_reference_states(self):
        """使用拉丁超立方抽样 (LHS) 生成均匀的参考状态 (t, N_o, N_d, SD)"""
        # 3 dimensions: t, no, nd
        sampler = qmc.LatinHypercube(d=3, seed=42)
        lhs_samples = sampler.random(n=self.n_reference_states)

        # 反归一化到实际的物理范围
        # t in [0, 1]
        t = lhs_samples[:, 0]
        # N_o in [0, 20]
        no = lhs_samples[:, 1] * 20.0
        # N_d in [0, 20]
        nd = lhs_samples[:, 2] * 20.0

        # 供需比 SD (加入 epsilon 防止除零)
        sd = no / (nd + 1e-6)

        return {'t': t, 'no': no, 'nd': nd, 'sd': sd}

    def _get_phenotype(self, platform_params):
        """
        将上层生成的定价公式映射为数值特征向量 (X)。
        """
        surge_func = platform_params['surge']
        subsidy_func = platform_params['subsidy']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 在参考状态上评估公式
            surge_vector = surge_func(
                self.reference_states['t'],
                self.reference_states['no'],
                self.reference_states['nd'],
                self.reference_states['sd']
            )
            subsidy_vector = subsidy_func(
                self.reference_states['t'],
                self.reference_states['no'],
                self.reference_states['nd'],
                self.reference_states['sd']
            )

            # 安全处理可能由非法公式产生的 nan 或 inf
            surge_vector = np.nan_to_num(np.array(surge_vector, dtype=float), nan=1.0, posinf=5.0, neginf=1.0)
            subsidy_vector = np.nan_to_num(np.array(subsidy_vector, dtype=float), nan=0.0, posinf=20.0, neginf=0.0)

            # 处理标量返回 (如果公式是常数)
            if surge_vector.ndim == 0:
                surge_vector = np.full(self.n_reference_states, surge_vector)
            if subsidy_vector.ndim == 0:
                subsidy_vector = np.full(self.n_reference_states, subsidy_vector)

        # 拼接作为该公式的指纹
        return np.concatenate([surge_vector, subsidy_vector])

    def update(self, archive_params, archive_fitness):
        """
        使用真实强化学习环境评估过的数据更新 GP 模型。

        Args:
            archive_params: list of platform_params dicts
            archive_fitness: numpy array of shape (N, 3) -> [Profit, Efficiency, Fairness]
        """
        if len(archive_params) < 5:
            # 样本太少不进行训练
            return

        X = np.array([self._get_phenotype(p) for p in archive_params])
        Y = np.array(archive_fitness)

        # 分别拟合三个目标
        self.gp_profit.fit(X, Y[:, 0])
        self.gp_efficiency.fit(X, Y[:, 1])
        self.gp_fairness.fit(X, Y[:, 2])

        self.is_trained = True

    def predict(self, platform_params):
        """预测均值和标准差"""
        if not self.is_trained:
            raise ValueError(
                "Surrogate model has not been trained yet. Need to evaluate some individuals in the real environment first.")

        x = self._get_phenotype(platform_params).reshape(1, -1)

        mu_p, std_p = self.gp_profit.predict(x, return_std=True)
        mu_e, std_e = self.gp_efficiency.predict(x, return_std=True)
        mu_f, std_f = self.gp_fairness.predict(x, return_std=True)

        mu = np.array([mu_p[0], mu_e[0], mu_f[0]])
        std = np.array([std_p[0], std_e[0], std_f[0]])

        return mu, std

    def calculate_ehvi_score(self, params, hof, ref_point, num_samples=30):
        """
        通过蒙特卡洛采样近似计算期望超体积改善 (EHVI)
        """
        mu, std = self.predict(params)

        if len(hof) == 0 or ref_point is None:
            # 初期没有帕累托前沿时，退化为均值探索
            return np.sum(mu)

        # 提取当前帕累托前沿的适应度 (转化为最小化视角)
        pf_fitnesses = np.array([ind.fitness.values for ind in hof])
        pf_minimized = -pf_fitnesses

        # 计算当前的超体积 HV(P)
        current_hv = hypervolume(pf_minimized, ref=ref_point)

        # 从代理模型预测的高斯分布中进行蒙特卡洛采样
        samples = np.random.normal(loc=mu, scale=std, size=(num_samples, len(mu)))

        hvi_sum = 0.0
        for sample in samples:
            sample_minimized = -sample

            # --- FIX 1: Reference Point Bound Check ---
            # If the sample is worse than the reference point in ANY dimension,
            # it contributes absolutely 0 to the hypervolume. Skip it to prevent crashes.
            if np.any(sample_minimized >= ref_point):
                continue

            # 判断采样点是否被当前的帕累托前沿支配
            is_dominated = False
            for pf_pt in pf_minimized:
                if np.all(pf_pt <= sample_minimized):
                    is_dominated = True
                    break

            if not is_dominated:
                # --- FIX 2: Strict Pareto Front Maintenance ---
                # Remove any points in the current front that are dominated BY the new sample
                # A point is dominated by the sample if all sample dimensions are <= the point's dimensions
                non_dominated_mask = ~np.all(pf_minimized >= sample_minimized, axis=1)
                filtered_pf = pf_minimized[non_dominated_mask]

                # Merge the filtered front with the valid new sample
                new_front = np.vstack((filtered_pf, sample_minimized))

                try:
                    new_hv = hypervolume(new_front, ref=ref_point)
                    hvi_sum += max(0.0, new_hv - current_hv)
                except Exception:
                    # Safe fallback: If DEAP throws an unexpected geometry error,
                    # assume 0 improvement rather than crashing a 72-hour training run.
                    pass

        # 返回积分的近似期望值
        return hvi_sum / num_samples