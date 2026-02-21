import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import warnings


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
        """生成一组固定的参考状态 (t, N_o, N_d, SD)，用于将符号公式转化为固定长度的特征向量"""
        rng = np.random.RandomState(42)  # 固定种子保证每次运行的参考状态一致

        # 归一化时间 t in [0, 1]
        t = rng.uniform(0, 1, self.n_reference_states)
        # 订单数量 N_o
        no = rng.uniform(0, 20, self.n_reference_states)
        # 司机数量 N_d
        nd = rng.uniform(0, 20, self.n_reference_states)
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

    def calculate_ucb(self, platform_params, kappa=2.0):
        """
        计算置信上限 (Upper Confidence Bound)，用于指导进化算法的探索与利用。
        公式: UCB = \mu + \kappa * \sigma
        """
        if not self.is_trained:
            # 如果未训练，返回极大值强制真实评估
            return np.array([float('inf'), float('inf'), float('inf')])

        mu, std = self.predict(platform_params)
        ucb_scores = mu + kappa * std
        return ucb_scores