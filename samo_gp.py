import os
import pickle
import random
import numpy as np
import operator
import math
import copy
from tqdm import tqdm

# 依赖库
from deap import base, creator, tools, gp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

# 引入本地模块
from env_core import CONFIG
from interface import BiLevelEvaluator


# --- 1. 定义 GP 语言原语 (Primitives) ---
# 使得生成的公式具有物理意义
def protectedDiv(left, right):
    # 保护除法，避免除以0
    return left / (right + 1e-4)


def protectedLog(x):
    # 保护对数，避免负数
    return np.log(np.abs(x) + 1e-4)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))


pset = gp.PrimitiveSet("MAIN", 3)
pset.renameArguments(ARG0='Demand')
pset.renameArguments(ARG1='Supply')
pset.renameArguments(ARG2='Time')

# 添加算子 (符号回归常用算子)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(protectedLog, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(max, 2)
pset.addPrimitive(min, 2)
# 添加常数
pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))

# --- 2. 定义多目标个体 ---
# 目标: Max(Profit), Max(ServiceRate), Max(Gini_Fairness)
# 注意：在 deap 中 weights 正数表示最大化，负数表示最小化
# Gini 系数我们希望越小越好 (越公平)，但为了统一方便，我们在 Evaluator 里返回的是 -Gini
# 所以这里统统 maximize
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)


class SurrogateModel:
    """
    代理模型包装器：负责将 GP 树转化为向量，并训练 GPR
    """

    def __init__(self, num_objectives=3, n_reference_points=20):
        self.num_objectives = num_objectives
        self.models = []
        self.scalers_X = StandardScaler()
        self.scalers_y = [StandardScaler() for _ in range(num_objectives)]

        # 初始化 3 个高斯过程 (每个目标一个)
        # Kernel: 常数核 * RBF核 (处理非线性关系) + 白噪声 (处理仿真噪声)
        for _ in range(num_objectives):
            kernel = C(1.0) * RBF(length_scale=1.0)
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, alpha=1e-2)
            self.models.append(gpr)

        # 生成固定的参考状态点，用于提取特征 (Phenotypic Characterization)
        # 特征向量 = 公式在这些点上的输出值
        rng = np.random.RandomState(42)
        # D, S, T
        self.ref_states = rng.rand(n_reference_points, 3)
        # 归一化 D, S 到常见范围 (0-10)
        self.ref_states[:, 0] *= 10.0
        self.ref_states[:, 1] *= 10.0

    def extract_features(self, func):
        """将 GP 函数转化为特征向量"""
        # 向量化计算
        try:
            # 尝试传入向量
            outputs = func(self.ref_states[:, 0], self.ref_states[:, 1], self.ref_states[:, 2])
            # 如果 func 返回标量 (常数函数)，则填充
            if np.isscalar(outputs):
                outputs = np.full(len(self.ref_states), outputs)
        except:
            # Fallback 循环
            outputs = np.array([func(d, s, t) for d, s, t in self.ref_states])

        # 替换 NaN/Inf
        outputs = np.nan_to_num(outputs, nan=0.0, posinf=10.0, neginf=-10.0)
        return outputs

    def train(self, archive):
        """利用历史真实评估数据训练模型"""
        if len(archive) < 10: return  # 数据太少不训练

        # 准备数据
        X = np.array([ind.features for ind in archive])
        Y = np.array([ind.fitness.values for ind in archive])

        # 归一化
        X_scaled = self.scalers_X.fit_transform(X)

        for i in range(self.num_objectives):
            y_col = Y[:, i].reshape(-1, 1)
            y_scaled = self.scalers_y[i].fit_transform(y_col).ravel()
            self.models[i].fit(X_scaled, y_scaled)

    def predict(self, individuals):
        """预测一组个体的 Fitness (Mean & Std)"""
        X = np.array([self.extract_features(gp.compile(ind, pset)) for ind in individuals])
        X_scaled = self.scalers_X.transform(X)

        means = []
        stds = []

        for i in range(self.num_objectives):
            mu, sigma = self.models[i].predict(X_scaled, return_std=True)
            # 反归一化 Mean
            mu_inv = self.scalers_y[i].inverse_transform(mu.reshape(-1, 1)).ravel()
            # Std 也要缩放回来
            sigma_inv = sigma * self.scalers_y[i].scale_

            means.append(mu_inv)
            stds.append(sigma_inv)

        return np.array(means).T, np.array(stds).T  # Shape: (N_ind, N_obj)


class SAMO_GP_Solver:
    """
    代理模型辅助的多目标遗传规划求解器
    """

    def __init__(self, simulator_path, pop_size=50, max_gens=20, eval_budget=500):
        self.evaluator = BiLevelEvaluator(simulator_path)
        self.pop_size = pop_size
        self.max_gens = max_gens
        self.eval_budget = eval_budget
        self.real_eval_count = 0

        # DEAP 工具箱配置
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=pset)

        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=pset)

        # 装饰器：限制树的高度 (防止公式过于复杂不可解释)
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))

        # 代理模型
        self.surrogate = SurrogateModel()
        self.archive = []  # 存储所有真实评估过的个体

    def real_evaluate_batch(self, individuals):
        """执行昂贵的真实评估"""
        for ind in individuals:
            if self.real_eval_count >= self.eval_budget:
                ind.fitness.values = (-1e9, -1e9, -1e9)  # 预算耗尽
                continue

            # 编译公式
            func = self.toolbox.compile(expr=ind)

            # [Core] 调用双层博弈评估器
            # Adaptation Epochs 设置为 3，模拟司机适应
            scores = self.evaluator.evaluate(func, strategy_type='GP', adaptation_epochs=3, test_episodes=1)

            ind.fitness.values = tuple(scores)

            # 存储特征以便代理模型训练
            ind.features = self.surrogate.extract_features(func)

            self.archive.append(ind)
            self.real_eval_count += 1

        return individuals

    def solve(self):
        print(f"Start SAMO-GP Evolution. Pop: {self.pop_size}, Budget: {self.eval_budget}")

        # 1. 初始化种群
        pop = self.toolbox.population(n=self.pop_size)

        # 2. 初始代：全部真实评估 (构建初始数据集)
        print(">>> Gen 0: Initial Real Evaluation...")
        self.real_evaluate_batch(pop)

        # 记录Pareto前沿
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("max", np.max, axis=0)
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "avg", "max"

        record = stats.compile(pop)
        logbook.record(gen=0, evals=self.real_eval_count, **record)
        print(logbook.stream)

        # 3. 进化循环
        for gen in range(1, self.max_gens + 1):
            if self.real_eval_count >= self.eval_budget:
                print("Evaluation budget exhausted.")
                break

            # A. 训练代理模型
            self.surrogate.train(self.archive)

            # B. 生成子代 (2倍种群大小，产生更多候选解)
            offspring = algorithms_varOr(pop, self.toolbox, lambda_=self.pop_size * 2, cxpb=0.7, mutpb=0.3)

            # C. 代理模型预筛选 (Infill Criterion)
            # 对所有子代进行预测
            pred_means, pred_stds = self.surrogate.predict(offspring)

            # 计算 UCB 得分 (Upper Confidence Bound)
            # 我们希望 Mean 大 (目标最大化) 且 Std 大 (探索未知)
            # 简单的加权和策略进行预排序
            # Normalized UCB Sum
            ucb_scores = []
            for i in range(len(offspring)):
                # 简单标量化：0.4*Profit + 0.3*Service + 0.3*Fairness
                # 注意：这里需要数据大概在一个量级，或者动态归一化。
                # 简化起见，直接用原始值（Profit较大，权重放小点）
                mean_score = 0.001 * pred_means[i][0] + 1.0 * pred_means[i][1] + 10.0 * pred_means[i][2]
                std_score = np.sum(pred_stds[i])
                ucb = mean_score + 0.5 * std_score  # alpha = 0.5
                ucb_scores.append(ucb)

                # 同时也先把预测均值赋给 fitness，用于下面的 NSGA2
                # 注意：这里只是为了 selection 能够运行，之后会用真实值覆盖
                offspring[i].fitness.values = tuple(pred_means[i])

            # D. 选择最有潜力的个体进行真实评估
            # 策略：选 UCB 得分最高的 Top 5 + 选 预测 Fitness 处于 Pareto Front 的 Top 5
            # 简化版：直接选 UCB 最高的 N 个
            n_infill = min(10, self.eval_budget - self.real_eval_count)  # 每代只跑 10 个真实仿真

            # 根据 UCB 排序
            sorted_indices = np.argsort(ucb_scores)[::-1]
            candidates = [offspring[i] for i in sorted_indices[:n_infill]]

            # E. 真实评估候选者
            print(f">>> Gen {gen}: Surrogate Infill ({n_infill} real evals)...")
            self.real_evaluate_batch(candidates)

            # F. 环境选择 (合并父代 + 所有子代)
            # 对于没有真实评估的子代，我们直接用预测值参与排序 (Semi-trust)
            # 或者，标准的 SAEA 通常只保留真实评估过的个体进入下一代 Archive，但 GP 为了维持多样性，
            # 我们保留所有 offspring，但在 NSGA2 中混合使用 (Real Fitness) 和 (Predicted Fitness)
            # 风险：预测误差可能误导搜索。
            # 稳健做法：只从 Archive (真实数据) + Candidates (刚评估的) 中选择下一代
            # 但这样种群会缩小。
            # 折中方案：下一代 = NSGA2(Pop + Candidates)
            # 这样保证下一代全是真实评估过的，质量有保证。

            combined = pop + candidates
            pop = self.toolbox.select(combined, k=self.pop_size)

            # 记录日志
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=self.real_eval_count, **record)
            print(logbook.stream)

            # 保存检查点
            if gen % 5 == 0:
                self.save_results(pop, logbook, f"checkpoint_gen_{gen}")

        # 4. 结束
        return pop, logbook

    def save_results(self, population, logbook, filename_prefix="samo_gp_final"):
        os.makedirs("experiment_results", exist_ok=True)

        # 提取 Pareto 前沿
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

        data = {
            'pareto_front': pareto_front,  # 包含个体公式和fitness
            'logbook': logbook,
            'archive': self.archive  # 所有历史数据
        }

        path = os.path.join("experiment_results", f"{filename_prefix}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Results saved to {path}")


# 辅助函数：变异/交叉生成子代
def algorithms_varOr(population, toolbox, lambda_, cxpb, mutpb):
    assert (cxpb + mutpb) <= 1.0, "The sum of the crossover and mutation probabilities must be smaller or equal to 1.0."
    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:  # Crossover
            ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Mutation
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:  # Replication
            offspring.append(toolbox.clone(random.choice(population)))
    return offspring


# --- 主入口 ---
if __name__ == "__main__":
    # 使用假数据或真实数据路径
    # 注意：这里需要你先生成 simulator_hex_....pkl 文件，或者用 interface 里生成的 dummy
    SIM_PATH = "dummy_simulator12.pkl"
    if not os.path.exists(SIM_PATH):
        # 尝试找真实路径
        real_path = 'model/generators/simulator_hex_scaling=0.004257843312339327_weekday.pkl'
        if os.path.exists(real_path):
            SIM_PATH = real_path
        else:
            # 生成 Dummy
            from interface import BiLevelEvaluator

            # Trigger dummy generation in interface
            os.system("python interface.py")

    solver = SAMO_GP_Solver(SIM_PATH, pop_size=20, max_gens=5, eval_budget=50)
    final_pop, logs = solver.solve()

    # 打印最优解
    pareto_front = tools.sortNondominated(final_pop, len(final_pop), first_front_only=True)[0]
    print("\n>>> Final Pareto Front Formulas:")
    for ind in pareto_front:
        print(f"Formula: {ind}")
        print(
            f"Fitness: Profit={ind.fitness.values[0]:.2f}, Service={ind.fitness.values[1]:.2f}, Gini={ind.fitness.values[2]:.4f}")
        print("-" * 30)