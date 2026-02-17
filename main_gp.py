import operator
import math
import random
import numpy as np
from deap import base, creator, tools, gp
from tqdm import tqdm
from shared_ppo import CONFIG, Trainer


class DynamicTrainer(Trainer):
    def run_dynamic_episode(self, strategy_func):
        state = self.env.reset()
        ep_score = 0

        dynamic_params = {
            'commission': 0.25,
            'lambda': np.ones((CONFIG['TIME_STEPS_PER_DAY'], 277)),
            'subsidy': np.zeros((CONFIG['TIME_STEPS_PER_DAY'], 277))
        }

        while True:
            t = self.env.time
            order_counts = np.zeros(self.env.n_zones)
            for o in self.env.pending_orders:
                if not o['matched']: order_counts[o['origin_idx']] += 1

            idle_mask = (self.env.driver_status == 0)
            driver_counts = np.bincount(self.env.driver_locations[idle_mask], minlength=self.env.n_zones)

            if t < CONFIG['TIME_STEPS_PER_DAY']:
                try:
                    obs_orders = order_counts / (order_counts.max() + 1e-6)
                    obs_drivers = driver_counts / (driver_counts.max() + 1e-6)
                    obs_time = t / CONFIG['TIME_STEPS_PER_DAY']

                    # 向量化计算
                    surges = np.array([strategy_func(o, d, obs_time) for o, d in zip(obs_orders, obs_drivers)])
                    surges = np.clip(surges, 1.0, 5.0)
                    dynamic_params['lambda'][t] = surges
                except:
                    dynamic_params['lambda'][t] = np.ones(277)

            actions = self.agent.select_actions(state)
            next_state, rewards, done, info = self.env.step(actions, dynamic_params)

            self.agent.buffer.rewards.append(rewards)
            # 加权得分
            step_score = info['step_profit'] * 1.0
            ep_score += step_score

            state = next_state
            if done: break

        self.agent.update()
        return ep_score


def protectedDiv(left, right):
    return left / (right + 1e-3)


pset = gp.PrimitiveSet("MAIN", 3)
pset.renameArguments(ARG0='Orders')
pset.renameArguments(ARG1='Drivers')
pset.renameArguments(ARG2='Time')
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(max, 2)
pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


class GP_Solver:
    def __init__(self, simulator_path, scenarios=None, pop_size=20, max_gens=10):
        self.trainer = DynamicTrainer(simulator_path, fixed_scenarios=scenarios)
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=pset)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr, pset=pset)

        self.pop_size = pop_size
        self.max_gens = max_gens
        self.history = {'score': []}

    def evaluate(self, individual):
        func = self.toolbox.compile(expr=individual)
        scores = []
        for _ in range(5):
            s = self.trainer.run_dynamic_episode(func)
            scores.append(s)
        return np.mean(scores),

    def solve(self):
        pop = self.toolbox.population(n=self.pop_size)

        fitnesses = list(map(self.evaluate, pop))
        for ind, fit in zip(pop, fitnesses): ind.fitness.values = fit

        for gen in range(self.max_gens):
            # 标准遗传操作 (无GPR)
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for ind in offspring:
                if random.random() < 0.2:
                    self.toolbox.mutate(ind)
                    del ind.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses): ind.fitness.values = fit

            pop[:] = offspring
            best_score = max([ind.fitness.values[0] for ind in pop])
            self.history['score'].append(best_score)
            print(f"GP Gen {gen}: Best Score {best_score:.2f}")

        # --- 新增：返回模型 ---
        best_ind = pop[0]
        model_artifacts = {
            'best_expr': best_ind,  # DEAP 的 individual 对象
            'driver_agent': self.trainer.agent.get_weights()
        }
        return self.history, model_artifacts

