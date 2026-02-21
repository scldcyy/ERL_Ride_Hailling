import numpy as np
from tqdm import tqdm
from scipy.stats.qmc import LatinHypercube
from shared_ppo import CONFIG, Trainer
import h3


def get_normalized_coords(hex_list):
    coords = []
    for h in hex_list:
        try:
            lat, lng = h3.cell_to_latlng(h)
        except:
            lat, lng = h3.h3_to_geo(h)
        coords.append([lat, lng])
    coords = np.array(coords)
    min_c = coords.min(axis=0)
    max_c = coords.max(axis=0)
    norm_coords = (coords - min_c) / (max_c - min_c + 1e-6)
    return norm_coords


def polynomial_mutation(p, bounds, eta=20, prob=0.1):
    if np.random.rand() > prob: return p
    mutant = np.copy(p)
    for i in range(len(p)):
        u = np.random.rand()
        delta = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
        mutant[i] += delta * (bounds[i, 1] - bounds[i, 0])
    return np.clip(mutant, bounds[:, 0], bounds[:, 1])


class SpatiotemporalStrategyEncoder:
    def __init__(self, zone_coords, n_hotspots=3):
        self.zone_coords = zone_coords
        self.n_zones = len(zone_coords)
        self.n_hotspots = n_hotspots
        # 非均匀时间分段 (5段)
        self.checkpoints = [0, 60, 120, 192, 240, 288]
        self.time_segments = len(self.checkpoints) - 1

        self.base_dim = 3
        self.hotspot_dim = 4
        self.segment_dim = n_hotspots * self.hotspot_dim
        self.dim = self.base_dim + self.time_segments * self.segment_dim

        self.bounds = []
        self.bounds.extend([(0.1, 0.35), (1.0, 1.5), (0.0, 2.0)])
        for _ in range(self.time_segments * n_hotspots):
            self.bounds.append((-1.0, 2.0))
            self.bounds.append((0.0, 1.0))
            self.bounds.append((0.0, 1.0))
            self.bounds.append((0.05, 0.3))
        self.bounds = np.array(self.bounds)

    def decode(self, gene):
        base_comm, base_surge, base_sub = gene[0], gene[1], gene[2]
        lambda_matrix = np.zeros((CONFIG['TIME_STEPS_PER_DAY'], self.n_zones))
        subsidy_matrix = np.zeros((CONFIG['TIME_STEPS_PER_DAY'], self.n_zones))

        for t in range(self.time_segments):
            start_idx = self.base_dim + t * self.segment_dim
            segment_gene = gene[start_idx: start_idx + self.segment_dim]
            spatial_surge = np.zeros(self.n_zones)
            spatial_sub = np.zeros(self.n_zones)

            for k in range(self.n_hotspots):
                w, cx, cy, sigma = segment_gene[k * 4: (k + 1) * 4]
                d2 = (self.zone_coords[:, 0] - cx) ** 2 + (self.zone_coords[:, 1] - cy) ** 2
                act = np.exp(-d2 / (2 * sigma ** 2))
                if w >= 0:
                    spatial_surge += w * act
                else:
                    spatial_sub += abs(w) * act

            t_start = self.checkpoints[t]
            t_end = self.checkpoints[t + 1]
            lambda_matrix[t_start:t_end, :] = np.clip(base_surge + spatial_surge, 1.0, 4.0)
            subsidy_matrix[t_start:t_end, :] = np.clip(base_sub + spatial_sub, 0.0, 10.0)

        return {'commission': base_comm, 'lambda': lambda_matrix, 'subsidy': subsidy_matrix}


class RBF_Solver:
    def __init__(self, simulator_path, scenarios=None, pop_size=10, max_gens=10):
        self.trainer = Trainer(simulator_path, fixed_scenarios=scenarios)
        hex_list = list(self.trainer.env.simulator.adjacency.keys())
        self.encoder = SpatiotemporalStrategyEncoder(get_normalized_coords(hex_list))
        self.pop_size = pop_size
        self.max_gens = max_gens
        self.history = {'score': []}

    def solve(self):
        sampler = LatinHypercube(d=self.encoder.dim)
        pop = sampler.random(n=self.pop_size) * (
                    self.encoder.bounds[:, 1] - self.encoder.bounds[:, 0]) + self.encoder.bounds[:, 0]

        objs = []
        for ind in tqdm(pop, desc="RBF Init"):
            params = self.encoder.decode(ind)
            res = self.trainer.train_and_evaluate(params, num_episodes=5)
            objs.append(res)
        objs = np.array(objs)

        for gen in range(self.max_gens):
            # 简单加权单目标: Profit + Comp*1000 - Wait*0.1
            fitness = objs[:, 0] * 1.0 + objs[:, 1] * 1000.0 - objs[:, 2] * 0.1
            best_idx = np.argmax(fitness)
            current_best = fitness[best_idx]
            self.history['score'].append(current_best)
            print(f"RBF Gen {gen}: Best Score {current_best:.2f}")

            # 标准进化操作 (无GPR)
            offspring = []
            for _ in range(self.pop_size):
                idx = np.random.randint(0, self.pop_size)
                mut = polynomial_mutation(pop[idx], self.encoder.bounds)
                offspring.append(mut)
            offspring = np.array(offspring)

            off_objs = []
            for ind in offspring:
                params = self.encoder.decode(ind)
                res = self.trainer.train_and_evaluate(params, num_episodes=5)
                off_objs.append(res)
            off_objs = np.array(off_objs)

            combined_pop = np.vstack((pop, offspring))
            combined_objs = np.vstack((objs, off_objs))
            combined_fit = combined_objs[:, 0] * 1.0 + combined_objs[:, 1] * 1000.0 - combined_objs[:, 2] * 0.1

            sorted_indices = np.argsort(combined_fit)[::-1]
            pop = combined_pop[sorted_indices[:self.pop_size]]
            objs = combined_objs[sorted_indices[:self.pop_size]]
        best_gene = pop[0]
        model_artifacts = {
            'best_gene': best_gene,
            'driver_agent': self.trainer.agent.get_weights()
        }
        return self.history, model_artifacts