from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

EPS = 1e-9
TERMINALS = ('t', 'no', 'nd', 'sd')


def arr(x):
    return np.asarray(x, dtype=float)


def safe_div(a, b):
    b = arr(b)
    b = np.where(np.abs(b) < 1e-6, 1e-6, b)
    return np.nan_to_num(arr(a) / b, nan=0.0, posinf=1e6, neginf=-1e6)


PRIMITIVES = {
    'add': (2, lambda a, b: np.nan_to_num(arr(a) + arr(b), nan=0.0)),
    'sub': (2, lambda a, b: np.nan_to_num(arr(a) - arr(b), nan=0.0)),
    'mul': (2, lambda a, b: np.nan_to_num(arr(a) * arr(b), nan=0.0, posinf=1e6, neginf=-1e6)),
    'div': (2, safe_div),
    'sin': (1, lambda a: np.sin(arr(a))),
    'cos': (1, lambda a: np.cos(arr(a))),
    'tanh': (1, lambda a: np.tanh(arr(a))),
    'log': (1, lambda a: np.log(np.abs(arr(a)) + 1e-6)),
    'sqrt': (1, lambda a: np.sqrt(np.abs(arr(a)))),
    'min': (2, lambda a, b: np.minimum(arr(a), arr(b))),
    'max': (2, lambda a, b: np.maximum(arr(a), arr(b))),
}


@dataclass
class Node:
    kind: str
    value: Any
    children: List['Node'] = field(default_factory=list)

    def clone(self):
        return Node(self.kind, copy.deepcopy(self.value), [c.clone() for c in self.children])

    def eval(self, ctx):
        if self.kind == 'const':
            ref = next((arr(ctx[k]) for k in ('no', 'nd', 'sd') if k in ctx), None)
            return np.asarray(float(self.value)) if ref is None else np.full_like(ref, float(self.value), dtype=float)
        if self.kind == 'var':
            return arr(ctx[self.value])
        arity, fn = PRIMITIVES[self.value]
        return fn(*[c.eval(ctx) for c in self.children[:arity]])

    def to_str(self):
        if self.kind == 'const':
            return f'{float(self.value):.4f}'
        if self.kind == 'var':
            return str(self.value)
        if len(self.children) == 1:
            return f'{self.value}({self.children[0].to_str()})'
        return f'{self.value}({self.children[0].to_str()}, {self.children[1].to_str()})'

    def paths(self, prefix=()):
        out = [(prefix, self)]
        for i, c in enumerate(self.children):
            out.extend(c.paths(prefix + (i,)))
        return out

    def subtree(self, path):
        node = self
        for idx in path:
            node = node.children[idx]
        return node

    def replace(self, path, new_subtree):
        if not path:
            return new_subtree.clone()
        root = self.clone()
        node = root
        for idx in path[:-1]:
            node = node.children[idx]
        node.children[path[-1]] = new_subtree.clone()
        return root


class TreeFactory:
    def __init__(self, max_depth=4, const_range=(-3.0, 3.0)):
        self.max_depth = max_depth
        self.const_range = const_range
        self.functions = list(PRIMITIVES)

    def terminal(self):
        return Node('const', random.uniform(*self.const_range)) if random.random() < 0.3 else Node('var', random.choice(TERMINALS))

    def random_tree(self, depth=0, force_function=False):
        if depth >= self.max_depth or (depth > 0 and not force_function and random.random() < 0.45):
            return self.terminal()
        name = random.choice(self.functions)
        arity = PRIMITIVES[name][0]
        return Node('func', name, [self.random_tree(depth + 1) for _ in range(arity)])

    def mutate(self, tree):
        path = random.choice([p for p, _ in tree.paths()])
        return tree.replace(path, self.random_tree())

    def crossover(self, a, b):
        pa = random.choice([p for p, _ in a.paths()])
        pb = random.choice([p for p, _ in b.paths()])
        return a.replace(pa, b.subtree(pb)), b.replace(pb, a.subtree(pa))


@dataclass
class Individual:
    surge_tree: Node
    subsidy_tree: Node
    fitness: Optional[np.ndarray] = None
    predicted_fitness: Optional[np.ndarray] = None
    ehvi: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone(self):
        return Individual(
            self.surge_tree.clone(),
            self.subsidy_tree.clone(),
            None if self.fitness is None else self.fitness.copy(),
            None if self.predicted_fitness is None else self.predicted_fitness.copy(),
            self.ehvi,
            copy.deepcopy(self.metadata),
        )

    def platform_params(self):
        return {
            'surge': lambda t, no, nd, sd: self.surge_tree.eval({'t': t, 'no': no, 'nd': nd, 'sd': sd}),
            'subsidy': lambda t, no, nd, sd: self.subsidy_tree.eval({'t': t, 'no': no, 'nd': nd, 'sd': sd}),
        }

    def formulas(self):
        return {'surge': self.surge_tree.to_str(), 'subsidy': self.subsidy_tree.to_str()}


def dominates(a, b):
    return np.all(a >= b - EPS) and np.any(a > b + EPS)


def fitness_of(ind):
    return ind.fitness if ind.fitness is not None else ind.predicted_fitness


def nondominated(points):
    keep = []
    for i, p in enumerate(points):
        if not any(i != j and dominates(points[j], p) for j in range(len(points))):
            keep.append(p)
    return np.asarray(keep, dtype=float)


def fast_sort(pop):
    fits = [np.asarray(fitness_of(ind), dtype=float) for ind in pop]
    S = [[] for _ in pop]
    n = [0] * len(pop)
    fronts = [[]]
    for p in range(len(pop)):
        for q in range(len(pop)):
            if p == q:
                continue
            if dominates(fits[p], fits[q]):
                S[p].append(q)
            elif dominates(fits[q], fits[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while i < len(fronts) and fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    nxt.append(q)
        if nxt:
            fronts.append(nxt)
        i += 1
    return fronts


def crowding(front, pop):
    if not front:
        return {}
    m = len(fitness_of(pop[front[0]]))
    dist = {idx: 0.0 for idx in front}
    if len(front) <= 2:
        return {idx: float('inf') for idx in front}
    for obj in range(m):
        order = sorted(front, key=lambda idx: float(fitness_of(pop[idx])[obj]))
        lo, hi = float(fitness_of(pop[order[0]])[obj]), float(fitness_of(pop[order[-1]])[obj])
        dist[order[0]] = dist[order[-1]] = float('inf')
        if abs(hi - lo) < EPS:
            continue
        for i in range(1, len(order) - 1):
            prev_v = float(fitness_of(pop[order[i - 1]])[obj])
            next_v = float(fitness_of(pop[order[i + 1]])[obj])
            dist[order[i]] += (next_v - prev_v) / (hi - lo + EPS)
    return dist


def nsga2_select(pop, size):
    out = []
    for front in fast_sort(pop):
        if len(out) + len(front) <= size:
            out.extend(pop[idx].clone() for idx in front)
        else:
            dist = crowding(front, pop)
            ranked = sorted(front, key=lambda idx: dist[idx], reverse=True)
            out.extend(pop[idx].clone() for idx in ranked[:size - len(out)])
            break
    return out


class PhenotypeExtractor:
    def __init__(self, ref_state_size=50, seed=42):
        rng = np.random.default_rng(seed)
        self.ref = []
        for _ in range(ref_state_size):
            t = float(rng.uniform(0.0, 1.0))
            no = float(rng.uniform(0.0, 40.0))
            nd = float(rng.uniform(1.0, 40.0))
            self.ref.append((t, no, nd, no / nd))

    def extract(self, ind):
        xs = []
        for t, no, nd, sd in self.ref:
            ctx = {'t': t, 'no': np.array([no]), 'nd': np.array([nd]), 'sd': np.array([sd])}
            xs.append(float(ind.surge_tree.eval(ctx)[0]))
            xs.append(float(ind.subsidy_tree.eval(ctx)[0]))
        return np.asarray(xs, dtype=float)


class MultiOutputGPSurrogate:
    def __init__(self, seed=42):
        self.seed = seed
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.models = []
        self.fitted = False

    def fit(self, X, Y):
        Xs = self.x_scaler.fit_transform(np.asarray(X, dtype=float))
        Ys = self.y_scaler.fit_transform(np.asarray(Y, dtype=float))
        self.models = []
        for i in range(Ys.shape[1]):
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-4)
            model = GaussianProcessRegressor(kernel=kernel, normalize_y=False, alpha=1e-6, n_restarts_optimizer=1, random_state=self.seed + i)
            model.fit(Xs, Ys[:, i])
            self.models.append(model)
        self.fitted = True

    def predict(self, X):
        Xs = self.x_scaler.transform(np.asarray(X, dtype=float))
        mus, stds = [], []
        for model in self.models:
            mu, std = model.predict(Xs, return_std=True)
            mus.append(mu)
            stds.append(std)
        mu = np.column_stack(mus)
        std = np.column_stack(stds) * np.asarray(self.y_scaler.scale_, dtype=float)
        return self.y_scaler.inverse_transform(mu), std


class EHVISelector:
    def __init__(self, seed=42, posterior_samples=32, hv_samples=1500):
        self.rng = np.random.default_rng(seed)
        self.posterior_samples = posterior_samples
        self.hv_samples = hv_samples

    def hypervolume(self, points, ref):
        points = nondominated(np.asarray(points, dtype=float))
        if points.size == 0:
            return 0.0
        upper = np.max(points, axis=0)
        span = upper - ref
        if np.any(span <= 0):
            return 0.0
        samples = self.rng.uniform(ref, upper, size=(self.hv_samples, points.shape[1]))
        dom = np.zeros(self.hv_samples, dtype=bool)
        for p in points:
            dom |= np.all(samples <= p + EPS, axis=1)
        return float(dom.mean() * np.prod(span))

    def score(self, means, stds, pf_points):
        means, stds = np.asarray(means, float), np.asarray(stds, float)
        if pf_points.size == 0:
            return np.linalg.norm(means, axis=1) + 0.1 * np.linalg.norm(stds, axis=1)
        ref = np.min(np.vstack([pf_points, means - 2 * stds]), axis=0) - 1e-3
        base_hv = self.hypervolume(pf_points, ref)
        scores = np.zeros(len(means), dtype=float)
        for i, (mu, sigma) in enumerate(zip(means, stds)):
            draws = self.rng.normal(mu, np.maximum(sigma, 1e-6), size=(self.posterior_samples, len(mu)))
            gains = []
            for y in draws:
                hv = self.hypervolume(np.vstack([pf_points, y]), ref)
                gains.append(max(0.0, hv - base_hv))
            scores[i] = float(np.mean(gains))
        return scores


class TransferMemory:
    def __init__(self, max_size=128):
        self.max_size = max_size
        self.records = []

    def add(self, phenotype, snapshot, fitness, formulas):
        self.records.append({
            'phenotype': np.asarray(phenotype, dtype=float).copy(),
            'snapshot': copy.deepcopy(snapshot),
            'fitness': np.asarray(fitness, dtype=float).copy(),
            'formulas': copy.deepcopy(formulas),
        })
        self.records = self.records[-self.max_size:]

    def nearest(self, phenotype):
        if not self.records:
            return None
        phenotype = np.asarray(phenotype, dtype=float)
        return min(self.records, key=lambda rec: float(np.linalg.norm(rec['phenotype'] - phenotype)))


class LowerLevelEvaluator:
    def __init__(self, trainer, extractor, eval_episodes=5, use_transfer=True, memory=None):
        self.trainer = trainer
        self.extractor = extractor
        self.eval_episodes = eval_episodes
        self.use_transfer = use_transfer
        self.memory = memory or TransferMemory()

    def evaluate(self, ind):
        phenotype = ind.metadata.get('phenotype')
        if phenotype is None:
            phenotype = self.extractor.extract(ind)
            ind.metadata['phenotype'] = phenotype
        record = self.memory.nearest(phenotype) if self.use_transfer else None
        self.trainer.prepare_for_new_policy(None if record is None else record['snapshot'])
        ind.fitness = np.asarray(self.trainer.train_and_evaluate(ind.platform_params(), num_episodes=self.eval_episodes, show_fig=False, save_model=False), dtype=float)
        self.memory.add(phenotype, self.trainer.get_policy_snapshot(), ind.fitness, ind.formulas())
        return ind.fitness


@dataclass
class TLConfig:
    population_size: int = 100
    generations: int = 100
    real_eval_quota: int = 5
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    max_tree_depth: int = 4
    tournament_size: int = 2
    ref_state_size: int = 50
    seed: int = 42
    use_surrogate: bool = True
    use_transfer: bool = True


class TLSAMOGP:
    def __init__(self, evaluator, config=None):
        self.cfg = config or TLConfig()
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.evaluator = evaluator
        self.factory = TreeFactory(max_depth=self.cfg.max_tree_depth)
        self.surrogate = MultiOutputGPSurrogate(seed=self.cfg.seed)
        self.selector = EHVISelector(seed=self.cfg.seed)
        self.archive = []
        self.history = []

    def random_individual(self):
        return Individual(self.factory.random_tree(force_function=True), self.factory.random_tree(force_function=True))

    def archive_arrays(self):
        X, Y = [], []
        for ind in self.archive:
            if ind.fitness is not None:
                X.append(ind.metadata['phenotype'])
                Y.append(ind.fitness)
        return np.asarray(X, dtype=float), np.asarray(Y, dtype=float)

    def fit_surrogate(self):
        if not self.cfg.use_surrogate:
            return False
        X, Y = self.archive_arrays()
        if len(X) < max(6, Y.shape[1] + 2 if Y.ndim == 2 else 6):
            return False
        self.surrogate.fit(X, Y)
        return True

    def tournament(self, pop, rank_map, crowd_map):
        cand = random.sample(list(enumerate(pop)), k=min(self.cfg.tournament_size, len(pop)))
        best = min(cand, key=lambda x: (rank_map.get(x[0], 10**9), -crowd_map.get(x[0], 0.0)))
        return best[1].clone()

    def offspring(self, pop):
        fronts = fast_sort(pop)
        rank_map, crowd_map = {}, {}
        for rank, front in enumerate(fronts):
            for idx in front:
                rank_map[idx] = rank
            crowd_map.update(crowding(front, pop))
        out = []
        while len(out) < len(pop):
            p1, p2 = self.tournament(pop, rank_map, crowd_map), self.tournament(pop, rank_map, crowd_map)
            c1, c2 = p1.clone(), p2.clone()
            if random.random() < self.cfg.crossover_rate:
                if random.random() < 0.5:
                    c1.surge_tree, c2.surge_tree = self.factory.crossover(c1.surge_tree, c2.surge_tree)
                if random.random() < 0.5:
                    c1.subsidy_tree, c2.subsidy_tree = self.factory.crossover(c1.subsidy_tree, c2.subsidy_tree)
            if random.random() < self.cfg.mutation_rate:
                if random.random() < 0.5:
                    c1.surge_tree = self.factory.mutate(c1.surge_tree)
                    c2.surge_tree = self.factory.mutate(c2.surge_tree)
                if random.random() < 0.5:
                    c1.subsidy_tree = self.factory.mutate(c1.subsidy_tree)
                    c2.subsidy_tree = self.factory.mutate(c2.subsidy_tree)
            c1.fitness = c2.fitness = None
            c1.predicted_fitness = c2.predicted_fitness = None
            out.extend([c1, c2])
        return out[:len(pop)]

    def pareto_front(self, pool):
        out = []
        for ind in pool:
            fit = fitness_of(ind)
            if fit is None:
                continue
            if not any(other is not ind and fitness_of(other) is not None and dominates(fitness_of(other), fit) for other in pool):
                out.append(ind)
        return out

    def surrogate_prescreen(self, offspring):
        if not (self.cfg.use_surrogate and self.surrogate.fitted):
            return offspring, offspring
        X = []
        for ind in offspring:
            ind.metadata['phenotype'] = self.evaluator.extractor.extract(ind)
            X.append(ind.metadata['phenotype'])
        means, stds = self.surrogate.predict(np.asarray(X, dtype=float))
        pf = self.pareto_front(self.archive)
        pf_points = np.asarray([ind.fitness for ind in pf if ind.fitness is not None], dtype=float)
        scores = self.selector.score(means, stds, pf_points)
        for ind, mu, s in zip(offspring, means, scores):
            ind.predicted_fitness = np.asarray(mu, dtype=float)
            ind.ehvi = float(s)
        ranked = sorted(offspring, key=lambda ind: ind.ehvi, reverse=True)
        real = ranked[:min(self.cfg.real_eval_quota, len(ranked))]
        for ind in ranked[len(real):]:
            ind.fitness = ind.predicted_fitness.copy()
        return real, offspring

    def initialize(self):
        pop = [self.random_individual() for _ in range(self.cfg.population_size)]
        for ind in pop:
            ind.metadata['phenotype'] = self.evaluator.extractor.extract(ind)
            self.evaluator.evaluate(ind)
            self.archive.append(ind.clone())
        self.fit_surrogate()
        return pop

    def run(self):
        pop = self.initialize()
        for gen in range(1, self.cfg.generations + 1):
            children = self.offspring(pop)
            real_eval, children = self.surrogate_prescreen(children)
            for ind in real_eval:
                self.evaluator.evaluate(ind)
                self.archive.append(ind.clone())
            self.fit_surrogate()
            pop = nsga2_select(pop + children, self.cfg.population_size)
            pf = self.pareto_front(self.archive)
            self.history.append({'generation': gen, 'archive_size': len(self.archive), 'pareto_size': len(pf)})
        return self.pareto_front(self.archive)


def build_shared_ppo_evaluator(simulator_path, eval_episodes=5, use_transfer=True, ref_state_size=50, seed=42):
    from shared_ppo import Trainer
    if not os.path.exists(simulator_path):
        raise FileNotFoundError(simulator_path)
    extractor = PhenotypeExtractor(ref_state_size=ref_state_size, seed=seed)
    trainer = Trainer(simulator_path)
    return LowerLevelEvaluator(trainer, extractor, eval_episodes=eval_episodes, use_transfer=use_transfer)


__all__ = [
    'TLConfig', 'TLSAMOGP', 'build_shared_ppo_evaluator', 'PhenotypeExtractor',
    'LowerLevelEvaluator', 'TransferMemory', 'Individual', 'Node',
]
