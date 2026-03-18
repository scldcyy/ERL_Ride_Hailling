from __future__ import annotations

import argparse
import copy
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from generate_simulator import PassengerSimulator
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

EPS = 1e-9
TERMINALS = ("t", "no", "nd", "sd")


# =========================
# Numeric helpers
# =========================


def _arr(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _safe_clip(x: Any, lo: float = -1e6, hi: float = 1e6) -> np.ndarray:
    return np.nan_to_num(np.clip(_arr(x), lo, hi), nan=0.0, posinf=hi, neginf=lo)


def _safe_div(a: Any, b: Any) -> np.ndarray:
    bb = _arr(b)
    bb = np.where(np.abs(bb) < 1e-6, 1e-6, bb)
    return _safe_clip(_arr(a) / bb)


def _as_vector(x: Any, ref: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return np.full_like(np.asarray(ref, dtype=float), float(arr), dtype=float)
    return np.asarray(arr, dtype=float)


def _phenotype_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0:
        return float("inf")
    return float(np.linalg.norm(a - b) / np.sqrt(a.size))


def _normalize_points(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    span = np.where(np.abs(hi - lo) < 1e-9, 1.0, hi - lo)
    return np.clip((points - lo) / span, 0.0, 1.0)


def _normalized_hv(points: np.ndarray, bounds_source: np.ndarray, num_samples: int, rng: np.random.Generator) -> float:
    pts = np.asarray(points, dtype=float)
    src = np.asarray(bounds_source, dtype=float)
    if pts.size == 0 or src.size == 0:
        return 0.0
    lo = np.min(src, axis=0)
    hi = np.max(src, axis=0)
    norm_pts = _normalize_points(pts, lo, hi)
    ref = np.zeros(norm_pts.shape[1], dtype=float)
    return float(hypervolume_mc(norm_pts, ref, num_samples=num_samples, rng=rng))


PRIMITIVES: Dict[str, Tuple[int, Callable[..., np.ndarray]]] = {
    "add": (2, lambda a, b: _safe_clip(_arr(a) + _arr(b))),
    "sub": (2, lambda a, b: _safe_clip(_arr(a) - _arr(b))),
    "mul": (2, lambda a, b: _safe_clip(_arr(a) * _arr(b))),
    "div": (2, _safe_div),
    "sin": (1, lambda a: _safe_clip(np.sin(_arr(a)))),
    "cos": (1, lambda a: _safe_clip(np.cos(_arr(a)))),
    "tanh": (1, lambda a: _safe_clip(np.tanh(_arr(a)))),
    "log": (1, lambda a: _safe_clip(np.log(np.abs(_arr(a)) + 1e-6))),
    "sqrt": (1, lambda a: _safe_clip(np.sqrt(np.abs(_arr(a))))),
    "min": (2, lambda a, b: _safe_clip(np.minimum(_arr(a), _arr(b)))),
    "max": (2, lambda a, b: _safe_clip(np.maximum(_arr(a), _arr(b)))),
}


# =========================
# GP tree and individual
# =========================


@dataclass
class GPNode:
    kind: str  # func | var | const
    value: Any
    children: List["GPNode"] = field(default_factory=list)

    def clone(self) -> "GPNode":
        return GPNode(self.kind, copy.deepcopy(self.value), [c.clone() for c in self.children])

    def evaluate(self, context: Dict[str, Any]) -> np.ndarray:
        if self.kind == "const":
            ref = None
            for key in ("no", "nd", "sd"):
                if key in context:
                    ref = _arr(context[key])
                    break
            if ref is None:
                return np.asarray(float(self.value), dtype=float)
            return np.full_like(ref, float(self.value), dtype=float)
        if self.kind == "var":
            return _arr(context[self.value])
        arity, fn = PRIMITIVES[self.value]
        args = [child.evaluate(context) for child in self.children[:arity]]
        return fn(*args)

    def to_string(self) -> str:
        if self.kind == "const":
            return f"{float(self.value):.4f}"
        if self.kind == "var":
            return str(self.value)
        if len(self.children) == 1:
            return f"{self.value}({self.children[0].to_string()})"
        return f"{self.value}({self.children[0].to_string()}, {self.children[1].to_string()})"

    def paths(self, prefix: Tuple[int, ...] = ()) -> List[Tuple[Tuple[int, ...], "GPNode"]]:
        out = [(prefix, self)]
        for i, c in enumerate(self.children):
            out.extend(c.paths(prefix + (i,)))
        return out

    def subtree(self, path: Tuple[int, ...]) -> "GPNode":
        node = self
        for idx in path:
            node = node.children[idx]
        return node

    def replace(self, path: Tuple[int, ...], new_subtree: "GPNode") -> "GPNode":
        if not path:
            return new_subtree.clone()
        root = self.clone()
        node = root
        for idx in path[:-1]:
            node = node.children[idx]
        node.children[path[-1]] = new_subtree.clone()
        return root


class GPTreeFactory:
    def __init__(self, max_depth: int = 4, const_range: Tuple[float, float] = (-3.0, 3.0)):
        self.max_depth = max_depth
        self.const_range = const_range
        self.functions = list(PRIMITIVES)

    def terminal(self) -> GPNode:
        if random.random() < 0.3:
            return GPNode("const", random.uniform(*self.const_range))
        return GPNode("var", random.choice(TERMINALS))

    def random_tree(self, depth: int = 0, force_function: bool = False) -> GPNode:
        if depth >= self.max_depth or (depth > 0 and not force_function and random.random() < 0.45):
            return self.terminal()
        name = random.choice(self.functions)
        arity = PRIMITIVES[name][0]
        return GPNode("func", name, [self.random_tree(depth + 1) for _ in range(arity)])

    def mutate(self, tree: GPNode) -> GPNode:
        path = random.choice([p for p, _ in tree.paths()])
        return tree.replace(path, self.random_tree())

    def crossover(self, a: GPNode, b: GPNode) -> Tuple[GPNode, GPNode]:
        pa = random.choice([p for p, _ in a.paths()])
        pb = random.choice([p for p, _ in b.paths()])
        return a.replace(pa, b.subtree(pb)), b.replace(pb, a.subtree(pa))


@dataclass
class PolicyIndividual:
    surge_tree: GPNode
    subsidy_tree: GPNode
    fitness: Optional[np.ndarray] = None
    predicted_fitness: Optional[np.ndarray] = None
    ehvi: float = 0.0
    evaluated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "PolicyIndividual":
        return PolicyIndividual(
            surge_tree=self.surge_tree.clone(),
            subsidy_tree=self.subsidy_tree.clone(),
            fitness=None if self.fitness is None else self.fitness.copy(),
            predicted_fitness=None if self.predicted_fitness is None else self.predicted_fitness.copy(),
            ehvi=float(self.ehvi),
            evaluated=bool(self.evaluated),
            metadata=copy.deepcopy(self.metadata),
        )

    def platform_params(self) -> Dict[str, Callable[[float, np.ndarray, np.ndarray, np.ndarray], np.ndarray]]:
        def surge_fn(t: float, no: np.ndarray, nd: np.ndarray, sd: np.ndarray) -> np.ndarray:
            return _as_vector(self.surge_tree.evaluate({"t": t, "no": no, "nd": nd, "sd": sd}), no)

        def subsidy_fn(t: float, no: np.ndarray, nd: np.ndarray, sd: np.ndarray) -> np.ndarray:
            return _as_vector(self.subsidy_tree.evaluate({"t": t, "no": no, "nd": nd, "sd": sd}), no)

        return {"surge": surge_fn, "subsidy": subsidy_fn}

    def formulas(self) -> Dict[str, str]:
        return {"surge": self.surge_tree.to_string(), "subsidy": self.subsidy_tree.to_string()}


# =========================
# Pareto and NSGA-II helpers
# =========================


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return np.all(a >= b - EPS) and np.any(a > b + EPS)


def fitness_of(ind: PolicyIndividual) -> np.ndarray:
    fit = ind.fitness if ind.fitness is not None else ind.predicted_fitness
    if fit is None:
        raise ValueError("Individual has neither fitness nor predicted_fitness.")
    return np.asarray(fit, dtype=float)


def filter_nondominated(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.empty((0, 0), dtype=float)
    keep = []
    for i, p in enumerate(pts):
        if not any(i != j and dominates(pts[j], p) for j in range(len(pts))):
            keep.append(p)
    return np.asarray(keep, dtype=float)


def pareto_front(population: Sequence[PolicyIndividual]) -> List[PolicyIndividual]:
    out = []
    for i, ind in enumerate(population):
        p = fitness_of(ind)
        if not any(i != j and dominates(fitness_of(other), p) for j, other in enumerate(population)):
            out.append(ind)
    return out


def fast_non_dominated_sort(population: Sequence[PolicyIndividual]) -> List[List[int]]:
    fits = [fitness_of(ind) for ind in population]
    S = [[] for _ in population]
    n = [0] * len(population)
    fronts: List[List[int]] = [[]]
    for p in range(len(population)):
        for q in range(len(population)):
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


def crowding_distance(front: Sequence[int], population: Sequence[PolicyIndividual]) -> Dict[int, float]:
    if not front:
        return {}
    m = len(fitness_of(population[front[0]]))
    dist = {idx: 0.0 for idx in front}
    if len(front) <= 2:
        return {idx: float("inf") for idx in front}
    for obj in range(m):
        order = sorted(front, key=lambda idx: float(fitness_of(population[idx])[obj]))
        lo = float(fitness_of(population[order[0]])[obj])
        hi = float(fitness_of(population[order[-1]])[obj])
        dist[order[0]] = dist[order[-1]] = float("inf")
        if abs(hi - lo) < EPS:
            continue
        for i in range(1, len(order) - 1):
            prev_v = float(fitness_of(population[order[i - 1]])[obj])
            next_v = float(fitness_of(population[order[i + 1]])[obj])
            dist[order[i]] += (next_v - prev_v) / (hi - lo + EPS)
    return dist


def nsga2_select(population: Sequence[PolicyIndividual], size: int) -> List[PolicyIndividual]:
    selected: List[PolicyIndividual] = []
    for front in fast_non_dominated_sort(population):
        if len(selected) + len(front) <= size:
            selected.extend(population[idx].clone() for idx in front)
        else:
            dist = crowding_distance(front, population)
            ranked = sorted(front, key=lambda idx: dist[idx], reverse=True)
            selected.extend(population[idx].clone() for idx in ranked[: size - len(selected)])
            break
    return selected


# =========================
# Phenotype, surrogate, EHVI
# =========================


@dataclass
class ReferenceStateSampler:
    size: int = 50
    no_range: Tuple[float, float] = (0.0, 40.0)
    nd_range: Tuple[float, float] = (1.0, 40.0)
    t_range: Tuple[float, float] = (0.0, 1.0)
    seed: int = 42

    def sample(self) -> List[Dict[str, float]]:
        rng = np.random.default_rng(self.seed)
        states: List[Dict[str, float]] = []
        for _ in range(self.size):
            t = float(rng.uniform(*self.t_range))
            no = float(rng.uniform(*self.no_range))
            nd = float(rng.uniform(*self.nd_range))
            sd = float(no / max(nd, 1e-6))
            states.append({"t": t, "no": no, "nd": nd, "sd": sd})
        return states


class PhenotypeExtractor:
    def __init__(self, reference_states: Sequence[Dict[str, float]]):
        self.reference_states = list(reference_states)

    def extract(self, individual: PolicyIndividual) -> np.ndarray:
        vals: List[float] = []
        for s in self.reference_states:
            no = np.asarray([s["no"]], dtype=float)
            nd = np.asarray([s["nd"]], dtype=float)
            sd = np.asarray([s["sd"]], dtype=float)
            t = float(s["t"])
            surge_val = _as_vector(individual.surge_tree.evaluate({"t": t, "no": no, "nd": nd, "sd": sd}), no)
            subsidy_val = _as_vector(individual.subsidy_tree.evaluate({"t": t, "no": no, "nd": nd, "sd": sd}), no)
            vals.append(float(surge_val[0]))
            vals.append(float(subsidy_val[0]))
        return np.asarray(vals, dtype=float)


class MultiOutputGPSurrogate:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.models: List[GaussianProcessRegressor] = []
        self.fitted = False

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        Xs = self.x_scaler.fit_transform(np.asarray(X, dtype=float))
        Ys = self.y_scaler.fit_transform(np.asarray(Y, dtype=float))
        self.models = []
        for i in range(Ys.shape[1]):
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-4)
            model = GaussianProcessRegressor(kernel=kernel, normalize_y=False, alpha=1e-6, n_restarts_optimizer=1, random_state=self.seed + i)
            model.fit(Xs, Ys[:, i])
            self.models.append(model)
        self.fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("Surrogate has not been fitted.")
        Xs = self.x_scaler.transform(np.asarray(X, dtype=float))
        mean_list, std_list = [], []
        for model in self.models:
            mu, std = model.predict(Xs, return_std=True)
            mean_list.append(mu)
            std_list.append(std)
        mean = np.column_stack(mean_list)
        std = np.column_stack(std_list)
        mean = self.y_scaler.inverse_transform(mean)
        std = std * np.asarray(self.y_scaler.scale_, dtype=float)
        return mean, std


def hypervolume_mc(points: np.ndarray, reference_point: np.ndarray, num_samples: int, rng: np.random.Generator) -> float:
    pts = filter_nondominated(np.asarray(points, dtype=float))
    if pts.size == 0:
        return 0.0
    ref = np.asarray(reference_point, dtype=float)
    upper = np.max(pts, axis=0)
    span = upper - ref
    if np.any(span <= 0):
        return 0.0
    samples = rng.uniform(low=ref, high=upper, size=(num_samples, pts.shape[1]))
    dominated_mask = np.zeros(num_samples, dtype=bool)
    for p in pts:
        dominated_mask |= np.all(samples <= p + EPS, axis=1)
    return float(np.mean(dominated_mask) * np.prod(span))


class EHVISelector:
    def __init__(self, num_posterior_samples: int = 64, num_hv_mc_samples: int = 2048, seed: int = 42):
        self.num_posterior_samples = num_posterior_samples
        self.num_hv_mc_samples = num_hv_mc_samples
        self.rng = np.random.default_rng(seed)

    def score(self, means: np.ndarray, stds: np.ndarray, pareto_points: np.ndarray) -> np.ndarray:
        means = np.asarray(means, dtype=float)
        stds = np.asarray(stds, dtype=float)
        pareto_points = np.asarray(pareto_points, dtype=float)
        if pareto_points.size == 0:
            lo = np.min(means - 2.0 * stds, axis=0)
            hi = np.max(means + 2.0 * stds, axis=0)
            norm_means = _normalize_points(means, lo, hi)
            norm_stds = np.asarray(stds, dtype=float) / np.where(np.abs(hi - lo) < 1e-9, 1.0, hi - lo)
            return np.linalg.norm(norm_means, axis=1) + 0.1 * np.linalg.norm(norm_stds, axis=1)
        lo = np.min(np.vstack([pareto_points, means - 2.0 * stds]), axis=0)
        hi = np.max(np.vstack([pareto_points, means + 2.0 * stds]), axis=0)
        norm_pf = _normalize_points(pareto_points, lo, hi)
        ref = np.zeros(norm_pf.shape[1], dtype=float)
        current_hv = hypervolume_mc(norm_pf, ref, self.num_hv_mc_samples, self.rng)
        scores = np.zeros(len(means), dtype=float)
        hv_samples = max(512, self.num_hv_mc_samples // 4)
        for i, (mu, sigma) in enumerate(zip(means, stds)):
            sigma = np.maximum(sigma, 1e-6)
            ys = self.rng.normal(loc=mu, scale=sigma, size=(self.num_posterior_samples, len(mu)))
            gains = []
            for y in ys:
                norm_y = _normalize_points(np.asarray([y], dtype=float), lo, hi)[0]
                cand_front = filter_nondominated(np.vstack([norm_pf, norm_y]))
                hv = hypervolume_mc(cand_front, ref, hv_samples, self.rng)
                gains.append(max(0.0, hv - current_hv))
            scores[i] = float(np.mean(gains))
        return scores


# =========================
# Transfer memory and evaluator
# =========================


@dataclass
class KnowledgeRecord:
    phenotype: np.ndarray
    policy_state: Dict[str, Any]
    policy_old_state: Dict[str, Any]
    fitness: np.ndarray
    formulas: Dict[str, str]


class TransferMemory:
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self.records: List[KnowledgeRecord] = []

    def add(self, record: KnowledgeRecord) -> None:
        self.records.append(record)
        if len(self.records) > self.max_size:
            self.records = self.records[-self.max_size :]

    def nearest_with_distance(self, phenotype: np.ndarray) -> Tuple[Optional[KnowledgeRecord], float]:
        if not self.records:
            return None, float("inf")
        target = np.asarray(phenotype, dtype=float)
        best = min(self.records, key=lambda r: _phenotype_distance(r.phenotype, target))
        return best, _phenotype_distance(best.phenotype, target)


class TransferWarmStarter:
    def __init__(self, memory: Optional[TransferMemory] = None, similarity_threshold: float = 1.0):
        self.memory = memory or TransferMemory()
        self.similarity_threshold = similarity_threshold

    def maybe_load(self, trainer: Any, phenotype: np.ndarray) -> bool:
        record, distance = self.memory.nearest_with_distance(phenotype)
        if record is None or distance > self.similarity_threshold:
            if hasattr(trainer, "reset_to_base_weights"):
                trainer.reset_to_base_weights()
            return False
        if hasattr(trainer, "reset_to_base_weights"):
            trainer.reset_to_base_weights()
        if hasattr(trainer, "agent"):
            trainer.agent.policy.load_state_dict(record.policy_state)
            trainer.agent.policy_old.load_state_dict(record.policy_old_state)
            if hasattr(trainer.agent, "buffer"):
                trainer.agent.buffer.clear()
            return True
        return False

    def store(self, trainer: Any, phenotype: np.ndarray, fitness: np.ndarray, formulas: Dict[str, str]) -> None:
        if not hasattr(trainer, "agent"):
            return
        record = KnowledgeRecord(
            phenotype=np.asarray(phenotype, dtype=float).copy(),
            policy_state=copy.deepcopy(trainer.agent.policy.state_dict()),
            policy_old_state=copy.deepcopy(trainer.agent.policy_old.state_dict()),
            fitness=np.asarray(fitness, dtype=float).copy(),
            formulas=copy.deepcopy(formulas),
        )
        self.memory.add(record)


class LowerLevelEvaluator:
    def __init__(self, trainer: Any, phenotype_extractor: PhenotypeExtractor, use_transfer: bool = True, eval_episodes: int = 5, memory_size: int = 128, similarity_threshold: float = 1.0):
        self.trainer = trainer
        self.phenotype_extractor = phenotype_extractor
        self.use_transfer = use_transfer
        self.eval_episodes = eval_episodes
        self.eval_times: List[float] = []
        self.warm_starter = TransferWarmStarter(TransferMemory(max_size=memory_size), similarity_threshold=similarity_threshold)
        if hasattr(self.trainer, "save"):
            self.trainer.save = lambda: None

    def evaluate(self, individual: PolicyIndividual) -> np.ndarray:
        phenotype = self.phenotype_extractor.extract(individual)
        if self.use_transfer:
            self.warm_starter.maybe_load(self.trainer, phenotype)
        elif hasattr(self.trainer, "reset_to_base_weights"):
            self.trainer.reset_to_base_weights()
        start = __import__("time").perf_counter()
        fitness = np.asarray(
            self.trainer.train_and_evaluate(individual.platform_params(), num_episodes=self.eval_episodes, show_fig=False),
            dtype=float,
        )
        elapsed = __import__("time").perf_counter() - start
        self.eval_times.append(float(elapsed))
        individual.fitness = fitness.copy()
        individual.evaluated = True
        individual.metadata["phenotype"] = phenotype
        individual.metadata["eval_time_sec"] = float(elapsed)
        if self.use_transfer:
            self.warm_starter.store(self.trainer, phenotype, fitness, individual.formulas())
        return fitness


# =========================
# Main algorithm
# =========================


@dataclass
class TLConfig:
    population_size: int = 16
    generations: int = 5
    real_eval_quota: int = 4
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    tournament_size: int = 2
    max_tree_depth: int = 4
    ref_state_size: int = 24
    random_seed: int = 7
    use_surrogate: bool = True
    use_transfer: bool = True
    transfer_similarity_threshold: float = 1.0


class TLSAMOGP:
    def __init__(self, evaluator: LowerLevelEvaluator, config: Optional[TLConfig] = None):
        self.config = config or TLConfig()
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        self.evaluator = evaluator
        self.tree_factory = GPTreeFactory(max_depth=self.config.max_tree_depth)
        self.surrogate = MultiOutputGPSurrogate(seed=self.config.random_seed)
        self.selector = EHVISelector(seed=self.config.random_seed)
        self.archive: List[PolicyIndividual] = []
        self.history: List[Dict[str, Any]] = []

    def _random_individual(self) -> PolicyIndividual:
        return PolicyIndividual(
            surge_tree=self.tree_factory.random_tree(force_function=True),
            subsidy_tree=self.tree_factory.random_tree(force_function=True),
        )

    def _archive_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        X, Y = [], []
        for ind in self.archive:
            if ind.fitness is None:
                continue
            pheno = ind.metadata.get("phenotype")
            if pheno is None:
                pheno = self.evaluator.phenotype_extractor.extract(ind)
                ind.metadata["phenotype"] = pheno
            X.append(pheno)
            Y.append(ind.fitness)
        if not X:
            return np.empty((0, 0), dtype=float), np.empty((0, 0), dtype=float)
        return np.asarray(X, dtype=float), np.asarray(Y, dtype=float)

    def _fit_surrogate(self) -> bool:
        if not self.config.use_surrogate:
            return False
        X, Y = self._archive_arrays()
        min_needed = max(5, 4)
        if len(X) < min_needed:
            return False
        self.surrogate.fit(X, Y)
        return True

    def _offspring(self, population: Sequence[PolicyIndividual]) -> List[PolicyIndividual]:
        fronts = fast_non_dominated_sort(population)
        rank = {}
        crowd = {}
        for r, front in enumerate(fronts):
            for idx in front:
                rank[idx] = r
            crowd.update(crowding_distance(front, population))

        def tour_select() -> PolicyIndividual:
            picks = random.sample(list(enumerate(population)), k=min(self.config.tournament_size, len(population)))
            picks.sort(key=lambda it: (rank.get(it[0], 10**9), -crowd.get(it[0], 0.0)))
            return picks[0][1].clone()

        children: List[PolicyIndividual] = []
        while len(children) < len(population):
            p1, p2 = tour_select(), tour_select()
            c1, c2 = p1.clone(), p2.clone()
            if random.random() < self.config.crossover_rate:
                c1.surge_tree, c2.surge_tree = self.tree_factory.crossover(c1.surge_tree, c2.surge_tree)
                c1.subsidy_tree, c2.subsidy_tree = self.tree_factory.crossover(c1.subsidy_tree, c2.subsidy_tree)
            if random.random() < self.config.mutation_rate:
                c1.surge_tree = self.tree_factory.mutate(c1.surge_tree)
            if random.random() < self.config.mutation_rate:
                c1.subsidy_tree = self.tree_factory.mutate(c1.subsidy_tree)
            if random.random() < self.config.mutation_rate:
                c2.surge_tree = self.tree_factory.mutate(c2.surge_tree)
            if random.random() < self.config.mutation_rate:
                c2.subsidy_tree = self.tree_factory.mutate(c2.subsidy_tree)
            c1.fitness = c1.predicted_fitness = None
            c2.fitness = c2.predicted_fitness = None
            c1.evaluated = c2.evaluated = False
            children.extend([c1, c2])
        return children[: len(population)]

    def _prescreen(self, offspring: Sequence[PolicyIndividual]) -> List[PolicyIndividual]:
        if not self.config.use_surrogate or not self.surrogate.fitted:
            return list(offspring)
        X = []
        for ind in offspring:
            pheno = self.evaluator.phenotype_extractor.extract(ind)
            ind.metadata["phenotype"] = pheno
            X.append(pheno)
        means, stds = self.surrogate.predict(np.asarray(X, dtype=float))
        real_pf = pareto_front(self.archive)
        pf_points = np.asarray([ind.fitness for ind in real_pf if ind.fitness is not None], dtype=float)
        scores = self.selector.score(means, stds, pf_points)
        for ind, mu, score in zip(offspring, means, scores):
            ind.predicted_fitness = np.asarray(mu, dtype=float)
            ind.ehvi = float(score)
        ranked = sorted(offspring, key=lambda ind: ind.ehvi, reverse=True)
        real_k = min(self.config.real_eval_quota, len(ranked))
        for ind in ranked[real_k:]:
            ind.fitness = ind.predicted_fitness.copy() if ind.predicted_fitness is not None else None
        return ranked[:real_k]

    def _evaluate_many(self, population: Sequence[PolicyIndividual]) -> None:
        for ind in population:
            if ind.fitness is None:
                self.evaluator.evaluate(ind)
                self.archive.append(ind.clone())

    def initialize(self) -> List[PolicyIndividual]:
        population = [self._random_individual() for _ in range(self.config.population_size)]
        self._evaluate_many(population)
        self._fit_surrogate()
        return population

    def run(self) -> List[PolicyIndividual]:
        population = self.initialize()
        for gen in range(1, self.config.generations + 1):
            offspring = self._offspring(population)
            for ind in self._prescreen(offspring):
                self.evaluator.evaluate(ind)
                self.archive.append(ind.clone())
            self._fit_surrogate()
            population = nsga2_select(list(population) + list(offspring), self.config.population_size)
            real_pf = pareto_front(self.archive)
            fits = np.asarray([ind.fitness for ind in self.archive if ind.fitness is not None], dtype=float)
            hv = 0.0
            if real_pf:
                pf_points = np.asarray([ind.fitness for ind in real_pf if ind.fitness is not None], dtype=float)
                hv = _normalized_hv(
                    pf_points,
                    fits if len(fits) else pf_points,
                    num_samples=1024,
                    rng=np.random.default_rng(self.config.random_seed + gen),
                )
            mean_eval = 0.0
            cum_eval = 0.0
            if hasattr(self.evaluator, 'eval_times') and getattr(self.evaluator, 'eval_times'):
                mean_eval = float(np.mean(self.evaluator.eval_times))
                cum_eval = float(np.sum(self.evaluator.eval_times))
            self.history.append({
                "generation": gen,
                "archive_size": len(self.archive),
                "pareto_size": len(real_pf),
                "hypervolume": float(hv),
                "best_profit": float(np.max(fits[:, 0])) if len(fits) else 0.0,
                "best_efficiency": float(np.max(fits[:, 1])) if len(fits) else 0.0,
                "best_fairness": float(np.max(fits[:, 2])) if len(fits) else 0.0,
                "real_evals": int(len(fits)),
                "mean_eval_time_sec": mean_eval,
                "cum_eval_time_sec": cum_eval,
            })
        return pareto_front(self.archive)


# =========================
# Lower-level wiring
# =========================


def build_shared_ppo_evaluator(simulator_path: str, eval_episodes: int = 1, use_transfer: bool = True, ref_state_size: int = 24, seed: int = 7, similarity_threshold: float = 1.0) -> LowerLevelEvaluator:
    from shared_ppo import Trainer

    if not os.path.exists(simulator_path):
        raise FileNotFoundError(f"Simulator file not found: {simulator_path}")
    trainer = Trainer(simulator_path=simulator_path)
    refs = ReferenceStateSampler(size=ref_state_size, seed=seed).sample()
    return LowerLevelEvaluator(
        trainer=trainer,
        phenotype_extractor=PhenotypeExtractor(refs),
        use_transfer=use_transfer,
        eval_episodes=eval_episodes,
        similarity_threshold=similarity_threshold,
    )


# =========================
# Dummy trainer for quick test
# =========================


class _DummyBuffer:
    def clear(self) -> None:
        return None


class _DummyStateful:
    def __init__(self) -> None:
        self._state = {"w": np.array([0.0], dtype=float)}

    def state_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._state)

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._state = copy.deepcopy(state)


class _DummyAgent:
    def __init__(self) -> None:
        self.policy = _DummyStateful()
        self.policy_old = _DummyStateful()
        self.buffer = _DummyBuffer()


class DummyTrainer:
    def __init__(self) -> None:
        self.agent = _DummyAgent()
        self._base_policy = self.agent.policy.state_dict()
        self._base_policy_old = self.agent.policy_old.state_dict()

    def reset_to_base_weights(self) -> None:
        self.agent.policy.load_state_dict(self._base_policy)
        self.agent.policy_old.load_state_dict(self._base_policy_old)
        self.agent.buffer.clear()

    def train_and_evaluate(self, platform_params: Dict[str, Callable], num_episodes: int = 1, show_fig: bool = False) -> np.ndarray:
        no = np.linspace(0.0, 12.0, 10)
        nd = np.linspace(1.0, 10.0, 10)
        sd = no / nd
        surge = np.asarray(platform_params["surge"](0.35, no, nd, sd), dtype=float)
        subsidy = np.asarray(platform_params["subsidy"](0.35, no, nd, sd), dtype=float)
        x = float(np.mean(surge))
        y = float(np.mean(subsidy))
        profit = -(x - 1.8) ** 2 - 0.25 * (y - 0.8) ** 2 + 5.0
        efficiency = -abs(x - 1.1) - 0.15 * abs(y - 0.4) + 2.5
        fairness = -abs(y - 0.6) - 0.2 * abs(x - 1.2) + 1.5
        self.agent.policy.load_state_dict({"w": np.array([x + y], dtype=float)})
        self.agent.policy_old.load_state_dict({"w": np.array([x + y], dtype=float)})
        return np.asarray([profit, efficiency, fairness], dtype=float)


# =========================
# CLI / minimal experiment
# =========================


def run_minimal_experiment(
    simulator_path: Optional[str] = None,
    population_size: int = 8,
    generations: int = 3,
    real_eval_quota: int = 3,
    eval_episodes: int = 1,
    use_surrogate: bool = True,
    use_transfer: bool = True,
    seed: int = 7,
) -> Dict[str, Any]:
    if simulator_path:
        evaluator = build_shared_ppo_evaluator(
            simulator_path=simulator_path,
            eval_episodes=eval_episodes,
            use_transfer=use_transfer,
            ref_state_size=24,
            seed=seed,
        )
        mode = "real"
    else:
        refs = ReferenceStateSampler(size=24, seed=seed).sample()
        evaluator = LowerLevelEvaluator(
            trainer=DummyTrainer(),
            phenotype_extractor=PhenotypeExtractor(refs),
            use_transfer=use_transfer,
            eval_episodes=1,
            similarity_threshold=1.0,
        )
        mode = "dummy"

    algo = TLSAMOGP(
        evaluator=evaluator,
        config=TLConfig(
            population_size=population_size,
            generations=generations,
            real_eval_quota=real_eval_quota,
            random_seed=seed,
            use_surrogate=use_surrogate,
            use_transfer=use_transfer,
            ref_state_size=24,
        ),
    )
    pareto = algo.run()
    result = {
        "mode": mode,
        "config": {
            "population_size": population_size,
            "generations": generations,
            "real_eval_quota": real_eval_quota,
            "eval_episodes": eval_episodes,
            "use_surrogate": use_surrogate,
            "use_transfer": use_transfer,
            "seed": seed,
            "simulator_path": simulator_path,
        },
        "history": algo.history,
        "pareto": [
            {
                "fitness": np.asarray(ind.fitness, dtype=float).round(6).tolist() if ind.fitness is not None else None,
                "formulas": ind.formulas(),
            }
            for ind in pareto
        ],
    }
    return result


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal TL-SAMOGP experiment.")
    parser.add_argument("--simulator_path", type=str, default=None, help="Path to simulator.pkl. If omitted, runs a dummy sanity-check experiment.")
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--real_eval_quota", type=int, default=3)
    parser.add_argument("--eval_episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--disable_surrogate", action="store_true")
    parser.add_argument("--disable_transfer", action="store_true")
    parser.add_argument("--save_json", type=str, default=None, help="Optional path to save experiment summary JSON.")
    return parser


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    summary = run_minimal_experiment(
        simulator_path=args.simulator_path,
        population_size=args.population,
        generations=args.generations,
        real_eval_quota=args.real_eval_quota,
        eval_episodes=args.eval_episodes,
        use_surrogate=not args.disable_surrogate,
        use_transfer=not args.disable_transfer,
        seed=args.seed,
    )

    print(f"[TL-SAMOGP] mode={summary['mode']}")
    print(f"[TL-SAMOGP] config={summary['config']}")
    for item in summary["history"]:
        print(
            "[Gen {generation}] archive={archive_size}, pareto={pareto_size}, hv={hypervolume}".format(
                **item
            )
        )
    print(f"[TL-SAMOGP] pareto_size={len(summary['pareto'])}")
    for i, item in enumerate(summary["pareto"][:5]):
        print(f"  - #{i}: fitness={item['fitness']}")
        print(f"    surge   = {item['formulas']['surge']}")
        print(f"    subsidy = {item['formulas']['subsidy']}")

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[TL-SAMOGP] summary saved to: {args.save_json}")
