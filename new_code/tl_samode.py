from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from generate_simulator import PassengerSimulator
import numpy as np

from tl_samogp import EHVISelector, MultiOutputGPSurrogate, nsga2_select, pareto_front
from upper_experiment_utils import archive_metrics, build_experiment_evaluator


@dataclass
class DEPolicyIndividual:
    params: np.ndarray
    fitness: Optional[np.ndarray] = None
    predicted_fitness: Optional[np.ndarray] = None
    ehvi: float = 0.0
    evaluated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "DEPolicyIndividual":
        return DEPolicyIndividual(
            params=np.asarray(self.params, dtype=float).copy(),
            fitness=None if self.fitness is None else np.asarray(self.fitness, dtype=float).copy(),
            predicted_fitness=None if self.predicted_fitness is None else np.asarray(self.predicted_fitness, dtype=float).copy(),
            ehvi=float(self.ehvi),
            evaluated=bool(self.evaluated),
            metadata=copy.deepcopy(self.metadata),
        )

    def platform_params(self):
        p = np.asarray(self.params, dtype=float)

        def surge(t, no, nd, sd):
            ratio = np.asarray(sd, dtype=float)
            return np.asarray(p[0] + p[1] * np.sin(p[2] * t + p[3] * ratio) + p[4] * ratio, dtype=float)

        def subsidy(t, no, nd, sd):
            ratio = np.asarray(sd, dtype=float)
            return np.asarray(
                p[5] + p[6] * np.sin(p[7] * t + p[8] * ratio) + p[9] * np.maximum(0.0, ratio - 1.0),
                dtype=float,
            )

        return {"surge": surge, "subsidy": subsidy}

    def formulas(self) -> Dict[str, str]:
        p = [float(x) for x in self.params]
        return {
            "surge": f"{p[0]:.4f} + {p[1]:.4f}*sin({p[2]:.4f}*t + {p[3]:.4f}*sd) + {p[4]:.4f}*sd",
            "subsidy": f"{p[5]:.4f} + {p[6]:.4f}*sin({p[7]:.4f}*t + {p[8]:.4f}*sd) + {p[9]:.4f}*max(sd-1,0)",
        }


@dataclass
class TLDEConfig:
    population_size: int = 16
    generations: int = 5
    real_eval_quota: int = 4
    differential_weight: float = 0.7
    crossover_rate: float = 0.9
    param_bounds: Sequence[tuple] = (
        (-2.0, 3.0), (-3.0, 3.0), (-6.0, 6.0), (-6.0, 6.0), (-3.0, 3.0),
        (0.0, 6.0), (-4.0, 4.0), (-6.0, 6.0), (-6.0, 6.0), (-4.0, 8.0),
    )
    random_seed: int = 7
    use_surrogate: bool = True
    use_transfer: bool = True


class TLSAMODE:
    def __init__(self, evaluator, config: Optional[TLDEConfig] = None):
        self.config = config or TLDEConfig()
        self.evaluator = evaluator
        self.rng = np.random.default_rng(self.config.random_seed)
        random.seed(self.config.random_seed)
        self.surrogate = MultiOutputGPSurrogate(seed=self.config.random_seed)
        self.selector = EHVISelector(seed=self.config.random_seed)
        self.archive: List[DEPolicyIndividual] = []
        self.history: List[Dict[str, Any]] = []

    def _random_individual(self) -> DEPolicyIndividual:
        vec = [self.rng.uniform(lo, hi) for lo, hi in self.config.param_bounds]
        return DEPolicyIndividual(params=np.asarray(vec, dtype=float))

    def _archive_arrays(self):
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
        if len(X) < 5:
            return False
        self.surrogate.fit(X, Y)
        return True

    def _evaluate(self, ind: DEPolicyIndividual) -> None:
        self.evaluator.evaluate(ind)
        self.archive.append(ind.clone())

    def _trial(self, population: Sequence[DEPolicyIndividual], idx: int) -> DEPolicyIndividual:
        n = len(population)
        pool = [j for j in range(n) if j != idx]
        a, b, c = self.rng.choice(pool, size=3, replace=False)
        x_a, x_b, x_c = population[a].params, population[b].params, population[c].params
        mutant = x_a + self.config.differential_weight * (x_b - x_c)
        bounds = np.asarray(self.config.param_bounds, dtype=float)
        mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])
        target = population[idx].params
        trial = target.copy()
        j_rand = self.rng.integers(len(target))
        mask = self.rng.random(len(target)) < self.config.crossover_rate
        mask[j_rand] = True
        trial[mask] = mutant[mask]
        return DEPolicyIndividual(params=trial)

    def _prescreen(self, offspring: Sequence[DEPolicyIndividual]) -> List[DEPolicyIndividual]:
        if not self.config.use_surrogate or not self.surrogate.fitted:
            return list(offspring)
        X = []
        for ind in offspring:
            ph = self.evaluator.phenotype_extractor.extract(ind)
            ind.metadata["phenotype"] = ph
            X.append(ph)
        means, stds = self.surrogate.predict(np.asarray(X, dtype=float))
        real_pf = pareto_front(self.archive)
        pf_points = np.asarray([ind.fitness for ind in real_pf if ind.fitness is not None], dtype=float)
        scores = self.selector.score(means, stds, pf_points)
        for ind, mu, score in zip(offspring, means, scores):
            ind.predicted_fitness = np.asarray(mu, dtype=float)
            ind.ehvi = float(score)
        ranked = sorted(offspring, key=lambda x: x.ehvi, reverse=True)
        real_k = min(self.config.real_eval_quota, len(ranked))
        for ind in ranked[real_k:]:
            if ind.predicted_fitness is not None:
                ind.fitness = ind.predicted_fitness.copy()
        return ranked[:real_k]

    def initialize(self) -> List[DEPolicyIndividual]:
        pop = [self._random_individual() for _ in range(self.config.population_size)]
        for ind in pop:
            self._evaluate(ind)
        self._fit_surrogate()
        return pop

    def run(self) -> List[DEPolicyIndividual]:
        population = self.initialize()
        for gen in range(1, self.config.generations + 1):
            offspring = [self._trial(population, i) for i in range(len(population))]
            for ind in self._prescreen(offspring):
                self._evaluate(ind)
            self._fit_surrogate()
            population = nsga2_select(list(population) + list(offspring), self.config.population_size)
            metrics = archive_metrics(self.archive, getattr(self.evaluator, "eval_times", []), seed=self.config.random_seed + gen)
            metrics["generation"] = gen
            self.history.append(metrics)
        return pareto_front(self.archive)


def run_minimal_tl_samode(
    simulator_path: Optional[str] = None,
    population_size: int = 8,
    generations: int = 3,
    real_eval_quota: int = 3,
    eval_episodes: int = 1,
    use_surrogate: bool = True,
    use_transfer: bool = True,
    seed: int = 7,
) -> Dict[str, Any]:
    evaluator = build_experiment_evaluator(
        simulator_path=simulator_path,
        eval_episodes=eval_episodes,
        use_transfer=use_transfer,
        ref_state_size=24,
        seed=seed,
    )
    algo = TLSAMODE(
        evaluator=evaluator,
        config=TLDEConfig(
            population_size=population_size,
            generations=generations,
            real_eval_quota=real_eval_quota,
            random_seed=seed,
            use_surrogate=use_surrogate,
            use_transfer=use_transfer,
        ),
    )
    pareto = algo.run()
    summary = archive_metrics(algo.archive, evaluator.eval_times, seed=seed)
    return {
        "algorithm": "TL-SAMODE",
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
        "summary": summary,
        "pareto": [
            {
                "fitness": None if ind.fitness is None else np.asarray(ind.fitness, dtype=float).round(6).tolist(),
                "formulas": ind.formulas(),
            }
            for ind in pareto
        ],
    }


def _build_argparser():
    parser = argparse.ArgumentParser(description="Run a minimal TL-SAMODE experiment.")
    parser.add_argument("--simulator_path", type=str, default=None)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--real_eval_quota", type=int, default=3)
    parser.add_argument("--eval_episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--disable_surrogate", action="store_true")
    parser.add_argument("--disable_transfer", action="store_true")
    parser.add_argument("--save_json", type=str, default=None)
    return parser


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    result = run_minimal_tl_samode(
        simulator_path=args.simulator_path,
        population_size=args.population,
        generations=args.generations,
        real_eval_quota=args.real_eval_quota,
        eval_episodes=args.eval_episodes,
        use_surrogate=not args.disable_surrogate,
        use_transfer=not args.disable_transfer,
        seed=args.seed,
    )
    print(f"[TL-SAMODE] summary={result['summary']}")
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[TL-SAMODE] saved to {args.save_json}")
