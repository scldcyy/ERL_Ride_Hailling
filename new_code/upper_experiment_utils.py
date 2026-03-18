from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple
from generate_simulator import PassengerSimulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests
from tl_samogp import DummyTrainer, ReferenceStateSampler, hypervolume_mc, pareto_front


def _phenotype_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0:
        return float('inf')
    return float(np.linalg.norm(a - b) / np.sqrt(a.size))


def _normalize_objectives(fits: np.ndarray, bounds_source: Optional[np.ndarray] = None) -> np.ndarray:
    fits = np.asarray(fits, dtype=float)
    if fits.size == 0:
        return fits
    source = fits if bounds_source is None else np.asarray(bounds_source, dtype=float)
    lo = np.min(source, axis=0)
    hi = np.max(source, axis=0)
    span = np.where(np.abs(hi - lo) < 1e-9, 1.0, hi - lo)
    return np.clip((fits - lo) / span, 0.0, 1.0)


def _read_expected_seeds(results_root: str) -> Optional[set[int]]:
    config_path = os.path.join(results_root, 'run_config.json')
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        seeds = data.get('seeds')
        if not seeds:
            return None
        return {int(seed) for seed in seeds}
    except Exception:
        return None


@dataclass
class SnapshotRecord:
    phenotype: np.ndarray
    snapshot: Dict[str, Any]
    fitness: np.ndarray
    descriptor: Dict[str, Any]


class SnapshotMemory:
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self.records: List[SnapshotRecord] = []

    def add(self, record: SnapshotRecord) -> None:
        self.records.append(record)
        if len(self.records) > self.max_size:
            self.records = self.records[-self.max_size:]

    def nearest_with_distance(self, phenotype: np.ndarray) -> Tuple[Optional[SnapshotRecord], float]:
        if not self.records:
            return None, float('inf')
        target = np.asarray(phenotype, dtype=float)
        best = min(self.records, key=lambda r: _phenotype_distance(r.phenotype, target))
        return best, _phenotype_distance(best.phenotype, target)


class FunctionalPhenotypeExtractor:
    def __init__(self, reference_states: Sequence[Dict[str, float]]):
        self.reference_states = list(reference_states)

    def extract(self, individual: Any) -> np.ndarray:
        policy = individual.platform_params() if hasattr(individual, 'platform_params') else individual.to_platform_params()
        vals: List[float] = []
        for s in self.reference_states:
            no = np.asarray([s['no']], dtype=float)
            nd = np.asarray([s['nd']], dtype=float)
            sd = np.asarray([s['sd']], dtype=float)
            t = float(s['t'])
            surge = np.asarray(policy['surge'](t, no, nd, sd), dtype=float)
            subsidy = np.asarray(policy['subsidy'](t, no, nd, sd), dtype=float)
            vals.extend([float(np.ravel(surge)[0]), float(np.ravel(subsidy)[0])])
        return np.asarray(vals, dtype=float)


class ExperimentEvaluator:
    def __init__(
        self,
        trainer: Any,
        phenotype_extractor: FunctionalPhenotypeExtractor,
        use_transfer: bool = True,
        eval_episodes: int = 3,
        memory_size: int = 128,
        similarity_threshold: float = 1.0,
    ):
        self.trainer = trainer
        self.phenotype_extractor = phenotype_extractor
        self.use_transfer = use_transfer
        self.eval_episodes = eval_episodes
        self.memory = SnapshotMemory(max_size=memory_size)
        self.similarity_threshold = similarity_threshold
        self.eval_times: List[float] = []
        self.eval_fitnesses: List[np.ndarray] = []
        if hasattr(self.trainer, 'save'):
            self.trainer.save = lambda *args, **kwargs: None

    def _snapshot(self) -> Optional[Dict[str, Any]]:
        if hasattr(self.trainer, 'get_policy_snapshot'):
            return self.trainer.get_policy_snapshot()
        if hasattr(self.trainer, 'agent'):
            snap = {
                'policy': copy.deepcopy(self.trainer.agent.policy.state_dict()),
                'policy_old': copy.deepcopy(self.trainer.agent.policy_old.state_dict()),
            }
            if hasattr(self.trainer.agent, 'optimizer'):
                snap['optimizer'] = copy.deepcopy(self.trainer.agent.optimizer.state_dict())
            return snap
        return None

    def _prepare(self, snapshot: Optional[Dict[str, Any]]) -> None:
        if hasattr(self.trainer, 'prepare_for_new_policy'):
            self.trainer.prepare_for_new_policy(snapshot)
            return
        if hasattr(self.trainer, 'reset_to_base_weights'):
            self.trainer.reset_to_base_weights()
        if snapshot is None or not hasattr(self.trainer, 'agent'):
            return
        self.trainer.agent.policy.load_state_dict(snapshot['policy'])
        self.trainer.agent.policy_old.load_state_dict(snapshot.get('policy_old', snapshot['policy']))
        if hasattr(self.trainer.agent, 'buffer'):
            self.trainer.agent.buffer.clear()
        if 'optimizer' in snapshot and hasattr(self.trainer.agent, 'optimizer'):
            try:
                self.trainer.agent.optimizer.load_state_dict(snapshot['optimizer'])
            except Exception:
                pass

    def evaluate(self, individual: Any) -> np.ndarray:
        phenotype = self.phenotype_extractor.extract(individual)
        snapshot = None
        warm_start_distance = float('inf')
        warm_start_used = False
        if self.use_transfer:
            rec, warm_start_distance = self.memory.nearest_with_distance(phenotype)
            if rec is not None and warm_start_distance <= self.similarity_threshold:
                snapshot = rec.snapshot
                warm_start_used = True
        self._prepare(snapshot)
        start = time.perf_counter()
        platform_params = individual.platform_params() if hasattr(individual, 'platform_params') else individual.to_platform_params()
        fitness = np.asarray(
            self.trainer.train_and_evaluate(platform_params, num_episodes=self.eval_episodes, show_fig=False),
            dtype=float,
        )
        elapsed = time.perf_counter() - start
        self.eval_times.append(float(elapsed))
        self.eval_fitnesses.append(fitness.copy())
        individual.fitness = fitness.copy()
        individual.evaluated = True
        individual.metadata['phenotype'] = phenotype.copy()
        individual.metadata['eval_time_sec'] = float(elapsed)
        individual.metadata['warm_start_distance'] = float(warm_start_distance)
        individual.metadata['warm_start_used'] = bool(warm_start_used)
        if self.use_transfer:
            snap = self._snapshot()
            if snap is not None:
                descriptor = individual.formulas() if hasattr(individual, 'formulas') else individual.formula()
                self.memory.add(SnapshotRecord(phenotype=phenotype.copy(), snapshot=snap, fitness=fitness.copy(), descriptor=copy.deepcopy(descriptor)))
        return fitness


def build_experiment_evaluator(
    simulator_path: Optional[str] = None,
    eval_episodes: int = 1,
    use_transfer: bool = True,
    ref_state_size: int = 24,
    seed: int = 7,
    similarity_threshold: float = 1.0,
) -> ExperimentEvaluator:
    refs = ReferenceStateSampler(size=ref_state_size, seed=seed).sample()
    extractor = FunctionalPhenotypeExtractor(refs)
    if simulator_path:
        from shared_ppo import Trainer
        trainer = Trainer(simulator_path=simulator_path)
    else:
        trainer = DummyTrainer()
    return ExperimentEvaluator(
        trainer=trainer,
        phenotype_extractor=extractor,
        use_transfer=use_transfer,
        eval_episodes=eval_episodes,
        similarity_threshold=similarity_threshold,
    )


def compute_hv_from_fitness(fits: np.ndarray, bounds_source: Optional[np.ndarray] = None, seed: int = 7) -> float:
    fits = np.asarray(fits, dtype=float)
    if fits.size == 0:
        return 0.0
    norm_fits = _normalize_objectives(fits, bounds_source=bounds_source)
    ref = np.zeros(norm_fits.shape[1], dtype=float)
    return float(hypervolume_mc(norm_fits, ref, num_samples=1024, rng=np.random.default_rng(seed)))


def archive_metrics(archive: Sequence[Any], eval_times: Optional[Sequence[float]] = None, seed: int = 7) -> Dict[str, float]:
    fits = np.asarray([np.asarray(ind.fitness, dtype=float) for ind in archive if getattr(ind, 'fitness', None) is not None], dtype=float)
    if fits.size == 0:
        return {
            'hypervolume': 0.0,
            'pareto_size': 0,
            'best_profit': 0.0,
            'best_efficiency': 0.0,
            'best_fairness': 0.0,
            'real_evals': 0,
            'mean_eval_time_sec': 0.0,
            'cum_eval_time_sec': 0.0,
        }
    pf = pareto_front(archive)
    pf_points = np.asarray([np.asarray(ind.fitness, dtype=float) for ind in pf if getattr(ind, 'fitness', None) is not None], dtype=float)
    hv = compute_hv_from_fitness(pf_points, bounds_source=fits, seed=seed) if pf_points.size else 0.0
    eval_times = list(eval_times or [])
    return {
        'hypervolume': float(hv),
        'pareto_size': int(len(pf_points)),
        'best_profit': float(np.max(fits[:, 0])),
        'best_efficiency': float(np.max(fits[:, 1])),
        'best_fairness': float(np.max(fits[:, 2])),
        'real_evals': int(len(fits)),
        'mean_eval_time_sec': float(np.mean(eval_times)) if eval_times else 0.0,
        'cum_eval_time_sec': float(np.sum(eval_times)) if eval_times else 0.0,
    }


class RunningScalarizer:
    def __init__(self, weights: Sequence[float] = (1.0, 1.0, 1.0)):
        self.weights = np.asarray(weights, dtype=float)
        self.min_v: Optional[np.ndarray] = None
        self.max_v: Optional[np.ndarray] = None

    def update(self, fits: np.ndarray) -> None:
        fits = np.asarray(fits, dtype=float)
        cur_min = np.min(fits, axis=0)
        cur_max = np.max(fits, axis=0)
        self.min_v = cur_min if self.min_v is None else np.minimum(self.min_v, cur_min)
        self.max_v = cur_max if self.max_v is None else np.maximum(self.max_v, cur_max)

    def score(self, fits: np.ndarray) -> np.ndarray:
        fits = np.asarray(fits, dtype=float)
        self.update(fits)
        den = np.maximum(self.max_v - self.min_v, 1e-6)
        norm = (fits - self.min_v) / den
        return norm @ (self.weights / np.sum(self.weights))


def save_run_json(result: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def _interp_curve(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.full_like(grid, np.nan, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x, uniq_idx = np.unique(x, return_index=True)
    y = y[uniq_idx]
    if len(x) == 1:
        return np.full_like(grid, y[0], dtype=float)
    return np.interp(grid, x, y)


def _load_all_results(results_root: str, algorithm_names: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, List[pd.DataFrame]], Dict[str, List[List[float]]]]:
    records: List[Dict[str, Any]] = []
    histories: Dict[str, List[pd.DataFrame]] = {}
    pareto_points: Dict[str, List[List[float]]] = {}
    expected_seeds = _read_expected_seeds(results_root)
    for algo in algorithm_names:
        algo_dir = os.path.join(results_root, algo)
        if not os.path.isdir(algo_dir):
            continue
        histories[algo] = []
        pareto_points[algo] = []
        json_files = sorted([os.path.join(algo_dir, x) for x in os.listdir(algo_dir) if x.endswith('.json')])
        for fp in json_files:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            seed = int(data.get('seed', -1))
            if expected_seeds is not None and seed not in expected_seeds:
                continue
            summary = copy.deepcopy(data.get('summary', {}))
            summary['algorithm'] = algo
            summary['seed'] = seed
            records.append(summary)
            histories[algo].append(pd.DataFrame(data.get('history', [])))
            for item in data.get('pareto', []):
                if item.get('fitness') is not None:
                    pareto_points[algo].append(item['fitness'])
    return records, histories, pareto_points


def _paired_cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if len(diff) <= 1:
        return 0.0
    std = float(np.std(diff, ddof=1))
    if std < 1e-12:
        return 0.0
    return float(np.mean(diff) / std)


def compute_statistical_tests(results_root: str, algorithm_names: Sequence[str], metrics: Sequence[str]) -> Dict[str, pd.DataFrame]:
    records, _, _ = _load_all_results(results_root, algorithm_names)
    final_df = pd.DataFrame(records)
    if final_df.empty:
        return {'pairwise': pd.DataFrame(), 'friedman': pd.DataFrame()}

    pair_rows = []
    friedman_rows = []
    for metric in metrics:
        pivot = final_df.pivot_table(index='seed', columns='algorithm', values=metric, aggfunc='mean')
        pivot = pivot[[c for c in algorithm_names if c in pivot.columns]].dropna(axis=0, how='any')
        if pivot.shape[0] >= 2 and pivot.shape[1] >= 3:
            stat, p = friedmanchisquare(*[pivot[c].to_numpy(dtype=float) for c in pivot.columns])
            friedman_rows.append({'metric': metric, 'num_seeds': int(pivot.shape[0]), 'num_algorithms': int(pivot.shape[1]), 'friedman_stat': float(stat), 'friedman_pvalue': float(p)})
        for a, b in combinations([c for c in algorithm_names if c in pivot.columns], 2):
            xy = pivot[[a, b]].dropna()
            if len(xy) < 2:
                continue
            x = xy[a].to_numpy(dtype=float)
            y = xy[b].to_numpy(dtype=float)
            t_p = float(ttest_rel(x, y).pvalue) if len(x) >= 2 else 1.0
            try:
                if np.allclose(x, y):
                    w_p, w_stat = 1.0, 0.0
                else:
                    w = wilcoxon(x, y, zero_method='wilcox', alternative='two-sided')
                    w_p, w_stat = float(w.pvalue), float(w.statistic)
            except Exception:
                w_p, w_stat = 1.0, 0.0
            pair_rows.append({'metric': metric, 'algo_a': a, 'algo_b': b, 'num_seeds': int(len(xy)), 'mean_a': float(np.mean(x)), 'mean_b': float(np.mean(y)), 'mean_diff_a_minus_b': float(np.mean(x - y)), 'paired_t_pvalue': t_p, 'wilcoxon_stat': w_stat, 'wilcoxon_pvalue': w_p, 'cohen_d_paired': _paired_cohen_d(x, y)})

    pair_df = pd.DataFrame(pair_rows)
    if not pair_df.empty:
        pair_df['paired_t_pvalue_holm'] = np.nan
        pair_df['wilcoxon_pvalue_holm'] = np.nan
        for metric in pair_df['metric'].unique():
            mask = pair_df['metric'] == metric
            pair_df.loc[mask, 'paired_t_pvalue_holm'] = multipletests(pair_df.loc[mask, 'paired_t_pvalue'], method='holm')[1]
            pair_df.loc[mask, 'wilcoxon_pvalue_holm'] = multipletests(pair_df.loc[mask, 'wilcoxon_pvalue'], method='holm')[1]
        pair_df.to_csv(os.path.join(results_root, 'pairwise_significance_tests.csv'), index=False)

    friedman_df = pd.DataFrame(friedman_rows)
    if not friedman_df.empty:
        friedman_df.to_csv(os.path.join(results_root, 'friedman_tests.csv'), index=False)
    return {'pairwise': pair_df, 'friedman': friedman_df}


def aggregate_results(results_root: str, algorithm_names: Sequence[str]) -> pd.DataFrame:
    os.makedirs(results_root, exist_ok=True)
    records, histories, pareto_points = _load_all_results(results_root, algorithm_names)
    final_df = pd.DataFrame(records)
    if not final_df.empty:
        final_df.sort_values(['algorithm', 'seed']).to_csv(os.path.join(results_root, 'final_metrics_by_seed.csv'), index=False)

    rows = []
    curve_metrics = ['hypervolume', 'best_profit', 'best_efficiency', 'best_fairness', 'pareto_size']
    overlay_curves: Dict[str, Dict[str, np.ndarray]] = {}

    for algo in algorithm_names:
        algo_dir = os.path.join(results_root, algo)
        if not os.path.isdir(algo_dir):
            continue
        algo_final = final_df[final_df['algorithm'] == algo].copy() if not final_df.empty else pd.DataFrame()
        if algo_final.empty:
            continue
        row = {'algorithm': algo}
        for col in ['hypervolume', 'pareto_size', 'best_profit', 'best_efficiency', 'best_fairness', 'mean_eval_time_sec', 'cum_eval_time_sec', 'real_evals']:
            row[f'{col}_mean'] = float(algo_final[col].mean())
            row[f'{col}_std'] = float(algo_final[col].std(ddof=0))
        rows.append(row)

        valid_histories = [h for h in histories.get(algo, []) if not h.empty and 'real_evals' in h.columns]
        max_x = max([int(h['real_evals'].max()) for h in valid_histories], default=0)
        if max_x > 0:
            grid = np.arange(1, max_x + 1)
            for metric in curve_metrics:
                curves = []
                for h in valid_histories:
                    if metric in h.columns:
                        curves.append(_interp_curve(h['real_evals'].to_numpy(dtype=float), h[metric].to_numpy(dtype=float), grid))
                if not curves:
                    continue
                arr = np.asarray(curves, dtype=float)
                mean = np.nanmean(arr, axis=0)
                std = np.nanstd(arr, axis=0)
                overlay_curves.setdefault(metric, {})[algo] = np.column_stack([grid, mean, std])
                plt.figure(figsize=(7, 4))
                plt.plot(grid, mean, label=algo)
                plt.fill_between(grid, mean - std, mean + std, alpha=0.2)
                plt.xlabel('Real evaluations')
                plt.ylabel(metric)
                plt.title(f'{algo} - {metric}')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(algo_dir, f'{metric}_curve.png'), dpi=160)
                plt.close()
                pd.DataFrame({'real_evals': grid, f'{metric}_mean': mean, f'{metric}_std': std}).to_csv(os.path.join(algo_dir, f'{metric}_curve.csv'), index=False)

        if pareto_points.get(algo):
            pts = np.asarray(pareto_points[algo], dtype=float)
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=12, alpha=0.7)
            ax.set_xlabel('Profit')
            ax.set_ylabel('Efficiency')
            ax.set_zlabel('Fairness')
            ax.set_title(f'{algo} Pareto points')
            plt.tight_layout()
            plt.savefig(os.path.join(algo_dir, 'pareto_front_3d.png'), dpi=160)
            plt.close()

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values('hypervolume_mean', ascending=False)
    summary_df.to_csv(os.path.join(results_root, 'comparison_summary.csv'), index=False)

    if not summary_df.empty:
        for metric in ['hypervolume_mean', 'best_profit_mean', 'best_efficiency_mean', 'best_fairness_mean', 'cum_eval_time_sec_mean']:
            plt.figure(figsize=(7, 4))
            plt.bar(summary_df['algorithm'], summary_df[metric])
            plt.ylabel(metric)
            plt.title(f'Algorithm comparison - {metric}')
            plt.xticks(rotation=20)
            plt.tight_layout()
            plt.savefig(os.path.join(results_root, f'{metric}_bar.png'), dpi=160)
            plt.close()

    for metric, algo_map in overlay_curves.items():
        plt.figure(figsize=(7, 4))
        for algo, arr in algo_map.items():
            grid, mean, std = arr[:, 0], arr[:, 1], arr[:, 2]
            plt.plot(grid, mean, label=algo)
            plt.fill_between(grid, mean - std, mean + std, alpha=0.15)
        plt.xlabel('Real evaluations')
        plt.ylabel(metric)
        plt.title(f'Comparison curve - {metric}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_root, f'comparison_{metric}_curve.png'), dpi=160)
        plt.close()

    compute_statistical_tests(results_root, algorithm_names, metrics=['hypervolume', 'best_profit', 'best_efficiency', 'best_fairness', 'cum_eval_time_sec'])
    return summary_df
