from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from tl_samogp import TLConfig, TLSAMOGP
from tl_samode import TLDEConfig, TLSAMODE
from upper_rl import UpperRLConfig, UpperLevelRL
from upper_policy_artifacts import (
    evaluate_saved_pareto_on_testset,
    save_policy_artifact,
    save_upper_rl_checkpoint,
    serialize_policies,
)
from upper_experiment_utils import aggregate_results, archive_metrics, build_experiment_evaluator, save_run_json


def _artifact_paths(results_root: str, algorithm_name: str, seed: int) -> Dict[str, str]:
    return {
        'pareto_artifact_rel': os.path.join('artifacts', f'seed_{seed}_pareto.json'),
        'pareto_artifact_abs': os.path.join(results_root, algorithm_name, 'artifacts', f'seed_{seed}_pareto.json'),
        'checkpoint_rel': os.path.join('checkpoints', f'seed_{seed}_upper_rl.pt'),
        'checkpoint_abs': os.path.join(results_root, algorithm_name, 'checkpoints', f'seed_{seed}_upper_rl.pt'),
    }


def run_tl_samogp(simulator_path: str | None, seed: int, args, algorithm_name: str) -> Dict[str, Any]:
    evaluator = build_experiment_evaluator(
        simulator_path=simulator_path,
        eval_episodes=args.eval_episodes,
        use_transfer=not args.disable_transfer_gp,
        ref_state_size=args.ref_state_size,
        seed=seed,
        similarity_threshold=args.transfer_similarity_threshold,
    )
    algo = TLSAMOGP(
        evaluator=evaluator,
        config=TLConfig(
            population_size=args.population,
            generations=args.generations,
            real_eval_quota=args.real_eval_quota,
            crossover_rate=args.gp_crossover,
            mutation_rate=args.gp_mutation,
            tournament_size=args.gp_tournament,
            max_tree_depth=args.gp_max_depth,
            ref_state_size=args.ref_state_size,
            random_seed=seed,
            use_surrogate=not args.disable_surrogate_gp,
            use_transfer=not args.disable_transfer_gp,
            transfer_similarity_threshold=args.transfer_similarity_threshold,
        ),
    )
    pareto = algo.run()
    summary = archive_metrics(algo.archive, evaluator.eval_times, seed=seed)
    serialized_pareto = serialize_policies(pareto)
    paths = _artifact_paths(args.results_root, algorithm_name, seed)
    save_policy_artifact(algorithm_name, seed, vars(args), serialized_pareto, paths['pareto_artifact_abs'])
    return {
        'algorithm': algorithm_name,
        'seed': seed,
        'config': vars(args),
        'history': algo.history,
        'summary': summary,
        'pareto': serialized_pareto,
        'artifacts': {
            'pareto_artifact': paths['pareto_artifact_rel'],
        },
    }


def run_tl_samode(simulator_path: str | None, seed: int, args, algorithm_name: str) -> Dict[str, Any]:
    evaluator = build_experiment_evaluator(
        simulator_path=simulator_path,
        eval_episodes=args.eval_episodes,
        use_transfer=not args.disable_transfer_de,
        ref_state_size=args.ref_state_size,
        seed=seed,
        similarity_threshold=args.transfer_similarity_threshold,
    )
    algo = TLSAMODE(
        evaluator=evaluator,
        config=TLDEConfig(
            population_size=args.population,
            generations=args.generations,
            real_eval_quota=args.real_eval_quota,
            differential_weight=args.de_f,
            crossover_rate=args.de_cr,
            random_seed=seed,
            use_surrogate=not args.disable_surrogate_de,
            use_transfer=not args.disable_transfer_de,
        ),
    )
    pareto = algo.run()
    summary = archive_metrics(algo.archive, evaluator.eval_times, seed=seed)
    serialized_pareto = serialize_policies(pareto)
    paths = _artifact_paths(args.results_root, algorithm_name, seed)
    save_policy_artifact(algorithm_name, seed, vars(args), serialized_pareto, paths['pareto_artifact_abs'])
    return {
        'algorithm': algorithm_name,
        'seed': seed,
        'config': vars(args),
        'history': algo.history,
        'summary': summary,
        'pareto': serialized_pareto,
        'artifacts': {
            'pareto_artifact': paths['pareto_artifact_rel'],
        },
    }


def run_upper_ppo(simulator_path: str | None, seed: int, args, algorithm_name: str) -> Dict[str, Any]:
    evaluator = build_experiment_evaluator(
        simulator_path=simulator_path,
        eval_episodes=args.eval_episodes,
        use_transfer=args.enable_transfer_rl,
        ref_state_size=args.ref_state_size,
        seed=seed,
        similarity_threshold=args.transfer_similarity_threshold,
    )
    algo = UpperLevelRL(
        evaluator=evaluator,
        config=UpperRLConfig(
            iterations=args.ppo_iterations,
            episodes_per_iter=args.ppo_episodes_per_iter,
            horizon=args.ppo_horizon,
            hidden_dim=args.ppo_hidden_dim,
            actor_lr=args.ppo_actor_lr,
            critic_lr=args.ppo_critic_lr,
            gamma=args.ppo_gamma,
            gae_lambda=args.ppo_gae_lambda,
            clip_eps=args.ppo_clip_eps,
            entropy_coef=args.ppo_entropy_coef,
            value_coef=args.ppo_value_coef,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.ppo_minibatch_size,
            random_seed=seed,
            use_transfer=args.enable_transfer_rl,
            preference_conditioned=not args.disable_preference_conditioned_ppo,
        ),
    )
    pareto = algo.run()
    summary = archive_metrics(algo.archive, evaluator.eval_times, seed=seed)
    serialized_pareto = serialize_policies(pareto)
    paths = _artifact_paths(args.results_root, algorithm_name, seed)
    save_policy_artifact(algorithm_name, seed, vars(args), serialized_pareto, paths['pareto_artifact_abs'])
    save_upper_rl_checkpoint(algo.policy, algo.optimizer, algorithm_name, seed, vars(args), paths['checkpoint_abs'])
    return {
        'algorithm': algorithm_name,
        'seed': seed,
        'config': vars(args),
        'history': algo.history,
        'summary': summary,
        'pareto': serialized_pareto,
        'artifacts': {
            'pareto_artifact': paths['pareto_artifact_rel'],
            'upper_rl_checkpoint': paths['checkpoint_rel'],
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run and compare TL-SAMOGP / TL-SAMODE / Upper-PPO.')
    parser.add_argument(
        '--simulator_path',
        type=str,
        default='generator/simulator_driver_nums=400_hex_scaling=0.017031373249357308_weekday.pkl',
        help='If omitted, uses the dummy lower-level evaluator.',
    )
    parser.add_argument('--results_root', type=str, default='upper_exp_results')
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 13, 23])
    parser.add_argument('--algorithms', type=str, nargs='+', default=['TL-SAMOGP', 'TL-SAMODE', 'Upper-PPO'])
    parser.add_argument('--eval_episodes', type=int, default=1)
    parser.add_argument('--test_simulator_path', type=str, default=None, help='Optional held-out simulator used for unified post-training reevaluation.')
    parser.add_argument('--test_eval_episodes', type=int, default=None, help='Evaluation episodes per saved upper-level policy on the held-out test simulator.')
    parser.add_argument('--test_results_root', type=str, default=None, help='Where to store held-out test-set summaries. Defaults to <results_root>/test_eval.')
    parser.add_argument('--test_max_policies', type=int, default=0, help='Optional cap on the number of saved Pareto policies reevaluated per run; 0 means all.')
    parser.add_argument('--test_only', action='store_true', help='Skip training and only reevaluate previously saved upper-level policies on the held-out test simulator.')
    parser.add_argument('--ref_state_size', type=int, default=50)
    parser.add_argument('--max_workers', type=int, default=2, help='Multi-thread workers. Single-GPU runs usually use 2.')
    parser.add_argument('--transfer_similarity_threshold', type=float, default=1.0, help='Phenotype-distance threshold for warm-start transfer.')

    parser.add_argument('--population', type=int, default=40)
    parser.add_argument('--generations', type=int, default=40)
    parser.add_argument('--real_eval_quota', type=int, default=5)

    parser.add_argument('--gp_crossover', type=float, default=0.9)
    parser.add_argument('--gp_mutation', type=float, default=0.1)
    parser.add_argument('--gp_tournament', type=int, default=2)
    parser.add_argument('--gp_max_depth', type=int, default=4)
    parser.add_argument('--disable_surrogate_gp', action='store_true')
    parser.add_argument('--disable_transfer_gp', action='store_true')

    parser.add_argument('--de_f', type=float, default=0.7)
    parser.add_argument('--de_cr', type=float, default=0.9)
    parser.add_argument('--disable_surrogate_de', action='store_true')
    parser.add_argument('--disable_transfer_de', action='store_true')

    parser.add_argument('--ppo_iterations', type=int, default=12)
    parser.add_argument('--ppo_episodes_per_iter', type=int, default=4)
    parser.add_argument('--ppo_horizon', type=int, default=5)
    parser.add_argument('--ppo_hidden_dim', type=int, default=64)
    parser.add_argument('--ppo_actor_lr', type=float, default=3e-4)
    parser.add_argument('--ppo_critic_lr', type=float, default=1e-3)
    parser.add_argument('--ppo_gamma', type=float, default=0.99)
    parser.add_argument('--ppo_gae_lambda', type=float, default=0.95)
    parser.add_argument('--ppo_clip_eps', type=float, default=0.2)
    parser.add_argument('--ppo_entropy_coef', type=float, default=0.01)
    parser.add_argument('--ppo_value_coef', type=float, default=0.5)
    parser.add_argument('--ppo_epochs', type=int, default=6)
    parser.add_argument('--ppo_minibatch_size', type=int, default=64)
    parser.add_argument('--enable_transfer_rl', action='store_true')
    parser.add_argument('--disable_preference_conditioned_ppo', action='store_true')
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()
    os.makedirs(args.results_root, exist_ok=True)

    runners = {
        'TL-SAMOGP': run_tl_samogp,
        'TL-SAMODE': run_tl_samode,
        'Upper-PPO': run_upper_ppo,
        'Upper-RL': run_upper_ppo,
        'MARL-PPO': run_upper_ppo,
    }

    tasks: List[Tuple[str, int]] = []
    for algo in args.algorithms:
        if algo not in runners:
            raise ValueError(f'Unknown algorithm: {algo}')
        os.makedirs(os.path.join(args.results_root, algo), exist_ok=True)
        for seed in args.seeds:
            tasks.append((algo, seed))

    if args.test_only and not args.test_simulator_path:
        raise ValueError('--test_only requires --test_simulator_path.')

    if not args.test_only:
        with open(os.path.join(args.results_root, 'run_config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)

        print(f'[Runner] total_tasks={len(tasks)}, max_workers={args.max_workers}')
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            future_map = {
                executor.submit(runners[algo], args.simulator_path, seed, args, algo): (algo, seed)
                for algo, seed in tasks
            }
            for fut in as_completed(future_map):
                algo, seed = future_map[fut]
                result = fut.result()
                save_path = os.path.join(args.results_root, algo, f'seed_{seed}.json')
                save_run_json(result, save_path)
                print(f'[{algo}] seed={seed} summary={result["summary"]}')

        summary_df = aggregate_results(args.results_root, args.algorithms)
        print('\n=== Comparison summary ===')
        if summary_df.empty:
            print('No results found.')
        else:
            print(summary_df.to_string(index=False))
            print(f"\nSaved summary to: {os.path.join(args.results_root, 'comparison_summary.csv')}")

    if args.test_simulator_path:
        test_eval_episodes = args.eval_episodes if args.test_eval_episodes is None else args.test_eval_episodes
        test_results_root = args.test_results_root or os.path.join(args.results_root, 'test_eval')
        test_summary_df = evaluate_saved_pareto_on_testset(
            results_root=args.results_root,
            algorithm_names=args.algorithms,
            seeds=args.seeds,
            test_simulator_path=args.test_simulator_path,
            test_eval_episodes=test_eval_episodes,
            ref_state_size=args.ref_state_size,
            test_root=test_results_root,
            max_policies=args.test_max_policies,
        )
        print('\n=== Held-out test summary ===')
        if test_summary_df.empty:
            print('No test results found.')
        else:
            print(test_summary_df.to_string(index=False))
            print(f"\nSaved test summary to: {os.path.join(test_results_root, 'comparison_summary.csv')}")
