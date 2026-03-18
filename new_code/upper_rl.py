from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from generate_simulator import PassengerSimulator
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from tl_samode import DEPolicyIndividual
from tl_samogp import dominates, pareto_front
from upper_experiment_utils import archive_metrics, build_experiment_evaluator



def _tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def _atanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -0.999999, 0.999999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


@dataclass
class PPOPolicyIndividual(DEPolicyIndividual):
    pass


@dataclass
class UpperRLConfig:
    iterations: int = 8
    episodes_per_iter: int = 2
    horizon: int = 3
    hidden_dim: int = 64
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    ppo_epochs: int = 6
    minibatch_size: int = 64
    init_log_std: float = -0.35
    reward_hv_coef: float = 2.0
    reward_scalar_coef: float = 0.6
    reward_nondom_bonus: float = 0.15
    random_seed: int = 7
    use_transfer: bool = False
    preference_conditioned: bool = True
    param_bounds: Tuple[Tuple[float, float], ...] = (
        (-2.0, 3.0), (-3.0, 3.0), (-6.0, 6.0), (-6.0, 6.0), (-3.0, 3.0),
        (0.0, 6.0), (-4.0, 4.0), (-6.0, 6.0), (-6.0, 6.0), (-4.0, 8.0),
    )


class RunningNormalizer:
    def __init__(self, dim: int):
        self.scale = np.ones(dim, dtype=float)

    def update(self, x: np.ndarray) -> None:
        self.scale = np.maximum(self.scale, np.abs(np.asarray(x, dtype=float)))

    def norm(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(np.asarray(x, dtype=float) / np.maximum(self.scale, 1e-6))


class UpperSearchEnv:
    def __init__(self, evaluator, config: UpperRLConfig):
        self.evaluator = evaluator
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.bounds = np.asarray(config.param_bounds, dtype=float)
        self.num_params = self.bounds.shape[0]
        self.obj_norm = RunningNormalizer(3)
        self.metric_norm = RunningNormalizer(4)
        self.episode_archive: List[PPOPolicyIndividual] = []
        self.global_archive: List[PPOPolicyIndividual] = []
        self.preference = np.ones(3, dtype=float) / 3.0
        self.step_idx = 0
        self.last_fitness = np.zeros(3, dtype=float)
        self.last_action = np.zeros(self.num_params, dtype=float)
        self.state_dim = 14 + (3 if self.config.preference_conditioned else 0)

    def _sample_preference(self) -> np.ndarray:
        if not self.config.preference_conditioned:
            return np.ones(3, dtype=float) / 3.0
        return self.rng.dirichlet(np.ones(3, dtype=float))

    def _action_to_params(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + 0.5 * (action + 1.0) * (hi - lo)

    def _make_individual(self, action: np.ndarray) -> PPOPolicyIndividual:
        return PPOPolicyIndividual(params=self._action_to_params(action))

    def _current_metrics(self) -> Dict[str, float]:
        if not self.episode_archive:
            return {'hypervolume': 0.0, 'pareto_size': 0.0, 'best_profit': 0.0, 'best_efficiency': 0.0, 'best_fairness': 0.0}
        return archive_metrics(self.episode_archive, eval_times=None, seed=self.config.random_seed + self.step_idx)

    def _state(self) -> np.ndarray:
        m = self._current_metrics()
        metric_vec = np.asarray([m['hypervolume'], m['best_profit'], m['best_efficiency'], m['best_fairness']], dtype=float)
        self.metric_norm.update(metric_vec)
        prog = np.asarray([
            self.step_idx / max(1, self.config.horizon),
            1.0 - self.step_idx / max(1, self.config.horizon),
            np.tanh(len(self.episode_archive) / max(1, self.config.horizon)),
        ], dtype=float)
        last_action_stats = np.asarray([
            float(np.mean(self.last_action)),
            float(np.std(self.last_action)),
            float(np.max(np.abs(self.last_action))),
            np.tanh(float(np.linalg.norm(self.last_action)) / np.sqrt(max(1, self.num_params))),
        ], dtype=float)
        parts = [prog, self.metric_norm.norm(metric_vec), self.obj_norm.norm(self.last_fitness), last_action_stats]
        if self.config.preference_conditioned:
            parts.append(self.preference)
        return np.concatenate(parts).astype(np.float32)

    def reset(self) -> np.ndarray:
        self.episode_archive = []
        self.preference = self._sample_preference()
        self.step_idx = 0
        self.last_fitness = np.zeros(3, dtype=float)
        self.last_action = np.zeros(self.num_params, dtype=float)
        return self._state()

    def step(self, action: np.ndarray):
        prev_hv = self._current_metrics()['hypervolume']
        ind = self._make_individual(action)
        fitness = np.asarray(self.evaluator.evaluate(ind), dtype=float)
        self.obj_norm.update(fitness)
        self.episode_archive.append(ind.clone())
        self.global_archive.append(ind.clone())
        cur_metrics = self._current_metrics()
        hv_gain = cur_metrics['hypervolume'] - prev_hv
        scalar_reward = float(np.dot(self.preference, self.obj_norm.norm(fitness)))
        is_nondom = not any(dominates(np.asarray(other.fitness, dtype=float), fitness) for other in self.episode_archive[:-1])
        reward = self.config.reward_scalar_coef * scalar_reward + self.config.reward_hv_coef * hv_gain + self.config.reward_nondom_bonus * float(is_nondom)
        self.last_fitness = fitness.copy()
        self.last_action = np.asarray(action, dtype=float).copy()
        self.step_idx += 1
        done = self.step_idx >= self.config.horizon
        info = {'fitness': fitness.copy(), 'metrics': copy.deepcopy(cur_metrics), 'individual': ind.clone()}
        return self._state(), float(reward), bool(done), info


class ContinuousActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, init_log_std: float = -0.35):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def _dist(self, state: torch.Tensor):
        mean = self.actor(state)
        std = torch.exp(torch.clamp(self.log_std, -5.0, 2.0))
        return Normal(mean, std)

    def act(self, state: torch.Tensor):
        dist = self._dist(state)
        pre_tanh = dist.rsample()
        action = _tanh(pre_tanh)
        logprob = dist.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action.detach(), logprob.sum(dim=-1).detach(), self.critic(state).squeeze(-1).detach()

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        dist = self._dist(state)
        pre_tanh = _atanh(action)
        logprob = dist.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + 1e-6)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(state).squeeze(-1)
        return logprob.sum(dim=-1), entropy, value


@dataclass
class RolloutBatch:
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)


class UpperLevelRL:
    def __init__(self, evaluator, config: Optional[UpperRLConfig] = None):
        self.config = config or UpperRLConfig()
        self.evaluator = evaluator
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        self.env = UpperSearchEnv(evaluator=evaluator, config=self.config)
        self.policy = ContinuousActorCritic(self.env.state_dim, len(self.config.param_bounds), self.config.hidden_dim, self.config.init_log_std).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.config.actor_lr},
            {'params': self.policy.critic.parameters(), 'lr': self.config.critic_lr},
            {'params': [self.policy.log_std], 'lr': self.config.actor_lr},
        ])
        self.archive: List[PPOPolicyIndividual] = []
        self.history: List[Dict[str, Any]] = []

    def _collect_rollouts(self) -> RolloutBatch:
        batch = RolloutBatch()
        for _ in range(self.config.episodes_per_iter):
            state = self.env.reset()
            for _step in range(self.config.horizon):
                s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action_t, logprob_t, value_t = self.policy.act(s)
                action = action_t.squeeze(0).cpu().numpy()
                next_state, reward, done, info = self.env.step(action)
                batch.states.append(state.copy())
                batch.actions.append(action.copy())
                batch.logprobs.append(float(logprob_t.item()))
                batch.rewards.append(float(reward))
                batch.dones.append(float(done))
                batch.values.append(float(value_t.item()))
                self.archive.append(info['individual'].clone())
                state = next_state
                if done:
                    break
        return batch

    def _advantages(self, batch: RolloutBatch):
        rewards = np.asarray(batch.rewards, dtype=float)
        dones = np.asarray(batch.dones, dtype=float)
        values = np.asarray(batch.values + [0.0], dtype=float)
        adv = np.zeros_like(rewards, dtype=float)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * values[t + 1] * next_nonterminal - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * next_nonterminal * gae
            adv[t] = gae
        returns = adv + values[:-1]
        if len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return torch.tensor(adv, dtype=torch.float32, device=self.device), torch.tensor(returns, dtype=torch.float32, device=self.device)

    def _update(self, batch: RolloutBatch) -> None:
        if not batch.states:
            return
        states = torch.tensor(np.asarray(batch.states, dtype=np.float32), device=self.device)
        actions = torch.tensor(np.asarray(batch.actions, dtype=np.float32), device=self.device)
        old_logprobs = torch.tensor(np.asarray(batch.logprobs, dtype=np.float32), device=self.device)
        adv, returns = self._advantages(batch)
        n = states.shape[0]
        mbs = min(self.config.minibatch_size, n)
        idxs = np.arange(n)
        for _ in range(self.config.ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, mbs):
                idx = idxs[start:start + mbs]
                logprob, entropy, values = self.policy.evaluate_actions(states[idx], actions[idx])
                ratios = torch.exp(logprob - old_logprobs[idx])
                surr1 = ratios * adv[idx]
                surr2 = torch.clamp(ratios, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * adv[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (returns[idx] - values).pow(2).mean()
                loss = actor_loss + self.config.value_coef * critic_loss - self.config.entropy_coef * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()

    def run(self) -> List[PPOPolicyIndividual]:
        for iteration in range(1, self.config.iterations + 1):
            batch = self._collect_rollouts()
            self._update(batch)
            metrics = archive_metrics(self.archive, getattr(self.evaluator, 'eval_times', []), seed=self.config.random_seed + iteration)
            metrics['generation'] = iteration
            self.history.append(metrics)
        return pareto_front(self.archive)


def run_minimal_upper_rl(simulator_path: Optional[str] = None, iterations: int = 4, episodes_per_iter: int = 2, horizon: int = 3, eval_episodes: int = 1, use_transfer: bool = False, seed: int = 7) -> Dict[str, Any]:
    evaluator = build_experiment_evaluator(simulator_path=simulator_path, eval_episodes=eval_episodes, use_transfer=use_transfer, ref_state_size=24, seed=seed)
    algo = UpperLevelRL(evaluator=evaluator, config=UpperRLConfig(iterations=iterations, episodes_per_iter=episodes_per_iter, horizon=horizon, random_seed=seed, use_transfer=use_transfer))
    pareto = algo.run()
    summary = archive_metrics(algo.archive, evaluator.eval_times, seed=seed)
    return {
        'algorithm': 'Upper-PPO',
        'config': {'iterations': iterations, 'episodes_per_iter': episodes_per_iter, 'horizon': horizon, 'eval_episodes': eval_episodes, 'use_transfer': use_transfer, 'seed': seed, 'simulator_path': simulator_path},
        'history': algo.history,
        'summary': summary,
        'pareto': [{'fitness': None if ind.fitness is None else np.asarray(ind.fitness, dtype=float).round(6).tolist(), 'formulas': ind.formulas()} for ind in pareto],
    }


def _build_argparser():
    parser = argparse.ArgumentParser(description='Run the strict upper-level PPO baseline.')
    parser.add_argument('--simulator_path', type=str, default=None)
    parser.add_argument('--iterations', type=int, default=4)
    parser.add_argument('--episodes_per_iter', type=int, default=2)
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--eval_episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--enable_transfer', action='store_true')
    parser.add_argument('--save_json', type=str, default=None)
    return parser


if __name__ == '__main__':
    args = _build_argparser().parse_args()
    result = run_minimal_upper_rl(simulator_path=args.simulator_path, iterations=args.iterations, episodes_per_iter=args.episodes_per_iter, horizon=args.horizon, eval_episodes=args.eval_episodes, use_transfer=args.enable_transfer, seed=args.seed)
    print(f"[Upper-PPO] summary={result['summary']}")
    if args.save_json:
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[Upper-PPO] saved to {args.save_json}")
