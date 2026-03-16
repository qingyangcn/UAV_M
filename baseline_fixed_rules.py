"""
Baseline: Fixed Rule Selection

Implements fixed rule baselines where rule_id is always the same value
(one of 0, 1, 2, 3, 4) for every decision.

Compatible with DecentralizedEventDrivenExecutor.

Rule descriptions (from UAV_ENVIRONMENT_11):
    0: Highest-priority / default rule
    1: Alternative rule 1
    2: EDF (Earliest Deadline First)
    3: Nearest pickup
    4: Slack per distance

Usage:
    # Run with fixed rule 3
    python baseline_fixed_rule.py --rule-id 3 --seed 42

    # Run all 5 fixed rules across multiple seeds
    python baseline_fixed_rule.py --all-rules --seeds 42,43,44

    # Run without MOPSO
    python baseline_fixed_rule.py --rule-id 0 --seed 42 --no-mopso
"""

import argparse
import math
import os
import sys
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from UAV_ENVIRONMENT_11 import ThreeObjectiveDroneDeliveryEnv
from U11_decentralized_execution import DecentralizedEventDrivenExecutor
from U11_ablation import _make_env, _compute_completion_stats

try:
    from U10_candidate_generator import MOPSOCandidateGenerator

    _HAS_MOPSO = True
except ImportError:
    _HAS_MOPSO = False


def make_fixed_rule_policy(rule_id: int) -> Callable[[dict], int]:
    """
    Factory function that returns a policy always selecting the given rule_id.

    Args:
        rule_id: Fixed rule identifier in {0, 1, 2, 3, 4}.

    Returns:
        A callable ``policy_fn(local_obs: dict) -> int`` compatible with
        ``DecentralizedEventDrivenExecutor``.
    """
    if rule_id not in range(5):
        raise ValueError(f"rule_id must be in {{0,1,2,3,4}}, got {rule_id}")

    def policy_fn(local_obs: dict) -> int:
        return rule_id

    policy_fn.__name__ = f"fixed_rule_{rule_id}_policy"
    return policy_fn


def run_episode(args, rule_id: int, seed: int) -> dict:
    """Run one episode with a fixed rule policy and return completion stats."""
    np.random.seed(seed)

    env = _make_env(args, order_cutoff_steps=0)

    if args.use_mopso and _HAS_MOPSO:
        candidate_generator = MOPSOCandidateGenerator(
            candidate_k=args.candidate_k,
            n_particles=30,
            n_iterations=10,
            max_orders=200,
            max_orders_per_drone=10,
            seed=seed,
        )
        env.set_candidate_generator(candidate_generator)

    fixed_policy = make_fixed_rule_policy(rule_id)

    executor = DecentralizedEventDrivenExecutor(
        env=env,
        policy_fn=fixed_policy,
        max_skip_steps=args.max_skip_steps,
        verbose=False,
    )

    executor.run_episode(max_steps=args.max_steps)

    stats = _compute_completion_stats(env)
    stats['seed'] = seed
    stats['rule_id'] = rule_id
    stats['policy'] = f'fixed_rule_{rule_id}'
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Baseline: Fixed Rule Selection"
    )

    parser.add_argument("--rule-id", type=int, default=2,
                        help="Fixed rule ID in {0,1,2,3,4} (required unless --all-rules)")
    parser.add_argument("--all-rules", action="store_true", default=True,
                        help="Run all 5 fixed rules (0..4) sequentially")
    parser.add_argument("--seed", type=int, default=21,
                        help="Random seed for a single episode run (default: 42)")
    parser.add_argument("--seeds", type=str, default='21',
                        help="Comma-separated seeds to run multiple episodes "
                             "(overrides --seed when provided)")
    parser.add_argument("--num-drones", type=int, default=20,
                        help="Number of drones (default: 20)")
    parser.add_argument("--obs-max-orders", type=int, default=400,
                        help="Maximum orders in observation (default: 400)")
    parser.add_argument("--top-k-merchants", type=int, default=100,
                        help="Top K merchants (default: 100)")
    parser.add_argument("--candidate-k", type=int, default=20,
                        help="Number of candidates per drone ")
    parser.add_argument("--enable-random-events", action="store_true", default=False,
                        help="Enable random events (default: False)")
    parser.add_argument("--max-skip-steps", type=int, default=1,
                        help="Max steps to skip when waiting for decisions (default: 1)")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum decision steps per episode (default: 500)")
    parser.add_argument("--use-mopso", action="store_true", default=True,
                        help="Use MOPSO candidate generator (default: True)")
    parser.add_argument("--no-mopso", dest="use_mopso", action="store_false",
                        help="Disable MOPSO candidate generator")

    args = parser.parse_args()

    if not args.all_rules and args.rule_id is None:
        parser.error("Provide --rule-id <0..4> or use --all-rules")

    rule_ids = list(range(5)) if args.all_rules else [args.rule_id]
    seeds = ([int(s.strip()) for s in args.seeds.split(',')]
             if args.seeds is not None else [args.seed])

    print("=" * 80)
    print("Baseline: Fixed Rule Selection")
    print(f"  rule_ids={rule_ids}  seeds={seeds}  candidate_k={args.candidate_k}  "
          f"use_mopso={args.use_mopso and _HAS_MOPSO}")
    print("=" * 80)

    all_stats = []
    for rule_id in rule_ids:
        rule_stats = []
        for seed in seeds:
            stats = run_episode(args, rule_id=rule_id, seed=seed)
            all_stats.append(stats)
            rule_stats.append(stats)

        if len(rule_stats) > 1:
            gc_values = [s['general_completion'] for s in rule_stats]
            sc_values = [s['serviceable_completion'] for s in rule_stats
                         if not math.isnan(s['serviceable_completion'])]
            print(f"  [rule_id={rule_id} mean]  "
                  f"mean_general_completion={float(np.mean(gc_values)):.4f}  "
                  f"mean_serviceable_completion="
                  f"{float(np.mean(sc_values)):.4f if sc_values else 'nan'}")

    print("=" * 80)
    return all_stats


if __name__ == "__main__":
    main()