"""
Baseline: Fixed Rule Selection

Implements a fixed-rule policy baseline that always selects the same rule_id
(0..4) at every decision point.

K (order_cutoff_steps) — Mode 1:
    The environment stops generating/accepting new orders K steps before the
    business-end step, but continues delivering already-accepted orders until
    the episode finishes.  K=0 means no early cutoff.

Compatible with DecentralizedEventDrivenExecutor.

Usage:
    # Run single episode with rule_id=3
    python baseline_fixed_rule.py --rule-id 3 --seed 42

    # Run with early order cutoff (K=19)
    python baseline_fixed_rule.py --rule-id 3 --seed 42 --order-cutoff-steps 19

    # Run multiple seeds
    python baseline_fixed_rule.py --rule-id 0 --seeds 42,43,44

    # Sweep all rule IDs
    for i in 0 1 2 3 4; do python baseline_fixed_rule.py --rule-id $i --seed 42; done
"""

import argparse
import os
import sys

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


VALID_RULE_IDS = range(5)


def make_fixed_rule_policy(rule_id: int):
    """Return a policy function that always selects the given rule_id."""
    def policy_fn(local_obs: dict) -> int:
        return rule_id
    return policy_fn


def run_episode(args, seed: int) -> dict:
    """Run one episode with a fixed-rule policy and return completion stats."""
    np.random.seed(seed)

    env = _make_env(args, order_cutoff_steps=args.order_cutoff_steps)

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

    executor = DecentralizedEventDrivenExecutor(
        env=env,
        policy_fn=make_fixed_rule_policy(args.rule_id),
        max_skip_steps=args.max_skip_steps,
        verbose=False,
    )

    executor.run_episode(max_steps=args.max_steps)

    stats = _compute_completion_stats(env)
    stats['seed'] = seed
    stats['policy'] = f'fixed_rule_{args.rule_id}'
    stats['rule_id'] = args.rule_id
    return stats


def print_stats(stats: dict):
    """Print completion statistics in a standard format."""
    print(f"  seed={stats['seed']}  rule_id={stats['rule_id']}  "
          f"generated_total={stats['generated_total']}  "
          f"completed_total={stats['completed_total']}  "
          f"general_completion={stats['general_completion']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Baseline: Fixed Rule Selection"
    )

    parser.add_argument("--rule-id", type=int, default=0,
                        help=f"Fixed rule ID to use at every decision point "
                             f"({min(VALID_RULE_IDS)}..{max(VALID_RULE_IDS)}, default: 0)")
    parser.add_argument("--seed", type=int, default=21,
                        help="Random seed for a single episode run (default: 21)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds to run multiple episodes "
                             "(overrides --seed when provided)")
    parser.add_argument("--num-drones", type=int, default=20,
                        help="Number of drones (default: 20)")
    parser.add_argument("--obs-max-orders", type=int, default=400,
                        help="Maximum orders in observation (default: 400)")
    parser.add_argument("--top-k-merchants", type=int, default=100,
                        help="Top K merchants (default: 100)")
    parser.add_argument("--candidate-k", type=int, default=20,
                        help="Number of candidates per drone")
    parser.add_argument("--order-cutoff-steps", type=int, default=0,
                        help="Stop generating/accepting orders this many steps before "
                             "business end (K, Mode 1; default: 0=disabled)")
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

    if args.rule_id not in VALID_RULE_IDS:
        parser.error(f"--rule-id must be in {min(VALID_RULE_IDS)}..{max(VALID_RULE_IDS)}, got {args.rule_id}")

    seeds = ([int(s.strip()) for s in args.seeds.split(',')]
             if args.seeds is not None else [args.seed])

    print("=" * 80)
    print(f"Baseline: Fixed Rule (rule_id={args.rule_id})")
    print(f"  Seeds: {seeds}  candidate_k={args.candidate_k}  "
          f"order_cutoff_steps={args.order_cutoff_steps}  "
          f"use_mopso={args.use_mopso and _HAS_MOPSO}")
    print("=" * 80)

    all_stats = []
    for seed in seeds:
        stats = run_episode(args, seed=seed)
        print_stats(stats)
        all_stats.append(stats)

    if len(all_stats) > 1:
        gc_values = [s['general_completion'] for s in all_stats]
        print("-" * 80)
        print(f"  mean_general_completion={float(np.mean(gc_values)):.4f}")

    print("=" * 80)
    return all_stats


if __name__ == "__main__":
    main()
