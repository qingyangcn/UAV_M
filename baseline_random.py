"""
Baseline: Random Candidate Selection

Implements a random policy baseline that selects a candidate order uniformly
at random from the drone's current candidate set at each decision point.

Compatible with DecentralizedEventDrivenExecutor (action_mode="candidate_index").

Usage:
    # Run single episode
    python baseline_random.py --seed 42

    # Run multiple seeds
    python baseline_random.py --seeds 42,43,44

    # Run without MOPSO candidate generation
    python baseline_random.py --seed 42 --no-mopso
"""

import argparse
import math
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


def policy_fn(local_obs: dict) -> int:
    """Random candidate policy: selects a valid candidate uniformly at random.

    Returns the index of a randomly chosen valid candidate from
    ``local_obs['candidates']``, or ``-1`` if no valid candidates exist.
    The validity of a candidate is indicated by feature index 0 (> 0.5 = valid).
    """
    candidates = local_obs.get('candidates')
    if candidates is None or candidates.size == 0:
        return -1
    valid_indices = np.where(candidates[:, 0] > 0.5)[0]
    if len(valid_indices) == 0:
        return -1
    return int(np.random.choice(valid_indices))


def run_episode(args, seed: int) -> dict:
    """Run one episode with random policy and return completion stats."""
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

    executor = DecentralizedEventDrivenExecutor(
        env=env,
        policy_fn=policy_fn,
        max_skip_steps=args.max_skip_steps,
        verbose=False,
        action_mode="candidate_index",
    )

    executor.run_episode(max_steps=args.max_steps)

    stats = _compute_completion_stats(env, sc_cutoff_steps=args.candidate_k)
    stats['seed'] = seed
    stats['policy'] = 'random_candidate'
    return stats


def print_stats(stats: dict):
    """Print completion statistics in a standard format."""
    sc = stats['serviceable_completion']
    sc_str = f"{sc:.4f}" if not math.isnan(sc) else "nan"
    print(f"  seed={stats['seed']}  "
          f"generated_total={stats['generated_total']}  "
          f"completed_total={stats['completed_total']}  "
          f"general_completion={stats['general_completion']:.4f}  "
          f"serviceable_generated={stats['serviceable_generated']}  "
          f"serviceable_completed={stats['serviceable_completed']}  "
          f"serviceable_completion={sc_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Baseline: Random Candidate Selection"
    )

    parser.add_argument("--seed", type=int, default=21,
                        help="Random seed for a single episode run (default: 42)")
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
    parser.add_argument("--cutoff-values", type=str, default="0",
                        help="Comma-separated K values to sweep in ablation mode (default: 0,6,12,18,24)")
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

    seeds = ([int(s.strip()) for s in args.seeds.split(',')]
             if args.seeds is not None else [args.seed])

    print("=" * 80)
    print("Baseline: Random Candidate Selection")
    print(f"  Seeds: {seeds}  candidate_k={args.candidate_k}  "
          f"use_mopso={args.use_mopso and _HAS_MOPSO}")
    print("=" * 80)

    all_stats = []
    for seed in seeds:
        stats = run_episode(args, seed=seed)
        print_stats(stats)
        all_stats.append(stats)

    if len(all_stats) > 1:
        gc_values = [s['general_completion'] for s in all_stats]
        sc_values = [s['serviceable_completion'] for s in all_stats
                     if not math.isnan(s['serviceable_completion'])]
        print("-" * 80)
        print(f"  mean_general_completion={float(np.mean(gc_values)):.4f}  "
              f"mean_serviceable_completion="
              f"{float(np.mean(sc_values)):.4f if sc_values else 'nan'}")

    print("=" * 80)
    return all_stats


if __name__ == "__main__":
    main()