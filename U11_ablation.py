"""
U11 Ablation & Sanity Check for Decentralized Event-Driven Execution

K (order_cutoff_steps) definition — **Mode 1 only**:
    The environment stops generating/accepting new orders K steps before the
    business-end step, but continues delivering already-accepted orders until
    the episode finishes.  K=0 means no early cutoff.

Usage:
    # Test with random policy (single episode)
    python U11_ablation.py

    # Test with trained policy
    python U11_ablation.py --model-path ./models/u10/ppo_u10_final.zip

    # Quick test (fewer steps)
    python U11_ablation.py --max-steps 100

    # Ablation: scan environment order_cutoff_steps (K) across multiple seeds
    python U11_ablation.py --ablation-cutoff --cutoff-values 0,6,12,18,24 --seeds 42,43 --csv-out out.csv
"""

import argparse
import csv
import math
import os
import sys
from collections import Counter

import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from UAV_ENVIRONMENT_11 import ThreeObjectiveDroneDeliveryEnv
from U11_decentralized_execution import DecentralizedEventDrivenExecutor

try:
    from U10_candidate_generator import MOPSOCandidateGenerator

    _HAS_MOPSO = True
except ImportError:
    _HAS_MOPSO = False


def random_policy(local_obs: dict) -> int:
    """Simple random policy for testing."""
    return np.random.randint(0, 5)


def load_trained_policy(model_path: str, vecnormalize_path: str = None):
    """
    Load a trained PPO policy.

    Supports the current Box/ndarray observation architecture: ``local_obs`` is a
    flat ``np.ndarray`` and ``obs_rms`` (from VecNormalize) is a single
    ``RunningMeanStd`` instance rather than a per-key dict.

    Args:
        model_path: Path to trained model (.zip file)
        vecnormalize_path: Path to VecNormalize stats (.pkl file)

    Returns:
        Policy function that takes local_obs (np.ndarray) and returns rule_id
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        raise RuntimeError("Please install stable-baselines3: pip install stable-baselines3")

    # Load model
    model = PPO.load(model_path)

    # Load VecNormalize stats if provided and apply during inference
    obs_rms = None
    clip_obs = 10.0
    epsilon = 1e-8

    if vecnormalize_path and os.path.exists(vecnormalize_path):
        try:
            try:
                import cloudpickle as _pkl_mod
            except ImportError:
                import pickle as _pkl_mod
            with open(vecnormalize_path, 'rb') as f:
                _vn = _pkl_mod.load(f)
            _obs_rms = getattr(_vn, 'obs_rms', None)
            _clip_obs = float(getattr(_vn, 'clip_obs', 10.0))
            _norm_obs = bool(getattr(_vn, 'norm_obs', True))
            _epsilon = float(getattr(_vn, 'epsilon', 1e-8))
            if _norm_obs and _obs_rms is not None:
                obs_rms = _obs_rms
                clip_obs = _clip_obs
                epsilon = _epsilon
                print(f"VecNormalize stats loaded from: {vecnormalize_path}")
                print(f"  norm_obs enabled, clip_obs={clip_obs}")
                _mean_avg = float(np.array(obs_rms.mean).mean())
                _std_avg = float(np.sqrt(np.array(obs_rms.var).mean()))
                print(f"  obs_rms: mean_avg={_mean_avg:.4f}, std_avg={_std_avg:.4f}")
            else:
                print(f"VecNormalize loaded from: {vecnormalize_path} "
                      f"(norm_obs={_norm_obs}, no normalization applied)")
        except Exception as e:
            print(f"Warning: Failed to load VecNormalize stats from {vecnormalize_path}: {e}")
            print("  Proceeding without observation normalization.")
    elif vecnormalize_path:
        print(f"Warning: VecNormalize stats file not found: {vecnormalize_path}")
        print("  Proceeding without observation normalization.")

    def policy_fn(local_obs: np.ndarray) -> int:
        """Wrapper function for trained policy (Box/ndarray observation space)."""
        if obs_rms is not None:
            # Apply VecNormalize observation normalization to match training preprocessing.
            # Replicates VecNormalize._normalize_obs: clip((obs - mean) / sqrt(var + eps), ±clip_obs)
            obs_to_predict = np.clip(
                (local_obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon),
                -clip_obs, clip_obs,
            ).astype(np.float32)
        else:
            obs_to_predict = local_obs
        action, _ = model.predict(obs_to_predict, deterministic=True)
        return int(action)

    return policy_fn


def _make_env(args, order_cutoff_steps: int = 0) -> ThreeObjectiveDroneDeliveryEnv:
    """Create environment with the given configuration."""
    return ThreeObjectiveDroneDeliveryEnv(
        grid_size=16,
        num_drones=args.num_drones,
        max_orders=args.obs_max_orders,
        num_bases=2,
        steps_per_hour=12,
        drone_max_capacity=10,
        top_k_merchants=args.top_k_merchants,
        reward_output_mode="scalar",
        enable_random_events=args.enable_random_events,
        debug_state_warnings=False,
        fixed_objective_weights=(0.5, 0.3, 0.2),
        num_candidates=args.candidate_k,
        rule_count=5,
        enable_diagnostics=False,
        energy_e0=0.1,
        energy_alpha=0.5,
        battery_return_threshold=10.0,
        multi_objective_mode="fixed",
        candidate_update_interval=8,
        candidate_fallback_enabled=False,
        order_cutoff_steps=order_cutoff_steps,
    )


def _compute_completion_stats(env: ThreeObjectiveDroneDeliveryEnv) -> dict:
    """Compute general completion statistics from a finished episode.

    Args:
        env: The finished environment.

    Returns:
        Dict with keys: generated_total, completed_total, general_completion.
    """
    generated_total = env.daily_stats['orders_generated']
    completed_total = env.daily_stats['orders_completed']
    general_completion = completed_total / generated_total if generated_total > 0 else 0.0

    return {
        'generated_total': generated_total,
        'completed_total': completed_total,
        'general_completion': general_completion,
    }


def run_single_episode(args, order_cutoff_steps: int, seed: int) -> dict:
    """Run one episode and return completion stats.

    The environment is created with *order_cutoff_steps* (K) so that order
    generation stops K steps before business-end (Mode 1).  Delivery of
    already-accepted orders continues until the episode finishes.
    """
    env = _make_env(args, order_cutoff_steps=order_cutoff_steps)

    if _HAS_MOPSO:
        candidate_generator = MOPSOCandidateGenerator(
            candidate_k=args.candidate_k,
            n_particles=30,
            n_iterations=10,
            max_orders=200,
            max_orders_per_drone=10,
            seed=seed,
        )
        env.set_candidate_generator(candidate_generator)

    policy_fn = random_policy

    executor = DecentralizedEventDrivenExecutor(
        env=env,
        policy_fn=policy_fn,
        max_skip_steps=args.max_skip_steps,
        verbose=False,
        track_action_stats=getattr(args, 'track_action_stats', False),
    )

    executor.run_episode(max_steps=args.max_steps, seed=seed)

    stats = _compute_completion_stats(env)
    stats['order_cutoff_steps'] = order_cutoff_steps
    stats['seed'] = seed

    if getattr(args, 'track_action_stats', False):
        action_stats = executor.get_action_stats()
        pct = {k: f'{v:.1f}%' for k, v in action_stats.to_percent().items()}
        print(f"  [seed={seed} K={order_cutoff_steps}] "
              f"rule_counts: {dict(action_stats.rule_counts)}  "
              f"rule_percent: {pct}  "
              f"n_decisions={action_stats.n_decisions}  "
              f"n_invalid_rule={action_stats.n_invalid_rule}  "
              f"n_empty_candidates={action_stats.n_empty_candidates}")

    return stats


def run_ablation_cutoff(args):
    """Run K-sweep ablation and write CSV."""
    cutoff_values = [int(v.strip()) for v in args.cutoff_values.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')]

    print("=" * 80)
    print("Ablation: Order Cutoff Steps (K) Sweep")
    print(f"  K values: {cutoff_values}")
    print(f"  Seeds:    {seeds}")
    print("=" * 80)

    rows = []
    fieldnames = [
        'order_cutoff_steps', 'seed',
        'generated_total', 'completed_total', 'general_completion',
    ]

    for K in cutoff_values:
        for seed in seeds:
            print(f"  Running K={K}, seed={seed} ...", end=' ', flush=True)
            row = run_single_episode(args, order_cutoff_steps=K, seed=seed)
            rows.append(row)
            print(f"GC={row['general_completion']:.4f}  "
                  f"generated={row['generated_total']}  completed={row['completed_total']}")

    if args.csv_out:
        with open(args.csv_out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row[k] for k in fieldnames})
        print(f"\nCSV written to: {args.csv_out}")

    # Aggregate per K
    from collections import defaultdict
    agg = defaultdict(lambda: {'gc': []})
    for row in rows:
        K = row['order_cutoff_steps']
        agg[K]['gc'].append(row['general_completion'])

    print("\n" + "=" * 80)
    print("Per-K aggregated means:")
    print(f"  {'K':>6}  {'mean_GC':>10}")
    summary = {}
    for K in cutoff_values:
        gc_list = agg[K]['gc']
        mean_gc = float(np.mean(gc_list)) if gc_list else float('nan')
        summary[K] = {'mean_gc': mean_gc}
        gc_str = f"{mean_gc:.4f}" if not math.isnan(mean_gc) else "nan"
        print(f"  {K:>6}  {gc_str:>10}")

    # Determine best K by GC
    valid_gc = {K: v['mean_gc'] for K, v in summary.items() if not math.isnan(v['mean_gc'])}

    if not valid_gc:
        print("\nInsufficient data to determine best K.")
        return

    K_g, best_gc_val = min(valid_gc.items(), key=lambda kv: (-kv[1], kv[0]))

    print("\n" + "=" * 80)
    print(f"Recommended K (best mean_GC={best_gc_val:.4f}): K={K_g}")

    print("=" * 80)


def run_sanity_check(args):
    """Run sanity check with specified configuration."""
    print("=" * 80)
    print("U11 Decentralized Execution Sanity Check")
    print("=" * 80)

    # Create environment
    print("\nCreating environment...")
    env = _make_env(args, order_cutoff_steps=args.order_cutoff_steps)

    # Create MOPSO candidate generator
    if _HAS_MOPSO:
        print("Creating MOPSO candidate generator...")
        candidate_generator = MOPSOCandidateGenerator(
            candidate_k=args.candidate_k,
            n_particles=30,
            n_iterations=10,
            max_orders=200,
            max_orders_per_drone=10,
            seed=args.seed,
        )
        env.set_candidate_generator(candidate_generator)

    # Choose policy
    if args.model_path:
        print(f"\nLoading trained policy from: {args.model_path}")
        policy_fn = load_trained_policy(args.model_path, args.vecnormalize_path)
        policy_name = "Trained Policy"
    else:
        print("\nUsing random policy for testing")
        policy_fn = random_policy
        policy_name = "Random Policy"

    # Create decentralized executor
    print(f"Creating decentralized executor with {policy_name}...")
    executor = DecentralizedEventDrivenExecutor(
        env=env,
        policy_fn=policy_fn,
        max_skip_steps=args.max_skip_steps,
        verbose=args.verbose,
    )

    # Run episode

    stats = executor.run_episode(max_steps=args.max_steps)


def main():
    """Parse arguments and run sanity check."""
    parser = argparse.ArgumentParser(
        description="U11 Decentralized Execution Sanity Check"
    )

    # Environment parameters
    parser.add_argument("--num-drones", type=int, default=20,
                        help="Number of drones (default: 10)")
    parser.add_argument("--obs-max-orders", type=int, default=400,
                        help="Maximum orders in observation (default: 200)")
    parser.add_argument("--top-k-merchants", type=int, default=100,
                        help="Top K merchants (default: 50)")
    parser.add_argument("--candidate-k", type=int, default=20,
                        help="Number of candidates per drone (default: 20)")
    parser.add_argument("--enable-random-events", action="store_true", default=False,
                        help="Enable random events (default: False)")

    # Executor parameters
    parser.add_argument("--max-skip-steps", type=int, default=1,
                        help="Max steps to skip when waiting for decisions (default: 10)")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum decision steps per episode (default: 500)")

    # Policy parameters
    parser.add_argument("--model-path", type=str, default='ppo_u11_1010000_steps.zip',
                        help="Path to trained model (.zip file) - if not provided, uses random policy")
    parser.add_argument("--vecnormalize-path", type=str, default='ppo_u11_vecnormalize_1010000_steps.pkl',
                        help="Path to VecNormalize stats (.pkl file)")

    # Order cutoff parameter
    parser.add_argument("--order-cutoff-steps", type=int, default=0,
                        help="Stop accepting orders this many steps before business end (default: 0=disabled)")

    # Ablation parameters
    parser.add_argument("--ablation-cutoff", action="store_true", default=False,
                        help="Enable K-sweep ablation mode: scan environment order_cutoff_steps "
                             "(stops order generation K steps before business end)")
    parser.add_argument("--cutoff-values", type=str, default="0",
                        help="Comma-separated environment order_cutoff_steps (K) values to sweep "
                             "in ablation mode (default: 0..60)")
    parser.add_argument("--seeds", type=str, default="21",
                        help="Comma-separated seed list for ablation mode (default: 42)")
    parser.add_argument("--csv-out", type=str, default=None,
                        help="Output CSV path for ablation results")

    # Other parameters
    parser.add_argument("--seed", type=int, default=21,
                        help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print detailed execution logs (default: False)")
    parser.add_argument("--track-action-stats", action="store_true", default=True,
                        help="Track and print rule selection distribution after each episode "
                             "(default: False)")

    args = parser.parse_args()

    if args.ablation_cutoff:
        try:
            run_ablation_cutoff(args)
        except Exception as e:
            print("\n" + "=" * 80)
            print("Ablation FAILED ✗")
            print("=" * 80)
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Run sanity check
        try:
            run_sanity_check(args)
        except Exception as e:
            print("\n" + "=" * 80)
            print("Sanity Check FAILED ✗")
            print("=" * 80)
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()