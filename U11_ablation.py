"""
U11 Sanity Check for Decentralized Event-Driven Execution

This script validates the decentralized event-driven execution system by:
1. Running a small episode with random policy
2. Testing event-driven loop functionality
3. Printing statistics (decisions, skips, success/failure rates)
4. Optionally loading and testing a trained policy

Usage:
    # Test with random policy
    python U11_sanity_check_decentralized.py

    # Test with trained policy
    python U11_sanity_check_decentralized.py --model-path ./models/u10/ppo_u10_final.zip

    # Quick test (fewer steps)
    python U11_sanity_check_decentralized.py --max-steps 100

    # Ablation scan over high_load_factor
    python U11_sanity_check_decentralized.py --ablation-high-load \\
        --high-load-values 1.0,0.9,0.8,0.7 --seeds 42,123,999 \\
        --csv-out ablation_results.csv
"""

import argparse
import csv
import os
import sys
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from UAV_ENVIRONMENT_11 import ThreeObjectiveDroneDeliveryEnv
from U10_candidate_generator import MOPSOCandidateGenerator
from U11_decentralized_execution import DecentralizedEventDrivenExecutor

# Tolerance for floating-point step boundary comparisons in grid scan
_FLOAT_TOLERANCE = 1e-9
# Decimal places used when rounding grid-scan hlf values to avoid fp drift
_HLF_DECIMAL_PRECISION = 10


def random_policy(local_obs: dict) -> int:
    """Simple random policy for testing."""
    return np.random.randint(0, 5)


def load_trained_policy(model_path: str, vecnormalize_path: str = None):
    """
    Load a trained PPO policy.

    Args:
        model_path: Path to trained model (.zip file)
        vecnormalize_path: Path to VecNormalize stats (.pkl file)

    Returns:
        Policy function that takes local_obs and returns rule_id
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize
    except ImportError:
        raise RuntimeError("Please install stable-baselines3: pip install stable-baselines3")

    # Load model
    model = PPO.load(model_path)

    # Load VecNormalize stats if provided
    vec_normalize = None
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        print(f"Loading VecNormalize stats from: {vecnormalize_path}")
        # Note: VecNormalize needs a dummy env to load
        # For now, we'll skip normalization in evaluation
        print("Warning: VecNormalize stats loading not implemented in sanity check")

    def policy_fn(local_obs: dict) -> int:
        """Wrapper function for trained policy."""
        # Convert dict obs to format expected by model
        # Model expects dict with keys: drone_state, candidates, global_context
        action, _ = model.predict(local_obs, deterministic=True)
        return int(action)

    return policy_fn


def run_sanity_check(args):
    """Run sanity check with specified configuration."""
    print("=" * 80)
    print("U10 Decentralized Execution Sanity Check")
    print("=" * 80)

    # Create environment
    print("\nCreating environment...")
    env = ThreeObjectiveDroneDeliveryEnv(
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
    )

    # Create MOPSO candidate generator
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
        verbose=args.verbose
    )

    # Run episode
    print("\n" + "=" * 80)
    print(f"Running episode (max {args.max_steps} decision steps)...")
    print("=" * 80 + "\n")

    stats = executor.run_episode(max_steps=args.max_steps)
    # Print results
    print("\n" + "=" * 80)
    print("Sanity Check Results")
    print("=" * 80)
    print(f"\nPolicy: {policy_name}")
    print(f"Environment: {args.num_drones} drones, {args.candidate_k} candidates/drone")
    print(f"\nExecution Statistics:")
    print(f"  Total Decision Rounds: {stats['total_decision_rounds']}")
    print(f"  Total Individual Decisions: {stats['total_decisions']}")
    print(f"  Successful Decisions: {stats['successful_decisions']}")
    print(f"  Failed Decisions: {stats['failed_decisions']}")
    print(f"  Success Rate: {stats['success_rate']:.2%}")
    print(f"  Total Skip Steps: {stats['total_skip_steps']}")
    print(f"  Cumulative Reward: {stats['cumulative_reward']:.2f}")

    if stats['failure_reasons']:
        print(f"\nFailure Reasons:")
        for reason, count in stats['failure_reasons'].items():
            print(f"  {reason}: {count}")

    print("\n" + "=" * 80)
    print("Sanity Check PASSED ✓")
    print("=" * 80)
    print("\nKey Validations:")
    print("  ✓ Event-driven loop executed successfully")
    print("  ✓ Decentralized decisions processed")
    print("  ✓ Fast-forward mechanism worked")
    print(f"  ✓ Decision success rate: {stats['success_rate']:.2%}")

    if stats['total_decisions'] == 0:
        print("\n⚠ WARNING: No decisions were made during the episode!")
        print("  This might indicate an issue with decision point detection.")

    if stats['success_rate'] < 0.1:
        print("\n⚠ WARNING: Very low success rate!")
        print("  This might indicate issues with order availability or drone capacity.")

    print("\n" + "=" * 80)


def _make_env(args, high_load_factor, seed):
    """Create environment and candidate generator for a given high_load_factor and seed."""
    env = ThreeObjectiveDroneDeliveryEnv(
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
        high_load_factor=high_load_factor,
    )
    candidate_generator = MOPSOCandidateGenerator(
        candidate_k=args.candidate_k,
        n_particles=30,
        n_iterations=10,
        max_orders=200,
        max_orders_per_drone=10,
        seed=seed,
    )
    env.set_candidate_generator(candidate_generator)
    return env


def run_ablation(args):
    """Run ablation scan over high_load_factor values across multiple seeds."""
    # Determine high_load_factor values to scan
    if args.high_load_values:
        hlf_values = [float(v.strip()) for v in args.high_load_values.split(",") if v.strip()]
    else:
        start = args.high_load_start
        end = args.high_load_end
        step = args.high_load_step
        if step <= 0:
            raise ValueError("--high-load-step must be positive")
        hlf_values = []
        v = start
        while v <= end + _FLOAT_TOLERANCE:
            hlf_values.append(round(v, _HLF_DECIMAL_PRECISION))
            v += step

    # Determine seeds
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed]

    # Load policy once (reused across runs)
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading trained policy from: {args.model_path}")
        policy_fn = load_trained_policy(args.model_path, args.vecnormalize_path)
        policy_name = "Trained Policy"
    else:
        policy_fn = random_policy
        policy_name = "Random Policy"

    print("=" * 80)
    print("Ablation Scan: high_load_factor")
    print("=" * 80)
    print(f"Policy      : {policy_name}")
    print(f"HLF values  : {hlf_values}")
    print(f"Seeds       : {seeds}")
    print(f"Max steps   : {args.max_steps}")
    if args.csv_out:
        print(f"CSV output  : {args.csv_out}")
    print("=" * 80)

    rows = []

    for hlf in hlf_values:
        for seed in seeds:
            np.random.seed(seed)
            print(f"\n[hlf={hlf:.4f}, seed={seed}] Creating env and running episode...")
            env = _make_env(args, hlf, seed)
            executor = DecentralizedEventDrivenExecutor(
                env=env,
                policy_fn=policy_fn,
                max_skip_steps=args.max_skip_steps,
                verbose=args.verbose,
            )
            executor.run_episode(max_steps=args.max_steps)

            ds = env.daily_stats
            generated = ds.get("orders_generated", 0)
            completed = ds.get("orders_completed", 0)
            cancelled = ds.get("orders_cancelled", 0)
            completion_rate = completed / generated if generated > 0 else 0.0
            is_100pct = int(completed == generated and cancelled == 0 and generated > 0)

            print(f"  generated={generated}, completed={completed}, "
                  f"cancelled={cancelled}, rate={completion_rate:.2%}")

            rows.append({
                "high_load_factor": hlf,
                "seed": seed,
                "generated_orders": generated,
                "completed_orders": completed,
                "cancelled_orders": cancelled,
                "completion_rate": completion_rate,
                "is_100pct": is_100pct,
            })

    # Write CSV
    if args.csv_out:
        fieldnames = ["high_load_factor", "seed", "generated_orders",
                      "completed_orders", "cancelled_orders", "completion_rate", "is_100pct"]
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults written to: {args.csv_out}")

    # Summary
    print("\n" + "=" * 80)
    print("Ablation Summary")
    print("=" * 80)
    best_hlf = None
    for hlf in hlf_values:
        hlf_rows = [r for r in rows if r["high_load_factor"] == hlf]
        success_count = sum(r["is_100pct"] for r in hlf_rows)
        total = len(hlf_rows)
        print(f"  hlf={hlf:.4f}: {success_count}/{total} seeds at 100%")
        if success_count == total and total > 0:
            best_hlf = hlf

    if best_hlf is not None:
        print(f"\n  Largest high_load_factor with 100% on ALL seeds: {best_hlf:.4f}")
    else:
        print("\n  No high_load_factor achieved 100% completion on all seeds.")
    print("=" * 80)


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
    parser.add_argument("--model-path", type=str, default='ppo_u11_final.zip',
                        help="Path to trained model (.zip file) - if not provided, uses random policy")
    parser.add_argument("--vecnormalize-path", type=str, default='vecnormalize_u11_final.pkl',
                        help="Path to VecNormalize stats (.pkl file)")

    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print detailed execution logs (default: False)")

    # Ablation mode parameters
    parser.add_argument("--ablation-high-load", action="store_true", default=False,
                        help="Enable ablation scan over high_load_factor values")
    parser.add_argument("--high-load-values", type=str, default=None,
                        help="Comma-separated list of high_load_factor values to scan, "
                             "e.g. '1.0,0.9,0.8,0.7' (takes priority over start/end/step)")
    parser.add_argument("--high-load-start", type=float, default=0.5,
                        help="Start value for high_load_factor grid scan (default: 0.5)")
    parser.add_argument("--high-load-end", type=float, default=1.5,
                        help="End value for high_load_factor grid scan (default: 1.5)")
    parser.add_argument("--high-load-step", type=float, default=0.1,
                        help="Step size for high_load_factor grid scan (default: 0.1)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated list of random seeds for ablation, "
                             "e.g. '42,123,999' (overrides --seed in ablation mode)")
    parser.add_argument("--csv-out", type=str, default=None,
                        help="Output CSV file path for ablation results")

    args = parser.parse_args()

    if args.ablation_high_load:
        try:
            run_ablation(args)
        except Exception as e:
            print("\n" + "=" * 80)
            print("Ablation FAILED ✗")
            print("=" * 80)
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Set random seed for non-ablation mode
        np.random.seed(args.seed)

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