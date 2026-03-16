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
"""

import argparse
import os
import random
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


def run_sanity_check(args):
    """Run sanity check with specified configuration."""
    print("=" * 80)
    print("U11 Decentralized Execution Sanity Check")
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
    print(f"  Decision Rounds:          {stats['decision_rounds']}")
    print(f"  Individual Decisions:     {stats['individual_decisions']}")
    print(f"  Actionable Decisions:     {stats['actionable_decisions']}  (order selected, commit attempted)")
    print(f"  Noop / Not Eligible:      {stats['noop_or_not_eligible']}  (no candidates, drone full, etc.)")
    print(f"  Commit Success:           {stats['commit_success']}")
    print(f"  Commit Fail (total):      {stats['failed_decisions'] - stats['noop_or_not_eligible']}")
    print(f"  Overall Success Rate:     {stats['success_rate']:.2%}  (commit_success / individual_decisions)")
    print(f"  Total Skip Steps:         {stats['total_skip_steps']}")
    print(f"  Cumulative Reward:        {stats['cumulative_reward']:.2f}")

    if stats['commit_fail_by_reason']:
        print(f"\nCommit Failure Reasons:")
        for reason, count in sorted(stats['commit_fail_by_reason'].items(),
                                    key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    elif stats['failure_reasons']:
        print(f"\nFailure Reasons (legacy):")
        for reason, count in sorted(stats['failure_reasons'].items(),
                                    key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    if stats['total_decisions'] == 0:
        print("\n⚠ WARNING: No decisions were made during the episode!")
        print("  This might indicate an issue with decision point detection.")

    if stats['success_rate'] < 0.1:
        print("\n⚠ WARNING: Very low success rate!")
        print("  This might indicate issues with order availability or drone capacity.")

    print("\n" + "=" * 80)


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

    args = parser.parse_args()

    # Set random seeds deterministically from CLI argument
    np.random.seed(args.seed)
    random.seed(args.seed)

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