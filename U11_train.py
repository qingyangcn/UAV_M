"""
U11 PPO Training with MOPSO Candidate Generation + Event-Driven Execution

This script implements a hierarchical decision-making system for UAV delivery with
two training modes:

1. EVENT_DRIVEN_SHARED_POLICY (Default - CTDE):
   - Centralized Training, Decentralized Execution
   - Trains a single shared policy that each drone can use independently
   - Action space: Discrete(5) - single rule_id
   - Observation: Rule-discriminant compact state (20-dim Box)
     Features: drone own-state, candidate task structure, rule-discriminant
     metrics (slack, distance, slack/distance) and global context.
   - Enables true decentralized execution during deployment
   - Each drone makes independent decisions using shared policy weights

2. CENTRAL_QUEUE (Compatibility):
   - Centralized queue-based decision making (FIFO drone order)
   - Action space: Discrete(5) - single rule_id for current drone
   - Observation: Same compact 20-dim Box as above
   - Processes drones one by one through central queue

UPPER LAYER (Candidate Generation):
- Uses MOPSOCandidateGenerator to generate candidate order sets for each drone
- MOPSO only generates candidates - it does NOT commit orders (READY -> ASSIGNED)
- Candidates are refreshed periodically (controlled by candidate_update_interval)
- Candidate generation uses multi-objective optimisation to suggest promising orders

LOWER LAYER (Rule Selection):
- Uses wrapper for event-driven decision making
- Action space: Discrete(5) - single rule_id
- Rules select orders from the filtered candidate sets
- Actual order assignment (READY -> ASSIGNED) happens via env.apply_rule_to_drone
- Wrapper automatically advances time when no decisions are needed

Environment: UAV_ENVIRONMENT_11.ThreeObjectiveDroneDeliveryEnv
- Observation exposed to PPO: flat Box(20) rule-discriminant compact state
- Reward: scalar (required for SB3 PPO)
- Candidates constrained by upper layer MOPSO suggestions

Training:
- Algorithm: PPO with MlpPolicy (flat Box observation)
- Normalization: VecNormalize for stability
- Checkpointing: Model and VecNormalize stats saved periodically

Usage:
    # Train with event-driven shared policy (CTDE - default)
    python U11_train.py --total-steps 200000 --seed 42 --num-drones 20 --candidate-k 20

    # Train with legacy central queue mode
    python U11_train.py --training-mode central_queue --total-steps 200000

    # Quick test run
    python U11_train.py --total-steps 1000
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import gymnasium as gym

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from UAV_ENVIRONMENT_11 import ThreeObjectiveDroneDeliveryEnv
from U10_candidate_generator import MOPSOCandidateGenerator
from U10_event_driven_single_uav_wrapper import EventDrivenSingleUAVWrapper
from U11_single_uav_training_wrapper import SingleUAVTrainingWrapper


def make_env(
        seed: int,
        num_drones: int,
        obs_max_orders: int,
        top_k_merchants: int,
        candidate_k: int,
        rule_count: int,
        enable_random_events: bool,
        debug_state_warnings: bool,
        max_skip_steps: int,
        candidate_update_interval: int,
        mopso_n_particles: int,
        mopso_n_iterations: int,
        mopso_max_orders: int,
        mopso_max_orders_per_drone: int,
        energy_e0: float,
        energy_alpha: float,
        battery_return_threshold: float,
        enable_diagnostics: bool,
        diagnostics_interval: int,
        training_mode: str = 'event_driven_shared_policy',
        drone_sampling: str = 'random',
) -> gym.Env:
    """
    Create U11 environment with MOPSO candidate generation and event-driven wrapper.

    Args:
        seed: Random seed
        num_drones: Number of drones
        obs_max_orders: Maximum orders in observation
        top_k_merchants: Top K merchants to include
        candidate_k: Number of candidates per drone
        rule_count: Number of rules (should be 5)
        enable_random_events: Enable random events
        debug_state_warnings: Enable debug warnings
        max_skip_steps: Max steps to skip in wrapper when waiting for decisions
        candidate_update_interval: How often to refresh candidates (0 = only on reset)
        mopso_n_particles: MOPSO particle count
        mopso_n_iterations: MOPSO iteration count
        mopso_max_orders: Max orders for MOPSO
        mopso_max_orders_per_drone: Max orders per drone in MOPSO
        energy_e0: Energy base consumption
        energy_alpha: Energy load coefficient
        battery_return_threshold: Battery threshold for return
        enable_diagnostics: Enable diagnostics
        diagnostics_interval: Diagnostics print interval
        training_mode: Training mode ('event_driven_shared_policy' or 'central_queue')
        drone_sampling: Drone sampling strategy ('random' or 'round_robin')

    Returns:
        Wrapped environment ready for training
    """
    # Create base environment
    env = ThreeObjectiveDroneDeliveryEnv(
        grid_size=16,
        num_drones=num_drones,
        max_orders=obs_max_orders,
        num_bases=2,
        steps_per_hour=12,
        drone_max_capacity=10,
        top_k_merchants=top_k_merchants,
        reward_output_mode="scalar",  # IMPORTANT: PPO requires scalar rewards
        enable_random_events=enable_random_events,
        debug_state_warnings=debug_state_warnings,
        fixed_objective_weights=(0.3, 0.2, 0.5),
        num_candidates=candidate_k,
        rule_count=rule_count,
        enable_diagnostics=enable_diagnostics,
        diagnostics_interval=diagnostics_interval,
        energy_e0=energy_e0,
        energy_alpha=energy_alpha,
        battery_return_threshold=battery_return_threshold,
        multi_objective_mode="fixed",
        candidate_update_interval=candidate_update_interval,
        candidate_fallback_enabled=False,  # Strict candidate filtering
    )

    # Create MOPSO candidate generator
    # Note: Order sharing across drones is always enabled for robustness
    candidate_generator = MOPSOCandidateGenerator(
        candidate_k=candidate_k,
        n_particles=mopso_n_particles,
        n_iterations=mopso_n_iterations,
        max_orders=mopso_max_orders,
        max_orders_per_drone=mopso_max_orders_per_drone,
        seed=seed,
        eta_speed_scale_assumption=0.7,
        eta_stop_service_steps=1,
    )

    # Bind candidate generator to environment
    env.set_candidate_generator(candidate_generator)

    # Wrap based on training mode
    if training_mode == 'event_driven_shared_policy':
        # NEW MODE: Single UAV training wrapper for shared policy
        # - Action space: Discrete(5)
        # - Observation: Local observation (drone_state, candidates, global_context)
        # - Enables decentralized execution during deployment
        env = SingleUAVTrainingWrapper(
            env,
            max_skip_steps=max_skip_steps,
            drone_sampling=drone_sampling,
            local_obs_only=False,
        )
    elif training_mode == 'central_queue':
        # LEGACY MODE: Event-driven single UAV wrapper (centralized queue)
        # - Action space: Discrete(5)
        # - Observation: Full observation with current_drone_id
        # - Uses central queue for decision making
        env = EventDrivenSingleUAVWrapper(
            env,
            max_skip_steps=max_skip_steps,
            local_observation=True,
        )
    else:
        raise ValueError(
            f"Unknown training_mode: {training_mode}. "
            f"Must be 'event_driven_shared_policy' or 'central_queue'"
        )

    return env


def train(args):
    """Main training function."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError as e:
        raise RuntimeError(
            "Please install stable-baselines3: pip install stable-baselines3"
        ) from e

    # Create directories
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # Environment factory
    def env_fn():
        return make_env(
            seed=args.seed,
            num_drones=args.num_drones,
            obs_max_orders=args.obs_max_orders,
            top_k_merchants=args.top_k_merchants,
            candidate_k=args.candidate_k,
            rule_count=args.rule_count,
            enable_random_events=args.enable_random_events,
            debug_state_warnings=args.debug_state_warnings,
            max_skip_steps=args.max_skip_steps,
            candidate_update_interval=args.candidate_update_interval,
            mopso_n_particles=args.mopso_n_particles,
            mopso_n_iterations=args.mopso_n_iterations,
            mopso_max_orders=args.mopso_max_orders,
            mopso_max_orders_per_drone=args.mopso_max_orders_per_drone,
            energy_e0=args.energy_e0,
            energy_alpha=args.energy_alpha,
            battery_return_threshold=args.battery_return_threshold,
            enable_diagnostics=args.enable_diagnostics,
            diagnostics_interval=args.diagnostics_interval,
            training_mode=args.training_mode,
            drone_sampling=args.drone_sampling,
        )

    # Create vectorized environment
    env = DummyVecEnv([env_fn])
    env = VecMonitor(env)

    # VecNormalize for training stability.
    # The observation is now a flat Box (compact rule-based state), so we do NOT
    # need to specify norm_obs_keys – VecNormalize normalises the entire vector.
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=args.gamma,
    )

    # Print configuration
    print("=" * 80)
    print("U11 PPO Training: MOPSO Candidate Generation + Event-Driven Single UAV")
    print("=" * 80)
    print(f"Training Mode: {args.training_mode}")
    if args.training_mode == 'event_driven_shared_policy':
        print(f"  → Shared Policy for Decentralized Execution (CTDE)")
        print(f"  → Drone Sampling: {args.drone_sampling}")
    print(f"Environment: UAV_ENVIRONMENT_11.ThreeObjectiveDroneDeliveryEnv")
    print(f"  num_drones={args.num_drones}, obs_max_orders={args.obs_max_orders}")
    print(f"  top_k_merchants={args.top_k_merchants}, candidate_k={args.candidate_k}")
    print(f"  rule_count={args.rule_count} (Discrete action space)")
    print(f"  candidate_update_interval={args.candidate_update_interval}")
    print(f"\nUpper Layer (Candidate Generation):")
    print(f"  Method: MOPSOCandidateGenerator")
    print(f"  MOPSO: particles={args.mopso_n_particles}, iterations={args.mopso_n_iterations}")
    print(f"  max_orders={args.mopso_max_orders}, max_per_drone={args.mopso_max_orders_per_drone}")
    print(f"  Note: MOPSO only generates candidates, does NOT commit READY->ASSIGNED")
    print(f"\nLower Layer (Rule Selection):")
    if args.training_mode == 'event_driven_shared_policy':
        print(f"  Wrapper: SingleUAVTrainingWrapper")
    else:
        print(f"  Wrapper: EventDrivenSingleUAVWrapper")
    print(f"  Action space: Discrete({args.rule_count})")
    print(f"  Observation: compact rule-based state (Box {env.get_attr('observation_space')[0].shape})")
    print(f"  max_skip_steps={args.max_skip_steps}")
    print(f"\nTraining:")
    print(f"  Algorithm: PPO with MlpPolicy")
    print(f"  VecNormalize: ENABLED (norm_obs=True, norm_reward=True)")
    print(f"  Total steps: {args.total_steps}")
    print(f"  Seed: {args.seed}")
    print(f"  enable_random_events={args.enable_random_events}")
    print(f"  enable_diagnostics={args.enable_diagnostics}, interval={args.diagnostics_interval}")
    print("=" * 80)

    # Print action/observation space info
    print(f"\nAction space: {env.get_attr('action_space')[0]}")
    print(f"Observation space: {env.get_attr('observation_space')[0]}")

    # Create PPO model with MlpPolicy (flat Box observation)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        verbose=1,
        tensorboard_log=args.log_dir,
        seed=args.seed,
        device='cuda',
    )

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.model_dir,
        name_prefix="ppo_u11",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # Train
    model.learn(
        total_timesteps=args.total_steps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(args.model_dir, "ppo_u11_final")
    model.save(final_path)
    print(f"\nSaved final model to: {final_path}")

    # Save VecNormalize statistics
    vecnormalize_path = os.path.join(args.model_dir, "vecnormalize_u11_final.pkl")
    env.save(vecnormalize_path)
    print(f"Saved VecNormalize statistics to: {vecnormalize_path}")
    print(f"Note: Load VecNormalize when evaluating: VecNormalize.load('{vecnormalize_path}', venv)")

    env.close()


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(
        description="U11 PPO Training with MOPSO Candidate Generation"
    )

    # Training parameters
    parser.add_argument("--total-steps", type=int, default=2000000,
                        help="Total training steps (default: 200000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--training-mode", type=str,
                        default='event_driven_shared_policy',
                        choices=['event_driven_shared_policy', 'central_queue'],
                        help="Training mode: event_driven_shared_policy (CTDE, default) or central_queue (legacy)")
    parser.add_argument("--drone-sampling", type=str,
                        default='random',
                        choices=['random', 'round_robin'],
                        help="Drone sampling strategy for training (default: random)")

    # Environment parameters
    parser.add_argument("--num-drones", type=int, default=20,
                        help="Number of drones (default: 20)")
    parser.add_argument("--obs-max-orders", type=int, default=200,
                        help="Maximum orders in observation (default: 400)")
    parser.add_argument("--top-k-merchants", type=int, default=50,
                        help="Top K merchants (default: 100)")
    parser.add_argument("--candidate-k", type=int, default=10,
                        help="Number of candidates per drone (default: 20)")
    parser.add_argument("--rule-count", type=int, default=5,
                        help="Number of rules for action space (default: 5)")
    parser.add_argument("--enable-random-events", action="store_true", default=False,
                        help="Enable random events (default: False)")
    parser.add_argument("--debug-state-warnings", action="store_true", default=False,
                        help="Enable debug state warnings (default: False)")

    # Wrapper parameters
    parser.add_argument("--max-skip-steps", type=int, default=10,
                        help="Max steps to skip when waiting for decisions (default: 10)")

    # Candidate generation parameters
    parser.add_argument("--candidate-update-interval", type=int, default=8,
                        help="Candidate refresh interval (0=only on reset, default: 8)")
    parser.add_argument("--mopso-n-particles", type=int, default=10,
                        help="MOPSO particle count (default: 30)")
    parser.add_argument("--mopso-n-iterations", type=int, default=3,
                        help="MOPSO iteration count (default: 10)")
    parser.add_argument("--mopso-max-orders", type=int, default=100,
                        help="Max orders for MOPSO (default: 200)")
    parser.add_argument("--mopso-max-orders-per-drone", type=int, default=10,
                        help="Max orders per drone in MOPSO (default: 10)")

    # Energy model parameters
    parser.add_argument("--energy-e0", type=float, default=0.1,
                        help="Energy base consumption (default: 0.1)")
    parser.add_argument("--energy-alpha", type=float, default=0.5,
                        help="Energy load coefficient (default: 0.5)")
    parser.add_argument("--battery-return-threshold", type=float, default=10.0,
                        help="Battery threshold for return (default: 10.0)")

    # Diagnostics
    parser.add_argument("--enable-diagnostics", action="store_true", default=False,
                        help="Enable environment diagnostics (default: False)")
    parser.add_argument("--diagnostics-interval", type=int, default=64,
                        help="Diagnostics print interval (default: 64)")

    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Steps per rollout (default: 2048)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Epochs per update (default: 10)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda (default: 0.95)")
    parser.add_argument("--clip-range", type=float, default=0.1,
                        help="PPO clip range (default: 0.1)")

    # Saving
    parser.add_argument("--save-freq", type=int, default=10000,
                        help="Checkpoint save frequency (default: 10000)")
    parser.add_argument("--log-dir", type=str, default="./logs/u11",
                        help="TensorBoard log directory (default: ./logs/u11)")
    parser.add_argument("--model-dir", type=str, default="./models/u11",
                        help="Model save directory (default: ./models/u11)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()