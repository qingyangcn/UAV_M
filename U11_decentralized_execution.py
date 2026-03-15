"""
Decentralized Event-Driven Execution for U10

This module implements the event-driven decentralized execution architecture where:
1. Each drone independently makes decisions using a shared policy
2. Decisions are made only when drones reach decision points (event-driven)
3. System fast-forwards time when no decisions are needed
4. Centralized arbitration ensures atomic order assignment (READY -> ASSIGNED)

Key Concepts:
- CTDE (Centralized Training, Decentralized Execution): Train one policy, execute independently per drone
- Event-driven: Only act when needed, not every time step
- Shared policy: All drones use the same policy weights but with their own local observations
- Centralized arbitration: Environment atomically handles order conflicts

Usage:
    # For evaluation/deployment with a trained policy
    executor = DecentralizedEventDrivenExecutor(env, policy)
    obs, info = executor.reset()

    while not done:
        obs, reward, terminated, truncated, info = executor.step()
        done = terminated or truncated
"""

import random
from collections import Counter
from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np
import gymnasium as gym


class ActionStats:
    """Per-episode action-selection statistics, collected when track_action_stats=True."""

    def __init__(self):
        self.rule_counts: Counter = Counter()
        self.n_decisions: int = 0
        self.n_invalid_rule: int = 0
        self.n_empty_candidates: int = 0

    def to_percent(self) -> dict:
        """Return each rule's share of all tracked decisions as a percentage."""
        total = sum(self.rule_counts.values())
        if total == 0:
            return {i: 0.0 for i in range(5)}
        return {k: 100.0 * v / total for k, v in self.rule_counts.items()}


class DecentralizedEventDrivenExecutor:
    """
    Event-driven decentralized execution manager for multi-UAV system.

    This executor:
    1. Detects drones at decision points using env.get_decision_drones()
    2. For each decision drone, extracts local observation and calls policy
    3. Submits rule_id to centralized arbitration via env.apply_rule_to_drone()
    4. Fast-forwards environment when no drones need decisions

    Args:
        env: The UAV environment (must have get_decision_drones, apply_rule_to_drone)
        policy_fn: Function that takes local_obs and returns rule_id (0-4)
        max_skip_steps: Maximum steps to skip when waiting for decisions
        verbose: Whether to print execution details
        track_action_stats: When True, collect per-rule selection statistics accessible
            via get_action_stats().  Defaults to False to preserve existing behaviour.
    """

    def __init__(
            self,
            env: gym.Env,
            policy_fn: Callable[[Dict], int],
            max_skip_steps: int = 10,
            verbose: bool = False,
            track_action_stats: bool = False,
    ):
        self.env = env
        self.policy_fn = policy_fn
        self.max_skip_steps = max_skip_steps
        self.verbose = verbose
        self.track_action_stats = track_action_stats

        # Unwrap environment to access methods
        self.unwrapped_env = env.unwrapped

        # Statistics
        self.total_decisions = 0
        self.total_decision_rounds = 0
        self.total_skip_steps = 0
        self.successful_decisions = 0
        self.failed_decisions = 0
        self.decision_failures_by_reason = {}

        # Fine-grained statistics
        self.actionable_decisions = 0       # rule selected an order, produced a commit attempt
        self.noop_or_not_eligible = 0       # no candidates, drone busy/full, not at decision point
        self.commit_fail_by_reason = {}     # commit attempts that failed, keyed by reason

        # Per-episode action-selection stats (populated only when track_action_stats=True)
        self._action_stats: ActionStats = ActionStats()

        # Episode state
        self.episode_active = False
        self.cumulative_reward = 0.0

        # Cached last observation (updated after every env.reset/step)
        self._last_obs = None

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """
        Reset environment and prepare for episode.

        Returns:
            observation: Full environment observation (for monitoring)
            info: Info dict with execution metadata
        """
        obs, info = self.env.reset(**kwargs)

        # Cache observation for local obs extraction
        self._last_obs = obs

        # Reset statistics
        self.total_decisions = 0
        self.total_decision_rounds = 0
        self.total_skip_steps = 0
        self.successful_decisions = 0
        self.failed_decisions = 0
        self.decision_failures_by_reason = {}
        self.actionable_decisions = 0
        self.noop_or_not_eligible = 0
        self.commit_fail_by_reason = {}
        self._action_stats = ActionStats()
        self.cumulative_reward = 0.0
        self.episode_active = True

        # Add executor info
        info['executor'] = self.get_statistics()

        if self.verbose:
            print("=" * 60)
            print("DecentralizedEventDrivenExecutor: Episode Started")
            print("=" * 60)

        return obs, info

    def step(self) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one decision round for all drones at decision points.

        This method:
        1. Gets all drones at decision points
        2. For each drone: extract local obs, call policy, submit to arbitrator
        3. Advances environment one step after all decisions
        4. If no decisions, fast-forwards until next decision event

        Returns:
            observation: Full environment observation
            reward: Accumulated reward from this step
            terminated: Whether episode terminated
            truncated: Whether episode truncated
            info: Info dict with decision details
        """
        if not self.episode_active:
            raise RuntimeError("Episode not active. Call reset() first.")

        # Get drones at decision points
        decision_drones = self.unwrapped_env.get_decision_drones()

        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if decision_drones:
            # Process all drones at decision points
            obs, reward, terminated, truncated, info = self._process_decision_round(decision_drones)
            total_reward += reward
        else:
            # No decisions needed, fast-forward to next decision event
            obs, reward, terminated, truncated, info = self._skip_to_next_decision()
            total_reward += reward

        # Update cumulative reward
        self.cumulative_reward += total_reward

        # Add executor statistics to info
        info['executor'] = self.get_statistics()

        # Check if episode ended
        if terminated or truncated:
            self.episode_active = False
            if self.verbose:
                print("=" * 60)
                print("DecentralizedEventDrivenExecutor: Episode Ended")
                print(f"  Total Decisions: {self.total_decisions}")
                print(f"  Successful: {self.successful_decisions}, Failed: {self.failed_decisions}")
                print(f"  Decision Rounds: {self.total_decision_rounds}")
                print(f"  Skip Steps: {self.total_skip_steps}")
                print(f"  Cumulative Reward: {self.cumulative_reward:.2f}")
                print("=" * 60)

        return obs, total_reward, terminated, truncated, info

    def _process_decision_round(
            self,
            decision_drones: List[int]
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Process one decision round for all drones at decision points.

        Args:
            decision_drones: List of drone IDs at decision points

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.total_decision_rounds += 1
        round_decisions = []

        if self.verbose:
            print(f"\n--- Decision Round {self.total_decision_rounds} ---")
            print(f"  Drones at decision points: {decision_drones}")

        # Get current observation for extracting local obs
        current_obs = self._get_current_observation()

        # Process each drone independently
        for drone_id in decision_drones:
            self.total_decisions += 1

            # Extract local observation for this drone
            local_obs = self._extract_local_observation(current_obs, drone_id)

            # Call policy to get rule_id
            rule_id = self.policy_fn(local_obs)

            # Track action stats when enabled
            if self.track_action_stats:
                self._action_stats.n_decisions += 1
                if not (0 <= rule_id < 5):
                    self._action_stats.n_invalid_rule += 1
                else:
                    self._action_stats.rule_counts[rule_id] += 1

            # Submit to centralized arbitrator (use with_info if available)
            if hasattr(self.unwrapped_env, 'apply_rule_to_drone_with_info'):
                success, decision_info = self.unwrapped_env.apply_rule_to_drone_with_info(
                    drone_id, rule_id
                )
                reason = decision_info.get('failure_reason') or 'unknown'
                order_id = decision_info.get('order_id')
            else:
                success = self.unwrapped_env.apply_rule_to_drone(drone_id, rule_id)
                reason = 'unknown'
                order_id = None

            # Noop / not-eligible reasons: no candidates, drone full, not at decision point
            _noop_reasons = {'no_order_selected', 'drone_at_capacity', 'not_at_decision_point',
                             'invalid_drone_id'}

            # Track result
            if success:
                self.successful_decisions += 1
                self.actionable_decisions += 1
                decision_result = "SUCCESS"
            else:
                self.failed_decisions += 1
                if reason in _noop_reasons:
                    self.noop_or_not_eligible += 1
                    if self.track_action_stats and reason == 'no_order_selected':
                        self._action_stats.n_empty_candidates += 1
                else:
                    # Actionable but commit failed
                    self.actionable_decisions += 1
                    self.commit_fail_by_reason[reason] = \
                        self.commit_fail_by_reason.get(reason, 0) + 1
                # Legacy failure reason tracking
                self.decision_failures_by_reason[reason] = \
                    self.decision_failures_by_reason.get(reason, 0) + 1
                decision_result = f"FAILED({reason})"

            round_decisions.append({
                'drone_id': drone_id,
                'rule_id': rule_id,
                'success': success,
                'order_id': order_id,
                'reason': reason if not success else None,
            })

            if self.verbose:
                print(f"  Drone {drone_id}: rule_id={rule_id}, result={decision_result}")

        # Advance environment one step after all decisions
        # Use dummy action (all zeros) to just advance time
        dummy_action = np.zeros(self.unwrapped_env.num_drones, dtype=np.int32)
        obs, reward, terminated, truncated, info = self.env.step(dummy_action)
        self._last_obs = obs

        # Add decision round info
        info['decision_round'] = {
            'round_number': self.total_decision_rounds,
            'num_decisions': len(decision_drones),
            'decisions': round_decisions
        }

        return obs, reward, terminated, truncated, info

    def _skip_to_next_decision(self) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Fast-forward environment until a decision event occurs.

        Returns:
            observation, accumulated_reward, terminated, truncated, info
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        obs = None
        info = {}

        for skip_step in range(self.max_skip_steps):
            # Advance environment with no-op
            dummy_action = np.zeros(self.unwrapped_env.num_drones, dtype=np.int32)
            obs, reward, terminated, truncated, info = self.env.step(dummy_action)
            self._last_obs = obs

            total_reward += reward
            self.total_skip_steps += 1

            # Check if episode ended
            if terminated or truncated:
                break

            # Check for decision events
            decision_drones = self.unwrapped_env.get_decision_drones()
            if decision_drones:
                # Found decision event, will be processed in next step() call
                break

        if self.verbose and self.total_skip_steps > 0:
            print(f"  Skipped {skip_step + 1} steps (no decisions)")

        info['skip_info'] = {
            'steps_skipped': skip_step + 1 if not (terminated or truncated) else skip_step,
            'reason': 'episode_end' if (terminated or truncated) else 'decision_event_found'
        }

        return obs, total_reward, terminated, truncated, info

    def _get_current_observation(self) -> Dict:
        """
        Get current observation from environment.

        Prefers the internally cached observation updated after every env.reset/step,
        falling back to calling the environment's observation method directly.

        Returns:
            Current observation dict
        """
        if self._last_obs is not None:
            return self._last_obs
        # Fallback: call observation method directly on unwrapped env
        if hasattr(self.unwrapped_env, '_get_obs'):
            return self.unwrapped_env._get_obs()
        elif hasattr(self.unwrapped_env, '_get_observation'):
            return self.unwrapped_env._get_observation()
        elif hasattr(self.unwrapped_env, 'last_obs'):
            return self.unwrapped_env.last_obs
        raise RuntimeError("Cannot obtain current observation from environment.")

    def _extract_local_observation(self, full_obs: Dict, drone_id: int) -> np.ndarray:
        """
        Return the compact rule-based state vector for a specific drone.

        Delegates to env.unwrapped._get_rule_based_state_for_drone(), which builds
        the 20-dimensional rule-discriminant feature vector directly from environment
        state. The full_obs argument is accepted for API compatibility but is not used.

        Args:
            full_obs: Full observation from environment (unused)
            drone_id: Drone ID to extract observation for

        Returns:
            np.ndarray of shape (RULE_BASED_STATE_DIM,), dtype=np.float32
        """
        return self.unwrapped_env._get_rule_based_state_for_drone(drone_id)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Dictionary with statistics including fine-grained decision metrics:
            - decision_rounds: number of rounds where at least one drone decided
            - individual_decisions: total per-drone decision calls
            - actionable_decisions: decisions that selected an order (commit attempted)
            - noop_or_not_eligible: decisions with no candidates / drone full / not eligible
            - commit_success: successful commit count (= successful_decisions)
            - commit_fail_by_reason: dict of reason -> count for failed commit attempts
            - failure_reasons: legacy dict (same as decision_failures_by_reason)
        """
        return {
            'total_decisions': self.total_decisions,
            'total_decision_rounds': self.total_decision_rounds,
            'total_skip_steps': self.total_skip_steps,
            'successful_decisions': self.successful_decisions,
            'failed_decisions': self.failed_decisions,
            'success_rate': (
                    self.successful_decisions / max(self.total_decisions, 1)
            ),
            'failure_reasons': dict(self.decision_failures_by_reason),
            'cumulative_reward': self.cumulative_reward,
            # Fine-grained metrics
            'decision_rounds': self.total_decision_rounds,
            'individual_decisions': self.total_decisions,
            'actionable_decisions': self.actionable_decisions,
            'noop_or_not_eligible': self.noop_or_not_eligible,
            'commit_success': self.successful_decisions,
            'commit_fail_by_reason': dict(self.commit_fail_by_reason),
        }

    def get_action_stats(self) -> ActionStats:
        """
        Return per-episode action-selection statistics.

        Only meaningful when ``track_action_stats=True`` was passed to the constructor.
        Returns an :class:`ActionStats` instance with:
        - ``rule_counts``: :class:`~collections.Counter` mapping rule_id → selection count
        - ``n_decisions``: total number of decisions tracked
        - ``n_invalid_rule``: decisions where policy returned a rule_id outside {0..4}
        - ``n_empty_candidates``: decisions where no candidate was available (no_order_selected)
        - ``to_percent()``: convenience method returning per-rule percentages
        """
        return self._action_stats

    def run_episode(self, max_steps: int = 10000, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a complete episode from start to finish.

        Args:
            max_steps: Maximum number of decision steps
            seed: Optional integer seed for deterministic episode execution.
                When provided, seeds both numpy and Python's random module before
                resetting the environment, ensuring reproducibility across runs.

        Returns:
            Episode statistics
        """
        if seed is not None:
            # Seed both the global numpy/random state (for policy and stochastic
            # library calls) and the environment's internal RNG (via reset(seed=)).
            # Both are required for full cross-component reproducibility:
            #   - np.random / random: policy outputs, any library-level sampling
            #   - env reset seed: environment-internal order generation, events
            np.random.seed(seed)
            random.seed(seed)
            obs, info = self.reset(seed=seed)
        else:
            obs, info = self.reset()

        for step_num in range(max_steps):
            obs, reward, terminated, truncated, info = self.step()

            if terminated or truncated:
                break

        return self.get_statistics()