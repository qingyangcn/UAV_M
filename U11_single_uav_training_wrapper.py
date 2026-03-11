"""
Single UAV Training Wrapper for Event-Driven Shared Policy

This wrapper enables training a shared policy that can be used by all drones
in a decentralized manner. It provides:

1. Discrete(5) action space (independent of number of drones N)
2. Local observation for the current decision drone
3. Drone sampling strategy for training data generation
4. Compatibility with SB3 PPO and other single-agent RL algorithms

Key Features:
- Action space: Discrete(5) - one rule_id per step
- Observation: Local observation (drone_state, candidates, global_context)
- Drone selection: Randomly samples from drones at decision points
- Episode handling: Advances until episode end or max steps reached

Usage:
    env = ThreeObjectiveDroneDeliveryEnv(...)
    env = SingleUAVTrainingWrapper(env)

    # Now env has Discrete(5) action space and local observations
    # Compatible with SB3 PPO:
    model = PPO("MultiInputPolicy", env, ...)
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SingleUAVTrainingWrapper(gym.Wrapper):
    """
    Training wrapper that converts multi-drone environment to single-drone interface.

    This wrapper:
    1. Converts action space from MultiDiscrete([R]*N) to Discrete(R)
    2. Provides local observation for the current decision drone
    3. Samples drones at decision points to generate training data
    4. Advances environment to next decision event when needed

    Args:
        env: The base UAV environment
        max_skip_steps: Maximum steps to skip when waiting for decisions
        drone_sampling: Strategy for selecting drones ('random', 'round_robin')
        local_obs_only: If True, only return local observation (no global info)
    """

    def __init__(
            self,
            env: gym.Env,
            max_skip_steps: int = 10,
            drone_sampling: str = 'random',
            local_obs_only: bool = False
    ):
        super().__init__(env)

        self.max_skip_steps = max_skip_steps
        self.drone_sampling = drone_sampling
        self.local_obs_only = local_obs_only

        # Get number of rules from environment
        if hasattr(env.unwrapped, 'rule_count'):
            self.rule_count = env.unwrapped.rule_count
        else:
            self.rule_count = 5  # Default

        # Override action space to single discrete action
        self.action_space = spaces.Discrete(self.rule_count)

        # Define local observation space
        # Based on UAV_ENVIRONMENT_11 observation structure:
        # - drone_state: 8 features (index 5 = cargo ratio, 0=empty/pickup, >0=has cargo/delivery)
        # - candidates: K x 12 features (K from env.num_candidates)
        # - global_context: 10 features (time=5, day_progress=1, resource_sat=1, weather=3)
        num_candidates = getattr(env.unwrapped, 'num_candidates', 20)

        self.observation_space = spaces.Dict({
            'drone_state': spaces.Box(
                low=0, high=1, shape=(8,), dtype=np.float32
            ),
            'candidates': spaces.Box(
                low=0, high=1, shape=(num_candidates, 12), dtype=np.float32
            ),
            'global_context': spaces.Box(
                low=0, high=1, shape=(10,), dtype=np.float32
            ),
        })

        # Current drone being processed
        self.current_drone_id: Optional[int] = None
        self.decision_queue: List[int] = []
        self.round_robin_index: int = 0

        # Track last observation from environment
        self.last_full_obs: Optional[Dict] = None
        self.last_info: Optional[Dict] = None

        # Statistics
        self.total_decisions = 0
        self.total_skip_steps = 0
        self.episode_steps = 0

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """
        Reset environment and get first local observation.

        Returns:
            observation: Local observation for first decision drone
            info: Info dictionary
        """
        # Reset underlying environment
        obs, info = self.env.reset(**kwargs)

        # Store full observation
        self.last_full_obs = obs
        self.last_info = info

        # Reset state
        self.current_drone_id = None
        self.decision_queue = []
        self.round_robin_index = 0
        self.total_decisions = 0
        self.total_skip_steps = 0
        self.episode_steps = 0

        # Get first decision drone
        self._populate_decision_queue()

        # If no decisions, skip forward
        if self.current_drone_id is None:
            obs, _, _, _, info = self._skip_to_next_decision()
            self.last_full_obs = obs
            self.last_info = info
            self._populate_decision_queue()

        # Get local observation
        local_obs = self._extract_local_observation(
            self.last_full_obs, self.current_drone_id
        )

        # Add metadata to info
        info['wrapper'] = {
            'current_drone_id': self.current_drone_id,
            'queue_length': len(self.decision_queue),
            'total_decisions': self.total_decisions,
            'total_skip_steps': self.total_skip_steps,
        }

        return local_obs, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute action for current drone and advance to next decision.

        Args:
            action: Rule ID (0-4) to apply to current drone

        Returns:
            observation: Next local observation
            reward: Accumulated reward
            terminated: Whether episode terminated
            truncated: Whether episode truncated
            info: Info dictionary
        """
        if self.current_drone_id is None:
            raise RuntimeError(
                "No current drone. This should not happen - likely a reset() issue."
            )

        drone_id = self.current_drone_id

        # Apply rule to drone through arbitrator
        success = self.env.unwrapped.apply_rule_to_drone(drone_id, action)

        # Advance environment one step
        dummy_action = np.zeros(self.env.unwrapped.num_drones, dtype=np.int32)
        obs, reward, terminated, truncated, info = self.env.step(dummy_action)

        # Store observation
        self.last_full_obs = obs
        self.last_info = info

        # Update statistics
        self.total_decisions += 1
        self.episode_steps += 1

        # Move to next drone
        self.current_drone_id = None
        self._populate_decision_queue()

        # If no more drones and episode not done, skip forward
        if self.current_drone_id is None and not (terminated or truncated):
            skip_obs, skip_reward, terminated, truncated, skip_info = \
                self._skip_to_next_decision()
            obs = skip_obs
            reward += skip_reward
            info.update(skip_info)
            self.last_full_obs = obs
            self.last_info = info
            self._populate_decision_queue()

        # Get next local observation
        local_obs = self._extract_local_observation(
            self.last_full_obs, self.current_drone_id
        )

        # Add metadata to info
        info['wrapper'] = {
            'last_drone_id': drone_id,
            'last_rule_id': action,
            'last_decision_success': success,
            'current_drone_id': self.current_drone_id,
            'queue_length': len(self.decision_queue),
            'total_decisions': self.total_decisions,
            'total_skip_steps': self.total_skip_steps,
            'episode_steps': self.episode_steps,
        }

        return local_obs, reward, terminated, truncated, info

    def _populate_decision_queue(self):
        """
        Populate decision queue with drones at decision points.

        Uses the configured drone_sampling strategy to select next drone.
        """
        if not self.decision_queue:
            # Get drones at decision points from environment
            decision_drones = self.env.unwrapped.get_decision_drones()

            if not decision_drones:
                self.current_drone_id = None
                return

            # Select drone based on sampling strategy
            if self.drone_sampling == 'random':
                # Randomly sample one drone
                selected_drone = np.random.choice(decision_drones)
                self.current_drone_id = selected_drone
                # Add remaining drones to queue for potential future use
                self.decision_queue = [d for d in decision_drones if d != selected_drone]

            elif self.drone_sampling == 'round_robin':
                # Sort for consistent ordering
                decision_drones = sorted(decision_drones)
                # Use round-robin index to select
                selected_idx = self.round_robin_index % len(decision_drones)
                selected_drone = decision_drones[selected_idx]
                self.current_drone_id = selected_drone
                self.round_robin_index += 1
                # Add remaining drones to queue
                self.decision_queue = [
                    d for d in decision_drones if d != selected_drone
                ]

            else:
                raise ValueError(f"Unknown drone_sampling strategy: {self.drone_sampling}")

        else:
            # Pop next drone from queue
            self.current_drone_id = self.decision_queue.pop(0)

    def _skip_to_next_decision(self) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Skip forward until a decision point appears or episode ends.

        Returns:
            observation, accumulated_reward, terminated, truncated, info
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        obs = self.last_full_obs
        info = {}

        for skip_step in range(self.max_skip_steps):
            # Advance environment with no-op
            dummy_action = np.zeros(self.env.unwrapped.num_drones, dtype=np.int32)
            obs, reward, terminated, truncated, info = self.env.step(dummy_action)

            total_reward += reward
            self.total_skip_steps += 1
            self.episode_steps += 1

            # Check if episode ended
            if terminated or truncated:
                break

            # Check for decision points
            decision_drones = self.env.unwrapped.get_decision_drones()
            if decision_drones:
                # Found decision points
                break

        info['skip_info'] = {
            'steps_skipped': skip_step + 1,
            'reason': 'episode_end' if (terminated or truncated) else 'decision_found'
        }

        return obs, total_reward, terminated, truncated, info

    def _extract_local_observation(
            self,
            full_obs: Optional[Dict],
            drone_id: Optional[int]
    ) -> Dict:
        """
        Extract local observation for a specific drone.

        Args:
            full_obs: Full observation from environment
            drone_id: Drone ID to extract observation for (None = return zeros)

        Returns:
            Local observation dict with keys: drone_state, candidates, global_context
        """
        if full_obs is None or drone_id is None:
            # Return zero observation
            num_candidates = self.observation_space['candidates'].shape[0]
            return {
                'drone_state': np.zeros(8, dtype=np.float32),
                'candidates': np.zeros((num_candidates, 12), dtype=np.float32),
                'global_context': np.zeros(10, dtype=np.float32),
            }

        # Extract drone's own state
        drone_state = full_obs['drones'][drone_id].astype(np.float32)

        # Extract drone's candidates
        candidates = full_obs['candidates'][drone_id].astype(np.float32)

        # Build global context (summary statistics)
        global_context = np.concatenate([
            full_obs['time'],  # 5 dims
            full_obs['day_progress'],  # 1 dim
            full_obs['resource_saturation'],  # 1 dim
            full_obs['weather_details'][:3],  # 3 dims (first 3 weather features)
        ]).astype(np.float32)

        local_obs = {
            'drone_state': drone_state,
            'candidates': candidates,
            'global_context': global_context,
        }

        return local_obs