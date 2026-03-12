"""
Event-Driven Single UAV Wrapper

This wrapper transforms the UAV environment from MultiDiscrete action space (one action per drone)
to a single-drone event-driven decision-making interface with Discrete(5) action space.

Key features:
1. Action space: Discrete(5) - single rule_id for one drone at a time
2. Observation: Homogeneous local observation for the current decision drone
3. Decision queue: Processes drones at decision points one by one
4. Time stepping: Automatically advances environment when no drones at decision points

This enables:
- Homogeneous policy parameter sharing across drones
- Action space independent of number of drones N
- Event-driven decision making (only act when needed)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Any, Tuple
from collections import deque


class EventDrivenSingleUAVWrapper(gym.Wrapper):
    """
    Event-driven single UAV decision wrapper.

    Converts MultiDiscrete([R]*N) action space to Discrete(R) by:
    1. Maintaining a decision queue of drones at decision points
    2. Each step() call processes one drone from the queue
    3. When queue is empty, advances the underlying environment

    Args:
        env: The base UAV environment
        max_skip_steps: Maximum number of environment steps to skip when
                       waiting for decision points (default: 10)
        local_observation: If True, extract local observation for current drone.
                          If False, return full observation with current_drone_id
                          (default: False, for gradual migration)
    """

    def __init__(
            self,
            env: gym.Env,
            max_skip_steps: int = 10,
            local_observation: bool = False
    ):
        super().__init__(env)

        self.max_skip_steps = max_skip_steps
        self.local_observation = local_observation

        # Get number of rules from environment
        if hasattr(env.unwrapped, 'rule_count'):
            self.rule_count = env.unwrapped.rule_count
        else:
            self.rule_count = 5  # Default

        # Override action space to single discrete action
        self.action_space = spaces.Discrete(self.rule_count)

        # Decision queue: stores drone IDs that need decisions
        self.decision_queue: deque = deque()

        # Current drone being processed
        self.current_drone_id: Optional[int] = None

        # Track last observation and info from environment
        self.last_obs: Optional[Dict] = None
        self.last_info: Optional[Dict] = None

        # Statistics
        self.total_decisions = 0
        self.total_skips = 0

        # Override observation space
        if self.local_observation:
            # TODO: Define local observation space
            # For now, keep original observation space
            pass
        else:
            # Add current_drone_id to observation space
            # Get original observation space
            original_spaces = dict(self.env.observation_space.spaces)
            # Add current_drone_id field
            original_spaces['current_drone_id'] = spaces.Box(
                low=-1, high=env.unwrapped.num_drones, shape=(1,), dtype=np.int32
            )
            # Override observation space with extended version
            self.observation_space = spaces.Dict(original_spaces)

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """
        Reset environment and initialize decision queue.

        Returns:
            observation: Observation for first decision drone
            info: Info dictionary with decision metadata
        """
        # Reset underlying environment
        obs, info = self.env.reset(**kwargs)

        # Store observation
        self.last_obs = obs
        self.last_info = info

        # Clear decision queue
        self.decision_queue.clear()

        # Populate initial decision queue
        self._populate_decision_queue()

        # Reset statistics
        self.total_decisions = 0
        self.total_skips = 0

        # Get first observation
        obs_out, info_out = self._get_current_observation()

        return obs_out, info_out

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one decision for the current drone.

        Args:
            action: Rule ID (0-4) to apply to current drone

        Returns:
            observation: Next observation
            reward: Reward from environment
            terminated: Whether episode terminated
            truncated: Whether episode truncated
            info: Info dictionary
        """
        # If no current drone, populate queue first
        if self.current_drone_id is None:
            self._populate_decision_queue()

            # If still no drones, skip environment forward
            if not self.decision_queue:
                obs, reward, terminated, truncated, info = self._skip_to_next_decision()
                return obs, reward, terminated, truncated, info

        # Apply action to current drone
        drone_id = self.current_drone_id

        # Create action array for underlying environment
        # All drones get action 0 (no-op or default), except current drone
        action_array = np.zeros(self.env.unwrapped.num_drones, dtype=np.int32)
        action_array[drone_id] = action

        # Step underlying environment with single-drone action
        # Note: We need to call apply_rule_to_drone directly to avoid
        # affecting other drones
        success = self.env.unwrapped.apply_rule_to_drone(drone_id, action)

        # Advance environment one time step to process events
        # Use a dummy action (all zeros) to just advance time
        dummy_action = np.zeros(self.env.unwrapped.num_drones, dtype=np.int32)
        obs, reward, terminated, truncated, info = self.env.step(dummy_action)

        # Store observation
        self.last_obs = obs
        self.last_info = info

        # Update statistics
        self.total_decisions += 1

        # Move to next drone in queue
        self.current_drone_id = None
        self._populate_decision_queue()

        # If no more drones in queue and episode not done, skip forward
        if not self.decision_queue and not (terminated or truncated):
            obs, reward, terminated, truncated, info = self._skip_to_next_decision()

        # Get observation for next drone
        obs_out, info_out = self._get_current_observation()

        # Merge info
        info_out.update(info)

        return obs_out, reward, terminated, truncated, info_out

    def _populate_decision_queue(self):
        """
        Populate decision queue with drones at decision points.
        Uses the environment's get_decision_drones() method.
        """
        if self.decision_queue:
            # Queue already has drones, pop next one
            if self.decision_queue:
                self.current_drone_id = self.decision_queue.popleft()
            return

        # Get drones at decision points from environment
        decision_drones = self.env.unwrapped.get_decision_drones()

        # Add to queue
        for drone_id in decision_drones:
            self.decision_queue.append(drone_id)

        # Set current drone
        if self.decision_queue:
            self.current_drone_id = self.decision_queue.popleft()
        else:
            self.current_drone_id = None

    def _skip_to_next_decision(self) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Skip forward in the environment until a decision point appears
        or max_skip_steps is reached or episode ends.

        Returns:
            observation, reward, terminated, truncated, info
        """
        total_reward = 0.0
        terminated = False
        truncated = False

        for skip_step in range(self.max_skip_steps):
            # Advance environment with no-op actions
            dummy_action = np.zeros(self.env.unwrapped.num_drones, dtype=np.int32)
            obs, reward, terminated, truncated, info = self.env.step(dummy_action)

            # Accumulate reward
            total_reward += reward

            # Store observation
            self.last_obs = obs
            self.last_info = info

            # Update skip count
            self.total_skips += 1

            # Check if episode ended
            if terminated or truncated:
                break

            # Check for decision points
            decision_drones = self.env.unwrapped.get_decision_drones()
            if decision_drones:
                # Found decision points, populate queue
                for drone_id in decision_drones:
                    self.decision_queue.append(drone_id)
                if self.decision_queue:
                    self.current_drone_id = self.decision_queue.popleft()
                break

        # Return observation via _get_current_observation to ensure consistency
        obs_out, info_out = self._get_current_observation()
        return obs_out, total_reward, terminated, truncated, info_out

    def _get_current_observation(self) -> Tuple[Dict, Dict]:
        """
        Get observation for current drone.

        Returns:
            observation: Observation dict
            info: Info dict with metadata
        """
        if self.last_obs is None:
            # No observation yet, return empty
            return {}, {}

        obs = self.last_obs.copy()
        info = self.last_info.copy() if self.last_info else {}

        if self.local_observation:
            # TODO: Extract local observation for current drone
            # For now, add current_drone_id to observation
            obs = self._extract_local_observation(obs, self.current_drone_id)
        else:
            # Add current_drone_id to observation for context
            obs['current_drone_id'] = np.array([self.current_drone_id if self.current_drone_id is not None else -1],
                                               dtype=np.int32)

        # Add decision queue info to metadata
        info['decision_queue_length'] = len(self.decision_queue)
        info['current_drone_id'] = self.current_drone_id if self.current_drone_id is not None else -1
        info['total_decisions'] = self.total_decisions
        info['total_skips'] = self.total_skips

        return obs, info

    def _extract_local_observation(self, obs: Dict, drone_id: Optional[int]) -> Dict:
        """
        Extract local observation for a specific drone.

        This creates a homogeneous observation that includes:
        - Drone's own state
        - Global statistics summary
        - Candidate orders features

        Args:
            obs: Full observation from environment
            drone_id: Drone ID to extract observation for

        Returns:
            Local observation dict
        """
        if drone_id is None or drone_id < 0:
            # No valid drone, return minimal observation
            return {
                'drone_state': np.zeros(8, dtype=np.float32),
                'candidates': np.zeros((self.env.unwrapped.num_candidates, 12), dtype=np.float32),
                'global_context': np.zeros(10, dtype=np.float32),
            }

        # Extract drone's own state
        drone_state = obs['drones'][drone_id]

        # Extract drone's candidates
        candidates = obs['candidates'][drone_id]

        # Build global context (summary statistics)
        global_context = np.concatenate([
            obs['time'],  # 5 dims
            obs['day_progress'],  # 1 dim
            obs['resource_saturation'],  # 1 dim
            obs['weather_details'][:3],  # 3 dims (first 3 weather features)
        ])

        local_obs = {
            'drone_state': drone_state,
            'candidates': candidates,
            'global_context': global_context,
        }

        return local_obs

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get wrapper statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'total_decisions': self.total_decisions,
            'total_skips': self.total_skips,
            'queue_length': len(self.decision_queue),
            'current_drone_id': self.current_drone_id,
        }