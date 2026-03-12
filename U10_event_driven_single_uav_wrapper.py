"""
Event-Driven Single UAV Wrapper (central-queue compatibility mode)

This wrapper is a compatibility shim for the ``central_queue`` training mode.
It exposes the same interface as ``SingleUAVTrainingWrapper`` but processes
drones in strict FIFO order through a centralised queue rather than sampling
them randomly.

Observation:
    Rule-discriminant compact state vector (20-dim Box, values in [0, 1]).
    Identical to the observation produced by SingleUAVTrainingWrapper.
    See env._get_rule_based_state_for_drone() for the feature layout.

Action:
    Discrete(R) — one rule_id for the currently-queued drone.

Usage:
    env = ThreeObjectiveDroneDeliveryEnv(...)
    env = EventDrivenSingleUAVWrapper(env, max_skip_steps=10)
    model = PPO("MlpPolicy", env, ...)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class EventDrivenSingleUAVWrapper(gym.Wrapper):
    """
    Central-queue variant of the single-UAV training wrapper.

    Drones at decision points are queued in FIFO order (sorted by drone_id).
    The wrapper advances the environment whenever the queue is empty and the
    episode has not ended.

    Args:
        env: The base UAV environment.
        max_skip_steps: Maximum environment steps to skip while waiting for
                        the next decision event.
        local_observation: Retained for API compatibility; observation is always
                           the compact rule-based state vector.
    """

    def __init__(
            self,
            env: gym.Env,
            max_skip_steps: int = 10,
            local_observation: bool = True,
    ):
        super().__init__(env)

        self.max_skip_steps = max_skip_steps
        self.local_observation = local_observation

        # Action space: single discrete rule_id
        rule_count = getattr(env.unwrapped, 'rule_count', 5)
        self.action_space = spaces.Discrete(rule_count)

        # Compact rule-based observation space (flat Box)
        state_dim = getattr(env.unwrapped, 'RULE_BASED_STATE_DIM', 20)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )

        # Internal state
        self.current_drone_id: Optional[int] = None
        self.decision_queue: List[int] = []
        self.last_full_obs: Optional[Dict] = None
        self.last_info: Optional[Dict] = None

        # Statistics
        self.total_decisions = 0
        self.total_skip_steps = 0
        self.episode_steps = 0

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        self.last_full_obs = obs
        self.last_info = info

        self.current_drone_id = None
        self.decision_queue = []
        self.total_decisions = 0
        self.total_skip_steps = 0
        self.episode_steps = 0

        self._populate_decision_queue()

        if self.current_drone_id is None:
            obs, _, _, _, info = self._skip_to_next_decision()
            self.last_full_obs = obs
            self.last_info = info
            self._populate_decision_queue()

        local_obs = self._get_local_obs(self.current_drone_id)

        info['wrapper'] = {
            'current_drone_id': self.current_drone_id,
            'queue_length': len(self.decision_queue),
            'total_decisions': self.total_decisions,
            'total_skip_steps': self.total_skip_steps,
        }
        return local_obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.current_drone_id is None:
            raise RuntimeError("No current drone queued. Call reset() first.")

        drone_id = self.current_drone_id
        success = self.env.unwrapped.apply_rule_to_drone(drone_id, action)

        dummy_action = np.zeros(self.env.unwrapped.num_drones, dtype=np.int32)
        obs, reward, terminated, truncated, info = self.env.step(dummy_action)
        self.last_full_obs = obs
        self.last_info = info
        self.total_decisions += 1
        self.episode_steps += 1

        self.current_drone_id = None
        self._populate_decision_queue()

        if self.current_drone_id is None and not (terminated or truncated):
            skip_obs, skip_reward, terminated, truncated, skip_info = \
                self._skip_to_next_decision()
            obs = skip_obs
            reward += skip_reward
            info.update(skip_info)
            self.last_full_obs = obs
            self.last_info = info
            self._populate_decision_queue()

        local_obs = self._get_local_obs(self.current_drone_id)

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _populate_decision_queue(self):
        """Fill decision queue with drones currently at decision points (FIFO)."""
        if not self.decision_queue:
            decision_drones = sorted(self.env.unwrapped.get_decision_drones())
            if not decision_drones:
                self.current_drone_id = None
                return
            self.current_drone_id = decision_drones[0]
            self.decision_queue = decision_drones[1:]
        else:
            self.current_drone_id = self.decision_queue.pop(0)

    def _skip_to_next_decision(self) -> Tuple[Dict, float, bool, bool, Dict]:
        """Advance environment until a decision event or episode end."""
        total_reward = 0.0
        terminated = False
        truncated = False
        obs = self.last_full_obs
        info = {}

        for skip_step in range(self.max_skip_steps):
            dummy_action = np.zeros(self.env.unwrapped.num_drones, dtype=np.int32)
            obs, reward, terminated, truncated, info = self.env.step(dummy_action)
            total_reward += reward
            self.total_skip_steps += 1
            self.episode_steps += 1

            if terminated or truncated:
                break

            if self.env.unwrapped.get_decision_drones():
                break

        info['skip_info'] = {
            'steps_skipped': skip_step + 1,
            'reason': 'episode_end' if (terminated or truncated) else 'decision_found',
        }
        return obs, total_reward, terminated, truncated, info

    def _get_local_obs(self, drone_id: Optional[int]) -> np.ndarray:
        """Return compact rule-based state for drone_id, or zeros if None."""
        state_dim = self.observation_space.shape[0]
        if drone_id is None:
            return np.zeros(state_dim, dtype=np.float32)
        return self.env.unwrapped._get_rule_based_state_for_drone(drone_id)
