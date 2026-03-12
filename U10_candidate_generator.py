"""
Candidate Generator for UAV Order Selection

This module provides candidate generation strategies for the layered UAV decision system.
The upper layer generates candidate order sets for each drone, and the lower layer
applies rules to select from these candidates.

Classes:
    CandidateGenerator: Base class for candidate generation
    NearestCandidateGenerator: Generates candidates based on distance
    EarliestDeadlineCandidateGenerator: Generates candidates based on deadline
    MixedHeuristicCandidateGenerator: Combines distance and deadline
    PSOMOPSOCandidateGenerator: Placeholder for future PSO/MOPSO integration
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import numpy as np
import math

if TYPE_CHECKING:
    from UAV_ENVIRONMENT_10 import ThreeObjectiveDroneDeliveryEnv


class CandidateGenerator(ABC):
    """
    Base class for candidate generation strategies.

    Subclasses should implement generate_candidates() to produce
    a dictionary mapping drone_id to a list of candidate order_ids.
    """

    def __init__(self, candidate_k: int = 20):
        """
        Initialize candidate generator.

        Args:
            candidate_k: Number of candidates to generate per drone
        """
        self.candidate_k = candidate_k

    @abstractmethod
    def generate_candidates(
            self,
            env: 'ThreeObjectiveDroneDeliveryEnv'
    ) -> Dict[int, List[int]]:
        """
        Generate candidate order sets for all drones.

        Args:
            env: The UAV environment instance

        Returns:
            Dictionary mapping drone_id to list of order_ids (candidates)
            Each list should contain up to candidate_k order_ids
        """
        pass

    def _get_active_orders(self, env: 'ThreeObjectiveDroneDeliveryEnv') -> List[int]:
        """Helper to get list of active order IDs from environment."""
        return list(env.active_orders)

    def _calculate_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two locations."""
        return math.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)


class NearestCandidateGenerator(CandidateGenerator):
    """Generate candidates based on nearest pickup distance."""

    def generate_candidates(
            self,
            env: 'ThreeObjectiveDroneDeliveryEnv'
    ) -> Dict[int, List[int]]:
        """
        For each drone, select K nearest orders by pickup location distance.

        Args:
            env: The UAV environment instance

        Returns:
            Dictionary mapping drone_id to list of nearest order_ids
        """
        candidates = {}
        active_orders = self._get_active_orders(env)

        for drone_id in range(env.num_drones):
            drone = env.drones[drone_id]
            drone_loc = drone['location']

            # Calculate distances to all active orders
            order_distances = []
            for order_id in active_orders:
                if order_id not in env.orders:
                    continue
                order = env.orders[order_id]
                merchant_loc = order['merchant_location']
                distance = self._calculate_distance(drone_loc, merchant_loc)
                order_distances.append((order_id, distance))

            # Sort by distance and take top K
            order_distances.sort(key=lambda x: x[1])
            candidates[drone_id] = [oid for oid, _ in order_distances[:self.candidate_k]]

        return candidates


class EarliestDeadlineCandidateGenerator(CandidateGenerator):
    """Generate candidates based on earliest delivery deadline."""

    def generate_candidates(
            self,
            env: 'ThreeObjectiveDroneDeliveryEnv'
    ) -> Dict[int, List[int]]:
        """
        For each drone, select K orders with earliest deadlines.

        Args:
            env: The UAV environment instance

        Returns:
            Dictionary mapping drone_id to list of order_ids with earliest deadlines
        """
        candidates = {}
        active_orders = self._get_active_orders(env)

        # Calculate deadlines for all active orders
        order_deadlines = []
        for order_id in active_orders:
            if order_id not in env.orders:
                continue
            order = env.orders[order_id]
            deadline = env._get_delivery_deadline_step(order)
            order_deadlines.append((order_id, deadline))

        # Sort by deadline
        order_deadlines.sort(key=lambda x: x[1])

        # Assign same candidate list to all drones
        # (each drone gets orders with earliest deadlines)
        top_k_orders = [oid for oid, _ in order_deadlines[:self.candidate_k]]
        for drone_id in range(env.num_drones):
            candidates[drone_id] = top_k_orders.copy()

        return candidates


class MixedHeuristicCandidateGenerator(CandidateGenerator):
    """
    Generate candidates using mixed heuristic (distance + deadline).

    This combines distance and deadline using a weighted score.
    """

    def __init__(
            self,
            candidate_k: int = 20,
            distance_weight: float = 0.5,
            deadline_weight: float = 0.5
    ):
        """
        Initialize mixed heuristic generator.

        Args:
            candidate_k: Number of candidates per drone
            distance_weight: Weight for distance component (0-1)
            deadline_weight: Weight for deadline component (0-1)
        """
        super().__init__(candidate_k)
        self.distance_weight = distance_weight
        self.deadline_weight = deadline_weight

    def generate_candidates(
            self,
            env: 'ThreeObjectiveDroneDeliveryEnv'
    ) -> Dict[int, List[int]]:
        """
        For each drone, select K orders based on weighted distance and deadline.

        Score = distance_weight * normalized_distance + deadline_weight * normalized_slack
        Lower score is better.

        Args:
            env: The UAV environment instance

        Returns:
            Dictionary mapping drone_id to list of order_ids
        """
        candidates = {}
        active_orders = self._get_active_orders(env)
        current_step = env.time_system.current_step

        for drone_id in range(env.num_drones):
            drone = env.drones[drone_id]
            drone_loc = drone['location']

            # Calculate scores for all active orders
            order_scores = []

            # First pass: collect raw values for normalization
            distances = []
            slacks = []
            for order_id in active_orders:
                if order_id not in env.orders:
                    continue
                order = env.orders[order_id]
                merchant_loc = order['merchant_location']
                distance = self._calculate_distance(drone_loc, merchant_loc)
                deadline = env._get_delivery_deadline_step(order)
                slack = deadline - current_step

                distances.append(distance)
                slacks.append(slack)

            if not distances:
                candidates[drone_id] = []
                continue

            # Normalize
            max_dist = max(distances) if distances else 1.0
            max_slack = max(slacks) if slacks else 1.0
            min_dist = min(distances) if distances else 0.0
            min_slack = min(slacks) if slacks else 0.0

            dist_range = max_dist - min_dist if max_dist > min_dist else 1.0
            slack_range = max_slack - min_slack if max_slack > min_slack else 1.0

            # Second pass: calculate normalized scores
            idx = 0
            for order_id in active_orders:
                if order_id not in env.orders:
                    continue

                # Normalize distance (0-1, lower is better)
                norm_distance = (distances[idx] - min_dist) / dist_range

                # Normalize slack (0-1, higher slack is worse for urgency)
                # Invert so that lower slack (more urgent) gets higher priority
                norm_urgency = 1.0 - ((slacks[idx] - min_slack) / slack_range)

                # Combined score (lower is better)
                score = (self.distance_weight * norm_distance +
                         self.deadline_weight * norm_urgency)

                order_scores.append((order_id, score))
                idx += 1

            # Sort by score and take top K
            order_scores.sort(key=lambda x: x[1])
            candidates[drone_id] = [oid for oid, _ in order_scores[:self.candidate_k]]

        return candidates


class PSOMOPSOCandidateGenerator(CandidateGenerator):
    """
    Placeholder for PSO/MOPSO-based candidate generation.

    This class provides an interface for future integration with
    Particle Swarm Optimization (PSO) or Multi-Objective PSO (MOPSO)
    algorithms for more sophisticated candidate generation.

    Current implementation falls back to mixed heuristic.
    """

    def __init__(
            self,
            candidate_k: int = 20,
            pso_params: Optional[Dict] = None
    ):
        """
        Initialize PSO/MOPSO generator.

        Args:
            candidate_k: Number of candidates per drone
            pso_params: Dictionary of PSO parameters (for future use)
        """
        super().__init__(candidate_k)
        self.pso_params = pso_params or {}
        # Fallback to mixed heuristic for now
        self._fallback = MixedHeuristicCandidateGenerator(candidate_k)

    def generate_candidates(
            self,
            env: 'ThreeObjectiveDroneDeliveryEnv'
    ) -> Dict[int, List[int]]:
        """
        Generate candidates using PSO/MOPSO (currently falls back to mixed heuristic).

        TODO: Implement actual PSO/MOPSO algorithm

        Args:
            env: The UAV environment instance

        Returns:
            Dictionary mapping drone_id to list of order_ids
        """
        # For now, use mixed heuristic as fallback
        # Future: Implement PSO/MOPSO optimization here
        return self._fallback.generate_candidates(env)


class MOPSOCandidateGenerator(CandidateGenerator):
    """
    MOPSO-based candidate generation for U10.

    Uses MOPSOPlanner from U7_mopso_dispatcher to generate candidate sets.
    This is the UPPER LAYER that only generates candidates - it does NOT
    commit orders (READY -> ASSIGNED). The actual assignment is handled by
    the lower layer (rule selection via EventDrivenSingleUAVWrapper).

    Key features:
    - Generates K candidates per drone using MOPSO optimization
    - Only uses READY, unassigned orders as input
    - Ensures no duplicate orders within a drone's candidate set
    - Allows same order across multiple drones' candidates for robustness
    - Falls back to heuristics when MOPSO returns insufficient candidates
    """

    def __init__(
            self,
            candidate_k: int = 20,
            n_particles: int = 30,
            n_iterations: int = 10,
            max_orders: int = 200,
            max_orders_per_drone: int = 10,
            seed: Optional[int] = None,
            **mopso_kwargs
    ):
        """
        Initialize MOPSO candidate generator.

        Args:
            candidate_k: Number of candidates per drone (K)
            n_particles: Number of MOPSO particles
            n_iterations: Number of MOPSO iterations
            max_orders: Maximum orders for MOPSO to consider
            max_orders_per_drone: Maximum orders per drone in MOPSO assignment
            seed: Random seed
            **mopso_kwargs: Additional arguments for MOPSOPlanner

        Note:
            Order sharing across drones' candidate sets is always allowed to provide
            robustness. This allows multiple drones to consider the same order, with
            the lower layer (rule selection) making the final choice.
        """
        super().__init__(candidate_k)
        self.max_orders = max_orders
        self.seed = seed

        # Import MOPSOPlanner
        try:
            from U7_mopso_dispatcher import MOPSOPlanner
            self.planner = MOPSOPlanner(
                n_particles=n_particles,
                n_iterations=n_iterations,
                max_orders=max_orders,
                max_orders_per_drone=max_orders_per_drone,
                seed=seed,
                **mopso_kwargs
            )
        except ImportError as e:
            raise ImportError(
                f"Failed to import MOPSOPlanner from U7_mopso_dispatcher: {e}. "
                "Make sure U7_mopso_dispatcher.py and U6_mopso_dispatcher.py are available."
            )

        # Fallback generator for padding
        self._fallback = MixedHeuristicCandidateGenerator(candidate_k)

    def generate_candidates(
            self,
            env: 'ThreeObjectiveDroneDeliveryEnv'
    ) -> Dict[int, List[int]]:
        """
        Generate candidate order sets for all drones using MOPSO.

        This method:
        1. Gets READY unassigned orders from environment
        2. Runs MOPSOPlanner._run_mopso to get assignment suggestions
        3. Extracts order_ids from MOPSO output for each drone
        4. Pads/truncates to exactly K candidates per drone
        5. Falls back to heuristics if MOPSO returns insufficient candidates

        Args:
            env: The UAV environment instance

        Returns:
            Dictionary mapping drone_id to list of order_ids (candidates)
            Each list contains up to candidate_k order_ids
        """
        # Get snapshots from environment
        ready_orders = env.get_ready_orders_snapshot(limit=self.max_orders)
        drones = env.get_drones_snapshot()
        merchants = env.get_merchants_snapshot()
        constraints = env.get_route_plan_constraints()
        objective_weights = getattr(env, 'objective_weights', np.array([0.33, 0.33, 0.34]))

        # Filter drones with capacity
        available_drones = []
        for d in drones:
            current_load = d.get('current_load', 0)
            max_capacity = d.get('max_capacity', 10)
            if current_load < max_capacity:
                available_drones.append(d)

        # If no drones or orders, return empty candidates
        if not available_drones or not ready_orders:
            return {drone_id: [] for drone_id in range(env.num_drones)}

        # Run MOPSO to get assignment (Dict[int, List[int]])
        try:
            mopso_assignment = self.planner._run_mopso(
                ready_orders[:self.max_orders],
                available_drones,
                merchants,
                constraints,
                objective_weights
            )
        except Exception as e:
            # If MOPSO fails, fall back to heuristic
            import warnings
            warnings.warn(f"MOPSO candidate generation failed: {e}. Falling back to heuristic.")
            return self._fallback.generate_candidates(env)

        # Build candidate sets
        candidates = {}
        ready_order_ids = {o['order_id'] for o in ready_orders}

        for drone_id in range(env.num_drones):
            # Get MOPSO suggestions for this drone
            mopso_orders = mopso_assignment.get(drone_id, [])

            # Filter to ensure orders are still READY and unassigned
            valid_orders = []
            seen = set()
            for oid in mopso_orders:
                if oid in ready_order_ids and oid not in seen:
                    valid_orders.append(oid)
                    seen.add(oid)

            # Pad if insufficient candidates
            if len(valid_orders) < self.candidate_k:
                valid_orders = self._pad_candidates(
                    env, drone_id, valid_orders, ready_order_ids, seen
                )

            # Truncate to K
            candidates[drone_id] = valid_orders[:self.candidate_k]

        return candidates

    def _pad_candidates(
            self,
            env: 'ThreeObjectiveDroneDeliveryEnv',
            drone_id: int,
            current_candidates: List[int],
            ready_order_ids: set,
            seen_orders: set
    ) -> List[int]:
        """
        Pad candidate list with heuristic choices when MOPSO returns insufficient candidates.

        Uses nearest distance heuristic to fill remaining slots.

        Args:
            env: Environment instance
            drone_id: Drone ID
            current_candidates: Current list of candidate order IDs
            ready_order_ids: Set of all available READY order IDs
            seen_orders: Set of order IDs already in candidates

        Returns:
            Padded list of candidate order IDs
        """
        drone = env.drones[drone_id]
        drone_loc = drone['location']

        # Calculate distances to all remaining ready orders
        order_distances = []
        for oid in ready_order_ids:
            if oid in seen_orders:
                continue

            # Safely access order
            order = env.orders.get(oid)
            if order is None:
                continue

            merchant_loc = order['merchant_location']
            distance = self._calculate_distance(drone_loc, merchant_loc)
            order_distances.append((oid, distance))

        # Sort by distance
        order_distances.sort(key=lambda x: x[1])

        # Add nearest orders until we have K candidates
        padded_candidates = current_candidates.copy()
        for oid, _ in order_distances:
            if len(padded_candidates) >= self.candidate_k:
                break
            padded_candidates.append(oid)
            seen_orders.add(oid)

        return padded_candidates