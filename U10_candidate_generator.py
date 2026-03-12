"""
Candidate Generator for Upper-Layer Task Assignment

This module provides the MOPSOCandidateGenerator used by the upper layer of the
hierarchical UAV scheduling system to suggest promising candidate order sets for
each drone.  The candidate sets constrain the lower-layer PPO rule selection so
that each drone only considers a focused subset of available orders.

The full MOPSO (Multi-Objective Particle Swarm Optimisation) implementation is
omitted here; this file supplies a functional passthrough fallback that returns
all active, unfinished orders as candidates for every drone.  Replace the body of
``generate_candidates`` with a real MOPSO implementation when available.
"""

from typing import Dict, List, Optional, Any


class MOPSOCandidateGenerator:
    """
    Upper-layer candidate generator (MOPSO-based).

    This implementation acts as a passthrough fallback: it returns all active
    orders as candidates for every drone.  Swap in a real MOPSO solver by
    overriding ``generate_candidates``.

    Args:
        candidate_k: Maximum candidates to return per drone.
        n_particles: (MOPSO) number of particles (informational, unused here).
        n_iterations: (MOPSO) iterations per call (informational, unused here).
        max_orders: Max orders fed to the optimiser (informational, unused here).
        max_orders_per_drone: Hard cap on candidates per drone.
        seed: Random seed.
        eta_speed_scale_assumption: Assumed speed fraction for ETA estimation.
        eta_stop_service_steps: Assumed service-stop overhead in steps.
    """

    def __init__(
            self,
            candidate_k: int = 20,
            n_particles: int = 30,
            n_iterations: int = 10,
            max_orders: int = 200,
            max_orders_per_drone: int = 10,
            seed: int = 42,
            eta_speed_scale_assumption: float = 0.7,
            eta_stop_service_steps: int = 1,
    ):
        self.candidate_k = candidate_k
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.max_orders = max_orders
        self.max_orders_per_drone = min(max_orders_per_drone, candidate_k)
        self.seed = seed
        self.eta_speed_scale_assumption = eta_speed_scale_assumption
        self.eta_stop_service_steps = eta_stop_service_steps

    def generate_candidates(self, env) -> Dict[int, List[int]]:
        """
        Generate candidate order sets for all drones.

        Fallback implementation: returns the K highest-priority active orders
        (urgent first, then by creation time) for every drone.

        Args:
            env: The UAV environment instance (ThreeObjectiveDroneDeliveryEnv).
                 Must expose ``active_orders``, ``orders``, ``num_drones``.

        Returns:
            Dict mapping drone_id -> list of order_ids (length ≤ candidate_k).
        """
        active_ids = list(env.active_orders)

        # Prioritise urgent orders and older orders
        def _priority(oid):
            o = env.orders.get(oid, {})
            urgent_score = 1000 if o.get('urgent', False) else 0
            age = env.time_system.current_step - o.get('creation_time', 0)
            return -(urgent_score + age)

        active_ids.sort(key=_priority)
        top_ids = active_ids[:min(self.max_orders, len(active_ids))]

        # Assign the same top-K list to every drone (simple fallback)
        k = min(self.candidate_k, len(top_ids))
        result: Dict[int, List[int]] = {}
        for drone_id in range(env.num_drones):
            result[drone_id] = top_ids[:k]

        return result
