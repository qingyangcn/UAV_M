"""
Test: deterministic order generation under fixed seed.

Verifies that:
1. Running two episodes with the same seed produces identical generated_total
   and the same creation_step sequence for all orders.
2. Enabling/disabling an external candidate generator does NOT affect the
   order generation sequence (env RNG is isolated).
"""

import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from UAV_ENVIRONMENT_11 import ThreeObjectiveDroneDeliveryEnv
from U11_decentralized_execution import DecentralizedEventDrivenExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_small_env(**kwargs) -> ThreeObjectiveDroneDeliveryEnv:
    """Create a small environment for fast testing."""
    defaults = dict(
        grid_size=16,
        num_drones=4,
        max_orders=100,
        num_bases=2,
        steps_per_hour=12,
        drone_max_capacity=10,
        top_k_merchants=20,
        reward_output_mode="scalar",
        enable_random_events=False,
        debug_state_warnings=False,
        fixed_objective_weights=(0.5, 0.3, 0.2),
        num_candidates=10,
        rule_count=5,
        enable_diagnostics=False,
        energy_e0=0.1,
        energy_alpha=0.5,
        battery_return_threshold=10.0,
        multi_objective_mode="fixed",
        candidate_update_interval=8,
        candidate_fallback_enabled=False,
        order_cutoff_steps=0,
    )
    defaults.update(kwargs)
    return ThreeObjectiveDroneDeliveryEnv(**defaults)


def _noop_policy(local_obs: dict) -> int:
    """Policy that always returns rule 0 (deterministic, no global RNG use)."""
    return 0


def _run_episode(seed: int, max_steps: int = 50, candidate_generator=None) -> dict:
    """Run a short episode and return reproducibility metrics."""
    env = _make_small_env()

    if candidate_generator is not None:
        env.set_candidate_generator(candidate_generator)

    executor = DecentralizedEventDrivenExecutor(
        env=env,
        policy_fn=_noop_policy,
        max_skip_steps=1,
        verbose=False,
    )
    executor.run_episode(max_steps=max_steps, seed=seed)

    generated_total = env.daily_stats['orders_generated']

    # Build a hash of all creation_step values in order-id order
    creation_steps = [
        env.orders[oid]['creation_step']
        for oid in sorted(env.orders.keys())
    ]
    seq_hash = hashlib.md5(str(creation_steps).encode()).hexdigest()

    return {'generated_total': generated_total, 'seq_hash': seq_hash}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Same seed → same order generation."""

    def test_same_seed_same_generated_total(self):
        """Two runs with identical seed must produce the same generated_total."""
        run1 = _run_episode(seed=42)
        run2 = _run_episode(seed=42)
        assert run1['generated_total'] == run2['generated_total'], (
            f"generated_total differs between runs with seed=42: "
            f"{run1['generated_total']} vs {run2['generated_total']}"
        )

    def test_same_seed_same_creation_sequence(self):
        """Two runs with identical seed must produce the same order creation_step sequence."""
        run1 = _run_episode(seed=42)
        run2 = _run_episode(seed=42)
        assert run1['seq_hash'] == run2['seq_hash'], (
            "Order creation_step sequence hash differs between runs with seed=42"
        )

    def test_different_seeds_different_results(self):
        """Different seeds should (almost certainly) produce different results."""
        run_a = _run_episode(seed=1)
        run_b = _run_episode(seed=9999)
        # Not a strict requirement but a sanity check—if they're equal that's suspicious
        # Use seq_hash for comparison since generated_total could coincidentally match
        assert run_a['seq_hash'] != run_b['seq_hash'], (
            "Different seeds produced identical order sequences (very unlikely if RNG works)"
        )

    def test_seed_isolation_from_external_generator(self):
        """Env order generation must be independent of whether a candidate generator is set.

        With the same env seed, generated_total and the creation_step sequence
        should be the same whether or not an external candidate generator is attached.
        """

        class _DummyGenerator:
            """Minimal stub that uses its own numpy global RNG heavily."""

            def generate_candidates(self, *args, **kwargs):
                import numpy as np
                # Consume from global RNG to simulate MOPSO interference
                _ = np.random.rand(1000)
                return {}

        dummy_gen = _DummyGenerator()

        run_no_gen = _run_episode(seed=7)
        run_with_gen = _run_episode(seed=7, candidate_generator=dummy_gen)

        assert run_no_gen['generated_total'] == run_with_gen['generated_total'], (
            f"generated_total changed when external generator was enabled: "
            f"no_gen={run_no_gen['generated_total']}, "
            f"with_gen={run_with_gen['generated_total']}"
        )
        assert run_no_gen['seq_hash'] == run_with_gen['seq_hash'], (
            "Order creation_step sequence changed when external generator was enabled"
        )
