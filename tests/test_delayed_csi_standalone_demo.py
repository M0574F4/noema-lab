from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = (
    ROOT
    / "demo_trainings"
    / "delayed_csi_predictive_allocation_standalone"
    / "demo.py"
)


def _load_demo():
    spec = importlib.util.spec_from_file_location("delayed_csi_standalone_demo", DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_channel_is_normalized_correlated_and_markov_valid():
    demo = _load_demo()
    _, gains, transition = demo.state_tables()
    assert np.allclose(np.mean(gains, axis=1), 1.0, atol=1e-12)
    assert np.allclose(np.sum(transition, axis=1), 1.0, atol=1e-12)
    batch = demo._trajectory_batch(seed_start=991001, trajectories=20, decisions=64)
    correlations = demo.delayed_correlations(batch)
    assert correlations["delayed_current_complex_correlation_magnitude"] > 0.75
    assert correlations["delayed_current_gain_correlation"] > 0.55


def test_water_filling_is_feasible_and_dominates_equal_power():
    demo = _load_demo()
    gains = np.random.default_rng(17).lognormal(size=(40, 16))
    water = demo.water_filling(gains)
    equal = np.ones_like(water)
    assert np.max(np.abs(np.sum(water, axis=1) - 16.0)) < 1e-9
    assert np.min(water) >= 0.0
    assert np.all(
        demo.spectral_efficiency(gains, water)
        >= demo.spectral_efficiency(gains, equal) - 1e-11
    )


def test_learned_policy_is_causal_and_power_feasible():
    demo = _load_demo()
    history, _ = demo.sample_independent_windows(24, 991101)
    model = demo.DelayedCsiPolicy()
    before = demo.learned_power(model, history)
    # Current/future channel values are deliberately unrelated to the policy API.
    current_and_future = np.random.default_rng(991102).normal(size=(24, 9, 16))
    current_and_future *= 1e6
    after = demo.learned_power(model, history.copy())
    assert current_and_future.shape == (24, 9, 16)
    assert np.array_equal(before, after)
    assert np.max(np.abs(np.sum(before, axis=1) - 16.0)) < 1e-5
    assert np.min(before) >= 0.0

    batch = demo._trajectory_batch(seed_start=991111, trajectories=2, decisions=5)
    evaluator_before = demo._method_powers(model, batch)["learned_predictive_allocator"]
    changed = dict(batch)
    changed["current_gains"] = np.flip(batch["current_gains"], axis=-1).copy() * 1e5
    changed["current_states"] = np.flip(batch["current_states"], axis=-1).copy()
    evaluator_after = demo._method_powers(model, changed)["learned_predictive_allocator"]
    assert np.array_equal(evaluator_before, evaluator_after)


def test_old_state_estimator_recovers_clean_two_snapshot_history():
    demo = _load_demo()
    response, _, _ = demo.state_tables()
    q_count = demo.CONFIG["channel"]["phase_states"]
    step = demo.CONFIG["channel"]["phase_step_states_per_slot"]
    states = np.arange(2 * q_count, dtype=np.int64)
    q = states // 2
    velocity = np.where(states % 2 == 0, -1, 1)
    previous_q = (q - step * velocity) % q_count
    history = np.stack((response[previous_q], response[q]), axis=1)
    estimated = demo.estimate_old_states(history)
    assert np.array_equal(estimated, states)
