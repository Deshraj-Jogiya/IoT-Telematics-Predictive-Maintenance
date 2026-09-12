import numpy as np

from models.jax_health_decay import compute_health_and_rul_jax


def _reference_python_loop(dt_days, anomaly_flags, alpha, beta, lambda_est):
    """The exact recurrence from predictive_maintenance.py's per-row loop,
    reproduced here as a plain Python reference to check the Jax version
    against."""
    fail_health = 25.0
    current_health = 100.0
    health_scores = []
    rul_days_list = []
    for dt, anomaly in zip(dt_days, anomaly_flags):
        decay_factor = np.exp(-(alpha * dt + beta * anomaly))
        current_health = current_health * decay_factor
        current_health = max(10.0, min(100.0, current_health))
        health_scores.append(current_health)

        if current_health <= fail_health:
            rul = 0.0
        else:
            rul = np.log(current_health / fail_health) / lambda_est
        rul = max(0.0, min(90.0, rul))
        rul_days_list.append(rul)
    return np.array(health_scores), np.array(rul_days_list)


def test_jax_scan_matches_reference_python_loop():
    rng = np.random.default_rng(42)
    dt_days = rng.uniform(0.01, 2.0, size=200)
    anomaly_flags = (rng.random(200) < 0.1).astype(float)
    alpha, beta, lambda_est = 0.009, 0.06, 0.018  # DEV-003's real params

    expected_health, expected_rul = _reference_python_loop(dt_days, anomaly_flags, alpha, beta, lambda_est)
    jax_health, jax_rul = compute_health_and_rul_jax(dt_days, anomaly_flags, alpha, beta, lambda_est)

    np.testing.assert_allclose(jax_health, expected_health, rtol=1e-5)
    np.testing.assert_allclose(jax_rul, expected_rul, rtol=1e-4)


def test_health_never_leaves_bounds():
    dt_days = np.full(50, 5.0)  # large gaps -> aggressive decay
    anomaly_flags = np.ones(50)
    health, rul = compute_health_and_rul_jax(dt_days, anomaly_flags, alpha=0.009, beta=0.06, lambda_est=0.018)
    assert (health >= 10.0).all() and (health <= 100.0).all()
    assert (rul >= 0.0).all() and (rul <= 90.0).all()
