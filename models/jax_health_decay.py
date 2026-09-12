"""A Jax-accelerated reimplementation of predictive_maintenance.py's
per-row Python loop for the exponential health-decay/RUL recurrence
(Health(t) = clamp(Health(t-1) * exp(-alpha*dt - beta*anomaly))).

That loop is a genuine sequential recurrence -- each step depends on
the previous health value -- which is exactly what jax.lax.scan is
for: a single JIT-compiled scan over the whole device history instead
of a Python-level iterrows() loop. Verified in
tests/test_jax_health_decay.py to produce numerically identical
results to the original implementation.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

FAIL_HEALTH = 25.0
MIN_HEALTH = 10.0
MAX_HEALTH = 100.0
MAX_RUL_DAYS = 90.0


@partial(jax.jit, static_argnames=())
def _decay_step(current_health, step_input):
    dt_days, anomaly, alpha, beta = step_input
    decay_factor = jnp.exp(-(alpha * dt_days + beta * anomaly))
    new_health = jnp.clip(current_health * decay_factor, MIN_HEALTH, MAX_HEALTH)
    return new_health, new_health


def _health_to_rul(health, lambda_est):
    rul = jnp.where(
        health <= FAIL_HEALTH,
        0.0,
        jnp.log(health / FAIL_HEALTH) / lambda_est,
    )
    return jnp.clip(rul, 0.0, MAX_RUL_DAYS)


def compute_health_and_rul_jax(dt_days: np.ndarray, anomaly_flags: np.ndarray, alpha: float, beta: float, lambda_est: float):
    """Runs the health-decay recurrence for one device's full history in
    a single compiled jax.lax.scan call, returning (health_scores, rul_days)
    as numpy arrays -- a drop-in replacement for the per-row Python loop
    in compute_anomalies_and_health() for the same device."""
    dt_days = jnp.asarray(dt_days, dtype=jnp.float32)
    anomaly_flags = jnp.asarray(anomaly_flags, dtype=jnp.float32)
    alpha_arr = jnp.full_like(dt_days, alpha)
    beta_arr = jnp.full_like(dt_days, beta)

    step_inputs = (dt_days, anomaly_flags, alpha_arr, beta_arr)
    _, health_scores = jax.lax.scan(_decay_step, jnp.asarray(100.0, dtype=jnp.float32), step_inputs)

    rul_days = jax.vmap(lambda h: _health_to_rul(h, lambda_est))(health_scores)

    return np.asarray(health_scores), np.asarray(rul_days)
