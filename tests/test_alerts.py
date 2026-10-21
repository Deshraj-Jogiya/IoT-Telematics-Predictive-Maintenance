import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
import predictive_maintenance as pm  # noqa: E402


def _readings(hours, device="DEV-003", vib=None):
    ts = pd.date_range("2026-03-15", periods=hours, freq="h")
    n = len(ts)
    return pd.DataFrame({
        "log_id": range(1, n + 1), "device_id": device, "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "battery_temp_c": 35.0, "motor_vibration_g": vib if vib is not None else np.full(n, 0.5),
        "voltage": 400.0, "rpm": 3000.0, "health_score": 100.0, "anomaly_flag": 0,
    })


def test_a_persisting_condition_alerts_at_most_once_every_four_hours():
    # DEV-003 wears fast, so it reaches critical health and stays there
    df, alerts = pm.compute_anomalies_and_health(_readings(24 * 200))
    critical = pd.to_datetime(pd.Series([a["alert_timestamp"] for a in alerts if a["alert_type"] == "Critical Health Degradation"]))
    assert len(critical) > 1
    assert critical.diff().dropna().min() >= pm.ALERT_REPEAT_INTERVAL


def test_both_rul_alert_levels_are_raised_as_health_falls():
    df, alerts = pm.compute_anomalies_and_health(_readings(24 * 200))
    types = [a["alert_type"] for a in alerts]
    first_of_each = {t: types.index(t) for t in set(types)}
    assert "Critical Health Degradation" in first_of_each and "Low Remaining Useful Life" in first_of_each


def test_detector_never_reads_the_planted_labels():
    import inspect
    src = inspect.getsource(pm.compute_anomalies_and_health)
    assert "injected_anomalies" not in src


def test_evaluation_counts_caught_spikes(tmp_path):
    import sqlite3
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE injected_anomalies (device_id TEXT, timestamp TEXT, kind TEXT)")
    conn.executemany("INSERT INTO injected_anomalies VALUES (?, ?, ?)",
                     [("DEV-001", "2026-03-15 01:00:00", "vibration"), ("DEV-001", "2026-03-15 03:00:00", "temperature")])
    conn.commit()
    conn.close()
    df = pd.DataFrame({"device_id": "DEV-001", "timestamp": [f"2026-03-15 0{h}:00:00" for h in range(5)],
                       "anomaly_flag": [0, 1, 1, 0, 0]})
    m = pm.evaluate_against_planted(str(db), df, out=str(tmp_path / "m.csv")).set_index("kind")
    assert m.loc["any", "caught"] == 1 and m.loc["any", "precision"] == 0.5 and m.loc["any", "recall"] == 0.5
