"""Export fleet health and alerts as flat CSV files for the Power BI report (powerbi/fleet_health_dashboard.pbix).

Writes to exports/:
- device_daily_health.csv: per device and day, closing health score, readings, anomalies, mean battery temperature and vibration
- maintenance_alerts.csv: every alert with its device, time, type, severity and remaining useful life
- device_status.csv: latest health score, latest remaining useful life and alert counts per device
"""

import os
import sqlite3

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE, "data", "iot_telemetry.db")
OUT_DIR = os.path.join(BASE, "exports")


def device_daily_health(conn):
    logs = pd.read_sql_query(
        "SELECT device_id, timestamp, battery_temp_c, motor_vibration_g, health_score, anomaly_flag FROM telemetry_logs",
        conn, parse_dates=["timestamp"])
    logs["date"] = logs["timestamp"].dt.date
    logs = logs.sort_values("timestamp")
    return (logs.groupby(["device_id", "date"])
                .agg(closing_health_score=("health_score", "last"), readings=("health_score", "size"),
                     anomalies=("anomaly_flag", "sum"), mean_battery_temp_c=("battery_temp_c", "mean"),
                     mean_vibration_g=("motor_vibration_g", "mean"))
                .round(2).reset_index())


def maintenance_alerts(conn):
    return pd.read_sql_query(
        "SELECT device_id, alert_timestamp, alert_type, severity, ROUND(remaining_useful_life_days, 1) AS rul_days "
        "FROM maintenance_alerts ORDER BY alert_timestamp", conn)


def device_status(daily, alerts):
    last = daily.sort_values("date").groupby("device_id").tail(1)[["device_id", "closing_health_score"]]
    rul = alerts.sort_values("alert_timestamp").groupby("device_id")["rul_days"].last().rename("latest_alert_rul_days")
    counts = alerts.pivot_table(index="device_id", columns="severity", values="alert_type", aggfunc="count", fill_value=0)
    counts.columns = [f"{c.lower()}_alerts" for c in counts.columns]
    return (last.set_index("device_id").join(rul).join(counts).fillna(0)
                .rename(columns={"closing_health_score": "latest_health_score"}).reset_index())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        daily, alerts = device_daily_health(conn), maintenance_alerts(conn)
    daily.to_csv(os.path.join(OUT_DIR, "device_daily_health.csv"), index=False)
    alerts.to_csv(os.path.join(OUT_DIR, "maintenance_alerts.csv"), index=False)
    device_status(daily, alerts).to_csv(os.path.join(OUT_DIR, "device_status.csv"), index=False)
    print(f"Wrote {len(daily)} device-days and {len(alerts)} alerts to {OUT_DIR}")


if __name__ == "__main__":
    main()
