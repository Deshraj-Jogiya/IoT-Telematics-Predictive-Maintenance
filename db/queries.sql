-- 1. Peak Vibration Hours
-- Identifies which hours of the day experience the highest average and maximum motor vibration.
-- Useful for identifying diurnal operational stress patterns.
SELECT 
    strftime('%H', timestamp) AS hour_of_day,
    COUNT(*) AS total_logs,
    ROUND(AVG(motor_vibration_g), 3) AS avg_vibration_g,
    ROUND(MAX(motor_vibration_g), 3) AS max_vibration_g
FROM telemetry_logs
GROUP BY hour_of_day
ORDER BY avg_vibration_g DESC;

-- 2. Rolling Average Voltage by Device
-- Uses SQLite window functions to compute a 10-period moving average of voltage to smooth out transient fluctuations.
SELECT 
    log_id,
    device_id,
    timestamp,
    voltage,
    ROUND(AVG(voltage) OVER (
        PARTITION BY device_id 
        ORDER BY timestamp 
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ), 3) AS rolling_avg_voltage_10p
FROM telemetry_logs
ORDER BY device_id, timestamp DESC
LIMIT 100;

-- 3. High Temperature Anomalies List
-- Retrieves logs where battery temperature exceeded 75°C and an anomaly was flagged.
SELECT 
    log_id,
    device_id,
    timestamp,
    battery_temp_c,
    health_score,
    anomaly_flag
FROM telemetry_logs
WHERE battery_temp_c > 75.0 AND anomaly_flag = 1
ORDER BY battery_temp_c DESC;

-- 4. Fleet Health Summary (Latest State)
-- Shows the most recent telemetry readings, current health score, and total cumulative anomalies per device.
WITH LatestTelemetry AS (
    SELECT 
        device_id,
        timestamp,
        battery_temp_c,
        motor_vibration_g,
        health_score,
        ROW_NUMBER() OVER (PARTITION BY device_id ORDER BY timestamp DESC) as rn
    FROM telemetry_logs
),
AnomalyCounts AS (
    SELECT 
        device_id,
        COUNT(*) AS total_anomalies
    FROM telemetry_logs
    WHERE anomaly_flag = 1
    GROUP BY device_id
)
SELECT 
    lt.device_id,
    lt.timestamp AS last_updated,
    ROUND(lt.battery_temp_c, 1) AS current_temp_c,
    ROUND(lt.motor_vibration_g, 2) AS current_vibration_g,
    ROUND(lt.health_score, 1) AS current_health_score,
    COALESCE(ac.total_anomalies, 0) AS cumulative_anomalies
FROM LatestTelemetry lt
LEFT JOIN AnomalyCounts ac ON lt.device_id = ac.device_id
WHERE lt.rn = 1
ORDER BY lt.health_score ASC;

-- 5. Critical Remaining Useful Life (RUL) Alerts
-- Returns the most recent high-severity alert for each device where Remaining Useful Life (RUL) has dropped below 14 days.
WITH LatestAlerts AS (
    SELECT 
        device_id,
        alert_timestamp,
        alert_type,
        remaining_useful_life_days,
        severity,
        ROW_NUMBER() OVER (PARTITION BY device_id ORDER BY alert_timestamp DESC) as rn
    FROM maintenance_alerts
    WHERE remaining_useful_life_days < 14.0 OR severity = 'CRITICAL'
)
SELECT 
    device_id,
    alert_timestamp,
    alert_type,
    ROUND(remaining_useful_life_days, 1) AS remaining_useful_life_days,
    severity
FROM LatestAlerts
WHERE rn = 1
ORDER BY remaining_useful_life_days ASC;

-- 6. Hourly Anomaly and Alert Distribution
-- Correlates telemetry anomalies and maintenance alerts generated on an hourly basis to track fleet failure events.
SELECT 
    strftime('%Y-%m-%d %H:00:00', t.timestamp) AS time_bucket,
    COUNT(DISTINCT t.log_id) AS total_telemetry_records,
    SUM(CASE WHEN t.anomaly_flag = 1 THEN 1 ELSE 0 END) AS total_anomalies,
    COUNT(DISTINCT a.alert_id) AS total_alerts_raised
FROM telemetry_logs t
LEFT JOIN maintenance_alerts a 
    ON t.device_id = a.device_id 
    AND strftime('%Y-%m-%d %H', t.timestamp) = strftime('%Y-%m-%d %H', a.alert_timestamp)
GROUP BY time_bucket
HAVING total_anomalies > 0 OR total_alerts_raised > 0
ORDER BY time_bucket DESC
LIMIT 50;
