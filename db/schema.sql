-- SQLite Schema for IoT Telematics & Predictive Maintenance

-- Telemetry Logs Table: Stores raw and processed sensor stream data
CREATE TABLE IF NOT EXISTS telemetry_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    battery_temp_c REAL NOT NULL,
    motor_vibration_g REAL NOT NULL,
    voltage REAL NOT NULL,
    rpm REAL NOT NULL,
    health_score REAL NOT NULL,
    anomaly_flag INTEGER DEFAULT 0
);

-- Maintenance Alerts Table: Stores system-generated predictive maintenance warnings
CREATE TABLE IF NOT EXISTS maintenance_alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL,
    alert_timestamp DATETIME NOT NULL,
    alert_type TEXT NOT NULL,
    remaining_useful_life_days REAL NOT NULL,
    severity TEXT NOT NULL
);

-- Create indexes for performance optimization on common query filters
CREATE INDEX IF NOT EXISTS idx_telemetry_device_time ON telemetry_logs (device_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_device_time ON maintenance_alerts (device_id, alert_timestamp);
