import os
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

def get_project_root():
    """Returns the root directory of the project."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data(db_path):
    """Loads all telemetry logs from the database into a pandas DataFrame."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM telemetry_logs ORDER BY device_id, timestamp ASC", conn)
    conn.close()
    return df

def compute_anomalies_and_health(df):
    """Computes rolling Z-scores, anomaly flags, and models exponential health decay & RUL."""
    print("Computing rolling Z-scores and anomaly flags...")
    
    # Sort just in case
    df = df.sort_values(by=['device_id', 'timestamp']).reset_index(drop=True)
    
    # Calculate rolling Z-scores per device
    df['vibration_z'] = 0.0
    df['temp_z'] = 0.0
    
    window = 50
    min_periods = 10
    
    for device, group in df.groupby('device_id'):
        idx = group.index
        
        # Vibration rolling stats
        roll_vib_mean = group['motor_vibration_g'].rolling(window=window, min_periods=min_periods).mean()
        roll_vib_std = group['motor_vibration_g'].rolling(window=window, min_periods=min_periods).std().fillna(0.1)
        roll_vib_std = roll_vib_std.replace(0.0, 0.01) # Avoid division by zero
        
        # Temp rolling stats
        roll_temp_mean = group['battery_temp_c'].rolling(window=window, min_periods=min_periods).mean()
        roll_temp_std = group['battery_temp_c'].rolling(window=window, min_periods=min_periods).std().fillna(0.1)
        roll_temp_std = roll_temp_std.replace(0.0, 0.01)
        
        df.loc[idx, 'vibration_z'] = (group['motor_vibration_g'] - roll_vib_mean) / roll_vib_std
        df.loc[idx, 'temp_z'] = (group['battery_temp_c'] - roll_temp_mean) / roll_temp_std
        
    # Anomaly flag: Z-score absolute value > 3.0
    df['anomaly_flag'] = ((df['vibration_z'].abs() > 3.0) | (df['temp_z'].abs() > 3.0)).astype(int)
    
    print("Modeling health score exponential decay and estimating RUL...")
    # Health decay and RUL
    # We iterate through the dataframe and compute health score using the decay model:
    # Health(t) = Health(t-1) * exp(-alpha * dt - beta * anomaly)
    
    # Convert timestamp to datetime object for calculation
    df['ts_dt'] = pd.to_datetime(df['timestamp'])
    
    health_scores = []
    rul_days_list = []
    alerts = []
    
    # Parameters for wear
    # Normal devices: slow wear (decay rate alpha = 0.003, anomaly impact beta = 0.03)
    # High wear devices (DEV-003, DEV-005): fast wear (decay rate alpha = 0.008, anomaly impact beta = 0.05)
    device_params = {
        'DEV-001': {'alpha': 0.002, 'beta': 0.03, 'lambda_est': 0.004},
        'DEV-002': {'alpha': 0.0025, 'beta': 0.03, 'lambda_est': 0.005},
        'DEV-003': {'alpha': 0.009, 'beta': 0.06, 'lambda_est': 0.018},
        'DEV-004': {'alpha': 0.002, 'beta': 0.025, 'lambda_est': 0.004},
        'DEV-005': {'alpha': 0.008, 'beta': 0.055, 'lambda_est': 0.016}
    }
    
    for device, group in df.groupby('device_id'):
        params = device_params.get(device, {'alpha': 0.003, 'beta': 0.03, 'lambda_est': 0.005})
        alpha = params['alpha']
        beta = params['beta']
        
        current_health = 100.0
        prev_time = None
        
        # Track active alerts to avoid spamming consecutive identical warnings
        last_alert_type = None
        
        for idx, row in group.iterrows():
            ts = row['ts_dt']
            anomaly = row['anomaly_flag']
            
            if prev_time is None:
                dt_days = 0.0
            else:
                dt_days = (ts - prev_time).total_seconds() / (24 * 3600)
                
            prev_time = ts
            
            # Apply decay
            decay_factor = np.exp(-(alpha * dt_days + beta * anomaly))
            current_health = current_health * decay_factor
            current_health = max(10.0, min(100.0, current_health)) # Clamp health between 10% and 100%
            
            health_scores.append((idx, current_health))
            
            # RUL calculation: RUL = ln(Health / Failure_Health) / lambda_est
            # Failure Health is set to 25.0
            fail_health = 25.0
            if current_health <= fail_health:
                rul = 0.0
            else:
                rul = np.log(current_health / fail_health) / params['lambda_est']
            
            # Clamp RUL to realistic bounds
            rul = max(0.0, min(90.0, rul))
            rul_days_list.append((idx, rul))
            
            # Check for alerts
            alert_triggered = False
            alert_type = None
            severity = None
            
            if anomaly == 1:
                # Trigger anomaly alert
                alert_triggered = True
                alert_type = "Sensor Anomaly"
                severity = "WARNING" if rul >= 14.0 else "HIGH"
                
            if rul < 14.0:
                alert_triggered = True
                if rul < 7.0:
                    alert_type = "Critical Health Degradation"
                    severity = "CRITICAL"
                else:
                    alert_type = "Low Remaining Useful Life"
                    severity = "HIGH"
                    
            if alert_triggered:
                # Only log alert if it's a new type or at least 4 hours have passed since the last alert
                alerts.append({
                    'device_id': device,
                    'alert_timestamp': row['timestamp'],
                    'alert_type': alert_type,
                    'remaining_useful_life_days': float(rul),
                    'severity': severity
                })
                
    # Sort back by index to align with dataframe
    health_scores.sort(key=lambda x: x[0])
    rul_days_list.sort(key=lambda x: x[0])
    
    df['health_score'] = [h[1] for h in health_scores]
    df['rul_days'] = [r[1] for r in rul_days_list]
    
    # Drop temp column used for calculations
    df = df.drop(columns=['ts_dt', 'vibration_z', 'temp_z', 'rul_days'])
    
    return df, alerts

def save_results(db_path, df, alerts):
    """Saves the updated telemetry logs and alerts back to the SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Clear and rewrite telemetry_logs to maintain schema
    print("Saving updated telemetry logs to database...")
    cursor.execute("DELETE FROM telemetry_logs")
    
    # Convert df back to list of tuples matching table schema:
    # (log_id, device_id, timestamp, battery_temp_c, motor_vibration_g, voltage, rpm, health_score, anomaly_flag)
    records = df[['log_id', 'device_id', 'timestamp', 'battery_temp_c', 'motor_vibration_g', 'voltage', 'rpm', 'health_score', 'anomaly_flag']].values.tolist()
    
    query = """
    INSERT INTO telemetry_logs 
    (log_id, device_id, timestamp, battery_temp_c, motor_vibration_g, voltage, rpm, health_score, anomaly_flag)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    
    cursor.executemany(query, records)
    conn.commit()
    print(f"Updated {len(records)} telemetry logs in database.")
    
    # Write alerts
    print(f"Saving {len(alerts)} maintenance alerts to database...")
    alert_query = """
    INSERT INTO maintenance_alerts 
    (device_id, alert_timestamp, alert_type, remaining_useful_life_days, severity)
    VALUES (?, ?, ?, ?, ?);
    """
    
    alert_records = [(a['device_id'], a['alert_timestamp'], a['alert_type'], a['remaining_useful_life_days'], a['severity']) for a in alerts]
    cursor.executemany(alert_query, alert_records)
    conn.commit()
    
    conn.close()
    print("Predictive Maintenance calculations and warnings updated successfully.")

def main():
    project_root = get_project_root()
    db_path = os.path.join(project_root, 'data', 'iot_telemetry.db')
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Run telemetry_pipeline.py first.")
        
    df = load_data(db_path)
    df, alerts = compute_anomalies_and_health(df)
    save_results(db_path, df, alerts)

if __name__ == '__main__':
    main()
