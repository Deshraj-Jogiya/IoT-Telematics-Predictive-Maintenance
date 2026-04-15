import os
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def get_project_root():
    """Returns the root directory of the project."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def initialize_database(db_path, schema_path):
    """Initializes the SQLite database with the schema."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Read and execute schema
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    
    cursor.executescript(schema_sql)
    
    # Clean tables for a fresh run
    cursor.execute("DELETE FROM telemetry_logs;")
    cursor.execute("DELETE FROM maintenance_alerts;")
    conn.commit()
    conn.close()
    print(f"Database initialized and cleared at: {db_path}")

def generate_telemetry_data():
    """Generates streaming-like telemetry data for 5 devices."""
    devices = ['DEV-001', 'DEV-002', 'DEV-003', 'DEV-004', 'DEV-005']
    start_time = datetime(2026, 3, 15, 0, 0, 0)
    end_time = datetime(2026, 4, 15, 10, 0, 0)
    
    # 20-minute intervals -> 3 logs per hour
    delta = timedelta(minutes=20)
    
    timestamps = []
    curr = start_time
    while curr <= end_time:
        timestamps.append(curr)
        curr += delta
        
    print(f"Generating {len(timestamps)} records per device (Total logs: {len(timestamps) * len(devices)})")
    
    # Random seed for reproducibility
    np.random.seed(42)
    
    all_records = []
    
    for device in devices:
        # Generate baseline parameters
        base_temp = 35.0
        base_vib = 1.5
        base_voltage = 24.0
        base_rpm = 3000.0
        
        # We model a gradual degradation in some devices (e.g., DEV-003 and DEV-005 have higher wear)
        wear_factor = 0.0
        is_high_wear = device in ['DEV-003', 'DEV-005']
        
        for idx, ts in enumerate(timestamps):
            # Gradual wear increases baseline temperature and vibration over time
            if is_high_wear:
                wear_factor = (idx / len(timestamps)) * 1.8
            
            # Normal distribution with noise
            temp = np.random.normal(base_temp + wear_factor * 2.0, 1.5)
            vib = np.random.normal(base_vib + wear_factor * 0.3, 0.15)
            voltage = np.random.normal(base_voltage - wear_factor * 0.5, 0.25)
            rpm = np.random.normal(base_rpm - wear_factor * 100.0, 50.0)
            
            # Inject anomalies (temperature spikes to ~80C, vibration to ~4.8g)
            # 1% chance for temperature anomaly
            if np.random.random() < 0.01:
                temp = np.random.normal(80.0, 3.0)
            # 1% chance for vibration anomaly
            if np.random.random() < 0.01:
                vib = np.random.normal(4.8, 0.4)
                
            # Default health score is initialized at 100.0, and anomaly flag is 0.
            # These will be updated by the predictive maintenance model.
            record = (
                device,
                ts.strftime('%Y-%m-%d %H:%M:%S'),
                float(temp),
                float(vib),
                float(voltage),
                float(rpm),
                100.0,  # initial health_score
                0       # initial anomaly_flag
            )
            all_records.append(record)
            
    # Sort records by timestamp to simulate streaming order
    all_records.sort(key=lambda x: x[1])
    return all_records

def write_to_db_micro_batches(db_path, records, batch_size=500):
    """Simulates writing streaming logs in micro-batches to the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = """
    INSERT INTO telemetry_logs 
    (device_id, timestamp, battery_temp_c, motor_vibration_g, voltage, rpm, health_score, anomaly_flag)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """
    
    total_records = len(records)
    for i in range(0, total_records, batch_size):
        batch = records[i:i+batch_size]
        cursor.executemany(query, batch)
        conn.commit()
        print(f"Ingested micro-batch: records {i} to {min(i + batch_size, total_records)} / {total_records}")
        
    conn.close()
    print("ETL Telemetry Pipeline finished successfully.")

def main():
    project_root = get_project_root()
    db_path = os.path.join(project_root, 'data', 'iot_telemetry.db')
    schema_path = os.path.join(project_root, 'db', 'schema.sql')
    
    initialize_database(db_path, schema_path)
    records = generate_telemetry_data()
    write_to_db_micro_batches(db_path, records)

if __name__ == '__main__':
    main()
