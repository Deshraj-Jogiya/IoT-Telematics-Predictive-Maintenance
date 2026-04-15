import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

def get_project_root():
    """Returns the root directory of the project."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data(db_path):
    """Loads all telemetry logs from the database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM telemetry_logs ORDER BY timestamp ASC", conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def generate_dashboard(df, output_path):
    """Generates a high-quality dark-mode neon dashboard visualization."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # ------------------ STYLING & PALETTE ------------------
    # Apply dark background
    plt.style.use('dark_background')
    
    # Color palette
    bg_color = '#0b0f19'       # Deep slate blue/black
    card_color = '#151c2c'     # Slightly lighter card background
    text_color = '#e2e8f0'     # Off-white
    grid_color = '#2a354f'     # Dark blue-grey grid lines
    
    # Neon accent colors
    neon_cyan = '#00f2fe'      # Temperature
    neon_magenta = '#ff007f'   # Anomaly / Vibration
    neon_green = '#39ff14'     # Health / Normal
    neon_yellow = '#ffe600'    # Warning
    
    # Configure global parameters
    plt.rcParams['figure.facecolor'] = bg_color
    plt.rcParams['axes.facecolor'] = card_color
    plt.rcParams['axes.edgecolor'] = grid_color
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = grid_color
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['text.color'] = text_color
    plt.rcParams['axes.labelcolor'] = text_color
    plt.rcParams['xtick.color'] = text_color
    plt.rcParams['ytick.color'] = text_color
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Create figure and layout (1 Top subplot, 2 Bottom subplots)
    fig = plt.figure(figsize=(16, 11), dpi=120)
    fig.suptitle('Power BI IoT Fleet Telematics & Predictive Maintenance Dashboard', 
                 fontsize=20, fontweight='bold', color=neon_cyan, y=0.96)
    
    # Create gridspec layout
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.25)
    
    # Add axes
    ax_temp = fig.add_subplot(gs[0, :])
    ax_anomaly = fig.add_subplot(gs[1, 0])
    ax_health = fig.add_subplot(gs[1, 1])
    
    # ------------------ (A) Real-Time Battery Temperature Stream ------------------
    print("Plotting Battery Temperature Stream...")
    # Resample to hourly average to make the line charts clean
    df_hourly = df.set_index('timestamp').groupby(['device_id']).resample('H').mean().reset_index()
    
    devices = sorted(df['device_id'].unique())
    device_colors = [neon_cyan, '#00bfff', neon_yellow, '#ffa500', neon_magenta]
    
    for device, color in zip(devices, device_colors):
        dev_data = df_hourly[df_hourly['device_id'] == device]
        ax_temp.plot(dev_data['timestamp'], dev_data['battery_temp_c'], 
                     label=device, color=color, linewidth=1.5, alpha=0.85)
        
    ax_temp.set_title('Hourly Fleet Battery Temperature Stream (Normal: ~35°C vs Anomalies)', 
                      fontsize=13, fontweight='bold', pad=12, color=text_color)
    ax_temp.set_ylabel('Battery Temp (°C)', fontsize=11)
    ax_temp.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax_temp.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax_temp.tick_params(axis='x', rotation=15)
    
    # Highlight anomaly zone
    ax_temp.axhline(y=75, color=neon_magenta, linestyle='--', alpha=0.6, label='Anomaly Threshold (75°C)')
    ax_temp.legend(loc='upper left', framealpha=0.2, facecolor=bg_color, edgecolor=grid_color)
    
    # ------------------ (B) Motor Vibration Z-Score Anomaly Scatter ------------------
    print("Plotting Motor Vibration Z-Score Anomaly Scatter...")
    # Recalculate Z-score for plotting
    df['vibration_z'] = 0.0
    for device in devices:
        idx = df[df['device_id'] == device].index
        vibs = df.loc[idx, 'motor_vibration_g']
        mean = vibs.rolling(window=50, min_periods=10).mean()
        std = vibs.rolling(window=50, min_periods=10).std().fillna(0.1).replace(0.0, 0.01)
        df.loc[idx, 'vibration_z'] = (vibs - mean) / std
        
    # Pick a subset of points (e.g. 1500 points randomly or step slice) to make the scatter plot clean
    df_scatter = df.iloc[::6].copy() # step every 6 records to reduce density
    
    # Separate normal and anomalous points
    normal_points = df_scatter[df_scatter['anomaly_flag'] == 0]
    anomaly_points = df_scatter[df_scatter['anomaly_flag'] == 1]
    
    ax_anomaly.scatter(normal_points['timestamp'], normal_points['vibration_z'], 
                       color='#00ffcc', alpha=0.35, s=12, label='Normal Telemetry')
    ax_anomaly.scatter(anomaly_points['timestamp'], anomaly_points['vibration_z'], 
                       color=neon_magenta, alpha=0.9, s=30, label='Anomaly Flagged (|Z| > 3.0)', 
                       edgecolors='white', linewidths=0.5)
    
    ax_anomaly.axhline(y=3.0, color=neon_yellow, linestyle=':', alpha=0.7, label='Upper Z-Limit (+3.0)')
    ax_anomaly.axhline(y=-3.0, color=neon_yellow, linestyle=':', alpha=0.7, label='Lower Z-Limit (-3.0)')
    
    ax_anomaly.set_title('Motor Vibration Z-Score & Outlier Detection', 
                         fontsize=13, fontweight='bold', pad=12)
    ax_anomaly.set_ylabel('Vibration Z-Score', fontsize=11)
    ax_anomaly.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax_anomaly.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax_anomaly.legend(loc='lower left', framealpha=0.2, facecolor=bg_color, edgecolor=grid_color)
    
    # ------------------ (C) Fleet Health Score Heatmap by Device ------------------
    print("Plotting Fleet Health Heatmap...")
    # Resample health to daily average per device
    df_daily_health = df.set_index('timestamp').groupby(['device_id']).resample('D')['health_score'].mean().reset_index()
    
    # Pivot to Device vs Date
    df_pivot = df_daily_health.pivot(index='device_id', columns='timestamp', values='health_score')
    
    # Format dates for columns
    df_pivot.columns = df_pivot.columns.strftime('%m-%d')
    
    # Plot heatmap using custom diverging palette (low health is red/orange, high health is cyan/green)
    cmap = sns.diverging_palette(10, 140, as_cmap=True) # Red to Green/Cyan
    
    # We display annot=False for clean look, but can format ticks
    sns.heatmap(df_pivot, ax=ax_health, cmap='RdYlGn', cbar_kws={'label': 'Health Score (%)'}, 
                vmin=20, vmax=100, linewidths=0.5, linecolor=bg_color)
    
    ax_health.set_title('Fleet Health Score (%) Heatmap by Device & Date', 
                        fontsize=13, fontweight='bold', pad=12)
    ax_health.set_ylabel('Device ID', fontsize=11)
    ax_health.set_xlabel('Date (Month-Day)', fontsize=11)
    
    # Clean up tick layouts
    plt.xticks(rotation=45)
    
    # Add footnote/metadata in bottom left
    fig.text(0.05, 0.02, 'Data Source: sqlite://data/iot_telemetry.db | Anomaly detection threshold: rolling Z-score | RUL based on exponential degradation model', 
             fontsize=9, color='#64748b', style='italic')
    
    # Save the dashboard graphic
    plt.savefig(output_path, bbox_inches='tight', facecolor=bg_color)
    plt.close()
    print(f"Dark-mode neon dashboard saved to: {output_path}")

def main():
    project_root = get_project_root()
    db_path = os.path.join(project_root, 'data', 'iot_telemetry.db')
    output_path = os.path.join(project_root, 'viz', 'powerbi_iot_dashboard.png')
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Run telemetry_pipeline.py and predictive_maintenance.py first.")
        
    df = load_data(db_path)
    generate_dashboard(df, output_path)

if __name__ == '__main__':
    main()
