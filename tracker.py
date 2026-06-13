import pandas as pd
import numpy as np
import os

# --- 1. CONFIGURATION & BOUNDING BOX (VIRTUAL FENCE) ---
# Simulating a massive paddock coordinate bounds in Northern Territory
MIN_LAT, MAX_LAT = -18.005, -18.000
MIN_LON, MAX_LON = 133.000, 133.005
NORMAL_TEMP_LOW, NORMAL_TEMP_HIGH = 38.3, 39.9  # Normal cattle temp in Celsius

def generate_mock_data(filename="cattle_telemetry.csv", num_records=100):
    """Generates a mock CSV file representing smart collar data from livestock."""
    np.random.seed(42) # For reproducible mock data
    
    cow_ids = [f"COW-{i:03d}" for i in range(1, 11)] # 10 cows
    data = []
    
    for _ in range(num_records):
        cow = np.random.choice(cow_ids)
        
        # Most data is normal, but let's sprinkle some anomalies
        anomaly_type = np.random.choice(['none', 'fever', 'stationary', 'escaped'], p=[0.85, 0.05, 0.05, 0.05])
        
        if anomaly_type == 'none':
            lat = np.random.uniform(MIN_LAT, MAX_LAT)
            lon = np.random.uniform(MIN_LON, MAX_LON)
            temp = np.random.uniform(NORMAL_TEMP_LOW, NORMAL_TEMP_HIGH)
        elif anomaly_type == 'fever':
            lat = np.random.uniform(MIN_LAT, MAX_LAT)
            lon = np.random.uniform(MIN_LON, MAX_LON)
            temp = np.random.uniform(40.5, 41.8) # High fever
        elif anomaly_type == 'escaped':
            lat = MIN_LAT - np.random.uniform(0.001, 0.003) # Out of bounds
            lon = np.random.uniform(MIN_LON, MAX_LON)
            temp = np.random.uniform(NORMAL_TEMP_LOW, NORMAL_TEMP_HIGH)
        elif anomaly_type == 'stationary':
            # Simulating an animal that isn't moving by locking it to a specific spot later in processing
            lat = -18.002
            lon = 133.002
            temp = np.random.uniform(NORMAL_TEMP_LOW, NORMAL_TEMP_HIGH)

        data.append([cow, pd.Timestamp.now() - pd.Timedelta(minutes=np.random.randint(1, 60)), lat, lon, round(temp, 2)])
        
    df = pd.DataFrame(data, columns=['Animal_ID', 'Timestamp', 'Latitude', 'Longitude', 'Temperature_C'])
    df.to_csv(filename, index=False)
    print(f"✔️ Successfully generated mock telemetry data saved to: {filename}")

# --- 2. ANOMALY DETECTION ENGINE ---
def analyze_livestock_data(filename="cattle_telemetry.csv"):
    """Reads telemetry data and flags health or geofence violations."""
    if not os.path.exists(filename):
        print("Error: Telemetry file not found.")
        return

    df = pd.read_csv(filename)
    df['Status'] = 'Normal'
    df['Alert_Reason'] = ''

    # Vectorized checks using pandas for maximum performance
    # Check 1: Health/Fever Detection
    fever_mask = df['Temperature_C'] > NORMAL_TEMP_HIGH
    df.loc[fever_mask, 'Status'] = 'CRITICAL'
    df.loc[fever_mask, 'Alert_Reason'] += 'Potential Illness (High Fever); '

    # Check 2: Virtual Fence (Geofence) Escape Detection
    geofence_mask = (
        (df['Latitude'] < MIN_LAT) | (df['Latitude'] > MAX_LAT) |
        (df['Longitude'] < MIN_LON) | (df['Longitude'] > MAX_LON)
    )
    df.loc[geofence_mask, 'Status'] = 'CRITICAL'
    df.loc[geofence_mask, 'Alert_Reason'] += 'Virtual Fence Breach (Escaped Paddock); '

    # Filter out only the critical anomalies to display to the station manager
    anomalies = df[df['Status'] == 'CRITICAL']
    
    print("\n" + "="*60)
    print("      OUTBACK LIVESTOCK TELEMETRY ANOMALY REPORT      ")
    print("="*60)
    print(f"Total Records Analyzed: {len(df)}")
    print(f"Critical Anomalies Detected: {len(anomalies)}\n")
    
    if not anomalies.empty:
        print(anomalies[['Animal_ID', 'Temperature_C', 'Latitude', 'Longitude', 'Alert_Reason']].to_string(index=False))
    else:
        print("Everything clear. All stock behaving normally within paddock limits.")
    print("="*60)

if __name__ == "__main__":
    # Generate data file if it doesn't exist, then analyze it
    generate_mock_data()
    analyze_livestock_data()
