"""
Local Backfill Script
Fetches 7 days of history and saves it to local_storage (Parquet).
Safe to run multiple times (deduplicates automatically).
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from dotenv import load_dotenv

# Import your local storage saver
# This is the magic part that handles the "don't overwrite" logic
try:
    from local_storage import save_data
except ImportError:
    print("❌ Critical Error: local_storage.py not found.")
    exit(1)

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger("local_backfill")

load_dotenv()

# Config
LAT = os.getenv("LAT", "33.6844") 
LON = os.getenv("LON", "73.0479")
OPEN_METEO_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Mappings
API_COL_MAPPING = {
    "pm10": "pm10",
    "pm2_5": "pm2_5",
    "ozone": "o3",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "carbon_monoxide": "co",
}

# --- Helper Functions ---

def calculate_aqi_subindex(concentration, breakpoints):
    if pd.isna(concentration) or concentration < 0: return 0.0
    for i in range(len(breakpoints) - 1):
        low_conc, low_aqi = breakpoints[i]
        high_conc, high_aqi = breakpoints[i + 1]
        if low_conc <= concentration <= high_conc:
            return round(((high_aqi - low_aqi) / (high_conc - low_conc)) * (concentration - low_conc) + low_aqi)
    return 500.0

def calculate_aqi(row):
    # PM2.5
    pm25_bp = [(0.0, 0), (12.0, 50), (35.4, 100), (55.4, 150), (150.4, 200), (250.4, 300), (350.4, 400), (500.4, 500)]
    idx_pm25 = calculate_aqi_subindex(row.get('pm2_5'), pm25_bp)
    
    # PM10
    pm10_bp = [(0.0, 0), (54.0, 50), (154.0, 100), (254.0, 150), (354.0, 200), (424.0, 300), (504.0, 400), (604.0, 500)]
    idx_pm10 = calculate_aqi_subindex(row.get('pm10'), pm10_bp)

    # Use max of PM2.5 and PM10 as primary drivers
    return max(idx_pm25, idx_pm10)

def run_local_backfill(days=7):
    logger.info(f"🔄 Starting local backfill for last {days} days...")

    # Calculate dates
    # Fetch up to "tomorrow" to ensure we get the latest forecast hours too
    end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join(API_COL_MAPPING.keys()),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC"
    }

    try:
        # 1. Fetch Data
        logger.info(f"   Requesting Open-Meteo ({start_date} to {end_date})...")
        r = requests.get(OPEN_METEO_URL, params=params)
        r.raise_for_status()
        data = r.json()

        # 2. Parse into DataFrame
        if "hourly" not in data:
            logger.error("❌ No hourly data found!")
            return

        hourly_data = {"timestamp": data["hourly"]["time"]}
        for api_col, my_col in API_COL_MAPPING.items():
            hourly_data[my_col] = data["hourly"][api_col]

        df = pd.DataFrame(hourly_data)
        
        # Clean Timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # 3. Add Features
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['ingestion_timestamp'] = pd.Timestamp.now(tz='UTC')

        # 4. Calculate AQI
        logger.info("📊 Calculating AQI...")
        df['aqi'] = df.apply(calculate_aqi, axis=1)

        # 5. Calculate AQI Change Rate
        # Sort is crucial for shift() to work correctly
        df = df.sort_values('timestamp')
        df['prev_aqi'] = df['aqi'].shift(1)
        df['aqi_change_rate'] = (df['aqi'] - df['prev_aqi']) / df['prev_aqi']
        
        # Clean up NaNs and Infs
        df['aqi_change_rate'] = df['aqi_change_rate'].fillna(0.0)
        df['aqi_change_rate'] = df['aqi_change_rate'].replace([np.inf, -np.inf], 0.0)
        
        df = df.drop(columns=['prev_aqi'])

        logger.info(f"✅ Processed {len(df)} rows.")

        # 6. Save to Local Storage
        # save_data() automatically handles duplicates!
        save_data(df)

    except Exception as e:
        logger.error(f"❌ Backfill failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # You can change 'days=7' to 'days=30' if you want more history
    run_local_backfill(days=7)