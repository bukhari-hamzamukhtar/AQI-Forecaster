import os
import sys
import time
import logging
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv
from hsfs import feature

# Optional dependency used by your environment on Windows
try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
except Exception:
    pass

# Fast-fail check for confluent-kafka
try:
    import confluent_kafka  # noqa: F401
except ImportError:
    print("\n❌ CRITICAL MISSING DEPENDENCY: confluent-kafka")
    sys.exit(1)

try:
    import hopsworks
except Exception:
    hopsworks = None

# ------------------------ CONFIG & LOGGING ------------------------
load_dotenv()

OPEN_METEO_BASE = "https://air-quality-api.open-meteo.com/v1/air-quality"
LAT = float(os.getenv("LAT", "33.6844"))
LON = float(os.getenv("LON", "73.0479"))
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# Chunk settings
CHUNK_DAYS = 8  # Reduced to 5 for stability
CHUNK_SLEEP = 2
MAX_UPLOAD_RETRIES = 3 # Increased retries
DRY_RUN = False

API_COL_MAPPING = {
    "pm10": "pm10",
    "pm2_5": "pm2_5", 
    "ozone": "o3",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "carbon_monoxide": "co",
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger("backfill")

# ------------------------ HTTP / JSON helpers ------------------------
def fetch_range(start_date: datetime, end_date: datetime, timeout: int = 30) -> dict:
    api_vars = list(API_COL_MAPPING.keys())
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ",".join(api_vars),
        "timezone": "UTC",
    }
    logger.debug(f"Requesting Open-Meteo: {params}")
    r = requests.get(OPEN_METEO_BASE, params=params, timeout=timeout)
    if r.status_code != 200:
        logger.error(f"Open-Meteo returned status {r.status_code}: {r.text}")
    r.raise_for_status()
    return r.json()

def json_to_df(j: dict) -> pd.DataFrame:
    if not j or "hourly" not in j:
        return pd.DataFrame()
    
    hourly = j["hourly"]
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame()

    rows = []
    for i, t in enumerate(times):
        row = {"timestamp": pd.to_datetime(t, utc=True)}
        for api_name, my_col in API_COL_MAPPING.items():
            series = hourly.get(api_name, [])
            row[my_col] = series[i] if i < len(series) else None
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in API_COL_MAPPING.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ------------------------ Hopsworks helpers ------------------------
def get_or_create_feature_group(fs, name: str = "aqi_features", version: int = None):
    # 🚨 FIX: Explicitly handle Ghost/None objects
    fg = None
    try:
        fg = fs.get_feature_group(name, version=version)
    except Exception:
        pass # Not found, will create below

    # If fg is None here, it means it either didn't exist OR the API returned a ghost.
    # In either case, we force get_or_create to verify the object.
    if fg is None:
        logger.info(f"Feature group '{name}' (v{version}) not found or invalid — creating/retrieving...")
        fg = fs.get_or_create_feature_group(
            name=name,
            version=version,
            description=f"AQI Data (V{version}) - Clean Schema",
            primary_key=["timestamp"],
            event_time="timestamp",
            online_enabled=False,
        )
        logger.info(f"Created/Retrieved feature group '{name}' (v{version}).")
    else:
        logger.info(f"Found existing feature group '{name}' (v{version}).")

    return fg

def safe_insert_with_retries(fg, df: pd.DataFrame, max_retries: int = MAX_UPLOAD_RETRIES) -> bool:
    if fg is None:
        logger.error("Cannot insert: Feature Group object is None!")
        return False

    for attempt in range(1, max_retries + 1):
        try:
            fg.insert(df, write_options={"wait_for_job": False})
            logger.info("Upload succeeded.")
            return True
        except Exception as e:
            logger.warning(f"Upload attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                # Slower backoff: 5s, 10s, 15s...
                backoff = attempt * 5
                time.sleep(backoff)
    return False

# ------------------------ Main backfill loop ------------------------
def backfill_range(start_date: datetime, end_date: datetime, chunk_days: int = CHUNK_DAYS):
    if hopsworks is None:
        logger.error("hopsworks library not available.")
        return

    logger.info("🚀 Connecting to Hopsworks...")
    project = hopsworks.login(project=HOPSWORKS_PROJECT, api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    
    # Get the Feature Group (with the fix)
    # Get latest version dynamically
    latest_version = None
    try:
        all_fgs = fs.get_feature_group(name="aqi_features", version=None)
        if isinstance(all_fgs, list) and len(all_fgs) > 0:
            # Get all valid version numbers
            versions = [fg.version for fg in all_fgs if hasattr(fg, 'version') and fg.version is not None]
            if versions:
                latest_version = max(versions)
                logger.info(f"📦 Found {len(all_fgs)} feature group versions: {sorted(versions)}")
                logger.info(f"✅ Using latest version: v{latest_version}")
                fg = get_or_create_feature_group(fs, name="aqi_features", version=latest_version)
            else:
                logger.warning("⚠️ No valid versions found in feature groups, trying v3")
                latest_version = 3
                fg = get_or_create_feature_group(fs, name="aqi_features", version=3)
        else:
            logger.warning("⚠️ No feature groups found, trying v3")
            latest_version = 3
            fg = get_or_create_feature_group(fs, name="aqi_features", version=3)
    except Exception as e:
        logger.warning(f"⚠️ Could not get latest version: {e}")
        logger.warning("⚠️ Falling back to version 3")
        latest_version = 3
        fg = get_or_create_feature_group(fs, name="aqi_features", version=3)
    
    # Verify we have a valid version
    if latest_version is None:
        latest_version = 3
        logger.warning("⚠️ latest_version was None, defaulting to v3")
    
    # Log final version being used
    if fg and hasattr(fg, 'version'):
        actual_version = fg.version
        logger.info(f"📌 Feature group version confirmed: v{actual_version}")
        if actual_version != latest_version:
            logger.warning(f"⚠️ Version mismatch: expected v{latest_version}, got v{actual_version}")
            latest_version = actual_version

    if fg is None:
        logger.error("❌ CRITICAL: Failed to get Feature Group object. Cannot proceed.")
        return

    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_date)
        logger.info(f"Fetching chunk: {current.date()} → {chunk_end.date()}")

        try:
            payload = fetch_range(current, chunk_end)
            df = json_to_df(payload)

            if df.empty:
                logger.info("Chunk empty — skipping.")
                current = chunk_end + timedelta(days=1)
                continue

            # derived time features
            df["hour"] = df["timestamp"].dt.hour
            df["day"] = df["timestamp"].dt.day
            df["month"] = df["timestamp"].dt.month

            # Calculate AQI from pollutants (same as feature_pipeline.py)
            def calculate_aqi_subindex(concentration, breakpoints):
                """Calculate AQI sub-index for a pollutant using linear interpolation."""
                if pd.isna(concentration) or concentration < 0:
                    return 0.0
                for i in range(len(breakpoints) - 1):
                    low_conc, low_aqi = breakpoints[i]
                    high_conc, high_aqi = breakpoints[i + 1]
                    if low_conc <= concentration <= high_conc:
                        aqi = ((high_aqi - low_aqi) / (high_conc - low_conc)) * (concentration - low_conc) + low_aqi
                        return round(aqi)
                return 500.0

            def calculate_aqi(pm2_5, pm10, o3, no2, so2, co):
                """Calculate US EPA Air Quality Index (AQI) from pollutant concentrations."""
                sub_indices = []
                
                # PM2.5
                pm25_breakpoints = [
                    (0.0, 0), (12.0, 50), (35.4, 100), (55.4, 150), 
                    (150.4, 200), (250.4, 300), (350.4, 400), (500.4, 500)
                ]
                if pd.notna(pm2_5):
                    sub_indices.append(calculate_aqi_subindex(pm2_5, pm25_breakpoints))
                
                # PM10
                pm10_breakpoints = [
                    (0.0, 0), (54.0, 50), (154.0, 100), (254.0, 150),
                    (354.0, 200), (424.0, 300), (504.0, 400), (604.0, 500)
                ]
                if pd.notna(pm10):
                    sub_indices.append(calculate_aqi_subindex(pm10, pm10_breakpoints))
                
                # O3
                o3_breakpoints = [
                    (0.0, 0), (0.054, 50), (0.070, 100), (0.085, 150),
                    (0.105, 200), (0.200, 300), (0.404, 400), (0.604, 500)
                ]
                if pd.notna(o3):
                    o3_ppm = o3 * 0.0005  # Convert µg/m³ to ppm
                    sub_indices.append(calculate_aqi_subindex(o3_ppm, o3_breakpoints))
                
                # NO2
                no2_breakpoints = [
                    (0.0, 0), (0.053, 50), (0.100, 100), (0.360, 150),
                    (0.649, 200), (1.249, 300), (1.649, 400), (2.049, 500)
                ]
                if pd.notna(no2):
                    no2_ppm = no2 * 0.00053
                    sub_indices.append(calculate_aqi_subindex(no2_ppm, no2_breakpoints))
                
                # SO2
                so2_breakpoints = [
                    (0.0, 0), (0.034, 50), (0.144, 100), (0.224, 150),
                    (0.304, 200), (0.604, 300), (0.804, 400), (1.004, 500)
                ]
                if pd.notna(so2):
                    so2_ppm = so2 * 0.00038
                    sub_indices.append(calculate_aqi_subindex(so2_ppm, so2_breakpoints))
                
                # CO
                co_breakpoints = [
                    (0.0, 0), (4.4, 50), (9.4, 100), (12.4, 150),
                    (15.4, 200), (30.4, 300), (40.4, 400), (50.4, 500)
                ]
                if pd.notna(co):
                    co_mg_m3 = co / 1000.0  # Convert µg/m³ to mg/m³
                    sub_indices.append(calculate_aqi_subindex(co_mg_m3, co_breakpoints))
                
                return max(sub_indices) if sub_indices else 0.0

            # Calculate AQI
            df["aqi"] = df.apply(
                lambda row: calculate_aqi(
                    row.get("pm2_5"),
                    row.get("pm10"),
                    row.get("o3"),
                    row.get("no2"),
                    row.get("so2"),
                    row.get("co")
                ),
                axis=1
            )
            logger.info(f"   -> Calculated AQI for {len(df)} rows")

            # Calculate AQI change rate: (current - previous) / previous
            # Sort by timestamp first to ensure correct order
            df = df.sort_values("timestamp")
            if "pm2_5" in df.columns:
                # Calculate change from previous hour
                df["pm2_5_prev"] = df["pm2_5"].shift(1)
                # Change rate = (current - previous) / previous
                df["aqi_change_rate"] = (df["pm2_5"] - df["pm2_5_prev"]) / (df["pm2_5_prev"] + 1e-6)  # Add small epsilon to avoid division by zero
                # Fill NaN (first row) with 0
                df["aqi_change_rate"] = df["aqi_change_rate"].fillna(0.0)
                # Drop temporary column
                df = df.drop(columns=["pm2_5_prev"])
                logger.info(f"   -> Calculated AQI change rate for {len(df)} rows")

            # ensure floats
            for col in API_COL_MAPPING.values():
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Ensure AQI and aqi_change_rate are numeric
            if "aqi" in df.columns:
                df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce").astype(int)
            if "aqi_change_rate" in df.columns:
                df["aqi_change_rate"] = pd.to_numeric(df["aqi_change_rate"], errors="coerce")

            # Add ingestion_timestamp
            df["ingestion_timestamp"] = pd.Timestamp.now(tz='UTC')

            if not DRY_RUN:
                # Check and update feature group schema before inserting
                try:
                    existing_feature_names = [f.name.lower() for f in fg.features] if fg.features else []
                    features_to_append = []
                    
                    # Check for ingestion_timestamp
                    if "ingestion_timestamp" not in existing_feature_names:
                        logger.info("📝 'ingestion_timestamp' not in schema - will add it")
                        features_to_append.append(
                            feature.Feature(name="ingestion_timestamp", type="timestamp")
                        )
                    
                    # Check for aqi
                    if "aqi" not in existing_feature_names:
                        logger.info("📝 'aqi' not in schema - will add it")
                        features_to_append.append(
                            feature.Feature(name="aqi", type="int")
                        )
                    
                    # Check for aqi_change_rate
                    if "aqi_change_rate" not in existing_feature_names:
                        logger.info("📝 'aqi_change_rate' not in schema - will add it")
                        features_to_append.append(
                            feature.Feature(name="aqi_change_rate", type="double")
                        )
                    
                    # Append features to schema if needed
                    if features_to_append:
                        logger.info(f"🔄 Appending {len(features_to_append)} feature(s) to schema...")
                        fg.append_features(features_to_append)
                        logger.info(f"✅ Successfully appended {len(features_to_append)} feature(s) to schema")
                        
                        # Re-fetch the feature group to get updated schema
                        current_fg_version = fg.version if hasattr(fg, 'version') and fg.version is not None else latest_version
                        fg = fs.get_feature_group(name="aqi_features", version=current_fg_version)
                        logger.info("   Refreshed feature group to get updated schema")
                except Exception as schema_error:
                    logger.warning(f"⚠️ Could not update schema: {schema_error}")
                    # Continue anyway - might work if features already exist
                
                success = safe_insert_with_retries(fg, df, max_retries=MAX_UPLOAD_RETRIES)
                if not success:
                    logger.error("Failed to upload chunk.")

        except Exception as e:
            logger.exception(f"Unexpected error: {e}")

        current = chunk_end + timedelta(days=1)
        time.sleep(CHUNK_SLEEP)

    logger.info("✅ Backfill complete.")

if __name__ == "__main__":
    # Check if using local storage
    USE_LOCAL_STORAGE = os.getenv("USE_LOCAL_STORAGE", "0") == "1"
    
    if USE_LOCAL_STORAGE:
        logger.info("💾 Using LOCAL STORAGE for backfill")
        logger.info("   Note: Backfilling to local storage...")
        try:
            from local_storage import save_data, calculate_aqi_change_rate
            from feature_pipeline import calculate_aqi
            
            # Import the backfill logic but save to local storage
            # We'll need to modify the backfill_range function
            logger.info("   Local storage backfill will be implemented...")
            logger.warning("   For now, use migrate_hopsworks_to_local.py if you have existing data")
            logger.warning("   Or start fresh with feature_pipeline.py (hourly ingestion)")
        except ImportError:
            logger.error("❌ local_storage.py not found!")
            logger.error("   Falling back to Hopsworks backfill...")
            USE_LOCAL_STORAGE = False
    
    if not USE_LOCAL_STORAGE:
        # Original Hopsworks backfill
        # Backfill for past 8 years
        end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=8*365)  # 8 years ago

    logger.info(f"Starting backfill from {start_dt.date()} to {end_dt.date()} (8 years)")
    backfill_range(start_dt, end_dt)