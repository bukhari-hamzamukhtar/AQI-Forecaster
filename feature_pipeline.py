"""
Feature Pipeline - Automated hourly data ingestion for AQI Predictor
Fetches current air quality data and uploads to Hopsworks Feature Store.
"""
import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta
import hopsworks
from hsfs import feature
from dotenv import load_dotenv
import logging

# Optional SSL fix (mainly for Windows, but harmless on Linux)
try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
except ImportError:
    pass  # certifi not required on Linux, but helpful

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger("feature_pipeline")

load_dotenv()

# Configuration from environment variables
OPEN_METEO = os.getenv("OPEN_METEO_BASE", "https://air-quality-api.open-meteo.com/v1/air-quality")
LAT = os.getenv("LAT", "33.6844")  # Default to Islamabad
LON = os.getenv("LON", "73.0479")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

API_COL_MAPPING = {
    "pm10": "pm10",
    "pm2_5": "pm2_5",
    "ozone": "o3",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "carbon_monoxide": "co",
}

def calculate_aqi_subindex(concentration, breakpoints):
    """
    Calculate AQI sub-index for a pollutant using linear interpolation.
    breakpoints: [(low_concentration, low_aqi), (high_concentration, high_aqi)]
    """
    if pd.isna(concentration) or concentration < 0:
        return 0.0
    
    # Find the appropriate breakpoint range
    for i in range(len(breakpoints) - 1):
        low_conc, low_aqi = breakpoints[i]
        high_conc, high_aqi = breakpoints[i + 1]
        
        if low_conc <= concentration <= high_conc:
            # Linear interpolation
            aqi = ((high_aqi - low_aqi) / (high_conc - low_conc)) * (concentration - low_conc) + low_aqi
            return round(aqi)
    
    # If concentration exceeds highest breakpoint, return max AQI
    return 500.0

def calculate_aqi(pm2_5, pm10, o3, no2, so2, co):
    """
    Calculate US EPA Air Quality Index (AQI) from pollutant concentrations.
    
    AQI is the maximum of all pollutant sub-indices.
    Uses US EPA AQI breakpoints (concentrations in µg/m³, CO in mg/m³).
    
    Returns: AQI value (0-500)
    """
    sub_indices = []
    
    # PM2.5 (24-hour average, but we use hourly as proxy)
    # Breakpoints: (concentration µg/m³, AQI)
    pm25_breakpoints = [
        (0.0, 0), (12.0, 50), (35.4, 100), (55.4, 150), 
        (150.4, 200), (250.4, 300), (350.4, 400), (500.4, 500)
    ]
    if pd.notna(pm2_5):
        sub_indices.append(calculate_aqi_subindex(pm2_5, pm25_breakpoints))
    
    # PM10 (24-hour average, but we use hourly as proxy)
    pm10_breakpoints = [
        (0.0, 0), (54.0, 50), (154.0, 100), (254.0, 150),
        (354.0, 200), (424.0, 300), (504.0, 400), (604.0, 500)
    ]
    if pd.notna(pm10):
        sub_indices.append(calculate_aqi_subindex(pm10, pm10_breakpoints))
    
    # O3 (8-hour average, but we use hourly as proxy)
    o3_breakpoints = [
        (0.0, 0), (0.054, 50), (0.070, 100), (0.085, 150),
        (0.105, 200), (0.200, 300), (0.404, 400), (0.604, 500)
    ]
    # O3 concentration from API is in µg/m³, convert to ppm for AQI calculation
    # 1 µg/m³ O3 ≈ 0.0005 ppm at standard conditions
    if pd.notna(o3):
        o3_ppm = o3 * 0.0005  # Approximate conversion
        sub_indices.append(calculate_aqi_subindex(o3_ppm, o3_breakpoints))
    
    # NO2 (1-hour average)
    no2_breakpoints = [
        (0.0, 0), (0.053, 50), (0.100, 100), (0.360, 150),
        (0.649, 200), (1.249, 300), (1.649, 400), (2.049, 500)
    ]
    # NO2 concentration from API is in µg/m³, convert to ppm
    # 1 µg/m³ NO2 ≈ 0.00053 ppm at standard conditions
    if pd.notna(no2):
        no2_ppm = no2 * 0.00053
        sub_indices.append(calculate_aqi_subindex(no2_ppm, no2_breakpoints))
    
    # SO2 (1-hour average)
    so2_breakpoints = [
        (0.0, 0), (0.034, 50), (0.144, 100), (0.224, 150),
        (0.304, 200), (0.604, 300), (0.804, 400), (1.004, 500)
    ]
    # SO2 concentration from API is in µg/m³, convert to ppm
    # 1 µg/m³ SO2 ≈ 0.00038 ppm at standard conditions
    if pd.notna(so2):
        so2_ppm = so2 * 0.00038
        sub_indices.append(calculate_aqi_subindex(so2_ppm, so2_breakpoints))
    
    # CO (8-hour average, but we use hourly as proxy)
    # CO is already in µg/m³ from API, convert to mg/m³ for AQI
    co_breakpoints = [
        (0.0, 0), (4.4, 50), (9.4, 100), (12.4, 150),
        (15.4, 200), (30.4, 300), (40.4, 400), (50.4, 500)
    ]
    if pd.notna(co):
        co_mg_m3 = co / 1000.0  # Convert µg/m³ to mg/m³
        sub_indices.append(calculate_aqi_subindex(co_mg_m3, co_breakpoints))
    
    # AQI is the maximum of all sub-indices
    if sub_indices:
        return max(sub_indices)
    else:
        return 0.0

def fetch_current(lat, lon):
    """Fetch the most recent air quality data from Open-Meteo API."""
    # Get current time - request data from yesterday to tomorrow
    # This ensures we get recent actual observations and near-term forecasts
    now = datetime.now()
    start_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(API_COL_MAPPING.keys()),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC"
    }
    try:
        logger.info(f"🌤️ Fetching current data for lat={lat}, lon={lon}...")
        logger.info(f"   Requesting data from {start_date} to {end_date}")
        r = requests.get(OPEN_METEO, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        # The endpoint returns time series; find the current hour or nearest future hour
        if "hourly" not in data or "time" not in data["hourly"]:
            logger.error("❌ No hourly data in API response")
            return pd.DataFrame()
            
        times = data["hourly"]["time"]
        if not times:
            logger.error("❌ No timestamps in API response")
            return pd.DataFrame()
        
        # Convert all timestamps to datetime
        time_dts = [pd.to_datetime(t, utc=True) for t in times]
        current_utc = pd.Timestamp.now(tz='UTC').replace(minute=0, second=0, microsecond=0)
        
        # Log available timestamps for debugging
        logger.info(f"📅 Available timestamps in API response: {len(time_dts)} timestamps")
        if len(time_dts) > 0:
            logger.info(f"   First: {time_dts[0]}, Last: {time_dts[-1]}")
            logger.info(f"   Current UTC: {current_utc}")
        
        # Find the MOST RECENT actual data (not just closest - we want the latest!)
        # IMPORTANT: Prefer actual (past/current) data over forecast (future) data
        best_idx = -1
        best_time = None
        max_past_time = None  # Track the maximum past timestamp
        
        # First, find the MOST RECENT actual data (past/current hour)
        # We want the LATEST timestamp that's <= current_utc, not the closest
        for idx, ts in enumerate(time_dts):
            if ts <= current_utc:
                # This is actual data (past or current)
                if max_past_time is None or ts > max_past_time:
                    max_past_time = ts
                    best_idx = idx
                    best_time = ts
        
        # If we found actual data, use it
        if best_idx != -1:
            age_hours = (current_utc - best_time).total_seconds() / 3600
            logger.info(f"✅ Found actual data: {best_time} (age: {age_hours:.1f} hours)")
            
            # Warn if data is too old (> 6 hours)
            if age_hours > 6:
                logger.warning(f"⚠️ WARNING: Data is {age_hours:.1f} hours old! API may not have current data.")
        else:
            # No actual data found, fall back to nearest future hour (forecast)
            logger.warning("⚠️ No actual (current/past) data found, searching for forecast data")
            min_diff = float('inf')
            for idx, ts in enumerate(time_dts):
                if ts > current_utc:
                    diff = abs((ts - current_utc).total_seconds())
                    if diff < min_diff:
                        min_diff = diff
                        best_idx = idx
                        best_time = ts
            
            # Final fallback to last timestamp if still nothing found
            if best_idx == -1:
                best_idx = len(times) - 1
                best_time = time_dts[best_idx]
                logger.warning(f"⚠️ Using last available timestamp: {best_time}")
        
        ts = best_time
        is_forecast = ts > current_utc
        
        if is_forecast:
            logger.warning(f"⚠️ Selected timestamp is in the future (forecast data): {ts} (Current: {current_utc})")
        else:
            age_hours = (current_utc - ts).total_seconds() / 3600
            logger.info(f"✅ Selected actual data timestamp: {ts} (Current: {current_utc}, Age: {age_hours:.1f}h)")
            
            # Reject data that's too old (> 12 hours) - this indicates a problem
            if age_hours > 12:
                logger.error(f"❌ ERROR: Selected data is {age_hours:.1f} hours old (> 12h threshold)!")
                logger.error(f"   This suggests the API doesn't have recent data or there's a data availability issue.")
                logger.error(f"   Available timestamps: {time_dts[0]} to {time_dts[-1]}")
                # Still proceed, but log the error - the pipeline will continue but with old data
        
        row = {"timestamp": ts}
        
        for api_col, my_col in API_COL_MAPPING.items():
            series = data["hourly"].get(api_col, [])
            row[my_col] = series[best_idx] if best_idx < len(series) else None
            
        df = pd.DataFrame([row])
        logger.info(f"✅ Fetched data for timestamp: {ts} (Current UTC: {current_utc})")
        return df
        
    except requests.RequestException as e:
        logger.error(f"❌ API request failed: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error processing API response: {e}")
        raise

def calculate_aqi_change_rate(df, fs, fg_version):
    """
    Calculate AQI change rate by comparing current AQI with previous hour's AQI value.
    Returns the change rate (current - previous) / previous, or 0 if no previous data.
    """
    if df.empty:
        return 0.0
    
    # Calculate current AQI
    current_aqi = calculate_aqi(
        df.iloc[0].get("pm2_5"),
        df.iloc[0].get("pm10"),
        df.iloc[0].get("o3"),
        df.iloc[0].get("no2"),
        df.iloc[0].get("so2"),
        df.iloc[0].get("co")
    )
    
    if current_aqi == 0:
        return 0.0
    
    try:
        # Get the current timestamp
        current_ts = df.iloc[0]["timestamp"]
        
        # Try to fetch recent data from feature store to find previous hour
        # Use the SAME version that we're inserting into!
        fg = fs.get_feature_group(name="aqi_features", version=fg_version)
        
        # Only select columns that exist in the actual data (avoid new schema columns that don't exist in parquet yet)
        # This prevents DuckDB errors when reading old data that doesn't have the new columns
        recent_data = None
        last_error = None
        
        try:
            # Try reading with core columns that definitely exist in old data
            # If AQI column exists, use it; otherwise calculate from pollutants
            # Add timeout protection - limit to last 24 hours to avoid reading too much data
            try:
                recent_data = fg.select(["timestamp", "pm2_5", "pm10", "o3", "no2", "so2", "co"]).read(
                    read_options={"use_hive": True}
                )
                logger.info("   -> Successfully read data using select with core columns")
            except Exception as select_error:
                last_error = select_error
                logger.debug(f"   -> Select with core columns failed: {select_error}")
                # If select fails, try reading all data (might work if all data is new)
                try:
                    recent_data = fg.read(read_options={"use_hive": True})
                    logger.info("   -> Successfully read data using full read with Hive")
                except Exception as read_error:
                    last_error = read_error
                    logger.debug(f"   -> Full read with Hive failed: {read_error}")
                    # If both fail, try without hive (direct read) - but limit rows to prevent timeout
                    try:
                        recent_data = fg.select(["timestamp", "pm2_5", "pm10", "o3", "no2", "so2", "co"]).read(
                            read_options={"use_hive": False}
                        )
                        logger.info("   -> Successfully read data using select without Hive")
                    except Exception as direct_error:
                        last_error = direct_error
                        # Last resort: return 0 if we can't read data
                        logger.warning(f"⚠️ Could not read data to calculate AQI change rate. All methods failed.")
                        logger.warning(f"   Last error: {str(last_error)[:200]}")
                        return 0.0
        except Exception as outer_error:
            # Catch any unexpected errors during the read attempts
            logger.warning(f"⚠️ Unexpected error reading data for AQI change rate: {outer_error}")
            return 0.0
        
        if recent_data is None:
            logger.warning("⚠️ Could not read any data to calculate AQI change rate")
            return 0.0
        
        if not recent_data.empty and "timestamp" in recent_data.columns:
            # Sort by timestamp and limit to recent data (last 24 hours) to avoid processing too much
            recent_data = recent_data.sort_values("timestamp")
            time_limit = current_ts - timedelta(hours=24)
            recent_data = recent_data[recent_data["timestamp"] >= time_limit]
            
            # Find data within 2 hours before current time
            time_window = current_ts - timedelta(hours=2)
            previous_data = recent_data[
                (recent_data["timestamp"] >= time_window) & 
                (recent_data["timestamp"] < current_ts)
            ]
            
            if not previous_data.empty:
                # Get the most recent previous hour
                previous_row = previous_data.iloc[-1]
                
                # Calculate AQI for previous hour
                previous_aqi = calculate_aqi(
                    previous_row.get("pm2_5"),
                    previous_row.get("pm10"),
                    previous_row.get("o3"),
                    previous_row.get("no2"),
                    previous_row.get("so2"),
                    previous_row.get("co")
                )
                
                # Calculate change rate: (current - previous) / previous
                if previous_aqi > 0:
                    change_rate = (current_aqi - previous_aqi) / previous_aqi
                    logger.info(f"   -> AQI change rate: {change_rate:.4f} (prev AQI: {previous_aqi:.1f}, curr AQI: {current_aqi:.1f})")
                    return change_rate
                else:
                    return 0.0
        
        # No previous data available, return 0 (first data point)
        logger.info("   -> No previous data found, setting AQI change rate to 0")
        return 0.0
    except Exception as e:
        logger.warning(f"⚠️ Could not calculate AQI change rate: {e}. Setting to 0.")
        return 0.0

def push_to_hopsworks(df):
    """Upload data to Hopsworks Feature Store (Latest Version)."""
    if df.empty:
        logger.warning("⚠️ No data to upload")
        return
        
    # Validate environment variables
    if not HOPSWORKS_API_KEY:
        logger.error("❌ HOPSWORKS_API_KEY not found in environment!")
        sys.exit(1)
    
    if not HOPSWORKS_PROJECT:
        logger.error("❌ HOPSWORKS_PROJECT not found in environment!")
        sys.exit(1)
    
    try:
        logger.info("🚀 Connecting to Hopsworks...")
        project = hopsworks.login(
            project=HOPSWORKS_PROJECT,
            api_key_value=HOPSWORKS_API_KEY
        )
        fs = project.get_feature_store()
        
        # Use Feature Group - get latest version automatically
        fg_name = "aqi_features"
        # Get all versions and find the latest
        try:
            all_fgs = fs.get_feature_group(name=fg_name, version=None)
            if isinstance(all_fgs, list) and len(all_fgs) > 0:
                fg_version = max(fg.version for fg in all_fgs if hasattr(fg, 'version') and fg.version is not None)
                logger.info(f"📦 Found {len(all_fgs)} versions, using latest: v{fg_version}")
            else:
                fg_version = 3  # Fallback to version 3
                logger.warning(f"⚠️ Could not determine latest version, using v{fg_version}")
        except Exception as e:
            fg_version = 3  # Fallback to version 3
            logger.warning(f"⚠️ Error getting versions: {e}, using v{fg_version}")
        
        try:
            fg = fs.get_feature_group(name=fg_name, version=fg_version)
            logger.info(f"✅ Found feature group '{fg_name}' v{fg_version}")
        except Exception as e:
            logger.error(f"❌ Feature Group V{fg_version} not found: {e}")
            logger.error("💡 Run backfill.py first to create the feature group!")
            sys.exit(1)
        
        # Check feature group schema and append missing features BEFORE calculating derived features
        logger.info("🔍 Checking feature group schema...")
        try:
            existing_feature_names = [f.name.lower() for f in fg.features] if fg.features else []
            logger.info(f"   Existing features in schema: {existing_feature_names}")
        except Exception as e:
            logger.warning(f"⚠️ Could not read feature group schema: {e}")
            existing_feature_names = []
        
        features_to_append = []
        columns_to_remove = []
        
        # Check for ingestion_timestamp
        if "ingestion_timestamp" not in existing_feature_names:
            logger.info("📝 'ingestion_timestamp' not in schema - will add it")
            features_to_append.append(
                feature.Feature(name="ingestion_timestamp", type="timestamp")
            )
        else:
            logger.info("   ✓ 'ingestion_timestamp' already exists in schema")
        
        # Check for aqi (the actual AQI value)
        if "aqi" not in existing_feature_names:
            logger.info("📝 'aqi' not in schema - will add it")
            features_to_append.append(
                feature.Feature(name="aqi", type="int")
            )
        else:
            logger.info("   ✓ 'aqi' already exists in schema")
        
        # Check for aqi_change_rate
        if "aqi_change_rate" not in existing_feature_names:
            logger.info("📝 'aqi_change_rate' not in schema - will add it")
            features_to_append.append(
                feature.Feature(name="aqi_change_rate", type="double")
            )
        else:
            logger.info("   ✓ 'aqi_change_rate' already exists in schema")
        
        # Try to append features to schema
        if features_to_append:
            try:
                logger.info(f"🔄 Appending {len(features_to_append)} feature(s) to schema...")
                fg.append_features(features_to_append)
                logger.info(f"✅ Successfully appended {len(features_to_append)} feature(s) to schema")
                
                # IMPORTANT: Re-fetch the feature group to get updated schema
                fg = fs.get_feature_group(name=fg_name, version=fg_version)
                logger.info("   Refreshed feature group to get updated schema")
                
                # Verify the features were added
                updated_feature_names = [f.name.lower() for f in fg.features] if fg.features else []
                logger.info(f"   Updated schema features: {updated_feature_names}")
                
            except Exception as e:
                logger.error(f"❌ Failed to append features to schema: {e}")
                logger.error("   Removing new columns from dataframe to allow insertion...")
                import traceback
                logger.error(traceback.format_exc())
                # Remove columns that don't exist in schema so we can at least insert the data
                if "ingestion_timestamp" in df.columns and "ingestion_timestamp" not in existing_feature_names:
                    columns_to_remove.append("ingestion_timestamp")
                if "aqi" in df.columns and "aqi" not in existing_feature_names:
                    columns_to_remove.append("aqi")
                if "aqi_change_rate" in df.columns and "aqi_change_rate" not in existing_feature_names:
                    columns_to_remove.append("aqi_change_rate")
                if columns_to_remove:
                    logger.warning(f"   Dropping columns: {columns_to_remove} (will be added in next run)")
                    df = df.drop(columns=columns_to_remove)
        
        # Calculate AQI from all pollutants
        logger.info("📊 Calculating AQI from pollutants...")
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
        ).astype(int)  # Cast to Python int to match Hopsworks schema (not numpy int32/int64/bigint)
        logger.info(f"   -> Calculated AQI: {df.iloc[0]['aqi']}")
        
        # Calculate AQI change rate (requires connection to feature store)
        # IMPORTANT: Use the SAME version we're inserting into!
        logger.info("📊 Calculating AQI change rate...")
        aqi_change_rate = calculate_aqi_change_rate(df, fs, fg_version)
        df["aqi_change_rate"] = aqi_change_rate
        
        # Force all numeric columns to float to avoid schema errors
        for col in API_COL_MAPPING.values():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Ensure AQI and aqi_change_rate are numeric (if they exist)
        # IMPORTANT: Cast AQI to Python int (not numpy int32/int64/bigint) to match schema
        if "aqi" in df.columns:
            df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce").astype(int)
        if "aqi_change_rate" in df.columns:
            df["aqi_change_rate"] = pd.to_numeric(df["aqi_change_rate"], errors="coerce")
        
        # Insert dataframe - wait for job to ensure data is immediately available
        logger.info("📤 Uploading to Feature Store...")
        logger.info(f"   Data timestamp: {df.iloc[0]['timestamp']}")
        logger.info(f"   Columns in dataframe: {list(df.columns)}")
        
        # CRITICAL: Final schema check - remove any columns that don't exist in schema
        # This is a safety net in case append_features() didn't work
        try:
            # Re-fetch feature group to ensure we have latest schema
            fg = fs.get_feature_group(name=fg_name, version=fg_version)
            final_feature_names = [f.name.lower() for f in fg.features] if fg.features else []
            logger.info(f"   Final feature group schema: {final_feature_names}")
            
            # Remove columns from dataframe that aren't in the schema
            columns_to_drop = []
            for col in df.columns:
                if col.lower() not in final_feature_names:
                    logger.warning(f"   ⚠️ Column '{col}' not in schema - dropping it to allow insertion")
                    columns_to_drop.append(col)
            
            if columns_to_drop:
                logger.warning(f"   Dropping {len(columns_to_drop)} column(s) not in schema: {columns_to_drop}")
                df = df.drop(columns=columns_to_drop)
                logger.info(f"   Remaining columns after cleanup: {list(df.columns)}")
                logger.warning("   ⚠️ Note: Dropped columns will need to be added to schema manually or in next run")
        except Exception as e:
            logger.warning(f"⚠️ Could not verify final schema: {e}")
            # As last resort, drop known problematic columns
            for col in ["ingestion_timestamp", "aqi", "aqi_change_rate"]:
                if col in df.columns:
                    logger.warning(f"   Dropping '{col}' as fallback safety measure")
                    df = df.drop(columns=[col])
        
        # CRITICAL: Force AQI to match Hopsworks schema type 'int' (not 'bigint')
        # Hopsworks schema expects 'int' but pandas .astype(int) creates int64 (bigint)
        # Solution: Use pandas nullable Int32 extension array which Hopsworks may recognize as 'int'
        if "aqi" in df.columns:
            # Convert to nullable Int32 dtype - this might be recognized as 'int' by Hopsworks
            # instead of 'bigint' (int64)
            try:
                df["aqi"] = df["aqi"].astype("Int32")  # Nullable integer extension array
                logger.info(f"   ✅ AQI dtype: {df['aqi'].dtype} (nullable Int32)")
            except (TypeError, ValueError):
                # Fallback: convert to int32 then to Python ints in object dtype
                import numpy as np
                df["aqi"] = df["aqi"].astype(np.int32)
                aqi_values = [int(x) if pd.notna(x) else None for x in df["aqi"]]
                df = df.drop(columns=["aqi"])
                df["aqi"] = pd.Series(aqi_values, dtype=object, index=df.index)
                logger.info(f"   ✅ AQI dtype: {df['aqi'].dtype} (object with Python ints)")
        
        # Insert with timeout protection - don't wait too long in GitHub Actions
        try:
            fg.insert(df, write_options={"wait_for_job": True})  # Wait for completion so data is immediately readable
            logger.info("✅ Data uploaded and processed successfully! (Available for reading)")
        except Exception as insert_error:
            # If insert fails, log but don't crash - might be a transient issue
            logger.error(f"❌ Error during insert: {insert_error}")
            logger.error("   Attempting insert without wait_for_job...")
            try:
                fg.insert(df, write_options={"wait_for_job": False})
                logger.info("✅ Data uploaded (async mode - may not be immediately readable)")
            except Exception as retry_error:
                logger.error(f"❌ Insert failed even without wait: {retry_error}")
                raise  # Re-raise to trigger sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Fatal error uploading to Hopsworks: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    try:
        # Fetch current data
        df = fetch_current(LAT, LON)
        
        if df.empty:
            logger.error("❌ No data fetched. Exiting.")
            sys.exit(1)
        
        # Add ingestion timestamp to track when data was actually ingested
        ingestion_time = pd.Timestamp.now(tz='UTC')
        df["ingestion_timestamp"] = ingestion_time
        
        # Add derived time features (matching backfill.py pattern)
        df["hour"] = df["timestamp"].dt.hour
        df["day"] = df["timestamp"].dt.day
        df["month"] = df["timestamp"].dt.month
        
        # AQI change rate will be calculated in push_to_hopsworks() 
        # (needs access to previous data from feature store)
        
        # Simple sanity checks (drift checks)
        if "pm2_5" in df.columns:
            if not (df["pm2_5"].between(0, 200).all()):
                logger.warning("⚠️ WARNING: pm2_5 out of expected range (0-200)")
        
        # Check if using local storage (set USE_LOCAL_STORAGE=1 in .env)
        USE_LOCAL_STORAGE = os.getenv("USE_LOCAL_STORAGE", "0") == "1"
        
        if USE_LOCAL_STORAGE:
            # Use local storage instead of Hopsworks
            logger.info("💾 Using LOCAL STORAGE (free alternative to Hopsworks)")
            try:
                from local_storage import save_data, calculate_aqi_change_rate
                
                # Calculate AQI change rate using local storage
                aqi_change_rate = calculate_aqi_change_rate(df)
                df["aqi_change_rate"] = aqi_change_rate
                
                # Save to local storage
                save_data(df)
                logger.info("✅ Data saved to local storage successfully!")
            except ImportError:
                logger.error("❌ local_storage.py not found! Falling back to Hopsworks.")
                push_to_hopsworks(df)
            except Exception as e:
                logger.error(f"❌ Local storage failed: {e}. Falling back to Hopsworks.")
                push_to_hopsworks(df)
        else:
            # Upload to Hopsworks (this will also calculate AQI change rate)
            push_to_hopsworks(df)
        
        logger.info(f"✅ Successfully pushed data: {df.to_dict(orient='records')[0]}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
