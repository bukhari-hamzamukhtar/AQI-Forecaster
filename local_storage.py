"""
Local Storage Module - Free alternative to Hopsworks Feature Store
Stores data in local Parquet files with the same structure as Hopsworks.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger("local_storage")

# Data directory
DATA_DIR = Path("local_data")
DATA_DIR.mkdir(exist_ok=True)

# Main data file
DATA_FILE = DATA_DIR / "aqi_features.parquet"
METADATA_FILE = DATA_DIR / "metadata.json"

def init_storage():
    """Initialize local storage - create empty file if it doesn't exist."""
    if not DATA_FILE.exists():
        # Create empty DataFrame with proper schema
        empty_df = pd.DataFrame(columns=[
            'timestamp', 'pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co',
            'hour', 'day', 'month', 'aqi', 'aqi_change_rate', 'ingestion_timestamp'
        ])
        empty_df['timestamp'] = pd.to_datetime(empty_df['timestamp'])
        # Use pyarrow engine for Parquet (faster and more reliable)
        try:
            empty_df.to_parquet(DATA_FILE, index=False, engine='pyarrow')
        except ImportError:
            # Fallback to fastparquet if pyarrow not available
            try:
                empty_df.to_parquet(DATA_FILE, index=False, engine='fastparquet')
            except ImportError:
                logger.error("❌ Neither pyarrow nor fastparquet installed!")
                logger.error("   Install with: pip install pyarrow")
                raise
        logger.info(f"✅ Created new local storage file: {DATA_FILE}")
    return DATA_FILE

def save_data(df):
    """
    Save/append data to local storage (equivalent to Hopsworks insert).
    
    Args:
        df: DataFrame with columns matching Hopsworks schema
    """
    if df.empty:
        logger.warning("⚠️ No data to save")
        return
    
    # Initialize if needed
    init_storage()
    
    # Load existing data
    if DATA_FILE.exists():
        try:
            # Try pyarrow first, fallback to fastparquet
            try:
                existing_df = pd.read_parquet(DATA_FILE, engine='pyarrow')
            except ImportError:
                existing_df = pd.read_parquet(DATA_FILE, engine='fastparquet')
        except Exception as e:
            logger.warning(f"⚠️ Could not read existing data: {e}. Creating new file.")
            existing_df = pd.DataFrame()
    else:
        existing_df = pd.DataFrame()
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    
    # Remove duplicates (same timestamp)
    if not existing_df.empty:
        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
        if existing_df['timestamp'].dt.tz is None:
            existing_df['timestamp'] = existing_df['timestamp'].dt.tz_localize('UTC')
        
        # Merge: update existing rows, append new ones
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    else:
        combined_df = df.copy()
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    
    # Save to parquet (use pyarrow engine)
    try:
        combined_df.to_parquet(DATA_FILE, index=False, engine='pyarrow')
    except ImportError:
        # Fallback to fastparquet if pyarrow not available
        try:
            combined_df.to_parquet(DATA_FILE, index=False, engine='fastparquet')
        except ImportError:
            logger.error("❌ Neither pyarrow nor fastparquet installed!")
            logger.error("   Install with: pip install pyarrow")
            raise
    logger.info(f"✅ Saved {len(df)} row(s). Total rows: {len(combined_df)}")
    
    return combined_df

def read_data(start_time=None, end_time=None, limit=None):
    """
    Read data from local storage (equivalent to Hopsworks read).
    
    Args:
        start_time: Optional start timestamp filter
        end_time: Optional end timestamp filter
        limit: Optional limit on number of rows
    
    Returns:
        DataFrame with all data
    """
    if not DATA_FILE.exists():
        logger.warning("⚠️ No local data file found. Returning empty DataFrame.")
        return pd.DataFrame()
    
    try:
        # Try pyarrow first, fallback to fastparquet
        try:
            df = pd.read_parquet(DATA_FILE, engine='pyarrow')
        except ImportError:
            df = pd.read_parquet(DATA_FILE, engine='fastparquet')
        
        if df.empty:
            return df
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        
        # Apply filters
        if start_time:
            start_time = pd.to_datetime(start_time)
            if start_time.tz is None:
                start_time = start_time.tz_localize('UTC')
            df = df[df['timestamp'] >= start_time]
        
        if end_time:
            end_time = pd.to_datetime(end_time)
            if end_time.tz is None:
                end_time = end_time.tz_localize('UTC')
            df = df[df['timestamp'] <= end_time]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Apply limit
        if limit:
            df = df.tail(limit)
        
        return df
    
    except Exception as e:
        logger.error(f"❌ Error reading local data: {e}")
        return pd.DataFrame()

def get_latest_data():
    """Get the most recent data point."""
    df = read_data()
    if df.empty:
        return pd.DataFrame()
    return df.tail(1)

def get_recent_data(hours=24):
    """Get recent data (last N hours)."""
    cutoff_time = pd.Timestamp.now(tz='UTC') - timedelta(hours=hours)
    return read_data(start_time=cutoff_time)

def calculate_aqi_change_rate(df):
    """
    Calculate AQI change rate by comparing with previous hour.
    Uses local storage to find previous hour's AQI.
    """
    if df.empty:
        return 0.0
    
    current_aqi = df.iloc[0].get('aqi', 0)
    if current_aqi == 0:
        return 0.0
    
    try:
        current_ts = pd.to_datetime(df.iloc[0]['timestamp'])
        if current_ts.tz is None:
            current_ts = current_ts.tz_localize('UTC')
        
        # Get previous hour's data
        prev_hour_ts = current_ts - timedelta(hours=1)
        prev_data = read_data(
            start_time=prev_hour_ts - timedelta(minutes=30),
            end_time=prev_hour_ts + timedelta(minutes=30)
        )
        
        if not prev_data.empty:
            prev_aqi = prev_data.iloc[-1].get('aqi', 0)
            if prev_aqi > 0:
                return (current_aqi - prev_aqi) / prev_aqi
        return 0.0
    except Exception as e:
        logger.warning(f"⚠️ Could not calculate AQI change rate: {e}")
        return 0.0

def get_data_stats():
    """Get statistics about stored data."""
    df = read_data()
    if df.empty:
        return {
            'total_rows': 0,
            'date_range': None,
            'latest_timestamp': None
        }
    
    return {
        'total_rows': len(df),
        'date_range': (df['timestamp'].min(), df['timestamp'].max()),
        'latest_timestamp': df['timestamp'].max()
    }

