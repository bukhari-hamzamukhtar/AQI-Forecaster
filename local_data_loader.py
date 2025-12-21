"""
Local Data Loader - Drop-in replacement for Hopsworks data loading
Use this when Hopsworks credits are exhausted.
"""
import pandas as pd
from datetime import timedelta
import logging

logger = logging.getLogger("local_data_loader")

try:
    from local_storage import read_data, get_recent_data, get_latest_data, get_data_stats
    LOCAL_STORAGE_AVAILABLE = True
except ImportError:
    LOCAL_STORAGE_AVAILABLE = False
    logger.warning("⚠️ local_storage.py not found. Install it to use local storage.")

def load_data_local():
    """
    Load all data from local storage (replacement for Hopsworks load_data).
    
    Returns:
        df, None (no project object needed for local storage)
    """
    if not LOCAL_STORAGE_AVAILABLE:
        logger.error("❌ Local storage not available. Please create local_storage.py")
        return None, None
    
    try:
        logger.info("💾 Loading data from LOCAL STORAGE...")
        df = read_data()
        
        if df.empty:
            logger.warning("⚠️ No data found in local storage. Run feature_pipeline.py first.")
            return None, None
        
        logger.info(f"✅ Loaded {len(df)} rows from {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df, None
    except Exception as e:
        logger.error(f"❌ Error loading local data: {e}")
        return None, None

def fetch_recent_data_local(hours=24):
    """
    Fetch recent data from local storage (replacement for Hopsworks fetch_recent_data).
    
    Args:
        hours: Number of hours of recent data to fetch
    
    Returns:
        DataFrame with recent data
    """
    if not LOCAL_STORAGE_AVAILABLE:
        return pd.DataFrame()
    
    try:
        df = get_recent_data(hours=hours)
        if not df.empty:
            df = df.sort_values("timestamp", ascending=False)
        return df
    except Exception as e:
        logger.error(f"❌ Error fetching recent data: {e}")
        return pd.DataFrame()

def fetch_training_data_local(days=7):
    """
    Fetch training data from local storage (replacement for Hopsworks fetch_training_data).
    
    Args:
        days: Number of days of training data
    
    Returns:
        DataFrame with training data
    """
    if not LOCAL_STORAGE_AVAILABLE:
        return pd.DataFrame()
    
    try:
        hours = days * 24
        df = get_recent_data(hours=hours)
        if not df.empty:
            df = df.sort_values("timestamp")
        return df
    except Exception as e:
        logger.error(f"❌ Error fetching training data: {e}")
        return pd.DataFrame()

def fetch_all_data_local(limit=None):
    """
    Fetch all data from local storage (replacement for Hopsworks fetch_all_data).
    
    Args:
        limit: Optional limit on number of rows
    
    Returns:
        DataFrame with all data
    """
    if not LOCAL_STORAGE_AVAILABLE:
        return pd.DataFrame()
    
    try:
        df = read_data(limit=limit)
        if not df.empty:
            df = df.sort_values("timestamp")
        return df
    except Exception as e:
        logger.error(f"❌ Error fetching all data: {e}")
        return pd.DataFrame()

