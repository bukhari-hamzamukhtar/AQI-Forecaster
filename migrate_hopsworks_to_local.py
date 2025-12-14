"""
Migration Script: Copy data from Hopsworks to Local Storage
Run this ONCE to migrate your existing Hopsworks data to local storage.
"""
import os
import pandas as pd
from dotenv import load_dotenv
import hopsworks
import logging
from local_storage import save_data, get_data_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("migration")

load_dotenv()

HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

def migrate_hopsworks_to_local():
    """Migrate all data from Hopsworks Feature Store to local storage."""
    
    if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT:
        logger.error("❌ HOPSWORKS_API_KEY or HOPSWORKS_PROJECT not found in .env")
        logger.error("   Cannot migrate without Hopsworks credentials.")
        return False
    
    try:
        logger.info("🚀 Connecting to Hopsworks...")
        project = hopsworks.login(project=HOPSWORKS_PROJECT, api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()
        
        # Get latest feature group version
        try:
            all_fgs = fs.get_feature_group(name="aqi_features", version=None)
            if isinstance(all_fgs, list) and len(all_fgs) > 0:
                latest_version = max(fg.version for fg in all_fgs if hasattr(fg, 'version') and fg.version is not None)
                logger.info(f"📦 Found {len(all_fgs)} versions, using latest: v{latest_version}")
                fg = fs.get_feature_group(name="aqi_features", version=latest_version)
            else:
                fg = fs.get_feature_group(name="aqi_features", version=3)
                latest_version = 3
        except Exception as e:
            logger.error(f"❌ Could not get feature group: {e}")
            return False
        
        logger.info("📥 Downloading all data from Hopsworks...")
        try:
            # Try without Hive first (faster)
            df = fg.read(read_options={"use_hive": False})
        except Exception:
            # Fall back to Hive
            logger.info("   -> Falling back to Hive engine...")
            df = fg.read(read_options={"use_hive": True})
        
        if df.empty:
            logger.warning("⚠️ No data found in Hopsworks Feature Store")
            return False
        
        logger.info(f"✅ Downloaded {len(df)} rows from Hopsworks")
        logger.info(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Save to local storage
        logger.info("💾 Saving to local storage...")
        save_data(df)
        
        # Verify
        stats = get_data_stats()
        logger.info(f"✅ Migration complete!")
        logger.info(f"   Total rows in local storage: {stats['total_rows']}")
        logger.info(f"   Date range: {stats['date_range']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("="*70)
    print("🔄 MIGRATING HOPSWORKS DATA TO LOCAL STORAGE")
    print("="*70)
    print()
    print("This script will:")
    print("  1. Connect to Hopsworks (if you still have access)")
    print("  2. Download all data from Feature Store")
    print("  3. Save it to local_data/aqi_features.parquet")
    print()
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        exit(0)
    
    success = migrate_hopsworks_to_local()
    
    if success:
        print()
        print("="*70)
        print("✅ MIGRATION SUCCESSFUL!")
        print("="*70)
        print()
        print("Next steps:")
        print("  1. Add USE_LOCAL_STORAGE=1 to your .env file")
        print("  2. Run: python feature_pipeline.py (for new data)")
        print("  3. Run: python training_pipeline.py (to train models)")
        print("  4. Run: streamlit run app.py (to use the app)")
    else:
        print()
        print("="*70)
        print("❌ MIGRATION FAILED")
        print("="*70)
        print()
        print("Possible reasons:")
        print("  - No Hopsworks credits left (can't connect)")
        print("  - No data in Hopsworks")
        print("  - Network/connection issues")
        print()
        print("Alternative: Start fresh with local storage")
        print("  Just run: python feature_pipeline.py (with USE_LOCAL_STORAGE=1)")

