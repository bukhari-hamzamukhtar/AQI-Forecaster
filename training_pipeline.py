import os
import certifi
import joblib
import pandas as pd
import numpy as np
import hopsworks
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from math import sqrt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, optimizers, callbacks
from dotenv import load_dotenv
import tempfile
import shutil
import warnings
import time
import sys
from datetime import timedelta
warnings.filterwarnings('ignore')

# Statistical time-series models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️  statsmodels not available. SARIMA will be skipped.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️  prophet not available. Prophet will be skipped.")

# -------------------------------------------------
# 🔧 WINDOWS SSL FIX (CRITICAL)
# -------------------------------------------------
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

load_dotenv()
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

def load_data():
    # Check if using local storage (set USE_LOCAL_STORAGE=1 in .env)
    USE_LOCAL_STORAGE = os.getenv("USE_LOCAL_STORAGE", "0") == "1"
    
    if USE_LOCAL_STORAGE:
        print("💾 Using LOCAL STORAGE (free alternative to Hopsworks)...")
        try:
            from local_data_loader import load_data_local
            df, project = load_data_local()
            return df, project
        except ImportError:
            print("⚠️ local_data_loader.py not found. Falling back to Hopsworks.")
        except Exception as e:
            print(f"⚠️ Local storage failed: {e}. Falling back to Hopsworks.")
    
    print("🚀 Connecting to Hopsworks...")
    project = hopsworks.login(project=HOPSWORKS_PROJECT, api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    
    # 1. Fetch Feature Group (Latest Version)
    try:
        # Get all versions and find the latest
        all_fgs = fs.get_feature_group(name="aqi_features", version=None)
        if isinstance(all_fgs, list) and len(all_fgs) > 0:
            latest_version = max(fg.version for fg in all_fgs if hasattr(fg, 'version') and fg.version is not None)
            print(f"📦 Found {len(all_fgs)} versions, using latest: v{latest_version}")
            fg = fs.get_feature_group(name="aqi_features", version=latest_version)
        else:
            # Fallback to version 3
            fg = fs.get_feature_group(name="aqi_features", version=3)
    except:
        print("❌ Error: Feature Group V3 not found. Did the backfill finish?")
        return None, None

    # 2. Create Feature View (Version 4 - Fresh Snapshot)
    try:
        feature_view = fs.get_feature_view(name="aqi_view", version=4)
        if feature_view is None:
            raise ValueError("Feature view not found")
    except:
        print("   -> Creating new Feature View...")
        try:
            feature_view = fs.create_feature_view(
                name="aqi_view",
                version=4,
                labels=[],  # No labels - we need pm2_5 as a feature for forecasting
                query=fg.select_all()
            )
        except Exception as e:
            print(f"❌ Error creating feature view: {e}")
            return None, None
    
    # Check if feature_view is still None after creation
    if feature_view is None:
        print("❌ Error: Feature view is None after creation attempt")
        return None, None
    
    # 3. Read Data (Using HIVE to prevent Windows Crash)
    print("📥 Downloading data (using safe engine)...")
    try:
        df = feature_view.get_batch_data(read_options={"use_hive": True})
        
        # Safety Check - if pm2_5 is missing, read directly from feature group
        if "pm2_5" not in df.columns:
            print("⚠️  'pm2_5' missing from feature view (likely marked as label). Reading from feature group directly...")
            df = fg.read(read_options={"use_hive": True})
    except Exception as e:
        print(f"❌ Read failed: {e}")
        print("   -> Trying to read directly from feature group...")
        try:
            df = fg.read(read_options={"use_hive": True})
        except Exception as e2:
            print(f"❌ Feature group read also failed: {e2}")
            return None, None
        
    # Final Safety Check
    if "pm2_5" not in df.columns:
        print(f"❌ CRITICAL: 'pm2_5' column missing. Columns found: {df.columns.tolist()}")
        return None, None
        
    return df, project

def feature_engineer_advanced(df):
    """
    Apply advanced time-series feature engineering:
    1. Lag Features (Windowing) - Capture historical patterns
    2. Velocity & Acceleration - Capture speed of change (gradients)
    3. Multi-Day Seasonality - Robust daily patterns
    4. Rolling Window Statistics - Capture trends and volatility
    5. Cyclical Time Encoding - Convert time features to sin/cos
    6. Domain Features - Weekend indicators, etc.
    """
    print("⚡ Applying Advanced Time-Series Feature Engineering...")
    
    # Ensure timestamp is datetime type
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort is critical for lag features (must be chronological)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # --------------------------------------
    # 1. LAG FEATURES (The History)
    # --------------------------------------
    # Capture the immediate trend (last 3 hours)
    df['pm2_5_lag1'] = df['pm2_5'].shift(1)
    df['pm2_5_lag2'] = df['pm2_5'].shift(2)
    df['pm2_5_lag3'] = df['pm2_5'].shift(3)
    
    # Capture the Daily Pattern (Same time yesterday, 2 days ago, 3 days ago)
    df['pm2_5_lag24'] = df['pm2_5'].shift(24)  # Same time yesterday
    df['pm2_5_lag48'] = df['pm2_5'].shift(48)  # Same time 2 days ago
    df['pm2_5_lag72'] = df['pm2_5'].shift(72)  # Same time 3 days ago
    
    # --------------------------------------
    # 2. VELOCITY & ACCELERATION (The "Slope")
    # --------------------------------------
    # Velocity: How much did it change in the last hour?
    df['diff_1h'] = df['pm2_5'].diff(1)
    
    # Velocity: How much did it change in the last 24 hours? (Daily Trend)
    df['diff_24h'] = df['pm2_5'].diff(24)
    
    # Acceleration: Is the change speeding up or slowing down?
    df['diff_1h_acc'] = df['diff_1h'].diff(1)
    
    # --------------------------------------
    # 3. MULTI-DAY SEASONALITY (The "Rhythm")
    # --------------------------------------
    # Baseline Profile: Average PM2.5 for this specific hour over the last 3 days
    # This creates a strong hint for the hourly shape
    df['recent_hour_avg'] = df[['pm2_5_lag24', 'pm2_5_lag48', 'pm2_5_lag72']].mean(axis=1)
    
    # --------------------------------------
    # 4. ROLLING WINDOW STATISTICS (The Trend)
    # --------------------------------------
    # Average of the last 24 hours (Smooths out noise)
    df['pm2_5_rolling_mean_24h'] = df['pm2_5'].rolling(window=24).mean()
    
    # Volatility (Standard Deviation) of last 24 hours
    df['pm2_5_rolling_std_24h'] = df['pm2_5'].rolling(window=24).std()
    
    # --------------------------------------
    # 5. CYCLICAL TIME ENCODING
    # --------------------------------------
    # Transform Hour (0-23) into cyclical features
    # This ensures hour 23 and hour 0 are close together (not far apart)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Transform Month (1-12) into cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Keep raw hour for Tree models (XGBoost/RandomForest prefer integer splits)
    df['hour_raw'] = df['hour']
    
    # --------------------------------------
    # 6. DOMAIN FEATURES
    # --------------------------------------
    # Is it a weekend? (Pollution is often lower on weekends)
    df['is_weekend'] = df['timestamp'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    
    print("   ✅ Lag features: pm2_5_lag1, pm2_5_lag2, pm2_5_lag3, pm2_5_lag24, pm2_5_lag48, pm2_5_lag72")
    print("   ✅ Velocity/Acceleration: diff_1h, diff_24h, diff_1h_acc")
    print("   ✅ Multi-day seasonality: recent_hour_avg")
    print("   ✅ Rolling stats: pm2_5_rolling_mean_24h, pm2_5_rolling_std_24h")
    print("   ✅ Cyclical time: hour_sin, hour_cos, month_sin, month_cos")
    print("   ✅ Raw features: hour_raw (for Tree models)")
    print("   ✅ Domain features: is_weekend")
    
    return df


def prepare_data(df, strategy='recursive'):
    """
    Prepare data for forecasting.
    
    Args:
        df: Input dataframe
        strategy: 'direct' (T+72) or 'recursive' (T+1, then loop 72 times)
    
    Returns:
        X_scaled, y, scaler, df
    """
    print("🛠 Preparing data for forecasting...")
    print(f"   📊 Strategy: {strategy.upper()}")
    df = df.sort_values("timestamp").reset_index(drop=True)  # Reset index to ensure clean 0-based indexing
    
    # --- APPLY ADVANCED FEATURE ENGINEERING ---
    df = feature_engineer_advanced(df)
    # -----------------------------------------
    
    if strategy == 'recursive':
        # RECURSIVE STRATEGY: Predict T+1 (next hour)
        # Then use that prediction to predict T+2, and loop 72 times
        print("   -> Creating target: PM2.5 value 1 hour ahead (for recursive forecasting)")
        df["pm2_5_next"] = df["pm2_5"].shift(-1)
        target_col = "pm2_5_next"
    else:
        # DIRECT STRATEGY: Predict T+72 directly (3 days ahead)
        FORECAST_HOURS = 72
        print(f"   -> Creating target: PM2.5 value {FORECAST_HOURS} hours (3 days) ahead")
        df["pm2_5_3day"] = df["pm2_5"].shift(-FORECAST_HOURS)
        target_col = "pm2_5_3day"
    
    # Drop NaNs created by Lags AND the Target Shift
    df = df.dropna().reset_index(drop=True)  # Reset index after dropna
    
    # Define feature list (Updated to include all engineered features)
    base_features = [
        # Original pollutants
        "pm2_5", "pm10", "o3", "no2", "so2", "co",
        
        # Lag features (Historical patterns)
        "pm2_5_lag1", "pm2_5_lag2", "pm2_5_lag3", 
        "pm2_5_lag24", "pm2_5_lag48", "pm2_5_lag72",
        
        # Velocity & Acceleration (Speed of change)
        "diff_1h", "diff_24h", "diff_1h_acc",
        
        # Multi-day seasonality (Robust daily patterns)
        "recent_hour_avg",
        
        # Rolling statistics (Trends and volatility)
        "pm2_5_rolling_mean_24h", "pm2_5_rolling_std_24h",
        
        # Time features (Cyclical + Raw for Tree models)
        "hour_sin", "hour_cos", "hour_raw",  # Hour: cyclical for MLP/SVM, raw for Trees
        "month_sin", "month_cos", "month",   # Month: cyclical + raw
        "day",  # Day of month
        
        # Domain features
        "is_weekend"
    ]
    features = base_features.copy()
    
    # Include AQI change rate if available (optional feature)
    if "aqi_change_rate" in df.columns:
        # Check if aqi_change_rate has valid values (not all NaN)
        if df["aqi_change_rate"].notna().sum() > 0:
            features.append("aqi_change_rate")
            print("   -> Including AQI change rate feature")
        else:
            print("   -> AQI change rate column exists but is empty, skipping")
    else:
        print("   -> AQI change rate not available, using base features only")
    
    # Ensure all features exist in dataframe
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < len(features):
        missing = set(features) - set(available_features)
        print(f"   ⚠️  Warning: Missing features {missing}, using available features only")
        features = available_features
    
    X = df[features]
    y = df[target_col]
    
    # Scale Data (Required for Neural Nets/SVM)
    print("⚖️ Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features, index=X.index)  # Preserve index
    
    print(f"   ✅ Final feature count: {len(features)} features")
    
    return X_scaled, y, scaler, df  # Return df with reset index for time-series models

def train_xgboost_optimized(X_train, y_train, X_val, y_val, n_iter=20, cv=3):
    """Train XGBoost with hyperparameter tuning."""
    print("🔍 Tuning XGBoost hyperparameters...")
    start_time = time.time()
    
    param_distributions = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2, 0.3]
    }
    
    base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    rmse_scorer = make_scorer(lambda y_true, y_pred: sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)
    
    total_combinations = n_iter * cv
    print(f"   Testing {n_iter} parameter combinations with {cv}-fold CV ({total_combinations} total fits)...")
    
    search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=rmse_scorer,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    print(f"   ✅ Best XGBoost params: {search.best_params_}")
    print(f"   📊 Best CV RMSE: {-search.best_score_:.4f} | Time: {elapsed_time:.1f}s")
    
    return search.best_estimator_


def train_random_forest_optimized(X_train, y_train, X_val, y_val, n_iter=20, cv=3):
    """Train Random Forest with hyperparameter tuning."""
    print("🔍 Tuning Random Forest hyperparameters...")
    start_time = time.time()
    
    param_distributions = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [5, 10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    rmse_scorer = make_scorer(lambda y_true, y_pred: sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)
    
    total_combinations = n_iter * cv
    print(f"   Testing {n_iter} parameter combinations with {cv}-fold CV ({total_combinations} total fits)...")
    
    search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=rmse_scorer,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    print(f"   ✅ Best Random Forest params: {search.best_params_}")
    print(f"   📊 Best CV RMSE: {-search.best_score_:.4f} | Time: {elapsed_time:.1f}s")
    
    return search.best_estimator_


from sklearn.svm import LinearSVR

def train_svm_optimized(X_train, y_train, X_val, y_val, n_iter=15, cv=3):
    """
    Train SVM using LinearSVR (FAST implementation).
    Replaces standard SVR to avoid 20+ hour training times.
    """
    print("🔍 Tuning SVM hyperparameters (Using LinearSVR for speed)...")
    start_time = time.time()
    
    # These 'loss' values are just settings built into LinearSVR. You don't need to define them.
    param_distributions = {
        'C': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
        'epsilon': [0.0, 0.05, 0.1, 0.2, 0.5],
        'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'], 
        'intercept_scaling': [1.0, 0.5, 2.0]
    }
    
    # max_iter=3000 ensures it finishes quickly
    base_model = LinearSVR(random_state=42, max_iter=3000, dual="auto")
    
    rmse_scorer = make_scorer(lambda y_true, y_pred: sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)
    
    # Cap iterations to 15 to ensure speed
    n_iter_safe = min(n_iter, 15)
    
    total_combinations = n_iter_safe * cv
    print(f"   Testing {n_iter_safe} parameter combinations with {cv}-fold CV ({total_combinations} total fits)...")
    
    search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=n_iter_safe,
        cv=cv,
        scoring=rmse_scorer,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    # Optimization: If dataset is huge (>20k rows), tune on a sample for speed
    if len(X_train) > 20000:
        print(f"   ⚠️ Large dataset detected ({len(X_train)} rows). Tuning on 10k sample for speed...")
        # using numpy (np) which is already imported in your file
        sample_indices = np.random.choice(len(X_train), 10000, replace=False)
        X_tune = X_train.iloc[sample_indices]
        y_tune = y_train.iloc[sample_indices]
        search.fit(X_tune, y_tune)
    else:
        search.fit(X_train, y_train)
        
    elapsed_time = time.time() - start_time
    
    best_params = search.best_params_
    
    # --- Margin Type Logic (Same as before) ---
    c_val = best_params['C']
    margin_type = "Hard" if c_val >= 100 else "Soft" if c_val <= 1 else "Medium"
    
    print(f"   ✅ Best SVM (Linear) params: {best_params}")
    print(f"   📊 Margin type: {margin_type} (C={c_val})")
    print(f"   📊 Best CV RMSE: {-search.best_score_:.4f} | Time: {elapsed_time:.1f}s")
    
    best_model = search.best_estimator_
    
    # If we sampled, refit on full data to ensure best performance
    if len(X_train) > 20000:
        print("   🔄 Retraining best model on FULL dataset...")
        best_model.fit(X_train, y_train)
        
    return best_model


def create_mlp_model(neurons_layer1=64, neurons_layer2=32, learning_rate=0.001, optimizer='adam', input_dim=None):
    """Create MLP model with configurable architecture."""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(neurons_layer1, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(neurons_layer2, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    
    if optimizer == 'adam':
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        opt = optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model


def train_mlp_optimized(X_train, y_train, X_val, y_val, n_iter=15):
    """Train Neural Network with hyperparameter tuning."""
    print("🔍 Tuning Neural Network hyperparameters...")
    start_time = time.time()
    
    param_distributions = {
        'neurons_layer1': [32, 64, 128, 256],
        'neurons_layer2': [16, 32, 64, 128],
        'learning_rate': [0.0001, 0.001, 0.01, 0.1],
        'optimizer': ['adam', 'rmsprop', 'sgd'],
        'batch_size': [16, 32, 64, 128],
        'epochs': [10, 20, 30, 50]
    }
    
    best_score = float('inf')
    best_model = None
    best_params = None
    
    # Sample parameter combinations
    np.random.seed(42)
    param_samples = []
    for _ in range(n_iter):
        sample = {}
        for key, values in param_distributions.items():
            sample[key] = np.random.choice(values)
        param_samples.append(sample)
    
    print(f"   Testing {n_iter} different architectures...")
    
    for i, params in enumerate(param_samples):
        iter_start = time.time()
        try:
            # Create model
            model = create_mlp_model(
                neurons_layer1=params['neurons_layer1'],
                neurons_layer2=params['neurons_layer2'],
                learning_rate=params['learning_rate'],
                optimizer=params['optimizer'],
                input_dim=X_train.shape[1]
            )
            
            # Early stopping callback
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                callbacks=[early_stop],
                verbose=0
            )
            
            # Evaluate on validation set
            val_pred = model.predict(X_val, verbose=0).flatten()
            val_rmse = sqrt(mean_squared_error(y_val, val_pred))
            
            iter_time = time.time() - iter_start
            progress = ((i + 1) / n_iter) * 100
            elapsed = time.time() - start_time
            avg_time_per_iter = elapsed / (i + 1)
            eta = avg_time_per_iter * (n_iter - (i + 1))
            
            if val_rmse < best_score:
                best_score = val_rmse
                best_model = model
                best_params = params.copy()  # Store params for later reference
                print(f"\r   [{i+1}/{n_iter}] {progress:.1f}% | New best! RMSE: {val_rmse:.4f} | Time: {iter_time:.1f}s | ETA: {eta:.0f}s", end='', flush=True)
            else:
                print(f"\r   [{i+1}/{n_iter}] {progress:.1f}% | RMSE: {val_rmse:.4f} | Time: {iter_time:.1f}s | ETA: {eta:.0f}s", end='', flush=True)
        
        except Exception as e:
            progress = ((i + 1) / n_iter) * 100
            print(f"\r   [{i+1}/{n_iter}] {progress:.1f}% | Failed: {str(e)[:50]}...", end='', flush=True)
            continue
    
    elapsed_time = time.time() - start_time
    print()  # New line after progress
    
    if best_model is None:
        print("   ⚠️ All MLP configurations failed, using default...")
        best_model = create_mlp_model(input_dim=X_train.shape[1])
        best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, verbose=0)
        best_params = {'neurons_layer1': 64, 'neurons_layer2': 32, 'learning_rate': 0.001, 'optimizer': 'adam'}
    
    # Store best params as attribute for later retrieval
    best_model.best_params_ = best_params
    
    print(f"   ✅ Best MLP params: {best_params}")
    print(f"   📊 Best Validation RMSE: {best_score:.4f} | Total Time: {elapsed_time:.1f}s")
    
    return best_model


def train_all_models(X_train, y_train, X_val, y_val, ts_train=None, ts_test=None, ts_train_df=None):
    """Train all models with hyperparameter tuning."""
    overall_start = time.time()
    print("\n🧠 Training models with automatic hyperparameter optimization...")
    print("="*70)
    
    # Split training data further for cross-validation
    # Use 80% of train for actual training, 20% for validation during tuning
    X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=False, random_state=42
    )
    
    # 1. XGBoost with hyperparameter tuning
    print("\n[1/6] XGBoost")
    model_start = time.time()
    xgb_m = train_xgboost_optimized(X_train_tune, y_train_tune, X_val_tune, y_val_tune, n_iter=25, cv=3)
    print(f"   ⏱️  XGBoost training completed in {time.time() - model_start:.1f}s")
    
    # 2. Random Forest with hyperparameter tuning
    print("\n[2/6] Random Forest")
    model_start = time.time()
    rf_m = train_random_forest_optimized(X_train_tune, y_train_tune, X_val_tune, y_val_tune, n_iter=25, cv=3)
    print(f"   ⏱️  Random Forest training completed in {time.time() - model_start:.1f}s")
    
    # 3. SVM with hard/soft margin and kernel selection
    print("\n[3/6] Support Vector Machine")
    model_start = time.time()
    svm_m = train_svm_optimized(X_train_tune, y_train_tune, X_val_tune, y_val_tune, n_iter=30, cv=3)
    print(f"   ⏱️  SVM training completed in {time.time() - model_start:.1f}s")
    
    # 4. Neural Network with architecture tuning
    print("\n[4/6] Neural Network (MLP)")
    model_start = time.time()
    mlp_m = train_mlp_optimized(X_train_tune, y_train_tune, X_val_tune, y_val_tune, n_iter=20)
    print(f"   ⏱️  Neural Network training completed in {time.time() - model_start:.1f}s")
    
    # 5. SARIMA (statistical time-series model)
    sarima_m = None
    if ts_train is not None:
        print("\n[5/6] SARIMA (Statistical Time-Series)")
        model_start = time.time()
        sarima_m = train_sarima(ts_train, ts_test)
        if sarima_m:
            print(f"   ⏱️  SARIMA training completed in {time.time() - model_start:.1f}s")
        else:
            print(f"   ⏱️  SARIMA skipped (time: {time.time() - model_start:.1f}s)")
    else:
        print("\n[5/6] SARIMA - Skipped (time-series data not provided)")
    
    # 6. Prophet (Facebook time-series model)
    prophet_m = None
    if ts_train_df is not None:
        print("\n[6/6] Prophet (Facebook Time-Series)")
        model_start = time.time()
        prophet_m = train_prophet(ts_train_df)
        if prophet_m:
            print(f"   ⏱️  Prophet training completed in {time.time() - model_start:.1f}s")
        else:
            print(f"   ⏱️  Prophet skipped (time: {time.time() - model_start:.1f}s)")
    else:
        print("\n[6/6] Prophet - Skipped (time-series data not provided)")
    
    total_time = time.time() - overall_start
    print("\n" + "="*70)
    print(f"✅ All models trained with optimized hyperparameters!")
    print(f"⏱️  Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print("="*70)
    
    return xgb_m, rf_m, svm_m, mlp_m, sarima_m, prophet_m


def prepare_time_series_data(df, train_indices, test_indices):
    """Prepare time-series data for SARIMA and Prophet models."""
    # df already has reset index (0-based) from prepare_data, and train_indices/test_indices match
    # Sort by timestamp to ensure chronological order
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    
    # Filter indices to only those that exist in df_sorted
    valid_train_indices = [idx for idx in train_indices if idx in df_sorted.index]
    valid_test_indices = [idx for idx in test_indices if idx in df_sorted.index]
    
    if len(valid_train_indices) == 0 or len(valid_test_indices) == 0:
        print(f"   ⚠️  Warning: Index mismatch. Train indices: {len(valid_train_indices)}, Test indices: {len(valid_test_indices)}")
        print(f"   -> df_sorted index range: {df_sorted.index.min()} to {df_sorted.index.max()}")
        print(f"   -> train_indices range: {min(train_indices)} to {max(train_indices)}")
        print(f"   -> test_indices range: {min(test_indices)} to {max(test_indices)}")
        # Fallback: use positional indexing
        train_size = int(len(df_sorted) * 0.8)
        valid_train_indices = df_sorted.index[:train_size]
        valid_test_indices = df_sorted.index[train_size:]
    
    # Extract PM2.5 time series with timestamps using the valid indices
    ts_train = df_sorted.loc[valid_train_indices, ['timestamp', 'pm2_5']].copy()
    ts_test = df_sorted.loc[valid_test_indices, ['timestamp', 'pm2_5']].copy()
    
    # Ensure timestamps are datetime
    ts_train['timestamp'] = pd.to_datetime(ts_train['timestamp'])
    ts_test['timestamp'] = pd.to_datetime(ts_test['timestamp'])
    
    # Sort by timestamp again to ensure chronological order
    ts_train = ts_train.sort_values('timestamp').reset_index(drop=True)
    ts_test = ts_test.sort_values('timestamp').reset_index(drop=True)
    
    # Set timestamp as index for SARIMA
    ts_train_indexed = ts_train.set_index('timestamp')['pm2_5']
    ts_test_indexed = ts_test.set_index('timestamp')['pm2_5']
    
    return ts_train_indexed, ts_test_indexed, ts_train, ts_test


def train_sarima(ts_train, ts_test):
    """Train SARIMA model with automatic parameter selection."""
    if not STATSMODELS_AVAILABLE:
        print("   ⚠️  statsmodels not available, skipping SARIMA")
        return None
    
    print("🔍 Training SARIMA model...")
    start_time = time.time()
    
    try:
        # Test for stationarity
        adf_result = adfuller(ts_train.dropna())
        is_stationary = adf_result[1] < 0.05
        
        if not is_stationary:
            print("   -> Data is non-stationary, will use differencing (d=1)")
            d = 1
        else:
            print("   -> Data appears stationary (d=0)")
            d = 0
        
        # Grid search for best SARIMA parameters
        # For hourly data: seasonal period = 24 (daily), 168 (weekly)
        best_aic = float('inf')
        best_model = None
        best_params = None
        
        # Try different parameter combinations
        p_values = [0, 1, 2]
        d_values = [d]  # Use determined d
        q_values = [0, 1, 2]
        P_values = [0, 1]
        D_values = [0, 1]  # Seasonal differencing
        Q_values = [0, 1]
        s = 24  # Hourly seasonality (24 hours)
        
        total_combinations = len(p_values) * len(d_values) * len(q_values) * len(P_values) * len(D_values) * len(Q_values)
        current = 0
        
        print(f"   -> Testing {total_combinations} SARIMA parameter combinations...")
        
        for p in p_values:
            for d_val in d_values:
                for q in q_values:
                    for P in P_values:
                        for D in D_values:
                            for Q in Q_values:
                                current += 1
                                try:
                                    model = SARIMAX(
                                        ts_train,
                                        order=(p, d_val, q),
                                        seasonal_order=(P, D, Q, s),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False
                                    )
                                    fitted_model = model.fit(disp=False, maxiter=50)
                                    
                                    if fitted_model.aic < best_aic:
                                        best_aic = fitted_model.aic
                                        best_model = fitted_model
                                        best_params = (p, d_val, q, P, D, Q, s)
                                        
                                    progress = (current / total_combinations) * 100
                                    print(f"\r   [{current}/{total_combinations}] {progress:.1f}% | Best AIC: {best_aic:.2f}", end='', flush=True)
                                
                                except Exception as e:
                                    continue
        
        print()  # New line
        
        if best_model is None:
            print("   ⚠️  Could not fit SARIMA model, using simple ARIMA(1,1,1)")
            try:
                model = ARIMA(ts_train, order=(1, 1, 1))
                best_model = model.fit(disp=False)
                best_params = (1, 1, 1, 0, 0, 0, 24)
            except:
                print("   ❌ Failed to fit fallback ARIMA model")
                return None
        
        elapsed_time = time.time() - start_time
        print(f"   ✅ SARIMA trained: order={best_params[:3]}, seasonal={best_params[3:6]}, period={best_params[6]}")
        print(f"   📊 AIC: {best_aic:.2f} | Time: {elapsed_time:.1f}s")
        
        # Store parameters for later reference
        best_model.best_params_ = best_params
        
        return best_model
    
    except Exception as e:
        print(f"   ❌ Error training SARIMA: {e}")
        return None


def train_prophet(ts_train_df):
    """Train Prophet model with hourly, daily, and weekly seasonality."""
    if not PROPHET_AVAILABLE:
        print("   ⚠️  prophet not available, skipping Prophet")
        return None
    
    print("🔍 Training Prophet model...")
    start_time = time.time()
    
    try:
        # Prepare data for Prophet (needs 'ds' and 'y' columns)
        prophet_data = ts_train_df[['timestamp', 'pm2_5']].copy()
        prophet_data.columns = ['ds', 'y']
        prophet_data = prophet_data.dropna()
        
        if len(prophet_data) < 48:  # Need at least 2 days of hourly data
            print("   ⚠️  Insufficient data for Prophet (need at least 48 hours)")
            return None
        
        # Create Prophet model with seasonalities
        model = Prophet(
            yearly_seasonality=False,  # Disable yearly (not enough data likely)
            weekly_seasonality=True,   # Weekly patterns (7 days)
            daily_seasonality=True,    # Daily patterns (24 hours)
            seasonality_mode='multiplicative',  # Multiplicative for AQI (better for pollution)
            changepoint_prior_scale=0.05,  # Regularization
            interval_width=0.95
        )
        
        # Add custom seasonalities if we have enough data
        if len(prophet_data) >= 168:  # At least 1 week
            model.add_seasonality(name='hourly', period=24, fourier_order=5)
        
        # Fit model
        model.fit(prophet_data)
        
        elapsed_time = time.time() - start_time
        print(f"   ✅ Prophet trained with daily & weekly seasonality")
        print(f"   📊 Time: {elapsed_time:.1f}s")
        
        return model
    
    except Exception as e:
        print(f"   ❌ Error training Prophet: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def register_winner(project, model, name, metrics, sample, scaler, description_suffix=""):
    """Save the winning model - supports both Hopsworks and local storage."""
    
    # Check if using local storage
    USE_LOCAL_STORAGE = os.getenv("USE_LOCAL_STORAGE", "0") == "1"
    
    print(f"💾 Saving winner: {name}...")
    
    # Save Model & Scaler
    model_file = f"aqi_best_model.pkl"
    scaler_file = "scaler.pkl"
    
    if hasattr(model, 'save'):
        # Keras/TensorFlow model
        model_file = "aqi_best_model.h5"
        model.save(model_file)
    else:
        # Scikit-learn/XGBoost model
        joblib.dump(model, model_file)
    
    joblib.dump(scaler, scaler_file)
    
    # Create description with metrics
    description = f"Winner: {name} | RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}{description_suffix}"
    
    if USE_LOCAL_STORAGE or project is None:
        # Save locally (no Hopsworks Model Registry)
        print("   💾 Saving to local files (no Model Registry)...")
        print(f"   ✅ Model saved: {model_file}")
        print(f"   ✅ Scaler saved: {scaler_file}")
        print(f"   📊 Metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")
        
        # Also save metadata to a text file
        metadata_file = "model_metadata.txt"
        with open(metadata_file, 'w') as f:
            f.write(f"Model: {name}\n")
            f.write(f"Description: {description}\n")
            f.write(f"RMSE: {metrics['rmse']:.4f}\n")
            f.write(f"MAE: {metrics['mae']:.4f}\n")
            f.write(f"R²: {metrics['r2']:.4f}\n")
        print(f"   ✅ Metadata saved: {metadata_file}")
        
    else:
        # Save to Hopsworks Model Registry
        mr = project.get_model_registry()
        
        # Create temporary directory for model artifacts
        temp_dir = tempfile.mkdtemp()
        try:
            # Copy model and scaler to temp directory
            import shutil
            if hasattr(model, 'save'):
                model.save(os.path.join(temp_dir, model_file))
            else:
                shutil.copy(model_file, os.path.join(temp_dir, model_file))
            shutil.copy(scaler_file, os.path.join(temp_dir, scaler_file))
            
            # Check if model exists, if so get the latest version and increment
            new_version = None
            try:
                existing_models = mr.get_models("aqi_best_model")
                if existing_models:
                    # Get the latest version number
                    latest_version = max(m.version for m in existing_models if hasattr(m, 'version'))
                    new_version = latest_version + 1
                    print(f"   -> Model exists (latest v{latest_version}), creating new version {new_version}...")
                else:
                    print("   -> Creating new model...")
            except Exception as e:
                # Model doesn't exist yet, will create version 1
                print("   -> Creating new model...")
            
            # Upload - save the entire directory
            m = mr.python.create_model(
                name="aqi_best_model",
                version=new_version,
                metrics=metrics,
                input_example=sample.iloc[0].to_dict(),
                description=description
            )
            m.save(temp_dir)  # Save the entire directory containing both files
            print("✅ Model & Scaler successfully uploaded to Hopsworks!")
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

def construct_features_for_recursive(df_historical, current_row, future_ts=None):
    """
    Simplified feature construction for recursive forecasting in training pipeline.
    Similar to app.py version but adapted for training context.
    """
    if future_ts is None:
        ts = pd.to_datetime(current_row['timestamp']) if 'timestamp' in current_row else pd.Timestamp.now()
    else:
        ts = pd.to_datetime(future_ts)
    
    df_hist = df_historical.sort_values('timestamp').reset_index(drop=True) if len(df_historical) > 0 else pd.DataFrame()
    current_pm25 = current_row.get('pm2_5', 0.0)
    
    features = {}
    features['pm2_5'] = current_row.get('pm2_5', 0.0)
    features['pm10'] = current_row.get('pm10', 0.0)
    features['o3'] = current_row.get('o3', 0.0)
    features['no2'] = current_row.get('no2', 0.0)
    features['so2'] = current_row.get('so2', 0.0)
    features['co'] = current_row.get('co', 0.0)
    
    # Lag features
    if len(df_hist) >= 72:
        features['pm2_5_lag1'] = df_hist.iloc[-1]['pm2_5'] if len(df_hist) >= 1 else current_pm25
        features['pm2_5_lag2'] = df_hist.iloc[-2]['pm2_5'] if len(df_hist) >= 2 else current_pm25
        features['pm2_5_lag3'] = df_hist.iloc[-3]['pm2_5'] if len(df_hist) >= 3 else current_pm25
        features['pm2_5_lag24'] = df_hist.iloc[-24]['pm2_5'] if len(df_hist) >= 24 else current_pm25
        features['pm2_5_lag48'] = df_hist.iloc[-48]['pm2_5'] if len(df_hist) >= 48 else current_pm25
        features['pm2_5_lag72'] = df_hist.iloc[-72]['pm2_5'] if len(df_hist) >= 72 else current_pm25
    else:
        features['pm2_5_lag1'] = current_pm25
        features['pm2_5_lag2'] = current_pm25
        features['pm2_5_lag3'] = current_pm25
        features['pm2_5_lag24'] = current_pm25
        features['pm2_5_lag48'] = current_pm25
        features['pm2_5_lag72'] = current_pm25
    
    # Velocity & Acceleration
    if len(df_hist) >= 24:
        pm25_1h_ago = df_hist.iloc[-1]['pm2_5'] if len(df_hist) >= 1 else current_pm25
        pm25_24h_ago = df_hist.iloc[-24]['pm2_5'] if len(df_hist) >= 24 else current_pm25
        pm25_2h_ago = df_hist.iloc[-2]['pm2_5'] if len(df_hist) >= 2 else current_pm25
        features['diff_1h'] = current_pm25 - pm25_1h_ago
        features['diff_24h'] = current_pm25 - pm25_24h_ago
        features['diff_1h_acc'] = (current_pm25 - pm25_1h_ago) - (pm25_1h_ago - pm25_2h_ago)
    else:
        features['diff_1h'] = 0.0
        features['diff_24h'] = 0.0
        features['diff_1h_acc'] = 0.0
    
    features['recent_hour_avg'] = np.mean([features['pm2_5_lag24'], features['pm2_5_lag48'], features['pm2_5_lag72']])
    
    # Rolling statistics
    if len(df_hist) >= 24:
        rolling_window = df_hist.iloc[-24:]['pm2_5']
        features['pm2_5_rolling_mean_24h'] = rolling_window.mean()
        features['pm2_5_rolling_std_24h'] = rolling_window.std() if len(rolling_window) > 1 else 0.0
    else:
        features['pm2_5_rolling_mean_24h'] = current_pm25
        features['pm2_5_rolling_std_24h'] = 0.0
    
    # Time features
    hour_val = ts.hour
    month_val = ts.month
    day_val = ts.day
    
    features['hour_sin'] = np.sin(2 * np.pi * hour_val / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour_val / 24)
    features['hour_raw'] = hour_val
    features['month_sin'] = np.sin(2 * np.pi * month_val / 12)
    features['month_cos'] = np.cos(2 * np.pi * month_val / 12)
    features['month'] = month_val
    features['day'] = day_val
    features['is_weekend'] = 1 if ts.dayofweek >= 5 else 0
    
    if 'aqi_change_rate' in current_row:
        features['aqi_change_rate'] = current_row['aqi_change_rate']
    elif len(df_hist) >= 2:
        prev_pm25 = df_hist.iloc[-1]['pm2_5']
        if prev_pm25 > 0:
            features['aqi_change_rate'] = (current_pm25 - prev_pm25) / prev_pm25
        else:
            features['aqi_change_rate'] = 0.0
    else:
        features['aqi_change_rate'] = 0.0
    
    base_features = [
        features['pm2_5'], features['pm10'], features['o3'], features['no2'], features['so2'], features['co'],
        features['pm2_5_lag1'], features['pm2_5_lag2'], features['pm2_5_lag3'],
        features['pm2_5_lag24'], features['pm2_5_lag48'], features['pm2_5_lag72'],
        features['diff_1h'], features['diff_24h'], features['diff_1h_acc'],
        features['recent_hour_avg'],
        features['pm2_5_rolling_mean_24h'], features['pm2_5_rolling_std_24h'],
        features['hour_sin'], features['hour_cos'], features['hour_raw'],
        features['month_sin'], features['month_cos'], features['month'],
        features['day'], features['is_weekend']
    ]
    
    return base_features, features


def recursive_forecast(model, scaler, initial_features_dict, df_historical, steps=72):
    """
    Perform recursive forecasting: predict T+1, then use that to predict T+2, etc.
    
    Args:
        model: Trained model (T+1 predictor)
        scaler: Feature scaler
        initial_features_dict: Dictionary with initial feature values
        df_historical: Historical data for computing lags/rolling stats
        steps: Number of steps ahead to forecast (default 72 for 3 days)
    
    Returns:
        List of predictions for each step
    """
    predictions = []
    current_row = initial_features_dict.copy()
    hist_context = df_historical.copy() if len(df_historical) > 0 else pd.DataFrame()
    
    # Get initial timestamp
    if 'timestamp' in current_row:
        current_ts = pd.to_datetime(current_row['timestamp'])
    else:
        current_ts = pd.Timestamp.now()
    
    # Store prediction history for recursive updates
    prediction_history = []
    
    for step in range(steps):
        # Calculate future timestamp for this step
        future_ts = current_ts + timedelta(hours=step + 1)
        
        # Construct features for this step
        base_features_list, features_dict = construct_features_for_recursive(
            hist_context, current_row, future_ts=future_ts
        )
        
        # Adjust feature count to match model expectations
        expected_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else len(base_features_list)
        
        if expected_features > 20:
            # New model with advanced features
            features = base_features_list.copy()
            if len(features) < expected_features:
                features.append(features_dict.get('aqi_change_rate', 0.0))
        elif expected_features == 10:
            # Old model with aqi_change_rate
            features = [
                current_row['pm2_5'], current_row['pm10'], current_row['o3'],
                current_row['no2'], current_row['so2'], current_row['co'],
                future_ts.hour, future_ts.day, future_ts.month,
                current_row.get('aqi_change_rate', 0.0)
            ]
        else:
            # Old model without aqi_change_rate
            features = [
                current_row['pm2_5'], current_row['pm10'], current_row['o3'],
                current_row['no2'], current_row['so2'], current_row['co'],
                future_ts.hour, future_ts.day, future_ts.month
            ]
        
        # Ensure correct feature count
        if len(features) != expected_features:
            if len(features) < expected_features:
                features.extend([0.0] * (expected_features - len(features)))
            else:
                features = features[:expected_features]
        
        # Make prediction
        input_data = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        pred_result = model.predict(input_scaled)
        
        if hasattr(pred_result, 'flatten'):
            prediction = pred_result.flatten()[0]
        else:
            prediction = pred_result[0] if isinstance(pred_result, (list, np.ndarray)) else pred_result
        
        predictions.append(prediction)
        prediction_history.append(prediction)
        
        # Update current_row for next iteration (recursive)
        # Use the prediction as the new PM2.5 value
        pm25_ratio = prediction / current_row['pm2_5'] if current_row['pm2_5'] > 0 else 1.0
        
        current_row['pm2_5'] = prediction
        current_row['pm10'] = current_row.get('pm10', 0.0) * pm25_ratio
        current_row['o3'] = current_row.get('o3', 0.0) * 0.995  # O3 decays slightly
        current_row['no2'] = current_row.get('no2', 0.0) * pm25_ratio
        current_row['so2'] = current_row.get('so2', 0.0) * pm25_ratio
        current_row['co'] = current_row.get('co', 0.0) * pm25_ratio
        
        # Update timestamp
        current_row['timestamp'] = future_ts
        
        # Update historical context with prediction (for next step's lags)
        if len(hist_context) > 0:
            # Add prediction to historical context
            new_row = hist_context.iloc[-1].copy()
            new_row['pm2_5'] = prediction
            new_row['timestamp'] = future_ts
            hist_context = pd.concat([hist_context, pd.DataFrame([new_row])], ignore_index=True)
            # Keep only last 72 hours for efficiency
            if len(hist_context) > 72:
                hist_context = hist_context.iloc[-72:].reset_index(drop=True)
    
    return predictions



if __name__ == "__main__":
    import sys
    
    # Allow strategy selection via command line argument
    strategy = 'recursive'  # Default to recursive
    if len(sys.argv) > 1:
        strategy = sys.argv[1].lower()
        if strategy not in ['direct', 'recursive']:
            print(f"⚠️  Unknown strategy '{strategy}', using 'recursive'")
            strategy = 'recursive'
    
    df, project = load_data()
    
    if df is not None:
        X, y, scaler, df_original = prepare_data(df, strategy=strategy)
        
        # Split Data (preserve indices for time-series models)
        train_indices = X.index[:int(len(X) * 0.8)]
        test_indices = X.index[int(len(X) * 0.8):]
        
        X_train, X_test = X.loc[train_indices], X.loc[test_indices]
        y_train, y_test = y.loc[train_indices], y.loc[test_indices]
        
        # Prepare time-series data for SARIMA and Prophet
        ts_train_indexed, ts_test_indexed, ts_train_df, ts_test_df = prepare_time_series_data(
            df_original, train_indices, test_indices
        )
        
        # Train
        xgb_m, rf_m, svm_m, mlp_m, sarima_m, prophet_m = train_all_models(
            X_train, y_train, X_test, y_test,
            ts_train=ts_train_indexed,
            ts_test=ts_test_indexed,
            ts_train_df=ts_train_df
        )
        
        # For recursive strategy: need to get actual T+72 values for comparison
        if strategy == 'recursive':
            # Get actual T+72 values from original dataframe
            df_test_actual = df_original.loc[test_indices].copy()
            df_test_actual = df_test_actual.sort_values('timestamp').reset_index(drop=True)
            # Get T+72 actual values (shift forward 72 hours)
            actual_t72 = []
            for idx in df_test_actual.index:
                # Find the row 72 hours ahead
                current_ts = pd.to_datetime(df_test_actual.loc[idx, 'timestamp'])
                future_ts = current_ts + timedelta(hours=72)
                # Find matching row in original dataframe
                future_rows = df_original[df_original['timestamp'] == future_ts]
                if len(future_rows) > 0:
                    actual_t72.append(future_rows.iloc[0]['pm2_5'])
                else:
                    # If not found, use last available value
                    actual_t72.append(df_test_actual.loc[idx, 'pm2_5'])
            y_test_t72 = pd.Series(actual_t72[:len(y_test)], index=y_test.index)
        else:
            y_test_t72 = y_test  # For direct strategy, use y_test as is
        
        # Evaluate all models with RMSE, MAE, and R-squared
        def evaluate_model(m, is_keras=False, is_timeseries=False, model_type=None, strategy=strategy):
            """Evaluate model. Handles ML models, time-series models, and limits recursive evaluation size."""
      
            # --- ⚡ FAST EVALUATION LIMIT ⚡ ---
            # Only evaluate on the last 100 test samples (approx 4 days)
            # This prevents the 10-hour wait time
            MAX_EVAL_SAMPLES = 100 
            # -----------------------------------

            if strategy == 'recursive' and not is_timeseries:
                # For recursive strategy: use recursive_forecast to get T+72 predictions
                pred_t72_list = []
                df_test_context = df_original.loc[test_indices].copy()
                df_test_context = df_test_context.sort_values('timestamp').reset_index(drop=True)
          
                # Get training data for historical context
                df_train_context = df_original.loc[train_indices].copy()
                df_train_context = df_train_context.sort_values('timestamp').reset_index(drop=True)
          
                # Determine start index to only process the last N samples
                total_samples = min(len(df_test_context), len(y_test_t72))
          
                # This logic skips the first ~13,900 rows and only does the last 100
                start_idx = max(0, total_samples - MAX_EVAL_SAMPLES)
          
                print(f"   ⚡ Fast Eval: Processing last {total_samples - start_idx} samples (skipping first {start_idx})...")

                for idx in range(start_idx, total_samples):
                    current_row = df_test_context.iloc[idx].to_dict()
                    # Use training data + test data up to current point as historical context
                    hist_context = pd.concat([df_train_context, df_test_context.iloc[:idx+1]], ignore_index=True)
                    hist_context = hist_context.sort_values('timestamp').reset_index(drop=True)
              
                    # Get T+72 prediction using recursive forecasting
                    # Get T+72 prediction using recursive forecasting
                    try:
                        pred_72h = recursive_forecast(m, scaler, current_row, hist_context, steps=72)
                        # Take the 72nd prediction (T+72)
                        pred_t72_list.append(pred_72h[-1] if len(pred_72h) > 0 else current_row['pm2_5'])
                    except Exception as e:
                        print(f"   ⚠️ Recursive forecast error at index {idx}: {e}")
                        pred_t72_list.append(current_row['pm2_5'])

                pred = np.array(pred_t72_list)
                # Compare against the corresponding LAST samples of y_test_t72
                y_compare = y_test_t72.values[start_idx:total_samples]

            elif is_timeseries:
                # Time-series models (SARIMA/Prophet)
                if model_type == "sarima":
                    try:
                        forecast = m.forecast(steps=len(y_test))
                        pred = forecast.values if hasattr(forecast, 'values') else np.array(forecast)
                        pred = pred[:len(y_test)]
                    except:
                        pred = np.full(len(y_test), ts_train_indexed.iloc[-1])
          
                elif model_type == "prophet":
                    try:
                        future = m.make_future_dataframe(periods=len(y_test), freq='H')
                        forecast = m.predict(future)
                        train_len = len(ts_train_df)
                        pred = forecast['yhat'].iloc[train_len:train_len + len(y_test)].values
                    except:
                        pred = np.full(len(y_test), ts_train_df['pm2_5'].iloc[-1])
                else:
                    pred = np.full(len(y_test), y_test.mean())
              
                y_compare = y_test.values

            else:
                # Direct Strategy (Standard ML) - Fast enough to do all
                pred = m.predict(X_test).flatten() if is_keras else m.predict(X_test)
                y_compare = y_test.values

            # Ensure shapes match
            min_len = min(len(pred), len(y_compare))
            pred = pred[:min_len]
            y_compare = y_compare[:min_len]
      
            mae = mean_absolute_error(y_compare, pred)
            mse = mean_squared_error(y_compare, pred)
            rmse = sqrt(mse)
            r2 = r2_score(y_compare, pred)
      
            return {"mae": mae, "rmse": rmse, "r2": r2}

        print("\n📊 Evaluating all models on test set...")
        
        # 1. Setup the results dictionary with the trained models
        results = {
            "XGBoost": {"m": xgb_m, "k": False, "ts": False},
            "RandomForest": {"m": rf_m, "k": False, "ts": False},
            "SVM": {"m": svm_m, "k": False, "ts": False},
            "NeuralNet": {"m": mlp_m, "k": True, "ts": False}
        }
        
        # Add time-series models if they exist
        if sarima_m is not None:
            results["SARIMA"] = {"m": sarima_m, "k": False, "ts": True, "ts_type": "sarima"}
        if prophet_m is not None:
            results["Prophet"] = {"m": prophet_m, "k": False, "ts": True, "ts_type": "prophet"}
        
        # 2. Run the Evaluation Loop
        for name in results:
            if results[name]["m"] is None:
                results[name].update({"mae": float('inf'), "rmse": float('inf'), "r2": -float('inf')})
                continue
            
            # This is where we actually CALL the function you wrote
            is_ts = results[name].get("ts", False)
            ts_type = results[name].get("ts_type", None)
            
            # Call evaluate_model
            metrics = evaluate_model(
                results[name]["m"],
                results[name]["k"],
                is_timeseries=is_ts,
                model_type=ts_type,
                strategy=strategy
            )
            
            # Save scores
            results[name].update(metrics)
        
        # 3. Print the Results Table
        print("\n" + "="*70)
        print("MODEL EVALUATION RESULTS")
        print("="*70)
        print(f"{'Model':<15} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
        print("-"*70)
        for name in results:
            if results[name]["m"] is not None:
                print(f"{name:<15} {results[name]['rmse']:<12.4f} {results[name]['mae']:<12.4f} {results[name]['r2']:<12.4f}")
            else:
                print(f"{name:<15} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
        print("="*70)
        
        # 4. Pick the Winner
        valid_results = {name: res for name, res in results.items() if res["m"] is not None}
        
        if not valid_results:
            print("❌ No valid models to compare!")
        else:
            # Sort by RMSE (Lower is better)
            best = min(valid_results, key=lambda k: valid_results[k]["rmse"])
            
            print(f"\n🏆 WINNER: {best}")
            print(f"   RMSE: {valid_results[best]['rmse']:.4f}")
            print(f"   MAE:  {valid_results[best]['mae']:.4f}")
            print(f"   R²:   {valid_results[best]['r2']:.4f}")
            
            # 5. Save/Register the Winner
            best_model = valid_results[best]["m"]
            
            # Create a simple description
            params_desc = ""
            if hasattr(best_model, 'best_params_'):
                params_desc = f" | Best Params: {best_model.best_params_}"
            
            # Create a dummy sample input for the registry
            sample_input = X_train.sample(1) if not valid_results[best].get("ts", False) else pd.DataFrame([[0]*len(X_train.columns)], columns=X_train.columns)
            
            register_winner(
                project, 
                best_model, 
                best, 
                {
                    "rmse": valid_results[best]["rmse"],
                    "mae": valid_results[best]["mae"],
                    "r2": valid_results[best]["r2"]
                }, 
                sample_input, 
                scaler,
                description_suffix=params_desc
            )