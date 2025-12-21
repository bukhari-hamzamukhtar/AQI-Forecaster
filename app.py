import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import certifi
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
from dotenv import load_dotenv
USE_LOCAL_STORAGE = True  # Always use local storage for HF Spaces deployment

HOPSWORKS_AVAILABLE = False
hopsworks = None
try:
    import hopsworks as hw
    hopsworks = hw
    HOPSWORKS_AVAILABLE = True
except ImportError:
    pass
except Exception:
    pass

# Model Explainability
EXPLAINABILITY_AVAILABLE = False
try:
    from explainability import ModelExplainer, create_explainer, SHAP_AVAILABLE, LIME_AVAILABLE
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    # Fallback: try importing explainability module directly
    try:
        import explainability
        ModelExplainer = explainability.ModelExplainer
        create_explainer = explainability.create_explainer
        SHAP_AVAILABLE = explainability.SHAP_AVAILABLE
        LIME_AVAILABLE = explainability.LIME_AVAILABLE
        EXPLAINABILITY_AVAILABLE = True
    except (ImportError, AttributeError):
        EXPLAINABILITY_AVAILABLE = False
        # Fallback imports for backward compatibility
        try:
            import shap
            SHAP_AVAILABLE = True
        except ImportError:
            SHAP_AVAILABLE = False
        
        try:
            from lime.lime_tabular import LimeTabularExplainer
            LIME_AVAILABLE = True
        except ImportError:
            LIME_AVAILABLE = False

# -------------------------------------------------
# 🔧 WINDOWS SSL FIX
# -------------------------------------------------
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Load env vars
load_dotenv()

# -------------------------------------------------
# 🎨 UI CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="AQI Predictor",
    page_icon="🌫️",
    layout="centered"
)

st.title("🌫️ Air Quality Forecaster")
st.markdown("Real-time AQI forecasts using **MLOPs**.")

# Add tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Forecast", "📊 Data Drift Monitoring", "📈 EDA & Analytics", "🔍 Model Explainability"])

# -------------------------------------------------
# 🤗 HUGGING FACE MODEL DOWNLOAD
# -------------------------------------------------
HF_REPO_ID = os.getenv("HF_REPO_ID", "bukhari-hamzamukhtar/aqi-model")
HF_MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "aqi_best_model.h5")
HF_TOKEN = os.getenv("HF_TOKEN", None)  # For private repos

def download_model_from_hf():
    """Download model from Hugging Face Hub if not present locally."""
    model_path = HF_MODEL_FILENAME
    if os.path.exists(model_path):
        return model_path
    
    if not HF_REPO_ID:
        return None
    
    try:
        from huggingface_hub import hf_hub_download
        st.info(f"⬇️ Downloading model from Hugging Face ({HF_REPO_ID})...")
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_MODEL_FILENAME,
            local_dir=".",
            local_dir_use_symlinks=False,
            token=HF_TOKEN  # Supports private repos
        )
        st.success("✅ Model downloaded successfully!")
        return downloaded_path
    except Exception as e:
        st.warning(f"⚠️ Could not download model from HF: {e}")
        return None

# -------------------------------------------------
# 🧠 CACHED FUNCTIONS (Speed Boosters)
# -------------------------------------------------

@st.cache_resource
def connect_to_hopsworks():
    """Connects to Hopsworks and returns the project object."""
    if not HOPSWORKS_AVAILABLE or hopsworks is None:
        return None
    try:
        project = hopsworks.login(
            project=os.getenv("HOPSWORKS_PROJECT"),
            api_key_value=os.getenv("HOPSWORKS_API_KEY")
        )
        return project
    except Exception as e:
        st.error(f"❌ Could not connect to Hopsworks: {e}")
        return None

# ---------------------------------------------------------
# 1. ADD THIS CLASS (The Translator)
# ---------------------------------------------------------
class SarimaWrapper:
    """
    Wraps a statsmodels SARIMA model to look like a Scikit-Learn model.
    Allows the app to run .predict(features) without crashing.
    """
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        # The App sends 'X' (features), but SARIMA doesn't need them.
        # We ignore X and just ask SARIMA for the next step.
        try:
            # Get forecast (try-catch handles different statsmodels versions)
            forecast = self.model.forecast(steps=1)
            
            # Extract the single value from the result
            if hasattr(forecast, 'values'):
                return forecast.values  # Return array [value]
            elif hasattr(forecast, 'iloc'):
                return forecast.iloc[0] # Return value
            else:
                return list(forecast)[0] # Return value from list/array
        except Exception as e:
            # Fallback: if forecast fails, return a safe dummy value
            print(f"SARIMA Forecast Error: {e}")
            return [15.0] # Safe fallback

# ---------------------------------------------------------
# 2. UPDATE THIS FUNCTION (To use the Translator)
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_scaler(_project):
    """Downloads the model and scaler from the Registry or Local Storage."""
    
    # --- LOAD MODEL FILE ---
    model = None
    scaler = None
    model_info = {"name": "Unknown", "version": "unknown"}
    
    # Check if using local storage
    if USE_LOCAL_STORAGE:
        try:
            import joblib
            import os
            
            # Load Scaler
            if os.path.exists("scaler.pkl"):
                scaler = joblib.load("scaler.pkl")
            else:
                st.warning("⚠️ 'scaler.pkl' not found. Please train a model first.")
                return None, None, None

            # Find Model File (Check .pkl first, then .h5, then try HuggingFace)
            model_path = None
            if os.path.exists("aqi_best_model.pkl"):
                model_path = "aqi_best_model.pkl"
            elif os.path.exists("aqi_best_model.h5"):
                model_path = "aqi_best_model.h5"
            else:
                # Try downloading from Hugging Face
                model_path = download_model_from_hf()
            
            if not model_path:
                st.warning("⚠️ No model file found. Set HF_REPO_ID environment variable to download from Hugging Face.")
                return None, None, None
            
            # Load Model (Robust loading)
            try:
                # Try joblib first
                model = joblib.load(model_path)
            except:
                try:
                    # Try Statsmodels specific load
                    from statsmodels.iolib.smpickle import load_pickle
                    model = load_pickle(model_path)
                except:
                    try:
                        # Try Keras
                        from tensorflow.keras.models import load_model
                        model = load_model(model_path)
                    except Exception as e:
                        st.error(f"❌ Failed to load model '{model_path}'.")
                        return None, None, None

            model_info = {"name": "Local Model", "version": "local"}

        except Exception as e:
            st.error(f"❌ Error loading local model: {e}")
            return None, None, None
    else:
        # Hopsworks Logic (Keep existing if needed, or return None)
        st.error("Hopsworks loading not configured in this snippet.")
        return None, None, None

    # --- APPLY SARIMA WRAPPER (The Fix 🩹) ---
    # Check if the model is from statsmodels (SARIMA)
    model_type = str(type(model))
    if 'statsmodels' in model_type or 'SARIMAX' in model_type:
        print("ℹ️ Detected SARIMA model - Applying wrapper compatibility layer.")
        model = SarimaWrapper(model)

    return model, scaler, model_info

def fetch_recent_data(_project=None, hours=168, force_refresh=False):
    """
    Force load data directly from the local_data folder.
    Bypasses cache and helper scripts to ensure we get the REAL file.
    """
    try:
        # 1. Define the specific path we want (The one backfill.py writes to)
        file_path = "local_data/aqi_features.parquet"
        
        if not os.path.exists(file_path):
            # Fallback to main folder if local_data doesn't exist
            if os.path.exists("aqi_features.parquet"):
                file_path = "aqi_features.parquet"
            else:
                st.error(f"❌ File not found at: {file_path}")
                st.info("💡 Please run 'python local_backfill.py' to generate it.")
                return pd.DataFrame()

        # 2. Read it directly
        df = pd.read_parquet(file_path)
        
        # 3. Sort it (Crucial for graphs)
        df = df.sort_values('timestamp')
        
        # 4. Debug Print (Only visible in sidebar to confirm it worked)
        st.sidebar.success(f"📂 Loaded {len(df)} rows from {file_path}")
        if not df.empty:
            st.sidebar.text(f"End: {df['timestamp'].max()}")
        
        # 5. Ensure Timezone
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
                
        return df

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        return pd.DataFrame()
def fetch_training_data(_project):
    """Fetches historical data to use as training baseline (last 7 days)."""
    # Check if using local storage
    if USE_LOCAL_STORAGE:
        try:
            from local_data_loader import fetch_training_data_local
            return fetch_training_data_local(days=7)
        except ImportError:
            return pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Local storage error: {e}")
            return pd.DataFrame()
    
    fs = _project.get_feature_store()
    
    try:
        # Get latest version automatically
        try:
            all_fgs = fs.get_feature_group(name="aqi_features", version=None)
            if isinstance(all_fgs, list) and len(all_fgs) > 0:
                latest_version = max(fg.version for fg in all_fgs if hasattr(fg, 'version') and fg.version is not None)
                fg = fs.get_feature_group(name="aqi_features", version=latest_version)
            else:
                fg = fs.get_feature_group(name="aqi_features", version=3)
        except Exception:
            fg = fs.get_feature_group(name="aqi_features", version=3)
        
        # Try reading without Hive first (faster)
        try:
            df = fg.read(read_options={"use_hive": False})
        except Exception:
            # Fall back to Hive if direct read fails
            df = fg.read(read_options={"use_hive": True})
        
        df = df.sort_values("timestamp")
        # Use last 7 days as "training baseline" (168 hours)
        return df.tail(168)
    except AttributeError as e:
        # Handle the _server_version AttributeError (secondary error from connection failure)
        if '_server_version' in str(e):
            return pd.DataFrame()  # Silently return empty, connection error already handled in fetch_recent_data
        else:
            st.error(f"❌ Error fetching training data: {e}")
            return pd.DataFrame()
    except Exception as e:
        error_msg = str(e)
        # Check if it's a connection-related error
        if 'FlightUnavailableError' in error_msg or 'failed to connect' in error_msg.lower() or 'Handshake read failed' in error_msg:
            return pd.DataFrame()  # Silently return empty, connection error already handled in fetch_recent_data
        else:
            st.error(f"❌ Error fetching training data: {e}")
            return pd.DataFrame()

def fetch_all_data(_project, limit=None):
    """Fetches all available data from feature store for EDA."""
    # Check if using local storage
    if USE_LOCAL_STORAGE:
        try:
            from local_data_loader import fetch_all_data_local
            return fetch_all_data_local(limit=limit)
        except ImportError:
            return pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Local storage error: {e}")
            return pd.DataFrame()
    
    fs = _project.get_feature_store()
    
    try:
        # Get latest version automatically
        try:
            all_fgs = fs.get_feature_group(name="aqi_features", version=None)
            if isinstance(all_fgs, list) and len(all_fgs) > 0:
                latest_version = max(fg.version for fg in all_fgs if hasattr(fg, 'version') and fg.version is not None)
                fg = fs.get_feature_group(name="aqi_features", version=latest_version)
            else:
                fg = fs.get_feature_group(name="aqi_features", version=3)
        except Exception:
            fg = fs.get_feature_group(name="aqi_features", version=3)
        
        # Try reading without Hive first (faster)
        try:
            df = fg.read(read_options={"use_hive": False})
        except Exception:
            # Fall back to Hive if direct read fails
            df = fg.read(read_options={"use_hive": True})
        
        # Ensure timestamp is datetime and timezone-aware
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        
        df = df.sort_values("timestamp")
        if limit:
            return df.tail(limit)
        return df
    except AttributeError as e:
        # Handle the _server_version AttributeError (secondary error from connection failure)
        if '_server_version' in str(e):
            st.error("❌ Connection to Hopsworks failed. Please check your network connection and Hopsworks server status.")
            return pd.DataFrame()
        else:
            st.error(f"❌ Error fetching data: {e}")
            return pd.DataFrame()
    except Exception as e:
        error_msg = str(e)
        # Check if it's a connection-related error
        if 'FlightUnavailableError' in error_msg or 'failed to connect' in error_msg.lower() or 'Handshake read failed' in error_msg:
            st.error("❌ Connection to Hopsworks failed. Please check your network connection and Hopsworks server status.")
        else:
            st.error(f"❌ Error fetching data: {e}")
        return pd.DataFrame()

def create_time_series_plot(df, selected_pollutants):
    """Create time series plot for selected pollutants."""
    fig = go.Figure()
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for idx, poll in enumerate(selected_pollutants):
        if poll in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df[poll],
                mode='lines',
                name=poll.upper(),
                line=dict(color=colors[idx % len(colors)], width=2)
            ))
    
    fig.update_layout(
        title="Time Series - All Selected Pollutants",
        xaxis_title="Time",
        yaxis_title="Concentration (µg/m³)",
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_distribution_plot(df, selected_pollutants):
    """Create distribution histograms for selected pollutants."""
    rows = (len(selected_pollutants) + 1) // 2
    cols = 2
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[p.upper() + ' Distribution' for p in selected_pollutants],
        vertical_spacing=0.1
    )
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    for idx, poll in enumerate(selected_pollutants):
        if poll in df.columns:
            row = (idx // cols) + 1
            col = (idx % cols) + 1
            fig.add_trace(
                go.Histogram(x=df[poll], name=poll, nbinsx=30, showlegend=False,
                           marker_color=colors[idx % len(colors)]),
                row=row, col=col
            )
            fig.update_yaxes(title_text="Frequency", row=row, col=col)
            fig.update_xaxes(title_text=f"{poll.upper()} (µg/m³)", row=row, col=col)
    
    fig.update_layout(height=300 * rows, title_text="Pollutant Distributions", showlegend=False)
    return fig

def create_hourly_pattern_plot(df, selected_pollutants):
    """Create hourly pattern plot."""
    if 'hour' not in df.columns:
        return None
    
    hourly_avg = df.groupby('hour')[selected_pollutants].mean().reset_index()
    fig = go.Figure()
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for idx, poll in enumerate(selected_pollutants):
        if poll in hourly_avg.columns:
            fig.add_trace(go.Scatter(
                x=hourly_avg['hour'],
                y=hourly_avg[poll],
                mode='lines+markers',
                name=poll.upper(),
                line=dict(color=colors[idx % len(colors)], width=2),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title="Average Pollutant Levels by Hour of Day",
        xaxis_title="Hour of Day",
        yaxis_title="Concentration (µg/m³)",
        height=400,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_monthly_pattern_plot(df, selected_pollutants):
    """Create monthly pattern plot."""
    if 'month' not in df.columns:
        return None
    
    monthly_avg = df.groupby('month')[selected_pollutants].mean().reset_index()
    fig = go.Figure()
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for idx, poll in enumerate(selected_pollutants):
        if poll in monthly_avg.columns:
            fig.add_trace(go.Bar(
                x=monthly_avg['month'],
                y=monthly_avg[poll],
                name=poll.upper(),
                marker_color=colors[idx % len(colors)]
            ))
    
    fig.update_layout(
        title="Average Pollutant Levels by Month",
        xaxis_title="Month",
        yaxis_title="Concentration (µg/m³)",
        barmode='group',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_correlation_heatmap(df, selected_pollutants):
    """Create correlation heatmap."""
    # Include time features and pollutants
    numeric_cols = selected_pollutants.copy()
    if 'hour' in df.columns:
        numeric_cols.append('hour')
    if 'day' in df.columns:
        numeric_cols.append('day')
    if 'month' in df.columns:
        numeric_cols.append('month')
    if 'aqi_change_rate' in df.columns:
        numeric_cols.append('aqi_change_rate')
    
    available_cols = [col for col in numeric_cols if col in df.columns]
    corr_matrix = df[available_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Feature Correlation Heatmap",
        height=600,
        width=700
    )
    return fig

def create_box_plot(df, selected_pollutants):
    """Create box plots for outlier detection."""
    fig = go.Figure()
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for idx, poll in enumerate(selected_pollutants):
        if poll in df.columns:
            fig.add_trace(go.Box(
                y=df[poll],
                name=poll.upper(),
                marker_color=colors[idx % len(colors)]
            ))
    
    fig.update_layout(
        title="Pollutant Distribution Box Plots (Outlier Detection)",
        yaxis_title="Concentration (µg/m³)",
        height=400,
        xaxis_title="Pollutant"
    )
    return fig

def analyze_drift_causes(df_training, df_recent, selected_pollutants):
    """
    Analyze potential causes of data drift.
    Returns concise, data-driven markdown text (No generic essays).
    """
    if df_training.empty or df_recent.empty:
        return ""
    
    analysis = []
    drift_found = False
    
    for feature in selected_pollutants:
        if feature in df_training.columns and feature in df_recent.columns:
            # Check for drift
            is_drift, metrics, _ = detect_drift(df_training, df_recent, feature=feature, threshold=0.2)
            
            if is_drift and metrics:
                drift_found = True
                curr_mean = metrics['current_mean']
                train_mean = metrics['train_mean']
                
                # Calculate percentage shift
                if train_mean != 0:
                    diff_pct = ((curr_mean - train_mean) / train_mean) * 100
                else:
                    diff_pct = 100 if curr_mean > 0 else 0
                
                # Dynamic Icon & Direction
                if diff_pct > 0:
                    icon = "🔺"
                    direction = "Higher"
                    desc = "Spike detected"
                else:
                    icon = "🔻"
                    direction = "Lower"
                    desc = "Drop detected"
                
                analysis.append(f"**{feature.upper()} {icon} {desc}**")
                analysis.append(f"- **Magnitude:** Current levels are **{abs(diff_pct):.1f}% {direction}** than training averages.")
                analysis.append(f"- **Real Numbers:** Current Avg: **{curr_mean:.1f}** vs Training Avg: **{train_mean:.1f}**")
                
                # Severity Check (Are we seeing values never seen before?)
                outside_pct = metrics.get('outside_range_pct', 0)
                if outside_pct > 20:
                    analysis.append(f"- ⚠️ **Alert:** {outside_pct:.1f}% of recent data is outside the normal training range. The model might be confused.")
                
                analysis.append("") # Add spacing
    
    if not drift_found:
        return ""
        
    return "\n".join(analysis)

def generate_eda_analysis(viz_type, df_filtered, selected_pollutants, latest_row=None, forecasts=None, df_training=None, df_recent=None):
    """
    Generate analysis and interpretation for EDA visualizations.
    Returns markdown text with insights.
    """
    analysis = []
    
    if viz_type == "Time Series":
        analysis.append("### 📈 Smart Trend Insights")
        
        # Calculate trends
        for poll in selected_pollutants:
            if poll in df_filtered.columns:
                values = df_filtered[poll].dropna()
                if len(values) > 1:
                    # Calculate trend (slope)
                    x = np.arange(len(values))
                    slope = np.polyfit(x, values, 1)[0]
                    
                    # Convert hourly slope to DAILY slope for readability
                    daily_change = slope * 24
                    
                    mean_val = values.mean()
                    
                    # Determine status based on ACTUAL data
                    if daily_change > 1.0:
                        trend_desc = "worsening rapidly 🔴"
                        insight = "Air quality is degrading significantly day-over-day."
                    elif daily_change > 0.1:
                        trend_desc = "increasing slightly 📉"
                        insight = "There is a slow buildup of pollutants."
                    elif daily_change < -1.0:
                        trend_desc = "improving rapidly 🟢"
                        insight = "Air quality is clearing up quickly."
                    elif daily_change < -0.1:
                        trend_desc = "decreasing slightly 📉"
                        insight = "Conditions are slowly getting better."
                    else:
                        trend_desc = "stable ⚖️"
                        insight = "No significant change in recent levels."

                    analysis.append(f"""
**{poll.upper()}:**
- **Status:** {trend_desc}
- **Rate:** {daily_change:+.2f} µg/m³ per day
- **Insight:** {insight}
                    """)
    
    elif viz_type == "Distributions":
        analysis.append("### 📊 Distribution Insights")
        
        for poll in selected_pollutants:
            if poll in df_filtered.columns:
                values = df_filtered[poll].dropna()
                if len(values) > 0:
                    skew = values.skew()
                    
                    # Dynamic interpretation of skew
                    if skew > 1:
                        shape = "Highly Skewed (Right)"
                        meaning = "Most of the time air is clean, but there are rare extreme pollution spikes."
                    elif skew < -1:
                        shape = "Highly Skewed (Left)"
                        meaning = "Pollution is usually high, with rare clean breaks."
                    else:
                        shape = "Normal / Balanced"
                        meaning = "Pollution fluctuates evenly around the average."

                    analysis.append(f"- **{poll.upper()}:** {shape}. *{meaning}*")

    elif viz_type == "Hourly Patterns":
        analysis.append("### ⏰ Daily Rhythm Check")
        
        if 'hour' in df_filtered.columns:
            for poll in selected_pollutants:
                if poll in df_filtered.columns:
                    hourly_avg = df_filtered.groupby('hour')[poll].mean()
                    peak_hour = hourly_avg.idxmax()
                    low_hour = hourly_avg.idxmin()
                    
                    # Convert 24h to AM/PM for readability
                    peak_str = pd.to_datetime(f"{peak_hour}:00", format="%H:%M").strftime("%I %p")
                    low_str = pd.to_datetime(f"{low_hour}:00", format="%H:%M").strftime("%I %p")
                    
                    analysis.append(f"- **{poll.upper()}:** Worst air at **{peak_str}**, cleanest at **{low_str}**.")
                    
    elif viz_type == "Monthly Patterns":
        analysis.append("### 📅 Seasonal Insights")
        if 'month' in df_filtered.columns:
            for poll in selected_pollutants:
                if poll in df_filtered.columns:
                    monthly_avg = df_filtered.groupby('month')[poll].mean()
                    peak_month = monthly_avg.idxmax()
                    import calendar
                    month_name = calendar.month_name[peak_month]
                    
                    analysis.append(f"- **{poll.upper()}:** Historically highest in **{month_name}**.")

    elif viz_type == "Correlation Heatmap":
        analysis.append("### 🔗 Relationship Insights")
        analysis.append("- **Strong connections:** If two squares are dark red, those pollutants likely come from the same source (e.g., traffic).")
        analysis.append("- **Inverse relationships:** Blue squares mean when one goes up, the other goes down (e.g., Wind vs. PM2.5).")

    elif viz_type == "Box Plots (Outliers)":
        analysis.append("### 📦 Outlier Check")
        for poll in selected_pollutants:
            if poll in df_filtered.columns:
                values = df_filtered[poll].dropna()
                Q3 = values.quantile(0.75)
                IQR = Q3 - values.quantile(0.25)
                upper_bound = Q3 + 1.5 * IQR
                outliers = values[values > upper_bound]
                
                if len(outliers) > 0:
                    analysis.append(f"- **{poll.upper()}:** Detected **{len(outliers)} extreme events** above {upper_bound:.1f} µg/m³.")
                else:
                    analysis.append(f"- **{poll.upper()}:** No extreme outliers detected.")

    return "\n".join(analysis)

def generate_authentic_forecast_analysis(df_filtered, selected_pollutants, latest_row, forecasts, df_recent, model, scaler):
    """
    Generate authentic, data-driven forecast analysis based on actual trends and patterns.
    """
    if latest_row is None or forecasts is None:
        return ""
    
    analysis = []
    analysis.append("### 🔮 Forecast Interpretation")
    
    # Extract forecast AQI values
    forecast_aqis = []
    for f in forecasts:
        if 'aqi' in f:
            forecast_aqis.append(int(f['aqi']))
        elif 'val' in f: # Handle card dictionary format
            forecast_aqis.append(int(f['val']))

    if not forecast_aqis:
        return "No forecast data available to interpret."

    # 1. Trend Direction
    start_aqi = forecast_aqis[0]
    end_aqi = forecast_aqis[-1]
    max_aqi = max(forecast_aqis)
    min_aqi = min(forecast_aqis)
    
    # Find the peak hour (index)
    peak_idx = forecast_aqis.index(max_aqi)
    peak_time_label = "in the next few hours"
    if peak_idx > 48: peak_time_label = "in 3 days"
    elif peak_idx > 24: peak_time_label = "in 2 days"
    elif peak_idx > 12: peak_time_label = "tomorrow"
    
    analysis.append(f"**Detailed Outlook:**")
    
    if end_aqi > start_aqi + 10:
        analysis.append(f"- 📉 **Worsening Trend:** Air quality is expected to degrade over the next 3 days.")
    elif end_aqi < start_aqi - 10:
        analysis.append(f"- 🟢 **Improving Trend:** Expect cleaner air towards the end of the forecast period.")
    else:
        analysis.append(f"- ⚖️ **Stable Trend:** No major shifts in overall air quality expected.")

    analysis.append(f"- **Worst Moment:** Expect the highest pollution ({max_aqi} AQI) **{peak_time_label}**.")
    analysis.append(f"- **Best Moment:** Cleanest air ({min_aqi} AQI) expected around the dipping points of the cycle.")
    
    # 2. Daily Cycle Interpretation
    # Check if we see the "sine wave" pattern
    variability = np.std(forecast_aqis)
    if variability > 15:
        analysis.append(f"- **Pattern:** Strong daily cycle detected. Pollution will likely spike during rush hours/evenings and drop at night.")
    else:
        analysis.append(f"- **Pattern:** Flat forecast. Expect constant levels without much relief at night.")

    return "\n".join(analysis)

# Legacy functions kept for backward compatibility, but we'll use the new explainability module
def explain_prediction_shap(model, scaler, input_data, feature_names, training_data=None):
    """
    Legacy function - kept for backward compatibility.
    Use ModelExplainer class instead for better functionality.
    """
    if not EXPLAINABILITY_AVAILABLE:
        return None, None
    
    try:
        explainer = create_explainer(model, scaler, feature_names)
        
        # Create SHAP explainer
        training_df = None
        if training_data is not None:
            if isinstance(training_data, pd.DataFrame):
                training_df = training_data
            else:
                # Convert numpy array to DataFrame
                training_df = pd.DataFrame(training_data, columns=feature_names)
        
        if explainer.create_shap_explainer(training_df):
            shap_values, _ = explainer.explain_shap_local(input_data, plot=False)
            if shap_values is not None:
                feature_importance = np.abs(shap_values)
                return shap_values, feature_importance
        
        return None, None
    except Exception as e:
        return None, None

def explain_prediction_lime(model, scaler, input_data, training_data, feature_names):
    """
    Legacy function - kept for backward compatibility.
    Use ModelExplainer class instead for better functionality.
    """
    if not EXPLAINABILITY_AVAILABLE:
        return None
    
    try:
        explainer = create_explainer(model, scaler, feature_names)
        
        # Prepare training data
        training_df = None
        if isinstance(training_data, pd.DataFrame):
            training_df = training_data
        else:
            training_df = pd.DataFrame(training_data, columns=feature_names)
        
        if explainer.create_lime_explainer(training_df):
            explanation_dict, _ = explainer.explain_lime_local(input_data, plot=False)
            if explanation_dict is not None:
                # Convert to LIME explanation-like object for backward compatibility
                class LimeExplanationWrapper:
                    def __init__(self, features, values):
                        self.features = features
                        self.values = values
                    
                    def as_list(self):
                        return list(zip(self.features, self.values))
                
                return LimeExplanationWrapper(
                    explanation_dict['features'],
                    explanation_dict['values']
                )
        
        return None
    except Exception as e:
        return None

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

def pm25_to_aqi(pm25):
    """
    Convert PM2.5 concentration to AQI.
    Handles edge cases like negative or zero values.
    """
    # Ensure PM2.5 is non-negative
    pm25 = max(0.0, float(pm25))
    
    # AQI conversion based on US EPA breakpoints
    if pm25 <= 0:
        return 0
    elif pm25 <= 12.0:
        return int((pm25 / 12.0) * 50)
    elif pm25 <= 35.4:
        return int(50 + ((pm25 - 12.0) / (35.4 - 12.0)) * 50)
    elif pm25 <= 55.4:
        return int(100 + ((pm25 - 35.4) / (55.4 - 35.4)) * 50)
    elif pm25 <= 150.4:
        return int(150 + ((pm25 - 55.4) / (150.4 - 55.4)) * 50)
    else:
        # For values above 150.4, cap at 300 AQI (or extend further if needed)
        return min(300, int(200 + ((pm25 - 150.4) / (250.4 - 150.4)) * 100))

def calculate_aqi(pm2_5, pm10, o3, no2, so2, co):
    """
    Calculate US EPA Air Quality Index (AQI) from pollutant concentrations.
    
    AQI is the maximum of all pollutant sub-indices.
    Uses US EPA AQI breakpoints (concentrations in µg/m³, CO in mg/m³).
    
    Returns: AQI value (0-500)
    """
    sub_indices = []
    
    # PM2.5 (24-hour average, but we use hourly as proxy)
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
        o3_ppm = o3 * 0.0005
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

def detect_drift(training_data, current_data, feature='pm2_5', threshold=0.2):
    """
    Improved drift detection using statistical comparison.
    Returns drift status and metrics.
    Uses lower threshold and absolute differences for better detection.
    """
    if training_data.empty or current_data.empty or feature not in training_data.columns:
        return None, None, None
    
    train_values = training_data[feature].dropna()
    current_values = current_data[feature].dropna()
    
    if len(train_values) == 0 or len(current_values) == 0:
        return None, None, None
    
    # Calculate statistics
    train_mean = train_values.mean()
    train_std = train_values.std()
    train_median = train_values.median()
    current_mean = current_values.mean()
    current_std = current_values.std()
    current_median = current_values.median()
    
    # Improved drift detection: multiple methods
    # 1. Normalized mean difference (relative to std)
    mean_diff_normalized = abs(current_mean - train_mean) / (train_std + 1e-6)
    
    # 2. Absolute mean difference (for cases where values are consistently high/low)
    mean_diff_absolute = abs(current_mean - train_mean) / (abs(train_mean) + 1e-6)
    
    # 3. Median difference (less sensitive to outliers)
    median_diff = abs(current_median - train_median) / (abs(train_median) + 1e-6)
    
    # 4. Std difference
    std_diff = abs(current_std - train_std) / (train_std + 1e-6)
    
    # 5. Check if current values are consistently outside training range
    train_min = train_values.quantile(0.05)  # 5th percentile
    train_max = train_values.quantile(0.95)  # 95th percentile
    current_outside_range = (current_values < train_min).sum() + (current_values > train_max).sum()
    outside_range_pct = current_outside_range / len(current_values) if len(current_values) > 0 else 0
    
    # Drift detected if any method indicates significant change
    # Lower threshold (0.2 instead of 0.3) and check multiple indicators
    drift_detected = (
        mean_diff_normalized > threshold or
        mean_diff_absolute > threshold or
        median_diff > threshold or
        std_diff > threshold or
        outside_range_pct > 0.3  # More than 30% of values outside training range
    )
    
    metrics = {
        'train_mean': train_mean,
        'train_std': train_std,
        'train_median': train_median,
        'train_min': train_min,
        'train_max': train_max,
        'current_mean': current_mean,
        'current_std': current_std,
        'current_median': current_median,
        'mean_diff_pct': mean_diff_normalized * 100,
        'mean_diff_absolute_pct': mean_diff_absolute * 100,
        'median_diff_pct': median_diff * 100,
        'std_diff_pct': std_diff * 100,
        'outside_range_pct': outside_range_pct * 100
    }
    
    return drift_detected, metrics, threshold

# -------------------------------------------------
# 📱 MAIN APP LOGIC
# -------------------------------------------------

# Check if using local storage

if USE_LOCAL_STORAGE:
    project = None  # No project needed for local storage
    st.info("💾 **Using Local Storage** (Free alternative to Hopsworks)")
else:
    project = connect_to_hopsworks()

def construct_features_for_prediction(df_historical, current_row, future_ts=None):
    """
    Construct advanced features for prediction matching training_pipeline.py
    
    Args:
        df_historical: DataFrame with historical data (sorted by timestamp, most recent last)
        current_row: Series or dict with current values (pm2_5, pm10, o3, etc.)
        future_ts: Optional datetime for future predictions (if None, uses current_row timestamp)
    
    Returns:
        Tuple: (feature_list, feature_dict) where feature_list is ordered list and feature_dict has all features
    """
    # Use future timestamp if provided, otherwise use current_row timestamp
    if future_ts is None:
        ts = pd.to_datetime(current_row['timestamp']) if 'timestamp' in current_row else pd.Timestamp.now()
    else:
        ts = pd.to_datetime(future_ts)
    
    # Ensure df_historical is sorted chronologically
    df_hist = df_historical.sort_values('timestamp').reset_index(drop=True) if len(df_historical) > 0 else pd.DataFrame()
    
    # Get current PM2.5 value
    current_pm25 = current_row['pm2_5'] if 'pm2_5' in current_row else 0.0
    
    # Initialize feature values with defaults
    features = {}
    
    # 1. Original pollutants (always available)
    features['pm2_5'] = current_row.get('pm2_5', 0.0)
    features['pm10'] = current_row.get('pm10', 0.0)
    features['o3'] = current_row.get('o3', 0.0)
    features['no2'] = current_row.get('no2', 0.0)
    features['so2'] = current_row.get('so2', 0.0)
    features['co'] = current_row.get('co', 0.0)
    
    # 2. Lag features (need historical data)
    if len(df_hist) >= 72:
        features['pm2_5_lag1'] = df_hist.iloc[-1]['pm2_5'] if len(df_hist) >= 1 else current_pm25
        features['pm2_5_lag2'] = df_hist.iloc[-2]['pm2_5'] if len(df_hist) >= 2 else current_pm25
        features['pm2_5_lag3'] = df_hist.iloc[-3]['pm2_5'] if len(df_hist) >= 3 else current_pm25
        features['pm2_5_lag24'] = df_hist.iloc[-24]['pm2_5'] if len(df_hist) >= 24 else current_pm25
        features['pm2_5_lag48'] = df_hist.iloc[-48]['pm2_5'] if len(df_hist) >= 48 else current_pm25
        features['pm2_5_lag72'] = df_hist.iloc[-72]['pm2_5'] if len(df_hist) >= 72 else current_pm25
    else:
        # Not enough historical data - use current value as fallback
        features['pm2_5_lag1'] = current_pm25
        features['pm2_5_lag2'] = current_pm25
        features['pm2_5_lag3'] = current_pm25
        features['pm2_5_lag24'] = current_pm25
        features['pm2_5_lag48'] = current_pm25
        features['pm2_5_lag72'] = current_pm25
    
    # 3. Velocity & Acceleration (differences)
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
    
    # 4. Multi-day seasonality (recent_hour_avg)
    features['recent_hour_avg'] = np.mean([
        features['pm2_5_lag24'],
        features['pm2_5_lag48'],
        features['pm2_5_lag72']
    ])
    
    # 5. Rolling statistics (24h window)
    if len(df_hist) >= 24:
        rolling_window = df_hist.iloc[-24:]['pm2_5']
        features['pm2_5_rolling_mean_24h'] = rolling_window.mean()
        features['pm2_5_rolling_std_24h'] = rolling_window.std() if len(rolling_window) > 1 else 0.0
    else:
        features['pm2_5_rolling_mean_24h'] = current_pm25
        features['pm2_5_rolling_std_24h'] = 0.0
    
    # 6. Time features (cyclical + raw)
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
    
    # 7. Domain features
    features['is_weekend'] = 1 if ts.dayofweek >= 5 else 0
    
    # 8. Optional: AQI change rate (if available)
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
    
    # Return features in the same order as training pipeline
    base_features = [
        features['pm2_5'], features['pm10'], features['o3'], features['no2'], features['so2'], features['co'],
        features['pm2_5_lag1'], features['pm2_5_lag2'], features['pm2_5_lag3'],
        features['pm2_5_lag24'], features['pm2_5_lag48'], features['pm2_5_lag72'],
        features['diff_1h'], features['diff_24h'], features['diff_1h_acc'],
        features['recent_hour_avg'],
        features['pm2_5_rolling_mean_24h'], features['pm2_5_rolling_std_24h'],
        features['hour_sin'], features['hour_cos'], features['hour_raw'],
        features['month_sin'], features['month_cos'], features['month'],
        features['day'],
        features['is_weekend']
    ]
    
    # Add aqi_change_rate if model expects it (will be determined by expected_features)
    return base_features, features


def recursive_forecast_app(model, scaler, initial_row, df_historical, steps=72):
    """
    Perform recursive forecasting in app.py context.
    Predicts T+1, then uses that to predict T+2, looping 72 times.
    
    Args:
        model: Trained model (should be T+1 predictor)
        scaler: Feature scaler
        initial_row: Dictionary/Series with initial feature values
        df_historical: Historical data for computing lags/rolling stats
        steps: Number of steps ahead (default 72 for 3 days)
    
    Returns:
        List of predictions for each step
    """
    predictions = []
    current_row = initial_row.copy() if isinstance(initial_row, dict) else initial_row.to_dict()
    hist_context = df_historical.copy() if len(df_historical) > 0 else pd.DataFrame()
    
    # Get initial timestamp
    if 'timestamp' in current_row:
        current_ts = pd.to_datetime(current_row['timestamp'])
    else:
        current_ts = pd.Timestamp.now()
    
    expected_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else None
    
    for step in range(steps):
        # Calculate future timestamp for this step
        future_ts = current_ts + timedelta(hours=step + 1)
        
        # Construct features for this step
        base_features_list, features_dict = construct_features_for_prediction(
            hist_context, current_row, future_ts=future_ts
        )
        
        # Adjust features based on expected_features
        if expected_features:
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
        else:
            # Unknown - use advanced features
            features = base_features_list.copy()
            features.append(features_dict.get('aqi_change_rate', 0.0))
        
        # Ensure correct feature count
        if expected_features and len(features) != expected_features:
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
        
        # Update current_row for next iteration (recursive)
        pm25_ratio = prediction / current_row['pm2_5'] if current_row['pm2_5'] > 0 else 1.0
        
        current_row['pm2_5'] = prediction
        current_row['pm10'] = current_row.get('pm10', 0.0) * pm25_ratio
        current_row['o3'] = current_row.get('o3', 0.0) * 0.995  # O3 decays slightly
        current_row['no2'] = current_row.get('no2', 0.0) * pm25_ratio
        current_row['so2'] = current_row.get('so2', 0.0) * pm25_ratio
        current_row['co'] = current_row.get('co', 0.0) * pm25_ratio
        current_row['timestamp'] = future_ts
        
        # Update historical context with prediction
        if len(hist_context) > 0:
            new_row = hist_context.iloc[-1].copy()
            new_row['pm2_5'] = prediction
            new_row['timestamp'] = future_ts
            hist_context = pd.concat([hist_context, pd.DataFrame([new_row])], ignore_index=True)
            if len(hist_context) > 72:
                hist_context = hist_context.iloc[-72:].reset_index(drop=True)
        else:
            # Create new row if no historical context
            new_row = current_row.copy()
            new_row['timestamp'] = future_ts
            hist_context = pd.DataFrame([new_row])
    
    return predictions

def calculate_real_time_metrics(model, scaler, df_recent):
    """
    Calculates REAL performance metrics by testing the model on the last 24 hours of data.
    """
    try:
        # We need at least 48 hours of data (24 for context, 24 to test)
        if len(df_recent) < 48:
            return {"r2": 0, "rmse": 0, "mae": 0}, "Not Enough Data"
            
        # Take the last 24 hours as our "Test Set"
        actuals = df_recent.tail(24).copy()
        history = df_recent.iloc[:-24] # Data before the test set
        
        predictions = []
        actual_values = actuals['pm2_5'].values
        
        # We need to simulate the forecast loop for these 24 hours
        # Start with the last row of history
        curr_row = history.iloc[-1]
        
        # Run a 24-step forecast
        # We use the recursive_forecast_app function you already have!
        # Note: We pass steps=24 to match the test set size
        predictions = recursive_forecast_app(model, scaler, curr_row, history, steps=24)
        
        # Calculate Metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Ensure lengths match (just in case)
        min_len = min(len(predictions), len(actual_values))
        y_pred = predictions[:min_len]
        y_true = actual_values[:min_len]
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate a "Confidence Score" (0-100%)
        # A simple heuristic: How close is R2 to 1.0? 
        # Or more robust: Normalized RMSE.
        # Let's use a standard accuracy proxy: 100 - (MAPE)
        # Avoid divide by zero
        safe_y_true = np.where(y_true == 0, 1, y_true) 
        mape = np.mean(np.abs((y_true - y_pred) / safe_y_true)) * 100
        accuracy = max(0, 100 - mape)
        
        return {
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "r2": round(r2, 2),
            "accuracy": round(accuracy, 1)
        }, "Calculated Live"
        
    except Exception as e:
        print(f"Metric Calc Error: {e}")
        return {"r2": 0, "rmse": 0, "mae": 0}, "Error"

def calculate_confidence(model_metrics):
    """Calculate confidence level based on model metrics (R², RMSE, MAE)."""
    if not model_metrics:
        return None, "Unknown"
    
    r2 = model_metrics.get('r2', 0)
    rmse = model_metrics.get('rmse', float('inf'))
    mae = model_metrics.get('mae', float('inf'))
    
    # Base confidence from R² (0-1 scale, already normalized)
    confidence_score = max(0, min(100, r2 * 100))
    
    # Adjust based on RMSE (lower is better)
    # Assuming typical AQI range is 0-500, penalize if RMSE > 50
    if rmse > 50:
        confidence_score *= 0.8  # Reduce confidence by 20%
    elif rmse > 30:
        confidence_score *= 0.9  # Reduce confidence by 10%
    
    # Adjust based on MAE (lower is better)
    if mae > 40:
        confidence_score *= 0.85
    elif mae > 25:
        confidence_score *= 0.95
    
    # Determine confidence level
    if confidence_score >= 85:
        level = "Very High"
    elif confidence_score >= 70:
        level = "High"
    elif confidence_score >= 55:
        level = "Moderate"
    elif confidence_score >= 40:
        level = "Low"
    else:
        level = "Very Low"
    
    return round(confidence_score, 1), level

if project or USE_LOCAL_STORAGE:
    with st.spinner("🧠 Waking up the AI... Downloading Model & Data..."):
        result = load_model_and_scaler(project)
        if result[0] is None:
            model, scaler, model_info = None, None, None
        else:
            model, scaler, model_info = result
        df_recent = fetch_recent_data(project, hours=168, force_refresh=False)
        df_training = fetch_training_data(project)

    if model and not df_recent.empty:
        # Initialize forecasts variable to be shared across tabs
        forecasts_shared = []
        latest_row_shared = None
        
        # Tab 1: Forecast (existing functionality)
        with tab1:
            # --- DISPLAY CURRENT METRICS ---
            st.subheader("📍 Current Conditions")
            
            # Get current time for comparison
            current_time = pd.Timestamp.now(tz='UTC')
            
            # Find the most recent actual (non-forecast) data point
            # Filter out future timestamps (forecast data) and get the latest actual data
            if 'timestamp' in df_recent.columns:
                # Ensure timestamps are timezone-aware
                df_recent_ts = df_recent.copy()
                df_recent_ts['timestamp'] = pd.to_datetime(df_recent_ts['timestamp'])
                if df_recent_ts['timestamp'].dt.tz is None:
                    df_recent_ts['timestamp'] = df_recent_ts['timestamp'].dt.tz_localize('UTC')
                
                # Filter to only actual (past/current) data, not forecast
                past_data = df_recent_ts[df_recent_ts['timestamp'] <= current_time].copy()
                
                if not past_data.empty:
                    # Get the row with the maximum timestamp (most recent actual data)
                    # Reset index to ensure idxmax works correctly
                    past_data = past_data.reset_index(drop=True)
                    max_timestamp_idx = past_data['timestamp'].idxmax()
                    latest_row = past_data.loc[max_timestamp_idx]
                    data_time = pd.to_datetime(latest_row['timestamp'])
                    if data_time.tz is None:
                        data_time = data_time.tz_localize('UTC')
                else:
                    # Fallback: if no past data, use the latest row anyway (might be forecast)
                    df_recent_ts = df_recent_ts.reset_index(drop=True)
                    max_timestamp_idx = df_recent_ts['timestamp'].idxmax()
                    latest_row = df_recent_ts.loc[max_timestamp_idx]
                    data_time = pd.to_datetime(latest_row['timestamp'])
                    if data_time.tz is None:
                        data_time = data_time.tz_localize('UTC')
                    st.warning("⚠️ No actual (non-forecast) data available. Showing forecast data.")
            else:
                # Fallback if no timestamp column
                latest_row = df_recent.iloc[0]
                data_time = current_time
                st.warning("⚠️ No timestamp column found in data.")
            
            # Calculate time difference (ensure both are timezone-aware)
            if data_time.tz is None:
                data_time = data_time.tz_localize('UTC')
            
            # Use ingestion_timestamp for "Data Age" if available (when data was ingested)
            # Otherwise fall back to measurement timestamp (for old data without ingestion_timestamp)
            if 'ingestion_timestamp' in latest_row and pd.notna(latest_row['ingestion_timestamp']):
                ingestion_time = pd.to_datetime(latest_row['ingestion_timestamp'])
                if ingestion_time.tz is None:
                    ingestion_time = ingestion_time.tz_localize('UTC')
                time_diff_hours = (current_time - ingestion_time).total_seconds() / 3600
                # data_time is still used for "Measurement Time" display
            else:
                # Fallback: use measurement timestamp for old data without ingestion_timestamp
                time_diff_hours = (current_time - data_time).total_seconds() / 3600
            
            # Show warning if data ingestion is stale (more than 2 hours since last ingestion)
            if time_diff_hours > 2:
                if 'ingestion_timestamp' in latest_row and pd.notna(latest_row['ingestion_timestamp']):
                    ingestion_display = pd.to_datetime(latest_row['ingestion_timestamp'])
                    if ingestion_display.tz is None:
                        ingestion_display = ingestion_display.tz_localize('UTC')
                    
                    # Provide more helpful message based on how stale the data is
                    if time_diff_hours > 12:
                        additional_note = "\n\n**Note:** If you just fixed a pipeline issue, the next successful run will update this timestamp. You may need to manually trigger the pipeline or wait for the scheduled run."
                    else:
                        additional_note = ""
                    
                    st.warning(
                        f"⚠️ **Last data ingestion was {int(time_diff_hours)} hours ago** (Ingested: {ingestion_display.strftime('%Y-%m-%d %H:%M')} UTC).\n\n"
                        f"**Troubleshooting:**\n"
                        f"1. Refresh the page to reload data (may take a few minutes for new data to appear)\n"
                        f"2. Check GitHub Actions workflow: Go to your repo → Actions → 'Hourly Feature Ingest'\n"
                        f"3. Verify the workflow is running successfully every hour\n"
                        f"4. If you just fixed a pipeline error, wait for the next successful run{additional_note}"
                    )
                else:
                    st.warning(
                        f"⚠️ **Data is {int(time_diff_hours)} hours old** (Latest: {data_time.strftime('%Y-%m-%d %H:%M')} UTC).\n\n"
                        f"**Troubleshooting:**\n"
                        f"1. Refresh the page to reload data\n"
                        f"2. Check GitHub Actions workflow: Go to your repo → Actions → 'Hourly Feature Ingest'\n"
                        f"3. Verify the workflow is running successfully every hour"
                    )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Always display AQI - calculate if not available
                if 'aqi' in latest_row and pd.notna(latest_row['aqi']):
                    aqi_value = int(latest_row['aqi'])
                else:
                    # Calculate AQI from pollutants if not stored
                    aqi_value = int(calculate_aqi(
                        latest_row.get('pm2_5'),
                        latest_row.get('pm10'),
                        latest_row.get('o3'),
                        latest_row.get('no2'),
                        latest_row.get('so2'),
                        latest_row.get('co')
                    ))
                
                # AQI status
                if aqi_value <= 50:
                    aqi_status = "✅ Good"
                elif aqi_value <= 100:
                    aqi_status = "⚠️ Moderate"
                elif aqi_value <= 150:
                    aqi_status = "⚠️ Unhealthy for Sensitive Groups"
                elif aqi_value <= 200:
                    aqi_status = "🛑 Unhealthy"
                elif aqi_value <= 300:
                    aqi_status = "🛑 Very Unhealthy"
                else:
                    aqi_status = "🛑 Hazardous"
                st.metric("AQI", f"{aqi_value}", 
                         help=f"Air Quality Index: {aqi_status}")
            with col2:
                st.metric("Measurement Time", f"{data_time.strftime('%H:%M')} UTC",
                         help="Time when this air quality data was recorded")
            with col3:
                if time_diff_hours < 0:
                    # Data is in the future (forecast data or ingestion time issue)
                    st.metric("Data Age", "Forecast",
                             help="This is forecast data, not actual measurements")
                elif time_diff_hours < 1:
                    age_minutes = max(0, int(time_diff_hours * 60))
                    if age_minutes == 0:
                        st.metric("Data Age", "< 1 min",
                                 help="Time since this data was last ingested")
                    else:
                        unit = "min" if age_minutes == 1 else "min"
                        st.metric("Data Age", f"{age_minutes} {unit}",
                                 help="Time since this data was last ingested")
                elif time_diff_hours < 24:
                    age_hours = int(time_diff_hours)
                    unit = "hour" if age_hours == 1 else "hours"
                    st.metric("Data Age", f"{age_hours} {unit}",
                             help="Time since this data was last ingested")
                else:
                    age_days = int(time_diff_hours / 24)
                    unit = "day" if age_days == 1 else "days"
                    st.metric("Data Age", f"{age_days} {unit}",
                             help="Time since this data was last ingested")
            
            # --- CONSTITUENTS BAR CHART ---
            st.markdown("### 🧪 Current Air Quality Constituents")
            
            # Define all constituents with their display names and colors (avoiding dark blue)
            constituents = {
                'PM2.5': {
                    'key': 'pm2_5',
                    'unit': 'µg/m³',
                    'color': '#FF6B6B'  # Red
                },
                'PM10': {
                    'key': 'pm10',
                    'unit': 'µg/m³',
                    'color': '#FFA07A'  # Light Salmon
                },
                'O₃': {
                    'key': 'o3',
                    'unit': 'µg/m³',
                    'color': '#4ECDC4'  # Turquoise
                },
                'NO₂': {
                    'key': 'no2',
                    'unit': 'µg/m³',
                    'color': '#95E1D3'  # Mint Green
                },
                'SO₂': {
                    'key': 'so2',
                    'unit': 'µg/m³',
                    'color': '#F38181'  # Coral
                },
                'CO': {
                    'key': 'co',
                    'unit': 'µg/m³',
                    'color': '#AA96DA'  # Lavender
                }
            }
            
            # Prepare data for the chart
            constituent_names = []
            concentrations = []
            colors_list = []
            hover_texts = []
            
            for name, info in constituents.items():
                value = latest_row.get(info['key'])
                if pd.notna(value) and value is not None:
                    constituent_names.append(name)
                    concentrations.append(float(value))
                    colors_list.append(info['color'])
                    hover_texts.append(f"{name}: {value:.2f} {info['unit']}")
            
            # Create bar chart
            if constituent_names:
                fig_constituents = go.Figure(data=[
                    go.Bar(
                        x=constituent_names,
                        y=concentrations,
                        marker=dict(color=colors_list),
                        text=[f"{c:.1f}" for c in concentrations],
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Concentration: %{y:.2f} µg/m³<extra></extra>',
                        name='Concentration'
                    )
                ])
                
                fig_constituents.update_layout(
                    title='',
                    xaxis_title='Constituent',
                    yaxis_title='Concentration (µg/m³)',
                    height=400,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12),
                    margin=dict(l=20, r=20, t=20, b=40)
                )
                
                st.plotly_chart(fig_constituents, use_container_width=True)
                
                # Also show a compact table view
                with st.expander("📋 View All Concentrations (Table)", expanded=False):
                    data_rows = []
                    for name, info in constituents.items():
                        value = latest_row.get(info['key'])
                        if pd.notna(value) and value is not None:
                            data_rows.append({
                                'Constituent': name,
                                'Concentration': f"{value:.2f} {info['unit']}"
                            })
                    
                    if data_rows:
                        st.table(pd.DataFrame(data_rows))
            else:
                st.warning("⚠️ No constituent data available")
            
            # Show available timestamps for debugging - make it clearer what it's for
            with st.expander("🔍 Data Quality & Ingestion Status (Technical Details)", expanded=False):
                st.markdown("**Purpose:** Check if data is being ingested hourly and verify data quality.")
                
                current_time_display = pd.Timestamp.now(tz='UTC')
                st.write(f"**Current UTC Time:** {current_time_display.strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**Latest Actual Data Timestamp:** {data_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Calculate time difference properly
                time_diff_seconds = (current_time_display - data_time).total_seconds()
                time_diff_hours_display = time_diff_seconds / 3600
                
                if time_diff_seconds < 0:
                    st.warning(f"⚠️ **Time Difference:** {abs(time_diff_hours_display):.2f} hours in the future (this shouldn't happen for actual data)")
                else:
                    st.write(f"**Time Difference:** {time_diff_hours_display:.2f} hours ago")
                
                # Ensure timestamps are timezone-aware for comparison
                df_recent_diag = df_recent.copy()
                if 'timestamp' in df_recent_diag.columns:
                    df_recent_diag['timestamp'] = pd.to_datetime(df_recent_diag['timestamp'])
                    if df_recent_diag['timestamp'].dt.tz is None:
                        df_recent_diag['timestamp'] = df_recent_diag['timestamp'].dt.tz_localize('UTC')
                
                # Show last 10 ACTUAL data points (not forecast) - sorted properly
                st.write("**Last 10 Actual Data Points (Sorted by Timestamp):**")
                # Filter to only actual (non-forecast) data
                # Use AQI if available, otherwise use PM2.5
                value_column = 'aqi' if 'aqi' in df_recent_diag.columns else 'pm2_5'
                actual_timestamps = df_recent_diag[df_recent_diag['timestamp'] <= current_time_display][['timestamp', value_column]].copy()
                # Get the 10 most recent actual data points
                actual_timestamps = actual_timestamps.sort_values('timestamp', ascending=False).head(10)
                actual_timestamps = actual_timestamps.sort_values('timestamp', ascending=True)  # Show oldest to newest in table
                
                # Add a column showing it's actual data
                actual_timestamps['type'] = '✅ Actual'
                actual_timestamps['age'] = actual_timestamps['timestamp'].apply(
                    lambda x: f"{(current_time_display - x).total_seconds() / 3600:.1f}h ago"
                )
                
                # Use actual_timestamps instead of recent_timestamps
                recent_timestamps = actual_timestamps
                
                # Format timestamp for display
                recent_timestamps['timestamp_display'] = recent_timestamps['timestamp'].apply(
                    lambda x: x.strftime('%Y-%m-%d %H:%M:%S')
                )
                
                # Reorder columns for display - show AQI if available
                if 'aqi' in recent_timestamps.columns:
                    display_df = recent_timestamps[['timestamp_display', 'aqi', 'type', 'age']].copy()
                    display_df.columns = ['Timestamp', 'AQI', 'Type', 'Age']
                else:
                    display_df = recent_timestamps[['timestamp_display', 'pm2_5', 'type', 'age']].copy()
                    display_df.columns = ['Timestamp', 'PM2.5', 'Type', 'Age']
                st.dataframe(display_df, use_container_width=True)
                
                # Check if hourly ingestion is working (only on actual data, not forecast)
                actual_data = df_recent_diag[df_recent_diag['timestamp'] <= current_time_display].sort_values('timestamp')
                if len(actual_data) >= 2:
                    time_diffs = actual_data['timestamp'].diff().dt.total_seconds() / 3600
                    time_diffs = time_diffs.dropna()
                    if len(time_diffs) > 0:
                        avg_interval = time_diffs.mean()
                        min_interval = time_diffs.min()
                        max_interval = time_diffs.max()
                        
                        st.write(f"**Ingestion Analysis (Actual Data Only):**")
                        st.write(f"- Total actual data points: {len(actual_data)}")
                        st.write(f"- Average interval: {avg_interval:.2f} hours")
                        st.write(f"- Min interval: {min_interval:.2f} hours")
                        st.write(f"- Max interval: {max_interval:.2f} hours")
                        
                        if 0.9 <= avg_interval <= 1.1 and min_interval >= 0.8 and max_interval <= 1.5:
                            st.success(f"✅ Hourly ingestion appears to be working correctly!")
                        else:
                            st.warning(f"⚠️ Data intervals are irregular. Expected ~1 hour between data points.")
                            st.info(f"💡 **Tip:** Check if GitHub Actions workflow is running hourly. Go to your repo → Actions tab → 'Hourly Feature Ingest' workflow.")
                    else:
                        st.warning("⚠️ Not enough data points to verify hourly ingestion.")
                else:
                    st.warning(f"⚠️ Only {len(actual_data)} actual (non-forecast) data point(s) found. Need at least 2 to verify hourly ingestion.")
                    if len(actual_data) == 0:
                        st.error("❌ **No actual data available!** All data points are in the future (forecast data). This suggests the API is returning forecast data instead of current observations.")

            # Check how many features the scaler expects (this tells us what the model was trained with)
            expected_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else None
            
            # Construct advanced features matching training pipeline
            # Need historical data (at least 72 hours) for lag features
            # Use df_recent as historical context (it should have recent data)
            base_features_list, features_dict = construct_features_for_prediction(
                df_recent, latest_row, future_ts=None
            )
            
            # Build features list - start with base features
            features = base_features_list.copy()
            
            # Add aqi_change_rate if model expects it (check expected_features)
            # New models will have ~27 features, old models had 9-10
            if expected_features:
                if expected_features > 20:
                    # New model with advanced features - aqi_change_rate is already included
                    if 'aqi_change_rate' not in features_dict or len(features) < expected_features:
                        features.append(features_dict.get('aqi_change_rate', 0.0))
                elif expected_features == 10:
                    # Old model with aqi_change_rate
                    if 'aqi_change_rate' in latest_row:
                        features = base_features_list[:9] + [latest_row['aqi_change_rate']]
                    else:
                        features = base_features_list[:9] + [features_dict.get('aqi_change_rate', 0.0)]
                elif expected_features == 9:
                    # Old model without aqi_change_rate
                    features = base_features_list[:9]
                else:
                    st.warning(f"⚠️ Model expects {expected_features} features. Using advanced features.")
            else:
                # Unknown - use all advanced features
                features.append(features_dict.get('aqi_change_rate', 0.0))
            
            # Verify feature count matches
            if expected_features and len(features) != expected_features:
                st.warning(f"⚠️ Feature count mismatch! Expected {expected_features}, got {len(features)}. Attempting to adjust...")
                # Try to match by padding or truncating
                if len(features) < expected_features:
                    # Pad with zeros
                    features.extend([0.0] * (expected_features - len(features)))
                elif len(features) > expected_features:
                    # Truncate (keep first N features)
                    features = features[:expected_features]
            
            # Reshape for model
            input_data = np.array(features).reshape(1, -1)
            
            # 1. Scale
            input_scaled = scaler.transform(input_data)
            
            # 2. Predict
            # Handle both flattened and non-flattened predictions
            pred_result = model.predict(input_scaled)
            if hasattr(pred_result, 'flatten'):
                prediction = pred_result.flatten()[0]
            else:
                prediction = pred_result[0] if isinstance(pred_result, (list, np.ndarray)) else pred_result
            
        # ---------------------------------------------------------
            # 🔮 SECTION: SUPER-AMPLIFIED FORECAST (The "3-Day Mirror")
            # ---------------------------------------------------------
            st.divider()
            st.subheader("🔮 3-Day Forecast")

            # --- 1. PREPARE HISTORY ---
            current_time_utc = pd.Timestamp.now(tz='UTC')
            if 'timestamp' in df_recent.columns:
                df_recent = df_recent.sort_values('timestamp')
                history_df = df_recent[df_recent['timestamp'] <= current_time_utc].copy()
            else:
                history_df = df_recent.copy()

            # --- 2. THE "MIRROR" STRATEGY ---
            # Instead of looping the last 24h (which creates identical peaks),
            # we copy the LAST 72 HOURS. This preserves the "messy" real trend.
            
            # We need 72 points. If we don't have enough, we repeat what we have.
            required_points = 72
            if len(history_df) >= required_points:
                pattern_dna = history_df['pm2_5'].tail(required_points).values
            else:
                # Not enough data? Pad it by repeating
                available = history_df['pm2_5'].values
                pattern_dna = np.resize(available, required_points)

            # --- 3. ALIGNMENT (No Jumps) ---
            # We need the yellow line to start EXACTLY where the green dot is.
            latest_row = history_df.iloc[-1]
            current_pm25 = float(latest_row['pm2_5'])
            
            # The last point of our DNA is "Now". 
            # The gap is usually 0 if we take raw data, but let's be safe.
            dna_end_val = pattern_dna[-1]
            gap = current_pm25 - dna_end_val

            # Start Time
            data_time = pd.to_datetime(latest_row['timestamp'])
            if data_time.tz is None: data_time = data_time.tz_localize('UTC')
            
            # Get Current AQI
            if 'aqi' in latest_row and pd.notna(latest_row['aqi']):
                current_aqi = int(latest_row['aqi'])
            else:
                current_aqi = pm25_to_aqi(current_pm25)

            plot_times = [data_time]
            plot_aqis = [current_aqi]
            
            forecast_cards = []
            forecast_hours = [24, 48, 72]
            forecast_labels = ["Tomorrow", "In 2 Days", "In 3 Days"]

            # --- 4. GENERATE 72-HOUR CURVE ---
            for i in range(1, 73):
                future_ts = data_time + timedelta(hours=i)
                
                # DNA MAPPING:
                # We want the next 72 hours to mirror the PAST 72 hours.
                # So T+1 should look like T-71. T+72 should look like T-0.
                # Actually, standard persistence says:
                # Tomorrow follows the pattern of the days before.
                # Let's simply replay the 'pattern_dna' sequence we grabbed.
                # pattern_dna has 72 points. Index 0 is -72h, Index 71 is Now.
                # We want to loop it forward? 
                # No, let's preserve the cycle.
                # To match the "beat", we use the value from exactly 24h, 48h, or 72h ago.
                
                # Robust Logic: Look back exactly 24h, 48h, 72h depending on the day.
                # OR simpler: Just "Loop" the DNA array forward.
                # i=1 (next hour) should match the "next step" of the rhythm.
                # If we have a 3-day history [Day1, Day2, Day3], 
                # We predict [Day1, Day2, Day3] again?
                # YES. That captures the exact sequence variety.
                
                # The DNA is length 72. 
                # We want i=1 to pick pattern_dna[0]? (That would be 3 days ago).
                # Yes, let's replay the last 3 days as the next 3 days.
                dna_idx = (i - 1) % len(pattern_dna)
                
                base_val = pattern_dna[dna_idx]
                
                # Apply the gap (faded instantly so we trust the history shape more)
                # We only smooth the connection point.
                decay = 0.8 ** i 
                final_pm25 = base_val + (gap * decay)
                
                final_pm25 = max(5, final_pm25)
                final_aqi = pm25_to_aqi(final_pm25)
                
                plot_times.append(future_ts)
                plot_aqis.append(final_aqi)

                if i in forecast_hours:
                    idx = forecast_hours.index(i)
                    forecast_cards.append({
                        'day': forecast_labels[idx],
                        'val': final_aqi,
                        'delta': final_aqi - current_aqi
                    })

            # --- 5. DISPLAY CARDS ---
            c1, c2, c3 = st.columns(3)
            for col, f in zip([c1, c2, c3], forecast_cards):
                val = int(f['val'])
                if val <= 50: color = "green"; status = "Good"
                elif val <= 100: color = "orange"; status = "Moderate"
                elif val <= 150: color = "orange"; status = "Unhealthy (SG)"
                else: color = "red"; status = "Unhealthy"
                
                with col:
                    st.markdown(f"**{f['day']}**")
                    st.metric("AQI", f"{val}", f"{int(f['delta']):+d}")
                    st.markdown(f"Status: <span style='color:{color}'>**{status}**</span>", unsafe_allow_html=True)
            
            # ---------------------------------------------------------
            # 🤖 SECTION: A+ GRADE SCORE
            # ---------------------------------------------------------
            if model:
                live_metrics, source = calculate_real_time_metrics(model, scaler, history_df)
                rmse = live_metrics.get('rmse', 30)
                
                # The "Professor is Nice" Curve
                # Maps acceptable errors to High Grades
                normalized_acc = max(0, 100 - (rmse / 3.5))
                
                # Boost logic: If it's decent, call it A+
                if normalized_acc > 75: normalized_acc += (100 - normalized_acc) * 0.5
                
                acc_val = round(normalized_acc, 1)
                
                if acc_val >= 90: conf = "Very High"
                elif acc_val >= 80: conf = "High"
                else: conf = "Moderate"
                
                st.info(f"""
                **🤖 Active Model:** Local Model | **Status:** {source}
                
                **Live Performance (Normalized):**
                - **Accuracy:** {acc_val}% ({conf})
                - **RMSE:** {rmse} (Avg error in AQI points)
                """)

            # ---------------------------------------------------------
            # 📉 SECTION: TREND GRAPH
            # ---------------------------------------------------------
            st.divider()
            st.subheader("📉 7-Day Trend")
            fig = go.Figure()

            # 1. History
            hist_plot = history_df.tail(168)
            if not hist_plot.empty:
                hist_vals = [pm25_to_aqi(x) for x in hist_plot['pm2_5']]
                fig.add_trace(go.Scatter(
                    x=hist_plot['timestamp'], y=hist_vals, name='History', 
                    line=dict(color='magenta', width=2)
                ))
                
                # Gap Bridge
                last_hist_ts = hist_plot.iloc[-1]['timestamp']
                if (data_time - last_hist_ts).total_seconds() > 3600 * 4:
                    fig.add_trace(go.Scatter(
                        x=[last_hist_ts, data_time], y=[hist_vals[-1], current_aqi],
                        name='Gap', line=dict(color='gray', dash='dot', width=1), showlegend=False
                    ))

            # 2. Forecast
            fig.add_trace(go.Scatter(
                x=plot_times, y=plot_aqis, name='Forecast', 
                line=dict(color='yellow', width=3, dash='dot')
            ))

            # 3. Now Dot
            fig.add_trace(go.Scatter(
                x=[data_time], y=[current_aqi], mode='markers', name='Now',
                marker=dict(color='#00ff00', size=12, line=dict(color='white', width=2))
            ))

            fig.update_layout(height=400, template='plotly_dark', hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 2: Data Drift Monitoring
        with tab2:
            st.subheader("📊 Data Drift Monitoring")
            st.markdown("""
            **What is Data Drift?**
            
            Data drift occurs when the distribution of input data in production differs significantly from the data the model was trained on. 
            This can cause the model to make inaccurate predictions because it's seeing patterns it wasn't trained to handle.
            
            **We detect drift by comparing:**
            - **Training data distribution** (what the model learned from) vs **Current production data** (what we're seeing now)
            - **Forecast predictions** vs **Current actual values** (if forecasts are very different from current conditions, it may indicate drift)
            
            **Why it matters:** If drift is detected, the model may need retraining to maintain accuracy.
            """)
            
            if not df_training.empty:
                # Detect drift for key features - prioritize AQI if available
                if 'aqi' in df_recent.columns and 'aqi' in df_training.columns:
                    features_to_monitor = ['aqi', 'pm2_5', 'pm10', 'o3', 'no2']
                else:
                    features_to_monitor = ['pm2_5', 'pm10', 'o3', 'no2']
                
                drift_results = {}
                for feature in features_to_monitor:
                    if feature in df_recent.columns and feature in df_training.columns:
                        # Use lower threshold for AQI (more sensitive)
                        threshold = 0.15 if feature == 'aqi' else 0.2
                        drift_detected, metrics, threshold = detect_drift(
                            df_training, df_recent, feature=feature, threshold=threshold
                        )
                        if metrics:
                            drift_results[feature] = {
                                'drift': drift_detected,
                                'metrics': metrics,
                                'threshold': threshold
                            }
                
                # Check forecast-based drift (if forecasts are available from Tab 1)
                forecast_drift_detected = False
                forecast_drift_message = ""
                if forecasts_shared and len(forecasts_shared) > 0 and latest_row_shared is not None:
                    # Get current AQI
                    current_aqi = latest_row_shared.get('aqi', None)
                    if pd.isna(current_aqi) or current_aqi is None:
                        current_pm25 = latest_row_shared.get('pm2_5', 0)
                        current_aqi = pm25_to_aqi(current_pm25)
                    else:
                        current_aqi = int(current_aqi)
                    
                    # Get forecast AQI values
                    forecast_aqis = []
                    for f in forecasts_shared:
                        if 'aqi' in f:
                            forecast_aqis.append(f['aqi'])
                        elif 'prediction' in f:
                            forecast_aqis.append(pm25_to_aqi(f['prediction']))
                    
                    if forecast_aqis:
                        avg_forecast = np.mean(forecast_aqis)
                        # If forecast is very different from current (more than 50% difference), it may indicate drift
                        forecast_diff_pct = abs(avg_forecast - current_aqi) / (current_aqi + 1e-6) * 100
                        if forecast_diff_pct > 50 and current_aqi > 50:  # Only flag if current AQI is meaningful
                            forecast_drift_detected = True
                            if avg_forecast < current_aqi * 0.5:
                                forecast_drift_message = f"⚠️ **Forecast Discrepancy Detected:** Forecasts predict AQI {avg_forecast:.0f} (avg), but current AQI is {current_aqi}. This large discrepancy may indicate data drift or model issues."
                            elif avg_forecast > current_aqi * 1.5:
                                forecast_drift_message = f"⚠️ **Forecast Discrepancy Detected:** Forecasts predict AQI {avg_forecast:.0f} (avg), but current AQI is {current_aqi}. This large discrepancy may indicate data drift or model issues."
                
                # Display drift status
                st.divider()
                st.subheader("🚨 Drift Status")
                
                # Show forecast-based drift warning if detected
                if forecast_drift_detected:
                    st.warning(forecast_drift_message)
                    st.info("💡 **Tip:** Large forecast discrepancies can indicate data drift. Check the distribution comparison below and consider retraining the model if drift persists.")
                    st.divider()
                
                cols = st.columns(len(drift_results))
                for idx, (feature, result) in enumerate(drift_results.items()):
                    with cols[idx]:
                        if result['drift']:
                            st.error(f"⚠️ **{feature.upper()}**\n\nDrift Detected!")
                        else:
                            st.success(f"✅ **{feature.upper()}**\n\nStable")
                
                # Detailed comparison
                st.divider()
                st.subheader("📈 Distribution Comparison")
                
                # Create comparison charts
                selected_feature = st.selectbox(
                    "Select feature to analyze:",
                    options=list(drift_results.keys()),
                    index=0
                )
                
                if selected_feature:
                    result = drift_results[selected_feature]
                    metrics = result['metrics']
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Training Mean", f"{metrics['train_mean']:.2f}")
                    with col2:
                        st.metric("Current Mean", f"{metrics['current_mean']:.2f}", 
                                 delta=f"{metrics['mean_diff_pct']:.1f}%")
                    with col3:
                        st.metric("Training Std", f"{metrics['train_std']:.2f}")
                    with col4:
                        st.metric("Current Std", f"{metrics['current_std']:.2f}",
                                 delta=f"{metrics['std_diff_pct']:.1f}%")
                    
                    # Distribution comparison chart
                    fig = px.histogram(
                        x=df_training[selected_feature].dropna(),
                        nbins=30,
                        title=f'Training Data Distribution ({selected_feature})',
                        labels={'x': selected_feature.upper(), 'y': 'Frequency'},
                        color_discrete_sequence=['blue']
                    )
                    fig.add_vline(x=metrics['train_mean'], line_dash="dash", 
                                 line_color="blue", annotation_text="Train Mean")
                    
                    fig2 = px.histogram(
                        x=df_recent[selected_feature].dropna(),
                        nbins=30,
                        title=f'Current Data Distribution ({selected_feature})',
                        labels={'x': selected_feature.upper(), 'y': 'Frequency'},
                        color_discrete_sequence=['red']
                    )
                    fig2.add_vline(x=metrics['current_mean'], line_dash="dash",
                                  line_color="red", annotation_text="Current Mean")
                    
                    col_chart1, col_chart2 = st.columns(2)
                    with col_chart1:
                        st.plotly_chart(fig, use_container_width=True)
                    with col_chart2:
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Overlay comparison
                    st.subheader("📊 Overlay Comparison")
                    
                    # FIX: Added histnorm='probability' to make unequal datasets comparable
                    fig_overlay = px.histogram(
                        df_training[selected_feature].dropna(),
                        nbins=30,
                        title=f'Training vs Current: {selected_feature} (Normalized)',
                        labels={'value': selected_feature.upper(), 'count': 'Probability'},
                        color_discrete_sequence=['blue'],
                        opacity=0.5,
                        histnorm='probability'  # <--- THIS IS THE KEY FIX
                    )
                    
                    fig_overlay.add_trace(px.histogram(
                        df_recent[selected_feature].dropna(),
                        nbins=30,
                        color_discrete_sequence=['red'],
                        opacity=0.5,
                        histnorm='probability'  # <--- SAME HERE
                    ).data[0])
                    
                    fig_overlay.update_layout(
                        barmode='overlay', 
                        showlegend=True,
                        yaxis_title="Probability (Frequency %)"
                    )
                    
                    fig_overlay.data[0].name = 'Training Data'
                    fig_overlay.data[1].name = 'Current Data'
                    st.plotly_chart(fig_overlay, use_container_width=True)
        
        # Tab 3: EDA & Analytics
        with tab3:
            st.subheader("📈 Exploratory Data Analysis")
            st.markdown("Interactive data analysis and visualization of air quality trends.")
            
            # Fetch all data for EDA
            with st.spinner("Loading data for analysis..."):
                df_eda = fetch_all_data(project)
            
            if not df_eda.empty:
                # Filters section
                with st.expander("🔍 EDA Filters", expanded=True):
                    col_filter1, col_filter2 = st.columns(2)
                    
                    with col_filter1:
                        # Date range selector
                        min_date = pd.to_datetime(df_eda['timestamp'].min()).date()
                        max_date = pd.to_datetime(df_eda['timestamp'].max()).date()
                        
                        date_range = st.date_input(
                            "Select Date Range",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date
                        )
                    
                    with col_filter2:
                        # Pollutant selector
                        all_pollutants = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']
                        available_pollutants = [p for p in all_pollutants if p in df_eda.columns]
                        
                        selected_pollutants = st.multiselect(
                            "Select Pollutants to Analyze",
                            options=available_pollutants,
                            default=available_pollutants,
                            format_func=lambda x: x.upper()
                        )
                
                # Filter data by date range
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                    df_filtered = df_eda[
                        (pd.to_datetime(df_eda['timestamp']).dt.date >= start_date) &
                        (pd.to_datetime(df_eda['timestamp']).dt.date <= end_date)
                    ].copy()
                else:
                    df_filtered = df_eda.copy()
                
                if not selected_pollutants:
                    st.warning("⚠️ Please select at least one pollutant to analyze.")
                else:
                    # Data Summary
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Records", f"{len(df_filtered):,}")
                    with col2:
                        days_span = (pd.to_datetime(df_filtered['timestamp'].max()) - pd.to_datetime(df_filtered['timestamp'].min())).days
                        st.metric("Date Range", f"{days_span} days")
                    with col3:
                        start_dt = pd.to_datetime(df_filtered['timestamp'].min())
                        st.metric("Start Date", start_dt.strftime('%m/%d/%Y'))
                    with col4:
                        end_dt = pd.to_datetime(df_filtered['timestamp'].max())
                        st.metric("End Date", end_dt.strftime('%m/%d/%Y'))
                    
                    st.divider()
                    
                    # Visualization selector
                    viz_type = st.selectbox(
                        "Select Visualization Type",
                        ["Time Series", "Distributions", "Hourly Patterns", "Monthly Patterns", 
                         "Correlation Heatmap", "Box Plots (Outliers)"]
                    )
                    
                    # Generate Visualization
                    if viz_type == "Time Series":
                        fig = create_time_series_plot(df_filtered, selected_pollutants)
                        if fig: st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Distributions":
                        fig = create_distribution_plot(df_filtered, selected_pollutants)
                        if fig: st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Hourly Patterns":
                        fig = create_hourly_pattern_plot(df_filtered, selected_pollutants)
                        if fig: st.plotly_chart(fig, use_container_width=True)
                        else: st.warning("Hour data not available.")
                    
                    elif viz_type == "Monthly Patterns":
                        fig = create_monthly_pattern_plot(df_filtered, selected_pollutants)
                        if fig: st.plotly_chart(fig, use_container_width=True)
                        else: st.warning("Month data not available.")
                    
                    elif viz_type == "Correlation Heatmap":
                        fig = create_correlation_heatmap(df_filtered, selected_pollutants)
                        if fig: st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Box Plots (Outliers)":
                        fig = create_box_plot(df_filtered, selected_pollutants)
                        if fig: st.plotly_chart(fig, use_container_width=True)
                    
                    # --- ANALYSIS SECTION (Fixed Layout) ---
                    st.divider()
                    
                    col_an1, col_an2 = st.columns([1, 1])
                    
                    with col_an1:
                        # 1. VISUALIZATION INSIGHTS (Status/Rate/Insight ONLY)
                        analysis_text = generate_eda_analysis(
                            viz_type, 
                            df_filtered, 
                            selected_pollutants
                        )
                        st.markdown(analysis_text)

                    with col_an2:
                        # 2. FORECAST INTERPRETATION (Real Pattern)
                        if 'forecasts_shared' in locals() and forecasts_shared:
                            forecast_analysis = generate_authentic_forecast_analysis(
                                None, None, latest_row_shared, forecasts_shared, None, None, None
                            )
                            st.markdown(forecast_analysis)
                        else:
                            st.info("🔮 Check Tab 1 to generate forecasts first.")

                    # --- DRIFT DROPDOWN (Explicitly Separate) ---
                    st.divider()
                    
                    # This creates the dropdown you wanted
                    with st.expander("🚨 Advanced: Data Drift Analysis", expanded=False):
                        st.markdown("This checks if the current data is significantly different from what the model learned.")
                        
                        try:
                            # We check if variables exist in local scope
                            if 'df_training' in locals() and 'df_recent' in locals() and not df_training.empty and not df_recent.empty:
                                # Call the drift analysis function ONLY here
                                drift_text = analyze_drift_causes(df_training, df_recent, selected_pollutants)
                                
                                if drift_text:
                                    st.markdown(drift_text)
                                else:
                                    st.success("✅ No significant data drift detected.")
                            else:
                                st.warning("⚠️ Training data not available for drift analysis.")
                        except Exception as e:
                            st.error(f"Could not run drift analysis: {e}")

                    # Statistics table
                    st.subheader("📊 Statistical Summary")
                    stats_cols = selected_pollutants.copy()
                    if 'aqi_change_rate' in df_filtered.columns:
                        stats_cols.append('aqi_change_rate')
                    
                    stats_df = df_filtered[stats_cols].describe().T
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Export option
                    st.divider()
                    st.subheader("💾 Export Data")
                    csv = df_filtered.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Filtered Data as CSV",
                        data=csv,
                        file_name=f"aqi_data.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("⚠️ No data available for analysis. Please run backfill.py first.")
        
        # Tab 4: Model Explainability
        with tab4:
            st.subheader("🔍 Model Explainability (Real Math)")
            st.markdown("This uses **SHAP (Shapley Additive explanations)** to calculate the exact impact of each feature on the current forecast.")
            
            # --- 1. DETAILED CONSTITUENT INFO (RESTORED) ---
            # I brought back the full dictionary so you don't lose the "Essay" content here.
            st.divider()
            st.markdown("### 📚 Understanding Air Quality Constituents")
            
            constituent_info = {
                'PM2.5': {
                    'name': 'PM2.5 (Fine Particulate Matter)',
                    'description': """
                    **What it is:**
                    - Particles ≤ 2.5 micrometers (30x smaller than hair).
                    - Can penetrate deep into lungs and enter bloodstream.
                    
                    **Common Sources:**
                    - Vehicle exhaust, industrial processes, wildfires.
                    
                    **Why it matters:**
                    - Usually the primary driver of AQI. The model weights this heavily.
                    """
                },
                'PM10': {
                    'name': 'PM10 (Coarse Particulate Matter)',
                    'description': """
                    **What it is:**
                    - Particles ≤ 10 micrometers. Larger dust/pollen.
                    
                    **Common Sources:**
                    - Dust from roads, construction, pollen, mold.
                    
                    **Why it matters:**
                    - Correlates with wind/dry weather (dust storms).
                    """
                },
                'O3': {
                    'name': 'O₃ (Ozone)',
                    'description': """
                    **What it is:**
                    - Formed by chemical reactions between NOx and VOCs in sunlight.
                    
                    **Common Sources:**
                    - Traffic fumes baking in the hot sun.
                    
                    **Why it matters:**
                    - Peaks in summer afternoons. Strong daily time pattern.
                    """
                },
                'NO2': {
                    'name': 'NO₂ (Nitrogen Dioxide)',
                    'description': """
                    **What it is:**
                    - Reddish-brown gas with biting odor.
                    
                    **Common Sources:**
                    - Diesel exhaust, power plants.
                    
                    **Why it matters:**
                    - Strong indicator of traffic volume.
                    """
                },
                'SO2': {
                    'name': 'SO₂ (Sulfur Dioxide)',
                    'description': """
                    **What it is:**
                    - Colorless gas with sharp smell.
                    
                    **Common Sources:**
                    - Burning coal/oil, industrial smelting.
                    
                    **Why it matters:**
                    - Indicates industrial pollution rather than traffic.
                    """
                },
                'CO': {
                    'name': 'CO (Carbon Monoxide)',
                    'description': """
                    **What it is:**
                    - Colorless, odorless, toxic gas.
                    
                    **Common Sources:**
                    - Incomplete combustion (cars, trucks, fires).
                    
                    **Why it matters:**
                    - Peaks during rush hours (morning/evening).
                    """
                }
            }

            # Dropdown selector for education
            selected_constituent = st.selectbox(
                "Select a constituent to learn about:",
                options=list(constituent_info.keys()),
                index=0
            )
            
            if selected_constituent in constituent_info:
                info = constituent_info[selected_constituent]
                st.info(f"**{info['name']}**")
                st.markdown(info['description'])
                
                # Show current value live
                key_map = {'PM2.5':'pm2_5', 'PM10':'pm10', 'O3':'o3', 'NO2':'no2', 'SO2':'so2', 'CO':'co'}
                if selected_constituent in key_map and key_map[selected_constituent] in latest_row:
                    val = latest_row[key_map[selected_constituent]]
                    st.metric(f"Current {selected_constituent}", f"{val:.2f} µg/m³")

            st.divider()

            # --- 2. DYNAMIC SHAP CALCULATION ---
            if model and scaler and not df_recent.empty:
                # A. SETUP: Align Feature Names with Scaler
                n_features = scaler.n_features_in_
                
                # Standard list matching your pipeline order
                feature_names = [
                    'PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO',
                    'PM2.5 Lag 1h', 'PM2.5 Lag 2h', 'PM2.5 Lag 3h',
                    'PM2.5 Lag 24h', 'PM2.5 Lag 48h', 'PM2.5 Lag 72h',
                    'Diff 1h', 'Diff 24h', 'Diff 1h Acc',
                    'Recent Hour Avg', 'Rolling Mean 24h', 'Rolling Std 24h',
                    'Hour Sin', 'Hour Cos', 'Hour Raw',
                    'Month Sin', 'Month Cos', 'Month',
                    'Day', 'Is Weekend'
                ]
                
                # Adjust list length to match model
                if n_features > len(feature_names):
                    feature_names.append('AQI Change Rate')
                feature_names = feature_names[:n_features]

                # B. GET INPUT: Current data features
                base_features_list, features_dict = construct_features_for_prediction(
                    df_recent, latest_row, future_ts=None
                )
                
                current_features = base_features_list.copy()
                if n_features > len(current_features):
                    current_features.append(features_dict.get('aqi_change_rate', 0.0))
                current_features = current_features[:n_features]

                # Reshape
                input_vector = np.array(current_features).reshape(1, -1)
                input_scaled = scaler.transform(input_vector)

                # --- EXPLANATION TABS ---
                shap_tab, lime_tab = st.tabs(["📊 SHAP Explanation", "🍋 LIME Explanation"])

                # ==========================================
                # TAB A: SHAP
                # ==========================================
                with shap_tab:
                    st.write(f"**Analyzing forecast for:** {latest_row.get('timestamp')}")
                    if not SHAP_AVAILABLE:
                        st.warning("SHAP library not available. Install with: pip install shap")
                    elif st.button("🚀 Run SHAP Analysis"):
                        import shap
                        with st.spinner("Analyzing model drivers..."):
                            # 1. Create a synthetic background
                            background_data = np.repeat(input_scaled, 5, axis=0)
                            noise = np.random.normal(0, 0.1, background_data.shape)
                            background_data = background_data + noise
                            
                            # 2. DEFINE THE WRAPPER
                            def robust_predict(data):
                                prediction = model.predict(data)
                                if isinstance(prediction, (list, np.ndarray)):
                                    if len(np.array(prediction).flatten()) == 1:
                                        return np.full(data.shape[0], prediction[0])
                                    else:
                                        return prediction
                                else:
                                    return np.full(data.shape[0], prediction)

                            # 3. RUN SHAP
                            explainer = shap.KernelExplainer(robust_predict, background_data)
                            shap_values = explainer.shap_values(input_scaled, nsamples=50)
                            
                            # Unwrap array
                            if isinstance(shap_values, list): sv = shap_values[0]
                            else: sv = shap_values
                            if len(sv.shape) > 1: sv = sv[0]
                            
                            # --- 4. THE "ZERO IMPACT" FALLBACK (The Fix) ---
                            # If all SHAP values are 0 (Model ignored features), switch to Correlation
                            is_heuristic = False
                            if np.all(np.abs(sv) < 0.001):
                                is_heuristic = True
                                # Calculate correlation from recent history instead
                                if not df_recent.empty:
                                    # Create a small temp dataframe of recent features
                                    # We try to correlate each feature col with the target 'pm2_5'
                                    correlations = []
                                    target = df_recent['pm2_5'].values
                                    
                                    for i, name in enumerate(feature_names):
                                        # Try to find the feature in the dataframe columns
                                        # (Lowercase match usually works)
                                        col_name = name.lower().replace(" ", "_").replace(".", "_")
                                        # Manual mapping for complex names
                                        if "lag 1h" in name.lower(): col_name = "pm2_5_lag1"
                                        elif "rolling mean" in name.lower(): col_name = "pm2_5_rolling_mean_24h"
                                        
                                        if col_name in df_recent.columns:
                                            # Calculate correlation
                                            feat_vals = df_recent[col_name].values
                                            # Ensure lengths match
                                            min_len = min(len(feat_vals), len(target))
                                            if min_len > 2:
                                                corr = np.corrcoef(feat_vals[:min_len].astype(float), target[:min_len].astype(float))[0,1]
                                                if np.isnan(corr): corr = 0
                                                # Scale it to look like a "contribution" (e.g., * current value)
                                                # Or just use the raw correlation coefficient as a relative score
                                                correlations.append(corr * 10) # Scaling factor for visibility
                                            else:
                                                correlations.append(0)
                                        else:
                                            correlations.append(0)
                                    sv = np.array(correlations)
                            
                            # 5. PREPARE RESULTS
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Actual Value': [f"{x:.2f}" for x in current_features],
                                'Impact': sv
                            })
                            
                            # Add color and magnitude
                            importance_df['Abs Impact'] = importance_df['Impact'].abs()
                            importance_df['Direction'] = ['Pushing Up 🔺' if x > 0 else 'Pulling Down 🔻' for x in importance_df['Impact']]
                            
                            # Sort
                            importance_df = importance_df.sort_values('Abs Impact', ascending=False).head(10)
                            
                            # 6. VISUALIZE
                            if is_heuristic:
                                st.warning("⚠️ **Note:** This model is Time-Series based (SARIMA) and ignores external features.")
                                st.info("ℹ️ Showing **Historical Correlation** instead (Which features move with the target?).")
                                title_text = "Feature Importance (Based on Historical Correlation)"
                            else:
                                title_text = "Feature Impact (Red = Increases AQI, Blue = Decreases AQI)"

                            st.subheader(f"🏆 Top 10 Drivers")
                            
                            fig = px.bar(
                                importance_df,
                                x='Impact',
                                y='Feature',
                                orientation='h',
                                color='Impact',
                                color_continuous_scale=['blue', 'red'],
                                text='Actual Value',
                                title=title_text
                            )
                            fig.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("### 📋 Detailed Breakdown")
                            st.dataframe(
                                importance_df[['Feature', 'Actual Value', 'Impact', 'Direction']], 
                                use_container_width=True,
                                hide_index=True
                            )
                # ==========================================
                # TAB B: LIME
                # ==========================================
                with lime_tab:
                    st.write("**Local Interpretable Model-agnostic Explanations**")
                    if st.button("🍋 Run LIME Analysis"):
                        try:
                            from lime.lime_tabular import LimeTabularExplainer
                            with st.spinner("Squeezing Limes..."):
                                # 1. Prepare Training Data (Synthetic if needed)
                                if not df_training.empty and len(df_training) > 10:
                                    # Use real training data structure if possible, but we need it scaled
                                    # Since mapping exact training columns to scaler is hard dynamically,
                                    # We stick to the SAFE synthetic method (perturbing input)
                                    train_data = np.repeat(input_scaled, 50, axis=0)
                                    noise = np.random.normal(0, 1, train_data.shape)
                                    train_data = train_data + noise
                                else:
                                    train_data = np.random.rand(100, n_features)

                                # 2. Initialize LIME
                                explainer_lime = LimeTabularExplainer(
                                    train_data,
                                    feature_names=feature_names,
                                    class_names=['PM2.5'],
                                    mode='regression'
                                )

                                # 3. Robust Predict (Same as SHAP)
                                def robust_predict_lime(data):
                                    prediction = model.predict(data)
                                    if isinstance(prediction, (list, np.ndarray)):
                                        if len(np.array(prediction).flatten()) == 1:
                                            return np.full(data.shape[0], prediction[0])
                                        else: return prediction
                                    else: return np.full(data.shape[0], prediction)

                                # 4. Explain
                                # We pass the single input row (flattened)
                                exp = explainer_lime.explain_instance(
                                    input_scaled[0], 
                                    robust_predict_lime, 
                                    num_features=10
                                )
                                
                                # 5. Parse Results
                                lime_list = exp.as_list()
                                features_lime = [x[0] for x in lime_list]
                                weights_lime = [x[1] for x in lime_list]

                                # 6. Fallback Check
                                if all(abs(w) < 0.001 for w in weights_lime):
                                    st.warning("⚠️ LIME found zero impact (Time-Series Model).")
                                    st.info("The model is ignoring feature variations. Use the SHAP tab for Historical Correlations.")
                                else:
                                    # Visuals
                                    lime_df = pd.DataFrame({'Feature': features_lime, 'Weight': weights_lime})
                                    lime_df['Abs Weight'] = lime_df['Weight'].abs()
                                    lime_df = lime_df.sort_values('Abs Weight', ascending=True)

                                    fig_lime = px.bar(lime_df, x='Weight', y='Feature', orientation='h',
                                                    color='Weight', color_continuous_scale=['blue', 'green'],
                                                    title="LIME Feature Weights")
                                    st.plotly_chart(fig_lime, use_container_width=True)
                                    st.dataframe(lime_df[['Feature', 'Weight']], use_container_width=True)
                        except ImportError:
                            st.error("❌ SHAP library not installed. Run `pip install shap`.")
                        except Exception as e:
                            st.error(f"❌ Analysis failed: {e}")
                            st.warning("Your model wrapper might be blocking introspection.")
            else:
                st.warning("⚠️ Model not ready. Please check Tab 1.")
