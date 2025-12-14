"""
Exploratory Data Analysis (EDA) for AQI Predictor
Analyzes trends, patterns, and relationships in the air quality data.
"""
import os
import certifi
import pandas as pd
import numpy as np
import hopsworks
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------
# 🔧 WINDOWS SSL FIX
# -------------------------------------------------
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

load_dotenv()

HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data():
    """Load data from Hopsworks Feature Store."""
    print("🚀 Connecting to Hopsworks...")
    try:
        project = hopsworks.login(project=HOPSWORKS_PROJECT, api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()
        
        # Get Feature Group
        # Get latest version
        all_fgs = fs.get_feature_group(name="aqi_features", version=None)
        if isinstance(all_fgs, list) and len(all_fgs) > 0:
            latest_version = max(fg.version for fg in all_fgs if hasattr(fg, 'version') and fg.version is not None)
            fg = fs.get_feature_group(name="aqi_features", version=latest_version)
        else:
            fg = fs.get_feature_group(name="aqi_features", version=3)
        
        # Read data
        print("📥 Downloading data...")
        df = fg.read(read_options={"use_hive": True})
        
        # Sort by timestamp
        df = df.sort_values("timestamp")
        
        print(f"✅ Loaded {len(df)} rows from {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def analyze_data_quality(df):
    """Analyze data quality: missing values, outliers, data types."""
    print("\n" + "="*70)
    print("📊 DATA QUALITY ANALYSIS")
    print("="*70)
    
    print(f"\n📏 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"📅 Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"⏱️  Time Span: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    
    print("\n🔍 Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("✅ No missing values!")
    
    print("\n📈 Data Types:")
    print(df.dtypes)
    
    print("\n📊 Basic Statistics:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'hour' in numeric_cols:
        numeric_cols.remove('hour')
        numeric_cols.remove('day')
        numeric_cols.remove('month')
    print(df[numeric_cols].describe())
    
    print("\n🔍 Outlier Detection (using IQR method):")
    for col in ['pm2_5', 'pm10', 'o3', 'no2']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            print(f"  {col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")

def analyze_temporal_trends(df):
    """Analyze temporal patterns: hourly, daily, monthly trends."""
    print("\n" + "="*70)
    print("⏰ TEMPORAL TREND ANALYSIS")
    print("="*70)
    
    # Set timestamp as index for easier time-based analysis
    df_time = df.set_index('timestamp').copy()
    
    # Hourly patterns
    print("\n🕐 Hourly Patterns:")
    hourly_avg = df.groupby('hour')[['pm2_5', 'pm10', 'o3']].mean()
    print(hourly_avg.round(2))
    
    # Daily patterns
    print("\n📅 Daily Patterns (by day of month):")
    daily_avg = df.groupby('day')[['pm2_5', 'pm10', 'o3']].mean()
    print(daily_avg.round(2))
    
    # Monthly patterns
    print("\n📆 Monthly Patterns:")
    monthly_avg = df.groupby('month')[['pm2_5', 'pm10', 'o3']].mean()
    print(monthly_avg.round(2))
    
    # Weekly patterns (day of week)
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['day_name'] = pd.to_datetime(df['timestamp']).dt.day_name()
    print("\n📆 Weekly Patterns (by day of week):")
    weekly_avg = df.groupby('day_name')[['pm2_5', 'pm10', 'o3']].mean()
    print(weekly_avg.round(2))
    
    return df

def analyze_correlations(df):
    """Analyze correlations between features."""
    print("\n" + "="*70)
    print("🔗 FEATURE CORRELATION ANALYSIS")
    print("="*70)
    
    # Select numeric columns for correlation
    numeric_cols = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co', 'hour', 'day', 'month']
    if 'aqi_change_rate' in df.columns:
        numeric_cols.append('aqi_change_rate')
    
    available_cols = [col for col in numeric_cols if col in df.columns]
    corr_matrix = df[available_cols].corr()
    
    print("\n📊 Correlation Matrix:")
    print(corr_matrix.round(3))
    
    # Find strongest correlations with PM2.5
    if 'pm2_5' in df.columns:
        print("\n🎯 Strongest Correlations with PM2.5:")
        pm25_corr = corr_matrix['pm2_5'].drop('pm2_5').sort_values(ascending=False)
        for feature, corr_value in pm25_corr.items():
            print(f"  {feature}: {corr_value:.3f}")
    
    return corr_matrix

def create_visualizations(df):
    """Create comprehensive visualizations."""
    print("\n" + "="*70)
    print("📈 CREATING VISUALIZATIONS")
    print("="*70)
    
    # Create output directory
    os.makedirs("eda_output", exist_ok=True)
    
    # 1. Time Series Plot - ALL POLLUTANTS
    print("\n1️⃣ Creating time series plot (ALL pollutants)...")
    # Get all available pollutants
    all_pollutants = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']
    available_pollutants = [p for p in all_pollutants if p in df.columns]
    
    # Create subplots - 3 rows, 2 columns for 6 pollutants
    rows = 3
    cols = 2
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[p.upper() + ' Over Time' for p in available_pollutants],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    for idx, poll in enumerate(available_pollutants):
        row = (idx // cols) + 1
        col = (idx % cols) + 1
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df[poll], 
                name=poll.upper(), 
                line=dict(color=colors[idx % len(colors)]),
                showlegend=False
            ),
            row=row, col=col
        )
        fig.update_yaxes(title_text=f"{poll.upper()} (µg/m³)", row=row, col=col)
        if row == rows:
            fig.update_xaxes(title_text="Time", row=row, col=col)
    
    fig.update_layout(height=900, title_text="Air Quality Time Series - All Pollutants", showlegend=False)
    fig.write_html("eda_output/time_series.html")
    print(f"   ✅ Saved: eda_output/time_series.html (included {len(available_pollutants)} pollutants)")
    
    # 2. Distribution Plots - ALL POLLUTANTS
    print("\n2️⃣ Creating distribution plots (ALL pollutants)...")
    all_pollutants = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']
    available_pollutants = [p for p in all_pollutants if p in df.columns]
    
    # Create subplots - 3 rows, 2 columns
    rows = 3
    cols = 2
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[p.upper() + ' Distribution' for p in available_pollutants],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    for idx, poll in enumerate(available_pollutants):
        row = (idx // cols) + 1
        col = (idx % cols) + 1
        fig.add_trace(
            go.Histogram(x=df[poll], name=poll, nbinsx=50, showlegend=False),
            row=row, col=col
        )
        fig.update_yaxes(title_text="Frequency", row=row, col=col)
        fig.update_xaxes(title_text=f"{poll.upper()} (µg/m³)", row=row, col=col)
    
    fig.update_layout(height=900, title_text="Pollutant Distributions - All Pollutants", showlegend=False)
    fig.write_html("eda_output/distributions.html")
    print(f"   ✅ Saved: eda_output/distributions.html (included {len(available_pollutants)} pollutants)")
    
    # 3. Hourly Patterns - ALL POLLUTANTS
    print("\n3️⃣ Creating hourly pattern plots (ALL pollutants)...")
    all_pollutants = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']
    available_pollutants = [p for p in all_pollutants if p in df.columns]
    hourly_avg = df.groupby('hour')[available_pollutants].mean().reset_index()
    
    fig = go.Figure()
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    for idx, col in enumerate(available_pollutants):
        fig.add_trace(go.Scatter(
            x=hourly_avg['hour'],
            y=hourly_avg[col],
            mode='lines+markers',
            name=col.upper(),
            line=dict(color=colors[idx % len(colors)], width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="Average Pollutant Levels by Hour of Day - All Pollutants",
        xaxis_title="Hour of Day",
        yaxis_title="Concentration (µg/m³)",
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.write_html("eda_output/hourly_patterns.html")
    print(f"   ✅ Saved: eda_output/hourly_patterns.html (included {len(available_pollutants)} pollutants)")
    
    # 4. Monthly Patterns - ALL POLLUTANTS
    print("\n4️⃣ Creating monthly pattern plots (ALL pollutants)...")
    all_pollutants = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']
    available_pollutants = [p for p in all_pollutants if p in df.columns]
    monthly_avg = df.groupby('month')[available_pollutants].mean().reset_index()
    
    fig = go.Figure()
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    for idx, col in enumerate(available_pollutants):
        fig.add_trace(go.Bar(
            x=monthly_avg['month'],
            y=monthly_avg[col],
            name=col.upper(),
            marker_color=colors[idx % len(colors)]
        ))
    
    fig.update_layout(
        title="Average Pollutant Levels by Month - All Pollutants",
        xaxis_title="Month",
        yaxis_title="Concentration (µg/m³)",
        barmode='group',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.write_html("eda_output/monthly_patterns.html")
    print(f"   ✅ Saved: eda_output/monthly_patterns.html (included {len(available_pollutants)} pollutants)")
    
    # 5. Correlation Heatmap
    print("\n5️⃣ Creating correlation heatmap...")
    numeric_cols = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co', 'hour', 'day', 'month']
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
    fig.write_html("eda_output/correlation_heatmap.html")
    print("   ✅ Saved: eda_output/correlation_heatmap.html")
    
    # 6. Box Plots for Outlier Detection - ALL POLLUTANTS
    print("\n6️⃣ Creating box plots (ALL pollutants)...")
    all_pollutants = ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']
    available_pollutants = [p for p in all_pollutants if p in df.columns]
    
    fig = go.Figure()
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    for idx, poll in enumerate(available_pollutants):
        fig.add_trace(go.Box(
            y=df[poll], 
            name=poll.upper(),
            marker_color=colors[idx % len(colors)]
        ))
    
    fig.update_layout(
        title="Pollutant Distribution Box Plots (Outlier Detection) - All Pollutants",
        yaxis_title="Concentration (µg/m³)",
        height=500,
        xaxis_title="Pollutant"
    )
    fig.write_html("eda_output/box_plots.html")
    print(f"   ✅ Saved: eda_output/box_plots.html (included {len(available_pollutants)} pollutants)")
    
    # 7. AQI Change Rate Analysis (if available)
    if 'aqi_change_rate' in df.columns:
        print("\n7️⃣ Creating AQI change rate analysis...")
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('AQI Change Rate Over Time', 'AQI Change Rate Distribution'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['aqi_change_rate'], name='Change Rate', 
                      line=dict(color='green')),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=df['aqi_change_rate'], name='Distribution', nbinsx=50),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Change Rate", row=2, col=1)
        fig.update_yaxes(title_text="Change Rate", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_layout(height=600, title_text="AQI Change Rate Analysis", showlegend=True)
        fig.write_html("eda_output/aqi_change_rate.html")
        print("   ✅ Saved: eda_output/aqi_change_rate.html")

def generate_summary_report(df):
    """Generate a summary report of findings."""
    print("\n" + "="*70)
    print("📋 EDA SUMMARY REPORT")
    print("="*70)
    
    report = []
    report.append("="*70)
    report.append("EXPLORATORY DATA ANALYSIS - SUMMARY REPORT")
    report.append("="*70)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\nDataset: {len(df)} rows, {df.shape[1]} columns")
    report.append(f"Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Key findings
    report.append("\n" + "-"*70)
    report.append("KEY FINDINGS")
    report.append("-"*70)
    
    # PM2.5 statistics
    if 'pm2_5' in df.columns:
        report.append(f"\n[STATS] PM2.5 Statistics:")
        report.append(f"  Mean: {df['pm2_5'].mean():.2f} ug/m3")
        report.append(f"  Median: {df['pm2_5'].median():.2f} ug/m3")
        report.append(f"  Std Dev: {df['pm2_5'].std():.2f} ug/m3")
        report.append(f"  Min: {df['pm2_5'].min():.2f} ug/m3")
        report.append(f"  Max: {df['pm2_5'].max():.2f} ug/m3")
    
    # Temporal patterns
    if 'hour' in df.columns:
        peak_hour = df.groupby('hour')['pm2_5'].mean().idxmax()
        report.append(f"\n[TEMPORAL] Temporal Patterns:")
        report.append(f"  Peak pollution hour: {peak_hour}:00")
    
    if 'month' in df.columns:
        peak_month = df.groupby('month')['pm2_5'].mean().idxmax()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        report.append(f"  Peak pollution month: {month_names[peak_month-1]}")
    
    # Correlations
    if 'pm2_5' in df.columns:
        numeric_cols = ['pm10', 'o3', 'no2', 'so2', 'co']
        available_cols = [col for col in numeric_cols if col in df.columns]
        if available_cols:
            corr_with_pm25 = df[['pm2_5'] + available_cols].corr()['pm2_5'].drop('pm2_5')
            strongest_corr = corr_with_pm25.abs().idxmax()
            report.append(f"\n[CORRELATION] Feature Correlations:")
            report.append(f"  Strongest correlation with PM2.5: {strongest_corr} ({corr_with_pm25[strongest_corr]:.3f})")
    
    # Data quality
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    report.append(f"\n[QUALITY] Data Quality:")
    report.append(f"  Missing data: {missing_pct:.2f}%")
    
    # Save report (use UTF-8 encoding to handle emoji characters on Windows)
    report_text = "\n".join(report)
    with open("eda_output/eda_summary_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print("\n".join(report))
    print(f"\n✅ Summary report saved to: eda_output/eda_summary_report.txt")

def main():
    """Main EDA execution function."""
    print("="*70)
    print("🔍 EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)
    
    # Load data
    df = load_data()
    if df is None or df.empty:
        print("❌ Cannot proceed without data. Please run backfill.py first.")
        return
    
    # Run analyses
    analyze_data_quality(df)
    df = analyze_temporal_trends(df)
    corr_matrix = analyze_correlations(df)
    create_visualizations(df)
    generate_summary_report(df)
    
    print("\n" + "="*70)
    print("✅ EDA COMPLETE!")
    print("="*70)
    print("\n📁 All visualizations and reports saved to: eda_output/")
    print("   - time_series.html")
    print("   - distributions.html")
    print("   - hourly_patterns.html")
    print("   - monthly_patterns.html")
    print("   - correlation_heatmap.html")
    print("   - box_plots.html")
    print("   - aqi_change_rate.html (if available)")
    print("   - eda_summary_report.txt")

if __name__ == "__main__":
    main()

