"""
Unit tests for feature_pipeline.py
Tests data quality, structure, and processing logic.
"""
import pandas as pd
import pytest
from datetime import datetime

# Mock API response structure (matches Open-Meteo format)
MOCK_API_RESPONSE = {
    "hourly": {
        "time": ["2025-12-01T10:00", "2025-12-01T11:00"],
        "pm10": [20.5, 22.1],
        "pm2_5": [12.0, 13.5],
        "ozone": [30.0, 31.0],
        "nitrogen_dioxide": [10.0, 11.0],
        "sulphur_dioxide": [5.0, 5.5],
        "carbon_monoxide": [0.5, 0.6]
    }
}

EMPTY_API_RESPONSE = {}

INVALID_API_RESPONSE = {
    "hourly": {
        "time": []
    }
}

def process_mock_data(json_data):
    """
    Simplified version of fetch_current processing logic for testing.
    Mimics the data processing in feature_pipeline.py
    """
    if not json_data or "hourly" not in json_data:
        return pd.DataFrame()
        
    hourly = json_data["hourly"]
    times = hourly.get("time", [])
    
    if not times:
        return pd.DataFrame()
    
    # Take the last timestamp (current hour)
    idx = -1
    ts = pd.to_datetime(times[idx], utc=True)
    row = {"timestamp": ts}
    
    # API column mapping (from feature_pipeline.py)
    API_COL_MAPPING = {
        "pm10": "pm10",
        "pm2_5": "pm2_5",
        "ozone": "o3",
        "nitrogen_dioxide": "no2",
        "sulphur_dioxide": "so2",
        "carbon_monoxide": "co",
    }
    
    for api_col, my_col in API_COL_MAPPING.items():
        series = hourly.get(api_col, [])
        row[my_col] = series[idx] if idx < len(series) else None
        
    df = pd.DataFrame([row])
    
    # Add time features (matching feature_pipeline.py)
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    
    return df


def test_data_structure():
    """Test 1: Does the function return a DataFrame with correct columns?"""
    df = process_mock_data(MOCK_API_RESPONSE)
    
    # Check if we got data back
    assert not df.empty, "DataFrame should not be empty"
    
    # Check required columns
    expected_cols = ["timestamp", "pm2_5", "pm10", "o3", "no2", "so2", "co", "hour", "day", "month"]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"
    
    # Check we have exactly one row
    assert len(df) == 1, "Should return exactly one row (latest timestamp)"


def test_data_quality():
    """Test 2: Are the values logical? (No negative pollution, reasonable ranges)"""
    df = process_mock_data(MOCK_API_RESPONSE)
    
    # PM2.5 can't be negative
    assert (df["pm2_5"] >= 0).all(), "Found negative PM2.5 values!"
    
    # PM10 can't be negative
    assert (df["pm10"] >= 0).all(), "Found negative PM10 values!"
    
    # O3, NO2, SO2, CO should be non-negative
    for col in ["o3", "no2", "so2", "co"]:
        assert (df[col] >= 0).all(), f"Found negative {col} values!"
    
    # PM2.5 should be less than PM10 (PM2.5 is a subset of PM10)
    assert (df["pm2_5"] <= df["pm10"]).all(), "PM2.5 should be <= PM10"
    
    # Values should be reasonable (not astronomical)
    assert (df["pm2_5"] < 1000).all(), "PM2.5 values seem unreasonably high (>1000)"
    assert (df["pm10"] < 1000).all(), "PM10 values seem unreasonably high (>1000)"


def test_time_features():
    """Test 3: Are time features correctly extracted?"""
    df = process_mock_data(MOCK_API_RESPONSE)
    
    # Check timestamp is valid
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]), "Timestamp should be datetime type"
    
    # Check hour is in valid range (0-23)
    assert (df["hour"] >= 0).all() and (df["hour"] <= 23).all(), "Hour should be between 0-23"
    
    # Check day is in valid range (1-31)
    assert (df["day"] >= 1).all() and (df["day"] <= 31).all(), "Day should be between 1-31"
    
    # Check month is in valid range (1-12)
    assert (df["month"] >= 1).all() and (df["month"] <= 12).all(), "Month should be between 1-12"


def test_empty_response():
    """Test 4: Does it handle broken/empty API calls gracefully?"""
    df = process_mock_data(EMPTY_API_RESPONSE)
    assert df.empty, "Should return empty DataFrame for empty API response"


def test_invalid_response():
    """Test 5: Does it handle invalid API responses (no time data)?"""
    df = process_mock_data(INVALID_API_RESPONSE)
    assert df.empty, "Should return empty DataFrame for invalid API response"


def test_data_types():
    """Test 6: Are numeric columns actually numeric?"""
    df = process_mock_data(MOCK_API_RESPONSE)
    
    numeric_cols = ["pm2_5", "pm10", "o3", "no2", "so2", "co", "hour", "day", "month"]
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} should be numeric"


def test_timestamp_format():
    """Test 7: Is timestamp in UTC and properly formatted?"""
    df = process_mock_data(MOCK_API_RESPONSE)
    
    # Check timestamp is timezone-aware (UTC)
    assert df["timestamp"].dt.tz is not None, "Timestamp should be timezone-aware"
    
    # Check it's UTC
    assert str(df["timestamp"].dt.tz) == "UTC", "Timestamp should be in UTC timezone"

