import pytest
import pandas as pd
import numpy as np

from core.features import (
    calc_obv_features,
    calc_vroc_features,
    calc_bb_features,
    calc_macd_features
)

@pytest.fixture
def mock_stock_data():
    dates = pd.date_range(start="2022-01-01", periods=100, freq="B")

    # Create deterministic data for predictable testing
    # A simple uptrend followed by a downtrend
    prices = np.concatenate([
        np.linspace(100, 150, 50),
        np.linspace(150, 100, 50)
    ])

    # Volume increasing then decreasing
    volumes = np.concatenate([
        np.linspace(1000, 5000, 50),
        np.linspace(5000, 1000, 50)
    ])

    df = pd.DataFrame({
        "Close": prices,
        "Volume": volumes
    }, index=dates)
    return df

def test_calc_obv_features(mock_stock_data):
    df = mock_stock_data
    feats = calc_obv_features(df, "AAPL")

    # Check if all columns are created
    expected_cols = [
        "AAPL_OBV_5d", "AAPL_OBV_20d",
        "AAPL_OBV_Signal_9d", "AAPL_OBV_Divergence"
    ]
    for col in expected_cols:
        assert col in feats.columns
        # Make sure no NaNs
        assert not feats[col].isna().any()

    # Basic logic check: during uptrend (first 50 days), OBV should be positive
    # OBV_5d at day 10 should be sum of 5 positive volumes
    assert feats["AAPL_OBV_5d"].iloc[10] > 0
    # during downtrend (last 50 days), OBV diffs should be negative
    assert feats["AAPL_OBV_5d"].iloc[80] < 0

def test_calc_vroc_features(mock_stock_data):
    df = mock_stock_data
    feats = calc_vroc_features(df, "AAPL")

    expected_cols = [
        "AAPL_VROC_5d", "AAPL_VROC_20d", "AAPL_VROC_MA3d"
    ]
    for col in expected_cols:
        assert col in feats.columns
        assert not feats[col].isna().any()

    # Logic check: VROC_5d should be 0 for the first 5 days since we fillna with 0
    assert feats["AAPL_VROC_5d"].iloc[0] == 0.0
    assert feats["AAPL_VROC_5d"].iloc[4] == 0.0
    # Day 5 should have a positive VROC because volume is increasing
    assert feats["AAPL_VROC_5d"].iloc[10] > 0.0

def test_calc_bb_features(mock_stock_data):
    df = mock_stock_data
    feats = calc_bb_features(df, "AAPL")

    expected_cols = [
        "AAPL_BB_Upper", "AAPL_BB_Lower",
        "AAPL_BB_Width_Pct", "AAPL_BB_Position"
    ]
    for col in expected_cols:
        assert col in feats.columns
        assert not feats[col].isna().any()

    # Logic check: Position is bounded and typically between 0 and 100,
    # but can exceed if price spikes outside bands.
    # First day std is 0, so UB=LB=Close, Position should be defaulted to 50
    assert feats["AAPL_BB_Position"].iloc[0] == 50.0

    # Upper band should always be >= Lower band
    assert (feats["AAPL_BB_Upper"] >= feats["AAPL_BB_Lower"]).all()

def test_calc_macd_features(mock_stock_data):
    df = mock_stock_data
    feats = calc_macd_features(df, "AAPL")

    expected_cols = [
        "AAPL_MACD", "AAPL_MACD_Signal",
        "AAPL_MACD_Histogram", "AAPL_MACD_Histogram_Cross"
    ]
    for col in expected_cols:
        assert col in feats.columns
        assert not feats[col].isna().any()

    # Cross should only be -1, 0, or 1
    assert set(feats["AAPL_MACD_Histogram_Cross"].unique()).issubset({-1.0, 0.0, 1.0})
