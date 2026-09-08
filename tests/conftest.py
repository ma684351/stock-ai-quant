import os
import shutil
import sys

# Add skills/stock-ai-analysis to sys.path so tests can import 'core'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'skills', 'stock-ai-analysis')))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session", autouse=True)
def ensure_data_dir():
    """テスト実行時に data ディレクトリが存在しない場合、data.example から安全に準備する"""
    if not os.path.exists("data") and os.path.exists("data.example"):
        shutil.copytree("data.example", "data")


@pytest.fixture
def dummy_market_data():
    """テスト用のダミー市場データ一式を生成するフィクスチャ"""
    np.random.seed(42)
    n_days = 120
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    # 株価データ (Close, Open, High, Low, Volume)
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_days)
    close_prices = base_price * np.cumprod(1 + returns)
    high_prices = close_prices * (1 + np.random.uniform(0.005, 0.02, n_days))
    low_prices = close_prices * (1 - np.random.uniform(0.005, 0.02, n_days))
    open_prices = close_prices * (1 + np.random.normal(0, 0.005, n_days))
    volumes = np.random.randint(1_000_000, 5_000_000, n_days)

    df_stock = pd.DataFrame(
        {
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Volume": volumes,
        },
        index=dates,
    )

    # S&P 500
    sp500_close = 5000.0 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n_days))
    df_sp500 = pd.DataFrame({"Close": sp500_close}, index=dates)

    # USD/JPY
    usdjpy_close = 150.0 * np.cumprod(1 + np.random.normal(0.0002, 0.005, n_days))
    df_usdjpy = pd.DataFrame({"Close": usdjpy_close}, index=dates)

    # Nikkei 225
    nikkei_close = 38000.0 * np.cumprod(1 + np.random.normal(0.0005, 0.012, n_days))
    df_nikkei = pd.DataFrame({"Close": nikkei_close}, index=dates)

    # 感情スコア
    df_sentiment = pd.DataFrame(
        {
            "Date": dates,
            "Sentiment_Score": np.random.uniform(-0.5, 0.8, n_days),
        }
    )

    # 財務データ（四半期ごとの数件）
    q_dates = dates[::30]
    df_fund = pd.DataFrame(
        {
            "Date": q_dates,
            "TTM_EPS": [5.0, 5.2, 5.5, 5.8][: len(q_dates)],
            "Fund_Rev_Growth_YoY": [0.12, 0.15, 0.14, 0.18][: len(q_dates)],
            "Fund_Net_Margin": [0.22, 0.23, 0.21, 0.24][: len(q_dates)],
            "Fund_Operating_Margin": [0.28, 0.30, 0.29, 0.31][: len(q_dates)],
            "Fund_Earnings_Surprise": [0.03, 0.05, -0.01, 0.04][: len(q_dates)],
            "Fund_Book_Value": [25.0, 26.0, 27.5, 29.0][: len(q_dates)],
            "Fund_ROE": [0.20, 0.22, 0.21, 0.24][: len(q_dates)],
            "Fund_ROA": [0.08, 0.09, 0.085, 0.095][: len(q_dates)],
            "Fund_PEG_Ratio": [1.5, 1.4, 1.6, 1.3][: len(q_dates)],
            "Fund_Dividend_Yield": [0.015, 0.016, 0.015, 0.017][: len(q_dates)],
            "Fund_Debt_to_Equity": [0.75, 0.70, 0.68, 0.65][: len(q_dates)],
        }
    )

    # 検索ボリューム (Attention Volume)
    attention_volume = pd.Series(
        np.random.randint(500, 10000, n_days).astype(float),
        index=dates,
        name="Attention_Volume",
    )

    # TNX (米10年債利回り)
    tnx_close = 4.0 + np.random.normal(0, 0.2, n_days)
    df_tnx = pd.DataFrame({"Close": tnx_close}, index=dates)

    # VIX (恐怖指数)
    vix_close = 18.0 + np.random.normal(0, 3.0, n_days)
    df_vix = pd.DataFrame({"Close": np.clip(vix_close, 10.0, 50.0)}, index=dates)

    # SOX (半導体株指数)
    sox_close = 4500.0 * np.cumprod(1 + np.random.normal(0.0008, 0.02, n_days))
    df_sox = pd.DataFrame({"Close": sox_close}, index=dates)

    # Oil (WTI原油先物)
    oil_close = 75.0 * np.cumprod(1 + np.random.normal(0.0003, 0.015, n_days))
    df_oil = pd.DataFrame({"Close": oil_close}, index=dates)

    # Gold (金先物)
    gold_close = 2500.0 * np.cumprod(1 + np.random.normal(0.0004, 0.01, n_days))
    df_gold = pd.DataFrame({"Close": gold_close}, index=dates)

    return {
        "df_stock": df_stock,
        "df_sp500": df_sp500,
        "df_usdjpy": df_usdjpy,
        "df_nikkei": df_nikkei,
        "df_daily_sentiment": df_sentiment,
        "df_fund": df_fund,
        "df_attention": attention_volume,
        "df_tnx": df_tnx,
        "df_vix": df_vix,
        "df_sox": df_sox,
        "df_oil": df_oil,
        "df_gold": df_gold,
    }
