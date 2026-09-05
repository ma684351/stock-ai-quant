import numpy as np
import pandas as pd
import pytest


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
        }
    )

    # 検索ボリューム (Attention Volume)
    attention_volume = pd.Series(
        np.random.randint(500, 10000, n_days).astype(float),
        index=dates,
        name="Attention_Volume",
    )

    return {
        "df_stock": df_stock,
        "df_sp500": df_sp500,
        "df_usdjpy": df_usdjpy,
        "df_nikkei": df_nikkei,
        "df_daily_sentiment": df_sentiment,
        "df_fund": df_fund,
        "df_attention": attention_volume,
    }
