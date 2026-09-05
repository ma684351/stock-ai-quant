import pandas as pd

from core.model import calculate_trade_price_guide


def test_calculate_trade_price_guide_buy(dummy_market_data):
    df_stock = dummy_market_data["df_stock"]
    latest_close = float(df_stock["Close"].iloc[-1])
    ticker = "TEST"
    df_latest = pd.DataFrame({f"{ticker}_Volatility_20d": [0.25]})

    guide = calculate_trade_price_guide(
        df_stock=df_stock,
        latest_close=latest_close,
        decision="BUY",
        prob=0.68,
        t_prefix=ticker,
        df_latest=df_latest,
    )

    assert guide["type"] == "BUY"
    assert "entry_range" in guide
    buy_low, buy_high = guide["entry_range"]
    assert buy_low <= buy_high <= latest_close + 1e-5
    assert guide["target_price"] > latest_close
    assert guide["target_return"] > 0
    assert guide["stop_loss"] < latest_close
    assert guide["loss_pct"] < 0
    assert guide["rr_ratio"] > 0


def test_calculate_trade_price_guide_sell(dummy_market_data):
    df_stock = dummy_market_data["df_stock"]
    latest_close = float(df_stock["Close"].iloc[-1])
    ticker = "TEST"
    df_latest = pd.DataFrame({f"{ticker}_Volatility_20d": [0.25]})

    guide = calculate_trade_price_guide(
        df_stock=df_stock,
        latest_close=latest_close,
        decision="SELL",
        prob=0.30,
        t_prefix=ticker,
        df_latest=df_latest,
    )

    assert guide["type"] == "SELL"
    assert "entry_range" in guide
    sell_low, sell_high = guide["entry_range"]
    assert sell_high >= sell_low >= latest_close - 1e-5
    assert guide["target_price"] < latest_close
    assert guide["target_return"] < 0
    assert guide["stop_loss"] > latest_close
    assert guide["loss_pct"] > 0
    assert guide["rr_ratio"] > 0


def test_calculate_trade_price_guide_hold(dummy_market_data):
    df_stock = dummy_market_data["df_stock"]
    latest_close = float(df_stock["Close"].iloc[-1])
    ticker = "TEST"
    df_latest = pd.DataFrame({f"{ticker}_Volatility_20d": [0.25]})

    guide = calculate_trade_price_guide(
        df_stock=df_stock,
        latest_close=latest_close,
        decision="HOLD",
        prob=0.51,
        t_prefix=ticker,
        df_latest=df_latest,
    )

    assert guide["type"] == "HOLD"
    assert guide["dip_buy_price"] < latest_close
    assert guide["dip_return"] < 0
    assert guide["breakout_price"] > latest_close
    assert guide["breakout_return"] > 0
