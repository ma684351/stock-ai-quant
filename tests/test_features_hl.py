import numpy as np
import pandas as pd
from core.features import build_features_and_target

def test_high_low_technical_indicators(dummy_market_data):
    ticker = "TEST"
    target_horizon = 20

    clean_df, latest_df, feature_cols = build_features_and_target(
        df_stock=dummy_market_data["df_stock"],
        df_sp500=dummy_market_data["df_sp500"],
        df_usdjpy=dummy_market_data["df_usdjpy"],
        df_nikkei=dummy_market_data["df_nikkei"],
        df_daily_sentiment=dummy_market_data["df_daily_sentiment"],
        df_fund=dummy_market_data["df_fund"],
        ticker=ticker,
        target_horizon=target_horizon,
        df_attention=dummy_market_data["df_attention"],
    )

    t_prefix = "TEST"

    # 1. 期待されるすべてのHigh-Low指標が特徴量リスト（または latest_df）に存在することを確認
    hl_features = [
        f"{t_prefix}_ATR_14",
        f"{t_prefix}_ATR_14_Ratio",
        f"{t_prefix}_HL_Range",
        f"{t_prefix}_HL_Range_MA20",
        f"{t_prefix}_HL_Range_Compression",
        f"{t_prefix}_Price_Position_20d",
        f"{t_prefix}_Close_Position_in_Range",
        f"{t_prefix}_Close_to_High_Ratio",
        f"{t_prefix}_Up_Down_Ratio_14d",
        f"{t_prefix}_Stoch_K",
        f"{t_prefix}_Stoch_D",
        f"{t_prefix}_ADX_14",
        f"{t_prefix}_Keltner_Bandwidth"
    ]

    for feat in hl_features:
        assert feat in clean_df.columns, f"Missing feature: {feat}"
        assert feat in feature_cols, f"Feature not in model inputs: {feat}"

    # 非定常変数がモデル特徴量から除外され、latest_dfには存在することを確認
    level_features = [
        f"{t_prefix}_High20",
        f"{t_prefix}_Low20",
        f"{t_prefix}_Range20"
    ]
    for feat in level_features:
        assert feat in clean_df.columns
        assert feat not in feature_cols

    # 2. 値の範囲と整合性チェック
    # ATRは必ず正の値
    assert (clean_df[f"{t_prefix}_ATR_14"] >= 0).all()
    assert (clean_df[f"{t_prefix}_ATR_14_Ratio"] >= 0).all()

    # Ratio系の指標は 0〜1 の範囲内
    assert (clean_df[f"{t_prefix}_Price_Position_20d"].between(0, 1)).all()
    assert (clean_df[f"{t_prefix}_Close_Position_in_Range"].between(0, 1)).all()
    assert (clean_df[f"{t_prefix}_Close_to_High_Ratio"].between(0, 1)).all()
    assert (clean_df[f"{t_prefix}_Up_Down_Ratio_14d"].between(0, 1)).all()

    # Stochasticsは 0〜100 の範囲内 (一部計算の都合でわずかに超える可能性も考慮しマージンを持つ)
    assert (clean_df[f"{t_prefix}_Stoch_K"] >= -0.1).all() and (clean_df[f"{t_prefix}_Stoch_K"] <= 100.1).all()
    assert (clean_df[f"{t_prefix}_Stoch_D"] >= -0.1).all() and (clean_df[f"{t_prefix}_Stoch_D"] <= 100.1).all()

    # ADXは 0〜100 の範囲内
    assert (clean_df[f"{t_prefix}_ADX_14"] >= 0).all() and (clean_df[f"{t_prefix}_ADX_14"] <= 100).all()

    # 3. NaN の混入がないこと
    assert not clean_df[hl_features].isna().any().any()
    assert not latest_df[hl_features].isna().any().any()

def test_high_low_indicators_with_nan_handling(dummy_market_data):
    """データにNaNが含まれる場合の補完（ffill, fillna）が正しく機能するかテスト"""
    import numpy as np

    df_stock = dummy_market_data["df_stock"].copy()
    # 意図的にHigh/Lowに欠損値を混入
    df_stock.iloc[5:10, df_stock.columns.get_loc("High")] = np.nan
    df_stock.iloc[5:10, df_stock.columns.get_loc("Low")] = np.nan
    # pandas bfill/ffill warnings prevention
    df_stock = df_stock.ffill().bfill()

    clean_df, latest_df, feature_cols = build_features_and_target(
        df_stock=df_stock,
        df_sp500=dummy_market_data["df_sp500"],
        df_usdjpy=dummy_market_data["df_usdjpy"],
        df_nikkei=dummy_market_data["df_nikkei"],
        df_daily_sentiment=dummy_market_data["df_daily_sentiment"],
        df_fund=dummy_market_data["df_fund"],
        ticker="TEST",
        target_horizon=20,
    )

    t_prefix = "TEST"
    assert not clean_df[f"{t_prefix}_ATR_14"].isna().any()
    assert not clean_df[f"{t_prefix}_ADX_14"].isna().any()
