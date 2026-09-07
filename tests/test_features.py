from core.features import build_features_and_target


def test_build_features_and_target(dummy_market_data):
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
        df_tnx=dummy_market_data["df_tnx"],
        df_vix=dummy_market_data["df_vix"],
        df_sox=dummy_market_data["df_sox"],
        df_oil=dummy_market_data["df_oil"],
        df_gold=dummy_market_data["df_gold"],
    )

    # 1. 出力行数と特徴量カラムリストの検証
    assert len(clean_df) > 0
    assert len(latest_df) == 1
    assert len(feature_cols) > 0
    assert "Target" not in feature_cols

    # 2. Target カラムの検証
    assert "Target" in clean_df.columns
    assert "Target" not in latest_df.columns
    assert set(clean_df["Target"].unique()).issubset({0, 1})

    # 3. NaN の混入がないこと
    assert not clean_df.isna().any().any()
    assert not latest_df.isna().any().any()

    # 4. 4大カテゴリの特徴量が揃っていること
    # [テクニカル]
    assert f"{ticker}_Return_1d" in clean_df.columns
    assert f"{ticker}_MA20_Ratio" in clean_df.columns
    assert f"{ticker}_RSI_14" in clean_df.columns
    # [マクロ]
    assert "SP500_Return_1d" in clean_df.columns
    assert "USDJPY_Return_5d" in clean_df.columns
    assert "Nikkei_Return_1d" in clean_df.columns
    assert "TNX_Return_1d" in clean_df.columns
    assert "TNX_MA20_Ratio" in clean_df.columns
    assert "VIX_Return_1d" in clean_df.columns
    assert "VIX_Over_25" in clean_df.columns
    assert "SOX_Return_1d" in clean_df.columns
    assert "Oil_Return_1d" in clean_df.columns
    assert "Gold_Return_1d" in clean_df.columns
    assert "Gold_Oil_Ratio_MA20_Ratio" in clean_df.columns
    # [感情]
    assert "News_Sentiment_Score" in clean_df.columns
    assert "News_Sentiment_Surprise" in clean_df.columns
    assert "News_Sentiment_x_RSI" in clean_df.columns
    # [検索・アクセスボリューム (Attention)]
    assert "Attention_Surprise_20d" in clean_df.columns
    assert "Attention_ZScore_60d" in clean_df.columns
    assert "Attention_x_Sentiment" in clean_df.columns
    assert "Attention_x_RSI" in clean_df.columns
    # [ファンダメンタルズ]
    assert "Fund_Dynamic_PE" in clean_df.columns
    assert "Fund_PE_Ratio_to_MA200" in clean_df.columns
    assert "Fund_PE_ZScore" in clean_df.columns
    assert "Fund_Dynamic_PBR" in clean_df.columns
    assert "Fund_PBR_Ratio_to_MA200" in clean_df.columns
    assert "Fund_PBR_ZScore" in clean_df.columns
    assert "Fund_ROE" in clean_df.columns
    assert "Fund_ROA" in clean_df.columns
    assert "Fund_PEG_Ratio" in clean_df.columns
    assert "Fund_Dividend_Yield" in clean_df.columns
    assert "Fund_Debt_to_Equity" in clean_df.columns
    assert "Fund_Yield_Spread" in clean_df.columns

    # 非定常指標（水準データ）が学習用 feature_cols から安全に除外されていること
    assert "Fund_Dynamic_PE" not in feature_cols
    assert "Fund_Dynamic_PBR" not in feature_cols
    assert "Fund_Book_Value" not in feature_cols
    assert "Fund_PBR_Ratio_to_MA200" in feature_cols
    assert "Fund_Yield_Spread" in feature_cols


def test_build_features_without_attention(dummy_market_data):
    """df_attention=None の場合でもデフォルト補完で正常に特徴量が構築できること"""
    clean_df, latest_df, feature_cols = build_features_and_target(
        df_stock=dummy_market_data["df_stock"],
        df_sp500=dummy_market_data["df_sp500"],
        df_usdjpy=dummy_market_data["df_usdjpy"],
        df_nikkei=dummy_market_data["df_nikkei"],
        df_daily_sentiment=dummy_market_data["df_daily_sentiment"],
        df_fund=dummy_market_data["df_fund"],
        ticker="TEST",
        target_horizon=20,
        df_attention=None,
    )
    assert "Attention_Surprise_20d" in clean_df.columns
    assert not clean_df.isna().any().any()
