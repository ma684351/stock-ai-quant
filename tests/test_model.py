from core.features import build_features_and_target
from core.model import predict_latest_signal, train_stock_model


def test_train_stock_model_and_signal_prediction(dummy_market_data):
    ticker = "TEST"
    df_stock = dummy_market_data["df_stock"]

    clean_df, latest_df, feature_cols = build_features_and_target(
        df_stock=df_stock,
        df_sp500=dummy_market_data["df_sp500"],
        df_usdjpy=dummy_market_data["df_usdjpy"],
        df_nikkei=dummy_market_data["df_nikkei"],
        df_daily_sentiment=dummy_market_data["df_daily_sentiment"],
        df_fund=dummy_market_data["df_fund"],
        ticker=ticker,
        target_horizon=20,
    )

    # 1. モデル学習と評価指標の生成
    model, metrics, best_thresh, df_imp = train_stock_model(clean_df, feature_cols, ticker=ticker, train_ratio=0.75)

    assert model is not None
    assert len(df_imp) == len(feature_cols)
    assert "accuracy" in metrics
    assert "auc" in metrics
    assert "cm" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.35 <= best_thresh <= 0.70

    # 2. 直近シグナル予測の検証
    latest_res = predict_latest_signal(
        model=model,
        df_latest=latest_df,
        df_stock=df_stock,
        ticker=ticker,
        feature_cols=feature_cols,
        threshold=best_thresh,
    )

    assert latest_res["ticker"] == ticker
    assert latest_res["decision"] in {"BUY", "HOLD", "SELL"}
    assert 0.0 <= latest_res["prob"] <= 1.0
    assert "price_guide" in latest_res
    assert latest_res["price_guide"]["type"] == latest_res["decision"]
