from core.features import build_features_and_target
from core.model import optimize_training_period, predict_latest_signal, train_stock_model


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


def test_train_stock_model_with_purging(dummy_market_data):
    import numpy as np
    import pandas as pd

    # 合成データでパージングの動作を確認
    n_samples = 120
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="B")
    feature_cols = ["Feat1", "Feat2"]
    df = pd.DataFrame(
        {
            "Feat1": np.random.randn(n_samples),
            "Feat2": np.random.randn(n_samples),
            "Target": np.random.choice([0, 1], size=n_samples),
        },
        index=dates,
    )

    model, metrics, thresh, df_imp = train_stock_model(
        df, feature_cols, ticker="TEST_PURGE", train_ratio=0.8, target_horizon=15
    )
    assert model is not None
    assert metrics["test_count"] == n_samples - int(n_samples * 0.8)


def test_optimize_training_period():
    import numpy as np
    import pandas as pd

    # 合成データ（600日分）を用意し、1.5y (375日) と 2y (500日) の比較検証
    n_samples = 600
    dates = pd.date_range("2022-01-01", periods=n_samples, freq="B")
    np.random.seed(42)
    feature_cols = ["Feat1", "Feat2", "Feat3"]
    df = pd.DataFrame(
        {
            "Feat1": np.random.randn(n_samples),
            "Feat2": np.random.randn(n_samples),
            "Feat3": np.random.randn(n_samples),
            "Target": np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6]),
        },
        index=dates,
    )

    best_period, model, metrics, best_thresh, df_imp, results = optimize_training_period(
        df,
        feature_cols,
        ticker="TEST_AUTO",
        candidate_periods=("1.5y", "2y", "3y"),
        train_ratio=0.8,
        verbose=True,
    )

    assert best_period in {"1.5y", "2y", "3y"}
    assert model is not None
    assert "auc" in metrics
    assert "precision" in metrics
    assert len(results) >= 2
    assert all("score" in r for r in results)
