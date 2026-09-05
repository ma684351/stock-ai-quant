import numpy as np
import pandas as pd

from core.data_loader import clean_ticker_name


def build_features_and_target(
    df_stock,
    df_sp500,
    df_usdjpy,
    df_nikkei,
    df_daily_sentiment,
    df_fund,
    ticker="AAPL",
    target_horizon=20,
    df_attention=None,
    df_jobs=None,
    df_tnx=None,
):
    """
    個別株テクニカル、マクロ指標（S&P500/為替/日経/米10年債利回り）、ニュース感情スコア、ファンダメンタルズ財務、
    検索・アクセスボリューム(Investor Attention)、および求人数(Hiring Data)からなる
    特徴量を構築し、20営業日後（約1ヶ月後）の正解ラベルを生成する
    """
    t_prefix = clean_ticker_name(ticker)
    base_df = pd.DataFrame(index=df_stock.index)
    base_df["Close"] = df_stock["Close"]
    base_df["Volume"] = df_stock["Volume"] if "Volume" in df_stock.columns else 0

    # マクロ指標のマージ
    base_df = base_df.merge(
        df_sp500[["Close"]].rename(columns={"Close": "SP500_Close"}), left_index=True, right_index=True, how="left"
    )
    base_df = base_df.merge(
        df_usdjpy[["Close"]].rename(columns={"Close": "USDJPY_Close"}), left_index=True, right_index=True, how="left"
    )
    base_df = base_df.merge(
        df_nikkei[["Close"]].rename(columns={"Close": "Nikkei_Close"}), left_index=True, right_index=True, how="left"
    )
    if df_tnx is not None and not df_tnx.empty:
        base_df = base_df.merge(
            df_tnx[["Close"]].rename(columns={"Close": "TNX_Close"}), left_index=True, right_index=True, how="left"
        )
        base_df["TNX_Close"] = base_df["TNX_Close"].ffill().bfill().fillna(0.0)
    else:
        base_df["TNX_Close"] = 0.0

    base_df["SP500_Close"] = base_df["SP500_Close"].ffill()
    base_df["USDJPY_Close"] = base_df["USDJPY_Close"].ffill()
    base_df["Nikkei_Close"] = base_df["Nikkei_Close"].ffill()

    # ニュース感情スコアのマージ
    base_df = base_df.merge(df_daily_sentiment.set_index("Date"), left_index=True, right_index=True, how="left")
    base_df["Sentiment_Score"] = base_df["Sentiment_Score"].fillna(0.0)

    # 検索ボリューム（Attention）のマージ
    if df_attention is not None and not df_attention.empty:
        base_df["Attention_Volume"] = df_attention.reindex(base_df.index).ffill().bfill().fillna(0.0)
    else:
        base_df["Attention_Volume"] = 0.0

    # 求人数（Hiring Data）のマージ
    if df_jobs is not None and not df_jobs.empty:
        base_df["Job_Openings"] = df_jobs.reindex(base_df.index).ffill().bfill().fillna(0.0)
    else:
        base_df["Job_Openings"] = 0.0

    # ファンダメンタルズ財務データの前方補完 (ffill)
    base_df = base_df.merge(df_fund.set_index("Date"), left_index=True, right_index=True, how="left")
    base_df["TTM_EPS"] = base_df["TTM_EPS"].ffill().bfill()
    base_df["Fund_Rev_Growth_YoY"] = base_df["Fund_Rev_Growth_YoY"].ffill().bfill()
    base_df["Fund_Net_Margin"] = base_df["Fund_Net_Margin"].ffill().bfill()
    base_df["Fund_Operating_Margin"] = base_df["Fund_Operating_Margin"].ffill().bfill()
    base_df["Fund_Earnings_Surprise"] = base_df["Fund_Earnings_Surprise"].ffill().bfill()
    if "Fund_Employees" in base_df.columns:
        base_df["Fund_Employees"] = base_df["Fund_Employees"].ffill().bfill().fillna(1000.0)
    else:
        base_df["Fund_Employees"] = 1000.0

    feats = pd.DataFrame(index=base_df.index)

    # [1] 個別株テクニカル
    feats[f"{t_prefix}_Return_1d"] = base_df["Close"].pct_change(1)
    feats[f"{t_prefix}_Return_5d"] = base_df["Close"].pct_change(5)
    feats[f"{t_prefix}_Volume_Change_1d"] = base_df["Volume"].pct_change(1)
    feats[f"{t_prefix}_MA5_Ratio"] = base_df["Close"] / base_df["Close"].rolling(5).mean() - 1.0
    feats[f"{t_prefix}_MA20_Ratio"] = base_df["Close"] / base_df["Close"].rolling(20).mean() - 1.0
    feats[f"{t_prefix}_Volatility_20d"] = feats[f"{t_prefix}_Return_1d"].rolling(20).std() * np.sqrt(252)

    delta = base_df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    feats[f"{t_prefix}_RSI_14"] = 100 - (100 / (1 + rs))

    # [2] マクロ指標
    feats["SP500_Return_1d"] = base_df["SP500_Close"].pct_change(1)
    feats["SP500_Return_5d"] = base_df["SP500_Close"].pct_change(5)
    feats["USDJPY_Return_1d"] = base_df["USDJPY_Close"].pct_change(1)
    feats["USDJPY_Return_5d"] = base_df["USDJPY_Close"].pct_change(5)
    feats["Nikkei_Return_1d"] = base_df["Nikkei_Close"].pct_change(1)
    feats["Nikkei_Return_5d"] = base_df["Nikkei_Close"].pct_change(5)
    feats["TNX_Return_1d"] = base_df["TNX_Close"].pct_change(1).fillna(0.0)
    feats["TNX_Return_5d"] = base_df["TNX_Close"].pct_change(5).fillna(0.0)
    tnx_ma20 = base_df["TNX_Close"].rolling(20, min_periods=5).mean() + 1e-6
    feats["TNX_MA20_Ratio"] = ((base_df["TNX_Close"] - tnx_ma20) / tnx_ma20).fillna(0.0)

    # [3] ニュース感情スコア
    feats["News_Sentiment_Score"] = base_df["Sentiment_Score"]
    feats["News_Sentiment_MA3d"] = base_df["Sentiment_Score"].rolling(3).mean()
    feats["News_Sentiment_Lag1"] = base_df["Sentiment_Score"].shift(1)

    # [4] 感情 × テクニカル複合
    feats["News_Sentiment_x_RSI"] = base_df["Sentiment_Score"] * ((feats[f"{t_prefix}_RSI_14"] - 50.0) / 25.0)
    feats["News_Sentiment_x_Return5d"] = base_df["Sentiment_Score"] * feats[f"{t_prefix}_Return_5d"]
    feats["News_Sentiment_Surprise"] = base_df["Sentiment_Score"] - feats["News_Sentiment_MA3d"]

    # [5] 検索・アクセスボリューム (Investor Attention)
    att_vol = base_df["Attention_Volume"]
    att_ma20 = att_vol.rolling(20, min_periods=5).mean() + 1.0
    att_ma60 = att_vol.rolling(60, min_periods=10).mean()
    att_std60 = att_vol.rolling(60, min_periods=10).std() + 1.0

    feats["Attention_Surprise_20d"] = ((att_vol - att_ma20) / att_ma20).fillna(0.0)
    feats["Attention_ZScore_60d"] = ((att_vol - att_ma60) / att_std60).fillna(0.0)
    feats["Attention_x_Sentiment"] = feats["Attention_Surprise_20d"] * base_df["Sentiment_Score"]
    feats["Attention_x_RSI"] = feats["Attention_ZScore_60d"] * ((feats[f"{t_prefix}_RSI_14"] - 50.0) / 25.0)

    # [6] 求人数・採用モメンタム (Hiring / Alternative Data)
    job_cnt = base_df["Job_Openings"]
    feats["Job_Openings_Count"] = job_cnt
    vol_ma20 = base_df["Volume"].rolling(20, min_periods=5).mean() + 1.0
    feats["Job_to_Volume_Ratio"] = (job_cnt / (vol_ma20 / 10000.0 + 1.0)).fillna(0.0)
    feats["Job_x_Rev_Growth"] = job_cnt * base_df["Fund_Rev_Growth_YoY"]
    feats["Job_x_RSI"] = (job_cnt / (job_cnt.rolling(60, min_periods=5).mean() + 1.0)) * (
        (feats[f"{t_prefix}_RSI_14"] - 50.0) / 25.0
    )
    # 従業員数に対する求人比率（組織拡大ペース %）
    emp_cnt = base_df["Fund_Employees"].clip(lower=10.0)
    feats["Job_to_Employee_Ratio"] = ((job_cnt / emp_cnt) * 100.0).clip(0.0, 50.0).fillna(0.0)
    feats["Fund_Employees"] = base_df["Fund_Employees"]

    # [7] ファンダメンタルズ財務
    feats["Fund_Dynamic_PE"] = base_df["Close"] / (base_df["TTM_EPS"] + 1e-9)
    feats["Fund_Earnings_Yield"] = (base_df["TTM_EPS"] + 1e-9) / base_df["Close"]
    pe_ma200 = feats["Fund_Dynamic_PE"].rolling(200, min_periods=20).mean()
    pe_std200 = feats["Fund_Dynamic_PE"].rolling(200, min_periods=20).std() + 1e-9
    feats["Fund_PE_Ratio_to_MA200"] = ((feats["Fund_Dynamic_PE"] - pe_ma200) / (pe_ma200 + 1e-9)).fillna(0.0)
    feats["Fund_PE_ZScore"] = ((feats["Fund_Dynamic_PE"] - pe_ma200) / pe_std200).fillna(0.0)
    feats["Fund_Rev_Growth_YoY"] = base_df["Fund_Rev_Growth_YoY"].clip(-0.5, 1.0)
    feats["Fund_Net_Margin"] = base_df["Fund_Net_Margin"].clip(-0.5, 0.8)
    feats["Fund_Earnings_Surprise"] = base_df["Fund_Earnings_Surprise"].clip(-0.5, 0.5)

    # [7] 正解ラベル (20営業日後 / 約1ヶ月後の終値 > 当日終値 なら 1, それ以外 0)
    feats["Target"] = (base_df["Close"].shift(-target_horizon) > base_df["Close"]).astype(int)

    # 直近（最新営業日）の推論用データ（直近日の行を確実に保持し、欠損値は直前行や0で安全に補完）
    feat_candidates = feats.drop(columns=["Target"])
    latest_row = feat_candidates.iloc[[-1]].copy()
    if latest_row.isna().any().any():
        latest_row = latest_row.combine_first(feat_candidates.ffill().iloc[[-1]]).fillna(0.0)
    latest_df = latest_row

    # 学習・評価用データ（未来のターゲットが確定している期間）
    clean_df = feats.iloc[:-target_horizon].dropna()

    # 株価水準（非定常）の直接リークを防ぐため、絶対値PE/益回り/従業員数は表示用に保持し、モデル学習は定常化指標を使用
    excluded_model_cols = {"Target", "Fund_Dynamic_PE", "Fund_Earnings_Yield", "Fund_Employees"}

    # 求人データの履歴十分性チェック:
    # 過去の学習期間において求人数の非ゼロ観測日が5日未満の場合、定数化による疑似相関（出来高逆数化）を防ぐためモデル学習から除外
    job_obs_count = int((clean_df["Job_Openings_Count"] > 0).sum()) if "Job_Openings_Count" in clean_df.columns else 0
    if job_obs_count < 5:
        excluded_model_cols.update({
            "Job_Openings_Count",
            "Job_to_Volume_Ratio",
            "Job_x_Rev_Growth",
            "Job_x_RSI",
            "Job_to_Employee_Ratio",
        })

    feature_cols = [c for c in clean_df.columns if c not in excluded_model_cols]
    return clean_df, latest_df, feature_cols
