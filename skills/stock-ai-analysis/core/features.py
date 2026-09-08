import numpy as np
import pandas as pd

from core.data_loader import clean_ticker_name


def calc_obv_features(df, t_prefix):
    """OBV (On-Balance Volume) - 出来高ベースのモメンタム"""
    close = df["Close"]
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(0, index=df.index)

    # Calculate daily OBV changes
    obv_change = np.where(close > close.shift(1), volume, np.where(close < close.shift(1), -volume, 0))
    obv_change = pd.Series(obv_change, index=df.index)

    # OBV is cumulative by definition, but we also want to extract 5d and 20d sums for the features
    obv = obv_change.cumsum()

    feats = pd.DataFrame(index=df.index)
    feats[f"{t_prefix}_OBV_5d"] = obv_change.rolling(5, min_periods=1).sum().fillna(0.0)
    feats[f"{t_prefix}_OBV_20d"] = obv_change.rolling(20, min_periods=1).sum().fillna(0.0)
    feats[f"{t_prefix}_OBV_Signal_9d"] = obv.rolling(9, min_periods=1).mean().fillna(0.0)

    # Divergence: Correlation between Close and OBV over 20 days
    # (If correlation is negative, divergence is occurring)
    corr_20d = close.rolling(20).corr(obv).fillna(0.0)
    feats[f"{t_prefix}_OBV_Divergence"] = corr_20d

    return feats


def calc_vroc_features(df, t_prefix):
    """VROC (Volume Rate of Change) - 出来高の変化率"""
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(0, index=df.index)

    feats = pd.DataFrame(index=df.index)

    # VROC_t = ((Volume_t - Volume_{t-n}) / Volume_{t-n}) * 100
    # Avoid division by zero
    vroc_5d = ((volume - volume.shift(5)) / (volume.shift(5) + 1e-9)) * 100
    vroc_20d = ((volume - volume.shift(20)) / (volume.shift(20) + 1e-9)) * 100

    feats[f"{t_prefix}_VROC_5d"] = vroc_5d.fillna(0.0).round(3)
    feats[f"{t_prefix}_VROC_20d"] = vroc_20d.fillna(0.0).round(3)
    feats[f"{t_prefix}_VROC_MA3d"] = vroc_5d.rolling(3, min_periods=1).mean().fillna(0.0).round(3)

    return feats


def calc_bb_features(df, t_prefix, window=20, num_std=2.0):
    """Bollinger Bands (BB) - ボラティリティと過熱度"""
    close = df["Close"]

    ma = close.rolling(window, min_periods=1).mean()
    std = close.rolling(window, min_periods=1).std()

    # On the first day, std is NaN. Fill with 0
    std = std.fillna(0.0)

    upper_band = ma + (std * num_std)
    lower_band = ma - (std * num_std)

    feats = pd.DataFrame(index=df.index)
    feats[f"{t_prefix}_BB_Upper"] = upper_band.fillna(0.0)
    feats[f"{t_prefix}_BB_Lower"] = lower_band.fillna(0.0)

    # Width Pct = (UB - LB) / MA * 100
    width_pct = (upper_band - lower_band) / (ma + 1e-9) * 100
    feats[f"{t_prefix}_BB_Width_Pct"] = width_pct.fillna(0.0)

    # Position = (Close - LB) / (UB - LB) * 100
    # If UB == LB, position is 50
    band_diff = upper_band - lower_band
    position = np.where(band_diff > 1e-9, (close - lower_band) / band_diff * 100, 50.0)
    feats[f"{t_prefix}_BB_Position"] = pd.Series(position, index=df.index).fillna(50.0)

    return feats


def calc_macd_features(df, t_prefix, fast_period=12, slow_period=26, signal_period=9):
    """MACD (Moving Average Convergence Divergence)"""
    close = df["Close"]

    # EMA calculation using pandas ewm
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line

    feats = pd.DataFrame(index=df.index)
    feats[f"{t_prefix}_MACD"] = macd_line.fillna(0.0)
    feats[f"{t_prefix}_MACD_Signal"] = signal_line.fillna(0.0)
    feats[f"{t_prefix}_MACD_Histogram"] = macd_histogram.fillna(0.0)

    # Histogram Cross: sign change of histogram (1 if changed to positive, -1 if changed to negative, 0 otherwise)
    # Actually, the requirement says "ヒストグラムの符号変化（ゼロクロス：売買シグナル）"
    # We can represent this as 1 (cross above 0), -1 (cross below 0), 0 (no cross)
    prev_hist = macd_histogram.shift(1)

    cross = np.zeros(len(df))
    cross[(macd_histogram > 0) & (prev_hist <= 0)] = 1.0
    cross[(macd_histogram < 0) & (prev_hist >= 0)] = -1.0

    feats[f"{t_prefix}_MACD_Histogram_Cross"] = pd.Series(cross, index=df.index).fillna(0.0)

    return feats


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
    df_tnx=None,
    df_vix=None,
    df_sox=None,
    df_oil=None,
    df_gold=None,
):
    """
    個別株テクニカル、マクロ指標（S&P500/為替/日経/米10年債/VIX/SOX/原油/金）、ニュース感情スコア、ファンダメンタルズ財務、
    検索・アクセスボリューム(Investor Attention)からなる
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

    if df_vix is not None and not df_vix.empty:
        base_df = base_df.merge(
            df_vix[["Close"]].rename(columns={"Close": "VIX_Close"}), left_index=True, right_index=True, how="left"
        )
        base_df["VIX_Close"] = base_df["VIX_Close"].ffill().bfill().fillna(20.0)
    else:
        base_df["VIX_Close"] = 20.0

    if df_sox is not None and not df_sox.empty:
        base_df = base_df.merge(
            df_sox[["Close"]].rename(columns={"Close": "SOX_Close"}), left_index=True, right_index=True, how="left"
        )
        base_df["SOX_Close"] = base_df["SOX_Close"].ffill().bfill().fillna(0.0)
    else:
        base_df["SOX_Close"] = 0.0

    if df_oil is not None and not df_oil.empty:
        base_df = base_df.merge(
            df_oil[["Close"]].rename(columns={"Close": "Oil_Close"}), left_index=True, right_index=True, how="left"
        )
        base_df["Oil_Close"] = base_df["Oil_Close"].ffill().bfill().fillna(0.0)
    else:
        base_df["Oil_Close"] = 0.0

    if df_gold is not None and not df_gold.empty:
        base_df = base_df.merge(
            df_gold[["Close"]].rename(columns={"Close": "Gold_Close"}), left_index=True, right_index=True, how="left"
        )
        base_df["Gold_Close"] = base_df["Gold_Close"].ffill().bfill().fillna(0.0)
    else:
        base_df["Gold_Close"] = 0.0

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

    # ファンダメンタルズ財務データの前方補完 (ffill)
    base_df = base_df.merge(df_fund.set_index("Date"), left_index=True, right_index=True, how="left")
    fund_cols_to_fill = [
        "TTM_EPS",
        "Fund_Rev_Growth_YoY",
        "Fund_Net_Margin",
        "Fund_Operating_Margin",
        "Fund_Earnings_Surprise",
        "Fund_Book_Value",
        "Fund_ROE",
        "Fund_ROA",
        "Fund_PEG_Ratio",
        "Fund_Dividend_Yield",
        "Fund_Debt_to_Equity",
    ]
    for fc in fund_cols_to_fill:
        if fc in base_df.columns:
            base_df[fc] = base_df[fc].ffill().bfill()
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

    # --- 新規追加: High/Low を活用したテクニカル指標群 ---
    if "High" in df_stock.columns and "Low" in df_stock.columns:
        high = df_stock["High"]
        low = df_stock["Low"]
        close = df_stock["Close"]

        # 1. ATR (Average True Range)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        feats[f"{t_prefix}_ATR_14"] = atr_14.ffill().fillna(0.0)
        feats[f"{t_prefix}_ATR_14_Ratio"] = (feats[f"{t_prefix}_ATR_14"] / (close + 1e-9)).ffill().fillna(0.0)

        # 2. ボラティリティ圧縮度
        hl_range = (high - low) / (close + 1e-9)
        feats[f"{t_prefix}_HL_Range"] = hl_range.ffill().fillna(0.0)
        feats[f"{t_prefix}_HL_Range_MA20"] = hl_range.rolling(20).mean().ffill().fillna(0.0)
        feats[f"{t_prefix}_HL_Range_Compression"] = (
            (feats[f"{t_prefix}_HL_Range"] / (feats[f"{t_prefix}_HL_Range_MA20"] + 1e-9)).ffill().fillna(0.0)
        )

        # 3. サポート・レジスタンス
        feats[f"{t_prefix}_High20"] = high.rolling(20).max().ffill().fillna(0.0)
        feats[f"{t_prefix}_Low20"] = low.rolling(20).min().ffill().fillna(0.0)
        feats[f"{t_prefix}_Range20"] = (feats[f"{t_prefix}_High20"] - feats[f"{t_prefix}_Low20"]).ffill().fillna(0.0)

        # 4. Price Position (価格位置)
        # 過去20日の高値安値に対する現在値の位置 (0〜1)
        feats[f"{t_prefix}_Price_Position_20d"] = (
            ((close - feats[f"{t_prefix}_Low20"]) / (feats[f"{t_prefix}_Range20"] + 1e-9))
            .clip(0, 1)
            .ffill()
            .fillna(0.5)
        )

        # 単日の高値安値に対する終値の位置 (0〜1)
        day_range = high - low
        feats[f"{t_prefix}_Close_Position_in_Range"] = (
            ((close - low) / (day_range + 1e-9)).clip(0, 1).ffill().fillna(0.5)
        )

        # 高値への接近度
        feats[f"{t_prefix}_Close_to_High_Ratio"] = ((high - close) / (day_range + 1e-9)).clip(0, 1).ffill().fillna(0.5)

        # 5. Directional Movement (上昇日の比率)
        up_day = (close > close.shift(1)).astype(float)
        feats[f"{t_prefix}_Up_Down_Ratio_14d"] = up_day.rolling(14).mean().ffill().fillna(0.5)

        # 6. Stochastics (%K / %D)
        stoch_k = (close - low.rolling(14).min()) / (high.rolling(14).max() - low.rolling(14).min() + 1e-9) * 100
        feats[f"{t_prefix}_Stoch_K"] = stoch_k.ffill().fillna(50.0)
        feats[f"{t_prefix}_Stoch_D"] = stoch_k.rolling(3).mean().ffill().fillna(50.0)

        # 7. ADX (Average Directional Index)
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        pos_dm_ser = pd.Series(pos_dm, index=close.index)
        neg_dm_ser = pd.Series(neg_dm, index=close.index)

        # Smoothed True Range and Directional Movement (using Wilder's smoothing approx with exponential moving average)
        # Using 14-day exponential moving average as standard ADX calculation
        atr_ema = tr.ewm(alpha=1 / 14, adjust=False).mean()
        pos_dm_ema = pos_dm_ser.ewm(alpha=1 / 14, adjust=False).mean()
        neg_dm_ema = neg_dm_ser.ewm(alpha=1 / 14, adjust=False).mean()

        pos_di = 100 * (pos_dm_ema / (atr_ema + 1e-9))
        neg_di = 100 * (neg_dm_ema / (atr_ema + 1e-9))

        dx = 100 * (abs(pos_di - neg_di) / (pos_di + neg_di + 1e-9))
        adx = dx.ewm(alpha=1 / 14, adjust=False).mean()

        feats[f"{t_prefix}_ADX_14"] = adx.ffill().fillna(0.0)

        # 8. Keltner Channel Bandwidth
        ema_20 = close.ewm(span=20, adjust=False).mean()
        keltner_upper = ema_20 + 2 * atr_14
        keltner_lower = ema_20 - 2 * atr_14
        feats[f"{t_prefix}_Keltner_Bandwidth"] = ((keltner_upper - keltner_lower) / (ema_20 + 1e-9)).ffill().fillna(0.0)

    # Add new technical indicators (OBV, VROC, BB, MACD)
    df_technical = base_df[["Close", "Volume"]]

    obv_feats = calc_obv_features(df_technical, t_prefix)
    vroc_feats = calc_vroc_features(df_technical, t_prefix)
    bb_feats = calc_bb_features(df_technical, t_prefix)
    macd_feats = calc_macd_features(df_technical, t_prefix)

    feats = pd.concat([feats, obv_feats, vroc_feats, bb_feats, macd_feats], axis=1)

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

    # VIX恐怖指数 (市場パニック・リスクオン/オフ)
    vix_ma20 = base_df["VIX_Close"].rolling(20, min_periods=5).mean() + 1e-6
    feats["VIX_Return_1d"] = base_df["VIX_Close"].pct_change(1).fillna(0.0)
    feats["VIX_Return_5d"] = base_df["VIX_Close"].pct_change(5).fillna(0.0)
    feats["VIX_MA20_Ratio"] = ((base_df["VIX_Close"] - vix_ma20) / vix_ma20).fillna(0.0)
    feats["VIX_Over_25"] = (base_df["VIX_Close"] >= 25.0).astype(float)

    # SOX半導体指数 (テック・半導体サイクル)
    sox_ma20 = base_df["SOX_Close"].rolling(20, min_periods=5).mean() + 1e-6
    feats["SOX_Return_1d"] = base_df["SOX_Close"].pct_change(1).fillna(0.0)
    feats["SOX_Return_5d"] = base_df["SOX_Close"].pct_change(5).fillna(0.0)
    feats["SOX_MA20_Ratio"] = ((base_df["SOX_Close"] - sox_ma20) / sox_ma20).fillna(0.0)

    # WTI原油先物 (エネルギー・インフレ・原価コスト)
    oil_ma20 = base_df["Oil_Close"].rolling(20, min_periods=5).mean() + 1e-6
    feats["Oil_Return_1d"] = base_df["Oil_Close"].pct_change(1).fillna(0.0)
    feats["Oil_Return_5d"] = base_df["Oil_Close"].pct_change(5).fillna(0.0)
    feats["Oil_MA20_Ratio"] = ((base_df["Oil_Close"] - oil_ma20) / oil_ma20).fillna(0.0)

    # 金先物 (安全資産・世界情勢・通貨信認)
    gold_ma20 = base_df["Gold_Close"].rolling(20, min_periods=5).mean() + 1e-6
    feats["Gold_Return_1d"] = base_df["Gold_Close"].pct_change(1).fillna(0.0)
    feats["Gold_Return_5d"] = base_df["Gold_Close"].pct_change(5).fillna(0.0)
    feats["Gold_MA20_Ratio"] = ((base_df["Gold_Close"] - gold_ma20) / gold_ma20).fillna(0.0)

    # 金/原油レシオ (Gold-to-Oil Ratio: 地政学危機 & リセッション先行指標)
    gold_oil_ratio = base_df["Gold_Close"] / (base_df["Oil_Close"] + 1e-6)
    go_ma20 = gold_oil_ratio.rolling(20, min_periods=5).mean() + 1e-6
    feats["Gold_Oil_Ratio_MA20_Ratio"] = ((gold_oil_ratio - go_ma20) / go_ma20).fillna(0.0)

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

    feats["Fund_Employees"] = base_df["Fund_Employees"]

    # [6] ファンダメンタルズ財務 & バリュエーション
    # 1. PER 関連
    feats["Fund_Dynamic_PE"] = base_df["Close"] / (base_df["TTM_EPS"] + 1e-9)
    feats["Fund_Earnings_Yield"] = (base_df["TTM_EPS"] + 1e-9) / base_df["Close"]
    pe_ma200 = feats["Fund_Dynamic_PE"].rolling(200, min_periods=20).mean()
    pe_std200 = feats["Fund_Dynamic_PE"].rolling(200, min_periods=20).std() + 1e-9
    feats["Fund_PE_Ratio_to_MA200"] = ((feats["Fund_Dynamic_PE"] - pe_ma200) / (pe_ma200 + 1e-9)).fillna(0.0)
    feats["Fund_PE_ZScore"] = ((feats["Fund_Dynamic_PE"] - pe_ma200) / pe_std200).fillna(0.0)

    # 2. PBR 関連 (動的PBR、200日平均比乖離率、Zスコア)
    book_val = base_df["Fund_Book_Value"].fillna(10.0) if "Fund_Book_Value" in base_df.columns else 10.0
    feats["Fund_Dynamic_PBR"] = base_df["Close"] / (book_val + 1e-9)
    pbr_ma200 = feats["Fund_Dynamic_PBR"].rolling(200, min_periods=20).mean()
    pbr_std200 = feats["Fund_Dynamic_PBR"].rolling(200, min_periods=20).std() + 1e-9
    feats["Fund_PBR_Ratio_to_MA200"] = ((feats["Fund_Dynamic_PBR"] - pbr_ma200) / (pbr_ma200 + 1e-9)).fillna(0.0)
    feats["Fund_PBR_ZScore"] = ((feats["Fund_Dynamic_PBR"] - pbr_ma200) / pbr_std200).fillna(0.0)

    # 3. 収益性・成長性・クオリティ
    feats["Fund_Rev_Growth_YoY"] = base_df["Fund_Rev_Growth_YoY"].clip(-0.5, 1.0)
    feats["Fund_Net_Margin"] = base_df["Fund_Net_Margin"].clip(-0.5, 0.8)
    feats["Fund_Operating_Margin"] = base_df["Fund_Operating_Margin"].clip(-0.5, 0.8)
    feats["Fund_Earnings_Surprise"] = base_df["Fund_Earnings_Surprise"].clip(-0.5, 0.5)

    if "Fund_ROE" in base_df.columns:
        feats["Fund_ROE"] = base_df["Fund_ROE"].clip(-0.5, 1.5).fillna(0.10)
    if "Fund_ROA" in base_df.columns:
        feats["Fund_ROA"] = base_df["Fund_ROA"].clip(-0.3, 0.5).fillna(0.04)
    if "Fund_PEG_Ratio" in base_df.columns:
        feats["Fund_PEG_Ratio"] = base_df["Fund_PEG_Ratio"].clip(0.0, 10.0).fillna(1.5)
    if "Fund_Debt_to_Equity" in base_df.columns:
        feats["Fund_Debt_to_Equity"] = base_df["Fund_Debt_to_Equity"].clip(0.0, 5.0).fillna(0.8)
    if "Fund_Dividend_Yield" in base_df.columns:
        feats["Fund_Dividend_Yield"] = base_df["Fund_Dividend_Yield"].clip(0.0, 0.15).fillna(0.0)

    # 4. イールドスプレッド (株式益回り - 米10年債利回り)
    if "TNX_Close" in base_df.columns:
        tnx_yield = (base_df["TNX_Close"] / 100.0).fillna(0.04)
        feats["Fund_Yield_Spread"] = (feats["Fund_Earnings_Yield"] - tnx_yield).clip(-0.10, 0.20)

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

    # 株価水準（非定常）の直接リークを防ぐため、絶対値PE/PBR/益回り/従業員数は表示用に保持し、モデル学習は定常化指標を使用
    excluded_model_cols = {
        "Target",
        "Fund_Dynamic_PE",
        "Fund_Earnings_Yield",
        "Fund_Employees",
        "Fund_Dynamic_PBR",
        "Fund_Book_Value",
    }

    if "High" in df_stock.columns and "Low" in df_stock.columns:
        # High/Low関連の非定常変数（レベル変数）もモデル学習から除外
        excluded_model_cols.update({f"{t_prefix}_High20", f"{t_prefix}_Low20", f"{t_prefix}_Range20"})

    feature_cols = [c for c in clean_df.columns if c not in excluded_model_cols]
    return clean_df, latest_df, feature_cols
