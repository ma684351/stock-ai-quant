import pandas as pd
import numpy as np

def build_features_and_target(df_stock, df_sp500, df_usdjpy, df_nikkei, df_daily_sentiment, df_fund, ticker="AAPL", target_horizon=20):
    """
    個別株テクニカル、マクロ指標、ニュース感情スコア、ファンダメンタルズ財務の
    4本柱からなる特徴量を構築し、20営業日後（約1ヶ月後）の正解ラベルを生成する
    """
    base_df = pd.DataFrame(index=df_stock.index)
    base_df['Close'] = df_stock['Close']
    base_df['Volume'] = df_stock['Volume'] if 'Volume' in df_stock.columns else 0
    
    # マクロ指標のマージ
    base_df = base_df.merge(df_sp500[['Close']].rename(columns={'Close': 'SP500_Close'}), left_index=True, right_index=True, how='left')
    base_df = base_df.merge(df_usdjpy[['Close']].rename(columns={'Close': 'USDJPY_Close'}), left_index=True, right_index=True, how='left')
    base_df = base_df.merge(df_nikkei[['Close']].rename(columns={'Close': 'Nikkei_Close'}), left_index=True, right_index=True, how='left')
    base_df['SP500_Close'] = base_df['SP500_Close'].ffill()
    base_df['USDJPY_Close'] = base_df['USDJPY_Close'].ffill()
    base_df['Nikkei_Close'] = base_df['Nikkei_Close'].ffill()
    
    # ニュース感情スコアのマージ
    base_df = base_df.merge(df_daily_sentiment.set_index('Date'), left_index=True, right_index=True, how='left')
    base_df['Sentiment_Score'] = base_df['Sentiment_Score'].fillna(0.0)
    
    # ファンダメンタルズ財務データの前方補完 (ffill)
    base_df = base_df.merge(df_fund.set_index('Date'), left_index=True, right_index=True, how='left')
    base_df['TTM_EPS'] = base_df['TTM_EPS'].ffill().bfill()
    base_df['Fund_Rev_Growth_YoY'] = base_df['Fund_Rev_Growth_YoY'].ffill().bfill()
    base_df['Fund_Net_Margin'] = base_df['Fund_Net_Margin'].ffill().bfill()
    base_df['Fund_Operating_Margin'] = base_df['Fund_Operating_Margin'].ffill().bfill()
    base_df['Fund_Earnings_Surprise'] = base_df['Fund_Earnings_Surprise'].ffill().bfill()
    
    feats = pd.DataFrame(index=base_df.index)
    
    # [1] 個別株テクニカル
    feats[f'{ticker}_Return_1d'] = base_df['Close'].pct_change(1)
    feats[f'{ticker}_Return_5d'] = base_df['Close'].pct_change(5)
    feats[f'{ticker}_Volume_Change_1d'] = base_df['Volume'].pct_change(1)
    feats[f'{ticker}_MA5_Ratio'] = base_df['Close'] / base_df['Close'].rolling(5).mean() - 1.0
    feats[f'{ticker}_MA20_Ratio'] = base_df['Close'] / base_df['Close'].rolling(20).mean() - 1.0
    feats[f'{ticker}_Volatility_20d'] = feats[f'{ticker}_Return_1d'].rolling(20).std() * np.sqrt(252)
    
    delta = base_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    feats[f'{ticker}_RSI_14'] = 100 - (100 / (1 + rs))
    
    # [2] マクロ指標
    feats['SP500_Return_1d'] = base_df['SP500_Close'].pct_change(1)
    feats['SP500_Return_5d'] = base_df['SP500_Close'].pct_change(5)
    feats['USDJPY_Return_1d'] = base_df['USDJPY_Close'].pct_change(1)
    feats['USDJPY_Return_5d'] = base_df['USDJPY_Close'].pct_change(5)
    feats['Nikkei_Return_1d'] = base_df['Nikkei_Close'].pct_change(1)
    feats['Nikkei_Return_5d'] = base_df['Nikkei_Close'].pct_change(5)
    
    # [3] ニュース感情スコア
    feats['News_Sentiment_Score'] = base_df['Sentiment_Score']
    feats['News_Sentiment_MA3d'] = base_df['Sentiment_Score'].rolling(3).mean()
    feats['News_Sentiment_Lag1'] = base_df['Sentiment_Score'].shift(1)
    
    # [4] 感情 × テクニカル複合
    feats['News_Sentiment_x_RSI'] = base_df['Sentiment_Score'] * ((feats[f'{ticker}_RSI_14'] - 50.0) / 25.0)
    feats['News_Sentiment_x_Return5d'] = base_df['Sentiment_Score'] * feats[f'{ticker}_Return_5d']
    feats['News_Sentiment_Surprise'] = base_df['Sentiment_Score'] - feats['News_Sentiment_MA3d']
    
    # [5] ファンダメンタルズ財務
    feats['Fund_Dynamic_PE'] = base_df['Close'] / (base_df['TTM_EPS'] + 1e-9)
    feats['Fund_Earnings_Yield'] = (base_df['TTM_EPS'] + 1e-9) / base_df['Close']
    feats['Fund_Rev_Growth_YoY'] = base_df['Fund_Rev_Growth_YoY']
    feats['Fund_Net_Margin'] = base_df['Fund_Net_Margin']
    feats['Fund_Earnings_Surprise'] = base_df['Fund_Earnings_Surprise']
    
    # [6] 正解ラベル (20営業日後 / 約1ヶ月後の終値 > 当日終値 なら 1, それ以外 0)
    feats['Target'] = (base_df['Close'].shift(-target_horizon) > base_df['Close']).astype(int)
    
    # 直近（最新営業日）の推論用データ
    latest_df = feats.drop(columns=['Target']).dropna().iloc[[-1]]
    
    # 学習・評価用データ（未来のターゲットが確定している期間）
    clean_df = feats.iloc[:-target_horizon].dropna()
    
    feature_cols = [c for c in clean_df.columns if c != 'Target']
    return clean_df, latest_df, feature_cols
