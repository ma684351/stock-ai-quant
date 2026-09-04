import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def train_stock_model(df_features, feature_cols, ticker="AAPL", train_ratio=0.8):
    """
    指定ティッカーのデータを用いて、時系列分割・クラス不均衡補正・正則化を行い
    最適な判定閾値を自動探索して LightGBM モデルを学習する
    """
    X = df_features[feature_cols]
    y = df_features['Target']
    
    split_idx = int(len(df_features) * train_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    neg_train = int(np.sum(y_train == 0))
    pos_train = int(np.sum(y_train == 1))
    scale_pos_weight = float(neg_train / pos_train) if pos_train > 0 else 1.0
    
    # メモリ連続化 (C-contiguous) を保証し、macOS OpenMP でのポインタ不正アクセスを防止
    X_train_arr = np.ascontiguousarray(X_train.values, dtype=np.float32)
    y_train_arr = np.ascontiguousarray(y_train.values, dtype=np.float32)
    X_test_arr = np.ascontiguousarray(X_test.values, dtype=np.float32)
    y_test_arr = np.ascontiguousarray(y_test.values, dtype=np.float32)
    
    lgb_train = lgb.Dataset(X_train_arr, y_train_arr, feature_name=list(feature_cols))
    lgb_test = lgb.Dataset(X_test_arr, y_test_arr, reference=lgb_train, feature_name=list(feature_cols))
    
    params = {
        'objective': 'binary',
        'metric': ['auc', 'binary_logloss'],
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 12,
        'max_depth': 4,
        'min_child_samples': 20,
        'feature_fraction': 0.65,
        'bagging_fraction': 0.75,
        'bagging_freq': 1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'num_threads': 1,
        'verbose': -1
    }
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=40, verbose=False),
        lgb.log_evaluation(period=0)
    ]
    
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=300,
        valid_sets=[lgb_train, lgb_test],
        callbacks=callbacks
    )
    
    # 判定閾値探索: 全押し（TN=0 または TP=0）を防ぐため、双方の再現率が最低20%以上確保される候補から選択
    y_train_proba = model.predict(X_train_arr)
    p25 = float(np.percentile(y_train_proba, 20))
    p75 = float(np.percentile(y_train_proba, 80))
    threshold_candidates = np.linspace(max(0.38, p25), min(0.62, p75), 41)
    
    best_thresh = 0.50
    best_score = -1.0
    for th in threshold_candidates:
        preds = (y_train_proba >= th).astype(int)
        rec_pos = recall_score(y_train, preds, pos_label=1, zero_division=0)
        rec_neg = recall_score(y_train, preds, pos_label=0, zero_division=0)
        bal_acc = 0.5 * (rec_pos + rec_neg)
        f1 = f1_score(y_train, preds, zero_division=0)
        
        if rec_pos >= 0.20 and rec_neg >= 0.20:
            score = 0.7 * bal_acc + 0.3 * f1
        else:
            score = (0.7 * bal_acc + 0.3 * f1) * 0.4
            
        if score > best_score:
            best_score = score
            best_thresh = float(th)
            
    # テスト評価
    y_pred_proba = model.predict(X_test_arr)
    y_pred = (y_pred_proba >= best_thresh).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'cm': cm,
        'threshold': best_thresh,
        'test_count': len(y_test)
    }
    
    # 特徴量重要度
    importance_gain = model.feature_importance(importance_type='gain')
    df_imp = pd.DataFrame({'Feature': feature_cols, 'Gain': importance_gain}).sort_values('Gain', ascending=False).reset_index(drop=True)
    
    return model, metrics, best_thresh, df_imp

def predict_latest_signal(model, df_latest, df_stock, ticker, feature_cols, threshold=0.50):
    """最新営業日（直近）のデータに対して今後1ヶ月の上昇確率と投資判断を判定する"""
    latest_feat = df_latest[feature_cols].iloc[[-1]]
    latest_date = latest_feat.index[0]
    latest_feat_arr = np.ascontiguousarray(latest_feat.values, dtype=np.float32)
    
    prob = float(model.predict(latest_feat_arr)[0])
    is_buy = prob >= threshold
    
    latest_close = float(df_stock.loc[latest_date, 'Close']) if latest_date in df_stock.index else 0.0
    dynamic_pe = float(latest_feat['Fund_Dynamic_PE'].values[0]) if 'Fund_Dynamic_PE' in latest_feat else None
    rev_growth = float(latest_feat['Fund_Rev_Growth_YoY'].values[0]) if 'Fund_Rev_Growth_YoY' in latest_feat else None
    
    ma20_col = f'{ticker}_MA20_Ratio'
    rsi_col = f'{ticker}_RSI_14'
    ma20_ratio = float(latest_feat[ma20_col].values[0]) if ma20_col in latest_feat else None
    rsi14 = float(latest_feat[rsi_col].values[0]) if rsi_col in latest_feat else None
    sentiment = float(latest_feat['News_Sentiment_Score'].values[0]) if 'News_Sentiment_Score' in latest_feat else None
    
    return {
        'ticker': ticker,
        'date': latest_date.strftime('%Y-%m-%d'),
        'close': latest_close,
        'dynamic_pe': dynamic_pe,
        'rev_growth': rev_growth,
        'ma20_ratio': ma20_ratio,
        'rsi14': rsi14,
        'sentiment': sentiment,
        'prob': prob,
        'threshold': threshold,
        'is_buy': is_buy
    }
