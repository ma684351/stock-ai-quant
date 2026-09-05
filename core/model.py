import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from core.data_loader import clean_ticker_name

def train_stock_model(df_features, feature_cols, ticker="AAPL", train_ratio=0.8):
    """
    指定ティッカーのデータを用いて、時系列分割・クラス不均衡補正・正則化を行い
    最適な判定閾値を自動探索して LightGBM モデルを学習する
    """
    X = df_features[feature_cols]
    y = df_features['Target']
    
    split_idx = int(len(df_features) * train_ratio)
    df_train_full = df_features.iloc[:split_idx]
    df_test = df_features.iloc[split_idx:]
    
    # 訓練データをさらに学習(80%)と検証(20%)に時系列分割し、テストデータのリークや早期終了誤爆を防止
    val_idx = int(len(df_train_full) * 0.80)
    df_tr = df_train_full.iloc[:val_idx]
    df_val = df_train_full.iloc[val_idx:]
    
    neg_train = int(np.sum(df_tr['Target'] == 0))
    pos_train = int(np.sum(df_tr['Target'] == 1))
    scale_pos_weight = float(neg_train / pos_train) if pos_train > 0 else 1.0
    
    # メモリ連続化 (C-contiguous) を保証し、macOS OpenMP でのポインタ不正アクセスを防止
    X_train_arr = np.ascontiguousarray(df_tr[feature_cols].values, dtype=np.float32)
    y_train_arr = np.ascontiguousarray(df_tr['Target'].values, dtype=np.float32)
    X_val_arr = np.ascontiguousarray(df_val[feature_cols].values, dtype=np.float32)
    y_val_arr = np.ascontiguousarray(df_val['Target'].values, dtype=np.float32)
    X_test_arr = np.ascontiguousarray(df_test[feature_cols].values, dtype=np.float32)
    y_test_arr = np.ascontiguousarray(df_test['Target'].values, dtype=np.float32)
    
    lgb_train = lgb.Dataset(X_train_arr, y_train_arr, feature_name=list(feature_cols))
    lgb_val = lgb.Dataset(X_val_arr, y_val_arr, reference=lgb_train, feature_name=list(feature_cols))
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 12,
        'max_depth': 4,
        'min_child_samples': 15,
        'feature_fraction': 0.65,
        'bagging_fraction': 0.80,
        'bagging_freq': 1,
        'reg_alpha': 0.5,
        'reg_lambda': 1.5,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'num_threads': 1,
        'verbose': -1
    }
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=35, verbose=False),
        lgb.log_evaluation(period=0)
    ]
    
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=250,
        valid_sets=[lgb_train, lgb_val],
        callbacks=callbacks
    )
    
    # 判定閾値探索: バリデーションセットで下落回避率(rec_neg)と適合率(precision)を重視して探索
    y_val_proba = model.predict(X_val_arr)
    threshold_candidates = np.linspace(0.40, 0.65, 51)
    
    best_thresh = 0.50
    best_score = -1.0
    for th in threshold_candidates:
        preds = (y_val_proba >= th).astype(int)
        rec_pos = recall_score(y_val_arr, preds, pos_label=1, zero_division=0)
        rec_neg = recall_score(y_val_arr, preds, pos_label=0, zero_division=0)
        prec = precision_score(y_val_arr, preds, zero_division=0)
        bal_acc = 0.5 * (rec_pos + rec_neg)
        f1 = f1_score(y_val_arr, preds, zero_division=0)
        
        # 下落的中率 (rec_neg) 25% 以上、上昇再現率 20% 以上を確保する健全な閾値を優先
        if rec_neg >= 0.25 and rec_pos >= 0.20:
            score = 0.35 * prec + 0.35 * bal_acc + 0.30 * f1
        else:
            score = (0.35 * prec + 0.35 * bal_acc + 0.30 * f1) * 0.3
            
        if score > best_score:
            best_score = score
            best_thresh = float(th)
            
    # テスト評価
    y_test = df_test['Target'].values
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
    """最新営業日（直近）のデータに対して今後1ヶ月の上昇・下落確率と3段階投資判断（BUY/HOLD/SELL）を判定する"""
    latest_feat = df_latest[feature_cols].iloc[[-1]]
    latest_date = latest_feat.index[0]
    latest_feat_arr = np.ascontiguousarray(latest_feat.values, dtype=np.float32)
    
    prob = float(model.predict(latest_feat_arr)[0])
    down_prob = 1.0 - prob
    
    buy_threshold = float(threshold)
    # 中立帯（HOLD）を適正に確保した売り閾値（下落確率が十分高い領域を検出）
    sell_threshold = min(buy_threshold - 0.08, round(1.0 - buy_threshold, 4))
    sell_threshold = max(0.35, min(0.46, sell_threshold))
    
    if prob >= buy_threshold:
        decision = "BUY"
        decision_label = "★【 買い (BUY) 】"
        action_desc = "上昇トレンド予測シグナル点灯！"
    elif prob <= sell_threshold:
        decision = "SELL"
        decision_label = "▼【 売り (SELL) 】"
        action_desc = "下落リスク警戒シグナル点灯！（利益確定・損切り、または空売り検討）"
    else:
        decision = "HOLD"
        decision_label = "◇【 様子見 (HOLD) 】"
        action_desc = "方向感に乏しく様子見推奨（中立レンジ）"
    
    latest_close = float(df_stock.loc[latest_date, 'Close']) if latest_date in df_stock.index else 0.0
    dynamic_pe = float(df_latest['Fund_Dynamic_PE'].iloc[-1]) if 'Fund_Dynamic_PE' in df_latest.columns else None
    rev_growth = float(df_latest['Fund_Rev_Growth_YoY'].iloc[-1]) if 'Fund_Rev_Growth_YoY' in df_latest.columns else None
    
    t_prefix = clean_ticker_name(ticker)
    ma20_col = f'{t_prefix}_MA20_Ratio'
    rsi_col = f'{t_prefix}_RSI_14'
    ma20_ratio = float(df_latest[ma20_col].iloc[-1]) if ma20_col in df_latest.columns else None
    rsi14 = float(df_latest[rsi_col].iloc[-1]) if rsi_col in df_latest.columns else None
    sentiment = float(df_latest['News_Sentiment_Score'].iloc[-1]) if 'News_Sentiment_Score' in df_latest.columns else None
    
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
        'down_prob': down_prob,
        'threshold': buy_threshold,
        'sell_threshold': sell_threshold,
        'decision': decision,
        'decision_label': decision_label,
        'action_desc': action_desc,
        'is_buy': (decision == "BUY"),
        'is_sell': (decision == "SELL"),
        'is_hold': (decision == "HOLD")
    }

