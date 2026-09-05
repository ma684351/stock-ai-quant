#!/usr/bin/env python3
"""
========================================================================================
米国株AI投資判断システム (Stock AI Quant Pipeline)
========================================================================================
FinBERT感情分析、マクロ3指標、テクニカル指標、Yahoo!ファイナンス四半期財務データを統合し、
各銘柄に個別最適化したLightGBMモデルを学習して今後1ヶ月（20営業日）の投資判断を出力します。

使用方法:
  1. 単一銘柄の分析:
     python stock_ai.py AAPL
     python stock_ai.py TSLA --deep-research

  2. 複数銘柄の一括比較・ランキング:
     python stock_ai.py --compare AAPL NVDA GOOGL MSFT TSLA

  3. 対話型モード (引数なし):
     python stock_ai.py
========================================================================================
"""

import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='resource_tracker')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.data_loader import fetch_market_data, fetch_macro_data, fetch_fundamentals_data, normalize_ticker, is_japanese_ticker
from core.research_agent import get_ticker_catalysts
from core.sentiment import generate_news_dataset, analyze_sentiment
from core.features import build_features_and_target
from core.model import train_stock_model, predict_latest_signal

def format_price(ticker: str, price: float) -> str:
    """銘柄に応じた通貨フォーマット（円 / ドル）を返す"""
    if is_japanese_ticker(ticker):
        return f"¥{price:,.1f}"
    return f"${price:,.2f}"

def analyze_single_stock(ticker: str, enable_deep_research: bool = False, force_refresh: bool = False, verbose: bool = True):
    """単一銘柄（日本株・米国株）のエンドツーエンド分析を実行し、診断結果を返す"""
    ticker = normalize_ticker(ticker)
    if verbose:
        market_label = "日本株 (東証)" if is_japanese_ticker(ticker) else "米国株 (US)"
        print("\n" + "=" * 70)
        print(f"【AI投資診断パイプライン実行開始: {ticker} ({market_label})】")
        print("=" * 70)
        
    # 1. カタリスト取得（キャッシュまたはディープリサーチ）
    catalysts = get_ticker_catalysts(ticker, enable_deep_research=enable_deep_research, force_refresh=force_refresh)
    
    # 2. 市場データ・マクロ指標の取得
    df_stock = fetch_market_data(ticker, period="2y")
    df_sp500, df_usdjpy, df_nikkei = fetch_macro_data(period="2y")
    
    # 3. ファンダメンタルズ財務データの取得
    df_fund = fetch_fundamentals_data(ticker)
    
    # 4. ニュース収集 & 金融NLP感情分析 (日本株: 日本語金融BERT / 米国株: FinBERT)
    df_news = generate_news_dataset(df_stock, ticker, catalysts=catalysts, n_supplementary=120)
    df_sentiment = analyze_sentiment(df_news, ticker=ticker, batch_size=16)

    
    # 5. 特徴量エンジニアリング & ターゲット定義 (20営業日後 / 約1ヶ月後)
    if verbose:
        print(f"[{ticker}] 4大カテゴリ（テクニカル×マクロ×感情×財務）の特徴量を構築中...")
    df_features, df_latest, feature_cols = build_features_and_target(
        df_stock, df_sp500, df_usdjpy, df_nikkei, df_sentiment, df_fund, ticker=ticker, target_horizon=20
    )
    
    # 6. LightGBMモデル学習 & 閾値探索
    if verbose:
        print(f"[{ticker}] LightGBMモデルを個別最適化して学習中...")
    model, metrics, best_thresh, df_imp = train_stock_model(
        df_features, feature_cols, ticker=ticker, train_ratio=0.8
    )
    
    # 7. 直近営業日の売買シグナル判定
    latest_res = predict_latest_signal(
        model, df_latest, df_stock, ticker, feature_cols, threshold=best_thresh
    )
    latest_res['metrics'] = metrics
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"【過去テストデータ評価結果（{ticker} / 1ヶ月後株価予測）】")
        print(f"  判定閾値 (Threshold)    : {metrics['threshold']:.4f} (>= {metrics['threshold']*100:.1f}% で上昇予測)")
        print(f"  正解率 (Accuracy)       : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
        print(f"  適合率 (Precision)      : {metrics['precision']:.4f}")
        print(f"  再現率 (Recall)         : {metrics['recall']:.4f}")
        print(f"  F1スコア (F1 Score)     : {metrics['f1']:.4f}")
        print(f"  ROC-AUC スコア          : {metrics['auc']:.4f}")
        cm = metrics['cm']
        print(f"  混同行列: TN={cm[0,0]} (下落的中), FP={cm[0,1]} (上昇誤予測), FN={cm[1,0]}, TP={cm[1,1]}")
        print("=" * 60)
        
        print("\n▼ 特徴量重要度 Top 8:")
        for idx, row in df_imp.head(8).iterrows():
            print(f"  {idx+1:2d}位 {row['Feature']:<25} Gain={row['Gain']:8.2f}")
            
        print("\n" + "=" * 60)
        print(f"【Step 8: 直近（最新営業日: {latest_res['date']}）のAI投資シグナル判定】")
        print("=" * 60)
        print(f"  ・{ticker} 直近終値          : {format_price(ticker, latest_res['close'])}")
        if latest_res['dynamic_pe']:
            print(f"  ・動的 PER (バリュエーション) : {latest_res['dynamic_pe']:.1f} 倍")
        if latest_res['rev_growth'] is not None:
            print(f"  ・四半期売上高成長率 (YoY)  : {latest_res['rev_growth']*100:+.2f}%")
        if latest_res['ma20_ratio'] is not None:
            print(f"  ・20日移動平均乖離率        : {latest_res['ma20_ratio']*100:+.2f}%")
        if latest_res['rsi14'] is not None:
            print(f"  ・14日 RSI (過熱度)         : {latest_res['rsi14']:.1f}")
        if latest_res['sentiment'] is not None:
            sentiment_model_name = "日本語金融BERT" if is_japanese_ticker(ticker) else "FinBERT"
            print(f"  ・ニュース感情スコア({sentiment_model_name}): {latest_res['sentiment']:+.3f}")
        print("  " + "-" * 56)
        print(f"  ・今後20営業日(約1ヶ月)予測確率: 上昇 {latest_res['prob']*100:.2f}% / 下落 {latest_res['down_prob']*100:.2f}%")
        print(f"    (判定基準: 上昇 {latest_res['threshold']*100:.1f}% 以上で買い、下落 {(1.0-latest_res['sell_threshold'])*100:.1f}% 以上で売り)")
        print("  " + "-" * 56)
        print(f"  {latest_res['decision_label']} {latest_res['action_desc']}")
        print("=" * 60 + "\n")
        
    return latest_res

def print_comparison_table(results):
    """複数銘柄の診断結果をランキング一覧表として出力する"""
    sorted_res = sorted(results, key=lambda x: x['prob'], reverse=True)
    
    print("\n" + "=" * 98)
    print("【AI投資判断 複数銘柄比較ランキングサマリー（今後1ヶ月の予測）】")
    print("=" * 98)
    header = f"{'順位':<4} {'銘柄':<10} {'現在株価':>12} {'動的PER':>9} {'14日RSI':>8} {'20日乖離':>9} {'1ヶ月上昇確率':>14}  {'AI投資シグナル':<18}"
    print(header)
    print("-" * 98)
    
    for i, r in enumerate(sorted_res):
        t = r['ticker']
        price_str = format_price(t, r['close'])
        pe_str = f"{r['dynamic_pe']:.1f}倍" if r['dynamic_pe'] else "N/A"
        rsi_str = f"{r['rsi14']:.1f}" if r['rsi14'] else "N/A"
        ma_str = f"{r['ma20_ratio']*100:+.1f}%" if r['ma20_ratio'] is not None else "N/A"
        prob_str = f"{r['prob']*100:.1f}%"
        decision = r.get('decision_label', '★【 買い (BUY) 】' if r.get('is_buy') else '◇【 様子見 (HOLD) 】')
        
        row = f"{i+1:2d}位  {t:<10} {price_str:>12} {pe_str:>9} {rsi_str:>8} {ma_str:>9} {prob_str:>14}  {decision:<18}"
        print(row)
    print("=" * 98 + "\n")

def main():
    parser = argparse.ArgumentParser(description="日米株AI投資判断パイプライン (日本語金融BERT/FinBERT + LightGBM)")
    parser.add_argument("ticker", nargs="?", default=None, help="分析したいティッカーシンボル (例: AAPL, NVDA, TSLA, 7203.T, 6758.T)")
    parser.add_argument("--compare", nargs="+", help="複数銘柄を一括比較・ランキング (例: --compare AAPL 7203.T NVDA 6758.T)")
    parser.add_argument("--deep-research", action="store_true", help="Antigravity を使用して新銘柄のカタリストを自律調査")
    parser.add_argument("--force", action="store_true", help="既存のカタリストキャッシュを無視して自律リサーチを再実行")
    
    args = parser.parse_args()
    
    # 1. 複数銘柄比較モード
    if args.compare:
        tickers = [normalize_ticker(t) for t in args.compare]
        print(f"\n[複数比較モード] 以下の {len(tickers)} 銘柄を順番に個別学習・診断します: {', '.join(tickers)}")
        results = []
        for t in tickers:
            try:
                res = analyze_single_stock(t, enable_deep_research=args.deep_research, force_refresh=args.force, verbose=False)
                results.append(res)
                print(f"  ✔ {t:<8} 完了 (現在値: {format_price(t, res['close'])}, 上昇確率: {res['prob']*100:.1f}%, 判定: {res['decision_label']})")
            except Exception as e:
                print(f"  ✘ {t:<8} エラー発生: {e}")
                
        if results:
            print_comparison_table(results)
        return

    # 2. 単一銘柄指定モード
    if args.ticker:
        analyze_single_stock(args.ticker, enable_deep_research=args.deep_research, force_refresh=args.force, verbose=True)
        return

    # 3. 対話型モード
    print("\n" + "=" * 60)
    print("【日米株AI投資判断システム (日本語金融BERT / FinBERT + LightGBM)】")
    print("=" * 60)
    user_input = input("分析したいティッカーシンボルを入力してください (例: AAPL, 7203.T, NVDA, 6758 または複数カンマ区切り): ").strip()
    
    if not user_input:
        print("ティッカーが入力されませんでした。終了します。")
        return
        
    if "," in user_input or " " in user_input:
        tickers = [normalize_ticker(t) for t in user_input.replace(",", " ").split() if t.strip()]
        results = []
        print(f"\n以下の {len(tickers)} 銘柄を順番に診断します: {', '.join(tickers)}")
        for t in tickers:
            try:
                res = analyze_single_stock(t, enable_deep_research=False, verbose=False)
                results.append(res)
                print(f"  ✔ {t:<8} 完了 (現在値: {format_price(t, res['close'])}, 上昇確率: {res['prob']*100:.1f}%, 判定: {res['decision_label']})")
            except Exception as e:
                print(f"  ✘ {t:<8} エラー: {e}")
        if results:
            print_comparison_table(results)
    else:
        analyze_single_stock(user_input, enable_deep_research=False, verbose=True)

if __name__ == "__main__":
    main()
