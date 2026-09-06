import json
import os
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

from core.data_loader import clean_ticker_name

BASE_DATA_DIR = os.environ.get(
    "STOCK_AI_DATA_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")),
)
CATALYST_DIR = os.path.join(BASE_DATA_DIR, "catalysts")


def load_catalysts(ticker: str) -> List[Tuple[str, str]]:
    """既存の確定カタリスト（キャッシュ）をロードする"""
    clean_t = clean_ticker_name(ticker)
    candidates = [
        os.path.join(CATALYST_DIR, f"{ticker.upper()}.json"),
        os.path.join(CATALYST_DIR, f"{clean_t}.json"),
        os.path.join(CATALYST_DIR, f"{ticker.split('.')[0].upper()}.json"),
    ]
    path = None
    for p in candidates:
        if os.path.exists(p):
            path = p
            break

    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[{ticker}] カタリストキャッシュから {len(data)} 件の確定イベントをロードしました")
            return [(str(item[0]), str(item[1])) for item in data]
        except Exception as e:
            print(f"[{ticker}] カタリスト読み込みエラー: {e}")
    return []


def save_catalysts(ticker: str, catalysts: List[Tuple[str, str]]):
    """カタリストを data/catalysts/{ticker}.json に自動保存する"""
    os.makedirs(CATALYST_DIR, exist_ok=True)
    clean_t = clean_ticker_name(ticker)
    path = os.path.join(CATALYST_DIR, f"{clean_t}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalysts, f, ensure_ascii=False, indent=2)
    print(f"[{ticker}] {len(catalysts)} 件のカタリストを {path} に保存しました")


def fetch_financial_catalysts_yfinance(ticker: str) -> List[Tuple[str, str]]:
    """
    [フォールバック] yfinance から四半期決算サプライズ、株式分割、最新ニュースを自動抽出する
    """
    print(f"  [Financial Engine] yfinance から '{ticker}' の確定決算・企業イベントを自動抽出中...")
    catalysts = []
    cutoff_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    try:
        import pandas as pd
        import yfinance as yf

        t = yf.Ticker(ticker)

        # 1. 四半期決算日・サプライズ実績
        try:
            ed = t.get_earnings_dates(limit=16)
            if ed is not None and not ed.empty:
                for idx, row in ed.iterrows():
                    dt_str = idx.strftime("%Y-%m-%d")
                    if dt_str < cutoff_date:
                        continue
                    reported = row.get("Reported EPS")
                    est = row.get("EPS Estimate")
                    surp = row.get("Surprise(%)")
                    if pd.notna(reported) and pd.notna(est):
                        surp_text = f" ({surp:+.1f}% surprise)" if pd.notna(surp) else ""
                        headline = (
                            f"{ticker} Q earnings: Reported EPS {reported:.2f} vs consensus {est:.2f}{surp_text}."
                        )
                        catalysts.append((dt_str, headline))
        except Exception:
            pass

        # 2. 株式分割イベント
        try:
            splits = t.splits
            if splits is not None and not splits.empty:
                for dt, ratio in splits.items():
                    dt_str = dt.strftime("%Y-%m-%d")
                    if dt_str >= cutoff_date:
                        catalysts.append((dt_str, f"{ticker} completes stock split with ratio {ratio}:1."))
        except Exception:
            pass

        # 3. 最新リアルニュース
        try:
            news = t.news
            if news:
                for n in news[:10]:
                    ts = n.get("providerPublishTime")
                    title = n.get("title")
                    if ts and title:
                        dt_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                        catalysts.append((dt_str, f"{ticker}: {title}"))
        except Exception:
            pass

    except Exception as e:
        print(f"  [Financial Engine] yfinance 取得エラー: {e}")

    # 重複除去 & ソート
    unique_catalysts = {}
    for d, c in catalysts:
        if d not in unique_catalysts:
            unique_catalysts[d] = c
    sorted_catalysts = sorted(unique_catalysts.items(), key=lambda x: x[0])

    if sorted_catalysts:
        print(f"  ✔ yfinance より {len(sorted_catalysts)} 件の決算・適時開示イベントを抽出しました")
        save_catalysts(ticker, sorted_catalysts)
    return sorted_catalysts


def get_ticker_catalysts(ticker: str) -> List[Tuple[str, str]]:
    """
    指定銘柄のカタリストを取得する。
    1. data/catalysts/{ticker}.json があればロード（Antigravity スキルにより高精度生成）
    2. なければ yfinance 確定決算エンジンで自動抽出
    """
    ticker = ticker.upper().strip()
    existing = load_catalysts(ticker)
    if existing:
        return existing

    print(
        f"[{ticker}] カタリストキャッシュがありません。yfinance自動抽出を実行します（※Antigravityスキルでリサーチすると最高精度のカタリストが生成されます）"
    )
    return fetch_financial_catalysts_yfinance(ticker)
