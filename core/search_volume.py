import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf

from core.data_loader import clean_ticker_name, is_japanese_ticker, normalize_ticker

# 日米主要銘柄の Wikipedia 記事名マッピング辞書 (lang, article_title)
KNOWN_WIKIPEDIA_PAGES = {
    # 米国株 (en.wikipedia.org)
    "AAPL": ("en", "Apple_Inc."),
    "NVDA": ("en", "Nvidia"),
    "GOOGL": ("en", "Alphabet_Inc."),
    "GOOG": ("en", "Alphabet_Inc."),
    "MSFT": ("en", "Microsoft"),
    "AMZN": ("en", "Amazon_(company)"),
    "TSLA": ("en", "Tesla,_Inc."),
    "META": ("en", "Meta_Platforms"),
    "GLW": ("en", "Corning_Inc."),
    "INTC": ("en", "Intel"),
    "AMD": ("en", "AMD"),
    "NFLX": ("en", "Netflix"),
    # 日本株 (ja.wikipedia.org)
    "7203": ("ja", "トヨタ自動車"),
    "7203.T": ("ja", "トヨタ自動車"),
    "6758": ("ja", "ソニーグループ"),
    "6758.T": ("ja", "ソニーグループ"),
    "7974": ("ja", "任天堂"),
    "7974.T": ("ja", "任天堂"),
    "3436": ("ja", "SUMCO"),
    "3436.T": ("ja", "SUMCO"),
    "4385": ("ja", "メルカリ_(企業)"),
    "4385.T": ("ja", "メルカリ_(企業)"),
    "6367": ("ja", "ダイキン工業"),
    "6367.T": ("ja", "ダイキン工業"),
    "6521": ("ja", "オキサイド_(企業)"),
    "6521.T": ("ja", "オキサイド_(企業)"),
    "6981": ("ja", "村田製作所"),
    "6981.T": ("ja", "村田製作所"),
    "9984": ("ja", "ソフトバンクグループ"),
    "9984.T": ("ja", "ソフトバンクグループ"),
    "6857": ("ja", "アドバンテスト"),
    "6857.T": ("ja", "アドバンテスト"),
    "8035": ("ja", "東京エレクトロン"),
    "8035.T": ("ja", "東京エレクトロン"),
}


def resolve_wikipedia_article(ticker: str) -> Tuple[str, str]:
    """ティッカーシンボルから対応する Wikipedia 言語コードと記事名を解決する"""
    norm_t = normalize_ticker(ticker)
    clean_t = norm_t.replace(".T", "")

    if norm_t in KNOWN_WIKIPEDIA_PAGES:
        return KNOWN_WIKIPEDIA_PAGES[norm_t]
    if clean_t in KNOWN_WIKIPEDIA_PAGES:
        return KNOWN_WIKIPEDIA_PAGES[clean_t]

    is_jp = is_japanese_ticker(norm_t)
    lang = "ja" if is_jp else "en"

    # 未登録銘柄は yfinance の会社名から抽出を試みる
    try:
        t_obj = yf.Ticker(norm_t)
        name = t_obj.info.get("shortName") or t_obj.info.get("longName") or clean_t
        # 余分な株式会社・Inc.・コーポレーションなどの接尾辞を簡易クリーン
        clean_name = re.sub(
            r"(\s*Co\.,?\s*Ltd\.?|\s*Inc\.?|\s*Corp\.?|\s*Corporation|\s*株式会社)",
            "",
            name,
            flags=re.IGNORECASE,
        ).strip()
        article = clean_name.replace(" ", "_")
        return lang, article
    except Exception:
        return lang, clean_t


def fetch_wikimedia_pageviews(
    lang: str,
    article: str,
    start_date: str,
    end_date: str,
    user_agent: str = "FinQuantPipeline/1.0 (https://github.com/ma684351/stock-ai-quant; contact@example.com)",
) -> Optional[dict]:
    """Wikimedia 公式 Pageviews REST API から指定記事の日次アクセス数を取得する"""
    encoded_article = urllib.parse.quote(article, safe="")
    project = f"{lang}.wikipedia.org"
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{project}/all-access/all-agents/{encoded_article}/daily/{start_date}/{end_date}"
    )

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return data
    except Exception:
        # 404（記事が見つからない）やネットワーク遮断時は None
        return None
    return None


def get_search_volume_series(
    ticker: str,
    df_stock: pd.DataFrame,
    cache_dir: str = "data/attention",
    verbose: bool = True,
) -> pd.Series:
    """
    指定ティッカーの検索・アクセスボリューム時系列 (日次PV) を取得する。
    ローカルキャッシュが存在すればキャッシュから高速読込し、無ければ Wikimedia API から取得する。
    取得不能時は 0 で安全に補完された Series を返す。
    """
    clean_t = clean_ticker_name(ticker)
    cache_path = os.path.join(cache_dir, f"{clean_t}.json")

    # 取得期間の決定（株価データの開始日〜終了日、または直近2年）
    if not df_stock.empty:
        start_dt = df_stock.index[0] - timedelta(days=60)  # MA60算出用に少し前から
        end_dt = df_stock.index[-1]
    else:
        end_dt = datetime.today()
        start_dt = end_dt - timedelta(days=730)

    start_str = start_dt.strftime("%Y%m%d")
    end_str = end_dt.strftime("%Y%m%d")

    raw_items = None

    # 1. キャッシュの確認
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                raw_items = json.load(f)
            if verbose:
                print(f"  ・[{ticker}] 検索ボリュームキャッシュをロードしました ({len(raw_items)}日分)")
        except Exception:
            raw_items = None

    # 2. キャッシュがない場合、APIから取得
    if raw_items is None:
        lang, article = resolve_wikipedia_article(ticker)
        if verbose:
            print(f"  ・[{ticker}] Wikimedia Pageviews APIからアクセス数取得中 ({lang}:{article})...")

        res_data = fetch_wikimedia_pageviews(lang, article, start_str, end_str)
        if res_data and "items" in res_data:
            raw_items = res_data["items"]
            # キャッシュ保存
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(raw_items, f, ensure_ascii=False, indent=2)
            if verbose:
                print(f"  ・[{ticker}] 検索ボリューム {len(raw_items)} 日分を取得・キャッシュ保存しました")

    # 3. Series の構築
    if raw_items:
        records = []
        for item in raw_items:
            # timestamp format: "2024010100"
            ts_str = str(item.get("timestamp", ""))[:8]
            views = float(item.get("views", 0))
            try:
                dt = pd.to_datetime(ts_str, format="%Y%m%d")
                records.append({"Date": dt, "Attention_Volume": views})
            except Exception:
                continue

        if records:
            df_pv = pd.DataFrame(records).set_index("Date").sort_index()
            # 重複日付の排除
            df_pv = df_pv[~df_pv.index.duplicated(keep="last")]
            # 株価データのインデックスに合わせて再インデックス & 前方補完
            if not df_stock.empty:
                s_pv = df_pv["Attention_Volume"].reindex(df_stock.index).ffill().bfill().fillna(0.0)
                return s_pv
            return df_pv["Attention_Volume"]

    # 4. 取得失敗時の安全なフォールバック（全ゼロ Series）
    if verbose:
        print(f"  ・[{ticker}] 検索ボリューム取得不可のため、中立値(0)で安全にフォールバックします")
    if not df_stock.empty:
        return pd.Series(0.0, index=df_stock.index, name="Attention_Volume")
    return pd.Series(dtype=float, name="Attention_Volume")
