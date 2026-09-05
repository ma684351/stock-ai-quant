import json
import os
import urllib.request
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

from core.data_loader import clean_ticker_name, normalize_ticker

DEFAULT_JOBS_MAPPING_FILE = "data/jobs_mapping.json"
DEFAULT_USER_AGENT = "FinQuantPipeline/1.0 (https://github.com/ma684351/stock-ai-quant; contact@example.com)"


def load_jobs_mappings(mapping_file: str = DEFAULT_JOBS_MAPPING_FILE) -> dict:
    """外部の求人ATSマッピングJSONを読み込む"""
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def fetch_greenhouse_jobs(token: str, user_agent: str = DEFAULT_USER_AGENT) -> Optional[int]:
    """Greenhouse 公開APIから有効求人数を取得する"""
    url = f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs"
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return len(data.get("jobs", []))
    except Exception:
        return None
    return None


def fetch_lever_jobs(token: str, user_agent: str = DEFAULT_USER_AGENT) -> Optional[int]:
    """Lever 公開APIから有効求人数を取得する"""
    url = f"https://api.lever.co/v0/postings/{token}"
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                if isinstance(data, list):
                    return len(data)
                return len(data.get("data", []))
    except Exception:
        return None
    return None


def fetch_workday_jobs(url: str, user_agent: str = DEFAULT_USER_AGENT) -> Optional[int]:
    """Workday CXS 公開APIから有効求人数 (total) を取得する"""
    headers = {
        "User-Agent": user_agent,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = json.dumps({"appliedFacets": {}, "limit": 1, "offset": 0, "searchText": ""}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                total = data.get("total")
                if total is not None:
                    return int(total)
    except Exception:
        return None
    return None


def fetch_company_job_openings(
    ticker: str,
    mapping_file: str = DEFAULT_JOBS_MAPPING_FILE,
) -> Tuple[Optional[int], Optional[str]]:
    """ティッカーからATSプラットフォームと求人数を取得する"""
    norm_t = normalize_ticker(ticker)
    clean_t = norm_t.replace(".T", "")

    mappings = load_jobs_mappings(mapping_file)
    cfg = mappings.get(norm_t) or mappings.get(clean_t)
    if not cfg:
        return None, None

    platform = cfg.get("platform", "").lower()
    if platform == "greenhouse":
        token = cfg.get("token")
        if token:
            count = fetch_greenhouse_jobs(token)
            return count, "Greenhouse"
    elif platform == "lever":
        token = cfg.get("token")
        if token:
            count = fetch_lever_jobs(token)
            return count, "Lever"
    elif platform == "workday":
        url = cfg.get("url")
        if url:
            count = fetch_workday_jobs(url)
            return count, "Workday"

    return None, None


def get_job_openings_series(
    ticker: str,
    df_stock: pd.DataFrame,
    cache_dir: str = "data/jobs",
    mapping_file: str = DEFAULT_JOBS_MAPPING_FILE,
    verbose: bool = True,
) -> pd.Series:
    """
    指定ティッカーの求人数時系列データを取得・生成する。
    ローカルキャッシュが存在すれば読み込み、無ければATS APIから取得してキャッシュ保存する。
    未登録・取得不可時は 0.0 で安全にフォールバックした Series を返す。
    """
    clean_t = clean_ticker_name(ticker)
    cache_path = os.path.join(cache_dir, f"{clean_t}.json")

    cached_data = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            if verbose:
                cnt = cached_data.get("count", 0)
                plat = cached_data.get("platform", "ATS")
                print(f"  ・[{ticker}] 求人数キャッシュをロードしました ({cnt} 件 / {plat})")
        except Exception:
            cached_data = None

    if cached_data is None:
        count, platform_name = fetch_company_job_openings(ticker, mapping_file=mapping_file)
        if count is not None:
            today_str = datetime.today().strftime("%Y-%m-%d")
            cached_data = {
                "ticker": clean_t,
                "count": count,
                "platform": platform_name,
                "last_updated": today_str,
                "history": [{"date": today_str, "count": count}],
            }
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cached_data, f, ensure_ascii=False, indent=2)
            if verbose:
                print(f"  ・[{ticker}] ATS公開APIから求人数 {count} 件 ({platform_name}) を取得・保存しました")

    if cached_data and "count" in cached_data:
        val = float(cached_data["count"])
        if not df_stock.empty:
            return pd.Series(val, index=df_stock.index, name="Job_Openings")
        return pd.Series([val], name="Job_Openings")

    if verbose:
        print(f"  ・[{ticker}] 求人データ未設定/取得不可のため、中立値(0)で安全にフォールバックします")
    if not df_stock.empty:
        return pd.Series(0.0, index=df_stock.index, name="Job_Openings")
    return pd.Series(dtype=float, name="Job_Openings")
