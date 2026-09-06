import json
import os
import urllib.request
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

from core.data_loader import clean_ticker_name, normalize_ticker

BASE_DATA_DIR = os.environ.get(
    "STOCK_AI_DATA_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")),
)
DEFAULT_JOBS_MAPPING_FILE = os.path.join(BASE_DATA_DIR, "jobs_mapping.json")
DEFAULT_CACHE_DIR = os.path.join(BASE_DATA_DIR, "jobs")
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
    cache_dir: str = DEFAULT_CACHE_DIR,
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

    today_str = datetime.today().strftime("%Y-%m-%d")

    # キャッシュのTTLチェック（日付が変わっていれば最新求人数を再取得してhistoryに追記）
    if cached_data is not None and cached_data.get("last_updated") != today_str:
        new_count, new_plat = fetch_company_job_openings(ticker, mapping_file=mapping_file)
        if new_count is not None:
            cached_data["count"] = new_count
            cached_data["platform"] = new_plat or cached_data.get("platform")
            cached_data["last_updated"] = today_str
            history = cached_data.get("history", [])
            dates_in_hist = [h.get("date") for h in history]
            if today_str in dates_in_hist:
                for h in history:
                    if h.get("date") == today_str:
                        h["count"] = new_count
            else:
                history.append({"date": today_str, "count": new_count})
            cached_data["history"] = history
            try:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cached_data, f, ensure_ascii=False, indent=2)
                if verbose:
                    print(f"  ・[{ticker}] 求人数キャッシュを本日の最新値 ({new_count} 件 / {new_plat}) に更新しました")
            except Exception:
                pass

    if cached_data is None:
        count, platform_name = fetch_company_job_openings(ticker, mapping_file=mapping_file)
        if count is not None:
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

    if cached_data and ("history" in cached_data or "count" in cached_data):
        if not df_stock.empty:
            s_jobs = pd.Series(0.0, index=df_stock.index, name="Job_Openings")
            history = cached_data.get("history", [])
            if history:
                for h in history:
                    h_dt = pd.to_datetime(h.get("date")).normalize()
                    if h_dt in s_jobs.index:
                        s_jobs.loc[h_dt] = float(h.get("count", 0))
                    elif h_dt >= s_jobs.index[0]:
                        # 最も近い直後の営業日にマッピング
                        later_days = s_jobs.index[s_jobs.index >= h_dt]
                        if len(later_days) > 0:
                            s_jobs.loc[later_days[0]] = float(h.get("count", 0))
                # 最初の観測日以降のみ前方補完 (ffill) し、過去への逆流 (bfill) は絶対に行わない
                first_obs = pd.to_datetime(history[0].get("date")).normalize()
                obs_mask = s_jobs.index >= first_obs
                s_jobs[obs_mask] = s_jobs[obs_mask].replace(0.0, None).ffill().fillna(0.0)
                # 最終営業日の最新値確認
                s_jobs.iloc[-1] = float(cached_data.get("count", 0))
            else:
                # 履歴がない場合は最新営業日にのみ求人数を配置
                s_jobs.iloc[-1] = float(cached_data.get("count", 0))
            return s_jobs
        return pd.Series([float(cached_data.get("count", 0))], name="Job_Openings")

    if verbose:
        print(f"  ・[{ticker}] 求人データ未設定/取得不可のため、中立値(0)で安全にフォールバックします")
    if not df_stock.empty:
        return pd.Series(0.0, index=df_stock.index, name="Job_Openings")
    return pd.Series(dtype=float, name="Job_Openings")
