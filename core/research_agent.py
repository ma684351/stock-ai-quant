import os
import sys
import json
import re
import shutil
import subprocess
from datetime import datetime
from typing import List, Tuple, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

CATALYST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "catalysts"))

def load_catalysts(ticker: str) -> List[Tuple[str, str]]:
    """既存の確定カタリスト（キャッシュ）をロードする"""
    path = os.path.join(CATALYST_DIR, f"{ticker.upper()}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[{ticker}] キャッシュから {len(data)} 件の確定カタリストをロードしました")
            return [(str(item[0]), str(item[1])) for item in data]
        except Exception as e:
            print(f"[{ticker}] カタリスト読み込みエラー: {e}")
    return []

def save_catalysts(ticker: str, catalysts: List[Tuple[str, str]]):
    """カタリストを data/catalysts/{ticker}.json に自動保存する"""
    os.makedirs(CATALYST_DIR, exist_ok=True)
    path = os.path.join(CATALYST_DIR, f"{ticker.upper()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalysts, f, ensure_ascii=False, indent=2)
    print(f"[{ticker}] {len(catalysts)} 件のカタリストを {path} に保存しました")

def _parse_catalysts_json(text: str) -> Optional[List[Tuple[str, str]]]:
    """テキストから JSON 配列を抽出・検証する"""
    text = text.strip()
    match = re.search(r'\[\s*\[.*\]\s*\]', text, re.DOTALL)
    if not match:
        # 単一配列の可能性もチェック
        match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list) and len(data) > 0:
                results = []
                for item in data:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        date_str = str(item[0]).strip()
                        content = str(item[1]).strip()
                        # 日付フォーマットの簡易チェック (YYYY-MM-DD)
                        if re.match(r'^\d{4}-\d{2}-\d{2}', date_str):
                            results.append((date_str[:10], content))
                if results:
                    results.sort(key=lambda x: x[0])
                    return results
        except Exception:
            pass
    return None

def _build_catalyst_prompt(ticker: str) -> str:
    """ディープリサーチ用の共通プロンプトを構築する"""
    return f"""You are a top-tier quantitative financial research analyst.
For US stock ticker '{ticker.upper()}', research and compile major historical catalysts from January 2024 to September 2026 that materially impacted the stock's price action.
Include:
- Quarterly earnings results (EPS beat/miss, revenue, guidance)
- Major product launches, keynotes, or technological breakthroughs
- CEO/leadership changes or significant executive compensation events
- Regulatory actions, antitrust lawsuits, or DOJ/SEC investigations
- Strategic acquisitions, divestitures, or multibillion-dollar commercial contracts
- Factory openings, delivery numbers, or major operational milestones

Output Format Requirement:
Return ONLY a valid JSON array of two-element arrays, with no markdown code blocks, backticks, or explanation:
[
  ["YYYY-MM-DD", "Concise factual English headline (50-100 characters)"]
]
Provide between 15 and 25 significant catalysts in strictly chronological order."""

def _find_agy_binary() -> Optional[str]:
    """ローカルにインストールされている Antigravity CLI (agy) バイナリを探索する"""
    # 1. PATH から探索
    path = shutil.which("agy")
    if path and os.path.exists(path):
        return path
    # 2. 代表的なインストールパス
    candidates = [
        "/opt/homebrew/bin/agy",
        "/usr/local/bin/agy",
        os.path.expanduser("~/.gemini/antigravity-cli/bin/agy"),
        os.path.expanduser("~/bin/agy"),
        "/opt/homebrew/Caskroom/antigravity-cli/1.1.26,5550154686791680/antigravity"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def run_antigravity_session_research(ticker: str) -> Optional[List[Tuple[str, str]]]:
    """
    ローカルの Antigravity ログインセッション (CLI: agy) を使用して自律ディープリサーチを実行する。
    APIキー不要で、ローカルに保存されている既存のログイン認証セッションを使用します。
    """
    agy_bin = _find_agy_binary()
    if not agy_bin:
        return None

    print(f"\n[Deep Research Agent] '{ticker}' の確定カタリストを自律調査中 (Engine: Antigravity Local Login Session)...")
    prompt = _build_catalyst_prompt(ticker)

    try:
        # ローカルのログインセッションを使って agy コマンドを実行（APIキー不要）
        # --effort low: 不要な長考・過剰な反復Web検索によるレートリミット遅延を防止
        # --dangerously-skip-permissions: ツール実行承認待ちによるブロックを回避
        # stdin=subprocess.DEVNULL: 対話モード誤判定防止
        cmd = [
            agy_bin,
            "--effort", "low",
            "--dangerously-skip-permissions",
            "--disable-slash-commands",
            "--print", prompt
        ]
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=300,
            check=False
        )
        if result.returncode == 0 and result.stdout:
            catalysts = _parse_catalysts_json(result.stdout)
            if catalysts and len(catalysts) >= 5:
                print(f"  ✔ Antigravity ログインセッションにより {len(catalysts)} 件の重要カタリストを自律特定しました")
                save_catalysts(ticker, catalysts)
                return catalysts
            else:
                print("  [Deep Research Agent] ログインセッションの応答パースに失敗しました。フォールバックを試行します。")
        else:
            stderr_msg = result.stderr.strip() if result.stderr else ""
            if stderr_msg:
                print(f"  [Deep Research Agent] agy 実行警告: {stderr_msg[:200]}")
    except subprocess.TimeoutExpired:
        print("  [Deep Research Agent] Antigravity ログインセッションの実行がタイムアウトしました。")
    except Exception as e:
        print(f"  [Deep Research Agent] ログインセッション実行エラー: {e}")

    return None

async def run_antigravity_sdk_research_async(ticker: str) -> Optional[List[Tuple[str, str]]]:
    """
    Antigravity Python SDK (google.antigravity) を呼び出して自律ディープリサーチを実行する
    (APIキーが設定されている場合のセカンダリエンジン)
    """
    # APIキーの取得 (.env または環境変数)
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # プロジェクトルートの .env ファイルが存在すれば読み込み
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY=") or line.startswith("GOOGLE_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

    if not api_key:
        return None

    print(f"\n[Deep Research Agent] '{ticker}' の確定カタリストを自律調査中 (Engine: Antigravity Python SDK)...")
    prompt = _build_catalyst_prompt(ticker)

    try:
        from google.antigravity import Agent, LocalAgentConfig, CapabilitiesConfig
        
        config_kwargs = {
            "system_instructions": "You are an autonomous financial research agent. Output strictly formatted JSON arrays.",
            "capabilities": CapabilitiesConfig(),
            "api_key": api_key
        }
            
        config = LocalAgentConfig(**config_kwargs)
        
        async with Agent(config) as agent:
            response = await agent.chat(prompt)
            text = ""
            async for token in response:
                text += token
                
            catalysts = _parse_catalysts_json(text)
            if catalysts and len(catalysts) >= 5:
                print(f"  ✔ Antigravity SDK により {len(catalysts)} 件の重要カタリストを自律特定しました")
                save_catalysts(ticker, catalysts)
                return catalysts
            else:
                print("  [Deep Research Agent] SDKレスポンスのパースに失敗しました。フォールバックを試行します。")
    except ImportError:
        print("  [Deep Research Agent] google-antigravity SDK が環境にインストールされていません。")
    except Exception as e:
        err_msg = str(e)
        if "API key is required" in err_msg or "API key not valid" in err_msg:
            print(f"  [Deep Research Notice] Antigravity SDK の実行には GEMINI_API_KEY が必要です: {err_msg}")
        else:
            print(f"  [Deep Research Agent] SDK実行エラー: {e}")
            
    return None

def run_antigravity_sdk_research(ticker: str) -> Optional[List[Tuple[str, str]]]:
    """SDKリサーチの非同期実行ラッパー"""
    try:
        import asyncio
        return asyncio.run(run_antigravity_sdk_research_async(ticker))
    except Exception as e:
        print(f"  [Deep Research Agent] 非同期実行エラー: {e}")
        return None

def fetch_financial_catalysts_yfinance(ticker: str) -> List[Tuple[str, str]]:
    """
    yfinance から過去の四半期決算サプライズ (Reported vs Consensus)、株式分割、ニュースを自動抽出する
    """
    print(f"  [Financial Fallback Engine] yfinance から '{ticker}' の確定決算・企業イベントを抽出中...")
    catalysts = []
    try:
        import yfinance as yf
        import pandas as pd
        t = yf.Ticker(ticker)
        
        # 1. 四半期決算日・サプライズ実績
        try:
            ed = t.get_earnings_dates(limit=16)
            if ed is not None and not ed.empty:
                for idx, row in ed.iterrows():
                    dt_str = idx.strftime("%Y-%m-%d")
                    # 2024年以降を対象
                    if dt_str < "2024-01-01":
                        continue
                    reported = row.get("Reported EPS")
                    est = row.get("EPS Estimate")
                    surp = row.get("Surprise(%)")
                    if pd.notna(reported) and pd.notna(est):
                        surp_text = f" ({surp:+.1f}% surprise)" if pd.notna(surp) else ""
                        headline = f"{ticker} Q earnings: Reported EPS ${reported:.2f} vs consensus ${est:.2f}{surp_text}."
                        catalysts.append((dt_str, headline))
        except Exception as e:
            pass

        # 2. 株式分割イベント
        try:
            splits = t.splits
            if splits is not None and not splits.empty:
                for dt, ratio in splits.items():
                    dt_str = dt.strftime("%Y-%m-%d")
                    if dt_str >= "2024-01-01":
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
                        dt_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
                        catalysts.append((dt_str, f"{ticker}: {title}"))
        except Exception:
            pass

    except Exception as e:
        print(f"  [Financial Fallback Engine] yfinance 取得エラー: {e}")

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

def run_deep_research(ticker: str) -> List[Tuple[str, str]]:
    """
    ディープリサーチ実行:
    1. Antigravity ローカルログインセッション (CLI: agy) による自律調査（APIキー不要）
    2. Antigravity Python SDK (google.antigravity with API key)
    3. yfinance 確定決算・開示イベントによる自動抽出フォールバック
    """
    ticker = ticker.upper().strip()
    
    # Engine 1: Antigravity ローカルログインセッション (agy)
    session_results = run_antigravity_session_research(ticker)
    if session_results:
        return session_results

    # Engine 2: Antigravity Python SDK (GEMINI_API_KEY 設定時)
    sdk_results = run_antigravity_sdk_research(ticker)
    if sdk_results:
        return sdk_results

    # Engine 3: yfinance 確定決算サプライズ自動抽出フォールバック
    yf_results = fetch_financial_catalysts_yfinance(ticker)
    if yf_results:
        return yf_results

    print(f"  [Deep Research Agent] '{ticker}' のカタリスト自動抽出が完了しませんでした")
    return []

def get_ticker_catalysts(ticker: str, enable_deep_research: bool = False, force_refresh: bool = False) -> List[Tuple[str, str]]:
    """キャッシュまたはディープリサーチからカタリストを取得する"""
    ticker = ticker.upper().strip()
    existing = load_catalysts(ticker)
    
    if force_refresh:
        return run_deep_research(ticker)

    if existing:
        if enable_deep_research:
            print(f"[{ticker}] 既存の確定カタリスト ({len(existing)} 件) を使用します (再調査時は --force を指定)")
        return existing

    if enable_deep_research:
        return run_deep_research(ticker)
    
    # ディープリサーチ未指定でもキャッシュがない場合は案内
    print(f"[{ticker}] 確定カタリストキャッシュがありません ('--deep-research' を指定すると自律AIリサーチが実行されます)")
    return []
