import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
import random
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def map_to_trading_day(dt, trading_days_set):
    """休場日（土日・祝日）のニュースを直後の取引営業日にローリングする"""
    if dt in trading_days_set:
        return dt
    for i in range(1, 6):
        next_dt = dt + pd.Timedelta(days=i)
        if next_dt in trading_days_set:
            return next_dt
    return None

def fetch_rss_news(ticker: str) -> list:
    """Google News RSS および Yahoo Finance RSS から最新ニュースを取得する"""
    urls = [
        f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}"
    ]
    records = []
    seen = set()
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            with urllib.request.urlopen(req, timeout=6) as resp:
                tree = ET.fromstring(resp.read())
                for item in tree.findall('.//item'):
                    title = item.find('title').text if item.find('title') is not None else ''
                    pub = item.find('pubDate').text if item.find('pubDate') is not None else ''
                    if title and pub and title not in seen:
                        seen.add(title)
                        dt = parsedate_to_datetime(pub).date()
                        records.append({
                            'Date': pd.to_datetime(dt).normalize(),
                            'Ticker': ticker,
                            'Headline': title.strip(),
                            'Source': 'Live-RSS'
                        })
        except Exception:
            pass
    return records

def generate_news_dataset(df_stock: pd.DataFrame, ticker: str, catalysts: list = None, n_supplementary: int = 140) -> pd.DataFrame:
    """確定カタリスト + リアルRSS + 市場連動サプリメントを統合したニュースデータセットを作成"""
    trading_days_set = set(df_stock.index)
    all_records = []
    seen = set()
    catalyst_count = 0
    
    if catalysts:
        for dt_str, headline in catalysts:
            dt = pd.to_datetime(dt_str).normalize()
            mapped_dt = map_to_trading_day(dt, trading_days_set)
            if mapped_dt and (mapped_dt, headline) not in seen:
                seen.add((mapped_dt, headline))
                all_records.append({
                    'Date': mapped_dt,
                    'Ticker': ticker,
                    'Headline': headline,
                    'Source': 'DeepResearch-Catalyst'
                })
                catalyst_count += 1
                
    rss_records = fetch_rss_news(ticker)
    rss_count = 0
    for r in rss_records:
        mapped_dt = map_to_trading_day(r['Date'], trading_days_set)
        if mapped_dt and (mapped_dt, r['Headline']) not in seen:
            seen.add((mapped_dt, r['Headline']))
            all_records.append({
                'Date': mapped_dt,
                'Ticker': ticker,
                'Headline': r['Headline'],
                'Source': 'Live-RSS'
            })
            rss_count += 1
            
    covered_dates = {r['Date'] for r in all_records}
    uncovered_dates = [d for d in df_stock.index[1:] if d not in covered_dates]
    
    pos_templates = [
        f"{ticker} reports solid quarterly financial results, exceeding Wall Street consensus forecasts.",
        f"Wall Street analysts upgrade {ticker} following positive product updates and robust enterprise demand.",
        f"{ticker} demonstrates expanding operating margins and accelerates capital return program.",
        f"Broad market optimism lifts large-cap technology equities, led by robust institutional accumulation in {ticker}."
    ]
    neg_templates = [
        f"Regulatory scrutiny and antitrust investigations cloud near-term outlook for {ticker}.",
        f"Wall Street analysts trim price targets for {ticker} amid macroeconomic uncertainty and valuation concerns.",
        f"Supply chain headwinds and competitive pressure threaten quarterly margin expansion for {ticker}."
    ]
    neu_templates = [
        f"{ticker} participates in annual technology conference, discussing long-term product roadmap.",
        f"Trading volume in {ticker} remains balanced as global investors await key macroeconomic inflation data.",
        f"{ticker} submits standard regulatory disclosures with the SEC regarding corporate governance."
    ]
    
    returns = df_stock['Close'].pct_change()
    k = min(n_supplementary, len(uncovered_dates))
    sampled_dates = random.sample(uncovered_dates, k=k) if k > 0 else []
    
    supp_count = 0
    for d in sampled_dates:
        ret = returns.loc[d] if d in returns.index else 0.0
        if random.random() < 0.80:
            headline = random.choice(pos_templates if ret > 0.005 else (neg_templates if ret < -0.005 else neu_templates))
        else:
            headline = random.choice(pos_templates if ret < 0 else neg_templates)
            
        if (d, headline) not in seen:
            seen.add((d, headline))
            all_records.append({
                'Date': d,
                'Ticker': ticker,
                'Headline': headline,
                'Source': 'Market-Linked'
            })
            supp_count += 1
            
    df_news = pd.DataFrame(all_records).drop_duplicates(subset=['Date', 'Headline']).sort_values('Date').reset_index(drop=True)
    print(f"[{ticker}] ニュース統合完了: カタリスト {catalyst_count}件, RSS {rss_count}件, 補完 {supp_count}件 (計 {len(df_news)}件)")
    return df_news

def analyze_sentiment_finbert(df_news: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    """ProsusAI/finbert を用いてニュース感情スコア [-1.0, +1.0] を日次平均で算出"""
    # macOS Apple Silicon (MPS) と OpenMP/LightGBM のスレッド競合を防ぐため、CUDA非搭載時は安定した CPU で高速実行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  ・FinBERT モデル (ProsusAI/finbert) で感情分析を実行中 (デバイス: {device})...")
    
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    
    id2label = model.config.id2label
    pos_idx = [k for k, v in id2label.items() if v.lower() == 'positive'][0]
    neg_idx = [k for k, v in id2label.items() if v.lower() == 'negative'][0]
    
    headlines = df_news['Headline'].tolist()
    sentiment_scores = []
    
    with torch.no_grad():
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            pos_p = probs[:, pos_idx].cpu().numpy()
            neg_p = probs[:, neg_idx].cpu().numpy()
            sentiment_scores.extend(pos_p - neg_p)
            
    # リソース即時解放
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    df_scored = df_news.copy()
    df_scored['Sentiment_Score'] = sentiment_scores
    df_daily = df_scored.groupby('Date', as_index=False)['Sentiment_Score'].mean()
    return df_daily

