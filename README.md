# 米国株AI投資判断システム (Stock AI Quant Pipeline)

金融感情分析モデル（FinBERT）、マクロ経済指標（S&P 500、ドル円、日経平均）、テクニカル指標、およびYahoo!ファイナンスのファンダメンタルズ財務データを統合し、各銘柄に個別最適化したLightGBMモデルで**今後1ヶ月（20営業日）の株価トレンド（上昇 / 下落）を予測**し、**最新営業日の投資判断（【買い】または【見送り】）**を自動判定する汎用AIクオンツシステムです。

---

## 📁 ディレクトリ構成 (Pattern A: モジュール設計)

```text
fin-sentiment-lgbm-pipeline/
├── stock_ai.py                 # 【メインCLI】全銘柄対応の分析・比較スクリプト
│
├── core/                       # コアロジック・パッケージ
│   ├── __init__.py
│   ├── research_agent.py       # 【Antigravity SDK】新銘柄のカタリスト自律リサーチ
│   ├── data_loader.py          # yfinance (株価・マクロ3指標・四半期財務) の自動取得
│   ├── sentiment.py            # リアルタイムRSS収集 & FinBERT感情分析
│   ├── features.py             # 4大カテゴリ（テクニカル×マクロ×感情×財務）の特徴量生成
│   └── model.py                # LightGBM個別学習、最適閾値探索、本日の投資判断推論
│
├── data/                       # データのキャッシュ・蓄積
│   ├── catalysts/              # 各銘柄の過去2年カタリスト (AAPL.json, NVDA.json, GOOGL.json...)
│   └── fundamentals/           # 四半期財務データ (AAPL.json, NVDA.json, GOOGL.json...)
│
├── notebooks/                  # Google Colab用個別ノートブック
│   ├── apple_stock_ai_pipeline.ipynb       # Apple (AAPL) 専用ノートブック
│   ├── nvidia_stock_ai_pipeline.ipynb      # NVIDIA (NVDA) 専用ノートブック
│   └── alphabet_stock_ai_pipeline.ipynb    # Alphabet (GOOGL) 専用ノートブック
│
├── .venv/                      # 独立仮想環境 (Python 3)
├── requirements.txt            # 依存パッケージ一覧
├── .gitignore                  # 除外設定
└── README.md                   # 本ドキュメント
```

---

## 🚀 実行方法（ローカル venv環境）

ローカルのグローバル環境を一切汚さずに、独立した仮想環境（`.venv`）で実行します。

### 1. 仮想環境の有効化

```bash
# プロジェクトフォルダへ移動
cd /Users/masami/project/fin-sentiment-lgbm-pipeline

# 仮想環境を有効化 (macOS / Linux)
source .venv/bin/activate
# ※ Windows (PowerShell) の場合: .venv\Scripts\Activate.ps1
```

### 2. コマンドライン実行

#### ① 単一銘柄の分析・最新シグナル診断
```bash
# 好きな銘柄のティッカーを指定（例: Apple, NVIDIA, Alphabet, Microsoft, Tesla など）
python stock_ai.py AAPL
python stock_ai.py NVDA
python stock_ai.py GOOGL
python stock_ai.py MSFT
python stock_ai.py TSLA
```

#### ② 複数銘柄の一括比較 ＆ ランキング一覧表示（★おすすめ）
気になる複数銘柄を順番に自動学習・推論し、**「今買うべき株ランキング一覧表」** を出力します。
```bash
python stock_ai.py --compare AAPL NVDA GOOGL MSFT TSLA AMZN META
```

出力例:
```text
========================================================================================
【AI投資判断 複数銘柄比較ランキングサマリー（今後1ヶ月の予測）】
========================================================================================
順位  銘柄    現在株価    動的PER   14日RSI   20日乖離    1ヶ月上昇確率   判定結果
----------------------------------------------------------------------------------------
 1位  GOOGL  $338.67    33.7倍      44.5     -1.4%           64.9%   ★【 買い (BUY) 】
 2位  AAPL   $320.70    39.5倍      64.5     +2.4%           53.8%   ◇【 見送り (HOLD) 】
 3位  NVDA   $231.17    49.2倍      54.2     +5.0%           30.3%   ◇【 見送り (HOLD) 】
========================================================================================
```

#### ③ Antigravity 自律ディープリサーチ (`--deep-research` / `--force`)
新しい銘柄や未登録銘柄で過去の確定イベント・決算カタリストを自律調査させたい場合：
```bash
# ローカル環境では API キー設定不要！Antigravity の既存ログインセッションを使って自動実行されます
python stock_ai.py TSLA --deep-research

# 既存キャッシュを無視して最新情報で再リサーチしたい場合
python stock_ai.py TSLA --deep-research --force

# （※CIやクラウド環境等で API キーを使用する場合は GEMINI_API_KEY を設定可能）
# export GEMINI_API_KEY="your-gemini-api-key"
```
* **ローカルログインセッション自動連携**: ローカル環境（Antigravity CLI / IDE）でログイン済みのセッションを自動認識し、APIキーなしで即座に自律ディープリサーチを実行します。
* **高可用性フォールバック**:
  1. ローカル Antigravity ログインセッション (`agy`)
  2. Antigravity Python SDK (`google-antigravity` + `GEMINI_API_KEY`)
  3. yfinance 財務開示自動抽出エンジン (オフライン/財務開示フォールバック)
* 調査された確定カタリストは `data/catalysts/{TICKER}.json` に自動保存され、2回目以降はキャッシュから高速ロードされてLightGBMとFinBERTの特徴量生成に即座に活用されます。

#### ④ 引数なしの対話型モード
```bash
python stock_ai.py
# プロンプトが表示されます:
# 分析したいティッカーシンボルを入力してください (例: AAPL, NVDA, TSLA または複数カンマ区切り):
```

---

## 🌐 Google Colab で実行する場合

ブラウザ上でGPUを使って動かしたい場合は、`notebooks/` フォルダ内の各ノートブックをそのままドラッグ＆ドロップして「すべてのセルを実行」してください。

* [**`apple_stock_ai_pipeline.ipynb`**](notebooks/apple_stock_ai_pipeline.ipynb)
* [**`nvidia_stock_ai_pipeline.ipynb`**](notebooks/nvidia_stock_ai_pipeline.ipynb)
* [**`alphabet_stock_ai_pipeline.ipynb`**](notebooks/alphabet_stock_ai_pipeline.ipynb)

---

## 🧠 モデル入力特徴量一覧（4大カテゴリ＋複合指標）

LightGBMモデルは、以下の4大カテゴリ（テクニカル×マクロ×感情×財務）および複合相互作用から構成される特徴量を用いて、今後1ヶ月（20営業日後）の株価上昇確率を推論します。

### 1. ファンダメンタルズ財務指標 (`core/data_loader.py`, `core/features.py`)
企業の稼ぐ力とバリュエーション（割安・割高）を日次株価と結合して動的に捉えます。

| 特徴量名 (Feature) | 内容・計算式 | クオンツ的解釈・意義 |
| :--- | :--- | :--- |
| **`Fund_Dynamic_PE`** | 当日終値 / 直近12ヶ月希薄化EPS (TTM) | **動的PER（株価収益率）**。静的な決算時PERではなく、毎日の終値変動に連動したリアルタイムの割高・割安度。 |
| **`Fund_Earnings_Yield`** | 1 / 動的PER = TTM EPS / 当日終値 | **株式益回り**。債券利回り等と比較可能な投資リターン利回り換算値。 |
| **`Fund_Rev_Growth_YoY`** | (当四半期売上高 - 前年同期売上高) / 前年同期売上高 | **四半期売上高成長率 (YoY)**。トップライン（事業規模）の拡大スピード。 |
| **`Fund_Net_Margin`** | 当期純利益 / 売上高 | **四半期純利益率**。売上を最終利益に残す収益性・マージン効率。 |
| **`Fund_Operating_Margin`**| 営業利益 / 売上高 | **四半期営業利益率**。本業の稼ぐ力。 |
| **`Fund_Earnings_Surprise`**| (実績EPS - アナリスト予想コンセンサス) / \|予想EPS\| | **決算サプライズ率**。市場予想に対する上振れ・下振れショック度合い。 |

### 2. 個別株テクニカル指標 (`core/features.py`)
株価のモメンタム（勢い）、移動平均との乖離、過熱感、ボラティリティを計測します。

| 特徴量名 (Feature) | 内容・計算式 | クオンツ的解釈・意義 |
| :--- | :--- | :--- |
| **`{TICKER}_Return_1d`** | 前日比変化率 ($P_t / P_{t-1} - 1$) | 日次モメンタム。 |
| **`{TICKER}_Return_5d`** | 5営業日前比変化率 ($P_t / P_{t-5} - 1$) | 週間スイングトレンド。 |
| **`{TICKER}_Volume_Change_1d`** | 出来高の前日比変化率 | 出来高の急増（大口機関投資家の参入・手仕舞いシグナル）。 |
| **`{TICKER}_MA5_Ratio`** | 終値 / 5日移動平均線 - 1 | 短期トレンドに対する株価の位置。 |
| **`{TICKER}_MA20_Ratio`** | 終値 / 20日移動平均線 - 1 | 中期トレンド（約1ヶ月線）に対する株価乖離率（リバーサル・過熱目安）。 |
| **`{TICKER}_Volatility_20d`** | 20日リターンの標準偏差 × $\sqrt{252}$ | 20営業日ヒストリカル・ボラティリティ（リスク水準・市場の不安心理）。 |
| **`{TICKER}_RSI_14`** | 14日相対力指数 (RSI) | 70以上で「買われすぎ（過熱警戒）」、30以下で「売られすぎ（底値圏）」。 |

### 3. グローバル・マクロ経済指標 (`core/data_loader.py`)
個別株単体では抗えない市場全体の地合い・為替・時差先行シグナルを捉えます。

| 特徴量名 (Feature) | 対象資産・指標 | クオンツ的解釈・意義 |
| :--- | :--- | :--- |
| **`SP500_Return_1d` / `5d`** | S&P 500指数 (`^GSPC`) | 米国株式市場全体の地合い・リスクオン/オフの波。 |
| **`USDJPY_Return_1d` / `5d`** | ドル円為替レート (`JPY=X`) | グローバル資金シフト・金利動向・為替リスク。 |
| **`Nikkei_Return_1d` / `5d`** | 日経平均株価 (`^N225`) | 米国市場オープン前に取引されるアジア市場の先行シグナル。 |

### 4. 金融ニュース感情指標 (FinBERT) & 複合指標 (`core/sentiment.py`, `core/features.py`)
自然言語処理モデル `ProsusAI/finbert` が金融テキストから抽出した感情スコア（-1.0〜+1.0）とテクニカル指標の相互作用。

| 特徴量名 (Feature) | 内容・計算式 | クオンツ的解釈・意義 |
| :--- | :--- | :--- |
| **`News_Sentiment_Score`** | 当日ニュースのFinBERT加重感情値 | 当日ヘッドラインのポジティブ（+）/ ネガティブ（-）度。 |
| **`News_Sentiment_MA3d`** | 感情スコアの3日移動平均 | 感情の慣性・トレンド（悪材料/好材料の継続性）。 |
| **`News_Sentiment_Lag1`** | 1営業日前の感情スコア | ニュース報道の翌日市場への浸透ラグ効果。 |
| **`News_Sentiment_x_RSI`** | 感情スコア × ((RSI - 50) / 25) | **感情×過熱度の相互作用**。過熱圏（RSI高）での好材料は天井打ちリスク、底値圏での好材料は急反発シグナル。 |
| **`News_Sentiment_x_Return5d`**| 感情スコア × 5日リターン | 株価の勢いとニュースの方向性の一致度（モメンタム追随か逆行か）。 |
| **`News_Sentiment_Surprise`** | 当日スコア - 3日移動平均 | **感情サプライズ**。平穏な状態から突如好材料/悪材料が出たインパクト。 |
