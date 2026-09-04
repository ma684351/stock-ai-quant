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

## 🧠 アーキテクチャの4本柱

1. **個別株テクニカル指標 (`core/features.py`)**:
   - 1日・5日リターン、出来高変化率、5日・20日移動平均乖離率、20日ボラティリティ、14日RSI。
2. **マクロ経済指標 (`core/data_loader.py`)**:
   - 米国市場の地合い（S&P 500: `^GSPC`）、グローバル為替（ドル円: `JPY=X`）、時差先行指標（日経平均: `^N225`）。
3. **ニュース感情分析 (`core/sentiment.py`)**:
   - 確定ヒストリカル・カタリスト ＋ リアルタイム金融RSS（Google News & Yahoo Finance）を `ProsusAI/finbert` で分析。
4. **ファンダメンタルズ財務指標 (`core/data_loader.py`)**:
   - 動的PER（リアルタイム割高・割安倍率）、株式益回り（1/PER）、四半期売上高成長率（YoY）、四半期純利益率、決算サプライズ率。
