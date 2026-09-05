---
name: stock-ai-analysis
description: >-
  日米株式（日本株・東証銘柄および米国株）のAI投資診断を実行し、今後1ヶ月の株価トレンド予測と3段階投資シグナル（★買い / ◇様子見 / ▼売り）を判定するスキル。
  銘柄の投資判断、株価予測、複数銘柄のランキング比較、および決算・開示の自律ディープリサーチを実行・解釈する際に使用する。
---

# 日米株AI投資診断スキル (Stock AI Analysis)

このスキルは、[`stock_ai.py`](file:///Users/masami/project/fin-sentiment-lgbm-pipeline/stock_ai.py) を使用して日米株式のマルチモーダルAI投資診断（テクニカル×マクロ経済×金融感情分析×財務ファンダメンタルズ）を実行し、今後1ヶ月（20営業日）のトレンド予測および3段階投資シグナルを判定・分析するための手順書です。

---

## 1. 実行環境と事前準備

本プロジェクトの仮想環境（`.venv`）内の Python インタプリタを使用します。

```bash
# 作業ディレクトリ
cd /Users/masami/project/fin-sentiment-lgbm-pipeline

# 仮想環境 Python のパス
.venv/bin/python stock_ai.py [引数...]
```

---

## 2. コマンドライン実行パターン

### ① 単一銘柄のAI投資診断
日本株は4桁数字コード（例: `7203`, `7974`, `2122`）または東証ティッカー（`7203.T`）、米国株はティッカーシンボル（`AAPL`, `NVDA`, `TSLA` など）を指定します。

```bash
# 日本株（自動的に 7203.T として認識、日本語金融BERTで感情分析）
.venv/bin/python stock_ai.py 7203

# 米国株（FinBERT で感情分析）
.venv/bin/python stock_ai.py AAPL
```

### ② 複数銘柄の一括比較・ランキング出力 (`--compare`)
気になる複数銘柄（日米混在可）を一括して個別学習・推論し、**「1ヶ月上昇確率順のランキング表」** を出力します。

```bash
.venv/bin/python stock_ai.py --compare 7203 2122 AAPL 7974
```

### ③ Antigravity 自律ディープリサーチ (`--deep-research`)
決算サプライズ・東証適時開示・為替影響などの過去カタリストを自律調査します。
**`--deep-research` を指定すると、既存キャッシュの有無に関わらず問答無用で最新調査を実行し、キャッシュを上書きします。**

```bash
# 新銘柄や、直近の適時開示・決算情報を最新化して診断したい場合
.venv/bin/python stock_ai.py 7203 --deep-research
.venv/bin/python stock_ai.py NVDA --deep-research
```

---

## 3. 出力結果の読み方と評価基準

### 投資シグナル判定（3段階）
小幅なノイズ相場での往復ビンタを防ぐため、中央に適正な中立帯（HOLD）を自動確保しています。

| シグナル | 状態 | 推奨アクション |
| :--- | :--- | :--- |
| **★【 買い (BUY) 】** | 上昇確率 $\ge$ 買い閾値 | 上昇トレンド予測。新規買い・押し目買い推奨 |
| **◇【 様子見 (HOLD) 】** | 中立レンジ | 方向感に乏しく様子見推奨（エントリー見送り） |
| **▼【 売り (SELL) 】** | 下落確率 $\ge$ 売り閾値 | 下落リスク警戒。利益確定・損切り、または空売り検討 |

### 過去テストデータ評価指標
- **ROC-AUC**: 0.50以上がランダムを上回る基準。0.60〜0.70超であれば高い予測力。
- **混同行列**:
  - `TN`: 下落予測的中
  - `FP`: 上昇誤予測（買い倒れ）
  - `FN`: 下落誤予測（買い逃し）
  - `TP`: 上昇予測的中
- **特徴量重要度 Top 8**:
  - `Fund_PE_Ratio_to_MA200`, `Fund_PE_ZScore`: バリュエーション過熱・割安度（定常化指標）
  - `Fund_Rev_Growth_YoY`: 売上高成長率
  - `Volatility_20d`, `RSI_14`: ボラティリティとモメンタム
  - `Nikkei_Return_5d`, `USDJPY_Return_5d`, `SP500_Return_5d`: マクロ環境

---

## 4. データ保存場所とキャッシュ構造

- **確定カタリスト**: `data/catalysts/{TICKER}.json`
  - 日本株は `7203_T.json`、米国株は `AAPL.json` の形式で保存。
  - `--deep-research` 実行時に自動更新されます。
- **四半期財務データ**: `data/fundamentals/{TICKER}.json`
  - yfinance から取得した年次・四半期財務諸表を自動補間して保存。
