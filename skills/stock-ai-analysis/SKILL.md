---
name: stock-ai-analysis
description: >-
  日米株式（日本株・東証銘柄および米国株）のAI投資診断を実行し、今後1ヶ月の株価トレンド予測と3段階投資シグナル（★買い / ◇様子見 / ▼売り）を判定するスキル。
  銘柄の投資判断、株価予測、複数銘柄のランキング比較、および決算・開示の自律ディープリサーチを実行・解釈する際に使用する。
  診断結果の提示時には、モデルの過去テストデータ評価結果（正解率、ROC-AUC、混同行列等）を参考情報として必ず表示する。
---

# 日米株AI投資診断スキル (Stock AI Analysis)

このスキルは、AIエージェント自身の強力な調査能力（Web検索・IR適時開示リサーチ）と、[`stock_ai.py`](stock_ai.py) のクオンツ機械学習エンジン（金融BERT×マクロ×テクニカル×財務LightGBM）を連携させ、日米株式の投資判断・1ヶ月株価予測・3段階シグナルを出力するための統合ワークフローです。

---

## 1. 全体アーキテクチャ（役割分担）

1. **AIエージェント（スキル）の役割**:
   - Web検索（`search_web`）やIR適時開示の調査を行い、最新の確定カタリスト（決算サプライズ、業績修正、大型新製品、自社株買い等）を自律抽出して `$DATA_DIR/catalysts/{TICKER}.json` に直接書き込みます。
2. **Python スクリプト (`stock_ai.py`) の役割**:
   - 生成されたカタリスト、リアルタイム株価・マクロ4指標（S&P 500/ドル円/日経平均/米10年債利回り TNX）、yfinance四半期財務データを統合し、金融BERT感情分析とLightGBM学習・閾値最適化・投資シグナル推論を高速実行します。

---

## 2. エージェントの自律実行手順

### Step 0: 実行環境とスクリプトパスの特定 (外部プロジェクト対応)

スキルがどのプロジェクトやディレクトリから呼び出されたかに応じて、実行リポジトリパス（`REPO_DIR`）を自動解決します。
**※ グローバル領域（$HOME 直下やシステム Python 等）には一切インストール・配置しません。**

1. **カレントディレクトリに `stock_ai.py` がある場合（本リポジトリ内での実行）**:
   - `REPO_DIR="."`
   - `DATA_DIR="data"`

2. **外部プロジェクトから呼び出された場合**:
   既存のローカルリポジトリパスを自動参照し、環境を一切汚さずに実行します：
   ```bash
   # リポジトリパスの解決（環境変数優先、または近隣ディレクトリ・Gitルートの自動検出）
   if [ -n "$STOCK_AI_DIR" ] && [ -d "$STOCK_AI_DIR" ]; then
     REPO_DIR="$STOCK_AI_DIR"
   elif [ -f "$(git rev-parse --show-toplevel 2>/dev/null)/stock_ai.py" ]; then
     REPO_DIR="$(git rev-parse --show-toplevel)"
   elif [ -f "../stock-ai-quant/stock_ai.py" ]; then
     REPO_DIR="../stock-ai-quant"
   elif [ -f "../fin-sentiment-lgbm-pipeline/stock_ai.py" ]; then
     REPO_DIR="../fin-sentiment-lgbm-pipeline"
   else
     # 手元にリポジトリがない新規マシン等の場合のみ、プロジェクトローカル配下に配置（グローバルは汚さない）
     REPO_DIR="./.stock-ai-quant"
     if [ ! -d "$REPO_DIR" ]; then
       git clone https://github.com/ma684351/stock-ai-quant.git "$REPO_DIR"
       cd "$REPO_DIR" && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cd -
     fi
   fi
   DATA_DIR="$REPO_DIR/data"
   ```

### Step 1: ディープリサーチ（カタリスト調査 & JSON生成）
**実行契機**:
- ユーザーから「〇〇をリサーチして」「最新開示を反映して」と指示された場合
- または、`$DATA_DIR/catalysts/{clean_ticker}.json` が存在しない新銘柄を診断する場合

**手順**:
1. 銘柄名・コード（例: トヨタ / 7203、ソニー / 6758、Apple / AAPL）を特定。
2. エージェント自身が `search_web` 等を使って、過去2年間の株価急変動・重要カタリストを15〜25件抽出：
   - 四半期決算発表（市場予想比、売上高、営業利益、通期上方修正・下方修正、増配、自社株買い）
   - 新製品・新技術発表、大型M&A、提携、主要設備投資
   - 不祥事、リコール、地政学・為替開示
3. `$DATA_DIR/catalysts/{clean_ticker}.json`（日本株例: `7203_T.json`、米国株例: `AAPL.json`）に以下のJSON配列として保存：
   ```json
   [
     ["2024-05-08", "2024年3月期通期決算発表、過去最高益更新および自社株買い発表"],
     ["2024-11-06", "2025年3月期第2四半期決算発表、円安効果とハイブリッド車好調で増益"]
   ]
   ```
4. **Wikipedia 記事名の確認と登録（Investor Attention 連携）**:
   - `$DATA_DIR/wikipedia_mapping.json` を開き、対象銘柄（ティッカーコード）が登録されているか確認。
   - 未登録の場合、Web検索等で正式な Wikipedia 記事名（例: 日本株なら `["ja", "メルカリ_(企業)"]`、米国株なら `["en", "Corning_Inc."]`) を特定し、`$DATA_DIR/wikipedia_mapping.json` に追記する。
   - ※ Pythonコード側に企業名をハードコードせず、スキル（エージェント）がこのJSONを自律管理します。
5. **採用求人ATSの確認と登録（Hiring Data 連携）**:
   - `$DATA_DIR/jobs_mapping.json` を開き、対象銘柄の求人ATS設定（Greenhouse / Lever / Workday）があるか確認。
   - 未登録の場合、企業の採用ページURL（例: `boards.greenhouse.io/{token}`, `jobs.lever.co/{token}`, `{tenant}.myworkdayjobs.com/...`）を特定し、`$DATA_DIR/jobs_mapping.json` に追記する。
   - ※ ATS非対応または独自採用サイトの企業は未登録のままで問題ありません（自動で中立値 0 件に安全フォールバック）。

### Step 2: AIクオンツ診断スクリプトの実行
リサーチが完了したら（または既存キャッシュを使用する場合）、仮想環境 Python で実行します。

```bash
# 【A. 本リポジトリ内で実行する場合 (カレントに stock_ai.py がある場合)】
.venv/bin/python stock_ai.py 7203
.venv/bin/python stock_ai.py --compare 7203 6758 AAPL 7974

# 【B. 外部プロジェクトから実行する場合 (REPO_DIR を参照)】
# ※ カレントディレクトリを変更せずに外部プロジェクトからそのまま実行可能
"$REPO_DIR/.venv/bin/python" "$REPO_DIR/stock_ai.py" 7203
"$REPO_DIR/.venv/bin/python" "$REPO_DIR/stock_ai.py" --compare 7203 6758 AAPL 7974
```

---

## 3. ユーザーへの回答フォーマット（必須4部構成）

診断結果をユーザーへ報告する際は、必ず以下の **4部構成** で提示してください。

### ① 📋 直近シグナル判定サマリー ＆ 実戦価格ガイド
- 最新株価、動的PER、売上高成長率(YoY)、20日乖離率、14日RSI、ニュース感情スコア
- 今後1ヶ月の予測確率（上昇% / 下落%）
- 3段階判定結果（★買い / ◇様子見 / ▼売り）
- **【AI実戦トレード価格ガイド】（★必須）**:
  - **買い時**: 推奨買いゾーン（押し目指値〜現在値）、1ヶ月目標株価（利確ターゲット）、防衛ライン（損切り目安）、リスクリワード比
  - **様子見時**: 押し目買い転換ライン（指値待ち）、上値ブレイクライン（順張り買いトリガー）
  - **売り時**: 戻り売りゾーン、下値ターゲット（買戻し目標）、踏み上げ防衛ライン

### ② 🔍 AIモデルのファクター分析
- 特徴量重要度 Top 5（Gain順）と、なぜその判断に至ったかの定性・クオンツ的根拠。

### ③ 📊 参考情報: AIモデルの過去バックテスト精度評価（★必須）
過去テストデータ（時系列分割で未来データとして評価）に対するモデルの汎化性能を明示します。
- **判定閾値 (Threshold)**
- **正解率 (Accuracy)**
- **適合率 (Precision)**: 「買い」と予測した局面が実際に上昇的中した割合
- **再現率 (Recall)** / **F1スコア (F1 Score)**
- **ROC-AUC スコア**: 0.50以上でランダムを上回る判別性能
- **混同行列 (Confusion Matrix)**: TN（下落的中）, FP（買い誤爆）, FN（買い逃し）, TP（上昇的中）
- **精度の総評**: この銘柄に対するモデル予測の信頼度コメント。

### ④ 💡 投資戦略・アクション推奨
- 現物保有者向け、新規買い検討者向け、アクティブトレーダー向けの具体的アクション。

---

## 4. ファイル管理

- **確定カタリスト**: `$DATA_DIR/catalysts/{TICKER}.json`（エージェントが自律調査・生成）
- **Wikipedia記事名マッピング**: `$DATA_DIR/wikipedia_mapping.json`（エージェントが自律追加・管理）
- **求人ATSマッピング**: `$DATA_DIR/jobs_mapping.json`（エージェントが自律追加・管理）
- **投資家関心（PV）キャッシュ**: `$DATA_DIR/attention/{TICKER}.json`（Wikimedia APIから自動保存）
- **求人数（Hiring）キャッシュ**: `$DATA_DIR/jobs/{TICKER}.json`（ATS公開APIから自動保存）
- **四半期財務データ**: `yfinance` から最新決算（売上高、純利益、EPS、利益率）を完全自動取得・動的補間
