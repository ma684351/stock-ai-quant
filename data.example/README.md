# data.example / データディレクトリ設定サンプル

本ディレクトリ（`data.example`）は、AI投資診断パイプラインで利用する外部マッピング設定や確定カタリストのサンプル構成です。

## セットアップ手順

プロジェクトをクローンした直後や初期セットアップ時は、このディレクトリをコピーして `data` ディレクトリを作成してください。

```bash
cp -r data.example data
```

※ `data/` ディレクトリは `.gitignore` に登録されており、実行時に自動生成される日次キャッシュ（Wikimedia PV、ATS求人数等）は Git にコミットされません。

## ディレクトリ構成

- **`wikipedia_mapping.json`**:
  ティッカーシンボルと Wikipedia 記事名（言語コード + 記事名）の対応辞書。Wikimedia Pageviews API で投資家関心度を取得する際に利用されます。
- **`jobs_mapping.json`**:
  企業の採用求人ATS（Greenhouse / Lever / Workday）の設定辞書。リアルタイム求人数を取得する際に利用されます。
- **`catalysts/`**:
  過去の重要カタリスト（決算サプライズ、新製品、大型開示等）のJSONファイル。Antigravityエージェント（スキル）により自動生成・更新されます。
- **`attention/`**:
  Wikipedia日次アクセス数の自動キャッシュ（実行時に自動生成）。
- **`jobs/`**:
  ATS日次求人数の自動キャッシュ（実行時に自動生成）。
