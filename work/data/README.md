# データ配置

実データはGitに含めず、次のフォルダに配置します。`.gitignore` により意図せず公開リポジトリへpushされません。

| フォルダ | 用途 | Git管理 |
| --- | --- | --- |
| `raw/` | 原本（PDF変換前テキスト、CSV、JSONLなど） | しない |
| `processed/` | チャンク化・正規化済みコーパス | しない |
| `evaluation/` | 実データ向け質問・正解ラベル | しない |
| `chroma/` | Chromaのローカル永続インデックス | しない |

コーパスはJSONLで各行に `id`、`text`、任意で `title` を含めます。

```json
{"id":"policy-001","title":"休暇規程","text":"..."}
```

評価セットは `id`、`question`、正解文書ID配列の `relevant_ids` を含めます。

```json
{"id":"q-001","question":"...","relevant_ids":["policy-001"]}
```

個人情報・APIキー・認証情報・社外秘文書はコミットしないでください。実行前に `git status --ignored` で除外状態を確認できます。

## 政府の公開情報を取得する

`config/public_sources/government_public_information.json`には、金融庁・日本銀行の公開ページURLだけを記録しています。本文はGitに含めません。以下を実行すると、原文HTMLを`raw/`、検索用コーパスを`processed/`に保存します。

```bash
uv run python scripts/download_public_sources.py
```

公開情報でも内容や掲載条件は変わり得るため、評価結果には取得日時とURLを併記してください。データを第三者へ再配布する場合は、元サイトの利用条件を確認してください。

検索結果を確認する例:

```bash
uv run python -m rag_lab inspect --profile local \
  --corpus data/processed/fsa_boj_public_information.jsonl \
  --query "日本銀行の物価安定の目標は何ですか" --method hybrid
```

精度評価を行う際は、`evaluation/`に質問と正解文書IDを自分で作成し、`evaluate`コマンドへ渡します。これらのファイルもGitには追加しません。
