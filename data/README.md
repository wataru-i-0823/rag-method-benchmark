# データ配置

実データはGitに含めず、次のフォルダに配置します。`.gitignore` により意図せず公開リポジトリへpushされません。

| フォルダ | 用途 | Git管理 |
| --- | --- | --- |
| `raw/` | 原本（PDF変換前テキスト、CSV、JSONLなど） | しない |
| `processed/` | チャンク化・正規化済みコーパス | しない |
| `evaluation/` | 実データ向け質問・正解ラベル | しない |
| `chroma/` | Chromaのローカル永続インデックス | しない |
| 直下の `example_*.jsonl` | 動作確認用の架空データ | する |

コーパスはJSONLで各行に `id`、`text`、任意で `title` を含めます。

```json
{"id":"policy-001","title":"休暇規程","text":"..."}
```

評価セットは `id`、`question`、正解文書ID配列の `relevant_ids` を含めます。

```json
{"id":"q-001","question":"...","relevant_ids":["policy-001"]}
```

個人情報・APIキー・認証情報・社外秘文書はコミットしないでください。実行前に `git status --ignored` で除外状態を確認できます。
