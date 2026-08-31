# データ配置

実データはGitに含めず、次のフォルダに配置します。`.gitignore` により意図せず公開リポジトリへpushされません。

| フォルダ | 用途 | Git管理 |
| --- | --- | --- |
| `raw/` | 原本（PDF変換前テキスト、CSV、JSONLなど） | しない |
| `processed/` | チャンク化・正規化済みコーパス | しない |
| `evaluation/` | 実データ向け質問・正解ラベル | しない |
| `chroma/` | Chromaのローカル永続インデックス | しない |

PDF原本は`raw/`に置きます。`scripts/ingest_pdfs.py`はまずPyMuPDFでPDF内部の文字を直接抽出し、文字が不足するページだけ任意でPaddleOCRを実行します。抽出済みJSONLも`processed/`に保存され、いずれもGit管理しません。

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

## PDFを取り込む

文字情報を持つPDFでは、OCRを使わずPyMuPDFの抽出結果を使います。

```bash
uv sync
uv run python scripts/ingest_pdfs.py data/raw/financial-report.pdf \
  --output data/processed/financial-report.jsonl
```

スキャンPDFなど、80文字未満しか取得できないページだけを日本語PaddleOCRで認識するには、Colab/LinuxでOCR追加依存関係を入れます。

```bash
uv sync --extra ocr
uv run python scripts/ingest_pdfs.py --input-dir data/raw/pdfs \
  --ocr-backend paddle --output data/processed/pdf_corpus.jsonl
```

出力の各行には`source_file`、`page_number`、`extraction_method`（`pymupdf`または`paddleocr`）を記録します。PDFの版やページを追跡できるよう、この情報は削除しないでください。
