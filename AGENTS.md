# RAG手法比較ラボ — エージェントガイド

RAGの取り込み、検索、評価、実行環境、Colabワークフローを変更する作業では、最初に [`skills/rag-lab-workflow/SKILL.md`](skills/rag-lab-workflow/SKILL.md) を読むこと。

## プロジェクトの境界

- コーパス／チャンク分割、埋め込みモデル、ベクトルストア、検索手法、オーケストレーション、評価は独立して選択・比較できる状態を保つ。
- 実行プロファイルを用途に応じて選ぶ。`local` はCPUのみの学習用、`colab` はGPUでのE5実験用、`cloud` は永続ストレージをマウントした環境用である。ローカルのベースラインにGPUやAPIキーを必須にしない。
- `data/raw/`、`data/processed/`、`data/evaluation/`、`data/chroma/`、`mlruns/`、`.env`、APIキーは非公開データとして扱う。内容をコミットまたは公開しない。
- 一時的な実行環境では、Chromaをコーパスと設定から再構築する。実行中のインデックスを環境間でコピーするのではなく、入力データと評価結果を保存する。

## 検証と引き渡し

- Python環境と依存関係には`uv`を使う。`pyproject.toml`の対応Pythonバージョンとドキュメントを一致させる。
- Retriever、プロファイルの振り分け、評価指標を変更した場合は、決定論的なテストを追加または更新する。プロジェクト環境で`python -m unittest discover -s tests -v`を実行する。
- Colab経路を変更する場合は、[`notebooks/rag_lab_colab.ipynb`](notebooks/rag_lab_colab.ipynb) が新規ランタイムでも実行でき、指定されたGoogle Driveのラボ用フォルダ以外へデータを書き込まないことを確認する。
- 公開GitHubへpushする前に、ステージ済み差分にシークレットと非公開データセットが含まれないことを確認する。外部公開には引き続きユーザーの認可が必要である。
