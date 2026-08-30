# 実行プロファイル

プロファイルは`rag_lab/profiles.py`で定義し、`--profile`で選択する。

| プロファイル | 埋め込み経路 | 保存ルール | 想定用途 |
| --- | --- | --- | --- |
| `local` | 決定論的なCPUベースラインである`chroma_hash` | `data/chroma/`はローカル保存・Git追跡外 | パイプラインの学習とGPU不要な手法比較 |
| `colab` | Sentence Transformersによる`intfloat/multilingual-e5-small`を使う`chroma_e5` | ColabでChromaを再構築し、コーパス・結果・MLflowトラッキングはGoogle Driveへ保存 | GPUでの意味検索実験 |
| `cloud` | GPUがあればGPU、なければCPUで`chroma_e5` | 永続ボリュームを指す`--chroma-path`を指定 | 本番に近い実験または定期実行 |

プロファイルを検索手法として扱わないこと。プロファイルはリソースとデフォルト手法を選ぶものであり、呼び出し側は`--methods`で統制した比較対象を指定できる。

新しいプロファイルを追加する場合は、`rag_lab/profiles.py`でデフォルト手法、Chromaのパス、埋め込みデバイスを定義し、READMEへ記載する。APIキーや外部モデルのダウンロードを必要としないプロファイル振り分けテストも追加する。
