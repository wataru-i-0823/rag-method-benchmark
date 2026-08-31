# プロンプトの運用

このフォルダは、検索・回答・GraphRAGで使う日本語プロンプトをコードから分離して管理する。`{{variable}}`は呼び出し側で置換する変数である。

| ファイル | 用途 | 主な変数 |
| --- | --- | --- |
| `hyde_ja.md` | 検索時の仮想文書生成 | `question` |
| `reverse_hyde_ja.md` | 文書ごとの仮想質問生成 | `title`, `context` |
| `rag_answer_ja.md` | 根拠付き最終回答 | `question`, `contexts` |
| `corpus2skill_cluster_summary_ja.md` | クラスタ／Skillノードの要約 | `contexts` |
| `graphrag_triple_extraction_ja.md` | エンティティ・関係抽出 | `source_id`, `page_number`, `context` |
| `graphrag_community_summary_ja.md` | グラフコミュニティの要約 | `entities`, `relationships` |

プロンプトを変更した実験では、ファイル内容のSHA-256とモデル名をMLflowのパラメータに記録する。評価時は、コーパス、チャンク設定、埋め込みモデル、検索方式、`k`を固定し、プロンプトだけを変更する。
