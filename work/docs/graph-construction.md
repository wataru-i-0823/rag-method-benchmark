# グラフ構築方式の比較

このラボでは、グラフの**保存先**をSQLiteで共通化し、グラフを作る方式だけを切り替える。SQLiteファイルは`data/graph/`に保存し、Gitには追加しない。

| 方式 | ノード | エッジ | 無料 | 得意な問い |
| --- | --- | --- | --- | --- |
| `semantic` | チャンク | 埋め込み類似度 | はい | 類似した規程・関連箇所の探索 |
| `llm_knowledge_graph` | エンティティとチャンク | LLM抽出した関係 | API次第 | 複数文書をまたぐ因果・主体・制度の問い |

## 無料: Semantic Graph

`semantic`は、各チャンクをE5-smallまたはBGE-M3で埋め込み、コサイン類似度が閾値以上のチャンク同士を接続する。各チャンクの上位`max_neighbors_per_chunk`件だけを残すため、全結合にはしない。

```text
チャンクA ── 類似度 0.82 ── チャンクB
     └──── 類似度 0.67 ── チャンクC
```

質問時は通常のHybrid検索で起点チャンクを選び、その隣接チャンクを加点する。関係ラベルは持たないため、「誰が何を決定したか」自体を推論する用途ではなく、関連根拠の取りこぼしを減らす用途で評価する。

## 将来: LLM Knowledge Graph

`llm_knowledge_graph`は、各チャンクをLLMへ渡してJSONトリプルを抽出する。

```text
[日本銀行] ── 目標として定める ──> [消費者物価前年比2%]
       └──── 金融政策を決定する ──> [金融政策決定会合]
```

同義語を名寄せした後、エンティティ、関係、根拠チャンク、関係の確信度をSQLiteへ保存する。さらにLeiden法でコミュニティを作り、コミュニティ要約を生成すれば、局所探索と全体探索を分けられる。

この方式はOpenAI互換APIなどのLLM APIを必要とする。キーは`.env`だけに置き、`GRAPH_RAG_API_KEY`、`GRAPH_RAG_BASE_URL`、`GRAPH_RAG_MODEL`で設定する。アプリ契約のChatGPT／ClaudeはAPIキーの代わりにはならない。抽出・要約プロンプトは`config/prompts/`に分離し、設定ファイルから指定する。

## 切替

`config/graph_backends.json`の`active_backend`を変更する。Colabでは、設定を確認してから明示的にバックエンド名を指定して構築する。

```bash
# E5-smallを使う無料グラフ
uv run python scripts/build_semantic_graph.py --config config/graph_backends.json --backend semantic_e5 --corpus data/processed/corpus.jsonl

# BGE-M3を使う無料グラフ
uv run python scripts/build_semantic_graph.py --config config/graph_backends.json --backend semantic_bge_m3 --corpus data/processed/corpus.jsonl
```

`llm_knowledge_graph`は設定を先に用意しているが、API接続・トリプル抽出・コミュニティ要約の実装とColabテストは別タスクとして追加する。無料版との比較では、コーパス、チャンク分割、評価質問、`k`を固定し、構築時間・エッジ数・Recall@k・MRR・nDCG@kをMLflowに記録する。
