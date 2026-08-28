# RAG 手法比較ラボ

同じコーパスと正解付き質問セットに対して複数の RAG 検索戦略を実行し、検索精度を比較するための軽量な実験基盤です。

## 対象手法

| 名前 | 内容 |
| --- | --- |
| `bm25` | キーワード検索（ベースライン） |
| `dense` | TF-IDF ベクトルのコサイン類似度による意味的な近似検索 |
| `hyde` | 検索時に仮想文書を生成してからDense検索するHyDE |
| `reverse_hyde` | 文書ごとの仮想質問を事前生成して検索するReverse HyDE |
| `hybrid` | BM25 と Dense の正規化スコア融合 |
| `advanced` | Hybrid 検索後にクエリ語の被覆率で再ランキング |
| `agentic` | 質問を分解し、複数回検索した結果を融合 |
| `graph` | 文書から抽出したエンティティ共有グラフで候補を拡張 |
| `corpus2skill` | 文書を階層スキルツリーへコンパイルし、ツリーを探索して取得 |

これは再現可能な比較用実装です。LLM による回答生成や外部 Embedding API を接続する前に、各方式の**検索精度**（Recall@k / MRR / nDCG@k）を公平に測れます。

## 実行

Python 3.10 以降のみで動きます。

```bash
python -m rag_lab evaluate --corpus data/example_corpus.jsonl --qa data/example_qa.jsonl --k 3
```

CSV と JSON の結果を `results/` に保存します。

```bash
python -m rag_lab evaluate --corpus data/example_corpus.jsonl --qa data/example_qa.jsonl \
  --methods bm25,dense,hybrid,advanced,agentic,graph,corpus2skill --k 3 --output results
```

## MLflowによる比較

MLflowを有効にすると、手法ごとにRecall@k、MRR、nDCG@k、平均レイテンシを記録します。コーパス本文はMLflowへ送らず、診断用の文書IDだけを保存します。

```bash
uv sync
uv run python -m rag_lab evaluate --corpus data/example_corpus.jsonl --qa data/example_qa.jsonl --mlflow
uv run mlflow ui --backend-store-uri mlruns
```

表示された `http://127.0.0.1:5000` でrunを比較できます。

特定の質問について、各方式が何を取得したかを確認できます。

```bash
python -m rag_lab inspect --corpus data/example_corpus.jsonl \
  --query "休暇申請の承認者は誰ですか" --method corpus2skill
```

## 入力形式

コーパスは JSONL で、各行に `id`、`text`、任意の `title` を持たせます。

```json
{"id":"leave-policy","title":"休暇規程","text":"年次有給休暇は直属の上長の承認を得て申請する。"}
```

評価質問は `question` と、正解文書IDの配列 `relevant_ids` を持たせます。

```json
{"id":"q1","question":"休暇申請の承認者は誰ですか","relevant_ids":["leave-policy"]}
```

## Corpus2Skill の解釈

`corpus2skill` は、文書を語彙ベクトルで類似するトピックに再帰的にまとめ、各ノードに `SKILL.md` 相当の要約を作る簡易実装です。問い合わせ時にはツリーの各階層で最も関連する枝だけをたどり、葉の文書を返します。研究版の LLM 要約・クラスタリングへ置換できるよう `Corpus2SkillRetriever` を独立させています。

## 比較時の注意

- 手法間で同じコーパス、同じ質問、同じ `k` を使う。
- 最初は検索指標を改善し、その後に回答の正確性・根拠忠実性を別の評価セットで測る。
- Corpus2Skill / Graph / Agentic は構築・実行コストも計測対象にする。`summary.json` に平均検索時間を出力します。
