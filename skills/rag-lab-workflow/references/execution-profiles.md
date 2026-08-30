# Execution profiles

Profiles are implemented in `rag_lab/profiles.py` and selected with `--profile`.

| Profile | Embedding route | Storage rule | Intended use |
| --- | --- | --- | --- |
| `local` | `chroma_hash`, a deterministic CPU baseline | `data/chroma/` is local and ignored | Learn the pipeline and compare non-GPU methods |
| `colab` | `chroma_e5` using `intfloat/multilingual-e5-small` with Sentence Transformers | Chroma is rebuilt on Colab; corpus, results, and MLflow tracking go to Google Drive | Semantic-search experiments on a GPU |
| `cloud` | `chroma_e5` on GPU when available, otherwise CPU | Supply `--chroma-path` pointing to a mounted persistent volume | Production-like or scheduled experiments |

Do not treat a profile as a retrieval method. It selects resources and default methods; callers can still set `--methods` to run a controlled comparison.

For a new profile, define its default methods, Chroma path, and embedding device in `rag_lab/profiles.py`, document it in the README, and add a profile-routing test that does not require an API key or external model download.
