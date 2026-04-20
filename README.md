# 🔍 Codebase RAG Benchmark

This project helps you **test and compare AI models** on your own codebase — completely **offline** (no OpenAI API needed).

If you’ve ever wondered:

* “Which embedding model is best for my code?”
* “Does reranking actually help?”
* “Which local LLM (like CodeLlama or DeepSeek) gives better answers?”

👉 This project answers those questions **systematically**.

---

# 🧠 What is this project (in simple terms)?

Imagine you have a big codebase (like HuggingFace Transformers).

Now you want to ask questions like:

> “How does AutoModel work?”

Instead of manually searching files, this system:

1. Reads your codebase
2. Breaks it into small chunks
3. Finds the most relevant parts
4. Uses an AI model to answer your question
5. **Evaluates how good the answer is**

---

# ⚙️ What does “RAG” mean?

RAG = **Retrieval Augmented Generation**

In simple words:

```text
Search → Select → Answer
```

* 🔍 Retrieval → Find relevant code
* 🧠 Generation → AI writes answer
* 📊 Evaluation → Check if answer is correct

---

# 🚀 What makes this project powerful?

✔ Works fully **offline**
✔ Supports **Ollama + Hugging Face models**
✔ Tests multiple models automatically
✔ Gives **objective metrics** (not guesswork)
✔ Modular (you can plug in new models easily)

---

# 📁 Project Structure (explained simply)

```text
benchmark/
│
├── embedders/        → Convert text into vectors (for search)
├── retriever/        → Find relevant chunks (FAISS + BM25)
├── rerankers/        → Improve ranking (very important)
├── llms/             → Generate answers
├── evaluation/       → Score results (quality check)
├── experiments/      → Configurations for experiments
├── data/             → Queries + processed code
│
└── run.py            → Main script (runs everything)
```

---

# 🧩 Key Components (in plain English)

## 1. Embedders (Search Brain)

They convert text into numbers so machines can compare meaning.

Examples:

* BGE (best quality)
* E5
* MiniLM
* Ollama embeddings

👉 Think: *“Understanding meaning”*

---

## 2. Retriever (Search Engine)

Finds relevant parts of code.

* FAISS → semantic search
* BM25 → keyword search

👉 Think: *Google for your codebase*

---

## 3. Reranker (Smart Filter) ⭐

This improves results a LOT.

It re-checks top results and ranks them better.

👉 Think: *“Second opinion before answering”*

---

## 4. LLM (Answer Generator)

Models like:

* CodeLlama
* DeepSeek Coder
* Phi-2

👉 These read context and generate answers.

---

## 5. Evaluation (Truth Checker)

This is the most important part.

It measures:

* Did we retrieve correct file?
* Is answer meaningful?
* Is answer based on context?

---

# 📊 Metrics (explained simply)

## 🔹 Retrieval Metrics

* **Recall@K** → Did we find the right file?
* **MRR** → Was it ranked at top?
* **nDCG** → How good was ranking?

---

## 🔹 Generation Metrics

* **Semantic similarity** → Does answer match truth?
* **Context relevance** → Is answer grounded in code?
* **Keyword overlap** → Basic correctness check

---

# 🧪 How experiments work

You define experiments in:

```text
experiments/configs.yaml
```

Example:

```yaml
- name: bge_with_reranker
  embedder: hf_bge_base
  reranker: bge_reranker_base
  llm: ollama_codellama
  retrieval_mode: hybrid
```

👉 Then the system tests everything automatically.

---

# ▶️ How to run this project

## Step 1 — Install dependencies

Create a clean virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

---

## Step 2 — Start Ollama (if using it)

```bash
ollama serve
```

Pull models:

```bash
ollama pull nomic-embed-text
ollama pull codellama
```

---

## Step 3 — Prepare data

You need:

* `repo_chunks.pkl` → processed code chunks
* `queries.json` → test questions

Each chunk should be a dict with at least:

```json
{"text": "...", "source": "path/to/file.py"}
```

---

## Step 4 — Run benchmark

```bash
python run.py
```

---

## 🧪 Detailed model + benchmark testing workflow

If you want a **repeatable process** to compare models (instead of just running once), use this checklist.

### A) Validate your dataset format first (quick sanity check)

The runner expects:

* `data/repo_chunks.pkl` as a list of dicts containing `text` and `source`
* `data/queries.json` as a list of dicts containing at least `query`

Run:

```bash
python - <<'PY'
import json, pickle

chunks = pickle.load(open("data/repo_chunks.pkl", "rb"))
queries = json.load(open("data/queries.json", "r", encoding="utf-8"))

print("chunks:", len(chunks), "queries:", len(queries))
print("first_chunk_keys:", sorted(chunks[0].keys()) if chunks else "EMPTY")
print("first_query_keys:", sorted(queries[0].keys()) if queries else "EMPTY")
PY
```

If `text`, `source`, or `query` are missing, fix data before benchmarking.

### B) Start with a small smoke test experiment

Before running the full matrix, temporarily keep only 1-2 experiments in `experiments/configs.yaml` (for example `baseline_minilm_codellama` and `bge_base_codellama`) and reduce runtime:

* `global.retrieval.top_k: 10`
* `global.retrieval.final_k: 2`
* keep only a small subset of queries in `data/queries.json`

Then run:

```bash
python run.py
```

This confirms your environment/model wiring works before long runs.

### C) Pull required Ollama models (only if used in config)

If your selected experiments include Ollama embedders/LLMs, make sure each model exists locally:

```bash
ollama list
ollama pull nomic-embed-text
ollama pull mxbai-embed-large
ollama pull codellama
ollama pull deepseek-coder
```

> Tip: You only need to pull models actually referenced by your active experiments.

### D) Run targeted benchmark groups (recommended order)

Use a staged approach so you can identify what improved results:

1. **Embedding baseline group**  
   Compare embedding models with same LLM + no reranker.
2. **Reranker group**  
   Keep embedder fixed and compare `none` vs reranker models.
3. **Retrieval mode group**  
   Compare `dense` vs `hybrid`.
4. **LLM group**  
   Keep retrieval stack fixed and swap only LLM.

After each group:

```bash
python run.py
cp results/results.json results/results_<group_name>.json
```

Example:

```bash
cp results/results.json results/results_embedding_group.json
```

### E) Compare outputs programmatically

Use this quick script to sort experiments by retrieval and generation quality:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("results/results.json")
data = json.loads(path.read_text())

ok = [x for x in data if "error" not in x]
if not ok:
    print("No successful experiments in", path)
    raise SystemExit(0)

def metric(item, *keys, default=0.0):
    cur = item
    for k in keys:
        cur = cur.get(k, {})
    return cur if isinstance(cur, (int, float)) else default

ranked = sorted(
    ok,
    key=lambda x: (
        metric(x, "retrieval", "recall@5"),
        metric(x, "retrieval", "mrr"),
        metric(x, "generation", "semantic_similarity"),
    ),
    reverse=True,
)

for r in ranked[:10]:
    print(
        f"{r['experiment']:<28} "
        f"recall@5={metric(r,'retrieval','recall@5'):.3f} "
        f"mrr={metric(r,'retrieval','mrr'):.3f} "
        f"sem_sim={metric(r,'generation','semantic_similarity'):.3f}"
    )
PY
```

### F) Recommended acceptance criteria for “best” model stack

Pick your winner using consistent gates, for example:

* Retrieval quality improves (`recall@5`, `MRR`) with no large latency jump.
* Generation quality improves (`semantic_similarity`, `context_relevance`).
* No experiment errors in `results/results.json`.
* Runtime is acceptable for your hardware.

### G) Run full benchmark only after smoke test passes

Once stable, restore full query set + full experiments and run:

```bash
python run.py
```

Keep the final artifact:

```bash
cp results/results.json results/results_full_run.json
```

---

## Step 5 — See results

Output:

```text
results/results.json
```

---

# 📊 Example Output

```json
{
  "experiment": "bge_base_codellama",
  "retrieval": {
    "recall@5": 0.82,
    "mrr": 0.61
  },
  "generation": {
    "semantic_similarity": 0.78,
    "context_relevance": 0.81
  }
}
```

---

# 🧠 How to interpret results

| Situation                       | Meaning                                  |
| ------------------------------- | ---------------------------------------- |
| High Recall, Low MRR            | Retrieval OK, ranking bad → use reranker |
| Low Recall                      | Embeddings/chunking bad                  |
| High Retrieval + Low Generation | LLM issue                                |
| All High                        | 🎯 Perfect pipeline                      |

---

# 🔥 Best Practices

✔ Always compare models on SAME dataset
✔ Use reranker (huge improvement)
✔ Try hybrid search (BM25 + embeddings)
✔ Use deterministic LLM for fair comparison

---

# ⚠️ Common Mistakes

❌ Changing data between experiments
❌ Using too few queries
❌ Ignoring evaluation metrics
❌ Not normalizing embeddings correctly

---

# 🛡️ Robustness improvements included

This project now includes a few practical safeguards:

* Input validation for config, queries, and chunks before experiment execution.
* Graceful per-experiment failure handling (a bad model/config no longer crashes the full run).
* Safe `top_k` handling to avoid asking retrieval for more items than available chunks.
* Reranker mapping fixed to preserve exact chunk identity even when texts are duplicated.
* Generation evaluator now falls back gracefully when semantic embedding model loading fails.

---

# 🧠 Real-world insight

From real experiments:

* Changing embedding → small improvement
* Adding reranker → BIG improvement
* Better chunking → HUGE improvement

👉 Most people optimize the wrong thing.

---

# 🚀 What you can do next

* Add more queries (50–100 recommended)
* Try different chunk sizes
* Add new models
* Build a UI dashboard
* Use your own codebase

---

# 🎯 Goal of this project

To help you:

✔ Build better AI systems
✔ Make data-driven decisions
✔ Understand RAG deeply
✔ Work fully offline

---

# 🙌 Final note

You’ve essentially built a **mini research framework for AI systems**.

This is how real companies test:

* search systems
* copilots
* retrieval + generation quality together
* AI assistants

---

If you understand this project →
👉 You understand modern AI pipelines.

---
