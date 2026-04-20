# 🔍 Codebase RAG Benchmark (Local + Offline)

This project benchmarks different embedding models, retrieval strategies, rerankers, and LLMs for **codebase question answering**.

## 🚀 Features

* HuggingFace + Ollama embeddings
* FAISS dense retrieval
* BM25 sparse retrieval (hybrid search)
* Cross-encoder reranking
* LLM evaluation (HF + Ollama)
* Retrieval + Generation metrics
* YAML-based experiment configs

## 🧠 Supported Models

### Embeddings

* BGE (base/large)
* E5 (base/large)
* MiniLM
* Ollama (nomic, mxbai)

### LLMs

* Codellama
* DeepSeek Coder
* Phi-2
* StarCoder

## 📊 Metrics

* Recall@K
* MRR
* nDCG
* Semantic similarity
* Context relevance

## ▶️ Run Benchmark

```bash
python run.py
```

## ⚙️ Config

Edit:

```
experiments/configs.yaml
```

## 📁 Structure

```
benchmark/
├── embedders/
├── retriever/
├── rerankers/
├── llms/
├── evaluation/
├── experiments/
├── data/
└── run.py
```

## 🧪 Goal

Evaluate RAG pipelines for code understanding using local/offline models.

---
