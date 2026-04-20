import os
import json
import yaml
import time
from typing import Dict, Any, List

from embedders.hf_embedder import HFEmbedder
from embedders.ollama_embedder import OllamaEmbedder

from retriever.faiss_index import FAISSIndex
from retriever.bm25 import BM25Retriever

from rerankers.hf_reranker import HFReranker

from llms.hf_llm import HFLLM
from llms.ollama_llm import OllamaLLM

from evaluation.retrieval_metrics import evaluate_retrieval, aggregate_metrics
from evaluation.generation_metrics import GenerationEvaluator
from data.build_chunks import build_chunks


# --------------------------------------------------
# 🔹 Load YAML config
# --------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at '{path}' must be a dictionary-like YAML object.")
    return cfg


# --------------------------------------------------
# 🔹 Load queries
# --------------------------------------------------
def load_queries(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    if not isinstance(queries, list):
        raise ValueError(f"Queries file at '{path}' must contain a JSON list.")
    return queries


# --------------------------------------------------
# 🔹 Load chunks (preprocessed)
# --------------------------------------------------
def load_chunks(path: str, rebuild_fn=None):
    import os
    import pickle

    # Case 1: File does not exist
    if not os.path.exists(path):
        if rebuild_fn:
            print("⚠️ Chunks file not found. Rebuilding...")
            return rebuild_fn()
        raise FileNotFoundError(f"Chunks file not found at '{path}'")

    try:
        with open(path, "rb") as f:
            chunks = pickle.load(f)

        # Case 2: Not a list
        if not isinstance(chunks, list):
            raise ValueError("Invalid format: not a list")

        # Case 3: Empty list
        if len(chunks) == 0:
            raise ValueError("Chunks list is empty")
        
        if len(chunks) < 10:
            raise ValueError("⚠️ Too few chunks, rebuilding...")

        return chunks

    except Exception as e:
        print(f"⚠️ Invalid or corrupted chunks file: {e}")

        if rebuild_fn:
            print("🔄 Rebuilding chunks...")
            return rebuild_fn()

        raise

def get_chunks(config):
    chunk_path = "data/repo_chunks.pkl"
    repo_path = os.path.expanduser(config["global"]["repo_path"])

    return load_chunks(
        chunk_path,
        rebuild_fn=lambda: build_chunks(repo_path, chunk_path)
    )

def validate_inputs(config: Dict[str, Any], chunks: List[Dict], queries: List[Dict]) -> None:
    required_top_level = ["global", "embedders", "llms", "rerankers", "experiments"]
    for key in required_top_level:
        if key not in config:
            raise KeyError(f"Missing required config section: '{key}'")

    if not chunks:
        raise ValueError("No chunks loaded. 'data/repo_chunks.pkl' appears empty.")
    if not queries:
        raise ValueError("No queries loaded. 'data/queries.json' appears empty.")

    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or "text" not in chunk or "source" not in chunk:
            raise ValueError(f"Chunk at index {i} must contain at least 'text' and 'source' keys.")

    for i, q in enumerate(queries):
        if not isinstance(q, dict) or "query" not in q:
            raise ValueError(f"Query at index {i} must contain a 'query' field.")


# --------------------------------------------------
# 🔹 Factory: Embedder
# --------------------------------------------------
def create_embedder(cfg, global_cfg):
    if cfg["type"] == "hf":
        return HFEmbedder(
            model_name=cfg["model_name"],
            batch_size=global_cfg["runtime"]["batch_size"],
        )

    elif cfg["type"] == "ollama":
        return OllamaEmbedder(
            model_name=cfg["model_name"]
        )

    else:
        raise ValueError("Unknown embedder type")


# --------------------------------------------------
# 🔹 Factory: LLM
# --------------------------------------------------
def create_llm(cfg, global_cfg):
    if cfg["type"] == "hf":
        return HFLLM(
            model_name=cfg["model_name"],
        )

    elif cfg["type"] == "ollama":
        llm = OllamaLLM(model_name=cfg["model_name"])
        llm.set_deterministic()  # important for benchmarking
        return llm

    else:
        raise ValueError("Unknown LLM type")


# --------------------------------------------------
# 🔹 Factory: Reranker
# --------------------------------------------------
def create_reranker(cfg):
    if cfg["type"] == "none":
        return None

    elif cfg["type"] == "hf":
        return HFReranker(model_name=cfg["model_name"])

    else:
        raise ValueError("Unknown reranker type")


# --------------------------------------------------
# 🔹 Main experiment runner
# --------------------------------------------------
def run_experiment(exp_cfg, config, chunks, queries):
    print(f"\n🚀 Running Experiment: {exp_cfg['name']}")

    # Initialize components
    embedder = create_embedder(
        config["embedders"][exp_cfg["embedder"]],
        config["global"]
    )

    llm = create_llm(
        config["llms"][exp_cfg["llm"]],
        config["global"]
    )

    reranker = create_reranker(
        config["rerankers"][exp_cfg["reranker"]]
    )

    retrieval_mode = exp_cfg["retrieval_mode"]
    if retrieval_mode not in {"dense", "hybrid"}:
        raise ValueError(f"Unsupported retrieval_mode='{retrieval_mode}' in experiment '{exp_cfg['name']}'")

    # ------------------------------------------
    # 🔹 Build embeddings + FAISS
    # ------------------------------------------
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed(texts)
    if embeddings.ndim != 2:
        raise ValueError(f"Embedder returned invalid embedding shape: {embeddings.shape}")

    index = FAISSIndex(dim=embeddings.shape[1])
    index.add(embeddings)

    # BM25 (for hybrid)
    bm25 = BM25Retriever(chunks) if retrieval_mode == "hybrid" else None

    # Evaluators
    gen_evaluator = GenerationEvaluator()

    retrieval_results = []
    generation_results = []

    # ------------------------------------------
    # 🔹 Loop queries
    # ------------------------------------------
    for q in queries:
        query = q["query"]
        relevant = q.get("relevant_files", [])
        reference_answer = q.get("answer", None)

        # ---- Dense retrieval
        q_emb = embedder.embed_query(query)
        requested_top_k = int(config["global"]["retrieval"]["top_k"])
        top_k = max(1, min(requested_top_k, len(chunks)))
        dense_indices = index.search_one(q_emb, top_k=top_k)

        # ---- Hybrid
        if retrieval_mode == "hybrid":
            bm25_indices = bm25.search(query, top_k=top_k)
            final_indices = BM25Retriever.hybrid_merge(
                dense_indices,
                bm25_indices,
                top_k=top_k,
            )
        else:
            final_indices = dense_indices

        retrieved_chunks = [chunks[i] for i in final_indices]

        # ---- Reranking
        if reranker:
            texts = [c["text"] for c in retrieved_chunks]
            _, reranked_indices, _ = reranker.rerank_with_indices(
                query,
                texts,
                top_k=config["global"]["retrieval"]["final_k"]
            )
            retrieved_chunks = [retrieved_chunks[i] for i in reranked_indices]
        else:
            retrieved_chunks = retrieved_chunks[:config["global"]["retrieval"]["final_k"]]

        # ---- Build context
        context = "\n\n".join([c["text"] for c in retrieved_chunks])

        # ---- Generate answer
        answer = llm.generate(query, context)

        # ---- Retrieval metrics
        retrieved_sources = [c["source"] for c in retrieved_chunks]
        r_metrics = evaluate_retrieval(
            retrieved_sources,
            relevant,
            ks=config["global"]["evaluation"]["ks"]
        )
        retrieval_results.append(r_metrics)

        # ---- Generation metrics
        g_metrics = gen_evaluator.evaluate(
            query=query,
            answer=answer,
            reference_answer=reference_answer,
            contexts=[c["text"] for c in retrieved_chunks]
        )
        generation_results.append(g_metrics)

    # ------------------------------------------
    # 🔹 Aggregate
    # ------------------------------------------
    final_retrieval = aggregate_metrics(retrieval_results)

    final_generation = {}
    if generation_results:
        keys = generation_results[0].keys()
        for k in keys:
            final_generation[k] = sum(d[k] for d in generation_results) / len(generation_results)

    return {
        "experiment": exp_cfg["name"],
        "retrieval": final_retrieval,
        "generation": final_generation,
    }


# --------------------------------------------------
# 🔹 Main
# --------------------------------------------------
def main():
    TARGET_EXP = "hf_e5_codellama_reranker"
    config = load_config("experiments/configs.yaml")

    queries = load_queries("data/queries.json")
    chunks = get_chunks(config)
    validate_inputs(config, chunks, queries)

    results = []

    for exp in config["experiments"]:
        if exp["name"] != TARGET_EXP:
            continue
        start = time.time()
        try:
            res = run_experiment(exp, config, chunks, queries)
        except Exception as e:
            res = {
                "experiment": exp.get("name", "unknown"),
                "error": str(e),
                "retrieval": {},
                "generation": {},
            }

        res["time"] = round(time.time() - start, 2)

        results.append(res)

        print("\n📊 Result:")
        print(json.dumps(res, indent=2))

    # Save results
    os.makedirs("results", exist_ok=True)

    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ All experiments completed!")


if __name__ == "__main__":
    main()
