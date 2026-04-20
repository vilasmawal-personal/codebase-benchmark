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


# --------------------------------------------------
# 🔹 Load YAML config
# --------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# --------------------------------------------------
# 🔹 Load queries
# --------------------------------------------------
def load_queries(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


# --------------------------------------------------
# 🔹 Load chunks (preprocessed)
# --------------------------------------------------
def load_chunks(path: str):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


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

    # ------------------------------------------
    # 🔹 Build embeddings + FAISS
    # ------------------------------------------
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed(texts)

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
        dense_indices = index.search_one(q_emb, top_k=config["global"]["retrieval"]["top_k"])

        # ---- Hybrid
        if retrieval_mode == "hybrid":
            bm25_indices = bm25.search(query, top_k=25)
            final_indices = BM25Retriever.hybrid_merge(dense_indices, bm25_indices)
        else:
            final_indices = dense_indices

        retrieved_chunks = [chunks[i] for i in final_indices]

        # ---- Reranking
        if reranker:
            texts = [c["text"] for c in retrieved_chunks]
            reranked_texts = reranker.rerank(
                query,
                texts,
                top_k=config["global"]["retrieval"]["final_k"]
            )

            # Map back
            retrieved_chunks = [
                next(c for c in retrieved_chunks if c["text"] == t)
                for t in reranked_texts
            ]
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
    config = load_config("experiments/configs.yaml")

    queries = load_queries("data/queries.json")
    chunks = load_chunks("data/repo_chunks.pkl")

    results = []

    for exp in config["experiments"]:
        start = time.time()

        res = run_experiment(exp, config, chunks, queries)

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
