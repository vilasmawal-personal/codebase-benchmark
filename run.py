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
from retriever.graph_builder import ASTGraphBuilder
from retriever.node_embedder import NodeEmbedder
from retriever.graph_rag import GraphRAGRetriever
from retriever.graph_store import GraphStore
from retriever.light_rag import LightRAGRetriever

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
    configured_repo_path = os.path.expanduser(config["global"].get("repo_path", ""))
    repo_path = configured_repo_path if os.path.isdir(configured_repo_path) else os.getcwd()

    chunk_cfg = config["global"].get("chunking", {})
    chunk_size = int(chunk_cfg.get("chunk_size", 500))
    overlap = int(chunk_cfg.get("overlap", 50))

    if repo_path != configured_repo_path:
        print(
            f"⚠️ repo_path '{configured_repo_path}' does not exist. "
            f"Falling back to current directory: {repo_path}"
        )

    return load_chunks(
        chunk_path,
        rebuild_fn=lambda: build_chunks(
            repo_path,
            chunk_path,
            chunk_size=chunk_size,
            overlap=overlap,
        )
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

    experiments = config.get("experiments", [])
    for i, exp in enumerate(experiments):
        if exp.get("embedder") not in config["embedders"]:
            raise KeyError(f"Experiment[{i}] uses unknown embedder: {exp.get('embedder')}")
        if exp.get("llm") not in config["llms"]:
            raise KeyError(f"Experiment[{i}] uses unknown llm: {exp.get('llm')}")
        if exp.get("reranker") not in config["rerankers"]:
            raise KeyError(f"Experiment[{i}] uses unknown reranker: {exp.get('reranker')}")


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

def build_or_load_graph(config):
    """
    Build graph once (or load if already stored)
    """

    repo_path = config["global"]["repo_path"]

    print("\n🧠 Initializing GraphRAG...")

    # Step 1: Build AST graph
    builder = ASTGraphBuilder()
    nodes, edges = builder.build_from_repo(repo_path)

    # Step 2: Embed nodes
    embedder = NodeEmbedder(
        model_name="microsoft/codebert-base",
        batch_size=16
    )

    nodes = embedder.embed_nodes(nodes)

    # Step 3: Store graph (optional but useful)
    store = GraphStore("graph_db")
    store.insert_graph(nodes, edges)

    # Step 4: Retriever
    graph_retriever = GraphRAGRetriever(
        nodes,
        edges,
        alpha=0.7,
        expand_hops=1
    )

    print("✅ GraphRAG ready")

    return graph_retriever, embedder

# --------------------------------------------------
# 🔹 Main experiment runner
# --------------------------------------------------
def run_experiment(exp_cfg, config, chunks, queries):
    """
    Runs a single experiment configuration:
    - Initializes models
    - Builds retrieval system
    - Executes queries
    - Computes metrics
    """

    print(f"\n🚀 Running Experiment: {exp_cfg['name']}")

    # ==================================================
    # 🔹 1. Initialize Components
    # ==================================================
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

    if retrieval_mode not in {"dense", "hybrid", "graph", "light_rag"}:
        raise ValueError(
            f"Unsupported retrieval_mode='{retrieval_mode}' "
            f"in experiment '{exp_cfg['name']}'"
        )

    # ==================================================
    # 🔹 2. Build Retrieval Backend
    # ==================================================
    graph_retriever = None
    graph_embedder = None
    index = None
    bm25 = None
    light_rag = None

    if retrieval_mode == "graph":
        graph_retriever, graph_embedder = build_or_load_graph(config)

    elif retrieval_mode == "light_rag":
        print("⚡ Initializing LightRAG...")
        light_rag = LightRAGRetriever(chunks)

    else:
        texts = [c["text"] for c in chunks]

        print("🔎 Building embeddings + FAISS index...")
        embeddings = embedder.embed(texts)

        index = FAISSIndex(dim=embeddings.shape[1])
        index.add(embeddings)

        if retrieval_mode == "hybrid":
            bm25 = BM25Retriever(chunks)

    # ==================================================
    # 🔹 3. Evaluators
    # ==================================================
    gen_evaluator = GenerationEvaluator()

    retrieval_results = []
    generation_results = []

    requested_top_k = int(config["global"]["retrieval"]["top_k"])
    final_k = int(config["global"]["retrieval"]["final_k"])

    # ==================================================
    # 🔹 4. Query Loop
    # ==================================================
    for i, q in enumerate(queries):
        print(f"🔍 Query {i+1}/{len(queries)}")

        query = q["query"]
        relevant = q.get("relevant_files", [])
        reference_answer = q.get("answer", None)

        # Consistent top_k
        top_k = requested_top_k if retrieval_mode == "graph" else max(
            1, min(requested_top_k, len(chunks))
        )

        # --------------------------------------------------
        # 🔹 Retrieval
        # --------------------------------------------------
        if retrieval_mode == "dense":
            q_emb = embedder.embed_query(query)
            final_indices = index.search_one(q_emb, top_k=top_k)

        elif retrieval_mode == "hybrid":
            q_emb = embedder.embed_query(query)

            dense_indices = index.search_one(q_emb, top_k=top_k)
            bm25_indices = bm25.search(query, top_k=top_k)

            final_indices = BM25Retriever.hybrid_merge(
                dense_indices,
                bm25_indices,
                top_k=top_k
            )

        elif retrieval_mode == "graph":
            final_indices = graph_retriever.search(
                query,
                graph_embedder,
                top_k=top_k
            )

        elif retrieval_mode == "light_rag":
            final_indices = light_rag.search(
                query,
                top_k=top_k
            )

        else:
            raise ValueError(f"Unhandled retrieval mode: {retrieval_mode}")

        # --------------------------------------------------
        # 🔹 Safety: fallback if no results
        # --------------------------------------------------
        if not final_indices:
            print("⚠️ No retrieval results, using fallback")
            final_indices = list(range(min(top_k, len(chunks))))

        # --------------------------------------------------
        # 🔹 Fetch Retrieved Content
        # --------------------------------------------------
        if retrieval_mode == "graph":
            retrieved_chunks = [
                {
                    "text": n["text"],
                    "source": n["file"]
                }
                for n in graph_retriever.get_nodes(final_indices)
            ]
        else:
            retrieved_chunks = [chunks[i] for i in final_indices if i < len(chunks)]

        # --------------------------------------------------
        # 🔹 Safety: ensure non-empty chunks
        # --------------------------------------------------
        if not retrieved_chunks:
            print("⚠️ Empty retrieved chunks after mapping, using fallback")
            retrieved_chunks = chunks[:min(top_k, len(chunks))]

        # --------------------------------------------------
        # 🔹 Reranking
        # --------------------------------------------------
        if reranker and retrieved_chunks:
            texts = [c["text"] for c in retrieved_chunks]

            _, reranked_indices, _ = reranker.rerank_with_indices(
                query,
                texts,
                top_k=min(final_k, len(texts))
            )

            retrieved_chunks = [retrieved_chunks[i] for i in reranked_indices]
        else:
            retrieved_chunks = retrieved_chunks[:final_k]

        # --------------------------------------------------
        # 🔹 Build Context
        # --------------------------------------------------
        context = "\n\n".join(c["text"] for c in retrieved_chunks)

        # --------------------------------------------------
        # 🔹 Generate Answer
        # --------------------------------------------------
        answer = llm.generate(query, context)

        # --------------------------------------------------
        # 🔹 Retrieval Metrics
        # --------------------------------------------------
        retrieved_sources = [c["source"] for c in retrieved_chunks]

        r_metrics = evaluate_retrieval(
            retrieved_sources,
            relevant,
            ks=config["global"]["evaluation"]["ks"]
        )

        retrieval_results.append(r_metrics)

        # --------------------------------------------------
        # 🔹 Generation Metrics
        # --------------------------------------------------
        g_metrics = gen_evaluator.evaluate(
            query=query,
            answer=answer,
            reference_answer=reference_answer,
            contexts=[c["text"] for c in retrieved_chunks]
        )

        generation_results.append(g_metrics)

    # ==================================================
    # 🔹 5. Aggregate Results
    # ==================================================
    final_retrieval = aggregate_metrics(retrieval_results)

    final_generation = {}
    if generation_results:
        for key in generation_results[0]:
            final_generation[key] = sum(
                d[key] for d in generation_results
            ) / len(generation_results)

    # ==================================================
    # 🔹 6. Return Results
    # ==================================================
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
    chunks = get_chunks(config)
    validate_inputs(config, chunks, queries)

    results = []

    target_experiments = {
        name.strip()
        for name in os.getenv("TARGET_EXPERIMENTS", "").split(",")
        if name.strip()
    }

    for exp in config["experiments"]:
        if target_experiments and exp["name"] not in target_experiments:
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

    with open("results/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n✅ All experiments completed!")


if __name__ == "__main__":
    main()
