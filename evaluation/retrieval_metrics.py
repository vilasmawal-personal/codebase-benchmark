from typing import List, Dict
import numpy as np


# --------------------------------------------------
# 🔹 Recall@K
# --------------------------------------------------
def recall_at_k(
    retrieved_sources: List[str],
    relevant_sources: List[str],
    k: int
) -> float:
    """
    Recall@K:
    Did we retrieve ANY relevant document in top-K?

    Args:
        retrieved_sources: list of retrieved file paths
        relevant_sources: list of ground-truth file names/paths
        k: cutoff

    Returns:
        float (0 or 1 for typical use)
    """

    retrieved_top_k = retrieved_sources[:k]

    for rel in relevant_sources:
        if any(rel in src for src in retrieved_top_k):
            return 1.0

    return 0.0


# --------------------------------------------------
# 🔹 Precision@K
# --------------------------------------------------
def precision_at_k(
    retrieved_sources: List[str],
    relevant_sources: List[str],
    k: int
) -> float:
    """
    Precision@K:
    How many retrieved docs are relevant?

    Returns:
        float in [0, 1]
    """

    retrieved_top_k = retrieved_sources[:k]

    hits = 0
    for src in retrieved_top_k:
        if any(rel in src for rel in relevant_sources):
            hits += 1

    return hits / k if k > 0 else 0.0


# --------------------------------------------------
# 🔹 Mean Reciprocal Rank (MRR)
# --------------------------------------------------
def reciprocal_rank(
    retrieved_sources: List[str],
    relevant_sources: List[str]
) -> float:
    """
    Reciprocal Rank:
    1 / rank of first relevant document
    """

    for i, src in enumerate(retrieved_sources):
        if any(rel in src for rel in relevant_sources):
            return 1.0 / (i + 1)

    return 0.0


def mean_reciprocal_rank(
    all_retrieved: List[List[str]],
    all_relevant: List[List[str]]
) -> float:
    """
    Mean Reciprocal Rank across queries
    """

    scores = [
        reciprocal_rank(ret, rel)
        for ret, rel in zip(all_retrieved, all_relevant)
    ]

    return float(np.mean(scores))


# --------------------------------------------------
# 🔹 DCG / nDCG
# --------------------------------------------------
def dcg_at_k(
    retrieved_sources: List[str],
    relevant_sources: List[str],
    k: int
) -> float:
    """
    Discounted Cumulative Gain
    """

    score = 0.0

    for i, src in enumerate(retrieved_sources[:k]):
        rel = 1 if any(r in src for r in relevant_sources) else 0
        score += rel / np.log2(i + 2)

    return score


def ndcg_at_k(
    retrieved_sources: List[str],
    relevant_sources: List[str],
    k: int
) -> float:
    """
    Normalized DCG
    """

    dcg = dcg_at_k(retrieved_sources, relevant_sources, k)

    # Ideal DCG (perfect ranking)
    ideal_rels = [1] * min(len(relevant_sources), k)
    ideal_dcg = sum(
        rel / np.log2(i + 2)
        for i, rel in enumerate(ideal_rels)
    )

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


# --------------------------------------------------
# 🔹 Hit Rate@K
# --------------------------------------------------
def hit_rate_at_k(
    retrieved_sources: List[str],
    relevant_sources: List[str],
    k: int
) -> float:
    """
    Same as recall@k but explicitly named for clarity
    """
    return recall_at_k(retrieved_sources, relevant_sources, k)


# --------------------------------------------------
# 🔹 Aggregate metrics for a single query
# --------------------------------------------------
def evaluate_retrieval(
    retrieved_sources: List[str],
    relevant_sources: List[str],
    ks: List[int] = [1, 3, 5, 10, 25]
) -> Dict[str, float]:
    """
    Compute multiple retrieval metrics for one query
    """

    results = {}

    for k in ks:
        results[f"recall@{k}"] = recall_at_k(
            retrieved_sources, relevant_sources, k
        )
        results[f"precision@{k}"] = precision_at_k(
            retrieved_sources, relevant_sources, k
        )
        results[f"ndcg@{k}"] = ndcg_at_k(
            retrieved_sources, relevant_sources, k
        )
        results[f"hit_rate@{k}"] = hit_rate_at_k(
            retrieved_sources, relevant_sources, k
        )

    results["mrr"] = reciprocal_rank(
        retrieved_sources, relevant_sources
    )

    return results


# --------------------------------------------------
# 🔹 Aggregate metrics across all queries
# --------------------------------------------------
def aggregate_metrics(
    all_results: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Average metrics across multiple queries
    """

    if not all_results:
        return {}

    keys = all_results[0].keys()

    aggregated = {}
    for key in keys:
        aggregated[key] = float(
            np.mean([res[key] for res in all_results])
        )

    return aggregated