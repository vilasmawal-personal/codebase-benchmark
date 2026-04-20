from typing import List, Dict, Optional
import numpy as np
import re

# Optional heavy libs (lazy usage)
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class GenerationEvaluator:
    """
    Evaluates generated answers using multiple metrics:

    1. Semantic similarity (answer vs ground truth)
    2. Context relevance (answer vs retrieved context)
    3. Keyword overlap (lightweight fallback)
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """
        Args:
            embedding_model (str): Model for semantic similarity
            device (str): cpu / cuda
        """
        self.embedding_model_name = embedding_model
        self.device = device

        self.model = None
        if SentenceTransformer is not None:
            self.model = SentenceTransformer(embedding_model, device=device)

    # -----------------------------
    # 🔹 Core: Semantic similarity
    # -----------------------------
    def _embed(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("SentenceTransformer not available")

        emb = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return emb

    def semantic_similarity(self, a: str, b: str) -> float:
        """
        Cosine similarity between two texts
        """
        emb = self._embed([a, b])
        return float(np.dot(emb[0], emb[1]))

    # -----------------------------
    # 🔹 Context relevance
    # -----------------------------
    def context_relevance(self, answer: str, contexts: List[str]) -> float:
        """
        Measures how well answer aligns with retrieved context
        """
        if not contexts:
            return 0.0

        joined_context = " ".join(contexts)

        try:
            return self.semantic_similarity(answer, joined_context)
        except Exception:
            return 0.0

    # -----------------------------
    # 🔹 Keyword overlap (fallback)
    # -----------------------------
    def keyword_overlap(self, answer: str, reference: str) -> float:
        """
        Simple lexical overlap score (Jaccard-like)
        """

        def tokenize(text):
            text = text.lower()
            text = re.sub(r"[^a-z0-9\s]", "", text)
            return set(text.split())

        a_tokens = tokenize(answer)
        r_tokens = tokenize(reference)

        if not a_tokens or not r_tokens:
            return 0.0

        intersection = len(a_tokens & r_tokens)
        union = len(a_tokens | r_tokens)

        return intersection / union

    # -----------------------------
    # 🔹 Combined evaluation
    # -----------------------------
    def evaluate(
        self,
        query: str,
        answer: str,
        reference_answer: Optional[str] = None,
        contexts: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Full evaluation of a generated answer.

        Args:
            query: original question
            answer: model output
            reference_answer: ground truth answer (optional)
            contexts: retrieved chunks

        Returns:
            dict of metrics
        """

        results = {}

        # Semantic similarity with reference
        if reference_answer:
            try:
                results["semantic_similarity"] = self.semantic_similarity(
                    answer, reference_answer
                )
            except Exception:
                results["semantic_similarity"] = 0.0

            results["keyword_overlap"] = self.keyword_overlap(
                answer, reference_answer
            )

        # Context relevance
        if contexts:
            results["context_relevance"] = self.context_relevance(
                answer, contexts
            )

        # Length sanity check (useful debug signal)
        results["answer_length"] = len(answer.split())

        return results

    # -----------------------------
    # 🔹 Batch evaluation
    # -----------------------------
    def evaluate_batch(
        self,
        queries: List[str],
        answers: List[str],
        references: Optional[List[str]] = None,
        contexts_list: Optional[List[List[str]]] = None,
    ) -> List[Dict[str, float]]:
        """
        Evaluate multiple samples
        """
        results = []

        for i in range(len(queries)):
            res = self.evaluate(
                query=queries[i],
                answer=answers[i],
                reference_answer=references[i] if references else None,
                contexts=contexts_list[i] if contexts_list else None,
            )
            results.append(res)

        return results