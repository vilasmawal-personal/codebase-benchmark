from typing import List, Tuple, Optional
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class HFReranker:
    """
    Hugging Face Cross-Encoder Reranker.

    Designed for models like:
    - BAAI/bge-reranker-base
    - BAAI/bge-reranker-large

    Takes (query, document) pairs and scores relevance.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512,
    ):
        """
        Args:
            model_name: HF reranker model
            device: "cpu", "cuda", or None (auto)
            batch_size: batch size for scoring
            max_length: max token length
        """

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)

        self.model.eval()

    # --------------------------------------------------
    # 🔹 Internal batching helper
    # --------------------------------------------------
    def _batch(self, items: List, batch_size: int):
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    # --------------------------------------------------
    # 🔹 Score (core function)
    # --------------------------------------------------
    def score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """
        Score query-document relevance.

        Args:
            query: input query
            documents: list of document texts

        Returns:
            List of relevance scores (same order as documents)
        """

        scores = []

        pairs = [(query, doc) for doc in documents]

        for batch in self._batch(pairs, self.batch_size):
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze(-1)

                # Convert to list
                batch_scores = logits.detach().cpu().numpy().tolist()

                # Handle single-item edge case
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]

                scores.extend(batch_scores)

        return scores

    # --------------------------------------------------
    # 🔹 Rerank (returns sorted docs)
    # --------------------------------------------------
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[str]:
        """
        Rerank documents by relevance.

        Args:
            query: input query
            documents: list of documents
            top_k: optional cutoff

        Returns:
            List of reranked documents
        """

        scores = self.score(query, documents)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        if top_k:
            ranked = ranked[:top_k]

        return [doc for doc, _ in ranked]

    # --------------------------------------------------
    # 🔹 Rerank with indices (IMPORTANT for pipeline)
    # --------------------------------------------------
    def rerank_with_indices(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> Tuple[List[str], List[int], List[float]]:
        """
        Rerank and keep original indices.

        Returns:
            documents: reranked docs
            indices: original positions
            scores: relevance scores
        """

        scores = self.score(query, documents)

        indexed = list(enumerate(documents))

        ranked = sorted(
            [(idx, doc, score) for (idx, doc), score in zip(indexed, scores)],
            key=lambda x: x[2],
            reverse=True,
        )

        if top_k:
            ranked = ranked[:top_k]

        docs = [r[1] for r in ranked]
        indices = [r[0] for r in ranked]
        scores = [r[2] for r in ranked]

        return docs, indices, scores

    def __repr__(self):
        return f"HFReranker(model_name={self.model_name}, device={self.device})"