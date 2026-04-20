from typing import List, Dict, Tuple, Optional
import numpy as np
import re
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    BM25-based sparse retriever.

    Works well for:
    - keyword matching
    - code search (function names, variables)
    - hybrid retrieval (with embeddings)
    """

    def __init__(
        self,
        documents: List[Dict],
        text_key: str = "text",
        preprocess: bool = True,
    ):
        """
        Args:
            documents: list of dicts with at least {text_key: str}
            text_key: key to extract text from document
            preprocess: whether to normalize text
        """

        self.documents = documents
        self.text_key = text_key
        self.preprocess = preprocess

        # Prepare corpus
        self.corpus = [doc[text_key] for doc in documents]

        # Tokenize
        self.tokenized_corpus = [self._tokenize(text) for text in self.corpus]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    # --------------------------------------------------
    # 🔹 Tokenization
    # --------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        """
        Basic tokenizer:
        - lowercase
        - remove special chars
        - split on whitespace

        You can improve this later for code-aware tokenization
        """

        if self.preprocess:
            text = text.lower()
            text = re.sub(r"[^a-z0-9_]+", " ", text)

        return text.split()

    # --------------------------------------------------
    # 🔹 Search
    # --------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 25,
    ) -> List[int]:
        """
        Retrieve top-k document indices

        Args:
            query: search query
            top_k: number of results

        Returns:
            List of indices (into self.documents)
        """

        tokenized_query = self._tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        return top_indices.tolist()

    # --------------------------------------------------
    # 🔹 Search with scores
    # --------------------------------------------------
    def search_with_scores(
        self,
        query: str,
        top_k: int = 25,
    ) -> List[Tuple[int, float]]:
        """
        Returns:
            List of (index, score)
        """

        tokenized_query = self._tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(int(idx), float(scores[idx])) for idx in top_indices]

    # --------------------------------------------------
    # 🔹 Get documents
    # --------------------------------------------------
    def get_documents(self, indices: List[int]) -> List[Dict]:
        """
        Fetch documents by indices
        """
        return [self.documents[i] for i in indices]

    # --------------------------------------------------
    # 🔹 Hybrid merge (BM25 + dense)
    # --------------------------------------------------
    @staticmethod
    def hybrid_merge(
        dense_indices: List[int],
        bm25_indices: List[int],
        top_k: int = 25,
    ) -> List[int]:
        """
        Merge dense + BM25 results

        Strategy:
        - Preserve order
        - Remove duplicates
        """

        seen = set()
        merged = []

        for idx in dense_indices + bm25_indices:
            if idx not in seen:
                seen.add(idx)
                merged.append(idx)

            if len(merged) >= top_k:
                break

        return merged

    def __repr__(self):
        return f"BM25Retriever(num_docs={len(self.documents)})"