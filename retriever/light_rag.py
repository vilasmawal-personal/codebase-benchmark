from typing import List, Dict, Set
from collections import defaultdict
import re
import math


class LightRAGRetriever:
    """
    Lightweight RAG using entity + keyword matching.

    Designed for codebases:
    - function names
    - class names
    - variables
    - identifiers

    Strategy:
    - Build inverted index
    - Tokenize query
    - Score chunks using token overlap + frequency
    """

    def __init__(self, chunks: List[Dict]):
        """
        Args:
            chunks: list of {"text": str, "source": str}
        """
        self.chunks = chunks

        # token -> set(chunk_indices)
        self.inverted_index: Dict[str, Set[int]] = defaultdict(set)

        # token -> document frequency
        self.df = defaultdict(int)

        self.N = len(chunks)

        self._build_index()

    # --------------------------------------------------
    # 🔹 Build inverted index
    # --------------------------------------------------
    def _build_index(self):
        """
        Build token → chunk mapping
        """

        print("⚡ Building LightRAG index...")

        for idx, chunk in enumerate(self.chunks):
            tokens = self._tokenize(chunk["text"])

            for token in tokens:
                self.inverted_index[token].add(idx)

        # compute document frequency
        for token, indices in self.inverted_index.items():
            self.df[token] = len(indices)

        print(f"✅ LightRAG index built with {len(self.inverted_index)} tokens")

    # --------------------------------------------------
    # 🔹 Tokenization (code-aware)
    # --------------------------------------------------
    def _tokenize(self, text: str) -> Set[str]:
        """
        Tokenize text with code awareness:
        - camelCase splitting
        - snake_case handling
        - identifier extraction
        """

        # Split camelCase → camel Case
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

        # Replace non-alphanumeric (keep _)
        text = re.sub(r"[^a-zA-Z0-9_]+", " ", text)

        tokens = text.lower().split()

        return set(tokens)

    # --------------------------------------------------
    # 🔹 IDF scoring
    # --------------------------------------------------
    def _idf(self, token: str) -> float:
        """
        Inverse document frequency
        """
        df = self.df.get(token, 0)
        if df == 0:
            return 0.0

        return math.log((self.N + 1) / (df + 1)) + 1

    # --------------------------------------------------
    # 🔹 Search
    # --------------------------------------------------
    def search(self, query: str, top_k: int = 25) -> List[int]:
        """
        Retrieve top-k chunk indices
        """

        query_tokens = self._tokenize(query)

        scores = defaultdict(float)

        # Score chunks
        for token in query_tokens:
            idf_score = self._idf(token)

            for idx in self.inverted_index.get(token, []):
                scores[idx] += idf_score

        # Fallback if no matches
        if not scores:
            return list(range(min(top_k, len(self.chunks))))

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [idx for idx, _ in ranked[:top_k]]

    # --------------------------------------------------
    # 🔹 Debug scoring
    # --------------------------------------------------
    def search_with_scores(self, query: str, top_k: int = 25):
        """
        Return (index, score)
        """
        query_tokens = self._tokenize(query)

        scores = defaultdict(float)

        for token in query_tokens:
            idf_score = self._idf(token)

            for idx in self.inverted_index.get(token, []):
                scores[idx] += idf_score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return ranked[:top_k]

    # --------------------------------------------------
    # 🔹 Utility
    # --------------------------------------------------
    def __repr__(self):
        return f"LightRAGRetriever(num_chunks={len(self.chunks)})"