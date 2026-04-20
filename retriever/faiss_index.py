from typing import List, Tuple, Optional
import numpy as np
import faiss
import os


class FAISSIndex:
    """
    Dense vector index using FAISS.

    Supports:
    - Cosine similarity (default, recommended)
    - L2 distance

    Works with embeddings from:
    - Hugging Face
    - Ollama
    """

    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
    ):
        """
        Args:
            dim: embedding dimension
            metric: "cosine" or "l2"
        """

        self.dim = dim
        self.metric = metric

        if metric == "cosine":
            # Inner product + normalized vectors = cosine similarity
            self.index = faiss.IndexFlatIP(dim)
            self.normalize = True

        elif metric == "l2":
            self.index = faiss.IndexFlatL2(dim)
            self.normalize = False

        else:
            raise ValueError("metric must be 'cosine' or 'l2'")

    # --------------------------------------------------
    # 🔹 Normalize embeddings (for cosine)
    # --------------------------------------------------
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return vectors / norms

    # --------------------------------------------------
    # 🔹 Add embeddings to index
    # --------------------------------------------------
    def add(self, embeddings: np.ndarray):
        """
        Add embeddings to FAISS index
        """

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")

        if self.normalize:
            embeddings = self._normalize(embeddings)

        self.index.add(embeddings)

    # --------------------------------------------------
    # 🔹 Search
    # --------------------------------------------------
    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 25,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            query_embeddings: shape (N, dim)
            top_k: number of results

        Returns:
            distances, indices
        """

        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype("float32")

        if self.normalize:
            query_embeddings = self._normalize(query_embeddings)

        distances, indices = self.index.search(query_embeddings, top_k)

        return distances, indices

    # --------------------------------------------------
    # 🔹 Search (single query helper)
    # --------------------------------------------------
    def search_one(
        self,
        query_embedding: np.ndarray,
        top_k: int = 25,
    ) -> List[int]:
        """
        Convenience method for single query
        """

        _, indices = self.search(query_embedding, top_k)
        return indices[0].tolist()

    # --------------------------------------------------
    # 🔹 Save index
    # --------------------------------------------------
    def save(self, path: str):
        """
        Save FAISS index to disk
        """

        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)

    # --------------------------------------------------
    # 🔹 Load index
    # --------------------------------------------------
    @classmethod
    def load(cls, path: str, metric: str = "cosine"):
        """
        Load FAISS index from disk
        """

        index = faiss.read_index(path)

        obj = cls(dim=index.d, metric=metric)
        obj.index = index

        return obj

    # --------------------------------------------------
    # 🔹 Get index size
    # --------------------------------------------------
    def __len__(self):
        return self.index.ntotal

    def __repr__(self):
        return f"FAISSIndex(dim={self.dim}, metric={self.metric}, size={len(self)})"