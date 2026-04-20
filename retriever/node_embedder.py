from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


class NodeEmbedder:
    """
    Embeds graph nodes using code-aware or general embedding models.

    Supported models:
    - microsoft/codebert-base
    - intfloat/e5-base-v2 / e5-large-v2
    - BAAI/bge-base-en / bge-large-en
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        batch_size: int = 16,
        normalize: bool = True,
    ):
        """
        Args:
            model_name: embedding model
            batch_size: embedding batch size
            normalize: normalize embeddings (recommended for cosine)
        """

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        self.model = SentenceTransformer(model_name)

    # --------------------------------------------------
    # 🔹 Public: embed nodes
    # --------------------------------------------------
    def embed_nodes(self, nodes: List[Dict]) -> List[Dict]:
        """
        Adds embeddings to nodes in-place.

        Each node gets:
            node["embedding"] = np.array
        """

        texts = [self._prepare_text(n) for n in nodes]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
        )

        for i, emb in enumerate(embeddings):
            nodes[i]["embedding"] = emb

        return nodes

    # --------------------------------------------------
    # 🔹 Embed query
    # --------------------------------------------------
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query (important for retrieval)
        """

        query = self._prepare_query(query)

        emb = self.model.encode(
            [query],
            normalize_embeddings=self.normalize
        )[0]

        return emb

    # --------------------------------------------------
    # 🔹 Internal: prepare node text
    # --------------------------------------------------
    def _prepare_text(self, node: Dict) -> str:
        """
        Build embedding text from node.

        Combines:
        - type
        - name
        - code snippet
        """

        node_type = node.get("type", "")
        name = node.get("name", "")
        text = node.get("text", "")

        # Truncate long code (important for speed)
        text = text[:2000]

        return f"{node_type}: {name}\n{text}"

    # --------------------------------------------------
    # 🔹 Internal: prepare query (model-specific)
    # --------------------------------------------------
    def _prepare_query(self, query: str) -> str:
        """
        Adjust query format for specific models
        """

        model_name = self.model_name.lower()

        # E5 models
        if "e5" in model_name:
            return f"query: {query}"

        # BGE models
        if "bge" in model_name:
            return f"Represent this sentence for retrieval: {query}"

        # CodeBERT / others
        return query

    # --------------------------------------------------
    # 🔹 Utility
    # --------------------------------------------------
    def get_embedding_matrix(self, nodes: List[Dict]) -> np.ndarray:
        """
        Convert node list → embedding matrix
        """

        return np.array([n["embedding"] for n in nodes])

    def __repr__(self):
        return f"NodeEmbedder(model_name={self.model_name})"