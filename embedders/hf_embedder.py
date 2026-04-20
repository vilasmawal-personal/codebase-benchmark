from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .base import BaseEmbedder


class HFEmbedder(BaseEmbedder):
    """
    Hugging Face embedder using sentence-transformers.

    Supports models like:
    - sentence-transformers/all-MiniLM-L6-v2
    - BAAI/bge-base-en
    - BAAI/bge-large-en
    - intfloat/e5-base-v2
    - intfloat/e5-large-v2
    """

    def __init__(
        self,
        model_name: str,
        device: str = None,
        batch_size: int = 32,
        normalize: bool = False,
    ):
        """
        Args:
            model_name (str): HF model name
            device (str): "cpu" or "cuda" (auto-detect if None)
            batch_size (int): Batch size for encoding
            normalize (bool): Whether to normalize embeddings
        """
        super().__init__(model_name, normalize)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.model = SentenceTransformer(model_name, device=self.device)

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Special handling:
        - E5 models require "query:" / "passage:" prefix
        - BGE models benefit from "Represent this sentence..." prefix (optional)
        """

        # Detect model type for prompt formatting
        if "e5" in self.model_name.lower():
            texts = [f"passage: {t}" for t in texts]

        elif "bge" in self.model_name.lower():
            # Optional but improves performance slightly
            texts = [f"Represent this sentence for retrieval: {t}" for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        embeddings = embeddings.astype("float32")

        return self.maybe_normalize(embeddings)