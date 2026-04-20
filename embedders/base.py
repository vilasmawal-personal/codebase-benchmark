from abc import ABC, abstractmethod
from typing import List, Union, Optional
import numpy as np


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedding models.

    Every embedder (Ollama, HuggingFace, etc.) must implement this interface.
    This ensures all embeddings are interchangeable in the benchmark pipeline.
    """

    def __init__(self, model_name: str, normalize: bool = False):
        """
        Args:
            model_name (str): Name or identifier of the embedding model
            normalize (bool): Whether to L2 normalize embeddings
        """
        self.model_name = model_name
        self.normalize = normalize

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (List[str]): Input text list

        Returns:
            np.ndarray: Shape (len(texts), embedding_dim)
        """
        pass

    def embed_query(self, text: str) -> np.ndarray:
        """
        Convenience method for embedding a single query.

        Args:
            text (str): Input query string

        Returns:
            np.ndarray: Shape (embedding_dim,)
        """
        return self.embed([text])[0]

    def maybe_normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply L2 normalization if enabled.

        Args:
            embeddings (np.ndarray): Raw embeddings

        Returns:
            np.ndarray: Normalized embeddings
        """
        if not self.normalize:
            return embeddings

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10  # avoid division by zero
        return embeddings / norms

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name}, normalize={self.normalize})"