from typing import List
import numpy as np
from tqdm import tqdm
import ollama

from .base import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    """
    Ollama-based embedding model.

    Supports models like:
    - nomic-embed-text
    - mxbai-embed-large

    Note:
    Ollama currently does not support true batching,
    so embeddings are generated one-by-one.
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        normalize: bool = True,
        show_progress: bool = True,
    ):
        """
        Args:
            model_name (str): Ollama embedding model name
            normalize (bool): Whether to normalize embeddings
            show_progress (bool): Show tqdm progress bar
        """
        super().__init__(model_name, normalize)
        self.show_progress = show_progress

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Since Ollama doesn't support batching,
        we process texts sequentially.
        """

        embeddings = []

        iterator = tqdm(texts, desc=f"Ollama Embedding ({self.model_name})") \
            if self.show_progress else texts

        for text in iterator:
            try:
                response = ollama.embeddings(
                    model=self.model_name,
                    prompt=text
                )

                emb = response["embedding"]
                embeddings.append(emb)

            except Exception as e:
                # Fail-safe: append zero vector to maintain alignment
                print(f"[Ollama ERROR] {e}")
                embeddings.append([0.0] * 768)  # fallback dimension

        embeddings = np.array(embeddings, dtype="float32")

        return self.maybe_normalize(embeddings)