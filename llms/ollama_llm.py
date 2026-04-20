from typing import Optional, Dict, Any
import ollama


class OllamaLLM:
    """
    Ollama-based LLM wrapper for code/text generation.

    Supports models like:
    - codellama
    - deepseek-coder
    - llama3 (if needed)

    Uses ollama.chat() API.
    """

    def __init__(
        self,
        model_name: str = "codellama",
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 256,
        system_prompt: Optional[str] = None,
    ):
        """
        Args:
            model_name: Ollama model name
            temperature: randomness
            top_p: nucleus sampling
            max_tokens: max tokens to generate
            system_prompt: optional system instruction
        """

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        self.system_prompt = system_prompt or (
            "You are a helpful AI assistant for understanding codebases. "
            "Answer strictly using the provided context. "
            "If unsure, say you don't know."
        )

    # --------------------------------------------------
    # 🔹 Prompt builder (RAG style)
    # --------------------------------------------------
    def build_messages(self, query: str, context: str):
        """
        Build chat-style messages for Ollama
        """

        user_prompt = f"""
        Context: {context}
        
        Question: {query}
        Answer:
        """

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return messages

    # --------------------------------------------------
    # 🔹 Generate (RAG)
    # --------------------------------------------------
    def generate(self, query: str, context: str) -> str:
        """
        Generate answer using query + retrieved context
        """

        messages = self.build_messages(query, context)

        try:
            response: Dict[str, Any] = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": self.max_tokens,
                },
            )

            return response["message"]["content"].strip()

        except Exception as e:
            print(f"[Ollama LLM ERROR] {e}")
            return ""

    # --------------------------------------------------
    # 🔹 Raw generation (no RAG)
    # --------------------------------------------------
    def generate_raw(self, prompt: str) -> str:
        """
        Direct prompt generation (no structured context)
        Useful for debugging or baseline testing
        """

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": self.max_tokens,
                },
            )

            return response["message"]["content"].strip()

        except Exception as e:
            print(f"[Ollama LLM ERROR] {e}")
            return ""

    # --------------------------------------------------
    # 🔹 Deterministic mode (for benchmarking)
    # --------------------------------------------------
    def set_deterministic(self):
        """
        Set parameters for reproducible outputs
        """

        self.temperature = 0.0
        self.top_p = 1.0

    def __repr__(self):
        return f"OllamaLLM(model_name={self.model_name})"