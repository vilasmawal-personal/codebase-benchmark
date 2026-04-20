from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class HFLLM:
    """
    Hugging Face LLM wrapper for text/code generation.

    Supports models like:
    - microsoft/phi-2
    - bigcode/starcoderbase
    - bigcode/starcoder2
    - deepseek-ai/deepseek-coder-1.3b-instruct
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
    ):
        """
        Args:
            model_name: HF model name
            device: "cpu", "cuda", or None (auto-detect)
            max_new_tokens: generation length
            temperature: randomness
            top_p: nucleus sampling
        """

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Fix for models without pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )

        if self.device == "cpu":
            self.model.to(self.device)

        self.model.eval()

    # --------------------------------------------------
    # 🔹 Prompt builder (RAG style)
    # --------------------------------------------------
    def build_prompt(self, query: str, context: str) -> str:
        """
        Standard RAG prompt format.
        You can tweak this later for better performance.
        """

        prompt = f"""You are a helpful AI assistant for code understanding.
        
        Use the following context to answer the question. 
        
        Context: {context}
        
        Question: {query}
        
        Answer:"""

        return prompt

    # --------------------------------------------------
    # 🔹 Generate
    # --------------------------------------------------
    def generate(self, query: str, context: str) -> str:
        """
        Generate answer using context + query
        """

        prompt = self.build_prompt(query, context)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # Remove prompt from output (important cleanup)
        answer = generated_text[len(prompt):].strip()

        return answer

    # --------------------------------------------------
    # 🔹 Simple generate (no context)
    # --------------------------------------------------
    def generate_raw(self, prompt: str) -> str:
        """
        Direct generation (no structured prompt)
        Useful for debugging
        """

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def __repr__(self):
        return f"HFLLM(model_name={self.model_name}, device={self.device})"