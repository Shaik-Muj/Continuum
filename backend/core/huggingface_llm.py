"""Hugging Face implementation of the Continuum LLM interface."""

import torch
from transformers import pipeline

from core.llm import LLM


class HuggingFaceLLM(LLM):
    """
    LLM implementation using a local Hugging Face model.
    """

    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"

        self.device = 0 if torch.cuda.is_available() else -1
        self.torch_dtype = torch.float16 if self.device == 0 else torch.float32

        self.pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def generate(self, prompt: str) -> str:
        """
        Generate a response using the local Hugging Face model.

        Args:
            prompt: Prompt to send to the model.

        Returns:
            Generated text response.
        """

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        result = self.pipeline(
            messages,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.1,
        )

        generated_messages = result[0]["generated_text"]

        return generated_messages[-1]["content"]