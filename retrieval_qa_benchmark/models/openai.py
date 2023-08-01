from __future__ import annotations

import os
from typing import Any, Optional

import openai

from retrieval_qa_benchmark.models.base import BaseLLM


class GPT(BaseLLM):
    @classmethod
    def build(
        cls,
        model_name: str = "text-davinci-003",
        api_base: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        api_key: str = os.getenv("OPENAI_API_KEY", ""),
    ) -> GPT:
        openai.api_base = api_base
        openai.api_key = api_key
        return cls(name=model_name, model=openai)

    def generate(
        self,
        text: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.8,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> str:
        completion = self.model.Completion.create(
            model="gpt-3.5-turbo",
            prompt="\n".join([system_prompt, text]),
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        return completion.choices[0].text


class ChatGPT(GPT):
    def generate(
        self,
        text: str = "gpt-3.5-turbo",
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.8,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> str:
        completion = self.model.ChatCompletion.create(
            model=self.name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        return completion.choices[0].message.content
