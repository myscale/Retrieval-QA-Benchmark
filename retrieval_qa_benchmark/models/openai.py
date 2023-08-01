from __future__ import annotations
import os
from typing import Any
import openai
from retrieval_qa_benchmark.models.base import BaseLLM


class GPT(BaseLLM):
    @classmethod
    def build(
        cls,
        model_name: str = "text-davinci-003",
        api_base: str = os.getenv("OPENAI_API_KEY"),
        api_key: str = os.getenv("OPENAI_API_BASE"),
    ) -> ChatGPT:
        openai.api_base = api_base
        openai.api_key = api_key
        return cls(name=model_name, model=openai)

    def generate(
        self,
        text: str,
        temperature=0.8,
        top_p=1.0,
        **kwargs: Any,
    ):
        completion = self.model.Completion.create(
            model="gpt-3.5-turbo",
            prompt=text,
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
        temperature=0.8,
        top_p=1.0,
        **kwargs: Any,
    ):
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
