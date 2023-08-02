from __future__ import annotations

import os
from typing import Any, Dict, Optional

import openai

from retrieval_qa_benchmark.schema import BaseLLM
from retrieval_qa_benchmark.utils.registry import REGISTRY


@REGISTRY.register_model("gpt35")
class GPT(BaseLLM):
    run_args: Dict[str, Any] = {}
    system_prompt: str = "You are a helpful assistant."

    @classmethod
    def build(
        cls,
        model_name: str = "text-davinci-003",
        api_base: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        api_key: str = os.getenv("OPENAI_API_KEY", ""),
        system_prompt: Optional[str] = None,
        run_args: Optional[Dict[str, Any]] = None,
    ) -> GPT:
        openai.api_base = api_base
        openai.api_key = api_key
        return cls(
            model_name=model_name,
            run_args=run_args or {},
            system_prompt=system_prompt or "",
        )

    def generate(
        self,
        text: str,
    ) -> str:
        completion = openai.Completion.create(
            model="gpt-3.5-turbo",
            prompt="\n".join([self.system_prompt, text]),
            **self.run_args,
        )
        return completion.choices[0].text


@REGISTRY.register_model("chatgpt35")
class ChatGPT(GPT):
    def generate(
        self,
        text: str = "",
    ) -> str:
        completion = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            **self.run_args,
        )
        return completion.choices[0].message.content
