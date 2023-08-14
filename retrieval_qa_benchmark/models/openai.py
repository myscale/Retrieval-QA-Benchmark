from __future__ import annotations

import os
from typing import Any, Dict, Optional

import openai

from retrieval_qa_benchmark.schema import BaseLLM, BaseLLMOutput
from retrieval_qa_benchmark.utils.registry import REGISTRY


@REGISTRY.register_model("remote-llm")
class RemoteLLM(BaseLLM):
    run_args: Dict[str, Any] = {}
    system_prompt: str = "You are a helpful assistant."

    @classmethod
    def build(
        cls,
        name: str = "llama2-13b-chat",
        api_base: str = os.getenv("OPENAI_API_BASE", "http://10.1.3.28:8990/v1"),
        api_key: str = os.getenv(
            "OPENAI_API_KEY", "sk-some-super-secret-key-you-will-never-know"
        ),
        system_prompt: Optional[str] = None,
        run_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> RemoteLLM:
        openai.api_base = api_base
        openai.api_key = api_key
        return cls(
            name=name,
            run_args=run_args or {},
            system_prompt=system_prompt or "",
            **kwargs,
        )

    def _generate(
        self,
        text: str,
    ) -> BaseLLMOutput:
        completion = openai.Completion.create(
            model=self.name,
            prompt="\n".join([self.system_prompt, text]),
            **self.run_args,
        )

        return BaseLLMOutput(
            generated=completion.choices[0].text,
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
        )


@REGISTRY.register_model("gpt35")
class GPT(RemoteLLM):
    @classmethod
    def build(
        cls,
        name: str = "text-davinci-003",
        api_base: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        api_key: str = os.getenv("OPENAI_API_KEY", ""),
        system_prompt: Optional[str] = None,
        run_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> GPT:
        openai.api_base = api_base
        openai.api_key = api_key
        return cls(
            name=name,
            run_args=run_args or {},
            system_prompt=system_prompt or "",
            **kwargs,
        )


@REGISTRY.register_model("chatgpt35")
class ChatGPT(GPT):
    def _generate(
        self,
        text: str = "",
    ) -> BaseLLMOutput:
        completion = openai.ChatCompletion.create(
            model=self.name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            **self.run_args,
        )
        return BaseLLMOutput(
            generated=completion.choices[0].message.content,
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
        )
