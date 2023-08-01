from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class BaseLLM(BaseModel):
    name: str
    model: Any

    @classmethod
    def build(cls, *args: Any, **kwargs: Any) -> BaseLLM:
        raise NotImplementedError

    def generate(
        self,
        text: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.8,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError
