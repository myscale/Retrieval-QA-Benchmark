from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class BaseLLM(BaseModel):
    model_name: str

    @classmethod
    def build(cls, *args: Any, **kwargs: Any) -> BaseLLM:
        raise NotImplementedError

    def generate(
        self,
        text: str,
    ) -> str:
        raise NotImplementedError
