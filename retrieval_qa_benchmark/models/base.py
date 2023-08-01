from __future__ import annotations
from typing import Any
from pydantic import BaseModel


class BaseLLM(BaseModel):
    name: str
    model: Any

    @classmethod
    def build(cls, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def generate(self, text: str, temperature=0, **kwargs):
        raise NotImplementedError
