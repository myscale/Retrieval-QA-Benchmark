from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel


class BaseEmbedding(BaseModel):
    name: str
    model: Any
    dim: int

    @classmethod
    def build(cls, *args: Any, **kwargs: Any) -> BaseEmbedding:
        raise NotImplementedError

    def encode(self, text: str) -> List[float]:
        raise NotImplementedError

    def batch_encode(self, text: List[str], batch_size: int = 32) -> List[List[float]]:
        raise NotImplementedError
