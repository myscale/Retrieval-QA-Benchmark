from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, Extra

from retrieval_qa_benchmark.schema.datatypes import QARecord


class BaseDataset(BaseModel):
    """Base class dataset"""

    name: str = "dataset"
    eval_set: List[QARecord] = []

    class Config:
        extra = Extra.forbid

    @classmethod
    def build(cls, *args: Any, **kwargs: Any) -> BaseDataset:
        raise NotImplementedError("Please implement a `.build` function")

    def __getitem__(self, index: int) -> QARecord:
        return self.eval_set[index]

    def __len__(self) -> int:
        return len(self.eval_set)
