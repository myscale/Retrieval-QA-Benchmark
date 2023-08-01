from __future__ import annotations

from typing import List

from pydantic import BaseModel, Extra

from retrieval_qa_benchmark.schema import KnowledgeRecord


class BaseDataStore(BaseModel):
    class Config:
        extra = Extra.allow

    @classmethod
    def build(
        cls,
    ) -> BaseDataStore:
        raise NotImplementedError

    def insert(self, data: List[KnowledgeRecord]) -> None:
        raise NotImplementedError

    def search(self, hint: str, k: int) -> List[KnowledgeRecord]:
        raise NotImplementedError
