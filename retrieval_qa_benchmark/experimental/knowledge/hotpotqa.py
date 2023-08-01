from typing import List

from retrieval_qa_benchmark.knowledge.base import BaseKnowledgebase
from retrieval_qa_benchmark.schema import KnowledgeRecord


class HotpotQAKnowledgeBase(BaseKnowledgebase):
    @classmethod
    def build(
        cls,
    ) -> BaseKnowledgebase:
        raise NotImplementedError("`build` must be implemented.")

    def retrieve(self, hint: str, k: int) -> List[KnowledgeRecord]:
        raise NotImplementedError("`retrieve` must be implemented.")
