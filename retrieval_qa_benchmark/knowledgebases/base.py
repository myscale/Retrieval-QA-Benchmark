from __future__ import annotations

from abc import abstractmethod
from typing import List, Optional, Tuple, Any
from pydantic import BaseModel

from retrieval_qa_benchmark.schema import KnowledgRecord


class BaseKnowledgebase(BaseModel):
    """Base knowledge base for retrieval tasks"""

    client: Any

    @classmethod
    def build(
        cls,
    ) -> BaseKnowledgebase:
        raise NotImplementedError("`build` must be implemented.")

    @abstractmethod
    def retrieve(self, hint: str, k: int) -> List[KnowledgRecord]:
        raise NotImplementedError("`retrieve` must be implemented.")

    def __getitem__()

    def rerank(self, records: List[KnowledgRecord], k: int) -> List[KnowledgRecord]:
        return records[:k]

    def stack(
        self,
        children: List[KnowledgRecord],
        k: int,
        root: Optional[List[KnowledgRecord]] = None,
    ) -> Tuple[List[KnowledgRecord], List[KnowledgRecord]]:
        if root is None:
            root = []
        result: List[KnowledgRecord] = []
        for r in children:
            result.extend(self.retrieve(r.context, k))
        root.extend(children)
        root = self.rerank(root, k)
        return root, result

    def chain(self, children: List[KnowledgRecord], k: int) -> List[KnowledgRecord]:
        result: List[KnowledgRecord] = []
        for r in children:
            result.extend(self.retrieve(r.context, k))
        return self.rerank(result, k)

    def __call__(self, hint: str, k: int) -> List[KnowledgRecord]:
        return self.rerank(self.retrieve(hint, k), k)


class KnowledgeTree(BaseModel):
    """KnowledgeBase Chain"""

    levels: List[BaseKnowledgebase]

    def __call__(self, hint: str, k: int) -> List[KnowledgRecord]:
        ret = self.levels[0](hint, k)
        for l in self.levels[1:]:
            ret = l.chain(ret)
        return ret


class KnowledgeStack(BaseModel):
    """Knowledge Beam Search"""

    stacks: List[BaseKnowledgebase]

    def __call__(self, hint: str, k: int) -> List[KnowledgRecord]:
        ret = self.stacks[0](hint, k)
        root = None
        for s in self.stacks[1:]:
            root, ret = s.stack(ret, k, root)
        ret.extend(root)
        ret = self.stacks[-1].rerank(ret)
        return ret
