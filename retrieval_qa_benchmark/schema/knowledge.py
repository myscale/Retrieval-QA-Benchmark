from __future__ import annotations
from typing import List, Optional, Tuple, Any
from pydantic import BaseModel, Extra
from retrieval_qa_benchmark.schema import KnowledgeRecord
from retrieval_qa_benchmark.embeddings.stembedding import BaseEmbedding


class BaseKnowledgebase(BaseModel):
    """Base knowledge base for retrieval tasks"""

    name: str
    retriever: BaseDataStore
    emb_model: BaseEmbedding

    class Config:
        extra = Extra.forbid

    @classmethod
    def build(
        cls,
        name: str,
        retriever: BaseDataStore,
        embedding: BaseEmbedding,
    ) -> BaseKnowledgebase:
        return cls(
            name=f"{name}_{embedding.name}", retriever=retriever, emb_model=embedding
        )

    def retrieve(self, hint: str, k: int) -> List[KnowledgeRecord]:
        return self.retriever.search(hint, k)

    def rerank(self, records: List[KnowledgeRecord], k: int) -> List[KnowledgeRecord]:
        return records[:k]

    def stack(
        self,
        children: List[KnowledgeRecord],
        k: int,
        root: Optional[List[KnowledgeRecord]] = None,
    ) -> Tuple[List[KnowledgeRecord], List[KnowledgeRecord]]:
        if root is None:
            root = []
        result: List[KnowledgeRecord] = []
        for r in children:
            result.extend(self.retrieve(r.context, k))
        root.extend(children)
        root = self.rerank(root, k)
        return root, result

    def chain(self, children: List[KnowledgeRecord], k: int) -> List[KnowledgeRecord]:
        result: List[KnowledgeRecord] = []
        for r in children:
            result.extend(self.retrieve(r.context, k))
        return self.rerank(result, k)

    def __call__(self, hint: str, k: int) -> List[KnowledgeRecord]:
        return self.rerank(self.retrieve(hint, k), k)


class KnowledgeTree(BaseModel):
    """KnowledgeBase Chain"""

    levels: List[BaseKnowledgebase]

    def __call__(self, hint: str, k: int) -> List[KnowledgeRecord]:
        ret = self.levels[0](hint, k)
        for l in self.levels[1:]:
            ret = l.chain(ret, k)
        return ret


class KnowledgeStack(BaseModel):
    """Knowledge Beam Search"""

    stacks: List[BaseKnowledgebase]

    def __call__(self, hint: str, k: int) -> List[KnowledgeRecord]:
        ret = self.stacks[0](hint, k)
        root: List[KnowledgeRecord] = []
        for s in self.stacks[1:]:
            root, ret = s.stack(ret, k, root)
        ret.extend(root)
        ret = self.stacks[-1].rerank(ret, k)
        return ret


class BaseDataStore(BaseModel):
    class Config:
        extra = Extra.allow

    @classmethod
    def build(cls, *args: Any, **kwargs: Any) -> BaseDataStore:
        raise NotImplementedError

    def insert(self, data: List[KnowledgeRecord]) -> None:
        raise NotImplementedError

    def search(self, hint: str, k: int) -> List[KnowledgeRecord]:
        raise NotImplementedError
