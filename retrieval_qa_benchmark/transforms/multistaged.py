from typing import Any

from retrieval_qa_benchmark.transforms.base import BaseContextTransform
from retrieval_qa_benchmark.transforms.search import RerankSearcher
from retrieval_qa_benchmark.utils.registry import REGISTRY


@REGISTRY.register_transform("RRFHybrid")
class ContextWithRRFHybrid(BaseContextTransform):
    rank_dict: dict = {"mpnet": 30, "bm25": 40}
    with_title: int = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._searcher = RerankSearcher(
            rank_dict=self.rank_dict,
            with_title=self.with_title,
        )
