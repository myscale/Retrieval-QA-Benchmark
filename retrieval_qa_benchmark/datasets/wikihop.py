from __future__ import annotations

from typing import Any, Dict, List, Optional

from retrieval_qa_benchmark.datasets.helper import build_hfdataset_internal
from retrieval_qa_benchmark.schema import BaseDataset, BaseTransform
from retrieval_qa_benchmark.utils.registry import REGISTRY


class WikiHopTransform(BaseTransform):
    qkey: str = "question"
    ckey: str = "candidates"
    akey: str = "answer"

    def transform_answer(self, data: Dict[str, Any], **params: Any) -> str:
        return data[self.akey]

    def transform_question(self, data: Dict[str, Any], **params: Any) -> str:
        question = data[self.qkey]
        return question

    def transform_type(self, data: Dict[str, Any], **params: Any) -> str:
        return "MCSA"

    def transform_choices(
        self, data: Dict[str, Any], **params: Any
    ) -> Optional[List[str]]:
        return data["candidates"]


@REGISTRY.register_dataset("wikihop")
class WikiHop(BaseDataset):
    """https://huggingface.co/datasets/wikihop
    WikiHop Dataset from Huggingface
    """

    @classmethod
    def build(
        cls,
        subset: str = "original",
    ) -> WikiHop:
        transform = WikiHopTransform()
        name, eval_set = build_hfdataset_internal(
            name=["wiki_hop", subset],
            eval_split="validation",
            transform=transform,
        )
        return cls(name=name, eval_set=eval_set)
