from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, List, Optional

from retrieval_qa_benchmark.datasets.helper import build_hfdataset_internal
from retrieval_qa_benchmark.schema import BaseDataset, BaseTransform
from retrieval_qa_benchmark.utils.registry import REGISTRY


class ARCTransform(BaseTransform):
    qkey: str = "question"
    ckey: str = "choices"
    akey: str = "answerKey"

    def transform_id(self, data: Dict[str, Any], **params: Any) -> str:
        return data["id"]

    def transform_answer(self, data: Dict[str, Any], **params: Any) -> str:
        d = {ck: ct for ck, ct, in zip(data["choices"]["label"], data["choices"]["text"])}
        return d[data[self.akey]]

    def transform_question(self, data: Dict[str, Any], **params: Any) -> str:
        question = data[self.qkey]
        return question

    def transform_type(self, data: Dict[str, Any], **params: Any) -> str:
        return "MCSA"

    def transform_choices(
        self, data: Dict[str, Any], **params: Any
    ) -> Optional[List[str]]:
        return data[self.ckey]["text"]


@REGISTRY.register_dataset("arc")
class ARC(BaseDataset):
    """https://huggingface.co/datasets/ai2_arc
    ARC Dataset from Huggingface
    """

    @classmethod
    def build(
        cls,
        subset: str = "ARC-Easy",
    ) -> ARC:
        transform = ARCTransform()
        name, eval_set = build_hfdataset_internal(
            name=["ai2_arc", subset],
            eval_split="test",
            transform=transform,
        )
        return cls(name=name, eval_set=eval_set)
