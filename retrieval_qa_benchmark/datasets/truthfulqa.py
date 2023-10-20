from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, List, Optional

from retrieval_qa_benchmark.datasets.helper import build_hfdataset_internal
from retrieval_qa_benchmark.schema import BaseDataset, BaseTransform
from retrieval_qa_benchmark.utils.registry import REGISTRY


class TruthfulQATransform(BaseTransform):
    qkey: str = "question"
    ckey: str = "choices"
    akey: str = "answerKey"
    metric: str = "mc2_targets"

    def transform_id(self, data: Dict[str, Any], **params: Any) -> str:
        return sha256((data[self.qkey]).encode("utf-8")).hexdigest()

    def transform_answer(self, data: Dict[str, Any], **params: Any) -> str:
        return '|'.join([c for c, l in zip(data["choices"], data["labels"]) if l > 0])

    def transform_question(self, data: Dict[str, Any], **params: Any) -> str:
        return data[self.qkey]

    def transform_type(self, data: Dict[str, Any], **params: Any) -> str:
        return "MCMA"

    def transform_choices(
        self, data: Dict[str, Any], **params: Any
    ) -> Optional[List[str]]:
        return data[self.ckey]["choices"]


@REGISTRY.register_dataset("truthful_qa_mc")
class TruthfulQA(BaseDataset):
    """https://huggingface.co/datasets/truthful_qa
    Truthful QA Dataset from Huggingface
    """

    @classmethod
    def build(
        cls,
        subset: str = "mc2_targets",
    ) -> TruthfulQA:
        assert subset in ["mc1_targets", "mc2_targets"]
        transform = TruthfulQATransform(metric=subset)
        name, eval_set = build_hfdataset_internal(
            name=["truthful_qa", "multiple_choice"],
            eval_split="validation",
            transform=transform,
        )
        return cls(name=name, eval_set=eval_set)
