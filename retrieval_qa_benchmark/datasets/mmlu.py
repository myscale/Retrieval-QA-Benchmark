from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, Optional, Sequence, List

from ast import literal_eval
from retrieval_qa_benchmark.schema import HFDataset
from retrieval_qa_benchmark.datasets.helper import build_hfdataset_internal
from retrieval_qa_benchmark.schema import BaseTransform
from retrieval_qa_benchmark.utils.registry import REGISTRY


class MMLUTransform(BaseTransform):
    qkey: str = "question"
    ckey: str = "choices"
    akey: str = "answer"

    def transform_id(self, data: Dict[str, Any], **params: Any) -> str:
        return sha256(
            (data[self.qkey] + "".join(data[self.ckey])).encode("utf-8")
        ).hexdigest()

    def transform_answer(self, data: Dict[str, Any], **params: Any) -> str:
        return data[self.ckey][data[self.akey]]

    def transform_question(self, data: Dict[str, Any], **params: Any) -> str:
        question = data[self.qkey]
        choices = "\n".join(
            [f"{chr(65+i)}. {v}" for i, v in enumerate(data[self.ckey])]
        )
        return f"{question}\n{choices}"

    def transform_type(self, data: Dict[str, Any], **params: Any) -> str:
        return "MCSA"

    def transform_choices(
        self, data: Dict[str, Any], **params: Any
    ) -> Optional[List[str]]:
        return data["choices"]


@REGISTRY.register_dataset("mmlu")
class MMLU(HFDataset):
    """https://huggingface.co/datasets/hotpot_qa
    Hotpot QA Dataset from Huggingface
    """

    @classmethod
    def build(
        cls,
        subset: str = "prehistory",
    ) -> MMLU:
        transform = MMLUTransform()
        name, eval_set = build_hfdataset_internal(
            name=["cais/mmlu", subset],
            eval_split="test",
            transform=transform,
        )
        return cls(name=name, eval_set=eval_set)
