from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, Optional, Sequence

from retrieval_qa_benchmark.datasets.base import HFDataset, build_hfdataset_internal
from retrieval_qa_benchmark.utils.transforms import (
    BaseTransform,
    MultipleChoiceTransform,
    TransformChain,
)


class MMLUTransform(BaseTransform):
    qkey: str = "question"
    ckey: str = "choices"
    akey: str = "answer"

    def transform_id(self, data: Dict[str, Any], **params: Any) -> str:
        return sha256(
            (data[self.qkey] + "".join(data[self.ckey])).encode("utf-8")
        ).hexdigest()

    def transform_answer(self, data: Dict[str, Any], **params: Any) -> str:
        return f"{chr(65 + data[self.akey])}. {data[self.ckey][data[self.akey]]}"

    def transform_question(self, data: Dict[str, Any], **params: Any) -> str:
        question = data[self.qkey]
        choices = "\t".join(
            [f"{chr(65+i)}. {v}" for i, v in enumerate(data[self.ckey])]
        )
        return f"{question}\n{choices}"

    def transform_type(self, data: Dict[str, Any], **params: Any) -> str:
        return "MCSA"


class MMLU(HFDataset):
    """https://huggingface.co/datasets/hotpot_qa
    Hotpot QA Dataset from Huggingface
    """

    @classmethod
    def build(
        cls,
        subset: str = "prehistory",
        extra_transforms: Optional[Sequence[BaseTransform]] = [
            MultipleChoiceTransform(
                prompt_prefix="Please answer with the letter of the correct answer.\n"
            ),
        ],
    ) -> MMLU:
        transform = MMLUTransform()
        if extra_transforms:
            transform = TransformChain(chain=[transform, *extra_transforms])  # type: ignore
        name, eval_set = build_hfdataset_internal(
            name=["cais/mmlu", subset],
            eval_split="test",
            transform=transform,
        )
        return cls(name=name, eval_set=eval_set)
