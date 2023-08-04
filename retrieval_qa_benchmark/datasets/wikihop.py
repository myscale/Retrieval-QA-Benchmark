from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, List

from retrieval_qa_benchmark.schema import HFDataset
from retrieval_qa_benchmark.datasets.helper import build_hfdataset_internal
from retrieval_qa_benchmark.transforms import (
    BaseTransform,
    MultipleChoiceTransform,
    TransformChain,
)
from retrieval_qa_benchmark.utils.registry import REGISTRY


class WikiHopTransform(BaseTransform):
    qkey: str = "question"
    ckey: str = "candidates"
    akey: str = "answer"

    def transform_answer(self, data: Dict[str, Any], **params: Any) -> str:
        return data[self.akey]

    def transform_choices(
        self, data: Dict[str, Any], **params: Any
    ) -> Optional[List[str]]:
        return data[self.ckey]

    def transform_question(self, data: Dict[str, Any], **params: Any) -> str:
        question = data[self.qkey]
        choices = "\t".join(
            [f"{chr(65+i)}. {v}." for i, v in enumerate(data[self.ckey])]
        )
        return f"Question: {question}\nChoices: {choices}"

    def transform_type(self, data: Dict[str, Any], **params: Any) -> str:
        return "MCSA"

    def transform_choices(
        self, data: Dict[str, Any], **params: Any
    ) -> Optional[List[str]]:
        return data["candidates"]


@REGISTRY.register_dataset("wikihop")
class WikiHop(HFDataset):
    """https://huggingface.co/datasets/hotpot_qa
    Hotpot QA Dataset from Huggingface
    """

    @classmethod
    def build(
        cls,
        subset: str = "original",
        extra_transforms: Optional[Sequence[BaseTransform]] = [
            MultipleChoiceTransform(
                prompt_prefix="Please answer with the letter of the correct answer.\n"
            ),
        ],
    ) -> WikiHop:
        transform = WikiHopTransform()
        if extra_transforms:
            transform = TransformChain(
                chain=[transform, *extra_transforms]
            )  # type: ignore
        name, eval_set = build_hfdataset_internal(
            name=["wiki_hop", subset],
            eval_split="validation",
            transform=transform,
        )
        return cls(name=name, eval_set=eval_set)
