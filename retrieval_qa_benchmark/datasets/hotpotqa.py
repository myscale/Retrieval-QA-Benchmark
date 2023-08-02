from __future__ import annotations

from typing import Optional, Sequence

from retrieval_qa_benchmark.schema import HFDataset
from retrieval_qa_benchmark.datasets import build_hfdataset_internal
from retrieval_qa_benchmark.transforms import (
    BaseTransform,
    MultipleChoiceTransform,
    TransformChain,
)
from retrieval_qa_benchmark.utils.registry import REGISTRY


@REGISTRY.register_dataset("hotpot_qa")
class HotpotQA(HFDataset):
    """https://huggingface.co/datasets/hotpot_qa
    Hotpot QA Dataset from Huggingface
    """

    @classmethod
    def build(
        cls,
        subset: str = "fullwiki",
        extra_transforms: Optional[Sequence[BaseTransform]] = [
            MultipleChoiceTransform(
                prompt_prefix="Please answer with the letter of the correct answer.\n"
            ),
        ],
    ) -> HotpotQA:
        transform = BaseTransform()
        if extra_transforms:
            transform = TransformChain(
                chain=[transform, *extra_transforms]
            )  # type: ignore
        name, eval_set = build_hfdataset_internal(
            name=["hotpot_qa", subset], eval_split="validation", transform=transform
        )
        return cls(name=name, eval_set=eval_set)
