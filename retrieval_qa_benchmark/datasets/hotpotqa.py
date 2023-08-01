from __future__ import annotations

from retrieval_qa_benchmark.datasets.base import HFDataset, build_hfdataset_internal
from retrieval_qa_benchmark.utils.transforms import BaseTransform


class HotpotQA(HFDataset):
    """https://huggingface.co/datasets/hotpot_qa
    Hotpot QA Dataset from Huggingface
    """

    @classmethod
    def build(cls, subset: str = "fullwiki") -> HotpotQA:
        transform = BaseTransform()
        name, eval_set = build_hfdataset_internal(
            name=["hotpot_qa", subset], eval_split="validation", transform=transform
        )
        return cls(name=name, eval_set=eval_set)
