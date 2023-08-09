from __future__ import annotations

from typing import Any, Dict, List, Optional

from retrieval_qa_benchmark.datasets.helper import build_hfdataset_internal
from retrieval_qa_benchmark.schema import BaseDataset, BaseTransform
from retrieval_qa_benchmark.utils.registry import REGISTRY


class HotpotQATransform(BaseTransform):
    def transform_choices(
        self, data: Dict[str, Any], **params: Any
    ) -> Optional[List[str]]:
        return None

    def transform_question(self, data: Dict[str, Any], **params: Any) -> str:
        return f"Question: {data['question']}\n"


@REGISTRY.register_dataset("hotpot_qa")
class HotpotQA(BaseDataset):
    """https://huggingface.co/datasets/hotpot_qa
    Hotpot QA Dataset from Huggingface
    """

    @classmethod
    def build(
        cls,
        subset: str = "fullwiki",
    ) -> HotpotQA:
        transform = HotpotQATransform()
        name, eval_set = build_hfdataset_internal(
            name=["hotpot_qa", subset], eval_split="validation", transform=transform
        )
        return cls(name=name, eval_set=eval_set)
