from typing import Any

from retrieval_qa_benchmark.schema import BaseDataset, QARecord
from retrieval_qa_benchmark.utils.registry import REGISTRY


@REGISTRY.register_dataset("Dummy")
class DummyDataset(BaseDataset):
    @classmethod
    def build(cls, lst: list[QARecord], **kwargs: Any) -> "DummyDataset":
        return cls(name="Dummy", eval_set=lst)
