from typing import Any, Dict

from retrieval_qa_benchmark.schema import BaseTransform
from retrieval_qa_benchmark.utils.registry import REGISTRY


@REGISTRY.register_transform("dummy1")
class Dummy_1(BaseTransform):
    def check_status(self, current: Dict[str, Any]) -> bool:
        return len(current["question"]) > 100

    def transform_question(self, data: Dict[str, Any], **params: Any) -> str:
        return "dummy1" + data["question"]


@REGISTRY.register_transform("dummy2")
class Dummy_2(BaseTransform):
    def transform_question(self, data: Dict[str, Any], **params: Any) -> str:
        return "dummy2" + data["question"]


@REGISTRY.register_transform("dummy3")
class Dummy_3(BaseTransform):
    def transform_question(self, data: Dict[str, Any], **params: Any) -> str:
        return "dummy3" + data["question"]
