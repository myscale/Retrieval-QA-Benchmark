from typing import Callable

from retrieval_qa_benchmark.schema import BaseEvaluator, QARecord
from retrieval_qa_benchmark.utils.registry import REGISTRY


def dummy_matcher(pred: str, gold: QARecord) -> float:
    return len(pred) / 100


@REGISTRY.register_evaluator("Dummy")
class DummyEvaluator(BaseEvaluator):
    matcher: Callable[[str, QARecord], float] = dummy_matcher
