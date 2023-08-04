from __future__ import annotations

from typing import List, Dict

from retrieval_qa_benchmark.utils.factory import *
from retrieval_qa_benchmark.schema import QARecord
from retrieval_qa_benchmark.protocols.base import BaseEvaluator
from retrieval_qa_benchmark.utils.registry import REGISTRY


def mcma_fuzzy_matcher(pred: str, gold: QARecord) -> bool:
    if gold.answer in pred:
        return True
    if f"{chr(65 + gold.choices.index(gold.answer))}." in pred:
        return True
    return False


@REGISTRY.register_evaluator("mcma")
class MCMAEvaluator(BaseEvaluator):
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> MCMAEvaluator:
        obj = super().from_config(config)
        obj.matcher = mcma_fuzzy_matcher
        return obj
