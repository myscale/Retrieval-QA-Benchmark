from __future__ import annotations

from typing import List, Dict

from retrieval_qa_benchmark.utils.factory import *
from retrieval_qa_benchmark.schema import QARecord
from retrieval_qa_benchmark.protocols.base import BaseEvaluator
from retrieval_qa_benchmark.utils.registry import REGISTRY


def mcsa_fuzzy_matcher(pred: str, gold: QARecord) -> bool:
    if gold.answer in pred:
        return True
    if f"{chr(65 + gold.choices.index(gold.answer))}." in pred:
        return True
    if pred[0] == f"{chr(65 + gold.choices.index(gold.answer))}":
        return true
    return False


@REGISTRY.register_evaluator("mcsa")
class MCSAEvaluator(BaseEvaluator):
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> MCSAEvaluator:
        obj = super().from_config(config)
        obj.matcher = mcsa_fuzzy_matcher
        return obj
