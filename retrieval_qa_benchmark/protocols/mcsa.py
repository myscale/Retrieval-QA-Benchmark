from __future__ import annotations

from typing import Any, Dict, cast

from retrieval_qa_benchmark.protocols.base import BaseEvaluator
from retrieval_qa_benchmark.schema import QARecord
from retrieval_qa_benchmark.utils.registry import REGISTRY


def mcsa_fuzzy_matcher(pred: str, gold: QARecord) -> bool:
    pred = pred.split("\n\n")[0]
    pred = ' '.join([p for p in pred.split(' ') if p != ''])
    if gold.answer in pred:
        return True
    if gold.choices is not None:
        gold_id = gold.choices.index(gold.answer)
        if f"{chr(65 + gold_id)}." in pred and sum([f"{chr(65 + n)}." in pred for n in range(len(gold.choices))]) == 1:
            return True
        if len(pred) == 1 and pred[0] == f"{chr(65 + gold.choices.index(gold.answer))}":
            return True
    return False


@REGISTRY.register_evaluator("mcsa")
class MCSAEvaluator(BaseEvaluator):
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> MCSAEvaluator:
        obj = super().from_config(config)
        obj.matcher = mcsa_fuzzy_matcher
        return cast(MCSAEvaluator, obj)
