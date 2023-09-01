from __future__ import annotations

from typing import Callable

from retrieval_qa_benchmark.schema import BaseEvaluator, QARecord
from retrieval_qa_benchmark.utils.registry import REGISTRY


def mcsa_fuzzy_matcher(pred_str: str, gold: QARecord) -> float:
    for pred in pred_str.split("\n\n"):
        pred = " ".join([p for p in pred.split(" ") if p != ""])
        if gold.answer in pred:
            return 1.0
        if gold.choices is not None:
            gold_id = gold.choices.index(gold.answer)
            if (
                f"{chr(65 + gold_id)}." in pred
                and sum([f"{chr(65 + n)}." in pred for n in range(len(gold.choices))]) == 1
            ):
                return 1.0
            if len(pred) == 1 and pred[0] == f"{chr(65 + gold.choices.index(gold.answer))}":
                return 1.0
    return 0.0


@REGISTRY.register_evaluator("mcsa")
class MCSAEvaluator(BaseEvaluator):
    matcher: Callable[[str, QARecord], float] = mcsa_fuzzy_matcher
