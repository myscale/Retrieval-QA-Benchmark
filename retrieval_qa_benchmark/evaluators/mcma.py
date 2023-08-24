from __future__ import annotations

from typing import Callable

from retrieval_qa_benchmark.schema import BaseEvaluator, QARecord
from retrieval_qa_benchmark.utils.registry import REGISTRY


def mcma_fuzzy_matcher(pred: str, gold: QARecord) -> float:
    pred = pred.split("\n\n")[0]
    pred = " ".join([p for p in pred.split(" ") if p != ""])
    gold_answer = gold.answer.split(",")
    cnt = 0
    for gans in gold_answer:
        if gans in pred:
            cnt += 1
        if gold.choices is not None:
            if (
                len(pred) == 1
                and pred[0] == f"{chr(65 + gold.choices.index(gold.answer))}"
            ):
                cnt += 1
    return cnt / len(gans)


@REGISTRY.register_evaluator("mcma")
class MCMAEvaluator(BaseEvaluator):
    matcher: Callable[[str, QARecord], float] = mcma_fuzzy_matcher
