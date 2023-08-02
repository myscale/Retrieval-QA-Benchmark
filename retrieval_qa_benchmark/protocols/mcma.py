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

@REGISTRY.register_evaluator('mcma')
class MCMAEvaluator(BaseEvaluator):
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> MCMAEvaluator:
        dataset = DatasetFactory.from_config(config["dataset"]).build()
        transform = TransformChainFactory(
            chain_config=[TransformFactory.from_config(c) for c in config["transform_chain"]]).build()
        model = ModelFactory.from_config(config['model']).build()
        return cls(dataset=dataset, transform=transform, out_file=config['out_file'], llm=model)