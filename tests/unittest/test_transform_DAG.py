from typing import Any, Dict
import pytest
import random
import yaml
from retrieval_qa_benchmark.schema import BaseTransform, TransformChain, QARecord
from retrieval_qa_benchmark.utils.factory import TransformChainFactory
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


@pytest.mark.parametrize("num_base", [random.randint(0, 100) for _ in range(10)])
def test_DAG_1(num_base):
    import math
    from os import path

    num_base = 20
    config = yaml.safe_load(
        open(
            path.join(
                path.dirname(path.realpath(__file__)), "assets/test_transform_DAG.yaml"
            )
        )
    )
    chain = TransformChainFactory(
        chain_config=config["evaluator"]["transform_chain"]
    ).build()
    d = chain(
        QARecord(
            id="test1", question="*" * num_base, answer="answer for test 1", type="open"
        )
    )
    assert (math.ceil((100 - num_base) / 12) + 1) * 12 + num_base == len(
        d.question
    ), "Execution count mismatched"
    assert d.question[:6] == "dummy3", "Final state should be dummy"


if __name__ == "__main__":
    test_DAG_1()
