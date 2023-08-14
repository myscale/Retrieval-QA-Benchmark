import random
from typing import Any, Dict

import pytest
import yaml

from retrieval_qa_benchmark.schema import BaseTransform, QARecord
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
def test_seq_1(num_base: int) -> None:
    from os import path

    num_base = 20
    config = yaml.safe_load(
        open(
            path.join(
                path.dirname(path.realpath(__file__)), "assets/test_transform_seq.yaml"
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
    assert chain.chain["0"].children == (chain.chain["1"], chain.chain["1"])
    assert d.question[:18] == "dummy3dummy2dummy1"


if __name__ == "__main__":
    test_seq_1(20)
