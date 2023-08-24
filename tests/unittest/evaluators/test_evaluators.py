import random
import sys
from os import path

import pytest
import yaml

from retrieval_qa_benchmark.schema import QARecord
from retrieval_qa_benchmark.utils.factory import EvaluatorFactory

sys.path.insert(0, path.dirname(path.split(__file__)[0]))
from dummy import *  # noqa: F403, E402


@pytest.mark.parametrize("num_base", [random.randint(20, 100) for _ in range(10)])
def test_evaluator_1(num_base: int) -> None:
    import math
    from os import path

    config = yaml.safe_load(
        open(
            path.join(
                path.dirname(path.realpath(__file__)), "../assets/test_evaluator.yaml"
            )
        )
    )
    config["evaluator"]["dataset"]["args"] = {}
    config["evaluator"]["dataset"]["args"]["lst"] = [
        QARecord(
            id=str(i), question=f"hello! record {i}", type="dummy", answer=f"record {i}"
        )
        for i in range(num_base)
    ]

    def matcher(x: str, y: QARecord) -> float:
        score = float(x[-len("hello world!") :] == "hello world!")
        x = x.split("hello world!")[0]
        ori = "hello!" + y.question.split("hello!")[-1]
        cnt = math.ceil((101 - len(ori)) / 12) * 12 + 12 + len(ori) % 6
        score += float(cnt == len(y.question))
        score += y.question.split("dummy1")[-1].split("hello! ")[-1] == y.answer
        return score / 3

    evaluator = EvaluatorFactory.from_config(config).build()
    evaluator.matcher = matcher
    score, res = evaluator()

    assert score == 100
