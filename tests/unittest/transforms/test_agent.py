import random
import sys
from os import path

import pytest
import yaml

from retrieval_qa_benchmark.schema import QAPrediction, QARecord
from retrieval_qa_benchmark.utils.factory import TransformGraphFactory

sys.path.insert(0, path.dirname(path.split(__file__)[0]))
from dummy import *  # noqa: F403, E402


@pytest.mark.parametrize(
    "dummy_questions",
    [f"Dummy Question #{random.randint(0, 65535)}" for _ in range(100)],
)
def test_agent(dummy_questions: str) -> None:
    from os import path

    record = QARecord(
        id=dummy_questions, question=dummy_questions, answer=dummy_questions, type="qa"
    )
    config = yaml.safe_load(
        open(
            path.join(
                path.dirname(path.realpath(__file__)),
                "../assets/test_agent.yaml",
            )
        )
    )
    transform = TransformGraphFactory.from_config(
        config=config["evaluator"]["transform"]
    ).build()
    out = transform(record)
    assert type(out) is QAPrediction
    assert out.generated == "The answer is dummy."
