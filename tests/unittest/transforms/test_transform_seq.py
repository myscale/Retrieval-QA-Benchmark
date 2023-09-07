import random
import sys
from os import path

import pytest
import yaml

from retrieval_qa_benchmark.schema import QARecord
from retrieval_qa_benchmark.utils.factory import TransformGraphFactory

sys.path.insert(0, path.dirname(path.split(__file__)[0]))
from dummy import *  # noqa: F403, E402


@pytest.mark.parametrize("num_base", [random.randint(0, 100) for _ in range(10)])
def test_seq(num_base: int) -> None:
    from os import path

    num_base = 20
    config = yaml.safe_load(
        open(
            path.join(
                path.dirname(path.realpath(__file__)),
                "../assets/test_transform_seq.yaml",
            )
        )
    )
    graph = TransformGraphFactory.from_config(
        config=config["evaluator"]["transform"]
    ).build()
    d = graph(
        QARecord(
            id="test1", question="*" * num_base, answer="answer for test 1", type="open"
        )
    )
    assert graph.nodes["0"].children == [graph.nodes["1"], graph.nodes["1"]]
    assert d.question[:18] == "dummy3dummy2dummy1"
