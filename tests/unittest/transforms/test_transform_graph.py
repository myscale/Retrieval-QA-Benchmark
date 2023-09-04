import random
import sys
from os import path

import pytest
import yaml

from retrieval_qa_benchmark.schema import QARecord
from retrieval_qa_benchmark.utils.factory import TransformGraphFactory

sys.path.insert(0, path.dirname(path.split(__file__)[0]))
from dummy import *  # noqa: F403, E402


@pytest.mark.parametrize("num_base", [random.randint(20, 100) for _ in range(100)])
def test_transfor_graph(num_base: int) -> None:
    import math
    from os import path

    config = yaml.safe_load(
        open(
            path.join(
                path.dirname(path.realpath(__file__)),
                "../assets/test_transform_graph.yaml",
            )
        )
    )
    graph = TransformGraphFactory(nodes_config=config["evaluator"]["transform"]).build()
    d = graph(
        QARecord(
            id="test1", question="*" * num_base, answer="answer for test 1", type="open"
        )
    )
    cnt = math.ceil((101 - num_base) / 12)
    assert cnt * 12 + 18, "Execution count mismatched"
    assert d.question[:6] == "dummy3", "Final state should be dummy"
