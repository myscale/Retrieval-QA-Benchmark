import sys
from os import path

import yaml

from retrieval_qa_benchmark.schema import QARecord
from retrieval_qa_benchmark.utils.factory import DatasetFactory

sys.path.insert(0, path.dirname(path.split(__file__)[0]))
from dummy import *  # noqa: F403, E402


def test_dataset_add() -> None:
    from os import path

    config = yaml.safe_load(
        open(
            path.join(
                path.dirname(path.realpath(__file__)), "../assets/test_dataset.yaml"
            )
        )
    )
    config["evaluator"]["dataset"][0]["args"] = {}
    config["evaluator"]["dataset"][0]["args"]["lst"] = [
        QARecord(
            id=str(i), question=f"hello! record {i}", type="dummy", answer=f"record {i}"
        )
        for i in range(10)
    ]
    config["evaluator"]["dataset"][1]["args"] = {}
    config["evaluator"]["dataset"][1]["args"]["lst"] = [
        QARecord(
            id=str(i), question=f"hello! record {i}", type="dummy", answer=f"record {i}"
        )
        for i in range(100, 110)
    ]
    d1 = DatasetFactory.from_config(config["evaluator"]["dataset"][0]).build()
    d2 = DatasetFactory.from_config(config["evaluator"]["dataset"][1]).build()
    d = d1 + d2
    assert len(d) == 20

    for n in range(20):
        if n < 10:
            assert d[n].id == str(n)
        else:
            assert d[n].id == str(90 + n)
