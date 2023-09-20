from typing import Any, Dict, Tuple

from retrieval_qa_benchmark.schema import QARecord
from retrieval_qa_benchmark.transforms.agents.base import AgentRouter, AgentTool
from retrieval_qa_benchmark.utils.registry import REGISTRY


@REGISTRY.register_transform("DummyTool#1")
class DummyTool1(AgentTool):
    name: str = "dummy1"
    descrption: str = "This is dummy tool 1"

    def execute_action(self, record: QARecord) -> Tuple[str, int, int]:
        return "I got some observation from dummy 1. ", 0, 0


@REGISTRY.register_transform("DummyTool#2")
class DummyTool2(AgentTool):
    name: str = "dummy2"
    descrption: str = "This is dummy tool 2"

    def execute_action(self, record: QARecord) -> Tuple[str, int, int]:
        return "I got some observation from dummy 2. ", 0, 0


@REGISTRY.register_transform("DummyTool#3")
class DummyTool3(AgentTool):
    name: str = "dummy3"
    descrption: str = "This is dummy tool 3"

    def execute_action(self, record: QARecord) -> Tuple[str, int, int]:
        return "I got some observation from dummy 3. ", 10, 20


@REGISTRY.register_transform("DummyRouter")
class DummyAgentRouter(AgentRouter):
    llm_model: Dict[str, Any] = {
        "type": "RandomParrot",
        "args": {
            "dummies": [
                (
                    "I need to take a look at dummy1.\nAction: dummy1\n"
                    "Action Input: some dummy input for dummy 1.\n"
                    "Observation: some hallucination..."
                ),
                (
                    "I need to take a look at dummy2.\nAction: dummy2\n"
                    "Action Input: some dummy input for dummy 2"
                ),
                (
                    "I need to take a look at dummy3.\nAction: dummy3\n"
                    "Action Input: some dummy input for dummy 3\n"
                    "Observation:"
                ),
                "I now know the answer.\nFinal Answer: The answer is dummy.",
            ]
        },
    }

    def execute_action(self, record: QARecord) -> Tuple[str, int, int]:
        p, pt, ct = super().execute_action(record)
        return p, pt, ct
