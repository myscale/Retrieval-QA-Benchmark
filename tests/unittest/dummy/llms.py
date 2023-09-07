import random
from typing import List

from retrieval_qa_benchmark.schema import BaseLLM
from retrieval_qa_benchmark.schema.model import BaseLLMOutput
from retrieval_qa_benchmark.utils.registry import REGISTRY


@REGISTRY.register_model("Dummy")
class DummyLLM(BaseLLM):
    name: str = "dummy"

    def generate(self, text: str) -> BaseLLMOutput:
        ret = f"{text}: hello world!"
        return BaseLLMOutput(
            generated=ret, prompt_tokens=len(text), completion_tokens=len(ret)
        )


@REGISTRY.register_model("RandomParrot")
class RandomParrotLLM(BaseLLM):
    name: str = "dummy_sql_agent"

    dummies: List[str] = []

    def generate(self, text: str) -> BaseLLMOutput:
        ret = random.choice(self.dummies)
        return BaseLLMOutput(
            generated=ret, prompt_tokens=len(text), completion_tokens=len(ret)
        )
