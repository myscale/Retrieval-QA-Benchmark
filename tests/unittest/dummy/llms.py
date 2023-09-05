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
