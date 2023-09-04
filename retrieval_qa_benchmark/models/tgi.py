from typing import Any, Optional, Dict
from huggingface_hub import InferenceClient
from retrieval_qa_benchmark.models import RemoteLLM
from retrieval_qa_benchmark.schema.datatypes import QARecord
from retrieval_qa_benchmark.schema.model import BaseLLMOutput
from retrieval_qa_benchmark.utils.registry import REGISTRY
from retrieval_qa_benchmark.utils.profiler import PROFILER


@REGISTRY.register_model("tgi")
class TGI_LLM(RemoteLLM):
    client: InferenceClient

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def build(
        cls,
        name: str = "llama2-13b-chat-hf",
        backend_url: str = "http://127.0.0.1:8080",
        system_prompt: Optional[str] = None,
        run_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "TGI_LLM":
        client = InferenceClient(model=backend_url)
        return cls(
            name=name,
            client=client,
            run_args=run_args or {},
            system_prompt=system_prompt or "",
            **kwargs,
        )
    def _generate(self, text: str) -> BaseLLMOutput:
        resp = self.client.text_generation(
            "\n".join([self.system_prompt, text]), 
            **self.run_args, 
            details=True,
            decoder_input_details=True,
        )
        return BaseLLMOutput(
            generated=resp.generated_text,
            completion_tokens=resp.details.generated_tokens,
            prompt_tokens=len(resp.details.prefill),
        )
