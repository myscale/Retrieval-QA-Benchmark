import re
import time
from typing import Any, Dict, Optional
from huggingface_hub import InferenceClient

from retrieval_qa_benchmark.schema.model import BaseLLM, BaseLLMOutput
from retrieval_qa_benchmark.utils.registry import REGISTRY
from retrieval_qa_benchmark.utils.profiler import PROFILER


@REGISTRY.register_model("tgi")
class TGI_LLM(BaseLLM):
    client: InferenceClient
    system_prompt: str = "You are a helpful assistant."
    boot_time_key: str = "model.tgi.boot.profile"
    completion_time_key: str = "model.tgi.completion.profile"
    stream: bool = False

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def build(
        cls,
        name: str = "llama2-13b-chat-hf",
        backend_url: str = "http://127.0.0.1:8080",
        system_prompt: Optional[str] = None,
        run_args: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> "TGI_LLM":
        client = InferenceClient(model=backend_url)
        return cls(
            name=name,
            client=client,
            run_args=run_args or {},
            system_prompt=system_prompt or "",
            stream=stream,
            **kwargs,
        )

    def _generate(self, text: str) -> BaseLLMOutput:
        if self.stream:
            pred_str = ""
            cnt = 0
            for name in [self.boot_time_key, self.completion_time_key]:
                if name not in self.counter:
                    PROFILER.counter[name] = 0
                if name not in self.accumulator:
                    PROFILER.accumulator[name] = 0
            t0 = time.time()
            stream = self.client.text_generation(
                "\n".join([self.system_prompt, text]),
                stream=True, details=True, **self.run_args)
            for i, token in enumerate(stream):
                if i == 0:
                    t_boot = time.time()
                if not token.token.special:
                    pred_str += token.token.text
                cnt += 1
            t_gen = (time.time() - t_boot) * 1000
            t_boot = (t_boot - t0) * 1000 # ms
            PROFILER.accumulator[self.boot_time_key] += t_boot
            PROFILER.accumulator[self.completion_time_key] += t_gen
            PROFILER.counter[self.boot_time_key] += 1
            PROFILER.counter[self.completion_time_key] += cnt - 1
            return BaseLLMOutput(
                    generated=pred_str,
                    completion_tokens=cnt,
                    # This is NOT returned by TGI so we just count words.
                    prompt_tokens=len(re.findall('\w+', pred_str)),
                )
        else:
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