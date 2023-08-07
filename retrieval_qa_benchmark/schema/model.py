from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from retrieval_qa_benchmark.utils.profiler import PROFILER

class BaseLLMOutput(BaseModel):
    generated: str
    prompt_tokens: int
    completion_tokens: int


class BaseLLM(BaseModel):
    model_name: str

    @property
    def tokenizer_type(self) -> str:
        return 'tiktoken'

    @classmethod
    def build(cls, *args: Any, **kwargs: Any) -> BaseLLM:
        raise NotImplementedError
    
    @PROFILER.profile_function("BaseModel.generate")
    def generate(
        self,
        text: str,
    ) -> BaseLLMOutput:
        return self._generate(text)

    def _generate(self, text:str) -> BaseLLMOutput:
        raise NotImplementedError