from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from retrieval_qa_benchmark.schema.datatypes import QARecord
from retrieval_qa_benchmark.utils.profiler import PROFILER


class BaseLLMOutput(BaseModel):
    generated: str
    prompt_tokens: int
    completion_tokens: int


class BaseLLM(BaseModel):
    model: str
    record_template: str = ("The following are multiple choice questions (with answers):\n\n"
                            "{context}Question: {question}\n{choices}Answer: ")
    context_template: str = "Context:\n{context}\n\n"

    @property
    def tokenizer_type(self) -> str:
        return "tiktoken"

    @classmethod
    def build(cls, *args: Any, **kwargs: Any) -> BaseLLM:
        raise NotImplementedError

    @PROFILER.profile_function("BaseModel.generate")
    def generate(
        self,
        text: str,
    ) -> BaseLLMOutput:
        return self._generate(self.convert_record(text))
    
    def convert_record(self, data: QARecord):
        choices = ""
        if data.choices:
            choices = "\t".join(
                    [f"{chr(65+i)}. {v}" for i, v in enumerate(data.choices)]
                )
            choices += "\n"
        context, context_str = [], ""
        if data.context:
            for i in range(len(data.context)):
                context.append(f"[{i + 1}] {data.context[i]}")
            context_str = self.context_template.format(context='\n'.join(context))
        return self.record_template.format(
            question=data.question,
            choices=choices,
            context=context_str
        )
        
    def _generate(self, text: str) -> BaseLLMOutput:
        raise NotImplementedError
