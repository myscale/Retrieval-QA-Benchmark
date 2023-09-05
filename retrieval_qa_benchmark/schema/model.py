from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel

from retrieval_qa_benchmark.schema.datatypes import QARecord


class BaseLLMOutput(BaseModel):
    generated: str
    """generated text in plain string"""
    prompt_tokens: int
    """number of input tokens"""
    completion_tokens: int
    """number of generated tokens"""


class BaseLLM(BaseModel):
    name: str
    """name of the model, like `gpt-3.5-turbo` or `llama2-13b-chat`"""
    record_template: str = (
        "The following are multiple choice questions (with answers) with context:\n\n"
        "{context}Question: {question}\n{choices}Answer: "
    )
    """template to convert :class:`QARecord` into string"""
    context_template: str = "Context:\n{context}\n\n"
    """template to inject contexts"""
    run_args: Dict[str, Any] = {}
    """Runtime keyword arguments"""

    @property
    def tokenizer_type(self) -> str:
        return "tiktoken"

    @classmethod
    def build(cls, **kwargs: Any) -> BaseLLM:
        return cls(**kwargs)

    def generate(
        self,
        text: QARecord,
    ) -> BaseLLMOutput:
        return self._generate(self.convert_record(text))

    def convert_record(self, data: QARecord) -> str:
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
            context_str = self.context_template.format(context="\n".join(context))
        return self.record_template.format(
            question=data.question, choices=choices, context=context_str
        )

    def _generate(self, text: str) -> BaseLLMOutput:
        raise NotImplementedError
