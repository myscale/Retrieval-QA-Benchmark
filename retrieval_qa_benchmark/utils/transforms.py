from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Sequence

from loguru import logger
from pydantic import BaseModel, Extra

from retrieval_qa_benchmark.schema import QARecord


class BaseTransform(BaseModel):
    """Base transformation to elements in datasets"""

    class Config:
        extra = Extra.allow

    @property
    def targets(self) -> Dict[str, Callable[[Dict[str, Any]], str]]:
        return {
            k: getattr(self, f"transform_{k}") for k in QARecord.model_fields.keys()
        }

    def transform_id(self, data: Dict[str, Any], **params: Any) -> str:
        return str(data["id"])

    def transform_question(self, data: Dict[str, Any], **params: Any) -> str:
        return str(data["question"])

    def transform_answer(self, data: Dict[str, Any], **params: Any) -> str:
        return str(data["answer"])

    def transform_type(self, data: Dict[str, Any], **params: Any) -> str:
        return str(data["type"])

    def __call__(self, data: Dict[str, Any]) -> QARecord:
        return QARecord(**{k: str(v) for k, v in self.chain(data).items()})

    def chain(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data_ = data.copy()
        result = {}
        for k, f in self.targets.items():
            try:
                result[k] = f(data_)
            except Exception as e:
                logger.error(f"Transform function failed on key `{k}`")
                raise e
        return result


class MultipleChoiceTransform(BaseTransform):
    sep_chr: str = "\n"
    prompt_prefix: str = ""
    prompt_suffix: str = ""

    def transform_question(self, data: Dict[str, Any], **params: Any) -> str:
        return self.sep_chr.join(
            [self.prompt_prefix, data["question"], self.prompt_suffix]
        )


class TransformChain(BaseModel):
    """Chain of transformations"""

    chain: Sequence[BaseTransform]

    def __call__(self, data: Dict[str, Any]) -> Any:
        for c in self.chain[:-1]:
            data = c.chain(data)
        return self.chain[-1](data)
