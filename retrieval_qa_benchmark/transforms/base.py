from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Extra

from retrieval_qa_benchmark.schema import QARecord
from retrieval_qa_benchmark.utils.registry import REGISTRY


@REGISTRY.register_extra_transform("base")
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

    def transform_choices(
        self, data: Dict[str, Any], **params: Any
    ) -> Optional[List[str]]:
        try:
            return data["choices"]
        except:
            return None

    def __call__(self, data: Dict[str, Any]) -> QARecord:
        return QARecord(
            **{k: v for k, v in self.chain(data).items() if v is not None}
        )

    def chain(self, data: Union[QARecord, Dict[str, Any]]) -> Dict[str, Any]:
        result = {}
        if type(data) is QARecord:
            data = data.model_dump()
        for k, f in self.targets.items():
            try:
                result[k] = f(data)
            except Exception as e:
                logger.error(f"Transform function failed on key `{k}`")
                raise e
        return result


class TransformChain(BaseModel):
    """Chain of transformations"""

    chain: Sequence[BaseTransform]

    def __call__(self, data: Dict[str, Any]) -> Any:
        for c in self.chain[:-1]:
            data = c.chain(data)
        return self.chain[-1](data)
