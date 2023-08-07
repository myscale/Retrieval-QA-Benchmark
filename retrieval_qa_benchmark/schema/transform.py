from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, List, Optional, Union, cast

from loguru import logger
from pydantic import BaseModel, Extra

from retrieval_qa_benchmark.schema.datatypes import QARecord
from retrieval_qa_benchmark.utils.registry import REGISTRY
from retrieval_qa_benchmark.utils.profiler import PROFILER

def get_func(obj: BaseTransform, name:str) -> Callable:
    method_name = f"transform_{name}"
    try:
        return getattr(obj, method_name)
    except AttributeError:
        def default_func(data):
            if name in data:
                return str(data[name])
            else:
                return ''
        return default_func


@REGISTRY.register_transform("base")
class BaseTransform(BaseModel):
    """Base transformation to elements in datasets"""

    class Config:
        extra = Extra.allow

    def targets(self) -> Dict[str, Callable[[Dict[str, Any]], str]]:
        return {
            k: get_func(self, k) for k in QARecord.model_fields.keys()
        }

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

    @PROFILER.profile_function("BaseTransform.chain")
    def chain(self, data: Union[QARecord, Dict[str, Any]]) -> Dict[str, Any]:
        result = {}
        if type(data) is QARecord:
            data = data.model_dump()
        for k, f in self.targets().items():
            try:
                result[k] = f(cast(Dict[str, Any], data))
            except Exception as e:
                logger.error(f"Transform function failed on key `{k}`")
                raise e
        return result


class TransformChain(BaseModel):
    """Chain of transformations"""

    chain: Sequence[BaseTransform]

    @PROFILER.profile_function("TransformChain.__call__")
    def __call__(self, data: Dict[str, Any]) -> Any:
        for c in self.chain[:-1]:
            data = c.chain(data)
        return self.chain[-1](data)
