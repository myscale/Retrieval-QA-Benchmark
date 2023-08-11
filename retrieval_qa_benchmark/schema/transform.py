from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, \
    Tuple, Union, cast, Generic
try:
    from typing import get_args, get_origin
# Compatibility
except ImportError:
    get_args = lambda t: getattr(t, '__args__', ()) \
                         if t is not Generic else Generic
    get_origin = lambda t: getattr(t, '__origin__', None)

from loguru import logger
from pydantic import BaseModel, Extra
from pydantic.fields import FieldInfo

from retrieval_qa_benchmark.schema.datatypes import QARecord, QAPrediction
from retrieval_qa_benchmark.utils.profiler import PROFILER
from retrieval_qa_benchmark.utils.registry import REGISTRY


def get_field_func(obj: BaseTransform, name: str, field: FieldInfo) -> Callable:
    method_name = f"transform_{name}"
    try:
        return getattr(obj, method_name)
    except AttributeError:

        def default_func(data: Dict[str, Any], **params) -> str:
            if name in data:
                return data[name]
            else:
                if field.default is not None:
                    return field.default
                if get_origin(field.annotation) is Union and \
                    type(None) in get_args(field.annotation):
                    return None
                return ""

        return default_func


@REGISTRY.register_transform("base")
class BaseTransform(BaseModel):
    """Base transformation to elements in datasets"""
    children: Tuple["BaseTransform"] = (None, None)

    class Config:
        extra = Extra.allow

    def field_targets(self) -> Dict[str, Callable[[Dict[str, Any]], str]]:
        return {k: get_field_func(self, k, field) for k, field in QARecord.model_fields.items()}

    def transform_choices(
        self, data: Dict[str, Any], **params: Any
    ) -> Optional[List[str]]:
        try:
            return data["choices"]
        except KeyError:
            return None
        
    def check_status(self, current: Dict[str, Any]) -> bool:
        return True

    def __call__(self, data: Dict[str, Any]) -> QARecord:
        status, data = self.chain(data)
        return status, QARecord(**{k: v for k, v in data.items() if v is not None})

    @PROFILER.profile_function("BaseTransform.chain")
    def chain(self, data: Union[QARecord, Dict[str, Any]]) -> Tuple["BaseTransform", Dict[str, Any]]:
        result = {}
        if type(data) is QARecord:
            data = data.model_dump()
        for k, f in self.field_targets().items():
            try:
                result[k] = f(cast(Dict[str, Any], data))
            except Exception as e:
                logger.error(f"Transform function failed on key `{k}`")
                raise e
        return self.children[self.check_status(result)], result

class TransformChain(BaseModel):
    """Chain of transformations"""
    entry_id: str
    chain: Dict[str, BaseTransform]
    
    @classmethod
    def build(cls, chain: Union[Dict[str, BaseTransform], List[BaseTransform]], entry_id:str="0") -> "TransformChain":
        _chain = {}
        if type(chain) is list:
            for i, n in enumerate(chain):
                if i > 0:
                    chain[str(i-1)].children = (n, n)
                _chain[str(i)] = n
        else:
            _chain = chain
        return cls(chain=_chain, entry_id=entry_id)

    @PROFILER.profile_function("TransformChain.__call__")
    def __call__(self, data: Dict[str, Any]) -> QARecord:
        if len(self.chain) > 0:
            ret = self.chain[self.entry_id]
            while ret is not None:
                ret, data = ret(data)
                if type(data) is QAPrediction:
                    data["stack"].append(data)
        return data
