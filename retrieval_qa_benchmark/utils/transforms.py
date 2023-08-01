from typing import Any, Callable, Dict, Sequence, Iterable

from pydantic import BaseModel, validator, Extra

from loguru import logger
from retrieval_qa_benchmark.schema import QARecord


def build_value_functions(
    keys: Iterable[str],
) -> Dict[str, Callable[[Dict[str, Any]], str]]:
    funcs = {}
    for k in keys:
        func: Callable[[Dict[str, Any]], Any] = lambda x: x[k]
        funcs[k] = func
    return funcs


class BaseTransform(BaseModel):
    """Base transformation to elements in datasets"""

    value_functions: Dict[str, Callable[[Dict[str, Any]], str]] = build_value_functions(
        QARecord.model_fields.keys()
    )

    class Config:
        extra = Extra.forbid

    @validator("value_functions", always=True)
    def check_value_functions(cls, value_functions: Dict[str, Any]) -> Any:
        if len(set(QARecord.model_fields.keys()) - set(value_functions)) != 0:
            raise KeyError(
                "Transform got unaligned keys between BaseRow and `value_functions`: "
                f"{set(QARecord.model_fields.keys())} vs. {set(value_functions.keys())}"
            )
        return value_functions

    def set_value_function(
        self, key: str, value: Callable[[Dict[str, Any]], Any]
    ) -> None:
        self.value_functions[key] = value

    def __call__(self, data: Dict[str, Any]) -> QARecord:
        return QARecord(**{k: str(v) for k, v in self.chain(data).items()})

    def chain(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for k, f in self.value_functions.items():
            try:
                result[k] = f(data)
            except Exception as e:
                raise KeyError(f"Value function failed on key `{k}`")
        return result


# class MultipleChoiceTransform(BaseTransform):
#     def chain(self, data: Dict[str, Any]) -> Dict[str, Any]:
#         pass


class TransformChain(BaseModel):
    """Chain of transformations"""

    chain: Sequence[BaseTransform]

    def __call__(self, data: Dict[str, Any]) -> Any:
        for c in self.chain[:-1]:
            data = c.chain(data)
        return c(data)
