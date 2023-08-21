from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Sequence,
    Union,
    cast,
    get_args,
    get_origin,
)

from loguru import logger
from pydantic import BaseModel, Extra
from pydantic.fields import FieldInfo

from retrieval_qa_benchmark.schema.datatypes import QAPrediction, QARecord
from retrieval_qa_benchmark.utils.profiler import PROFILER
from retrieval_qa_benchmark.utils.registry import REGISTRY


def get_field_func(obj: BaseTransform, name: str, field: FieldInfo) -> Callable:
    """Get default function for a field

    It will try to looking up the input data dict.
    If None or not in dictionary, It will return the field's default if possible.
    Or just return None or empty string

    :param obj: `Transform` object to be wrapped on
    :type obj: BaseTransform
    :param name: target name of the field
    :type name: str
    :param field: field object
    :type field: FieldInfo
    :return: a function that gives the default value
    :rtype: Callable
    """
    method_name = f"transform_{name}"
    try:
        return getattr(obj, method_name)
    except AttributeError:

        def default_func(data: Dict[str, Any], **params: Any) -> Optional[str]:
            if name in data:
                return data[name]
            else:
                if field.default is not None:
                    return field.default
                if get_origin(field.annotation) is Union and type(None) in get_args(
                    field.annotation
                ):
                    return None
                return ""

        return default_func


@REGISTRY.register_transform("base")
class BaseTransform(BaseModel):
    """Base transform object.

    This framework is driven by :class:`BaseTransform`.
    A :class:`BaseTransform` will always takes :class:`QARecord` as input,
    and outputs a new :class:`QARecord`.

    ** Principle of design:
    
    1. Make every transform as a minimal and atomic operation to :class:`QARecord`
    2. Only alter the fields it needs to change in a single :class:`BaseTransform`
    """

    next: Sequence[Optional["BaseTransform"]] = [None, None]
    """list of next status"""

    class Config:
        extra = Extra.allow

    def field_targets(self) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
        """get collection of all transform function of this transform

        :return: Dictionary of transform function to fields
        :rtype: Dict[str, Callable[[Dict[str, Any]], Any]]
        """
        return {
            k: get_field_func(self, k, field)
            for k, field in QARecord.model_fields.items()
        }

    def transform_choices(
        self, data: Dict[str, Any], **params: Any
    ) -> Optional[List[str]]:
        """Special transform function to choices in :class:`QARecord`

        :param data: input :class:`QARecord` as dictionary
        :type data: Dict[str, Any]
        :return: transformed choices
        :rtype: Optional[List[str]]
        """
        try:
            return data["choices"]
        except KeyError:
            return None

    def check_status(self, current: Dict[str, Any]) -> int:
        """Check the status after all transform functions

        :param current: Current transformed :class:`QARecord` as dictionary
        :type current: Dict[str, Any]
        :return: the next state ID in :attribute:`BaseTransform.next`
        :rtype: int
        """
        return 1

    def __call__(
        self, data: Dict[str, Any]
    ) -> Tuple[Optional["BaseTransform"], QARecord]:
        """you can call :class:`BaseTransform` as functions

        :return: a transformed :class:`QARecord`
        :rtype: Tuple[Optional["BaseTransform"], QARecord]
        """
        next_, data = self.chain(data)
        return next_, QARecord(**{k: v for k, v in data.items() if v is not None})

    def chain(
        self, data: Union[QARecord, Dict[str, Any]]
    ) -> Tuple[Optional["BaseTransform"], Dict[str, Any]]:
        """chainable function for :class:`TransformChain`

        :raises e: transform will raise exception once one of transform function fails.
        :return: next state and the returned :class:`QARecord`
        :rtype: Tuple[Optional["BaseTransform"], Dict[str, Any]]
        """
        result = {}
        if type(data) is QARecord:
            data = data.model_dump()
        for k, f in self.field_targets().items():
            try:
                result[k] = f(cast(Dict[str, Any], data))
            except Exception as e:
                logger.error(f"Transform function failed on key `{k}`")
                raise e
        return self.next[self.check_status(result)], result


class TransformGraph(BaseModel):
    """Callable graph for :class:`BaseTransform`

.. graphviz::
   digraph foo {
      "bar" -> "baz";
   }
    """

    entry_id: str
    chain: Dict[str, BaseTransform]

    @classmethod
    def build(
        cls,
        chain: Dict[str, BaseTransform],
        entry_id: str = "0",
    ) -> "TransformGraph":
        return cls(chain=chain, entry_id=entry_id)

    @PROFILER.profile_function("transform.TransformChain.profile")
    def __call__(self, data: Dict[str, Any]) -> QARecord:
        if len(self.chain) > 0:
            ret: Optional["BaseTransform"] = self.chain[self.entry_id]
            while ret is not None:
                ret, data = ret(data)  # type: ignore
                if type(data) is QAPrediction:
                    data["stack"].append(data)
        if isinstance(data, QARecord):
            return data
        else:
            try:
                return QAPrediction(**data)
            except Exception:
                return QARecord(**data)
