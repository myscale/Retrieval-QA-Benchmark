from __future__ import annotations

from typing import Any, List, Sequence, Tuple, Union

from datasets import load_dataset
from loguru import logger
from pydantic import BaseModel, Extra
from tqdm import tqdm

from retrieval_qa_benchmark.schema import QARecord
from retrieval_qa_benchmark.transforms import BaseTransform, TransformChain
from retrieval_qa_benchmark.utils.registry import REGISTRY


def build_hfdataset_internal(
    name: Union[str, Sequence[str]],
    eval_split: str = "validation",
    transform: Union[BaseTransform, TransformChain] = BaseTransform(),
    **kwargs: Any,
) -> Tuple[str, List[QARecord]]:
    if type(name) is str:
        name = [name]
    data = load_dataset(*name, **kwargs)[eval_split]
    try:
        eval_set: List[QARecord] = [
            transform(d) for d in tqdm(data, desc="Converting dataset...")
        ]
        return f"{'.'.join(name)}-{eval_split}", eval_set
    except Exception as e:
        logger.error(f"Failed to parse data whose first row is like: \n{data[0]}")
        raise e


class BaseDataset(BaseModel):
    """Base class dataset"""

    name: str = "dataset"
    eval_set: List[QARecord] = []

    class Config:
        extra = Extra.forbid

    @classmethod
    def build(cls, *args: Any, **kwargs: Any) -> BaseDataset:
        raise NotImplementedError("Please implement a `.build` function")

    def __getitem__(self, index: int) -> QARecord:
        return self.eval_set[index]

    def __len__(self) -> int:
        return len(self.eval_set)


class HFDataset(BaseDataset):
    """Base class for huggingface datasets"""

    @classmethod
    def build_(
        cls,
        name: Union[str, Sequence[str]],
        eval_split: str = "validation",
        extra_transforms: Union[BaseTransform, TransformChain] = BaseTransform(),
        **kwargs: Any,
    ) -> HFDataset:
        name, eval_set = build_hfdataset_internal(
            name, eval_split, extra_transforms, **kwargs
        )
        return cls(name=name, eval_set=eval_set)
