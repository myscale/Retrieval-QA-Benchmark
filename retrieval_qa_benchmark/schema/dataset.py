from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, Extra

from retrieval_qa_benchmark.schema.datatypes import QARecord


class BaseDataset(BaseModel):
    """Dataset's Baseclass
    
    Dataset should always output :class:`QARecord`
    with ``__getitem__`` method
    """

    name: str = "dataset"
    """Name of this dataset"""
    eval_set: List[QARecord] = []
    """Data to be evaluated. The data is 
        transformed with its built-in transform."""

    class Config:
        extra = Extra.forbid
        """No extra field allowed"""

    @classmethod
    def build(cls, *args: Any, **kwargs: Any) -> BaseDataset:
        """build dataset

        :raises NotImplementedError: user should implement this
        :return: dataset that iterate over ``List[QARecord]``
        :rtype: BaseDataset
        """
        raise NotImplementedError("Please implement a `.build` function")

    def __getitem__(self, index: int) -> QARecord:
        """_summary_

        :param index: _description_
        :type index: int
        :return: _description_
        :rtype: QARecord
        :meta public:
        """
        return self.eval_set[index]

    def __len__(self) -> int:
        """_summary_

        :return: _description_
        :rtype: int
        :meta public:
        """
        return len(self.eval_set)
