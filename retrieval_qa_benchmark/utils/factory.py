from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

from pydantic import BaseModel, Extra

from retrieval_qa_benchmark.schema import (
    BaseDataset,
    BaseLLM,
    BaseTransform,
    TransformChain,
)
from retrieval_qa_benchmark.utils.registry import REGISTRY


class BaseFactory(BaseModel):
    """
    """
    type: str
    args: Optional[Dict[str, Any]] = {}
    run_args: Optional[Dict[str, Any]] = {}

    class Config:
        extra = Extra.ignore

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseFactory:
        type = config["type"]
        args = config["args"] if "args" in config else {}
        run_args = config["run_args"] if "run_args" in config else {}
        return cls(type=type, args=args, run_args=run_args)

    def build(self) -> Any:
        raise NotImplementedError


class DatasetFactory(BaseFactory):
    """
    """
    def build(self) -> BaseDataset:
        return REGISTRY.Datasets[self.type].build(**self.args)


class TransformFactory(BaseFactory):
    """
    """
    @classmethod
    def from_config(cls, id: str, config: Dict[str, Any]) -> BaseFactory:
        type = config["type"]
        args = config["args"] if "args" in config else {}
        run_args = config["run_args"] if "run_args" in config else {}
        return cls(id=id, type=type, args=args, run_args=run_args)

    def build(self) -> BaseTransform:
        return REGISTRY.Transforms[self.type](**self.args)


class TransformChainFactory(BaseModel):
    """
    """
    chain_config: Union[Dict[str, Any], Sequence[Any]] = []

    def build(self) -> TransformChain:
        if "chain" in self.chain_config and len(self.chain_config["chain"]) > 0:
            chain_config = self.chain_config["chain"]
            if type(chain_config) in [list, tuple]:
                entry_id = "0"
                transforms = {
                    str(i): TransformFactory.from_config(str(i), c).build()
                    for i, c in enumerate(chain_config)
                }
                for i in range(len(chain_config)):
                    if i > 0:
                        transforms[str(i - 1)].children = (
                            transforms[str(i)],
                            transforms[str(i)],
                        )
                print(transforms)
            else:
                entry_id = self.chain_config["entry_id"]
                transforms = {
                    k: TransformFactory.from_config(k, c).build()
                    for k, c in chain_config.items()
                }
                for k, c in chain_config.items():
                    transforms[k].children = (
                        transforms[c["children"][0]]
                        if c["children"][0] is not None
                        else None,
                        transforms[c["children"][1]]
                        if c["children"][1] is not None
                        else None,
                    )
            assert (
                entry_id != ""
            ), "Entry ID must not be empty for dictionary of transforms"
            assert (
                entry_id in transforms
            ), "Entry ID must be in keys of transform dictionary"
        else:
            entry_id = ""
            transforms = {}
        return TransformChain(entry_id=entry_id, chain=transforms)


class ModelFactory(BaseFactory):
    """
    """
    def build(self) -> BaseLLM:
        return REGISTRY.LLMs[self.type].build(**self.args, run_args=self.run_args)
