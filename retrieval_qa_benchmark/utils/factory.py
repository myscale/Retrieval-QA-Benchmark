from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, List

from pydantic import BaseModel, Extra

from retrieval_qa_benchmark.schema import BaseDataset
from retrieval_qa_benchmark.schema import BaseLLM
from retrieval_qa_benchmark.schema import BaseTransform, TransformChain

from retrieval_qa_benchmark.utils.registry import REGISTRY


class BaseFactory(BaseModel):
    name: str
    args: Optional[Dict[str, Any]] = {}
    run_args: Optional[Dict[str, Any]] = {}

    class Config:
        extra = Extra.ignore

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseFactory:
        name = config["name"]
        args = config["args"] if "args" in config else {}
        run_args = config["run_args"] if "run_args" in config else {}
        return cls(name=name, args=args, run_args=run_args)

    def build(self) -> Any:
        raise NotImplementedError


class DatasetFactory(BaseFactory):
    def build(self) -> BaseDataset:
        return REGISTRY.Datasets[self.name].build(**self.args)


class TransformFactory(BaseFactory):
    def build(self) -> BaseTransform:
        return REGISTRY.Transforms[self.name](**self.args)


class TransformChainFactory(BaseModel):
    chain_config: Sequence[TransformFactory] = []

    def build(self) -> TransformChain:
        transforms = [c.build() for c in self.chain_config]
        return TransformChain(chain=transforms)


class ModelFactory(BaseFactory):
    def build(self) -> BaseLLM:
        return REGISTRY.Models[self.name].build(**self.args, run_args=self.run_args)
