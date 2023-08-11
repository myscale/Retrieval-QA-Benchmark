import time
from typing import Any, Callable, Dict

from pydantic import BaseModel, Extra


class Profiler(BaseModel):
    """"""
    counter: Dict[str, int] = {}
    accumulator: Dict[str, float] = {}

    class Config:
        extra = Extra.forbid

    def profile_function(self, name: str) -> Callable:
        def decorator(f: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                t0 = time.time()
                ret = f(*args, **kwargs)
                t1 = time.time()
                if name not in self.counter:
                    self.counter[name] = 0
                if name not in self.accumulator:
                    self.accumulator[name] = 0
                self.counter[name] += 1
                self.accumulator[name] += (t1 - t0) * 1000
                return ret

            return wrapper

        return decorator

    def profile_dataset(self, dataset_name: str) -> Callable:
        def decorator(dataset: Any) -> Any:
            dataset.__getitem__ = self.profile_function(
                f"dataset.{dataset_name}.profile"
            )(dataset.__getitem__)
            return dataset

        return decorator

    def profile_transform(self, transform_name: str) -> Callable:
        def decorator(transform: Any) -> Any:
            transform.chain = self.profile_function(
                f"transform.{transform_name}.profile"
            )(transform.chain)
            return transform

        return decorator

    def profile_model(self, model_name: str) -> Callable:
        def decorator(model: Any) -> Any:
            model.generate = self.profile_function(f"model.{model_name}.profile")(
                model.generate
            )
            return model

        return decorator

    def clear(self):
        self.counter = {}
        self.accumulator = {}

    def __str__(self) -> str:
        ret = "\t======== Profile Result ========\n"
        for k in self.counter:
            ret += (
                f"{k:50s}\tCount:{self.counter[k]: 8d}\t"
                f"AvgTime:{self.accumulator[k]/self.counter[k]:8.2f} ms\n"
            )
        return ret


PROFILER = Profiler()
