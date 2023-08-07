from typing import Any, Callable, Dict

import time
from functools import wraps
from pydantic import BaseModel, Extra


class Profiler(BaseModel):

    counter: Dict[str, int] = {}
    accumulator: Dict[str, float] = {}

    class Config:
        extra = Extra.forbid

    def profile_function(self, name: str) -> Callable:
        def decorator(f: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                t0 = time.time()
                ret = f(*args, **kwargs)
                t1 = time.time()
                if name not in self.counter:
                    self.counter[name] = 0
                if name not in self.accumulator:
                    self.accumulator[name] = 0
                self.counter[name] += 1
                self.accumulator[name] += t1 - t0
                return ret
            return wrapper

        return decorator

    def __str__(self) -> str:
        ret = '\t======== Profile Result ========\n'
        for k in self.counter:
            ret += f'{k:30s}\tCount:{self.counter[k]: 8d}\tAvgTime:{self.accumulator[k] * 1000/self.counter[k]:8.2f} ms\n'
        return ret

PROFILER = Profiler()
