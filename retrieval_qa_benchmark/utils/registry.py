from typing import Any, Callable, Dict

from pydantic import BaseModel, Extra


class Registry(BaseModel):
    Datasets: Dict[str, Any] = {}
    Transforms: Dict[str, Any] = {}
    Models: Dict[str, Any] = {}
    Evaluators: Dict[str, Any] = {}

    class Config:
        extra = Extra.forbid

    def register_dataset(self, name: str) -> Callable:
        def decorator(f: Callable) -> Callable:
            self.Datasets[name] = f
            return f

        return decorator

    def register_model(self, name: str) -> Callable:
        def decorator(f: Callable) -> Callable:
            self.Models[name] = f
            return f

        return decorator

    def register_evaluator(self, name: str) -> Callable:
        def decorator(f: Callable) -> Callable:
            self.Evaluators[name] = f
            return f

        return decorator

    def register_transform(self, name: str) -> Callable:
        def decorator(f: Callable) -> Callable:
            self.Transforms[name] = f
            return f

        return decorator

    def __str__(self) -> str:
        return "\n".join(
            [
                f"{n}:\n" + "\n".join([f"\t{_n}" for _n in getattr(self, n)])
                for n in self.model_fields.keys()
            ]
        )


REGISTRY = Registry()
