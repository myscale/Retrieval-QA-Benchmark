from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Extra
from tqdm import tqdm

from retrieval_qa_benchmark.schema import (
    BaseDataset,
    BaseLLM,
    QAPrediction,
    QARecord,
    TransformGraph,
)
from retrieval_qa_benchmark.utils.factory import (
    DatasetFactory,
    ModelFactory,
    TransformGraphFactory,
)
from retrieval_qa_benchmark.utils.profiler import PROFILER


class BaseEvaluator(BaseModel):
    """Base class for evaluators"""

    dataset: BaseDataset
    llm: BaseLLM
    transform: TransformGraph
    matcher: Callable[[str, QARecord], bool] = lambda x, y: x == y.answer  # noqa: E731
    out_file: Optional[str] = None

    class Config:
        extra = Extra.forbid

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseEvaluator:
        config = config["evaluator"]
        if type(config["dataset"]) is list:
            dataset = DatasetFactory.from_config(config["dataset"][0]).build()
            for c in config["dataset"][1:]:
                dataset += DatasetFactory.from_config(c).build()
        else:
            dataset = DatasetFactory.from_config(config["dataset"]).build()
        if "transform" in config:
            transform = TransformGraphFactory(nodes_config=config["transform"]).build()
        else:
            transform = TransformGraphFactory(nodes_config={}).build()
        model = ModelFactory.from_config(config["model"]).build()
        out_file = config["out_file"] if "out_file" in config else None
        return cls(dataset=dataset, transform=transform, llm=model, out_file=out_file)

    def __call__(self) -> Tuple[float, List[QAPrediction]]:
        result: List[QAPrediction] = []
        cnt = 0
        for d in tqdm(self.dataset.eval_set, desc="Evaluating"):
            try:
                d_ = self.transform(d)
                pred = self.llm.generate(d_)
                mtch = self.matcher(pred.generated, d_)
                if mtch:
                    cnt += 1
                prompt_tokens = pred.prompt_tokens
                completion_tokens = pred.completion_tokens
                if d_.stack and len(d_.stack) > 0:
                    prompt_tokens += sum([p.prompt_tokens for p in d_.stack if p])
                    completion_tokens += sum(
                        [p.completion_tokens for p in d_.stack if p]
                    )
                profile_avg = {
                    k: PROFILER.accumulator[k] / PROFILER.counter[k]
                    for k in PROFILER.accumulator.keys()
                }
                result.append(
                    QAPrediction(
                        **d_.model_dump(),
                        pred=pred.generated,
                        matched=mtch,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        profile_avg=profile_avg,
                        profile_count=PROFILER.counter,
                        profile_time=PROFILER.accumulator,
                    )
                )
                PROFILER.clear()
            except Exception as e:
                logger.error(f"Failed to evaluate record {str(d)}")
                raise e
        acc = 100 * cnt / len(self.dataset)
        logger.info(
            f"Evaluation finished! Executed Evaluator:{type(self)} on "
            f"Dataset:{self.dataset.name} with Model:{self.llm.name}. "
            f"Accuracy: {acc:.2f}%"
        )
        if self.out_file:
            with open(self.out_file, "w") as f:
                f.write("\n".join([r.model_dump_json() for r in result]))
        return acc, result
