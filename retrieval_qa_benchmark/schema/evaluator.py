from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Extra
from tqdm import tqdm

from retrieval_qa_benchmark.schema.dataset import BaseDataset
from retrieval_qa_benchmark.schema.datatypes import (
    QAPrediction,
    QARecord,
)
from retrieval_qa_benchmark.schema.model import BaseLLM
from retrieval_qa_benchmark.schema.transform import TransformGraph
from retrieval_qa_benchmark.utils.profiler import PROFILER


def default_matcher(x: str, y: QARecord) -> float:
    return float(x == y.answer)


class BaseEvaluator(BaseModel):
    """Base class for evaluators"""

    dataset: BaseDataset
    llm: BaseLLM
    transform: TransformGraph
    matcher: Callable[[str, QARecord], float] = default_matcher
    out_file: Optional[str] = None

    class Config:
        extra = Extra.forbid

    def __call__(self) -> Tuple[float, List[QAPrediction]]:
        PROFILER.clear()
        result: List[QAPrediction] = []
        for d in tqdm(self.dataset.eval_set, desc="Evaluating"):
            try:
                d_ = self.transform(d)
                pred = self.llm(d_)
                mtch = self.matcher(pred.generated, d_)
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
        score = 100 * sum([p.matched for p in result]) / len(self.dataset)
        logger.info(
            f"Evaluation finished! Executed Evaluator:{type(self)} on "
            f"Dataset:{self.dataset.name} with Model:{self.llm.name}. "
            f"Score: {score:.2f}%"
        )
        if self.out_file:
            with open(self.out_file, "w") as f:
                f.write("\n".join([r.model_dump_json() for r in result]))
        return score, result
