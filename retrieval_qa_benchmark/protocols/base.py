from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from loguru import logger
from pydantic import BaseModel, Extra
from tqdm import tqdm

from retrieval_qa_benchmark.schema import (
    BaseDataset,
    BaseLLM,
    BaseTransform,
    QAPrediction,
    QARecord,
    TransformChain,
)
from retrieval_qa_benchmark.utils.factory import (
    DatasetFactory,
    ModelFactory,
    TransformChainFactory,
    TransformFactory,
)


class BaseEvaluator(BaseModel):
    """Base class for evaluators"""

    dataset: BaseDataset
    llm: BaseLLM
    transform: Union[BaseTransform, TransformChain]
    matcher: Callable[[str, QARecord], bool] = lambda x, y: x == y.answer  # noqa: E731
    out_file: Optional[str] = None

    class Config:
        extra = Extra.forbid

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseEvaluator:
        config = config["evaluator"]
        dataset = DatasetFactory.from_config(config["dataset"]).build()
        if "transform_chain" in config:
            transform = TransformChainFactory(
                chain_config=config["transform_chain"]
            ).build()
        else:
            transform = TransformChainFactory(chain_config=[]).build()
        model = ModelFactory.from_config(config["model"]).build()
        out_file = config["out_file"] if "out_file" in config else None
        return cls(dataset=dataset, transform=transform, llm=model, out_file=out_file)

    def __call__(self) -> Tuple[float, List[QAPrediction]]:
        result: List[QAPrediction] = []
        cnt = 0
        for d in tqdm(self.dataset.eval_set, desc="Evaluating"):
            try:
                d_ = self.transform(d)
                pred = self.llm.generate(d_.question)
                mtch = self.matcher(pred.generated, d_)
                if mtch:
                    cnt += 1
                result.append(
                    QAPrediction(
                        **d_.model_dump(),
                        pred=pred.generated,
                        matched=mtch,
                        prompt_tokens=pred.prompt_tokens,
                        completion_tokens=pred.completion_tokens,
                    )
                )
            except Exception as e:
                logger.error(f"Failed to evaluate record {str(d)}")
                raise e
        acc = 100 * cnt / len(self.dataset)
        logger.info(
            f"Evaluation finished! Executed Evaluator:{type(self)} on "
            f"Dataset:{self.dataset.name} with Model:{self.llm.model}. "
            f"Accuracy: {acc:.2f}%"
        )
        if self.out_file:
            with open(self.out_file, "w") as f:
                f.write("\n".join([r.model_dump_json() for r in result]))
        return acc, result
