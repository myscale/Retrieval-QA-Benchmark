from __future__ import annotations

from typing import Union, Callable, Optional, Tuple, List
from pydantic import BaseModel, Extra

from loguru import logger
from tqdm import tqdm
from retrieval_qa_benchmark.transforms.base import BaseTransform, TransformChain
from retrieval_qa_benchmark.schema import BaseDataset
from retrieval_qa_benchmark.schema import BaseLLM
from retrieval_qa_benchmark.schema import QAPrediction


class BaseEvaluator(BaseModel):
    """Base class for evaluators"""
    dataset: BaseDataset
    llm: BaseLLM
    transform: Union[BaseTransform, TransformChain]
    matcher: Callable[[str, str], bool]
    out_file: Optional[str] = None
    
    class Config:
        extra = Extra.forbid

    def __call__(self) -> Tuple[float, List[QAPrediction]]:
        result: List[QAPrediction] = []
        cnt = 0
        for d in tqdm(self.dataset, desc="Evaluating"):
            d_ = self.transform(d)
            pred = self.llm.generate(d_.question)
            mtch = self.matcher(pred, d_)
            if mtch:
                cnt += 1
            result.append(QAPrediction(**d.model_dump(), pred=pred, matched=mtch))
        acc = 100 * cnt/len(self.dataset)
        logger.info(f"Evaluation finished! Executed Evaluator:{type(self)} on "
                    f"Dataset:{self.dataset.name} with Model:{self.llm.model_name}. "
                    f"Accuracy: {acc:.2f}%")
        if self.out_file:
            with open(self.out_file, 'w') as f:
                f.write('\n'.join([r.model_dump_json() for r in result]))
        return acc, result