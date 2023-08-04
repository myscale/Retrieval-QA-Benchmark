from typing import Union, Sequence, Tuple, Any, List
from datasets import load_dataset
from tqdm import tqdm
from loguru import logger
from retrieval_qa_benchmark.transforms.base import BaseTransform, TransformChain
from retrieval_qa_benchmark.schema import QARecord


def build_hfdataset_internal(
    name: Union[str, Sequence[str]],
    eval_split: str = "validation",
    transform: Union[BaseTransform, TransformChain] = BaseTransform(),
    **kwargs: Any,
) -> Tuple[str, List[QARecord]]:
    if type(name) is str:
        name = [name]
    data = load_dataset(*name, **kwargs)[eval_split]
    try:
        eval_set: List[QARecord] = [
            transform(d) for d in tqdm(data, desc="Converting dataset...")
        ]
        return f"{'.'.join(name)}-{eval_split}", eval_set
    except Exception as e:
        logger.error(f"Failed to parse data whose first row is like: \n{data[0]}")
        raise e
