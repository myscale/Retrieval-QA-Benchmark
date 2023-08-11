from typing import Any, List, Sequence, Tuple, Union

from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from retrieval_qa_benchmark.schema import QARecord
from retrieval_qa_benchmark.schema.transform import BaseTransform, TransformChain


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
            transform(d)[1] for d in tqdm(data, desc="Converting dataset...")
        ]
        return f"{'.'.join(name)}-{eval_split}", eval_set
    except Exception as e:
        logger.error(f"Failed to parse data whose first row is like: \n{data[0]}")
        raise e
