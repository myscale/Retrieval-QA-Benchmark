from typing import Any, Sequence, Tuple

from retrieval_qa_benchmark.transforms.base import BaseContextTransform
from retrieval_qa_benchmark.transforms.search import (
    FaissElSearchBM25HybridSearcher,
)
from retrieval_qa_benchmark.utils.registry import REGISTRY


@REGISTRY.register_transform("FaissESHybrid")
class ContextWithFaissESHybrid(BaseContextTransform):
    """_summary_

    :inherited-members:
    :param BaseContextTransform: _description_
    :type BaseContextTransform: _type_
    """

    nprobe: int = 128
    el_host: str
    el_auth: Tuple[str, str]
    index_path: str = "data/indexes/Cohere_mpnet/IVFSQ_L2.index"
    embedding_name: str = "paraphrase-multilingual-mpnet-base-v2"
    dataset_name: Sequence[str] = ["Cohere/wikipedia-22-12-en-embeddings"]
    num_filtered: int = 100
    is_raw_rank: bool = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._searcher = FaissElSearchBM25HybridSearcher(
            embedding_name=self.embedding_name,
            index_path=self.index_path,
            template=self.context_template,
            el_host=self.el_host,
            el_auth=self.el_auth,
            num_filtered=self.num_filtered,
            is_raw_rank=self.is_raw_rank,
            nprobe=self.nprobe,
            dataset_name=self.dataset_name,
            dataset_split="train",
        )
