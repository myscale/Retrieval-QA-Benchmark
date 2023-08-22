from typing import Any, Sequence, Tuple

from retrieval_qa_benchmark.transforms.base import BaseContextTransform
from retrieval_qa_benchmark.transforms.search import (
    ElSearchSearcher,
    FaissSearcher,
    MyScaleSearcher,
)
from retrieval_qa_benchmark.utils.registry import REGISTRY


@REGISTRY.register_transform("Faiss")
class ContextWithFaiss(BaseContextTransform):
    """_summary_

    :inherited-members:
    :param BaseContextTransform: _description_
    :type BaseContextTransform: _type_
    """

    nprobe: int = 128
    index_path: str = (
        "data/indexes/Cohere_mpnet/IVFSQ_L2.index"
    )
    embedding_name: str = "paraphrase-multilingual-mpnet-base-v2"
    dataset_name: Sequence[str] = ["Cohere/wikipedia-22-12-en-embeddings"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._searcher = FaissSearcher(
            embedding_name=self.embedding_name,
            index_path=self.index_path,
            template=self.context_template,
            nprobe=self.nprobe,
            dataset_name=self.dataset_name,
            dataset_split="train",
        )


@REGISTRY.register_transform("ElasticBM25")
class ContextWithElasticBM25(BaseContextTransform):
    el_host: str
    el_auth: Tuple[str, str]
    dataset_name: Sequence[str] = ["Cohere/wikipedia-22-12-en-embeddings"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._searcher = ElSearchSearcher(
            template=self.context_template,
            el_host=self.el_host,
            el_auth=self.el_auth,
            dataset_name=self.dataset_name,
            dataset_split="train",
        )


@REGISTRY.register_transform("MyScale")
class ContextWithMyScale(BaseContextTransform):
    msc_host: str
    msc_port: int
    msc_user: str
    msc_pass: str
    two_staged: bool = False
    kw_topk: int = 10
    num_filtered: int = 100
    embedding_name: str = "paraphrase-multilingual-mpnet-base-v2"
    table_name: str = "default.Wikipedia"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._searcher = MyScaleSearcher(
            template=self.context_template,
            embedding_name=self.embedding_name,
            host=self.msc_host,
            port=self.msc_port,
            username=self.msc_user,
            password=self.msc_pass,
            table_name=self.table_name,
            two_staged=self.two_staged,
            num_filtered=self.num_filtered,
            kw_topk=self.kw_topk,
        )
