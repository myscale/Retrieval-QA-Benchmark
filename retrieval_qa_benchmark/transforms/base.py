from typing import Any, Dict, List, Sequence, Tuple

from retrieval_qa_benchmark.schema.transform import BaseTransform
from retrieval_qa_benchmark.transforms.search import (
    ElSearchBM25Searcher,
    FaissSearcher,
    MyScaleSearcher,
)
from retrieval_qa_benchmark.transforms.search.base import BaseSearcher
from retrieval_qa_benchmark.utils.registry import REGISTRY


class BaseContextTransform(BaseTransform):
    sep_chr: str = "\n"
    num_selected: int = 5
    context_template: str = "{title} | {paragraph}"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._searcher = BaseSearcher()

    def preproc_question4query(self, data: Dict[str, Any]) -> str:
        question = data["question"]
        choices = " | ".join(data["choices"])
        return "\n".join([question, choices])

    def transform_context(self, data: Dict[str, Any], **params: Any) -> List[str]:
        context = self._searcher(
            [self.preproc_question4query(data)],
            self.num_selected,
            context=[data["context"]],
        )
        return context


@REGISTRY.register_transform("Faiss")
class ContextWithFaiss(BaseContextTransform):
    nprobe: int = 128
    index_path: str = (
        "/mnt/workspaces/yongqij/evals/data/indexes/Cohere_mpnet/IVFSQ_L2.index"
    )
    embedding_name: str = "paraphrase-multilingual-mpnet-base-v2"
    dataset_name: Sequence[str] = ["Cohere/wikipedia-22-12-en-embeddings"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._searcher = FaissSearcher(
            model_name=self.embedding_name,
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
        self._searcher = ElSearchBM25Searcher(
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
            model_name=self.embedding_name,
            host=self.msc_host,
            port=self.msc_port,
            username=self.msc_user,
            password=self.msc_pass,
            table_name=self.table_name,
            two_staged=self.two_staged,
            num_filtered=self.num_filtered,
            kw_topk=self.kw_topk,
        )
