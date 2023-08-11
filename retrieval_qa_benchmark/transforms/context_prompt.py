from typing import Any, Dict, Iterable, List, Union

from retrieval_qa_benchmark.schema.transform import BaseTransform
from retrieval_qa_benchmark.utils.registry import REGISTRY

from .myscale_retrieval.myscale_search import MyScaleSearch


@REGISTRY.register_transform("add_myscale_retrieval")
class AddMyScaleRetrievalTransform(BaseTransform):
    sep_chr: str = "\n"
    prompt_prefix: str = "Context:"
    num_filtered: int = 100
    num_selected: int = 5
    with_title: bool = True
    rank_dict: dict = {"mpnet": 30, "bm25": 40}
    index_path: str = (
        "/mnt/workspaces/yongqij/evals/data/indexes/Cohere_mpnet/IVFSQ_L2.index"
    )
    embedding_name: str = "paraphrase-multilingual-mpnet-base-v2"
    dataset_name: Iterable[str] = ["Cohere/wikipedia-22-12-en-embeddings"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._retrieval: MyScaleSearch = MyScaleSearch(
            index_path=self.index_path,
            model_name=self.embedding_name,
            dataset_name=self.dataset_name,
        )

    class Config:
        arbitrary_types_allowed = True

    def transform_context(self, data: Dict[str, Any], **params: Any) -> List[str]:
        question = data["question"]
        choices = "\n".join(
            [f"{chr(65+i)}. {v}" for i, v in enumerate(data["choices"])]
        )
        context = self._retrieval(
            "\n".join([question, choices]),
            self.num_filtered,
            self.num_selected,
            self.with_title,
            self.rank_dict,
            simple=True,
        )
        return context
