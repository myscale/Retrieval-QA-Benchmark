from typing import Any, Dict, Iterable, List, Union

from retrieval_qa_benchmark.schema.transform import BaseTransform
from retrieval_qa_benchmark.utils.registry import REGISTRY

from .myscale_retrieval.myscale_search import MyScaleSearch


@REGISTRY.register_transform("add_context")
class AddContextTransform(BaseTransform):
    sep_chr: str = "\n"
    prompt_prefix: str = "Context:"
    prompt_context: Union[str, list] = ""

    def transform_question(self, data: Dict[str, Any], **params: Any) -> str:
        question = data["question"]
        query = data["raw_question"]
        question = insert_context(
            question, query, self.sep_chr, self.prompt_prefix, self.prompt_context
        )
        return question


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
    model_name: str = "paraphrase-multilingual-mpnet-base-v2"
    dataset_name: Iterable[str] = ["Cohere/wikipedia-22-12-en-embeddings"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._retrieval: MyScaleSearch = MyScaleSearch(
            index_path=self.index_path,
            model_name=self.model_name,
            dataset_name=self.dataset_name,
        )

    class Config:
        arbitrary_types_allowed = True

    def transform_question(self, data: Dict[str, Any], **params: Any) -> str:
        question = data["question"]
        query = data["raw_question"]
        context = self._retrieval(
            query,
            self.num_filtered,
            self.num_selected,
            self.with_title,
            self.rank_dict,
            simple=True,
        )
        question = insert_context(
            question, query, self.sep_chr, self.prompt_prefix, context
        )
        return question


def insert_context(
    question: str,
    query: str,
    sep_chr: str,
    context_prompt: str,
    context: Union[List[Any], str],
) -> str:
    query = query.split("\n")[0]
    prefix, suffix = question.split(query)
    context_part = [context_prompt]
    if isinstance(context, str):
        context_part.append(context)
    else:
        context_num = context_part.count("\n")
        for i in range(len(context)):
            context_part.append(f"[{context_num + i + 1}] {context[i]}")
    context_part_ = sep_chr.join(context_part)
    return prefix + sep_chr.join([f"{context_part_}\n", query]) + suffix
