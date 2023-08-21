from typing import Any, Dict, List

from retrieval_qa_benchmark.schema import BaseTransform
from retrieval_qa_benchmark.transforms.search.base import BaseSearcher


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

