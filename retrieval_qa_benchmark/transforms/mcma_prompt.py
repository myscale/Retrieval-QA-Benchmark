from typing import Any, Dict, Optional, List

from retrieval_qa_benchmark.transforms import BaseTransform
from retrieval_qa_benchmark.utils.registry import REGISTRY


@REGISTRY.register_extra_transform("mcsa_prompt")
class MultipleChoiceTransform(BaseTransform):
    sep_chr: str = "\n"
    prompt_prefix: str = (
        "Please answer with the letter of the correct answer. "
        "Note that there may be more than one correct answer for the question.\n"
    )
    prompt_suffix: str = ""

    def transform_question(self, data: Dict[str, Any], **params: Any) -> str:
        return self.sep_chr.join(
            [self.prompt_prefix, f"{data['question']}", self.prompt_suffix]
        )

    def transform_choices(self, data: Dict[str, Any], **params: Any) -> Optional[List[str]]:
        return data['choices']
