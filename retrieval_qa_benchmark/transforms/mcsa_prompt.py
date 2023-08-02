from typing import Any, Dict

from retrieval_qa_benchmark.transforms import BaseTransform
from retrieval_qa_benchmark.utils.registry import REGISTRY


@REGISTRY.register_extra_transform("mcsa_prompt")
class MultipleChoiceTransform(BaseTransform):
    sep_chr: str = "\n"
    prompt_prefix: str = "Please answer with the letter of the correct answer.\n"
    prompt_suffix: str = ""

    def transform_question(self, data: Dict[str, Any], **params: Any) -> str:
        return self.sep_chr.join(
            [self.prompt_prefix, data["question"], self.prompt_suffix]
        )
