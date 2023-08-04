import pandas as pd
from pydantic import field_validator
from typing import Any, Dict, Optional, List

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

    def transform_choices(self, data: Dict[str, Any], **params: Any) -> Optional[List[str]]:
        return data['choices']
    
@REGISTRY.register_extra_transform("mcsa_fewshot_mmlu")
class FewshotMultipleChoiceTransform(BaseTransform):
    example_csv: str
    sep_chr: str = "\n"
    
    @field_validator("example_csv")
    @classmethod
    def convert_csv(cls, example_csv) -> Any:
        example_csv = pd.read_csv(example_csv, names=("Question", "A", "B", "C", "D", "Answer"))
        
        fewshot = []
        for n in range(len(example_csv)):
            x = example_csv.iloc[n]    
            choices = "\t".join(
                [f"{chr(65+i)}. {v}" for i, v in enumerate(x[['A', 'B', 'C', 'D']])]
            )
            fewshot.append(f"Question: {x['Question']}\nChoices: {choices}\nAnwser: {x['Answer']}\n")
        return '\n'.join(fewshot)
    
    
    def transform_question(self, data: Dict[str, Any], **params: Any) -> str:
        return self.sep_chr.join(
            [self.example_csv, data["question"]]
        )
