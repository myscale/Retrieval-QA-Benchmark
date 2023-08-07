from typing import Any, Dict, Optional, List, Union

from retrieval_qa_benchmark.transforms import BaseTransform
from retrieval_qa_benchmark.utils.registry import REGISTRY

from .myscale_retrieval.myscale_search import MyScaleSearch


@REGISTRY.register_extra_transform("add_context")
class AddContextTransform(BaseTransform):
    sep_chr: str = "\n"
    prompt_prefix: str = "Context:"
    prompt_context: Union[str, list] = ""

    def transform_question(self, data: Dict[str, Any], **params: Any) -> str:
        question = data["question"]
        question = insert_context(question, self.sep_chr, self.prompt_prefix, self.prompt_context)
        return question

    def transform_choices(self, data: Dict[str, Any], **params: Any) -> Optional[List[str]]:
        return data['choices']
    
    
@REGISTRY.register_extra_transform("add_myscale_retrieval")
class AddMyScaleRetrievalTransform(BaseTransform):
    sep_chr: str = "\n"
    prompt_prefix: str = "Context:"
    num_filtered: int = 100
    num_selected: int = 5
    with_title: bool = True
    rank_dict: dict = {'mpnet':30, 'bm25':40}
    retrieval: Any = MyScaleSearch()

    def transform_question(self, data: Dict[str, Any], **params: Any) -> str:
        question = data["question"]
        query = question['query']
        context = self.retrieval(query, 
                                 self.num_filtered, 
                                 self.num_selected,
                                 self.with_title,
                                 self.rank_dict,
                                 simple = True,)
        question = insert_context(question, self.sep_chr, self.prompt_prefix, context)
        return question

    def transform_choices(self, data: Dict[str, Any], **params: Any) -> Optional[List[str]]:
        return data['choices']
    
    
def insert_context(question, sep_chr, prompt_prefix, prompt_context):
    context_pos = question.find(prompt_prefix)
    question_pos = question.find('Question:')
    assert question_pos != -1
    if context_pos == -1:
        context_pos == question_pos
        no_context = True
    prefix_part = question[0:context_pos]
    context_part = question[context_pos:question_pos]
    question_part = question[question_pos:]
    if no_context:
        context_part = prompt_prefix + sep_chr
    else:
        assert context_part.endswith(sep_chr)
        context_part = context_part[0:-len(sep_chr)]
    if isinstance(prompt_context, str):
        context_part += prompt_context
    else:
        context_num = context_part.count('\n')
        for i in range(len(prompt_context)):
            context_part += f'[{context_num + i}] {prompt_context[i]}' + sep_chr
    context_part += sep_chr
    return prefix_part + context_part + question_part
