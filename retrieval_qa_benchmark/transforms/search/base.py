from pydantic import BaseModel, Extra
import re
from loguru import logger
from datasets import load_dataset
import numpy as np
from parse import parse
from typing import Union, List, Any, Tuple, Sequence
from abc import abstractmethod

class Entry(BaseModel):
    rank: int
    paragraph_id: int
    title: str
    paragraph: str

class BaseSearcher(BaseModel):
    """"""
    template: str = "{title} | {paragraph}"
    
    class Config:
        extra = Extra.allow
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def __call__(
        self,
        question: Union[str, List[str]],
        num_selected: int = 15,
        context: List[str] = None,
    ) -> List[str]:
        if type(question) is str:
            question = [question]
        D_list, entry_list = self.search(question, num_selected, context=context)
        return self.format(entry_list, D_list)
    
    def parse_context(self, context: List[str]) -> Tuple[List[float], List[Entry]]:
        entry_list: List[Entry] = []
        D_list: List[float] = []
        for i, c in enumerate(context):
            d = parse(self.template, c)
            entry_list.append(Entry(rank=i, paragraph_id=i, title=d['title'], paragraph=d['paragraph']))
            D_list.append(i+1/len(context))
        return D_list, entry_list
        
    @abstractmethod
    def search(self, query_list: list, num_selected: int, context: List[str] = None) -> Tuple[List[float], Union[List[Entry], List[List[Entry]]]]:
        raise NotImplementedError    

    def format(self, entry_list: Union[List[Entry], List[List[Entry]]], D_list: List[float]) -> List[str]:
        result_list: List[str] = []
        for entries, d in zip(entry_list, D_list):
            if type(entries) is Entry:
                result_list.append(self.template.format(title=entries.title, paragraph=entries.paragraph, distance=d))
            else:
                results = []
                for entry in entries:
                    results.append(self.template.format(title=entry.title, paragraph=entry.paragraph, distance=d))
                result_list.extend(results)
        return result_list
    
    
class PluginVectorSearcher(BaseSearcher):
    """
    """
    dataset_name: Sequence[str] = ["Cohere/wikipedia-22-12-en-embeddings"]
    dataset_split: str = "train"
    
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # TODO @mpskex: expand supported knowledge base
        assert self.dataset_name in (
            ["wikipedia", "20220301.en"],
            ["Cohere/wikipedia-22-12-en-embeddings"],
        )
        logger.info("load dataset...")
        self.dataset = load_dataset(*self.dataset_name, split=self.dataset_split)
    
    
    def para_id_to_entry(self, para_id: str, start_para_list: List[int]) -> Tuple[str, str]:
        para_id_ = int(para_id)
        if start_para_list is None:
            title = self.dataset[para_id_]["title"]
            para = self.dataset[para_id_]["text"]
        else:
            import bisect

            title_id = bisect.bisect(start_para_list, para_id_)
            title = self.dataset[title_id - 1]["title"]
            para = [
                para
                for para in self.dataset[title_id - 1]["text"].split("\n\n")
                if len(re.split(" |\n", para)) > 5
            ][para_id_ - start_para_list[title_id - 1]]
        return title, para

    def para_id_list_to_entry(self, para_id_list: List[Any]) -> Union[List[Entry], List[List[Entry]]]:
        start_para_list = None
        entry_list = []
        if type(para_id_list[0]) == list or type(para_id_list[0]) == np.ndarray:
            for paras_id in para_id_list:
                entries = []
                for i in range(len(paras_id)):
                    para_id = paras_id[i]
                    title, para = self.para_id_to_entry(para_id, start_para_list)
                    entries.append(Entry(rank=i, paragraph_id=para_id, title=title, paragraph=para))
                entry_list.append(entries)
        else:
            for i in range(len(para_id_list)):
                para_id = para_id_list[i]
                title, para = self.para_id_to_entry(para_id, start_para_list)
                entry_list.append(Entry(rank=i, paragraph_id=para_id, title=title, paragraph=para))  # type: ignore
        return entry_list