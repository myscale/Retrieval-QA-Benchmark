import re
from typing import Any, List, Optional, Sequence, Tuple

from datasets import load_dataset
from loguru import logger
from parse import parse
from pydantic import BaseModel, Extra


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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        question: List[str],
        num_selected: int = 15,
        context: Optional[List[List[str]]] = None,
    ) -> List[str]:
        if context:
            assert len(question) == len(context)
        D_list, entry_list = self.search(question, num_selected, context=context)
        return self.format(entry_list, D_list)

    def parse_context(
        self, context: List[List[str]]
    ) -> Tuple[List[List[float]], List[List[Entry]]]:
        entry_list: List[List[Entry]] = []
        D_list: List[List[float]] = []
        for con in context:
            temp_d_list = []
            temp_entries = []
            for i, c in enumerate(con):
                d = parse(self.template, c)
                temp_entries.append(
                    Entry(
                        rank=i,
                        paragraph_id=i,
                        title=d["title"],
                        paragraph=d["paragraph"],
                    )
                )
                temp_d_list.append(i + 1 / len(context))
            D_list.append(temp_d_list)
            entry_list.append(temp_entries)
        return D_list, entry_list

    def search(
        self,
        query_list: List[str],
        num_selected: int,
        context: Optional[List[List[str]]] = None,
    ) -> Tuple[List[List[float]], List[List[Entry]]]:
        raise NotImplementedError

    def format(
        self, entry_list: List[List[Entry]], D_list: List[List[float]]
    ) -> List[str]:
        result_list: List[str] = []
        for entries, dlist in zip(entry_list, D_list):
            results = []
            for entry, d in zip(entries, dlist):
                results.append(
                    self.template.format(
                        title=entry.title, paragraph=entry.paragraph, distance=d
                    )
                )
            result_list.extend(results)
        return result_list


class PluginVectorSearcher(BaseSearcher):
    """ """

    dataset_name: Sequence[str] = ["Cohere/wikipedia-22-12-en-embeddings"]
    dataset_split: str = "train"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # TODO @mpskex: expand supported knowledge base
        assert self.dataset_name in (
            ["wikipedia", "20220301.en"],
            ["Cohere/wikipedia-22-12-en-embeddings"],
        )
        logger.info("load dataset...")
        self.dataset = load_dataset(*self.dataset_name, split=self.dataset_split)

    def para_id_to_entry(
        self, para_id: int, start_para_list: Optional[List[int]]
    ) -> Tuple[str, str]:
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

    def para_id_list_to_entry(self, para_id_list: List[List[int]]) -> List[List[Entry]]:
        start_para_list = None
        entry_list = []
        for paras_id in para_id_list:
            entries = []
            for i in range(len(paras_id)):
                para_id = paras_id[i]
                title, para = self.para_id_to_entry(para_id, start_para_list)
                entries.append(
                    Entry(rank=i, paragraph_id=para_id, title=title, paragraph=para)
                )
            entry_list.append(entries)
        return entry_list
