from string import Template
from typing import Any, Iterable, List, Tuple, Union

import faiss
import nltk
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from retrieval_qa_benchmark.utils.profiler import PROFILER

from .utils_search import index_search, para_id_list_to_entry, rrf_hybrid_search


class MyScaleSearch(object):
    def __init__(
        self,
        index_path: str,
        model_name: str,
        dataset_name: Iterable[str] = ["Cohere/wikipedia-22-12-en-embeddings"],
        dataset_split: str = "train",
    ):
        assert dataset_name in (
            ["wikipedia", "20220301.en"],
            ["Cohere/wikipedia-22-12-en-embeddings"],
        )
        self.dataset_name = dataset_name
        self.index_path = index_path
        self.model_name = model_name
        nltk.download("stopwords")
        nltk.download("punkt")
        nltk.download("averaged_perceptron_tagger")
        nltk.download("wordnet")
        print("load dataset...")
        self.dataset = load_dataset(*dataset_name, split=dataset_split)
        self.dataset_split = dataset_split
        print("load index...")
        self.index = faiss.read_index(index_path)
        print("load mpnet model...")
        self.model = SentenceTransformer(model_name)

    @PROFILER.profile_function("database.MyScaleSearch.profile")
    def __call__(
        self,
        question: Union[str, List[str]],
        num_filtered: int = 100,
        num_selected: int = 15,
        with_title: bool = True,
        rank_dict: dict = {"mpnet": 30, "bm25": 40},
        simple: bool = False,
        show_progress: bool = False,
    ) -> List[Any]:
        if type(question) == str:
            entry_list = self.filtered_hybrid_search(
                [question],
                num_filtered,
                num_selected,
                with_title,
                rank_dict,
                show_progress,
            )
            entry_list = entry_list[0]
        else:
            entry_list = self.filtered_hybrid_search(
                question,
                num_filtered,
                num_selected,
                with_title,
                rank_dict,
                show_progress,
            )
        if simple:
            template = "$title | $para"
            s = Template(template)
            result_list: Union[List[str], List[List[str]]] = []
            for entries in entry_list:
                if type(entries) == tuple:
                    result_list.append(s.substitute(title=entries[2], para=entries[3]))
                else:
                    results = []
                    for entry in entries:
                        results.append(s.substitute(title=entry[2], para=entry[3]))
                    result_list.append(results)  # type: ignore
            return result_list
        return entry_list

    @PROFILER.profile_function("database.MyScaleSearch.emb_filter.profile")
    def emb_filter(
        self, query_list: list, num_filtered: int
    ) -> Tuple[Any, Any]:  # (rank, para_id, title, para)
        if type(query_list[0]) == str:
            query_list = self.model.encode(query_list)
        assert type(query_list[0]) == np.ndarray
        D_list, para_id_list = index_search(
            query_list, self.index, self.index_path, num_filtered
        )
        entry_list = para_id_list_to_entry(
            para_id_list, self.dataset, self.dataset_name
        )
        return D_list, entry_list
