from typing import List, Optional, Sequence, Tuple

from loguru import logger

from .base import Entry, PluginVectorSearcher
from .utils import text_preprocess


class ElSearchBM25Searcher(PluginVectorSearcher):
    el_host: str
    el_auth: Tuple[str, str]
    dataset_name: Sequence[str] = ["Cohere/wikipedia-22-12-en-embeddings"]
    dataset_split: str = "train"

    def search(
        self,
        query_list: list,
        num_selected: int,
        context: Optional[List[List[str]]] = None,
    ) -> Tuple[List[List[float]], List[List[Entry]]]:
        if context is not None and context not in [[], [None]]:
            logger.warning("Ignoring context data in elastic search search...")
        return self.bm25_filter(query_list=query_list, num_selected=num_selected)

    def bm25_filter(
        self, query_list: List[str], num_selected: int
    ) -> Tuple[List[List[float]], List[List[Entry]]]:
        from elasticsearch import Elasticsearch

        es = Elasticsearch(hosts=self.el_host, basic_auth=self.el_auth)
        para_id_list = []
        score_list = []
        for i in range(len(query_list)):
            query = query_list[i]
            query_pp = ' '.join(text_preprocess(query))
            query_ = {"match": {"context": query_pp}}
            result = es.search(index="wiki-index", query=query_, size=num_selected)
            para_ids = [int(item["_id"]) for item in result["hits"]["hits"]]
            scores = [float(item["_score"]) for item in result["hits"]["hits"]]
            para_id_list.append(para_ids)
            score_list.append(scores)
        entry_list = self.para_id_list_to_entry(para_id_list)
        return score_list, entry_list
