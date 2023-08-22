from typing import List, Optional, Sequence, Tuple, Callable

from loguru import logger

from .base import Entry, PluginVectorSearcher
from .utils import text_preprocess


class ElSearchSearcher(PluginVectorSearcher):
    """Elastic searcher"""

    el_host: str
    """hostname to elastic search backend"""
    el_auth: Tuple[str, str]
    """auth tuple for elastic search"""
    dataset_name: Sequence[str] = ["Cohere/wikipedia-22-12-en-embeddings"]
    """dataset name for plugin dataset"""
    dataset_split: str = "train"
    """split for that dataset"""
    text_preprocess: Callable = text_preprocess

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
        """BM25 search

        :param query_list: list of queries
        :type query_list: List[str]
        :param num_selected: number of returned context
        :type num_selected: int
        :return: distances and entries
        :rtype: Tuple[List[List[float]], List[List[Entry]]]
        """
        from elasticsearch import Elasticsearch

        es = Elasticsearch(hosts=self.el_host, basic_auth=self.el_auth)
        para_id_list = []
        score_list = []
        for i in range(len(query_list)):
            query = query_list[i]
            query_pp = ' '.join(self.text_preprocess(query))
            query_ = {"match": {"context": query_pp}}
            result = es.search(index="wiki-index", query=query_, size=num_selected)
            para_ids = [int(item["_id"]) for item in result["hits"]["hits"]]
            scores = [float(item["_score"]) for item in result["hits"]["hits"]]
            para_id_list.append(para_ids)
            score_list.append(scores)
        entry_list = self.para_id_list_to_entry(para_id_list)
        return score_list, entry_list
