from typing import List, Tuple, Sequence, Any, Union
from loguru import logger
from .base import PluginVectorSearcher, Entry
from .utils import text_preprocess

class ElSearchBM25Searcher(PluginVectorSearcher):
    el_host: str
    el_auth: Sequence[str]
    dataset_name: Sequence[str] = ["Cohere/wikipedia-22-12-en-embeddings"]
    dataset_split: str = "train"
    
    def search(self, query_list: list, num_selected: int, context: List[str] = None) -> Tuple[Any, Any]:
        if context:
            logger.warning(f"Ignoring context data in faiss search: [{', '.join(context[0]+['...'])}]")
        return self.bm25_filter(query_list=query_list, num_selected=num_selected)
    
    def bm25_filter(
        self,
        query_list: List[str],
        num_filtered: int
    ) -> Tuple[List[float], Union[List[Entry], List[List[Entry]]]]:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(
            hosts = self.el_host,
            basic_auth=self.el_auth
            )
        para_id_list = []
        score_list = []
        for i in range(len(query_list)):
            query = query_list[i]
            query_pp = text_preprocess(query)
            query = {
                'match': {
                    'context': query_pp
                    }
                }
            result = es.search(index='wiki-index', query=query, size=num_filtered)
            para_ids = [int(item['_id']) for item in result['hits']['hits']]
            scores = [item['_score'] for item in result['hits']['hits']]
            para_id_list.append(para_ids)
            score_list.append(scores)
        entry_list = self.para_id_list_to_entry(para_id_list)
        return score_list, entry_list
    