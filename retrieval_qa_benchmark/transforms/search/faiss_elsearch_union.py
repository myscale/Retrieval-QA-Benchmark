import os
from typing import Any, List, Optional, Tuple, Sequence

import faiss
import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer

from retrieval_qa_benchmark.utils.profiler import PROFILER

from .base import Entry, PluginVectorSearcher

from .utils import text_preprocess


class FaissElSearchBM25UnionSearcher(PluginVectorSearcher):
    model_name: str
    index_path: str
    nprobe: int = 128
    el_host: str
    el_auth: Tuple[str, str]
    dataset_name: Sequence[str] = ["Cohere/wikipedia-22-12-en-embeddings"]
    dataset_split: str = "train"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        logger.info("load index...")
        self.index = faiss.read_index(self.index_path)
        logger.info("load mpnet model...")
        self.model = SentenceTransformer(self.model_name)
        
    def search(
        self,
        query_list: list,
        num_selected: int,
        context: Optional[List[List[str]]] = None,
    ) -> Tuple[List[List[float]], List[List[Entry]]]:
        if context is not None and context not in [[], [None]]:
            logger.warning("Ignoring context data in faiss elastic search search...")
        return self.faiss_bm25_union_filter(query_list=query_list, num_selected=num_selected)

    @PROFILER.profile_function("database.FaissSearch.emb_filter.profile")
    def emb_filter(
        self, query_list: List[str], num_selected: int
    ) -> Tuple[List[List[float]], List[List[int]]]:
        if type(query_list[0]) == str:
            query_list = self.model.encode(query_list)
        assert type(query_list[0]) == np.ndarray
        D_list, para_id_list = self.index_search(query_list, num_selected)
        return D_list, para_id_list

    @PROFILER.profile_function("database.FaissSearch.index_search.profile")
    def index_search(
        self, query_list: List[str], num_selected: int
    ) -> Tuple[List[List[float]], List[List[int]]]:
        if os.path.split(self.index_path)[-1] == "IVFSQ_IP.index":
            faiss.normalize_L2(query_list)
        self.index.nprobe = self.nprobe
        D_list, para_id_list = self.index.search(query_list, num_selected)
        return D_list, para_id_list
    
    @PROFILER.profile_function("database.FaissSearch.bm25_filter.profile")
    def bm25_filter(
        self, query_list: List[str], num_selected: int
    ) -> Tuple[List[List[float]], List[List[int]]]:
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
        return score_list, para_id_list
    

    def faiss_bm25_union_filter(
        self, query_list: List[str], num_selected: int
    ) -> Tuple[List[List[float]], List[List[Entry]]]:
        D_list, para_id_list_emb = self.emb_filter(query_list, num_selected)
        score_list, para_id_list_bm25 = self.bm25_filter(query_list, num_selected)
        para_id_list = []
        rank_list = []
        for j in range(len(query_list)):
            para_ids_emb = para_id_list_emb[j]
            para_ids_bm25 = para_id_list_bm25[j]
            para_ids = []
            ranks = []
            for i in range(num_selected):
                para_id = para_ids_emb[i]
                if para_id not in para_ids:
                    para_ids.append(para_id)
                    ranks.append(i+1)
                if len(para_ids) >= num_selected:
                    break
                para_id = para_ids_bm25[i]
                if para_id not in para_ids:
                    para_ids.append(para_id)
                    ranks.append(i+1)
                if len(para_ids) >= num_selected:
                    break
            if len(para_ids) < num_selected:
                logger.warning(f"Only {len(para_ids)} unique paragraphs found, less than {num_selected}")
            para_id_list.append(para_ids)
            rank_list.append(ranks)
        entry_list = self.para_id_list_to_entry(para_id_list)
        return rank_list, entry_list
