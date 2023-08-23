import os
from typing import Any, List, Optional, Tuple, Sequence, Callable

import faiss
import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer

from retrieval_qa_benchmark.utils.profiler import PROFILER

from .base import Entry, PluginVectorSearcher

from .utils import text_preprocess


class FaissElSearchBM25HybridSearcher(PluginVectorSearcher):
    model_name: str
    index_path: str
    nprobe: int = 128
    el_host: str
    el_auth: Tuple[str, str]
    dataset_name: Sequence[str] = ["Cohere/wikipedia-22-12-en-embeddings"]
    dataset_split: str = "train"
    num_filtered: int
    is_raw_rank: bool
    text_preprocess: Callable = text_preprocess

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
        return self.faiss_bm25_hybrid_filter(query_list=query_list, num_selected=num_selected, num_filtered=self.num_filtered, is_raw_rank=self.is_raw_rank)

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
            query_pp = ' '.join(self.text_preprocess(query))
            query_ = {"match": {"context": query_pp}}
            result = es.search(index="wiki-index", query=query_, size=num_selected)
            para_ids = [int(item["_id"]) for item in result["hits"]["hits"]]
            scores = [float(item["_score"]) for item in result["hits"]["hits"]]
            para_id_list.append(para_ids)
            score_list.append(scores)
        return score_list, para_id_list
    
    
    def faiss_bm25_hybrid_filter(
        self, query_list: List[str], num_selected: int, num_filtered: int, is_raw_rank: bool
    ) -> Tuple[List[List[float]], List[List[Entry]]]:
        
        def rrf(rank_list: List[Any], k_list: List[int] = [40,40]) -> Optional[float]:
            score_rrf = None
            for rank, k in zip(rank_list, k_list):
                if score_rrf is None:
                    score_rrf = 1 / (k + rank)
                else:
                    score_rrf += 1 / (k + rank)
            return score_rrf
        
        D_list, para_id_list_emb = self.emb_filter(query_list, num_filtered)
        score_list, para_id_list_bm25 = self.bm25_filter(query_list, num_filtered)
        para_id_list = []
        rrf_list = []
        for j in range(len(query_list)):
            result_df = pd.DataFrame()
            para_ids_emb = para_id_list_emb[j].tolist()
            para_ids_bm25 = para_id_list_bm25[j]
            para_ids_interscetion = list(set.intersection(set(para_ids_emb), set(para_ids_bm25)))
            raw_rank_emb = []
            raw_rank_bm25 = []
            for para_id in para_ids_interscetion:
                idx_emb = para_ids_emb.index(para_id)
                raw_rank_emb.append(idx_emb + 1)
                idx_bm25 = para_ids_bm25.index(para_id)
                raw_rank_bm25.append(idx_bm25 + 1)
            result_df['para_id'] = para_ids_interscetion
            result_df['raw_rank_emb'] = raw_rank_emb
            result_df['raw_rank_bm25'] = raw_rank_bm25
            pro_rank_emb = result_df['raw_rank_emb'].rank(method='min').values.reshape(-1)
            pro_rank_bm25 = result_df['raw_rank_bm25'].rank(method='min').values.reshape(-1)
            result_df['pro_rank_emb'] = pro_rank_emb
            result_df['pro_rank_bm25'] = pro_rank_bm25
            if is_raw_rank:
                rank_names = ['raw_rank_emb', 'raw_rank_bm25']
            else:
                rank_names = ['pro_rank_emb', 'pro_rank_bm25']
            rank_list = [result_df[rank_name].values.reshape(-1) for rank_name in rank_names]
            score_rrf = rrf(rank_list)
            result_df['score_rrf'] = score_rrf
            if len(result_df) < num_selected:
                logger.warning(f"Only {len(result_df)} unique paragraphs found, less than {num_selected}")
                para_ids = (
                    result_df.sort_values(by="score_rrf")["para_id"]
                    .head(len(result_df))
                    .values
                )
            else:
                para_ids = (
                    result_df.sort_values(by="score_rrf")["para_id"]
                    .head(num_selected)
                    .values
                )
            para_id_list.append(para_ids)
            rrf_list.append(score_rrf)
        entry_list = self.para_id_list_to_entry(para_id_list)
        return rrf_list, entry_list
