from typing import Iterable
import faiss
from datasets import load_dataset
import numpy as np
from sentence_transformers import SentenceTransformer
from string import Template

from .utils_search import *


class MyScaleSearch(object):
    
    def __init__(
        self,
        dataset_name: Iterable[str] = [f"Cohere/wikipedia-22-12-en-embeddings"],
        dataset_split: str = 'train',
        index_path: str = '/mnt/workspaces/yongqij/evals/data/indexes/Cohere_mpnet/IVFSQ_L2.index',
        model_name: str = 'paraphrase-multilingual-mpnet-base-v2',
    ):
        assert dataset_name in (['wikipedia','20220301.en'], [f"Cohere/wikipedia-22-12-en-embeddings"])
        self.dataset_name = dataset_name
        self.index_path = index_path
        self.model_name = model_name
        print('load dataset...')
        self.dataset = load_dataset(*dataset_name, split=dataset_split)
        self.dataset_split = dataset_split
        print('load index...')
        self.index = faiss.read_index(index_path)
        print('load mpnet model...')
        self.model = SentenceTransformer(model_name)
        print('load Colbert model...')
        self.Colbert_init()
        

    def __call__(
        self, 
        question: str or list[str], 
        num_filtered: int = 100,
        num_selected: int = 15,
        with_title: bool = True,
        rank_dict: dict = {'mpnet':30, 'bm25':40},
        simple: bool = False,
        show_progress: bool = False,
    ):
        if type(question) == str:
            entry_list = self.filtered_hybrid_search([question], num_filtered, num_selected, with_title, rank_dict, show_progress)
            entry_list = entry_list[0]
        else:
            entry_list = self.filtered_hybrid_search(question, num_filtered, num_selected, with_title, rank_dict, show_progress)
        if simple:
            template = '$title | $para'
            s = Template(template)
            result_list = []
            for entries in entry_list:
                if type(entries) == tuple:
                    result_list.append(s.substitute(title=entries[2], para=entries[3]))
                else:
                    results = []
                    for entry in entries:
                        results.append(s.substitute(title=entry[2], para=entry[3]))
                    result_list.append(results)
            return result_list
        return entry_list
        
    def Colbert_init(
        self,
        num_gpu: int = 1
    ):
        from transformers import AutoTokenizer, AutoConfig
        from HF_Colbert import HF_ColBERT
        from multiprocessing import current_process
        global worker_id
        # global worker_id, colbert_model, colbert_tokenizer
        if num_gpu > 1:
            worker_id = (current_process()._identity[0] - 1) % num_gpu
        else:
            worker_id = 0
        colbert_tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
        loadedConfig  = AutoConfig.from_pretrained("colbert-ir/colbertv2.0")
        loadedConfig.dim = 128
        colbert_model = HF_ColBERT.from_pretrained("colbert-ir/colbertv2.0", loadedConfig).to(f'cuda:{worker_id}')
        colbert_model.eval()
        self.colbert_tokenizer = colbert_tokenizer
        self.colbert_model = colbert_model
    
    
    def emb_filter(
        self, 
        query_list: list, 
        num_filtered: int
    ) -> list[list[tuple]]: # (rank, para_id, title, para)
        if type(query_list[0]) == str:
            query_list = self.model.encode(query_list)
        assert type(query_list[0]) == np.ndarray
        D_list, para_id_list = index_search(query_list, self.index, self.index_path, num_filtered)
        entry_list = para_id_list_to_entry(para_id_list, self.dataset, self.dataset_name)
        return D_list, entry_list
    

    def hybrid_search(
        self, 
        question_list: list[str],
        entry_list: list[list[tuple]],
        num_selected: int,
        with_title: bool = True,
        rank_dict: dict = {'mpnet':30, 'bm25':40},
        show_progress: bool = False,
    ) -> list[list[tuple]]: # (rank, para_id, title, para)
        _entry_list = rrf_hybrid_search(question_list, entry_list, num_selected, with_title, rank_dict, self.colbert_tokenizer, self.colbert_model, show_progress)
        return _entry_list
    
    
    def filtered_hybrid_search(
        self,
        question_list: list[str],
        num_filtered: int,
        num_selected: int,
        with_title: bool = True,
        rank_dict: dict = {'mpnet':30, 'bm25':40},
        show_progress: bool = False,
    ) -> list[list[tuple]]:
        D_list, entry_list = self.emb_filter(question_list, num_filtered)
        _entry_list = self.hybrid_search(question_list, entry_list, num_selected, with_title, rank_dict, show_progress)
        return _entry_list