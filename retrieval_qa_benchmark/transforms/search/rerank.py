from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger

from retrieval_qa_benchmark.utils.profiler import PROFILER

from .base import BaseSearcher, Entry
from .utils import text_preprocess


def SimMax(query: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
    # (q_seq_sz, dim) * (b_sz, d_seq_sz, dim).T = (b_sz, q_seq_sz, d_seq_sz)
    mat = torch.matmul(query.unsqueeze(0), embeddings.permute(0, 2, 1))
    score = mat.amax(2).sum(1).data.cpu()
    return score


def Colbert_single(args: Any, is_filter: bool = True) -> Tuple[Any, Any]:
    with torch.no_grad():
        i, sentences, query, tokenizer, model, worker_id = args
        query = F.normalize(query, p=2, dim=1)
        if is_filter:
            import string

            puncts = string.punctuation
            punct_tokens = set()
            for punct in puncts:
                punct_tokens.update(tokenizer(punct)["input_ids"][1:-1])
        scores = np.zeros(len(sentences))
        for j in range(len(sentences)):
            sentence = sentences[j]
            tokenizes = tokenizer(
                sentence, return_tensors="pt", truncation=True, max_length=512
            )["input_ids"].to(f"cuda:{worker_id}")
            tokenizes[0][1] = 2
            embeddings = model(tokenizes)["pooler_output"]
            embeddings = F.normalize(embeddings, p=2, dim=2)
            if is_filter:
                tokenizes = tokenizes[0].cpu().numpy()
                for idx in range(len(tokenizes)):
                    if tokenizes[idx] in punct_tokens:
                        embeddings[0][idx] = 0
            scores[j] = SimMax(query, embeddings)
            break
        return i, scores


class RerankSearcher(BaseSearcher):
    """"""

    rank_dict: Dict[str, int] = {"mpnet": 30, "bm25": 40}
    with_title: bool = True

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        logger.info("load Colbert model...")
        self.Colbert_init()

    def Colbert_init(self, num_gpu: int = 1) -> None:
        from multiprocessing import current_process

        from transformers import AutoConfig, AutoTokenizer

        from .colbert import HF_ColBERT

        worker_id = 0
        if num_gpu > 1:
            worker_id = (current_process()._identity[0] - 1) % num_gpu

        colbert_tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
        loadedConfig = AutoConfig.from_pretrained("colbert-ir/colbertv2.0")
        loadedConfig.dim = 128
        colbert_model = HF_ColBERT.from_pretrained(
            "colbert-ir/colbertv2.0", loadedConfig
        ).to(f"cuda:{worker_id}")
        colbert_model.eval()
        self.colbert_tokenizer = colbert_tokenizer
        self.colbert_model = colbert_model

    def search(
        self,
        query_list: List[str],
        num_selected: int,
        context: Optional[List[List[str]]] = None,
    ) -> Tuple[List[List[float]], List[List[Entry]]]:
        assert (
            context is not None
        ), "Second Stage Searcher should always work with non-empty context!"
        assert len(query_list) == len(
            context
        ), "Query list length should be equal to number of context list!"
        D_list, entry_list = self.parse_context(context)
        return D_list, self.stage2_search(query_list, entry_list, num_selected)

    @PROFILER.profile_function("database.RerankSearcher.stage2_search.profile")
    def stage2_search(
        self, question_list: List[str], entry_list: List[List[Entry]], num_selected: int
    ) -> List[List[Entry]]:
        entry_list_ = self.rrf_hybrid_search(
            question_list,
            entry_list,
            num_selected,
        )
        return entry_list_

    def bm25(
        self, keywords: List[str], words_para_list: List[List[str]]
    ) -> Tuple[Union[List[Entry], List[List[Entry]]], List[float]]:
        from rank_bm25 import BM25Okapi

        bm25_para = BM25Okapi(words_para_list)
        scores = bm25_para.get_scores(keywords)
        rank = (
            pd.DataFrame(scores)
            .rank(ascending=False, method="average")
            .values.reshape(-1)
        )
        return rank, scores

    def rank_result(
        self,
        question: str,
        entries: List[Entry],
    ) -> pd.DataFrame:
        rank_emb = np.array(range(1, len(entries) + 1), dtype=np.int32)  # noqa: F841
        para_id = []
        title = []
        para = []
        for entry in entries:
            para_id.append(entry.paragraph_id)
            title.append(entry.title)
            para.append(entry.paragraph)
        keywords = text_preprocess(question)
        words_para_list = []
        if not self.with_title:
            words_para_list = [text_preprocess(_para) for _para in para]
        else:
            words_para_list = [
                text_preprocess(_title) + text_preprocess(_para)
                for _title, _para in zip(title, para)
            ]
        db_names = ["para_id", "rank_emb"]
        for rank_name in self.rank_dict.keys():
            if rank_name == "mpnet":
                db_names.append("rank_emb")
            elif rank_name == "bm25":
                rank_bm25, score_bm25 = self.bm25(keywords, words_para_list)
                db_names.extend(["rank_bm25", "score_bm25"])
            elif rank_name == "colbert":
                rank_col, score_col = self.rank_colbert(question, entries)
                db_names.extend(["rank_col", "score_col"])
            else:
                raise ValueError(f"rank_name {rank_name} is not supported")
        db_names.extend(["title", "para"])
        result_db = pd.DataFrame()
        for name in db_names:
            result_db[name] = eval(name)
        return result_db

    def rank_colbert(
        self,
        question: str,
        entries: List[Entry],
        work_id: int = 0,
        batch_size: int = 1,
    ) -> Tuple[Union[List[Entry], List[List[Entry]]], List[float]]:
        question = "# " + question
        sentences = [f"{entry.title}\n{entry.paragraph}" for entry in entries]
        sentences = ["# " + sentence for sentence in sentences]
        scores = np.zeros(len(sentences))
        q_token_ids = self.colbert_tokenizer(question, return_tensors="pt")[
            "input_ids"
        ].to(f"cuda:{0}")
        q_token_ids[0][1] = 1
        query = self.colbert_model(q_token_ids)["pooler_output"][0]
        for i, score in map(
            Colbert_single,
            map(
                lambda i: (
                    i,
                    sentences[i : min(i + batch_size, len(sentences))],
                    query,
                    self.colbert_tokenizer,
                    self.colbert_model,
                    work_id,
                ),
                range(0, len(sentences), batch_size),
            ),
        ):
            scores[i : min(i + batch_size, len(sentences))] = score
        q_token_ids = q_token_ids[0].cpu().numpy()
        rank = (
            pd.DataFrame(scores)
            .rank(ascending=False, method="average")
            .values.reshape(-1)
        )
        return rank, scores.tolist()

    def rrf_hybrid_search(
        self,
        question_list: List[str],
        entry_list: List[List[Entry]],
        num_selected: int,
    ) -> List[List[Entry]]:
        _entry_list = []
        for i in range(len(question_list)):
            result_db = self.rank_result(
                question_list[i],
                entry_list[i],
            )
            result_db = self.rrf_result(result_db)
            paras_id = (
                result_db.sort_values(by="rank_rrf")["para_id"]
                .head(num_selected)
                .values
            )
            titles = (
                result_db.sort_values(by="rank_rrf")["title"].head(num_selected).values
            )
            paras = (
                result_db.sort_values(by="rank_rrf")["para"].head(num_selected).values
            )
            entries = []
            for j in range(len(paras_id)):
                entries.append(
                    Entry(
                        rank=j,
                        paragraph_id=paras_id[j],
                        title=titles[j],
                        paragraph=paras[j],
                    )
                )
            _entry_list.append(entries)
        return _entry_list

    def rrf_result(self, result_db: Dict[str, Any]) -> Dict[str, Any]:
        _dict = {"mpnet": "rank_emb", "bm25": "rank_bm25", "colbert": "rank_col"}
        ranks = []
        rrf_coefficients = []
        for rank_name in self.rank_dict.keys():
            ranks.append(result_db[_dict[rank_name]].values)
            rrf_coefficients.append(self.rank_dict[rank_name])
        score_rrf = self.rrf(ranks, rrf_coefficients)
        rank_rrf = (
            pd.DataFrame(score_rrf)
            .rank(ascending=False, method="average")
            .values.reshape(-1)
        )
        rrf_db = result_db
        rrf_db["rank_rrf"] = rank_rrf
        rrf_db["score_rrf"] = score_rrf
        return rrf_db

    def rrf(self, rank_list: list[int], k_list: List[int]) -> Optional[float]:
        score_rrf = None
        for rank, k in zip(rank_list, k_list):
            if score_rrf is None:
                score_rrf = 1 / (k + rank)
            else:
                score_rrf += 1 / (k + rank)
        return score_rrf
