from typing import Any, List, Optional, Tuple

from clickhouse_connect import get_client
from loguru import logger
from rake_nltk import Rake
from sentence_transformers import SentenceTransformer

from retrieval_qa_benchmark.utils.profiler import PROFILER

from .base import BaseSearcher, Entry


def is_sql_safe(word: str) -> bool:
    for c in [")", "'", ",", "("]:
        if c in word:
            return False
    return True


class MyScaleSearcher(BaseSearcher):
    """"""

    host: str
    port: int
    model_name: str
    username: str = "default"
    password: str = ""
    table_name: str = "Wikipedia"
    two_staged: bool = False
    num_filtered: int = 100
    kw_topk: int = 10

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        logger.info("connecting to myscale backend...")
        self.client = get_client(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
        )
        logger.info("load mpnet model...")
        self.model = SentenceTransformer(self.model_name)
        self.ke_model = Rake()

    @PROFILER.profile_function("database.MyScaleSearcher.search.profile")
    def search(
        self,
        query_list: list,
        num_selected: int,
        context: Optional[List[List[str]]] = None,
    ) -> Tuple[List[List[float]], List[List[Entry]]]:
        assert len(query_list) == 1, "MyScale currently does not support batch mode."
        if context is not None and context not in [[], [None]]:
            logger.warning("Ignoring context data in myscale search...")
        if type(query_list[0]) == str:
            emb_list = self.model.encode(query_list)
        query = f"""SELECT d, title, text FROM {self.table_name}
            ORDER BY distance(emb, 
                [{','.join(map(str, emb_list[0].tolist()))}]) AS d
            LIMIT {self.num_filtered if self.two_staged else num_selected}
            """
        if self.two_staged:
            self.ke_model.extract_keywords_from_text(query_list[0])
            terms = [w for w in self.ke_model.get_ranked_phrases() if is_sql_safe(w)][
                : self.kw_topk
            ]
            terms_pattern = [f"(?i){x}" for x in terms]
            query = (
                f"SELECT tempt.text AS text, tempt.title AS title, "
                f"distance1 + distance2 + tempt.d AS d "
                f"FROM ({query}) tempt "
                f"ORDER BY "
                f"length(multiMatchAllIndices(arrayStringConcat("
                f"[tempt.title, tempt.text], ' '), {terms_pattern})) "
                f"AS distance1 DESC, "
                f"log(1 + countMatches(arrayStringConcat([tempt.title, "
                f"tempt.text], ' '), '(?i)({'|'.join(terms)})')) "
                f"AS distance2, d DESC LIMIT {num_selected}"
            )
        result = self.retrieve(query)
        entry_list = [
            [
                Entry(rank=i, paragraph_id=i, title=r["title"], paragraph=r["text"])
                for i, r in enumerate(result)
            ]
        ]
        D_list = [[float(r["d"]) for r in result]]
        return D_list, entry_list

    @PROFILER.profile_function("database.MyScaleSearcher.retrieve.profile")
    def retrieve(self, query: str) -> List[Any]:
        client = get_client(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
        )
        return [r for r in client.query(query).named_results()]
