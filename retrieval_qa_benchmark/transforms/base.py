from typing import Any, Dict, Iterable, List

from retrieval_qa_benchmark.schema.transform import BaseTransform
from retrieval_qa_benchmark.utils.registry import REGISTRY

from retrieval_qa_benchmark.transforms.search import FaissSearch, ElSearchBM25Searcher


class BaseContextTransform(BaseTransform):
    sep_chr: str = "\n"
    num_selected: int = 5
    context_template: str = "{title} | {paragraph}"
    
    def preproc_question4query(self, data: Dict[str, Any]) -> str:
        question = data["question"]
        choices = " | ".join(data["choices"])
        return "\n".join([question, choices])
    
    def transform_context(self, data: Dict[str, Any], **params: Any) -> List[str]:
        context = self._searcher(
            self.preproc_question4query(data),
            self.num_selected,
            context=data['context']
        )
        return context

@REGISTRY.register_transform("Faiss")
class ContextWithFaiss(BaseContextTransform):
    nprobe: int = 128
    index_path: str = (
        "/mnt/workspaces/yongqij/evals/data/indexes/Cohere_mpnet/IVFSQ_L2.index"
    )
    embedding_name: str = "paraphrase-multilingual-mpnet-base-v2"
    dataset_name: Iterable[str] = ["Cohere/wikipedia-22-12-en-embeddings"]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._searcher = FaissSearch(
                model_name=self.embedding_name,
                index_path=self.index_path,
                template=self.context_template,
                nprobe=self.nprobe,
                dataset_name=self.dataset_name,
                dataset_split='train',)
    
@REGISTRY.register_transform("ElasticBM25")
class ContextWithElasticBM25(BaseContextTransform):
    el_host: str
    el_auth: str
    embedding_name: str = "paraphrase-multilingual-mpnet-base-v2"
    dataset_name: Iterable[str] = ["Cohere/wikipedia-22-12-en-embeddings"]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._searcher = ElSearchBM25Searcher(
                template=self.context_template,
                el_host=self.el_host,
                el_auth=self.el_auth,
                dataset_name=self.dataset_name,
                dataset_split='train')