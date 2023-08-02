from __future__ import annotations

import re
from typing import List

from sentence_transformers import SentenceTransformer

from retrieval_qa_benchmark.schema import BaseEmbedding


class STEmbedding(BaseEmbedding):
    """sentence transformer text embedding"""

    name: str
    model: SentenceTransformer

    @classmethod
    def build(
        cls, name: str = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    ) -> STEmbedding:
        model = SentenceTransformer(
            "sentence-transformers/distiluse-base-multilingual-cased-v2"
        )
        name = re.sub(r"[^a-zA-Z0-9]", "_", name)
        dim = model.encode("test").squeeze().shape[0]
        return cls(name=name, model=model, dim=dim)

    def encode(self, text: str) -> List[float]:
        return self.model.encode(text).squeeze().numpy().tolist()

    def batch_encode(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        return (
            self.model.encode(texts, batch_size=batch_size).squeeze().numpy().tolist()
        )
