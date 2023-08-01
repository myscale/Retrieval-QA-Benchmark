from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Extra, validator


class QARecord(BaseModel):
    """Base data record for questioning & answering"""

    id: str
    question: str
    answer: str
    type: str

    class Config:
        extra = Extra.forbid


class KnowledgeRecord(BaseModel):
    """Base data record for knowledge retrieval"""

    id: str
    title: Optional[str]
    context: str
    embedding: List[float]
    embedding_dim: int

    class Config:
        extra = Extra.forbid

    @validator("embedding", "embedding_dim", always=True)
    def check_dim(cls, values: Dict[str, Any]) -> Dict:
        embedding = values["embedding"]
        embedding_dim = values["embedding_dims"]
        if type(embedding) == np.ndarray:
            assert (
                len(embedding.shape) == 1
            ), "Embedding must be 1-dimentional array of floats"
            assert embedding.dtype in [
                np.float32,
                np.float16,
            ], "Embedding's data type must be one of `float32`, `float16`"
        assert len(embedding) == embedding_dim, (
            f"Embedding's shape should be ({embedding_dim},)"
            " but ({len(embedding)},) found."
        )
        return values
