from typing import Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, Extra


class QARecord(BaseModel):
    """Base data record for questioning & answering"""

    id: str
    question: str
    answer: str
    type: str
    context: Optional[Sequence[str]] = None
    choices: Optional[Sequence[str]] = None
    stack: Optional[List["QAPrediction"]] = []

    class Config:
        extra = Extra.forbid


class QAPrediction(QARecord):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    pred: str
    matched: float
    profile_time: Optional[Dict[str, Union[int, float]]] = {}
    profile_count: Optional[Dict[str, int]] = {}
    profile_avg: Optional[Dict[str, float]] = {}
