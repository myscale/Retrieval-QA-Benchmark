from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from pydantic import BaseModel, Extra, validator, model_validator


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
    matched: bool