from pydantic import BaseModel


class QARecord(BaseModel):
    """Base data row"""

    id: str
    question: str
    answer: str
    type: str
