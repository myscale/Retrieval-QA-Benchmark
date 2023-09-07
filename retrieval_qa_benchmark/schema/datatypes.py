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
    stack: Optional[List["LLMHistory"]] = []

    class Config:
        extra = Extra.forbid


class QAPrediction(QARecord):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generated: str
    matched: float = 0.0
    profile_time: Optional[Dict[str, Union[int, float]]] = {}
    profile_count: Optional[Dict[str, int]] = {}
    profile_avg: Optional[Dict[str, float]] = {}


class BaseLLMOutput(BaseModel):
    generated: str
    prompt_tokens: int
    completion_tokens: int


class ToolHistory(BaseModel):
    """Tool call history"""

    thought: str = ""
    """rationale step from LLM"""
    tool: Optional[str] = None
    """function called in this history"""
    tool_inputs: Union[str, dict, None] = None
    """Input for this tool call"""
    result: Optional[str] = None
    """Output from this function call"""


class LLMHistory(BaseLLMOutput):
    """LLM output history"""

    created_by: str = "default"
    """Which node creates this """
    comment: str = ""
    """extra comments to this generation"""
    extra: Union[ToolHistory, None] = None
