from typing import Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, Extra


class QARecord(BaseModel):
    """Base data record for questioning & answering"""

    id: str
    """identifier for this record"""
    question: str
    """question to ask in string"""
    answer: str
    """the true answer from the dataset"""
    type: str
    """type of this question. can be one of ['mcsa', 'mcma']
    mcsa: multiple choice single answer
    mcma: multiple choice multiple answer
    """
    context: Optional[Sequence[str]] = None
    """list of context strings that are retrieved from db or other sources"""
    choices: Optional[Sequence[str]] = None
    """choices where model should be choosing from. only present in ['mcsa', 'mcma']"""
    stack: Optional[List["LLMHistory"]] = []
    """stacked intermediate prediction results (for multi-hop qa pipelines)"""

    class Config:
        extra = Extra.forbid


class QAPrediction(QARecord):
    """Base prediction result for questioning & answering"""

    prompt_tokens: int = 0
    """number of input tokens"""
    completion_tokens: int = 0
    """number of generated tokens"""
    generated: str
    """output from the model, is compared with the true answer in :class:`QARecord`"""
    matched: float = 0.0
    """match score that measures how accurate this prediction is to the answer"""
    profile_time: Optional[Dict[str, Union[int, float]]] = {}
    """accumulated time profiling regarding to each function"""
    profile_count: Optional[Dict[str, int]] = {}
    """accumulated number of execution to each profiled functions"""
    profile_avg: Optional[Dict[str, float]] = {}
    """calculated averaged time consumption. equals to time / count."""


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
