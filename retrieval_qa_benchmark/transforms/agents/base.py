from typing import Any, Dict, Tuple, Optional, List
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.agents.mrkl.output_parser import MRKLOutputParser
from langchain.agents.agent_toolkits.sql.prompt import SQL_PREFIX, SQL_SUFFIX
from langchain.schema import AgentAction, AgentFinish, OutputParserException

from retrieval_qa_benchmark.schema import BaseTransform, QARecord, QAPrediction
from retrieval_qa_benchmark.transforms.base import BaseLLMTransform
from retrieval_qa_benchmark.utils.registry import REGISTRY


class AgentTool(BaseTransform):
    name: str
    """name of this tool"""
    descrption: str
    """description of this tool"""
        

class AgentDecider(BaseLLMTransform):
    """Agent Decision with LangChain MRKL Agent Prompts"""

    children: List[Optional[AgentTool]] = [None, None]
    """Agent's children must be tools"""
    
    record_template: str = "Question: {question}\n{choices}"
    """Template to format records"""
    

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tool_map = {n.name: n for n in self.children if n}
        self.agent_template = self._build_agent_template()
        self._output_parser = MRKLOutputParser()
        
        # overrides the template to format `QARecord`
        self._llm.record_template = self.record_template
        
    def _build_agent_template(self) -> str:
        tool_strings = "\n".join([f"{name}: {n.descrption}" for name, n in self._tool_map.items()])
        tool_names = ", ".join([tool for tool in self._tool_map])
        format_instructions = FORMAT_INSTRUCTIONS.format(tool_names=tool_names)
        return "\n\n".join([PREFIX, tool_strings, format_instructions, SUFFIX])
        
    def check_status(self, current: Dict[str, Any]) -> int:
        return super().check_status(current)

    def _format_agent_template(self, q:str, stacked: List[str]) -> str:
        return self.agent_template.format(input=q,
                                          agent_scratchpad='\n'.join(stacked))

    def __call__(
        self, data: Dict[str, Any]
    ) -> Tuple[Optional[BaseTransform], QAPrediction]:
        """you can call :class:`BaseTransform` as functions

        :return: a transformed :class:`QARecord`
        :rtype: Tuple[Optional["BaseTransform"], QARecord]
        """
        record = QARecord(**{k: v for k, v in data.items() if v is not None})
        # format
        q = self._llm.convert_record(record)
        
        stacked = []
        if 'stack' in data and len(data['stack']) > 0:
            # get all previous generation from stacked
            stacked = [d['pred'] for d in data['stack']]
        
        # fill the template
        q = self._format_agent_template(q, stacked)
        
        # wrap this with internal call
        pred = self._llm.generate(q)
        
        decision = self._output_parser.parse(pred.generated)
        
        # This will push this generated text into stack
        p = QAPrediction(**record.model_dump(),
                         pred=pred.generated,
                         prompt_tokens=pred.prompt_tokens,
                         completion_tokens=pred.completion_tokens,
                         )
        if type(decision) is AgentFinish:
            return None, p
        elif type(decision) is AgentAction:
            return self._tool_map[decision.tool], p
        else:
            raise OutputParserException("Output decision has unidentified type.")