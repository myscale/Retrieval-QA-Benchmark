from typing import Any, Dict, List, Optional, Tuple, Union

from langchain.agents.mrkl.output_parser import MRKLOutputParser
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.schema import AgentAction, AgentFinish, OutputParserException

from retrieval_qa_benchmark.schema import (
    BaseTransform,
    LLMHistory,
    QAPrediction,
    QARecord,
    ToolHistory,
)
from retrieval_qa_benchmark.transforms.base import BaseLLMTransform
from retrieval_qa_benchmark.utils.registry import REGISTRY


class AgentBase(BaseTransform):
    verbose: bool = False
    """If true, then agent will print output to stdout"""

    def execute_action(self, record: QARecord) -> Tuple[str, int, int]:
        """execute action for agent components

        :param record: data record to be processed
        :type record: QARecord
        :return: (generated file, number of prompt tokens, number of generated tokens)
        :rtype: Tuple[str, int, int]
        """
        return "This is a text", 0, 0

    def get_next_state(self, generate: str) -> Tuple[Optional[BaseTransform], str]:
        return self.children[0], generate

    def parse_extra(self, generate: str) -> Union[ToolHistory, None]:
        return None

    def __call__(
        self, data: Dict[str, Any]
    ) -> Tuple[Optional[BaseTransform], Union[QARecord, QAPrediction]]:
        """you can call :class:`BaseTransform` as functions

        :return: a transformed :class:`QARecord`
        :rtype: Tuple[Optional["BaseTransform"], QARecord]
        """
        record = QARecord(**{k: v for k, v in data.items() if v is not None})

        pred, ptok, ctok = self.execute_action(record)
        if self.verbose:
            print(
                f"\033[93m// generated @ {type(self)}-{id(self)} // -> \n\033[0m", pred
            )
        next, pred = self.get_next_state(pred)

        # This will push this generated text into stack
        if not record.stack:
            record.stack = []
        if next:
            extra = self.parse_extra(pred)
            record.stack.append(
                LLMHistory(
                    generated=pred,
                    prompt_tokens=ptok,
                    completion_tokens=ctok,
                    created_by=f"{type(self)}@{id(self)}",
                    extra=extra,
                )
            )
        else:
            record = QAPrediction(
                **record.model_dump(),
                generated=pred,
                prompt_tokens=ptok,
                completion_tokens=ctok,
            )
        return next, record


class AgentTool(AgentBase):
    name: str
    """name of this tool"""
    descrption: str
    """description of this tool"""

    def parse_extra(self, generate: str) -> ToolHistory | None:
        return ToolHistory(result=generate)

    def set_children(self, children: List[BaseTransform | None]) -> None:
        """Set children for transform

        :param children: next nodes to execute
        :type children: List[BaseTransform  |  None]
        """
        for n in children:
            if n:
                assert isinstance(
                    n, AgentRouter
                ), "Tools should always return to Routers"
        super().set_children(children)


@REGISTRY.register_transform("MRKLAgent")
class AgentRouter(AgentBase, BaseLLMTransform):
    """Agent Routing with LangChain MRKL Agent Prompts"""

    record_template: str = "{question}\n{choices}"
    """Template to format records"""

    prefix: str = PREFIX
    """Template prefix for agent"""

    suffix: str = SUFFIX
    """Template suffix for agent"""

    format_instructions: str = FORMAT_INSTRUCTIONS
    """Instruction to teach LLM what is the output format"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._output_parser = MRKLOutputParser()

        # overrides the template to format `QARecord`
        self._llm.record_template = self.record_template

    def set_children(self, children: List[BaseTransform | None]) -> None:
        """Set children for transform

        :param children: next nodes to execute
        :type children: List[BaseTransform  |  None]
        """
        for n in children:
            if n:
                assert isinstance(n, AgentRouter) or isinstance(n, AgentTool), (
                    "Router's children should be either Router or Tool. "
                    f"Got {type(n)} instead."
                )
        super().set_children(children)
        self._tool_map = {n.name: n for n in self.children if isinstance(n, AgentTool)}
        self.agent_template = self.build_agent_template()

    def parse_extra(self, generate: str) -> Union[ToolHistory, None]:
        decision = self._output_parser.parse(generate)
        if type(decision) is AgentAction:
            if type(decision.tool_input) is str:
                decision.tool_input = decision.tool_input.split("\nObservation: ")[0]
            return ToolHistory(
                thought=decision.log.split("\n")[0],
                tool=decision.tool,
                tool_inputs=decision.tool_input,
            )
        else:
            return None

    def build_agent_template(self) -> str:
        tool_strings = "\n".join(
            [f"{name}: {n.descrption}" for name, n in self._tool_map.items()]
        )
        tool_names = ", ".join([tool for tool in self._tool_map])
        format_instructions = self.format_instructions.format(tool_names=tool_names)
        return "\n\n".join(
            [self.prefix, tool_strings, format_instructions, self.suffix]
        )

    def format_agent_template(self, q: str, stacked: List[str]) -> str:
        return self.agent_template.format(input=q, agent_scratchpad="\n".join(stacked))

    def get_next_state(self, generated: str) -> Tuple[Optional[BaseTransform], str]:
        decision = self._output_parser.parse(generated)
        if type(decision) is AgentFinish:
            if self.verbose:
                print(
                    f"\033[96mThe Final Answer is "
                    f"{decision.return_values['output']}\033[0m"
                )
            return None, decision.return_values["output"]
        elif type(decision) is AgentAction:
            return self._tool_map[decision.tool], generated
        else:
            raise OutputParserException("Output decision has unidentified type.")

    def execute_action(self, record: QARecord) -> Tuple[str, int, int]:
        stacked: List[str] = []
        if record.stack and len(record.stack) > 0:
            # get all previous generation from stacked
            for d in record.stack:
                if d.extra:
                    if d.extra.tool:
                        stacked.append(
                            f"{'Thoughts:' if len(stacked) > 0 else ''} "
                            f"{d.extra.thought}\n"
                            f"Action: {d.extra.tool}\n"
                            f"Action Input: {d.extra.tool_inputs}"
                        )
                    elif d.extra.result:
                        stacked.append(f"Observation: {d.extra.result}")

        # fill the template
        q = self._llm.convert_record(record)
        q = self.format_agent_template(q, stacked)

        if len(stacked) > 0:
            q += "\nThought: "
        if self.verbose:
            print(f"\033[92m{q}\033[0m")
        # wrap this with internal call
        pred = self._llm.generate(q)
        return pred.generated, pred.prompt_tokens, pred.completion_tokens
