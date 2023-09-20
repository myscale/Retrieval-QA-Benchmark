from typing import Any, List, Optional, Tuple

from langchain.agents.agent_toolkits.sql.prompt import SQL_PREFIX, SQL_SUFFIX
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.tools.sql_database.prompt import QUERY_CHECKER
from langchain.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLDataBaseTool,
)
from langchain.utilities.sql_database import SQLDatabase
from sqlalchemy import MetaData, create_engine

from retrieval_qa_benchmark.schema import BaseTransform, QARecord
from retrieval_qa_benchmark.transforms.agents.base import AgentRouter, AgentTool
from retrieval_qa_benchmark.transforms.base import BaseLLMTransform
from retrieval_qa_benchmark.utils.registry import REGISTRY

LIST_SQL_DATABASE_NAME = "sql_db_list_tables"
INFO_SQL_DATABASE_NAME = "sql_db_schema"
QUERY_SQL_DATABASE_NAME = "sql_db_query"
QUERY_CHECKER_SQL_NAME = "sql_db_query_checker"

list_sql_database_tool_description = (
    "Input is an empty string, output is a comma separated list "
    "of tables in the database."
)

info_sql_database_tool_description = (
    "Input to this tool is a comma-separated list of tables, output is the "
    "schema and sample rows for those tables. "
    "Be sure that the tables actually exist by calling "
    f"{LIST_SQL_DATABASE_NAME} first! "
    "Example Input: 'table1, table2, table3'"
)

query_sql_database_tool_description = (
    "Input to this tool is a detailed and correct SQL query, output is a "
    "result from the database. If the query is not correct, an error message "
    "will be returned. If an error is returned, rewrite the query, check the "
    "query, and try again. If you encounter an issue with Unknown column "
    f"'xxxx' in 'field list', using {INFO_SQL_DATABASE_NAME} "
    "to query the correct table fields."
)

query_sql_checker_tool_description = (
    "Use this tool to double check if your query is correct before executing "
    "it. Always use this tool before executing a query with "
    f"{QUERY_SQL_DATABASE_NAME}!"
)

tool_desc_map = {
    LIST_SQL_DATABASE_NAME: list_sql_database_tool_description,
    INFO_SQL_DATABASE_NAME: info_sql_database_tool_description,
    QUERY_SQL_DATABASE_NAME: query_sql_database_tool_description,
    QUERY_CHECKER_SQL_NAME: query_sql_checker_tool_description,
}


class LangChainSQLAgentTool(AgentTool):
    url: str
    """URL string to create engines"""
    name: str = "langchain_sql_tool"
    """name for this tool"""
    descrption: str = "This is the base class of langchain sql tool"
    """prompt description to this tool"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._engine = create_engine(self.url)
        self._metadata = MetaData(bind=self._engine)
        self._db = SQLDatabase(self._engine, None, self._metadata)
        self._tool: Any = None

    def set_children(self, children: List[Optional[BaseTransform]]) -> None:
        assert len(children) == 1, f"{self.name} should only have one outgoing route."
        super().set_children(children)


@REGISTRY.register_transform("LangChainSQLAgent")
class LangChainSQLAgentRouter(AgentRouter):
    """Agent Decision with LangChain SQL Agent Prompts"""

    sql_dialect: str = "SQL"
    """SQL dialect that helps the LLM understand which SQL its working on"""
    sql_topk: int = 5
    """Maximum retrieved context from database"""
    prefix: str = SQL_PREFIX
    """Template prefix for agent"""
    suffix: str = SQL_SUFFIX
    """Template suffix for agent"""

    format_instructions: str = FORMAT_INSTRUCTIONS
    """Instruction to teach LLM what is the output format"""

    def format_agent_template(self, q: str, stacked: List[str]) -> str:
        return self.agent_template.format(
            dialect=self.sql_dialect,
            top_k=self.sql_topk,
            input=q,
            agent_scratchpad="\n".join(stacked),
        )


@REGISTRY.register_transform("LangChainListSQLTool")
class LangChainListSQLDB(LangChainSQLAgentTool):
    name: str = LIST_SQL_DATABASE_NAME
    descrption: str = list_sql_database_tool_description

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tool = ListSQLDatabaseTool(
            name=self.name,
            db=self._db,
            description=list_sql_database_tool_description,
        )

    def execute_action(self, record: QARecord) -> Tuple[str, int, int]:
        assert (
            record.stack and len(record.stack) > 0
        ), "LLM should always call tools after routers."
        assert (
            record.stack[-1].extra is not None
        ), "LLM must return valid output with tool call."
        assert (
            record.stack[-1].extra.tool_inputs is not None
        ), "LLM must provides input for LangChain ListSQL Tool."
        out = self._tool.run(record.stack[-1].extra.tool_inputs)
        return f"Here are some available tables: {out}", 0, 0


@REGISTRY.register_transform("LangChainInfoSQLTool")
class LangChainInfoSQLDB(LangChainSQLAgentTool):
    name: str = INFO_SQL_DATABASE_NAME
    descrption: str = info_sql_database_tool_description

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tool = InfoSQLDatabaseTool(
            name=self.name,
            db=self._db,
            description=info_sql_database_tool_description,
        )

    def execute_action(self, record: QARecord) -> Tuple[str, int, int]:
        assert (
            record.stack and len(record.stack) > 0
        ), "LLM should always call tools after routers."
        assert (
            record.stack[-1].extra is not None
        ), "LLM must return valid output with tool call."
        assert (
            record.stack[-1].extra.tool_inputs is not None
        ), "LLM must provides input for LangChain InfoSQL Tool."
        return (
            "Here is the table schema: "
            f"{self._tool.run(record.stack[-1].extra.tool_inputs)}",
            0,
            0,
        )


@REGISTRY.register_transform("LangChainQuerySQLTool")
class LangChainQuerySQLDB(LangChainSQLAgentTool):
    name: str = QUERY_SQL_DATABASE_NAME
    descrption: str = query_sql_database_tool_description

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tool = QuerySQLDataBaseTool(
            name=self.name,
            db=self._db,
            description=query_sql_database_tool_description,
        )

    def execute_action(self, record: QARecord) -> Tuple[str, int, int]:
        assert (
            record.stack and len(record.stack) > 0
        ), "LLM should always call tools after routers."
        assert (
            record.stack[-1].extra is not None
        ), "LLM must return valid output with tool call."
        assert (
            record.stack[-1].extra.tool_inputs is not None
        ), "LLM must provides input for LangChain QuerySQL Tool."
        return (
            "This is the result I got from SQL database: "
            + self._tool.run(record.stack[-1].extra.tool_inputs),
            0,
            0,
        )


@REGISTRY.register_transform("LangChainSQLCheckerTool")
class LangChainSQLChecker(LangChainSQLAgentTool, BaseLLMTransform):
    sql_dialect: str = "SQL"
    name: str = QUERY_CHECKER_SQL_NAME
    descrption: str = query_sql_checker_tool_description
    checker_prompt: str = QUERY_CHECKER

    def execute_action(self, record: QARecord) -> Tuple[str, int, int]:
        assert (
            record.stack and len(record.stack) > 0
        ), "LLM should always call tools after routers."
        assert (
            record.stack[-1].extra is not None
        ), "LLM must return valid output with tool call."
        assert (
            record.stack[-1].extra.tool_inputs is not None
        ), "LLM must provides input for LangChain SQLChecker Tool."
        q = self.checker_prompt.format(
            dialect=self.sql_dialect, query=record.stack[-1].extra.tool_inputs
        )
        out = self._llm.generate(q)
        return (
            "This is the revised SQL: " + out.generated,
            out.prompt_tokens,
            out.completion_tokens,
        )
