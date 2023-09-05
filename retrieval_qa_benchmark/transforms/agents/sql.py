from typing import Any, List
from sqlalchemy import create_engine, MetaData
from langchain.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain.utilities.sql_database import SQLDatabase
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.agents.mrkl.output_parser import MRKLOutputParser
from langchain.agents.agent_toolkits.sql.prompt import SQL_PREFIX, SQL_SUFFIX
from langchain.schema import AgentAction, AgentFinish, OutputParserException

from retrieval_qa_benchmark.transforms.base import BaseLLMTransform
from retrieval_qa_benchmark.schema import BaseTransform, QARecord, QAPrediction
from retrieval_qa_benchmark.transforms.agents.base import AgentDecider, AgentTool
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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._engine = create_engine(self.url)
        self._metadata = MetaData(bind=self._engine)
        self._db = SQLDatabase(self._engine, None, self._metadata)
        self._tool = None


class LangChainSQLAgentThought(AgentDecider):
    """Agent Decision with LangChain SQL Agent Prompts"""

    sql_dialect: str = "SQL"
    """SQL dialect that helps the LLM understand which SQL its working on"""
    sql_topk: int = 5
    """Maximum retrieved context from database"""
    

    def _build_agent_template(self) -> str:
        tool_strings = "\n".join([f"{name}: {n.descrption}" for name, n in self._tool_map.items()])
        tool_names = ", ".join([tool for tool in tool_desc_map])
        format_instructions = FORMAT_INSTRUCTIONS.format(tool_names=tool_names)
        return "\n\n".join([SQL_PREFIX, tool_strings, format_instructions, SQL_SUFFIX])

    def _format_agent_template(self, q:str, stacked: List[str]) -> str:
        return self.agent_template.format(dialect=self.sql_dialect, 
                                       top_k=self.sql_topk,
                                       input=q,
                                       agent_scratchpad='\n'.join(stacked))
        

@REGISTRY.register_transform("langchain-listsql")
class LangChainListSQLDB(LangChainSQLAgentTool):
    
    name: str = LIST_SQL_DATABASE_NAME
    descrption: str = list_sql_database_tool_description
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tool = ListSQLDatabaseTool(
            name=LIST_SQL_DATABASE_NAME,
            db=self._db,
            description=list_sql_database_tool_description,
        )


@REGISTRY.register_transform("langchain-infosql")
class LangChainInfoSQLDB(LangChainSQLAgentTool):
    name:str = INFO_SQL_DATABASE_NAME
    descrption: str = info_sql_database_tool_description
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tool = InfoSQLDatabaseTool(
            name=INFO_SQL_DATABASE_NAME,
            db=self._db,
            description=info_sql_database_tool_description,
        )


@REGISTRY.register_transform("langchain-querysql")
class LangChainQuerySQLDB(LangChainSQLAgentTool):
    name: str = QUERY_SQL_DATABASE_NAME
    descrption: str = query_sql_database_tool_description
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tool = QuerySQLDataBaseTool(
            name=QUERY_SQL_DATABASE_NAME,
            db=self._db,
            description=query_sql_database_tool_description,
        )


@REGISTRY.register_transform("langchain-sqlchecker")
class LangChainSQLChecker(LangChainSQLAgentTool, BaseLLMTransform):
    name: str = QUERY_CHECKER_SQL_NAME
    descrption: str = query_sql_checker_tool_description
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # self._tool = QuerySQLCheckerTool(
        #     db=self._db, llm=self._llm, description=query_sql_checker_tool_description
        # )
