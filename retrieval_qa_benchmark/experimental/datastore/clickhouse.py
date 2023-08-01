from __future__ import annotations

from typing import List, Optional

from sqlalchemy import MetaData, Table, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from retrieval_qa_benchmark.schema import (
    BaseDataStore,
    BaseKnowledgebase,
    KnowledgeRecord,
)
from retrieval_qa_benchmark.experimental.datastore.db_helper import knowlegde_db_factory


class ClickhouseDatastore(BaseDataStore):
    client: Engine
    metadata: MetaData
    table: Table

    @classmethod
    def build(
        cls,
        knowledge_base: BaseKnowledgebase,
        host: str = "localhost",
        port: int = 8123,
        user: Optional[str] = None,
        password: Optional[str] = None,
        secure: bool = False,
    ) -> ClickhouseDatastore:
        conn_str = "clickhouse://"
        cred_str = ""
        if user:
            cred_str += user
            if password:
                cred_str += f":{password}"
        if cred_str != "":
            cred_str += "@"
        conn_str += cred_str + host
        if port:
            conn_str += f":{str(port)}"
        conn_str += "/default?protocol=http"
        if secure:
            conn_str += "s"
        client = create_engine(conn_str)
        metadata = MetaData(bind=client)
        table = knowlegde_db_factory(client, metadata, knowledge_base)
        metadata.create_all(client)
        return cls(client=client, metadata=metadata, table=table)

    def insert(self, data: List[KnowledgeRecord]) -> None:
        with Session(self.client) as session:
            knowledge = [self.table(**d.model_dump()) for d in data]
            session.add_all(knowledge)
            session.commit()

    def search(self, hint: str, k: int) -> List[KnowledgeRecord]:
        with self.client.begin():
            # connection.execute(text(command))
            return []
