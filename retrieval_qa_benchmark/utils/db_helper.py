from sqlalchemy import MetaData, Column, Table, types, CheckConstraint
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base
from retrieval_qa_benchmark.schema import BaseKnowledgebase


def knowlegde_db_factory(
    engine: Engine, metadata: MetaData, knowledge_base: BaseKnowledgebase
) -> Table:
    table_name = knowledge_base.name
    Base: Table = declarative_base(bind=engine, metadata=metadata)

    class Knowledge(Base):
        __tablename__: str = table_name

        id: Column = Column("id", types.String, nullable=False, primary_key=True)
        context: Column = Column("context", types.String, nullable=False)
        embedding: Column = Column(
            "embedding", types.ARRAY(types.Float), nullable=False
        )
        embedding_dim: Column = Column("embedding_dim", types.Integer, nullable=False)
        title: Column = Column("title", types.String, nullable=True)
        emb_len_check: CheckConstraint = CheckConstraint(
            "emb_len_check", f"length(embedding) = {knowledge_base.emb_model.dim}"
        )

        def __repr__(self) -> str:
            return f"Knowledge(id={self.id!r}, context={self.context[:20]!r}, title={self.title[:20]!r})"

    return Knowledge
