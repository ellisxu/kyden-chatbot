from langchain.memory import SQLiteEntityStore


class SQLiteEntityStore_Bugfix(SQLiteEntityStore):
    session_id: str = "default"
    table_name: str = "memory_store"

    """Temporarily fix an issue (https://github.com/langchain-ai/langchain/issues/6091)
    by adding 'conn: object = None'
    """
    conn: object = None
