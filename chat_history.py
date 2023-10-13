from typing import List, Any, Optional

from langchain.pydantic_v1 import BaseModel, Field, Required
from langchain.schema import (
    BaseChatMessageHistory,
)
from langchain.schema.messages import BaseMessage, _message_to_dict, messages_from_dict
import sqlite3
import json


class SqliteChatMessageHistory(BaseChatMessageHistory, BaseModel):
    session_id: str = "default"
    table_name: str = "memory_store"
    conn: sqlite3.Connection = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        session_id: str,
        db_file: str = "./memorystore/chat_message_history.db",
        table_name: str = "memory_store",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.conn = sqlite3.connect(db_file)
        self.session_id = session_id
        self.table_name = table_name

        self._create_table_if_not_exists()

    @property
    def messages(self):
        fetch_messages = f"""
            SELECT message FROM {self.table_name} WHERE session_id = ?
        """
        with self.conn:
            cursor = self.conn.execute(fetch_messages, (self.session_id,))
            records = cursor.fetchall()
            items = [json.loads(record[0]) for record in records]
            messages = messages_from_dict(items)
            return messages

    def add_message(self, message: BaseMessage) -> None:
        add_message = f"""
            INSERT INTO {self.table_name} (session_id, message) VALUES (?, ?)
        """
        jsonstr = json.dumps(_message_to_dict(message))
        with self.conn:
            self.conn.execute(add_message, (self.session_id, jsonstr))

    def clear(self):
        """Clear session memory from db"""
        clear_message = f"""
          DELETE FROM {self.table_name}
          WHERE session_id = ?
        """
        with self.conn:
            self.conn.execute(clear_message, (self.session_id,))

    def _create_table_if_not_exists(self) -> None:
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message TEXT,
                updated_time TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """

        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(create_table_query)
            # cursor.execute(create_update_time_trigger)
