from langchain.memory import ChatMessageHistory
from chat_history import SqliteChatMessageHistory
import sqlite3
from test_cases.toolkits import delete_file_and_dir
import pytest
import uuid


def table_exists(conn: sqlite3.Connection, table_name: str):
    cursor = conn.cursor()
    cursor.execute(
        """ SELECT count(name) FROM sqlite_master WHERE type='table' AND name=? """,
        (table_name,),
    )

    # If the count is 1, then table exists
    if cursor.fetchone()[0] == 1:
        return True
    return False


MEMORY_DB_FILE_DIR = "./test_cases/memorystore/sqlite/"
MEMORY_DB_FILE_NAME = "chat_message_history.db"
FAKE_SESSION_ID = str(uuid.uuid1())


@pytest.fixture(scope="module")
def chatHistory() -> SqliteChatMessageHistory:
    chatHistory = SqliteChatMessageHistory(
        session_id=FAKE_SESSION_ID, db_file=MEMORY_DB_FILE_DIR + MEMORY_DB_FILE_NAME
    )
    yield chatHistory
    delete_file_and_dir(MEMORY_DB_FILE_DIR)


def test_SqliteChatMessageHistory(chatHistory: SqliteChatMessageHistory):
    assert chatHistory is not None
    assert chatHistory.conn is not None
    assert chatHistory.session_id == FAKE_SESSION_ID
    assert table_exists(conn=chatHistory.conn, table_name=chatHistory.table_name)

    human_message = "Hi!"
    ai_message = "What's up?"
    chatHistory.add_user_message(human_message)
    chatHistory.add_ai_message(ai_message)

    messages = chatHistory.messages
    assert 2 == len(messages)
    assert "human" == messages[0].type
    assert human_message == messages[0].content
    assert "ai" == messages[1].type
    assert ai_message == messages[1].content

    chatHistory_2 = SqliteChatMessageHistory(
        session_id=FAKE_SESSION_ID, db_file=MEMORY_DB_FILE_DIR + MEMORY_DB_FILE_NAME
    )
    messages = chatHistory_2.messages
    assert 2 == len(messages)
    assert "human" == messages[0].type
    assert human_message == messages[0].content
    assert "ai" == messages[1].type
    assert ai_message == messages[1].content

    new_session_id = str(uuid.uuid1())
    chatHistory_3 = SqliteChatMessageHistory(
        session_id=new_session_id, db_file=MEMORY_DB_FILE_DIR + MEMORY_DB_FILE_NAME
    )
    messages = chatHistory_3.messages
    assert 0 == len(messages)

    human_message = "Hello! Nice to meet you."
    ai_message = "Hi! Glad to see you too."
    chatHistory_3.add_user_message(human_message)
    chatHistory_3.add_ai_message(ai_message)

    messages = chatHistory_3.messages
    assert 2 == len(messages)
    assert "human" == messages[0].type
    assert human_message == messages[0].content
    assert "ai" == messages[1].type
    assert ai_message == messages[1].content

    human_message_2 = "How you doing."
    ai_message_2 = "Very good. Thanks for asking."
    chatHistory_3.add_user_message(human_message_2)
    chatHistory_3.add_ai_message(ai_message_2)

    messages = chatHistory_3.messages
    assert 4 == len(messages)
    assert "human" == messages[0].type
    assert human_message == messages[0].content
    assert "ai" == messages[1].type
    assert ai_message == messages[1].content
    assert "human" == messages[2].type
    assert human_message_2 == messages[2].content
    assert "ai" == messages[3].type
    assert ai_message_2 == messages[3].content

    chatHistory_3.clear()
    assert 0 == len(chatHistory_3.messages)
    assert 2 == len(chatHistory.messages)

    chatHistory.clear()
    assert 0 == len(chatHistory.messages)
    assert 0 == len(chatHistory_2.messages)
