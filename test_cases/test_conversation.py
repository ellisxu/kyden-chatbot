import pytest
import asyncio
from conversation import MemoryHandler, Question, Conversation
from test_cases.toolkits import delete_file_and_dir
from langchain.memory import ConversationEntityMemory
import os
import openai
import uuid
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # Read the local .env file

openai.api_key = os.environ["OPENAI_API_KEY"]

MEMORY_DB_FILE_DIR = "./test_cases/memorystore/sqlite/"
MEMORY_DB_FILE_NAME = "chat_message_history.db"


def test_MemoryHandler_from_session():
    handler = MemoryHandler(db_file=MEMORY_DB_FILE_DIR + MEMORY_DB_FILE_NAME)
    memory_1 = handler.from_session(
        session_id=str(uuid.uuid1()), k=2, return_messages=True
    )
    memory_2 = handler.from_session(
        session_id=str(uuid.uuid1()), k=4, return_messages=True
    )

    memory_1.save_context({"input": "hi"}, {"output": "whats up"})
    memory_1.save_context({"input": "not much you"}, {"output": "not much"})
    memory_1.save_context({"input": "La La La"}, {"output": "Hah Hah Hah"})

    memory_2.save_context({"input": "hi"}, {"output": "whats up"})
    memory_2.save_context({"input": "not much you"}, {"output": "not much"})
    memory_2.save_context({"input": "La La La"}, {"output": "Hah Hah Hah"})

    record_1 = memory_1.load_memory_variables({})
    record_2 = memory_2.load_memory_variables({})
    
    assert 4 == len(record_1.get("chat_history"))
    assert 6 == len(record_2.get("chat_history"))
 
    delete_file_and_dir(directory_path=MEMORY_DB_FILE_DIR)


BASE_SYSTEM_MESSAGE = """"""
STUFF_PROMPTS = [
    {
        "key": "prompt",
        "variables": {
            "context": "This is a context for testing.",
            "question": "This is a question.",
        },
        "result": f"""System: {BASE_SYSTEM_MESSAGE}Use the following pieces of context to answer the users question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
This is a context for testing.
Human: This is a question.""",
    }
]

MAP_REDUCE_PROMPTS = [
    {
        "key": "question_prompt",
        "variables": {
            "context": "This is a context for testing.",
            "question": "This is a question.",
        },
        "result": f"""System: {BASE_SYSTEM_MESSAGE}Use the following portion of a long document to see if any of the text is relevant to answer the question. 
Return any relevant text verbatim.
______________________
This is a context for testing.
Human: This is a question.""",
    },
    {
        "key": "combine_prompt",
        "variables": {
            "summaries": "Here are the summaries for testing.",
            "question": "This is a question.",
        },
        "result": f"""System: {BASE_SYSTEM_MESSAGE}Given the following extracted parts of a long document and a question, create a final answer. 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
______________________
Here are the summaries for testing.
Human: This is a question.""",
    },
]

REFINE_CHAIN_PROMPTS = [
    {
        "key": "question_prompt",
        "variables": {
            "context_str": "This is a context for testing.",
            "question": "This is a question.",
        },
        "result": f"""System: {BASE_SYSTEM_MESSAGE}Context information is below.
------------
This is a context for testing.
------------
Given the context information and not prior knowledge, answer any questions
Human: This is a question.""",
    },
    {
        "key": "refine_prompt",
        "variables": {
            "context_str": "This is a context for testing.",
            "question": "This is a question.",
            "existing_answer": "This is the existing answer.",
        },
        "result": f"""Human: This is a question.
AI: This is the existing answer.
Human: We have the opportunity to refine the existing answer (only if needed) with some more context below.
------------
This is a context for testing.
------------
Given the new context, refine the original answer to better answer the question. If the context isn't useful, return the original answer.""",
    },
]

MAP_RERANK_PROMPTS = [
    {
        "key": "prompt",
        "variables": {
            "context": "This is a context for testing.",
            "question": "This is a question.",
        },
        "result": f"""{BASE_SYSTEM_MESSAGE}Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

In addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:

Question: [question here]
Helpful Answer: [answer here]
Score: [score between 0 and 100]

How to determine the score:
- Higher is a better answer
- Better responds fully to the asked question, with sufficient level of detail
- If you do not know the answer based on the context, that should be a score of 0
- Don't be overconfident!

Example #1

Context:
---------
Apples are red
---------
Question: what color are apples?
Helpful Answer: red
Score: 100

Example #2

Context:
---------
it was night and the witness forgot his glasses. he was not sure if it was a sports car or an suv
---------
Question: what type was the car?
Helpful Answer: a sports car or an suv
Score: 60

Example #3

Context:
---------
Pears are either red or orange
---------
Question: what color are apples?
Helpful Answer: This document does not answer the question
Score: 0

Begin!

Context:
---------
This is a context for testing.
---------
Question: This is a question.
Helpful Answer:""",
    }
]


def test_Conversation_prompts():
    prompts = Conversation._prompts(chain_type="stuff")
    assert 1 == len(prompts)
    for item in STUFF_PROMPTS:
        assert prompts[item["key"]]
        assert item["result"] == prompts[item["key"]].format(**item["variables"])

    prompts = Conversation._prompts(chain_type="map_reduce")
    assert 2 == len(prompts)
    for item in MAP_REDUCE_PROMPTS:
        assert prompts[item["key"]]
        assert item["result"] == prompts[item["key"]].format(**item["variables"])

    prompts = Conversation._prompts(chain_type="refine")
    assert 2 == len(prompts)
    for item in REFINE_CHAIN_PROMPTS:
        assert prompts[item["key"]]
        assert item["result"] == prompts[item["key"]].format(**item["variables"])

    prompts = Conversation._prompts(chain_type="map_rerank")
    assert 1 == len(prompts)
    for item in MAP_RERANK_PROMPTS:
        assert prompts[item["key"]]
        assert item["result"] == prompts[item["key"]].format(**item["variables"])
