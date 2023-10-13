import asyncio
import re

from typing import Mapping, Protocol, Dict, Any, Optional
from pydantic import BaseModel, validate_arguments, Field, validator
from functools import partial
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema.language_model import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.chains import (
    ConversationalRetrievalChain,
    SequentialChain,
    OpenAIModerationChain,
)
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain.chains.question_answering import (
    # map_reduce_prompt,
    refine_prompts,
    stuff_prompt,
    map_rerank_prompt,
)
from langchain.callbacks.manager import AsyncCallbackManagerForChainRun
from content_manager import ContentManager
from chat_history import SqliteChatMessageHistory
from errors import PolicyViolationError


class MemoryHandler(BaseModel):
    db_file: str = "./memorystore/chat_message_history.db"

    @validate_arguments
    def from_session(
        self,
        session_id: str,
        k: int = 5,  # Number of messages to store in buffer.
        return_messages: bool = False,
        llm: Optional[BaseLanguageModel] = None,
        input_key: Optional[str] = None,
        verbose: bool = False,
    ) -> BaseChatMemory:
        chat_history = SqliteChatMessageHistory(
            session_id=session_id, db_file=self.db_file
        )
        if not llm:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, verbose=verbose)

        return ConversationBufferWindowMemory(
            chat_memory=chat_history,
            memory_key="chat_history",
            k=k,
            input_key=input_key,
            return_messages=return_messages,
        )


class Question(BaseModel):
    session_id: str
    message: str

    @validator("session_id")
    def validate_session_id(cls, v):
        pattern = r"^[a-fA-F0-9]{64}$"
        if not re.match(pattern, v):
            raise ValueError(f"{v} does not match the required session-id format")
        return v


class PromptCallable(Protocol):
    def __call__(self, **kwargs: any) -> dict[str, ChatPromptTemplate]:
        pass


class KydenModerationChain(OpenAIModerationChain):
    def _moderate(self, text: str, results: dict) -> str:
        if results["flagged"]:
            error_str = "Text was found that violates our content policy."
            if self.error:
                raise PolicyViolationError(error_str)
            else:
                return error_str
        return text

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        # This is a temporary workaround to make the _acall function
        # asynchronous.
        func = partial(self._call, inputs)
        return await asyncio.get_event_loop().run_in_executor(None, func)


class Conversation:
    prompt_input_key: str = "question"

    @classmethod
    @validate_arguments
    async def chat(
        cls,
        question: Question,
        retriever_search_type: str = "similarity",
        retriever_search_kwargs: dict = Field(default_factory=dict),
        combine_docs_chain_type: str = "stuff",
        verbose: bool = False,
    ) -> dict[str, any]:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, verbose=verbose)
        memory = MemoryHandler().from_session(
            session_id=question.session_id,
            return_messages=True,
            llm=llm,
            input_key=cls.prompt_input_key,
            verbose=verbose,
        )
        # retriever = ContentManager().vectordb.as_retriever(
        #     search_type=retriever_search_type, verbose=verbose
        # )
        retriever = ContentManager().as_self_query_retriever(
            llm=llm,
            search_type=retriever_search_type,
            search_kwargs=retriever_search_kwargs,
            verbose=verbose,
        )
        chat_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            memory=memory,
            chain_type=combine_docs_chain_type,
            combine_docs_chain_kwargs=cls._prompts(chain_type=combine_docs_chain_type),
            verbose=verbose,
        )

        return await chat_chain.acall({cls.prompt_input_key: question.message})

    @classmethod
    @validate_arguments
    async def chat_with_moderation(
        cls,
        question: Question,
        retriever_search_type: str = "similarity",
        retriever_search_kwargs: dict = Field(default_factory=dict),
        combine_docs_chain_type: str = "stuff",
        verbose: bool = False,
    ) -> dict[str, any]:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, verbose=verbose)
        memory = MemoryHandler().from_session(
            session_id=question.session_id,
            return_messages=True,
            llm=llm,
            input_key=cls.prompt_input_key,
            verbose=verbose,
        )

        retriever = ContentManager().as_self_query_retriever(
            llm=llm,
            search_type=retriever_search_type,
            search_kwargs=retriever_search_kwargs,
            verbose=verbose,
        )
        chat_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            memory=memory,
            chain_type=combine_docs_chain_type,
            combine_docs_chain_kwargs=cls._prompts(chain_type=combine_docs_chain_type),
            verbose=verbose,
        )

        moderation_chain = KydenModerationChain(
            error=True, output_key=cls.prompt_input_key
        )
        chain = SequentialChain(
            chains=[moderation_chain, chat_chain], input_variables=["input"]
        )

        return await chain.acall(question.message)

    @classmethod
    def _prompts(cls, chain_type: str = "stuff") -> dict[str, ChatPromptTemplate]:
        prompt_mapping: Mapping[str, PromptCallable] = {
            "stuff": cls._stuff_chain_prompts,
            "map_reduce": cls._map_reduce_chain_prompts,
            "refine": cls._refine_chain_prompts,
            "map_rerank": cls._map_rerank_chain_prompts,
        }
        return prompt_mapping[chain_type]()

    base_system_message: str = """"""

    @classmethod
    def _stuff_chain_prompts(cls) -> dict[str, ChatPromptTemplate]:
        system_template = cls.base_system_message + stuff_prompt.system_template
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        return {"prompt": ChatPromptTemplate.from_messages(messages)}

    @classmethod
    def _map_reduce_chain_prompts(cls) -> dict[str, ChatPromptTemplate]:
        system_template = (
            cls.base_system_message
            + """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
Return any relevant text verbatim.
______________________
{context}"""
        )

        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        CHAT_QUESTION_PROMPT = ChatPromptTemplate.from_messages(messages)

        system_template = (
            cls.base_system_message
            + """Given the following extracted parts of a long document and a question, create a final answer. 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
______________________
{summaries}"""
        )
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        CHAT_COMBINE_PROMPT = ChatPromptTemplate.from_messages(messages)

        return {
            "question_prompt": CHAT_QUESTION_PROMPT,
            "combine_prompt": CHAT_COMBINE_PROMPT,
        }

    @classmethod
    def _refine_chain_prompts(cls) -> dict[str, ChatPromptTemplate]:
        messages = [
            HumanMessagePromptTemplate.from_template("{question}"),
            AIMessagePromptTemplate.from_template("{existing_answer}"),
            HumanMessagePromptTemplate.from_template(refine_prompts.refine_template),
        ]
        CHAT_REFINE_PROMPT = ChatPromptTemplate.from_messages(messages)

        chat_qa_prompt_template = (
            cls.base_system_message + refine_prompts.chat_qa_prompt_template
        )
        messages = [
            SystemMessagePromptTemplate.from_template(chat_qa_prompt_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        CHAT_QUESTION_PROMPT = ChatPromptTemplate.from_messages(messages)

        return {
            "question_prompt": CHAT_QUESTION_PROMPT,
            "refine_prompt": CHAT_REFINE_PROMPT,
        }

    @classmethod
    def _map_rerank_chain_prompts(cls) -> dict[str, ChatPromptTemplate]:
        prompt_template = cls.base_system_message + map_rerank_prompt.prompt_template
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
            output_parser=map_rerank_prompt.output_parser,
        )

        return {"prompt": PROMPT}
