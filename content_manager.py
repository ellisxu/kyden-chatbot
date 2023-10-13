import os
import fnmatch
import json
from pydantic import BaseModel, Field
from typing import List, Optional, Any, cast

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.schema import BaseRetriever, Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.callbacks.manager import AsyncCallbackManagerForRetrieverRun
from langchain.chains.query_constructor.ir import StructuredQuery
from content_loader import ContentLoader
from datetime import datetime, timezone


class AsyncSelfQueryRetriever(SelfQueryRetriever):
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        inputs = self.llm_chain.prep_inputs({"query": query})

        structured_query = cast(
            StructuredQuery,
            self.llm_chain.prompt.output_parser.parse(
                await self.llm_chain.apredict(
                    callbacks=run_manager.get_child(), **inputs
                )
            ),
        )
        if self.verbose:
            print(structured_query)
        new_query, new_kwargs = self.structured_query_translator.visit_structured_query(
            structured_query
        )
        if structured_query.limit is not None:
            new_kwargs["k"] = structured_query.limit

        if self.use_original_query:
            new_query = query

        search_kwargs = {**self.search_kwargs, **new_kwargs}
        docs = await self.vectorstore.asearch(
            new_query, self.search_type, **search_kwargs
        )
        return docs


class FileForEmbedding(BaseModel):
    """Specifies the required metadata when using ContentManager to manage embedding."""

    file: str
    update_time: datetime
    IDs: List[str]
    is_valid: bool = True

    @classmethod
    def from_dict(cls, data):
        update_time = datetime.fromisoformat(data["update_time"].rstrip("Z")).replace(
            tzinfo=timezone.utc
        )
        return cls(
            file=data["file"],
            update_time=update_time,
            IDs=data["IDs"],
            is_valid=data["is_valid"],
        )

    def to_dict(self):
        return {
            "file": self.file,
            "update_time": self.update_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "IDs": self.IDs,
            "is_valid": self.is_valid,
        }


class ContentManager(BaseModel):
    """A class used to manage embedding data.
    Place markdown docs to the path specified by the "original_content_path" field,
    ContentManager will traverse all the docs and handle embedding, adding, updating, and deletion properly.
    """

    class Config:
        arbitrary_types_allowed = True

    original_content_path: str = "./original_content/"
    embedding_record_file: str = "embedding.json"
    persist_directory: str = "./vectorstore/chroma/"
    collection_name: str = "kyden-chatbot"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    separators: List[str] = ["\n\n", "\n", "(?<=\\. )", " ", ""]
    embedding: Optional[OpenAIEmbeddings] = None
    vectordb: Optional[Chroma] = None
    splitter: Optional[TextSplitter] = None

    def __init__(
        self,
        original_content_path: Optional[str] = None,
        embedding_record_file: Optional[str] = None,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        if original_content_path:
            self.original_content_path = original_content_path
        if embedding_record_file:
            self.embedding_record_file = embedding_record_file
        if persist_directory:
            self.persist_directory = persist_directory
        if collection_name:
            self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if separators:
            self.separators = separators

        self.embedding = OpenAIEmbeddings()
        self.vectordb = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding,
            persist_directory=self.persist_directory,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    def trigger_embedding(self):
        """Traverse the docs in the directory specified by the "original_content_path" field,
        compare them with the records in the embedding.json file to determine proper operations (embedding, adding, updating, deleting),
        then evoke those operations.
        """
        all_files = self._traverse_original_content(self.original_content_path)
        embedding_dict: dict[str, FileForEmbedding] = {}

        if os.path.exists(f"{self.original_content_path}{self.embedding_record_file}"):
            with open(
                f"{self.original_content_path}{self.embedding_record_file}", "r"
            ) as file:
                data_list = json.load(file)
                for item in data_list:
                    obj = FileForEmbedding.from_dict(item)
                    embedding_dict[obj.file] = obj

        deleting_list: list[FileForEmbedding] = []
        adding_list: list[FileForEmbedding] = []
        for item in all_files:
            if item.file in embedding_dict:
                record = embedding_dict[item.file]
                item.IDs = record.IDs

                if not item.is_valid and record.is_valid:
                    deleting_list.append(item)
                elif not record.is_valid and item.is_valid:
                    adding_list.append(item)
                elif item.update_time > record.update_time:
                    deleting_list.append(item)
                    adding_list.append(item)

            elif item.is_valid:
                adding_list.append(item)

        self._delete_embedding(all_files=deleting_list)
        self._embedding(all_files=adding_list)

        with open(
            f"{self.original_content_path}{self.embedding_record_file}", "w"
        ) as file:
            file.write(json.dumps([item.to_dict() for item in all_files], indent=4))

    def _embedding(self, all_files: list[FileForEmbedding]) -> list[FileForEmbedding]:
        try:
            for item in all_files:
                splits = self.splitter.split_documents(
                    ContentLoader(file_path=item.file).load()
                )
                IDs = self.vectordb.add_documents(documents=splits)
                item.IDs = IDs
        finally:
            # Maintain backwards compatibility with chromadb < 0.4.0
            self.vectordb.persist()

        return all_files

    def _delete_embedding(
        self, all_files: list[FileForEmbedding]
    ) -> list[FileForEmbedding]:
        try:
            ids: list[str] = []
            for item in all_files:
                ids.extend(item.IDs)
                item.IDs.clear()

            self.vectordb.delete(ids=ids)
        finally:
            # Maintain backwards compatibility with chromadb < 0.4.0
            self.vectordb.persist()

        return all_files

    def _traverse_original_content(self, path) -> list[FileForEmbedding]:
        all_files: list[FileForEmbedding] = []
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in fnmatch.filter(filenames, "*.md"):
                file_path = os.path.join(dirpath, filename)
                metadata = ContentLoader(file_path=file_path).load()[0].metadata

                update_time = datetime.fromisoformat(
                    metadata["date"].rstrip("Z")
                ).replace(tzinfo=timezone.utc)
                is_valid = True if metadata.get("isValid") == 1 else False

                all_files.append(
                    FileForEmbedding(
                        file=file_path,
                        update_time=update_time,
                        IDs=[],
                        is_valid=is_valid,
                    )
                )

        return all_files

    def as_self_query_retriever(
        self,
        llm: BaseLanguageModel,
        search_type: str = "similarity",
        search_kwargs: dict = Field(default_factory=dict),
        verbose: bool = False,
    ) -> BaseRetriever:
        metadata_field_info = [
            AttributeInfo(
                name="category",
                description="""The category of the content. There are multiple categories:
1. Web Page Content - The content within this category encompasses all the elements typically found on the portfolio website, such as an introduction, the owner's profile, a sitemap, and the contents of each page.
2. Article - Tech articles.
3. Project - Tech project summaries.
4. Youtube Video Subtitles - Subtitles from relevant Youtube videos""",
                type="string",
            ),
            AttributeInfo(
                name="title",
                description="""The title of the content.""",
                type="string",
            ),
            AttributeInfo(
                name="description",
                description="""The brief description of the content.""",
                type="string",
            ),
            AttributeInfo(
                name="date",
                description="""The update time of the content. The time strings are in the ISO 8601 format, specifically using the structure 'YYYY-MM-DDTHH:MM:SS.mmmZ', where:
- YYYY is the 4-digit year.
- MM is the 2-digit month.
- DD is the 2-digit day of the month.
- T is a literal separator between the date and time.
- HH is the 2-digit hour using a 24-hour clock.
- MM is the 2-digit minutes.
- SS is the 2-digit seconds.
- mmm is the 3-digit milliseconds.
- Z indicates the UTC timezone.""",
                type="string",
            ),
            AttributeInfo(
                name="author",
                description="""The author of the content.""",
                type="string",
            ),
            AttributeInfo(
                name="github",
                description="""The corresponding Github address of the content.""",
                type="string",
            ),
            AttributeInfo(
                name="link",
                description="""The corresponding Youtube link of the content.""",
                type="string",
            ),
        ]
        document_content_description = """All documents are from Kyden Hsui's portfolio website. The contents can be divided into multiple categories:
1. Web Page Content - The content within this category encompasses all the elements typically found on the portfolio website, such as an introduction, the owner's profile, a sitemap, and the contents of each page.
2. Article - Tech articles.
3. Project - Tech project summaries.
4. Youtube Video Subtitles - Subtitles from relevant Youtube videos"""

        return AsyncSelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=self.vectordb,
            document_contents=document_content_description,
            metadata_field_info=metadata_field_info,
            verbose=verbose,
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
