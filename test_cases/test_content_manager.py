import pytest
from content_manager import ContentManager, FileForEmbedding, ContentLoader
from test_cases.toolkits import delete_file_and_dir
import os
import zipfile
import json
import uuid
import openai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # Read the local .env file

openai.api_key = os.environ["OPENAI_API_KEY"]


ORIGINAL_CONTENT_PATH_GENERAL = "./test_cases/materials/general/"
ORIGINAL_CONTENT_PATH_SPECIFICS = "./test_cases/materials/specifics/"
PERSIST_DIRECTORY = "./test_cases/vectorstore/chroma/"
PERSIST_DIRECTORY_FOR_DEL = "./test_cases/vectorstore/chroma_for_del/"
COLLECTION_NAME = "content-manager-test"


# def delete_file_and_dir(directory_path):
#     for filename in os.listdir(directory_path):
#         file_path = os.path.join(directory_path, filename)

#         try:
#             if os.path.isfile(file_path) or os.path.islink(file_path):
#                 os.unlink(file_path)
#             elif os.path.isdir(file_path):
#                 shutil.rmtree(file_path)
#         except Exception as e:
#             print(f"Failed to delete {file_path}. Reason: {e}")


@pytest.fixture(scope="module")
def manager() -> ContentManager:
    manager = ContentManager(
        original_content_path=ORIGINAL_CONTENT_PATH_GENERAL,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
    )
    yield manager
    delete_file_and_dir(PERSIST_DIRECTORY)


def test_traverse_original_content(manager: ContentManager):
    all_files = manager._traverse_original_content(ORIGINAL_CONTENT_PATH_GENERAL)

    assert all_files
    assert len(all_files) == 3

    all_file_paths = [item.file for item in all_files]
    assert f"{ORIGINAL_CONTENT_PATH_GENERAL}content01.md" in all_file_paths
    assert f"{ORIGINAL_CONTENT_PATH_GENERAL}content02.md" in all_file_paths
    assert f"{ORIGINAL_CONTENT_PATH_GENERAL}content03.md" in all_file_paths


@pytest.mark.skip(reason="Reduce OpenAI usage.")
def test_embedding():
    manager = ContentManager(
        original_content_path=ORIGINAL_CONTENT_PATH_GENERAL,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=f"{COLLECTION_NAME}-embedding-func",
    )
    all_files = manager._traverse_original_content(ORIGINAL_CONTENT_PATH_GENERAL)
    results = manager._embedding(all_files=all_files)
    assert len(all_files) == len(results)
    assert 9 == manager.vectordb._collection.count()

    for item in results:
        if item.file == f"{ORIGINAL_CONTENT_PATH_GENERAL}content01.md":
            assert len(item.IDs) == 1
        elif item.file == f"{ORIGINAL_CONTENT_PATH_GENERAL}content02.md":
            assert len(item.IDs) == 1
        elif item.file == f"{ORIGINAL_CONTENT_PATH_GENERAL}content03.md":
            assert len(item.IDs) == 7


@pytest.fixture(scope="function")
def manager_for_del() -> ContentManager:
    zip_path = f"{PERSIST_DIRECTORY_FOR_DEL}/../for_del_test.zip"

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(PERSIST_DIRECTORY_FOR_DEL)

    manager = ContentManager(
        original_content_path=ORIGINAL_CONTENT_PATH_GENERAL,
        persist_directory=PERSIST_DIRECTORY_FOR_DEL,
        collection_name=f"{COLLECTION_NAME}-delete-embedding-func",
    )
    yield manager
    delete_file_and_dir(PERSIST_DIRECTORY_FOR_DEL)


def test_delete_embedding(manager_for_del: ContentManager):
    data_list = [
        {
            "file": "./test_cases/materials/content01.md",
            "update_time": "2023-05-23T14:57:07.322Z",
            "IDs": ["47273bd4-3a75-11ee-938b-8c8590ad4c67"],
            "is_valid": True,
        },
        {
            "file": "./test_cases/materials/content03.md",
            "update_time": "2023-05-23T14:57:07.322Z",
            "IDs": [
                "47b17092-3a75-11ee-938b-8c8590ad4c67",
                "47b17146-3a75-11ee-938b-8c8590ad4c67",
                "47b17182-3a75-11ee-938b-8c8590ad4c67",
                "47b171b4-3a75-11ee-938b-8c8590ad4c67",
                "47b171dc-3a75-11ee-938b-8c8590ad4c67",
                "47b1720e-3a75-11ee-938b-8c8590ad4c67",
                "47b17236-3a75-11ee-938b-8c8590ad4c67",
            ],
            "is_valid": True,
        },
        {
            "file": "./test_cases/materials/content02.md",
            "update_time": "2023-05-23T14:57:07.322Z",
            "IDs": [
                "485589de-3a75-11ee-938b-8c8590ad4c67",
            ],
            "is_valid": True,
        },
    ]
    all_files = [FileForEmbedding.from_dict(data) for data in data_list]
    deleted_files = manager_for_del._delete_embedding(all_files=all_files[0:1])
    assert 8 == manager_for_del.vectordb._collection.count()
    for item in deleted_files:
        assert 0 == len(item.IDs)

    deleted_files = manager_for_del._delete_embedding(all_files=all_files[1:2])
    assert 1 == manager_for_del.vectordb._collection.count()
    for item in deleted_files:
        assert 0 == len(item.IDs)

    deleted_files = manager_for_del._delete_embedding(all_files=all_files[2:])
    assert 0 == manager_for_del.vectordb._collection.count()
    for item in deleted_files:
        assert 0 == len(item.IDs)


@pytest.fixture(scope="function")
def manager_for_trigger() -> ContentManager:
    class ContentManagerForTesting(ContentManager):
        class Config:
            extra = "allow"
            allow_mutation = True
            arbitrary_types_allowed = True

    manager = ContentManagerForTesting(
        original_content_path=ORIGINAL_CONTENT_PATH_SPECIFICS,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
    )
    # manager = ContentManager(
    #     original_content_path=ORIGINAL_CONTENT_PATH_SPECIFICS,
    #     persist_directory=PERSIST_DIRECTORY,
    #     collection_name=COLLECTION_NAME,
    # )
    yield manager
    # delete_file_and_dir(ORIGINAL_CONTENT_PATH_SPECIFICS)


def test_trigger_embedding(manager_for_trigger: ContentManager):
    def _embedding(self, all_files: list[FileForEmbedding]) -> list[FileForEmbedding]:
        for item in all_files:
            splits = self.splitter.split_documents(
                ContentLoader(file_path=item.file).load()
            )
            item.IDs = [str(uuid.uuid1()) for _ in splits]

        return all_files

    def _delete_embedding(
        self, all_files: list[FileForEmbedding]
    ) -> list[FileForEmbedding]:
        for item in all_files:
            item.IDs.clear()

        return all_files

    manager_for_trigger._embedding = _embedding.__get__(
        manager_for_trigger, ContentManager
    )
    manager_for_trigger._delete_embedding = _delete_embedding.__get__(
        manager_for_trigger, ContentManager
    )

    def execute(mock_file: str):
        zip_path = f"{ORIGINAL_CONTENT_PATH_SPECIFICS}../{mock_file}"
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(ORIGINAL_CONTENT_PATH_SPECIFICS)
        manager_for_trigger.trigger_embedding()
        yield "start"
        delete_file_and_dir(ORIGINAL_CONTENT_PATH_SPECIFICS)
        yield "end"

    BEGIN = END = next

    def get_embedding_list() -> list[FileForEmbedding]:
        assert os.path.exists(
            f"{ORIGINAL_CONTENT_PATH_SPECIFICS}{manager_for_trigger.embedding_record_file}"
        )

        embedding_list = []
        with open(
            f"{manager_for_trigger.original_content_path}{manager_for_trigger.embedding_record_file}"
        ) as file:
            data_list = json.load(file)
            embedding_list = [FileForEmbedding.from_dict(item) for item in data_list]

        return embedding_list

    # Test 01: The first time to add only valid new docs
    test_01 = execute(mock_file="test_trigger_embedding-01.zip")
    BEGIN(test_01)
    embedding_list = get_embedding_list()

    assert 3 == len(embedding_list)
    for item in embedding_list:
        if item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content01.md":
            assert len(item.IDs) == 1
            assert item.is_valid
        elif item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content02.md":
            assert len(item.IDs) == 1
            assert item.is_valid
        elif item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content03.md":
            assert len(item.IDs) == 7
            assert item.is_valid
    END(test_01)

    # Test 02: The first time to add both valid and invalid new docs
    test_02 = execute(mock_file="test_trigger_embedding-02.zip")
    BEGIN(test_02)
    embedding_list = get_embedding_list()

    assert 4 == len(embedding_list)
    for item in embedding_list:
        if item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content01.md":
            assert len(item.IDs) == 1
            assert item.is_valid
        elif item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content02.md":
            assert len(item.IDs) == 1
            assert item.is_valid
        elif item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content03.md":
            assert len(item.IDs) == 7
            assert item.is_valid
        elif item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content04.md":
            assert len(item.IDs) == 0
            assert not item.is_valid
    END(test_02)

    # Test 03: Add new docs when there are already some docs
    test_03 = execute(mock_file="test_trigger_embedding-03.zip")
    BEGIN(test_03)
    embedding_list = get_embedding_list()

    assert 6 == len(embedding_list)
    for item in embedding_list:
        if item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content01.md":
            assert len(item.IDs) == 1
            assert item.is_valid
        elif item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content02.md":
            assert len(item.IDs) == 1
            assert item.is_valid
        elif item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content03.md":
            assert len(item.IDs) == 7
            assert item.is_valid
        elif item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content04.md":
            assert len(item.IDs) == 0
            assert not item.is_valid
        elif item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content05.md":
            assert len(item.IDs) == 7
            assert item.is_valid
        elif item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content06.md":
            assert len(item.IDs) == 0
            assert not item.is_valid
    END(test_03)

    # Test 04: Update docs
    test_04 = execute(mock_file="test_trigger_embedding-04.zip")
    BEGIN(test_04)
    embedding_list = get_embedding_list()

    assert 4 == len(embedding_list)
    for item in embedding_list:
        if item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content01.md":
            assert len(item.IDs) == 1
            assert item.is_valid
        elif item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content02.md":
            assert len(item.IDs) == 1
            assert item.is_valid
        elif item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content03.md":
            assert len(item.IDs) == 3
            assert item.is_valid
        elif item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content04.md":
            assert len(item.IDs) == 7
            assert item.is_valid
    END(test_04)

    # Test 05: Delete docs (by changing their in_valid fields)
    test_05 = execute(mock_file="test_trigger_embedding-05.zip")
    BEGIN(test_05)
    embedding_list = get_embedding_list()

    assert 4 == len(embedding_list)
    for item in embedding_list:
        if item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content01.md":
            assert len(item.IDs) == 1
            assert item.is_valid
        elif item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content02.md":
            assert len(item.IDs) == 1
            assert item.is_valid
        elif item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content03.md":
            assert len(item.IDs) == 0
            assert not item.is_valid
        elif item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content04.md":
            assert len(item.IDs) == 0
            assert not item.is_valid
    END(test_05)

    # Test 06: Add, update, and delete docs at the same time
    test_06 = execute(mock_file="test_trigger_embedding-06.zip")
    BEGIN(test_06)
    embedding_list = get_embedding_list()

    assert 5 == len(embedding_list)
    for item in embedding_list:
        if item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content01.md":
            assert len(item.IDs) == 0
            assert not item.is_valid
        elif item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content02.md":
            assert len(item.IDs) == 1
            assert item.is_valid
        elif item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content03.md":
            assert len(item.IDs) == 3
            assert item.is_valid
        elif item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content04.md":
            assert len(item.IDs) == 7
            assert item.is_valid
        elif item.file == f"{ORIGINAL_CONTENT_PATH_SPECIFICS}content05.md":
            assert len(item.IDs) == 2
            assert item.is_valid
    END(test_06)
