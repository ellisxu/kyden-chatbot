from content_loader import ContentLoader
import pytest

# from datetime import date

CONTENT_01_PATH = "./test_cases/materials/general/content01.md"


@pytest.fixture
def content_01_text() -> str:
    with open(CONTENT_01_PATH, "r") as f:
        text = f.read()
        return text


def test_parse_front_matter(content_01_text):
    loader = ContentLoader("")
    metadata = loader.parse_front_matter(content_01_text)

    assert metadata is not None
    assert len(metadata) == 5
    assert metadata["title"] == "My Blog Post"
    assert metadata["date"] == "2023-05-23T14:57:07.322Z"
    assert metadata["author"] == "John Doe"
    assert metadata["isValid"] == 1


def test_load():
    loader = ContentLoader(CONTENT_01_PATH)
    docs = loader.load()

    assert docs
    assert len(docs) == 1

    page_content = docs[0].page_content
    metadata = docs[0].metadata

    assert (
        page_content
        == f"""---
category: ''
title: 'My Blog Post'
date: '2023-05-23T14:57:07.322Z'
author: 'John Doe'
isValid: 1
---

Here's the content of the blog post."""
    )
    assert len(metadata) == 6
    assert metadata["source"] == "./test_cases/materials/general/content01.md"
    assert metadata["title"] == "My Blog Post"
    assert metadata["date"] == "2023-05-23T14:57:07.322Z"
    assert metadata["author"] == "John Doe"
    assert metadata["isValid"] == 1
