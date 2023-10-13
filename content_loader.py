from typing import List
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
import yaml

class ContentLoader(TextLoader):
    def __init__(self, file_path: str, encoding: str | None = None, autodetect_encoding: bool = False):
        super().__init__(file_path, encoding, autodetect_encoding)

    
    def load(self) -> List[Document]:
        """Load a doc file, then parse and use its YFM(YAML Front Matter) as metadata
        """
        docs = super().load()
        text = docs[0].page_content
        metadata = docs[0].metadata
        
        if (text_metadata := self.parse_front_matter(text)):
            metadata.update(text_metadata)

        return [Document(page_content=text, metadata=metadata)]
    
    def parse_front_matter(self, content: str):
        # Check if the content starts and has at least one '---' separator
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                front_matter = yaml.safe_load(parts[1])
                return front_matter
        return None

