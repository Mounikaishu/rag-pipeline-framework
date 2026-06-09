from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from config.settings import Settings


class Chunker:

    def __init__(self):

        self.text_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=
                Settings.CHUNK_SIZE,

                chunk_overlap=
                Settings.CHUNK_OVERLAP
            )
        )

    def split_documents(
        self,
        documents
    ):
        return self.text_splitter.split_documents(
            documents
        )