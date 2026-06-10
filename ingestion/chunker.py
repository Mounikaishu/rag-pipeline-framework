from langchain_experimental.text_splitter import (
    SemanticChunker
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)


class Chunker:

    def __init__(self):

        self.embeddings = (
            HuggingFaceEmbeddings(
                model_name=
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        )

        self.semantic_chunker = (
            SemanticChunker(
                self.embeddings
            )
        )

    def split_documents(
        self,
        documents
    ):

        # ----------------------
        # GENERIC SEMANTIC CHUNKING
        # ----------------------

        final_chunks = (
            self.semantic_chunker
            .split_documents(
                documents
            )
        )

        print(
            "Chunks created:",
            len(final_chunks)
        )

        return final_chunks

