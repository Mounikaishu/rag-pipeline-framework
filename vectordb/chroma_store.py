from langchain_community.vectorstores import (
    Chroma
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from config.settings import (
    Settings
)


class ChromaStore:

    def __init__(
        self,
        persist_directory
    ):

        self.embedding_model = (
            HuggingFaceEmbeddings(
                model_name=
                Settings.EMBEDDING_MODEL
            )
        )

        self.persist_directory = (
            persist_directory
        )

    def create_vector_store(
        self,
        chunks
    ):

        vector_store = Chroma.from_documents(
            documents=chunks,

            embedding=
            self.embedding_model,

            persist_directory=
            self.persist_directory
        )

        return vector_store