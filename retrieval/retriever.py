from langchain_community.vectorstores import (
    Chroma
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from config.settings import (
    Settings
)


class Retriever:

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

        self.vector_store = Chroma(
            persist_directory=
            persist_directory,

            embedding_function=
            self.embedding_model
        )

    def retrieve(
        self,
        query,
        k=3
    ):

        results = (
    self.vector_store
    .max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=10
    )
)

        return results