
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

        self.vector_store = (
            Chroma(
                persist_directory=
                persist_directory,

                embedding_function=
                self.embedding_model
            )
        )

    def retrieve(
        self,
        query,
        k=5
    ):

        print(
            "\nRunning Vector Search..."
        )

        try:

            results = (
                self.vector_store
                .similarity_search_with_score(
                    query,
                    k=k
                )
            )

            docs = []

            for doc, score in results:

                print(
                    f"Vector Score:"
                    f" {score:.4f}"
                )

                docs.append(
                    doc
                )

            print(
                "\nVector Retrieved:",
                len(docs)
            )

            return docs

        except Exception as e:

            print(
                "\nVector Search Error:",
                e
            )

            return []
