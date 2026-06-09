from langchain_experimental.text_splitter import (
    SemanticChunker
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_core.documents import (
    Document
)

from config.settings import (
    Settings
)


class Chunker:

    def __init__(self):

        self.headers = [
            "Career Objective",
            "Education",
            "Projects",
            "Skills",
            "Achievements",
            "Certifications",
            "Experience"
        ]

        self.embedding_model = (
            HuggingFaceEmbeddings(
                model_name=
                Settings.EMBEDDING_MODEL
            )
        )

        self.semantic_splitter = (
            SemanticChunker(
                embeddings=
                self.embedding_model,

                breakpoint_threshold_type=
                "percentile"
            )
        )

    def split_documents(
        self,
        documents
    ):

        all_chunks = []

        for doc in documents:

            text = doc.page_content

            sections = (
                self._header_chunk(
                    text
                )
            )

            for section in sections:

                semantic_chunks = (
                    self.semantic_splitter
                    .create_documents(
                        [section]
                    )
                )

                all_chunks.extend(
                    semantic_chunks
                )

        return all_chunks

    def _header_chunk(
        self,
        text
    ):

        sections = []

        current_section = ""

        lines = text.split("\n")

        for line in lines:

            line = line.strip()

            if line in self.headers:

                if current_section:

                    sections.append(
                        current_section
                    )

                current_section = (
                    line + "\n"
                )

            else:

                current_section += (
                    line + "\n"
                )

        if current_section:

            sections.append(
                current_section
            )

        return sections