from langchain_experimental.text_splitter import (
    SemanticChunker
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_core.documents import (
    Document
)

import re


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

        full_text = "\n".join(
            [
                doc.page_content
                for doc in documents
            ]
        )

        # ----------------------
        # SECTION HEADERS
        # ----------------------

        headers = [
            "Career Objective",
            "Education",
            "Projects",
            "Skills",
            "Achievements",
            "Certifications",
            "Experience",
            "Internships"
        ]

        pattern = (
            "("
            + "|".join(headers)
            + ")"
        )

        splits = re.split(
            pattern,
            full_text
        )

        section_docs = []

        current_header = None

        for split in splits:

            split = split.strip()

            if not split:
                continue

            # If header found
            if split in headers:

                current_header = split

            else:

                content = (
                    f"{current_header}\n"
                    f"{split}"
                )

                section_docs.append(
                    Document(
                        page_content=
                        content,

                        metadata={
                            "section":
                            current_header
                        }
                    )
                )

        # ----------------------
        # SEMANTIC CHUNKING
        # ----------------------

        final_chunks = []

        for section_doc in section_docs:

            semantic_chunks = (
                self.semantic_chunker
                .split_documents(
                    [section_doc]
                )
            )

            final_chunks.extend(
                semantic_chunks
            )

        print(
            "Chunks created:",
            len(final_chunks)
        )

        return final_chunks