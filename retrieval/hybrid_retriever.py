
from rank_bm25 import (
    BM25Okapi
)

from retrieval.retriever import (
    Retriever
)


class HybridRetriever:

    def __init__(
        self,
        persist_directory
    ):

        self.vector_retriever = (
            Retriever(
                persist_directory=
                persist_directory
            )
        )

    def retrieve(
        self,
        query,
        chunks,
        k=10
    ):

        print(
            "\nHYBRID RETRIEVER RUNNING"
        )

        query_lower = (
            query.lower()
        )

        # -------------------
        # VECTOR SEARCH
        # -------------------

        vector_docs = (
            self.vector_retriever
            .retrieve(
                query=query,
                k=k
            )
        )

        vector_scores = {}

        for rank, doc in enumerate(
            vector_docs
        ):

            vector_scores[
                doc.page_content
            ] = (
                1 / (rank + 1)
            )

        # -------------------
        # BM25 SEARCH
        # -------------------

        tokenized_chunks = [

            doc.page_content
            .lower()
            .split()

            for doc
            in chunks
        ]

        bm25 = (
            BM25Okapi(
                tokenized_chunks
            )
        )

        tokenized_query = (
            query.lower()
            .split()
        )

        bm25_scores = (
            bm25.get_scores(
                tokenized_query
            )
        )

        ranked_indices = sorted(
            range(
                len(
                    bm25_scores
                )
            ),
            key=lambda i:
            bm25_scores[i],
            reverse=True
        )[:k]

        bm25_docs = [
            chunks[i]
            for i
            in ranked_indices
        ]

        # -------------------
        # COMBINE SCORES
        # -------------------

        combined_scores = {}

        all_docs = (
            vector_docs
            + bm25_docs
        )

        for rank, doc in enumerate(
            bm25_docs
        ):

            text = (
                doc.page_content
            )

            bm25_score = (
                1 / (rank + 1)
            )

            vector_score = (
                vector_scores
                .get(text, 0)
            )

            combined_scores[
                text
            ] = (

                0.7
                *
                vector_score

                +

                0.3
                *
                bm25_score
            )

        # -------------------
        # DEDUPLICATE
        # -------------------

        unique_docs = {}

        for doc in all_docs:

            text = (
                doc.page_content
            )

            if (
                text
                not in unique_docs
            ):

                unique_docs[
                    text
                ] = doc

        final_docs = sorted(
            unique_docs.values(),
            key=lambda doc:
            combined_scores.get(
                doc.page_content,
                0
            ),
            reverse=True
        )

        # -------------------
        # METADATA BOOSTING
        # -------------------

        skill_keywords = [
            "coding",
            "programming",
            "language",
            "skills",
            "technical"
        ]

        if any(
            keyword
            in query_lower
            for keyword
            in skill_keywords
        ):

            boosted_docs = []
            other_docs = []

            for doc in final_docs:

                section = str(
                    doc.metadata.get(
                        "section",
                        ""
                    )
                ).lower()

                if (
                    section
                    == "skills"
                ):

                    boosted_docs.append(
                        doc
                    )

                else:

                    other_docs.append(
                        doc
                    )

            final_docs = (
                boosted_docs
                + other_docs
            )

        # -------------------
        # LIMIT TOP K
        # -------------------

        final_docs = (
            final_docs[:k]
        )

        print(
            "\nHybrid Retrieved:",
            len(final_docs)
        )

        for i, doc in enumerate(
            final_docs
        ):

            print(
                f"\nDoc {i+1}"
            )

            print(
                "Section:",
                doc.metadata.get(
                    "section",
                    "N/A"
                )
            )

            print(
                doc.page_content[
                    :150
                ]
            )

        return final_docs
