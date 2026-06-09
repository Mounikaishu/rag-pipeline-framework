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
            "\nNEW HYBRID RETRIEVER RUNNING"
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

        # -------------------
        # BM25 SEARCH
        # -------------------

        tokenized_chunks = [
            doc.page_content.split()
            for doc in chunks
        ]

        bm25 = (
            BM25Okapi(
                tokenized_chunks
            )
        )

        tokenized_query = (
            query.split()
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
            for i in ranked_indices
        ]

        # -------------------
        # MERGE RESULTS
        # -------------------

        combined_docs = (
            vector_docs
            + bm25_docs
        )

        unique_docs = []

        seen = set()

        for doc in combined_docs:

            text = (
                doc.page_content
            )

            if text not in seen:

                seen.add(
                    text
                )

                unique_docs.append(
                    doc
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

            for doc in unique_docs:

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

            unique_docs = (
                boosted_docs
                + other_docs
            )

        # -------------------
        # DEBUG PRINT
        # -------------------

        print(
            "\nHybrid Retrieved docs:",
            len(unique_docs)
        )

        for doc in unique_docs:

            print(
                "\nSECTION:",
                doc.metadata.get(
                    "section"
                )
            )

            print(
                doc.page_content[
                    :200
                ]
            )

        return unique_docs