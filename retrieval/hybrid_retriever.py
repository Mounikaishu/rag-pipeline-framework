
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
        k=5
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

        print(
            "\nRunning Vector Search..."
        )

        vector_docs = (
            self.vector_retriever
            .retrieve(
                query=query,
                k=k
            )
        )

        print(
            "\nVector Retrieved:",
            len(vector_docs)
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

        print(
            "\nRunning BM25 Search..."
        )

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
        )[:3]

        bm25_docs = [
            chunks[i]
            for i
            in ranked_indices
        ]

        print(
            "\nBM25 Retrieved:",
            len(bm25_docs)
        )

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

            if (
                text
                not in seen
            ):

                seen.add(
                    text
                )

                unique_docs.append(
                    doc
                )

        # -------------------
        # HYBRID SCORE
        # -------------------

        combined_scores = {}

        for rank, doc in enumerate(
            unique_docs
        ):

            text = (
                doc.page_content
            )

            vector_score = (
                vector_scores
                .get(
                    text,
                    0
                )
            )

            bm25_score = 0

            if (
                doc
                in bm25_docs
            ):

                bm25_rank = (
                    bm25_docs.index(
                        doc
                    )
                )

                bm25_score = (
                    1 /
                    (
                        bm25_rank
                        + 1
                    )
                )

            final_score = (

                0.7
                *
                vector_score

                +

                0.3
                *
                bm25_score
            )

            # -------------------
            # DOMAIN BOOSTING
            # -------------------

            text_lower = (
                text.lower()
            )

            # batting intent
            if any(
                word in query_lower
                for word in [
                    "runs",
                    "scorer",
                    "batting",
                    "orange cap",
                    "highest runs",
                    "most runs"
                ]
            ):

                if (
                    "batting statistics"
                    in text_lower
                ):

                    final_score += 2.5

            # bowling intent
            elif any(
                word in query_lower
                for word in [
                    "wickets",
                    "bowling",
                    "best bowler"
                ]
            ):

                if (
                    "bowling statistics"
                    in text_lower
                ):

                    final_score += 2.5

            combined_scores[
                text
            ] = final_score

        # -------------------
        # SORT DOCS
        # -------------------

        final_docs = sorted(
            unique_docs,
            key=lambda doc:
            combined_scores.get(
                doc.page_content,
                0
            ),
            reverse=True
        )

        final_docs = (
            final_docs[:k]
        )

        # -------------------
        # DEBUG PRINT
        # -------------------

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
