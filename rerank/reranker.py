
from sentence_transformers import (
    CrossEncoder
)


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(
        self,
        query,
        retrieved_docs,
        top_k=4
    ):

        # ------------------------
        # FILTER VALID DOCS
        # ------------------------

        pairs = []

        valid_docs = []

        for doc in retrieved_docs:

            if hasattr(
                doc,
                "page_content"
            ):

                pairs.append(
                    (
                        query,
                        doc.page_content
                    )
                )

                valid_docs.append(
                    doc
                )

            else:

                print(
                    "\nSkipping invalid doc:",
                    type(doc)
                )

        # ------------------------
        # SAFETY CHECK
        # ------------------------

        if len(
            valid_docs
        ) == 0:

            return {

                "docs": [],

                "best_score":
                0.0,

                "confidence":
                0.0,

                "score_gap":
                0.0
            }

        # ------------------------
        # SCORE DOCS
        # ------------------------

        scores = (
            self.model.predict(
                pairs
            )
        )

        scored_docs = list(
            zip(
                valid_docs,
                scores
            )
        )

        scored_docs.sort(
            key=lambda x: x[1],
            reverse=True
        )

        best_score = (
            scored_docs[0][1]
        )

        second_score = (

            scored_docs[1][1]

            if len(
                scored_docs
            ) > 1

            else best_score
        )

        score_gap = (
            best_score
            - second_score
        )

        confidence = max(
            0,
            min(
                100,
                (
                    (
                        best_score
                        + 10
                    )
                    / 20
                ) * 100
            )
        )

        print(
            "\nScore Gap:",
            score_gap
        )

        for doc, score in scored_docs:

            print(
                "Rerank Score:",
                score
            )

        # ------------------------
        # KEEP TOP K DOCS
        # ------------------------

        reranked_docs = [

            doc

            for doc, score
            in scored_docs[
                :top_k
            ]
        ]

        return {

            "docs":
            reranked_docs,

            "best_score":
            float(
                best_score
            ),

            "confidence":
            float(
                confidence
            ),

            "score_gap":
            float(
                score_gap
            )
        }
