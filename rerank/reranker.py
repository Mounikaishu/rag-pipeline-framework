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
        top_k=3
    ):

        pairs = [
            (
                query,
                doc.page_content
            )
            for doc
            in retrieved_docs
        ]

        scores = (
            self.model.predict(
                pairs
            )
        )

        scored_docs = list(
            zip(
                retrieved_docs,
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

        reranked_docs = []

        for doc, score in scored_docs:

            print(
                "Rerank Score:",
                score
            )

            if score >= (
                best_score - 2.0
            ):

                reranked_docs.append(
                    doc
                )

        if (
            len(
                reranked_docs
            )
            == 0
        ):

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