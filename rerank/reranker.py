
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

        # ------------------
        # EMPTY DOC SAFETY
        # ------------------

        if len(retrieved_docs) == 0:

            return {
                "docs": [],
                "best_score": 0.0,
                "score_gap": 0.0,
                "confidence": 0.0
            }

        query_lower = (
            query.lower()
        )

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

        scored_docs = []

        for doc, score in zip(
            retrieved_docs,
            scores
        ):

            section = str(
                doc.metadata.get(
                    "section",
                    ""
                )
            ).lower()

            # ------------------
            # SECTION BONUS
            # ------------------

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

                if (
                    section
                    == "skills"
                ):

                    score += 2.0

            scored_docs.append(
                (
                    doc,
                    score
                )
            )

        # ------------------
        # SORT BY SCORE
        # ------------------

        scored_docs.sort(
            key=lambda x: x[1],
            reverse=True
        )

        best_score = (
            scored_docs[0][1]
        )

        second_score = (
            scored_docs[1][1]
            if len(scored_docs) > 1
            else best_score
        )

        score_gap = (
            best_score
            - second_score
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

        if len(
            reranked_docs
        ) == 0:

            reranked_docs = [
                doc
                for doc, score
                in scored_docs[:top_k]
            ]

        # ------------------
        # CONFIDENCE SCORE
        # ------------------

        confidence = max(
            0,
            min(
                100,
                (best_score + 15) * 5
            )
        )

        print(
            "Confidence:",
            confidence
        )

        # ------------------
        # RETURN
        # ------------------

        return {

            "docs":
            reranked_docs,

            "best_score":
            float(best_score),

            "score_gap":
            float(score_gap),

            "confidence":
            float(confidence)
        }

