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
            for doc in retrieved_docs
        ]

        scores = self.model.predict(
            pairs
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

        # Best score
        best_score = (
            scored_docs[0][1]
        )

        reranked_docs = []

        for doc, score in scored_docs:

            print(
                "Rerank Score:",
                score
            )

            # Relative threshold
            if score >= (
                best_score - 0.5
            ):

                reranked_docs.append(
                    doc
                )

        # fallback
        if len(reranked_docs) == 0:

            reranked_docs = [
                doc
                for doc, score
                in scored_docs[:top_k]
            ]

        return reranked_docs