class Refiner:

    def refine(
        self,
        reranked_docs
    ):

        refined_docs = []

        for doc in reranked_docs:

            text = (
                doc.page_content
            )

            lines = text.split(
                "\n"
            )

            filtered_lines = []

            for line in lines:

                line = line.strip()

                # remove tiny/noisy lines
                if len(line) > 20:

                    filtered_lines.append(
                        line
                    )

            refined_text = (
                "\n".join(
                    filtered_lines
                )
            )

            doc.page_content = (
                refined_text
            )

            refined_docs.append(
                doc
            )

        return refined_docs