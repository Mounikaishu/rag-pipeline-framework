from generation.llm import (
    LLMModel
)


class Generator:

    def __init__(self):

        self.llm = (
            LLMModel()
            .get_llm()
        )

    def generate_answer(
        self,
        query,
        retrieved_docs
    ):

        context = "\n\n".join(
            [
                doc.page_content
                for doc in retrieved_docs
            ]
        )

        # ADD HERE
        print("\nRetrieved Context:\n")
        print(context)

        prompt = f"""
You are a helpful AI assistant.

Answer the user's question
ONLY using the provided context.

If answer is not present,
say:
"I could not find this information in the documents."

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm.invoke(
            prompt
        )

        return response.content