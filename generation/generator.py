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

If information exists
in the context,
extract it completely.

Do NOT ignore relevant details.

If multiple relevant
items exist,
include all of them.

Do not guess or
invent information.

If answer is not present,
say exactly:

"I could not find this information in the documents."

QUESTION:
{query}

CONTEXT:
{context}

ANSWER:
"""

        response = self.llm.invoke(
            prompt
        )

        return response.content