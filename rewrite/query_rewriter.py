from generation.llm import (
    LLMModel
)


class QueryRewriter:

    def __init__(self):

        self.llm = (
            LLMModel()
            .get_llm()
        )

    def rewrite(
        self,
        query
    ):

        prompt = f"""
You are a query rewriting assistant for a RAG system.

Rewrite the query ONLY to improve retrieval.

STRICT RULES:
- Preserve original meaning exactly
- Do NOT add new words like
  "notable", "important",
  "best", "top"
- Prefer document language
- Make query more retrieval-friendly
- Keep it concise

Examples:

User Query:
What projects has she done?

Rewritten Query:
List the projects mentioned in the resume

User Query:
Tell me skills

Rewritten Query:
List the skills mentioned in the resume

User Query:
{query}

Rewritten Query:
"""

        response = (
            self.llm.invoke(
                prompt
            )
        )

        rewritten_query = (
            response.content
            .strip()
        )

        return rewritten_query