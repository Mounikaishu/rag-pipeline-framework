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
        query,
        chat_history=""
    ):

        query = (
            query
            .strip()
        )

        has_history = (
            len(
                str(chat_history)
            ) > 5
        )

        prompt = f"""
You are an intelligent query rewriting
assistant for a conversational
RAG system.

TASK:
Rewrite ONLY to improve retrieval.

CORE RULE:
Preserve user intent EXACTLY.

STRICT RULES:
- NEVER answer the question
- NEVER invent information
- NEVER assume entities
- NEVER add new conditions
- NEVER change numbers
- NEVER change meaning
- Keep rewriting minimal

Conversation history should ONLY
be used when the query contains:

- pronouns
- omitted references
- follow-up questions
- missing subjects

Examples:

User:
Who scored highest runs in IPL?

Rewrite:
Who scored highest runs in IPL?

User:
Which players scored
more than 5000 runs?

Rewrite:
Which IPL players scored
more than 5000 runs?

User:
How many centuries
they have scored?

Rewrite:
How many centuries have
the previously discussed
players scored?

IMPORTANT:
If query is already clear,
rewrite minimally.

Conversation History:
{chat_history if has_history else "None"}

Current Query:
{query}

Output:
Only rewritten query text.
"""

        response = (
            self.llm.invoke(
                prompt
            )
        )

        rewritten_query = (
            response.content
            .strip()
            .split("\n")[0]
        )

        # -------------------
        # fallback
        # -------------------

        if (
            len(
                rewritten_query
            ) == 0
        ):

            rewritten_query = (
                query
            )

        # -------------------
        # conversational fallback
        # -------------------

        pronouns = [
            "he",
            "she",
            "they",
            "them",
            "his",
            "her",
            "it"
        ]

        query_lower = (
            query.lower()
        )

        has_pronoun = any(
            p in query_lower
            for p in pronouns
        )

        if (
            has_pronoun
            and has_history
            and rewritten_query.lower()
            == query.lower()
        ):

            rewritten_query = (
                f"{query} "
                f"about previously discussed topic"
            )

        print(
            "\nRewritten Query:",
            rewritten_query
        )

        return (
            rewritten_query
        )