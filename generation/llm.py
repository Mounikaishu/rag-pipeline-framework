from langchain_groq import ChatGroq
from config.settings import (
    Settings
)


class LLMModel:

    def __init__(self):

        self.llm = ChatGroq(
            groq_api_key=
            Settings.GROQ_API_KEY,

            model_name=
            Settings.LLM_MODEL,

            temperature=0
        )

    def get_llm(self):

        return self.llm