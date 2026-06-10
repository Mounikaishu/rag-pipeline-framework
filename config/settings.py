from dotenv import load_dotenv
import os

load_dotenv()


class Settings:

    GROQ_API_KEY = os.getenv(
        "GROQ_API_KEY"
    )

    LLM_MODEL = os.getenv(
        "LLM_MODEL",
        "llama-3.1-8b-instant"
    )

    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL",
        "all-MiniLM-L6-v2"
    )

    CHUNK_SIZE = int(
        os.getenv(
            "CHUNK_SIZE",
            500
        )
    )

    CHUNK_OVERLAP = int(
        os.getenv(
            "CHUNK_OVERLAP",
            100
        )
    )