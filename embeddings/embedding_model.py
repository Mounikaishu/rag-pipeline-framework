from sentence_transformers import SentenceTransformer
from config.settings import Settings


class EmbeddingModel:

    def __init__(self):
        self.model = SentenceTransformer(
            Settings.EMBEDDING_MODEL
        )

    def get_model(self):
        return self.model