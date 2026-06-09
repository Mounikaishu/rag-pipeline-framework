from embeddings.embedding_model import EmbeddingModel


embedding_model = EmbeddingModel()

model = embedding_model.get_model()

embedding = model.encode(
    "What is SQL join?"
)

print(len(embedding))
print(embedding[:10])