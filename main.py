from ingestion.pdf_loader import (
    PDFLoader
)

from ingestion.chunker import (
    Chunker
)

from vectordb.chroma_store import (
    ChromaStore
)

from graph.rag_graph import (
    build_graph
)


# --------------------------
# STEP 1: LOAD PDF
# --------------------------

loader = PDFLoader()

documents = loader.load_pdf(
    "./cvx (2).pdf"
)

print(
    "Loaded pages:",
    len(documents)
)


# --------------------------
# STEP 2: CHUNKING
# --------------------------

chunker = Chunker()

chunks = chunker.split_documents(
    documents
)

print(
    "Chunks created:",
    len(chunks)
)


# --------------------------
# STEP 3: STORE
# --------------------------

vector_store = ChromaStore(
    persist_directory=
    "./test_db"
)

db = (
    vector_store
    .create_vector_store(
        chunks
    )
)

print(
    "Embeddings stored!"
)


# --------------------------
# STEP 4: RUN GRAPH
# --------------------------

graph = (
    build_graph()
)

query = (
    "What internships has she completed?"
)

result = (
    graph.invoke(
        {
            "query":
            query,

            "chunks":
            chunks,

            "retry_count":
            0
        }
    )
)
print(
    "\nFinal Answer:\n"
)

print(
    result["answer"]
)