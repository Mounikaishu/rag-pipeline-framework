from ingestion.pdf_loader import (
    PDFLoader
)

from ingestion.chunker import (
    Chunker
)

from vectordb.chroma_store import (
    ChromaStore
)

from retrieval.retriever import (
    Retriever
)

from generation.generator import (
    Generator
)

from rerank.reranker import (
    Reranker
)

from refine.refiner import (
    Refiner
)

from rewrite.query_rewriter import (
    QueryRewriter
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
# STEP 4: QUERY REWRITE
# --------------------------

query = (
    "What projects has she done?"
)

rewriter = (
    QueryRewriter()
)

rewritten_query = (
    rewriter.rewrite(
        query
    )
)

print(
    "\nOriginal Query:",
    query
)

print(
    "Rewritten Query:",
    rewritten_query
)


# --------------------------
# STEP 5: RETRIEVE
# --------------------------

retriever = Retriever(
    persist_directory=
    "./test_db"
)

retrieved_docs = (
    retriever.retrieve(
        query=
        rewritten_query,
        k=10
    )
)

print(
    "\nRetrieved docs:",
    len(retrieved_docs)
)
# --------------------------
# STEP 5: RERANK
# --------------------------

reranker = Reranker()

reranked_docs = (
    reranker.rerank(
        query=rewritten_query,
        retrieved_docs=
        retrieved_docs,
        top_k=3
    )
)

print(
    "\nReranked docs:",
    len(reranked_docs)
)


# --------------------------
# STEP 6: REFINE
# --------------------------

refiner = Refiner()

refined_docs = (
    refiner.refine(
        reranked_docs
    )
)

print(
    "\nRefined docs:",
    len(refined_docs)
)


# --------------------------
# STEP 7: GENERATE
# --------------------------

generator = Generator()

answer = (
    generator
    .generate_answer(
        query=rewritten_query,
        retrieved_docs=
        refined_docs
    )
)

print("\nFinal Answer:\n")

print(answer)