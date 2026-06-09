from typing import TypedDict

from langgraph.graph import (
    StateGraph,
    END
)

from rewrite.query_rewriter import (
    QueryRewriter
)

from retrieval.hybrid_retriever import (
    HybridRetriever
)

from rerank.reranker import (
    Reranker
)

from graph.router import (
    route_after_rerank
)

from refine.refiner import (
    Refiner
)

from generation.generator import (
    Generator
)


# -------------------------
# STATE
# -------------------------

class GraphState(
    TypedDict
):
    query: str
    chunks: list
    rewritten_query: str
    retrieved_docs: list
    reranked_docs: list
    refined_docs: list
    best_score: float
    score_gap: float
    retry_count: int
    answer: str

# -------------------------
# NODES
# -------------------------

def rewrite_node(
    state: GraphState
):

    rewriter = (
        QueryRewriter()
    )

    rewritten_query = (
        rewriter.rewrite(
            state["query"]
        )
    )

    print(
        "\nRewritten Query:",
        rewritten_query
    )
    retry_count = (
    state.get(
        "retry_count",
        0
    ) + 1
)
    return {
    "rewritten_query":
    rewritten_query,

    "retry_count":
    retry_count
}


def retrieve_node(
    state: GraphState
):

    retriever = (
        HybridRetriever(
            persist_directory=
            "./test_db"
        )
    )

    retrieved_docs = (
        retriever.retrieve(
            query=
            state[
                "rewritten_query"
            ],

            chunks=
            state[
                "chunks"
            ],

            k=10
        )
    )

    print(
        "\nRetrieved docs:",
        len(retrieved_docs)
    )

    return {
        "retrieved_docs":
        retrieved_docs
    }


def rerank_node(
    state: GraphState
):

    reranker = (
        Reranker()
    )

    rerank_result = (
        reranker.rerank(
            query=
            state[
                "rewritten_query"
            ],

            retrieved_docs=
            state[
                "retrieved_docs"
            ]
        )
    )

    reranked_docs = (
        rerank_result[
            "docs"
        ]
    )

    best_score = (
        rerank_result[
            "best_score"
        ]
    )

    score_gap = (
        rerank_result[
            "score_gap"
        ]
    )

    print(
        "\nBest rerank score:",
        best_score
    )

    print(
        "Score gap:",
        score_gap
    )

    print(
        "\nReranked docs:",
        len(reranked_docs)
    )

    return {
        "reranked_docs":
        reranked_docs,

        "best_score":
        best_score,

        "score_gap":
        score_gap
    }


def refine_node(
    state: GraphState
):

    refiner = (
        Refiner()
    )

    refined_docs = (
        refiner.refine(
            state[
                "reranked_docs"
            ]
        )
    )

    print(
        "\nRefined docs:",
        len(refined_docs)
    )

    return {
        "refined_docs":
        refined_docs
    }


def generate_node(
    state: GraphState
):

    generator = (
        Generator()
    )

    answer = (
        generator
        .generate_answer(
            query=
            state[
                "rewritten_query"
            ],

            retrieved_docs=
            state[
                "refined_docs"
            ]
        )
    )

    return {
        "answer":
        answer
    }


# -------------------------
# BUILD GRAPH
# -------------------------

def build_graph():

    workflow = (
        StateGraph(
            GraphState
        )
    )

    # Nodes
    workflow.add_node(
        "rewrite",
        rewrite_node
    )

    workflow.add_node(
        "retrieve",
        retrieve_node
    )

    workflow.add_node(
        "rerank",
        rerank_node
    )

    workflow.add_node(
        "refine",
        refine_node
    )

    workflow.add_node(
        "generate",
        generate_node
    )

    # Entry point
    workflow.set_entry_point(
        "rewrite"
    )

    # Edges
    workflow.add_edge(
        "rewrite",
        "retrieve"
    )

    workflow.add_edge(
        "retrieve",
        "rerank"
    )

    # Conditional routing
    workflow.add_conditional_edges(
        "rerank",
        route_after_rerank,
        {
            "rewrite":
            "rewrite",

            "refine":
            "refine"
        }
    )

    workflow.add_edge(
        "refine",
        "generate"
    )

    workflow.add_edge(
        "generate",
        END
    )

    return workflow.compile()