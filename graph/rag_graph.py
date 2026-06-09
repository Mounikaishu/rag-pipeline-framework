from typing import TypedDict

from langgraph.graph import (
    StateGraph,
    END
)

from rewrite.query_rewriter import (
    QueryRewriter
)

from retrieval.retriever import (
    Retriever
)

from rerank.reranker import (
    Reranker
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
    rewritten_query: str
    retrieved_docs: list
    reranked_docs: list
    refined_docs: list
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

    return {
        "rewritten_query":
        rewritten_query
    }


def retrieve_node(
    state: GraphState
):

    retriever = (
        Retriever(
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

    reranked_docs = (
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

    print(
        "\nReranked docs:",
        len(reranked_docs)
    )

    return {
        "reranked_docs":
        reranked_docs
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

    workflow.set_entry_point(
        "rewrite"
    )

    workflow.add_edge(
        "rewrite",
        "retrieve"
    )

    workflow.add_edge(
        "retrieve",
        "rerank"
    )

    workflow.add_edge(
        "rerank",
        "refine"
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