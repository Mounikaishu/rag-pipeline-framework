def route_after_rerank(
    state
):

    score_gap = (
        state[
            "score_gap"
        ]
    )

    retry_count = (
        state.get(
            "retry_count",
            0
        )
    )

    # max retries
    if retry_count >= 2:

        print(
            "\nMax retries reached → continuing"
        )

        return (
            "refine"
        )

    # weak confidence
    if score_gap < 1:

        print(
            "\nLow confidence → retrying..."
        )

        return (
            "rewrite"
        )

    print(
        "\nGood retrieval → continue"
    )

    return (
        "refine"
    )