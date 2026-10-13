"""
Web Search Tool
===============
Lightweight external security-context retrieval for
AI-IoT-Agentic-Security.

Important:
    UNSW-NB15 remains the ONLY ML dataset used by the project.

    This tool is NOT used for ML training.
    It is only used to retrieve external security context.
"""

import requests


# ============================================================
# CONFIGURATION
# ============================================================

DDG_API_URL = "https://api.duckduckgo.com/"

HEADERS = {
    "User-Agent": (
        "AI-IoT-Agentic-Security/1.0 "
        "(security research project)"
    )
}


# ============================================================
# WEB SEARCH
# ============================================================

def web_search(query: str, max_results: int = 5):
    """
    Search for external security information.

    Returns a list of structured results.
    """

    if not query or not query.strip():
        return []

    params = {
        "q": query,
        "format": "json",
        "no_html": "1",
        "skip_disambig": "1"
    }

    try:

        response = requests.get(
            DDG_API_URL,
            params=params,
            headers=HEADERS,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

    except requests.Timeout:

        print("[ERROR] Web search timed out.")
        return []

    except requests.RequestException as exc:

        print(
            f"[ERROR] Web search request failed: {exc}"
        )

        return []

    except ValueError:

        print(
            "[ERROR] Search provider returned invalid JSON."
        )

        return []

    results = []

    # ========================================================
    # DIRECT ABSTRACT RESULT
    # ========================================================

    abstract = data.get(
        "AbstractText",
        ""
    ).strip()

    abstract_url = data.get(
        "AbstractURL",
        ""
    ).strip()

    abstract_source = data.get(
        "AbstractSource",
        ""
    ).strip()

    heading = data.get(
        "Heading",
        ""
    ).strip()

    if abstract:

        results.append(
            {
                "title": heading or query,
                "url": abstract_url,
                "snippet": abstract,
                "source": abstract_source
            }
        )

    # ========================================================
    # RELATED TOPICS
    # ========================================================

    def extract_topics(items):

        for item in items:

            if len(results) >= max_results:
                return

            if not isinstance(item, dict):
                continue

            text = item.get(
                "Text",
                ""
            ).strip()

            url = item.get(
                "FirstURL",
                ""
            ).strip()

            if text:

                results.append(
                    {
                        "title": text[:120],
                        "url": url,
                        "snippet": text,
                        "source": "DuckDuckGo"
                    }
                )

    related_topics = data.get(
        "RelatedTopics",
        []
    )

    extract_topics(related_topics)

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return results[:max_results]


# ============================================================
# SECURITY SEARCH
# ============================================================

def security_search(
    topic: str,
    max_results: int = 5
):
    """
    Search for security-related context.

    This information is NOT used as a training dataset.
    """

    query = (
        f"{topic} "
        "IoT cybersecurity intrusion detection "
        "vulnerability"
    )

    return web_search(
        query=query,
        max_results=max_results
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def print_results(
    query: str,
    results: list
):

    print("=" * 60)
    print("WEB SEARCH TOOL")
    print("=" * 60)

    print(f"Query: {query}")
    print(
        f"Results returned: {len(results)}"
    )

    if not results:

        print()
        print(
            "[WARNING] No structured results "
            "were returned by the search provider."
        )

        return

    print()

    for index, result in enumerate(
        results,
        start=1
    ):

        print("-" * 60)

        print(f"Result {index}")
        print(
            f"Title   : {result.get('title', '')}"
        )
        print(
            f"URL     : {result.get('url', '')}"
        )
        print(
            f"Source  : {result.get('source', '')}"
        )
        print(
            f"Snippet : {result.get('snippet', '')}"
        )

    print("-" * 60)


# ============================================================
# MAIN TEST
# ============================================================

def main():

    query = "IoT intrusion detection security"

    print("=" * 60)
    print("WEB SEARCH TOOL")
    print("=" * 60)

    print(
        f"Searching for: {query}"
    )

    print()

    results = web_search(
        query=query,
        max_results=5
    )

    print(
        f"Results returned: {len(results)}"
    )

    if not results:

        print()
        print(
            "[WARNING] No results returned."
        )

        print(
            "The search provider may not provide "
            "Instant Answer data for this query."
        )

        return

    print()

    for index, result in enumerate(
        results,
        start=1
    ):

        print("-" * 60)

        print(
            f"Result {index}"
        )

        print(
            f"Title   : {result.get('title', '')}"
        )

        print(
            f"URL     : {result.get('url', '')}"
        )

        print(
            f"Source  : {result.get('source', '')}"
        )

        print(
            f"Snippet : {result.get('snippet', '')}"
        )

    print("-" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()