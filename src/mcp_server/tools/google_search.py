from langchain_core.tools import tool
import serpapi

from open_deep_research.utils.api_keys import (
    get_api_key,
    ApiKeyEnvVar
)

def perform_serpapi_search(engine: str, params: dict, snippet_path: list = None) -> str:
    """Generic search function using SerpAPI."""

    client = serpapi.Client(get_api_key(ApiKeyEnvVar.SERPAPI))
    search = client.search({
        **params,
        "engine": engine,
    })
    results = search.get_dict()

    # If snippet_path is specified, traverse to get the snippet
    if snippet_path:
        try:
            data = results
            for key in snippet_path:
                if isinstance(data, list):
                    data = data[0] if data else {}
                data = data.get(key, {})
            return data if isinstance(data, str) else str(data)
        except Exception:
            return "Failed to extract snippet."
    return str(results.get("organic_results", results))


@tool
def search_serpapi(query: str) -> str:
    """Search Google using SerpAPI and return the first snippet."""
    return perform_serpapi_search(
        engine="google",
        params={"q": query},
        snippet_path=["organic_results", "snippet"]
    )


@tool
def search_autocomplete(query: str) -> str:
    """Use SerpAPI Google Autocomplete to suggest completions for a query."""
    result = perform_serpapi_search(
        engine="google_autocomplete",
        params={"q": query}
    )
    suggestions = [s['value'] for s in result.get('suggestions', [])] if isinstance(result, dict) else []
    return ", ".join(suggestions) if suggestions else "No autocomplete suggestions found."


@tool
def search_google_scholar(query: str) -> str:
    """Use SerpAPI Google Scholar to search academic content."""
    results = perform_serpapi_search(
        engine="google_scholar",
        params={"q": query}
    )
    if isinstance(results, dict):
        entries = results.get("organic_results", [])
        if entries:
            top = entries[0]
            return f"{top.get('title', 'No title')}: {top.get('snippet', 'No snippet')}"
    return "No scholar results found."


@tool
def search_duckduckgo(query: str) -> str:
    """Use SerpAPI DuckDuckGo to search the web."""
    return perform_serpapi_search(
        engine="duckduckgo",
        params={"q": query},
        snippet_path=["organic_results", "snippet"]
    )


@tool
def search_bing(query: str) -> str:
    """Searches the Bing search engine for the given query using SerpAPI."""
    return perform_serpapi_search(
        engine="bing",
        params={"q": query},
        snippet_path=["organic_results", "snippet"]
    )


@tool
def search_baidu(query: str) -> str:
    """Searches the Baidu search engine for the given query using SerpAPI."""
    return perform_serpapi_search(
        engine="baidu",
        params={"q": query},
        snippet_path=["organic_results", "snippet"]
    )


@tool
def search_walmart(query: str) -> str:
    """Searches Walmart for the given product query using SerpAPI."""
    return perform_serpapi_search(
        engine="walmart",
        params={"query": query},
        snippet_path=["organic_results", "title"]  # Walmart có thể không có snippet, lấy title
    )
