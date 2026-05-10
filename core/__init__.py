def __getattr__(name):
    if name in ("search", "SearchRequest", "SearchResponse"):
        from core.search_engine import search, SearchRequest, SearchResponse
        return {"search": search, "SearchRequest": SearchRequest, "SearchResponse": SearchResponse}[name]
    raise AttributeError(f"module 'core' has no attribute {name!r}")
