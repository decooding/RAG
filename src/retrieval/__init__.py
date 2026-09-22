# Package src.retrieval
from src.retrieval.models import SearchResult


def retrieve(*args, **kwargs):
    from src.retrieval.search_engine import retrieve as _retrieve
    return _retrieve(*args, **kwargs)


def get_search_engine(*args, **kwargs):
    from src.retrieval.search_engine import RAGSearchEngine
    return RAGSearchEngine(*args, **kwargs)


__all__ = ["SearchResult", "retrieve", "get_search_engine"]
