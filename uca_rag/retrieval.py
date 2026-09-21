from __future__ import annotations

import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .hierarchy import HierarchicalCorpus, HierarchyConfig, SummaryGenerator, build_hierarchy, expand_retrieval_units
from .prompts import build_search_queries


@dataclass
class Document:
    text: str
    source: str
    metadata: dict[str, str] = field(default_factory=dict)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _html_to_structured_text(root) -> str:
    """Keep headings visible so the in-memory hierarchy can recover web sections."""
    parts: list[str] = []
    for element in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table"]):
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        if element.name.startswith("h"):
            parts.append(f"{'#' * int(element.name[1])} {text}")
        elif element.name == "li":
            parts.append(f"- {text}")
        else:
            parts.append(text)
    return "\n\n".join(parts) or root.get_text(" ", strip=True)


def _model_cache_folder() -> str | None:
    configured = os.getenv("UCA_RAG_MODEL_CACHE")
    return str(Path(configured).expanduser()) if configured else None


@lru_cache(maxsize=None)
def _load_embedding_model(model_name: str, cache_folder: str | None):
    from sentence_transformers import SentenceTransformer

    kwargs = {"cache_folder": cache_folder} if cache_folder else {}
    return SentenceTransformer(model_name, **kwargs)


@lru_cache(maxsize=None)
def _load_reranker_model(model_name: str, cache_folder: str | None):
    from sentence_transformers import CrossEncoder

    kwargs = {"cache_folder": cache_folder} if cache_folder else {}
    return CrossEncoder(model_name, **kwargs)


class ExcelIndexer:
    def __init__(
        self,
        directory: str | Path | None,
        hierarchy_config: HierarchyConfig | None = None,
        processed_dir: str | Path | None = None,
        summary_generator: SummaryGenerator | None = None,
    ):
        self.directory = Path(directory) if directory is not None else None
        self.hierarchy_config = hierarchy_config or HierarchyConfig.from_environment()
        self.processed_dir = Path(processed_dir) if processed_dir is not None else None
        self.summary_generator = summary_generator
        self.documents: list[Document] = []

    def build(self) -> "ExcelIndexer":
        import pandas as pd

        self.documents = []
        if self.directory is None or not self.directory.exists():
            return self
        for workbook in sorted(self.directory.glob("*.xls*")):
            sheets = pd.read_excel(workbook, sheet_name=None)
            for sheet_name, frame in sheets.items():
                frame = frame.dropna(how="all")
                for row_number, row in frame.iterrows():
                    values = [f"{column}: {value}" for column, value in row.items() if pd.notna(value)]
                    if values:
                        self.documents.append(
                            Document(
                                text=" | ".join(values),
                                source=f"{workbook.name}#{sheet_name}!row-{row_number + 2}",
                                metadata={"workbook": workbook.name, "sheet": str(sheet_name)},
                            )
                        )
        return self


class _LexicalIndex:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.term_sets = [set(_tokens(document.text)) for document in documents]
        self.bm25 = None
        if not documents:
            return
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi([_tokens(document.text) for document in documents])
        except ImportError:
            self.bm25 = None

    def search(self, query: str, limit: int) -> list[tuple[Document, float]]:
        query_terms = set(_tokens(query))
        if self.bm25 is not None:
            scores = self.bm25.get_scores(list(query_terms))
            ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:limit]
            return [(self.documents[index], float(score)) for index, score in ranked if score > 0]

        scored = []
        for document, terms in zip(self.documents, self.term_sets):
            if not terms:
                continue
            overlap = len(query_terms & terms)
            score = overlap / math.sqrt(max(len(query_terms) * len(terms), 1))
            if score:
                scored.append((document, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]


class _DenseIndex:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.model = None
        self.vectors = None
        if not documents:
            return
        try:
            self.model = _load_embedding_model("all-MiniLM-L6-v2", _model_cache_folder())
            self.vectors = self.model.encode([document.text for document in documents], normalize_embeddings=True)
        except Exception:
            self.vectors = None

    def search(self, query: str, limit: int) -> list[tuple[Document, float]]:
        if not self.documents:
            return []
        if self.model is None:
            return _LexicalIndex(self.documents).search(query, limit)
        query_vector = self.model.encode([query], normalize_embeddings=True)[0]
        scored = [(document, float(vector @ query_vector)) for document, vector in zip(self.documents, self.vectors)]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = None
        try:
            self.model = _load_reranker_model(model_name, _model_cache_folder())
        except Exception:
            pass

    def rerank(self, query: str, candidates: list[tuple[Document, float]], limit: int) -> list[tuple[Document, float]]:
        if not candidates:
            return []
        if self.model is None:
            return sorted(candidates, key=lambda item: item[1], reverse=True)[:limit]
        pairs = [(query, document.text) for document, _ in candidates]
        scores = self.model.predict(pairs)
        ranked = [(document, float(score)) for (document, _), score in zip(candidates, scores)]
        return sorted(ranked, key=lambda item: item[1], reverse=True)[:limit]


class HybridRetriever:
    def __init__(self, index: ExcelIndexer | list[Document], reranker: CrossEncoderReranker | None = None):
        self.documents = index.documents if isinstance(index, ExcelIndexer) else index
        self.lexical = _LexicalIndex(self.documents)
        self.dense = _DenseIndex(self.documents)
        self.reranker = reranker or (CrossEncoderReranker() if self.documents else None)

    def retrieve(self, query: str, limit: int = 8) -> list[Document]:
        return self.retrieve_many([query], limit=limit)

    def retrieve_many(self, queries: Iterable[str], limit: int = 8) -> list[Document]:
        if not self.documents:
            return []
        query_list = list(dict.fromkeys(query for query in queries if query.strip()))
        if not query_list:
            return []
        with ThreadPoolExecutor(max_workers=2) as executor:
            lexical_futures = [executor.submit(self.lexical.search, query, limit * 2) for query in query_list]
            dense_futures = [executor.submit(self.dense.search, query, limit * 2) for query in query_list]
            lexical_results = [future.result() for future in lexical_futures]
            dense_results = [future.result() for future in dense_futures]

        # Rank fusion avoids treating lexical and cosine scores as if they shared a scale.
        candidates: dict[str, tuple[Document, float]] = {}
        for query_results in (lexical_results, dense_results):
            for results in query_results:
                for rank, (document, _) in enumerate(results, start=1):
                    score = 1.0 / (60 + rank)
                    current = candidates.get(document.source)
                    candidates[document.source] = (document, score + (current[1] if current else 0.0))
        rerank_query = " ".join(query_list)
        return [document for document, _ in self.reranker.rerank(rerank_query, list(candidates.values()), limit)]


class WebSearchRetriever:
    def __init__(self, max_results: int = 5, timeout: float = 10.0, chunk_size: int = 1200, chunk_overlap: int = 180):
        self.max_results = max_results
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split(self, text: str) -> list[str]:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            return splitter.split_text(text)
        except ImportError:
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunks.append(text[start:end])
                if end == len(text):
                    break
                start = end - self.chunk_overlap
            return chunks

    def _fetch_result_text(
        self,
        result: dict,
        query: str,
        requests_module,
        beautifulsoup_module,
        *,
        prefer_full_text: bool = False,
    ) -> tuple[str, str, str]:
        url = result.get("href", query)
        title = result.get("title", "")
        snippet = " ".join(filter(None, [title, result.get("body")]))
        text = ""
        if url.startswith(("http://", "https://")):
            try:
                response = requests_module.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; UCA-RAG/1.0)"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                soup = beautifulsoup_module(response.content, "html.parser")
                for element in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form"]):
                    element.decompose()
                content_root = soup.find("article") or soup.find("main") or soup.body or soup
                text = _html_to_structured_text(content_root)
            except Exception:
                text = ""
        if not prefer_full_text and len(text) < 300:
            text = snippet
        return url, title, text

    def _documents_from_result(self, result: dict, query: str, requests_module, beautifulsoup_module) -> list[Document]:
        url, title, text = self._fetch_result_text(result, query, requests_module, beautifulsoup_module)
        if not text:
            return []

        chunks = self._split(text)
        return [
            Document(
                text=chunk,
                source=f"{url}#chunk-{chunk_number}",
                metadata={
                    "parent_source": url,
                    "title": title,
                    "query": query,
                    "chunk": str(chunk_number),
                },
            )
            for chunk_number, chunk in enumerate(chunks)
        ]

    def _source_document_from_result(self, result: dict, query: str, requests_module, beautifulsoup_module) -> Document | None:
        url, title, text = self._fetch_result_text(
            result, query, requests_module, beautifulsoup_module, prefer_full_text=True
        )
        if not text:
            return None
        return Document(
            text=text,
            source=url,
            metadata={"parent_source": url, "title": title, "query": query, "level": "document"},
        )

    def retrieve(self, queries: Iterable[str]) -> list[Document]:
        try:
            from ddgs import DDGS
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return []
        documents: list[Document] = []
        seen_urls: set[str] = set()
        try:
            with DDGS() as client:
                for query in queries:
                    for result in client.text(query, max_results=self.max_results):
                        url = result.get("href", "")
                        if url in seen_urls:
                            continue
                        seen_urls.add(url)
                        documents.extend(self._documents_from_result(result, query, requests, BeautifulSoup))
        except Exception:
            return documents
        return documents

    def retrieve_sources(self, queries: Iterable[str]) -> list[Document]:
        try:
            from ddgs import DDGS
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return []
        documents: list[Document] = []
        seen_urls: set[str] = set()
        try:
            with DDGS() as client:
                for query in queries:
                    for result in client.text(query, max_results=self.max_results):
                        url = result.get("href", "")
                        if url in seen_urls:
                            continue
                        seen_urls.add(url)
                        document = self._source_document_from_result(result, query, requests, BeautifulSoup)
                        if document is not None:
                            documents.append(document)
        except Exception:
            return documents
        return documents


def _select_parent_chunks(documents: list[Document], limit: int, max_chunks_per_parent: int = 2) -> list[Document]:
    selected: list[Document] = []
    parent_counts: dict[str, int] = {}
    for document in documents:
        parent = document.metadata.get("parent_source", document.source)
        if parent_counts.get(parent, 0) >= max_chunks_per_parent:
            continue
        selected.append(document)
        parent_counts[parent] = parent_counts.get(parent, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def build_context(
    request,
    index: ExcelIndexer,
    enable_web: bool = True,
    limit: int = 8,
    internal_retriever: HybridRetriever | None = None,
    queries: Iterable[str] | None = None,
) -> str:
    query_list = list(queries or build_search_queries(request.control_action, request.source_controller, request.target))
    retriever = internal_retriever or HybridRetriever(index)
    internal = retriever.retrieve_many(query_list, limit=limit)
    web_sources = WebSearchRetriever(max_results=10).retrieve_sources(query_list) if enable_web else []
    web_corpus = (
        build_hierarchy(web_sources, index.hierarchy_config, index.summary_generator)
        if web_sources
        else HierarchicalCorpus(config=index.hierarchy_config)
    )
    web_documents = web_corpus.retrieval_documents()
    web = HybridRetriever(web_documents).retrieve_many(query_list, limit=limit * 2) if web_documents else []
    internal_context = "\n".join(document.text for document in internal)
    web_context = expand_retrieval_units(web, web_corpus) if web else ""
    context = "\n".join(part for part in (internal_context, web_context) if part)
    if context:
        return context
    selected = []
    if not selected:
        return request.context or "No supporting evidence was retrieved."
    return "\n".join(document.text for document in selected)
