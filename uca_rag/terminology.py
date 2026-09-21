from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .models import UCARequest
from .model_cache import cached_causal_lm
from .terminology_prompts import TERMINOLOGY_SELECTION_SYSTEM_PROMPT
from .retrieval import Document, WebSearchRetriever


@dataclass
class TerminologyDecision:
    search_terms: list[str]
    query: str
    existing_title_index: int | None = None
    merged_title: str | None = None


@dataclass
class TerminologyEntry:
    title: str
    summary: str

    def as_dict(self) -> dict[str, str]:
        return {"title": self.title, "summary": self.summary}


class TerminologyLLM(Protocol):
    def decide(self, request: UCARequest, titles: list[str]) -> TerminologyDecision: ...

    def summarize(self, title: str, content: str) -> str: ...


class TerminologyStore:
    """A small JSON store for titles and their human-readable summaries."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> list[TerminologyEntry]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as file:
            raw_entries = json.load(file)
        if not isinstance(raw_entries, list):
            raise ValueError("terminology store must contain a JSON array")
        return [TerminologyEntry(str(item["title"]), str(item.get("summary", ""))) for item in raw_entries]

    def save(self, entries: list[TerminologyEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump([entry.as_dict() for entry in entries], file, indent=2, ensure_ascii=False)
            file.write("\n")
        temporary_path.replace(self.path)


class TerminologyWebSearch:
    def __init__(self, max_results: int = 10, timeout: float = 10.0):
        self.searcher = WebSearchRetriever(max_results=max_results, timeout=timeout)

    def search(self, query: str) -> list[Document]:
        return self.searcher.retrieve_sources([query])


class TerminologyPipeline:
    def __init__(self, llm: TerminologyLLM, store: TerminologyStore, web_search: TerminologyWebSearch | None = None):
        self.llm = llm
        self.store = store
        self.web_search = web_search or TerminologyWebSearch(max_results=10)

    def run(self, request: UCARequest) -> list[TerminologyEntry]:
        entries = self.store.load()
        titles = [entry.title for entry in entries]
        decision = self.llm.decide(request, titles)

        if decision.existing_title_index is not None:
            if not 0 <= decision.existing_title_index < len(entries):
                raise ValueError("LLM returned an invalid existing title index")
            if decision.merged_title:
                entries[decision.existing_title_index].title = decision.merged_title
            self.store.save(entries)
            return entries

        for term in _unique_terms(
            [request.control_action, request.source_controller, request.target]
        ):
            documents = self.web_search.search(term)
            content = _content_for_summary(documents)
            if not content:
                continue
            entries.append(TerminologyEntry(title=term, summary=self.llm.summarize(term, content)))
        self.store.save(entries)
        return entries


def _unique_terms(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for term in terms:
        normalized = term.strip()
        if normalized and normalized.casefold() not in seen:
            seen.add(normalized.casefold())
            result.append(normalized)
    return result


def _content_for_summary(documents: list[Document], max_characters: int = 80_000) -> str:
    parts: list[str] = []
    characters = 0
    for document in documents:
        part = document.text
        remaining = max_characters - characters
        if remaining <= 0:
            break
        parts.append(part[:remaining])
        characters += len(part)
    return "\n\n".join(parts)


class HuggingFaceTerminologyLLM:
    """Local Llama adapter. It creates a fresh chat for selection and summary."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct", **load_kwargs: Any):
        self.tokenizer, self.model = cached_causal_lm(model_name)

    def _invoke(self, system: str, user: str, max_new_tokens: int = 512) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

    def decide(self, request: UCARequest, titles: list[str]) -> TerminologyDecision:
        prompt = (
            "Identify only technical or domain-specific terms in Control Action, From, and To that benefit "
            "from web research. Exclude ordinary words such as driver. Prefer the obscure source/target "
            "system names. Build one concise search query. Compare against the existing titles and detect "
            "an equivalent title including abbreviations, expanded names, spelling variants, and aliases. "
            "Return JSON only with search_terms (array), query (string), existing_title_index (integer or null), "
            "and merged_title (string or null). Existing titles are names only; do not infer from summaries.\n\n"
            f"Control Action: {request.control_action}\nFrom: {request.source_controller}\nTo: {request.target}\n"
            f"Existing titles: {json.dumps(titles, ensure_ascii=False)}"
        )
        data = _parse_json(self._invoke(TERMINOLOGY_SELECTION_SYSTEM_PROMPT, prompt))
        return TerminologyDecision(
            search_terms=[str(item) for item in data.get("search_terms", [])],
            query=str(data.get("query", "")),
            existing_title_index=data.get("existing_title_index"),
            merged_title=data.get("merged_title"),
        )

    def summarize(self, title: str, content: str) -> str:
        prompt = (
            f"Term: {title}\n\nWeb content:\n{content}\n\n"
            "Write a concise, factual technical summary of this term based only on the supplied content. "
            "Mention conflicting or missing information when relevant."
        )
        return self._invoke("You summarize retrieved technical web content.", prompt, max_new_tokens=768)


def _parse_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("terminology LLM did not return a JSON object")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("terminology LLM returned a JSON value other than an object")
    return value