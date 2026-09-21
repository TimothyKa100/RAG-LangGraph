from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

from .model_cache import cached_causal_lm, clear_causal_lm_cache


@dataclass(frozen=True)
class HierarchyConfig:
    semantic_chunk_min_tokens: int = 300
    semantic_chunk_target_tokens: int = 500
    semantic_chunk_max_tokens: int = 700
    retrieval_unit_min_tokens: int = 50
    retrieval_unit_target_tokens: int = 100
    retrieval_unit_max_tokens: int = 150
    chunk_overlap_tokens: int = 0
    enable_document_summaries: bool = False
    enable_section_summaries: bool = False
    enable_chunk_summaries: bool = False
    context_expansion_enabled: bool = True
    adjacent_units_before: int = 1
    adjacent_units_after: int = 1
    max_context_tokens: int = 5000
    summary_model: str = "meta-llama/Llama-3.2-1B-Instruct"

    @classmethod
    def from_environment(cls) -> "HierarchyConfig":
        values = {}
        for field_name, field_value in cls.__dataclass_fields__.items():
            raw = os.getenv(field_name.upper())
            if raw is None:
                continue
            default = field_value.default
            if isinstance(default, bool):
                values[field_name] = raw.lower() in {"1", "true", "yes", "on"}
            elif isinstance(default, str):
                values[field_name] = raw
            else:
                values[field_name] = int(raw)
        return cls(**values)


@dataclass
class HierarchyNode:
    id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)
    summary: str = ""

    def as_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "summary": self.summary, "metadata": self.metadata}


@dataclass
class HierarchicalCorpus:
    documents: list[HierarchyNode] = field(default_factory=list)
    sections: list[HierarchyNode] = field(default_factory=list)
    chunks: list[HierarchyNode] = field(default_factory=list)
    retrieval_units: list[HierarchyNode] = field(default_factory=list)
    config: HierarchyConfig = field(default_factory=HierarchyConfig)

    def nodes_by_id(self) -> dict[str, HierarchyNode]:
        return {node.id: node for node in self.documents + self.sections + self.chunks + self.retrieval_units}

    def retrieval_documents(self):
        from .retrieval import Document

        return [Document(unit.text, unit.metadata.get("source", unit.id), {**unit.metadata, "hierarchy_id": unit.id}) for unit in self.retrieval_units]

    def persist(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for name, nodes in (("documents", self.documents), ("sections", self.sections), ("chunks", self.chunks), ("retrieval_units", self.retrieval_units)):
            with (target / f"{name}.jsonl").open("w", encoding="utf-8") as file:
                for node in nodes:
                    file.write(json.dumps(node.as_dict(), ensure_ascii=False) + "\n")

    def statistics(self) -> dict[str, float | int]:
        return {
            "documents": len(self.documents),
            "sections": len(self.sections),
            "semantic_chunks": len(self.chunks),
            "retrieval_units": len(self.retrieval_units),
            "average_section_tokens": _average_tokens(self.sections),
            "average_chunk_tokens": _average_tokens(self.chunks),
            "average_retrieval_unit_tokens": _average_tokens(self.retrieval_units),
        }

    def debug_report(self) -> str:
        lines = []
        for document in self.documents:
            lines.append(f"DOCUMENT: {document.id}\nTITLE: {document.metadata.get('document_title', '')}")
            for section in [item for item in self.sections if item.metadata["document_id"] == document.id]:
                lines.append(f"\nSECTION: {section.id}\nTITLE: {section.metadata['section_title']}")
                for chunk in [item for item in self.chunks if item.metadata["section_id"] == section.id]:
                    lines.append(f"\nCHUNK: {chunk.id}\nTOKENS: {len(_words(chunk.text))}")
                    for unit in [item for item in self.retrieval_units if item.metadata["chunk_id"] == chunk.id]:
                        lines.append(f"\nUNIT: {unit.id}\nTOKENS: {len(_words(unit.text))}\nTEXT:\n{unit.text}")
                if section.summary:
                    lines.append(f"\nSECTION SUMMARY:\n{section.summary}")
        return "\n".join(lines)


class SummaryGenerator(Protocol):
    def summarize(self, text: str, title: str) -> str: ...


class HuggingFaceSummaryGenerator:
    """Generate summaries from the complete original text using the cached 1B model."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        max_new_tokens: int = 256,
        max_input_characters: int = 48000,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_input_characters = max_input_characters
        self.tokenizer, self.model = cached_causal_lm(model_name)

    def summarize(self, text: str, title: str) -> str:
        source_text = text[: self.max_input_characters]
        messages = [
            {
                "role": "system",
                "content": "Summarize the supplied source text accurately and concisely. Use only the source text and preserve important terminology.",
            },
            {"role": "user", "content": f"Full source text:\n{source_text}\n\nSummary:"},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()

    def release(self) -> None:
        self.tokenizer = None
        self.model = None
        clear_causal_lm_cache()


def build_hierarchy(
    source_documents: Iterable,
    config: HierarchyConfig | None = None,
    summary_generator: SummaryGenerator | None = None,
) -> HierarchicalCorpus:
    config = config or HierarchyConfig()
    corpus = HierarchicalCorpus(config=config)
    for document_index, source in enumerate(source_documents, start=1):
        document_id = f"doc_{document_index:03d}"
        title = source.metadata.get("title") or source.metadata.get("workbook") or Path(source.source).stem or document_id
        document = HierarchyNode(
            document_id,
            source.text,
            {**source.metadata, "document_id": document_id, "document_title": title, "source": source.source, "level": "document"},
            summary_generator.summarize(source.text, title) if config.enable_document_summaries and summary_generator else "",
        )
        corpus.documents.append(document)
        sections = _detect_sections(source.text)
        for section_index, (section_title, section_text) in enumerate(sections, start=1):
            section_id = f"{document_id}_sec_{section_index:03d}"
            section = HierarchyNode(
                section_id,
                section_text,
                {"document_id": document_id, "document_title": title, "section_title": section_title, "section_index": str(section_index), "level": "section"},
                summary_generator.summarize(section_text, section_title) if config.enable_section_summaries and summary_generator else "",
            )
            corpus.sections.append(section)
            section_chunks = _make_chunks(section_text, config)
            for chunk_index, chunk_text in enumerate(section_chunks, start=1):
                chunk_id = f"{section_id}_chunk_{chunk_index:03d}"
                chunk = HierarchyNode(
                    chunk_id,
                    chunk_text,
                    {"document_id": document_id, "section_id": section_id, "section_title": section_title, "chunk_index": str(chunk_index), "parent_id": section_id, "level": "chunk"},
                    summary_generator.summarize(chunk_text, section_title) if config.enable_chunk_summaries and summary_generator else "",
                )
                corpus.chunks.append(chunk)
                for unit_index, unit_text in enumerate(_make_units(chunk_text, config), start=1):
                    unit_id = f"{chunk_id}_unit_{unit_index:03d}"
                    corpus.retrieval_units.append(
                        HierarchyNode(
                            unit_id,
                            unit_text,
                            {"document_id": document_id, "document_title": title, "section_id": section_id, "section_title": section_title, "chunk_id": chunk_id, "chunk_index": str(chunk_index), "retrieval_unit_id": unit_id, "unit_index": str(unit_index), "parent_id": chunk_id, "source": source.source, "level": "retrieval_unit"},
                        )
                    )
    return corpus


def expand_retrieval_units(retrieved: list, corpus: HierarchicalCorpus, config: HierarchyConfig | None = None) -> str:
    config = config or corpus.config
    if not retrieved:
        return ""
    by_id = corpus.nodes_by_id()
    units_by_chunk: dict[str, list[HierarchyNode]] = {}
    for unit in corpus.retrieval_units:
        units_by_chunk.setdefault(unit.metadata["chunk_id"], []).append(unit)
    selected: list[HierarchyNode] = []
    seen: set[str] = set()
    for result in retrieved:
        unit_id = result.metadata.get("hierarchy_id") or result.metadata.get("retrieval_unit_id")
        unit = by_id.get(unit_id)
        if unit is None:
            unit = next((item for item in corpus.retrieval_units if item.text == result.text), None)
        if unit is None:
            continue
        chunk_id = unit.metadata["chunk_id"]
        siblings = units_by_chunk[chunk_id]
        unit_position = siblings.index(unit)
        candidates = [unit]
        if config.context_expansion_enabled:
            start = max(0, unit_position - config.adjacent_units_before)
            end = unit_position + config.adjacent_units_after + 1
            candidates.extend(siblings[start:end])
            candidates.append(by_id[chunk_id])
            section = by_id[unit.metadata["section_id"]]
            if section.summary:
                candidates.append(HierarchyNode(section.id + "_summary", section.summary, section.metadata))
        for candidate in candidates:
            if candidate.id not in seen:
                seen.add(candidate.id)
                selected.append(candidate)

    parts: list[str] = []
    token_count = 0
    for node in selected:
        text = node.summary if node.id.endswith("_summary") else node.text
        count = len(_words(text))
        if token_count + count > config.max_context_tokens:
            continue
        parts.append(text)
        token_count += count
    return "\n".join(parts)


def _detect_sections(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    headings = [(index, line.lstrip("# ").strip()) for index, line in enumerate(lines) if re.match(r"^\s{0,3}#{1,6}\s+\S", line)]
    if not headings:
        return [("Section 1", text.strip())]
    sections = []
    for heading_index, (line_index, title) in enumerate(headings):
        end = headings[heading_index + 1][0] if heading_index + 1 < len(headings) else len(lines)
        body = "\n".join(lines[line_index + 1:end]).strip()
        if body:
            sections.append((title, body))
    return sections or [("Section 1", text.strip())]


def _make_chunks(text: str, config: HierarchyConfig) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else []
    chunks, current = [], []
    current_count = 0
    for paragraph in paragraphs:
        paragraph_count = len(_words(paragraph))
        if current and current_count + paragraph_count > config.semantic_chunk_max_tokens:
            chunks.append("\n\n".join(current))
            overlap = current[-1] if config.chunk_overlap_tokens and len(_words(current[-1])) <= config.chunk_overlap_tokens else ""
            current, current_count = ([overlap] if overlap else []), len(_words(overlap))
        current.append(paragraph)
        current_count += paragraph_count
        if current_count >= config.semantic_chunk_target_tokens:
            chunks.append("\n\n".join(current))
            current, current_count = [], 0
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _make_units(text: str, config: HierarchyConfig) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if not sentences:
        return [text.strip()] if text.strip() else []
    units, current = [], []
    current_count = 0
    for sentence in sentences:
        count = len(_words(sentence))
        if current and current_count + count > config.retrieval_unit_max_tokens:
            units.append(" ".join(current))
            current, current_count = [], 0
        current.append(sentence)
        current_count += count
        if current_count >= config.retrieval_unit_target_tokens:
            units.append(" ".join(current))
            current, current_count = [], 0
    if current:
        units.append(" ".join(current))
    return units


def _words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text)


def _average_tokens(nodes: list[HierarchyNode]) -> float:
    return round(sum(len(_words(node.text)) for node in nodes) / len(nodes), 2) if nodes else 0.0


def _summary(text: str, title: str) -> str:
    words = _words(text)
    return f"{title}: {' '.join(words[:45])}" if words else title