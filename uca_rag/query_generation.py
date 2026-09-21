from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .models import UCARequest
from .model_cache import cached_causal_lm, clear_causal_lm_cache

logger = logging.getLogger(__name__)


QUERY_GENERATION_SYSTEM_PROMPT = """You are a search-query generation model for Systems-Theoretic Process Analysis (STPA).

Your task is to transform a structured STPA control action into a set of high-quality web search queries that can retrieve technical literature, documentation, examples, and explanations relevant to the control relationship.

INPUT

You will receive:

{"control_action": "", "from": "<source/controller>", "to": "<destination/controlled process>"}

IMPORTANT CONSTRAINTS

- The input contains ONLY the control action, source, and destination.
- Do not invent specific system behaviour, hazards, consequences, components, or technical details that are not supported by the input.
- You may introduce reasonable synonyms, abbreviations, and general STPA terminology.
- Preserve the meaning of the original control action and the relationship between "from" and "to".
- Queries should be suitable for a web search engine, not natural-language questions to an LLM.
- Prefer concise technical search queries.
- Include "STPA" or relevant STPA terminology where appropriate.
- Do not make every query a simple paraphrase of the original input.

Generate exactly 2 direct queries, 1 query for each of the six failure modes (Not Provided, Provided Incorrectly, Provided But Not Needed, Provided Too Early, Provided Too Late, Stopped Providing Too Soon), 3 relationship queries, and 2 terminology-diversified queries. Consider all six failure modes without assuming any is the actual category.

Return ONLY valid JSON with exactly these keys: direct_queries, failure_mode_queries, relationship_queries, terminology_queries. The failure_mode_queries keys are: not_provided, provided_incorrectly, provided_but_not_needed, provided_too_early, provided_too_late, stopped_providing_too_soon."""

FAILURE_MODE_KEYS = (
    "not_provided",
    "provided_incorrectly",
    "provided_but_not_needed",
    "provided_too_early",
    "provided_too_late",
    "stopped_providing_too_soon",
)
QUERY_CATEGORIES = ("direct", "failure_mode", "relationship", "terminology")


@dataclass
class GeneratedQueries:
    direct_queries: list[str] = field(default_factory=list)
    failure_mode_queries: dict[str, list[str]] = field(default_factory=dict)
    relationship_queries: list[str] = field(default_factory=list)
    terminology_queries: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "direct_queries": self.direct_queries,
            "failure_mode_queries": self.failure_mode_queries,
            "relationship_queries": self.relationship_queries,
            "terminology_queries": self.terminology_queries,
        }

    def flatten(self, enabled_categories: tuple[str, ...] = QUERY_CATEGORIES) -> list[str]:
        queries: list[str] = []
        if "direct" in enabled_categories:
            queries.extend(self.direct_queries)
        if "failure_mode" in enabled_categories:
            queries.extend(query for values in self.failure_mode_queries.values() for query in values)
        if "relationship" in enabled_categories:
            queries.extend(self.relationship_queries)
        if "terminology" in enabled_categories:
            queries.extend(self.terminology_queries)
        return _unique_queries(queries)


class QueryGenerationError(ValueError):
    pass


class QueryGenerator(Protocol):
    model_name: str

    def generate(self, request: UCARequest) -> GeneratedQueries: ...


def validate_generated_queries(value: Any) -> GeneratedQueries:
    if not isinstance(value, dict):
        raise QueryGenerationError("query generation output must be a JSON object")
    required = {"direct_queries", "failure_mode_queries", "relationship_queries", "terminology_queries"}
    if set(value) != required:
        raise QueryGenerationError(f"query generation output keys must be exactly {sorted(required)}")
    failure_modes = value["failure_mode_queries"]
    if not isinstance(failure_modes, dict) or set(failure_modes) != set(FAILURE_MODE_KEYS):
        raise QueryGenerationError("all six failure-mode query keys are required")

    try:
        result = GeneratedQueries(
            direct_queries=_clean_query_list(value["direct_queries"], "direct_queries"),
            failure_mode_queries={key: _clean_query_list(failure_modes[key], key) for key in FAILURE_MODE_KEYS},
            relationship_queries=_clean_query_list(value["relationship_queries"], "relationship_queries"),
            terminology_queries=_clean_query_list(value["terminology_queries"], "terminology_queries"),
        )
    except (TypeError, QueryGenerationError) as error:
        logger.warning("query generation validation failed: %s", error)
        raise
    if not result.flatten():
        raise QueryGenerationError("query generation produced no usable queries")
    return result


def _clean_query_list(value: Any, field_name: str) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise QueryGenerationError(f"{field_name} must be an array")
    cleaned = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            logger.warning("discarding malformed query in %s", field_name)
            continue
        query = " ".join(item.split())
        if query not in cleaned:
            cleaned.append(query)
    return cleaned


def _unique_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for query in queries:
        key = query.casefold()
        if key not in seen:
            seen.add(key)
            result.append(query)
    return result


def parse_query_json(text: str) -> GeneratedQueries:
    candidate = text.strip()
    if not candidate:
        logger.warning("query generation returned empty output")
        raise QueryGenerationError("query generation returned empty output")
    if not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as error:
        logger.warning("query generation returned malformed JSON: %s", error)
        raise QueryGenerationError("query generation returned malformed JSON") from error
    return validate_generated_queries(value)


class HuggingFaceQueryGenerator:
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    def __init__(self, model_name: str = model_name, temperature: float = 0.0, max_new_tokens: int = 768, **load_kwargs: Any):
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.tokenizer, self.model = cached_causal_lm(model_name)

    def generate(self, request: UCARequest) -> GeneratedQueries:
        payload = json.dumps(
            {"control_action": request.control_action, "from": request.source_controller, "to": request.target},
            ensure_ascii=False,
        )
        messages = [
            {"role": "system", "content": QUERY_GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "min_new_tokens": 32,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
        if self.tokenizer.pad_token_id is not None:
            generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        outputs = self.model.generate(**inputs, **generation_kwargs)
        text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return parse_query_json(text)

    def release(self) -> None:
        self.tokenizer = None
        self.model = None
        clear_causal_lm_cache()