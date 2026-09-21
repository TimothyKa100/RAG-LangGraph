import json

import pytest

from uca_rag.models import UCARequest
from uca_rag.query_generation import (
    FAILURE_MODE_KEYS,
    QueryGenerationError,
    QUERY_GENERATION_SYSTEM_PROMPT,
    parse_query_json,
    validate_generated_queries,
)
from uca_rag.model_cache import load_causal_lm


def valid_payload():
    return {
        "direct_queries": ["direct query", "direct query", "  ", 4],
        "failure_mode_queries": {key: [f"{key} query"] for key in FAILURE_MODE_KEYS},
        "relationship_queries": ["relationship query"],
        "terminology_queries": ["terminology query"],
    }


def test_valid_json_has_all_failure_modes_and_removes_duplicates_and_empty_values():
    result = validate_generated_queries(valid_payload())

    assert set(result.failure_mode_queries) == set(FAILURE_MODE_KEYS)
    assert result.direct_queries == ["direct query"]
    assert len(result.flatten()) == 9


def test_single_query_strings_are_normalized_to_arrays():
    payload = valid_payload()
    payload["failure_mode_queries"]["not_provided"] = "single query"

    result = validate_generated_queries(payload)

    assert result.failure_mode_queries["not_provided"] == ["single query"]


def test_malformed_json_is_rejected():
    with pytest.raises(QueryGenerationError):
        parse_query_json("not json")


def test_query_json_can_be_wrapped_in_explanatory_text():
    result = parse_query_json(f"```json\n{json.dumps(valid_payload())}\n```")

    assert result.direct_queries == ["direct query"]


def test_empty_query_generation_is_rejected_explicitly():
    with pytest.raises(QueryGenerationError, match="empty output"):
        parse_query_json("   ")


def test_missing_failure_mode_is_rejected():
    payload = valid_payload()
    payload["failure_mode_queries"].pop("provided_too_late")

    with pytest.raises(QueryGenerationError):
        validate_generated_queries(payload)


def test_query_prompt_only_requests_structured_uca_fields():
    assert '"control_action"' in QUERY_GENERATION_SYSTEM_PROMPT
    assert '"from"' in QUERY_GENERATION_SYSTEM_PROMPT
    assert '"to"' in QUERY_GENERATION_SYSTEM_PROMPT
    assert "Not Provided" in QUERY_GENERATION_SYSTEM_PROMPT


def test_model_cache_is_keyed_by_model_and_reuses_loaded_pair(monkeypatch):
    load_causal_lm.cache_clear()
    calls = []

    class FakeTokenizer:
        pass

    class FakeModel:
        def eval(self):
            return self

    monkeypatch.setitem(__import__("sys").modules, "transformers", type("Transformers", (), {
        "AutoTokenizer": type("TokenizerLoader", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: calls.append(("tokenizer", args[0])) or FakeTokenizer())}),
        "AutoModelForCausalLM": type("ModelLoader", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: calls.append(("model", args[0])) or FakeModel())}),
    }))

    first = load_causal_lm("test-model")
    second = load_causal_lm("test-model")

    assert first is second
    assert calls == [("tokenizer", "test-model"), ("model", "test-model")]