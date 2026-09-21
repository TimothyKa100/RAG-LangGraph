from pathlib import Path

import pandas as pd

from uca_rag import graph as graph_module
from uca_rag.graph import QueryRecordStore, build_uca_graph
from uca_rag.models import GeneratedUCAResponse, UCARequest, diagnose_jsonl_line, request_from_jsonl_line
from uca_rag.prompts import GENERATION_SYSTEM_PROMPT, build_generation_prompt
from uca_rag.query_generation import GeneratedQueries
from uca_rag.retrieval import ExcelIndexer, HybridRetriever, WebSearchRetriever


def test_generation_prompt_keeps_evidence_guidance_with_runtime_context() -> None:
    prompt = build_generation_prompt("Activate Warning", "ALKS", "Other Vehicle Systems", "warning evidence")

    assert "warning evidence" in prompt
    assert "Use a precise context from the evidence." in prompt
    assert '"Control Action": "Activate Warning"' in prompt
    assert "input JSON below is the only control action to analyze" in prompt
    assert "context:h" not in prompt
    assert "warning evidence" not in GENERATION_SYSTEM_PROMPT
    assert "- context" not in GENERATION_SYSTEM_PROMPT


def test_generated_response_schema_does_not_require_context() -> None:
    generated = GeneratedUCAResponse()

    assert "context" not in generated.model_dump()


def test_request_is_extracted_from_training_jsonl_line() -> None:
    line = '{"messages": [{"role": "system", "content": "STPA instructions"}, {"role": "user", "content": "\\"Control Action\\": \\"Activate Hazard Warning\\", \\"From\\": \\"Automated Lane Keeping System\\", \\"To\\": \\"Other Vehicle Systems\\""}, {"role": "assistant", "content": "{}"}]}'

    request = request_from_jsonl_line(line)

    assert request.control_action == "Activate Hazard Warning"
    assert request.source_controller == "Automated Lane Keeping System"
    assert request.target == "Other Vehicle Systems"


def test_jsonl_diagnostics_do_not_include_record_content() -> None:
    line = '{"messages": [{"role": "user", "content": "\\"Control Action\\": \\"Activate Hazard Warning\\", \\"From\\": \\"ALKS\\", \\"To\\": \\"Other Vehicle Systems\\""}]}'

    diagnostic = diagnose_jsonl_line(line, 1)

    assert diagnostic["ok"] is True
    assert diagnostic["stage"] == "wrapped user content JSON"
    assert "Activate Hazard Warning" not in str(diagnostic)


def test_empty_request_fields_are_allowed() -> None:
    request = UCARequest.model_validate({"control_action": "", "from": "", "to": ""})

    assert request.control_action == ""
    assert request.source_controller == ""
    assert request.target == ""


def test_web_result_fetches_and_chunks_full_page() -> None:
    class FakeResponse:
        content = b"<html><body><article>" + (b"Accident report detail. " * 100) + b"</article></body></html>"

        def raise_for_status(self) -> None:
            return None

    class FakeRequests:
        @staticmethod
        def get(*args, **kwargs):
            return FakeResponse()

    from bs4 import BeautifulSoup

    documents = WebSearchRetriever(chunk_size=100, chunk_overlap=20)._documents_from_result(
        {"href": "https://example.com/report", "title": "Accident report", "body": "short snippet"},
        "accident report",
        FakeRequests,
        BeautifulSoup,
    )

    assert len(documents) > 1
    assert all(document.metadata["parent_source"] == "https://example.com/report" for document in documents)
    assert documents[0].source.endswith("#chunk-0")


def test_web_source_preserves_headings_for_in_memory_hierarchy():
    class FakeResponse:
        content = b"""<html><body><article>
        <h1>System Description</h1>
        <p>The controller sends a control action to the controlled process.</p>
        <h2>Safety Constraints</h2>
        <p>The action must be provided at the correct time.</p>
        </article></body></html>"""

        def raise_for_status(self) -> None:
            return None

    class FakeRequests:
        @staticmethod
        def get(*args, **kwargs):
            return FakeResponse()

    from bs4 import BeautifulSoup
    from uca_rag.hierarchy import build_hierarchy

    searcher = WebSearchRetriever()
    source = searcher._source_document_from_result(
        {"href": "https://example.com/stpa", "title": "STPA", "body": "short"},
        "STPA controller",
        FakeRequests,
        BeautifulSoup,
    )
    corpus = build_hierarchy([source])

    assert source is not None
    assert [section.metadata["section_title"] for section in corpus.sections] == ["System Description", "Safety Constraints"]
    assert all(unit.metadata["source"] == "https://example.com/stpa" for unit in corpus.retrieval_units)


def test_empty_internal_index_is_supported() -> None:
    index = ExcelIndexer(None).build()
    assert not index.documents
    assert HybridRetriever(index).retrieve("any query") == []


def test_web_only_graph_does_not_load_summary_model(monkeypatch) -> None:
    loaded_summary_models = []

    class FakeSummaryGenerator:
        def __init__(self, model_name):
            loaded_summary_models.append(model_name)

    class FakeQueryGenerator:
        model_name = "query-model"

        def __init__(self, model_name):
            self.model_name = model_name

    monkeypatch.setattr(graph_module, "HuggingFaceSummaryGenerator", FakeSummaryGenerator)
    monkeypatch.setattr(graph_module, "HuggingFaceQueryGenerator", FakeQueryGenerator)
    monkeypatch.setattr(graph_module, "local_generator", lambda model: lambda request, context: {})

    graph_module.build_default_graph(
        None,
        model="generation-model",
        query_model="query-model",
        summary_model="summary-model",
        enable_web=False,
    )

    assert loaded_summary_models == []


def test_hybrid_retriever_accepts_multiple_queries() -> None:
    documents = [
        {"text": "ALKS warning timing requirement", "source": "one"},
        {"text": "radio frequency communication", "source": "two"},
    ]
    from uca_rag.retrieval import Document

    retriever = HybridRetriever([Document(**document) for document in documents])

    results = retriever.retrieve_many(["warning timing", "radio frequency"])

    assert {document.source for document in results} == {"one", "two"}


def test_retrieved_context_contains_text_without_document_metadata() -> None:
    from uca_rag.retrieval import Document, build_context

    document = Document(
        "Evidence text only",
        "https://example.test/evidence",
        {"title": "Evidence title", "query": "internal query"},
    )
    index = ExcelIndexer(None).build()
    context = build_context(
        UCARequest(control_action="Action", **{"from": "Source", "to": "Target"}),
        index,
        enable_web=False,
        internal_retriever=HybridRetriever([document]),
        queries=["evidence"],
    )

    assert context == "Evidence text only"
    assert "https://example.test/evidence" not in context
    assert "Evidence title" not in context
    assert "internal query" not in context


def test_excel_sheets_are_indexed_and_context_reaches_generator(tmp_path: Path) -> None:
    workbook = tmp_path / "incidents.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame(
            {"Scenario": ["A following vehicle approaches a stopped ALKS vehicle."], "Finding": ["Hazard warning was activated too late."]}
        ).to_excel(writer, sheet_name="Accidents", index=False)
        pd.DataFrame({"Rule": ["The warning should be generated within 5 seconds."]}).to_excel(
            writer, sheet_name="Requirements", index=False
        )

    index = ExcelIndexer(tmp_path).build()
    retriever = HybridRetriever(index)
    seen = {}

    def generator(request, context):
        seen["context"] = context
        return {"context": context, "Not Provided": [], "Provided Incorrectly": [], "Provided but Not Needed": [], "Provided too Early": [], "Provided too Late": ["evidence-backed UCA"], "Provided too Long": [], "Stopped Providing too Soon": []}

    graph = build_uca_graph(retriever, generator, enable_web=False)
    result = graph.invoke(UCARequest(control_action="Activate Hazard Warning", **{"from": "ALKS", "to": "Other Vehicle Systems"}).model_dump(by_alias=True))

    assert "activated too late" in seen["context"].lower()
    assert result["response"]["Provided too Late"] == ["evidence-backed UCA"]


def test_retrieved_context_is_added_without_mutating_generator_response() -> None:
    generated_response = {"Not Provided": [], "Provided too Late": ["generated UCA"]}

    def generator(request, context):
        return generated_response

    graph = build_uca_graph(HybridRetriever(ExcelIndexer(None).build()), generator, enable_web=False)
    result = graph.invoke(
        UCARequest(control_action="Activate Warning", **{"from": "ALKS", "to": "Other Vehicle Systems"}).model_dump(
            by_alias=True
        )
    )

    assert result["response"]["context"] == "No supporting evidence was retrieved."
    assert "context" not in generated_response


def test_generated_queries_are_passed_to_existing_retriever_and_saved(tmp_path: Path) -> None:
    class FakeQueryGenerator:
        model_name = "test-query-model"

        def generate(self, request):
            assert request.control_action == "Activate Warning"
            return GeneratedQueries(
                direct_queries=["activate warning ALKS"],
                failure_mode_queries={key: [f"{key} activate warning"] for key in (
                    "not_provided", "provided_incorrectly", "provided_but_not_needed",
                    "provided_too_early", "provided_too_late", "stopped_providing_too_soon",
                )},
                relationship_queries=["ALKS controller controlled process"],
                terminology_queries=["ALKS safety constraint"],
            )

    class FakeRetriever:
        def __init__(self):
            self.queries = None

        def retrieve_many(self, queries, limit=8):
            self.queries = list(queries)
            return []

    retriever = FakeRetriever()
    graph = build_uca_graph(
        retriever,
        lambda request, context: {"Not Provided": []},
        enable_web=False,
        query_generator=FakeQueryGenerator(),
        query_record_store=QueryRecordStore(tmp_path),
    )
    result = graph.invoke({"uca_id": "uca-1", "request": {"control_action": "Activate Warning", "from": "ALKS", "to": "Vehicle"}})

    assert len(retriever.queries) == 9
    assert len(result["generated_queries"]["failure_mode_queries"]) == 6
    saved = (tmp_path / "queries.jsonl").read_text(encoding="utf-8")
    assert '"uca_id": "uca-1"' in saved
    assert '"model": "test-query-model"' in saved
