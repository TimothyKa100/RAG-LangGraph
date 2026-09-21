import json

from uca_rag.models import UCARequest
from uca_rag.terminology import Document, TerminologyDecision, TerminologyEntry, TerminologyPipeline, TerminologyStore


class FakeLLM:
    def __init__(self, decision):
        self.decision = decision
        self.titles = None
        self.summaries = []

    def decide(self, request, titles):
        self.titles = titles
        return self.decision

    def summarize(self, title, content):
        self.summaries.append((title, content))
        return f"Summary of {title}"


class FakeSearch:
    def __init__(self):
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return [
            Document(
                "Technical details",
                "https://example.test/term#chunk-0",
                {"parent_source": "https://example.test/term", "title": "Term page"},
            )
        ]


def test_new_term_is_searched_and_stored_without_embeddings(tmp_path):
    llm = FakeLLM(TerminologyDecision(["ignored by the pipeline"], "ignored query"))
    search = FakeSearch()
    store = TerminologyStore(tmp_path / "terminology.json")

    entries = TerminologyPipeline(llm, store, search).run(
        UCARequest(control_action="RadioFrequencyChange", **{"from": "ATCTower", "to": "Flight Handling"})
    )

    assert llm.titles == []
    assert search.queries == ["RadioFrequencyChange", "ATCTower", "Flight Handling"]
    assert [entry.title for entry in entries] == ["RadioFrequencyChange", "ATCTower", "Flight Handling"]
    assert all(summary_content == "Technical details" for _, summary_content in llm.summaries)
    assert json.loads((tmp_path / "terminology.json").read_text()) == [
        {"title": "RadioFrequencyChange", "summary": "Summary of RadioFrequencyChange"},
        {"title": "ATCTower", "summary": "Summary of ATCTower"},
        {"title": "Flight Handling", "summary": "Summary of Flight Handling"},
    ]


def test_equivalent_title_is_updated_without_duplicate_search(tmp_path):
    store = TerminologyStore(tmp_path / "terminology.json")
    store.save([TerminologyEntry("Automated Lane Keeping System", "existing")])
    llm = FakeLLM(TerminologyDecision([], "", existing_title_index=0, merged_title="ALKS, Automated Lane Keeping System"))
    search = FakeSearch()

    entries = TerminologyPipeline(llm, store, search).run(UCARequest(control_action="driver", **{"from": "ALKS", "to": "vehicle"}))

    assert llm.titles == ["Automated Lane Keeping System"]
    assert search.queries == []
    assert entries[0].title == "ALKS, Automated Lane Keeping System"