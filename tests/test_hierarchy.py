from uca_rag.hierarchy import HierarchyConfig, build_hierarchy, expand_retrieval_units
from uca_rag.retrieval import Document, HybridRetriever


def sample_source() -> Document:
    return Document(
        text=(
            "# Control Structure\n\n"
            "The controller sends a control action to the controlled process. "
            "The controlled process changes state and returns feedback to the controller.\n\n"
            "# Safety Constraints\n\n"
            "The control action must be provided at the correct time. "
            "The controller must preserve the safety constraint during operation."
        ),
        source="stpa.md",
        metadata={"title": "Example STPA Document"},
    )


def test_hierarchy_has_sections_chunks_units_and_parent_relationships():
    config = HierarchyConfig(semantic_chunk_target_tokens=10, semantic_chunk_max_tokens=30, retrieval_unit_target_tokens=10)
    corpus = build_hierarchy([sample_source()], config)

    assert len(corpus.documents) == 1
    assert [section.metadata["section_title"] for section in corpus.sections] == ["Control Structure", "Safety Constraints"]
    assert corpus.chunks
    assert corpus.retrieval_units
    sections = {section.id for section in corpus.sections}
    chunks = {chunk.id for chunk in corpus.chunks}
    documents = {document.id for document in corpus.documents}
    assert all(unit.metadata["parent_id"] in chunks for unit in corpus.retrieval_units)
    assert all(chunk.metadata["parent_id"] in sections for chunk in corpus.chunks)
    assert all(section.metadata["document_id"] in documents for section in corpus.sections)


def test_hierarchy_ids_are_deterministic_and_source_text_is_preserved():
    config = HierarchyConfig(semantic_chunk_target_tokens=10, semantic_chunk_max_tokens=30, retrieval_unit_target_tokens=10)
    first = build_hierarchy([sample_source()], config)
    second = build_hierarchy([sample_source()], config)

    assert [unit.id for unit in first.retrieval_units] == [unit.id for unit in second.retrieval_units]
    source = sample_source().text
    assert all(unit.text in source for unit in first.retrieval_units)


def test_hierarchy_persistence_and_statistics(tmp_path):
    corpus = build_hierarchy([sample_source()])
    corpus.persist(tmp_path)

    assert (tmp_path / "documents.jsonl").exists()
    assert (tmp_path / "sections.jsonl").exists()
    assert (tmp_path / "chunks.jsonl").exists()
    assert (tmp_path / "retrieval_units.jsonl").exists()
    assert corpus.statistics()["retrieval_units"] >= 1


def test_configured_summary_generator_receives_full_source_text():
    class FakeSummaryGenerator:
        def __init__(self):
            self.calls = []

        def summarize(self, text, title):
            self.calls.append((text, title))
            return f"Summary for {title}"

    source = sample_source()
    generator = FakeSummaryGenerator()
    config = HierarchyConfig(enable_document_summaries=True, enable_section_summaries=True, enable_chunk_summaries=True)
    corpus = build_hierarchy([source], config, generator)

    assert generator.calls[0] == (source.text, "Example STPA Document")
    assert corpus.documents[0].summary == "Summary for Example STPA Document"
    assert all(node.summary for node in corpus.sections + corpus.chunks)


def test_expanded_context_contains_text_without_hierarchy_metadata():
    config = HierarchyConfig(
        semantic_chunk_target_tokens=10,
        semantic_chunk_max_tokens=30,
        retrieval_unit_target_tokens=10,
    )
    corpus = build_hierarchy([sample_source()], config)
    result = corpus.retrieval_documents()[0]
    context = expand_retrieval_units([result], corpus, config)

    assert "#" not in context
    assert "doc_001" not in context
    assert "retrieval_unit" not in context
    assert result.text in context


def test_expansion_deduplicates_parent_context_and_obeys_budget():
    config = HierarchyConfig(
        semantic_chunk_target_tokens=10,
        semantic_chunk_max_tokens=30,
        retrieval_unit_target_tokens=10,
        max_context_tokens=24,
        enable_section_summaries=True,
    )
    corpus = build_hierarchy([sample_source()], config)
    indexed = corpus.retrieval_documents()
    retriever = HybridRetriever(indexed)
    retrieved = retriever.retrieve_many(["controller control action", "safety constraint"], limit=2)
    context = expand_retrieval_units(retrieved + retrieved, corpus, config)

    assert context
    assert context.count("chunk_") >= 1
    assert len(context.split()) <= 24 * 2
