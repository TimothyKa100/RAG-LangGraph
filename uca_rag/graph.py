from __future__ import annotations

import json
import hashlib
import re
from dataclasses import replace
from pathlib import Path
from typing import Callable, TypedDict

from .models import GeneratedUCAResponse, UCARequest, UCAResponse
from .prompts import GENERATION_SYSTEM_PROMPT, build_generation_prompt, build_search_queries
from .retrieval import ExcelIndexer, HybridRetriever, build_context
from .query_generation import GeneratedQueries, HuggingFaceQueryGenerator, QueryGenerator
from .hierarchy import HierarchyConfig, HuggingFaceSummaryGenerator
from .model_cache import cached_causal_lm, clear_causal_lm_cache
from .terminology import TerminologyLLM, TerminologyPipeline, TerminologyStore, TerminologyWebSearch


UCAState = TypedDict(
    "UCAState",
    {
        "request": dict,
        "control_action": str,
        "from": str,
        "to": str,
        "uca_id": str,
        "context": str,
        "response": dict,
        "terminology": list[dict],
        "generated_queries": dict,
    },
    total=False,
)


Generator = Callable[[UCARequest, str], dict | UCAResponse]


class QueryRecordStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def save(self, uca_id: str, request: UCARequest, queries: GeneratedQueries, model_name: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "queries.jsonl"
        record = {
            "uca_id": uca_id,
            "input": {"control_action": request.control_action, "from": request.source_controller, "to": request.target},
            "queries": queries.as_dict(),
            "query_count": len(queries.flatten()),
            "validation_status": "valid",
            "model": model_name,
        }
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_uca_graph(
    retriever,
    generator: Generator,
    *,
    enable_web: bool = True,
    internal_index=None,
    query_generator: QueryGenerator | None = None,
    query_categories: tuple[str, ...] = ("direct", "failure_mode", "relationship", "terminology"),
    query_record_store: QueryRecordStore | None = None,
    include_retrieval: bool = True,
    include_generation: bool = True,
    release_query_model: bool = True,
    release_generator: bool = True,
):
    from langgraph.graph import END, START, StateGraph

    internal_retriever = HybridRetriever(internal_index) if internal_index is not None else retriever

    def request_from_state(state: UCAState) -> UCARequest:
        payload = state.get("request") or state
        return UCARequest.model_validate(payload)

    def retrieve_context(state: UCAState) -> dict:
        request = request_from_state(state)
        try:
            generated_queries = query_generator.generate(request) if query_generator else None
        finally:
            if release_query_model and query_generator is not None:
                release = getattr(query_generator, "release", None)
                if release:
                    release()
        queries = generated_queries.flatten(query_categories) if generated_queries else build_search_queries(
            request.control_action, request.source_controller, request.target
        )
        if generated_queries and query_record_store:
            default_uca_id = hashlib.sha256(
                json.dumps(
                    {"control_action": request.control_action, "from": request.source_controller, "to": request.target},
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:16]
            query_record_store.save(
                str(state.get("uca_id") or state.get("request", {}).get("uca_id") or default_uca_id),
                request,
                generated_queries,
                query_generator.model_name,
            )
        if internal_index is not None:
            context = build_context(
                request, internal_index, enable_web=enable_web, internal_retriever=internal_retriever, queries=queries
            )
        else:
            internal = internal_retriever.retrieve_many(queries)
            context = "\n".join(item.text for item in internal) or request.context
        result = {"context": context or "No supporting evidence was retrieved."}
        if generated_queries:
            result["generated_queries"] = generated_queries.as_dict()
        return result

    def generate_ucas(state: UCAState) -> dict:
        request = request_from_state(state)
        try:
            result = generator(request, state["context"])
        finally:
            if release_generator:
                release = getattr(generator, "release", None)
                if release:
                    release()
        generated_response = result.model_dump(by_alias=True) if isinstance(result, UCAResponse) else result
        response = {**generated_response, "context": state["context"]}
        return {"response": response}

    workflow = StateGraph(UCAState)
    if include_retrieval:
        workflow.add_node("retrieve_context", retrieve_context)
    if include_generation:
        workflow.add_node("generate_ucas", generate_ucas)
    if include_retrieval and include_generation:
        workflow.add_edge(START, "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_ucas")
    elif include_retrieval:
        workflow.add_edge(START, "retrieve_context")
        workflow.add_edge("retrieve_context", END)
    elif include_generation:
        workflow.add_edge(START, "generate_ucas")
        workflow.add_edge("generate_ucas", END)
    else:
        raise ValueError("graph must include retrieval or generation")
    return workflow.compile()


def prompt_generator(request: UCARequest, context: str) -> dict:
    return {
        "system": GENERATION_SYSTEM_PROMPT,
        "user": build_generation_prompt(request.control_action, request.source_controller, request.target, context),
    }


def build_prompt_graph(internal_dir: str | None, enable_web: bool = False):
    index = ExcelIndexer(internal_dir).build()
    return build_uca_graph(None, prompt_generator, enable_web=enable_web, internal_index=index)


def local_generator(model_path: str = "outputs/fine-tune-merged", max_new_tokens: int = 768) -> Generator:
    tokenizer = None
    model = None

    def generate(request: UCARequest, context: str) -> UCAResponse:
        nonlocal tokenizer, model
        if tokenizer is None or model is None:
            tokenizer, model = cached_causal_lm(model_path)
        messages = [
            ("system", GENERATION_SYSTEM_PROMPT),
            ("user", build_generation_prompt(request.control_action, request.source_controller, request.target, context)),
        ]
        chat_messages = [{"role": role, "content": content} for role, content in messages]
        inputs = tokenizer.apply_chat_template(
            chat_messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        raw = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
        if match:
            candidate = match.group(1)
        else:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            candidate = match.group(0) if match else ""
        if not candidate:
            preview = raw[:1000].encode("unicode_escape", errors="backslashreplace").decode("ascii")
            raise ValueError(f"local UCA model did not return a JSON object; decoded output={preview!r}")
        try:
            generated = GeneratedUCAResponse.model_validate(json.loads(candidate))
        except (json.JSONDecodeError, ValueError) as error:
            preview = raw[:1000].encode("unicode_escape", errors="backslashreplace").decode("ascii")
            raise ValueError(f"local UCA model returned invalid UCA JSON; decoded output={preview!r}") from error
        return UCAResponse.model_validate({**generated.model_dump(by_alias=True), "context": context})

    def release() -> None:
        nonlocal tokenizer, model
        tokenizer = None
        model = None
        clear_causal_lm_cache()

    generate.release = release

    return generate


def build_default_graph(
    internal_dir: str | None,
    model: str = "outputs/fine-tune-merged",
    enable_web: bool = True,
    query_model: str | None = "meta-llama/Llama-3.2-3B-Instruct",
    query_record_dir: str | None = None,
    query_categories: tuple[str, ...] = ("direct", "failure_mode", "relationship", "terminology"),
    processed_dir: str | None = "data/processed",
    hierarchy_config: HierarchyConfig | None = None,
    summary_model: str | None = None,
):
    effective_hierarchy_config = hierarchy_config or HierarchyConfig.from_environment()
    if summary_model:
        effective_hierarchy_config = replace(effective_hierarchy_config, summary_model=summary_model)
    summary_generator = (
        HuggingFaceSummaryGenerator(effective_hierarchy_config.summary_model)
        if internal_dir is not None
        and (
            effective_hierarchy_config.enable_document_summaries
            or effective_hierarchy_config.enable_section_summaries
            or effective_hierarchy_config.enable_chunk_summaries
        )
        else None
    )
    index = ExcelIndexer(
        internal_dir,
        hierarchy_config=effective_hierarchy_config,
        processed_dir=processed_dir,
        summary_generator=summary_generator,
    ).build()
    if summary_generator is not None:
        release = getattr(summary_generator, "release", None)
        if release:
            release()
        index.summary_generator = None
        clear_causal_lm_cache()
    query_generator = HuggingFaceQueryGenerator(query_model) if query_model else None
    record_store = QueryRecordStore(query_record_dir) if query_record_dir else None
    return build_uca_graph(
        None,
        local_generator(model),
        enable_web=enable_web,
        internal_index=index,
        query_generator=query_generator,
        query_categories=query_categories,
        query_record_store=record_store,
    )


def build_default_batch_graphs(
    internal_dir: str | None,
    model: str = "outputs/fine-tune-merged",
    enable_web: bool = True,
    query_model: str | None = "meta-llama/Llama-3.2-3B-Instruct",
    query_record_dir: str | None = None,
    query_categories: tuple[str, ...] = ("direct", "failure_mode", "relationship", "terminology"),
    processed_dir: str | None = "data/processed",
    hierarchy_config: HierarchyConfig | None = None,
    summary_model: str | None = None,
):
    """Build retrieval and generation graphs for memory-bounded batch processing."""
    effective_hierarchy_config = hierarchy_config or HierarchyConfig.from_environment()
    if summary_model:
        effective_hierarchy_config = replace(effective_hierarchy_config, summary_model=summary_model)
    summary_generator = (
        HuggingFaceSummaryGenerator(effective_hierarchy_config.summary_model)
        if internal_dir is not None
        and (
            effective_hierarchy_config.enable_document_summaries
            or effective_hierarchy_config.enable_section_summaries
            or effective_hierarchy_config.enable_chunk_summaries
        )
        else None
    )
    index = ExcelIndexer(
        internal_dir,
        hierarchy_config=effective_hierarchy_config,
        processed_dir=processed_dir,
        summary_generator=summary_generator,
    ).build()
    if summary_generator is not None:
        release = getattr(summary_generator, "release", None)
        if release:
            release()
        index.summary_generator = None
        clear_causal_lm_cache()

    query_generator = HuggingFaceQueryGenerator(query_model) if query_model else None
    record_store = QueryRecordStore(query_record_dir) if query_record_dir else None
    final_generator = local_generator(model)
    retrieval_graph = build_uca_graph(
        None,
        lambda request, context: {},
        enable_web=enable_web,
        internal_index=index,
        query_generator=query_generator,
        query_categories=query_categories,
        query_record_store=record_store,
        include_generation=False,
        release_query_model=False,
    )
    generation_graph = build_uca_graph(
        None,
        final_generator,
        include_retrieval=False,
        release_generator=False,
    )

    def release_query_models() -> None:
        if query_generator is not None:
            query_release = getattr(query_generator, "release", None)
            if query_release:
                query_release()

    def release() -> None:
        release_query_models()
        final_release = getattr(final_generator, "release", None)
        if final_release:
            final_release()
        clear_causal_lm_cache()

    return retrieval_graph, generation_graph, release_query_models, release


def build_terminology_graph(
    llm: TerminologyLLM,
    store_path: str,
    *,
    web_search: TerminologyWebSearch | None = None,
):
    from langgraph.graph import END, START, StateGraph

    pipeline = TerminologyPipeline(llm, TerminologyStore(store_path), web_search)

    def discover_terminology(state: UCAState) -> dict:
        payload = state.get("request") or state
        request = UCARequest.model_validate(payload)
        entries = pipeline.run(request)
        return {"terminology": [entry.as_dict() for entry in entries]}

    workflow = StateGraph(UCAState)
    workflow.add_node("discover_terminology", discover_terminology)
    workflow.add_edge(START, "discover_terminology")
    workflow.add_edge("discover_terminology", END)
    return workflow.compile()


def build_default_terminology_graph(
    store_path: str = "data/terminology.json",
    model: str = "meta-llama/Llama-3.2-3B-Instruct",
):
    from .terminology import HuggingFaceTerminologyLLM

    return build_terminology_graph(HuggingFaceTerminologyLLM(model), store_path)
