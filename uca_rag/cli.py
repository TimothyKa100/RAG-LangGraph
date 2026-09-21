from __future__ import annotations

import argparse
import json

from .graph import build_default_batch_graphs, build_default_graph, build_default_terminology_graph, build_prompt_graph
from .hierarchy import HierarchyConfig
from .models import UCARequest, diagnose_jsonl_line, request_from_jsonl_line


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate STPA UCA statements with hybrid RAG")
    parser.add_argument("--internal-dir", default="data/internal")
    parser.add_argument("--no-internal", action="store_true", help="Skip Excel retrieval and use web retrieval only")
    parser.add_argument("--input-jsonl", help="Process each training-style JSONL record")
    parser.add_argument("--control-action")
    parser.add_argument("--from", dest="source_controller")
    parser.add_argument("--to", dest="target")
    parser.add_argument("--context", default="")
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--prompt-only", action="store_true", help="Output the prompt without calling the local generation model")
    parser.add_argument("--debug-jsonl", action="store_true", help="Validate JSONL locally without printing record contents")
    parser.add_argument("--terminology", action="store_true", help="Run the standalone obscure-terminology search system")
    parser.add_argument("--terminology-store", default="data/terminology.json")
    parser.add_argument("--terminology-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--query-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--generation-model", default="outputs/fine-tune-merged")
    parser.add_argument("--summary-model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--query-record-dir", default="data/query_records")
    parser.add_argument("--query-categories", default="direct,failure_mode,relationship,terminology")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--no-context-expansion", action="store_true")
    args = parser.parse_args()
    if args.debug_jsonl:
        if not args.input_jsonl:
            parser.error("--debug-jsonl requires --input-jsonl")
        with open(args.input_jsonl, encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if line.strip():
                    print(json.dumps(diagnose_jsonl_line(line, line_number), ensure_ascii=True))
        return
    if args.input_jsonl:
        requests = []
        with open(args.input_jsonl, encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if line.strip():
                    try:
                        requests.append(request_from_jsonl_line(line))
                    except (ValueError, json.JSONDecodeError) as error:
                        parser.error(f"invalid JSONL record on line {line_number}: {error}")
    elif all((args.control_action, args.source_controller, args.target)):
        requests = [UCARequest(control_action=args.control_action, source_controller=args.source_controller, target=args.target, context=args.context)]
    else:
        parser.error("provide --input-jsonl or all of --control-action, --from, and --to")

    if args.terminology:
        graph = build_default_terminology_graph(args.terminology_store, args.terminology_model)
        for request in requests:
            result = graph.invoke({"request": request.model_dump(by_alias=True)})
            print(json.dumps(result["terminology"], indent=None if args.input_jsonl else 2, ensure_ascii=False))
        return

    if args.input_jsonl and not args.prompt_only:
        retrieval_graph, generation_graph, release_query_models, release_models = build_default_batch_graphs(
            None if args.no_internal else args.internal_dir,
            model=args.generation_model,
            enable_web=not args.no_web,
            query_model=args.query_model,
            summary_model=args.summary_model,
            query_record_dir=args.query_record_dir,
            query_categories=tuple(filter(None, args.query_categories.split(","))),
            processed_dir=args.processed_dir,
            hierarchy_config=HierarchyConfig(context_expansion_enabled=False) if args.no_context_expansion else None,
        )
        try:
            retrieved = [
                retrieval_graph.invoke({"request": request.model_dump(by_alias=True)}) for request in requests
            ]
            release_query_models()
            for request, state in zip(requests, retrieved):
                result = generation_graph.invoke(
                    {"request": request.model_dump(by_alias=True), "context": state["context"]}
                )
                print(json.dumps(result["response"], indent=None, ensure_ascii=False))
        finally:
            release_models()
        return

    graph = (
        build_prompt_graph(None if args.no_internal else args.internal_dir, enable_web=not args.no_web)
        if args.prompt_only
        else build_default_graph(
            None if args.no_internal else args.internal_dir,
            model=args.generation_model,
            enable_web=not args.no_web,
            query_model=args.query_model,
            summary_model=args.summary_model,
            query_record_dir=args.query_record_dir,
            query_categories=tuple(filter(None, args.query_categories.split(","))),
            processed_dir=args.processed_dir,
            hierarchy_config=HierarchyConfig(context_expansion_enabled=False) if args.no_context_expansion else None,
        )
    )
    for request in requests:
        result = graph.invoke({"request": request.model_dump(by_alias=True)})
        print(json.dumps(result["response"], indent=None if args.input_jsonl else 2, ensure_ascii=False))


if __name__ == "__main__":
    main()
