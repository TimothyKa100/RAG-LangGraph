import os

from fastapi import FastAPI

from functools import lru_cache

from .graph import build_default_graph, build_default_terminology_graph
from .models import UCARequest

app = FastAPI(title="UCA RAG LangGraph")


@app.post("/generate-uca")
def generate_uca(request: UCARequest):
    graph = build_default_graph(
        "data/internal",
        model=os.getenv("UCA_GENERATION_MODEL", "outputs/fine-tune-merged"),
        query_model=os.getenv("UCA_QUERY_MODEL", "meta-llama/Llama-3.2-3B-Instruct"),
        summary_model=os.getenv("UCA_SUMMARY_MODEL", "meta-llama/Llama-3.2-1B-Instruct"),
        query_record_dir=os.getenv("UCA_QUERY_RECORD_DIR", "data/query_records"),
        query_categories=tuple(
            category for category in os.getenv(
                "UCA_QUERY_CATEGORIES", "direct,failure_mode,relationship,terminology"
            ).split(",") if category
        ),
    )
    result = graph.invoke({"request": request.model_dump(by_alias=True)})
    return result["response"]


@lru_cache(maxsize=1)
def _terminology_graph():
    return build_default_terminology_graph()


@app.post("/discover-terminology")
def discover_terminology(request: UCARequest):
    result = _terminology_graph().invoke({"request": request.model_dump(by_alias=True)})
    return {"terminology": result["terminology"]}
