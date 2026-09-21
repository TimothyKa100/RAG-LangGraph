# UCA RAG with LangGraph

A two-step RAG workflow for STPA unsafe control action generation.

The main UCA path now performs one structured multi-query generation call with `meta-llama/Llama-3.2-3B-Instruct`, then passes the selected flattened queries into the existing hybrid retrieval implementation. The original BM25, dense embedding, RRF, and reranking components remain in place.

The project also includes a separate terminology discovery system. It uses a local `meta-llama/Llama-3.2-3B-Instruct` model to select obscure terms, checks only existing title names for aliases before searching, retrieves up to 10 DDGS results, fetches their page content, and stores `{title, summary}` records in a JSON file. It does not use embeddings. The terminology system is intentionally separate from the relational UCA retrieval path.

## Architecture

1. **Retrieve context**: build multi-query accident-report searches from `control_action`, `from`, and `to`; search the internal Excel corpus and DDGS; fetch readable page content with `requests` and BeautifulSoup; split web pages into overlapping child chunks linked to their parent URL; merge BM25 and dense semantic results; rerank with a cross-encoder; write the selected evidence to `context`.
2. **Generate UCAs**: pass the enriched request and evidence to a structured LLM prompt that returns the seven UCA categories.

Internal workbooks are read sheet-by-sheet. Each non-empty row becomes a retrievable document with workbook and sheet metadata.

Web results are fetched from the DDGS URLs rather than using snippets alone. Main/article text is extracted, split into overlapping chunks, and each chunk retains its parent URL and title. Retrieval limits the number of chunks selected from one parent page so a single page does not dominate the context.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[retrieval,llm,api,dev]"
```

The default UCA generator is the local Hugging Face model at `outputs/fine-tune-merged`. Query generation uses `meta-llama/Llama-3.2-3B-Instruct`; optional hierarchy summaries use `meta-llama/Llama-3.2-1B-Instruct`. The optional retrieval dependencies enable DDGS, BM25, sentence-transformer embeddings, and cross-encoder reranking.

SentenceTransformer and CrossEncoder instances are reused within a process, including when processing multiple JSONL records. Their weights use the normal Hugging Face/Sentence Transformers persistent cache. To choose an explicit shared cache directory, set `UCA_RAG_MODEL_CACHE` before running:

```powershell
$env:UCA_RAG_MODEL_CACHE = "$PWD\.model-cache"
```

Install the local model dependencies:

```powershell
pip install -e ".[retrieval,terminology,api,dev]"
```

On a Slurm server, the provided `run_uca_web.slurm` script updates the activated environment from `requirements.txt` before running the job.

All local causal-LM adapters reuse the same in-process tokenizer/model pair for each model path. `UCA_RAG_MODEL_CACHE` controls the Hugging Face cache directory.

## Run

```powershell
uca-rag --internal-dir .\data\internal --control-action "Activate Hazard Warning" --from "Automated Lane Keeping System" --to "Other Vehicle Systems"
```

To process the training-style JSONL format directly, use `--input-jsonl`. The loader extracts the last `user` message from each record and reads its `Control Action`, `From`, and `To` fields:

```powershell
uca-rag --internal-dir .\data\internal --input-jsonl .\data\uca-input.jsonl
```

The command writes one JSON response per input line. The `assistant` message in the source record is not used as input; it is the expected example output.

To retrieve local evidence and output the generated prompt without calling the final model, use `--prompt-only`:

```powershell
& .\.venv\Scripts\python.exe -m uca_rag.cli `
	--internal-dir .\data\internal `
	--input-jsonl .\data\uca-input.jsonl `
	--prompt-only `
	--no-web
```

This mode does not require the final generation model. It outputs `system`, `user`, and `context` fields for each input record. `--no-web` keeps retrieval inside the local Excel corpus.

To test before any Excel files are available, use `--no-internal`. This runs web retrieval and prompt generation only:

```powershell
& .\.venv\Scripts\python.exe -m uca_rag.cli `
	--no-internal `
	--input-jsonl .\data\uca-input.jsonl `
	--prompt-only
```

For a completely offline smoke test with neither Excel nor web retrieval, combine `--no-internal` and `--no-web`. The generated prompt will contain the fallback message `No supporting evidence was retrieved.`

For an API:

```powershell
uvicorn uca_rag.api:app --reload
```

API model paths can be overridden with `UCA_GENERATION_MODEL`, `UCA_QUERY_MODEL`, and `UCA_SUMMARY_MODEL`.

Run the standalone terminology system from the CLI with the terminology dependencies installed:

```powershell
pip install -e ".[terminology,retrieval,dev]"
uca-rag --terminology --terminology-store .\data\terminology.json --control-action "RadioFrequencyChange" --from "ATCTower" --to "Flight Handling"
```

It can also be called at `POST /discover-terminology` using the same request fields as `/generate-uca`.

The query generator can be configured with `UCA_QUERY_MODEL`, `UCA_QUERY_RECORD_DIR`, and `UCA_QUERY_CATEGORIES` (comma-separated values from `direct`, `failure_mode`, `relationship`, and `terminology`). Generated query records are appended to `queries.jsonl` and include the UCA ID when supplied in graph state.

Excel preprocessing writes the hierarchy to `data/processed` by default as `documents.jsonl`, `sections.jsonl`, `chunks.jsonl`, and `retrieval_units.jsonl`. Configure hierarchy values with environment variables matching the `HierarchyConfig` fields, such as `SEMANTIC_CHUNK_TARGET_TOKENS`, `RETRIEVAL_UNIT_TARGET_TOKENS`, `MAX_CONTEXT_TOKENS`, `ADJACENT_UNITS_BEFORE`, and `ADJACENT_UNITS_AFTER`. The `HierarchyConfig` object can also be passed directly to `ExcelIndexer` or `build_default_graph`.

The resulting hierarchy supports these ablations through configuration: retrieval units only, units plus parent chunk, units plus neighbors, and units plus neighbors plus section summaries.

POST `/generate-uca` with:

```json
{"control_action":"Activate Hazard Warning","from":"Automated Lane Keeping System","to":"Other Vehicle Systems","context":""}
```
