#!/usr/bin/env bash
#SBATCH --job-name=uca-web-rag
#SBATCH --partition=long
#SBATCH --account=standard
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:L40:1
#SBATCH --qos=normal

set -euo pipefail

# Submit from the repository root, or override PROJECT_DIR when submitting.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
INPUT_JSONL="${INPUT_JSONL:-}"
DEFAULT_INPUT_JSONL="$PROJECT_DIR/test.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/slurm/$SLURM_JOB_ID}"
MODEL_CACHE="${UCA_RAG_MODEL_CACHE:-$PROJECT_DIR/.model-cache}"

mkdir -p "$PROJECT_DIR/logs" "$OUTPUT_DIR" "$MODEL_CACHE"
cd "$PROJECT_DIR"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "Virtual environment not found: $VENV_DIR" >&2
    echo "Create it on the cluster and install: pip install -e '.[retrieval,terminology,dev]'" >&2
    exit 1
fi
source "$VENV_DIR/bin/activate"

if [[ ! -f "$PROJECT_DIR/requirements.txt" ]]; then
    echo "Requirements file not found: $PROJECT_DIR/requirements.txt" >&2
    exit 1
fi

echo "Updating Python environment from requirements.txt"
python -m pip install --upgrade pip
python -m pip install -r "$PROJECT_DIR/requirements.txt"
python -m pip install --no-deps -e "$PROJECT_DIR"

if [[ -z "$INPUT_JSONL" && -f "$DEFAULT_INPUT_JSONL" ]]; then
    INPUT_JSONL="$DEFAULT_INPUT_JSONL"
fi

# Local models. The model cache is shared by query generation, terminology,
# summaries, and final generation within this process.
export UCA_RAG_MODEL_CACHE="$MODEL_CACHE"
export UCA_GENERATION_MODEL="${UCA_GENERATION_MODEL:-$PROJECT_DIR/outputs/fine-tune-merged}"
export UCA_QUERY_MODEL="${UCA_QUERY_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
export UCA_SUMMARY_MODEL="${UCA_SUMMARY_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"

# Web hierarchy is transient. Enable section summaries for expanded web context;
# no Excel files are read because --no-internal is passed below.
export ENABLE_SECTION_SUMMARIES="${ENABLE_SECTION_SUMMARIES:-true}"
export CONTEXT_EXPANSION_ENABLED="${CONTEXT_EXPANSION_ENABLED:-true}"
export MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-3000}"
export ADJACENT_UNITS_BEFORE="${ADJACENT_UNITS_BEFORE:-1}"
export ADJACENT_UNITS_AFTER="${ADJACENT_UNITS_AFTER:-1}"

# Keep query records and final responses separate for each Slurm job.
QUERY_RECORD_DIR="${QUERY_RECORD_DIR:-$OUTPUT_DIR/query_records}"
PROCESSED_DIR="${PROCESSED_DIR:-$OUTPUT_DIR/processed}"

if [[ -n "$INPUT_JSONL" ]]; then
    if [[ ! -f "$INPUT_JSONL" ]]; then
        echo "Input JSONL not found: $INPUT_JSONL" >&2
        exit 1
    fi
    python -m uca_rag.cli \
        --no-internal \
        --input-jsonl "$INPUT_JSONL" \
        --generation-model "$UCA_GENERATION_MODEL" \
        --query-model "$UCA_QUERY_MODEL" \
        --summary-model "$UCA_SUMMARY_MODEL" \
        --query-record-dir "$QUERY_RECORD_DIR" \
        --processed-dir "$PROCESSED_DIR" \
        --query-categories "${UCA_QUERY_CATEGORIES:-direct,failure_mode,relationship,terminology}" \
        > "$OUTPUT_DIR/results.jsonl"
else
    : "${CONTROL_ACTION:?Set CONTROL_ACTION when INPUT_JSONL is not used}"
    : "${SOURCE_CONTROLLER:?Set SOURCE_CONTROLLER when INPUT_JSONL is not used}"
    : "${TARGET:?Set TARGET when INPUT_JSONL is not used}"

    python -m uca_rag.cli \
        --no-internal \
        --control-action "$CONTROL_ACTION" \
        --from "$SOURCE_CONTROLLER" \
        --to "$TARGET" \
        --context "${CONTEXT:-}" \
        --generation-model "$UCA_GENERATION_MODEL" \
        --query-model "$UCA_QUERY_MODEL" \
        --summary-model "$UCA_SUMMARY_MODEL" \
        --query-record-dir "$QUERY_RECORD_DIR" \
        --processed-dir "$PROCESSED_DIR" \
        --query-categories "${UCA_QUERY_CATEGORIES:-direct,failure_mode,relationship,terminology}" \
        > "$OUTPUT_DIR/results.json"
fi

echo "Completed UCA web RAG job. Results: $OUTPUT_DIR"
