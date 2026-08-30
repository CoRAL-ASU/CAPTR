#!/bin/bash

set -euo pipefail

if [ -f .env ]; then
	set -a
	. ./.env
	set +a
fi

GEMMA3_MODEL="${GEMMA3_MODEL:-google/gemma-3-27b-it}"
QWEN3VL_MODEL="${CAPTR_QWEN3VL_MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
PARTIAL_INPUT_MODEL_MMTABQA="${CAPTR_PARTIAL_INPUT_MODEL_MMTABQA:-$GEMMA3_MODEL}"
INTERLEAVED_MODEL_MMTABQA="${CAPTR_INTERLEAVED_MODEL_MMTABQA:-$GEMMA3_MODEL}"
PARTIAL_INPUT_MODEL_MMTBENCH="${CAPTR_PARTIAL_INPUT_MODEL_MMTBENCH:-$QWEN3VL_MODEL}"
INTERLEAVED_MODEL_MMTBENCH="${CAPTR_INTERLEAVED_MODEL_MMTBENCH:-$QWEN3VL_MODEL}"
N_EXAMPLES_MMTBENCH="${N_EXAMPLES_MMTBENCH:-700}"

run_baselines() {
	local dataset_label="$1"
	local partial_model="$2"
	local interleaved_model="$3"
	shift 3
	local extra_args=("$@")

	echo ""
	echo "=========================================="
	echo "Running ${dataset_label} baselines"
	echo "=========================================="

	echo "Running partial input baseline for ${dataset_label}"
	python3 baselines/partial_input_baseline.py --model="${partial_model}" "${extra_args[@]}"

	echo "Running interleaved baseline for ${dataset_label}"
	python3 baselines/interleaved_baseline.py --model="${interleaved_model}" "${extra_args[@]}"
}

run_baselines "MMTabQA" "${PARTIAL_INPUT_MODEL_MMTABQA}" "${INTERLEAVED_MODEL_MMTABQA}" --num_gpus=2

run_baselines "MMTabBench" "${PARTIAL_INPUT_MODEL_MMTBENCH}" "${INTERLEAVED_MODEL_MMTBENCH}" --num_gpus=2 --mmtabreal --n_examples="${N_EXAMPLES_MMTBENCH}"

echo ""
echo "All baseline runs complete."