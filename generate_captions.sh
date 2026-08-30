#!/bin/bash

set -euo pipefail

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

if [ -z "${MMTABQA_HF_DATASET_BASE_PATH:-}" ]; then
  echo "MMTABQA_HF_DATASET_BASE_PATH must be set"
  exit 1
fi

if [ -z "${MMTABQA_IMAGE_BASE_PATH:-}" ]; then
  echo "MMTABQA_IMAGE_BASE_PATH must be set"
  exit 1
fi

if [ -z "${MMT_BENCH_BASE_PATH:-}" ]; then
  echo "MMT_BENCH_BASE_PATH must be set"
  exit 1
fi

N_EXAMPLES="${N_EXAMPLES:-700}"
CAPTION_MODES="detailed naive short_with_context"

run_caption_generation() {
  local target_dataset="$1"
  local dataset_base_path="$2"
  local output_base_path="$3"
  local title="$4"

  echo ""
  echo "Generating captions for ${title}..."
  echo ""

  for mode in ${CAPTION_MODES}; do
    echo "================================"
    echo "Generating ${title} captions with mode: ${mode}"
    echo "================================"
    python3 mmtabqa/create_caption_dataset.py \
      --target_dataset "${target_dataset}" \
      --caption_mode "${mode}" \
      --dataset_base_path "${dataset_base_path}" \
      --output_base_path "${output_base_path}" \
      --n_examples "${N_EXAMPLES}"
    echo ""
    echo "✓ Caption generation complete for ${title} (${mode})"
    echo ""
  done
}

run_caption_generation \
  "mmtabqa" \
  "${MMTABQA_HF_DATASET_BASE_PATH}" \
  "${MMTABQA_HF_DATASET_BASE_PATH}" \
  "MMTabQA"

run_caption_generation \
  "mmtbench" \
  "${MMT_BENCH_BASE_PATH}/hf_dataset" \
  "${MMT_BENCH_BASE_PATH}/converted_to_hf_dataset" \
  "MMTabBench"

echo "ALL CAPTION GENERATION COMPLETE!"
