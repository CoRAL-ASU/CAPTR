#!/bin/bash
# This script evaluates:
# 1. Gemma3-27B (original model)
# 2. Qwen3-VL-32B-Instruct
# 
# Each model is evaluated with:
# - Different captioning strategies (naive, short context, detailed)
# - Oracle filtering upper bound
# - Baselines (interleaved, partial input, retrieval)
#
# For new model evaluations, we sample n=700 examples per dataset split.

if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi

GEMMA3_MODEL="${GEMMA3_MODEL:-google/gemma-3-27b-it}"
QWEN3VL_MODEL="${CAPTR_QWEN3VL_MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
PARTIAL_INPUT_MODEL="${CAPTR_PARTIAL_INPUT_MODEL:-$GEMMA3_MODEL}"
INTERLEAVED_MODEL="${CAPTR_INTERLEAVED_MODEL:-$GEMMA3_MODEL}"

##########################################################################################################################################
# GEMMA3-27B EXPERIMENTS (full dataset)
##########################################################################################################################################
echo ""
echo "=========================================="
echo "Running Gemma3-27B experiments (full dataset)"
echo "=========================================="

# naive captions + Interleaved Reasoner
uv run run_model.py --config_file="run_configs/gemma3/naive_captioned_interleaved_reasoner.json"

# short context captions + Interleaved Reasoner
uv run run_model.py --config_file="run_configs/gemma3/short_captioned_interleaved_reasoner.json"

# detailed captions + Interleaved Reasoner
uv run run_model.py --config_file="run_configs/gemma3/detailed_captioned_interleaved_reasoner.json"

# Upper Bound: CAPTR oracle filtering + interleaved reasoner
uv run run_model.py --config_file="run_configs/gemma3/upper_bound_oracle_filter_interleaved_reasoner.json"

##########################################################################################################################################
# QWEN3-VL-32B EXPERIMENTS (n=700 examples per split)
##########################################################################################################################################
echo ""
echo "=========================================="
echo "Running Qwen3-VL-32B experiments (n=700 per split)"
echo "=========================================="

# naive captions + Interleaved Reasoner
uv run run_model.py --config_file="run_configs/qwen3vl/naive_captioned_interleaved_reasoner.json"

# short context captions + Interleaved Reasoner
uv run run_model.py --config_file="run_configs/qwen3vl/short_captioned_interleaved_reasoner.json"

# detailed captions + Interleaved Reasoner
uv run run_model.py --config_file="run_configs/qwen3vl/detailed_captioned_interleaved_reasoner.json"

# Upper Bound: CAPTR oracle filtering + interleaved reasoner
uv run run_model.py --config_file="run_configs/qwen3vl/upper_bound_oracle_filter_interleaved_reasoner.json"


##########################################################################################################################################
# BASELINES
##########################################################################################################################################
echo ""
echo "=========================================="
echo "Running Baselines"
echo "=========================================="

cd baselines

##########################################################################################################################################
# Gemma3-27B Baselines (full dataset)
##########################################################################################################################################
echo ""
echo "--- Gemma3-27B Baselines ---"

# Partial input baseline
uv run python partial_input_baseline.py --model="$PARTIAL_INPUT_MODEL" --num_gpus=2

# Interleaved baseline
uv run python interleaved_baseline.py --model="$INTERLEAVED_MODEL" --num_gpus=2


##########################################################################################################################################
# Qwen3-VL Baselines (n=700 examples)
##########################################################################################################################################
echo ""
echo "--- Qwen3-VL Baselines (n=700) ---"

# Partial input baseline
uv run python partial_input_baseline.py --model="$QWEN3VL_MODEL" --num_gpus=2 --n_examples=700

# Interleaved baseline
uv run python interleaved_baseline.py --model="$QWEN3VL_MODEL" --num_gpus=2 --n_examples=700


##########################################################################################################################################
# RETRIEVAL BASELINES
# Step 1: Run retrieval to get relevant rows (uses ColQwen2 model)
# Step 2: Run interleaved baseline with retrieved rows
##########################################################################################################################################
echo ""
echo "=========================================="
echo "Running Retrieval Baselines"
echo "=========================================="

DATASETS=("HybridQA" "WikiTQ" "WikiSQL" "FetaQA")

# --- Row-wise Retrieval ---
echo ""
echo "--- Row-wise Retrieval (full dataset for Gemma3) ---"
for dataset in "${DATASETS[@]}"; do
    echo "Running row-wise retrieval for $dataset..."
    uv run python retrieval.py --retrieval_fn "row_wise_retrieval" --dataset_name "$dataset" --output_dir "data_gemma3/"
done

# Run interleaved baseline with retrieval results (Gemma3)
echo "Running Gemma3 interleaved baseline with row-wise retrieval results..."
uv run python interleaved_baseline.py --model="$INTERLEAVED_MODEL" --num_gpus=2 --retrieval_output_dir="data_gemma3/row_wise_retrieval"

echo ""
echo "--- Row-wise Retrieval (n=700 for Qwen3-VL) ---"
for dataset in "${DATASETS[@]}"; do
    echo "Running row-wise retrieval for $dataset (n=700)..."
    uv run python retrieval.py --retrieval_fn "row_wise_retrieval" --dataset_name "$dataset" --output_dir "data_qwen3vl_n700/" --n_examples=700
done

# Run interleaved baseline with retrieval results (Qwen3-VL)
echo "Running Qwen3-VL interleaved baseline with row-wise retrieval results..."
uv run python interleaved_baseline.py --model="$QWEN3VL_MODEL" --num_gpus=2 --n_examples=700 --retrieval_output_dir="data_qwen3vl_n700/row_wise_retrieval"


# --- Union Retrieval (text + image) ---
echo ""
echo "--- Union Retrieval (full dataset for Gemma3) ---"
for dataset in "${DATASETS[@]}"; do
    echo "Running union retrieval for $dataset..."
    uv run python retrieval.py --retrieval_fn "row_union_image_retrieval" --dataset_name "$dataset" --output_dir "data_gemma3/"
done

# Run interleaved baseline with retrieval results (Gemma3)
echo "Running Gemma3 interleaved baseline with union retrieval results..."
uv run python interleaved_baseline.py --model="$INTERLEAVED_MODEL" --num_gpus=2 --retrieval_output_dir="data_gemma3/row_union_image_retrieval"

echo ""
echo "--- Union Retrieval (n=700 for Qwen3-VL) ---"
for dataset in "${DATASETS[@]}"; do
    echo "Running union retrieval for $dataset (n=700)..."
    uv run python retrieval.py --retrieval_fn "row_union_image_retrieval" --dataset_name "$dataset" --output_dir "data_qwen3vl_n700/" --n_examples=700
done

# Run interleaved baseline with retrieval results (Qwen3-VL)
echo "Running Qwen3-VL interleaved baseline with union retrieval results..."
uv run python interleaved_baseline.py --model="$QWEN3VL_MODEL" --num_gpus=2 --n_examples=700 --retrieval_output_dir="data_qwen3vl_n700/row_union_image_retrieval"

# move back to the root directory
cd ..
echo "Done! :)"