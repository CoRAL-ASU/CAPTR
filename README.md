# CAPTR: Caption-based Context Pruning for Multimodal Tabular Reasoning

<!--[![Paper](https://img.shields.io/badge/Paper-ACL%202025-blue)](TODO)-->

> **TL;DR:** CAPTR reduces context length by up to 80% while improving multimodal table reasoning accuracy by up to 8 points through caption-based pruning. It outperforms retrieval and full-context baselines on MMTabQA and MMTBench.

<!--![CAPTR Pipeline](CAPTR.png)-->

## Overview

Vision-Language Models (VLMs) struggle with multimodal reasoning when tables contain hundreds of images. High token counts lead to quadratic computational costs and models get distracted by irrelevant visual context.

**CAPTR** (Caption-based Context Pruning for Tabular Reasoning) solves this by using image captions as a lightweight textual proxy to prune irrelevant rows and columns before multimodal reasoning.


### Key Results

- 📉 **Up to 80% context reduction**
- 📈 **Up to 8 point accuracy improvement** over competitive baselines
- 🎯 **LLM-as-a-judge** used when exact match is too strict for image-based entity answers
- 🚀 Evaluated on **MMTabQA** and **MMTBench** with multiple multimodal models

### Why CAPTR?
Traditional approaches fail when scaling to tables with many images:
- Full Context (Interleaved): Feeding all images results in massive context windows (often 80k+ tokens), causing high latency and model distraction.
- Standard Retrieval (RAG): Retrieval methods typically fail to capture the structural dependencies inherent in tabular data (e.g., row/column alignment), often retrieving irrelevant images or missing connected cells.
- Visual Token Pruning: Methods like SparseVLM focus on compressing tokens within an image, but do not remove semantically irrelevant images entirely.

CAPTR addresses these by operating in two stages:
- Relevance Pruning (Text Space): Images are temporarily replaced by short, context-aware captions. An LLM uses SQL and semantic reasoning to select only the rows and columns relevant to the user's query.
- Multimodal Reasoning (Visual Space): The original images are restored into the pruned table structure. The VLM reasons over this compact, high-signal context.

## Installation
### Setup

```bash
# Clone the repository
git clone <repository-url>
cd CAPTR

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Create `.env` from the provided example and fill values
# Then edit `.env` and fill in tokens and dataset paths, e.g.:
# HF_TOKEN=<your_huggingface_token>
# MMTABQA_IMAGE_BASE_PATH=/path/to/MMTabQA_Dataset/IMAGES/
# MMTABQA_HF_DATASET_BASE_PATH=/path/to/converted_to_hf_dataset/
# MMTABQA_BASE_PATH=/path/to/MMTabQA_Dataset/
# MMT_BENCH_BASE_PATH=/path/to/MMTBench
```

### Dataset Download

We evaluate CAPTR on two benchmarks: **MMTabQA** and **MMTBench**. See [datasets/README.md](datasets/README.md) for detailed download and preprocessing instructions.

```bash
# Prepare MMTabQA dataset (downloads images from MEGA, unzips them)
cd datasets/MMTabQA_Dataset
./prepare_mmtabqa.sh
```

### Generate Captions
Use the provided helper script to generate captions for both datasets (MMTabQA and MMTBench).

```bash
bash generate_captions.sh
```

By default `generate_captions.sh` will run caption generation for both MMTabQA and MMTBench using the paths from your `.env` file. See [mmtabqa/README.md](mmtabqa/README.md) for dataset-specific details.

## Quick Start

```bash
bash run_captr.sh
```

To run only the baseline suite for MMTabQA and MMTabBench, use [run_baselines_mmtabqa_mmtbench.sh](run_baselines_mmtabqa_mmtbench.sh).

### Manually running CAPTR

```bash
# Run CAPTR with short context captions on MMTabQA. Results will be saved in `results/gemma3-mmtabqa-captioned-short-interleaved-reasoner/`
uv run run_model.py --config_file="run_configs/gemma3/short_captioned_interleaved_reasoner.json"
```

For details on customizing configurations (e.g., changing the pruning strategy or VLM), see [run_configs/README.md](run_configs/README.md). Baseline usage is documented in [baselines/README.md](baselines/README.md).



## Project Structure
Most important folders:
```
CAPTR/
├── run_model.py                   # Main pipeline executor
├── run_all_experiments.sh         # Main experiment launcher
├── run_baselines_mmtabqa_mmtbench.sh # Baseline launcher for MMTabQA/MMTabBench
├── baselines/                     # Partial-input, interleaved, retrieval, and SparseVLM baselines
├── datasets/                      # Dataset download and preprocessing scripts
├── generation/                    # Model-specific generation wrappers
├── mmtabqa/                       # Caption dataset creation and MMTabQA/MMTabBench loaders
├── nsql/                          # SQL execution, parsing, and database helpers
├── prompts/                       # Prompt templates used by pruning/reasoning steps
├── run_configs/                   # Experiment configuration JSON files
├── scripts/                       # Pipeline steps, evaluation, and analysis utilities
├── statistics/                    # Tables, plots, and analysis scripts for the paper
├── utils/                         # Shared runtime, normalization, matching, and evaluation helpers
└── README.md                      # This file
```

CAPTR uses LLM-as-a-Judge for semantic evaluation when exact match is too strict for image-based entity answers. For analysis scripts and paper tables, see [statistics/README.md](statistics/README.md).



## License

The code of this repository is distributed with the Apache License 2.0.
