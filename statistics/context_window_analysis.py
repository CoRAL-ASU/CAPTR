import json
import os
import re
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer


BASELINE_RESULTS_DIR = "../mmtabqa/baselines/results/gemma-3-27b-it/"
# Oracle baseline:
# PIPELINE_RESULTS_DIR = "/home/leon/Uni/Module/Master/FS4/Thesis/H-STAR/results/gemma3-mmtabqa-gold-entity-replaced-baseline-interleaved-reasoner/"

PIPELINE_RESULTS_DIR = "../results/gemma3-mmtabqa-captioned-short-interleaved-reasoner"


MODEL_NAME = "google/gemma-3-27b-it"
TOKENS_PER_IMAGE = 256

IMAGE_TAG_PATTERN = re.compile(r"<image_")

PIPELINE_STEPS_FILES = {
    "step1": "step1_col_sql.json",
    "step2": "step2_col_text.json",
    "step3": "step3_row_sql.json",
    "step4": "step4_row_text.json",
    "reasoning": "interleaved_reasoning_output.json",
}

MMTABQA_SUB_DATASETS = ["wikitq", "wikisql", "fetaqa", "hybridqa"]
MMTABQA_DATASET_SPLITS = ["VQ", "AQ", "EQ", "IQ"]


def load_captr_data(results_base_dir):
    """
    Load CAPTR pipeline data from the results directory structure.

    This function traverses the CAPTR results directory to load data from all
    pipeline steps across different MMTabQA sub-datasets and splits. It organizes
    the data hierarchically by sub-dataset, example ID, and pipeline step.

    Args:
        results_base_dir (str): Base directory containing CAPTR pipeline results. Expected structure: {base_dir}/{sub_dataset}_{split}/{step_file}

    Returns:
        dict: Nested dictionary with structure:
              {sub_dataset: {example_id: {step_name: step_data}}}
              where step_data contains the pipeline output for that step.

    Example:
        >>> data = load_captr_data("/path/to/results")
        >>> print(data["wikitq"]["VQ"]["example_1"]["step1"])
        {'prompt': '...', 'generations': ['...']}
    """
    print(f"Loading CAPTR data from base directory: {results_base_dir}")
    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for sub_dataset in MMTABQA_SUB_DATASETS:
        for split in MMTABQA_DATASET_SPLITS:
            print(f"Loading CAPTR data for {sub_dataset}_{split}")
            split_dir = os.path.join(results_base_dir, f"{sub_dataset}_{split}")
            if not os.path.isdir(split_dir):
                print(f"Directory not found, skipping: {sub_dataset}_{split}  ({split_dir})")
                continue

            for step_name, filename in tqdm(
                PIPELINE_STEPS_FILES.items(), desc=f"Loading CAPTR data for {sub_dataset}_{split}"
            ):
                filepath = os.path.join(split_dir, filename)
                if not os.path.exists(filepath):
                    print(f"File not found, skipping: {filepath}")
                    continue

                with open(filepath, "r") as f:
                    step_data = json.load(f)
                for eid, data in step_data.items():
                    prompt = data["content"] if "content" in data else data["prompt"]

                    # if prompt is a list, take the first element
                    if isinstance(prompt, list):
                        prompt = prompt[0]["text"]

                    all_data[sub_dataset][split][eid][step_name] = {
                        "prompt": prompt,
                        "generations": data["generations"],
                    }
    return all_data


def count_tokens_for_prompt(prompt, tokenizer):
    """Counts tokens for a given prompt, which can be a string or an interleaved list."""

    token_count = len(tokenizer.encode(prompt))
    image_count = len(IMAGE_TAG_PATTERN.findall(prompt))
    token_count += image_count * TOKENS_PER_IMAGE
    return token_count


def count_captr_tokens(captr_data, tokenizer):
    """Counts tokens for the processed CAPTR data."""
    print("Counting tokens for CAPTR data...")
    token_counts = {}

    for dataset, splits in tqdm(captr_data.items(), desc="Counting CAPTR tokens"):
        token_counts[dataset] = {}
        for split, examples in splits.items():
            token_counts[dataset][split] = {}
            dataset_summed_inputs, dataset_summed_outputs = [], []
            dataset_max_inputs, dataset_max_outputs = [], []

            for eid, steps in tqdm(examples.items(), desc=f"Counting CAPTR tokens for {dataset}/{split}"):
                step_input_tokens, step_output_tokens = [], []

                # prepare dict
                for step_name in steps.keys():
                    if step_name not in token_counts[dataset][split]:
                        token_counts[dataset][split][step_name] = {}
                    token_counts[dataset][split][step_name][eid] = {}

                # count tokens for each step
                for step_name, data in steps.items():
                    prompt = data["prompt"]

                    input_count = count_tokens_for_prompt(prompt, tokenizer)
                    token_counts[dataset][split][step_name][eid]["input_tokens"] = input_count
                    step_input_tokens.append(input_count)

                    # TODO: what if we have multiple generations? How do we count the input/output tokens then?
                    output_count = len(tokenizer.encode(data["generations"][0], add_special_tokens=False))
                    token_counts[dataset][split][step_name][eid]["output_tokens"] = output_count
                    step_output_tokens.append(output_count)

                dataset_summed_inputs.append(sum(step_input_tokens))
                dataset_max_inputs.append(max(step_input_tokens))
                dataset_summed_outputs.append(sum(step_output_tokens))
                dataset_max_outputs.append(max(step_output_tokens))

            token_counts[dataset][split]["summed_input"] = dataset_summed_inputs
            token_counts[dataset][split]["summed_output"] = dataset_summed_outputs
            token_counts[dataset][split]["max_input"] = dataset_max_inputs
            token_counts[dataset][split]["max_output"] = dataset_max_outputs
    return token_counts


def load_baseline_data(base_dir):
    """Loads baseline data, merging splits for each dataset."""
    print(f"Loading baseline data from {base_dir}")
    all_data = defaultdict(dict)

    mapping_sub_dataset_to_baseline = {
        "wikitq": "WikiTQ",
        "wikisql": "WikiSQL",
        "fetaqa": "FetaQA",
        "hybridqa": "HybridQA",
    }

    for sub_dataset in MMTABQA_SUB_DATASETS:
        for split in MMTABQA_DATASET_SPLITS:
            filename = f"{mapping_sub_dataset_to_baseline[sub_dataset]}_{split}_interleaved.json"
            filepath = os.path.join(base_dir, filename)
            if not os.path.exists(filepath):
                print(f"Baseline file not found, skipping: {filepath}")
                continue

            with open(filepath, "r") as f:
                split_data = json.load(f)
            all_data[sub_dataset].update(split_data)
    return all_data


def count_baseline_tokens(baseline_data, tokenizer):
    """Counts tokens for the processed baseline data."""
    print("Counting tokens for baseline data...")
    token_counts = defaultdict(lambda: defaultdict(list))

    for dataset, examples in baseline_data.items():
        print(f"counting tokens for {dataset}")
        for eid, data in tqdm(examples.items(), desc=f"Counting baseline tokens for {dataset}"):
            # Count tokens in input
            content: List[Dict[str, str]] = data["content"]
            input_count = 0
            for item in content:
                if item["type"] == "image":
                    input_count += TOKENS_PER_IMAGE
                else:
                    input_count += len(tokenizer.encode(item["text"]))
            token_counts[dataset]["input"].append(input_count)

            # Count tokens in output
            output_count = len(tokenizer.encode(data["generations"][0], add_special_tokens=False))
            token_counts[dataset]["output"].append(output_count)
    return token_counts


def plot_histograms(data, title, save_path):
    """Generates and saves a histogram for the given data."""
    if not data:
        print(f"No data to plot for {title}. Skipping.")
        return

    plt.figure(figsize=(10, 6))
    avg, med, maximum = np.mean(data), np.median(data), np.max(data)
    plt.hist(data, bins=50, alpha=0.75, label=f"Avg: {avg:.2f}, Median: {med:.2f}, Max: {maximum}")
    plt.title(title, fontsize=16)
    plt.xlabel("Token Count", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved histogram to {save_path}")


def run_captr_analysis(tokenizer):
    """Main function for CAPTR analysis."""
    output_dir = os.path.join(PIPELINE_RESULTS_DIR, "token_analysis")
    os.makedirs(output_dir, exist_ok=True)

    captr_data = load_captr_data(PIPELINE_RESULTS_DIR)
    if not captr_data:
        print("No CAPTR data loaded. Exiting analysis.")
        return
    print(f"Loaded data for {len(captr_data)} datasets: {list(captr_data.keys())}")
    token_counts = count_captr_tokens(captr_data, tokenizer)

    all_datasets_stats = {
        "summed_input": [],
        "summed_output": [],
        "max_input": [],
    }

    for dataset, splits in token_counts.items():
        dataset_dir = os.path.join(output_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        for split, counts in splits.items():
            print(f"\n---- Statistics for {dataset} - {split} ----")

            summed_inputs = counts["summed_input"]
            if summed_inputs:
                avg_input = np.mean(summed_inputs)
                median_input = np.median(summed_inputs)
                print("-- Input tokens (all steps summed) --")
                print(f"Average: {avg_input:.2f}")
                print(f"Median: {median_input:.2f}")
                print(f"90%: {np.percentile(summed_inputs, 90):.2f}")  # fmt: skip
                all_datasets_stats["summed_input"].extend(summed_inputs)

            summed_outputs = counts["summed_output"]
            if summed_outputs:
                avg_output = np.mean(summed_outputs)
                median_output = np.median(summed_outputs)
                print("-- Output tokens (all steps summed) --")
                print(f"Average: {avg_output:.2f}")
                print(f"Median: {median_output:.2f}")
                print(f"90%: {np.percentile(summed_outputs, 90):.2f}")  # fmt: skip
                all_datasets_stats["summed_output"].extend(summed_outputs)

            max_inputs = counts["max_input"]
            if max_inputs:
                print("-- Input tokens (max across all steps; e.g. [1,3,2,1] would be 3) - REQUIRED CONTEXT WINDOW SIZE --")  # fmt: skip
                print(f"Max: {np.max(max_inputs)}")
                print(f"Median: {np.median(max_inputs):.2f}")
                print(f"90%: {np.percentile(max_inputs, 90):.2f}")
                all_datasets_stats["max_input"].extend(max_inputs)

            for category, data in counts.items():
                if isinstance(data, list):
                    title = f"H-STAR: {dataset}_{split} - {category.replace('_', ' ').title()}"
                    save_path = os.path.join(dataset_dir, f"{split}_{category}.png")
                    plot_histograms(data, title, save_path)
                elif isinstance(data, dict):
                    input_data = [v["input_tokens"] for v in data.values()]
                    title_input = f"H-STAR: {dataset}_{split} - {category.replace('_', ' ').title()} Input Tokens"
                    save_path_input = os.path.join(dataset_dir, f"{split}_{category}_input.png")
                    plot_histograms(input_data, title_input, save_path_input)

                    output_data = [v["output_tokens"] for v in data.values()]
                    title_output = f"H-STAR: {dataset}_{split} - {category.replace('_', ' ').title()} Output Tokens"
                    save_path_output = os.path.join(dataset_dir, f"{split}_{category}_output.png")
                    plot_histograms(output_data, title_output, save_path_output)

    print("\n---- Overall Statistics (All Datasets & Splits) ----")
    if all_datasets_stats["summed_input"]:
        avg_input_all = np.mean(all_datasets_stats["summed_input"])
        median_input_all = np.median(all_datasets_stats["summed_input"])
        print("-- Input tokens (all steps summed) --")
        print(f"Average: {avg_input_all:.2f}")
        print(f"Median: {median_input_all:.2f}")
        print(f"90%: {np.percentile(all_datasets_stats["summed_input"], 90):.2f}")  # fmt: skip

    if all_datasets_stats["summed_output"]:
        avg_output_all = np.mean(all_datasets_stats["summed_output"])
        median_output_all = np.median(all_datasets_stats["summed_output"])
        print("-- Output tokens (all steps summed) --")
        print(f"Average: {avg_output_all:.2f}")
        print(f"Median: {median_output_all:.2f}")
        print(f"90%: {np.percentile(all_datasets_stats["summed_output"], 90):.2f}")  # fmt: skip

    if all_datasets_stats["max_input"]:
        max_input_all = np.max(all_datasets_stats["max_input"])
        print("-- Input tokens (max across all steps; e.g. [1,3,2,1] would be 3) - REQUIRED CONTEXT WINDOW SIZE --")  # fmt: skip
        print(f"Max: {max_input_all}")
        print(f"Median: {np.median(all_datasets_stats['max_input']):.2f}")
        print(f"90%: {np.percentile(all_datasets_stats["max_input"], 90):.2f}")  # fmt: skip


def run_baseline_analysis(tokenizer):
    """Main function for baseline analysis."""
    output_dir = os.path.join(PIPELINE_RESULTS_DIR, "token_analysis", "baselines", "interleaved_reasoner")
    os.makedirs(output_dir, exist_ok=True)

    baseline_data = load_baseline_data(BASELINE_RESULTS_DIR)
    token_counts = count_baseline_tokens(baseline_data, tokenizer)

    all_input_tokens = []
    all_output_tokens = []

    for dataset, counts in token_counts.items():
        dataset_dir = os.path.join(output_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        print(f"\n---- Baseline Statistics for {dataset} ----")
        input_tokens = counts["input"]
        if input_tokens:
            print("-- Input tokens --")
            print(f"Average: {np.mean(input_tokens):.2f}")
            print(f"Median: {np.median(input_tokens):.2f}")
            print(f"90%: {np.percentile(input_tokens, 90):.2f}")
            print(f"Max: {np.max(input_tokens)}")
            all_input_tokens.extend(input_tokens)

        output_tokens = counts["output"]
        if output_tokens:
            print("-- Output tokens --")
            print(f"Average: {np.mean(output_tokens):.2f}")
            print(f"Median: {np.median(output_tokens):.2f}")
            print(f"90%: {np.percentile(output_tokens, 90):.2f}")
            print(f"Max: {np.max(output_tokens)}")
            all_output_tokens.extend(output_tokens)

        for category, data in counts.items():
            title = f"Baseline 'interleaved_reasoner': {dataset} - {category.title()} Tokens"
            save_path = os.path.join(dataset_dir, f"{dataset}_{category}.png")
            plot_histograms(data, title, save_path)

    # Stats across all datasets:
    print("\n---- Overall Baseline Statistics (All Datasets & Splits) ----")
    print("-- Input tokens --")
    print(f"Average: {np.mean(all_input_tokens):.2f}")
    print(f"Median: {np.median(all_input_tokens):.2f}")
    print(f"90%: {np.percentile(all_input_tokens, 90):.2f}")
    print(f"Max: {np.max(all_input_tokens)}")
    print("-- Output tokens --")
    print(f"Average: {np.mean(all_output_tokens):.2f}")
    print(f"Median: {np.median(all_output_tokens):.2f}")
    print(f"90%: {np.percentile(all_output_tokens, 90):.2f}")
    print(f"Max: {np.max(all_output_tokens)}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 1. Analyse H-STAR
    run_captr_analysis(tokenizer)

    print("\n\n\n\n\n\n")

    # 2. Analyse Baselines
    run_baseline_analysis(tokenizer)


if __name__ == "__main__":
    main()
