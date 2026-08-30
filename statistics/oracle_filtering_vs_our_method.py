import json
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np


ORACLE_BASELINE_RESULTS_DIR = "../results/gemma3-mmtabqa-gold-entity-replaced-baseline-interleaved-reasoner/"
CAPTION_BASED_FILTERING_RESULTS_DIR = "../results/gemma3-mmtabqa-captioned-short-interleaved-reasoner"

MMTABQA_SUB_DATASETS = ["wikitq", "wikisql", "fetaqa", "hybridqa"]
MMTABQA_DATASET_SPLITS = ["VQ", "AQ", "EQ", "IQ"]


def load_filtered_tables(results_base_dir):
    """
    Load the filtered tables from the results directory structure.
    """
    all_data = defaultdict(lambda: defaultdict(dict))

    for sub_dataset in MMTABQA_SUB_DATASETS:
        for split in MMTABQA_DATASET_SPLITS:
            print(f"Loading CAPTR data for {sub_dataset}_{split}")
            split_dir = os.path.join(results_base_dir, f"{sub_dataset}_{split}")
            if not os.path.isdir(split_dir):
                print(f"Directory not found, skipping: {sub_dataset}_{split}  ({split_dir})")
                continue

            filepath = os.path.join(split_dir, "step4_row_text.json")

            with open(filepath, "r") as f:
                step_data = json.load(f)
            for eid, data in step_data.items():
                rows = data["rows"]
                cols = data["cols"]

                all_data[sub_dataset][split][eid] = {
                    "cols": cols,
                    "rows": rows,
                }
    return all_data


def plot_histogram(counts1, counts2, total_examples, title, xlabel, output_path, percentage_labels=True):
    freq1 = Counter(counts1)
    freq2 = Counter(counts2)

    all_keys = list(set(freq1.keys()) | set(freq2.keys()))
    if not all_keys:
        max_val = 0
    else:
        max_val = max(all_keys)

    num_categories = max_val + 1
    x = np.arange(num_categories)
    if percentage_labels:
        plot_width = max(12, num_categories)
        fig, ax = plt.subplots(figsize=(plot_width, 6))
    else:
        fig, ax = plt.subplots(figsize=(12, 6))

    width = 0.35

    # Convert to percentages
    y1 = [100 * freq1.get(i, 0) / total_examples for i in x]
    bars1 = ax.bar(x - width / 2, y1, width, label="Missing (FN)")

    y2 = [100 * freq2.get(i, 0) / total_examples for i in x]
    bars2 = ax.bar(x + width / 2, y2, width, label="Too Much (FP)")

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

    if percentage_labels:
        add_labels(bars1)
        add_labels(bars2)

    # Set y-axis to percentage scale with grid lines
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_yticklabels([f"{i}%" for i in range(0, 101, 10)])
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    ax.set_ylabel("Percentage of Examples")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.legend()

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def main():
    output_dir = os.path.join(CAPTION_BASED_FILTERING_RESULTS_DIR, "analysis_vs_oracle_filtering")
    os.makedirs(output_dir, exist_ok=True)

    caption_based_filtered_tables = load_filtered_tables(CAPTION_BASED_FILTERING_RESULTS_DIR)
    oracle_filtered_tables = load_filtered_tables(ORACLE_BASELINE_RESULTS_DIR)

    print("""
Recall: how many relevant rows/cols the filtering missed.
Precision: how many irrelevant rows/cols the filtering kept.

Recall is the more important metric: if rows/columns relevant to answer the query are filtered out (recall lower 1), it becomes impossible to answer the question correctly, whereas extra irrelevant data (low precision) only adds noise but doesn't remove the necessary information.
""")

    global_missing_rows_counts = []
    global_too_much_rows_counts = []
    global_missing_cols_counts = []
    global_too_much_cols_counts = []
    global_total_examples_count = 0
    global_perfect_rows_count = 0
    global_perfect_cols_count = 0
    global_perfect_rows_and_cols_count = 0
    global_perfect_recall_rows_count = 0
    global_perfect_precision_rows_count = 0
    global_perfect_recall_cols_count = 0
    global_perfect_precision_cols_count = 0

    global_intersecting_rows = 0
    global_oracle_rows = 0
    global_caption_rows = 0
    global_intersecting_cols = 0
    global_oracle_cols = 0
    global_caption_cols = 0

    for sub_dataset in MMTABQA_SUB_DATASETS:
        sub_dataset_output_dir = os.path.join(output_dir, sub_dataset)
        os.makedirs(sub_dataset_output_dir, exist_ok=True)

        for split in MMTABQA_DATASET_SPLITS:
            if (
                split not in caption_based_filtered_tables[sub_dataset]
                or split not in oracle_filtered_tables[sub_dataset]
            ):
                print(f"Skipping {sub_dataset}_{split} as it's not present in both results.")
                continue

            print(f"\n--- Analyzing {sub_dataset} - {split} ---")

            oracle_split_data = oracle_filtered_tables[sub_dataset][split]
            caption_split_data = caption_based_filtered_tables[sub_dataset][split]
            eids = oracle_split_data.keys()
            total_examples = len(eids)

            missing_rows_counts, too_much_rows_counts = [], []
            missing_cols_counts, too_much_cols_counts = [], []

            perfect_recall_rows_count, perfect_precision_rows_count = 0, 0
            perfect_recall_cols_count, perfect_precision_cols_count = 0, 0

            perfect_rows_count = 0
            perfect_cols_count = 0
            perfect_rows_and_cols_count = 0

            total_oracle_rows, total_caption_rows, total_intersecting_rows = 0, 0, 0
            total_oracle_cols, total_caption_cols, total_intersecting_cols = 0, 0, 0

            for eid in eids:
                oracle_data = oracle_split_data[eid]
                oracle_rows = set(oracle_data["rows"])
                oracle_cols = set(oracle_data["cols"])

                caption_data = caption_split_data.get(eid, {})
                caption_rows = set(caption_data.get("rows", []))
                caption_cols = set(caption_data.get("cols", []))

                missing_rows = len(oracle_rows - caption_rows)
                too_much_rows = len(caption_rows - oracle_rows)
                missing_cols = len(oracle_cols - caption_cols)
                too_much_cols = len(caption_cols - oracle_cols)

                if missing_rows == 0:
                    perfect_recall_rows_count += 1
                if too_much_rows == 0:
                    perfect_precision_rows_count += 1
                if missing_cols == 0:
                    perfect_recall_cols_count += 1
                if too_much_cols == 0:
                    perfect_precision_cols_count += 1

                if missing_rows == 0 and too_much_rows == 0:
                    perfect_rows_count += 1
                if missing_cols == 0 and too_much_cols == 0:
                    perfect_cols_count += 1
                if missing_rows == 0 and missing_cols == 0 and too_much_rows == 0 and too_much_cols == 0:
                    perfect_rows_and_cols_count += 1

                missing_rows_counts.append(missing_rows)
                too_much_rows_counts.append(too_much_rows)
                missing_cols_counts.append(missing_cols)
                too_much_cols_counts.append(too_much_cols)

                total_oracle_rows += len(oracle_rows)
                total_caption_rows += len(caption_rows)
                total_intersecting_rows += len(oracle_rows.intersection(caption_rows))

                total_oracle_cols += len(oracle_cols)
                total_caption_cols += len(caption_cols)
                total_intersecting_cols += len(oracle_cols.intersection(caption_cols))

            recall_rows = total_intersecting_rows / total_oracle_rows
            precision_rows = total_intersecting_rows / total_caption_rows
            print(f"[Rows] Recall: {recall_rows:.4f}, Precision: {precision_rows:.4f}")

            perfect_recall_rows_percentage = (perfect_recall_rows_count / total_examples) * 100
            perfect_precision_rows_percentage = (perfect_precision_rows_count / total_examples) * 100
            print(
                f"[Rows] Examples with perfect recall: {perfect_recall_rows_percentage:.2f}% | Examples with perfect precision: {perfect_precision_rows_percentage:.2f}%"
            )

            recall_cols = total_intersecting_cols / total_oracle_cols
            precision_cols = total_intersecting_cols / total_caption_cols
            print(f"[Cols] Recall: {recall_cols:.4f}, Precision: {precision_cols:.4f}")

            perfect_recall_cols_percentage = (perfect_recall_cols_count / total_examples) * 100
            perfect_precision_cols_percentage = (perfect_precision_cols_count / total_examples) * 100
            print(
                f"[Cols] Examples with perfect recall: {perfect_recall_cols_percentage:.2f}% | Examples with perfect precision: {perfect_precision_cols_percentage:.2f}%"
            )

            perfect_rows_percentage = (perfect_rows_count / total_examples) * 100
            perfect_cols_percentage = (perfect_cols_count / total_examples) * 100
            perfect_rows_and_cols_percentage = (perfect_rows_and_cols_count / total_examples) * 100
            print(
                f"Examples with perfect rows: {perfect_rows_percentage:.2f}% | Examples with perfect cols: {perfect_cols_percentage:.2f}% | Examples with perfect rows and cols: {perfect_rows_and_cols_percentage:.2f}% (happened in {perfect_rows_and_cols_count} out of {total_examples} examples)"  # fmt: skip
            )

            # Plot row analysis (w/o percentage labels)
            plot_path_rows = os.path.join(sub_dataset_output_dir, f"{split}_rows_histogram.png")
            plot_histogram(
                missing_rows_counts,
                too_much_rows_counts,
                total_examples,
                f"{sub_dataset} - {split} - Row Analysis",
                "Number of Rows missing / too much",
                plot_path_rows,
                percentage_labels=False,
            )

            # Plot column analysis (w/o percentage labels)
            plot_path_cols = os.path.join(sub_dataset_output_dir, f"{split}_cols_histogram.png")
            plot_histogram(
                missing_cols_counts,
                too_much_cols_counts,
                total_examples,
                f"{sub_dataset} - {split} - Column Analysis",
                "Number of Columns missing / too much",
                plot_path_cols,
                percentage_labels=False,
            )

            # Plot row analysis (w/ percentage labels)
            percentage_labels_dir = os.path.join(sub_dataset_output_dir, "percentage_labels")
            os.makedirs(percentage_labels_dir, exist_ok=True)
            plot_path_rows_percentage = os.path.join(percentage_labels_dir, f"{split}_rows_histogram.png")  # fmt: skip
            plot_histogram(
                missing_rows_counts,
                too_much_rows_counts,
                total_examples,
                f"{sub_dataset} - {split} - Row Analysis",
                "Number of Rows missing / too much",
                plot_path_rows_percentage,
                percentage_labels=True,
            )

            # Plot column analysis (w/ percentage labels)
            plot_path_cols_percentage = os.path.join(percentage_labels_dir, f"{split}_cols_histogram.png")  # fmt: skip
            plot_histogram(
                missing_cols_counts,
                too_much_cols_counts,
                total_examples,
                f"{sub_dataset} - {split} - Column Analysis",
                "Number of Columns missing / too much",
                plot_path_cols_percentage,
                percentage_labels=True,
            )

            # Track for global statistics
            global_missing_rows_counts.extend(missing_rows_counts)
            global_too_much_rows_counts.extend(too_much_rows_counts)
            global_missing_cols_counts.extend(missing_cols_counts)
            global_too_much_cols_counts.extend(too_much_cols_counts)
            global_total_examples_count += total_examples
            global_perfect_rows_count += perfect_rows_count
            global_perfect_cols_count += perfect_cols_count
            global_perfect_rows_and_cols_count += perfect_rows_and_cols_count
            global_perfect_recall_rows_count += perfect_recall_rows_count
            global_perfect_precision_rows_count += perfect_precision_rows_count
            global_perfect_recall_cols_count += perfect_recall_cols_count
            global_perfect_precision_cols_count += perfect_precision_cols_count
            global_intersecting_rows += total_intersecting_rows
            global_oracle_rows += total_oracle_rows
            global_caption_rows += total_caption_rows
            global_intersecting_cols += total_intersecting_cols
            global_oracle_cols += total_oracle_cols
            global_caption_cols += total_caption_cols

    # print global statistics
    print(f"Global perfect rows: {global_perfect_rows_count} / {global_total_examples_count} = {global_perfect_rows_count / global_total_examples_count * 100:.2f}%")  # fmt: skip
    print(f"Global perfect cols: {global_perfect_cols_count} / {global_total_examples_count} = {global_perfect_cols_count / global_total_examples_count * 100:.2f}%")  # fmt: skip
    print(f"Global perfect rows and cols: {global_perfect_rows_and_cols_count} / {global_total_examples_count} = {global_perfect_rows_and_cols_count / global_total_examples_count * 100:.2f}%")  # fmt: skip
    print(f"Global perfect recall rows: {global_perfect_recall_rows_count} / {global_total_examples_count} = {global_perfect_recall_rows_count / global_total_examples_count * 100:.2f}%")  # fmt: skip
    print(f"Global perfect precision rows: {global_perfect_precision_rows_count} / {global_total_examples_count} = {global_perfect_precision_rows_count / global_total_examples_count * 100:.2f}%")  # fmt: skip
    print(f"Global perfect recall cols: {global_perfect_recall_cols_count} / {global_total_examples_count} = {global_perfect_recall_cols_count / global_total_examples_count * 100:.2f}%")  # fmt: skip
    print(f"Global perfect precision cols: {global_perfect_precision_cols_count} / {global_total_examples_count} = {global_perfect_precision_cols_count / global_total_examples_count * 100:.2f}%")  # fmt: skip

    # Global rows/cols precision and recall
    global_recall_rows = global_intersecting_rows / global_oracle_rows
    global_precision_rows = global_intersecting_rows / global_caption_rows
    print(f"Global rows recall: {global_recall_rows:.4f}, Global rows precision: {global_precision_rows:.4f}")
    global_recall_cols = global_intersecting_cols / global_oracle_cols
    global_precision_cols = global_intersecting_cols / global_caption_cols
    print(f"Global cols recall: {global_recall_cols:.4f}, Global cols precision: {global_precision_cols:.4f}")

    # Plot global histogram: Rows (w/o percentage labels)
    plot_histogram(
        global_missing_rows_counts,
        global_too_much_rows_counts,
        global_total_examples_count,
        "Global Row Analysis",
        "Number of Rows missing / too much",
        os.path.join(output_dir, "rows_histogram.png"),
        percentage_labels=False,
    )

    # Plot global histogram: Cols (w/o percentage labels)
    plot_histogram(
        global_missing_cols_counts,
        global_too_much_cols_counts,
        global_total_examples_count,
        "Global Column Analysis",
        "Number of Columns missing / too much",
        os.path.join(output_dir, "cols_histogram.png"),
        percentage_labels=False,
    )


if __name__ == "__main__":
    main()
