"""
Code to handle loading all the different datasets for step 1.

This code is used in the col_sql.py script.
"""

import os
import random
from argparse import ArgumentParser
from pathlib import Path

import dotenv

from datasets import load_from_disk
from mmtabqa.load_mmtabqa_utils import load_mmtabqa_dataset
from utils.utils import load_dataset_from_args


dotenv.load_dotenv("../.env")

MMTABQA_BASE_PATH = os.getenv("MMTABQA_BASE_PATH")
MMT_BENCH_BASE_PATH = os.getenv("MMT_BENCH_BASE_PATH")

DATASET_PATHS = {
    "wikitq": {
        "base_path": MMTABQA_BASE_PATH,
        "AQ": "mmtabqa_wikitq_hf_dataset_explicit_ans_mention",
        "EQ": "mmtabqa_wikitq_hf_dataset_explicit_questions",
        "VQ": "mmtabqa_wikitq_hf_dataset_visual_questions",
        "IQ": "mmtabqa_wikitq_hf_dataset_implicit_questions",
    },
    "wikisql": {
        "base_path": MMTABQA_BASE_PATH,
        "AQ": "mmtabqa_wikisql_hf_dataset_explicit_ans_mention",
        "IQ": "mmtabqa_wikisql_hf_dataset_implicit_questions",
        "EQ": "mmtabqa_wikisql_hf_dataset_explicit_questions",
        "VQ": "mmtabqa_wikisql_hf_dataset_visual_questions",
    },
    "fetaqa": {
        "base_path": MMTABQA_BASE_PATH,
        "AQ": "mmtabqa_fetaqa_hf_dataset_explicit_ans_mention",
        "IQ": "mmtabqa_fetaqa_hf_dataset_implicit_questions",
        "EQ": "mmtabqa_fetaqa_hf_dataset_explicit_questions",
        "VQ": "mmtabqa_fetaqa_hf_dataset_visual_questions",
    },
    "hybridqa": {
        "base_path": MMTABQA_BASE_PATH,
        "AQ": "mmtabqa_hybridqa_hf_dataset_explicit_ans_mention",
        "EQ": "mmtabqa_hybridqa_hf_dataset_explicit_questions",
        "VQ": "mmtabqa_hybridqa_hf_dataset_visual_questions",
        "IQ": "mmtabqa_hybridqa_hf_dataset_implicit_questions",
    },
    "mmtbench": {
        "base_path": MMT_BENCH_BASE_PATH,
        "AQ": "mmtabreal_AQ",
        "EQ": "mmtabreal_EQ",
        "IQ": "mmtabreal_IQ",
        "VQ": "mmtabreal_VQ",
    },
}


def mmtabqa_dataset(
    args: ArgumentParser,
    load_captioned: bool = False,
    partial_input_baseline: bool = False,
    gold_entity_replaced_images: bool = False,
):
    dataset_path = DATASET_PATHS[args.mmtabqa_sub_dataset][args.dataset_split]
    base_path = Path(DATASET_PATHS[args.mmtabqa_sub_dataset]["base_path"])

    # MMTabReal (mmtbench) datasets are in hf_dataset/ directory instead of converted_to_hf_dataset/
    is_mmtabreal = args.mmtabqa_sub_dataset == "mmtbench"

    if load_captioned:
        if is_mmtabreal:
            # For MMTabReal, check if captioned version exists, otherwise use hf_dataset
            captioned_path = base_path / "converted_to_hf_dataset" / "mmtabqa_captioned" / args.caption_type / dataset_path
            print(f"📂 Checking for captioned dataset at: {captioned_path}")
            print(f"   exists: {captioned_path.exists()}")
            if captioned_path.exists():
                dataset_path = captioned_path
                print(f"✅ Using captioned dataset")
            else:
                print(f"\n⚠️  WARNING: Captioned MMTabReal dataset not found at {captioned_path}")
                print(f"    Using raw dataset from hf_dataset/ directory instead.")
                print(f"    To use captions, generate them first (see mmtabqa/README.md)\n")
                dataset_path = base_path / "hf_dataset" / dataset_path
            dataset = load_from_disk(dataset_path)
        else:
            dataset_path = base_path / "converted_to_hf_dataset" / "mmtabqa_captioned" / args.caption_type / dataset_path  # fmt: skip
            dataset = load_from_disk(dataset_path)
    elif partial_input_baseline:
        if is_mmtabreal:
            dataset_path = base_path / "hf_dataset" / dataset_path
        else:
            dataset_path = base_path / "converted_to_hf_dataset" / dataset_path
        dataset = load_mmtabqa_dataset(
            dataset_path=dataset_path,
            num_proc=10,
            partial_input_baseline=True,
            load_images=False,
        )
    elif gold_entity_replaced_images:
        if is_mmtabreal:
            dataset_path = base_path / "hf_dataset" / dataset_path
        else:
            dataset_path = base_path / "converted_to_hf_dataset" / dataset_path
        dataset = load_mmtabqa_dataset(
            dataset_path=dataset_path,
            num_proc=10,
            gold_entity_replaced_images=True,
            dataset_name=args.mmtabqa_sub_dataset,
            mmtabqa_dataset_dir=str(base_path),
        )
    else:
        raise ValueError("One of load_captioned, partial_input_baseline, and gold_entity_replaced_images must be True")

    dataset = dataset.rename_column("table", "table_with_metadata")

    return dataset


def load_default_hstar_dataset(args: ArgumentParser):
    """
    load default H-STAR datasets (WikiTableQuestions and TabFact).
    """
    dataset = load_dataset_from_args(args)

    def _add_metadata_to_dataset(example):
        new_rows = []
        for row in example["table"]["rows"]:
            new_rows.append(
                {
                    "content": row,
                    "type": ["text"] * len(row),
                }
            )
        example["table"]["rows"] = new_rows
        return example

    # The dataset contains a "table" and the "rows" are directly the list of values. We need to add the metadata to the dataset.
    dataset = dataset.map(_add_metadata_to_dataset)

    # rename the "table" key to "table_with_metadata"
    dataset = dataset.rename_column("table", "table_with_metadata")

    return dataset


def _sanity_check_data_item_loading(data_item, args):
    # Sanity check: check that each data_item now follows our new expected format:
    #   - "table" is not present anymore but is called "table_with_metadata"
    #   - the "rows" in "table_with_metadata" now contain a "content" list and "type" list which are of same length.
    assert "table" not in data_item, "table should not be present in the data_item"
    assert "table_with_metadata" in data_item, "table_with_metadata should be present in the data_item"
    assert len(data_item["table_with_metadata"]["rows"][0]["content"]) == len(data_item["table_with_metadata"]["rows"][0]["type"]), "the number of content and type lists should be the same"  # fmt: skip
    assert all(isinstance(content, str) for content in data_item["table_with_metadata"]["rows"][0]["content"]), "the content list should contain only strings"  # fmt: skip
    assert all(isinstance(type, str) for type in data_item["table_with_metadata"]["rows"][0]["type"]), "the type list should contain only strings"  # fmt: skip


def sample_dataset(dataset, n_examples: int = None, seed: int = 42):
    """
    Deterministically sample n_examples from a dataset.

    Args:
        dataset: HuggingFace dataset to sample from
        n_examples: Number of examples to sample. If None or >= len(dataset), returns full dataset.
        seed: Random seed for reproducibility

    Returns:
        Sampled dataset (or original if n_examples is None/larger than dataset)
    """
    if n_examples is None or n_examples >= len(dataset):
        return dataset

    random.seed(seed)
    total_size = len(dataset)
    sample_size = min(n_examples, total_size)
    indices = random.sample(range(total_size), sample_size)
    indices.sort()  # Sort to maintain some order

    print(f"📌 Sampling {sample_size} examples from {total_size} total (seed={seed})")
    return dataset.select(indices)


def load_dataset_step_1(args: ArgumentParser):
    # if dataset is mmtabqa, we need to load the dataset differently
    if args.dataset_version == "partial":
        print("\n\n📌 Loading the PARTIAL MMTabQA dataset for H-STAR. This means: every unique image in an MMTabQA example will be repalced with \"ENTITY-1\", \"ENTITY-2\", ... strings \n\n")  # fmt: skip
        dataset = mmtabqa_dataset(args, partial_input_baseline=True)
    elif args.dataset_version == "captioned":
        print("\n\n📌 Loading the CAPTIONED MMTabQA dataset for H-STAR. This means: every image in MMTabQA will be replaced with its caption (needs the captions to be generated first; see `./mmtabqa/README.md`).\n\n")  # fmt: skip
        dataset = mmtabqa_dataset(args, load_captioned=True)
    elif args.dataset_version == "gold_entity_replaced":
        print("\n\n📌 Loading the GOLD ENTITY REPLACED IMAGES MMTabQA dataset for H-STAR. This means: every image in MMTabQA will be replaced with its original entity.\n\n")  # fmt: skip
        dataset = mmtabqa_dataset(args, gold_entity_replaced_images=True)
    else:
        dataset = load_default_hstar_dataset(args)

    # Apply n_examples sampling if specified
    n_examples = getattr(args, "n_examples", None)
    if n_examples is not None:
        dataset = sample_dataset(dataset, n_examples=n_examples, seed=42)

    col_dict = {}
    for eid in range(len(dataset)):
        data_item = dataset[eid]
        _sanity_check_data_item_loading(data_item, args)
        col_dict[str(eid)] = {"ori_data_item": data_item}

    return col_dict
