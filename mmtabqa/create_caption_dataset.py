"""
Example script demonstrating how to create and load captioned MMTabQA datasets.
"""

import argparse
import os
import subprocess
import sys

from datasets import load_from_disk
from load_mmtabqa_utils import create_captioned_dataset

from utils.runtime_env import configure_runtime_environment, load_repo_dotenv

configure_runtime_environment()
load_repo_dotenv()


def create_captioned_dataset_helper(
    dataset_base_path,
    dataset_paths,
    caption_mode,
    image_base_path,
    output_base_path=None,
    run_isolated_subprocesses=True,
    passthrough_args=None,
    n_examples=None,
):
    """
    Create captioned datasets.
    
    Args:
        dataset_base_path: Base path where input datasets are located
        dataset_paths: Dict mapping dataset names to their relative paths
        caption_mode: Caption generation mode
        image_base_path: Base path for images (can be None if images are in dataset)
        output_base_path: Custom base path for saving (defaults to dataset_base_path)
    """
    if output_base_path is None:
        output_base_path = dataset_base_path
    if passthrough_args is None:
        passthrough_args = []
    
    # Initialize cache dict once, then reuse across all datasets (AQ/EQ/VQ/IQ)
    share_between_dataset_dict = {}
    
    for dataset_name, dataset_path in dataset_paths.items():
        save_path = os.path.join(output_base_path, "mmtabqa_captioned", caption_mode, dataset_path)

        if os.path.exists(save_path):
            print(f"[SKIP] {dataset_name} already exists at {save_path}")
            continue

        if run_isolated_subprocesses:
            print(f"=== Creating Captioned Dataset {dataset_name} (isolated subprocess) ===")
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--caption_mode",
                caption_mode,
                "--single_dataset_name",
                dataset_name,
                "--single_dataset_path",
                dataset_path,
            ]
            if n_examples is not None:
                cmd.extend(["--n_examples", str(n_examples)])
            cmd.extend(passthrough_args)
            result = subprocess.run(cmd, env=os.environ.copy())
            if result.returncode != 0:
                raise RuntimeError(
                    f"Caption generation failed for {dataset_name} in mode {caption_mode} with exit code {result.returncode}"
                )
        else:
            print(f"=== Creating Captioned Dataset {dataset_name} ===")
            # Pass the accumulated cache dict to avoid re-captioning images from previous datasets
            share_between_dataset_dict = create_captioned_dataset(
                dataset_path=os.path.join(dataset_base_path, dataset_path),
                output_path=save_path,
                caption_mode=caption_mode,
                image_base_path=image_base_path,
                share_between_dataset_dict=share_between_dataset_dict,
                num_proc=10,
                num_gpus=1,
                n_examples=n_examples,
            )

        print("\n=== Loading Captioned Dataset ===")
        captioned_dataset = load_from_disk(save_path)

        print(f"Dataset contains {len(captioned_dataset)} examples")

        example = captioned_dataset[0]
        print("\nExample entry:")
        print(f"Question: {example['question']}")
        print(f"Answer: {example['answer_text']}")
        print(f"Table ID: {example['table_id']}")
        print(f"Page Title: {example['table']['page_title']}")
        print(f"Headers: {example['table']['header']}")

        print("\nFirst few table rows:")
        for i, row in enumerate(example["table"]["rows"][:3]):
            print(f"Row {i + 1}:")
            for j, (cell_type, content) in enumerate(zip(row["type"], row["content"])):
                print(f"  Cell {j + 1} ({cell_type}): {content[:100]}{'...' if len(str(content)) > 100 else ''}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--caption_mode", type=str, choices=["naive", "detailed", "short_with_context"]
    )
    parser.add_argument("--target_dataset", type=str, choices=["mmtabqa", "mmtbench"], default="mmtbench")
    parser.add_argument("--dataset_base_path", type=str, default=None)
    parser.add_argument("--output_base_path", type=str, default=None)
    parser.add_argument("--image_base_path", type=str, default=None)
    parser.add_argument("--n_examples", type=int, default=None)
    parser.add_argument("--single_dataset_name", type=str, default=None)
    parser.add_argument("--single_dataset_path", type=str, default=None)
    args = parser.parse_args()

    dataset_base_path_mmtabqa = os.getenv("MMTABQA_HF_DATASET_BASE_PATH")
    image_base_path_mmtabqa = os.getenv("MMTABQA_IMAGE_BASE_PATH")
    dataset_paths_mmtabqa = {
        "AQ-WikiTQ": "mmtabqa_wikitq_hf_dataset_explicit_ans_mention",
        "EQ-WikiTQ": "mmtabqa_wikitq_hf_dataset_explicit_questions",
        "VQ-WikiTQ": "mmtabqa_wikitq_hf_dataset_visual_questions",
        "IQ-WikiTQ": "mmtabqa_wikitq_hf_dataset_implicit_questions",
        "AQ-WikiSQL": "mmtabqa_wikisql_hf_dataset_explicit_ans_mention",
        "EQ-WikiSQL": "mmtabqa_wikisql_hf_dataset_explicit_questions",
        "VQ-WikiSQL": "mmtabqa_wikisql_hf_dataset_visual_questions",
        "IQ-WikiSQL": "mmtabqa_wikisql_hf_dataset_implicit_questions",
        "AQ-FetaQA": "mmtabqa_fetaqa_hf_dataset_explicit_ans_mention",
        "EQ-FetaQA": "mmtabqa_fetaqa_hf_dataset_explicit_questions",
        "VQ-FetaQA": "mmtabqa_fetaqa_hf_dataset_visual_questions",
        "IQ-FetaQA": "mmtabqa_fetaqa_hf_dataset_implicit_questions",
        "AQ-HybridQA": "mmtabqa_hybridqa_hf_dataset_explicit_ans_mention",
        "EQ-HybridQA": "mmtabqa_hybridqa_hf_dataset_explicit_questions",
        "VQ-HybridQA": "mmtabqa_hybridqa_hf_dataset_visual_questions",
        "IQ-HybridQA": "mmtabqa_hybridqa_hf_dataset_implicit_questions",
    }

    # MMTabReal (MMTBench)
    mmtbench_base_path = os.getenv("MMT_BENCH_BASE_PATH")
    dataset_base_path_mmtbench = os.path.join(mmtbench_base_path, "hf_dataset") if mmtbench_base_path else None
    output_base_path_mmtbench = (
        os.path.join(mmtbench_base_path, "converted_to_hf_dataset") if mmtbench_base_path else None
    )
    dataset_paths_mmtbench = {
        "MMTabReal - VQ": "mmtabreal_VQ",
        "MMTabReal - AQ": "mmtabreal_AQ",
        "MMTabReal - EQ": "mmtabreal_EQ",
        "MMTabReal - IQ": "mmtabreal_IQ",
    }

    if args.target_dataset == "mmtabqa":
        dataset_base_path = args.dataset_base_path or dataset_base_path_mmtabqa
        output_base_path = args.output_base_path or dataset_base_path
        image_base_path = args.image_base_path or image_base_path_mmtabqa
        dataset_paths = dataset_paths_mmtabqa
    else:
        dataset_base_path = args.dataset_base_path or dataset_base_path_mmtbench
        output_base_path = args.output_base_path or output_base_path_mmtbench
        image_base_path = args.image_base_path
        dataset_paths = dataset_paths_mmtbench

    if dataset_base_path is None:
        raise ValueError("dataset base path is not set. Use --dataset_base_path or set the dataset env var.")
    if output_base_path is None:
        raise ValueError("output base path is not set. Use --output_base_path or set the dataset env var.")

    passthrough_args = ["--target_dataset", args.target_dataset]
    if args.dataset_base_path:
        passthrough_args.extend(["--dataset_base_path", args.dataset_base_path])
    if args.output_base_path:
        passthrough_args.extend(["--output_base_path", args.output_base_path])
    if args.image_base_path:
        passthrough_args.extend(["--image_base_path", args.image_base_path])
    if args.n_examples is not None:
        passthrough_args.extend(["--n_examples", str(args.n_examples)])

    run_isolated_subprocesses = False  # Disabled to allow cache sharing across datasets (AQ/EQ/VQ/IQ)
    if args.single_dataset_name and args.single_dataset_path:
        dataset_paths = {args.single_dataset_name: args.single_dataset_path}
        run_isolated_subprocesses = False

    # MMTBench uses image paths embedded in the dataset. MMTabQA uses image_base_path.
    create_captioned_dataset_helper(
        dataset_base_path,
        dataset_paths,
        args.caption_mode,
        image_base_path,
        output_base_path=output_base_path,
        run_isolated_subprocesses=run_isolated_subprocesses,
        passthrough_args=passthrough_args,
        n_examples=args.n_examples,
    )


if __name__ == "__main__":
    main()
