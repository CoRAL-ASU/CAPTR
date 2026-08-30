import argparse
import json
import os
import pathlib
import random
import sys
from datetime import datetime
from multiprocessing import Pool

# Add project root to path to enable imports from scripts/
project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# Remove it first if it exists to ensure it's at position 0
if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)

import dotenv
import pandas as pd
from tqdm import tqdm

from utils.runtime_env import load_repo_dotenv

load_repo_dotenv()

from baselines.exact_match import EvaluationMetrics
from generation.generator_vllm_gemma3 import VLLMGeneratorGemma3
from generation.generator_vllm_qwen3vl import VLLMGeneratorQwen3VL
from scripts.final_evaluate import final_evaluate_default
from scripts.llm_as_a_judge import evaluate_AQ_via_LLM_as_a_judge


# Make paths optional - handle case where env vars are not set
MMTABQA_HF_DATASET_BASE_PATH_STR = os.getenv("MMTABQA_HF_DATASET_BASE_PATH")
MMTABQA_HF_DATASET_BASE_PATH = pathlib.Path(MMTABQA_HF_DATASET_BASE_PATH_STR) if MMTABQA_HF_DATASET_BASE_PATH_STR else None

DATASET_PATHS = {}
if MMTABQA_HF_DATASET_BASE_PATH:
    DATASET_PATHS = {
        "WikiTQ": {
            "AQ": MMTABQA_HF_DATASET_BASE_PATH / "mmtabqa_wikitq_hf_dataset_explicit_ans_mention",
            "EQ": MMTABQA_HF_DATASET_BASE_PATH / "mmtabqa_wikitq_hf_dataset_explicit_questions",
            "VQ": MMTABQA_HF_DATASET_BASE_PATH / "mmtabqa_wikitq_hf_dataset_visual_questions",
            "IQ": MMTABQA_HF_DATASET_BASE_PATH / "mmtabqa_wikitq_hf_dataset_implicit_questions",
        },
        "WikiSQL": {
            "AQ": MMTABQA_HF_DATASET_BASE_PATH / "mmtabqa_wikisql_hf_dataset_explicit_ans_mention",
            "EQ": MMTABQA_HF_DATASET_BASE_PATH / "mmtabqa_wikisql_hf_dataset_explicit_questions",
            "IQ": MMTABQA_HF_DATASET_BASE_PATH / "mmtabqa_wikisql_hf_dataset_implicit_questions",
            "VQ": MMTABQA_HF_DATASET_BASE_PATH / "mmtabqa_wikisql_hf_dataset_visual_questions",
        },
        "FetaQA": {
            "AQ": MMTABQA_HF_DATASET_BASE_PATH / "mmtabqa_fetaqa_hf_dataset_explicit_ans_mention",
            "EQ": MMTABQA_HF_DATASET_BASE_PATH / "mmtabqa_fetaqa_hf_dataset_explicit_questions",
            "IQ": MMTABQA_HF_DATASET_BASE_PATH / "mmtabqa_fetaqa_hf_dataset_implicit_questions",
            "VQ": MMTABQA_HF_DATASET_BASE_PATH / "mmtabqa_fetaqa_hf_dataset_visual_questions",
        },
        "HybridQA": {
            "AQ": MMTABQA_HF_DATASET_BASE_PATH / "mmtabqa_hybridqa_hf_dataset_explicit_ans_mention",
            "EQ": MMTABQA_HF_DATASET_BASE_PATH / "mmtabqa_hybridqa_hf_dataset_explicit_questions",
            "VQ": MMTABQA_HF_DATASET_BASE_PATH / "mmtabqa_hybridqa_hf_dataset_visual_questions",
            "IQ": MMTABQA_HF_DATASET_BASE_PATH / "mmtabqa_hybridqa_hf_dataset_implicit_questions",
        },
    }

MMTABQA_BASE_PATH_STR = os.getenv("MMTABQA_BASE_PATH")
MMTABQA_BASE_PATH = pathlib.Path(MMTABQA_BASE_PATH_STR) if MMTABQA_BASE_PATH_STR else None


def resolve_local_model_path(model_name: str) -> str:
    """
    Canonicalize local Hugging Face cache snapshot paths while leaving repo IDs untouched.

    If the path exists, it is resolved to an absolute canonical path. If it points at a
    snapshot directory with a truncated hash prefix, the unique matching snapshot is used.
    """
    if not model_name:
        return model_name

    model_path = pathlib.Path(model_name)
    if model_path.exists():
        return str(model_path.resolve())

    if model_path.parent.name == "snapshots":
        snapshot_matches = [candidate for candidate in model_path.parent.glob(f"{model_path.name}*") if candidate.is_dir()]
        if len(snapshot_matches) == 1:
            return str(snapshot_matches[0].resolve())

    return model_name


def read_local_model_architectures(model_name: str) -> list[str]:
    """
    Read architectures from a local HF-style model directory if available.
    Returns an empty list when unavailable.
    """
    model_path = pathlib.Path(model_name)
    if not model_path.exists() or not model_path.is_dir():
        return []

    config_path = model_path / "config.json"
    if not config_path.exists():
        return []

    try:
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
        architectures = config.get("architectures", [])
        return architectures if isinstance(architectures, list) else []
    except Exception:
        return []

# Load table metadata only if base path is available
if MMTABQA_BASE_PATH:
    WIKITQ_TABLE_METADATA = pd.read_csv(
        MMTABQA_BASE_PATH / "WikiTableQuestions" / "table-metadata.tsv",
        sep="\t",
    ).set_index("contextId")
else:
    WIKITQ_TABLE_METADATA = None


def get_metadata(example):
    """
    Get a string representing the metadata for a table example.
    """
    question = example["question"]
    page_title = example["table"]["page_title"]
    section_title = example["table"]["section_title"]

    if section_title and section_title != "":
        table_metadata = f"Table related to {section_title} in context of {page_title}."
    else:
        table_metadata = f"Table in context of {page_title}."

    return table_metadata, question


################################ Dataset Sampling #################################


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

    print(f"Sampling {sample_size} examples from {total_size} total (seed={seed})")
    return dataset.select(indices)


################################ vllm model call #################################


def get_vllm_generator(
    model_name: str = "google/gemma-3-27b-it",
    prompt_mode: str = "text-image",
    number_of_gpus: int = 2,
    limit_mm_per_prompt: dict = None,
    system_prompt: str = None,
    context_window_size: int = 90000,
    gpu_memory_utilization: float = 0.9,
):
    """
    Get a vLLM generator for the specified model.
    Automatically selects the correct generator class based on model name.

    Args:
        model_name: HuggingFace model name (e.g., "google/gemma-3-27b-it", "Qwen/Qwen3-VL-32B-Instruct")
        prompt_mode: Prompt mode ("text-image" for multimodal)
        number_of_gpus: Number of GPUs for tensor parallelism
        limit_mm_per_prompt: Maximum images per prompt
        system_prompt: System prompt (None to disable)
        context_window_size: Maximum context window size
        gpu_memory_utilization: Fraction of GPU memory to reserve for vLLM

    Returns:
        Tuple of (generator, model_name_short)
    """
    # Don't override cache dirs - they're already set in the calling script
    # Just ensure the token is preserved if it exists
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    model_name = resolve_local_model_path(model_name)
    architectures = read_local_model_architectures(model_name)

    if "Qwen3_5ForConditionalGeneration" in architectures:
        warning_message = (
            "Selected model architecture 'Qwen3_5ForConditionalGeneration' may not be supported by your current vLLM build. "
            "Proceeding with model initialization and letting vLLM decide. "
            "If loading fails, use a supported VL checkpoint (for example Qwen3-VL / Qwen2.5-VL) "
            "or upgrade vLLM to a version that supports Qwen3.5. "
            f"Resolved model path: {model_name}"
        )
        if os.getenv("STRICT_MODEL_ARCH_CHECK", "0") == "1":
            raise ValueError(warning_message)
        print(f"Warning: {warning_message}")
    
    args = argparse.Namespace()
    args.engine = "vllm"
    args.vllm_model_name = model_name
    args.number_of_gpus = number_of_gpus
    args.max_api_total_tokens = context_window_size
    args.prompt_mode = prompt_mode
    args.gpu_memory_utilization = gpu_memory_utilization
    args.prompt_style = ""  # not needed for image captioning (but if not specified we get an error)
    args.seed = 42  # also not needed (but if not specified we get an error)

    model_name_short = model_name.split("/")[-1]

    # Select generator class based on model name
    model_name_lower = model_name.lower()
    if "gemma" in model_name_lower:
        generator_cls = VLLMGeneratorGemma3
    elif (
        "qwen3-vl" in model_name_lower
        or "qwen3vl" in model_name_lower
        or "qwen3.5" in model_name_lower
        or "qwen3_5" in model_name_lower
        or "qwen3_5forconditionalgeneration" in " ".join(architectures).lower()
    ):
        generator_cls = VLLMGeneratorQwen3VL
    else:
        # Default to Gemma3 for backwards compatibility
        generator_cls = VLLMGeneratorGemma3

    print(f"Using {generator_cls.__name__} for model: {model_name}")

    if limit_mm_per_prompt:
        return generator_cls(
            args,
            limit_mm_per_prompt=limit_mm_per_prompt,
            system_prompt=system_prompt,
            gpu_memory_utilization=gpu_memory_utilization,
        ), model_name_short
    return generator_cls(
        args,
        system_prompt=system_prompt,
        gpu_memory_utilization=gpu_memory_utilization,
    ), model_name_short


# Backwards compatibility alias
def get_gemma3_generator(
    model_name: str = "google/gemma-3-27b-it",
    prompt_mode: str = "text-image",
    number_of_gpus: int = 2,
    limit_mm_per_prompt: dict = None,
    system_prompt: str = None,
    context_window_size: int = 90000,
    gpu_memory_utilization: float = 0.75,
):
    """Backwards compatible wrapper for get_vllm_generator."""
    return get_vllm_generator(
        model_name=model_name,
        prompt_mode=prompt_mode,
        number_of_gpus=number_of_gpus,
        limit_mm_per_prompt=limit_mm_per_prompt,
        system_prompt=system_prompt,
        context_window_size=context_window_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def store_generations(outputs, model_name_short, dataset_name, dataset_split_name, mode="partial_input"):
    # create results folder if it doesn't exist
    os.makedirs("results", exist_ok=True)
    os.makedirs(f"results/{model_name_short}", exist_ok=True)
    with open(f"results/{model_name_short}/{dataset_name}_{dataset_split_name}_{mode}.json", "w") as f:
        json.dump(outputs, f)


################################## evaluation #################################


def _evaluate_mmtabqa_rouge_bleu(outputs):
    """
    Calculate the rouge, bleu and bleurt scores for mmtabqa datasets
    """

    predictions = []
    ground_truths = []
    for i in outputs:
        prediction = outputs[i]["generations"][0]
        prediction = prediction.split("Step 2:")[-1].strip()
        if prediction.startswith("['"):  # don't know if these are necessary
            prediction = prediction[2:-2]
        if prediction and prediction[0] == "[":
            prediction = prediction[1:-1]
        predictions.append(prediction)
        ground_truths.append(outputs[i]["answer_text"])

    evaluator = EvaluationMetrics()
    # bleu_score = evaluator.get_bleu_score(predictions, ground_truths)
    rouge_score = evaluator.get_rouge_score(predictions, ground_truths)
    # bleurt_score = evaluator.get_bleurt_score(predictions, ground_truths)

    rouge_l = rouge_score["rougeL"]

    print(f"\n\nROUGE-L Score: {rouge_l}\n\n\n\n")
    return rouge_l


def _evaluate_single_example_em(args):
    i, output = args
    generated_text = output["generations"][0]
    gold_answer = output["answer_text"]

    prediction = generated_text.split("Step 2:")[-1].strip()
    if prediction.startswith("['"):
        prediction = prediction[2:-2]
    if prediction and prediction[0] == "[":
        prediction = prediction[1:-1]
    pred_answer = prediction

    if isinstance(gold_answer, list):
        gold_answer = ", ".join([str(_) for _ in gold_answer])
    evaluator = EvaluationMetrics()
    score = evaluator.compute_exact_match(gold_answer, pred_answer, dataset="wikitq")

    # Return debug info for first few examples
    debug_info = None
    if int(i) < 10:
        debug_info = {
            "example_id": i,
            "question": output["question"],
            "generated_text": generated_text,
            "pred_answer": pred_answer,
            "gold": gold_answer,
            "score": score,
        }

    return score, debug_info


def _evaluate_mmtabqa_em(outputs):
    """
    Calculate the EM score for mmtabqa datasets
    """
    # Prepare arguments for parallel processing
    args_list = [(i, outputs[str(i)]) for i in outputs]

    # Process in parallel with progress bar
    with Pool(processes=8) as pool:
        results = list(
            tqdm(pool.imap(_evaluate_single_example_em, args_list), total=len(args_list), desc="Evaluating examples")
        )

    # Collect results
    acc = 0
    for score, debug_info in results:
        acc += score

        # Print debug info for first few examples
        if debug_info:
            print(f"\n\n\n\nExample {debug_info['example_id']}:")
            print(f"Question: '{debug_info['question']}'")
            print(f"Generated text: '{debug_info['generated_text']}'")
            print(f"pred_answer type: {type(debug_info['pred_answer'])}")
            print(f"\nPredicted answer: '{debug_info['pred_answer']}'")
            print(f"Gold: '{debug_info['gold']}'")
            print(f"gold type: {type(debug_info['gold'])}")
            print(f"gold[0] == pred_answer: {debug_info['gold'][0] == debug_info['pred_answer']}")
            print(f"Score: {debug_info['score']}")

    print(f"\n\nCorrect Samples: {acc}; Total Samples: {len(outputs)}")
    print(f"Accuracy: {100 * acc / len(outputs):.2f}\n\n\n\n")
    return acc


def evaluate(
    outputs, generator, dataset_name, dataset_split, model_name_short, mode="partial_input", use_llm_as_judge=True
):
    """
    Evaluate the outputs of a baseline. This function uses 2 ways to evaluate:
    1. EM metric as implemented in the MMTabQA paper
    2. EM metric as implemented in the H-STAR paper

    For AQ questions, we use LLM-as-a-judge to evaluate (in addition to the default EM / ROUGE-L metrics).

    Args:
        outputs: dict of dicts, each containing the output of a single example: {"0": {"question": str, "answer_text": str, "generations": list[str]}, ...}
        generator: (vllm) generator
        dataset_name: name of the dataset
        dataset_split: name of the dataset split
        model_name: name of the model
        use_llm_as_judge: whether to use LLM-as-a-judge to evaluate
    """

    if use_llm_as_judge:
        readme_file = f"results/results_{model_name_short}_{mode}.md"
        evaluate_AQ_via_LLM_as_a_judge(
            outputs=outputs,
            generator=generator,
            mode="mmtabqa_baseline",
            readme_file=readme_file,
            mmtabqa_sub_dataset=dataset_name,
            mmtabqa_dataset_split=dataset_split,
            results_file=f"results/{model_name_short}/{dataset_name}_{dataset_split}_{mode}.json",
        )
        # Skip standard EM metrics when using LLM-as-a-judge
        return

    if dataset_name == "WikiTQ" or dataset_name == "WikiSQL" or dataset_name == "HybridQA" or dataset_name == "MMTabReal":
        acc_mmtabqa = _evaluate_mmtabqa_em(outputs)
        acc_hstar = final_evaluate_default(
            outputs,
            mode="mmtabqa_baseline",
            dataset=dataset_name,
            mmtabqa_sub_dataset=dataset_name,
            mmtabqa_dataset_split=dataset_split,
        )

        # Use appropriate metric names based on mode
        if mode == "sparsevlm":
            accuracy = 100 * acc_hstar / len(outputs)
            method_name = "SparseVLM"
            print(f"\n\n\nExact Match Accuracy: {accuracy:.2f}%\n\n\n")
            
            with open(f"results/results_{mode}.md", "a") as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {model_name_short}: {dataset_name} - {dataset_split} - {method_name} - {accuracy:.2f}%\n")
        else:
            print(f"\n\n\nAccuracy (MMTabQA): {100 * acc_mmtabqa / len(outputs):.2f}\nAccuracy (H-STAR): {100 * acc_hstar / len(outputs):.2f}\n\n\n")  # fmt: skip
            
            with open(f"results/results_{mode}.md", "a") as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {model_name_short}: {dataset_name} - {dataset_split} - Accuracy: {100 * acc_mmtabqa / len(outputs):.2f}% (MMTabQA) - {100 * acc_hstar / len(outputs):.2f}% (H-STAR)\n")  # fmt: skip

    elif dataset_name == "FetaQA":
        # bleu_score, rouge_score, bleurt_score = _evaluate_mmtabqa_rouge_bleu(outputs)
        rouge_l = _evaluate_mmtabqa_rouge_bleu(outputs)
        print(f"\n\n\nROUGE-L Score: {rouge_l}\n\n\n\n")
        with open(f"results/results_{mode}.md", "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {model_name_short}: {dataset_name} - {dataset_split}: ROUGE-L Score: {rouge_l}\n")  # fmt: skip

    else:
        raise ValueError(f"Dataset name '{dataset_name}' is not a valid dataset name")
