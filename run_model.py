import argparse
import copy
import json
import os
import re
from argparse import Namespace
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from utils.runtime_env import configure_runtime_environment, expand_env_values

configure_runtime_environment()
from scripts.col_sql import main as col_sql_main
from scripts.col_text import main as col_text_main
from scripts.final_evaluate import main as final_evaluate_main
from scripts.reason_sql import main as reason_sql_main
from scripts.reason_text import main as reason_text_main
from scripts.row_sql import main as row_sql_main
from scripts.row_text import main as row_text_main
from utils.utils import get_generator_and_tokenizer

ROOT_DIR = os.path.join(os.path.dirname(__file__))
TOKENIZER_FALSE = "export TOKENIZERS_PARALLELISM=false\n"

OUTPUT_FILES = {
    1: "step1_col_sql.json",
    2: "step2_col_text.json",
    3: "step3_row_sql.json",
    4: "step4_row_text.json",
    5: "hstar_reasoning_step5_sql_reason.json",
    6: "hstar_reasoning_step6_text_reason.json",
}

README_TEMPLATE = """# Experiment: {experiment_name}

## Configuration
- Source: {config_path}
- Dataset: {dataset_name}
- Engine: {engine}

## Steps Executed
{steps_executed}

## Steps Skipped
{steps_skipped}
"""


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    config = expand_env_values(config)

    # Override with config values, falling back to existing (default) if key is missing
    parsed_config = {
        "engine": config["engine"],
        "dataset_name": config["dataset"],
        "dataset_version": config.get("dataset_version", None),
        "dataset_split": config.get("dataset_split", "test"),  # default for wikitq is "test"
        "experiment_name": config["experiment_name"],
        "run_wikitq_sql_col": "step1" in config["run_steps"],
        "run_wikitq_text_col": "step2" in config["run_steps"],
        "run_wikitq_sql_row": "step3" in config["run_steps"],
        "run_wikitq_text_row": "step4" in config["run_steps"],
        "run_wikitq_sql_reason": "step5" in config["run_steps"],
        "run_wikitq_text_reason": "step6" in config["run_steps"],
        "run_final_evaluation": config["run_final_evaluation"],
        "skip_step1": "step1" in config["skip_steps"],
        "skip_step2": "step2" in config["skip_steps"],
        "skip_step3": "step3" in config["skip_steps"],
        "skip_step4": "step4" in config["skip_steps"],
        "skip_step5": "step5" in config["skip_steps"],
        "config_path": config_path,
        "load_images_for_reasoning": config.get("load_images_for_reasoning", False),
        "use_mmtabqa_interleaved_prompt": config.get("use_mmtabqa_interleaved_prompt", False),
        "temperature": config.get("temperature", 0.3),
        "max_generation_tokens": config.get("max_generation_tokens", 1024),
        # Add vllm specific arguments
        "vllm_model_name": config.get("vllm_model_name"),
        "number_of_gpus": config.get("number_of_gpus"),
        "prompt_mode": config.get("prompt_mode"),
        "gpu_memory_utilization": config.get("gpu_memory_utilization", None),
        # misc arguments
        "caption_type": config.get("caption_type"),
        "filtering_as_remover": config.get(
            "filtering_as_remover", False
        ),  # If true: remove irrelevant cols/rows. If false: select relevant cols/rows
        "include_captions_in_reasoner": config.get("include_captions_in_reasoner", False),
        "reasoner_answer_only": config.get("reasoner_answer_only", None),
        "reason_temperature": config.get("reason_temperature", 0.0),
        "reason_sampling_n": config.get("reason_sampling_n", 1),
        "reason_max_generation_tokens": config.get("reason_max_generation_tokens", 512),
        # Sampling arguments
        "n_examples": config.get("n_examples", None),  # Number of examples to sample (None = all)
    }

    if parsed_config["reasoner_answer_only"] is None:
        model_name = str(parsed_config.get("vllm_model_name", "")).lower()
        parsed_config["reasoner_answer_only"] = (
            parsed_config.get("dataset_name") == "mmtbench_only"
            and ("qwen3.5" in model_name or "qwen3_5" in model_name)
        )

    if parsed_config["filtering_as_remover"]:
        parsed_config["experiment_name"] = f"{parsed_config['experiment_name']}/filtering_as_remover"

    # sanity check skipping:
    if parsed_config["skip_step1"] and parsed_config["run_wikitq_sql_col"]:
        raise ValueError("`skip_step1` is true but so is `run_wikitq_sql_col`. This makes no sense: either set to skip or run")  # fmt: skip
    if parsed_config["skip_step2"] and parsed_config["run_wikitq_text_col"]:
        raise ValueError("`skip_step2` is true but so is `run_wikitq_text_col`. This makes no sense: either set to skip or run")  # fmt: skip
    if parsed_config["skip_step3"] and parsed_config["run_wikitq_sql_row"]:
        raise ValueError("`skip_step3` is true but so is `run_wikitq_sql_row`. This makes no sense: either set to skip or run")  # fmt: skip
    if parsed_config["skip_step4"] and parsed_config["run_wikitq_text_row"]:
        raise ValueError("`skip_step4` is true but so is `run_wikitq_text_row`. This makes no sense: either set to skip or run")  # fmt: skip
    if parsed_config["skip_step5"] and parsed_config["run_wikitq_sql_reason"]:
        raise ValueError("`skip_step5` is true but so is `run_wikitq_sql_reason`. This makes no sense: either set to skip or run")  # fmt: skip

    return parsed_config


def _set_default_max_api_total_tokens(config):
    """Set a safe model-aware default unless the config explicitly overrides it."""
    if config.get("max_api_total_tokens") is not None:
        return

    model_name = str(config.get("vllm_model_name", "")).lower()
    if "gemma" in model_name:
        # Gemma captioned prompts are the longest in this pipeline.
        config["max_api_total_tokens"] = 65000
    elif "qwen3-vl" in model_name or "qwen" in model_name:
        config["max_api_total_tokens"] = 55000
    else:
        config["max_api_total_tokens"] = 50000


def create_step_args(config, **kwargs):
    args = Namespace(
        dataset=config["dataset_name"],
        dataset_version=config.get("dataset_version", None),
        dataset_split=config["dataset_split"],
        engine=config["engine"],
        save_dir=f"results/{config['experiment_name']}",
        vllm_model_name=config.get("vllm_model_name"),
        number_of_gpus=config.get("number_of_gpus"),
        prompt_mode=config.get("prompt_mode"),
        load_images_for_reasoning=config.get("load_images_for_reasoning", False),
        caption_type=config.get("caption_type"),
        temperature=config.get("temperature", 0.3),
        max_generation_tokens=config.get("max_generation_tokens", 1024),
        max_api_total_tokens=config.get("max_api_total_tokens", 20000),
        top_p=config.get("top_p", 1.0),
        stop_tokens=config.get("stop_tokens", ""),
        mmtabqa_sub_dataset=config.get("mmtabqa_sub_dataset"),
        mmtabqa_dataset_split=config.get("mmtabqa_dataset_split"),
        filtering_as_remover=config.get("filtering_as_remover", False),
        reasoner_answer_only=config.get("reasoner_answer_only", False),
        n_examples=config.get("n_examples"),  # Number of examples to sample (None = all)
    )

    for key, value in kwargs.items():
        setattr(args, key, value)

    args.prompt_file = os.path.join(ROOT_DIR, args.prompt_file)
    args.save_dir = os.path.join(ROOT_DIR, args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    return args


def write_readme(config):
    # Create README.md in save_dir with experiment details
    steps = [
        (config["run_wikitq_sql_col"], config["skip_step1"], "Step 1: Column Selection (SQL)"),
        (config["run_wikitq_text_col"], config["skip_step2"], "Step 2: Column Selection (Text)"),
        (config["run_wikitq_sql_row"], config["skip_step3"], "Step 3: Row Selection (SQL)"),
        (config["run_wikitq_text_row"], config["skip_step4"], "Step 4: Row Selection (Text)"),
        (config["run_wikitq_sql_reason"], config["skip_step5"], "Step 5: Reason Generation (SQL)"),
        (config["run_wikitq_text_reason"], False, "Step 6: Reason Generation (Text)"),  # Step 6 cannot be skipped
        (config["run_final_evaluation"], False, "Final Evaluation"),
    ]

    readme_steps_executed = ["  - " + step[2] for step in steps if step[0] and not step[1]]
    readme_steps_skipped = ["  - " + step[2] for step in steps if step[1]]

    readme_steps_executed_str = "\n".join(readme_steps_executed) if readme_steps_executed else "  - None"
    readme_steps_skipped_str = "\n".join(readme_steps_skipped) if readme_steps_skipped else "  - None"

    readme_content = README_TEMPLATE.format(
        experiment_name=config["experiment_name"],
        config_path=config["config_path"],
        dataset_name=config["dataset_name"],
        engine=config["engine"],
        steps_executed=readme_steps_executed_str,
        steps_skipped=readme_steps_skipped_str,
    )

    # Create results directory if it doesn't exist
    os.makedirs(f"results/{config['experiment_name']}", exist_ok=True)

    # Write README.md
    with open(f"results/{config['experiment_name']}/README.md", "w") as f:
        f.write(readme_content)


def _normalize_rows_for_metrics(rows):
    normalized_rows = []
    for row in rows:
        if isinstance(row, dict):
            normalized_rows.append(row.get("content", []))
        else:
            normalized_rows.append(row)
    return normalized_rows


def _token_count(values):
    joined = " ".join(str(v) for v in values if v is not None)
    return len(joined.split())


def _parse_row_index(row_name):
    if isinstance(row_name, int):
        return row_name
    if isinstance(row_name, str):
        match = re.search(r"\d+", row_name)
        if match:
            return int(match.group(0))
    return None


def append_intermediate_pruning_metrics(config):
    """
    Append intermediate pruning stats to the split README:
    - % row pruned
    - % col pruned
    - % tokens pruned
    """
    step2_path = f"results/{config['experiment_name']}/{OUTPUT_FILES[2]}"
    step4_path = f"results/{config['experiment_name']}/{OUTPUT_FILES[4]}"
    readme_path = f"results/{config['experiment_name']}/README.md"

    if not os.path.exists(step2_path) or not os.path.exists(step4_path):
        print("Skipping intermediate pruning metrics: step2/step4 outputs not found.")
        return

    with open(step2_path, "r") as f:
        step2_data = json.load(f)
    with open(step4_path, "r") as f:
        step4_data = json.load(f)

    total_rows_before = 0
    total_rows_after = 0
    total_cols_before = 0
    total_cols_after = 0
    total_tokens_before = 0
    total_tokens_after = 0

    common_ids = sorted(set(step2_data.keys()).intersection(step4_data.keys()), key=lambda x: int(x))
    if not common_ids:
        print("Skipping intermediate pruning metrics: no common example ids between step2 and step4.")
        return

    for eid in common_ids:
        ori_data_item = step2_data[eid].get("ori_data_item", {})
        table_with_metadata = ori_data_item.get("table_with_metadata", {})
        headers = table_with_metadata.get("header", [])
        rows = _normalize_rows_for_metrics(table_with_metadata.get("rows", []))

        if not headers or not rows:
            continue

        header_to_idx = {str(h).lower(): i for i, h in enumerate(headers)}

        selected_cols = step2_data[eid].get("cols", [])
        selected_col_indices = []
        for col in selected_cols:
            if str(col).lower() == "row_id":
                continue
            idx = header_to_idx.get(str(col).lower())
            if idx is not None and idx not in selected_col_indices:
                selected_col_indices.append(idx)
        if not selected_col_indices:
            selected_col_indices = list(range(len(headers)))

        selected_rows_raw = step4_data[eid].get("rows", [])
        selected_row_indices = []
        for row_name in selected_rows_raw:
            row_idx = _parse_row_index(row_name)
            if row_idx is not None and 0 <= row_idx < len(rows):
                selected_row_indices.append(row_idx)
        selected_row_indices = sorted(set(selected_row_indices))
        if not selected_row_indices:
            selected_row_indices = list(range(len(rows)))

        total_rows_before += len(rows)
        total_rows_after += len(selected_row_indices)
        total_cols_before += len(headers)
        total_cols_after += len(selected_col_indices)

        tokens_before = _token_count(headers)
        for row in rows:
            tokens_before += _token_count(row)
        total_tokens_before += tokens_before

        pruned_headers = [headers[i] for i in selected_col_indices]
        tokens_after = _token_count(pruned_headers)
        for row_idx in selected_row_indices:
            row = rows[row_idx]
            filtered_row = [row[i] if i < len(row) else "" for i in selected_col_indices]
            tokens_after += _token_count(filtered_row)
        total_tokens_after += tokens_after

    if total_rows_before == 0 or total_cols_before == 0 or total_tokens_before == 0:
        print("Skipping intermediate pruning metrics: insufficient table content.")
        return

    row_pruned_pct = 100 * (1 - (total_rows_after / total_rows_before))
    col_pruned_pct = 100 * (1 - (total_cols_after / total_cols_before))
    token_pruned_pct = 100 * (1 - (total_tokens_after / total_tokens_before))

    print("\nIntermediate Pruning Metrics")
    print(f"rows before: {total_rows_before}; rows after: {total_rows_after}; % row pruned: {row_pruned_pct:.2f}")
    print(f"cols before: {total_cols_before}; cols after: {total_cols_after}; % col pruned: {col_pruned_pct:.2f}")
    print(f"tokens before: {total_tokens_before}; tokens after: {total_tokens_after}; % tokens pruned: {token_pruned_pct:.2f}")

    with open(readme_path, "a") as f:
        f.write("\n\nIntermediate Pruning Metrics\n")
        f.write(f"rows before: {total_rows_before}; rows after: {total_rows_after}\n")
        f.write(f"cols before: {total_cols_before}; cols after: {total_cols_after}\n")
        f.write(f"tokens before: {total_tokens_before}; tokens after: {total_tokens_after}\n")
        f.write(f"% row pruned: {row_pruned_pct:.2f}\n")
        f.write(f"% col pruned: {col_pruned_pct:.2f}\n")
        f.write(f"% tokens pruned: {token_pruned_pct:.2f}\n")

    print("Appended intermediate pruning metrics to README.")


def append_sql_execution_metrics(config):
    """
    Append SQL execution metrics (step 3) to the split README:
    - % sql execution accuracy by example (at least one valid SQL generation)
    - % sql execution accuracy by generation
    """
    step3_path = f"results/{config['experiment_name']}/{OUTPUT_FILES[3]}"
    readme_path = f"results/{config['experiment_name']}/README.md"

    if not os.path.exists(step3_path):
        print("Skipping SQL execution metrics: step3 output not found.")
        return

    with open(step3_path, "r") as f:
        step3_data = json.load(f)

    attempted_generations = 0
    successful_generations = 0
    successful_examples = 0
    total_examples = len(step3_data)

    # Fast path: use execution status emitted by step3 directly.
    has_inline_status = any("sql_exec_success" in item for item in step3_data.values())
    if has_inline_status:
        for eid in sorted(step3_data.keys(), key=lambda x: int(x)):
            item = step3_data[eid]
            success_flags = item.get("sql_exec_success", [])
            if not isinstance(success_flags, list):
                success_flags = []

            attempted_generations += len(success_flags)
            successful_generations += sum(1 for flag in success_flags if flag)
            if any(success_flags):
                successful_examples += 1
    else:
        # Backward-compatible fallback for older results that don't store status.
        from nsql.database import NeuralDB
        from nsql.sql_exec import Executor
        from utils.normalizer import extract_first_sql_statement, post_process_sql
        from utils.utils import metadata_to_table

        for eid in sorted(step3_data.keys(), key=lambda x: int(x)):
            item = step3_data[eid]
            g_data_item = copy.deepcopy(item.get("ori_data_item", {}))

            try:
                metadata_to_table(g_data_item)
                db = NeuralDB(
                    tables=[
                        {
                            "title": g_data_item["table"]["page_title"],
                            "table": g_data_item["table"],
                        }
                    ]
                )
            except Exception:
                continue

            generations = item.get("generations_pure", [])
            if not isinstance(generations, list):
                generations = []

            example_success = False
            for generation in generations:
                attempted_generations += 1
                try:
                    sql = extract_first_sql_statement(generation)
                    if not sql:
                        continue
                    if re.search(r"^\s*select\s+from\s+w\s*;?\s*$", sql, flags=re.IGNORECASE):
                        sql = "SELECT row_id FROM w;"

                    norm_sql = post_process_sql(
                        sql_str=sql,
                        df=db.get_table_df(),
                        process_program_with_fuzzy_match_on_db=True,
                        table_title=db.get_table_title(),
                    )

                    executor = Executor(None, "")
                    executor.sql_exec(norm_sql, db, verbose=False)
                    successful_generations += 1
                    example_success = True
                except Exception:
                    continue

            if example_success:
                successful_examples += 1

    if total_examples == 0:
        print("Skipping SQL execution metrics: no step3 examples found.")
        return

    example_accuracy = 100 * (successful_examples / total_examples)
    generation_accuracy = 0.0
    if attempted_generations > 0:
        generation_accuracy = 100 * (successful_generations / attempted_generations)
    failed_generations = max(0, attempted_generations - successful_generations)

    with open(readme_path, "a") as f:
        f.write("\n\nSQL execution Accuracy\n")
        f.write(f"sql_exec_success: {successful_generations}\n")
        f.write(f"sql_exec_failures: {failed_generations}\n")
        f.write(f"sql_exec_total: {attempted_generations}\n")
        f.write(f"sql_exec_accuracy: {generation_accuracy:.2f}%\n")
        f.write(f"sql_exec_example_accuracy: {example_accuracy:.2f}%\n")

    print("\nSQL Execution Metrics")
    print(f"sql_exec_success: {successful_generations}")
    print(f"sql_exec_failures: {failed_generations}")
    print(f"sql_exec_total: {attempted_generations}")
    print(f"sql_exec_accuracy: {generation_accuracy:.2f}%")
    print(f"sql_exec_example_accuracy: {example_accuracy:.2f}%")
    print("Appended SQL execution metrics to README.")


def append_image_not_reintroduced_ablation(config, generator, final_evaluate_optional_args, output_file):
    """
    Evaluate the no-image-reintroduced ablation and append metrics to split README.
    This reuses the interleaved parsing format but keeps step-6 text-only (no image reintroduction).
    """
    readme_path = f"results/{config['experiment_name']}/README.md"
    ablation_readme_path = f"results/{config['experiment_name']}/_ablation_no_image_eval.tmp.md"
    if os.path.exists(ablation_readme_path):
        os.remove(ablation_readme_path)

    results_file = f"results/{config['experiment_name']}/{output_file}"
    final_evaluate_main(
        results_file=results_file,
        mode=final_evaluate_optional_args.get("mode", "hstar"),
        dataset=config["dataset_name"],
        readme_file=ablation_readme_path,
        mmtabqa_sub_dataset=final_evaluate_optional_args.get("mmtabqa_sub_dataset", None),
        mmtabqa_dataset_split=final_evaluate_optional_args.get("mmtabqa_dataset_split", None),
        generator=generator,
        use_llm_as_judge=final_evaluate_optional_args.get("use_llm_as_judge", False),
    )

    em_block = None
    judge_block = None
    if os.path.exists(ablation_readme_path):
        with open(ablation_readme_path, "r") as f:
            tmp_text = f.read()

        block_pattern = re.compile(
            r"Correct Samples:\s*(\d+)\s*;\s*Total Samples:\s*(\d+)\s*\nAccuracy:\s*([0-9.]+%?)",
            re.MULTILINE,
        )
        blocks = block_pattern.findall(tmp_text)
        if len(blocks) >= 1:
            em_block = blocks[0]
        if len(blocks) >= 2:
            judge_block = blocks[1]

        os.remove(ablation_readme_path)

    with open(readme_path, "a") as f:
        f.write("\n\nImage not reintroduced\n")
        f.write(f"File: {results_file}\n")
        if em_block is not None:
            f.write("EM numbers\n")
            f.write(f"Correct Samples: {em_block[0]}; Total Samples: {em_block[1]}\n")
            f.write(f"Accuracy: {em_block[2]}\n")
        else:
            f.write("EM numbers\n")
            f.write("Correct Samples: N/A; Total Samples: N/A\n")
            f.write("Accuracy: N/A\n")

        if judge_block is not None:
            f.write("LLM as a judge numbers\n")
            f.write(f"Correct Samples: {judge_block[0]}; Total Samples: {judge_block[1]}\n")
            f.write(f"Accuracy: {judge_block[2]}\n")
        else:
            f.write("LLM as a judge numbers\n")
            f.write("Correct Samples: N/A; Total Samples: N/A\n")
            f.write("Accuracy: N/A\n")

    print("Appended no-image-reintroduced ablation metrics to README.")


def print_config_summary(config):
    print("---------------------------------------")
    print(f"Running experiment: {config['experiment_name']} with {config['engine']} on {config['dataset_name']}")
    print(f"Run Step 1 (col_sql): {config['run_wikitq_sql_col']}    Skip Step 1: {config['skip_step1']}")
    print(f"Run Step 2 (col_text): {config['run_wikitq_text_col']}    Skip Step 2: {config['skip_step2']}")
    print(f"Run Step 3 (row_sql): {config['run_wikitq_sql_row']}   Skip Step 3: {config['skip_step3']}")
    print(f"Run Step 4 (row_text): {config['run_wikitq_text_row']}    Skip Step 4: {config['skip_step4']}")
    print(f"Run Step 5 (sql_reason): {config['run_wikitq_sql_reason']}    Skip Step 5: {config['skip_step5']}")
    print(f"Run Step 6 (text_reason): {config['run_wikitq_text_reason']}    Step 6 cannot be skipped")
    print(f"Run Final Evaluation: {config['run_final_evaluation']}")
    print("---------------------------------------")


def run_step_1(config, generator, tokenizer, **config_kwargs):
    if os.path.exists(f"results/{config['experiment_name']}/{OUTPUT_FILES[1]}"):
        print("\n\n⚠️⚠️⚠️ Step 1 already executed. Skipping... ⚠️⚠️⚠️\n\n")
        return

    if config["run_wikitq_sql_col"]:
        print("\nrunning step 1: col_sql\n")

        if config["skip_step1"]:
            raise ValueError("Step 1 set to skip but RUN_WIKITQ_SQL_COL is True")

        is_qwen3_vl = "qwen3-vl" in str(config.get("vllm_model_name", "")).lower()
        if config["filtering_as_remover"]:
            prompt_file = "prompts/filtering_as_remover/col_select_sql.txt"
        else:
            prompt_file = "prompts/col_select_sql.txt"

        default_sql_sampling_n = 1 if is_qwen3_vl else 2
        sql_sampling_n = int(config.get("sql_sampling_n", default_sql_sampling_n))
        default_sql_temperature = 0.0 if is_qwen3_vl else config.get("temperature", 0.3)
        sql_temperature = float(config.get("sql_temperature", default_sql_temperature))

        step_args = create_step_args(
            config,
            prompt_file=prompt_file,
            sampling_n=sql_sampling_n,
            temperature=sql_temperature,
            skip_step_2=config["skip_step2"],
            save_file_name=OUTPUT_FILES[1],
            prompt_style="create_table_select_full_table",
            generate_type="col",
            n_shots=8,
            seed=42,
            **config_kwargs,
        )

        col_sql_main(step_args, generator, tokenizer)

        print("\n\n\nfinished col_sql")
    else:
        print("\n\n\nskipping col_sql\n\n\n")


def run_step_2(config, generator, tokenizer, **config_kwargs):
    if os.path.exists(f"results/{config['experiment_name']}/{OUTPUT_FILES[2]}"):
        print("\n\n⚠️⚠️⚠️ Step 2 already executed. Skipping... ⚠️⚠️⚠️\n\n")
        return

    if config["run_wikitq_text_col"]:
        if config["skip_step2"]:
            raise ValueError("Step 2 set to skip but RUN_WIKITQ_TEXT_COL is True")
        input_program_file = OUTPUT_FILES[1]
        print("\nrunning step 2: col_text\n")

        if config["filtering_as_remover"]:
            prompt_file = "prompts/filtering_as_remover/col_select_text.txt"
        else:
            prompt_file = "prompts/col_select_text.txt"

        step_args = create_step_args(
            config,
            prompt_file=prompt_file,
            input_program_file=input_program_file,
            sampling_n=2,
            skip_step_1=config["skip_step1"],
            save_file_name=OUTPUT_FILES[2],
            prompt_style="transpose",
            generate_type="col",
            n_shots=8,
            seed=42,
            **config_kwargs,
        )
        col_text_main(step_args, generator, tokenizer)

        print("\n\n\nfinished col_text\n\n\n")
    else:
        print("\n\n\nskipping col_text\n\n\n")


def run_step_3(config, generator, tokenizer, **config_kwargs):
    if os.path.exists(f"results/{config['experiment_name']}/{OUTPUT_FILES[3]}"):
        print("\n\n⚠️⚠️⚠️ Step 3 already executed. Skipping... ⚠️⚠️⚠️\n\n")
        return

    if config["run_wikitq_sql_row"]:
        if config["skip_step3"]:
            raise ValueError("Step 3 set to skip but RUN_WIKITQ_SQL_ROW is True")

        input_program_file = OUTPUT_FILES[2] if not config["skip_step2"] else OUTPUT_FILES[1]

        is_qwen3_vl = "qwen3-vl" in str(config.get("vllm_model_name", "")).lower()
        if config["filtering_as_remover"]:
            prompt_file = "prompts/filtering_as_remover/row_select_sql.txt"
        else:
            prompt_file = "prompts/row_select_sql.txt"

        print("\nrunning step 3: row_sql\n")
        default_sql_sampling_n = 1 if is_qwen3_vl else 2
        sql_sampling_n = int(config.get("sql_sampling_n", default_sql_sampling_n))
        default_sql_temperature = 0.0 if is_qwen3_vl else 0.3
        sql_temperature = float(config.get("sql_temperature", default_sql_temperature))

        step_args = create_step_args(
            config,
            prompt_file=prompt_file,
            input_program_file=input_program_file,
            max_generation_tokens=512,
            temperature=sql_temperature,
            sampling_n=sql_sampling_n,
            skip_step_4=config["skip_step4"],
            save_file_name=OUTPUT_FILES[3],
            prompt_style="create_table_select_full_table",
            generate_type="row",
            n_shots=8,
            seed=42,
            **config_kwargs,
        )
        row_sql_main(step_args, generator, tokenizer)

        print("\n\n\nfinished row_sql\n\n\n")
    else:
        print("\n\n\nskipping row_sql\n\n\n")


def run_step_4(config, generator, tokenizer, **config_kwargs):
    if os.path.exists(f"results/{config['experiment_name']}/{OUTPUT_FILES[4]}"):
        print("\n\n⚠️⚠️⚠️ Step 4 already executed. Skipping... ⚠️⚠️⚠️\n\n")
        return

    if config["run_wikitq_text_row"]:
        if config["skip_step4"]:
            raise ValueError("Step 4 set to skip but RUN_WIKITQ_TEXT_ROW is True")
        input_program_file = (
            OUTPUT_FILES[3]
            if not config["skip_step3"]
            else OUTPUT_FILES[2]
            if not config["skip_step2"]
            else OUTPUT_FILES[1]
        )

        if config["filtering_as_remover"]:
            prompt_file = "prompts/filtering_as_remover/row_select_text.txt"
        else:
            prompt_file = "prompts/row_select_text.txt"

        print("\nrunning step 4: row_text\n")
        step_args = create_step_args(
            config,
            prompt_file=prompt_file,
            input_program_file=input_program_file,
            max_generation_tokens=512,
            temperature=0.4,
            sampling_n=2,
            skip_step_3=config["skip_step3"],
            save_file_name=OUTPUT_FILES[4],
            prompt_style="text_full_table",
            generate_type="verification",
            n_shots=8,
            seed=42,
            **config_kwargs,
        )
        row_text_main(step_args, generator, tokenizer)

        print("\n\n\nfinished row_text\n\n\n")
    else:
        print("\n\n\nskipping row_text\n\n\n")


def run_step_5(config, generator, tokenizer, **config_kwargs):
    if os.path.exists(f"results/{config['experiment_name']}/{OUTPUT_FILES[5]}"):
        print("\n\n⚠️⚠️⚠️ Step 5 already executed. Skipping... ⚠️⚠️⚠️\n\n")
        return

    if config["run_wikitq_sql_reason"]:
        if config["skip_step5"]:
            raise ValueError("Step 5 set to skip but RUN_WIKITQ_SQL_REASON is True")

        input_program_file = (
            OUTPUT_FILES[4]
            if not config["skip_step4"]
            else OUTPUT_FILES[3]
            if not config["skip_step3"]
            else OUTPUT_FILES[2]
            if not config["skip_step2"]
            else OUTPUT_FILES[1]
        )

        prompt_file = "prompts/sql_reason_wtq.txt"

        print("\nrunning step 5: reason_sql\n")
        step_args = create_step_args(
            config,
            prompt_file=prompt_file,
            input_program_file=input_program_file,
            max_generation_tokens=512,
            temperature=0.1,
            sampling_n=1,
            save_file_name=OUTPUT_FILES[5],
            prompt_style="create_table_select_full_table",
            generate_type="col",
            n_shots=8,
            seed=42,
            **config_kwargs,
        )
        reason_sql_main(step_args, generator, tokenizer)

        print("\n\n\nfinished reason_sql\n\n\n")
    else:
        print("\n\n\nskipping reason_sql\n\n\n")


def run_step_6(config, generator, tokenizer, output_file, mmtabqa_sub_dataset=None, **config_kwargs) -> bool:
    if os.path.exists(f"results/{config['experiment_name']}/{output_file}"):
        print("\n\n⚠️⚠️⚠️ Step 6 already executed. Skipping... ⚠️⚠️⚠️\n\n")
        return True

    if config["run_wikitq_text_reason"]:
        input_program_file = (
            OUTPUT_FILES[5]
            if not config["skip_step5"]
            else OUTPUT_FILES[4]
            if not config["skip_step4"]
            else OUTPUT_FILES[3]
            if not config["skip_step3"]
            else OUTPUT_FILES[2]
            if not config["skip_step2"]
            else OUTPUT_FILES[1]
        )

        prompt_file = "prompts/text_reason_wtq.txt"

        print("\nrunning step 6: reason_text\n")
        step_args = create_step_args(
            config,
            prompt_file=prompt_file,
            input_program_file=input_program_file,
            max_generation_tokens=int(config.get("reason_max_generation_tokens", 512)),
            temperature=float(config.get("reason_temperature", 0.0)),
            sampling_n=int(config.get("reason_sampling_n", 1)),
            save_file_name=output_file,
            skip_step_5=config["skip_step5"],
            mmtabqa_sub_dataset=mmtabqa_sub_dataset,
            use_mmtabqa_interleaved_prompt=config["use_mmtabqa_interleaved_prompt"],
            prompt_style="text_full_table",
            generate_type="col",
            n_shots=8,
            seed=42,
            **config_kwargs,
        )
        reason_text_main(step_args, generator, tokenizer)

        print("\n\n\nfinished reason_text\n\n\n")
    else:
        print("\n\n\nskipping reason_text\n\n\n")
    return False


def run_final_evaluation(config, generator, output_file, **final_evaluate_optional_args):
    if config["run_final_evaluation"]:
        final_evaluate_main(
            results_file=f"results/{config['experiment_name']}/{output_file}",
            mode=final_evaluate_optional_args.get("mode", "hstar"),
            dataset=config["dataset_name"],
            readme_file=f"results/{config['experiment_name']}/README.md",
            mmtabqa_sub_dataset=final_evaluate_optional_args.get("mmtabqa_sub_dataset", None),
            mmtabqa_dataset_split=final_evaluate_optional_args.get("mmtabqa_dataset_split", None),
            generator=generator,
            use_llm_as_judge=final_evaluate_optional_args.get("use_llm_as_judge", False),
        )

        print("\n\n\nfinished final evaluation\n\n\n")
    else:
        print("\n\n\nskipping final evaluation\n\n\n")


def run_config(
    config, generator, tokenizer, final_evaluate_optional_args={}, mmtabqa_sub_dataset=None, **config_kwargs
):
    # The first 4 steps are our filtering steps
    run_step_1(config, generator, tokenizer, **config_kwargs)
    run_step_2(config, generator, tokenizer, **config_kwargs)
    run_step_3(config, generator, tokenizer, **config_kwargs)
    run_step_4(config, generator, tokenizer, **config_kwargs)

    # Now we have our filtered tables and we can apply the reasoner (interleaved from MMTabQA, H-STAR step 5 / 6 for text only)
    if config["use_mmtabqa_interleaved_prompt"]:
        output_file = "interleaved_reasoning_output.json"

        add_col_values_to_prompt = "add_col_values_to_prompt" in config_kwargs and config_kwargs["add_col_values_to_prompt"]  # fmt: skip
        add_caption_to_reasoner_v1 = "add_caption_to_reasoner_v1" in config_kwargs and config_kwargs["add_caption_to_reasoner_v1"]  # fmt: skip
        add_caption_to_reasoner_v2 = "add_caption_to_reasoner_v2" in config_kwargs and config_kwargs["add_caption_to_reasoner_v2"]  # fmt: skip
        assert not (add_col_values_to_prompt and add_caption_to_reasoner_v1 and add_caption_to_reasoner_v2), "Only one of add_col_values_to_prompt, add_caption_to_reasoner_v1, or add_caption_to_reasoner_v2 can be True"  # fmt: skip
        if add_col_values_to_prompt:
            output_file = "interleaved_reasoning_output_with_col_values.json"
        elif add_caption_to_reasoner_v1:
            output_file = "interleaved_reasoning_output_with_captions.json"
        elif add_caption_to_reasoner_v2:
            output_file = "interleaved_reasoning_output_with_captions_v2.json"

        skipped_step_6 = run_step_6(config, generator, tokenizer, output_file, mmtabqa_sub_dataset, **config_kwargs)
    else:
        run_step_5(config, generator, tokenizer, **config_kwargs)
        output_file = OUTPUT_FILES[6]
        skipped_step_6 = run_step_6(config, generator, tokenizer, output_file, mmtabqa_sub_dataset, **config_kwargs)

    # Now we have our prediction -> run evaluation
    # if not skipped_step_6:
    #     run_final_evaluation(config, generator, output_file, **final_evaluate_optional_args)
    # else:
    #     print("skipping evaluation since step 6 was skipped, thus we evaluated the results already.")
    run_final_evaluation(config, generator, output_file, **final_evaluate_optional_args)

    # Optional ablation: keep interleaved parsing format but do NOT reintroduce images in step 6.
    if config["use_mmtabqa_interleaved_prompt"] and config.get("load_images_for_reasoning", False):
        ablation_output_file = "interleaved_reasoning_output_no_image_reintroduced.json"
        original_load_images = config.get("load_images_for_reasoning", False)
        try:
            config["load_images_for_reasoning"] = False
            run_step_6(
                config,
                generator,
                tokenizer,
                ablation_output_file,
                mmtabqa_sub_dataset,
                **config_kwargs,
            )
        finally:
            config["load_images_for_reasoning"] = original_load_images

        append_image_not_reintroduced_ablation(
            config,
            generator,
            final_evaluate_optional_args,
            ablation_output_file,
        )

    append_intermediate_pruning_metrics(config)
    append_sql_execution_metrics(config)


def config_to_generator_args(config):
    return Namespace(
        engine=config["engine"],
        vllm_model_name=config["vllm_model_name"],
        number_of_gpus=config["number_of_gpus"],
        max_api_total_tokens=config["max_api_total_tokens"],
        prompt_mode=config["prompt_mode"],
        gpu_memory_utilization=config.get("gpu_memory_utilization", None),
    )


def run_all_multimodal_datasets(config):
    DATASETS = ["wikitq", "wikisql", "fetaqa", "hybridqa"]
    SPLITS = ["VQ", "AQ", "EQ", "IQ"]
    original_experiment_name = config["experiment_name"]

    use_system_prompt = False if "gemma" in config["vllm_model_name"] else True

    generator_args = config_to_generator_args(config)
    generator, tokenizer = get_generator_and_tokenizer(
        generator_args, limit_mm_per_prompt={"image": 200}, use_system_prompt=use_system_prompt
    )

    for sub_dataset in DATASETS:
        for split in SPLITS:
            # Only run captioned interleaved modes for the MMTabReal (mmtbench) dataset.
            # For all other sub-datasets we keep only the `no_caption` mode to avoid
            # generating `interleaved_reasoning_output_with_captions*.json` files.
            if config["include_captions_in_reasoner"] and sub_dataset == "mmtbench":
                caption_in_reasoner_modes = ["no_caption", "caption_v1", "caption_v2"]
            else:
                caption_in_reasoner_modes = ["no_caption"]

            for caption_in_reasoner_mode in caption_in_reasoner_modes:
                # Eval showed that col value is only beneficial for WikiSQL
                add_col_values_to_prompt = True if sub_dataset == "wikisql" else False
                config_kwargs = {
                    "mmtabqa_dataset_split": split,
                    "mmtabqa_sub_dataset": sub_dataset,
                    "add_col_values_to_prompt": add_col_values_to_prompt,
                    "add_caption_to_reasoner_v1": caption_in_reasoner_mode == "caption_v1",
                    "add_caption_to_reasoner_v2": caption_in_reasoner_mode == "caption_v2",
                }

                final_evaluate_optional_args = {
                    "mmtabqa_dataset_split": split,
                    "mmtabqa_sub_dataset": sub_dataset,
                }
                if config["use_mmtabqa_interleaved_prompt"]:
                    final_evaluate_optional_args["mode"] = "interleaved_reasoner"

                # use LLM-as-a-judge for evaluation. This is the default. If you don't want to use it, set it to False.
                final_evaluate_optional_args["use_llm_as_judge"] = True

                config["experiment_name"] = f"{original_experiment_name}/{sub_dataset}_{split}"
                config["dataset_split"] = split
                print(f"\n\n============================\nRunning {config['dataset_name']} with {sub_dataset} and {config['dataset_split']}\n============================\n\n")  # fmt: skip
                config["mmtabqa_sub_dataset"] = sub_dataset
                run_config(
                    config=config,
                    generator=generator,
                    tokenizer=tokenizer,
                    final_evaluate_optional_args=final_evaluate_optional_args,
                    **config_kwargs,
                )


def main():
    print("run_model: start")
    parser = argparse.ArgumentParser(description="Run H-STAR pipeline with a configuration file.")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the JSON configuration file (e.g., run_configs/wikitq_full_hstar.json).",
    )
    # optional argument to retry from last step if aborted
    parser.add_argument(
        "--retry_from_last_step",
        type=bool,
        default=True,
        help="Retry from last step if the process was aborted (e.g. because it was killed because it ran on the yolo partition on slurm)",
    )
    args = parser.parse_args()
    config_path = args.config_file
    config = load_config(config_path)
    _set_default_max_api_total_tokens(config)

    # Captioned datasets still retain image_path metadata, so the final reasoning step
    # can re-load the original images if requested.

    print_config_summary(config)
    write_readme(config)

    if config["dataset_name"] == "all_multimodal_datasets":
        run_all_multimodal_datasets(config)
    elif config["dataset_name"] == "mmtbench_only":
        # Run only MMTabReal (mmtbench) for all splits
        original_experiment_name = config["experiment_name"]
        all_splits = ["VQ", "AQ", "EQ", "IQ"]

        # Skip split if its result folder already exists.
        pending_splits = []
        for split in all_splits:
            split_result_dir = f"results/{original_experiment_name}/mmtbench_{split}"
            if os.path.isdir(split_result_dir):
                print(f"Skipping MMTabReal {split}: result folder already exists at {split_result_dir}")
            else:
                pending_splits.append(split)

        if not pending_splits:
            print("All MMTabReal splits already have result folders. Nothing to run.")
            return

        use_system_prompt = False if "gemma" in config["vllm_model_name"] else True
        generator_args = config_to_generator_args(config)
        generator, tokenizer = get_generator_and_tokenizer(
            generator_args, limit_mm_per_prompt={"image": 330}, use_system_prompt=use_system_prompt
        )

        for split in pending_splits:
            config_kwargs = {
                "mmtabqa_dataset_split": split,
                "mmtabqa_sub_dataset": "mmtbench",
                "add_col_values_to_prompt": False,
                "add_caption_to_reasoner_v1": False,
                "add_caption_to_reasoner_v2": False,
            }
            
            final_evaluate_optional_args = {
                "mmtabqa_dataset_split": split,
                "mmtabqa_sub_dataset": "mmtbench",
            }
            if config["use_mmtabqa_interleaved_prompt"]:
                final_evaluate_optional_args["mode"] = "interleaved_reasoner"
            
            final_evaluate_optional_args["use_llm_as_judge"] = True
            
            config["experiment_name"] = f"{original_experiment_name}/mmtbench_{split}"
            config["dataset_split"] = split
            print(f"\n\n============================\nRunning MMTabReal with {split} split\n============================\n\n")
            config["mmtabqa_sub_dataset"] = "mmtbench"
            run_config(
                config=config,
                generator=generator,
                tokenizer=tokenizer,
                final_evaluate_optional_args=final_evaluate_optional_args,
                **config_kwargs,
            )
    else:
        generator_args = config_to_generator_args(config)
        generator, tokenizer = get_generator_and_tokenizer(generator_args)
        run_config(config, generator, tokenizer)


if __name__ == "__main__":
    main()
