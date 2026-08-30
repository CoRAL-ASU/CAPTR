"""
General utilities.
"""

import argparse
import copy
import json
import os
import re
from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Iterable

from datasets import load_dataset
from generation.generator import Generator
from generation.generator_gpt import GPTGenerator
from generation.generator_vllm_gemma3 import VLLMGeneratorGemma3
from generation.generator_vllm_qwen import VLLMGeneratorQwen
from generation.generator_vllm_qwen3vl import VLLMGeneratorQwen3VL
from nsql.database import NeuralDB
from transformers import AutoTokenizer

from utils.normalizer import normalize_column_name, normalize_headers


ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")


def build_passage_context(
    g_data_item, selected_rows=None, selected_cols=None, max_chars_in_total_passage_context=25000
):
    """
    This function builds the passage context for HybridQA.
    Input:
    - g_data_item (dict): the g_data_item
    - selected_rows (list): the selected rows; start counting at 0; must look like this: ["row 0", "row 4", ...]
    - selected_cols (list): the selected cols; might look like this: ["order year", "powertrain (engine/transmission)"]

    Output:
    - passage_context (str): the passage context
    """
    passages = g_data_item.get("passages", None)

    if passages is None:
        return None

    # 0. Convert passages to a list of dicts that has 2 entries: linked_cell & text
    # passages_list might look like this: [{"linked_cell": [3, 1], "text": "The order year is 2017."}, ...]
    passages_list = [
        {"linked_cell": passages["linked_cell"][idx], "text": passages["text"][idx]}
        for idx in range(len(passages["linked_cell"]))
    ]

    # 1. filter the passages based on the selected rows and cols
    old_row_id_to_new_row_id_map = {}
    if selected_rows is not None:
        selected_rows_nums = [int(row.split(" ")[1]) for row in selected_rows]
        selected_rows_nums.sort()
        old_row_id_to_new_row_id_map = dict(zip(selected_rows_nums, range(len(selected_rows_nums))))
        passages_list = [passage for passage in passages_list if passage["linked_cell"][0]-1 in selected_rows_nums]  # -1 because linked_cells rows begin to count at 1, not 0  # fmt: skip

    headers_lowercase = [h.lower() for h in g_data_item["table_with_metadata"]["header"]]
    original_col_num_to_name_map = dict(
        zip(range(len(headers_lowercase)), headers_lowercase)
    )  # columns begin to count at 0
    if selected_cols is not None:
        selected_cols_lowercase = [col.lower() for col in selected_cols if col != "row_id"]
        selected_cols_nums = [headers_lowercase.index(col) for col in selected_cols_lowercase if col in headers_lowercase]  # fmt: skip
        passages_list = [passage for passage in passages_list if passage["linked_cell"][1] in selected_cols_nums]  # fmt: skip

    if passages_list is None or len(passages_list) == 0:
        return None

    # 2. deduplicate passages with the same text to only link it once
    text_to_locations = defaultdict(list)
    for passage in passages_list:
        linked_cell_row_id = passage["linked_cell"][0]
        if old_row_id_to_new_row_id_map != {}:
            row_id = old_row_id_to_new_row_id_map[linked_cell_row_id - 1]
        else:
            row_id = linked_cell_row_id - 1
        col_id = passage["linked_cell"][1]
        col_name = original_col_num_to_name_map[col_id]

        text_to_locations[passage["text"]].append((row_id, col_name))

    # 3. create the passage context text
    passage_context = ""
    for text, locations in text_to_locations.items():
        if len(locations) == 1:
            row_id, col_name = locations[0]
            passage_context += f"Passage related to entity in row {row_id} and column {col_name}: {text}\n"
        else:
            location_str = " and ".join([f"row {row_id}, col {col_name}" for row_id, col_name in locations])
            passage_context += f"Passage related to entity in {location_str}: {text}\n"

    if len(passage_context) > max_chars_in_total_passage_context:
        print(
            f"Truncated passage context to {max_chars_in_total_passage_context} characters. Passages removed: {len(passage_context) - max_chars_in_total_passage_context}"
        )
        passage_context = f"{passage_context[:max_chars_in_total_passage_context]} ... [TRUNCATED]"

    return passage_context


def final_col_extraction(g_dict):
    """
    Final operation to get the columns: If the list of predicted columns is empty, use all columns.
    In the original H-STAR pipeline, this is done after step 2, i.e. after the text-based column extraction generation.
    Since we allow to skip step 2, we need to be able to call this after step 1, hence, I have this in a seperate function.

    Modifies g_dict inplace.
    """
    for g_eid in range(len(g_dict)):
        # Filter Columns
        g_data_item = copy.deepcopy(g_dict[str(g_eid)]["ori_data_item"])
        if g_dict[str(g_eid)]["cols"] == []:
            metadata_to_table(g_data_item)
            db = NeuralDB(
                tables=[
                    {
                        "title": g_data_item["table"]["page_title"],
                        "table": g_data_item["table"],
                    }
                ]
            )
            table = db.get_table_df()
            g_dict[str(g_eid)]["cols"] = list(table.columns)
            print(
                f"No columns found for eid#{g_eid}, wtqid#{g_data_item['id']}. Using all columns. Columns are: {g_dict[str(g_eid)]['cols']}"
            )


def final_row_extraction(g_dict):
    """
    Final operation to get the rows.

    Modifies g_dict inplace.
    """
    pattern_row_num = r"\d+"
    pattern_row_num = re.compile(pattern_row_num, re.S)

    for g_eid in range(len(g_dict)):
        pred = g_dict[str(g_eid)]["rows"]
        g_data_item = copy.deepcopy(g_dict[str(g_eid)]["ori_data_item"])
        metadata_to_table(g_data_item)
        row_list = []
        if pred != [""]:
            row_list = [str(pattern_row_num.search(x).group()) for x in pred if pattern_row_num.search(x)]
            if row_list == []:
                row_list = [str(j) for j in range(len(g_data_item["table"]["rows"]))]
                row_unique = set(row_list)
                row_list = list(row_unique)
                print(
                    f"No rows found for eid#{g_eid}, wtqid#{g_data_item['id']}. Using all rows. Row list is: {row_list}"
                )
            else:
                row_unique = set(row_list)
                row_list = list(row_unique)

        # If "rows" is empty, use all rows
        if pred == [] or pred == [""]:
            db = NeuralDB(
                tables=[
                    {
                        "title": g_data_item["table"]["page_title"],
                        "table": g_data_item["table"],
                    }
                ]
            )
            table = db.get_table_df()
            row_list = [str(j) for j in range(len(table.row_id))]
            row_unique = set(row_list)
            row_list = list(row_unique)

        g_dict[str(g_eid)]["rows"] = row_list


def metadata_to_table_with_image_placeholders(data_item):
    """
    This function converts the table_with_metadata into a table that H-STAR pipeline can use. It works like `metadata_to_table` BUT: it replaces the image cells with a unique placeholder.
    It also returns a map from the placeholder to the image path.
    """

    table_with_metadata = data_item["table_with_metadata"]
    image_map = {}
    caption_map = {}
    image_counter = 0
    
    # Handle images in column headers
    new_header = []
    for header_val in table_with_metadata["header"]:
        header_str = str(header_val)
        # Check if header is an image path (starts with "images/" or contains image file extensions)
        if header_str.startswith("images/") or any(header_str.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif"]):
            placeholder = f"image_header_{image_counter}"
            image_map[placeholder] = header_str
            caption_map[placeholder] = header_str  # Use the path as caption if no caption available
            image_counter += 1
            new_header.append(placeholder)
        else:
            new_header.append(header_val)

    normalized_new_header = []
    existing_names = set()
    for header_idx, header_val in enumerate(new_header):
        if isinstance(header_val, str) and header_val.startswith("image_header_"):
            normalized_new_header.append(header_val)
            existing_names.add(header_val)
        else:
            normalized_new_header.append(normalize_column_name(header_val, header_idx, existing_names))
    new_header = normalized_new_header
    
    new_rows = []
    for row_data in table_with_metadata["rows"]:
        new_row = []
        
        # Handle both dict format (with image_path) and list format (captioned)
        if isinstance(row_data, dict):
            # Dict format: has "content", "type", "image_path"
            image_paths = row_data.get("image_path", [])
            content_list = row_data["content"]
        else:
            # List format: captioned dataset, no image paths
            content_list = row_data
            image_paths = []
        
        for i, content in enumerate(content_list):
            # Check if this cell has image path(s).
            raw_image_path = image_paths[i] if i < len(image_paths) else ""
            if raw_image_path:
                cell_image_paths = []
                for p in str(raw_image_path).split("|"):
                    token = str(p).strip()
                    if not token:
                        continue
                    if token.startswith("TEXT:"):
                        continue
                    if token.startswith("IMAGES:"):
                        token = token[7:]
                    token = token.strip().replace("\\", "/")
                    if token:
                        cell_image_paths.append(token)

                # Keep textual context for mixed cells and append placeholders for all images.
                cell_text = ""
                if isinstance(content, list):
                    text_parts = [str(x).strip() for x in content if isinstance(x, str) and str(x).strip()]
                    cell_text = " ".join(text_parts)
                elif content is not None:
                    cell_text = str(content).strip()

                cell_placeholders = []
                for cell_image_path in cell_image_paths:
                    placeholder = f"<image_{image_counter}>"
                    image_map[placeholder] = cell_image_path
                    caption_map[placeholder] = cell_text if cell_text else cell_image_path
                    image_counter += 1
                    cell_placeholders.append(placeholder)

                placeholder_text = " ".join(cell_placeholders)
                if cell_text:
                    new_row.append(f"{cell_text} {placeholder_text}".strip())
                else:
                    new_row.append(placeholder_text)
            else:
                new_row.append(content)
        new_rows.append(new_row)

    data_item["table"] = {
        "page_title": table_with_metadata["page_title"],
        "header": new_header,
        "rows": new_rows,
    }

    return image_map, caption_map


def metadata_to_table(g_data_item):
    """
    Create a "table" inplace in the g_data_item (from the "table_with_metadata" key).
    """
    table_with_metadata = g_data_item["table_with_metadata"]
    rows = table_with_metadata["rows"]
    normalized_rows = []
    for row in rows:
        if isinstance(row, dict):
            normalized_rows.append(row.get("content", []))
        else:
            normalized_rows.append(row)

    g_data_item["table"] = {
        "page_title": table_with_metadata["page_title"],
        "header": normalize_headers(table_with_metadata["header"]),
        "rows": normalized_rows,
    }

    g_data_item["table_with_metadata"]["header"] = normalize_headers(table_with_metadata["header"])


# Define a custom type for boolean arguments
def str_to_bool(val: str) -> bool:
    """Converts a string representation of truth to true (True) or false (False).
    True values are 'y', 'yes', 't', 'true', 'on', '1', 'True'.
    False values are 'n', 'no', 'f', 'false', 'off', '0', 'False'.
    Raises argparse.ArgumentTypeError if 'val' is anything else.
    """
    val_lower = val.lower()
    if val_lower in ("y", "yes", "t", "true", "on", "1", "True"):
        return True
    elif val_lower in ("n", "no", "f", "false", "off", "0", "False"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected (e.g., 'true', 'false'), got '{val}'")


def load_dataset_from_args(args: ArgumentParser):
    # Load dataset
    dataset = load_data_split(args.dataset, args.dataset_split)
    if args.dataset == "fetaqa" and args.dataset_split == "test":
        dataset = []
        with open(os.path.join(ROOT_DIR, "utils", "fetaqa", "fetaQA-v1_test.jsonl"), "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                dic = json.loads(line)
                feta_id = dic["feta_id"]
                caption = dic["table_page_title"]
                question = dic["question"]
                answer = dic["answer"]
                sub_title = dic["table_section_title"]  # noqa: F841
                header = dic["table_array"][0]
                rows = dic["table_array"][1:]
                data = {
                    "id": feta_id,
                    "table": {
                        "id": feta_id,
                        "header": header,
                        "rows": rows,
                        "page_title": caption,
                    },
                    "question": question,
                    "answer": answer,
                }
                dataset.append(data)

    # For TabFact test split, we load the small test set (about 2k examples) to test,
    # since it is expensive to test on full set
    if args.dataset == "tab_fact" and args.dataset_split == "test":
        dataset = []
        with open(os.path.join(ROOT_DIR, "utils", "tab_fact", "small_test.jsonl"), "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                dic = json.loads(line)
                id = dic["table_id"]
                caption = dic["table_caption"]
                question = dic["statement"]
                answer_text = dic["label"]
                header = dic["table_text"][0]
                rows = dic["table_text"][1:]

                data = {
                    "id": i,
                    "table": {
                        "id": id,
                        "header": header,
                        "rows": rows,
                        "page_title": caption,
                    },
                    "question": question,
                    "answer_text": answer_text,
                }
                dataset.append(data)

    return dataset


def _join_prompt_parts(generate_prompt, few_shot_prompt, initial_response, is_row_text_step):
    if is_row_text_step:
        # Step 4 is different as it contains the answer from the previous step
        prompt1, prompt2 = generate_prompt.split("<initial response>")
        prompt = few_shot_prompt + "\n\n" + prompt1 + f"Initial Response:{initial_response}" + "\n" + prompt2
    else:
        prompt = few_shot_prompt + "\n\n" + generate_prompt

    return prompt


def create_prompt(
    tokenizer,
    generator: Generator,
    g_data_item,
    few_shot_prompt,
    args,
    is_row_text_step=False,
    initial_response=None,
    max_tokens_buffer=0,
    passage_context=None,
):
    max_prompt_tokens = args.max_api_total_tokens - args.max_generation_tokens - max_tokens_buffer

    # 1. Create the prompt unshortened & check if it fits the context window
    generate_prompt = generator.build_generate_prompt(data_item=g_data_item, generate_type=(args.generate_type,), args=args, passage_context=passage_context)  # fmt: skip
    prompt = _join_prompt_parts(generate_prompt, few_shot_prompt, initial_response, is_row_text_step)

    # 2. If the prompt is too long, shrink it
    if len(tokenizer.tokenize(prompt)) >= max_prompt_tokens:
        original_num_rows = g_data_item["table"].shape[0]
        num_rows = original_num_rows
        while len(tokenizer.tokenize(prompt)) >= max_prompt_tokens:
            num_rows -= 1
            generate_prompt = generator.build_generate_prompt(
                data_item=g_data_item,
                generate_type=(args.generate_type,),
                num_rows=num_rows,
                args=args,
                passage_context=passage_context,
            )
            prompt = _join_prompt_parts(generate_prompt, few_shot_prompt, initial_response, is_row_text_step)

        print(f"⚠️⚠️⚠️ Attention: Shrinking the prompt to fit the max prompt tokens. We now only use {num_rows} rows of the original {original_num_rows} rows. ⚠️⚠️⚠️")  # fmt: skip

    return prompt


def load_data_split(dataset_to_load, split, data_dir=os.path.join(ROOT_DIR, "datasets/")):
    dataset_split_loaded = load_dataset(
        path=os.path.join(data_dir, "{}.py".format(dataset_to_load)),
        cache_dir=os.path.join(data_dir, "data"),
        trust_remote_code=True,
    )[split]

    # unify names of keys
    if dataset_to_load in [
        "wikitq",
        "has_squall",
        "missing_squall",
        "wikitq",
        "wikitq_sql_solvable",
        "wikitq_sql_unsolvable",
        "wikitq_sql_unsolvable_but_in_squall",
        "wikitq_scalability_ori",
        "wikitq_scalability_100rows",
        "wikitq_scalability_200rows",
        "wikitq_scalability_500rows",
        "wikitq_robustness",
        "fetaqa",
    ]:
        pass
    elif dataset_to_load == "tab_fact":
        new_dataset_split_loaded = []
        for data_item in dataset_split_loaded:
            data_item["question"] = data_item["statement"]
            data_item["answer_text"] = data_item["label"]
            data_item["table"]["page_title"] = data_item["table"]["caption"]
            new_dataset_split_loaded.append(data_item)
        dataset_split_loaded = new_dataset_split_loaded
    elif dataset_to_load == "hybridqa":
        new_dataset_split_loaded = []
        for data_item in dataset_split_loaded:
            data_item["table"]["page_title"] = data_item["context"].split(" | ")[0]
            new_dataset_split_loaded.append(data_item)
        dataset_split_loaded = new_dataset_split_loaded
    elif dataset_to_load == "mmqa":
        new_dataset_split_loaded = []
        for data_item in dataset_split_loaded:
            data_item["table"]["page_title"] = data_item["table"]["title"]
            new_dataset_split_loaded.append(data_item)
        dataset_split_loaded = new_dataset_split_loaded
    else:
        raise ValueError(f"{dataset_to_load} dataset is not supported now.")
    return dataset_split_loaded


def pprint_dict(dic):
    print(json.dumps(dic, indent=2))


def flatten(nested_list):
    for x in nested_list:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def get_generator_and_tokenizer(args: ArgumentParser, limit_mm_per_prompt=None, use_system_prompt=True):
    if args.engine == "vllm":
        assert args.vllm_model_name is not None, "vllm_model_name is required when engine is vllm"
        assert args.number_of_gpus is not None, "number_of_gpus is required when engine is vllm"

        # Select appropriate generator class based on model name
        model_name_lower = args.vllm_model_name.lower()
        if "gemma" in model_name_lower:
            vllm_generator_cls = VLLMGeneratorGemma3
        elif (
            "qwen3-vl" in model_name_lower
            or "qwen3vl" in model_name_lower
            or "qwen3.5" in model_name_lower
            or "qwen3_5" in model_name_lower
        ):
            vllm_generator_cls = VLLMGeneratorQwen3VL
        else:
            vllm_generator_cls = VLLMGeneratorQwen
        print(f"Using {vllm_generator_cls.__name__} for vllm generation")

        generator_optional_kwargs = {}
        gpu_memory_utilization = getattr(args, "gpu_memory_utilization", None)
        if gpu_memory_utilization is not None:
            generator_optional_kwargs["gpu_memory_utilization"] = gpu_memory_utilization

        if use_system_prompt:
            if limit_mm_per_prompt is None:
                generator = vllm_generator_cls(args, **generator_optional_kwargs)
            else:
                generator = vllm_generator_cls(
                    args,
                    limit_mm_per_prompt=limit_mm_per_prompt,
                    **generator_optional_kwargs,
                )
        else:
            if limit_mm_per_prompt is None:
                generator = vllm_generator_cls(args, system_prompt=None, **generator_optional_kwargs)
            else:
                generator = vllm_generator_cls(
                    args,
                    limit_mm_per_prompt=limit_mm_per_prompt,
                    system_prompt=None,
                    **generator_optional_kwargs,
                )

        tokenizer = generator.tokenizer
    else:
        generator = GPTGenerator(args)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(ROOT_DIR, "utils", "gpt2")
        )

    return generator, tokenizer
