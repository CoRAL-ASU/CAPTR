import argparse
import json
import os
import re
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict

from utils.runtime_env import configure_runtime_environment, load_repo_dotenv

configure_runtime_environment()

import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError
from tqdm import tqdm

import datasets

# Add parent directory to path to import generation module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from baselines.baselines_utils import sample_dataset
from generation.generator_vllm_gemma3 import VLLMGeneratorGemma3

load_repo_dotenv()


def _batch_load_images_in_example(batch: Dict[str, list], image_base_path=None) -> Dict[str, list]:
    """
    Helper function to process a BATCH of examples from the dataset.
    It iterates through the examples in the batch, finds entries marked as 'image' in the nested table,
    and replaces the filename in the 'content' list with a loaded PIL Image object.
    To keep track of the image name, we add am "image_path" key to each row.

    This function is designed to be used with `datasets.Dataset.set_transform()`.
    """

    # Get the 'table' column from the batch. It's a list of table dictionaries.
    table_column = batch["table"]
    processed_table_column = []

    for table_dict in table_column:
        processed_rows = []

        # The logic for processing a SINGLE table_dict remains the same as before.
        for row_dict in table_dict["rows"]:
            processed_content_list = []
            image_path_list = []
            processed_types = []

            for cell_type, cell_content in zip(row_dict["type"], row_dict["content"]):
                # Handle cells with multiple images or image/text combinations
                if "image" in cell_type and isinstance(cell_content, str):
                    # Parse complex cell formats like "TEXT:Valencia|IMAGES:/path1.png|/path2.png"
                    text_part = ""
                    image_paths = []
                    
                    if "TEXT:" in cell_content and "IMAGES:" in cell_content:
                        parts = cell_content.split("|")
                        for part in parts:
                            if part.startswith("TEXT:"):
                                text_part = part[5:]  # Remove "TEXT:" prefix
                            elif part.startswith("IMAGES:"):
                                # Multiple images can be separated by |
                                image_paths.append(part[7:].replace("\\", "/"))  # Remove "IMAGES:" prefix and normalize path
                            elif part.strip():  # Additional image paths without prefix
                                image_paths.append(part.replace("\\", "/"))
                    elif "|" in cell_content:
                        # Multiple image paths separated by |
                        image_paths = [p.strip().replace("\\", "/") for p in cell_content.split("|") if p.strip()]
                    else:
                        # Single image
                        image_paths = [cell_content.replace("\\", "/")]
                    
                    # Load all images in this cell
                    cell_images = []
                    cell_image_paths = []
                    for img_path in image_paths:
                        if image_base_path is None:
                            full_image_path = img_path
                        else:
                            full_image_path = os.path.join(image_base_path, img_path)

                        try:
                            image = Image.open(full_image_path).convert("RGB")
                            image = image.resize((256, 256), Image.Resampling.LANCZOS)
                            image.load()
                            cell_images.append(image)
                            cell_image_paths.append(full_image_path)
                        except FileNotFoundError:
                            print(f"⚠️ Image not found at {full_image_path}. Skipping this image. ⚠️")
                        except UnidentifiedImageError:
                            print(f"⚠️ Unidentified image: {full_image_path}. Skipping this image. ⚠️")
                        except OSError:
                            print(f"⚠️ Image file is truncated: {full_image_path}. Trying to load it with LOAD_TRUNCATED_IMAGES flag")  # fmt: skip
                            ImageFile.LOAD_TRUNCATED_IMAGES = True
                            try:
                                image = Image.open(full_image_path).convert("RGB")
                                image = image.resize((256, 256), Image.Resampling.LANCZOS)
                                image.load()
                                cell_images.append(image)
                                cell_image_paths.append(full_image_path)
                            except OSError:
                                print(f"⚠️⚠️⚠️ Image file is still truncated: {full_image_path}. Skipping this image. ⚠️⚠️⚠️")
                            finally:
                                ImageFile.LOAD_TRUNCATED_IMAGES = False
                    
                    # Store cell content: combine text and images as a list
                    if text_part and cell_images:
                        # Cell has both text and images
                        processed_content_list.append([text_part] + cell_images)
                        processed_types.append("mixed")
                        image_path_list.append("|".join(cell_image_paths) if cell_image_paths else "")
                    elif cell_images:
                        # Cell has only images
                        if len(cell_images) == 1:
                            processed_content_list.append(cell_images[0])
                            processed_types.append("image")
                            image_path_list.append(cell_image_paths[0] if cell_image_paths else "")
                        else:
                            processed_content_list.append(cell_images)
                            processed_types.append("multi_image")
                            image_path_list.append("|".join(cell_image_paths) if cell_image_paths else "")
                    else:
                        # All images failed to load
                        processed_content_list.append(text_part if text_part else "IMAGE")
                        processed_types.append("text")
                        image_path_list.append("")
                else:
                    processed_content_list.append(cell_content)
                    processed_types.append(cell_type)
                    image_path_list.append("")

            processed_rows.append(
                {"type": processed_types, "content": processed_content_list, "image_path": image_path_list}
            )

        # Update the 'rows' in the individual table_dict and add it to our results list.
        table_dict["rows"] = processed_rows
        processed_table_column.append(table_dict)

    # 2. Replace the original 'table' column in the batch with the new one containing PIL Images.
    batch["table"] = processed_table_column

    # 3. Return the entire modified batch.
    return batch


def _load_partial_input_baseline_in_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper function to process one example for the partial input baseline.
    Replaces image filenames with {ENTITY-X} placeholders, where X is incremented
    for each unique image in the table.
    """
    table = example["table"]
    image_to_entity = {}
    entity_counter = 1

    processed_rows = []
    for row_dict in table["rows"]:
        processed_content_list = []
        image_path_list = []
        for cell_type, cell_content in zip(row_dict["type"], row_dict["content"]):
            if cell_type == "image" and isinstance(cell_content, str):
                if cell_content not in image_to_entity:
                    image_to_entity[cell_content] = f"{{ENTITY-{entity_counter}}}"
                    entity_counter += 1
                processed_content_list.append(image_to_entity[cell_content])
                image_path_list.append(cell_content)
            else:
                processed_content_list.append(cell_content)
                image_path_list.append("")

        new_row = {
            "type": ["text"] * len(processed_content_list),
            "content": processed_content_list,
            "image_path": image_path_list,
        }
        processed_rows.append(new_row)

    example["table"]["rows"] = processed_rows
    return example


def _load_gold_entity_replaced_images_in_example_mmtabqa(dataset, num_proc, **kwargs):
    """
    MMTabQA: Replace the images with their original text entity. Store the image paths in image_path.
    """

    dataset_name = kwargs.get("mmtabqa_sub_dataset")
    if dataset_name is None:
        dataset_name = kwargs.get("dataset_name")

    base_dir_value = os.getenv("MMTABQA_BASE_PATH")
    if not base_dir_value:
        raise ValueError("MMTABQA_BASE_PATH must be set to load MMTabQA datasets.")
    base_dir = Path(base_dir_value)
    dataset_name_to_dir = {
        "wikitq": "WikiTableQuestions",
        "wikisql": "WikiSQL",
        "fetaqa": "FetaQA",
        "hybridqa": "HybridQA",
    }

    image_id_to_original_string_map = json.load(open(base_dir / dataset_name_to_dir[dataset_name] / "image_id_to_original_string.json"))  # fmt:skip
    example_id_to_original_table_df = pd.read_json(base_dir / dataset_name_to_dir[dataset_name] / "tables.jsonl", lines=True).set_index("table_id")  # fmt:skip
    image_id_to_image_path = json.load(open(base_dir / "image_id_to_image_path.json"))  # fmt:skip

    IMG_TAG_PATTERN = re.compile(r"\{IMG-\{([^}]+)\}\}")

    def _load_gold_entity_replaced_images_in_example_single_example(example: Dict[str, Any]) -> Dict[str, Any]:
        table_id = example["table_id"]
        table_df = example_id_to_original_table_df.loc[table_id]
        table_array = table_df["table_array"]

        processed_rows = []
        for table_row in table_array[1:]:  # Skip first row as it is the header
            type_list = []
            content_list = []
            image_path_list = []
            for cell_content in table_row:
                match = IMG_TAG_PATTERN.search(str(cell_content))
                if match:
                    # Case 1: Cell contains an image tag. Do the following:
                    # 1) Replace with the original string
                    # 2) Add image path to "image_path"
                    image_id = f"{{IMG-{{{match.group(1)}}}}}"
                    type_list.append("text")
                    content_list.append(image_id_to_original_string_map.get(image_id))
                    image_path_list.append(image_id_to_image_path.get(image_id))
                else:
                    # Case 2: Otherwise, append the text content
                    type_list.append("text")
                    content_list.append(cell_content)
                    image_path_list.append("")

            new_row = {"type": type_list, "content": content_list, "image_path": image_path_list}

            processed_rows.append(new_row)

        example["table"]["rows"] = processed_rows
        return example

    dataset = dataset.map(
        _load_gold_entity_replaced_images_in_example_single_example,
        num_proc=num_proc,
        desc="Applying gold entity replaced images",
        load_from_cache_file=False,
    )

    return dataset


def _load_gold_entity_replaced_images_in_example_mmtbench(dataset, num_proc, **kwargs):
    """
    MMTBench: Replace the images with their original text entity. Store the image paths in image_path.
    """

    raise NotImplementedError(
        "MMTBench oracle not supported. MMTBench's tables do not match the final MMTBench tables with images. Mapping between non-oracle and oracle tables is impossible."
    )


def _load_gold_entity_replaced_images_in_example(dataset, num_proc, **kwargs):
    """
    Replace the images with their original text entity. Store the image paths in image_path.
    """

    dataset_name = kwargs.get("mmtabqa_sub_dataset")
    if dataset_name is None:
        dataset_name = kwargs.get("dataset_name")

    if dataset_name == "wikitq" or dataset_name == "wikisql" or dataset_name == "fetaqa" or dataset_name == "hybridqa":
        return _load_gold_entity_replaced_images_in_example_mmtabqa(dataset, num_proc, **kwargs)
    elif dataset_name == "mmtbench":
        return _load_gold_entity_replaced_images_in_example_mmtbench(dataset, num_proc, **kwargs)
    else:
        raise ValueError(f"Dataset name {dataset_name} not supported")


def load_mmtabqa_dataset(
    dataset_path: str,
    image_base_path: str = None,
    num_proc: int = 1,
    load_images: bool = False,
    partial_input_baseline: bool = False,
    gold_entity_replaced_images: bool = False,
    **kwargs,
) -> datasets.Dataset:
    """
    Loads MMTabQA as Hugging Face dataset & replaces image filenames in the nested
    table structure with loaded Pillow Image objects.

    Args:
        dataset_path (str): The path to the saved Hugging Face dataset directory.
        image_base_path (str): The base directory where the image files (e.g., '0687f95dc2990e0.png') are stored. (only needed when load_images is True)
        num_proc (int): The number of processes to use for parallel processing.
        load_images (bool): Whether to load the images into the dataset.
        partial_input_baseline (bool): Don't load images but instead replace the images with "ENTITY-X" placeholders (for the partial input baseline of the MMTabQA paper).
        gold_entity_replaced_images (bool): Whether to replace the images with the gold entity placeholders.
            if True, mmtabqa_dataset_path and dataset_name must be provided in kwargs

    Returns:
        datasets.Dataset: A Hugging Face dataset with the following structure:
        - "table" is a list of table dictionaries with the following structure:
            - "table"["rows"] is a list of row dictionaries containing "type", "content" and "image_path" (image_path is only present if load_images=True)
            - "table"["section_title"] is a string
            - "table"["page_title"] is a string
            - "table"["header"] is a list of strings

    """
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset path not found at: {dataset_path}")
    if sum([load_images, partial_input_baseline, gold_entity_replaced_images]) > 1:
        raise ValueError("Only one of load_images, partial_input_baseline, gold_entity_replaced_images can be True")

    # 1. Load the dataset from disk
    print(f"Loading dataset from {dataset_path}  ...")
    dataset = datasets.load_from_disk(dataset_path)

    # 3. Apply the mapping function to each example in the dataset
    if load_images:
        print("Setting on-the-fly transform to load images...")
        transform_function = partial(_batch_load_images_in_example, image_base_path=image_base_path)
        dataset.set_transform(transform_function)

    elif partial_input_baseline:
        dataset = dataset.map(
            _load_partial_input_baseline_in_example,
            num_proc=num_proc,
            desc="Applying partial input baseline",
            load_from_cache_file=False,
        )

    elif gold_entity_replaced_images:
        dataset = _load_gold_entity_replaced_images_in_example(dataset, num_proc=num_proc, **kwargs)

    return dataset


def _infer_num_columns(rows):
    if not rows:
        return 0
    first_row = rows[0]
    if isinstance(first_row, dict):
        return len(first_row.get("content", []))
    if isinstance(first_row, list):
        return len(first_row)
    return 0


def _normalize_header_name(header_value: Any, col_idx: int) -> str:
    header_str = "" if header_value is None else str(header_value).strip()
    header_lower = header_str.lower()

    if not header_str or header_lower.startswith("unnamed:"):
        return f"header_{col_idx + 1}"

    if header_str.startswith("images/") or any(header_lower.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"]):
        base_name = os.path.basename(header_str)
        stem = os.path.splitext(base_name)[0].replace("_", " ").strip()
        return stem if stem else f"header_image_{col_idx + 1}"

    return header_str


def _build_normalized_headers(raw_headers, rows) -> list[str]:
    expected_cols = _infer_num_columns(rows)
    normalized_headers = []

    for col_idx in range(expected_cols):
        raw_header = raw_headers[col_idx] if col_idx < len(raw_headers) else ""
        normalized_headers.append(_normalize_header_name(raw_header, col_idx))

    return normalized_headers


def _create_captions_with_context(
    dataset: datasets.Dataset,
    dataset_raw: datasets.Dataset,
    vllm_generator: VLLMGeneratorGemma3,
    detailed: bool,
    share_between_dataset_dict: Dict[str, Any],
) -> callable:
    """
    Creates captions for each image in the dataset, considering the context of the table and row. Each image occurrence is captioned individually.
    """
    cell_to_caption: Dict[tuple, str] = share_between_dataset_dict.get("cell_to_caption", {})
    processed_table_ids = share_between_dataset_dict.get("processed_table_ids", set())
    current_index = 0
    shard_size = 500  # Increased from 200 to reduce image I/O overhead

    while current_index < len(dataset):
        end_index = min(current_index + shard_size, len(dataset))
        print(
            f"\nProcessing shard {current_index}:{end_index}. Total: {len(dataset)}. Progress: {current_index / len(dataset):.2%}"
        )

        inputs_for_vllm = {}
        input_keys = []

        for example_idx in tqdm(
            range(current_index, end_index),
            desc=f"Collecting images and contexts from shard {current_index}:{end_index}",
        ):
            table_id = dataset[example_idx]["table_id"]
            if table_id in processed_table_ids:
                continue
            processed_table_ids.add(table_id)

            table_with_images = dataset[example_idx]["table"]
            table_raw = dataset_raw[example_idx]["table"]

            example_passages = dataset_raw[example_idx].get("passages", [])

            page_title = table_raw.get("page_title", "")
            section_title = table_raw.get("section_title", "")
            headers = _build_normalized_headers(table_raw.get("header", []), table_with_images["rows"])

            for row_idx, row_with_images in enumerate(table_with_images["rows"]):
                for col_idx, cell_type in enumerate(row_with_images["type"]):
                    # Handle image, multi_image, and mixed types
                    if cell_type in ["image", "multi_image", "mixed"]:
                        cell_content = row_with_images["content"][col_idx]
                        
                        # Collect images to caption
                        images_to_caption = []
                        text_part = None
                        
                        if cell_type == "image":
                            images_to_caption = [cell_content]
                        elif cell_type == "multi_image":
                            images_to_caption = cell_content  # Already a list
                        elif cell_type == "mixed":
                            # First element is text, rest are images
                            text_part = cell_content[0] if cell_content else None
                            images_to_caption = cell_content[1:] if len(cell_content) > 1 else []
                        
                        # Caption each image separately
                        for img_idx, image_to_caption in enumerate(images_to_caption):
                            content = []
                            if detailed:
                                prompt_text = "Your task is to generate a detailed, structured caption for an image that is part of a table. The caption should describe the main objects in the image, their attributes, and any relationships between them. Do not include any text other than the caption itself.\n\nTable Context:\n"
                            else:
                                prompt_text = "Your task is to generate a short caption for an image within a table cell. Output ONLY a comma-separated list of 3-7 keywords that describe the image content. These keywords should describe the main visual elements of the image, prioritizing those most relevant to the data in the image's corresponding table row. Output format: keyword1, keyword2, keyword3, ...\n\nTable Context:\n"

                            if page_title:
                                prompt_text += f"Page Title: {page_title}\n"
                            if section_title and section_title != "":
                                prompt_text += f"Section Title: {section_title}\n"
                            if headers:
                                prompt_text += f"Column Headers: {', '.join(headers)}\n\n"

                            # --- For HybridQA ---
                            # example_passages is a dict looking like this: {"id": [1, 2, ...], "text": ["...", "...", ...], "type": ["text", "mm", ...], "linked_cell": [[row_idx, col_idx], [row_idx, col_idx], ...]}
                            # For MMTabReal, passages doesn't exist so example_passages is an empty list
                            linked_passages = []
                            if isinstance(example_passages, dict) and "linked_cell" in example_passages:
                                for i, linked_cell in enumerate(example_passages["linked_cell"]):
                                    if linked_cell == [
                                        row_idx + 1,
                                        col_idx,
                                    ]:  # need to offset by 1 because the header row is not included
                                        linked_passages.append(example_passages["text"][i])

                            if linked_passages:
                                prompt_text += "\nLinked Passage Context:\n"
                                for passage_text in linked_passages:
                                    prompt_text += f"- {passage_text}\n"
                                prompt_text += "\n"
                            # --- For HybridQA ---

                            prompt_text += "Row Context:\n"
                            image_column_header = None

                            for i, val in enumerate(row_with_images["content"]):
                                h = headers[i] if i < len(headers) else f"header_{i + 1}"
                                if row_with_images["type"][i] in ["image", "multi_image", "mixed"]:
                                    if i == col_idx:
                                        image_column_header = h
                                    else:
                                        # Include other images in context
                                        other_images = []
                                        if row_with_images["type"][i] == "image":
                                            other_images = [val]
                                        elif row_with_images["type"][i] == "multi_image":
                                            other_images = val
                                        elif row_with_images["type"][i] == "mixed":
                                            other_images = val[1:] if len(val) > 1 else []
                                        
                                        if other_images:
                                            prompt_text += f"{h}: "
                                            content.append({"type": "text", "text": prompt_text})
                                            for other_img in other_images:
                                                content.append({"type": "image", "image": other_img})
                                            prompt_text = "\n"
                                else:
                                    prompt_text += f"{h}: {val}\n"

                            if image_column_header is None:
                                image_column_header = headers[col_idx] if col_idx < len(headers) else f"header_{col_idx + 1}"

                            if detailed:
                                prompt_text += f"\nGiven this context, please provide a detailed caption for this image that is part of the {image_column_header} column: "
                            else:
                                prompt_text += f"\nBased on the table and row context, generate a comma-separated list of keywords for the following image from the '{image_column_header}' column. Output only the keywords and nothing else. "

                            content.append({"type": "text", "text": prompt_text})
                            content.append({"type": "image", "image": image_to_caption})

                            request_key = f"{table_id}_{row_idx}_{col_idx}_{img_idx}"
                            inputs_for_vllm[request_key] = {"content": content}
                            input_keys.append((table_id, row_idx, col_idx, img_idx))

        if inputs_for_vllm:
            print(f"Found {len(inputs_for_vllm)} images to caption in shard.")
            print("Generating captions...")

            # generate captions
            vllm_generator.generate_batch_pass(inputs_for_vllm)
            print(f"Generated {len(inputs_for_vllm)} captions.")

            # store captions in map: (example_id, row_idx, col_idx, img_idx) -> caption
            for i, key in enumerate(input_keys):
                request_key = f"{key[0]}_{key[1]}_{key[2]}_{key[3]}"
                caption = inputs_for_vllm[request_key]["generations"][0]
                cell_to_caption[key] = caption
                if i == 0:
                    print(f"Example caption: {caption}")

        current_index = end_index

    def _replace_images_with_context_captions(example: Dict[str, Any]) -> Dict[str, Any]:
        table_id = example["table_id"]
        table = example["table"]
        processed_rows = []

        for row_idx, row in enumerate(table["rows"]):
            # Check if row is a list (raw format) or dict (processed format with images)
            if isinstance(row, list):
                # Raw format: rows are lists of cell values
                processed_row = []
                for col_idx, cell_value in enumerate(row):
                    if isinstance(cell_value, str) and ("TEXT:" in cell_value or "IMAGES:" in cell_value):
                        # Parse the TEXT:|IMAGES: format
                        text_part = ""
                        img_count = 0
                        
                        parts = cell_value.split("|")
                        for part in parts:
                            part = part.strip()
                            if part.startswith("TEXT:"):
                                text_part = part[5:]
                            elif part.startswith("IMAGES:") or "/" in part or "\\" in part:
                                # This is an image path
                                img_count += 1
                        
                        # Get captions for this cell
                        captions = []
                        for img_idx in range(img_count):
                            caption = cell_to_caption.get((table_id, row_idx, col_idx, img_idx))
                            if caption:
                                captions.append(caption)
                        
                        # Combine text and captions
                        if text_part and captions:
                            # Mixed: text + captions
                            new_value = text_part + " | " + " | ".join(captions)
                        elif captions:
                            # Only captions (image-only cell)
                            new_value = " | ".join(captions)
                        elif text_part:
                            # Only text (captions failed)
                            new_value = text_part
                        else:
                            # Everything failed
                            new_value = "IMAGE"
                        
                        processed_row.append(new_value)
                    else:
                        # Regular text cell, keep as-is
                        processed_row.append(cell_value)
                
                processed_rows.append(processed_row)
            
            else:
                # Dict format (from dataset with images loaded): has "type", "content", "image_path" keys
                processed_content_list = []
                processed_type_list = []
                image_path_list = []

                for col_idx, (cell_type, cell_content) in enumerate(zip(row["type"], row["content"])):
                    if cell_type in ["image", "multi_image", "mixed"]:
                        # Collect captions for this cell
                        captions = []
                        text_part = None
                        # Extract image path from content (raw dataset has paths as strings)
                        if cell_type == "image" and isinstance(cell_content, str):
                            original_image_path = cell_content
                        elif cell_type == "multi_image" and isinstance(cell_content, list):
                            # For multi_image, join paths with |
                            original_image_path = "|".join([p for p in cell_content if isinstance(p, str)])
                        elif cell_type == "mixed" and isinstance(cell_content, list):
                            # For mixed, extract image paths from the list (skip first element which is text)
                            original_image_path = "|".join([p for p in cell_content[1:] if isinstance(p, str)])
                        else:
                            # Fallback: try to get from row's image_path field if it exists
                            original_image_path = row.get("image_path", [""] * len(row["content"]))[col_idx] if "image_path" in row else ""
                        img_paths = []
                        
                        # Determine how many images are in this cell
                        num_images = 0
                        if cell_type == "image":
                            num_images = 1
                        elif cell_type == "multi_image":
                            num_images = len(cell_content) if isinstance(cell_content, list) else 0
                        elif cell_type == "mixed":
                            text_part = cell_content[0] if cell_content else None
                            num_images = len(cell_content) - 1 if isinstance(cell_content, list) and len(cell_content) > 1 else 0
                        
                        # Get all captions for this cell
                        for img_idx in range(num_images):
                            caption = cell_to_caption.get((table_id, row_idx, col_idx, img_idx))
                            if caption:
                                captions.append(caption)
                        
                        # Combine results
                        if text_part and captions:
                            # Mixed: combine text and captions
                            combined = text_part + " | " + " | ".join(captions)
                            processed_content_list.append(combined)
                            processed_type_list.append("text")
                            image_path_list.append(original_image_path)
                        elif captions:
                            # Only captions (single or multiple)
                            combined = " | ".join(captions) if len(captions) > 1 else captions[0]
                            processed_content_list.append(combined)
                            processed_type_list.append("text")
                            image_path_list.append(original_image_path)
                        elif text_part:
                            # Only text (all images failed)
                            processed_content_list.append(text_part)
                            processed_type_list.append("text")
                            image_path_list.append(original_image_path)
                        else:
                            # Everything failed
                            processed_content_list.append("IMAGE")
                            processed_type_list.append("text")
                            image_path_list.append(original_image_path)
                    else:
                        processed_content_list.append(cell_content)
                        processed_type_list.append(cell_type)
                        image_path_list.append("")

                processed_rows.append(
                    {"type": processed_type_list, "content": processed_content_list, "image_path": image_path_list}
                )

        example["table"]["rows"] = processed_rows
        example["table"]["header"] = _build_normalized_headers(example["table"].get("header", []), processed_rows)
        return example

    # save the share_between_dataset_dict (this serves as a global cache state so that we don't have to re-caption tables that we already have captions for)
    share_between_dataset_dict["cell_to_caption"] = cell_to_caption
    share_between_dataset_dict["processed_table_ids"] = processed_table_ids

    return _replace_images_with_context_captions, share_between_dataset_dict


def _create_naive_captions(
    dataset: datasets.Dataset,
    dataset_raw: datasets.Dataset,
    vllm_generator: VLLMGeneratorGemma3,
    share_between_dataset_dict: Dict[str, Any],
) -> Dict[str, Any]:
    image_path_to_caption = share_between_dataset_dict.get("image_path_to_caption", {})
    images_seen = share_between_dataset_dict.get("images_seen", set())
    processed_table_ids = share_between_dataset_dict.get("processed_table_ids", set())  # ADDED to skip re-captioning
    current_index = 0
    shard_size = 300  # Increased from 100 to reduce image I/O overhead

    while current_index < len(dataset):
        # We will load A LOT of images into memory. So we will process the dataset in shards.
        end_index = min(current_index + shard_size, len(dataset))
        print(f"\nProcessing shard {current_index}:{end_index}. Total: {len(dataset)}. Progress: {current_index / len(dataset):.2%}")  # fmt:skip

        # Process each example in the shard
        all_images = []
        all_image_paths = []
        for example_idx in tqdm(
            range(current_index, end_index),
            desc=f"Collecting images from shard {current_index}:{end_index} (total: {len(dataset)})",
        ):
            # ADDED: Skip tables that have already been processed (e.g., same table in AQ/EQ/VQ/IQ splits)
            table_id = dataset[example_idx]["table_id"]
            if table_id in processed_table_ids:
                continue
            processed_table_ids.add(table_id)
            
            # Get raw example to access original image paths
            raw_table = dataset_raw[example_idx]["table"]
            table = dataset[example_idx]["table"]

            # Collect images with their corresponding paths
            for row_idx, (row_dict, raw_row_dict) in enumerate(zip(table["rows"], raw_table["rows"])):
                for col_idx in range(len(row_dict["type"])):
                    cell_type = row_dict["type"][col_idx]
                    if cell_type == "image":
                        path_to_image = raw_row_dict["content"][col_idx]
                        if path_to_image not in images_seen:
                            images_seen.add(path_to_image)
                            all_images.append(row_dict["content"][col_idx])
                            all_image_paths.append(path_to_image)

        if all_images:
            print(f"Found {len(all_images)} unique images in shard")
            print("Generating captions...")
            try:
                # Prepare inputs for vLLM vision-language model
                inputs = {}
                for img_idx, img in enumerate(
                    tqdm(all_images, total=len(all_images), desc="Preparing input prompts for vLLM")
                ):
                    inputs[str(img_idx)] = {
                        "content": [
                            {
                                "type": "text",
                                "text": "Summarise this picture in 10 words or less. Do not include any text other than the caption and do not ask any follow-up questions. Your answer should only include the caption.",
                            },
                            {
                                "type": "image",
                                "image": img,
                            },
                        ]
                    }

                vllm_generator.generate_batch_pass(inputs)

                print(f"Generated {len(inputs)} captions. Examples:")
                print(f"First:    {inputs['0']['generations'][0]}")

                for image_idx, img in enumerate(all_images):
                    path = all_image_paths[image_idx]
                    caption = inputs[str(image_idx)]["generations"][0]
                    image_path_to_caption[path] = caption

            except Exception as e:
                raise Exception(f"Error generating captions: {e}")

        current_index = end_index

    # Apply caption replacement to dataset. Add image_path to each row.
    def _replace_images_with_captions(example: Dict[str, Any]) -> Dict[str, Any]:
        table = example["table"]
        processed_rows = []

        for row in table["rows"]:
            processed_content_list = []
            processed_type_list = []
            image_path_list = []

            for cell_type, cell_content in zip(row["type"], row["content"]):
                if cell_type == "image":
                    if cell_content not in image_path_to_caption:
                        # If it isn't in the mapping, we skipped it because it through an OSError during image loading
                        processed_content_list.append("IMAGE")
                        processed_type_list.append("text")
                        image_path_list.append("")
                    else:
                        # Replace image with caption using the original image path
                        caption = image_path_to_caption[cell_content]
                        processed_content_list.append(caption)
                        processed_type_list.append("text")
                        image_path_list.append(cell_content)
                else:
                    processed_content_list.append(cell_content)
                    processed_type_list.append(cell_type)
                    image_path_list.append("")

            processed_rows.append(
                {"type": processed_type_list, "content": processed_content_list, "image_path": image_path_list}
            )

        example["table"]["rows"] = processed_rows
        example["table"]["header"] = _build_normalized_headers(example["table"].get("header", []), processed_rows)
        return example

    # save the share_between_dataset_dict (this serves as a global cache state so that we don't have to re-caption images that we already have captions for)
    share_between_dataset_dict["image_path_to_caption"] = image_path_to_caption
    share_between_dataset_dict["images_seen"] = images_seen
    share_between_dataset_dict["processed_table_ids"] = processed_table_ids  # ADDED to persist table_id tracking

    return _replace_images_with_captions, share_between_dataset_dict


def create_captioned_dataset(
    dataset_path: str,
    output_path: str,
    share_between_dataset_dict: Dict[str, Any],
    image_base_path: str = None,
    caption_mode: str = "naive",
    num_proc: int = 10,
    num_gpus: int = 1,
    n_examples: int = None,
) -> None:
    """
    Creates a captioned version of the MMTabQA dataset where images are replaced with their captions.
    Final stored dataset has the following features: features: ['id', 'question', 'answer_text', 'table_id', 'table'].
    - "table" is a list of table dictionaries.
    - "table"["rows"] is a list of row dictionaries.
    - "table"["rows"]["type"] is a list of cell types (always "text")
    - "table"["rows"]["content"] is a list of cell contents (str)
    - "table"["rows"]["image_path"] is a list of image paths.

    Args:
        dataset_path (str): Path to the original HuggingFace dataset directory
        image_base_path (str): Base directory where image files are stored
        output_path (str): Path where the captioned dataset will be saved
        share_between_dataset_dict: Dict[str, Any]. Share already processed images between datasets. First call: must be empty, subsequent calls: use output of previous call as input.
        image_base_path (str): Base directory where image files are stored. If None, we assume that the image paths are already included in the dataset table cells.
        caption_mode (str): Mode of captioning. "naive", "detailed", "short_with_context"
        num_proc (int): Number of processes for dataset processing
        num_gpus (int): Number of GPUs to use for generation

    caption_modes:
        - "naive": don't provide any context. Prompt is like: "Summarize this image: <image>" (actual prompt is a bit longer)
        - "short_with_context": tell the model to create a short caption of the model. Input contains the information that the image is part of a table, the table name, the column names and the row values that this image is part of.
        - "detailed": like "short_with_context" but instructed to create structured captions (all objects and their relations)
    """

    # 0. set max_generation_tokens
    if caption_mode == "naive":
        max_generation_tokens = 50
    elif caption_mode == "short_with_context":
        max_generation_tokens = 50
    elif caption_mode == "detailed":
        max_generation_tokens = 256  # Reduced from 512 to speed up generation (saves ~50% time for detailed mode)

    # 1. init the Gemma 3 vllm generator (using local snapshot to avoid downloading)
    args = argparse.Namespace()
    args.engine = "vllm"
    args.vllm_model_name = os.getenv("CAPTR_GEMMA3_MODEL_ID", "google/gemma-3-27b-it")
    args.number_of_gpus = num_gpus
    args.max_api_total_tokens = 20000
    args.max_generation_tokens = max_generation_tokens
    args.prompt_mode = "text-image"
    args.prompt_style = ""
    args.seed = 42

    vllm_generator = VLLMGeneratorGemma3(
        args, limit_mm_per_prompt={"image": 50}, gpu_memory_utilization=0.85, system_prompt=None
    ) 

    # 2. Prepare data:
    # 2.1 Load the original dataset without images first to get image filenames
    print("Loading dataset without images to extract filenames...")
    dataset_raw = load_mmtabqa_dataset(
        dataset_path=dataset_path,
        image_base_path=image_base_path,
        num_proc=num_proc,
        load_images=False,
        partial_input_baseline=False,
    )

    # 2.2 Load the dataset with images for caption generation
    print("Loading dataset with images...")
    dataset = load_mmtabqa_dataset(
        dataset_path=dataset_path,
        image_base_path=image_base_path,
        num_proc=num_proc,
        load_images=True,
        partial_input_baseline=False,
    )

    if n_examples is not None:
        print(f"Sampling captioning dataset to {n_examples} examples (seed=42) before caption creation.")
        dataset_raw = sample_dataset(dataset_raw, n_examples=n_examples, seed=42)
        dataset = sample_dataset(dataset, n_examples=n_examples, seed=42)

    # Now that we have the dataset with images, we can create the captions according to the caption_mode.
    # Each captioning function returns a function to replace the images with captions.
    if caption_mode == "naive":
        replace_images_with_captions_fn, share_between_dataset_dict = _create_naive_captions(
            dataset, dataset_raw, vllm_generator, share_between_dataset_dict
        )
    elif caption_mode == "short_with_context":
        replace_images_with_captions_fn, share_between_dataset_dict = _create_captions_with_context(
            dataset, dataset_raw, vllm_generator, detailed=False, share_between_dataset_dict=share_between_dataset_dict
        )
    elif caption_mode == "detailed":
        replace_images_with_captions_fn, share_between_dataset_dict = _create_captions_with_context(
            dataset, dataset_raw, vllm_generator, detailed=True, share_between_dataset_dict=share_between_dataset_dict
        )

    # Apply the captioning function to the dataset
    captioned_dataset = dataset_raw.map(
        replace_images_with_captions_fn, num_proc=num_proc, desc="Replacing images with captions"
    )

    print(f"Saving captioned dataset to {output_path}...")
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    captioned_dataset.save_to_disk(output_path)

    # sanity checks
    assert len(captioned_dataset) == len(dataset_raw), (
        "The captioned dataset has a different number of examples than the original dataset"
    )
    for example_idx in range(len(captioned_dataset)):
        assert len(captioned_dataset[example_idx]["table"]["rows"]) == len(dataset_raw[example_idx]["table"]["rows"]), "The captioned dataset has a different number of rows than the original dataset"  # fmt: skip
        for row in captioned_dataset[example_idx]["table"]["rows"]:
            assert "type" in row, "The captioned dataset has a row without a 'type' column"
            assert "content" in row, "The captioned dataset has a row without a 'content' column"
            assert "image_path" in row, "The captioned dataset has a row without a 'image_path' column"

    print("\n\nDone! Sanity check: All images have been replaced with captions.\n\n")

    return share_between_dataset_dict
