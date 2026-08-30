"""
Final Text Reasoning
"""

import copy
import json
import os
from argparse import ArgumentParser

import dotenv
from PIL import Image, ImageFile

from baselines.interleaved_baseline import create_prompt_for_our_pipeline
from generation.generator import Generator
from nsql.database import NeuralDB
from utils.utils import (
    build_passage_context,
    create_prompt,
    metadata_to_table,
    metadata_to_table_with_image_placeholders,
)


dotenv.load_dotenv("../.env")


def load_dataset_step_6(args: ArgumentParser):
    with open(os.path.join(args.save_dir, args.input_program_file), "r") as f:
        data = json.load(f)

    row_col_dict = {
        str(eid): {
            "rows": data[str(eid)]["rows"],
            "cols": data[str(eid)]["cols"],
            "data_item": data[str(eid)]["ori_data_item"],
            "reason_sql_string": data[str(eid)]["reason_sql_string"] if not args.skip_step_5 else "",
        }
        for eid in range(len(data))
    }

    return row_col_dict


def prepare_g_dict(g_dict, row_col, g_eid: str, with_image_placeholders: bool = False):
    """
    This function prepares the g_dict (i.e. the output dict).
    """
    g_data_item = row_col[g_eid]["data_item"]
    g_dict[g_eid] = {
        "generations": [],
        "ori_data_item": copy.deepcopy(g_data_item),
    }
    if with_image_placeholders:
        image_map, caption_map = metadata_to_table_with_image_placeholders(g_data_item)
    else:
        image_map, caption_map = None, None
        metadata_to_table(g_data_item)
    db = NeuralDB(
        tables=[
            {
                "title": g_data_item["table"]["page_title"],
                "table": g_data_item["table"],
            }
        ]
    )
    g_data_item["title"] = db.get_table_title()
    g_data_item["table"] = db.get_table_df()
    return g_data_item, image_map, caption_map


def process_examples_with_images(
    args, generator: Generator, row_col, tokenizer, use_mmtabqa_interleaved_prompt: bool = False
):
    """
    Process examples sequentially for step 6 (final text-based reasoning). Load images and interleave the prompt.

    final prediction is in g_dict[g_eid]["generations"][0]
    """
    assert args.prompt_mode == "omni" or args.prompt_mode == "text-image", "Text reasoning WITH IMAGES only works with omni or text-image prompt mode."  # fmt: skip

    complete_g_dict = {}
    few_shot_prompt = generator.build_few_shot_prompt_from_file(file_path=args.prompt_file, n_shots=args.n_shots)
    no_image_placeholder_warning_emitted = False
    disable_cot = getattr(args, "reasoner_answer_only", False)

    current_index = 0
    shard_size = 20  # Reduced from 300 to match baseline memory efficiency

    while current_index < len(row_col):
        g_dict = {}
        end_index = min(current_index + shard_size, len(row_col))
        shard_image_items = 0
        shard_examples_with_images = 0
        print(f"\nProcessing shard {current_index}:{end_index}. Total: {len(row_col)}. Progress: {current_index / len(row_col):.2%}")  # fmt: skip
        for g_eid in range(current_index, end_index):
            g_eid = str(g_eid)
            g_data_item, image_map, caption_map = prepare_g_dict(g_dict, row_col, g_eid, with_image_placeholders=True)
            if not image_map and not no_image_placeholder_warning_emitted:
                print(
                    "⚠️ load_images_for_reasoning=True but no image placeholders were found in the table data. "
                    "Image reintroduction is skipped for these examples (common with captioned datasets)."
                )
                no_image_placeholder_warning_emitted = True

            ### Filter columns ###
            selected_cols = copy.deepcopy(row_col[g_eid]["cols"])

            if "row_id" in selected_cols:
                selected_cols.pop(0)

            df = g_data_item["table"][selected_cols]

            # get all unique text-values values contained in the column from the df
            if args.add_col_values_to_prompt:
                col_values = {col: df[col].unique().tolist() for col in selected_cols}

                # clear values in col_values from image placeholders
                all_image_placeholders = list(image_map.keys())
                for col, values in col_values.items():
                    cleaned_values = []
                    for value in values:
                        value_str = str(value)
                        for placeholder in all_image_placeholders:
                            value_str = value_str.replace(placeholder, " ")
                        value_str = " ".join(value_str.split()).strip()
                        if value_str:
                            cleaned_values.append(value_str)
                    col_values[col] = list(dict.fromkeys(cleaned_values))
            else:
                col_values = None

            ### Filter rows ###
            selected_rows = row_col[g_eid]["rows"]
            try:
                indices = [int(i) for i in selected_rows]
                if any(index >= len(df) for index in indices):
                    raise IndexError("Index out of bounds")
                df = df[df.index.isin(indices)]
            except IndexError:
                df = df

            g_data_item["table"] = df

            if use_mmtabqa_interleaved_prompt:
                prompt = create_prompt_for_our_pipeline(
                    data_item=g_data_item,
                    dataset_name=args.mmtabqa_sub_dataset,
                    selected_rows=selected_rows,
                    selected_cols=selected_cols,
                    col_values=col_values,
                    disable_cot=disable_cot,
                )
            else:
                select_rows_with_row_ids = [f"row {idx}" for idx in selected_rows]
                passage_context = build_passage_context(
                    g_data_item, selected_rows=select_rows_with_row_ids, selected_cols=selected_cols
                )
                prompt = create_prompt(tokenizer, generator, g_data_item, few_shot_prompt, args, passage_context=passage_context)  # fmt: skip
                prompt += "\nNo additional evidence provided.\n"

            # ----- Hacky prompt modifications for some ablations with captioning during interleaved reasoning -----
            if args.add_caption_to_reasoner_v2:
                prompt = prompt.replace(
                    "The table consists of data in the form of text and images.",
                    "The table consists of data in the form of text and images. Every table cell that contains an image has also a caption that describes the image.",
                )
                prompt = prompt.replace(
                    "Now, based upon the examples given above, you must understand the text and images given in the table and follow the steps 1-2 to answer the question corresponding to the table represented bt the data.",
                    "Now, based upon the examples given above, you must understand the text and images given in the table and follow the steps 1-2 to answer the question corresponding to the table represented bt the data. If useful, use the caption describing the image that is within the same table cell.",
                )
            # ----- Hacky prompt modifications for some ablations with captioning during interleaved reasoning -----

            prompt_for_storage = copy.deepcopy(prompt)

            # --- Build the multimodal prompt ---
            # Now that we have the prompt, we need to replace the placeholders with image and bring the prompt into in the interleaved format.
            # Problem: the order of the placeholders in the prompt may not be the same as in image_map
            content = [{"type": "text", "text": prompt}]

            for unique_image_placeholder, image_path in image_map.items():
                # since order of placeholders in prompt may not be the same as in image_map, we need to iterate over all text in content and find the image_placeholder and replace it.
                for i, item in enumerate(content):
                    if item["type"] == "text" and unique_image_placeholder in item["text"]:
                        text_parts = item["text"].split(unique_image_placeholder, 1)
                        if len(text_parts) < 2:
                            continue

                        if args.add_caption_to_reasoner_v1:
                            caption = caption_map.get(unique_image_placeholder, "")
                            text_parts[0] = f"{text_parts[0]} Image Caption: '{caption}'"
                            prompt_for_storage = prompt_for_storage.replace(unique_image_placeholder, f"Image Caption: '{caption}'{unique_image_placeholder}")  # fmt: skip

                        if args.add_caption_to_reasoner_v2:
                            caption = caption_map.get(unique_image_placeholder, "")
                            caption = " ".join([value.strip() for value in caption.split(",")])
                            text_parts[0] = f"{text_parts[0]} {caption} "
                            prompt_for_storage = prompt_for_storage.replace(unique_image_placeholder, f"{caption} {unique_image_placeholder}")  # fmt: skip

                        image_base_path = os.getenv("MMTABQA_IMAGE_BASE_PATH")
                        image_path_str = str(image_path).strip()
                        if "|" in image_path_str:
                            image_path_str = image_path_str.split("|", 1)[0].strip()
                        resolved_image_path = image_path_str
                        if image_base_path and not os.path.isabs(image_path_str):
                            resolved_image_path = os.path.join(image_base_path, image_path_str)
                        try:
                            image = Image.open(resolved_image_path).convert("RGB")
                            content[i] = {"type": "text", "text": text_parts[0]}  # text before image
                            content.insert(i + 1, {"type": "image", "image": image})  # image
                            content.insert(i + 2, {"type": "text", "text": text_parts[1]})  # text after image

                        except OSError:
                            print(f"⚠️ Image file is truncated: {image_path}. Trying to load it with LOAD_TRUNCATED_IMAGES flag")  # fmt: skip
                            ImageFile.LOAD_TRUNCATED_IMAGES = True
                            try:
                                image = Image.open(resolved_image_path).convert("RGB")
                                content[i] = {"type": "text", "text": text_parts[0]}  # text before image
                                content.insert(i + 1, {"type": "image", "image": image})  # image
                                content.insert(i + 2, {"type": "text", "text": text_parts[1]})  # text after image
                            except OSError:
                                print(f"⚠️⚠️⚠️ Image file is still truncated: {image_path}. Skipping this image. ⚠️⚠️⚠️")
                                content[i] = {"type": "text", "text": text_parts[0]}  # text before image
                                content.insert(i + 1, {"type": "text", "text": "IMAGE"})  # image-text-replacement
                                content.insert(i + 2, {"type": "text", "text": text_parts[1]})  # text after image
                            finally:
                                ImageFile.LOAD_TRUNCATED_IMAGES = False
                        except Exception as e:
                            print(f"⚠️ Failed to load image {image_path}: {e}. Keeping text-only placeholder replacement.")
                            content[i] = {"type": "text", "text": text_parts[0]}
                            content.insert(i + 1, {"type": "text", "text": "IMAGE"})
                            content.insert(i + 2, {"type": "text", "text": text_parts[1]})

                        continue

            # Finally, store the content in the g_dict
            example_image_items = sum(1 for x in content if isinstance(x, dict) and x.get("type") == "image")
            g_dict[g_eid]["num_images_attached"] = example_image_items
            shard_image_items += example_image_items
            if example_image_items > 0:
                shard_examples_with_images += 1
            g_dict[g_eid]["content"] = content
            g_dict[g_eid]["prompt_for_storage"] = prompt_for_storage

        print(
            f"Shard multimodal summary: examples with images {shard_examples_with_images}/{end_index - current_index}; "
            f"total attached image items {shard_image_items}."
        )

        generator.generate_batch_pass(g_dict=g_dict, args=args)

        # For saving, replace the content object with the text-only version
        for g_eid_key in g_dict:
            if "prompt_for_storage" in g_dict[g_eid_key]:
                g_dict[g_eid_key]["content"] = g_dict[g_eid_key].pop("prompt_for_storage")

        complete_g_dict.update(g_dict)
        
        # Clean up images from memory after each shard
        for eid in g_dict.keys():
            if "content" in g_dict[eid]:
                content_items = g_dict[eid]["content"]
                if isinstance(content_items, list):
                    for item in content_items:
                        if isinstance(item, dict) and item.get("type") == "image" and "image" in item:
                            del item["image"]
        
        del g_dict
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        current_index = end_index

    return complete_g_dict


def process_examples(args, generator: Generator, row_col, tokenizer, use_mmtabqa_interleaved_prompt: bool = False):
    """
    Process examples sequentially for step 6 (final text-based reasoning).

    final prediction is in g_dict[g_eid]["generations"][0]
    """
    g_dict = {}
    few_shot_prompt = generator.build_few_shot_prompt_from_file(file_path=args.prompt_file, n_shots=args.n_shots)
    disable_cot = getattr(args, "reasoner_answer_only", False)

    # 1. Prepare the batch of requests
    for g_eid in range(len(row_col)):
        g_eid = str(g_eid)
        g_data_item, image_map, caption_map = prepare_g_dict(g_dict, row_col, g_eid)

        ### Filter columns ###
        selected_cols = copy.deepcopy(row_col[g_eid]["cols"])

        if "row_id" in selected_cols:
            selected_cols.pop(0)

        df = g_data_item["table"][selected_cols]

        ### Filter rows ###
        selected_rows = row_col[g_eid]["rows"]
        try:
            indices = [int(i) for i in selected_rows]
            if any(index >= len(df) for index in indices):
                raise IndexError("Index out of bounds")
            df = df[df.index.isin(indices)]
        except IndexError:
            df = df

        g_data_item["table"] = df

        if use_mmtabqa_interleaved_prompt:
            prompt = create_prompt_for_our_pipeline(
                data_item=g_data_item,
                dataset_name=args.mmtabqa_sub_dataset,
                selected_rows=selected_rows,
                selected_cols=selected_cols,
                disable_cot=disable_cot,
            )
        else:
            # Additional Evidence
            reason_sql_string = row_col[g_eid]["reason_sql_string"]
            if args.skip_step_5:
                additional_evidence = "\nNo additional evidence provided.\n "
            else:
                if reason_sql_string and reason_sql_string != "":
                    additional_evidence = "\n" + reason_sql_string
                else:
                    additional_evidence = "\nNo additional evidence provided.\n "

            # Build the prompt
            max_tokens_buffer = len(tokenizer.tokenize(additional_evidence))
            select_rows_with_row_ids = [f"row {idx}" for idx in selected_rows]
            passage_context = build_passage_context(
                g_data_item, selected_rows=select_rows_with_row_ids, selected_cols=selected_cols
            )
            prompt = create_prompt(tokenizer, generator, g_data_item, few_shot_prompt, args, passage_context=passage_context, max_tokens_buffer=max_tokens_buffer)  # fmt: skip
            prompt += additional_evidence

        if args.prompt_mode == "text-only":
            g_dict[g_eid]["prompt"] = prompt
        elif args.prompt_mode == "omni" or args.prompt_mode == "text-image":
            g_dict[g_eid]["content"] = [{"type": "text", "text": prompt}]

    # 2. Generate the responses
    generator.generate_batch_pass(g_dict=g_dict, args=args)

    return g_dict


def process_examples_for_fetaqa(args, generator: Generator, row_col, tokenizer):
    """
    Process examples sequentially for step 6 (final text-based reasoning).

    final prediction is in g_dict[g_eid]["generations"][0]
    """
    g_dict = {}
    disable_cot = getattr(args, "reasoner_answer_only", False)

    for g_eid in range(len(row_col)):
        g_eid = str(g_eid)
        g_data_item, image_map, caption_map = prepare_g_dict(g_dict, row_col, g_eid)

        ### Filter columns ###
        selected_cols = copy.deepcopy(row_col[g_eid]["cols"])

        if "row_id" in selected_cols:
            selected_cols.pop(0)

        df = g_data_item["table"][selected_cols]

        ### Filter rows ###
        selected_rows = row_col[g_eid]["rows"]
        try:
            indices = [int(i) for i in selected_rows]
            if any(index >= len(df) for index in indices):
                raise IndexError("Index out of bounds")
            df = df[df.index.isin(indices)]
        except IndexError:
            df = df

        g_data_item["table"] = df

        prompt = create_prompt_for_our_pipeline(
            g_data_item,
            args.mmtabqa_sub_dataset,
            selected_rows=selected_rows,
            selected_cols=selected_cols,
            disable_cot=disable_cot,
        )

        # Additional Evidence
        reason_sql_string = row_col[g_eid]["reason_sql_string"]
        if not args.skip_step_5 and reason_sql_string and reason_sql_string != "":
            prompt += "\n" + reason_sql_string

        if args.prompt_mode == "text-only":
            g_dict[g_eid]["prompt"] = prompt
        elif args.prompt_mode == "omni" or args.prompt_mode == "text-image":
            g_dict[g_eid]["content"] = [{"type": "text", "text": prompt}]

    generator.generate_batch_pass(g_dict=g_dict, args=args)

    return g_dict


def main(args, generator, tokenizer):
    row_col_dict = load_dataset_step_6(args)

    # Annotate
    print("\n******* Annotating *******")
    if args.load_images_for_reasoning:
        g_dict = process_examples_with_images(
            args,
            generator,
            row_col_dict,
            tokenizer,
            use_mmtabqa_interleaved_prompt=args.use_mmtabqa_interleaved_prompt,
        )
    else:
        if args.mmtabqa_sub_dataset == "fetaqa":
            print("⚠️ Working with FetaQA: Changing the prompt file and format to one equivalent to the interleaved baseline prompt. This is necessary because FetaQA has a different answer format.")  # fmt: skip
            g_dict = process_examples_for_fetaqa(args, generator, row_col_dict, tokenizer)
        else:
                g_dict = process_examples(
                    args,
                    generator,
                    row_col_dict,
                    tokenizer,
                    use_mmtabqa_interleaved_prompt=args.use_mmtabqa_interleaved_prompt,
                )

    # Save annotation results
    with open(os.path.join(args.save_dir, args.save_file_name), "w") as f:
        json.dump(g_dict, f, indent=4)

    return g_dict


if __name__ == "__main__":
    raise NotImplementedError(
        "No longer callable directly. Please run thorugh run_model.py - which will call the main function."
    )
