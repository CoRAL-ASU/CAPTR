"""
SQL based Column Extraction
"""

import copy
import json
import os
import re
import traceback

from tqdm import tqdm

from generation.generator import Generator
from nsql.database import NeuralDB
from scripts.load_hf_dataset_step1 import load_dataset_step_1
from utils.utils import build_passage_context, create_prompt, final_col_extraction, metadata_to_table


def step_1_sql_col_extraction(g_dict, args):
    """
    Extract columns after the SQL generation (step 1). Assumes that g_dict is a dictionary with keys from 0 to n-1, where n is the number of examples. Each key maps to a dictionary with the following structure:
    {
        "generations": [list of SQL generations],
        "ori_data_item": {
            "table_with_metadata": {
                "page_title": str,
                "header": list of column names,
                "rows": list of {
                    "content": list of values,
                    "type": list of "text",
                    "image_path": list of image paths (optional)
                }
            }
        }
    }

    Modifies g_dict inplace.
    Adds the following keys to g_dict:
    - cols: list of columns
    """

    # Check that g_dict follows the expected structure
    for g_eid in range(len(g_dict)):
        assert str(g_eid) in g_dict, f"g_dict does not have key {g_eid}"
        for key in ["generations", "ori_data_item"]:
            assert key in g_dict[str(g_eid)], f"g_dict[{g_eid}] does not have key '{key}'"
        assert "table_with_metadata" in g_dict[str(g_eid)]["ori_data_item"], f"g_dict[{g_eid}]['ori_data_item'] does not have key 'table_with_metadata'"  # fmt: skip
        for key in ["page_title", "header", "rows"]:
            assert key in g_dict[str(g_eid)]["ori_data_item"]["table_with_metadata"], f"g_dict[{g_eid}]['ori_data_item']['table_with_metadata'] does not have key '{key}'"  # fmt: skip

    pattern_col = r"(f_col\(\[(.*?)\]\))"
    pattern_col = re.compile(pattern_col, re.S)

    for g_eid in range(len(g_dict)):
        pred_col = ""
        try:
            pred_cols = []
            num_generations = min(getattr(args, "sampling_n", 2), len(g_dict[str(g_eid)]["generations"]))
            # Extract Columns
            for n in range(num_generations):
                try:
                    pred_col = re.findall(pattern_col, g_dict[str(g_eid)]["generations"][n])[0][1]
                    pred_col = pred_col.replace("'", "")
                    pred_col = pred_col.split(", ")
                    pred_cols.append(pred_col)

                except Exception:
                    pass

            pred_col = list(set().union(*pred_cols))
            g_data_item = copy.deepcopy(g_dict[str(g_eid)]["ori_data_item"])
            metadata_to_table(g_data_item)
            db = NeuralDB(
                tables=[
                    {
                        "title": g_data_item["table"]["page_title"],
                        "table": g_data_item["table"],
                    }
                ]
            )

            if pred_col != []:
                table = db.get_table_df()
                if args.filtering_as_remover:
                    # invert the filtering, i.e. set filtered_pred to the columns not in pred_col
                    filtered_pred = [value for value in table.columns if value not in pred_col]
                else:
                    filtered_pred = [value for value in table.columns if value in pred_col]

                g_dict[str(g_eid)]["cols"] = filtered_pred
            else:
                if args.filtering_as_remover:
                    # in case we don't have any columns, we set the columns to all columns since we don't filter out anything
                    g_dict[str(g_eid)]["cols"] = table.columns.tolist()
                else:
                    g_dict[str(g_eid)]["cols"] = []

        except Exception as e:
            print(f"Error processing example {g_eid}: {e}")
            traceback.print_exc()


def prepare_g_dict(g_dict, col_dict, g_eid: str):
    """
    This function prepares the g_dict (i.e. the output dict) for the SQL column extraction and returns the g_data_item, which contains the table and the title.
    """
    g_data_item = col_dict[g_eid]["ori_data_item"]
    g_dict[g_eid] = {
        "generations": [],
        "ori_data_item": copy.deepcopy(g_data_item),
    }
    metadata_to_table(g_data_item)
    db = NeuralDB(
        tables=[
            {
                "title": g_data_item["table"]["page_title"],
                "table": g_data_item["table"],
            }
        ]
    )
    g_data_item["table"] = db.get_table_df()
    g_data_item["title"] = db.get_table_title()
    return g_data_item


def process_examples(args, generator: Generator, col_dict, tokenizer):
    """
    Process examples sequentially for step 1 (SQL column extraction).
    """
    g_dict = {}
    few_shot_prompt = generator.build_few_shot_prompt_from_file(file_path=args.prompt_file, n_shots=args.n_shots)

    # 1. Prepare the batch of requests
    for g_eid in tqdm(range(len(col_dict)), desc="Processing examples"):
        g_eid = str(g_eid)
        g_data_item = prepare_g_dict(g_dict, col_dict, g_eid)

        # Build the prompt
        passage_context = build_passage_context(g_data_item, selected_rows=None, selected_cols=None)
        prompt = create_prompt(tokenizer, generator, g_data_item, few_shot_prompt, args, passage_context=passage_context)  # fmt: skip

        if args.prompt_mode == "text-only":
            g_dict[g_eid]["prompt"] = prompt
        elif args.prompt_mode == "omni" or args.prompt_mode == "text-image":
            g_dict[g_eid]["content"] = [{"type": "text", "text": prompt}]

    # 2. Generate the responses
    generator.generate_batch_pass(g_dict=g_dict, args=args)
    step_1_sql_col_extraction(g_dict, args)
    if args.skip_step_2:
        final_col_extraction(g_dict)

    return g_dict


def main(args, generator, tokenizer):
    col_dict = load_dataset_step_1(args)

    # Process examples
    print("\n******* Processing Examples *******")
    g_dict = process_examples(args, generator, col_dict, tokenizer)

    # Save results
    with open(os.path.join(args.save_dir, args.save_file_name), "w") as f:
        json.dump(g_dict, f, indent=4)


if __name__ == "__main__":
    raise NotImplementedError(
        "No longer callable directly. Please run thorugh run_model.py - which will call the main function."
    )
