"""
Text based Column Extraction
"""

import copy
import json
import os
from argparse import ArgumentParser

import regex as re

from generation.generator import Generator
from nsql.database import NeuralDB
from scripts.col_sql import load_dataset_step_1
from utils.utils import build_passage_context, create_prompt, final_col_extraction, metadata_to_table


def load_dataset_step_2(args: ArgumentParser):
    with open(os.path.join(args.save_dir, args.input_program_file), "r") as f:
        data = json.load(f)

    # From Step 1, we need to get 1) the original data item and 2) the predicted SQL columns
    col_dict = {
        str(eid): {
            "sql_cols": data[str(eid)]["cols"],
            "ori_data_item": data[str(eid)]["ori_data_item"],
        }
        for eid in range(len(data))
    }

    return col_dict


def step_2_text_col_extraction(g_dict, args):
    """
    Extract columns after the col-text-extraction generation (step 2). Assumes that g_dict is a dictionary with keys from 0 to n-1, where n is the number of examples. Each key maps to a dictionary with the following structure:
    {
        "generations": [list of text generations],
        "ori_data_item": {
            "table": {
                "page_title": str,
                "header": list of column names,
                "rows": list of lists of values
            }
        }
    }

    Modifies g_dict inplace. Adds the following keys to g_dict:
    - cols: list of columns
    """
    pattern_col = r"(f_col\(\[(.*?)\]\))"
    pattern_col = re.compile(pattern_col, re.S)

    for g_eid in range(len(g_dict)):
        try:
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
            table = db.get_table_df()

            # Extract Columns
            preds = []
            for n in range(args.sampling_n):
                try:
                    pred = re.findall(pattern_col, g_dict[str(g_eid)]["generations"][n])[0][1]
                    pred = pred.split(", ")
                    new_pred = []
                    for i in pred:
                        if i.startswith("'"):
                            i = i[1:]

                        if i.endswith("'"):
                            i = i[:-1]
                        new_pred.append(i)
                    preds.append(new_pred)
                except:
                    pass

            if args.filtering_as_remover:
                preds = list({item for sublist in preds for item in sublist})  # flatten to 1D
                preds = [value for value in table.columns if value not in preds]
                # Union the sql columns with the text-reasoning columns (if you want to evaluate intersection, you need to set this to .intersection)
                pred = list(set(preds).union(g_dict[str(g_eid)]["sql_cols"]))
            else:
                # Unite the SQL columns (step 1) with the text-reasoning columns (step 2)
                pred = list(set().union(*preds, g_dict[str(g_eid)]["sql_cols"]))

            filtered_pred = [value for value in table.columns if value in pred]

            g_dict[str(g_eid)]["cols"] = filtered_pred
        except Exception as e:
            print(f"eid#{g_eid}, wtqid#{g_data_item['id']} with error: {e}")


def prepare_g_dict(g_dict, col_dict, g_eid: int, args):
    """
    This function prepares the g_dict (i.e. the output dict).
    """
    g_data_item = col_dict[g_eid]["ori_data_item"]
    g_dict[g_eid] = {
        "generations": [],
        "sql_cols": col_dict[g_eid]["sql_cols"] if not args.skip_step_1 else [],
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
    Process examples sequentially for step 2 (text-based column extraction).
    """
    g_dict = {}
    few_shot_prompt = generator.build_few_shot_prompt_from_file(file_path=args.prompt_file, n_shots=args.n_shots)

    # 1. Prepare the batch of requests
    for g_eid in [str(i) for i in range(len(col_dict))]:
        # prepare the output dict
        g_data_item = prepare_g_dict(g_dict, col_dict, g_eid, args)

        # Build the prompt
        passage_context = build_passage_context(g_data_item, selected_rows=None, selected_cols=None)
        prompt = create_prompt(tokenizer, generator, g_data_item, few_shot_prompt, args, passage_context=passage_context)  # fmt: skip

        # Generate
        if args.prompt_mode == "text-only":
            g_dict[g_eid]["prompt"] = prompt
        elif args.prompt_mode == "omni" or args.prompt_mode == "text-image":
            g_dict[g_eid]["content"] = [{"type": "text", "text": prompt}]

    # 2. Generate the responses
    generator.generate_batch_pass(g_dict=g_dict, args=args)
    step_2_text_col_extraction(g_dict, args)
    final_col_extraction(g_dict)

    return g_dict


def main(args, generator, tokenizer):
    # Load dataset
    if args.skip_step_1:
        col_dict = load_dataset_step_1(args)
    else:
        col_dict = load_dataset_step_2(args)

    # Annotate
    print("\n******* Annotating *******")
    g_dict = process_examples(args, generator, col_dict, tokenizer)

    # Save annotation results
    with open(os.path.join(args.save_dir, args.save_file_name), "w") as f:
        dict_keys = list(g_dict.keys())
        dict_keys.sort()
        sorted_g_dict = {i: g_dict[i] for i in dict_keys}
        json.dump(sorted_g_dict, f, indent=4)


if __name__ == "__main__":
    raise NotImplementedError(
        "No longer callable directly. Please run thorugh run_model.py - which will call the main function."
    )
