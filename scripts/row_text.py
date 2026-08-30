"""
Text based Row Extraction
"""

import copy
import json
import os
from argparse import ArgumentParser

import regex as re

from generation.generator import Generator
from nsql.database import NeuralDB
from utils.utils import build_passage_context, create_prompt, final_row_extraction, metadata_to_table


def load_dataset_step_4(args: ArgumentParser):
    with open(os.path.join(args.save_dir, args.input_program_file), "r") as f:
        data = json.load(f)

    row_dict = {
        str(eid): {
            "rows_sql": data[str(eid)]["rows"] if not args.skip_step_3 else [],
            "cols": data[str(eid)]["cols"],
            "data_item": data[str(eid)]["ori_data_item"],
            "rows_sql_inverted": data[str(eid)]["rows_sql_inverted"] if "rows_sql_inverted" in data[str(eid)] else [],
        }
        for eid in range(len(data))
    }

    return row_dict


def step_4_text_row_extraction(g_dict, args):
    """
    Process examples sequentially for step 4 (text-based row extraction).

    Modifies g_dict inplace. Adds the following keys to g_dict:
    - rows: list of rows
    """
    pattern_row = r"(f_row\(\[(.*?)\]\))"
    pattern_row = re.compile(pattern_row, re.S)

    for g_eid in range(len(g_dict)):
        g_eid = str(g_eid)
        try:
            preds = []
            in_exception = False
            for n in range(2):
                try:
                    pred = re.findall(pattern_row, g_dict[g_eid]["generations"][n])[0][1]
                    if pred == "*":
                        pred = ""
                        for i in range(len(g_dict[g_eid]["data_item"]["table"]["rows"])):
                            pred += f"row {i}"
                            if i != len(g_dict[g_eid]["data_item"]["table"]["rows"]) - 1:
                                pred += ", "

                        pred = pred.split(", ")
                        preds.append(pred)
                    else:
                        pred = pred.split(", ")
                        preds.append(pred)
                except Exception:
                    in_exception = True
                    pred = g_dict[g_eid]["rows_sql"]
                    # If the prediction couldn't be parsed and the SQL rows are empty, use all rows
                    if pred == [] or pred == "":
                        pred = ""
                        for i in range(len(g_dict[g_eid]["data_item"]["table"]["rows"])):
                            pred += f"row {i}"
                            if i != len(g_dict[g_eid]["data_item"]["table"]["rows"]) - 1:
                                pred += ", "
                        pred = pred.split(", ")
                    preds.append(pred)

            if args.filtering_as_remover and not in_exception:
                # invert the filtering, i.e. set pred to the rows not in pred
                preds = list({item for sublist in preds for item in sublist})  # flatten to 1D
                inverted_preds = []
                for i in range(len(g_dict[g_eid]["data_item"]["table"]["rows"])):
                    row_id_name = f"row {i}"
                    if row_id_name not in preds:
                        inverted_preds.append(row_id_name)

                # Union the sql rows with the text-reasoning rows (if you want to evaluate intersection, you need to set this to .intersection)
                pred = list(set(inverted_preds).union(g_dict[g_eid]["rows_sql"]))

            else:
                # Unite the SQL rows (step 3) with the text-reasoning rows (step 4)
                pred = list(set().union(*preds, g_dict[g_eid]["rows_sql"]))

            g_dict[g_eid]["rows"] = pred

        except Exception as e:
            print(f"Error in step 4 for example {g_eid}: {e}. Keeping the SQL rows.")
            g_dict[g_eid]["rows"] = g_dict[g_eid]["rows_sql"]


def prepare_g_dict(g_dict, data, g_eid: str):
    """
    This function prepares the g_dict (i.e. the output dict).
    """
    g_data_item = data[g_eid]["data_item"]
    g_dict[g_eid] = {
        "generations": [],
        "cols": data[g_eid]["cols"],
        "rows_sql": data[g_eid]["rows_sql"],
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


def process_examples(args, generator: Generator, row_dict, tokenizer):
    """
    Process examples sequentially for step 4 (text-based row extraction).
    """
    g_dict = {}
    few_shot_prompt = generator.build_few_shot_prompt_from_file(file_path=args.prompt_file, n_shots=args.n_shots)

    # 1. Prepare the batch of requests
    for g_eid in range(len(row_dict)):
        g_eid = str(g_eid)
        g_data_item = prepare_g_dict(g_dict, row_dict, g_eid)

        df = g_data_item["table"]

        if args.skip_step_3:
            initial_response = "[]"
        else:
            if args.filtering_as_remover:
                initial_response = row_dict[g_eid]["rows_sql_inverted"]
            else:
                initial_response = row_dict[g_eid]["rows_sql"]

        # Get the selected columns and filter the table accordingly
        selected_cols = copy.deepcopy(row_dict[g_eid]["cols"])
        df = g_data_item["table"][selected_cols]
        g_data_item["table"] = df

        passage_context = build_passage_context(g_data_item, selected_rows=None, selected_cols=selected_cols)
        prompt = create_prompt(tokenizer, generator, g_data_item, few_shot_prompt, args,is_row_text_step=True, initial_response=initial_response, passage_context=passage_context)  # fmt: skip

        if args.prompt_mode == "text-only":
            g_dict[g_eid]["prompt"] = prompt
        elif args.prompt_mode == "omni" or args.prompt_mode == "text-image":
            g_dict[g_eid]["content"] = [{"type": "text", "text": prompt}]

    # 2. Generate the responses
    generator.generate_batch_pass(g_dict=g_dict, args=args)
    step_4_text_row_extraction(g_dict, args)
    final_row_extraction(g_dict)
    return g_dict


def main(args, generator, tokenizer):
    row_dict = load_dataset_step_4(args)

    # Enter dataset size for inference: range(0, len(dataset))
    print("\n******* Annotating *******")
    g_dict = process_examples(args, generator, row_dict, tokenizer)

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
