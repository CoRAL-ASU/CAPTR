"""
SQL based Row Extraction
"""

import copy
import json
import os
import re
from argparse import ArgumentParser

from generation.generator import Generator
from nsql.database import NeuralDB
from nsql.parser import extract_rows
from nsql.sql_exec import Executor
from utils.normalizer import extract_first_sql_statement, post_process_sql
from utils.utils import build_passage_context, create_prompt, final_row_extraction, metadata_to_table


def load_dataset_step_3(args: ArgumentParser):
    with open(os.path.join(args.save_dir, args.input_program_file), "r") as f:
        data = json.load(f)

    col_dict = {
        str(eid): {
            "cols": data[str(eid)]["cols"],
            "data_item": data[str(eid)]["ori_data_item"],
        }
        for eid in range(len(data))
    }

    return col_dict


def step_3_sql_row_extraction(g_dict, args):
    """
    Extract columns after the col-text-extraction generation (step 3).

    Modifies g_dict inplace. Adds the following keys to g_dict:
    - rows: list of rows
    """
    pattern_row = r"(f_row\(\[(.*?)\]\))"
    pattern_row_num = r"\d+"
    pattern_row = re.compile(pattern_row, re.S)
    pattern_row_num = re.compile(pattern_row_num, re.S)

    for g_eid in range(len(g_dict)):
        g_eid = str(g_eid)
        g_data_item = copy.deepcopy(g_dict[g_eid]["ori_data_item"])
        metadata_to_table(g_data_item)

        sql_exec_success = []
        sql_exec_errors = []

        db = NeuralDB(
            tables=[
                {
                    "title": g_data_item["table"]["page_title"],
                    "table": g_data_item["table"],
                }
            ]
        )
        g_data_item["title"] = db.get_table_title()
        table = db.get_table_df()

        # Execute the SQL to get the statements to get the rows, i.e. add: "f_row(['row 0'])" to the end of the generation
        response_list = copy.deepcopy(g_dict[g_eid]["generations_pure"])

        for n in range(len(response_list)):
            exec_rows = []
            keys = ""
            try:
                executor = Executor(args, keys)
                sql = extract_first_sql_statement(response_list[n])
                if not sql:
                    raise ValueError("No executable SQL statement found in generation")
                if re.search(r"^\s*select\s+from\s+w\s*;?\s*$", sql, flags=re.IGNORECASE):
                    # Common malformed output with no projection; default to all rows.
                    sql = "SELECT row_id FROM w;"
                norm_sql = post_process_sql(
                    sql_str=sql,
                    df=db.get_table_df(),
                    process_program_with_fuzzy_match_on_db=True,
                    table_title=g_data_item["title"],
                )
                exec_answer = executor.sql_exec(norm_sql, db, verbose=False)
                exec_rows = extract_rows(exec_answer)
                response_list[n] += f" f_row({str(exec_rows)})"
                sql_exec_success.append(True)
                sql_exec_errors.append("")
            except Exception as e:
                print(f"SQL execution failed. Error: {e}")
                # Keep downstream stages usable even when SQL is malformed.
                fallback_rows = [f"row {i}" for i in table.index.tolist()]
                response_list[n] += f" f_row({str(fallback_rows)})"
                sql_exec_success.append(False)
                sql_exec_errors.append(str(e))

        g_dict[g_eid]["generations"] = response_list
        g_dict[g_eid]["sql_exec_success"] = sql_exec_success
        g_dict[g_eid]["sql_exec_errors"] = sql_exec_errors

        # Extract Rows
        preds = []
        num_generations = min(getattr(args, "sampling_n", 2), len(g_dict[g_eid]["generations"]))
        for n in range(num_generations):
            try:
                pred = re.findall(pattern_row, g_dict[g_eid]["generations"][n])[0][1]
                if pred == "*":
                    pred = ""
                    for i in range(len(table.index)):
                        pred += f"row {i}"
                        if i != len(table.index) - 1:
                            pred += ", "
                pred = pred.replace("'", "")
                pred = pred.split(", ")
                preds.append(pred)

            except Exception as e:
                print(f"Error in step 3 sql row extraction: {e}")
                pass

        pred = list({item for sublist in preds for item in sublist})  # flatten to 1D

        if args.filtering_as_remover:
            g_dict[g_eid]["rows_sql_inverted"] = copy.deepcopy(pred)
            # invert the filtering, i.e. set pred to the rows not in pred
            pred = [f"row {value}" for value in table.index if f"row {value}" not in pred]

        g_dict[g_eid]["rows"] = pred


def prepare_g_dict(g_dict, col_dict, g_eid: str):
    """
    This function prepares the g_dict (i.e. the output dict).
    """
    g_data_item = col_dict[g_eid]["data_item"]
    g_dict[g_eid] = {
        "generations": [],  # LLM generation + post processing, i.e. the f_row ([...])
        "generations_pure": [],  # LLM generation
        "cols": col_dict[g_eid]["cols"],
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
    Process examples sequentially for step 3 (SQL row extraction).
    """
    g_dict = {}
    few_shot_prompt = generator.build_few_shot_prompt_from_file(file_path=args.prompt_file, n_shots=args.n_shots)

    # 1. Prepare the batch of requests
    for g_eid in range(len(col_dict)):
        g_eid = str(g_eid)
        g_data_item = prepare_g_dict(g_dict, col_dict, g_eid)

        # Get the selected columns and filter the table accordinly
        selected_cols = copy.deepcopy(col_dict[g_eid]["cols"])

        if "row_id" not in selected_cols:
            selected_cols.insert(0, "row_id")

        df = g_data_item["table"][selected_cols]
        g_data_item["table"] = df

        # Build the prompt
        passage_context = build_passage_context(g_data_item, selected_rows=None, selected_cols=selected_cols)
        prompt = create_prompt(tokenizer, generator, g_data_item, few_shot_prompt, args, passage_context=passage_context)  # fmt: skip

        if args.prompt_mode == "text-only":
            g_dict[g_eid]["prompt"] = prompt
        elif args.prompt_mode == "omni" or args.prompt_mode == "text-image":
            g_dict[g_eid]["content"] = [{"type": "text", "text": prompt}]

    # 2. Generate the responses
    generator.generate_batch_pass(g_dict=g_dict, args=args)

    # 3. Rename "generations" to "generations_pure"
    for g_eid in range(len(g_dict)):
        g_eid = str(g_eid)
        g_dict[g_eid]["generations_pure"] = g_dict[g_eid]["generations"]
        g_dict[g_eid]["generations"] = []

    step_3_sql_row_extraction(g_dict, args)
    if args.skip_step_4:
        final_row_extraction(g_dict)

    return g_dict


def main(args, generator, tokenizer):
    col_dict = load_dataset_step_3(args)

    print("\n******* Annotating *******")
    g_dict = process_examples(args, generator, col_dict, tokenizer)

    with open(os.path.join(args.save_dir, args.save_file_name), "w") as f:
        dict_keys = list(g_dict.keys())
        dict_keys.sort()
        sorted_g_dict = {i: g_dict[i] for i in dict_keys}
        json.dump(sorted_g_dict, f, indent=4)


if __name__ == "__main__":
    raise NotImplementedError(
        "No longer callable directly. Please run thorugh run_model.py - which will call the main function."
    )
