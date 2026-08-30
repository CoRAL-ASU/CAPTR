"""
SQL based Reasoning
"""

import copy
import json
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from generation.generator import Generator
from nsql.database import NeuralDB
from nsql.sql_exec import Executor
from utils.normalizer import extract_first_sql_statement, post_process_sql
from utils.utils import build_passage_context, create_prompt, metadata_to_table


def load_dataset_step_5(args: ArgumentParser):
    with open(os.path.join(args.save_dir, args.input_program_file), "r") as f:
        data = json.load(f)

    data_dict = {
        str(eid): {
            "rows": data[str(eid)]["rows"],
            "cols": data[str(eid)]["cols"],
            "data_item": data[str(eid)]["ori_data_item"],
        }
        for eid in range(len(data))
    }

    return data_dict


def step_5_sql_reasoning(g_dict, args):
    """
    Process examples sequentially for step 5 (SQL reasoning).

    Modifies g_dict inplace. Adds the following keys to g_dict:
    - reason_sql_string: string of the reason SQL
    """
    executor = Executor(args, "")

    for g_eid in range(len(g_dict)):
        g_eid = str(g_eid)
        g_data_item = copy.deepcopy(g_dict[g_eid]["ori_data_item"])
        metadata_to_table(g_data_item)
        db = NeuralDB(
            tables=[
                {
                    "title": g_data_item["table"]["page_title"],
                    "table": g_data_item["table"],
                }
            ]
        )

        df = db.get_table_df()
        title = db.get_table_title()

        string = ""
        try:
            nsql = g_dict[g_eid]["generations"][0]
            sql = extract_first_sql_statement(nsql)
            if not sql:
                raise ValueError("No executable SQL statement found in generation")
            norm_sql = post_process_sql(
                sql_str=sql,
                df=df,
                process_program_with_fuzzy_match_on_db=True,
                table_title=title,
            )
            exec_answer = executor.sql_exec(norm_sql, db, verbose=False)

            # Convert dictionary to DataFrame
            new_df = pd.DataFrame(exec_answer["rows"], columns=exec_answer["header"])
            if "row_id" in new_df.columns.tolist():
                new_df.drop(columns="row_id", inplace=True)
            if "index" in new_df.columns.tolist():
                new_df.drop(columns="index", inplace=True)
            if np.array_equal(new_df.values, df.values):
                continue

            string = "Here is an additional evidence to help the answering process.\nAdditional Evidence:\n/*\n"
            string += "col : " + " | ".join(new_df.columns) + "\n"
            for row_id, row in new_df.iloc[: len(new_df)].iterrows():
                string += f"row {row_id} : "
                for column_id, header in enumerate(new_df.columns):
                    string += str(row[header])
                    if column_id != len(new_df.columns) - 1:
                        string += " | "
                string += "\n"
            string += "*/\n"

        except Exception as e:
            print(f"Error in step 5 sql reasoning (g_eid: {g_eid}): {e}")
            pass

        g_dict[g_eid]["reason_sql_string"] = string


def prepare_g_dict(g_dict, data, g_eid: str):
    """
    This function prepares the g_dict (i.e. the output dict).
    """
    g_data_item = data[g_eid]["data_item"]
    g_dict[g_eid] = {
        "generations": [],
        "cols": data[g_eid]["cols"],
        "rows": data[g_eid]["rows"],
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


def process_examples(args, generator: Generator, data, tokenizer):
    """
    Process examples sequentially for step 5 (SQL reasoning).
    """
    g_dict = {}
    few_shot_prompt = generator.build_few_shot_prompt_from_file(file_path=args.prompt_file, n_shots=args.n_shots)

    # 1. Prepare the batch of requests
    for g_eid in range(len(data)):
        g_eid = str(g_eid)
        g_data_item = prepare_g_dict(g_dict, data, g_eid)

        # Get the selected columns and filter the table accordinly
        selected_cols = copy.deepcopy(data[g_eid]["cols"])

        if "row_id" not in selected_cols:
            selected_cols.insert(0, "row_id")

        df = g_data_item["table"][selected_cols]

        # Filter the table according to the selected rows
        selected_rows = data[g_eid]["rows"]
        try:
            indices = [int(i) for i in selected_rows]
            if any(index >= len(df) for index in indices):
                raise IndexError("Index out of bounds")
            df = df[df.index.isin(indices)]
        except IndexError:
            df = df

        g_data_item["table"] = df

        # Build the prompt
        selected_rows_with_row_ids = [f"row {idx}" for idx in selected_rows]
        passage_context = build_passage_context(
            g_data_item, selected_rows=selected_rows_with_row_ids, selected_cols=selected_cols
        )
        prompt = create_prompt(tokenizer, generator, g_data_item, few_shot_prompt, args, passage_context=passage_context)  # fmt: skip

        if args.prompt_mode == "text-only":
            g_dict[g_eid]["prompt"] = prompt
        elif args.prompt_mode == "omni" or args.prompt_mode == "text-image":
            g_dict[g_eid]["content"] = [{"type": "text", "text": prompt}]

            assert not args.load_images_for_reasoning, (
                "Images and SQL don't work together: You are executing SQL reasoning, but you are also loading images (--load_images_for_reasoning). This is not supported."
            )

    # 2. Generate the responses
    generator.generate_batch_pass(g_dict=g_dict, args=args)
    step_5_sql_reasoning(g_dict, args)

    return g_dict


def main(args, generator, tokenizer):
    data_dict = load_dataset_step_5(args)

    # Annotate
    print("\n******* Annotating *******")
    g_dict = process_examples(args, generator, data_dict, tokenizer)

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
