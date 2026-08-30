# Python script to transform MMTabQA into a Hugging Face dataset. Based partially on https://github.com/MMTabQA/mmtabqa/blob/main/src/modelling/image-captioning/step1.py
import json
import os
import pathlib
from typing import List

import dotenv
import pandas as pd

import datasets


dotenv.load_dotenv("../.env")


Question_Type_Map = {
    "Explicit Question": "EQ",
    "Visual-Based Question": "VQ",
    "Answer-Mention Question": "AQ",
    "Implicit Question": "IQ",
}


GOLD_COL_NAME_TO_CSV_COLNAME_MAP = {
    "weather": {
        "High": "Hi",
    },
    "sparkline-column": {
        "Sales Trend Pattern": "Sales Trend",
        "Margin % Trend Pattern": "Margin Trend",
    },
    "sparkline-charts-column-darkblue": {
        "Profit Trend Pattern": "Profit Trend",
        "12 Month Revenue Pattern": "Goal Trend",
        "Sales Trend Pattern": "Sales Trend",
    },
    "presidential_elctions_4": {
        "Net gain/loss of president's party[aq]": "Net gain/loss of president's party",
        "Unnamed: 4": "Net gain/loss of president's party.1",
    },
    # ...
    # TODO: complete this list. Can get all the values via executing:
    # python3 create_dataset_mmtbench.py
}


def main():
    data_dir = pathlib.Path(os.getenv("MMT_BENCH_BASE_PATH"))
    # data_dir = pathlib.Path("/tmp/mmtbench")

    # Load data
    all_tables_dir = data_dir / "All_Tables"
    all_tables_dir_list = [p for p in all_tables_dir.iterdir() if p.is_dir()]

    create_dataset(all_tables_dir_list, data_dir)


def _load_table_csv(table_dir, table_name, list_of_tables_to_rotate: List[str]):
    table_csv = pd.read_csv(table_dir / f"{table_name}.csv")
    columns = table_csv.columns.tolist()
    duplicate_columns = [col for col in columns if col.endswith(".1")]

    # remove duplicates
    if len(duplicate_columns) == len(columns) // 2:
        # print(f"Found {len(duplicate_columns)} duplicate columns: {duplicate_columns}")
        table_csv = table_csv.drop(columns=duplicate_columns)

    # rotate table if needed
    if table_name in list_of_tables_to_rotate:
        print(f"Rotating table: {table_name}")
        table_csv = table_csv.T
        new_columns = table_csv.iloc[0]
        table_csv = table_csv[1:]
        table_csv.columns = new_columns

    # TODO: if column header is an image or is empty: Use captioning
    has_image_column = False
    for column in table_csv.columns:
        if str(column).startswith("images/"):
            has_image_column = True
            break
    if has_image_column:
        print(f"Table {table_name} has image column")

    return table_csv


def create_dataset(
    all_tables_dir_list,
    data_dir,
):
    features = datasets.Features(
        {
            "id": datasets.Value("string"),
            "question": datasets.Value("string"),
            "answer_text": datasets.Sequence(datasets.Value("string")),
            "table_id": datasets.Value("string"),
            "table": {
                "section_title": datasets.Value("string"),
                "page_title": datasets.Value("string"),
                "header": datasets.Sequence(datasets.Value("string")),
                # The rows do NOT contain the images. We only keep the image paths there.
                "rows": datasets.Sequence(
                    datasets.Sequence(  # Each row is a list of cells
                        {
                            "type": datasets.Value("string"),
                            "content": datasets.Value("string"),  # This will hold the text or the image path
                        }
                    )
                ),
            },
        }
    )

    def mmtabqa_to_hf_dataset_generator(all_tables, target_question_type):
        """
        Generator that yields structured dictionaries for the Hugging Face dataset.
        Each yielded item represents one question and its multimodal table.
        """

        skipped_count = 0
        total_count = 0

        for table_dir in all_tables:
            table_name = table_dir.name  # e.g. world_stock_markets

            if table_name == "List_of_Major_League_Baseball_players__Wa_Wh_1":
                # fix incoherent table name
                table_csv = pd.read_csv(table_dir / "List_of_Major_League_Baseball_players_Wa_Wh_.csv")
            else:
                table_csv = _load_table_csv(table_dir, table_name, ["wind 3", "30 year temp for toronto"])

            # TODO:
            # assert that table_csv and original_text_tables have the same number of columns and rows
            original_text_tables = pd.read_excel(data_dir / "Upper_Bound" / f"{table_name}.xlsx")
            # assert len(table_csv.columns) == len(original_text_tables.columns), f"{table_name}: Number of columns in table_csv and original_text_tables do not match. {len(table_csv.columns)} != {len(original_text_tables)}\n\ntable_csv:\n{table_csv}\n\noriginal_text_tables.columns: {original_text_tables.columns}"  # fmt: skip
            # assert len(table_csv) == len(original_text_tables), f"{table_name}: Number of rows in table_csv and original_text_tables do not match. {len(table_csv)} != {len(original_text_tables)}\n\ntable_csv: {table_csv}\n\noriginal_text_tables: {original_text_tables}"  # fmt: skip
            total_count += 1
            if total_count > 490:
                print(f"total_count: {total_count}    skipped_count: {skipped_count}")
            if len(table_csv.columns) != len(original_text_tables.columns) or len(table_csv) != len(
                original_text_tables
            ):
                skipped_count += 1
                continue

            # rename the columns so that they match. Note: They may still be not in the same order.
            if table_name in GOLD_COL_NAME_TO_CSV_COLNAME_MAP:
                original_text_tables.columns = [
                    GOLD_COL_NAME_TO_CSV_COLNAME_MAP[table_name].get(col, col) for col in original_text_tables.columns
                ]

            # assert that the table_csv and original_text_tables have the same column names
            if set(table_csv.columns.tolist()) != set(original_text_tables.columns.tolist()):
                print(f"{table_name}: Column names in table_csv and original_text_tables do not match.\n{original_text_tables.columns.tolist()} !=\n{table_csv.columns.tolist()}\n\n\n")  # fmt: skip

            if table_name == "List_of_geochronologic_names":
                # fix incoherent file name
                questions_for_table = json.load(open(data_dir / "Questions" / f"{table_name}.JSON"))
            else:
                questions_for_table = json.load(open(data_dir / "Questions" / f"{table_name}.json"))
            questions_metadata = json.load(open(data_dir / "Question-Metadata" / f"{table_name}.json"))

            if len(questions_for_table) != len(questions_metadata):
                print(f"{table_name}: Number of questions for table is not equal to number of questions metadata. {len(questions_for_table)} != {len(questions_metadata)}")  # fmt: skip
                continue

            for question_number, question_dict in enumerate(questions_for_table):
                # 1. check if question is of the right type
                question_type_attr_names = [
                    "question Type",
                    "Question Type",
                    f"Question {question_number + 1} Type",
                    f"question {question_number + 1} Type",
                    f"Question{question_number + 1} Type",
                    f"Question {question_number} Type",
                    "Questio6 Type",
                    f"Question{question_number} Type",
                    f"Question {question_number + 2} Type",
                    f"question {question_number + 1} type",
                    "question type",
                ]
                question_type = None
                for question_type_attr_name in question_type_attr_names:
                    if question_type_attr_name in questions_metadata[question_number]:
                        question_type = questions_metadata[question_number][question_type_attr_name]
                        break

                if question_type is None:
                    print(f"\nQuestion {question_number} of Table '{table_name}' has no question type. Question dict for this question: {question_dict}")  # fmt: skip
                    print(f"question_number: {question_number}")
                    print(f"\n\nquestions_metadata: {questions_metadata[question_number]}")
                    print(f"\n\nall questions_metadata: {questions_metadata}")
                    print(f"\n\nquestion_type_attr_names: {question_type_attr_names}")

                question_type = Question_Type_Map[question_type]
                if question_type != target_question_type:
                    continue

                # 2. get question answer
                try:
                    question = question_dict["question"]
                except KeyError:
                    print(f"\nQuestion {question_number} of Table '{table_name}' has no question. Question dict for this question: {question_dict}")  # fmt: skip
                    continue
                answer_text = question_dict["answer"]

                headers = table_csv.columns.tolist()

                # 3. convert table rows
                rows = []
                for _, row in table_csv.iterrows():
                    row_dict = []
                    for header in headers:
                        cell_content = row[header]
                        if str(cell_content).startswith("images/"):
                            image_path = os.path.join(table_dir, cell_content)
                            assert os.path.exists(image_path), f"Image not found in image_dir: {image_path}"
                            row_dict.append({"type": "image", "content": image_path})
                        else:
                            row_dict.append({"type": "text", "content": cell_content})
                    rows.append(row_dict)

                yield {
                    "id": f"{table_name}_{question_number}",
                    "question": question,
                    "answer_text": [answer_text],
                    "table_id": table_name,
                    "table": {
                        "page_title": table_name.replace("_", " "),
                        "section_title": "",  # don't have section titles for MMTBench
                        "header": headers,  # easy
                        "rows": rows,
                    },
                }

    # Create datasets for EQ, AQ, ...
    for target_question_type in [
        "EQ",
        "AQ",
        "VQ",
        "IQ",
    ]:
        print(f"Creating Hugging Face Dataset for {target_question_type} ...")

        # Create dataset
        hf_dataset = datasets.Dataset.from_generator(
            generator=mmtabqa_to_hf_dataset_generator,
            features=features,
            split=datasets.Split.TEST,
            gen_kwargs={"all_tables": all_tables_dir_list, "target_question_type": target_question_type},
        )

        # save to disk
        print("Dataset created successfully!")
        print(hf_dataset)
        print("\nExample entry:")
        print(hf_dataset[0])
        save_dir = os.path.join(data_dir, "converted_to_hf_dataset")
        print(f"\nSaving dataset to disk: {save_dir} ...")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        hf_dataset.save_to_disk(os.path.join(save_dir, f"mmtbench_{target_question_type}"))

    print("Done.")


if __name__ == "__main__":
    main()
