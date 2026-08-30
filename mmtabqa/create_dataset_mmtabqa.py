# Python script to transform MMTabQA into a Hugging Face dataset. Based partially on https://github.com/MMTabQA/mmtabqa/blob/main/src/modelling/image-captioning/step1.py
import json
import os
import re

import pandas as pd

import datasets

from utils.runtime_env import load_repo_dotenv

load_repo_dotenv()


IMG_TAG_PATTERN = re.compile(r"\{IMG-\{([^}]+)\}\}")


def main():
    data_dir = os.getenv("MMTABQA_BASE_PATH")
    image_dir = os.getenv("MMTABQA_IMAGE_BASE_PATH")

    if not data_dir:
        raise ValueError("MMTABQA_BASE_PATH must be set before creating the dataset.")
    if not image_dir:
        raise ValueError("MMTABQA_IMAGE_BASE_PATH must be set before creating the dataset.")

    mmtabqa_datasets_dirs = {
        "wikitq": "WikiTableQuestions/",
        "wikisql": "WikiSQL/",
        "fetaqa": "FetaQA/",
        "hybridqa": "HybridQA/",
    }
    for dataset_name, dataset_dir in mmtabqa_datasets_dirs.items():
        # 2. Load data
        dataset_dir = os.path.join(data_dir, dataset_dir)
        (
            explicit_ans_mention,
            explicit_questions,
            visual_questions,
            implicit_questions,
            table_metadata,
            tables,
            image_id_to_image_path,
            mm_passages,
            text_passages,
        ) = load_data(dataset_dir, data_dir, dataset_name=dataset_name)

        # 3. Create dataset
        create_dataset(
            explicit_ans_mention=explicit_ans_mention,
            explicit_questions=explicit_questions,
            visual_questions=visual_questions,
            implicit_questions=implicit_questions,
            metadata_df=table_metadata,
            tables_df=tables,
            image_dir=image_dir,
            image_id_to_image_path=image_id_to_image_path,
            data_dir=data_dir,
            dataset_name=dataset_name,
            mm_passages=mm_passages,
            text_passages=text_passages,
        )


def load_data(dataset_dir, data_dir, dataset_name):
    """
    Loads all necessary data files from the WikiTableQuestions directory.
    Returns:
        explicit_ans_mention: pd.DataFrame - Answer-Mention Questions
        explicit_questions: pd.DataFrame - Explicit Questions
        visual_questions: pd.DataFrame - Visual Questions
        image_id_to_original_string: dict - Mapping of image IDs to original strings
        image_id_to_qid: dict - Mapping of image IDs to question IDs
        image_id_to_wikipedia_link: dict - Mapping of image IDs to Wikipedia links
        table_metadata: pd.DataFrame - Table metadata
        tables: pd.DataFrame - Tables
    """
    print("Loading data (AQ, EQ, VQ, IQ)...")
    explicit_ans_mention = pd.read_json(os.path.join(dataset_dir, "explicit_ans_mention.jsonl"), lines=True).set_index("question_id")  # AQ  # fmt: skip
    explicit_questions = pd.read_json(os.path.join(dataset_dir, "explicit_questions.jsonl"), lines=True).set_index("question_id")  # EQ  # fmt: skip
    visual_questions = pd.read_json(os.path.join(dataset_dir, "visual_questions.jsonl"), lines=True).set_index("question_id")  # VQ  # fmt: skip
    implicit_questions = pd.read_json(os.path.join(dataset_dir, "implicit_questions.jsonl"), lines=True).set_index("question_id")  # fmt: skip

    if dataset_name == "wikitq":
        table_metadata = pd.read_csv(os.path.join(dataset_dir, "table-metadata.tsv"), sep="\t").set_index("contextId")
    else:
        table_metadata = None

    tables = pd.read_json(os.path.join(dataset_dir, "tables.jsonl"), lines=True).set_index("table_id")
    image_id_to_image_path = json.load(open(os.path.join(data_dir, "image_id_to_image_path.json")))

    mm_passages = None
    text_passages = None
    if dataset_name == "hybridqa":
        print("Loading HybridQA passage data...")
        with open(os.path.join(data_dir, "HybridQA/mm_passages.json"), "r") as f:
            mm_passages = json.load(f)
        with open(os.path.join(data_dir, "HybridQA/text_passages.json"), "r") as f:
            text_passages = json.load(f)

    return (
        explicit_ans_mention,
        explicit_questions,
        visual_questions,
        implicit_questions,
        table_metadata,
        tables,
        image_id_to_image_path,
        mm_passages,
        text_passages,
    )


def create_dataset(
    explicit_ans_mention,
    explicit_questions,
    visual_questions,
    implicit_questions,
    metadata_df,
    tables_df,
    image_dir,
    image_id_to_image_path,
    data_dir,
    dataset_name,
    mm_passages,
    text_passages,
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
            "passages": datasets.Sequence(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "type": datasets.Value("string"),  # 'mm' or 'text'
                    "linked_cell": datasets.Sequence(datasets.Value("int32"), length=2),  # [row, col]
                }
            ),
        }
    )

    # Define the generator function here so that it has all relevant dataframes. We then only shard the "df" in the gen_kwargs.
    def mmtabqa_to_hf_dataset_generator(questions_df):
        """
        Generator that yields structured dictionaries for the Hugging Face dataset.
        Each yielded item represents one question and its multimodal table.
        """

        for q_id, row in questions_df.iterrows():
            table_id = row["table_context"]

            table_info = tables_df.loc[table_id]
            table_array = table_info["table_array"]

            # Get table metadata
            if dataset_name == "wikitq":
                wtq_table_metadata_id = table_id.replace("WTQ/", "")
                metadata = metadata_df.loc[wtq_table_metadata_id]
                page_title = metadata["title"]
                section_title = metadata["headers"]
            elif dataset_name == "wikisql":
                page_title = table_info["page_title"]
                section_title = table_info["section_title"]
            elif dataset_name == "fetaqa":
                page_title = table_info["table_page_title"]
                section_title = table_info["table_section_title"]
            elif dataset_name == "hybridqa":
                page_title = ""
                section_title = table_info.get("url", "").split("/")[-1].replace("_", " ")

            # Get headers from the first table_array row
            headers = table_array[0]

            processed_rows = []
            for table_row in table_array[1:]:  # Skip first row as it is the header
                new_row = []
                # new_row should have text and the image path like this:
                # [
                #   {'type': 'text', 'content': 'Here is the first part of the sequence.'},
                #   {'type': 'image', 'content': '/path/to/your/image1.jpg'},
                #   ...
                # ]
                for cell_content in table_row:
                    match = IMG_TAG_PATTERN.search(cell_content)
                    if match:
                        # Case 1: Cell contains an image tag
                        image_id = f"{{IMG-{{{match.group(1)}}}}}"
                        image_path = image_id_to_image_path.get(image_id)
                        assert image_path, f"Image ID not in map: {image_id}"
                        full_image_path = os.path.join(image_dir, image_path)
                        assert os.path.exists(full_image_path), f"Image not found in image_dir: {image_path}"
                        new_row.append({"type": "image", "content": image_path})
                    else:
                        # Case 2: Otherwise, append the text content
                        new_row.append({"type": "text", "content": cell_content})
                processed_rows.append(new_row)

            # Process passages for HybridQA
            processed_passages = []
            if dataset_name == "hybridqa" and "cells_to_link" in table_info:
                for link_info in table_info["cells_to_link"]:
                    row_idx, col_idx, linked_cells = link_info
                    # The passage ID might be a single string or a list/tuple
                    passage_ids_to_process = (
                        linked_cells if isinstance(linked_cells, (list, tuple)) else [linked_cells]
                    )

                    for p_id in passage_ids_to_process:
                        passage_text = ""
                        if p_id in mm_passages:
                            passage_text = mm_passages[p_id]
                            passage_source_type = "mm"
                        elif p_id in text_passages:
                            passage_text = text_passages[p_id]
                            passage_source_type = "text"

                        if passage_text:
                            processed_passages.append(
                                {
                                    "id": p_id,
                                    "text": passage_text,
                                    "type": passage_source_type,
                                    "linked_cell": [row_idx, col_idx],  # [row, col]
                                }
                            )

            if dataset_name == "wikitq":
                answer_text = row["answer"].split("|")
            elif dataset_name == "wikisql":
                answer_text = row["answer"]
            elif dataset_name == "fetaqa":
                answer_text = [row["answer"]]
            elif dataset_name == "hybridqa":
                answer_text = [row["answer-text"]]

            yield {
                "id": q_id,
                "question": row["question"],
                "answer_text": answer_text,
                "table_id": table_id,
                "table": {
                    "page_title": page_title,
                    "section_title": section_title,
                    "header": headers,
                    "rows": processed_rows,
                },
                "passages": processed_passages,
            }

    # Create datasets for EQ, AQ, ...
    for df_name, df in [
        ("explicit_ans_mention", explicit_ans_mention),
        ("explicit_questions", explicit_questions),
        ("visual_questions", visual_questions),
        ("implicit_questions", implicit_questions),
    ]:
        print(f"Creating Hugging Face Dataset for {df_name} ...")

        # Create dataset
        hf_dataset = datasets.Dataset.from_generator(
            generator=mmtabqa_to_hf_dataset_generator,
            features=features,
            split=datasets.Split.TEST,
            gen_kwargs={"questions_df": df},
        )

        # save to disk
        print("Dataset created successfully!")
        print(hf_dataset)
        print("\nExample entry:")
        print(hf_dataset[0])
        print("\nSaving dataset to disk...")
        save_dir = os.path.join(data_dir, "converted_to_hf_dataset")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        hf_dataset.save_to_disk(os.path.join(save_dir, f"mmtabqa_{dataset_name}_hf_dataset_{df_name}"))

    print("Done.")


if __name__ == "__main__":
    main()
