# Python script to transform HTML tables into a Hugging Face dataset
import json
import os
import pathlib
from typing import List, Tuple, Optional

import datasets
print(datasets.__version__)
from bs4 import BeautifulSoup

from utils.runtime_env import load_repo_dotenv

load_repo_dotenv()

BASE_DATASET = os.getenv("MMT_BENCH_BASE_PATH")
if not BASE_DATASET:
    raise ValueError("MMT_BENCH_BASE_PATH must be set before creating the MMTabReal dataset.")
QUESTIONS_DIR = os.path.join(BASE_DATASET, "Questions")
METADATA_DIR = os.path.join(BASE_DATASET, "Question-Metadata")
TABLES_DIR = os.path.join(BASE_DATASET, "all")
OUTPUT_DIR = os.path.join(BASE_DATASET, "hf_dataset")


Question_Type_Map = {
    "Explicit Question": "EQ",
    "Visual-Based Question": "VQ",
    "Answer-Mention Question": "AQ",
    "Implicit Question": "IQ",
}


def load_json(path):
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_table_from_html(html_path: str, table_name: str) -> Tuple[Optional[List[str]], Optional[List[List[dict]]]]:
    """
    Extracts first table from HTML file.
    - First row becomes header
    - Supports text, images, and mixed cells
    - Supports multiple images per cell
    - Removes hyperlinks
    
    Returns:
        Tuple of (headers, rows) or (None, None) if table not found
    """
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
    except Exception as e:
        print(f"Error reading HTML file {table_name}: {e}")
        return None, None

    table = soup.find("table")
    if not table:
        print(f"No <table> tag found in {table_name}")
        return None, None

    all_rows = table.find_all("tr")
    if not all_rows:
        print(f"No <tr> rows found in table for {table_name}")
        return None, None

    # Find first non-empty row to use as header
    header_row_idx = None
    headers = []
    
    for idx, row in enumerate(all_rows):
        cells = row.find_all(["th", "td"])
        if cells:
            # Found a row with cells - use it as header
            for cell in cells:
                # Remove hyperlinks
                for a in cell.find_all("a"):
                    a.unwrap()
                headers.append(cell.get_text(" ", strip=True))
            header_row_idx = idx
            break
    
    if not headers or header_row_idx is None:
        print(f"No cells found in any row for {table_name}")
        return None, None

    # Extract data rows (all rows after header)
    rows = []
    for tr in all_rows[header_row_idx + 1:]:
        row = []
        for cell in tr.find_all(["td", "th"]):
            # Remove hyperlinks
            for a in cell.find_all("a"):
                a.unwrap()

            # Collect images
            images = []
            for img in cell.find_all("img"):
                src = img.get("src")
                if src:
                    # Convert relative path to absolute path
                    image_path = os.path.join(os.path.dirname(html_path), src)
                    images.append(image_path)

            # Remove images before text extraction
            for img in cell.find_all("img"):
                img.extract()

            text = cell.get_text(" ", strip=True)

            # Build cell based on content type
            if images and text:
                row.append({
                    "type": "image/text",
                    "text_content": text,
                    "image_content": images
                })
            elif images:
                row.append({
                    "type": "image",
                    "content": images
                })
            else:
                row.append({
                    "type": "text",
                    "content": text
                })

        if row:
            rows.append(row)

    # Return even if no data rows - some tables might only have headers
    return headers, rows


def normalize_qa(item: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract question and answer from a question dictionary.
    Handles various key naming conventions.
    """
    question = None
    answer = None

    for k, v in item.items():
        k_lower = k.lower()
        if "question" in k_lower and not question:
            question = str(v)
        elif "answer" in k_lower and not answer:
            answer = str(v)

    return question, answer


def get_question_type(question_metadata: dict, question_number: int, table_name: str) -> Optional[str]:
    """
    Extract question type from metadata with various fallback key patterns.
    """
    question_type_attr_names = [
        "question Type",
        "Question Type",
        f"Question {question_number + 1} Type",
        f"question {question_number + 1} Type",
        f"Question{question_number + 1} Type",
        f"Question {question_number} Type",
        f"Question{question_number} Type",
        f"Question {question_number + 2} Type",
        f"question {question_number + 1} type",
        "question type",
        "Questio6 Type",
    ]
    
    for attr_name in question_type_attr_names:
        if attr_name in question_metadata:
            raw_type = question_metadata[attr_name]
            return Question_Type_Map.get(raw_type, raw_type)
    
    print(f"Warning: No question type found for question {question_number} in table {table_name}")
    return None


def create_dataset():
    """
    Main function to create Hugging Face dataset from HTML tables and JSON questions.
    """
    # Define dataset features
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
                "rows": datasets.Sequence(
                    datasets.Sequence(
                        {
                            "type": datasets.Value("string"),
                            "content": datasets.Value("string"),
                        }
                    )
                ),
            },
        }
    )

    def dataset_generator(target_question_type: Optional[str] = None):
        """
        Generator that yields structured dictionaries for the Hugging Face dataset.
        
        Args:
            target_question_type: If specified, only yield questions of this type (EQ, VQ, AQ, IQ)
        """
        skipped_count = 0
        total_count = 0
        processed_count = 0

        # Iterate through all question files
        for file in os.listdir(QUESTIONS_DIR):
            if not file.endswith(".json"):
                continue

            table_name = os.path.splitext(file)[0]
            total_count += 1

            # Paths
            questions_path = os.path.join(QUESTIONS_DIR, file)
            html_path = os.path.join(TABLES_DIR, table_name, f"{table_name}.html")
            metadata_path = os.path.join(METADATA_DIR, file)

            # Check if HTML file exists
            if not os.path.exists(html_path):
                print(f"Skipping {table_name}: HTML file not found")
                skipped_count += 1
                continue

            # Load questions
            try:
                questions = load_json(questions_path)
            except Exception as e:
                print(f"Error loading questions for {table_name}: {e}")
                skipped_count += 1
                continue

            # Load metadata (optional)
            metadata = None
            if os.path.exists(metadata_path):
                try:
                    metadata = load_json(metadata_path)
                except Exception as e:
                    print(f"Warning: Could not load metadata for {table_name}: {e}")

            # Extract table from HTML
            headers, rows = extract_table_from_html(html_path, table_name)
            if headers is None:
                print(f"Skipping {table_name}: Could not extract table from HTML")
                skipped_count += 1
                continue
            
            # Allow tables with only headers or no rows - they may still be valid
            if not headers:
                print(f"Skipping {table_name}: No headers found")
                skipped_count += 1
                continue

            # Process each question
            for question_idx, question_item in enumerate(questions):
                question, answer = normalize_qa(question_item)
                
                if not question or not answer:
                    continue

                # Check question type if metadata exists and filtering is requested
                if target_question_type and metadata:
                    if question_idx < len(metadata):
                        q_type = get_question_type(metadata[question_idx], question_idx, table_name)
                        if q_type != target_question_type:
                            continue
                    else:
                        # No metadata for this question, skip if filtering
                        continue

                # Convert rows to the format expected by the dataset
                # Flatten complex cells to simple type/content structure
                formatted_rows = []
                for row in rows:
                    formatted_row = []
                    for cell in row:
                        if cell["type"] == "image":
                            # Multiple images in one cell - use first or concatenate paths
                            images = cell["content"]
                            if isinstance(images, list):
                                content = "|".join(images)  # Join multiple image paths
                            else:
                                content = images
                            formatted_row.append({"type": "image", "content": content})
                        elif cell["type"] == "image/text":
                            # Mixed cell - store as special type with combined content
                            text_part = cell.get("text_content", "")
                            images_part = cell.get("image_content", [])
                            if isinstance(images_part, list):
                                images_str = "|".join(images_part)
                            else:
                                images_str = images_part
                            content = f"TEXT:{text_part}|IMAGES:{images_str}"
                            formatted_row.append({"type": "image/text", "content": content})
                        else:  # text
                            formatted_row.append({"type": "text", "content": str(cell.get("content", ""))})
                    formatted_rows.append(formatted_row)

                yield {
                    "id": f"{table_name}_{question_idx + 1}",
                    "question": question,
                    "answer_text": [answer],
                    "table_id": table_name,
                    "table": {
                        "page_title": table_name.replace("_", " "),
                        "section_title": "",
                        "header": headers,
                        "rows": formatted_rows,
                    },
                }
                processed_count += 1

        print(f"\nDataset creation summary:")
        print(f"  Total tables found: {total_count}")
        print(f"  Tables skipped: {skipped_count}")
        print(f"  Questions processed: {processed_count}")

    # Create datasets for each question type
    question_types = ["EQ", "VQ", "AQ", "IQ"]
    
    for q_type in question_types:
        print(f"\n{'='*60}")
        print(f"Creating dataset for question type: {q_type}")
        print(f"{'='*60}")
        
        try:
            hf_dataset = datasets.Dataset.from_generator(
                generator=dataset_generator,
                features=features,
                split=datasets.Split.TEST,
                gen_kwargs={"target_question_type": q_type},
            )

            print(f"\nDataset created successfully for {q_type}!")
            print(f"Total examples: {len(hf_dataset)}")
            
            if len(hf_dataset) > 0:
                print(f"\nExample entry:")
                print(hf_dataset[0])
                
                # Save to disk
                save_path = os.path.join(OUTPUT_DIR, f"mmtabreal_{q_type}")
                print(f"\nSaving dataset to: {save_path}")
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                hf_dataset.save_to_disk(save_path)
                print(f"✓ Saved {q_type} dataset")
            else:
                print(f"⚠ No examples found for {q_type}")
                
        except Exception as e:
            print(f"✗ Error creating dataset for {q_type}: {e}")
            import traceback
            traceback.print_exc()

    # Also create a combined dataset with all question types
    print(f"\n{'='*60}")
    print(f"Creating combined dataset (all question types)")
    print(f"{'='*60}")
    
    try:
        hf_dataset_all = datasets.Dataset.from_generator(
            generator=dataset_generator,
            features=features,
            split=datasets.Split.TEST,
            gen_kwargs={"target_question_type": None},  # No filtering
        )

        print(f"\nCombined dataset created successfully!")
        print(f"Total examples: {len(hf_dataset_all)}")
        
        if len(hf_dataset_all) > 0:
            print(f"\nExample entry:")
            print(hf_dataset_all[0])
            
            # Save to disk
            save_path = os.path.join(OUTPUT_DIR, "mmtabreal_all")
            print(f"\nSaving combined dataset to: {save_path}")
            hf_dataset_all.save_to_disk(save_path)
            print(f"✓ Saved combined dataset")
            
    except Exception as e:
        print(f"✗ Error creating combined dataset: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print("Dataset creation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    create_dataset()
