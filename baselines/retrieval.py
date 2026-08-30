# The retrieval pipeline works as follows:
# 1. Execute this script to retrieve the relevant rows
#     - This script stores the retrieved rows as a dataset, i.e. we now have a new dataset that has question along with relevant rows.
# 2. Execute the interleaved reasoner script, i.e. call: `python interleaved_baseline.py --retrieval_output_dir <path_to_retrieved_rows_dataset>`

import argparse
import math
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import ColQwen2ForRetrieval, ColQwen2Processor

from baselines.baselines_utils import DATASET_PATHS, sample_dataset
from datasets import load_from_disk
from mmtabqa.load_mmtabqa_utils import load_mmtabqa_dataset

load_dotenv(".env")


class RowToImageConverter:
    """Converts a table row (text and images), including the header, into a PIL Image."""

    def __init__(self, font_size=14, padding=10, cell_padding=5, max_cell_width=200, max_image_height=100):
        self.font_size = font_size
        self.padding = padding
        self.cell_padding = cell_padding
        self.max_cell_width = max_cell_width
        self.max_image_height = max_image_height
        self.font = self._load_font(font_size)

    def _load_font(self, font_size):
        for font_name in ["DejaVuSans.ttf", "arial.ttf", "LiberationSans-Regular.ttf"]:
            try:
                return ImageFont.truetype(font_name, font_size)
            except IOError:
                continue
        return ImageFont.load_default()

    def _get_text_size(self, text, font):
        if not text:
            return 0, 0
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        try:
            width = int(math.ceil(draw.textlength(text, font=font)))
            # Use consistent height based on standard characters
            bbox = draw.textbbox((0, 0), "Ay", font=font)
            height = bbox[3] - bbox[1]
        except Exception:
            # Fallback
            bbox = draw.textbbox((0, 0), text, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
        return width, height

    def _wrap_text(self, text, max_width):
        lines = []
        current_line = ""
        words = str(text).split()

        if not words:
            return [""]

        for word in words:
            test_line = f"{current_line} {word}".strip()
            width, _ = self._get_text_size(test_line, self.font)
            if width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines

    def convert(self, header, row_data_input):
        row_data = list(row_data_input)

        if len(header) != len(row_data):
            while len(row_data) < len(header):
                row_data.append("")
            if len(row_data) > len(header):
                row_data = row_data[: len(header)]

        # 1. Calculate dims
        col_widths = []
        header_content = []
        row_content = []

        for i, (h, cell) in enumerate(zip(header, row_data)):
            # Header
            h_lines = self._wrap_text(h, self.max_cell_width)
            h_width = max((self._get_text_size(line, self.font)[0] for line in h_lines), default=0)
            header_content.append(h_lines)

            # Row
            if isinstance(cell, Image.Image):
                img = cell.copy()
                img_width, img_height = img.size
                if img_height > self.max_image_height:
                    ratio = self.max_image_height / img_height
                    img_width = int(img_width * ratio)
                    img_height = self.max_image_height
                    img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)

                r_width = img_width
                row_content.append(img)
            else:
                r_lines = self._wrap_text(cell, self.max_cell_width)
                r_width = max((self._get_text_size(line, self.font)[0] for line in r_lines), default=0)
                row_content.append(r_lines)

            col_widths.append(max(h_width, r_width) + 2 * self.cell_padding)

        # 2. Determine final row heights
        _, lh = self._get_text_size("Ay", self.font)
        line_height = lh + 2  # Add spacing

        header_height = 0
        row_height = 0

        for h_lines, r_item in zip(header_content, row_content):
            header_height = max(header_height, len(h_lines) * line_height)
            if isinstance(r_item, Image.Image):
                row_height = max(row_height, r_item.height)
            else:
                row_height = max(row_height, len(r_item) * line_height)

        header_height += 2 * self.cell_padding
        row_height += 2 * self.cell_padding

        # 3. Create and draw image
        total_width = sum(col_widths) + 2 * self.padding
        total_height = header_height + row_height + 2 * self.padding

        img = Image.new("RGB", (total_width, total_height), color="white")
        draw = ImageDraw.Draw(img)

        x_offset = self.padding
        for i in range(len(header)):
            width = col_widths[i]

            # Draw Header (Background and Text)
            y_header_start = self.padding
            draw.rectangle(
                [x_offset, y_header_start, x_offset + width, y_header_start + header_height], fill=(240, 240, 240)
            )

            y_text = y_header_start + self.cell_padding
            for line in header_content[i]:
                draw.text((x_offset + self.cell_padding, y_text), line, fill="black", font=self.font)
                y_text += line_height

            # Draw Row (Image or Text)
            y_row_start = self.padding + header_height
            r_item = row_content[i]

            if isinstance(r_item, Image.Image):
                y_img = y_row_start + (row_height - r_item.height) // 2
                x_img = x_offset + self.cell_padding
                try:
                    # Handle image pasting robustly (Transparency and Modes)
                    if r_item.mode == "RGBA" or (r_item.mode == "P" and "transparency" in r_item.info):
                        if r_item.mode != "RGBA":
                            r_item = r_item.convert("RGBA")
                        img.paste(r_item, (x_img, int(y_img)), mask=r_item)
                    else:
                        if r_item.mode != "RGB":
                            r_item = r_item.convert("RGB")
                        img.paste(r_item, (x_img, int(y_img)))

                except Exception:
                    draw.text(
                        (x_offset + self.cell_padding, y_row_start + self.cell_padding),
                        "[Image Error]",
                        fill="red",
                        font=self.font,
                    )
            else:
                y_text = y_row_start + self.cell_padding
                for line in r_item:
                    draw.text((x_offset + self.cell_padding, y_text), line, fill="black", font=self.font)
                    y_text += line_height

            # Draw borders
            draw.rectangle([x_offset, self.padding, x_offset + width, total_height - self.padding], outline="gray")

            x_offset += width

        return img


def load_colqwen_model():
    """Loads the ColQwen model and processor."""
    model_name = "vidore/colqwen2-v1.0-hf"
    model = ColQwen2ForRetrieval.from_pretrained(model_name, device_map="auto")
    model.eval()
    processor = ColQwen2Processor.from_pretrained(model_name)
    return model, processor


######################################################## First Retrieval Way: text-image-union retrieval ########################################################
def _row_to_text(header, row_content):
    """Converts a table row to text representation."""
    text_parts = []
    for h, cell in zip(header, row_content):
        # Only include text content, skip images
        if not isinstance(cell, Image.Image):
            text_parts.append(f"{h}: {cell}")
    return " | ".join(text_parts)


def row_union_image_retrieval(
    mmtabqa_dataset,
    colqwen_model,
    colqwen_processor,
    row_to_image_converter,
    top_k,
    batch_size,
):
    """
    Retrieve through union of text retrieval results and image retrieval results.

    1. Retrieve top-k relevant rows through text parts of table rows
    2. Retrieve top-k relevant rows through all images in the table
    3. Return the union of both sets
    """
    selected_rows = {}

    # Step 1: Pre-process examples
    all_batch_data_texts = []
    all_batch_data_images = []

    for i in range(len(mmtabqa_dataset)):
        example = mmtabqa_dataset[i]
        header = example["table"]["header"]
        rows = example["table"]["rows"]
        question = example["question"]

        row_texts = []
        valid_row_indices_text = []
        all_images_in_table = []
        image_to_row_idx = []

        for row_idx, row in enumerate(rows):
            text = _row_to_text(header, row["content"])
            row_texts.append(text)
            valid_row_indices_text.append(row_idx)
            images_in_row = [cell for cell in row["content"] if isinstance(cell, Image.Image)]
            for img in images_in_row:
                if img.width > 256 or img.height > 256:
                    if img.width > img.height:
                        new_width = 256
                        new_height = int(img.height * (256 / img.width))
                    else:
                        new_height = 256
                        new_width = int(img.width * (256 / img.height))
                    img = img.resize((new_width, new_height))
                all_images_in_table.append(img)
                image_to_row_idx.append(row_idx)

        all_batch_data_texts.append(
            {
                "example_idx": i,
                "question": question,
                "row_texts": row_texts,
                "valid_indices": valid_row_indices_text,
                "num_rows": len(row_texts),
            }
        )

        all_batch_data_images.append(
            {
                "example_idx": i,
                "question": question,
                "all_images": all_images_in_table,
                "image_to_row_idx": image_to_row_idx,
                "num_rows": len(all_images_in_table),
            }
        )

    # Calculate batches
    batches_start_end_texts = _get_batches_start_end(all_batch_data_texts, batch_size)
    batches_start_end_images = _get_batches_start_end(all_batch_data_images, batch_size)

    # Step 2: Text-based retrieval
    text_top_k_results = {}
    for batch_idx in tqdm(range(len(batches_start_end_texts)), desc="Text retrieval batches"):
        batch_start, batch_end = batches_start_end_texts[batch_idx]
        batch_data = all_batch_data_texts[batch_start:batch_end]

        # Collect all row texts and questions from all examples in batch
        all_row_texts = []
        all_questions = []
        example_to_rows_mapping = []  # Maps example to (start_idx, end_idx) in all_row_texts

        current_idx = 0
        for data in batch_data:
            num_rows = len(data["row_texts"])
            all_row_texts.extend(data["row_texts"])
            all_questions.append(data["question"])
            example_to_rows_mapping.append((current_idx, current_idx + num_rows))
            current_idx += num_rows

        # Embed all row texts at once
        inputs_text = colqwen_processor(text=all_row_texts, return_tensors="pt", padding=True).to(colqwen_model.device)
        with torch.no_grad():
            text_embeddings = colqwen_model(**inputs_text).embeddings

        # Embed all questions at once
        inputs_questions = colqwen_processor(text=all_questions, return_tensors="pt", padding=True).to(
            colqwen_model.device
        )
        with torch.no_grad():
            query_embeddings = colqwen_model(**inputs_questions).embeddings

        # For each example, compute scores and get top-k
        for idx, data in enumerate(batch_data):
            start_idx, end_idx = example_to_rows_mapping[idx]
            example_text_embeddings = text_embeddings[start_idx:end_idx]
            example_query_embedding = query_embeddings[idx : idx + 1]

            # Calculate scores
            scores_tensor = colqwen_processor.score_retrieval(example_query_embedding, example_text_embeddings)

            # Get top-k indices
            k = min(top_k, example_text_embeddings.shape[0])
            top_k_indices = torch.topk(scores_tensor[0], k, dim=0).indices.tolist()

            # Map to original row indices
            valid_indices = data["valid_indices"]
            top_k_rows_from_text = {valid_indices[idx] for idx in top_k_indices if idx < len(valid_indices)}
            text_top_k_results[data["example_idx"]] = top_k_rows_from_text

        # Clean up
        del text_embeddings, query_embeddings, inputs_text, inputs_questions
        torch.cuda.empty_cache()

    # Step 3: Image-based retrieval
    image_top_k_results = {}
    for batch_idx in tqdm(range(len(batches_start_end_images)), desc="Image retrieval batches"):
        batch_start, batch_end = batches_start_end_images[batch_idx]
        batch_data = all_batch_data_images[batch_start:batch_end]

        # Collect all images and questions from all examples in batch
        all_images = []
        all_questions = []
        example_to_images_mapping = []  # Maps example to (start_idx, end_idx) in all_images

        current_idx = 0
        for data in batch_data:
            num_images = len(data["all_images"])
            all_images.extend(data["all_images"])
            all_questions.append(data["question"])
            example_to_images_mapping.append((current_idx, current_idx + num_images))
            current_idx += num_images

        inputs_images = colqwen_processor(images=all_images, return_tensors="pt").to(colqwen_model.device)
        with torch.no_grad():
            image_embeddings = colqwen_model(**inputs_images).embeddings

        # Embed all questions at once
        inputs_questions = colqwen_processor(text=all_questions, return_tensors="pt", padding=True).to(
            colqwen_model.device
        )
        with torch.no_grad():
            query_embeddings = colqwen_model(**inputs_questions).embeddings

        # For each example, compute scores and get top-k
        for idx, data in enumerate(batch_data):
            start_idx, end_idx = example_to_images_mapping[idx]

            # Skip if no images for this example
            if start_idx == end_idx:
                image_top_k_results[data["example_idx"]] = set()
                continue

            example_image_embeddings = image_embeddings[start_idx:end_idx]
            example_query_embedding = query_embeddings[idx : idx + 1]

            # Calculate scores
            scores_tensor = colqwen_processor.score_retrieval(example_query_embedding, example_image_embeddings)

            # Get top-k images
            k = min(top_k, example_image_embeddings.shape[0])
            top_k_image_indices = torch.topk(scores_tensor[0], k, dim=0).indices.tolist()

            # Map top-k images back to rows
            image_to_row_idx = data["image_to_row_idx"]
            top_k_rows_from_images = {
                image_to_row_idx[idx] for idx in top_k_image_indices if idx < len(image_to_row_idx)
            }
            image_top_k_results[data["example_idx"]] = top_k_rows_from_images

        # Clean up
        del image_embeddings, query_embeddings, inputs_images, inputs_questions
        torch.cuda.empty_cache()

    # Step 4: Union of text and image results
    for i in range(len(mmtabqa_dataset)):
        text_rows = text_top_k_results.get(i, set())
        image_rows = image_top_k_results.get(i, set())
        union_rows = sorted(text_rows.union(image_rows))
        selected_rows[i] = union_rows

    return selected_rows


######################################################## Second Retrieval Way: Row-wise Retrieval, i.e. row->image -> embedding -> retrieval ########################################################
def _get_batches_start_end(all_batch_data, batch_size):
    batches_start_end = []
    current_start = 0
    current_row_count = 0

    for i, data in enumerate(all_batch_data):
        num_rows = data["num_rows"]

        # If adding this example would exceed max_rows_at_once, finalize current batch
        if current_row_count > 0 and current_row_count + num_rows > batch_size:
            batches_start_end.append((current_start, i))
            current_start = i
            current_row_count = 0

        current_row_count += num_rows

    if current_start < len(all_batch_data):
        batches_start_end.append((current_start, len(all_batch_data)))

    return batches_start_end


def row_wise_retrieval(
    mmtabqa_dataset,
    colqwen_model,
    colqwen_processor,
    row_to_image_converter,
    top_k,
    batch_size,
):
    """
    Retrieve through row-wise retrieval (by converting each row to an image).
    """
    selected_rows = {}

    # Step 1: Pre-process examples and convert rows to images
    all_batch_data = []
    for i in range(0, len(mmtabqa_dataset)):
        example = mmtabqa_dataset[i]
        try:
            header = example["table"]["header"]
            rows = example["table"]["rows"]
            question = example["question"]

            # Convert rows to images
            row_images = []
            valid_indices = []
            for row_idx, row in enumerate(rows):
                try:
                    img = row_to_image_converter.convert(header, row["content"])
                    row_images.append(img)
                    valid_indices.append(row_idx)
                except Exception:
                    continue

            if not row_images:
                selected_rows[i] = []
                continue

            all_batch_data.append(
                {
                    "example_idx": i,
                    "question": question,
                    "row_images": row_images,
                    "valid_indices": valid_indices,
                    "num_rows": len(row_images),
                }
            )
        except Exception as e:
            print(f"Error processing example {i} (ID: {example.get('id', 'N/A')}): {e}. Skipping.")
            selected_rows[i] = []

    # calcuate the batches: create a list of tuples of (start_idx, end_idx) so that the sum of rows is less than batch_size
    batches_start_end = _get_batches_start_end(all_batch_data, batch_size)

    for batch_idx in tqdm(
        range(0, len(batches_start_end)),
        desc="Completed batches",
    ):
        batch_start, batch_end = batches_start_end[batch_idx]
        batch_data = all_batch_data[batch_start:batch_end]

        # Step 2: Put all images and questions together in 1 list -> batch process this & later split back to original examples
        all_row_images = []
        all_questions = []
        example_to_rows_mapping = []  # Maps example to (start_idx, end_idx) in all_row_images

        current_idx = 0
        for data in batch_data:
            num_rows = len(data["row_images"])
            all_row_images.extend(data["row_images"])
            all_questions.append(data["question"])
            example_to_rows_mapping.append((current_idx, current_idx + num_rows))
            current_idx += num_rows

        # Step 3: Row Image Embeddings
        image_embeddings = []

        for i in range(0, len(all_row_images), batch_size):
            batch = all_row_images[i : i + batch_size]
            inputs_images = colqwen_processor(images=batch, return_tensors="pt").to(colqwen_model.device)
            with torch.no_grad():
                image_embeddings.append(colqwen_model(**inputs_images).embeddings)

        # flatten
        image_embeddings = torch.cat(image_embeddings, dim=0)

        # Step 4: Question Embeddings
        inputs_text = colqwen_processor(text=all_questions, return_tensors="pt", padding=True).to(colqwen_model.device)
        with torch.no_grad():
            query_embeddings = colqwen_model(**inputs_text).embeddings

        # Step 5: For each example, compute scores for its rows and get top-k
        for idx, data in enumerate(batch_data):
            try:
                start_idx, end_idx = example_to_rows_mapping[idx]
                example_image_embeddings = image_embeddings[start_idx:end_idx]
                example_query_embedding = query_embeddings[idx : idx + 1]

                # Calculate scores
                scores_tensor = colqwen_processor.score_retrieval(example_query_embedding, example_image_embeddings)

                # Get top-k indices
                k = min(top_k, example_image_embeddings.shape[0])
                top_k_indices = torch.topk(scores_tensor[0], k, dim=0).indices.tolist()

                # Map to original table indices and sort
                valid_indices = data["valid_indices"]
                retrieved_indices = sorted([valid_indices[idx] for idx in top_k_indices if idx < len(valid_indices)])
                selected_rows[data["example_idx"]] = retrieved_indices

            except Exception as e:
                print(f"Error computing scores for example {data['example_idx']}: {e}. Skipping.")
                selected_rows[data["example_idx"]] = []

        # Clean up:
        del image_embeddings, query_embeddings, inputs_images, inputs_text
        torch.cuda.empty_cache()
    return selected_rows


########################################################################################################################################################################


def main_rag_filter_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (small subset).")
    parser.add_argument(
        "--retrieval_fn",
        type=str,
        default="row_wise_retrieval",
        help="Retrieval function to use.",
        choices=["row_wise_retrieval", "row_union_image_retrieval"],
    )
    parser.add_argument(
        "--dataset_name", type=str, help="Which dataset to use.", choices=["HybridQA", "WikiTQ", "WikiSQL", "FetaQA"]
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=None,
        help="Number of examples to sample from each dataset split. If None, use all examples.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load ColQwen model
    colqwen_model, colqwen_processor = load_colqwen_model()
    row_to_image_converter = RowToImageConverter()

    retrieval_functions = {
        "row_wise_retrieval": {
            "retrieval_fn": row_wise_retrieval,
            "batch_size": 64,
        },
        "row_union_image_retrieval": {
            "retrieval_fn": row_union_image_retrieval,
            "batch_size": 64,
        },
    }

    filter_fn_name = args.retrieval_fn
    filter_fn = retrieval_functions[filter_fn_name]["retrieval_fn"]
    batch_size = retrieval_functions[filter_fn_name]["batch_size"]

    dataset_name = args.dataset_name
    dataset_splits = DATASET_PATHS[dataset_name]

    for split_name, dataset_path in dataset_splits.items():
        print(f"\n--- Processing {dataset_name}-{split_name} ---")

        output_path = Path(args.output_dir) / filter_fn_name / f"{dataset_name}_{split_name}_top{args.top_k}"
        if output_path.exists():
            print(f"⚠️ {output_path} already exists. Skipping.")
            continue

        # Load dataset with images for retrieval
        print("Loading dataset with images for retrieval...")
        dataset_with_images = load_mmtabqa_dataset(
            dataset_path, image_base_path=os.getenv("MMTABQA_IMAGE_BASE_PATH"), load_images=True
        )

        # Apply sampling if n_examples is specified
        if args.n_examples is not None:
            dataset_with_images = sample_dataset(dataset_with_images, n_examples=args.n_examples, seed=42)

        if args.debug:
            dataset_with_images = dataset_with_images.select(range(min(30, len(dataset_with_images))))

        # Retrieve selected row indices in shards to not run OOM
        current_index = 0
        shard_size = 500
        all_selected_rows = {}

        while current_index < len(dataset_with_images):
            end_index = min(current_index + shard_size, len(dataset_with_images))
            print(
                f"\nProcessing shard {current_index}:{end_index}. Total: {len(dataset_with_images)}. Progress: {current_index / len(dataset_with_images):.2%}"
            )

            shard_dataset = dataset_with_images.select(range(current_index, end_index))
            shard_selected_rows = filter_fn(
                shard_dataset,
                colqwen_model,
                colqwen_processor,
                row_to_image_converter,
                args.top_k,
                batch_size,
            )

            # Adjust indices to match global dataset indices
            for shard_idx, row_indices in shard_selected_rows.items():
                global_idx = current_index + shard_idx
                all_selected_rows[global_idx] = row_indices

            current_index = end_index

        # Load original dataset without images
        print("Loading original dataset without images...")
        dataset_original = load_from_disk(str(dataset_path))

        # Apply same sampling to original dataset
        if args.n_examples is not None:
            dataset_original = sample_dataset(dataset_original, n_examples=args.n_examples, seed=42)

        if args.debug:
            dataset_original = dataset_original.select(range(min(30, len(dataset_original))))

        # Apply row filtering transformation
        def filter_rows(example, idx):
            indices = all_selected_rows.get(idx, [])
            example["table"]["rows"] = [example["table"]["rows"][i] for i in indices if i < len(example["table"]["rows"])]  # fmt: skip
            example["retrieved_row_indices"] = indices
            return example

        print("Filtering dataset...")
        filtered_dataset = dataset_original.map(filter_rows, with_indices=True, desc="Filtering rows")

        # Save
        print(f"Saving to {output_path}...")
        filtered_dataset.save_to_disk(output_path)
        print("Done!")


if __name__ == "__main__":
    main_rag_filter_pipeline()
