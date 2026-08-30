from tqdm import tqdm

import datasets


def main():
    # Base directory where the HF datasets are stored, as you specified.
    base_dir = "CAPTR/datasets/MMTabQA_Dataset/converted_to_hf_dataset/"

    dataset_paths = {
        "AQ-WikiTQ": "mmtabqa_wikitq_hf_dataset_explicit_ans_mention",
        "EQ-WikiTQ": "mmtabqa_wikitq_hf_dataset_explicit_questions",
        "VQ-WikiTQ": "mmtabqa_wikitq_hf_dataset_visual_questions",
        "IQ-WikiTQ": "mmtabqa_wikitq_hf_dataset_implicit_questions",
        "AQ-WikiSQL": "mmtabqa_wikisql_hf_dataset_explicit_ans_mention",
        "EQ-WikiSQL": "mmtabqa_wikisql_hf_dataset_explicit_questions",
        "VQ-WikiSQL": "mmtabqa_wikisql_hf_dataset_visual_questions",
        "IQ-WikiSQL": "mmtabqa_wikisql_hf_dataset_implicit_questions",
        "AQ-FetaQA": "mmtabqa_fetaqa_hf_dataset_explicit_ans_mention",
        "EQ-FetaQA": "mmtabqa_fetaqa_hf_dataset_explicit_questions",
        "VQ-FetaQA": "mmtabqa_fetaqa_hf_dataset_visual_questions",
        "IQ-FetaQA": "mmtabqa_fetaqa_hf_dataset_implicit_questions",
    }

    max_images = 0
    example_with_max_images = None
    dataset_with_max_images = None

    for dataset_name, dataset_path in dataset_paths.items():
        print(f"Processing dataset: {dataset_name}")
        dataset = datasets.load_from_disk(base_dir + dataset_path)

        # Iterate through each example in the dataset
        for example in tqdm(dataset, desc="Analyzing examples"):
            image_count = 0
            # The table rows are a list of lists of dicts
            for row in example["table"]["rows"]:
                # row = {"type": [...], ...}
                for type_list in row["type"]:
                    if "image" in type_list:
                        image_count += 1

            if image_count > max_images:
                max_images = image_count
                example_with_max_images = example
                dataset_with_max_images = dataset_name

    print("\n=====================================")
    print(f"Maximum number of images in a single example: {max_images}")
    print(f"Example ID with max images: {example_with_max_images['id']}")
    print(f"Dataset with max images: {dataset_with_max_images}")
    print(f"\n\n\nexample: {example_with_max_images}")
    print("=====================================")


if __name__ == "__main__":
    main()
