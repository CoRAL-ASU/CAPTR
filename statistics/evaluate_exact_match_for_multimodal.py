from typing import List

from baselines.baselines_utils import DATASET_PATHS
from mmtabqa.load_mmtabqa_utils import load_mmtabqa_dataset


def how_often_is_label_not_in_table(dataset):
    count = 0
    for example in dataset:
        gold_answers: List[str] = example["answer_text"]
        table_rows = [row["content"] for row in example["table"]["rows"]]
        table_rows_flattened = [item for sublist in table_rows for item in sublist]
        if not any(gold_answer in table_rows_flattened for gold_answer in gold_answers):
            count += 1

    return count


for dataset_name, dataset_splits in DATASET_PATHS.items():
    total_count = 0
    total_length = 0
    for split_name, dataset_path in dataset_splits.items():
        print(f"Processing {dataset_name}-{split_name} dataset...")

        mmtabqa_dataset = load_mmtabqa_dataset(dataset_path, num_proc=10, partial_input_baseline=True)

        count = how_often_is_label_not_in_table(mmtabqa_dataset)
        print(
            f"How often is the label not in the table: {count} from {len(mmtabqa_dataset)}, i.e.: {count / len(mmtabqa_dataset):.2%}"
        )

        total_count += count
        total_length += len(mmtabqa_dataset)

    print(f"\nTotal: {total_count} from {total_length}, i.e.: {total_count / total_length:.2%}%")
