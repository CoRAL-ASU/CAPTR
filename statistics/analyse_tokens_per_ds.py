import tiktoken
from tqdm import tqdm

from utils.utils import load_data_split


def count_tokens(text):
    """Count tokens using GPT tokenizer"""
    encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoder.encode(text))


def analyze_dataset_tokens(dataset_name):
    """Analyze token counts for a dataset"""
    total_tokens = 0
    max_tokens = 0
    num_examples = 0

    # Load dataset
    try:
        if dataset_name == "wikitq":
            dataset = load_data_split(dataset_name, "test")
        elif dataset_name == "tab_fact":
            dataset = load_data_split(dataset_name, "test")
    except Exception as e:
        print(f"Error loading {dataset_name}: {str(e)}")
        return None

    # Process each example
    for example in tqdm(dataset, desc=f"Processing {dataset_name}"):
        try:
            # Count tokens based on dataset structure
            if dataset_name == "wikitq":
                question_tokens = count_tokens(example["question"])
                table_tokens = count_tokens(
                    f"Title: {example['table']['page_title']}\n"
                    + f"Headers: {', '.join(example['table']['header'])}\n"
                    + f"Rows: {str(example['table']['rows'])}"
                )
                example_tokens = question_tokens + table_tokens

            elif dataset_name == "tab_fact":
                statement_tokens = count_tokens(example["statement"])
                table_tokens = count_tokens(
                    f"Caption: {example['table']['caption']}\n"
                    + f"Headers: {', '.join(example['table']['header'])}\n"
                    + f"Rows: {str(example['table']['rows'])}"
                )
                example_tokens = statement_tokens + table_tokens

            elif dataset_name == "fetaqa":
                question_tokens = count_tokens(example["question"])
                table_tokens = count_tokens(
                    f"Title: {example['table']['page_title']}\n"
                    + f"Headers: {', '.join(example['table']['header'])}\n"
                    + f"Rows: {str(example['table']['rows'])}"
                )
                example_tokens = question_tokens + table_tokens

            total_tokens += example_tokens
            max_tokens = max(max_tokens, example_tokens)
            num_examples += 1

        except Exception as e:
            print(f"Error processing example in {dataset_name}: {str(e)}")
            continue

    return {
        "dataset": dataset_name,
        "total_tokens": total_tokens,
        "avg_tokens": total_tokens / num_examples if num_examples > 0 else 0,
        "max_tokens": max_tokens,
        "num_examples": num_examples,
    }


def main():
    datasets_to_analyze = ["wikitq", "tab_fact"]
    results = []

    input_cost = 0.15 / 1_000_000  # $0.15 per 1M tokens
    output_cost = 0.6 / 1_000_000  # $0.6 per 1M tokens
    output_tokens = 300  # we roughly have 300 output tokens for each example

    for dataset in datasets_to_analyze:
        stats = analyze_dataset_tokens(dataset)
        if stats:
            results.append(stats)

        print(f"{dataset} has #{stats['num_examples']} examples with a total of {stats['total_tokens']} tokens")

        # Step 1: Col Select SQL
        prompt_file = "prompts/col_select_sql.txt"
        with open(prompt_file, "r") as f:
            prompt_text = f.read()
        prompt_tokens = count_tokens(prompt_text)
        print("\nStep 1: Col Select SQL")
        print(f"Prompt tokens: {prompt_tokens:,}")
        input_step_1_tokens = prompt_tokens * stats["num_examples"] + stats["total_tokens"]
        output_step_1_tokens = stats["num_examples"] * output_tokens
        step_1_cost = input_step_1_tokens * input_cost + output_step_1_tokens * output_cost

        step_1_per_example = input_step_1_tokens / stats["num_examples"]
        print(f"Input tokens: {input_step_1_tokens:,}    per example: {step_1_per_example:,}")
        print(f"Output tokens: {output_step_1_tokens:,}")
        print(f"Step 1 cost: ${step_1_cost:.2f}")

        # Step 2: Col Select Text
        prompt_file = "prompts/col_select_text.txt"
        with open(prompt_file, "r") as f:
            prompt_text = f.read()
        prompt_tokens = count_tokens(prompt_text)
        print("\nStep 2: Col Select Text")
        print(f"Prompt tokens: {prompt_tokens:,}")
        input_step_2_tokens = prompt_tokens * stats["num_examples"] + stats["total_tokens"]
        output_step_2_tokens = stats["num_examples"] * output_tokens
        step_2_cost = input_step_2_tokens * input_cost + output_step_2_tokens * output_cost

        step_2_per_example = input_step_2_tokens / stats["num_examples"]
        print(f"Input tokens: {input_step_2_tokens:,}    per example: {step_2_per_example:,}")
        print(f"Output tokens: {output_step_2_tokens:,}")
        print(f"Step 2 cost: ${step_2_cost:.2f}")

        # Step 3: Row Select SQL
        prompt_file = "prompts/row_select_sql.txt"
        with open(prompt_file, "r") as f:
            prompt_text = f.read()
        prompt_tokens = count_tokens(prompt_text)
        print("\nStep 3: Row Select SQL")
        print(f"Prompt tokens: {prompt_tokens:,}")
        input_step_3_tokens = prompt_tokens * stats["num_examples"] + stats["total_tokens"]
        output_step_3_tokens = stats["num_examples"] * output_tokens
        step_3_cost = input_step_3_tokens * input_cost + output_step_3_tokens * output_cost

        step_3_per_example = input_step_3_tokens / stats["num_examples"]
        print(f"Input tokens: {input_step_3_tokens:,}    per example: {step_3_per_example:,}")
        print(f"Output tokens: {output_step_3_tokens:,}")
        print(f"Step 3 cost: ${step_3_cost:.2f}")

        # Step 4: Row Select Text
        prompt_file = "prompts/row_select_text.txt"
        with open(prompt_file, "r") as f:
            prompt_text = f.read()
        prompt_tokens = count_tokens(prompt_text)
        print("\nStep 4: Row Select Text")
        print(f"Prompt tokens: {prompt_tokens:,}")
        input_step_4_tokens = prompt_tokens * stats["num_examples"] + stats["total_tokens"]
        output_step_4_tokens = stats["num_examples"] * output_tokens
        step_4_cost = input_step_4_tokens * input_cost + output_step_4_tokens * output_cost

        step_4_per_example = input_step_4_tokens / stats["num_examples"]
        print(f"Input tokens: {input_step_4_tokens:,}    per example: {step_4_per_example:,}")
        print(f"Output tokens: {output_step_4_tokens:,}")
        print(f"Step 4 cost: ${step_4_cost:.2f}")

        # Step 5: Reason Text
        prompt_file = "prompts/sql_reason_wtq.txt"
        with open(prompt_file, "r") as f:
            prompt_text = f.read()
        prompt_tokens = count_tokens(prompt_text)
        print("\nStep 5: Reason Text")
        print(f"Prompt tokens: {prompt_tokens:,}")
        input_step_5_tokens = prompt_tokens * stats["num_examples"] + stats["total_tokens"]
        output_step_5_tokens = stats["num_examples"] * output_tokens
        step_5_cost = input_step_5_tokens * input_cost + output_step_5_tokens * output_cost

        step_5_per_example = input_step_5_tokens / stats["num_examples"]
        print(f"Input tokens: {input_step_5_tokens:,}    per example: {step_5_per_example:,}")
        print(f"Output tokens: {output_step_5_tokens:,}")
        print(f"Step 5 cost: ${step_5_cost:.2f}")

        # Step 6: Reason SQL
        prompt_file = "prompts/sql_reason_wtq.txt"
        with open(prompt_file, "r") as f:
            prompt_text = f.read()
        prompt_tokens = count_tokens(prompt_text)
        print("\nStep 6: Reason SQL")
        print(f"Prompt tokens: {prompt_tokens:,}")
        input_step_6_tokens = prompt_tokens * stats["num_examples"] + stats["total_tokens"]
        output_step_6_tokens = stats["num_examples"] * output_tokens
        step_6_cost = input_step_6_tokens * input_cost + output_step_6_tokens * output_cost

        step_6_per_example = input_step_6_tokens / stats["num_examples"]
        print(f"Input tokens: {input_step_6_tokens:,}    per example: {step_6_per_example:,}")
        print(f"Output tokens: {output_step_6_tokens:,}")
        print(f"Step 6 cost: ${step_6_cost:.2f}")

        # Total cost
        total_cost = step_1_cost + step_2_cost + step_3_cost + step_4_cost + step_5_cost + step_6_cost
        print(f"\nTotal cost: ${total_cost:.2f}")

        print(
            f"total tokens per example: {step_1_per_example + step_2_per_example + step_3_per_example + step_4_per_example + step_5_per_example + step_6_per_example}"
        )


if __name__ == "__main__":
    main()
