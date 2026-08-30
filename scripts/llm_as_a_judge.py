from datetime import datetime
from tqdm import tqdm

from generation.generator import Generator


def evaluate_AQ_via_LLM_as_a_judge(
    outputs, generator: Generator, mode, readme_file, mmtabqa_sub_dataset, mmtabqa_dataset_split, results_file
):
    """
    Use LLM-as-a-judge for AQ questions.
    """

    if mmtabqa_sub_dataset == "fetaqa" and mode == "hstar":
        mode = "interleaved_reasoner"

    accuracy, acc, total_samples = _evaluate_AQ_via_LLM_as_a_judge_CAPTR(outputs, generator, mode)

    print("\n\n\n----- LLM-as-a-judge Evaluation Results -----")
    if mmtabqa_sub_dataset is not None and mmtabqa_dataset_split is not None:
        print(f"----- MMTabQA Sub-Dataset: {mmtabqa_sub_dataset} / {mmtabqa_dataset_split} -----")

    print(f"Correct Samples: {acc}; Total Samples: {total_samples}")
    print(f"Accuracy: {accuracy:.2f}%")

    # 6. Add the accuracy to the README file
    if readme_file is not None:
        with open(readme_file, "a") as f:
            # Determine method name and model from results_file path
            # results_file format: results/{model_name}/{dataset}_{split}_{mode}.json
            parts = results_file.split('/')
            model_name = parts[1] if len(parts) > 1 else "unknown"
            
            # Use clean format for sparsevlm mode
            if "sparsevlm" in mode or "sparsevlm" in results_file:
                method_name = "SparseVLM (LLM-as-judge)"
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {model_name}: {mmtabqa_sub_dataset} - {mmtabqa_dataset_split} - {method_name} - {accuracy:.2f}%\n")
            else:
                # Original format for other modes
                if mmtabqa_sub_dataset is not None and mmtabqa_dataset_split is not None:
                    f.write(f"\n\n{mode} - {mmtabqa_sub_dataset} - {mmtabqa_dataset_split} (LLM-as-a-judge)\n")
                else:
                    f.write(f"\n\n{mode} - LLM-as-a-judge\n")
                f.write(f"File: {results_file}\n")
                f.write(f"Correct Samples: {acc}; Total Samples: {total_samples}\n")
                f.write(f"Accuracy: {accuracy:.2f}%\n")


def _evaluate_AQ_via_LLM_as_a_judge_CAPTR(outputs, generator: Generator, mode):
    batch_requests = []
    for i in tqdm(outputs, desc="Preparing LLM-as-a-judge evaluation"):
        output = outputs[str(i)]

        # 1. Extract predicted answer, gold answer, and question
        if mode == "hstar":
            if "the answer is: " in output["generations"][0]:
                pred_answer_split = output["generations"][0].split("the answer is: ")[1]
                pred_answer = pred_answer_split.split('"')[1:2]
                if len(pred_answer) == 0:
                    pred_answer = pred_answer_split
            else:
                pred_answer = output["generations"][0]

            gold_answer = output["ori_data_item"]["answer_text"]
            question = output["ori_data_item"]["question"]
            page_title = output["ori_data_item"]["table_with_metadata"]["page_title"]
            section_title = output["ori_data_item"]["table_with_metadata"]["section_title"]

        elif mode == "mmtabqa_baseline":
            assert len(output["generations"]) == 1, (
                f"MMTabQA baseline should only have one generation, but got: {len(output['generations'])} many generations."
            )

            # Some original baseline prompts had wrong spaces. We fixed this but for compatibility, we check for both.
            if "Step 2 :" in output["generations"][0]:
                pred_answer = output["generations"][0].split("Step 2 :")[-1].strip()
            else:
                pred_answer = output["generations"][0].split("Step 2:")[-1].strip()
            pred_answer = [pred_answer]  # the evaluator expects a list
            gold_answer = output["answer_text"]
            question = output["question"]
            page_title = output["page_title"]
            section_title = output["section_title"]

        elif mode == "interleaved_reasoner":
            if "Step 2 :" in output["generations"][0]:
                pred_answer = output["generations"][0].split("Step 2 :")[-1].strip()
            else:
                pred_answer = output["generations"][0].split("Step 2:")[-1].strip()
            pred_answer = [pred_answer]  # the evaluator expects a list
            gold_answer = output["ori_data_item"]["answer_text"]
            question = output["ori_data_item"]["question"]
            page_title = output["ori_data_item"]["table_with_metadata"]["page_title"]
            section_title = output["ori_data_item"]["table_with_metadata"]["section_title"]

        else:
            raise ValueError(f"Invalid mode: {mode}")

        # 2. Add the request to batch_requests
        batch_requests.append(
            {
                "oracle_answer_list": gold_answer,
                "our_answer": pred_answer[0],
                "question": question,
                "page_title": page_title,
                "section_title": section_title,
            }
        )

    # 4. Evaluate using LLM-as-a-judge
    print(f"Evaluating {len(batch_requests)} examples using LLM-as-a-judge...")
    print(f"first batch_request: {batch_requests[0]}")
    scores = generator.evaluate_llm_as_a_judge(batch_requests)

    # 5. Calculate accuracy
    acc = sum(scores)
    total_samples = len(outputs)
    accuracy = 100 * acc / total_samples
    return accuracy, acc, total_samples
