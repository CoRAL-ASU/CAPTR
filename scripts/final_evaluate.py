import json
import traceback

import nltk
from tqdm import tqdm
nltk.download('punkt_tab')
from generation.generator import Generator
from scripts.llm_as_a_judge import evaluate_AQ_via_LLM_as_a_judge
from utils.evaluator import Evaluator


def main(
    results_file,
    mode,
    dataset=None,
    readme_file=None,
    mmtabqa_sub_dataset=None,
    mmtabqa_dataset_split=None,
    generator: Generator = None,
    use_llm_as_judge=False,
):
    print(f"loading results file: {results_file}")
    with open(results_file, "r") as f:
        outputs = json.load(f)

    print("Using default evaluation")
    if mmtabqa_sub_dataset == "fetaqa":
        print("Evaluating FetaQA")
        final_evaluate_fetaqa(outputs, mode, readme_file, mmtabqa_sub_dataset, mmtabqa_dataset_split, results_file)
    else:
        print("Evaluating default")
        if mode == "hstar" and "wikitq" in dataset:
            dataset = "wikitq"

        final_evaluate_default(
            outputs, mode, dataset, readme_file, mmtabqa_sub_dataset, mmtabqa_dataset_split, results_file
        )

    if use_llm_as_judge:
        print("Using LLM-as-a-judge evaluation")
        evaluate_AQ_via_LLM_as_a_judge(
            outputs,
            generator,
            mode,
            readme_file,
            mmtabqa_sub_dataset,
            mmtabqa_dataset_split,
            results_file,
        )


def final_evaluate_fetaqa(
    outputs, mode, readme_file=None, mmtabqa_sub_dataset=None, mmtabqa_dataset_split=None, results_file=None
):
    # not used by the baselne models. They call the same metrics but in a different function. Metric is equal.
    try:
        nltk.data.find("tokenizers/punkt")
    except (OSError, LookupError):
        nltk.download("punkt")

    import evaluate

    rouge = evaluate.load("rouge")

    predictions = []
    references = []
    index_error_count = 0

    for i in tqdm(outputs, desc="Evaluating FetaQA"):
        output = outputs[str(i)]
        # Baselines & H-STAR & interleaved reasoner all use prompts that have "Step 2:"
        pred = output["generations"][0].split("Step 2:")[-1].strip()
        predictions.append(pred)
        references.append(output["ori_data_item"]["answer_text"][0])

    # ROUGE
    processed_preds_rouge, processed_golds_rouge = postprocess_text(predictions, references, "rouge")
    rouge_results = rouge.compute(predictions=processed_preds_rouge, references=processed_golds_rouge)

    if index_error_count > 0:
        print(f"IndexError occurred {index_error_count} times.")

    print("\n\n\n----- FetaQA Evaluation Results -----")
    if mmtabqa_sub_dataset is not None and mmtabqa_dataset_split is not None:
        print(f"----- MMTabQA Sub-Dataset: {mmtabqa_sub_dataset} / {mmtabqa_dataset_split} -----")

    print(f"Total Samples: {len(outputs)}")
    print(f"ROUGE: {rouge_results}")
    print(f"ROUGE-L: {rouge_results['rougeL']}")

    if readme_file is not None:
        with open(readme_file, "a") as f:
            if mmtabqa_sub_dataset is not None and mmtabqa_dataset_split is not None:
                f.write(f"\n\n{mode} - {mmtabqa_sub_dataset} - {mmtabqa_dataset_split} (FetaQA)\n")
            else:
                f.write(f"\n\n{mode} - FetaQA\n")
            f.write(f"File: {results_file}\n")
            f.write(f"Total Samples: {len(outputs)}\n")
            f.write(f"ROUGE-L: {rouge_results['rougeL']}                      ROUGE: {rouge_results}\n")


def final_evaluate_default(
    outputs,
    mode,
    dataset=None,
    readme_file=None,
    mmtabqa_sub_dataset=None,
    mmtabqa_dataset_split=None,
    results_file=None,
):
    """
    This is the default evaluation function for H-STAR and MMTabQA. It handles the default text-only WikiTQ and TabFact and also the MMTabQA-WikiTQ, MMTabQA-WikiSQL.
    For FetaQA, we have a separate evaluation function as it doesn't use Exact Match.
    """
    acc = 0
    index_error_count = 0
    for i in tqdm(outputs, desc="Evaluating"):
        try:
            output = outputs[str(i)]
            if mode == "hstar":
                if "the answer is: " in output["generations"][0]:
                    pred_answer = output["generations"][0].split("the answer is: ")[1]
                else:
                    pred_answer = output["generations"][0]

                # We did not evaluate tab fact but keep it here in case someone wants to evaluate the text-only TabFact like it was evaluated in H-STAR.
                if dataset == "tab_fact":
                    if pred_answer == "true." or pred_answer == "true":
                        pred_answer = [1]
                    elif pred_answer == "false" or pred_answer == "false.":
                        pred_answer = [0]
                else:
                    pred_answer = pred_answer.split('"')[1:2]

                gold_answer = output["ori_data_item"]["answer_text"]
                question = output["ori_data_item"]["question"]

            elif mode == "mmtabqa_baseline":
                assert len(output["generations"]) == 1, (
                    f"MMTabQA baseline should only have one generation, but got: {len(output['generations'])} many generations."
                )

                # always use the wikitq metric
                dataset = "wikitq"
                # Some original baseline prompts had wrong spaces. We fixed this but for compatibility, we check for both.
                if "Step 2 :" in output["generations"][0]:
                    pred_answer = output["generations"][0].split("Step 2 :")[-1].strip()
                else:
                    pred_answer = output["generations"][0].split("Step 2:")[-1].strip()
                pred_answer = [pred_answer]  # the evaluator expects a list
                gold_answer = output["answer_text"]
                question = output["question"]

            elif mode == "interleaved_reasoner":
                dataset = "wikitq"

                if "Step 2 :" in output["generations"][0]:
                    pred_answer = output["generations"][0].split("Step 2 :")[-1].strip()
                else:
                    pred_answer = output["generations"][0].split("Step 2:")[-1].strip()
                pred_answer = [pred_answer]  # the evaluator expects a list
                gold_answer = output["ori_data_item"]["answer_text"]
                question = output["ori_data_item"]["question"]
            else:
                raise ValueError(f"Invalid mode: {mode}")

            # if gold_answer == "['1911']":
            # print(f"gold_answer: {gold_answer}   pred_answer: {pred_answer}")
            # if pred answer length more then 1
            if len(pred_answer) > 1:
                print(f"gold_answer: {gold_answer}   pred_answer: {pred_answer}")

            # Score is either 1 or 0
            score = Evaluator().evaluate(pred_answer, gold_answer, dataset=dataset, question=question)
            acc += score
        except IndexError as e:
            if str(e) == "list index out of range":
                print(f"IndexError while processing {i} (likely due to wrong answer format provided by the LLM)")
                print(f"\nanswer given by the LLM:\n\n{output['generations'][0]}\n\n")
                index_error_count += 1
                continue
            else:
                print(traceback.format_exc())

    if index_error_count > 0:
        print(f"IndexError occurred {index_error_count} times.")

    if mmtabqa_sub_dataset is not None and mmtabqa_dataset_split is not None:
        print(f"\n\n\n----- MMTabQA Sub-Dataset: {mmtabqa_sub_dataset} / {mmtabqa_dataset_split} -----")

    print(f"Correct Samples: {acc}; Total Samples: {len(outputs)}")
    print(f"Accuracy: {100 * acc / len(outputs):.2f}")

    # Add the accuracy to the README file
    if readme_file is not None:
        with open(readme_file, "a") as f:
            if mmtabqa_sub_dataset is not None and mmtabqa_dataset_split is not None:
                f.write(f"\n\n{mode} - {mmtabqa_sub_dataset} - {mmtabqa_dataset_split}\n")
            else:
                f.write(f"\n\n{mode}\n")
            f.write(f"File: {results_file}\n")
            f.write(f"Correct Samples: {acc}; Total Samples: {len(outputs)}\n")
            f.write(f"Accuracy: {100 * acc / len(outputs):.2f}\n")

    return acc


if __name__ == "__main__":
    raise NotImplementedError("Calling this file directly is not supported anymore")


# postprocess_text refers to the https://github.com/Yale-LILY/FeTaQA/blob/main/end2end/train.py
def postprocess_text(preds, labels, metric_name):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    if metric_name == "rouge":
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    elif metric_name == "sacrebleu":  # sacrebleu
        labels = [[label] for label in labels]
    elif metric_name == "bleu":
        preds = [pred.split(" ") for pred in preds]
        labels = [[label.split(" ")] for label in labels]
    else:
        pass

    return preds, labels
