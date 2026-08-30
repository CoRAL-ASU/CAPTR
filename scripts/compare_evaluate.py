import argparse
import json
import traceback

from utils.evaluator import Evaluator


def main(results_file1, results_file2, dataset):
    with open(results_file1, "r") as f:
        outputs1 = json.load(f)
    with open(results_file2, "r") as f:
        outputs2 = json.load(f)

    if "wikitq" in dataset:
        dataset = "wikitq"

    acc1 = 0
    acc2 = 0
    differences = []

    for i in outputs1:
        try:
            output1 = outputs1[str(i)]
            output2 = outputs2[str(i)]

            # Process first file
            if dataset == "wikitq":
                try:
                    pred_answer1 = output1["generations"][0].split("the answer is: ")[1]
                    pred_answer1 = pred_answer1.split('"')[1:2]
                except IndexError:
                    print(f"\nIndexError in Model 1 for index {i}")
                    print(f"Model 1 generation: {output1['generations'][0]}")
                    raise

                try:
                    pred_answer2 = output2["generations"][0].split("the answer is: ")[1]
                    pred_answer2 = pred_answer2.split('"')[1:2]
                except IndexError:
                    print(f"\nIndexError in Model 2 for index {i}")
                    print(f"Model 2 generation: {output2['generations'][0]}")
                    raise

            elif dataset == "tab_fact":
                try:
                    pred_answer1 = output1["generations"][0].lower().split("statement is: ")[1].replace(" ", "")
                    if pred_answer1 == "true." or pred_answer1 == "true":
                        pred_answer1 = [1]
                    elif pred_answer1 == "false" or pred_answer1 == "false.":
                        pred_answer1 = [0]
                except IndexError:
                    print(f"\nIndexError in Model 1 for index {i}")
                    print(f"Model 1 generation: {output1['generations'][0]}")
                    raise

                try:
                    pred_answer2 = output2["generations"][0].lower().split("statement is: ")[1].replace(" ", "")
                    if pred_answer2 == "true." or pred_answer2 == "true":
                        pred_answer2 = [1]
                    elif pred_answer2 == "false" or pred_answer2 == "false.":
                        pred_answer2 = [0]
                except IndexError:
                    print(f"\nIndexError in Model 2 for index {i}")
                    print(f"Model 2 generation: {output2['generations'][0]}")
                    raise

            gold_answer = output1["ori_data_item"]["answer_text"]

            # Score is either 1 or 0
            score1 = Evaluator().evaluate(
                pred_answer1, gold_answer, dataset=dataset, question=output1["ori_data_item"]["question"]
            )
            score2 = Evaluator().evaluate(
                pred_answer2, gold_answer, dataset=dataset, question=output2["ori_data_item"]["question"]
            )

            acc1 += score1
            acc2 += score2

            # If one is correct and the other isn't, record the difference
            if score1 != score2:
                differences.append(
                    {
                        "index": i,
                        "question": output1["ori_data_item"]["question"],
                        "gold_answer": gold_answer,
                        "model1_answer": output1["generations"][0],
                        "model2_answer": output2["generations"][0],
                        "model1_correct": bool(score1),
                        "model2_correct": bool(score2),
                    }
                )

        except IndexError as e:
            if str(e) == "list index out of range":
                print(f"\nDetailed information for index {i}:")
                print(f"Question: {output1['ori_data_item']['question']}")
                print(f"Gold Answer: {output1['ori_data_item']['answer_text']}")
                print(f"Model 1 Answer: {output1['generations'][0]}")
                print(f"Model 2 Answer: {output2['generations'][0]}")
                print("-" * 80)
                continue
            else:
                print(traceback.format_exc())

    print("\nResults for first model:")
    print(f"Correct Samples: {acc1}; Total Samples: {len(outputs1)}")
    print(f"Accuracy: {100 * acc1 / len(outputs1):.2f}")

    print("\nResults for second model:")
    print(f"Correct Samples: {acc2}; Total Samples: {len(outputs2)}")
    print(f"Accuracy: {100 * acc2 / len(outputs2):.2f}")

    print(f"\nFound {len(differences)} cases where models disagreed:")
    for diff in differences:
        print("\n\n\n" + "=" * 80)
        print(f"Index: {diff['index']}")
        print(f"Question: {diff['question']}")
        print(f"Gold Answer: {diff['gold_answer']}")
        print(f"Model 1 Answer: {diff['model1_answer']} (Correct: {diff['model1_correct']})")
        print(f"Model 2 Answer: {diff['model2_answer']} (Correct: {diff['model2_correct']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare results from two model outputs.")
    parser.add_argument("--results_file1", type=str, required=True, help="Path to the first results file.")
    parser.add_argument("--results_file2", type=str, required=True, help="Path to the second results file.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    args = parser.parse_args()

    main(args.results_file1, args.results_file2, args.dataset)
