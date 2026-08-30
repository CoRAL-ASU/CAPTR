"""Extract evaluation scores into a proper CSV table.

Columns: Model folder name, then one column per table category (from folders.json).
Values: Accuracy (% correct) for that category in that model.

Usage:
    python scripts/scores_to_csv_tables.py --input evaluation_results_by_category.json --out evaluation_scores_table.csv
"""
import argparse
import json
import csv
import os


def extract_table_from_json(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = data.get("models", {}) if isinstance(data, dict) else {}
    
    # Collect all categories
    all_categories = set()
    model_results = []
    
    for model_key, model_result in models.items():
        # Extract model name from path (just the final folder name)
        model_name = model_key.split("/")[-1] if "/" in model_key else model_key
        
        grouped = model_result.get("grouped", {})
        if not isinstance(grouped, dict):
            continue
        
        # Calculate accuracy per category
        category_scores = {}
        for cat, bucket in grouped.items():
            if not isinstance(bucket, list):
                continue
            
            correct = 0
            total = 0
            for entry in bucket:
                if isinstance(entry, dict) and "judge_score" in entry:
                    correct += entry["judge_score"]
                    total += 1
            
            if total > 0:
                accuracy = (100 * correct / total)
                category_scores[cat] = accuracy
                all_categories.add(cat)
            else:
                category_scores[cat] = None
        
        model_results.append({
            "model_name": model_name,
            "scores": category_scores
        })
    
    return model_results, sorted(all_categories)


def write_csv_table(model_results, categories, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header row: Model name + all categories
        header = ["Model"] + categories
        writer.writerow(header)
        
        # Data rows
        for model in model_results:
            row = [model["model_name"]]
            for cat in categories:
                score = model["scores"].get(cat)
                # Format score as "XX.XX%" or empty if no data
                if score is not None:
                    row.append(f"{score:.2f}%")
                else:
                    row.append("")
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=os.getenv("CAPTR_EVALUATION_RESULTS_JSON", "evaluation_results_by_category.json"),
        help="Path to evaluation_results_by_category.json",
    )
    parser.add_argument("--out", default="evaluation_scores_table.csv", help="Output CSV path")
    args = parser.parse_args()

    model_results, categories = extract_table_from_json(args.input)
    write_csv_table(model_results, categories, args.out)
    print(f"Wrote {len(model_results)} models × {len(categories)} categories to {args.out}")


if __name__ == "__main__":
    main()
