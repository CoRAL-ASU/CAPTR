#!/usr/bin/env python3
"""
Evaluation script for MMTabReal results.
Computes Exact Match, Substring Match, and F1 Score metrics.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter


def normalize_answer(s: str) -> str:
    """Normalize answer text for comparison."""
    if not isinstance(s, str):
        s = str(s)
    
    # Convert to lowercase
    s = s.lower()
    
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    
    # Remove punctuation
    s = re.sub(r'[^\w\s]', ' ', s)
    
    # Remove extra whitespace
    s = ' '.join(s.split())
    
    return s.strip()


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Check if prediction exactly matches ground truth (after normalization)."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def substring_match_score(prediction: str, ground_truth: str) -> float:
    """Check if one answer is a substring of the other (after normalization)."""
    pred_norm = normalize_answer(prediction)
    truth_norm = normalize_answer(ground_truth)
    
    if not pred_norm or not truth_norm:
        return 0.0
    
    # Check if either is a substring of the other
    if pred_norm in truth_norm or truth_norm in pred_norm:
        return 1.0
    return 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score based on word overlap."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1


def evaluate_file(file_path: Path) -> Dict[str, float]:
    """Evaluate a single JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    exact_matches = []
    substring_matches = []
    f1_scores = []
    
    for idx, entry in data.items():
        # Get ground truth and prediction
        ground_truths = entry.get('answer_text', [])
        predictions = entry.get('generations', [])
        
        if not ground_truths or not predictions:
            continue
        
        # Take first answer from each (most common case)
        ground_truth = ground_truths[0] if isinstance(ground_truths, list) else ground_truths
        prediction = predictions[0] if isinstance(predictions, list) else predictions
        
        # Skip if answer is UNKNOWN or empty
        if not ground_truth or not prediction:
            continue
        
        # Calculate metrics
        em = exact_match_score(prediction, ground_truth)
        sm = substring_match_score(prediction, ground_truth)
        f1 = f1_score(prediction, ground_truth)
        
        exact_matches.append(em)
        substring_matches.append(sm)
        f1_scores.append(f1)
    
    # Calculate averages
    n = len(exact_matches)
    if n == 0:
        return {
            'exact_match': 0.0,
            'substring_match': 0.0,
            'f1_score': 0.0,
            'count': 0
        }
    
    return {
        'exact_match': sum(exact_matches) / n * 100,
        'substring_match': sum(substring_matches) / n * 100,
        'f1_score': sum(f1_scores) / n,  # Keep as decimal (0-1)
        'count': n
    }


def evaluate_directory(dir_path: Path, mode: str) -> Dict[str, Dict[str, float]]:
    """Evaluate all JSON files in a directory."""
    results = {}
    
    # Find all model directories
    for model_dir in dir_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        model_results = {}
        
        # Find all JSON files for this model
        for json_file in model_dir.glob('*.json'):
            # Extract question type from filename
            # e.g., MMTabReal_AQ_partial_input.json -> AQ
            match = re.search(r'MMTabReal_([A-Z]+)_', json_file.name)
            if match:
                question_type = match.group(1)
            else:
                question_type = json_file.stem
            
            # Evaluate this file
            file_results = evaluate_file(json_file)
            model_results[question_type] = file_results
        
        if model_results:
            results[model_name] = model_results
    
    return results


def print_results(results: Dict[str, Dict[str, float]], mode: str):
    """Pretty print evaluation results."""
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS - {mode.upper()}")
    print(f"{'='*80}\n")
    
    for model_name, model_results in results.items():
        print(f"\nModel: {model_name}")
        print("-" * 80)
        print(f"{'Question Type':<15} {'Count':<10} {'Exact Match':<15} {'Substring Match':<18} {'F1 Score':<15}")
        print("-" * 80)
        
        all_em = []
        all_sm = []
        all_f1 = []
        all_counts = []
        
        for q_type, metrics in sorted(model_results.items()):
            print(f"{q_type:<15} {metrics['count']:<10} "
                  f"{metrics['exact_match']:>13.2f}% "
                  f"{metrics['substring_match']:>16.2f}% "
                  f"{metrics['f1_score']:>14.4f}")
            
            all_em.append(metrics['exact_match'])
            all_sm.append(metrics['substring_match'])
            all_f1.append(metrics['f1_score'])
            all_counts.append(metrics['count'])
        
        # Print averages
        if all_em:
            print("-" * 80)
            avg_em = sum(all_em) / len(all_em)
            avg_sm = sum(all_sm) / len(all_sm)
            avg_f1 = sum(all_f1) / len(all_f1)
            total_count = sum(all_counts)
            
            print(f"{'AVERAGE':<15} {total_count:<10} "
                  f"{avg_em:>13.2f}% "
                  f"{avg_sm:>16.2f}% "
                  f"{avg_f1:>14.4f}")
            print("-" * 80)


def main():
    base_dir = Path(__file__).parent
    
    # Evaluate partial input results
    partial_dir = base_dir / 'partial_results_mmtabreal'
    if partial_dir.exists():
        partial_results = evaluate_directory(partial_dir, 'partial')
        print_results(partial_results, 'partial')
    else:
        print(f"Warning: {partial_dir} not found")
    
    # Evaluate interleaved results
    interleaved_dir = base_dir / 'interleaved_results_mmtabreal'
    if interleaved_dir.exists():
        interleaved_results = evaluate_directory(interleaved_dir, 'interleaved')
        print_results(interleaved_results, 'interleaved')
    else:
        print(f"Warning: {interleaved_dir} not found")
    
    # Comparison if both exist
    if partial_dir.exists() and interleaved_dir.exists():
        print(f"\n{'='*80}")
        print("COMPARISON: PARTIAL vs INTERLEAVED")
        print(f"{'='*80}\n")
        
        for model_name in partial_results.keys():
            if model_name in interleaved_results:
                print(f"\nModel: {model_name}")
                print("-" * 80)
                print(f"{'Metric':<20} {'Partial':<15} {'Interleaved':<15} {'Difference':<15}")
                print("-" * 80)
                
                # Calculate aggregated metrics
                partial_metrics = partial_results[model_name]
                interleaved_metrics = interleaved_results[model_name]
                
                partial_em = sum(m['exact_match'] for m in partial_metrics.values()) / len(partial_metrics)
                partial_sm = sum(m['substring_match'] for m in partial_metrics.values()) / len(partial_metrics)
                partial_f1 = sum(m['f1_score'] for m in partial_metrics.values()) / len(partial_metrics)
                
                interleaved_em = sum(m['exact_match'] for m in interleaved_metrics.values()) / len(interleaved_metrics)
                interleaved_sm = sum(m['substring_match'] for m in interleaved_metrics.values()) / len(interleaved_metrics)
                interleaved_f1 = sum(m['f1_score'] for m in interleaved_metrics.values()) / len(interleaved_metrics)
                
                print(f"{'Exact Match':<20} {partial_em:>13.2f}% {interleaved_em:>13.2f}% {(interleaved_em - partial_em):>+13.2f}%")
                print(f"{'Substring Match':<20} {partial_sm:>13.2f}% {interleaved_sm:>13.2f}% {(interleaved_sm - partial_sm):>+13.2f}%")
                print(f"{'F1 Score':<20} {partial_f1:>14.4f} {interleaved_f1:>14.4f} {(interleaved_f1 - partial_f1):>+14.4f}")
                print("-" * 80)


if __name__ == '__main__':
    main()
