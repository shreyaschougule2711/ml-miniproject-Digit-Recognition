"""
Evaluation benchmark for multi-digit handwritten number recognition.

Constructs multi-digit composite test sequences using held-out MNIST test digits
and synthetic handwriting samples to measure:
1. Segmentation Success Rate (correct number of digits extracted).
2. Per-digit Classification Accuracy.
3. Exact-match Full-Number Accuracy.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from typing import List, Dict, Any

import config
from src.data_loader import download_or_load_raw_mnist
from src.multi_digit import predict_multi_digit_number


def create_composite_number_from_mnist(
    digits_list: List[int],
    X_test: np.ndarray,
    y_test: np.ndarray,
    spacing_px: int = 14,
    height: int = 80
) -> np.ndarray:
    """
    Creates a multi-digit test image by stitching untouched MNIST test digits
    with natural horizontal spacing and padding.
    """
    canvas_parts = []
    for d in digits_list:
        indices = np.where(y_test == d)[0]
        # Choose a random sample for that digit
        chosen_idx = np.random.choice(indices)
        digit_28x28 = X_test[chosen_idx].reshape(28, 28)
        
        # Add side padding for spacing
        spaced_digit = np.pad(digit_28x28, ((0, 0), (spacing_px // 2, spacing_px // 2)), mode='constant')
        canvas_parts.append(spaced_digit)

    combined_digits = np.hstack(canvas_parts)
    # Pad vertically to standard banner height
    pad_top = (height - 28) // 2
    pad_bottom = height - 28 - pad_top
    composite = np.pad(combined_digits, ((pad_top, pad_bottom), (20, 20)), mode='constant')
    return composite


def run_multidigit_benchmark(num_trials_per_length: int = 10) -> Dict[str, Any]:
    """
    Evaluates multi-digit recognition across numbers of varying lengths (1 to 5 digits).
    """
    print("\n=======================================================")
    print("   MULTI-DIGIT NUMBER RECOGNITION BENCHMARK EVALUATION")
    print("=======================================================")
    
    # Load untouched test split
    X_all, y_all = download_or_load_raw_mnist()
    X_test = X_all[60000:]
    y_test = y_all[60000:]

    test_numbers = [
        # Single digit
        "5", "8",
        # 2-digit
        "12", "45", "99",
        # 3-digit
        "108", "256", "789",
        # 4-digit
        "2026", "1234",
        # 5-digit
        "12345"
    ]

    results_table = []
    total_samples = 0
    total_digits = 0
    correct_digits = 0
    exact_matches = 0
    segmentation_successes = 0

    for num_str in test_numbers:
        target_digits = [int(ch) for ch in num_str]
        
        for trial in range(num_trials_per_length):
            total_samples += 1
            total_digits += len(target_digits)
            
            img = create_composite_number_from_mnist(target_digits, X_test, y_test)
            pred_res = predict_multi_digit_number(img, model_key="rbf_svm")

            is_seg_ok = (pred_res["digit_count"] == len(target_digits))
            if is_seg_ok:
                segmentation_successes += 1

            is_exact = (pred_res["predicted_number_str"] == num_str)
            if is_exact:
                exact_matches += 1

            # Count per-digit matches
            pred_digits = [d["predicted_digit"] for d in pred_res.get("digits", [])]
            for idx, target_d in enumerate(target_digits):
                if idx < len(pred_digits) and pred_digits[idx] == target_d:
                    correct_digits += 1

            results_table.append({
                "Target Number": num_str,
                "Length": len(num_str),
                "Predicted Number": pred_res["predicted_number_str"],
                "Segmentation Success": is_seg_ok,
                "Exact Match": is_exact,
                "Detected Count": pred_res["digit_count"],
                "Confidence (%)": pred_res.get("average_confidence_percentage", 0.0)
            })

    df = pd.DataFrame(results_table)
    seg_rate = (segmentation_successes / total_samples) * 100
    digit_acc = (correct_digits / total_digits) * 100
    exact_match_acc = (exact_matches / total_samples) * 100

    print(f"\n[Multi-Digit Benchmark Summary]")
    print(f"Total Test Sequences Evaluated:   {total_samples}")
    print(f"Total Individual Digits Tested:   {total_digits}")
    print(f"Segmentation Success Rate:        {seg_rate:.2f}%")
    print(f"Per-Digit Classification Accuracy: {digit_acc:.2f}%")
    print(f"Exact-Match Number Accuracy:       {exact_match_acc:.2f}%\n")

    summary = {
        "total_test_sequences": total_samples,
        "total_individual_digits": total_digits,
        "segmentation_success_rate": round(seg_rate, 2),
        "per_digit_accuracy": round(digit_acc, 2),
        "exact_match_accuracy": round(exact_match_acc, 2),
        "sample_cases": results_table[:15]
    }

    # Save to results/
    out_csv = config.RESULTS_DIR / "multidigit_benchmark.csv"
    df.to_csv(out_csv, index=False)
    print(f"[Multi-Digit Benchmark] Saved full test results to {out_csv}")
    return summary


if __name__ == "__main__":
    run_multidigit_benchmark(num_trials_per_length=5)
