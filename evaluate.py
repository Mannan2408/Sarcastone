"""
SarcasTone — Evaluation Module
================================
Computes Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
for trained models. Generates comparative analysis across all 3 models.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


def evaluate_model(model: nn.Module, test_loader, model_name: str,
                   input_mode: str = "fusion") -> dict:
    """
    Evaluate a trained model on the test set.

    Returns:
        Dict with accuracy, precision, recall, f1, confusion_matrix.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            if input_mode == "text":
                inputs = batch["text_embedding"].to(device)
                logits = model(inputs)
            elif input_mode == "audio":
                inputs = batch["audio_features"].to(device)
                logits = model(inputs)
            else:  # fusion
                text_emb = batch["text_embedding"].to(device)
                audio_feat = batch["audio_features"].to(device)
                logits = model(text_emb, audio_feat)

            labels = batch["label"].numpy()
            preds = (torch.sigmoid(logits).cpu().numpy() >= 0.5).astype(int).flatten()

            all_preds.extend(preds)
            all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Handle edge case where test set might be very small
    if len(all_preds) == 0:
        print(f"\n⚠️  {model_name}: No test samples to evaluate")
        return {
            "name": model_name,
            "accuracy": 0, "precision": 0, "recall": 0, "f1": 0,
            "confusion_matrix": np.zeros((2, 2)),
        }

    results = {
        "name": model_name,
        "accuracy": accuracy_score(all_labels, all_preds) * 100,
        "precision": precision_score(all_labels, all_preds, zero_division=0) * 100,
        "recall": recall_score(all_labels, all_preds, zero_division=0) * 100,
        "f1": f1_score(all_labels, all_preds, zero_division=0) * 100,
        "confusion_matrix": confusion_matrix(all_labels, all_preds, labels=[0, 1]),
    }

    print(f"\n--- {model_name} ---")
    print(f"  Accuracy:  {results['accuracy']:.1f}%")
    print(f"  Precision: {results['precision']:.1f}%")
    print(f"  Recall:    {results['recall']:.1f}%")
    print(f"  F1-Score:  {results['f1']:.1f}%")
    print(f"  Confusion Matrix:")
    cm = results["confusion_matrix"]
    print(f"    Predicted:   Non-Sarc  Sarcastic")
    if cm.shape == (2, 2):
        print(f"    Non-Sarc  │  {cm[0][0]:5d}    {cm[0][1]:5d}")
        print(f"    Sarcastic │  {cm[1][0]:5d}    {cm[1][1]:5d}")
    else:
        print(f"    {cm}")

    return results


def print_comparison(text_results: dict, audio_results: dict, fusion_results: dict):
    """Print a comparative table of all 3 models."""
    print("\n" + "=" * 70)
    print("  COMPARATIVE ANALYSIS: Text-Only vs Audio-Only vs Fusion")
    print("=" * 70)
    print(f"  {'Metric':<15} {'Text-Only':>12} {'Audio-Only':>12} {'Fusion':>12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")

    for metric in ["accuracy", "precision", "recall", "f1"]:
        t = text_results[metric]
        a = audio_results[metric]
        f = fusion_results[metric]
        best = max(t, a, f)
        label = metric.capitalize()

        t_mark = " ★" if t == best else ""
        a_mark = " ★" if a == best else ""
        f_mark = " ★" if f == best else ""

        print(f"  {label:<15} {t:>10.1f}%{t_mark} {a:>10.1f}%{a_mark} {f:>10.1f}%{f_mark}")

    print(f"\n  ★ = Best performing model for that metric")
    print("=" * 70)
