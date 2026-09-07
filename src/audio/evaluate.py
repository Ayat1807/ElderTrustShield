"""
Evaluation metrics: EER, both naive (0.5-threshold) and EER-calibrated
accuracy, confusion matrix, and per-attack-algorithm breakdown.

EER is threshold-independent and is the metric to trust for this task,
especially under class imbalance (this shard is ~77% spoof / 23%
bonafide). Naive 0.5-threshold accuracy can look badly imbalanced between
bonafide/spoof recall even when EER indicates reasonable separation —
confirmed on this project's own zero-shot run (24% bonafide recall at 0.5
threshold vs. ~81% balanced recall at the EER-optimal threshold). Always
report EER; report calibrated accuracy if a single operating-point number
is needed for a real decision.
"""
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve


def compute_metrics(true_labels, predicted_labels, bonafide_scores, attack_algorithms):
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    bonafide_scores = np.asarray(bonafide_scores)
    attack_algorithms = np.asarray(attack_algorithms)

    naive_accuracy = accuracy_score(true_labels, predicted_labels)
    naive_matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])

    bona_fide_targets = (true_labels == 0).astype(np.int32)
    fpr, tpr, thresholds = roc_curve(bona_fide_targets, bonafide_scores)
    eer_index = np.nanargmin(np.abs(fpr - (1.0 - tpr)))
    eer = (fpr[eer_index] + (1.0 - tpr[eer_index])) / 2.0
    eer_threshold = thresholds[eer_index]

    calibrated_bonafide_pred = (bonafide_scores >= eer_threshold).astype(int)
    calibrated_predictions = 1 - calibrated_bonafide_pred  # back to 0=bonafide,1=spoof convention
    calibrated_accuracy = accuracy_score(true_labels, calibrated_predictions)
    calibrated_matrix = confusion_matrix(true_labels, calibrated_predictions, labels=[0, 1])

    groups = defaultdict(list)
    for index, aa in enumerate(attack_algorithms):
        groups[aa].append(index)
    breakdown = {}
    for aa in sorted(groups):
        idx = np.asarray(groups[aa])
        breakdown[aa] = {
            "files": len(idx),
            "accuracy": accuracy_score(true_labels[idx], predicted_labels[idx]),
            "bonafide_count": int(np.sum(true_labels[idx] == 0)),
            "spoof_count": int(np.sum(true_labels[idx] == 1)),
        }

    return {
        "files_evaluated": len(true_labels),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "naive_accuracy": naive_accuracy,
        "naive_confusion_matrix": naive_matrix,
        "calibrated_accuracy": calibrated_accuracy,
        "calibrated_confusion_matrix": calibrated_matrix,
        "breakdown_by_attack_algorithm": breakdown,
    }


def print_report(metrics: dict) -> None:
    print("OVERALL_RESULTS")
    print("FILES_EVALUATED=", metrics["files_evaluated"])
    print(f"EER={metrics['eer']:.6f} ({metrics['eer']*100:.2f}%)")
    print(f"NAIVE_ACCURACY (0.5 threshold)={metrics['naive_accuracy']:.6f}")
    print(f"CALIBRATED_ACCURACY (EER threshold)={metrics['calibrated_accuracy']:.6f}")
    print()
    print("NAIVE_CONFUSION_MATRIX (rows: true [bonafide, spoof], cols: predicted [bonafide, spoof])")
    print(metrics["naive_confusion_matrix"])
    print()
    print("CALIBRATED_CONFUSION_MATRIX")
    print(metrics["calibrated_confusion_matrix"])
    print()
    print("BREAKDOWN_BY_ATTACK_ALGORITHM")
    print(f"{'attack_algorithm':<20}{'files':>8}{'accuracy':>14}{'bonafide':>12}{'spoof':>10}")
    for aa, row in metrics["breakdown_by_attack_algorithm"].items():
        print(f"{aa:<20}{row['files']:>8}{row['accuracy']:>14.6f}{row['bonafide_count']:>12}{row['spoof_count']:>10}")
