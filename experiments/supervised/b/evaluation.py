"""Classification diagnostics and hard-label retrieval using the legacy evaluator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import (average_precision_score, accuracy_score, hamming_loss,
                             multilabel_confusion_matrix, precision_recall_fscore_support, f1_score)

from wbm_retrieval.common import csv_rows, write_csv, write_json
from wbm_retrieval.evaluation import evaluate_ranking
from .suite import make_loader


@torch.no_grad()
def predict(model, dataset, config, device):
    model.eval()
    ids, labels, probabilities = [], [], []
    total, count = 0., 0
    for sample_ids, images, targets in make_loader(dataset, config):
        logits = model(images.to(device))
        loss = F.binary_cross_entropy_with_logits(logits, targets.to(device), reduction="sum")
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite classifier logits/loss")
        total += loss.item()
        count += targets.numel()
        ids.append(sample_ids.numpy())
        labels.append(targets.numpy().astype(np.uint8))
        probabilities.append(logits.sigmoid().cpu().numpy())
    return np.concatenate(ids), np.concatenate(labels), np.concatenate(probabilities), total / count


def classification_metrics(labels, probabilities, threshold=0.5, groups=True):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if (labels.ndim != 2 or labels.shape[1] != 8 or probabilities.shape != labels.shape or not len(labels)
            or not np.isin(labels, [0, 1]).all() or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))):
        raise ValueError("Expected nonempty Nx8 binary labels and finite probabilities in [0,1]")
    predicted = probabilities >= threshold
    precision, recall, f1, support = precision_recall_fscore_support(labels, predicted, average=None, zero_division=0)
    micro = precision_recall_fscore_support(labels, predicted, average="micro", zero_division=0)
    ap = [float(average_precision_score(labels[:, k], probabilities[:, k])) if support[k] else None for k in range(8)]
    values = {"macro_f1": float(f1.mean()), "micro_f1": float(micro[2]), "micro_precision": float(micro[0]),
              "micro_recall": float(micro[1]), "macro_precision": float(precision.mean()),
              "macro_recall": float(recall.mean()), "subset_accuracy": float(accuracy_score(labels, predicted)),
              "hamming_loss": float(hamming_loss(labels, predicted)),
              "samples_f1": float(f1_score(labels, predicted, average="samples", zero_division=0)),
              "macro_ap": float(np.mean([value for value in ap if value is not None])) if any(v is not None for v in ap) else None,
              "empty_prediction_rate": float((predicted.sum(1) == 0).mean()),
              "label_count_mae": float(np.abs(predicted.sum(1).astype(float) - labels.sum(1)).mean())}
    result = {"num_maps": len(labels), "threshold": threshold, "metrics": values,
              "undefined_policy": "P/R/F1 zero_division=0; AP with no positives is null and excluded from macro_ap",
              "ap_defined_classes": sum(value is not None for value in ap),
              "per_class": [{"label": k, "precision": float(precision[k]), "recall": float(recall[k]),
                             "f1": float(f1[k]), "support": int(support[k]), "ap": ap[k]} for k in range(8)],
              "confusion_matrices": multilabel_confusion_matrix(labels, predicted, labels=np.arange(8)).tolist()}
    if groups:
        result["groups"] = {}
        for name, mask in (("single", labels.sum(1) == 1), ("mixed", labels.sum(1) > 1)):
            result["groups"][name] = classification_metrics(labels[mask], probabilities[mask], threshold, False) if mask.any() else None
    return result


def save_classification(output, ids, labels, probabilities, loss, threshold):
    output = Path(output)
    result = classification_metrics(labels, probabilities, threshold)
    result["metrics"]["bce"] = float(loss)
    write_json(output / "metrics.json", result)
    write_csv(output / "per_class_metrics.csv", result["per_class"], list(result["per_class"][0]))
    predicted = (probabilities >= threshold).astype(np.uint8)
    np.savez_compressed(output / "predictions.npz", sample_ids=ids, true_labels=labels,
                        probabilities=probabilities, predicted_labels=predicted)
    rows = []
    for i, sample_id in enumerate(ids):
        row = {"sample_id": int(sample_id), "exact_match": bool(np.array_equal(labels[i], predicted[i]))}
        for k in range(8):
            row.update({f"true_{k}": int(labels[i, k]), f"pred_{k}": int(predicted[i, k]), f"prob_{k}": float(probabilities[i, k])})
        rows.append(row)
    write_csv(output / "predictions.csv", rows, list(rows[0]))
    return result


def hard_containment(query, candidates):
    query, candidates = np.asarray(query), np.asarray(candidates)
    if not np.isin(query, [0, 1]).all() or not np.isin(candidates, [0, 1]).all():
        raise ValueError("Retrieval accepts hard binary labels only")
    count = int(query.sum())
    if count == 0:
        return np.zeros(len(candidates), dtype=float)
    return np.sum(candidates.astype(float) * query.astype(float), axis=1) / count


def rank_predictions(ids, predicted, protocol):
    """Only predicted labels and candidate IDs affect ranking, never protocol.labels."""
    ids, predicted = np.asarray(ids), np.asarray(predicted)
    if len(set(ids.tolist())) != len(ids) or predicted.shape != (len(ids), 8):
        raise ValueError("Predictions must have unique map IDs and eight label columns")
    lookup = {int(sample_id): i for i, sample_id in enumerate(ids)}
    audits = []
    for q in protocol.queries:
        candidates = np.asarray(protocol.candidates[q], dtype=np.int64)
        query = predicted[lookup[q]]
        scores = hard_containment(query, predicted[[lookup[int(c)] for c in candidates]])
        order = np.lexsort((candidates, -scores))
        audits.append({"query_id": q, "empty_prediction": bool(query.sum() == 0),
                       "distinct_scores": len(np.unique(scores)), "top_score_ties": int((scores == scores.max()).sum()),
                       "all_zero_scores": bool((scores == 0).all())})
        yield [(q, rank, int(candidates[i]), float(scores[i])) for rank, i in enumerate(order, 1)], audits[-1]


def retrieval_report(output, ids, probabilities, protocol, config, bootstrap=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    import csv
    audits = []
    with (output / "rankings.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "rank", "candidate_id", "similarity_score"])
        for rows, audit in rank_predictions(ids, probabilities >= config["threshold"], protocol):
            writer.writerows(rows)
            audits.append(audit)
    settings = dict(config["evaluation"])
    if bootstrap is not None:
        settings["bootstrap_samples"] = bootstrap
    result = evaluate_ranking(output / "rankings.csv", protocol, settings, config["seed"])
    write_csv(output / "tie_audit.csv", audits, list(audits[0]))
    write_json(output / "retrieval_info.json", {"score": "predicted_query_label_containment",
               "threshold": config["threshold"], "empty_query_policy": "all candidates score zero; retain query",
               "ties": "candidate_id ascending; no probability tie-break",
               "num_queries": len(audits), "num_candidates": [len(protocol.candidates[q]) for q in protocol.queries],
               "empty_query_rate": float(np.mean([a["empty_prediction"] for a in audits])),
               "all_zero_scores_rate": float(np.mean([a["all_zero_scores"] for a in audits])),
               "mean_distinct_scores": float(np.mean([a["distinct_scores"] for a in audits])),
               "mean_top_score_ties": float(np.mean([a["top_score_ties"] for a in audits]))})
    per_query = csv_rows(output / "per_query_metrics.csv")
    groups = {}
    for name, condition in (("single", lambda count: count == 1), ("mixed", lambda count: count > 1)):
        rows = [r for r in per_query if condition(int(np.sum(protocol.labels[int(r["query_id"])])))]
        metrics = {}
        for key in result["metrics"]:
            values = [float(row[key]) for row in rows if row[key] != ""]
            metrics[key] = float(np.mean(values)) if values else None
        groups[name] = {"num_queries": len(rows), "metrics": metrics}
    write_json(output / "group_metrics.json", groups)
    return result
