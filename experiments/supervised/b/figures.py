"""Headless PNG/PDF reports; figures are regenerated from saved predictions."""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from wbm_retrieval.common import read_json, write_json


def save(figure, output, name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"{name}.{suffix}", dpi=160, bbox_inches="tight")
    plt.close(figure)


def learning_curves(folder):
    folder = Path(folder)
    history = read_json(folder / "history.json")
    epochs = [r["epoch"] for r in history]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    ax = axes[0, 0]
    ax.plot(epochs, [r["train_loss"] for r in history], label="Online train BCE", alpha=.65)
    ax.plot(epochs, [r["train_eval"]["bce"] for r in history], label="Fixed train subset (eval)")
    ax.plot(epochs, [r["valid_loss"] for r in history], label="Validation BCE")
    ax.set_title("BCE loss (raw epochs)")
    for ax, metric, title in zip(axes.flat[1:], ("micro_f1", "macro_f1", "subset_accuracy"),
                                  ("Micro-F1", "Macro-F1", "Exact match")):
        ax.plot(epochs, [r["train_eval"][metric] for r in history], label="Fixed train subset (eval)")
        ax.plot(epochs, [r["valid_metrics"][metric] for r in history], label="Validation")
        ax.set_title(title)
        ax.set_ylim(-.03, 1.03)
    summary = read_json(folder / "training_summary.json")
    for ax in axes.flat:
        ax.axvline(summary["best_classifier_epoch"], color="gray", linestyle="--", label="Selected epoch")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    figure.suptitle(f"{folder.name}: classifier learning curves")
    save(figure, folder / "figures", "learning_curves")
    sampled = [r for r in history if r["valid_retrieval"] is not None]
    if sampled:
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for ax, metric in zip(axes, ("ContainmentNDCG@10", "mAP_full")):
            ax.plot([r["epoch"] for r in sampled], [r["valid_retrieval"][metric] for r in sampled], marker="o")
            ax.set(title=metric, xlabel="Epoch", ylim=(-.03, 1.03))
            ax.grid(alpha=.2)
        figure.suptitle("Fixed validation retrieval pool (monitoring only)")
        save(figure, folder / "figures", "valid_retrieval_curves")


def classification_figures(folder, split, root):
    folder, root = Path(folder), Path(root)
    source = folder / "classification" / split
    with np.load(source / "predictions.npz", allow_pickle=False) as arrays:
        ids, true, predicted, probabilities = (arrays[k] for k in ("sample_ids", "true_labels", "predicted_labels", "probabilities"))
    report = read_json(source / "metrics.json")
    output = folder / "figures" / split
    names = [f"label_{k}" for k in range(8)]  # Do not guess the source class-name mapping.
    figure, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    x = np.arange(8)
    for offset, key in zip((-.25, 0, .25), ("precision", "recall", "f1")):
        ax.bar(x + offset, [r[key] for r in report["per_class"]], width=.25, label=key)
    ax.set_xticks(x, [f"{name}\nn={int(true[:, k].sum())}" for k, name in enumerate(names)])
    ax.set(ylim=(0, 1.08), title=f"{split}: per-class classification")
    ax.legend()
    save(figure, output, "per_class_metrics")
    figure, axes = plt.subplots(2, 4, figsize=(12, 6), layout="constrained")
    for k, ax in enumerate(axes.flat):
        matrix = np.asarray(report["confusion_matrices"][k])
        ax.imshow(matrix, cmap="Blues")
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, str(value), ha="center", va="center", color="white" if value > matrix.max()/2 else "black")
        ax.set(title=names[k], xlabel="Predicted", ylabel="True", xticks=[0, 1], yticks=[0, 1])
    figure.suptitle(f"{split}: eight binary confusion matrices (counts)")
    save(figure, output, "multilabel_confusion")
    figure, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    for k in range(8):
        ap = report["per_class"][k]["ap"]
        if ap is None:
            continue
        precision, recall, _ = precision_recall_curve(true[:, k], probabilities[:, k])
        ax.step(recall, precision, where="post", label=f"{names[k]} AP={ap:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.03), title=f"{split}: per-class PR curves")
    if any(r["ap"] is not None for r in report["per_class"]):
        ax.legend(fontsize=9)
    save(figure, output, "precision_recall")
    matrix = np.zeros((9, 9), dtype=int)
    np.add.at(matrix, (true.sum(1).astype(int), predicted.sum(1).astype(int)), 1)
    figure, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    ax.imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        if value:
            ax.text(col, row, str(value), ha="center", va="center", color="white" if value > matrix.max()/2 else "black")
    ax.set(title=f"{split}: label counts", xlabel="Predicted count", ylabel="True count", xticks=range(9), yticks=range(9))
    save(figure, output, "label_count_confusion")
    masks = {"correct": (true == predicted).all(1), "missed": (true > predicted).any(1),
             "false_positive": (predicted > true).any(1)}
    # First four sorted IDs per group: deterministic, no manual cherry-picking.
    choices = {group: np.flatnonzero(mask)[np.argsort(ids[mask])][:4].tolist() for group, mask in masks.items()}
    write_json(source / "case_selection.json", {"rule": "first four ascending sample IDs per group; groups may overlap",
               "sample_ids": {group: [int(ids[i]) for i in indices] for group, indices in choices.items()}})
    cache_ids = np.load(root / "cache" / f"{split}_ids.npy")
    images = np.load(root / "cache" / f"{split}_whole.npy", mmap_mode="r")
    lookup = {int(i): row for row, i in enumerate(cache_ids)}
    figure, axes = plt.subplots(3, 4, figsize=(14, 11), layout="constrained")
    label_text = lambda row: ",".join(map(str, np.flatnonzero(row))) or "empty"
    for row, (group, indices) in enumerate(choices.items()):
        for col in range(4):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(indices):
                ax.set_title(f"{group}: no additional example", fontsize=9)
                continue
            i = indices[col]
            ax.imshow(images[lookup[int(ids[i])], 0], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{group} | map {ids[i]}\nTrue: {label_text(true[i])}; Pred: {label_text(predicted[i])}", fontsize=9)
            probs = " ".join(f"{v:.2f}" for v in probabilities[i])
            ax.text(.5, -.04, f"p[0:8]: {probs}", transform=ax.transAxes, ha="center", fontsize=6)
    figure.suptitle(f"{split}: complete map inputs and label predictions")
    save(figure, output, "prediction_examples")


def budget_comparison(root, split, rows):
    if not rows:
        return
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    x = np.arange(len(rows))
    for ax, keys, title in ((axes[0], ("macro_f1", "micro_f1", "subset_accuracy"), "Classification"),
                             (axes[1], ("ContainmentNDCG@10", "mAP_full"), "Retrieval")):
        width = .8 / len(keys)
        for index, key in enumerate(keys):
            values = [row.get(key) if row.get(key) is not None else np.nan for row in rows]
            ax.bar(x + (index - (len(keys)-1)/2)*width, values, width, label=key)
        ax.set_xticks(x, [row["method"].replace("supervised_", "") for row in rows])
        ax.set(ylim=(0, 1.08), title=title, xlabel="Training label budget")
        ax.legend(fontsize=8)
    figure.suptitle(f"{split}: selected classifiers, hard-label retrieval")
    save(figure, Path(root) / "figures", f"{split}_budget_comparison")
