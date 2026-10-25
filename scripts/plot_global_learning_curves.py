#!/usr/bin/env python3
"""Plot saved train/validation losses for the 5%, 10%, and 50% global models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = (("global_5pct", "5%"), ("global_10pct", "10%"), ("global_50pct", "50%"))


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_history(path):
    with path.open(encoding="utf-8") as handle:
        history = json.load(handle)
    if not isinstance(history, list) or not history:
        raise ValueError(f"{path}: expected a nonempty list of epoch records")
    rows = []
    for index, row in enumerate(history, 1):
        try:
            epoch = row["epoch"]
            train, valid = float(row["train_loss"]), float(row["valid_loss"])
            if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
                raise ValueError("epoch must be a positive integer")
            if not all(math.isfinite(value) and value >= 0 for value in (train, valid)):
                raise ValueError("losses must be finite and nonnegative")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: invalid record {index}: {error}") from error
        rows.append((epoch, train, valid))
    rows.sort(key=lambda row: row[0])
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError(f"{path}: duplicate epochs")
    return rows


def draw_curve(ax, rows, label):
    from matplotlib.ticker import MaxNLocator

    epochs, train, valid = zip(*rows)
    best_epoch, _, best_loss = min(rows, key=lambda row: row[2])
    ax.plot(epochs, train, color="#2563eb", linewidth=1.8, label="Training loss")
    ax.plot(epochs, valid, color="#ea580c", linewidth=1.8, label="Validation loss")
    if len(rows) == 1:
        ax.scatter(epochs, train, color="#2563eb")
        ax.scatter(epochs, valid, color="#ea580c")
    ax.axvline(best_epoch, color="#64748b", linestyle="--", linewidth=1, alpha=0.8)
    ax.scatter([best_epoch], [best_loss], color="#ea580c", edgecolors="white", s=65, zorder=5)
    ax.set_title(f"Label budget: {label}\nBest validation: {best_loss:.6g} at epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.2)
    ax.legend(loc="best", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-experiment", default="artifacts/experiment_a_top4_opening_seed2026",
                        help="Experiment containing models/global_*pct/history.json; project-relative or absolute")
    parser.add_argument("--output-dir", help="Project-relative or absolute; default: <source>/figures/learning_curves")
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution (default: 180)")
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    source = project_path(args.source_experiment)
    output = project_path(args.output_dir) if args.output_dir else source / "figures" / "learning_curves"
    histories = {}
    # Validate every input before writing any figures. No model or GPU is needed.
    for method, _ in METHODS:
        path = source / "models" / method / "history.json"
        try:
            histories[method] = read_history(path)
        except (OSError, ValueError) as error:
            parser.error(str(error))

    import matplotlib
    matplotlib.use("Agg")  # Save figures on servers without a display.
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10, "axes.titlesize": 11, "savefig.facecolor": "white"}):
        figure, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
        for ax, (method, label) in zip(axes, METHODS):
            rows = histories[method]
            draw_curve(ax, rows, label)
            best = min(rows, key=lambda row: row[2])
            print(f"{method}: {len(rows)} recorded epochs; best epoch={best[0]}, "
                  f"best valid_loss={best[2]:.6g}; "
                  f"final train_loss={rows[-1][1]:.6g}, final valid_loss={rows[-1][2]:.6g}")
            single, single_ax = plt.subplots(figsize=(7, 5), layout="constrained")
            draw_curve(single_ax, rows, label)
            single.savefig(output / f"{method}_loss.png", dpi=args.dpi)
            plt.close(single)
        figure.suptitle("Supervised whole-map learning curves", fontsize=15)
        figure.text(0.5, 0.025, "Raw epoch losses (no smoothing). Training pairs are resampled; validation pairs are fixed.",
                    ha="center", fontsize=10, color="#475569")
        figure.tight_layout(rect=(0, 0.07, 1, 0.94))
        for suffix in ("png", "pdf"):
            figure.savefig(output / f"global_learning_curves.{suffix}", dpi=args.dpi)
        plt.close(figure)
    print(f"Saved 3 individual PNGs and combined PNG/PDF to: {output}")


if __name__ == "__main__":
    main()
