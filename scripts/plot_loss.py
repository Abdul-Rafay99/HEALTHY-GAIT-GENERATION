"""Plot training and validation loss from the saved CSV history."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_history(history_path: Path):
    epochs = []
    train_loss = []
    val_loss = []

    with history_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))

    return epochs, train_loss, val_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot gait autoencoder loss curves")
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("logs/training_history.csv"),
        help="Path to the saved CSV history file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/loss_curve.png"),
        help="Where to save the plot image",
    )
    args = parser.parse_args()

    if not args.history.exists():
        raise FileNotFoundError(f"History file not found: {args.history}")

    epochs, train_loss, val_loss = read_history(args.history)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    plt.plot(epochs, val_loss, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Gait Autoencoder Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
