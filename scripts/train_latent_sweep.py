"""Train the gait autoencoder across a sweep of latent dimensions.

Runs a fixed number of epochs for each size in LATENT_SIZES, sequentially,
each into its own checkpoint/history path so results don't collide or get
concatenated together the way a single shared history file would.

Usage:
    python scripts/train_latent_sweep.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_autoencoder import TrainConfig, fit

LATENT_SIZES = [512, 128, 64]
EPOCHS_PER_RUN = 100


def main() -> None:
    results = {}

    for latent_dim in LATENT_SIZES:
        run_name = f"latent{latent_dim}"
        print(f"\n{'=' * 60}")
        print(f"Starting run: {run_name} (latent_dim={latent_dim}, epochs={EPOCHS_PER_RUN})")
        print(f"{'=' * 60}\n")

        config = TrainConfig(
            latent_dim=latent_dim,
            epochs=EPOCHS_PER_RUN,
            checkpoint_dir=PROJECT_ROOT / "checkpoints" / run_name,
            history_file=PROJECT_ROOT / "logs" / f"{run_name}_history.csv",
        )

        try:
            history = fit(config)
            results[run_name] = history
            print(f"\nFinished {run_name}: best_val_loss={history['best_val_loss']:.6f}")
        except Exception:
            print(f"\nRun {run_name} FAILED:")
            traceback.print_exc()
            results[run_name] = None

    print(f"\n{'=' * 60}")
    print("Sweep complete. Summary:")
    print(f"{'=' * 60}")
    for run_name, history in results.items():
        if history is None:
            print(f"  {run_name}: FAILED")
        else:
            print(f"  {run_name}: best_val_loss={history['best_val_loss']:.6f}")


if __name__ == "__main__":
    main()
