"""Training loop for the gait encoder-decoder autoencoder."""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import nullcontext
import csv
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
from src.models.decoder import GaitDecoder
from src.models.encoder import GaitEncoder


def resolve_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def configure_torch_for_device(device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


@dataclass
class TrainConfig:
    train_split: Path = DATASET_DIR / "train.txt"
    val_split: Path = DATASET_DIR / "val.txt"
    motion_dir: Path = DATASET_DIR / "joints_unscaled"
    computed_label_dir: Path = DATASET_DIR / "computed_labels_w30"
    window_size: int = 30
    batch_size: int = 64
    num_workers: int = field(default_factory=lambda: min(8, max(2, os.cpu_count() or 2)))
    latent_dim: int = 256
    input_channels: int = 3
    output_time_steps: int = 30
    output_joints: int = 22
    output_channels: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 3
    lr_scheduler_min_lr: float = 1e-6
    epochs: int = 200
    device: str = field(default_factory=lambda: os.getenv("TRAIN_DEVICE", resolve_default_device()))
    use_normalized_input: bool = False
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    history_file: Path = PROJECT_ROOT / "logs" / "training_history.csv"
    amp: bool = True
    log_every: int = 50


class LazyGaitWindowDataset(Dataset):
    """Load one gait window at a time instead of materializing the whole split.

    The original dataset class eagerly loads and normalizes every motion during
    initialization. That is correct, but it makes startup slow on laptops.
    This version keeps only the split index in memory and reads a single window
    on demand.
    """

    def __init__(self, split_path: Path, motion_dir: Path, window_size: int, use_normalized_input: bool) -> None:
        self.motion_dir = motion_dir
        self.window_size = window_size
        self.use_normalized_input = use_normalized_input
        self.sample_ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        self.motion_lengths = []
        self.window_index = []
        self._motion_cache = {}

        for motion_id in self.sample_ids:
            motion_path = self.motion_dir / f"{motion_id}.npy"
            motion = np.load(motion_path, mmap_mode="r")
            motion_length = int(motion.shape[0])
            self.motion_lengths.append(motion_length)

            num_windows = max(motion_length - self.window_size, 0)
            for start in range(num_windows):
                self.window_index.append((motion_id, start))

    def __len__(self) -> int:
        return len(self.window_index)

    def _load_motion(self, motion_id: str) -> np.ndarray:
        cached = self._motion_cache.get(motion_id)
        if cached is None:
            cached = np.load(self.motion_dir / f"{motion_id}.npy", mmap_mode="r")
            self._motion_cache[motion_id] = cached
        return cached

    def __getitem__(self, index: int):
        motion_id, start = self.window_index[index]
        motion = self._load_motion(motion_id)
        window = np.asarray(motion[start : start + self.window_size], dtype=np.float32)
        motion_tensor = torch.from_numpy(window)

        if self.use_normalized_input:
            from dataset.normalization import normalize_skeletons

            normalized = normalize_skeletons(motion_tensor.unsqueeze(0))[0]
            return motion_tensor, normalized, torch.empty(0)

        return motion_tensor, motion_tensor, torch.empty(0)


def build_dataloader(split_path: Path, config: TrainConfig, shuffle: bool) -> DataLoader:
    dataset = LazyGaitWindowDataset(split_path, config.motion_dir, config.window_size, config.use_normalized_input)
    device = torch.device(config.device)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=config.num_workers > 0,
    )


def train_one_epoch(
    encoder: nn.Module,
    decoder: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_normalized_input: bool,
    amp_enabled: bool,
    log_every: int,
) -> float:
    encoder.train()
    decoder.train()

    running_loss = 0.0
    total_batches = 0
    device_type = device.type
    use_amp = amp_enabled and device_type in {"cuda", "mps"}
    amp_dtype = get_amp_dtype(device) if use_amp else None

    for step, (motion, motion_normalized, _) in enumerate(loader, start=1):
        inputs = motion_normalized if use_normalized_input else motion
        targets = inputs

        inputs = inputs.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)

        amp_context = torch.autocast(device_type=device_type, dtype=amp_dtype) if use_amp else nullcontext()
        with amp_context:
            latent = encoder(inputs)
            reconstruction = decoder(latent)
            loss = criterion(reconstruction, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

        if log_every > 0 and step % log_every == 0:
            print(f"  batch {step:05d} | loss={loss.item():.6f}")

    return running_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(
    encoder: nn.Module,
    decoder: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_normalized_input: bool,
    amp_enabled: bool,
) -> float:
    encoder.eval()
    decoder.eval()

    running_loss = 0.0
    total_batches = 0

    amp_dtype = get_amp_dtype(device) if (amp_enabled and device.type in {"cuda", "mps"}) else None

    for motion, motion_normalized, _ in loader:
        inputs = motion_normalized if use_normalized_input else motion
        targets = inputs

        inputs = inputs.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)

        amp_context = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype is not None else nullcontext()
        with amp_context:
            latent = encoder(inputs)
            reconstruction = decoder(latent)
            loss = criterion(reconstruction, targets)

        running_loss += loss.item()
        total_batches += 1

    return running_loss / max(total_batches, 1)


def fit(config: TrainConfig) -> Dict[str, float]:
    device = torch.device(config.device)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.history_file.parent.mkdir(parents=True, exist_ok=True)

    configure_torch_for_device(device)

    train_loader = build_dataloader(config.train_split, config, shuffle=True)
    val_loader = build_dataloader(config.val_split, config, shuffle=False)

    output_joints = 24 if config.use_normalized_input else 22
    encoder = GaitEncoder(input_channels=config.input_channels, latent_dim=config.latent_dim).to(device)
    decoder = GaitDecoder(
        latent_dim=config.latent_dim,
        output_time_steps=config.output_time_steps,
        output_joints=output_joints,
        output_channels=config.output_channels,
    ).to(device)

    print(
        f"Using device={device.type}, train_windows={len(train_loader.dataset)}, val_windows={len(val_loader.dataset)}, "
        f"batch_size={config.batch_size}, amp={config.amp}, normalized={config.use_normalized_input}"
    )

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        min_lr=config.lr_scheduler_min_lr,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    history: Dict[str, float] = {}

    history_exists = config.history_file.exists()
    with config.history_file.open("a", newline="") as history_handle:
        writer = csv.DictWriter(history_handle, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        if not history_exists:
            writer.writeheader()

        for epoch in range(1, config.epochs + 1):
            train_loss = train_one_epoch(
                encoder,
                decoder,
                train_loader,
                optimizer,
                criterion,
                device,
                config.use_normalized_input,
                config.amp,
                config.log_every,
            )
            val_loss = evaluate(
                encoder,
                decoder,
                val_loader,
                criterion,
                device,
                config.use_normalized_input,
                config.amp,
            )

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"] = train_loss
            history["val_loss"] = val_loss
            history["lr"] = current_lr

            writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr})
            history_handle.flush()

            checkpoint = {
                "epoch": epoch,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "config": config.__dict__,
            }

            torch.save(checkpoint, config.checkpoint_dir / "latest_autoencoder.pt")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, config.checkpoint_dir / "best_autoencoder.pt")

            print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={current_lr:.2e}")

    history["best_val_loss"] = best_val_loss
    return history


if __name__ == "__main__":
    fit(TrainConfig())
