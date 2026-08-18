"""Training loop for the augmented-to-healthy gait denoising autoencoder.

Reuses the same GaitEncoder/GaitDecoder architecture as train_autoencoder.py
unchanged - only what feeds in as input vs. target differs. Input is one of
the 3 pathological augmentations (dataset/augmentations.py) applied to a real
window; target is the healthy (unaugmented) normalized representation of
that same window. Every real window is deterministically expanded into 3
training examples, one per augmentation, each epoch.
"""

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
import sys

if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from normalization import normalize_skeletons  # type: ignore
from src.data.augmentations import PATHOLOGICAL_AUGMENTATIONS
from src.models.decoder import GaitDecoder
from src.models.encoder import GaitEncoder
from src.training.train_autoencoder import (
    configure_torch_for_device,
    get_amp_dtype,
    resolve_default_device,
)


@dataclass
class DenoisingTrainConfig:
    train_split: Path = DATASET_DIR / "train.txt"
    val_split: Path = DATASET_DIR / "val.txt"
    motion_dir: Path = DATASET_DIR / "joints_unscaled"
    window_size: int = 30
    # Larger than train_autoencoder.py's default (64): the model is tiny
    # (~2M params) and the dataset is 3x larger here (one entry per
    # augmentation per window), so fewer/bigger steps trades nothing on a
    # GPU this size, and cuts per-epoch kernel-launch overhead.
    batch_size: int = 128
    # Each item does real CPU work now (augmentation on top of
    # normalize_skeletons), unlike plain reconstruction - a higher worker
    # cap keeps that off the critical path on a well-cored training machine.
    num_workers: int = field(default_factory=lambda: min(16, max(2, os.cpu_count() or 2)))
    latent_dim: int = 256
    input_channels: int = 3
    output_time_steps: int = 30
    output_joints: int = 24  # augmentations only operate on the normalized (24-joint) representation
    output_channels: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 3
    lr_scheduler_min_lr: float = 1e-6
    epochs: int = 200
    device: str = field(default_factory=lambda: os.getenv("TRAIN_DEVICE", resolve_default_device()))
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints" / "denoiser"
    history_file: Path = PROJECT_ROOT / "logs" / "denoiser_history.csv"
    amp: bool = True
    log_every: int = 50


class DenoisingGaitWindowDataset(Dataset):
    """Windows the same way the reconstruction dataset does, but each real
    window is expanded into 3 entries - one per pathological augmentation -
    so every window is deterministically paired with every augmentation type
    exactly once per epoch.

    Returns (augmented_input, healthy_target) pairs. The healthy target is
    always the normalized (root-centered) representation of the real,
    unaugmented window - never the augmented input's own possible variants.
    """

    def __init__(self, split_path: Path, motion_dir: Path, window_size: int) -> None:
        self.motion_dir = motion_dir
        self.window_size = window_size
        self.sample_ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        self.augmentation_names = list(PATHOLOGICAL_AUGMENTATIONS.keys())
        self.window_index = []
        self._motion_cache = {}

        for motion_id in self.sample_ids:
            motion_path = self.motion_dir / f"{motion_id}.npy"
            motion = np.load(motion_path, mmap_mode="r")
            motion_length = int(motion.shape[0])

            num_windows = max(motion_length - self.window_size, 0)
            for start in range(num_windows):
                for augmentation_name in self.augmentation_names:
                    self.window_index.append((motion_id, start, augmentation_name))

    def __len__(self) -> int:
        return len(self.window_index)

    def _load_motion(self, motion_id: str) -> np.ndarray:
        cached = self._motion_cache.get(motion_id)
        if cached is None:
            cached = np.load(self.motion_dir / f"{motion_id}.npy", mmap_mode="r")
            self._motion_cache[motion_id] = cached
        return cached

    def __getitem__(self, index: int):
        motion_id, start, augmentation_name = self.window_index[index]
        motion = self._load_motion(motion_id)
        window = np.asarray(motion[start : start + self.window_size], dtype=np.float32)
        motion_tensor = torch.from_numpy(window)

        healthy = normalize_skeletons(motion_tensor.unsqueeze(0))[0]  # [T, 24, 3]
        augment_fn = PATHOLOGICAL_AUGMENTATIONS[augmentation_name]
        augmented = augment_fn(healthy.detach().cpu().numpy())
        augmented_tensor = torch.from_numpy(augmented)

        return augmented_tensor, healthy


def build_dataloader(split_path: Path, config: DenoisingTrainConfig, shuffle: bool) -> DataLoader:
    dataset = DenoisingGaitWindowDataset(split_path, config.motion_dir, config.window_size)
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

    for step, (augmented, healthy) in enumerate(loader, start=1):
        inputs = augmented.to(device=device, dtype=torch.float32)
        targets = healthy.to(device=device, dtype=torch.float32)

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
    amp_enabled: bool,
) -> float:
    encoder.eval()
    decoder.eval()

    running_loss = 0.0
    total_batches = 0

    amp_dtype = get_amp_dtype(device) if (amp_enabled and device.type in {"cuda", "mps"}) else None

    for augmented, healthy in loader:
        inputs = augmented.to(device=device, dtype=torch.float32)
        targets = healthy.to(device=device, dtype=torch.float32)

        amp_context = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype is not None else nullcontext()
        with amp_context:
            latent = encoder(inputs)
            reconstruction = decoder(latent)
            loss = criterion(reconstruction, targets)

        running_loss += loss.item()
        total_batches += 1

    return running_loss / max(total_batches, 1)


def fit(config: DenoisingTrainConfig) -> Dict[str, float]:
    device = torch.device(config.device)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.history_file.parent.mkdir(parents=True, exist_ok=True)

    configure_torch_for_device(device)

    train_loader = build_dataloader(config.train_split, config, shuffle=True)
    val_loader = build_dataloader(config.val_split, config, shuffle=False)

    encoder = GaitEncoder(
        input_joints=config.output_joints,
        input_channels=config.input_channels,
        latent_dim=config.latent_dim,
    ).to(device)
    decoder = GaitDecoder(
        latent_dim=config.latent_dim,
        output_time_steps=config.output_time_steps,
        output_joints=config.output_joints,
        output_channels=config.output_channels,
    ).to(device)

    print(
        f"Using device={device.type}, train_windows={len(train_loader.dataset)}, val_windows={len(val_loader.dataset)}, "
        f"batch_size={config.batch_size}, amp={config.amp} (denoising: augmented -> healthy, 3x window expansion)"
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
                config.amp,
                config.log_every,
            )
            val_loss = evaluate(
                encoder,
                decoder,
                val_loader,
                criterion,
                device,
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

            torch.save(checkpoint, config.checkpoint_dir / "latest_denoiser.pt")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, config.checkpoint_dir / "best_denoiser.pt")

            print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={current_lr:.2e}")

    history["best_val_loss"] = best_val_loss
    return history


if __name__ == "__main__":
    fit(DenoisingTrainConfig())
