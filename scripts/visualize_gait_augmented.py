"""Generate one original gait visualization and three augmented visualizations.

This script uses the dataset's normalization utilities to keep the skeleton
human-centered, then denormalizes the sequence back into global coordinates so
the walking motion remains visible. Each sample produces four GIFs:

- original
- parkinsonian_gait
- spatial_asymmetry
- postural_shift

Use `--all-split` to render every sample in a split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from normalization import denormalize_skeletons, get_start_poses, normalize_skeletons  # type: ignore


JOINT_NAMES = [
    "PELVIS",
    "HIP_RIGHT",
    "HIP_LEFT",
    "NAVEL",
    "KNEE_RIGHT",
    "KNEE_LEFT",
    "CHEST_LOWER",
    "ANKLE_RIGHT",
    "ANKLE_LEFT",
    "CHEST_UPPER",
    "TOES_RIGHT",
    "TOES_LEFT",
    "NECK",
    "CLAVICLE_RIGHT",
    "CLAVICLE_LEFT",
    "HEAD",
    "SHOULDER_RIGHT",
    "SHOULDER_LEFT",
    "ELBOW_RIGHT",
    "ELBOW_LEFT",
    "WRIST_RIGHT",
    "WRIST_LEFT",
]


SIDE_BONES = [
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (9, 13), (13, 16), (16, 18), (18, 20),
    (9, 14), (14, 17), (17, 19), (19, 21),
]


VIEW_AXES = {
    "side": (2, 1),
    "front": (0, 1),
    "top": (0, 2),
}

POSE_OFFSET = 2
BODY_JOINTS = slice(POSE_OFFSET, None)


def load_motion(sample_id: str) -> np.ndarray:
    motion_path = DATASET_DIR / "joints_unscaled" / f"{sample_id}.npy"
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion = np.load(motion_path)
    if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
        raise ValueError(f"Expected motion shape [T, 22, 3], got {motion.shape}")
    return motion.astype(np.float32)


def get_sample_ids(split: str) -> list[str]:
    split_path = DATASET_DIR / f"{split}.txt"
    ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"No sample ids found in {split_path}")
    return ids


def get_sample_id(index: int | None, sample_id: str | None, split: str) -> str:
    if sample_id is not None:
        return sample_id

    sample_ids = get_sample_ids(split)
    if index is None:
        return sample_ids[0]

    if index < 0 or index >= len(sample_ids):
        raise IndexError(f"Index {index} is out of range for split '{split}' with {len(sample_ids)} samples")

    return sample_ids[index]


def normalize_motion(motion: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    motion_tensor = torch.tensor(motion[None], dtype=torch.float32)
    normalized = normalize_skeletons(motion_tensor)
    startpos, startorient = get_start_poses(motion_tensor)
    return (
        normalized[0].detach().cpu().numpy(),
        startpos.detach().cpu().numpy(),
        startorient.detach().cpu().numpy(),
    )


def denormalize_motion(normalized: np.ndarray, startpos: np.ndarray, startorient: np.ndarray) -> np.ndarray:
    normalized_tensor = torch.tensor(normalized[None], dtype=torch.float32)
    startpos_tensor = torch.tensor(startpos, dtype=torch.float32)
    startorient_tensor = torch.tensor(startorient, dtype=torch.float32)
    global_motion = denormalize_skeletons(normalized_tensor, startpos_tensor, startorient_tensor)
    return global_motion[0].detach().cpu().numpy().astype(np.float32)


def copy_normalized(normalized: np.ndarray) -> np.ndarray:
    return np.array(normalized, copy=True)


def pose_view(normalized: np.ndarray) -> np.ndarray:
    return normalized[:, BODY_JOINTS, :]


def scale_chain(pose: np.ndarray, parent: int, chain: list[int], factors: tuple[float, float, float]) -> None:
    for joint in chain:
        offset = pose[:, joint, :] - pose[:, parent, :]
        pose[:, joint, :] = pose[:, parent, :] + offset * np.array(factors, dtype=np.float32)
        parent = joint


def rotate_subset_x(pose: np.ndarray, joints: list[int], center_joint: int, angle_deg: float) -> None:
    theta = np.deg2rad(angle_deg)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    center = pose[:, center_joint:center_joint + 1, :]
    rel = pose[:, joints, :] - center
    y = rel[:, :, 1]
    z = rel[:, :, 2]
    rel[:, :, 1] = cos_theta * y - sin_theta * z
    rel[:, :, 2] = sin_theta * y + cos_theta * z
    pose[:, joints, :] = center + rel


def add_tremor(pose: np.ndarray, joints: list[int], amplitude: float = 0.006, frequency_hz: float = 5.0) -> None:
    frame_times = np.arange(pose.shape[0], dtype=np.float32) / 20.0
    for offset_index, joint in enumerate(joints):
        phase = offset_index * np.pi / 3.0
        tremor_y = amplitude * np.sin(2.0 * np.pi * frequency_hz * frame_times + phase)
        tremor_z = amplitude * np.cos(2.0 * np.pi * frequency_hz * frame_times + phase)
        pose[:, joint, 1] += tremor_y
        pose[:, joint, 2] += tremor_z


def flatten_feet(pose: np.ndarray, side: str, factor: float = 0.82) -> None:
    if side == "right":
        ankle, toe = 7, 10
    else:
        ankle, toe = 8, 11
    pose[:, toe, :] = pose[:, ankle, :] + (pose[:, toe, :] - pose[:, ankle, :]) * np.array([1.0, 0.85, factor], dtype=np.float32)


def parkinsonian_gait(normalized: np.ndarray) -> np.ndarray:
    augmented = copy_normalized(normalized)
    pose = pose_view(augmented)

    # Keep the global trajectory stable; shorten the visible stride through the lower-limb pose.
    augmented[:, 0, 2] *= 0.82
    augmented[:, 0, 0] *= 0.96
    augmented[:, 0, 1] *= 0.96

    # Smaller, shuffly lower-body motion.
    scale_chain(pose, 0, [1, 4, 7, 10], (1.0, 0.90, 0.68))
    scale_chain(pose, 0, [2, 5, 8, 11], (1.0, 0.90, 0.68))
    flatten_feet(pose, "right", factor=0.62)
    flatten_feet(pose, "left", factor=0.62)

    # Reduced arm swing.
    scale_chain(pose, 9, [13, 16, 18, 20], (1.0, 0.95, 0.45))
    scale_chain(pose, 9, [14, 17, 19, 21], (1.0, 0.95, 0.45))

    # Low-amplitude tremor in the forearms and hands.
    add_tremor(pose, [18, 19, 20, 21], amplitude=0.0045, frequency_hz=5.0)

    return augmented


def spatial_asymmetry(normalized: np.ndarray) -> np.ndarray:
    augmented = copy_normalized(normalized)
    pose = pose_view(augmented)

    # Stronger reduction on the right leg and right arm.
    scale_chain(pose, 1, [4, 7, 10], (1.0, 0.90, 0.68))
    scale_chain(pose, 9, [16, 18, 20], (1.0, 0.95, 0.78))

    # Keep the left side mostly intact, but make the asymmetry more readable with a mild right hip/knee reduction.
    scale_chain(pose, 2, [5, 8, 11], (1.0, 0.96, 0.93))

    return augmented


def postural_shift(normalized: np.ndarray) -> np.ndarray:
    augmented = copy_normalized(normalized)
    pose = pose_view(augmented)

    # Forward flexion of the trunk.
    upper_body = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    rotate_subset_x(pose, upper_body, center_joint=0, angle_deg=15.0)

    # Slight compensatory hip/knee flexion.
    scale_chain(pose, 0, [1, 4, 7, 10], (1.0, 0.94, 0.90))
    scale_chain(pose, 0, [2, 5, 8, 11], (1.0, 0.94, 0.90))

    return augmented


def identity_variant(normalized: np.ndarray) -> np.ndarray:
    return copy_normalized(normalized)


def project_pose(pose: np.ndarray, view: str) -> np.ndarray:
    if view not in VIEW_AXES:
        raise ValueError(f"Unknown view '{view}'. Expected one of {sorted(VIEW_AXES)}")
    x_idx, y_idx = VIEW_AXES[view]
    return pose[:, :, [x_idx, y_idx]]


def compute_limits(projected_frames: np.ndarray, zoom: float) -> tuple[float, float, float, float]:
    flat = projected_frames.reshape(-1, 2)
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    center = (mins + maxs) / 2.0
    half_range = ((maxs - mins).max() / 2.0) * zoom
    half_range = max(float(half_range), 1e-3)
    return (
        float(center[0] - half_range),
        float(center[0] + half_range),
        float(center[1] - half_range),
        float(center[1] + half_range),
    )


def style_axes(ax, view: str, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#fbfbf7")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(f"Gait movement - {view} view", fontsize=14, pad=14)


def render_gif(projected_frames: np.ndarray, output_path: Path, fps: int, view: str, title: str, zoom: float = 0.85) -> None:
    x_min, x_max, y_min, y_max = compute_limits(projected_frames, zoom=zoom)

    fig, ax = plt.subplots(figsize=(6.8, 8.6))
    style_axes(ax, view, x_min, x_max, y_min, y_max)

    bone_lines = []
    for bone_index, _ in enumerate(SIDE_BONES):
        color = "#1f77b4" if bone_index < 12 else "#ff7f0e"
        line, = ax.plot([], [], linewidth=4.5, color=color, solid_capstyle="round")
        bone_lines.append(line)

    joints = ax.scatter([], [], s=48, color="#111111", zorder=4)
    trail_line, = ax.plot([], [], linewidth=2.5, color="#666666", alpha=0.65)
    frame_label = ax.text(0.02, 0.96, title, transform=ax.transAxes, fontsize=12, va="top")
    step_label = ax.text(0.02, 0.90, "", transform=ax.transAxes, fontsize=11, va="top")

    trail: list[np.ndarray] = []

    def init():
        for line in bone_lines:
            line.set_data([], [])
        joints.set_offsets(np.empty((0, 2)))
        trail_line.set_data([], [])
        frame_label.set_text(title)
        step_label.set_text("")
        return bone_lines + [joints, trail_line, frame_label, step_label]

    def update(frame_index: int):
        skeleton = projected_frames[frame_index]
        xs = skeleton[:, 0]
        ys = skeleton[:, 1]

        for line, (start, end) in zip(bone_lines, SIDE_BONES):
            line.set_data([xs[start], xs[end]], [ys[start], ys[end]])

        joints.set_offsets(np.column_stack([xs, ys]))
        trail.append(skeleton[0])
        trail_arr = np.asarray(trail)
        trail_line.set_data(trail_arr[:, 0], trail_arr[:, 1])
        frame_label.set_text(title)
        step_label.set_text(f"Frame {frame_index + 1}/{len(projected_frames)}")
        return bone_lines + [joints, trail_line, frame_label, step_label]

    animation = FuncAnimation(
        fig,
        update,
        frames=len(projected_frames),
        init_func=init,
        interval=max(1, 1000 // max(fps, 1)),
        blit=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def render_frames(projected_frames: np.ndarray, output_dir: Path, view: str, title: str, zoom: float = 0.85) -> None:
    x_min, x_max, y_min, y_max = compute_limits(projected_frames, zoom=zoom)
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_index, skeleton in enumerate(projected_frames):
        fig, ax = plt.subplots(figsize=(6.8, 8.6))
        style_axes(ax, view, x_min, x_max, y_min, y_max)

        xs = skeleton[:, 0]
        ys = skeleton[:, 1]
        ax.scatter(xs, ys, s=48, color="#111111", zorder=4)
        for bone_index, (start, end) in enumerate(SIDE_BONES):
            color = "#1f77b4" if bone_index < 12 else "#ff7f0e"
            ax.plot([xs[start], xs[end]], [ys[start], ys[end]], linewidth=4.5, color=color, solid_capstyle="round")

        ax.text(0.02, 0.96, title, transform=ax.transAxes, fontsize=12, va="top")
        ax.text(0.02, 0.90, f"Frame {frame_index + 1}/{len(projected_frames)}", transform=ax.transAxes, fontsize=11, va="top")
        fig.tight_layout()
        fig.savefig(output_dir / f"frame_{frame_index:04d}.png", dpi=180)
        plt.close(fig)


VariantFn = Callable[[np.ndarray], np.ndarray]


VARIANTS: list[tuple[str, str, VariantFn]] = [
    ("original", "Original", identity_variant),
    ("parkinsonian_gait", "Parkinsonian gait", parkinsonian_gait),
    ("spatial_asymmetry", "Spatial asymmetry", spatial_asymmetry),
    ("postural_shift", "Postural shift", postural_shift),
]


def build_global_motion(motion: np.ndarray, augmentation_fn: VariantFn) -> np.ndarray:
    normalized, startpos, startorient = normalize_motion(motion)
    augmented = augmentation_fn(normalized)
    return denormalize_motion(augmented, startpos, startorient)


def save_variants_for_sample(
    sample_id: str,
    output_dir: Path,
    fps: int,
    stride: int,
    view: str,
    frames_only: bool,
) -> None:
    motion = load_motion(sample_id)
    print(f"Loaded sample {sample_id} with shape {motion.shape}")

    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    for variant_name, title, transform in VARIANTS:
        global_motion = build_global_motion(motion, transform)
        sampled_motion = global_motion[:: max(stride, 1)]
        projected = project_pose(sampled_motion, view)
        if frames_only:
            render_frames(projected, sample_dir / variant_name, view, title)
        else:
            render_gif(projected, sample_dir / f"{variant_name}.gif", fps, view, title)
        print(f"Saved {variant_name} for sample {sample_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate original and augmented gait visualizations")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split to use")
    parser.add_argument("--index", type=int, default=0, help="Zero-based index in the split file")
    parser.add_argument("--sample-id", type=str, default=None, help="Explicit sample id to load")
    parser.add_argument("--all-split", action="store_true", help="Render every sample in the selected split")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/augmented_gifs"), help="Directory for GIF outputs")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the GIFs")
    parser.add_argument("--stride", type=int, default=2, help="Keep every n-th frame to make motion easier to see")
    parser.add_argument("--view", type=str, default="side", choices=sorted(VIEW_AXES), help="Projection view for the skeleton")
    parser.add_argument("--frames-only", action="store_true", help="Save PNG frames instead of GIFs")
    args = parser.parse_args()

    if args.all_split:
        for sample_id in get_sample_ids(args.split):
            save_variants_for_sample(sample_id, args.output_dir, args.fps, args.stride, args.view, args.frames_only)
    else:
        sample_id = get_sample_id(args.index, args.sample_id, args.split)
        save_variants_for_sample(sample_id, args.output_dir, args.fps, args.stride, args.view, args.frames_only)


if __name__ == "__main__":
    main()
