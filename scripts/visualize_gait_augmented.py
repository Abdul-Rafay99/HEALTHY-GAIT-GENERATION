"""Generate one original gait visualization and three augmented visualizations.

This script uses the dataset's normalization utilities to keep the skeleton
human-centered, then denormalizes the sequence back into global coordinates so
the walking motion remains visible. Each sample produces four GIFs:

- original
- parkinsonian_gait
- spatial_asymmetry
- postural_shift

Since gait is cyclic, only a handful of steps near the start of the sequence
are rendered (see `--steps`) instead of the full walk, which keeps the camera
zoomed in on the skeleton instead of the whole traversed distance.

Use `--all-split` to render every sample in a split.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from scipy.signal import find_peaks
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from normalization import denormalize_skeletons, get_start_poses, normalize_skeletons  # type: ignore
from src.data.augmentations import (
    BODY_JOINTS,
    POSE_OFFSET,
    copy_normalized,
    identity_variant,
    parkinsonian_gait,
    pose_view,
    postural_shift,
    spatial_asymmetry,
)


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

SOURCE_FPS = 20.0  # dataset/description.md: "Frames are recorded with 20Hz"
DEFAULT_STEPS_TO_SHOW = 4
STEP_BOUNDARY_MIN_FRAME_GAP = 5
FALLBACK_STEP_DURATION_S = 0.55
MIN_STEP_PROGRESS_M = 0.12  # minimum net forward displacement expected per real step

FIGURE_SIZE = (6.8, 8.6)
CANVAS_AREA_SQIN = FIGURE_SIZE[0] * FIGURE_SIZE[1]
MIN_FIGURE_ASPECT = 0.45  # x/y bounds so a near-flat or near-vertical clip doesn't produce a sliver figure
MAX_FIGURE_ASPECT = 1.8


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


def detect_step_boundaries(normalized: np.ndarray) -> np.ndarray:
    """Frame indices where the feet cross underneath the body, i.e. one boundary per step.

    Mirrors the (unused) step-detection in dataset/attribute_computation.py, but
    scales the height threshold to the sample's own foot-distance range instead
    of a fixed value, since it runs on the human-centered pose here rather than
    the raw joint data that script was tuned against.
    """
    pose = pose_view(normalized)
    foot_distance = np.abs(pose[:, 10, 2] - pose[:, 11, 2])  # TOES_RIGHT vs TOES_LEFT, forward axis
    threshold = max(float(foot_distance.max()) * 0.25, 1e-3)
    peaks, _ = find_peaks(-foot_distance, distance=STEP_BOUNDARY_MIN_FRAME_GAP, height=-threshold)
    return peaks


def select_step_window(motion: np.ndarray, num_steps: int) -> tuple[int, int]:
    """Pick a [start, end) frame range covering `num_steps` steps of real forward
    walking near the start of the sequence."""
    normalized, _, _ = normalize_motion(motion)
    boundaries = detect_step_boundaries(normalized)

    if len(boundaries) > num_steps:
        min_progress = MIN_STEP_PROGRESS_M * num_steps
        # Some recordings open with a settling/weight-shift period before real
        # forward walking begins, which still registers as foot-crossing "steps"
        # here - skip past those by requiring genuine net pelvis displacement
        # (in raw/world coordinates) across the candidate window.
        for i in range(len(boundaries) - num_steps):
            b_start, b_end = boundaries[i], boundaries[i + num_steps]
            progress = float(np.linalg.norm(motion[b_end, 0, :] - motion[b_start, 0, :]))
            if progress >= min_progress:
                return int(b_start), min(int(b_end) + 1, motion.shape[0])

        # Nothing met the progress bar - fall back to the first detected window anyway.
        start, end = int(boundaries[0]), int(boundaries[num_steps])
        return start, min(end + 1, motion.shape[0])

    # Too few clean footfalls detected (short or noisy sample) - fall back to a
    # fixed-duration window sized for the requested step count.
    fallback_frames = int(round(num_steps * FALLBACK_STEP_DURATION_S * SOURCE_FPS))
    return 0, min(fallback_frames, motion.shape[0])


def project_pose(pose: np.ndarray, view: str) -> np.ndarray:
    if view not in VIEW_AXES:
        raise ValueError(f"Unknown view '{view}'. Expected one of {sorted(VIEW_AXES)}")
    x_idx, y_idx = VIEW_AXES[view]
    return pose[:, :, [x_idx, y_idx]]


def compute_limits(projected_frames: np.ndarray, zoom: float) -> tuple[float, float, float, float]:
    # zoom scales the natural bounding box directly (no forced square), so keep it
    # >= 1.0 - the figure is sized to this box's own aspect ratio via
    # figure_size_for_limits, so there's no slack dimension left to absorb a crop.
    flat = projected_frames.reshape(-1, 2)
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)
    center = (mins + maxs) / 2.0
    half_x = max(float(maxs[0] - mins[0]) / 2.0, 1e-3) * zoom
    half_y = max(float(maxs[1] - mins[1]) / 2.0, 1e-3) * zoom
    return (
        float(center[0] - half_x),
        float(center[0] + half_x),
        float(center[1] - half_y),
        float(center[1] + half_y),
    )


def figure_size_for_limits(x_min: float, x_max: float, y_min: float, y_max: float) -> tuple[float, float]:
    """Size the figure to match this clip's own data aspect ratio (clamped) so the
    skeleton fills the canvas instead of being letterboxed inside a fixed shape."""
    aspect = (x_max - x_min) / max(y_max - y_min, 1e-6)
    aspect = min(max(aspect, MIN_FIGURE_ASPECT), MAX_FIGURE_ASPECT)
    height = math.sqrt(CANVAS_AREA_SQIN / aspect)
    width = CANVAS_AREA_SQIN / height
    return width, height


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


def render_gif(projected_frames: np.ndarray, output_path: Path, fps: int, view: str, title: str, zoom: float = 1.06) -> None:
    x_min, x_max, y_min, y_max = compute_limits(projected_frames, zoom=zoom)

    fig, ax = plt.subplots(figsize=figure_size_for_limits(x_min, x_max, y_min, y_max))
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


def render_frames(projected_frames: np.ndarray, output_dir: Path, view: str, title: str, zoom: float = 1.06) -> None:
    x_min, x_max, y_min, y_max = compute_limits(projected_frames, zoom=zoom)
    output_dir.mkdir(parents=True, exist_ok=True)
    figsize = figure_size_for_limits(x_min, x_max, y_min, y_max)

    for frame_index, skeleton in enumerate(projected_frames):
        fig, ax = plt.subplots(figsize=figsize)
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
    num_steps: int,
) -> None:
    motion = load_motion(sample_id)
    print(f"Loaded sample {sample_id} with shape {motion.shape}")

    start, end = select_step_window(motion, num_steps)
    motion = motion[start:end]
    print(f"Showing frames {start}:{end} (~{num_steps} steps) for sample {sample_id}")

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
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS_TO_SHOW, help="Number of gait steps to render near the start of the sequence (gait is cyclic, so 3-5 is usually enough)")
    args = parser.parse_args()

    if args.all_split:
        for sample_id in get_sample_ids(args.split):
            save_variants_for_sample(sample_id, args.output_dir, args.fps, args.stride, args.view, args.frames_only, args.steps)
    else:
        sample_id = get_sample_id(args.index, args.sample_id, args.split)
        save_variants_for_sample(sample_id, args.output_dir, args.fps, args.stride, args.view, args.frames_only, args.steps)


if __name__ == "__main__":
    main()
