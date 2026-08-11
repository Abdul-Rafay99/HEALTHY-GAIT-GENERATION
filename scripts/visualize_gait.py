"""Visualize a gait sequence as a clear human-like stick figure animation.

The dataset stores raw world coordinates and also provides a normalized,
human-centered representation in `normalization.py`. The normalized pose is
much easier to read visually because it removes the large global translation
and keeps the body centered.

By default this script:
- loads one motion sequence from the dataset,
- converts it to the normalized pose space,
- projects the skeleton into a side view,
- and saves a GIF or a folder of PNG frames.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

from normalization import normalize_skeletons  # type: ignore


SKELETON_BONES = [
    (2, 5),
    (5, 8),
    (8, 11),
    (1, 4),
    (4, 7),
    (7, 10),
    (15, 12),
    (12, 9),
    (9, 6),
    (6, 3),
    (3, 0),
    (17, 19),
    (19, 21),
    (16, 18),
    (18, 20),
    (10, 11),
    (1, 2),
    (13, 14),
]


VIEW_AXES = {
    "side": (2, 1),   # forward vs up: makes walking direction read left-to-right
    "front": (0, 1),  # left-right vs up
    "top": (0, 2),    # left-right vs forward
}


def load_motion(sample_id: str) -> np.ndarray:
    motion_path = DATASET_DIR / "joints_unscaled" / f"{sample_id}.npy"
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion = np.load(motion_path)
    if motion.ndim != 3 or motion.shape[1] != 22 or motion.shape[2] != 3:
        raise ValueError(f"Expected motion shape [T, 22, 3], got {motion.shape}")
    return motion


def get_sample_id(index: int | None, sample_id: str | None, split: str) -> str:
    if sample_id is not None:
        return sample_id

    split_path = DATASET_DIR / f"{split}.txt"
    ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"No sample ids found in {split_path}")

    if index is None:
        return ids[0]

    if index < 0 or index >= len(ids):
        raise IndexError(f"Index {index} is out of range for split '{split}' with {len(ids)} samples")

    return ids[index]


def normalize_motion(motion: np.ndarray) -> np.ndarray:
    """Return the normalized pose portion [T, 22, 3]."""
    motion_tensor = torch.tensor(motion[None], dtype=torch.float32)
    normalized = normalize_skeletons(motion_tensor)[0].detach().cpu().numpy()
    return normalized[:, 2:, :]


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


def render_gif(projected_frames: np.ndarray, output_path: Path, fps: int) -> None:
    x_min, x_max, y_min, y_max = compute_limits(projected_frames, zoom=0.85)

    fig, ax = plt.subplots(figsize=(6.5, 8.0))
    style_axes(ax, "side", x_min, x_max, y_min, y_max)

    bone_lines = []
    for bone_index, _ in enumerate(SKELETON_BONES):
        color = "#1f77b4" if bone_index < 11 else "#ff7f0e"
        line, = ax.plot([], [], linewidth=4, color=color, solid_capstyle="round")
        bone_lines.append(line)

    joints = ax.scatter([], [], s=45, color="#111111", zorder=4)
    frame_label = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=12, va="top")

    def init():
        for line in bone_lines:
            line.set_data([], [])
        joints.set_offsets(np.empty((0, 2)))
        frame_label.set_text("")
        return bone_lines + [joints, frame_label]

    def update(frame_index: int):
        skeleton = projected_frames[frame_index]
        xs = skeleton[:, 0]
        ys = skeleton[:, 1]

        for line, (start, end) in zip(bone_lines, SKELETON_BONES):
            line.set_data([xs[start], xs[end]], [ys[start], ys[end]])

        joints.set_offsets(np.column_stack([xs, ys]))
        frame_label.set_text(f"Frame {frame_index + 1}/{len(projected_frames)}")
        return bone_lines + [joints, frame_label]

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


def render_frames(projected_frames: np.ndarray, output_dir: Path) -> None:
    x_min, x_max, y_min, y_max = compute_limits(projected_frames, zoom=0.85)
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_index, skeleton in enumerate(projected_frames):
        fig, ax = plt.subplots(figsize=(6.5, 8.0))
        style_axes(ax, "side", x_min, x_max, y_min, y_max)

        xs = skeleton[:, 0]
        ys = skeleton[:, 1]
        ax.scatter(xs, ys, s=45, color="#111111", zorder=4)

        for bone_index, (start, end) in enumerate(SKELETON_BONES):
            color = "#1f77b4" if bone_index < 11 else "#ff7f0e"
            ax.plot([xs[start], xs[end]], [ys[start], ys[end]], linewidth=4, color=color, solid_capstyle="round")

        ax.text(0.02, 0.96, f"Frame {frame_index + 1}/{len(projected_frames)}", transform=ax.transAxes, fontsize=12, va="top")
        fig.tight_layout()
        fig.savefig(output_dir / f"frame_{frame_index:04d}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a gait sequence as a normalized skeleton animation")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split to use")
    parser.add_argument("--index", type=int, default=0, help="Zero-based index in the split file")
    parser.add_argument("--sample-id", type=str, default=None, help="Explicit sample id to load")
    parser.add_argument("--output", type=Path, default=Path("outputs/gait_animation.gif"), help="Output GIF path")
    parser.add_argument("--fps", type=int, default=12, help="Frames per second for the GIF")
    parser.add_argument("--stride", type=int, default=2, help="Keep every n-th frame to make motion easier to see")
    parser.add_argument("--view", type=str, default="side", choices=sorted(VIEW_AXES), help="Projection view for the skeleton")
    parser.add_argument("--raw", action="store_true", help="Use raw world coordinates instead of normalized pose")
    parser.add_argument("--frames-only", action="store_true", help="Save PNG frames instead of a GIF")
    parser.add_argument("--frames-dir", type=Path, default=Path("outputs/gait_frames"), help="Directory for PNG frames")
    args = parser.parse_args()

    sample_id = get_sample_id(args.index, args.sample_id, args.split)
    motion = load_motion(sample_id)
    print(f"Loaded sample {sample_id} with shape {motion.shape}")

    if args.raw:
        pose = motion
    else:
        pose = normalize_motion(motion)

    pose = pose[:: max(args.stride, 1)]
    projected = project_pose(pose, args.view)

    if args.frames_only:
        render_frames(projected, args.frames_dir)
        print(f"Saved frames to {args.frames_dir}")
    else:
        render_gif(projected, args.output, args.fps)
        print(f"Saved animation to {args.output}")


if __name__ == "__main__":
    main()
