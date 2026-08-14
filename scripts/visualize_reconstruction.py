"""Visualize ground truth vs. model reconstruction, both in a root-centered view.

Every frame - ground truth and reconstruction alike - is expressed relative to
the body's own root (pelvis) and facing direction, via the dataset's
normalize_skeletons utility. The pelvis therefore sits at the origin in every
frame in both panels: the skeleton walks in place instead of translating
across the screen, so joint articulation is visible on its own, without being
confounded by forward travel, camera framing, or the trajectory-reconstruction
math in denormalize_skeletons (neither panel needs it).

Reconstruction is produced by feeding the model raw motion in non-overlapping
windows matching its trained window size, exactly as it saw data during
training, then concatenating the raw reconstructed windows and root-centering
the result with the *same* root_centered_pose() function used for ground
truth - not a parallel reimplementation.

The full recorded sequence is rendered (as many whole model-windows as fit),
not a trimmed excerpt.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch
import torch.serialization

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from normalization import normalize_skeletons  # type: ignore
from src.models.encoder.gait_encoder import GaitEncoder
from src.models.decoder.gait_decoder import GaitDecoder


JOINT_NAMES = [
    "PELVIS", "HIP_RIGHT", "HIP_LEFT", "NAVEL", "KNEE_RIGHT", "KNEE_LEFT",
    "CHEST_LOWER", "ANKLE_RIGHT", "ANKLE_LEFT", "CHEST_UPPER", "TOES_RIGHT",
    "TOES_LEFT", "NECK", "CLAVICLE_RIGHT", "CLAVICLE_LEFT", "HEAD",
    "SHOULDER_RIGHT", "SHOULDER_LEFT", "ELBOW_RIGHT", "ELBOW_LEFT",
    "WRIST_RIGHT", "WRIST_LEFT",
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

# Per-panel canvas budget for the side-by-side figure.
PANEL_CANVAS_AREA_SQIN = 6.8 * 8.6
MIN_FIGURE_ASPECT = 0.45
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


def root_centered_pose(motion: np.ndarray) -> np.ndarray:
    """Root-center and body-orient every frame independently.

    normalize_skeletons subtracts each frame's own pelvis position from every
    joint, then rotates into a frame built from that instant's hip/shoulder
    vectors. The result: joint 0 (PELVIS) is exactly (0, 0, 0) in every frame,
    and every other joint's position reflects only true articulation relative
    to the body's own root and facing direction - no translation, no turning
    of the whole body, none of the world-frame trajectory. Used identically
    for ground truth and for the model's reconstructed motion below, so both
    panels are put through the exact same transform.
    """
    motion_tensor = torch.tensor(motion[None], dtype=torch.float32)
    normalized = normalize_skeletons(motion_tensor)[0].detach().cpu().numpy()
    return normalized[:, BODY_JOINTS, :]  # [T, 22, 3]


def load_model(ckpt_path: str, device: torch.device):
    print(f"Loading checkpoint from {ckpt_path}...")
    with torch.serialization.safe_globals([pathlib.PosixPath]):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = checkpoint.get("config", {})
    latent_dim = cfg.get("latent_dim", 256)
    time_steps = cfg.get("output_time_steps", 30)
    in_channels = cfg.get("input_channels", 3)
    out_channels = cfg.get("output_channels", 3)
    use_normalized_input = cfg.get("use_normalized_input", False)
    # Derived the same way fit() derives it, not read from the checkpoint's
    # saved "output_joints" field - that field is a stale TrainConfig default
    # (always 22), never actually synced to what use_normalized_input made
    # the model's real joint count at construction time.
    out_joints = 24 if use_normalized_input else 22

    encoder = GaitEncoder(input_joints=out_joints, input_channels=in_channels, latent_dim=latent_dim).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()

    decoder = GaitDecoder(
        latent_dim=latent_dim,
        output_time_steps=time_steps,
        output_joints=out_joints,
        output_channels=out_channels,
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    decoder.eval()

    return encoder, decoder, time_steps, use_normalized_input


def reconstruct_root_centered_pose(
    motion: np.ndarray,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    device: torch.device,
    time_steps: int,
    use_normalized_input: bool,
) -> np.ndarray:
    """Run the model over `motion` in non-overlapping time_steps-frame windows
    (matching how it was trained), then root-center the reconstruction with
    the same root_centered_pose() used for ground truth.

    If the model was trained on raw motion (use_normalized_input=False), its
    output is raw world-frame motion, so it goes through root_centered_pose()
    directly. If it was trained on the normalized representation, its output
    is already local/root-relative, so the body-joint slice is used as-is.
    """
    recon_raw_windows = []
    recon_local_windows = []
    num_windows = motion.shape[0] // time_steps

    for w in range(num_windows):
        window_raw = motion[w * time_steps : (w + 1) * time_steps]
        with torch.no_grad():
            if use_normalized_input:
                window_normalized = normalize_skeletons(torch.tensor(window_raw[None], dtype=torch.float32)).to(device)
                latent = encoder(window_normalized)
                output = decoder(latent)  # [1, T, 24, 3] local representation
                recon_local_windows.append(output[0, :, BODY_JOINTS, :].detach().cpu().numpy())
            else:
                input_tensor = torch.tensor(window_raw[None], dtype=torch.float32).to(device)
                latent = encoder(input_tensor)
                output = decoder(latent)  # [1, T, 22, 3] raw representation
                recon_raw_windows.append(output[0].detach().cpu().numpy())

    if use_normalized_input:
        return np.concatenate(recon_local_windows, axis=0)
    recon_raw = np.concatenate(recon_raw_windows, axis=0)
    return root_centered_pose(recon_raw)


def articulation_energy(pose: np.ndarray) -> float:
    """Mean frame-to-frame displacement of the non-root joints (meters) - the
    root is excluded since it's pinned to zero by construction and would only
    dilute the signal."""
    if pose.shape[0] < 2:
        return 0.0
    diffs = np.diff(pose[:, 1:, :], axis=0)
    return float(np.linalg.norm(diffs, axis=-1).mean())


def project_pose(pose: np.ndarray, view: str) -> np.ndarray:
    if view not in VIEW_AXES:
        raise ValueError(f"Unknown view '{view}'. Expected one of {sorted(VIEW_AXES)}")
    x_idx, y_idx = VIEW_AXES[view]
    return pose[:, :, [x_idx, y_idx]]


def compute_limits(projected_frames: np.ndarray, zoom: float) -> tuple[float, float, float, float]:
    """Bounding box across every frame, for a single panel's own data only (not
    shared with the other panel, so one can't distort the other's framing).
    Since the pelvis is pinned to the origin, this is bounded by body size
    alone - never grows with sequence length."""
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
    """Size the two-panel figure to match this box's own aspect ratio (clamped)
    so the skeleton fills each panel instead of being letterboxed."""
    aspect = (x_max - x_min) / max(y_max - y_min, 1e-6)
    aspect = min(max(aspect, MIN_FIGURE_ASPECT), MAX_FIGURE_ASPECT)
    panel_height = math.sqrt(PANEL_CANVAS_AREA_SQIN / aspect)
    panel_width = PANEL_CANVAS_AREA_SQIN / panel_height
    return panel_width * 2, panel_height


def style_axes(ax, x_min: float, x_max: float, y_min: float, y_max: float, title: str) -> None:
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
    ax.set_title(title, fontsize=13, pad=14)


def render_comparison_gif(
    proj_orig: np.ndarray,
    proj_recon: np.ndarray,
    output_path: Path,
    fps: int,
    view: str,
    sample_id: str,
    zoom: float = 1.15,
) -> None:
    """Side-by-side GIF, each panel framed from only its own data."""
    orig_limits = compute_limits(proj_orig, zoom=zoom)
    recon_limits = compute_limits(proj_recon, zoom=zoom)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size_for_limits(*orig_limits))
    style_axes(ax1, *orig_limits, f"{sample_id} - Ground truth ({view} view)")
    style_axes(ax2, *recon_limits, f"{sample_id} - Reconstruction ({view} view)")

    def setup_skeleton(ax):
        bone_lines = []
        for bone_index, _ in enumerate(SIDE_BONES):
            color = "#1f77b4" if bone_index < 12 else "#ff7f0e"
            line, = ax.plot([], [], linewidth=4.5, color=color, solid_capstyle="round")
            bone_lines.append(line)
        joints = ax.scatter([], [], s=48, color="#111111", zorder=4)
        step_label = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=11, va="top")
        return bone_lines, joints, step_label

    lines_orig, joints_orig, label_orig = setup_skeleton(ax1)
    lines_recon, joints_recon, label_recon = setup_skeleton(ax2)

    def init():
        for line in lines_orig + lines_recon:
            line.set_data([], [])
        joints_orig.set_offsets(np.empty((0, 2)))
        joints_recon.set_offsets(np.empty((0, 2)))
        label_orig.set_text("")
        label_recon.set_text("")
        return lines_orig + lines_recon + [joints_orig, joints_recon, label_orig, label_recon]

    def update(frame_index: int):
        def update_ax(skeleton, lines, joints_scatter, label):
            xs = skeleton[:, 0]
            ys = skeleton[:, 1]
            for line, (start, end) in zip(lines, SIDE_BONES):
                line.set_data([xs[start], xs[end]], [ys[start], ys[end]])
            joints_scatter.set_offsets(np.column_stack([xs, ys]))
            label.set_text(f"Frame {frame_index + 1}/{len(proj_orig)}")

        update_ax(proj_orig[frame_index], lines_orig, joints_orig, label_orig)
        update_ax(proj_recon[frame_index], lines_recon, joints_recon, label_recon)
        return lines_orig + lines_recon + [joints_orig, joints_recon, label_orig, label_recon]

    animation = FuncAnimation(
        fig, update, frames=len(proj_orig), init_func=init,
        interval=max(1, 1000 // max(fps, 1)), blit=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_comparison_for_sample(
    sample_id: str,
    output_dir: Path,
    fps: int,
    stride: int,
    view: str,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    device: torch.device,
    time_steps: int,
    use_normalized_input: bool,
) -> None:
    motion = load_motion(sample_id)
    print(f"Loaded sample {sample_id} with shape {motion.shape} ({motion.shape[0]} frames)")

    num_windows = motion.shape[0] // time_steps
    if num_windows == 0:
        print(f"  Skipping {sample_id}: sequence shorter than the model's {time_steps}-frame window.")
        return
    usable_frames = num_windows * time_steps
    motion = motion[:usable_frames]

    orig_pose = root_centered_pose(motion)
    recon_pose = reconstruct_root_centered_pose(motion, encoder, decoder, device, time_steps, use_normalized_input)

    orig_energy = articulation_energy(orig_pose)
    recon_energy = articulation_energy(recon_pose)
    ratio = recon_energy / max(orig_energy, 1e-8)
    print(
        f"  articulation energy (mean frame-to-frame non-root joint displacement, meters): "
        f"ground truth={orig_energy:.4f}  reconstruction={recon_energy:.4f}  "
        f"(reconstruction retains {ratio:.1%})"
    )

    orig_sampled = orig_pose[:: max(stride, 1)]
    recon_sampled = recon_pose[:: max(stride, 1)]

    proj_orig = project_pose(orig_sampled, view)
    proj_recon = project_pose(recon_sampled, view)

    output_path = output_dir / f"{sample_id}_comparison.gif"
    render_comparison_gif(proj_orig, proj_recon, output_path, fps, view, sample_id)
    print(f"  Saved {orig_sampled.shape[0]} frames -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize root-centered ground truth vs. model reconstruction")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to use")
    parser.add_argument("--index", type=int, default=None, help="Zero-based index in the split file")
    parser.add_argument("--sample-id", type=str, default=None, help="Explicit sample id to load")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to render, starting from the top of the split (ignored if --sample-id is given)")
    parser.add_argument("--ckpt", type=str, default=str(PROJECT_ROOT / "checkpoints" / "best_autoencoder.pt"), help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "ground_truth", help="Directory for GIF outputs")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the GIF")
    parser.add_argument("--stride", type=int, default=1, help="Keep every n-th frame (1 = every frame)")
    parser.add_argument("--view", type=str, default="side", choices=sorted(VIEW_AXES), help="Projection view for the skeleton")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    try:
        encoder, decoder, time_steps, use_normalized_input = load_model(args.ckpt, device)
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint. \nDetails: {e}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_id is not None or args.index is not None:
        sample_ids = [get_sample_id(args.index, args.sample_id, args.split)]
    else:
        sample_ids = get_sample_ids(args.split)[: args.num_samples]

    for sample_id in sample_ids:
        save_comparison_for_sample(
            sample_id, args.output_dir, args.fps, args.stride, args.view,
            encoder, decoder, device, time_steps, use_normalized_input,
        )


if __name__ == "__main__":
    main()
