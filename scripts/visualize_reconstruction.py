"""Visualize a side-by-side comparison of the original gait and the model reconstruction.

This script uses the robust normalization and rendering utilities from the 
augmented visualizer. It truncates the motion to the model's expected window 
size, passes it through the encoder-decoder, and denormalizes both sequences 
back to global coordinates for an accurate side-by-side visual comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pathlib
import torch.serialization

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch
import torch.nn as nn

# Path setup to ensure dataset and src modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from normalization import denormalize_skeletons, get_start_poses, normalize_skeletons  # type: ignore
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


def project_pose(pose: np.ndarray, view: str) -> np.ndarray:
    if view not in VIEW_AXES:
        raise ValueError(f"Unknown view '{view}'. Expected one of {sorted(VIEW_AXES)}")
    x_idx, y_idx = VIEW_AXES[view]
    return pose[:, :, [x_idx, y_idx]]

def compute_shared_limits(proj_a: np.ndarray, proj_b: np.ndarray, zoom: float) -> tuple[float, float, float, float]:
    """Computes bounding box limits shared by both skeletons to keep scaling identical."""
    flat = np.concatenate([proj_a.reshape(-1, 2), proj_b.reshape(-1, 2)], axis=0)
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

def render_comparison_gif(
    proj_orig: np.ndarray, 
    proj_recon: np.ndarray, 
    output_path: Path, 
    fps: int, 
    view: str, 
    zoom: float = 0.85
) -> None:
    """Renders a side-by-side GIF of the original and reconstructed skeletons."""
    x_min, x_max, y_min, y_max = compute_shared_limits(proj_orig, proj_recon, zoom=zoom)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 8.6))
    style_axes(ax1, view, x_min, x_max, y_min, y_max)
    style_axes(ax2, view, x_min, x_max, y_min, y_max)

    def setup_skeleton(ax, title):
        bone_lines = []
        for bone_index, _ in enumerate(SIDE_BONES):
            color = "#1f77b4" if bone_index < 12 else "#ff7f0e"
            line, = ax.plot([], [], linewidth=4.5, color=color, solid_capstyle="round")
            bone_lines.append(line)
        
        joints = ax.scatter([], [], s=48, color="#111111", zorder=4)
        trail_line, = ax.plot([], [], linewidth=2.5, color="#666666", alpha=0.65)
        
        ax.text(0.02, 0.96, title, transform=ax.transAxes, fontsize=12, va="top", fontweight="bold")
        step_label = ax.text(0.02, 0.90, "", transform=ax.transAxes, fontsize=11, va="top")
        
        return bone_lines, joints, trail_line, step_label

    lines_orig, joints_orig, trail_orig, label_orig = setup_skeleton(ax1, f"Ground Truth ({view} view)")
    lines_recon, joints_recon, trail_recon, label_recon = setup_skeleton(ax2, f"Model Reconstruction ({view} view)")

    trail_pts_orig: list[np.ndarray] = []
    trail_pts_recon: list[np.ndarray] = []

    def init():
        for line in lines_orig + lines_recon:
            line.set_data([], [])
        joints_orig.set_offsets(np.empty((0, 2)))
        joints_recon.set_offsets(np.empty((0, 2)))
        trail_orig.set_data([], [])
        trail_recon.set_data([], [])
        label_orig.set_text("")
        label_recon.set_text("")
        return lines_orig + lines_recon + [joints_orig, joints_recon, trail_orig, trail_recon, label_orig, label_recon]

    def update(frame_index: int):
        def update_ax(skeleton, lines, joints_scatter, trail_line, trail_list, label):
            xs = skeleton[:, 0]
            ys = skeleton[:, 1]
            for line, (start, end) in zip(lines, SIDE_BONES):
                line.set_data([xs[start], xs[end]], [ys[start], ys[end]])
            joints_scatter.set_offsets(np.column_stack([xs, ys]))
            
            trail_list.append(skeleton[0])
            trail_arr = np.asarray(trail_list)
            trail_line.set_data(trail_arr[:, 0], trail_arr[:, 1])
            label.set_text(f"Frame {frame_index + 1}/{len(proj_orig)}")

        update_ax(proj_orig[frame_index], lines_orig, joints_orig, trail_orig, trail_pts_orig, label_orig)
        update_ax(proj_recon[frame_index], lines_recon, joints_recon, trail_recon, trail_pts_recon, label_recon)

        return lines_orig + lines_recon + [joints_orig, joints_recon, trail_orig, trail_recon, label_orig, label_recon]

    animation = FuncAnimation(
        fig, update, frames=len(proj_orig), init_func=init,
        interval=max(1, 1000 // max(fps, 1)), blit=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct and visualize a batch of gait sequences")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to use")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples to generate")
    parser.add_argument("--ckpt", type=str, default=str(PROJECT_ROOT / "checkpoints/best_autoencoder.pt"), help="Path to weights")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/reconstructions", help="Output directory for GIFs")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the GIF")
    parser.add_argument("--stride", type=int, default=2, help="Keep every n-th frame")
    parser.add_argument("--view", type=str, default="side", choices=sorted(VIEW_AXES), help="Projection view")
    args = parser.parse_args()

    # 1. Initialize Device & Load Checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading checkpoint from {args.ckpt}...")
    
    try:
        # We use safe_globals to safely allow pathlib.PosixPath from the saved config
        with torch.serialization.safe_globals([pathlib.PosixPath]):
            # Set weights_only=False to prevent the unpickler error
            checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
            
        ckpt_config = checkpoint.get("config", {})
        
        latent_dim = ckpt_config.get("latent_dim", 256)
        time_steps = ckpt_config.get("output_time_steps", 30)
        out_joints = ckpt_config.get("output_joints", 22)
        in_channels = ckpt_config.get("input_channels", 3)
        out_channels = ckpt_config.get("output_channels", 3)
        use_normalized_input = ckpt_config.get("use_normalized_input", False)
        
        encoder = GaitEncoder(input_channels=in_channels, latent_dim=latent_dim).to(device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder.eval()
        
        decoder = GaitDecoder(
            latent_dim=latent_dim,
            output_time_steps=time_steps,
            output_joints=out_joints,
            output_channels=out_channels,
        ).to(device)
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        decoder.eval()

    except Exception as e:
        print(f"ERROR: Model inference failed. \nDetails: {e}")
        print("Exiting visualizer to prevent generating a false comparison.")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = get_sample_ids(args.split)
    num_to_process = min(args.num_samples, len(sample_ids))
    
    print(f"Generating {num_to_process} visualizations in {args.output_dir} ...\n")
    
    for i in range(num_to_process):
        sample_id = sample_ids[i]
        motion_raw = load_motion(sample_id)
        print(f"[{i+1}/{num_to_process}] Processing {sample_id} (shape: {motion_raw.shape})...")

        orig_globals = []
        recon_globals = []
        
        num_windows = len(motion_raw) // time_steps
        if num_windows == 0:
            print(f"  Skipping {sample_id}: sequence too short.")
            continue
            
        for w in range(num_windows):
            start = w * time_steps
            end = start + time_steps
            window_raw = motion_raw[start:end]
            orig_globals.append(window_raw)
            
            with torch.no_grad():
                if use_normalized_input:
                    normalized, startpos, startorient = normalize_motion(window_raw)
                    input_tensor = torch.tensor(normalized[None], dtype=torch.float32).to(device)
                    latent = encoder(input_tensor)
                    output_tensor = decoder(latent)
                    recon_normalized = output_tensor[0].cpu().numpy()
                    window_recon_global = denormalize_motion(recon_normalized, startpos, startorient)
                else:
                    input_tensor = torch.tensor(window_raw[None], dtype=torch.float32).to(device)
                    latent = encoder(input_tensor)
                    output_tensor = decoder(latent)
                    window_recon_global = output_tensor[0].cpu().numpy()
                    
            recon_globals.append(window_recon_global)

        orig_global = np.concatenate(orig_globals, axis=0)
        recon_global = np.concatenate(recon_globals, axis=0)

        orig_sampled = orig_global[:: max(args.stride, 1)]
        recon_sampled = recon_global[:: max(args.stride, 1)]
        
        proj_orig = project_pose(orig_sampled, args.view)
        proj_recon = project_pose(recon_sampled, args.view)

        output_path = args.output_dir / f"{sample_id}_comparison.gif"
        render_comparison_gif(proj_orig, proj_recon, output_path, args.fps, args.view)
        print(f"  Saved -> {output_path}")

    print("\nBatch visualization complete!")

if __name__ == "__main__":
    main()