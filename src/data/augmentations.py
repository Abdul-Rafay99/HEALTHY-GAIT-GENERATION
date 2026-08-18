"""Gait augmentation transforms simulating specific pathological impairments.

These operate on the normalized (root-centered, per-frame local) skeleton
representation produced by normalize_skeletons - the same [T, 24, 3] tensor
already used as both input and target when training on normalized data. Each
function takes that representation and returns an augmented copy, leaving the
input untouched.

Row 0 of the normalized tensor is delta_root_local (frame-to-frame root
displacement) and row 1 is forward_local (facing-direction turn rate); the
22 real body joints live at BODY_JOINTS (index 2 onward), in JOINT_NAMES
order:

    0 PELVIS         6  CHEST_LOWER     12 NECK             18 ELBOW_RIGHT
    1 HIP_RIGHT       7  ANKLE_RIGHT     13 CLAVICLE_RIGHT   19 ELBOW_LEFT
    2 HIP_LEFT        8  ANKLE_LEFT      14 CLAVICLE_LEFT    20 WRIST_RIGHT
    3 NAVEL           9  CHEST_UPPER     15 HEAD             21 WRIST_LEFT
    4 KNEE_RIGHT      10 TOES_RIGHT      16 SHOULDER_RIGHT
    5 KNEE_LEFT       11 TOES_LEFT       17 SHOULDER_LEFT
"""

from __future__ import annotations

import numpy as np


POSE_OFFSET = 2
BODY_JOINTS = slice(POSE_OFFSET, None)


def pose_view(normalized: np.ndarray) -> np.ndarray:
    return normalized[:, BODY_JOINTS, :]


def copy_normalized(normalized: np.ndarray) -> np.ndarray:
    return np.array(normalized, copy=True)


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
    scale_chain(pose, 1, [4, 7, 10], (1.0, 0.96, 0.93))

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


# The 3 pathological impairments only - used by denoising training, where
# every input must be an augmented (not healthy) sample.
PATHOLOGICAL_AUGMENTATIONS = {
    "parkinsonian_gait": parkinsonian_gait,
    "spatial_asymmetry": spatial_asymmetry,
    "postural_shift": postural_shift,
}
