"""Camera-angle offset for tilted-camera mocap takes.

Manual-only: the user supplies pitch / roll in degrees via the UI's
"Camera Angle Offset" controls, and we left-multiply the resulting
rotation into global_rots. Auto-detection was removed — every body-
internal signal we tried (foot-plane, spine chain, head-up axis) was
either dominated by SAM's pose-fitting compensations or failed on
common clip types. The manual sliders are the honest tool here; if
SAM ever exposes camera rotation directly in the NPZ, automatic
correction can be revisited.

Where it sits in the pipeline:
    modes.py:run_body loads global_rots, joint_coords, cam_t from the
    NPZ, runs the camera-pitch detrend on cam_t, then calls
    apply_correction on global_rots before compute_local_rotations.

Frame:
    global_rots lives in MODEL frame (Y-up Z-back). R is applied
    directly to global_rots via left-multiply. apply_correction does
    NOT rotate joint_coords or cam_t — that matches the live-preview
    path (which only writes Root rotation keyframes; locomotion stays
    in its original direction) and gives bake/preview parity.

Composition:
    pitch is rotation about X (side axis), roll is rotation about Z
    (forward axis in BVH Y-up Z-back). Composed as
    R = R_z(roll) @ R_x(pitch) in MODEL frame — pitch applied first
    so "pitch then roll" is the natural mental model.
"""
import numpy as np


def _build_euler_correction(pitch_deg, roll_deg):
    p = np.radians(pitch_deg)
    r = np.radians(roll_deg)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    Rx = np.array([[1, 0, 0],
                   [0, cp, -sp],
                   [0, sp,  cp]], dtype=np.float64)
    Rz = np.array([[cr, -sr, 0],
                   [sr,  cr, 0],
                   [0,   0,  1]], dtype=np.float64)
    return Rz @ Rx


def resolve_correction(pitch_deg=0.0, roll_deg=0.0):
    """Build the camera-angle correction rotation, or None when zero."""
    if abs(pitch_deg) < 1e-6 and abs(roll_deg) < 1e-6:
        return None
    return _build_euler_correction(pitch_deg, roll_deg)


def apply_correction(global_rots, joint_coords, cam_t, R_correct):
    """Left-multiply R into global_rots. Returns (global_rots,
    joint_coords, cam_t) — the latter two are passed through unchanged.

    Why only global_rots: the lean lives entirely in per-frame body
    rotations (compute_local_rotations puts the cancelled tilt onto
    the root joint's local rotation). joint_coords and cam_t are read
    by foot_grounding for stance detection and floor metric — leaving
    them untouched means foot grounding works on the original (un-
    rotated) data, which is consistent with the live-preview path
    (which only writes Root rotation keyframes; locomotion stays in
    its original direction). Bake and live preview now agree exactly.
    Trade-off: for very tilted clips the foot grounding's floor-
    reference may sit a couple cm off from the actual rotated feet —
    bridged by the per-rig height offset slider.

    Rest pose (R0) is unchanged: compute_local_rotations' R0[parent]
    @ G[parent]^T term cancels R on every non-root joint — only the
    root bone's local rotation picks up R, which is exactly what
    "rotate the skeleton's world frame" means.
    """
    if R_correct is None:
        return global_rots, joint_coords, cam_t
    global_rots = np.einsum('ij,fkjl->fkil', R_correct, global_rots)
    return global_rots, joint_coords, cam_t


def resolve_head_correction(pitch_deg=0.0, roll_deg=0.0):
    """Head-bone-only variant of resolve_correction. Same math (R =
    Rz(roll) @ Rx(pitch) in MODEL frame), exposed as a separate
    function so the call site in modes.py reads cleanly."""
    return resolve_correction(pitch_deg=pitch_deg, roll_deg=roll_deg)


def apply_head_correction_local(local_rots, R_head, names,
                                head_name="Head"):
    """Left-multiply R_head onto the Head bone's BVH-local rotation
    on every frame. Returns local_rots (modified in-place + returned
    for symmetry).

    Runs AFTER prune_skeleton so local_rots is already in BVH joint
    order. Mirrors the body camera offset's *effective* behavior:
    body's R applied to global_rots, after compute_local_rotations
    and prune_skeleton, lands as L_new[Hips_BVH] = R @ L_old[Hips_BVH]
    (direct left-multiply, no conjugation, because Hips is the kept-
    tree root and has no parent). Doing the equivalent for Head
    requires bypassing compute_local_rotations entirely — applying R
    via global_rots would otherwise produce L_new[Head] = M @ R @
    M^-1 @ L_old[Head] where M = R0[Neck] @ G[Neck].T, a per-frame
    conjugation that rotates R's axis (a pure Z-roll could end up as
    a Y-yaw-like rotation in Head's frame, which presents as "wrong
    direction" or "ignored").

    Direct L modification keeps the rotation axis fixed in BVH MODEL
    frame, matching exactly what body offset does to Hips. Direction
    convention now identical: same R input → same direction tilt.

    Silent no-op when head_name isn't in `names` (face-only or
    skeletons without a Head bone).
    """
    if R_head is None:
        return local_rots
    try:
        head_idx = names.index(head_name)
    except ValueError:
        return local_rots
    # einsum over the (frames, 3, 3) slice for the single Head joint.
    local_rots[:, head_idx, :, :] = np.einsum(
        'ij,fjk->fik', R_head, local_rots[:, head_idx, :, :])
    return local_rots
