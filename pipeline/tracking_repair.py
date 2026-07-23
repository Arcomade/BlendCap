# tracking_repair.py
# ==================
# Detect and repair tracking-failure frames in SAM 3D Body output.
#
# Two failure modes from SAM 3D Body's per-frame pose head:
#
#   A. Limb-length anomaly — bbox cropped a limb, partial detection, etc.
#      Symptom: parent->child distance changes vs. its own median across
#      the clip.  A real bone doesn't stretch frame-to-frame; the model's
#      output bone does when the input is bad.
#
#   B. Pose-head regression to the rest pose mean — the OOD-pose failure.
#      When SAM gives up under uncertainty (extreme kicks, unusual stances)
#      it outputs near the training-set mean, which is close to T-pose.
#      Limb lengths stay correct (T-pose has correct anatomy), so signal A
#      misses these.  The signature is: mean angular distance from the rest
#      pose drops sharply for a few frames, then recovers.
#
# Repair: bidirectional matrix-aware interpolation.  Convert boundary
# frames matrix -> quaternion -> SLERP -> matrix.  Position channel uses
# linear interpolation.  Open-ended bad runs (clip starts or ends inside
# a bad run) are filled by holding the nearest good frame.

import numpy as np

from .bvh_math import (
    _rotmats_to_quats_xyzw,
    _slerp_quats_xyzw,
    quats_xyzw_to_rotmats,
)


def _detect_bad_frames(global_rots, joint_coords, parents, R0):
    """Boolean (N,) mask of frames flagged by signals A or B.

    Returns (bad_mask, n_limb, n_tpose) for diagnostic printing.
    """
    N = global_rots.shape[0]
    if N < 4:
        return np.zeros(N, dtype=bool), 0, 0

    # --- Signal A: limb-length stability ---
    # Use ALL parented joints; fingers are fine in this signal because
    # the median-baseline is per-bone, so naturally-short bones don't
    # bias the comparison.
    n_joints = global_rots.shape[1]
    bone_pairs = [(c, parents[c]) for c in range(n_joints) if parents[c] >= 0]

    bone_lens = np.empty((N, len(bone_pairs)), dtype=np.float64)
    for k, (c, p) in enumerate(bone_pairs):
        bone_lens[:, k] = np.linalg.norm(joint_coords[:, c] - joint_coords[:, p],
                                         axis=1)

    bone_med = np.maximum(np.median(bone_lens, axis=0), 1e-6)            # (n_bones,)
    rel_dev = np.abs(bone_lens - bone_med[None, :]) / bone_med[None, :]  # (N, n_bones)
    mean_dev = rel_dev.mean(axis=1)                                       # (N,)

    dev_med = float(np.median(mean_dev))
    dev_mad = float(np.median(np.abs(mean_dev - dev_med)))
    dev_threshold = max(dev_med + 6.0 * dev_mad, 0.10)
    limb_bad = mean_dev > dev_threshold

    # --- Signal B: pose-head regression to rest pose ---
    # Per-joint angular distance from rest: angle(R0[j]^T @ G[f, j]).
    # When the pose head regresses to its training mean, this distance
    # collapses across most joints simultaneously — caught by a sharp
    # local-minimum vs. a 7-frame median baseline.
    R0_T = R0.transpose(0, 2, 1)
    delta = np.einsum('jab,fjbc->fjac', R0_T, global_rots)
    traces = delta[:, :, 0, 0] + delta[:, :, 1, 1] + delta[:, :, 2, 2]
    cos_a = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    angles = np.arccos(cos_a)
    mean_angle = angles.mean(axis=1)

    if N >= 7:
        # 7-frame rolling median, edge-padded
        baseline = np.empty(N, dtype=np.float64)
        for f in range(N):
            lo = max(0, f - 3); hi = min(N, f + 4)
            baseline[f] = np.median(mean_angle[lo:hi])
        rel_drop = (baseline - mean_angle) / np.maximum(baseline, 1e-3)
        tpose_bad = rel_drop > 0.40
    else:
        tpose_bad = np.zeros(N, dtype=bool)
    # NaN/Inf is always bad
    nan_bad = ~(np.isfinite(global_rots).all(axis=(1, 2, 3)) &
                np.isfinite(joint_coords).all(axis=(1, 2)))

    bad = limb_bad | tpose_bad | nan_bad
    return bad, int(limb_bad.sum()), int(tpose_bad.sum())


def _bad_runs(bad):
    """List of (start, end) tuples for each contiguous bad run, end exclusive."""
    runs = []
    N = len(bad)
    i = 0
    while i < N:
        if not bad[i]:
            i += 1
            continue
        j = i
        while j < N and bad[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def clean_tracking_failures(global_rots, joint_coords, parents, R0):
    """Detect and repair tracking-failure frames.

    Returns (gr_out, jc_out, runs) where runs is a list of (start, end)
    inclusive-exclusive bad-frame intervals (for sidecar JSON).
    """
    N = global_rots.shape[0]
    if N < 4:
        return global_rots.copy(), joint_coords.copy(), []

    bad, n_limb, n_tpose = _detect_bad_frames(global_rots, joint_coords,
                                              parents, R0)
    n_bad = int(bad.sum())
    if n_bad == 0:
        print("       Clean tracking: no glitches detected")
        return global_rots.copy(), joint_coords.copy(), []

    print(f"       Clean tracking: {n_bad}/{N} frames flagged "
          f"(limb-anomaly={n_limb}  rest-collapse={n_tpose})")

    good_idx = np.where(~bad)[0]
    if len(good_idx) == 0:
        print("       WARNING: all frames flagged — skipping repair")
        return global_rots.copy(), joint_coords.copy(), []

    runs = _bad_runs(bad)
    gr_out = global_rots.copy()
    jc_out = joint_coords.copy()

    # Convert only the boundary good frames to quaternions on demand —
    # avoids converting everything when most of the clip is clean.
    quat_cache = {}
    def _quat_at(frame):
        if frame not in quat_cache:
            quat_cache[frame] = _rotmats_to_quats_xyzw(global_rots[frame])
        return quat_cache[frame]

    for i, j in runs:
        prev_arr = good_idx[good_idx < i]
        next_arr = good_idx[good_idx >= j]

        if len(prev_arr) == 0 and len(next_arr) == 0:
            continue
        if len(prev_arr) == 0:
            ng = int(next_arr[0])
            gr_out[i:j] = global_rots[ng]
            jc_out[i:j] = joint_coords[ng]
        elif len(next_arr) == 0:
            pg = int(prev_arr[-1])
            gr_out[i:j] = global_rots[pg]
            jc_out[i:j] = joint_coords[pg]
        else:
            pg = int(prev_arr[-1])
            ng = int(next_arr[0])
            span = float(ng - pg)
            q_pg = _quat_at(pg)
            q_ng = _quat_at(ng)
            for f in range(i, j):
                t = (f - pg) / span
                q_t = _slerp_quats_xyzw(q_pg, q_ng, t)
                gr_out[f] = quats_xyzw_to_rotmats(q_t)
                jc_out[f] = (1.0 - t) * joint_coords[pg] + t * joint_coords[ng]

    print(f"       Clean tracking: repaired {n_bad} frames across "
          f"{len(runs)} run(s)")
    return gr_out, jc_out, runs
