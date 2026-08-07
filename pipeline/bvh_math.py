# bvh_math.py
# ===========
# Pure-math primitives shared across the BVH export pipeline:
# rotation conversions, FK, T-pose accumulation, and temporal smoothing.
# No project state, no I/O, no skeleton-specific knowledge — anything in
# here works on raw numpy arrays.
#
# Verified FK math (error < 0.015 against pred_joint_coords):
#   world_pos[j] = world_pos[parent] + global_rots[parent] @ offsets[j]
#   local_rot[j] = global_rots[parent]^T @ global_rots[j]

import numpy as np

try:
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =====================================================================
#  ROTATION MATH (vectorized)
# =====================================================================

def _rotmats_to_quats_xyzw(R):
    """(..., 3, 3) rotation matrices -> (..., 4) xyzw quaternions.

    Shepperd's stable method: pick the largest of trace, m00, m11, m22 to
    avoid square-root cancellation.
    """
    R = np.asarray(R, dtype=np.float64)
    flat = R.reshape(-1, 3, 3)
    n = len(flat)
    q = np.empty((n, 4), dtype=np.float64)

    m00 = flat[:, 0, 0]; m11 = flat[:, 1, 1]; m22 = flat[:, 2, 2]
    trace = m00 + m11 + m22

    case_w = trace > 0
    case_x = (~case_w) & (m00 >= m11) & (m00 >= m22)
    case_y = (~case_w) & (~case_x) & (m11 >= m22)
    case_z = ~(case_w | case_x | case_y)

    if np.any(case_w):
        s = np.sqrt(trace[case_w] + 1.0) * 2.0
        q[case_w, 3] = 0.25 * s
        q[case_w, 0] = (flat[case_w, 2, 1] - flat[case_w, 1, 2]) / s
        q[case_w, 1] = (flat[case_w, 0, 2] - flat[case_w, 2, 0]) / s
        q[case_w, 2] = (flat[case_w, 1, 0] - flat[case_w, 0, 1]) / s
    if np.any(case_x):
        s = np.sqrt(1.0 + m00[case_x] - m11[case_x] - m22[case_x]) * 2.0
        q[case_x, 3] = (flat[case_x, 2, 1] - flat[case_x, 1, 2]) / s
        q[case_x, 0] = 0.25 * s
        q[case_x, 1] = (flat[case_x, 0, 1] + flat[case_x, 1, 0]) / s
        q[case_x, 2] = (flat[case_x, 0, 2] + flat[case_x, 2, 0]) / s
    if np.any(case_y):
        s = np.sqrt(1.0 + m11[case_y] - m00[case_y] - m22[case_y]) * 2.0
        q[case_y, 3] = (flat[case_y, 0, 2] - flat[case_y, 2, 0]) / s
        q[case_y, 0] = (flat[case_y, 0, 1] + flat[case_y, 1, 0]) / s
        q[case_y, 1] = 0.25 * s
        q[case_y, 2] = (flat[case_y, 1, 2] + flat[case_y, 2, 1]) / s
    if np.any(case_z):
        s = np.sqrt(1.0 + m22[case_z] - m00[case_z] - m11[case_z]) * 2.0
        q[case_z, 3] = (flat[case_z, 1, 0] - flat[case_z, 0, 1]) / s
        q[case_z, 0] = (flat[case_z, 0, 2] + flat[case_z, 2, 0]) / s
        q[case_z, 1] = (flat[case_z, 1, 2] + flat[case_z, 2, 1]) / s
        q[case_z, 2] = 0.25 * s

    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(norms, 1e-12)
    return q.reshape(R.shape[:-2] + (4,))


def _slerp_quats_xyzw(q0, q1, t):
    """SLERP between two (..., 4) xyzw quaternion arrays at scalar t in [0, 1]."""
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1s = np.where(dot < 0, -q1, q1)
    dot_abs = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot_abs)
    sin_theta = np.sin(theta)
    near_par = sin_theta < 1e-6
    safe_sin = np.where(near_par, 1.0, sin_theta)
    w0 = np.where(near_par, 1.0 - t, np.sin((1.0 - t) * theta) / safe_sin)
    w1 = np.where(near_par, t,        np.sin(t * theta)         / safe_sin)
    out = w0 * q0 + w1 * q1s
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    return out / np.maximum(norms, 1e-10)


def quats_xyzw_to_rotmats(q):
    """Convert (N, 4) xyzw quaternions to (N, 3, 3) rotation matrices (batch)."""
    q = np.asarray(q, dtype=np.float64)
    single = q.ndim == 1
    if single:
        q = q[None]

    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    q = q / norms
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.empty((len(q), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (yy + zz)
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = 1 - 2 * (xx + zz)
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = 1 - 2 * (xx + yy)

    return R[0] if single else R


def rotmats_to_euler_zyx_deg(R):
    """(N, 3, 3) rotation matrices -> (N, 3) [Zrot, Yrot, Xrot] in degrees (batch)."""
    R = np.asarray(R, dtype=np.float64)
    single = R.ndim == 2
    if single:
        R = R[None]

    sy = np.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
    singular = sy < 1e-6

    # Non-singular
    x = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    y = np.arctan2(-R[:, 2, 0], sy)
    z = np.arctan2(R[:, 1, 0], R[:, 0, 0])

    # Singular overrides
    if np.any(singular):
        x[singular] = np.arctan2(-R[singular, 1, 2], R[singular, 1, 1])
        z[singular] = 0.0

    result = np.degrees(np.stack([z, y, x], axis=1))
    return result[0] if single else result


def euler_zyx_deg_to_rotmats(angles):
    """(N, 3) [Zrot, Yrot, Xrot] degrees -> (N, 3, 3) rotation matrices (batch).

    Inverse of rotmats_to_euler_zyx_deg.  Composes Rz @ Ry @ Rx.
    """
    angles = np.asarray(angles, dtype=np.float64)
    single = angles.ndim == 1
    if single:
        angles = angles[None]
    rad = np.radians(angles)
    cz, cy, cx = np.cos(rad[:, 0]), np.cos(rad[:, 1]), np.cos(rad[:, 2])
    sz, sy, sx = np.sin(rad[:, 0]), np.sin(rad[:, 1]), np.sin(rad[:, 2])

    R = np.empty((len(angles), 3, 3), dtype=np.float64)
    R[:, 0, 0] = cz * cy
    R[:, 0, 1] = cz * sy * sx - sz * cx
    R[:, 0, 2] = cz * sy * cx + sz * sx
    R[:, 1, 0] = sz * cy
    R[:, 1, 1] = sz * sy * sx + cz * cx
    R[:, 1, 2] = sz * sy * cx - cz * sx
    R[:, 2, 0] = -sy
    R[:, 2, 1] = cy * sx
    R[:, 2, 2] = cy * cx
    return R[0] if single else R


# =====================================================================
#  T-POSE FROM ACCUMULATED PREROTATIONS
# =====================================================================

def compute_tpose_globals(parents, prerot_quats):
    """
    Accumulate prerotation quaternions through the hierarchy to get
    T/A-pose global rotation matrices.

    T[j] = T[parent] @ prerot_mat[j]
    """
    n = len(parents)
    prerot_mats = quats_xyzw_to_rotmats(prerot_quats)

    T = np.zeros((n, 3, 3), dtype=np.float64)
    for j in range(n):
        p = parents[j]
        T[j] = prerot_mats[j] if p < 0 else T[p] @ prerot_mats[j]

    return T


# =====================================================================
#  COMPUTE WORLD POSITIONS (verified FK)
# =====================================================================

def compute_world_positions(parents, offsets_cm, global_rots_frame):
    """
    Verified FK: pos[j] = pos[parent] + G[parent] @ offset[j]
    Works for both frame-0 globals and T-pose globals.
    """
    n = len(parents)
    pos = np.zeros((n, 3), dtype=np.float64)

    for j in range(n):
        p = parents[j]
        if p < 0:
            pos[j] = offsets_cm[j]
        else:
            pos[j] = pos[p] + global_rots_frame[p] @ offsets_cm[j]

    return pos


# =====================================================================
#  GLOBAL -> LOCAL ROTATIONS (vectorized over frames)
# =====================================================================

def compute_local_rotations(global_rots, parents, R0):
    """
    Local rotations where the rest pose (R0) maps to identity.
    Vectorized: eliminates the per-frame Python loop.

    Formula:
      L[f,j] = R0[parent] @ G[f,parent]^T @ G[f,j] @ R0[j]^T
    For root (parent < 0):
      L[f,root] = G[f,root] @ R0[root]^T
    """
    n_frames, n_joints = global_rots.shape[:2]
    local = np.empty((n_frames, n_joints, 3, 3), dtype=np.float64)

    R0_T = R0.transpose(0, 2, 1)  # (J, 3, 3)

    for j in range(n_joints):
        p = parents[j]
        if p < 0:
            # (F,3,3) @ (3,3) -> (F,3,3)
            local[:, j] = global_rots[:, j] @ R0_T[j]
        else:
            # R0[p] @ G[f,p]^T @ G[f,j] @ R0_T[j]
            # G[f,p]^T: transpose last two dims -> (F,3,3)
            GpT = global_rots[:, p].transpose(0, 2, 1)
            local[:, j] = R0[p] @ GpT @ global_rots[:, j] @ R0_T[j]

    return local


# =====================================================================
#  SMOOTHING
# =====================================================================

def seconds_to_savgol_window(seconds, fps, min_frames=5):
    """Convert a duration in seconds to an odd Savgol window in frames.

    Returns 0 if seconds <= 0 (smoothing disabled). The result is
    always odd (Savgol requirement) and at least `min_frames` (Savgol
    degenerates below that). FPS-aware: same seconds value gives
    equivalent perceptual smoothing at any capture rate.
    """
    if seconds <= 0:
        return 0
    w = int(round(seconds * float(fps)))
    if w < min_frames:
        w = min_frames
    if w % 2 == 0:
        w += 1
    return w


def seconds_to_sigma_frames(seconds, fps):
    """Convert a Gaussian sigma in seconds to sigma in frames.

    Returns 0.0 if seconds <= 0 (smoothing disabled). FPS-aware.
    """
    if seconds <= 0:
        return 0.0
    return seconds * float(fps)


def smooth_rotations(local_rots, window, polyorder=3):
    """
    Savitzky-Golay on rotation matrix elements, then batch SVD re-orthogonalization.
    """
    if not HAS_SCIPY:
        print("       WARNING: scipy not installed, skipping smoothing")
        return local_rots

    n_frames, n_joints = local_rots.shape[:2]
    if n_frames < window:
        print(f"       Not enough frames ({n_frames}) for window={window}, skipping")
        return local_rots

    smoothed = local_rots.copy()
    print(f"       Smoothing {n_joints} joints (window={window}, poly={polyorder})...")

    # Savitzky-Golay per joint, all 9 elements at once
    for j in range(n_joints):
        flat = smoothed[:, j].reshape(n_frames, 9)
        for k in range(9):
            flat[:, k] = savgol_filter(flat[:, k], window, polyorder)
        smoothed[:, j] = flat.reshape(n_frames, 3, 3)

    # Batch SVD re-orthogonalization: all frames × all joints at once
    flat_all = smoothed.reshape(-1, 3, 3)
    U, _, Vt = np.linalg.svd(flat_all)
    R = U @ Vt
    # Fix reflections
    dets = np.linalg.det(R)
    neg_mask = dets < 0
    if np.any(neg_mask):
        U[neg_mask, :, -1] *= -1
        R[neg_mask] = U[neg_mask] @ Vt[neg_mask]
    smoothed = R.reshape(n_frames, n_joints, 3, 3)

    return smoothed


def smooth_positions(positions, sigma, depth_axis=2, skip_axes=()):
    """Gaussian smoothing on root positions. Depth axis gets 2x sigma.
    Axes in skip_axes pass through untouched."""
    if not HAS_SCIPY or positions.shape[0] < 4:
        return positions

    smoothed = positions.copy()
    for axis in range(3):
        if axis in skip_axes:
            continue
        s = sigma * 2.0 if axis == depth_axis else sigma
        smoothed[:, axis] = gaussian_filter1d(positions[:, axis], sigma=s)
    return smoothed


def _savgol_masked_2d(data, mask, window, polyorder=2):
    """Apply savgol_filter to masked rows of a 2D array (N, C) in-place."""
    n_valid = mask.sum()
    if n_valid < window:
        return
    poly = min(polyorder, min(window, n_valid) - 1)
    w = min(window, n_valid)
    vals = data[mask]  # (n_valid, C)
    for ch in range(vals.shape[1]):
        vals[:, ch] = savgol_filter(vals[:, ch], w, poly)
    data[mask] = vals


# Face smoothing: centered Savitzky-Golay convolution. Zero phase shift
# by construction (symmetric kernel), so lipsync stays frame-accurate.
# polyorder=3 preserves sharp transitions (blinks, lip plosives) better
# than a moving average while attenuating high-frequency jitter. Window
# is specified in seconds (FPS-independent) and converted to a frame
# window at runtime via seconds_to_savgol_window().
# Must stay in sync with the ARKit subprocess
# (pipeline/smooth_face_npz.py routes through this same function).
_FACE_SAVGOL_POLY = 3


# Region -> ARKit blendshape names. Used for per-region dead-zone
# denoising downstream of smoothing. Mirrors the same region set as
# the legacy retarget face_denoise (jaw/gaze/lids/brows/lips/cheeks/
# nose/tongue); duplicated in blendcap/operators_library.py for the
# ARKit in-process path. tongue got its own region once tongueOut
# became a live capture channel (pixel-detected at export time).
ARKIT_REGION_BLENDSHAPES = {
    "jaw":    ["jawOpen", "jawForward", "jawLeft", "jawRight", "mouthClose"],
    "gaze":   ["eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
               "eyeLookUpLeft", "eyeLookDownRight", "eyeLookInRight",
               "eyeLookOutRight", "eyeLookUpRight"],
    "lids":   ["eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft",
               "eyeSquintRight", "eyeWideLeft", "eyeWideRight"],
    "brows":  ["browDownLeft", "browDownRight", "browInnerUp",
               "browOuterUpLeft", "browOuterUpRight"],
    "lips":   ["mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
               "mouthSmileLeft", "mouthSmileRight",
               "mouthFrownLeft", "mouthFrownRight",
               "mouthDimpleLeft", "mouthDimpleRight",
               "mouthStretchLeft", "mouthStretchRight",
               "mouthRollLower", "mouthRollUpper",
               "mouthShrugLower", "mouthShrugUpper",
               "mouthPressLeft", "mouthPressRight",
               "mouthLowerDownLeft", "mouthLowerDownRight",
               "mouthUpperUpLeft", "mouthUpperUpRight"],
    "cheeks": ["cheekPuff", "cheekSquintLeft", "cheekSquintRight"],
    "nose":   ["noseSneerLeft", "noseSneerRight"],
    "tongue": ["tongueOut"],
}


def smooth_face_data(face_data, window_s=0.23, window=None):
    """Centered Savitzky-Golay smoothing on face data arrays (in-place).

    Zero phase shift, FPS-independent: window_s in seconds is converted
    to an odd frame window at runtime using face_data['fps']. Applied
    to detected frames only; undetected frames are left as zeros.
    window_s <= 0 disables. The legacy `window` arg is ignored and
    kept only for back-compat with old call sites.
    """
    if window_s <= 0:
        return
    if not HAS_SCIPY:
        print("       Face smoothing skipped: scipy not installed "
              "(pip install scipy)")
        return

    fps = float(face_data.get("fps", 30.0))
    win = seconds_to_savgol_window(window_s, fps, min_frames=5)
    if win <= 0:
        return

    detected = face_data["detected"]
    mask = detected.astype(bool)
    n_det = int(mask.sum())
    if n_det < win:
        print(f"       Face smoothing skipped: only {n_det} detected "
              f"frame(s), need >= {win}")
        return

    print(f"       Smoothing face data (Savitzky-Golay, "
          f"window={win} frames ({window_s:.2f} s at {fps:.1f} fps), "
          f"poly={_FACE_SAVGOL_POLY})...")

    # Blendshapes (N, 52)
    _savgol_masked_2d(face_data["blendshapes"], mask, win, _FACE_SAVGOL_POLY)

    # Landmarks (N, 478, 3) -> flatten to (N, 1434) for vectorized smoothing
    lm = face_data["landmarks"]
    n_frames = lm.shape[0]
    lm_flat = lm.reshape(n_frames, -1)
    _savgol_masked_2d(lm_flat, mask, win, _FACE_SAVGOL_POLY)
    face_data["landmarks"] = lm_flat.reshape(n_frames, 478, 3)

    # Eye gaze (N, 4)
    _savgol_masked_2d(face_data["eye_gaze"], mask, win, _FACE_SAVGOL_POLY)


def amplify_face_blendshapes(face_data, region_multipliers, bs_names):
    """In-place per-region amplitude scaling on blendshape values.

    For each region with multiplier != 1.0, every blendshape value in
    that region's ARKit group is multiplied by the value. Operates on
    face_data['blendshapes'] (N, 52). Order independent of denoise
    (apply after denoise so dead-zoned zeros stay zero).

    region_multipliers: dict {region_name: float multiplier, 1.0 = pass-through}.
    bs_names: list of 52 strings, the blendshape name for each column.
    """
    bs_arr = face_data.get("blendshapes")
    if bs_arr is None or bs_arr.size == 0:
        return
    name_to_col = {name: i for i, name in enumerate(bs_names)}
    affected_regions = []
    for region, mult in region_multipliers.items():
        if mult == 1.0:
            continue
        cols = []
        for bs_name in ARKIT_REGION_BLENDSHAPES.get(region, []):
            col = name_to_col.get(bs_name)
            if col is not None:
                cols.append(col)
        if not cols:
            continue
        bs_arr[:, cols] *= mult
        affected_regions.append(f"{region}={mult:.2f}")
    if affected_regions:
        print(f"       Face amp applied: " + ", ".join(affected_regions))


def denoise_face_blendshapes(face_data, region_thresholds, bs_names):
    """In-place per-region SOFT dead-zone on blendshape values.

    Smooth ramp from 0 (at |x|=T) to 1x (at |x|=2T), so values
    straddling the threshold no longer snap on/off frame-to-frame.
    Below T -> 0; above 2T -> unchanged; in between -> scaled by
    smoothstep((|x|-T)/T). Operates on face_data['blendshapes'] (N, 52).

    region_thresholds: dict {region_name: float threshold in [0,1]}.
    bs_names: list of 52 strings, the blendshape name for each column.
    """
    bs_arr = face_data.get("blendshapes")
    if bs_arr is None or bs_arr.size == 0:
        return
    name_to_col = {name: i for i, name in enumerate(bs_names)}
    affected_regions = []
    for region, threshold in region_thresholds.items():
        if threshold <= 0:
            continue
        cols = []
        for bs_name in ARKIT_REGION_BLENDSHAPES.get(region, []):
            col = name_to_col.get(bs_name)
            if col is not None:
                cols.append(col)
        if not cols:
            continue
        sub = bs_arr[:, cols]
        # Smoothstep ramp on |x| / threshold:
        #   t = (|x| - thr) / thr   clipped to [0,1]
        #   scale = 3t^2 - 2t^3  (smoothstep)
        # scale is 0 at |x|<=thr, 1 at |x|>=2*thr, smooth between.
        t = np.clip((np.abs(sub) - threshold) / threshold, 0.0, 1.0)
        scale = t * t * (3.0 - 2.0 * t)
        bs_arr[:, cols] = sub * scale
        affected_regions.append(f"{region}={threshold:.3f}")
    if affected_regions:
        print(f"       Face denoise applied (soft): "
              + ", ".join(affected_regions))
