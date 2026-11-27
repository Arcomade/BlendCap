"""World-space head stabilization: per-axis dead-zone on head wobble.

Monocular capture leaves a visible nodding/shaking wobble on the head.
Forensics on a real problem clip (2026-08-26) showed WHERE it lives,
and it is not where intuition says: the head/neck LOCAL channels carry
only ~a third of it. The solver smears small per-frame orientation
errors across the whole chain (hips ~1.4deg, each spine joint
0.2-0.9deg, neck/head ~0.6-1.4deg, partially anti-correlated), and the
lever arm sums them into ~1deg RMS / 2-3deg peak wobble of the head's
WORLD orientation at ~1.5-3 Hz. Gating the head/neck local rotations
therefore cannot remove it — the first version of this module did
exactly that and measurably didn't.

So this stabilizes the head where the eye judges it: in WORLD space.

    1. FK the chain to get the head's world rotation per frame.
    2. Baseline it (zero-phase Butterworth on the quaternion track,
       cutoff 1.5 Hz): deliberate head carriage and gestures stay in
       the baseline, the wobble lands in the residual.
    3. Split the residual rotation into components about the
       baseline head's own axes:
           nod  = about head-local X (pitch, "yes")
           turn = about head-local Y (yaw, "no")
           tilt = about head-local Z (roll, ear-to-shoulder)
    4. Per-axis smoothstep dead-zone (threshold -> 2x, in degrees):
       sub-threshold wobble collapses to the baseline, larger motion
       passes through. A deliberate nod is 5-15deg — far above any
       sane threshold — while the capture wobble is 1-3deg.
    5. Recompose and write ONLY the Head bone's local rotation:
       local = world(parent)^-1 @ stabilized_world. The head counter-
       rotates against whatever the chain below sends up; body
       channels are untouched.

The anti-pop scale treatment WIDENS the pass-through region (max of
raw and Gaussian-smoothed scale) instead of blurring it — a plain blur
dilutes short scale peaks and was shaving ~20% off fast real nods.

Runs LAST among the chain-modifying stages (after depth denoise, FDF,
footskate, and body grounding): every one of those rotates or
translates joints under the head and re-creates world-space head
wobble — gating right after smoothing measurably did almost nothing
once the rest of the pipeline had run.
"""

import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d as _gauss1d
    from scipy.signal import butter as _butter, filtfilt as _filtfilt
    from scipy.spatial.transform import Rotation as _Rot
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# Baseline for the world-orientation gate: zero-phase Butterworth
# low-pass, NOT a Gaussian. The baseline must hold everything the
# actor MEANS to do (head carriage and deliberate gestures live at
# <= ~2 Hz) while excluding the capture wobble (~1.5-3 Hz, strongest
# at 3 Hz) — on a clip where the head moves a lot, a Gaussian's
# shallow rolloff dumps the gestures into the residual, their
# amplitude blows past the threshold, and the wobble riding on top
# passes through with them (measured: only ~20% wobble reduction on
# the reference problem clip). An order-4 Butterworth keeps gestures
# below the cutoff in the baseline while the wobble above it lands in
# the residual, which the threshold kills even DURING gestures.
# filtfilt = zero phase, no lag.
#
# Cutoff tuned on the reference clip (24 fps, constant head motion):
# 2.0 Hz floors the wobble at ~0.52deg RMS no matter the thresholds
# (sub-2Hz wobble hides in the baseline); 1.5 Hz matches it at default
# thresholds and reaches 0.34-0.43deg when the sliders go up, with
# head carriage preserved within ~0.7deg. 1.2 Hz gains little more
# and starts degrading the default-threshold case.
_HEAD_BASELINE_CUTOFF_HZ = 1.5
_HEAD_BASELINE_ORDER = 4

# Anti-pop transition width for the suppression on/off boundary,
# in frames.
_SCALE_SIGMA_FRAMES = 2.0

_AXIS_LABELS = ("nod", "turn", "tilt")  # head-local X, Y, Z


def _smooth_quats(q, fps):
    """Zero-phase Butterworth low-pass on a quaternion track (N,4)
    with sign continuity, renormalized. Valid for the small angular
    dispersions a baseline deals in. Falls back to the input when the
    clip is too short to filter."""
    q = q.copy()
    for i in range(1, q.shape[0]):
        if np.dot(q[i], q[i - 1]) < 0.0:
            q[i] = -q[i]
    nyq = fps * 0.5
    cutoff = min(_HEAD_BASELINE_CUTOFF_HZ, nyq * 0.9)
    b, a = _butter(_HEAD_BASELINE_ORDER, cutoff / nyq)
    # (filtfilt pads each end with 3 * max(len(a), len(b)) samples, 15
    # here, and refuses a track that is not longer than that. This
    # test used to count 3 * (max - 1) = 12, so a take of 13, 14 or 15
    # frames got past it and stopped the whole export in filtfilt: the
    # user's 15-frame capture, 2026-11-24.)
    padlen = 3 * max(len(a), len(b))
    if q.shape[0] <= padlen:
        return q
    sm = np.stack([_filtfilt(b, a, q[:, k]) for k in range(4)], axis=1)
    norm = np.linalg.norm(sm, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return sm / norm


def _gate_scale(abs_r, threshold):
    """Smoothstep dead-zone scale in [0,1] with widened pass-through."""
    t = np.clip((abs_r - threshold) / threshold, 0.0, 1.0)
    scale = 3.0 * t * t - 2.0 * t * t * t
    return np.maximum(scale,
                      _gauss1d(scale, sigma=_SCALE_SIGMA_FRAMES))


def apply_head_rotation_denoise(local_rots, names, parents, fps,
                                nod_deg, turn_deg, tilt_deg):
    """In-place world-space head stabilization (see module docstring).

    local_rots: (n_frames, n_joints, 3, 3) BVH local rotations.
    parents:    per-joint parent index array (-1 for the root), same
                indexing as names.
    Only the Head joint's local rotation is modified.
    """
    thresholds = np.radians(
        [max(0.0, float(nod_deg)), max(0.0, float(turn_deg)),
         max(0.0, float(tilt_deg))])
    if not np.any(thresholds > 0):
        return
    if not _HAS_SCIPY:
        print("       Head noise filtering skipped: scipy not installed "
              "(pip install scipy)")
        return
    if "Head" not in names:
        return
    n_frames = local_rots.shape[0]
    if n_frames < 5:
        return

    head = names.index("Head")
    chain = []
    j = head
    while j != -1:
        chain.append(j)
        j = int(parents[j])
    chain.reverse()  # root .. Head

    # World rotation of the head (and of its parent, needed to write
    # the corrected local back).
    Rw = np.tile(np.eye(3), (n_frames, 1, 1))
    Rw_parent = None
    for j in chain:
        if j == head:
            Rw_parent = Rw.copy()
        Rw = Rw @ local_rots[:, j]

    q = _Rot.from_matrix(Rw).as_quat()
    R_base = _Rot.from_quat(_smooth_quats(q, fps)).as_matrix()

    # Residual rotation in the BASELINE head's own frame: columns of
    # R_base are the baseline head's world axes, so the rotvec of
    # base^T @ world, expressed in base coordinates, has components
    # exactly about head-local X/Y/Z = nod/turn/tilt.
    R_resid = np.transpose(R_base, (0, 2, 1)) @ Rw
    rv = _Rot.from_matrix(R_resid).as_rotvec()  # (N, 3), base frame

    kept = np.zeros_like(rv)
    for ax in range(3):
        thr = thresholds[ax]
        comp = rv[:, ax]
        if thr <= 0.0:
            kept[:, ax] = comp  # axis not filtered: passes through
            continue
        abs_r = np.abs(comp)
        p90 = float(np.degrees(np.percentile(abs_r, 90)))
        mx = float(np.degrees(abs_r.max()))
        above = int(np.count_nonzero(abs_r > thr))
        print(f"       Head noise [world/{_AXIS_LABELS[ax]}]: "
              f"thr={np.degrees(thr):.1f}deg  wobble "
              f"p90={p90:.2f}deg max={mx:.2f}deg  "
              f"{above}/{n_frames} frames above thr")
        kept[:, ax] = comp * _gate_scale(abs_r, thr)

    Rw_new = R_base @ _Rot.from_rotvec(kept).as_matrix()
    local_rots[:, head] = (np.transpose(Rw_parent, (0, 2, 1))
                           @ Rw_new)
