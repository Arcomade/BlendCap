"""Per-tip depth residual-amplitude dead-zone for body animation.

Monocular body capture (SAM 3D Body) is noisiest along the camera-depth
axis (BVH Z, which becomes Blender Y after BVH import). The wobble is
small per-frame but visually obvious on the leaf tips of the skeleton —
feet (driven by shin rotation), hands (driven by forearm rotation), and
head (driven by head rotation). The root translation is usually clean
enough after smoothing, but we expose a knob for it too in case a clip
needs it.

This module mirrors the face Noise Filtering pattern (smoothstep dead-
zone with a T→2T fade) but operates on the RESIDUAL AGAINST A SMOOTHED
BASELINE instead of a raw value or per-frame velocity. The face filter
gets away with thresholding on the raw value because blendshape=0 is a
meaningful baseline. Position data has no natural "off" baseline, but
a locally-smoothed copy of the same series IS the baseline we want:
real slow motion (locomotion, body sway) lives in the baseline, and
the jitter is precisely what the smoothing can't capture.

Algorithm per tip (depth axis only):
    baseline(t) = Gaussian low-pass of raw depth, sigma ~0.3s
    residual(t) = raw_z(t) - baseline(t)
    scale(t)    = smoothstep((|residual| - T) / T) clamped to [0, 1]
    scale(t)    = Gaussian(scale, sigma=2 frames)   # soften the transition
    z_new(t)    = baseline(t) + residual(t) * scale(t)

The scale smoothing exists to kill the "pop" at the boundary between
suppressed (stationary) and pass-through (real motion) regions —
without it, a residual jumping from below T to above 2T in one frame
flips scale 0→1 in one frame too, which reads as a snap. A 2-frame
sigma spreads that transition over ~5 frames (~200ms at 24fps).

Earlier attempts used a Savitzky-Golay baseline at polyorder=3 over a
~7-frame window. That baseline was too flexible — a cubic over 7 frames
fits high-frequency wobble into the polynomial itself, leaving almost
nothing in the residual for the dead-zone to bite. Gaussian is
monotonic and predictable; sigma controls how aggressively real motion
stays in the baseline vs leaks into the residual.

Discrimination:
  - Slow walk: baseline tracks the walk, residual is tiny -> suppressed,
    walk passes through cleanly in the baseline term.
  - Per-frame depth wobble: baseline ignores it, residual is large in
    amplitude but high-frequency -> suppressed.
  - Sudden real motion (kick, head turn): baseline lags briefly, residual
    is large -> passes through above 2T.

Threshold T is in METERS of depth-residual amplitude (= "how much
wobble is noise vs signal"). The smoothstep fade-in/out matches the
face Noise Filtering UX.

The dead-zoned depth series is then baked back into the chain:
  - Feet: 2-bone IK on UpLeg + Leg (reuses _solve_2bone_ik_kovar from
    footskate_cleanup), with foot-bone rotation compensation to preserve
    world heel-toe roll.
  - Hands: 2-bone IK on Arm + ForeArm, with hand-bone rotation
    compensation to preserve wrist orientation.
  - Head: 1-bone swing delta on Head against a virtual 15cm tip
    extending along the head's bone-local Y. No child compensation (Head
    is a leaf in the body skeleton — face bones get attached post-bake
    in write_bvh).
  - Root: direct residual dead-zone on root_pos[:, 2]. No IK needed.

Runs AFTER smooth_rotations and BEFORE footskate_cleanup so footskate
sees cleaner depth input and remains the final word on floor contact.
"""

import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from .bvh_math import seconds_to_sigma_frames
from .footskate_cleanup import (
    _local_to_global_bvh,
    _fk_world_positions_bvh,
    _solve_2bone_ik_kovar,
    _rotation_between,
)


# Virtual head-tip offset in the head bone's LOCAL Y direction. The
# head bone is a leaf in the body skeleton (HeadTop SAM joint was
# dropped during pruning and emitted as an End Site), so it has no
# stored tip length we can read; 15cm is the canonical forehead-to-
# crown distance and gives a sensible lever arm for the swing-delta.
_HEAD_TIP_LOCAL_M = np.array([0.0, 0.15, 0.0])

# Gaussian sigma (in seconds) for the baseline against which depth
# residuals are measured. The earlier Savgol(0.3s, polyorder=3) baseline
# was too flexible — a cubic over ~7 frames fits high-frequency wobble
# into the polynomial instead of leaving it in the residual, which
# silently zeroed out the filter's effect. Gaussian is monotonic and
# predictable: sigma=0.3s puts a low-pass cutoff around 0.5 Hz, well
# below locomotion timing (1-2 Hz). Real strides have ~30 cm amplitude
# so even attenuated they sit well above any sane threshold and pass
# through; per-frame wobble (~0.5-2 cm amplitude, 5+ Hz) gets cut by
# 90%+ in the baseline and shows up large in the residual, where the
# threshold catches it.
_BASELINE_SIGMA_SEC = 0.3

# Gaussian sigma (in FRAMES) applied to the per-frame `scale` signal
# AFTER the smoothstep dead-zone, before multiplying it into the
# residual. Without this, the residual can jump from below T to above
# 2T in a single frame at the boundary between stationary and real
# motion, which makes the smoothstep transition from scale=0 to scale=1
# in one frame too — that reads as a subtle "pop" at the noise/motion
# boundary. A 2-frame Gaussian spreads the suppression-on/off
# transition over ~5 frames (~200 ms at 24 fps), turning the snap into
# a gentle fade. Specified in frames (not seconds) because the goal is
# a fixed number of transition frames regardless of capture rate.
_SCALE_SMOOTHING_SIGMA_FRAMES = 2.0

# Strength of the mid-joint (knee / elbow) depth correction relative
# to the tip threshold. With only the tip denoised, the 2-bone IK lands
# the foot/hand at its smoothed depth but the parent chain has to
# rotate to get there — the wiggle that's been lifted off the tip
# transfers up into the knee/elbow. Solving both targets exactly is
# over-constrained on a 2-bone chain, so we use the smoothed mid-joint
# position to derive a smoother POLE DIRECTION for the IK solver,
# which controls which way the chain bends but not whether the tip
# lands at its target. 1/3 keeps the correction soft: tip remains the
# authoritative target, mid-joint is a best-effort nudge.
_MID_JOINT_STRENGTH_RATIO = 1.0 / 3.0

_ARM_CHAINS = (
    {'side': 'L', 'upper': 'LeftArm',  'lower': 'LeftForeArm',  'tip': 'LeftHand'},
    {'side': 'R', 'upper': 'RightArm', 'lower': 'RightForeArm', 'tip': 'RightHand'},
)

_LEG_CHAINS = (
    {'side': 'L', 'upper': 'LeftUpLeg',  'lower': 'LeftLeg',  'tip': 'LeftFoot'},
    {'side': 'R', 'upper': 'RightUpLeg', 'lower': 'RightLeg', 'tip': 'RightFoot'},
)


def _smoothed_baseline(series_1d, sigma_frames):
    """Gaussian low-pass for the residual baseline.

    Returns a copy of `series_1d` if scipy isn't available or sigma is
    too small to do anything useful. Monotonic frequency response —
    sigma controls how much real motion stays in the baseline vs leaks
    into the residual.
    """
    n = series_1d.shape[0]
    if not _HAS_SCIPY or n < 3 or sigma_frames < 0.5:
        return series_1d.copy()
    return gaussian_filter1d(series_1d, sigma=sigma_frames)


def _smoothstep_deadzone_residual(series_1d, threshold, baseline_sigma_frames):
    """Smoothstep dead-zone on the residual against a Gaussian baseline.

    Below |residual| <= threshold: residual scales to 0 (jitter killed,
    output collapses to the baseline).
    Above |residual| >= 2*threshold: residual passes through (real fast
    motion the baseline couldn't follow).
    Between: smoothstep 3t^2 - 2t^3, t = (|residual| - thr) / thr.

    The per-frame scale is then temporally smoothed by a small Gaussian
    (sigma = _SCALE_SMOOTHING_SIGMA_FRAMES) to spread the
    suppression-on/off transition over a few frames, eliminating the
    "pop" at the noise/motion boundary.

    Returns (output, residual_abs) so callers can print diagnostics
    without recomputing the baseline.
    """
    n = series_1d.shape[0]
    if n < 2 or threshold <= 0.0:
        return series_1d.copy(), np.zeros_like(series_1d)
    baseline = _smoothed_baseline(series_1d, baseline_sigma_frames)
    residual = series_1d - baseline
    abs_r = np.abs(residual)
    t = np.clip((abs_r - threshold) / threshold, 0.0, 1.0)
    scale = 3.0 * t * t - 2.0 * t * t * t
    if _HAS_SCIPY and n >= 3:
        scale = gaussian_filter1d(scale, sigma=_SCALE_SMOOTHING_SIGMA_FRAMES)
    return baseline + residual * scale, abs_r


def _residual_stats(abs_r, threshold):
    """Format a one-line diagnostic for the residual amplitudes vs threshold."""
    if abs_r.size == 0:
        return "no data"
    p50 = float(np.percentile(abs_r, 50))
    p90 = float(np.percentile(abs_r, 90))
    mx = float(abs_r.max())
    above = int(np.count_nonzero(abs_r > threshold))
    return (f"residual amp p50={p50*100:.2f}cm p90={p90*100:.2f}cm "
            f"max={mx*100:.2f}cm vs thr={threshold*100:.2f}cm; "
            f"{above}/{abs_r.size} frames above thr")


def _emit_progress(progress_range, local_frac, msg):
    if not progress_range:
        return
    lo, hi = progress_range
    pct = lo + max(0.0, min(1.0, local_frac)) * (hi - lo)
    print(f"[PROGRESS {int(pct)}] {msg}", flush=True)


def _resolve_chain(names, chain):
    """Return (up_idx, lo_idx, tip_idx) or None if any name missing."""
    try:
        return (names.index(chain['upper']),
                names.index(chain['lower']),
                names.index(chain['tip']))
    except ValueError:
        return None


def _apply_two_bone_chain(parents, bvh_offsets, local_rots, hips_pos,
                          up_idx, lo_idx, tip_idx, threshold_m,
                          baseline_sigma_frames,
                          progress_range, tick_msg):
    """Denoise a 2-bone IK chain's tip depth and back-solve UpLeg/Leg
    (or Arm/ForeArm) rotations. Tip bone's local rotation is compensated
    so its world rotation is preserved across the parent re-orientation.
    Returns the number of frames modified.
    """
    n_frames = local_rots.shape[0]
    if n_frames < 2:
        return 0

    # Precompute current world positions of the three chain joints.
    up_world = np.empty((n_frames, 3))
    lo_world = np.empty((n_frames, 3))
    tip_world = np.empty((n_frames, 3))
    g_up = np.empty((n_frames, 3, 3))
    g_lo = np.empty((n_frames, 3, 3))
    parent_up = parents[up_idx]
    g_parent_up = np.empty((n_frames, 3, 3))

    tick_every = max(1, n_frames // 20)
    for f in range(n_frames):
        globals_f = _local_to_global_bvh(parents, local_rots[f])
        wpos = _fk_world_positions_bvh(parents, bvh_offsets, globals_f,
                                       hips_pos[f])
        up_world[f] = wpos[up_idx]
        lo_world[f] = wpos[lo_idx]
        tip_world[f] = wpos[tip_idx]
        g_up[f] = globals_f[up_idx]
        g_lo[f] = globals_f[lo_idx]
        g_parent_up[f] = globals_f[parent_up]
        if f % tick_every == 0:
            _emit_progress(progress_range, f / max(n_frames, 1),
                           f"{tick_msg}: FK precompute")

    # Dead-zone the tip's depth (BVH Z = index 2) against its smoothed
    # baseline. Slow real motion lives in the baseline term and passes
    # through; sub-amplitude wobble around the baseline is suppressed.
    tip_z_new, abs_r = _smoothstep_deadzone_residual(
        tip_world[:, 2], threshold_m, baseline_sigma_frames)
    print(f"         {tick_msg}: {_residual_stats(abs_r, threshold_m)}")

    # Mid-joint (knee/elbow) depth denoise at reduced strength. The
    # smoothed mid-joint position becomes the IK solver's pole hint,
    # nudging the knee/elbow toward its denoised depth without breaking
    # the tip target.
    mid_threshold_m = threshold_m * _MID_JOINT_STRENGTH_RATIO
    mid_z_new, mid_abs_r = _smoothstep_deadzone_residual(
        lo_world[:, 2], mid_threshold_m, baseline_sigma_frames)
    print(f"         {tick_msg} mid-joint: "
          f"{_residual_stats(mid_abs_r, mid_threshold_m)}")

    # Skip everything if neither tip nor mid-joint changed (e.g., very
    # short clip or thresholds high enough to leave all residuals above
    # 2T).
    if (np.allclose(tip_z_new, tip_world[:, 2], atol=1e-9)
            and np.allclose(mid_z_new, lo_world[:, 2], atol=1e-9)):
        return 0

    n_modified = 0
    for f in range(n_frames):
        target = tip_world[f].copy()
        target[2] = tip_z_new[f]
        desired_mid = lo_world[f].copy()
        desired_mid[2] = mid_z_new[f]
        if (np.allclose(target, tip_world[f], atol=1e-9)
                and np.allclose(desired_mid, lo_world[f], atol=1e-9)):
            continue

        # Current knee/elbow angle (interior).
        v_lo_to_up = up_world[f] - lo_world[f]
        v_lo_to_tip = tip_world[f] - lo_world[f]
        l1_curr = float(np.linalg.norm(v_lo_to_up))
        l2_curr = float(np.linalg.norm(v_lo_to_tip))
        if l1_curr < 1e-6 or l2_curr < 1e-6:
            continue
        cos_mid = float(np.clip(
            np.dot(v_lo_to_up, v_lo_to_tip) / (l1_curr * l2_curr),
            -1.0, 1.0))
        current_mid_angle = float(np.arccos(cos_mid))

        # Pole direction uses the smoothed mid-joint position so the
        # IK's bend-plane choice tracks the denoised knee/elbow depth.
        pole_dir = desired_mid - up_world[f]

        R_upper_world, R_lower_world, _ = _solve_2bone_ik_kovar(
            up_world[f], target,
            bvh_offsets[lo_idx], bvh_offsets[tip_idx],
            pole_dir, current_mid_angle,
            g_up[f], g_lo[f])

        R_upper_local_new = g_parent_up[f].T @ R_upper_world
        R_lower_local_new = R_upper_world.T @ R_lower_world

        # Preserve tip bone's WORLD rotation (heel-toe roll on foot,
        # wrist orientation on hand) across the parent re-orientation.
        R_lower_world_old = g_lo[f]
        tip_local_old = local_rots[f, tip_idx]
        R_tip_local_new = (R_lower_world.T @ R_lower_world_old
                           @ tip_local_old)

        local_rots[f, up_idx] = R_upper_local_new
        local_rots[f, lo_idx] = R_lower_local_new
        local_rots[f, tip_idx] = R_tip_local_new
        n_modified += 1

        if f % tick_every == 0:
            _emit_progress(progress_range, f / max(n_frames, 1),
                           f"{tick_msg}: IK back-solve")
    return n_modified


def _apply_head_chain(parents, bvh_offsets, local_rots, hips_pos,
                      head_idx, threshold_m, baseline_sigma_frames,
                      progress_range):
    """Denoise the head's virtual-tip depth via swing-delta on Head's
    world rotation. Head has no body-skeleton children, so no descendant
    compensation is needed (face bones are wired into the BVH later in
    write_bvh against Head's basis directly).
    """
    n_frames = local_rots.shape[0]
    if n_frames < 2:
        return 0

    head_tip_world = np.empty((n_frames, 3))
    head_head_world = np.empty((n_frames, 3))
    g_head = np.empty((n_frames, 3, 3))
    g_parent = np.empty((n_frames, 3, 3))
    parent_head = parents[head_idx]

    tick_every = max(1, n_frames // 20)
    for f in range(n_frames):
        globals_f = _local_to_global_bvh(parents, local_rots[f])
        wpos = _fk_world_positions_bvh(parents, bvh_offsets, globals_f,
                                       hips_pos[f])
        head_head_world[f] = wpos[head_idx]
        g_head[f] = globals_f[head_idx]
        g_parent[f] = globals_f[parent_head]
        head_tip_world[f] = wpos[head_idx] + g_head[f] @ _HEAD_TIP_LOCAL_M
        if f % tick_every == 0:
            _emit_progress(progress_range, f / max(n_frames, 1),
                           "Depth denoise H: FK precompute")

    tip_z_new, abs_r = _smoothstep_deadzone_residual(
        head_tip_world[:, 2], threshold_m, baseline_sigma_frames)
    print(f"         Depth denoise head: {_residual_stats(abs_r, threshold_m)}")
    if np.allclose(tip_z_new, head_tip_world[:, 2], atol=1e-9):
        return 0

    n_modified = 0
    for f in range(n_frames):
        if np.isclose(tip_z_new[f], head_tip_world[f, 2], atol=1e-9):
            continue
        new_tip = head_tip_world[f].copy()
        new_tip[2] = tip_z_new[f]
        old_dir = head_tip_world[f] - head_head_world[f]
        new_dir = new_tip - head_head_world[f]
        if np.linalg.norm(old_dir) < 1e-9 or np.linalg.norm(new_dir) < 1e-9:
            continue
        swing = _rotation_between(old_dir, new_dir)
        R_head_world_new = swing @ g_head[f]
        local_rots[f, head_idx] = g_parent[f].T @ R_head_world_new
        n_modified += 1

        if f % tick_every == 0:
            _emit_progress(progress_range, f / max(n_frames, 1),
                           "Depth denoise H: swing-delta")
    return n_modified


def apply_body_depth_denoise(parents, bvh_offsets, local_rots, hips_pos,
                              root_pos, names, fps,
                              feet_thr_m=0.0, hands_thr_m=0.0,
                              head_thr_m=0.0, root_thr_m=0.0,
                              progress_range=None):
    """Per-tip residual-amplitude dead-zone with smoothstep ramp on the
    BVH-frame depth (Z) axis.

    Modifies `local_rots` (limb chains) and `root_pos` (root translation)
    IN PLACE. Each threshold is in METERS of depth-residual amplitude
    (= "how much wobble around the locally-smoothed baseline counts as
    noise"); set to 0.0 to disable that tip's denoise. Below threshold
    the residual is fully suppressed (output collapses to the baseline);
    above 2x threshold the residual passes through (real motion that the
    baseline can't follow); between the two the residual is scaled via
    smoothstep.

    parents, bvh_offsets, names: post-prune skeleton.
    local_rots: (n_frames, K, 3, 3), post-rotation-smoothing.
    hips_pos:   (n_frames, 3) Hips world position (only Y is non-zero
                at this pipeline stage; X/Z root motion lives in root_pos
                and gets stitched in later by insert_synthetic_root).
    root_pos:   (n_frames, 3) Root world translation in BVH frame
                (X = right, Y = up, Z = back/depth).
    fps:        capture rate, used to convert _BASELINE_WINDOW_SEC into
                an FPS-independent odd Savgol frame window.
    progress_range: optional (lo, hi) percent span for [PROGRESS ..] emits.
    """
    n_frames = local_rots.shape[0]
    any_on = (feet_thr_m > 0.0 or hands_thr_m > 0.0
              or head_thr_m > 0.0 or root_thr_m > 0.0)
    if n_frames < 2 or not any_on:
        return

    baseline_sigma_frames = seconds_to_sigma_frames(
        _BASELINE_SIGMA_SEC, fps)
    if not _HAS_SCIPY:
        print("       Depth denoise: scipy not installed, skipping "
              "(baseline can't be computed)")
        return

    enabled = []
    if feet_thr_m > 0.0:
        enabled.append(f"feet={feet_thr_m*100:.1f}cm")
    if hands_thr_m > 0.0:
        enabled.append(f"hands={hands_thr_m*100:.1f}cm")
    if head_thr_m > 0.0:
        enabled.append(f"head={head_thr_m*100:.1f}cm")
    if root_thr_m > 0.0:
        enabled.append(f"root={root_thr_m*100:.1f}cm")
    print(f"       Depth denoise: {', '.join(enabled)} residual-amplitude "
          f"threshold (Gaussian baseline sigma {_BASELINE_SIGMA_SEC:.2f}s = "
          f"{baseline_sigma_frames:.1f} frames at {fps:.1f} fps)")

    # ----- Root translation depth -----
    if root_thr_m > 0.0:
        z_new, abs_r = _smoothstep_deadzone_residual(
            root_pos[:, 2], root_thr_m, baseline_sigma_frames)
        print(f"         Depth denoise root: {_residual_stats(abs_r, root_thr_m)}")
        delta_total = float(np.sum(np.abs(z_new - root_pos[:, 2])))
        root_pos[:, 2] = z_new
        print(f"         Root: total depth delta absorbed = {delta_total*100:.2f}cm")

    # ----- 2-bone chains (legs, arms) -----
    chain_specs = []
    if feet_thr_m > 0.0:
        for ch in _LEG_CHAINS:
            chain_specs.append((ch, feet_thr_m, f"Depth denoise foot {ch['side']}"))
    if hands_thr_m > 0.0:
        for ch in _ARM_CHAINS:
            chain_specs.append((ch, hands_thr_m, f"Depth denoise hand {ch['side']}"))

    n_two_bone = 0
    for ch, thr, msg in chain_specs:
        idxs = _resolve_chain(names, ch)
        if idxs is None:
            print(f"         Skip {ch['side']} {ch['tip']}: chain bones not "
                  f"in skeleton ({ch['upper']}/{ch['lower']}/{ch['tip']})")
            continue
        up_idx, lo_idx, tip_idx = idxs
        modified = _apply_two_bone_chain(
            parents, bvh_offsets, local_rots, hips_pos,
            up_idx, lo_idx, tip_idx, thr,
            baseline_sigma_frames,
            progress_range, msg)
        n_two_bone += modified
        print(f"         {msg}: modified {modified} frames")

    # ----- Head -----
    if head_thr_m > 0.0:
        try:
            head_idx = names.index("Head")
        except ValueError:
            print("         Skip Head: bone not in skeleton")
            head_idx = None
        if head_idx is not None:
            modified = _apply_head_chain(
                parents, bvh_offsets, local_rots, hips_pos,
                head_idx, head_thr_m, baseline_sigma_frames,
                progress_range)
            print(f"         Depth denoise head: modified {modified} frames")
