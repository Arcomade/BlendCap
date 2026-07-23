"""Per-foot stance detection + body-grounding correction.

Computes a per-frame Y delta that, when added to body_y in modes.py,
locks each foot-plant moment to the floor. Extends animation_shift_y
(which grounds the lowest foot only) by also grounding intermediate
plants that float a few cm above due to SAM's monocular jump/squat
ambiguity.

Why this is in the pipeline (not a Blender post-import operator):
- Bakes corrections into the BVH file on disk, so any tool that imports
  the BVH (Blender, Maya, MotionBuilder, Unreal) sees a grounded take.
- Reuses translation + joint_coords already computed for animation
  _shift_y; no extra FK pass required.
- Same NumPy data flow as the rest of modes.py.

Algorithm:
1. Per-foot 3D world position per frame = translation + jc_feet (BVH-Y-up)
2. Per-foot velocity = 3D frame-to-frame displacement
3. Stance = run of consecutive frames with low velocity AND low Y
4. Correction per stance = (floor_y - min(foot_y during stance))
5. Combine both feet — when both planted simultaneously at different
   perceived heights, take the SMALLER-MAGNITUDE correction so neither
   foot punches through the floor (the higher foot stays slightly above
   floor; accepted residual)
6. Smooth at stance boundaries via linear blend over BLEND_FRAMES

Floor reference: per-foot global minimum Y observed across the take
(equals what animation_shift_y already grounds to FLOOR_CLEARANCE_M).
By construction, the lowest stance's correction is ~zero — only higher
stances need lowering.
"""
import numpy as np

from scipy.ndimage import gaussian_filter1d, median_filter, minimum_filter1d


# Sensitivity-mapped detection thresholds. The user-facing "Sensitivity"
# slider in the Blender UI (and --foot-sensitivity CLI flag) maps to
# three internal thresholds via piecewise-linear interpolation between
# three calibration points:
#
#   sensitivity=0.0 (strict):     V=0.5 m/s, H=0.05m, MIN=0.21s
#   sensitivity=0.5 (default):    V=2.0 m/s, H=0.15m, MIN=0.083s
#   sensitivity=1.0 (aggressive): V=4.0 m/s, H=0.30m, MIN=0.04s
#
# Higher sensitivity → looser thresholds → catches more frames as foot
# contacts. Lower sensitivity → stricter → only obvious contacts. The
# 0.5 = current historical defaults so existing clips behave the same
# unless the user moves the slider. MIN_STANCE is in seconds (not
# frames) so the threshold is framerate-independent — converted to
# frames at runtime using the clip's fps.
_SENS_BREAKPOINTS = [
    # (sensitivity, V_THRESH_MPS, H_THRESH_M, MIN_STANCE_SEC)
    (0.0, 0.5, 0.05, 0.21),
    (0.5, 2.0, 0.15, 0.083),
    (1.0, 4.0, 0.30, 0.04),
]

BLEND_FRAMES = 3
CORRECTION_SMOOTH_SIGMA = 2.0

# Pre-smoothing on the bottom-of-foot Y series before stance detection
# and per-stance min math. SAM produces frame-to-frame Y jitter of a
# few cm on tracked joints; without this, adjacent stances pick up
# different "minimum" Y values from noise spikes and produce different
# corrections, which in turn drives visible body bounce between stances.
# A small Gaussian (sigma~2 frames) stabilizes per-stance values without
# blurring real motion (stances span many more frames than the kernel).
# Applied to bottom-of-foot Y only — toe XYZ used for velocity stays
# raw, since smoothing velocity weakens swing-vs-stance discrimination.
FOOT_Y_SMOOTH_SIGMA = 2.0

# Robust detection mode (used internally by Feet Define the Floor):
# the stance detector's height-cap base switches from the
# per-foot global minimum (which a single glitched frame below the
# true floor drags down — lowering the cap's absolute position and
# potentially disqualifying every real plant) to a low percentile of
# the foot-Y series. 5% ignores brief dips while still sitting at or
# below the planted level on any clip whose feet spend at least a
# twentieth of the take on the ground.
ROBUST_FLOOR_PERCENTILE = 5.0

# Robust mode also runs a short median filter on the foot-Y series
# BEFORE the Gaussian smooth. The Gaussian SMEARS a brief deep glitch
# into the neighboring (genuinely planted) frames, which would poison
# the per-stance minima and the stance floor from inside the stances;
# a median with this window removes spikes up to ~2 frames long
# outright instead of spreading them. Longer sustained misreads
# survive — those are the velocity gate's job.
ROBUST_FLOOR_MEDIAN_FRAMES = 5

# Rolling window (seconds) for the LOCAL height base used by
# Feet-Define-the-Floor stance detection. A global base (min or
# percentile of the whole take) breaks under exactly the conditions
# FDF targets: sustained camera drift, or long low spans elsewhere in
# the clip, shift the global base away from the local planted level
# and disqualify real plants. A rolling minimum tracks the local
# level instead.
#
# The base is SHARED across both feet (rolling min of the LOWER foot),
# not per-foot. Per-foot bases backfire on held one-leg poses
# (scorpion, leg-over-head): a foot held still in the air longer than
# the window becomes its own reference, passes the height gate, and
# gets yanked to the floor. With the shared base the standing foot
# anchors the reference at the floor, so the raised foot stays
# correctly rejected — while camera drift, which moves both feet
# together, is still tracked. Known residual: BOTH feet held still
# off the floor for longer than the window (sitting on furniture with
# dangling feet, aerial hangs) can still read as planted.
LOCAL_BASE_WINDOW_SEC = 2.0

# --- Feet Define the Floor (--feet-define-floor) tunables ---
# Longest plant-free gap still treated as a ballistic flight. 1.0 s of
# hang time is a 1.23 m apex — beyond any unassisted jump — so longer
# gaps are considered "not a jump" (sitting, lying, feet hidden) and
# fall back to the video's vertical motion, anchored at both ends.
FDF_MAX_FLIGHT_SEC = 1.0
G_MPS2 = 9.81
# A short plant-free gap only counts as a FLIGHT if the VIDEO shows
# the body rising during it. Foot speed and pose geometry cannot
# separate a jump from a crouch in dynamic footage (martial-arts
# drops are violent, trained jumps are smooth, and a tuck looks like
# a crouch from the inside) — but the direction of travel on screen
# can: jumps go up, drops go down. The video is untrustworthy for HOW
# HIGH (that stays hang-time physics), yet its rise/fall over a
# sub-second window is reliable evidence of WHETHER. The rise is
# measured against the straight line between the gap's edges (so slow
# camera drift cancels) and must clear both an absolute floor and a
# fraction of the apex the hang time implies — a gap claiming a big
# jump must show a proportionally big on-screen rise.
FDF_FLIGHT_RISE_MIN_M = 0.06
FDF_FLIGHT_RISE_FRACTION = 0.2

# Tuck-assisted flight vote. The model chronically under-reports the
# root's rise during real jumps — the very distrust FDF is built on —
# so a genuine jump can fail the rise bar above and get its arc
# flattened (legs visibly tuck, root never leaves the ground). The
# tuck itself is measurable: BOTH feet rising sharply in the BODY
# frame during the gap (legs fold / heels lift). That evidence alone
# must never declare a jump — a deep squat also raises body-frame
# feet — so it only RELAXES the required on-screen rise, never the
# direction: with tuck evidence the video must still rise, but only
# by FDF_TUCK_VIDEO_RELIEF x the normal bar, floored at
# FDF_TUCK_VIDEO_FLOOR_M. Squats stay safe (their on-screen root
# holds level or DROPS, failing even the relaxed bar), and because
# the bar scales with apex ~ T^2, longer gaps still demand big rises
# even when assisted.
FDF_TUCK_RISE_MIN_M = 0.12
FDF_TUCK_VIDEO_RELIEF = 0.3
FDF_TUCK_VIDEO_FLOOR_M = 0.015

# Soft-plant reconstruction for lost-stance crouches. A no-rise gap
# whose feet stayed horizontally still is a squat/kneel whose stances
# the detector lost — NOT a held pose. Holding the level there is
# what turned squats into levitation: the legs fold, the root refuses
# to descend, and the feet get pulled off the floor. When a foot's
# toe XZ stays within FDF_SOFTPLANT_HORIZ_M of its gap-median AND the
# reconstruction meets the neighboring plant levels within
# FDF_SOFTPLANT_EDGE_M at both edges (feet-hidden hallucinations fail
# this seam check), that foot is treated as planted and v follows it
# per-frame exactly as planted frames do. Applies to short no-rise
# gaps AND long gaps (a 3 s squat exceeds FDF_MAX_FLIGHT_SEC). Feet
# that traveled (rolls, crawls, missed steps) refuse the stillness
# test and keep the old behavior.
FDF_SOFTPLANT_HORIZ_M = 0.15
FDF_SOFTPLANT_EDGE_M = 0.08

# When a gap borrows its anchor levels from the surrounding plants,
# step this many frames deeper into each stance. The frames right at
# a gap's edge are the ones whose perceived foot height was already
# drifting (that drift is usually why the gap exists), so the
# edge-most v values are contaminated; a few frames in, they're clean.
FDF_ANCHOR_BACKOFF_FRAMES = 4
# Light smoothing on the rebuilt vertical: softens left/right stance
# handoffs during walking (the two feet sit at slightly different
# body-frame heights) without visibly flattening jump arcs.
FDF_SMOOTH_SIGMA = 2.0

# SAM joint indices for ankles AND toes. Bottom-of-foot per frame =
# min(ankle_y, toe_y). Why both, not just one:
# - During heel strike the ankle dips below the toe; during push-off
#   the toe dips below the ankle. Either alone misses half the cycle.
# - animation_shift_y already grounds the lowest of all 4 to floor, so
#   our reference must use the same set to compose cleanly with that
#   global shift (otherwise our "floor" is at a different Y than the
#   pipeline's actual ground plane).
# - Per-frame min(ankle, toe) tracks the actual ground-contact point
#   regardless of foot pose, which is what "where should the foot be"
#   needs to know.
LEFT_ANKLE_SAM_IDX = 5
LEFT_TOE_SAM_IDX = 7
RIGHT_ANKLE_SAM_IDX = 21
RIGHT_TOE_SAM_IDX = 23


def sensitivity_to_thresholds(sensitivity, fps):
    """Map a sensitivity value in [0, 1] to (v_thresh_mps, h_thresh_m,
    min_stance_frames) using piecewise-linear interpolation between
    the calibration breakpoints in _SENS_BREAKPOINTS.

    fps is used to convert min_stance from seconds (framerate-
    independent) to frames (what the detector loop needs).
    """
    s = max(0.0, min(1.0, float(sensitivity)))
    bps = _SENS_BREAKPOINTS
    # Find segment
    for i in range(len(bps) - 1):
        s0, v0, h0, m0 = bps[i]
        s1, v1, h1, m1 = bps[i + 1]
        if s0 <= s <= s1:
            t = (s - s0) / (s1 - s0) if s1 > s0 else 0.0
            v = v0 + t * (v1 - v0)
            h = h0 + t * (h1 - h0)
            m_sec = m0 + t * (m1 - m0)
            min_frames = max(1, int(round(m_sec * max(fps, 1.0))))
            return float(v), float(h), int(min_frames)
    # Fallback (shouldn't reach here given the [0, 1] clamp + breakpoints
    # spanning that range, but be safe)
    s_last, v_last, h_last, m_last = bps[-1]
    return float(v_last), float(h_last), max(1, int(round(m_last * fps)))


def detect_stances_per_foot(joint_coords, translation, fps, sensitivity=0.5,
                            robust_floor=False, local_height_base=False):
    """Detect per-foot stance phases on the BVH source data.

    Shared primitive used by Pass 1 (body grounding correction) and
    Pass 2 (per-foot FK grounding via 2-bone IK). Both passes need the
    same stance lists and the same smoothed bottom-of-foot Y series for
    anchor-frame selection, so it lives here.

    robust_floor: base the planted-height cap on a low percentile of
        the foot-Y series instead of its global minimum, so a brief
        glitch below the true floor can't move the cap (see
        ROBUST_FLOOR_PERCENTILE).
    local_height_base: base the cap on a ROLLING minimum instead of a
        whole-take statistic, so sustained vertical drift (moving
        camera) can't disqualify real plants. Used by Feet Define the
        Floor, where the plants themselves are the ground truth (see
        LOCAL_BASE_WINDOW_SEC). Takes precedence over robust_floor's
        percentile base (the median pre-filter still applies).

    Returns:
        (left_stances, right_stances, foot_y)
        - left_stances, right_stances: lists of (start, end) inclusive
          frame index pairs.
        - foot_y: (n_frames, 2) smoothed bottom-of-foot Y per side
          (col 0 = L, col 1 = R), in BVH-Y-up coords (same series the
          stance detection ran on).
    """
    n_frames = joint_coords.shape[0]
    if n_frames < 2:
        return [], [], np.zeros((n_frames, 2))

    # All 4 foot joints in BVH-Y-up. Same flip convention as modes.py:
    # SAM Y-down -> BVH Y-up (idx 1 *= -1), SAM Z-forward -> BVH Z-back
    # (idx 2 *= -1). X stays. Order: [L_ankle, L_toe, R_ankle, R_toe].
    flip = np.array([1.0, -1.0, -1.0])
    foot_joint_indices = [
        LEFT_ANKLE_SAM_IDX, LEFT_TOE_SAM_IDX,
        RIGHT_ANKLE_SAM_IDX, RIGHT_TOE_SAM_IDX,
    ]
    jc_all_feet = joint_coords[:, foot_joint_indices, :] * flip
    all_feet_world_xyz = translation[:, None, :] + jc_all_feet
    # Shape: (n_frames, 4, 3) — frame, joint (L_ank,L_toe,R_ank,R_toe), axis

    # Bottom-of-foot Y per side per frame: min(ankle_y, toe_y). Tracks
    # the actual ground-contact point regardless of foot pose (heel
    # strike, foot flat, push-off all yield the right "lowest part of
    # the foot" Y). Shape: (n_frames, 2).
    foot_y = np.stack([
        all_feet_world_xyz[:, [0, 1], 1].min(axis=1),  # L bottom-of-foot
        all_feet_world_xyz[:, [2, 3], 1].min(axis=1),  # R bottom-of-foot
    ], axis=1)

    # Robust mode: median-filter first so a brief deep glitch is
    # removed outright rather than smeared into adjacent planted
    # frames by the Gaussian below (see ROBUST_FLOOR_MEDIAN_FRAMES).
    if robust_floor and ROBUST_FLOOR_MEDIAN_FRAMES > 1:
        foot_y = median_filter(foot_y,
                               size=(ROBUST_FLOOR_MEDIAN_FRAMES, 1),
                               mode='nearest')

    # Smooth foot_y along the time axis. Stabilizes per-stance min
    # values across the take so adjacent stances produce similar
    # corrections (instead of differing by a few cm due to noise),
    # which is the dominant remaining source of body bounce between
    # stances. See FOOT_Y_SMOOTH_SIGMA comment at module top for
    # rationale and tradeoffs.
    if FOOT_Y_SMOOTH_SIGMA > 0:
        foot_y = gaussian_filter1d(foot_y, sigma=FOOT_Y_SMOOTH_SIGMA, axis=0)

    # For velocity, use the TOE XYZ. The toe is more stable than ankle
    # during a stance (ankle Y varies more from foot roll), so toe
    # velocity better reflects "is this foot moving as a unit." Shape:
    # (n_frames, 2, 3).
    toe_xyz = np.stack([
        all_feet_world_xyz[:, 1],  # L toe
        all_feet_world_xyz[:, 3],  # R toe
    ], axis=1)
    deltas = np.diff(toe_xyz, axis=0)
    velocities = np.linalg.norm(deltas, axis=2)
    velocities = np.concatenate([velocities[:1], velocities], axis=0)

    # Height-cap base per foot, as an (n_frames, 2) series. Classic:
    # the global minimum (a glitch below the floor drags the cap down
    # with it, and every real plant can end up above the cap — no
    # stances at all). Robust: a low percentile, so brief dips don't
    # move the cap. Local: a rolling minimum, so slow vertical drift
    # moves the cap WITH the footage instead of against it.
    if local_height_base:
        w = max(3, int(round(LOCAL_BASE_WINDOW_SEC * max(fps, 1e-6))))
        if w % 2 == 0:
            w += 1
        # Shared cross-foot base: rolling min of the LOWER foot. See
        # LOCAL_BASE_WINDOW_SEC comment for why it must not be
        # per-foot (held one-leg poses).
        shared = minimum_filter1d(foot_y.min(axis=1), size=w,
                                  mode='nearest')
        base_y = np.stack([shared, shared], axis=1)
    elif robust_floor:
        base_y = np.broadcast_to(
            np.percentile(foot_y, ROBUST_FLOOR_PERCENTILE, axis=0),
            foot_y.shape)
    else:
        base_y = np.broadcast_to(foot_y.min(axis=0), foot_y.shape)
    v_thresh_mps, h_thresh_m, min_stance_frames = sensitivity_to_thresholds(
        sensitivity, fps)
    v_thresh_per_frame = v_thresh_mps / max(fps, 1e-6)

    # Per-foot planted boolean: low velocity AND near the per-foot base Y.
    planted = np.zeros((n_frames, 2), dtype=bool)
    for foot_idx in range(2):
        planted[:, foot_idx] = (
            (velocities[:, foot_idx] < v_thresh_per_frame)
            & (foot_y[:, foot_idx] < base_y[:, foot_idx] + h_thresh_m)
        )

    left_stances = _find_runs(planted[:, 0], min_stance_frames)
    right_stances = _find_runs(planted[:, 1], min_stance_frames)

    # Y-only velocity extension. The XYZ-based pass above is the primary
    # signal — it discriminates real plants from lateral kicks/slides via
    # full 3D motion. But the absolute threshold is brittle to depth-axis
    # velocity inflation (MoGe-s vs MoGe-l focal mismatch, or any future
    # pipeline-version difference that shifts cam_t.z scale). Symptom is
    # a 1-2 frame anchor stance whose XYZ velocity sits just above the
    # threshold, so it drops out — and the gap to the next detected
    # stance becomes too large to bridge.
    #
    # Extension recovers these anchors by walking outward from each
    # detected stance and including adjacent frames where Y-only velocity
    # is below threshold AND the foot is still below the same height cap.
    # This NEVER creates a stance from scratch — extension requires an
    # XYZ-detected anchor — so kicks/slides that XYZ correctly rejected
    # can't be reintroduced. The height cap also keeps extension from
    # walking into descent frames during a real swing (foot is too high).
    y_deltas = np.abs(deltas[:, :, 1])
    y_velocities = np.concatenate([y_deltas[:1], y_deltas], axis=0)
    left_stances = _extend_stances_y_only(
        left_stances, y_velocities[:, 0], foot_y[:, 0],
        base_y[:, 0] + h_thresh_m,
        v_thresh_per_frame, n_frames)
    right_stances = _extend_stances_y_only(
        right_stances, y_velocities[:, 1], foot_y[:, 1],
        base_y[:, 1] + h_thresh_m,
        v_thresh_per_frame, n_frames)
    return left_stances, right_stances, foot_y


def compute_per_stance_correction(joint_coords, translation, fps,
                                   sensitivity=0.5):
    """Return per-frame Y delta to add to body_y for foot-plant grounding.

    joint_coords: (n_frames, n_joints, 3) — SAM-frame joint positions.
    translation: (n_frames, 3) — body translation in BVH-Y-up world
        (smoothed, centered) — same array used for animation_shift_y.
    fps: float — frame rate, used to convert m/s threshold to m/frame.

    Returns: (n_frames,) ndarray of Y corrections. Mostly negative
        (stances need the body lowered to land on floor); zero outside
        stances and outside blend regions.
    """
    left_stances, right_stances, foot_y = detect_stances_per_foot(
        joint_coords, translation, fps, sensitivity=sensitivity)
    n_frames = foot_y.shape[0]

    if not left_stances and not right_stances:
        print("       Foot grounding: no plants detected "
              "(thresholds too tight or all-airborne clip)")
        return np.zeros(n_frames)

    floor_y = float(foot_y.min())
    left_corr = _stance_corrections(left_stances, foot_y[:, 0], floor_y)
    right_corr = _stance_corrections(right_stances, foot_y[:, 1], floor_y)

    print(f"       Foot grounding: {len(left_stances)} L stances, "
          f"{len(right_stances)} R stances "
          f"(floor_y={floor_y*100:+.1f}cm)")
    for i, (s, e, c) in enumerate(left_corr):
        print(f"         L[{i}] f={s}..{e} delta={c*100:+.1f}cm")
    for i, (s, e, c) in enumerate(right_corr):
        print(f"         R[{i}] f={s}..{e} delta={c*100:+.1f}cm")

    correction = _build_correction_array(
        left_corr, right_corr, n_frames, BLEND_FRAMES)

    # Smooth the per-frame correction. Stance detection toggles on/off
    # near the velocity/height thresholds, which causes the raw
    # correction to jitter frame-to-frame and the body to bounce. A
    # small Gaussian smooth (sigma~2 frames) damps that without
    # noticeably softening the actual stance corrections (which span
    # many frames).
    if CORRECTION_SMOOTH_SIGMA > 0:
        correction = gaussian_filter1d(correction, sigma=CORRECTION_SMOOTH_SIGMA)

    return correction


def compute_feet_defined_vertical(joint_coords, translation, fps,
                                  sensitivity=0.5):
    """Feet Define the Floor: rebuild the body's vertical translation
    from foot plants + ballistic flight, discarding the video's
    vertical channel (which cannot distinguish a jump from camera
    motion or an elevator).

    Returns a per-frame replacement for translation[:, 1] ("v"), or
    None when no plants were detected (feature falls back to classic
    behavior).

    Construction, in the same world series stance detection uses
    (v + body-frame bottom-of-foot Y):
      - Planted frames: v = -(lowest planted foot's body-frame Y), so
        the weight-bearing foot sits exactly at level 0. Where the
        video thought the ground was never enters the math.
      - Gaps up to FDF_MAX_FLIGHT_SEC where the video shows the body
        RISING during the gap (see FDF_FLIGHT_RISE_MIN_M): a gravity
        arc between the takeoff and landing plant levels, apex =
        g*T^2/8 for hang time T. The video only votes on WHETHER it
        was a jump; hang time, not the camera, decides how high.
        When BOTH feet rise sharply in the body frame (a tuck), the
        required on-screen rise is relaxed — but never the direction
        (see the FDF_TUCK_* comments at module top) — so real jumps
        whose root-rise the model under-reported still get their arc.
      - Grounded gaps (either length) whose feet stayed horizontally
        still AND whose reconstruction meets the neighboring plant
        levels at both edges: soft-plant — v follows the still feet
        per-frame exactly as planted frames do. This is the squat/
        kneel case: the stances were lost, not the contact. (See the
        FDF_SOFTPLANT_* comments at module top.)
      - Remaining gaps up to FDF_MAX_FLIGHT_SEC (a stance-detection
        blink during a held pose, not a jump): hold the level,
        blending between the surrounding plants. No arc.
      - Remaining longer gaps (sitting, lying, feet hidden) and the
        spans before the first / after the last plant: the video's
        vertical motion, offset-anchored to the neighboring plant
        level(s), so the classic behavior survives where feet give
        no evidence.

    Stance detection runs with the median pre-filter (glitch-proof
    plants are strictly better when the plants ARE the floor) and the
    ROLLING height base, so sustained camera drift — the very thing
    this mode removes — can't disqualify the plants it needs.
    """
    left_stances, right_stances, foot_y = detect_stances_per_foot(
        joint_coords, translation, fps, sensitivity=sensitivity,
        robust_floor=True, local_height_base=True)
    if not left_stances and not right_stances:
        return None

    n = foot_y.shape[0]
    video_y = translation[:, 1].astype(np.float64).copy()
    # Body-frame bottom-of-foot: strip the video translation back out
    # of the (smoothed) world series so plants can be re-anchored.
    jc_foot = foot_y - video_y[:, None]
    # World toe XZ per foot — the stillness evidence for soft-plants
    # (the same joint the stance detector's velocity gate reads).
    toe_xz = np.stack(
        [joint_coords[:, LEFT_TOE_SAM_IDX][:, (0, 2)]
         + translation[:, (0, 2)],
         joint_coords[:, RIGHT_TOE_SAM_IDX][:, (0, 2)]
         + translation[:, (0, 2)]], axis=1)

    planted = np.zeros((n, 2), dtype=bool)
    for stances, col in ((left_stances, 0), (right_stances, 1)):
        for s, e in stances:
            planted[s:e + 1, col] = True
    any_plant = planted.any(axis=1)

    v = np.zeros(n, dtype=np.float64)
    masked = np.where(planted, jc_foot, np.inf)
    v[any_plant] = -masked[any_plant].min(axis=1)

    # The flight gate reads the VIDEO's vertical during each gap (see
    # the FDF_FLIGHT_RISE_* comments). Nothing to precompute here —
    # the rise is measured per gap against its own edge baseline.

    idx = np.flatnonzero(any_plant)
    first, last = int(idx[0]), int(idx[-1])

    # Leading / trailing plant-free spans: video shape, anchored to
    # the nearest plant.
    if first > 0:
        v[:first] = video_y[:first] + (v[first] - video_y[first])
    if last < n - 1:
        v[last + 1:] = video_y[last + 1:] + (v[last] - video_y[last])

    # Interior gaps: flight arcs, soft-plants, held levels, or
    # anchored-video fallback.
    n_flights = 0
    n_assisted = 0
    n_soft = 0
    n_holds = 0
    n_fallbacks = 0
    max_apex = 0.0
    f = first
    while f <= last:
        if any_plant[f]:
            f += 1
            continue
        g0 = f
        g1 = f
        while g1 + 1 <= last and not any_plant[g1 + 1]:
            g1 += 1
        a, b = g0 - 1, g1 + 1          # planted anchor frames
        va = _anchor_value(v, any_plant, a, -1)
        vb = _anchor_value(v, any_plant, b, +1)
        span = b - a
        hang_sec = span / max(fps, 1e-6)
        ts = (np.arange(g0, g1 + 1) - a) / float(span)
        base = va + (vb - va) * ts
        apex_pred = G_MPS2 * hang_sec * hang_sec / 8.0
        # Flight vote: did the body visibly rise on screen during the
        # gap? Measured against the straight line between the gap's
        # edges so slow camera drift cancels out.
        video_base = video_y[a] + (video_y[b] - video_y[a]) * ts
        rise = float((video_y[g0:g1 + 1] - video_base).max())
        rise_needed = max(FDF_FLIGHT_RISE_MIN_M,
                          FDF_FLIGHT_RISE_FRACTION * apex_pred)
        # Tuck evidence: BOTH feet rising in the body frame (legs
        # fold / heels lift), measured against the same edge-line
        # baseline as the video rise. Relaxes the video bar but never
        # the direction — see the FDF_TUCK_* comments at module top.
        tuck_rise = np.inf
        for c in (0, 1):
            jc_base = (jc_foot[a, c]
                       + (jc_foot[b, c] - jc_foot[a, c]) * ts)
            tuck_rise = min(tuck_rise, float(
                (jc_foot[g0:g1 + 1, c] - jc_base).max()))
        tucked = tuck_rise > FDF_TUCK_RISE_MIN_M
        assisted = ((not rise > rise_needed) and tucked
                    and rise > max(FDF_TUCK_VIDEO_FLOOR_M,
                                   FDF_TUCK_VIDEO_RELIEF * rise_needed))
        took_off = rise > rise_needed or assisted
        if hang_sec <= FDF_MAX_FLIGHT_SEC and took_off:
            v[g0:g1 + 1] = base + 4.0 * apex_pred * ts * (1.0 - ts)
            n_flights += 1
            if assisted:
                n_assisted += 1
            max_apex = max(max_apex, apex_pred)
        else:
            # Grounded gap. Soft-plant first: feet that stayed
            # horizontally still are plants the detector LOST
            # (squat/kneel), not a held pose — follow them per-frame
            # instead of holding a level the legs are about to fold
            # under. The edge seam check rejects feet-hidden
            # hallucinations (their level teleports at the seams).
            v_soft = None
            still_cols = _still_feet_cols(toe_xz, g0, g1)
            if still_cols:
                cand = -jc_foot[g0:g1 + 1][:, still_cols].min(axis=1)
                if (abs(float(cand[0]) - va) <= FDF_SOFTPLANT_EDGE_M
                        and abs(float(cand[-1]) - vb)
                        <= FDF_SOFTPLANT_EDGE_M):
                    v_soft = cand
            if v_soft is not None:
                v[g0:g1 + 1] = v_soft
                n_soft += 1
            elif hang_sec <= FDF_MAX_FLIGHT_SEC:
                # No on-screen rise, feet not verifiably still: a
                # detection blink during a held pose. Hold the level
                # instead of launching the performer.
                v[g0:g1 + 1] = base
                n_holds += 1
            else:
                # Not a plausible jump: keep the video's motion shape,
                # blended between the two anchor offsets.
                off_a = va - video_y[a]
                off_b = vb - video_y[b]
                v[g0:g1 + 1] = (video_y[g0:g1 + 1]
                                + off_a * (1.0 - ts) + off_b * ts)
                n_fallbacks += 1
        f = g1 + 2

    if FDF_SMOOTH_SIGMA > 0:
        v = gaussian_filter1d(v, sigma=FDF_SMOOTH_SIGMA)

    print(f"       Feet-defined vertical: {len(left_stances)} L / "
          f"{len(right_stances)} R stances, {n_flights} flight(s) "
          f"({n_assisted} tuck-assisted, max apex {max_apex*100:.0f} cm), "
          f"{n_soft} soft-plant(s), {n_holds} held blink(s), "
          f"{n_fallbacks} non-flight gap(s) kept video motion")
    return v


def _anchor_value(v, any_plant, i, direction):
    """Anchor level for a plant-free gap: step up to
    FDF_ANCHOR_BACKOFF_FRAMES deeper into the neighboring stance
    (direction -1 = takeoff side, +1 = landing side) and read v there.
    The edge-most planted frames often carry the same perception drift
    that opened the gap; a few frames in, the level is clean."""
    j = i
    for _ in range(FDF_ANCHOR_BACKOFF_FRAMES):
        k = j + direction
        if 0 <= k < len(v) and any_plant[k]:
            j = k
        else:
            break
    return v[j]


def _still_feet_cols(toe_xz, g0, g1):
    """Feet whose world toe XZ stays within FDF_SOFTPLANT_HORIZ_M of
    its gap-median across [g0, g1] inclusive — evidence of a ground
    contact the stance detector lost (squat/kneel), as opposed to
    feet that traveled (rolls, crawls, real steps). toe_xz is
    (n, 2 feet, 2). Returns the list of still foot columns."""
    cols = []
    for c in (0, 1):
        seg = toe_xz[g0:g1 + 1, c]
        med = np.median(seg, axis=0)
        dev = np.hypot(seg[:, 0] - med[0], seg[:, 1] - med[1])
        if float(dev.max()) <= FDF_SOFTPLANT_HORIZ_M:
            cols.append(c)
    return cols


def _find_runs(arr, min_len):
    """Return list of (start, end) inclusive index pairs for runs of
    True with length >= min_len."""
    n = len(arr)
    runs = []
    i = 0
    while i < n:
        if arr[i]:
            j = i
            while j < n and arr[j]:
                j += 1
            if j - i >= min_len:
                runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def _extend_stances_y_only(stances, vy, foot_y, height_cap,
                            v_thresh, n_frames):
    """Extend XYZ-detected stances into adjacent frames where Y-only
    velocity is below threshold AND foot is below the (per-frame)
    height cap. Merges extended stances that overlap or touch. See the
    call site in detect_stances_per_foot for the rationale."""
    if not stances:
        return []
    extended = []
    for s_start, s_end in stances:
        new_start = s_start
        while new_start > 0:
            f = new_start - 1
            if vy[f] < v_thresh and foot_y[f] < height_cap[f]:
                new_start = f
            else:
                break
        new_end = s_end
        while new_end < n_frames - 1:
            f = new_end + 1
            if vy[f] < v_thresh and foot_y[f] < height_cap[f]:
                new_end = f
            else:
                break
        extended.append((new_start, new_end))
    return _merge_adjacent_stances(extended)


def _merge_adjacent_stances(stances):
    """Sort and merge stances that overlap or touch (end + 1 == start)."""
    if not stances:
        return []
    s = sorted(stances)
    merged = [s[0]]
    for start, end in s[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _stance_corrections(stances, foot_y_series, floor_y):
    """For each stance, compute (start, end, correction) tuple.

    correction = floor_y - MIN(foot_y during stance). Grounds the
    LOWEST frame of the stance to the floor; other frames in the
    stance stay at or above floor (the spread within the stance is
    preserved). Using MIN instead of median is what keeps the lowest
    moment from being pushed BELOW floor — when median > min, a
    median-based correction would shift everything down by
    (median - min), pushing the min frame underground. With min, the
    correction is exactly what's needed to bring the lowest moment
    of the stance up to floor level, no more.

    Side benefit: when the entire clip is detected as one stance
    (low-motion takes), this becomes correction = 0 (since min ==
    global floor by definition), making the per-stance pass a clean
    no-op on top of animation_shift_y's global grounding.
    """
    out = []
    for start, end in stances:
        mn = float(np.min(foot_y_series[start:end + 1]))
        correction = floor_y - mn
        out.append((start, end, correction))
    return out


def _build_correction_array(left_corr, right_corr, n_frames, blend_frames):
    """Build per-frame correction array from per-foot stance lists.

    Combination rule: when both feet are simultaneously planted, take
    the SMALLER-MAGNITUDE correction so neither foot punches through
    the floor. The higher foot stays slightly above floor (accepted
    residual; can be cleaned up by a future Pass 2 IK lock).

    Smoothing: at stance edges, blend toward 0 over `blend_frames` so
    Hips doesn't snap. Between two nearby stances (gap < 2*blend_frames),
    interpolate by inverse distance instead of decaying to zero and back.
    """
    left_arr = [None] * n_frames
    right_arr = [None] * n_frames
    for s, e, c in left_corr:
        for f in range(s, e + 1):
            if 0 <= f < n_frames:
                left_arr[f] = c
    for s, e, c in right_corr:
        for f in range(s, e + 1):
            if 0 <= f < n_frames:
                right_arr[f] = c

    raw = [None] * n_frames
    for i in range(n_frames):
        l = left_arr[i]
        r = right_arr[i]
        if l is not None and r is not None:
            raw[i] = l if abs(l) <= abs(r) else r
        elif l is not None:
            raw[i] = l
        elif r is not None:
            raw[i] = r

    smoothed = np.zeros(n_frames)
    for i in range(n_frames):
        if raw[i] is not None:
            smoothed[i] = raw[i]
            continue
        left_val, left_dist = None, None
        for k in range(1, blend_frames + 1):
            j = i - k
            if j < 0:
                break
            if raw[j] is not None:
                left_val = raw[j]
                left_dist = k
                break
        right_val, right_dist = None, None
        for k in range(1, blend_frames + 1):
            j = i + k
            if j >= n_frames:
                break
            if raw[j] is not None:
                right_val = raw[j]
                right_dist = k
                break
        if left_val is None and right_val is None:
            smoothed[i] = 0.0
        elif left_val is not None and right_val is None:
            blend = 1.0 - (left_dist / (blend_frames + 1))
            smoothed[i] = left_val * blend
        elif right_val is not None and left_val is None:
            blend = 1.0 - (right_dist / (blend_frames + 1))
            smoothed[i] = right_val * blend
        else:
            total = left_dist + right_dist
            w_l = right_dist / total
            w_r = left_dist / total
            smoothed[i] = left_val * w_l + right_val * w_r
    return smoothed
