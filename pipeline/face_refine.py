# face_refine.py
# ==============
# Recomputes the face blendshape channels MediaPipe's blendshape head
# gets wrong, from data the capture already saves:
#
#   Tier 1 (geometric): jawOpen from CHIN DROP instead of lip gap, the
#     lip-gap remainder routed into mouthUpperUp*/mouthLowerDown*,
#     noseSneer* from the nasolabial complex, browDown* from brow-to-eye
#     drop + glabella narrowing, mouthFrown* from corner depression —
#     all measured on rigidly-stabilized landmarks as deltas from the
#     actor's own neutral (auto-detected from the clip).
#   Tier 2 (pixel gate): the mouth_stats stored at capture time
#     (save_face_data.py) tell teeth apart from open-mouth cavity, so a
#     teeth smile can't read as a gaping jaw even when landmarks drift.
#     Gate only ever REDUCES jawOpen, never raises it, and is skipped
#     when the lips are sealed (a closed-lips jaw drop is real jaw —
#     that's what mouthClose is for).
#   Tier 3 (solver): box-constrained least-squares fit of ICT-FaceKit
#     expression shapes (pipeline/face_solver.py) for whole-mouth
#     consistency. Falls back to Tier 1 values when the baked basis is
#     missing.
#
# Runs on the raw NPZ data BEFORE smoothing/denoise/amp — wired into
# npz_to_bvh.py and pipeline/smooth_face_npz.py behind --face-refine.
# The NPZ on disk is never modified; refinement is in-memory at export,
# so different settings never require re-running inference.
#
# Modes:
#   off        pass-through (MediaPipe values untouched)
#   geometric  Tier 1 + Tier 2
#   full       Tier 1 + Tier 2 + Tier 3 (default; auto-falls back to
#              geometric when pipeline/ict_face_basis.npz is absent)

import json
import os

import numpy as np

from pipeline import face_landmark_topology as topo

REFINE_MODES = ("off", "geometric", "full")
DEFAULT_MODE = "full"

# ── Tuning constants ─────────────────────────────────────────────────
# All distances are in canonical units (interocular = 1). "FULL" values
# are the measured delta that should map to channel value 1.0; ICT-
# FaceKit's shape magnitudes informed the initial numbers.


class RefineConfig:
    # jawOpen from chin drop, as a fraction of face height (forehead->
    # chin). A fully open jaw drops the chin ~24% of face height.
    JAW_FULL_FRAC = 0.24
    JAW_DEADZONE_FRAC = 0.012

    # Lip / brow / sneer channels are LOCAL DIFFERENTIALS (lip vs
    # subnasale, lip vs chin, brow vs eye corners, wing vs nose column)
    # — absolute positions carry head-pose drift on real footage. The
    # *_DEADZONE values are static floors; the effective deadzone per
    # channel is raised adaptively to the clip's own noise level,
    # measured on its neutral frames (see _adaptive_ramp).
    UPPER_UP_FULL = 0.13       # lip-to-subnasale shrink at mouthUpperUp=1
    LOWER_DOWN_FULL = 0.10     # lip-to-chin shrink at mouthLowerDown=1
    LIP_DEADZONE = 0.008

    SNEER_FULL = 0.055         # wing-vs-column + lip-vs-subnasale rise
    SNEER_DEADZONE = 0.008

    BROW_FULL = 0.055          # brow drop (+ narrowing term) at =1
    BROW_DEADZONE = 0.007
    BROW_NARROW_WEIGHT = 0.5

    FROWN_FULL = 0.055         # center-relative corner drop at =1
    FROWN_DEADZONE = 0.006

    # mouthClose: inner-lip gap expected at jawOpen=1 (canonical units);
    # closure = how much of that expected gap the lips are holding shut.
    GAP_PER_JAW = 0.50
    CLOSE_MIN_JAW = 0.05

    # Jaw fusion. The inner-lip GAP is the honest openness signal (huge
    # on real opens, zero on puckers). The chin/solver estimate only
    # contributes the sealed-lips jaw-drop case, capped (lips can't stay
    # sealed past ~half jaw) and guarded against puckers — in 2D a
    # pucker is the identical twin of a sealed jaw drop (whole lower
    # face translates down), but a pucker NARROWS the mouth 0.15-0.3 IO
    # while jaw drops don't.
    JAW_GAP_DZ = 0.02
    JAW_GAP_FULL = 0.45        # inner gap at jawOpen=1
    SEALED_JAW_CAP = 0.55
    PUCKER_LO = 0.06           # mouth-width shrink where guard starts
    PUCKER_HI = 0.20           # ... and where it fully mutes the sealed path

    # Tier-2 gate (see refine notes above). cavity_norm units: mouth-
    # interior open-pixel area / interocular^2.
    GATE_CAVITY_LO = 0.006
    GATE_CAVITY_HI = 0.030
    GATE_FLOOR = 0.15          # never remove more than 85% of jawOpen
    GATE_MIN_GAP = 0.05        # lips-parted threshold for gating
    TEETH_CLAMP_NORM = 0.020   # teeth area that triggers the hard clamp
    TEETH_CLAMP_JAW = 0.10
    # Central-band variants (mouth_stats v2): the middle 60% of the
    # mouth is where teeth vs cavity is unambiguous — wide teeth-bares
    # have dark buccal-corridor CORNERS that fooled the whole-polygon
    # gate. Thresholds are ~0.6x the whole-polygon ones (smaller area).
    GATE_C_LO = 0.004
    GATE_C_HI = 0.020
    TEETH_C_NORM = 0.012

    # tongueOut detection (mouth_stats v2 central band). A protruding
    # tongue fills the mouth center with LIT FLESH: bright (unlike the
    # shadowed in-mouth cavity, which stays dark no matter how red) and
    # saturated (unlike teeth). Lips never enter the sampled polygon;
    # gums only appear welded to strong teeth evidence, which vetoes.
    # Thresholds are lighting-adaptive off the same skin/lip references
    # as the jaw gate. Calibrated on "Tongue Test Video.mp4".
    TONGUE_FLESH_V = 0.62      # x skin V median, floor 100 (lit flesh)
    TONGUE_FLESH_S_HI = 1.35   # x lip S median, ceil 150 (still flesh)
    TONGUE_GAP_LO = 0.08       # lips-parted gate
    TONGUE_GAP_HI = 0.18
    TONGUE_BAND_LO = 0.025     # central-band area gate (vs interocular^2)
    TONGUE_BAND_HI = 0.060
    TONGUE_TEETH_VETO_LO = 0.18   # full grin = gums -> veto ramps in
    TONGUE_TEETH_VETO_HI = 0.32
    TONGUE_CAV_VETO_LO = 0.12     # dark center = open mouth, not tongue
    TONGUE_CAV_VETO_HI = 0.30
    # Flesh-fraction gates, calibrated on real true/false runs. The
    # central band separates everted-lip speech (0.43-0.60 flesh) from
    # tongues (0.70+). The MIDDLE THIRD carries a raised bar because
    # warm lighting lifts teeth saturation past the teeth cut and they
    # count as flesh: a warm-lit teeth grin measures 0.79 mid-flesh vs
    # 0.87-1.00 for every clean blep on the calibration clips.
    # FAVOR-TEETH POLICY: ambiguous grin-vs-tongue frames resolve to
    # teeth — a missed marginal tongue beats a tongue poking through a
    # smile, and end-user footage skews warm/noisy. Teeth+tongue
    # combos where flesh does not dominate the middle third are
    # deliberate casualties of that bar.
    TONGUE_FLESH_GATE_LO = 0.55      # central band
    TONGUE_FLESH_GATE_HI = 0.70
    TONGUE_FLESH_GATE_MID_LO = 0.80  # middle third (raised bar)
    TONGUE_FLESH_GATE_MID_HI = 0.92
    TONGUE_RAMP_LO = 0.15      # gated evidence -> channel value
    TONGUE_RAMP_HI = 0.55
    # A real blep holds for ~0.3s+ (shortest real run on the
    # calibration clips: 8 frames at 24fps); FP fragments — blip
    # noise, the 3-4 frame head of a grin cresting the flesh bar
    # before the teeth fully bare — all ran 4 frames or less.
    TONGUE_MIN_RUN = 5         # frames (~0.2s at 24fps)
    # A protruding tongue also HOLDS the lips apart, so on tongue
    # frames the inner gap over-reads the jaw: lips-hugging bleps read
    # as gapes. jawOpen is instead capped by the chin-driven solver
    # estimate — low when only the lips are parted around the tongue.
    # On full downward bleps the tongue drapes OVER the chin and the
    # chin landmarks ride it, so that estimate over-reads too: the
    # ceiling bounds every tongue frame at a partway-open mouth (the
    # tongue itself fills the view; no signal can measure the mandible
    # behind it). The floor is a cap minimum, not a jaw minimum —
    # closed-mouth slight stick-outs stay closed on their own gap
    # signal.
    TONGUE_JAW_CHIN_K = 1.15   # cap = K x solver jawOpen
    TONGUE_JAW_FLOOR = 0.12    # cap never drops below this
    TONGUE_JAW_CEIL = 0.45     # ...nor rises above this

    # Face-size reliability: below REL_IO_MIN pixels of interocular the
    # landmarks are too coarse to out-measure MediaPipe's own head, so
    # the refined values crossfade back to the original MediaPipe
    # values (full fallback at/below the MIN, full refine at/above the
    # GOOD size). Typical face-only captures run 150px+; tiny faces in
    # full-body shots run 20-60px.
    REL_IO_MIN = 45.0
    REL_IO_GOOD = 110.0

    mode = DEFAULT_MODE
    neutral_frame = -1


# Channels each tier owns. In geometric mode GEO_CHANNELS are replaced;
# in full mode SOLVER_CHANNELS come from the ICT fit and the rest of
# GEO_CHANNELS stay geometric. The split follows each channel's
# strongest estimator (validated in dev_tests/test_face_refine.py):
# the solver recovers chin/nose-driven shapes cleanly (jaw family,
# sneer) but splits weight between collinear lip/brow shapes, where the
# direct geometric measurements are more reliable.
GEO_CHANNELS = [
    "jawOpen", "mouthClose",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "noseSneerLeft", "noseSneerRight",
    "browDownLeft", "browDownRight",
    "mouthFrownLeft", "mouthFrownRight",
]
SOLVER_CHANNELS = [
    "jawOpen", "mouthClose",
    "noseSneerLeft", "noseSneerRight",
    "jawLeft", "jawRight",
]


def _smoothstep(x, lo, hi):
    t = np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _ramp(delta, deadzone, full):
    """Map a measured delta to [0,1]: 0 below deadzone, 1 at `full`."""
    return np.clip((delta - deadzone) / max(full - deadzone, 1e-9),
                   0.0, 1.0)


def _median3(x):
    """3-tap median filter (kills single-frame flicker), edge-padded."""
    if len(x) < 3:
        return x.copy()
    stacked = np.stack([np.r_[x[0], x[:-1]], x, np.r_[x[1:], x[-1]]])
    return np.median(stacked, axis=0)


def fuse_jaw(inner_gap, width_shrink, sealed_src, cfg=RefineConfig):
    """Gap-anchored jaw fusion (see RefineConfig jaw-fusion notes).

    inner_gap:    (N,) inner-lip gap, canonical units
    width_shrink: (N,) mouth-width narrowing vs neutral (>= 0)
    sealed_src:   (N,) chin-based jaw estimate (solver or geometric) —
                  only trusted for the sealed-lips case, capped and
                  pucker-guarded.
    Returns (jaw (N,), pucker_guard (N,))."""
    gap_jaw = _ramp(inner_gap, cfg.JAW_GAP_DZ, cfg.JAW_GAP_FULL)
    pucker_guard = 1.0 - _smoothstep(width_shrink, cfg.PUCKER_LO,
                                     cfg.PUCKER_HI)
    sealed = np.minimum(sealed_src, cfg.SEALED_JAW_CAP) * pucker_guard \
        * (1.0 - _smoothstep(inner_gap, 0.10, 0.25))
    return np.maximum(gap_jaw, sealed), pucker_guard


def chin_cap_jaw(jaw, tongue, sealed_src, cfg=RefineConfig):
    """Tongue frames: cap the gap-anchored jaw by chin evidence.

    jaw:        (N,) jawOpen after fusion/gating
    tongue:     (N,) tongueOut channel (0-1), doubles as blend weight
    sealed_src: (N,) chin-driven solver jawOpen estimate
    Returns (jaw (N,), capped mask (N,))."""
    cap = np.clip(cfg.TONGUE_JAW_CHIN_K * sealed_src,
                  cfg.TONGUE_JAW_FLOOR, cfg.TONGUE_JAW_CEIL)
    w = np.clip(tongue, 0.0, 1.0)
    capped = (w > 0.2) & (jaw > cap)
    return jaw + w * (np.minimum(jaw, cap) - jaw), capped


def _adaptive_ramp(raw, neutral_idx, static_dz, full, dz_cap_frac=0.5):
    """Baseline-subtract + noise-adaptive deadzone + ramp to [0,1].

    The clip's own neutral frames define zero (their median) and the
    noise floor (p95 spread): whatever residual drift the differential
    measure still carries on THIS clip sets the deadzone, capped at
    dz_cap_frac * full so an expressive 'neutral' can't kill the
    channel. Returns (values, baseline, deadzone)."""
    if neutral_idx is not None and len(neutral_idx) >= 3:
        ref = raw[neutral_idx]
        base = float(np.median(ref))
        noise = float(np.percentile(np.abs(ref - base), 95))
    else:
        base, noise = 0.0, 0.0
    dz = min(max(static_dz, 1.5 * noise), dz_cap_frac * full)
    return _ramp(raw - base, dz, full), base, dz


# =====================================================================
#  TIER 1 — geometric channels
# =====================================================================

def compute_geometric_channels(stab, template, detected, face_height,
                               neutral_frames=None, cfg=RefineConfig):
    """Returns (channels dict name->(N,), measures dict, diag dict).

    Every channel is a LOCAL DIFFERENTIAL (distance between nearby
    landmark groups) — absolute positions vs the template drift with
    residual head pose on real footage. Each raw signal is then
    baseline-subtracted and deadzoned against the clip's own neutral
    frames (_adaptive_ramp)."""
    det = np.asarray(detected, dtype=bool)
    T = template

    def y_mean(idx_list, arr):
        return arr[:, idx_list, 1].mean(axis=1)

    def y_mean_t(idx_list):
        return float(T[idx_list, 1].mean())

    def rise_rel(part_idx, ref_idx):
        """How much `part` rose relative to `ref` since the template
        (positive = part moved up relative to its local anchor)."""
        d_f = y_mean(part_idx, stab) - y_mean(ref_idx, stab)
        d_t = y_mean_t(part_idx) - y_mean_t(ref_idx)
        return d_f - d_t

    diag = {}

    def ramp(name, raw, static_dz, full):
        vals, base, dz = _adaptive_ramp(raw, neutral_frames, static_dz,
                                        full)
        diag[name] = {"baseline": round(base, 4),
                      "deadzone": round(dz, 4)}
        return vals

    # ── Common-mode drift (for the chin only — the one channel that
    # has no adjacent rigid anchor): vertical shift of the nose column,
    # scaled by height below the eye line, deadzoned against noise.
    common = y_mean(topo.NOSE_COLUMN, stab) - y_mean_t(topo.NOSE_COLUMN)
    common = np.sign(common) * np.maximum(np.abs(common) - 0.006, 0.0)
    h_nose = max(-y_mean_t(topo.NOSE_COLUMN), 0.15)
    h_chin = max(-y_mean_t(topo.CHIN_CLUSTER), 0.0)
    # Cap the extrapolation ratio: it also multiplies column NOISE, and
    # an uncapped ~2.4x turned the jaw's neutral-frame noise floor into
    # a deadzone that swallowed real jaw motion on real footage.
    drift_ratio = min(h_chin / h_nose, 1.6)

    # ── Jaw: chin drop, in face-height units ──────────────────────
    chin_drop = (y_mean_t(topo.CHIN_CLUSTER)
                 - y_mean(topo.CHIN_CLUSTER, stab)) \
        + common * drift_ratio
    jaw = ramp("jawOpen", chin_drop,
               cfg.JAW_DEADZONE_FRAC * face_height,
               cfg.JAW_FULL_FRAC * face_height)

    # ── Inner-lip gap (differential by construction) ──────────────
    gap = np.zeros(len(stab))
    for u, l in topo.INNER_LIP_PAIRS:
        gap += np.clip(stab[:, u, 1] - stab[:, l, 1], 0.0, None)
    gap /= len(topo.INNER_LIP_PAIRS)
    gap_t = 0.0
    for u, l in topo.INNER_LIP_PAIRS:
        gap_t += max(float(T[u, 1] - T[l, 1]), 0.0)
    gap = np.clip(gap - gap_t / len(topo.INNER_LIP_PAIRS), 0.0, None)

    # ── Lip-part channels ─────────────────────────────────────────
    # Upper lip rises toward the subnasale (which barely moves with
    # the lips); lower lip drops toward the chin. Measuring against
    # the chin makes mouthLowerDown jaw-independent automatically: a
    # jaw drop moves chin MORE than lip (ride ~0.8), so the lip-chin
    # distance grows and the channel stays quiet.
    upper_up_l = ramp("mouthUpperUpLeft",
                      rise_rel(topo.UPPER_LIP_INNER_LEFT, topo.SUBNASALE),
                      cfg.LIP_DEADZONE, cfg.UPPER_UP_FULL)
    upper_up_r = ramp("mouthUpperUpRight",
                      rise_rel(topo.UPPER_LIP_INNER_RIGHT,
                               topo.SUBNASALE),
                      cfg.LIP_DEADZONE, cfg.UPPER_UP_FULL)
    # mouthLowerDown is gap-gated: with lips SEALED over a jaw drop the
    # lower lip rides UP with the seal while the chin falls, so the
    # lip-chin distance shrinks and reads as a (false) depressor. A
    # real mouthLowerDown always parts the lips / bares lower teeth.
    lip_gap_gate = _smoothstep(gap, 0.02, 0.06)
    lower_down_l = ramp("mouthLowerDownLeft",
                        -rise_rel(topo.LOWER_LIP_INNER_LEFT,
                                  topo.CHIN_CLUSTER),
                        cfg.LIP_DEADZONE, cfg.LOWER_DOWN_FULL) \
        * lip_gap_gate
    lower_down_r = ramp("mouthLowerDownRight",
                        -rise_rel(topo.LOWER_LIP_INNER_RIGHT,
                                  topo.CHIN_CLUSTER),
                        cfg.LIP_DEADZONE, cfg.LOWER_DOWN_FULL) \
        * lip_gap_gate

    # ── Sneer: wing rise vs nose column + lateral lip vs subnasale ─
    sneer_l = ramp("noseSneerLeft",
                   0.6 * rise_rel(topo.NOSE_WING_LEFT, topo.NOSE_COLUMN)
                   + 0.6 * rise_rel(topo.UPPER_LIP_OUTER_LEFT,
                                    topo.SUBNASALE),
                   cfg.SNEER_DEADZONE, cfg.SNEER_FULL)
    sneer_r = ramp("noseSneerRight",
                   0.6 * rise_rel(topo.NOSE_WING_RIGHT, topo.NOSE_COLUMN)
                   + 0.6 * rise_rel(topo.UPPER_LIP_OUTER_RIGHT,
                                    topo.SUBNASALE),
                   cfg.SNEER_DEADZONE, cfg.SNEER_FULL)

    # ── Brow down: drop vs eye corners + glabella narrowing ───────
    narrow = (float(T[topo.LEFT_BROW_INNER_LOW, 0]
                    - T[topo.RIGHT_BROW_INNER_LOW, 0])
              - (stab[:, topo.LEFT_BROW_INNER_LOW, 0]
                 - stab[:, topo.RIGHT_BROW_INNER_LOW, 0]))
    narrow = np.clip(narrow, 0.0, None)
    brow_l = ramp("browDownLeft",
                  -rise_rel(topo.LEFT_BROW_LOWER, topo.EYE_CORNERS_LEFT)
                  + cfg.BROW_NARROW_WEIGHT * narrow,
                  cfg.BROW_DEADZONE, cfg.BROW_FULL)
    brow_r = ramp("browDownRight",
                  -rise_rel(topo.RIGHT_BROW_LOWER,
                            topo.EYE_CORNERS_RIGHT)
                  + cfg.BROW_NARROW_WEIGHT * narrow,
                  cfg.BROW_DEADZONE, cfg.BROW_FULL)

    # ── Frown: corner drop relative to the outer-lip center ───────
    lip_c = [topo.UPPER_LIP_OUTER_C, topo.LOWER_LIP_OUTER_C]
    frown_l = ramp("mouthFrownLeft",
                   -rise_rel([topo.MOUTH_CORNER_LEFT], lip_c),
                   cfg.FROWN_DEADZONE, cfg.FROWN_FULL)
    frown_r = ramp("mouthFrownRight",
                   -rise_rel([topo.MOUTH_CORNER_RIGHT], lip_c),
                   cfg.FROWN_DEADZONE, cfg.FROWN_FULL)

    channels = {
        "jawOpen": jaw,
        "mouthUpperUpLeft": upper_up_l,
        "mouthUpperUpRight": upper_up_r,
        "mouthLowerDownLeft": lower_down_l,
        "mouthLowerDownRight": lower_down_r,
        "noseSneerLeft": sneer_l,
        "noseSneerRight": sneer_r,
        "browDownLeft": brow_l,
        "browDownRight": brow_r,
        "mouthFrownLeft": frown_l,
        "mouthFrownRight": frown_r,
    }
    for name in channels:
        channels[name] = np.where(det, channels[name], 0.0)

    # Mouth-width shrink vs neutral (positive = narrower). Puckers
    # narrow the mouth hard; jaw drops don't — the jaw fusion uses this
    # to tell them apart.
    width = np.linalg.norm(stab[:, topo.MOUTH_CORNER_LEFT, :2]
                           - stab[:, topo.MOUTH_CORNER_RIGHT, :2],
                           axis=1)
    width_t = float(np.linalg.norm(T[topo.MOUTH_CORNER_LEFT, :2]
                                   - T[topo.MOUTH_CORNER_RIGHT, :2]))
    width_shrink = np.clip(width_t - width, 0.0, None)

    measures = {"chin_drop": chin_drop, "inner_gap": gap,
                "width_shrink": width_shrink}
    return channels, measures, diag


# =====================================================================
#  TIER 2 — mouth-pixel gate
# =====================================================================

# Layout must match save_face_data.extract_mouth_stats().
MOUTH_STATS_N = 82              # v1 width; v2 +central band; v3 +thirds
MOUTH_STATS_N_V2 = 147
MOUTH_STATS_N_V3 = 342
_HIST_OFFSET = 18
_HIST_BINS = 8          # 8 V-bins x 8 S-bins, V-major, bin width 32
_IDX_AREA, _IDX_IO = 0, 1
_IDX_FRAC_RED = 13
_IDX_SKIN_V, _IDX_SKIN_S = 16, 17
_IDX_C_AREA = 82
_HIST_C_OFFSET = 83
# Vertical thirds of the central band (v3): (area_idx, hist_offset).
_THIRDS = {"top": (147, 148), "mid": (212, 213), "bot": (277, 278)}


def _hist_fraction(hist, v_below=None, v_above=None, s_below=None):
    """Integrate the (N, 8, 8) V x S histogram with linear partial bins."""
    n = len(hist)
    v_edges = np.arange(_HIST_BINS + 1) * (256.0 / _HIST_BINS)
    w_v = np.ones(_HIST_BINS)
    if v_below is not None:
        w_v = np.clip((v_below - v_edges[:-1]) / (256.0 / _HIST_BINS),
                      0.0, 1.0)
    elif v_above is not None:
        w_v = np.clip((v_edges[1:] - v_above) / (256.0 / _HIST_BINS),
                      0.0, 1.0)
    w_s = np.ones(_HIST_BINS)
    if s_below is not None:
        w_s = np.clip((s_below - v_edges[:-1]) / (256.0 / _HIST_BINS),
                      0.0, 1.0)
    return np.einsum("nvs,v,s->n", hist.reshape(n, _HIST_BINS,
                                                _HIST_BINS), w_v, w_s)


def compute_mouth_gate(mouth_stats, inner_gap, detected,
                       cfg=RefineConfig):
    """Tier-2 gate from capture-time pixel statistics.

    Returns (gate (N,) in [GATE_FLOOR..1], teeth_clamp (N,) bool,
    info dict) — or (None, None, info) when stats are unusable."""
    info = {}
    if mouth_stats is None:
        return None, None, {"reason": "no mouth_stats in NPZ"}
    stats = np.asarray(mouth_stats, dtype=np.float64)
    if stats.ndim != 2 or stats.shape[1] < MOUTH_STATS_N:
        return None, None, {"reason": f"unexpected mouth_stats shape "
                                      f"{stats.shape}"}

    det = np.asarray(detected, dtype=bool)
    io_px = stats[:, _IDX_IO]
    valid = det & (io_px > 1.0)
    if valid.sum() < 5:
        return None, None, {"reason": "too few frames with pixel stats"}

    area_norm = np.zeros(len(stats))
    area_norm[valid] = stats[valid, _IDX_AREA] / (io_px[valid] ** 2)

    # Adaptive thresholds from the clip's own skin exposure.
    skin_v = stats[valid, _IDX_SKIN_V]
    skin_v_med = float(np.median(skin_v[skin_v > 1.0])) \
        if (skin_v > 1.0).any() else 128.0
    skin_s = stats[valid, _IDX_SKIN_S]
    skin_s_med = float(np.median(skin_s[skin_s > 0.0])) \
        if (skin_s > 0.0).any() else 80.0
    t_cavity = float(np.clip(0.42 * skin_v_med, 30.0, 90.0))
    # Teeth signature is DESATURATION, not out-brighting the skin —
    # teeth sit in the mouth's shadow, so on well-lit footage they are
    # usually darker than lit skin but far greyer than lips/gums.
    t_teeth_v = float(np.clip(0.55 * skin_v_med, 90.0, 160.0))
    t_teeth_s = float(np.clip(0.55 * skin_s_med, 30.0, 60.0))

    # Central band (mouth_stats v2) is the trustworthy region: wide
    # teeth-bares are bright in the CENTER with dark corridor corners
    # that fooled the whole-polygon statistics. Fall back to the whole
    # polygon on v1 captures.
    central = stats.shape[1] >= MOUTH_STATS_N_V2
    if central:
        hist = stats[:, _HIST_C_OFFSET:_HIST_C_OFFSET + 64]
        band_norm = np.zeros(len(stats))
        band_norm[valid] = (stats[valid, _IDX_C_AREA]
                            / io_px[valid] ** 2)
        lo, hi = cfg.GATE_C_LO, cfg.GATE_C_HI
        teeth_thresh = cfg.TEETH_C_NORM
    else:
        hist = stats[:, _HIST_OFFSET:_HIST_OFFSET + 64]
        band_norm = area_norm
        lo, hi = cfg.GATE_CAVITY_LO, cfg.GATE_CAVITY_HI
        teeth_thresh = cfg.TEETH_CLAMP_NORM

    frac_cavity = _hist_fraction(hist, v_below=t_cavity)
    frac_teeth = _hist_fraction(hist, v_above=t_teeth_v,
                                s_below=t_teeth_s)

    # Open evidence is DARKNESS only. The stored red fraction was
    # originally added as tongue evidence, but bared GUMS are just as
    # red — on real teeth-bares it counted the gums as an open mouth
    # and held the gate open. A tongue-filled mouth still shows a dark
    # cavity ring, so darkness alone is sufficient.
    cavity_norm = band_norm * frac_cavity
    teeth_norm = band_norm * frac_teeth

    g = _smoothstep(cavity_norm, lo, hi)
    g = _median3(g)
    gate = cfg.GATE_FLOOR + (1.0 - cfg.GATE_FLOOR) * g

    # Lips sealed -> no gating (mouthClose territory; pixels can't see
    # a closed-lips jaw drop).
    gate = np.where(inner_gap > cfg.GATE_MIN_GAP, gate, 1.0)
    gate = np.where(valid, gate, 1.0)

    # Teeth clamp: bright-desaturated CENTER with no central cavity =
    # tooth rows in contact -> jaw is hard-capped, whatever the corners
    # look like.
    teeth_clamp = valid & (inner_gap > cfg.GATE_MIN_GAP) \
        & (_median3(teeth_norm.astype(np.float64)) > teeth_thresh) \
        & (cavity_norm < lo)

    info = {
        "skin_v_median": skin_v_med,
        "t_cavity": t_cavity, "t_teeth_v": t_teeth_v,
        "t_teeth_s": t_teeth_s,
        "central_band": bool(central),
        "gated_frames": int((gate < 0.999).sum()),
        "teeth_clamped_frames": int(teeth_clamp.sum()),
    }
    return gate, teeth_clamp, info


def compute_tongue_out(mouth_stats, inner_gap, detected,
                       cfg=RefineConfig):
    """Pixel-based tongueOut channel from the central-band statistics.

    Signature of a protruding tongue: the mouth CENTER filled with lit,
    saturated flesh — bright enough to be outside the mouth's shadow
    (a resting tongue in an open mouth reads dark), saturated enough to
    not be teeth, inside the inner-lip polygon so it can't be lips, and
    without the strong teeth evidence that accompanies visible gums.
    Tuned to FAVOR TEETH over tongue: frames where teeth could account
    for the flesh (middle third not flesh-dominated) resolve to no
    tongue, and sub-TONGUE_MIN_RUN blips are dropped.

    Returns (values (N,), info dict) or (None, info) when the stats
    can't support detection (v1 capture / too few frames)."""
    info = {}
    if mouth_stats is None:
        return None, {"reason": "no mouth_stats"}
    stats = np.asarray(mouth_stats, dtype=np.float64)
    if stats.ndim != 2 or stats.shape[1] < MOUTH_STATS_N_V3:
        return None, {"reason": "needs mouth_stats v3 (vertical thirds)"}

    det = np.asarray(detected, dtype=bool)
    io_px = stats[:, _IDX_IO]
    valid = det & (io_px > 1.0)
    if valid.sum() < 5:
        return None, {"reason": "too few frames with pixel stats"}

    skin_v = stats[valid, _IDX_SKIN_V]
    skin_v_med = float(np.median(skin_v[skin_v > 1.0])) \
        if (skin_v > 1.0).any() else 128.0
    skin_s = stats[valid, _IDX_SKIN_S]
    skin_s_med = float(np.median(skin_s[skin_s > 0.0])) \
        if (skin_s > 0.0).any() else 80.0
    lip_s = stats[valid, 15]
    lip_s_med = float(np.median(lip_s[lip_s > 0.0])) \
        if (lip_s > 0.0).any() else 110.0

    t_flesh_v = max(cfg.TONGUE_FLESH_V * skin_v_med, 100.0)
    t_flesh_s_hi = min(cfg.TONGUE_FLESH_S_HI * lip_s_med, 150.0)
    t_teeth_v = float(np.clip(0.55 * skin_v_med, 90.0, 160.0))
    t_teeth_s = float(np.clip(0.55 * skin_s_med, 30.0, 60.0))
    t_cavity = float(np.clip(0.42 * skin_v_med, 30.0, 90.0))

    # The MIDDLE third of the central band is the deciding region:
    # top-to-bottom, a blep reads flesh/flesh/flesh, teeth+tongue reads
    # teeth/flesh/flesh, but an everted lower lip during speech reads
    # teeth/dark/flesh — judging flesh in the middle rejects the lip.
    a_mid, h_mid = _THIRDS["mid"]
    hist_mid = stats[:, h_mid:h_mid + 64]
    band = np.zeros(len(stats))
    band[valid] = stats[valid, a_mid] / io_px[valid] ** 2
    hist_central = stats[:, _HIST_C_OFFSET:_HIST_C_OFFSET + 64]

    # Lit flesh = (V >= t_flesh_v, S <= flesh_s_hi) minus the teeth
    # part of that box (V >= t_flesh_v, S <= t_teeth_s).
    def flesh_frac(h):
        lit = _hist_fraction(h, v_above=t_flesh_v, s_below=t_flesh_s_hi)
        lit_teeth = _hist_fraction(h, v_above=t_flesh_v,
                                   s_below=t_teeth_s)
        return np.clip(lit - lit_teeth, 0.0, 1.0)

    f_flesh_mid = flesh_frac(hist_mid)
    f_flesh_cen = flesh_frac(hist_central)
    f_cav = _hist_fraction(hist_mid, v_below=t_cavity)
    # Gum veto judges the WHOLE central band — a full grin's teeth can
    # sit above or below the middle third.
    f_teeth = _hist_fraction(hist_central, v_above=t_teeth_v,
                             s_below=t_teeth_s)

    # A tongue must DOMINATE with flesh both in the deciding middle
    # third (raised bar — warm teeth read as flesh, see RefineConfig)
    # and across the central band — everted-lip speech frames top out
    # around 0.6 on both.
    evidence = _smoothstep(f_flesh_mid, cfg.TONGUE_FLESH_GATE_MID_LO,
                           cfg.TONGUE_FLESH_GATE_MID_HI) \
        * _smoothstep(f_flesh_cen, cfg.TONGUE_FLESH_GATE_LO,
                      cfg.TONGUE_FLESH_GATE_HI) \
        * _smoothstep(inner_gap, cfg.TONGUE_GAP_LO, cfg.TONGUE_GAP_HI) \
        * _smoothstep(band, cfg.TONGUE_BAND_LO / 3.0,
                      cfg.TONGUE_BAND_HI / 3.0) \
        * (1.0 - _smoothstep(f_teeth, cfg.TONGUE_TEETH_VETO_LO,
                             cfg.TONGUE_TEETH_VETO_HI)) \
        * (1.0 - _smoothstep(f_cav, cfg.TONGUE_CAV_VETO_LO,
                             cfg.TONGUE_CAV_VETO_HI))
    vals = _ramp(_median3(evidence), cfg.TONGUE_RAMP_LO,
                 cfg.TONGUE_RAMP_HI)
    vals = np.where(valid, vals, 0.0)

    # Zero active runs shorter than TONGUE_MIN_RUN: FP blips and grin
    # heads survive the median filter at full strength but never hold
    # as long as a real blep.
    act = np.r_[False, vals > 0.05, False]
    edges = np.flatnonzero(np.diff(act.astype(np.int8)))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < cfg.TONGUE_MIN_RUN:
            vals[a:b] = 0.0

    info = {"t_flesh_v": round(t_flesh_v, 1),
            "t_flesh_s_hi": round(t_flesh_s_hi, 1),
            "active_frames": int((vals > 0.1).sum()),
            "p95": round(float(np.percentile(vals[det], 95)), 3)
            if det.any() else 0.0}
    return vals, info


# =====================================================================
#  ORCHESTRATION
# =====================================================================

def apply_face_refine(face_data, mode=None, neutral_frame=-1,
                      report_path="", cfg=RefineConfig):
    """Refine face_data['blendshapes'] in place. Returns a report dict.

    face_data: dict from pipeline.face_export.load_face_data — uses
        'landmarks', 'blendshapes', 'detected', 'bs_name_to_idx', and
        (when present) 'mouth_stats'.
    """
    mode = (mode or cfg.mode).lower()
    if mode not in REFINE_MODES:
        print(f"       WARNING: unknown --face-refine '{mode}', "
              f"using '{DEFAULT_MODE}'")
        mode = DEFAULT_MODE
    report = {"mode": mode}
    if mode == "off":
        print("       Face refine: off (MediaPipe values pass through)")
        return report

    landmarks = face_data["landmarks"]
    blendshapes = face_data["blendshapes"]
    detected = face_data["detected"].astype(bool)
    bs_idx = face_data["bs_name_to_idx"]

    if not detected.any():
        print("       Face refine: no detected frames — skipped")
        report["skipped"] = "no detected frames"
        return report

    # MediaPipe's own activity signal helps pick neutral frames.
    non_neutral_cols = [i for n, i in bs_idx.items()
                        if not str(n).startswith("_")
                        and not str(n).startswith("eyeLook")]
    mp_sum = blendshapes[:, non_neutral_cols].sum(axis=1)

    try:
        st = topo.stabilize_landmarks(landmarks, detected,
                                      mp_blendshape_sum=mp_sum,
                                      neutral_frame=neutral_frame)
    except ValueError as e:
        print(f"       Face refine: stabilization failed ({e}) — skipped")
        report["skipped"] = str(e)
        return report
    stab, template = st["stab"], st["template"]
    report["neutral_frames"] = st["neutral_frames"][:12]
    report["face_height"] = st["face_height"]

    # Per-frame tracked face size (pixels) -> refine reliability. Tiny
    # faces (full-body shots) crossfade back to MediaPipe's original
    # values instead of amplifying landmark noise.
    eye_l_px = landmarks[:, topo.LEFT_EYE_RING, :2].mean(axis=1)
    eye_r_px = landmarks[:, topo.RIGHT_EYE_RING, :2].mean(axis=1)
    io_px = np.linalg.norm(eye_l_px - eye_r_px, axis=1)
    reliability = _smoothstep(io_px, cfg.REL_IO_MIN, cfg.REL_IO_GOOD)
    reliability = np.where(detected, reliability, 0.0)
    io_med = float(np.median(io_px[detected]))
    rel_med = float(np.median(reliability[detected]))
    report["face_size"] = {"io_px_median": round(io_med, 1),
                           "reliability_median": round(rel_med, 2)}
    if rel_med < 0.999:
        print(f"       Face size: {io_med:.0f}px interocular -> refine "
              f"reliability {rel_med:.2f}"
              + (" (falling back to MediaPipe values)"
                 if rel_med < 0.05 else " (partial blend with MediaPipe)"
                 if rel_med < 0.95 else ""))

    neutral_idx = np.array([i for i in st["neutral_frames"]
                            if detected[i]], dtype=int)
    geo, measures, geo_diag = compute_geometric_channels(
        stab, template, detected, st["face_height"],
        neutral_frames=neutral_idx, cfg=cfg)
    report["geo"] = geo_diag
    inner_gap = measures["inner_gap"]

    gate, teeth_clamp, gate_info = compute_mouth_gate(
        face_data.get("mouth_stats"), inner_gap, detected, cfg)
    report["gate"] = gate_info
    if gate is None:
        print(f"       Face refine: pixel gate unavailable "
              f"({gate_info.get('reason', '?')}) — landmarks only")

    # tongueOut is computed BEFORE the jaw composition because it must
    # exempt the teeth/cavity gate: a protruding tongue FILLS the mouth
    # with lit flesh (no dark cavity), which reads exactly like a
    # teeth-smile and force-closed the jaw during bleps — the tongue
    # then exited through the shut teeth. No UI toggle by design;
    # BLENDCAP_FACE_TONGUE=0 is the support-case kill switch, same
    # pattern as BLENDCAP_NO_ROCM.
    if os.environ.get("BLENDCAP_FACE_TONGUE", "1").lower() \
            in ("0", "off", "false"):
        tongue, tongue_info = None, {"reason": "disabled via "
                                               "BLENDCAP_FACE_TONGUE"}
    else:
        tongue, tongue_info = compute_tongue_out(
            face_data.get("mouth_stats"), inner_gap, detected, cfg)
    report["tongue"] = tongue_info

    # ── Tier 3 ────────────────────────────────────────────────────
    solver_vals = {}
    if mode == "full":
        from pipeline import face_solver
        basis = face_solver.load_basis()
        if basis is None:
            print("       Face refine: ict_face_basis.npz missing — "
                  "falling back to geometric mode")
            report["solver"] = "basis missing"
        else:
            weights, names, diag = face_solver.solve_expressions(
                stab, template, detected, basis)
            # Baseline subtraction: the solver explains the actor-vs-ICT
            # identity difference with CONSTANT expression weights (on
            # real footage jawOpen idled at ~0.28, holding the rig's
            # mouth open at rest). The clip's neutral frames define
            # zero-expression — subtract their median and rescale so
            # 1.0 stays reachable.
            if len(neutral_idx) >= 3:
                base_w = np.minimum(
                    np.median(weights[neutral_idx], axis=0), 0.8)
                weights = np.clip(
                    (weights - base_w[None]) / (1.0 - base_w[None]),
                    0.0, 1.0)
                diag["baseline_max"] = round(float(base_w.max()), 3)
                diag["baseline"] = {
                    names[k]: round(float(base_w[k]), 3)
                    for k in np.argsort(base_w)[::-1][:6]
                    if base_w[k] > 0.05}
            report["solver"] = diag
            print(f"       ICT solve: {diag['n_solved']} frames, "
                  f"median residual {diag['resid_ratio_median']:.2f}, "
                  f"saturated {diag['saturated_frac'] * 100:.1f}%")
            # Per-channel noise floor, same idea as the geometric tier:
            # whatever a channel does on the clip's own neutral frames
            # is noise — deadzone it out and rescale so 1.0 survives.
            diag["channel_deadzones"] = {}
            for ch in SOLVER_CHANNELS:
                if ch not in names:
                    continue
                w = weights[:, names.index(ch)].astype(np.float64)
                if len(neutral_idx) >= 3:
                    dz = min(1.3 * float(
                        np.percentile(w[neutral_idx], 95)), 0.30)
                    if dz > 0.01:
                        w = np.clip((w - dz) / (1.0 - dz), 0.0, 1.0)
                        diag["channel_deadzones"][ch] = round(dz, 3)
                solver_vals[ch] = w

    # ── Compose final channels ────────────────────────────────────
    final = {}
    for ch in GEO_CHANNELS:
        if ch in ("mouthClose", "jawOpen"):
            continue                       # composed below
        src = "geo"
        vals = geo.get(ch)
        if ch in solver_vals:
            vals, src = solver_vals[ch], "solver"
        if vals is None:
            continue
        final[ch] = (vals, src)

    # ── Jaw: gap-anchored fusion (fuse_jaw) ───────────────────────
    # The inner-lip gap is the honest openness signal: huge on real
    # opens, zero on puckers, and independent of chin-landmark smear.
    # The chin/solver estimate contributes ONLY the sealed-lips case.
    sealed_src = solver_vals.get("jawOpen", geo.get("jawOpen"))
    jaw_src_name = "solver" if "jawOpen" in solver_vals else "geo"
    jaw, pucker_guard = fuse_jaw(inner_gap, measures["width_shrink"],
                                 sealed_src, cfg)
    src = f"gap+{jaw_src_name}"
    report["pucker_guarded_frames"] = int(
        ((pucker_guard < 0.95) & detected).sum())

    for ch in ("jawLeft", "jawRight"):     # solver-only extras, guarded
        if ch in solver_vals:
            final[ch] = (solver_vals[ch] * pucker_guard, "solver")

    # Tier-2 gate acts on jawOpen (and clamps for strong teeth evidence).
    # A detected tongue exempts both: tongue-out physically requires an
    # open jaw, and tongue flesh is exactly what the cavity gate can't
    # tell from a closed mouth.
    if gate is not None:
        gate_eff = gate if tongue is None else np.maximum(gate, tongue)
        jaw = jaw * gate_eff
        if teeth_clamp is not None:
            clamp_eff = teeth_clamp if tongue is None \
                else (teeth_clamp & (tongue < 0.3))
            jaw = np.where(clamp_eff,
                           np.minimum(jaw, cfg.TEETH_CLAMP_JAW), jaw)
        src += "+gate"
    # ...but the ungated gap ALSO over-reads on tongue frames — the
    # tongue itself holds the lips apart — so cap by the chin-driven
    # solver estimate (chin_cap_jaw). Solver only: the geometric
    # chin-drop runs too muted to anchor a cap without re-closing
    # real bleps.
    if tongue is not None and jaw_src_name == "solver":
        jaw, capped = chin_cap_jaw(jaw, tongue, sealed_src, cfg)
        report["tongue_chin_capped_frames"] = int(
            (capped & detected).sum())
        src += "+chincap"
    final["jawOpen"] = (jaw, src)

    # mouthClose: lips holding shut against an open jaw. Purely
    # derived from (gated jaw x lip closure) — the solver's own
    # mouthClose is too weak after baseline correction to be
    # trusted as a cap, and the derivation is already gap-safe.
    expected_gap = cfg.GAP_PER_JAW * jaw
    closure = np.clip(1.0 - inner_gap / np.maximum(expected_gap,
                                                   0.02), 0.0, 1.0)
    m_close = np.where(jaw > cfg.CLOSE_MIN_JAW, jaw * closure, 0.0)
    final["mouthClose"] = (m_close, "derived")

    # ── tongueOut: pixel-only channel MediaPipe doesn't provide ────
    # (computed above, before the jaw composition, so it can exempt
    # the jaw gate). Appended as a NEW column; every consumer matches
    # channels by name, and the source rig's calibration already
    # carries a tongueOut pose, so the Tongue bones pick it up
    # automatically.
    if tongue is not None:
        if "tongueOut" not in bs_idx:
            blendshapes = np.concatenate(
                [blendshapes, np.zeros((len(blendshapes), 1),
                                       dtype=blendshapes.dtype)], axis=1)
            face_data["blendshapes"] = blendshapes
            bs_idx["tongueOut"] = blendshapes.shape[1] - 1
            if "blendshape_names" in face_data:
                face_data["blendshape_names"].append("tongueOut")
        final["tongueOut"] = (tongue, "pixels")
        if tongue_info.get("active_frames", 0):
            print(f"       tongueOut: {tongue_info['active_frames']} "
                  f"active frames (p95 {tongue_info['p95']:.2f})")

    # ── Write into the blendshape array ───────────────────────────
    # Refined values crossfade with the original MediaPipe values by
    # per-frame face-size reliability (tiny face -> MediaPipe wins).
    applied = {}
    for ch, (vals, src) in final.items():
        col = bs_idx.get(ch)
        if col is None:
            continue
        col = int(col)
        old = blendshapes[:, col].copy()
        refined = np.clip(np.nan_to_num(vals), 0.0, 1.0)
        blended = reliability * refined + (1.0 - reliability) * old
        new = np.where(detected, blended, old)
        blendshapes[:, col] = new
        applied[ch] = {
            "source": src,
            "mp_p95": round(float(np.percentile(old[detected], 95)), 3),
            "new_p95": round(float(np.percentile(new[detected], 95)), 3),
        }
    report["channels"] = applied

    n_gated = gate_info.get("gated_frames", 0) if gate is not None else 0
    print(f"       Face refine ({mode}): {len(applied)} channels "
          f"recomputed, {n_gated} frames jaw-gated"
          + (f", {gate_info.get('teeth_clamped_frames', 0)} teeth-clamped"
             if gate is not None else ""))

    if report_path:
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"       Refine report -> {report_path}")
        except OSError as e:
            print(f"       WARNING: could not write refine report: {e}")
    return report
