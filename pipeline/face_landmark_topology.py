# face_landmark_topology.py
# =========================
# MediaPipe FaceLandmarker (478-point) topology constants and the
# stabilization used by the face-refine stage (pipeline/face_refine.py)
# and the ICT solver (pipeline/face_solver.py).
#
# SIDE CONVENTION — verified against the installed mediapipe 0.10.32
# FaceLandmarksConnections constants: the subject's LEFT eye is the
# 362/263 index group (iris 473-477), the subject's RIGHT eye is the
# 33/133 group (iris 468-472). "Left"/"Right" everywhere in this module
# means the SUBJECT's left/right — matching ARKit blendshape naming
# (browDownLeft = the subject's left brow), NOT image-space sides.
#
# Stabilization puts every frame's landmarks into one canonical face
# frame: origin at eye-mid, interocular distance = 1, +X toward the
# subject's LEFT, +Y up, +Z out of the face. Rotation/translation/scale
# are removed with a similarity Kabsch fit over expression-RIGID
# landmarks only (eye corners, nose bridge, forehead), so expression
# motion survives and head motion doesn't.

import numpy as np

# ── Semantic index sets (subject-relative sides) ─────────────────────
LEFT_EYE_RING = [263, 249, 390, 373, 374, 380, 381, 382,
                 362, 398, 384, 385, 386, 387, 388, 466]
RIGHT_EYE_RING = [33, 7, 163, 144, 145, 153, 154, 155,
                  133, 173, 157, 158, 159, 160, 161, 246]

# Brow rows, inner -> outer. "lower" = the row along the brow's lower
# edge (closest to the eye), the one that drops in a frown.
LEFT_BROW_LOWER = [285, 295, 282, 283, 276]
LEFT_BROW_UPPER = [336, 296, 334, 293, 300]
RIGHT_BROW_LOWER = [55, 65, 52, 53, 46]
RIGHT_BROW_UPPER = [107, 66, 105, 63, 70]
LEFT_BROW_INNER = 336        # by the glabella
RIGHT_BROW_INNER = 107
LEFT_BROW_INNER_LOW = 285
RIGHT_BROW_INNER_LOW = 55

# Lips. Ring lists are in continuous drawing order (usable as polygons).
LIPS_OUTER_RING = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                   291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER_RING = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
                   308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
MOUTH_CORNER_LEFT = 291      # outer corner, subject left
MOUTH_CORNER_RIGHT = 61
MOUTH_CORNER_LEFT_IN = 308
MOUTH_CORNER_RIGHT_IN = 78
UPPER_LIP_OUTER_C = 0
LOWER_LIP_OUTER_C = 17
UPPER_LIP_INNER_C = 13
LOWER_LIP_INNER_C = 14
# Center-biased side clusters (the third, cornermost point of each
# inner-lip triad — 310/318/80/88 — is excluded: smiles lift it hard,
# which over-drove mouthUpperUp/mouthLowerDown on real footage).
UPPER_LIP_INNER_LEFT = [312, 311]
LOWER_LIP_INNER_LEFT = [317, 402]
UPPER_LIP_INNER_RIGHT = [82, 81]
LOWER_LIP_INNER_RIGHT = [87, 178]
UPPER_LIP_OUTER_LEFT = [267, 269, 270]
UPPER_LIP_OUTER_RIGHT = [37, 39, 40]
# Vertical inner-lip pairs (upper_idx, lower_idx), center + 3 per side.
INNER_LIP_PAIRS = [(13, 14),
                   (82, 87), (81, 178), (80, 88),
                   (312, 317), (311, 402), (310, 318)]

# Nose. Wing groups sit just above/behind each nostril.
NOSE_TIP = 1
NOSE_UNDER = 2
NOSE_BRIDGE = 168
NOSE_WING_LEFT = [294, 327, 278, 440]
NOSE_WING_RIGHT = [64, 98, 48, 220]
# Nose column (lower bridge -> tip): the common-mode reference for the
# lower face. Expression moves it far less than lips/chin/wings, but
# residual out-of-plane pitch (which the similarity stabilization can't
# fully remove, especially on small faces) shifts it together with the
# whole lower face — so its vertical delta measures perspective drift.
NOSE_COLUMN = [1, 4, 5, 195]

# Chin: menton (152) plus its face-oval neighbors for a robust centroid.
CHIN_CENTER = 152
CHIN_CLUSTER = [152, 148, 176, 377, 400]
FOREHEAD_TOP = 10

# Cheek reference points (for exposure sampling in the mouth-stats
# extractor; roughly mid-cheek, outside the smile fold).
CHEEK_LEFT = 425
CHEEK_RIGHT = 205

# Local differential references. Real footage taught us (see
# dev_tests/analyze_face_clip.py output on Face Example .mp4) that
# ABSOLUTE landmark positions vs the neutral template carry ~0.02-0.05
# interocular units of head-pose-correlated drift — expression-sized
# noise. Distances between NEARBY points cancel that drift, so every
# geometric channel measures against a local anchor instead:
#   upper lip  -> subnasale (nose base, barely moves with lips)
#   lower lip  -> chin cluster (rides the jaw WITH the lip)
#   brows      -> eye corners (rigid, adjacent)
#   nose wings -> nose column (adjacent, rises far less in a sneer)
SUBNASALE = [2, 97, 326]         # nose base center + nostril bottoms
EYE_CORNERS_LEFT = [362, 263]
EYE_CORNERS_RIGHT = [33, 133]

# Landmarks that barely move under facial expression: eye corners, nose
# bridge column, upper forehead. Used for the rigid Kabsch fit. (Chin,
# lips, brows, nose wings are deliberately absent.)
RIGID_INDICES = [33, 133, 362, 263,           # eye corners
                 168, 6, 197, 195,            # nose bridge column
                 10, 109, 338, 67, 297,       # upper forehead arc
                 151, 9, 8,                   # forehead / glabella mid
                 127, 356, 234, 454]          # temple / upper oval sides

N_LANDMARKS = 478


# ── Similarity Kabsch ────────────────────────────────────────────────

def similarity_fit(src, dst, z_weight=1.0):
    """Least-squares similarity transform mapping src -> dst.
    Both (P, 3). Returns (s, R, t) with dst ~ s * R @ src + t.
    z_weight < 1 downweights the z axis (MediaPipe z is the noisiest)."""
    w = np.array([1.0, 1.0, float(z_weight)])
    src_w = src * w
    dst_w = dst * w
    mu_s = src_w.mean(axis=0)
    mu_d = dst_w.mean(axis=0)
    a = src_w - mu_s
    b = dst_w - mu_d
    cov = b.T @ a
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    var = (a ** 2).sum()
    s = (S * np.diag(D)).sum() / max(var, 1e-12)
    # Recompute translation in unweighted space so the transform applies
    # to raw coordinates.
    t = dst.mean(axis=0) - s * R @ src.mean(axis=0)
    return s, R, t


def canonical_frame_from(landmarks):
    """Build the canonical face frame from one set of landmarks.
    Returns (origin, R_rows, scale): canonical = ((p - origin) @ R_rows.T) * scale.
    +X = subject left, +Y = up (chin->bridge), +Z = out of the face."""
    eye_l = landmarks[LEFT_EYE_RING].mean(axis=0)
    eye_r = landmarks[RIGHT_EYE_RING].mean(axis=0)
    origin = (eye_l + eye_r) / 2.0
    left = eye_l - eye_r
    io = np.linalg.norm(left)
    left = left / max(io, 1e-9)
    up = landmarks[NOSE_BRIDGE] - landmarks[CHIN_CENTER]
    up = up - left * (up @ left)
    up = up / max(np.linalg.norm(up), 1e-9)
    fwd = np.cross(left, up)
    R_rows = np.stack([left, up, fwd], axis=0)
    return origin, R_rows, 1.0 / max(io, 1e-9)


def to_canonical(landmarks, origin, R_rows, scale):
    return ((landmarks - origin) @ R_rows.T) * scale


# ── Stabilization + neutral template ─────────────────────────────────

def compute_activity(canon_landmarks, detected, mp_blendshape_sum=None):
    """Per-frame expression-activity score used to find neutral frames.
    canon_landmarks: (N, 478, 3) canonical-ish (interocular ~ 1).
    Lower = more neutral. Undetected frames get +inf."""
    n = len(canon_landmarks)
    act = np.full(n, np.inf, dtype=np.float64)
    det = detected.astype(bool)
    if not det.any():
        return act
    lm = canon_landmarks[det]

    # Inner-lip gap (any mouth opening is activity).
    gap = np.zeros(det.sum())
    for u, l in INNER_LIP_PAIRS:
        gap += np.abs(lm[:, u, 1] - lm[:, l, 1])
    gap /= len(INNER_LIP_PAIRS)

    # Mouth-corner deviation from the clip median (smile/frown/stretch).
    corners = lm[:, [MOUTH_CORNER_LEFT, MOUTH_CORNER_RIGHT], :2]
    corner_med = np.median(corners, axis=0)
    corner_dev = np.linalg.norm(corners - corner_med, axis=2).mean(axis=1)

    # Brow deviation from clip median.
    brows = lm[:, LEFT_BROW_LOWER + RIGHT_BROW_LOWER, 1]
    brow_med = np.median(brows, axis=0)
    brow_dev = np.abs(brows - brow_med).mean(axis=1)

    score = 2.0 * gap + 1.5 * corner_dev + 1.0 * brow_dev
    if mp_blendshape_sum is not None:
        s = mp_blendshape_sum[det]
        s_scale = max(np.percentile(s, 90), 1e-6)
        score = score + 0.15 * (s / s_scale)
    act[det] = score
    return act


def stabilize_landmarks(landmarks, detected, mp_blendshape_sum=None,
                        neutral_frame=-1, n_neutral_max=60):
    """Two-pass stabilization.

    landmarks: (N, 478, 3) raw pixel-space landmarks from the face NPZ.
    detected:  (N,) bool.
    mp_blendshape_sum: optional (N,) sum of MediaPipe's own blendshape
        values — extra signal for picking neutral frames.
    neutral_frame: manual neutral frame index (-1 = automatic).

    Returns dict:
        stab            (N, 478, 3) canonical-frame landmarks (zeros on
                        undetected frames)
        template        (478, 3) neutral template, canonical frame
        neutral_frames  indices used for the template
        face_height     template forehead->chin distance (canonical units)
    """
    landmarks = np.asarray(landmarks, dtype=np.float64)
    det = np.asarray(detected, dtype=bool)
    n = len(landmarks)
    det_idx = np.flatnonzero(det)
    if len(det_idx) == 0:
        raise ValueError("stabilize_landmarks: no detected frames")

    # ── Pass A: everything into the first detected frame's canonical
    # frame (good enough to measure activity scale-free).
    ref = landmarks[det_idx[0]]
    o0, R0, s0 = canonical_frame_from(ref)
    ref_c = to_canonical(ref, o0, R0, s0)
    rig_ref = ref_c[RIGID_INDICES]

    stab_a = np.zeros_like(landmarks)
    for i in det_idx:
        s, R, t = similarity_fit(landmarks[i][RIGID_INDICES], rig_ref,
                                 z_weight=0.5)
        stab_a[i] = (s * (R @ landmarks[i].T)).T + t

    # ── Neutral frame selection.
    if neutral_frame >= 0:
        lo = max(0, neutral_frame - 2)
        hi = min(n, neutral_frame + 3)
        neutral_frames = [i for i in range(lo, hi) if det[i]]
        if not neutral_frames:
            neutral_frames = [int(det_idx[0])]
    else:
        act = compute_activity(stab_a, det, mp_blendshape_sum)
        order = np.argsort(act)
        n_pick = int(np.clip(len(det_idx) // 10, 5, n_neutral_max))
        n_pick = min(n_pick, len(det_idx))
        neutral_frames = sorted(int(i) for i in order[:n_pick])

    template_a = np.median(stab_a[neutral_frames], axis=0)

    # Re-canonicalize the template itself (its own eye-mid/axes) so the
    # canonical frame is defined by the NEUTRAL pose, not frame 0.
    ot, Rt, st = canonical_frame_from(template_a)
    template = to_canonical(template_a, ot, Rt, st)
    rig_t = template[RIGID_INDICES]

    # ── Pass B: fit every frame straight onto the neutral template.
    stab = np.zeros_like(landmarks)
    for i in det_idx:
        s, R, t = similarity_fit(landmarks[i][RIGID_INDICES], rig_t,
                                 z_weight=0.5)
        stab[i] = (s * (R @ landmarks[i].T)).T + t

    face_height = float(np.linalg.norm(template[FOREHEAD_TOP]
                                       - template[CHIN_CENTER]))

    # Side sanity: subject-left mouth corner must sit at +X in the
    # canonical frame. If this ever fires, the side convention upstream
    # changed — warn loudly rather than silently swap L/R channels.
    if template[MOUTH_CORNER_LEFT, 0] <= template[MOUTH_CORNER_RIGHT, 0]:
        print("  WARNING: canonical-frame side check failed "
              "(left mouth corner not at +X). Left/Right face channels "
              "may be swapped — please report this clip.")

    return {
        "stab": stab,
        "template": template,
        "neutral_frames": neutral_frames,
        "face_height": face_height,
    }
