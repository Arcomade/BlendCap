# face_export.py
# ==============
# Build face bone animation data from a calibration JSON exported by
# extract_face_calibration.py. The calibration carries the source rig's
# actual bone topology AND per-ARKit-shape pose deltas (translation +
# rotation), so the resulting BVH is a recording of the source rig's
# face poses, replayed under the input MediaPipe blendshape coefficients.
#
# No hardcoded face template — the bone set is whatever the calibration
# contains. No coordinate assumptions about the source rig's bone
# orientations: deltas are stored in parent-local frame, which is
# invariant under the global Z-up -> Y-up axis swap to BVH conventions.

import json
import math

import numpy as np


AVERAGE_IPD_METERS = 0.063


# Axis swap: source armature is Blender Z-up; BVH (and the rest of this
# pipeline) is Y-up. Maps (X right, Y forward-back, Z up) into
# (X right, Y up, Z forward).
#   BVH.x =  source.x
#   BVH.y =  source.z
#   BVH.z = -source.y     (source -Y forward -> BVH +Z forward)
R_AXIS = np.array([
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],
    [0.0, -1.0, 0.0],
], dtype=np.float64)


def face_ipd_for_head_width(target_head_width, face_calibration):
    """Convert a desired head width (TEMPLE-to-TEMPLE, in meters) into
    the face_ipd_target value that will produce that head width when
    the source rig's skeleton is uniformly scaled. Uses the SOURCE
    rig's own temple-to-IPD ratio so the final BVH temple span matches
    target_head_width exactly regardless of how the source rig is
    proportioned — and the source rig's per-bone positions are
    preserved untouched (retarget maps still work as authored).

    Fallback chain when LeftTemple/RightTemple are missing:
        Temple -> Ear -> max X extent of the skeleton.

    face_calibration: dict from load_face_calibration() — needs both
        the `skeleton` block (for per-bone X positions) and `_meta`
        (for source_ipd).
    """
    skeleton = face_calibration.get("skeleton") or []
    meta = face_calibration.get("_meta") or {}
    source_ipd = float(meta.get("source_ipd", 0.0))
    if not skeleton or source_ipd <= 1e-6:
        return float(target_head_width)

    by_name = {entry["name"]: entry for entry in skeleton}

    def _pair_width(left, right):
        if left not in by_name or right not in by_name:
            return 0.0
        return abs(by_name[left]["head"][0] - by_name[right]["head"][0])

    source_head_width = _pair_width("LeftTemple", "RightTemple")
    label = "temple-to-temple"
    if source_head_width <= 1e-6:
        source_head_width = _pair_width("LeftEar", "RightEar")
        label = "ear-to-ear (Temple bones absent)"
    if source_head_width <= 1e-6:
        xs = []
        for entry in skeleton:
            xs.append(entry["head"][0])
            xs.append(entry["tail"][0])
        source_head_width = float(max(xs) - min(xs)) if xs else 0.0
        label = "max-X extent (Temple/Ear bones absent)"
    if source_head_width <= 1e-6:
        return float(target_head_width)

    print(f"       Source head width ({label}): "
          f"{source_head_width*100:.2f} cm at source IPD "
          f"{source_ipd*100:.2f} cm")
    return float(target_head_width) * source_ipd / source_head_width


# Visual chain links: bone name -> name of the bone its tail should point at.
# Used purely for the BVH skeleton's appearance — Blender draws each leaf
# bone from its head to its End Site offset, so by aiming each chain link
# at its neighbor we get a connected face wireframe (lip loop, brow arcs,
# nose bridge, jaw corners) instead of a constellation of disconnected
# octahedrons. Animation is unaffected — all bones still parent to Head
# with world-frame deltas.
#
# Bones NOT in this dict (or whose chain target is missing from the rig)
# fall back to a short capped stub in the source-rig tail direction.
CHAIN_LINKS = {
    # Lip bones intentionally left out — they display better as singleton
    # stubs (small oriented dots) than as a connected mouth loop.

    # Brow ridge (visible edge) — outer -> inner per side.
    "LeftBrow":          "LeftBrow1",
    "LeftBrow1":         "LeftBrow2",
    "LeftBrow2":         "LeftBrow3",
    "LeftBrow3":         "LeftBrowInner",
    "RightBrow":         "RightBrow1",
    "RightBrow1":        "RightBrow2",
    "RightBrow2":        "RightBrow3",
    "RightBrow3":        "RightBrowInner",

    # Brow muscle (under-skin arc).
    "LeftBrowMuscle":    "LeftBrowMuscle1",
    "LeftBrowMuscle1":   "LeftBrowMuscle2",
    "LeftBrowMuscle2":   "LeftBrowMuscle3",
    "RightBrowMuscle":   "RightBrowMuscle1",
    "RightBrowMuscle1":  "RightBrowMuscle2",
    "RightBrowMuscle2":  "RightBrowMuscle3",

    # Nose bridge (top of bridge -> tip area -> underside chain).
    "Nose":              "Nose1",
    "Nose1":             "Nose2",
    "Nose2":             "Nose3",
    "Nose3":             "Nose4",

    # Nostrils (side -> nostril).
    "LeftNostril":       "LeftNostril1",
    "RightNostril":      "RightNostril1",

    # Jaw corners (corner -> mid -> chin pad).
    "LeftJawCorner":     "LeftJawCorner1",
    "LeftJawCorner1":    "LeftChinPad",
    "RightJawCorner":    "RightJawCorner1",
    "RightJawCorner1":   "RightChinPad",

    # Jaw center descending into chin.
    "Jaw":               "Chin",
    "Chin":              "Chin1",

    # Forehead (optional — only if the rig has these bones).
    "LeftForehead":      "LeftForehead1",
    "LeftForehead1":     "LeftForehead2",
    "RightForehead":     "RightForehead1",
    "RightForehead1":    "RightForehead2",

    # Ears (optional — only if the rig has these bones).
    "LeftEar":           "LeftEar1",
    "LeftEar1":          "LeftEar2",
    "LeftEar2":          "LeftEar3",
    "LeftEar3":          "LeftEar4",
    "RightEar":          "RightEar1",
    "RightEar1":         "RightEar2",
    "RightEar2":         "RightEar3",
    "RightEar3":         "RightEar4",

    # Tongue (only Tongue is in the user's 63, but link if .001/.002 exist).
    "Tongue":            "Tongue1",
    "Tongue1":           "Tongue2",
}




# =====================================================================
#  CALIBRATION LOADING
# =====================================================================

def load_face_calibration(json_path):
    """Load the calibration JSON. Returns the full dict (with `skeleton`,
    `calibration`, `_meta` keys); pass straight to compute_face_bone_data."""
    with open(json_path) as f:
        data = json.load(f)
    if "skeleton" not in data or "calibration" not in data:
        raise RuntimeError(
            f"Calibration JSON missing required keys; got {list(data.keys())}. "
            "Re-run extract_face_calibration.py against the source rig.")
    n_bones = len(data["skeleton"])
    n_shapes = len(data["calibration"])
    print(f"  Loaded face calibration: {n_bones} bones, {n_shapes} shapes")
    return data


# =====================================================================
#  MEDIAPIPE FACE NPZ LOADING
# =====================================================================

def first_detected(detected):
    """Return index of first True in detected array."""
    indices = np.flatnonzero(detected)
    return int(indices[0]) if len(indices) > 0 else 0


def load_face_data(face_npz_path):
    """Load face NPZ produced by save_face_data.py and return a dict."""
    print(f"       Loading face data: {face_npz_path}")
    face = np.load(str(face_npz_path), allow_pickle=True)

    data = {
        "landmarks": face["face_landmarks"].astype(np.float64),
        "blendshapes": face["face_blendshapes"].astype(np.float64),
        "eye_gaze": face["eye_gaze"].astype(np.float64),
        "detected": face["face_detected"].astype(bool),
        "frame_indices": face["frame_indices"].astype(int),
        "fps": float(face["fps"]),
    }
    data["bs_name_to_idx"] = {name: i for i, name in enumerate(face["blendshape_names"])}
    # Ordered name list kept alongside the index dict — the refine
    # stage may APPEND channels (tongueOut), and writers need the
    # updated ordering to stay in sync with the widened array.
    data["blendshape_names"] = [str(n) for n in face["blendshape_names"]]
    data["face_transform"] = (face["face_transform"].astype(np.float64)
                               if "face_transform" in face else None)
    # Raw mouth pixel stats (NPZ v2+, save_face_data.extract_mouth_stats)
    # — consumed by pipeline/face_refine.py. Absent in older captures.
    data["mouth_stats"] = (face["mouth_stats"].astype(np.float64)
                           if "mouth_stats" in face else None)

    n = len(data["landmarks"])
    print(f"       {n} frames, {data['detected'].sum()} with face detected, "
          f"{data['fps']:.1f} FPS")
    return data


def compute_head_local_landmarks(face_data, eye_anchors=None):
    """
    Transform face landmarks from pixel space to head-local meters for BVH.

    eye_anchors: optional (sam_eye_L, sam_eye_R) in BVH Head-local space.
        When provided ("both" mode), anchors face directly to SAM eyes.
        When None ("face-only" mode), uses face_transform fallback.
    """
    landmarks = face_data["landmarks"]   # (N, 478, 3)
    detected = face_data["detected"]
    n_frames = len(landmarks)
    first_det = first_detected(detected)

    head_local = np.zeros((n_frames, 478, 3), dtype=np.float64)
    mask = detected.astype(bool)
    if not mask.any():
        return head_local, None

    lm_det = landmarks[mask]                          # (D, 478, 3)

    if eye_anchors is not None:
        # ---- Anchor approach: map MP eyes -> SAM eyes directly ----
        sam_eye_L, sam_eye_R = eye_anchors
        sam_mid = (sam_eye_L + sam_eye_R) / 2
        sam_ipd = np.linalg.norm(sam_eye_L - sam_eye_R)
        sam_right = sam_eye_R - sam_eye_L
        sam_angle = np.arctan2(sam_right[1], sam_right[0])

        # Eye contour indices (stable, not iris which tracks gaze)
        L_EYE = [33, 7, 163, 144, 145, 153, 154, 155,
                 133, 173, 157, 158, 159, 160, 161, 246]
        R_EYE = [362, 382, 381, 380, 374, 373, 390, 249,
                 263, 466, 388, 387, 386, 385, 384, 398]

        mp_eye_L = lm_det[:, L_EYE, :].mean(axis=1)  # (D, 3)
        mp_eye_R = lm_det[:, R_EYE, :].mean(axis=1)  # (D, 3)
        mp_mid = (mp_eye_L + mp_eye_R) / 2            # (D, 3)

        mp_ipd = np.sqrt(np.sum((mp_eye_L[:, :2] - mp_eye_R[:, :2])**2,
                                axis=1))
        mp_ipd = np.maximum(mp_ipd, 1.0)
        mpp = sam_ipd / mp_ipd                        # (D,) meters per pixel

        print(f"       SAM IPD: {sam_ipd * 100:.1f}cm, "
              f"MP IPD: {mp_ipd[0]:.1f}px, scale: {mpp[0]:.6f} m/px")

        centered = lm_det - mp_mid[:, None, :]
        scaled = centered * mpp[:, None, None]

        # Pixel axes: +X = screen right = char LEFT, +Y = down, +Z = depth
        # BVH Head-local: +X = char RIGHT, +Y = up, +Z = forward
        hl = np.empty_like(scaled)
        hl[:, :, 0] = -scaled[:, :, 0]
        hl[:, :, 1] = scaled[:, :, 1]
        hl[:, :, 2] = -scaled[:, :, 2]

        # Per-frame roll correction
        mp_eye_L_hl = hl[:, L_EYE, :].mean(axis=1)
        mp_eye_R_hl = hl[:, R_EYE, :].mean(axis=1)
        mp_right_hl = mp_eye_R_hl - mp_eye_L_hl
        mp_angles = np.arctan2(mp_right_hl[:, 1], mp_right_hl[:, 0])
        thetas = sam_angle - mp_angles
        cos_t = np.cos(thetas)[:, None]
        sin_t = np.sin(thetas)[:, None]

        rotated = hl.copy()
        rotated[:, :, 0] = cos_t * hl[:, :, 0] - sin_t * hl[:, :, 1]
        rotated[:, :, 1] = sin_t * hl[:, :, 0] + cos_t * hl[:, :, 1]
        rotated += sam_mid
        head_local[mask] = rotated

        print(f"       Eye mid: ({sam_mid[0]:.4f}, {sam_mid[1]:.4f}, "
              f"{sam_mid[2]:.4f})")
        print(f"       Roll correction: {np.degrees(thetas[0]):+.1f} deg")

        return head_local, None

    # ---- Fallback: face_transform (face-only mode) ----
    ref_lm = landmarks[first_det]
    ipd_px = np.linalg.norm(ref_lm[468, :2] - ref_lm[473, :2])
    if ipd_px < 3.0:
        ipd_px = np.linalg.norm(ref_lm[234, :2] - ref_lm[454, :2]) * 0.4
    meters_per_pixel = AVERAGE_IPD_METERS / max(ipd_px, 1.0)
    print(f"       IPD: {ipd_px:.1f}px, scale: {meters_per_pixel:.6f} m/px")

    head_joint_frac = 0.20
    cx = lm_det[:, 168, 0:1]
    face_h = lm_det[:, 152, 1:2] - lm_det[:, 10, 1:2]
    cy = lm_det[:, 152, 1:2] - head_joint_frac * face_h

    hl = np.empty_like(lm_det)
    hl[:, :, 0] = (lm_det[:, :, 0] - cx) * meters_per_pixel
    hl[:, :, 1] = -(lm_det[:, :, 1] - cy) * meters_per_pixel
    hl[:, :, 2] = -lm_det[:, :, 2] * meters_per_pixel + 0.05

    face_transform = face_data.get("face_transform")
    if face_transform is not None:
        ft_det = face_transform[mask]
        for fi in range(len(ft_det)):
            rot_inv = ft_det[fi, :3, :3].T
            hl[fi] = (rot_inv @ hl[fi].T).T
        print(f"       Face rotation corrected via face_transform")

    head_local[mask] = hl

    fd = head_local[first_det]
    print(f"       Head-local (meters): "
          f"X [{fd[:,0].min():.4f}, {fd[:,0].max():.4f}]  "
          f"Y [{fd[:,1].min():.4f}, {fd[:,1].max():.4f}]  "
          f"Z [{fd[:,2].min():.4f}, {fd[:,2].max():.4f}]")
    return head_local, None


# =====================================================================
#  ROTATION MATH HELPERS
# =====================================================================

def _axis_angle_to_matrix(aa):
    """Rotation vector (axis * angle) -> 3x3 rotation matrix via Rodrigues."""
    angle = float(np.linalg.norm(aa))
    if angle < 1e-9:
        return np.eye(3)
    axis = aa / angle
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]], dtype=np.float64)
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _matrix_to_euler_zyx_deg(R):
    """3x3 rotation matrix -> (Z, Y, X) Euler angles in degrees.
    Matches the BVH channel order 'Zrotation Yrotation Xrotation'."""
    sy = -R[2, 0]
    if abs(sy) < 0.99999:
        pitch_y = math.asin(sy)
        yaw_z = math.atan2(R[1, 0], R[0, 0])
        roll_x = math.atan2(R[2, 1], R[2, 2])
    else:
        pitch_y = math.copysign(math.pi / 2.0, sy)
        yaw_z = math.atan2(-R[0, 1], R[1, 1])
        roll_x = 0.0
    return (math.degrees(yaw_z), math.degrees(pitch_y), math.degrees(roll_x))


# =====================================================================
#  FACE BONE DATA
# =====================================================================

def compute_face_bone_data(face_data, head_local_landmarks, rest_face=None,
                           face_boost=2.0, face_calibration=None,
                           face_ipd_target=None):
    """Build the face bone data dict consumed by bvh_writer.write_bvh.

    face_calibration: dict from load_face_calibration(). Required —
        without it, the bone topology is unknown.
    face_boost: scalar multiplier on all blendshape-driven deltas.
    face_ipd_target: target face IPD in meters. Drives the absolute size
        of the face cluster — body+face mode passes a body-derived value
        (so face proportions track the body's head size) and face-only
        mode passes AVERAGE_IPD_METERS (so absolute size is constant
        across face-only captures). If None, falls back to the tracked
        face's IPD (legacy behavior, sensitive to landmark noise).
    """
    if face_calibration is None or "skeleton" not in face_calibration:
        raise RuntimeError(
            "compute_face_bone_data requires face_calibration with skeleton "
            "topology. Re-run extract_face_calibration.py on the source rig.")

    skeleton = face_calibration["skeleton"]
    cal = face_calibration["calibration"]
    meta = face_calibration.get("_meta", {})
    source_ipd = float(meta.get("source_ipd", 0.0))

    blendshapes = face_data["blendshapes"]   # (N, 52)
    detected = face_data["detected"]
    bs_idx = face_data["bs_name_to_idx"]
    n_frames = len(blendshapes)
    mask = detected.astype(bool)

    first_det = first_detected(detected)
    rest_lm = rest_face if rest_face is not None else head_local_landmarks[first_det]

    # ---- Eye anchor in BVH head-local frame ----
    our_eye_L = rest_lm[468]
    our_eye_R = rest_lm[473]
    our_mid = (our_eye_L + our_eye_R) * 0.5
    our_ipd = float(np.linalg.norm(our_eye_L - our_eye_R))

    # ---- Resolve final face IPD ----
    # face_ipd_target wins when supplied — body+face passes a value tied
    # to the body's head segment so face size scales with the body, and
    # face-only passes AVERAGE_IPD_METERS so absolute size is stable
    # across separate face captures. The tracked-IPD fallback exists for
    # the no-target legacy path; it varies with landmark noise and is
    # the source of the "sometimes really small / sometimes really big"
    # head sizing the new pipeline used to produce.
    if face_ipd_target is not None and face_ipd_target > 1e-6:
        target_ipd = float(face_ipd_target)
        ipd_source = "body-anchored" if face_ipd_target != AVERAGE_IPD_METERS \
            else "anatomical"
    elif our_ipd > 1e-6:
        target_ipd = our_ipd
        ipd_source = "tracked"
    else:
        target_ipd = AVERAGE_IPD_METERS
        ipd_source = "fallback"
    print(f"       Face target IPD: {target_ipd*100:.2f} cm ({ipd_source}); "
          f"source IPD: {source_ipd*100:.2f} cm; "
          f"tracked IPD: {our_ipd*100:.2f} cm")

    # ---- Convert source skeleton (Z-up armature) -> BVH (Y-up) head-local ----
    # Inter-bone geometry comes 1:1 from the source rig's calibration —
    # retarget maps depend on per-bone rest positions matching the
    # source rig, so the only thing this function adjusts is the
    # OVERALL face scale and where the cluster sits in head-local
    # space. Per-bone positions, ratios, and relative offsets are
    # preserved untouched.
    if source_ipd > 1e-6:
        scale = target_ipd / source_ipd
    else:
        scale = 1.0
        print(f"       WARNING: source IPD missing; using scale=1.0")

    # Anchor source bones into BVH head-local space. Three independent
    # axes so we can place each one where it makes sense for the
    # downstream rig — uniform-shift would force unwanted compromises
    # on source rigs whose internal bone positions don't match
    # anatomical norms (e.g. rigs that place Temple ~15 cm behind
    # LeftEye in source space, which would dump the temple
    # cluster way past the back of the body's head if we Z-anchored
    # at the eyes too):
    #   X — pin source eye-mid X to our_mid X. With a symmetric source
    #       rig this is just an X=0 anchor.
    #   Y — pin source eye-mid Y to our_mid Y, so the rendered eye
    #       height matches the tracked head's eye height (body+face
    #       lines up against SAM's eye anchors; face-only lines up
    #       against the MP iris midpoint).
    #   Z — pin source Jaw HEAD's Z to the body's Head-joint Z (0 in
    #       head-local), so the face's Jaw bone sits on the same Z
    #       column as the body's Head bone visual. With Blender drawing
    #       each bone's "fat part" ~1/8 from its head, putting Jaw's
    #       head at Head joint's Z brings Jaw's and Head's fat parts
    #       into the same column — which is the alignment the user
    #       wants. Falls back to eye-Z anchor if Jaw is missing from
    #       the calibration.
    src_eye_L = src_eye_R = None
    src_jaw = None
    for entry in skeleton:
        if entry["name"] == "LeftEye":
            src_eye_L = np.array(entry["head"], dtype=np.float64)
        elif entry["name"] == "RightEye":
            src_eye_R = np.array(entry["head"], dtype=np.float64)
        elif entry["name"] == "Jaw":
            src_jaw = np.array(entry["head"], dtype=np.float64)
    if src_eye_L is None or src_eye_R is None:
        raise RuntimeError(
            "Source skeleton missing LeftEye / RightEye — cannot anchor face "
            "to the tracked head.")

    def _axis_swap(v):
        return R_AXIS @ np.asarray(v, dtype=np.float64)
    src_eye_mid_bvh = _axis_swap((src_eye_L + src_eye_R) * 0.5) * scale
    if src_jaw is not None:
        src_jaw_bvh = _axis_swap(src_jaw) * scale
        z_anchor_source = src_jaw_bvh[2]
        z_anchor_target = 0.0
        z_anchor_label = "Jaw head Z -> 0 (body Head joint)"
    else:
        z_anchor_source = src_eye_mid_bvh[2]
        z_anchor_target = our_mid[2]
        z_anchor_label = "eye-mid Z -> our_mid Z (no Jaw bone)"
    shift = np.array([
        our_mid[0] - src_eye_mid_bvh[0],
        our_mid[1] - src_eye_mid_bvh[1],
        z_anchor_target - z_anchor_source,
    ], dtype=np.float64)
    print(f"       Anchor: eye-mid X/Y -> our_mid X/Y; {z_anchor_label}")

    rest_head = {}    # name -> position in BVH head-local (after scale + shift)
    tail_pos = {}
    parent_of = {}
    children = {"Head": []}

    # Flat structure: every face bone is a direct child of Head, in the
    # order the calibration JSON lists them. Source-rig parent chains
    # are intentionally NOT reconstructed — each bone's animation is
    # world-frame, captured against rest, applied independently.
    for entry in skeleton:
        name = entry["name"]
        head_bvh = _axis_swap(entry["head"]) * scale + shift
        tail_bvh = _axis_swap(entry["tail"]) * scale + shift
        rest_head[name] = head_bvh
        tail_pos[name] = tail_bvh
        parent_of[name] = "Head"
        children["Head"].append(name)

    print(f"       Face scale {scale:.3f}x (target IPD {target_ipd*100:.2f} cm / "
          f"source IPD {source_ipd*100:.2f} cm); rest positions from source rig")

    bone_order = list(children["Head"])

    # ---- Parent-relative offsets (BVH OFFSET values) ----
    # All bones have parent = Head, so each offset is just the bone's
    # head position in head-local BVH space.
    offsets = {name: rest_head[name].copy() for name in bone_order}

    # ---- End sites for leaf bones ----
    # Every face bone is a leaf under Head (flat hierarchy), so each one
    # needs its own end-site to define a tail direction. Two policies:
    #   1. If the bone is in CHAIN_LINKS and its target exists in the
    #      rig, point the tail at the next bone's head — connects the
    #      chain visually (lip loop, brow arcs, nose bridge, etc.).
    #      Not capped — chain neighbors are naturally close.
    #   2. Otherwise (singletons / chain ends / unknown bones), use the
    #      source rig's tail direction, capped at 1.5 cm so the stubs
    #      stay in the same size range as typical chain segments
    #      (brow / lip / nose links are 1–1.3 cm).
    MAX_STUB_LEN = 0.015
    end_sites = {}
    for name in bone_order:
        next_name = CHAIN_LINKS.get(name)
        if next_name and next_name in rest_head:
            chain_vec = rest_head[next_name] - rest_head[name]
            if float(np.linalg.norm(chain_vec)) > 1e-6:
                # Valid chain link — tail at the neighbor's head.
                end_sites[name] = chain_vec
                continue
            # Chain neighbor sits at the same head position (rigify
            # control rigs sometimes collapse short chains like
            # tongue / tongue.001 to one point). Fall through to the
            # singleton stub so we don't emit a zero-length End Site,
            # which Blender renders as a default-sized fallback bone.
        # Singleton stub in the source rig's tail direction, capped.
        vec = tail_pos[name] - rest_head[name]
        mag = float(np.linalg.norm(vec))
        if mag > MAX_STUB_LEN and mag > 1e-9:
            vec = vec * (MAX_STUB_LEN / mag)
        elif mag < 1e-9:
            vec = np.array([0.0, MAX_STUB_LEN, 0.0])
        end_sites[name] = vec

    # ---- Per-shape per-bone deltas. Both translation and rotation are
    #      stored in the source armature's WORLD frame (Z-up). Apply the
    #      Z-up -> Y-up axis swap to map them into BVH world (= head-local,
    #      since every face bone is parented to Head). Translation also
    #      gets the IPD scale factor; rotation is scale-invariant. ----
    bone_t_drivers = {}   # name -> [(shape, np.array([dt_x, dt_y, dt_z]))]
    bone_r_drivers = {}   # name -> [(shape, np.array([rv_x, rv_y, rv_z]))]
    n_t = n_r = 0

    for shape_name, bone_data in cal.items():
        if shape_name.startswith("_"):
            continue
        if shape_name not in bs_idx:
            continue
        for bone_name, entry in bone_data.items():
            if bone_name not in offsets:
                continue
            if "t" in entry:
                t_src = np.array(entry["t"], dtype=np.float64)
                t_bvh = (R_AXIS @ t_src) * scale
                bone_t_drivers.setdefault(bone_name, []).append((shape_name, t_bvh))
                n_t += 1
            if "r" in entry:
                # `r` is axis * angle in source world. R_AXIS rotates the
                # axis into BVH world; the magnitude (rotation angle) is
                # preserved because R_AXIS is orthogonal.
                r_src = np.array(entry["r"], dtype=np.float64)
                r_bvh = R_AXIS @ r_src
                bone_r_drivers.setdefault(bone_name, []).append((shape_name, r_bvh))
                n_r += 1

    # ---- Per-frame animation ----
    frames_dict = {}
    for name in bone_order:
        fr = np.zeros((n_frames, 6), dtype=np.float64)

        # Translation: rest offset + sum(weight * delta_t)
        total_t = np.zeros((n_frames, 3), dtype=np.float64)
        for shape_name, dt in bone_t_drivers.get(name, []):
            idx = bs_idx.get(shape_name, -1)
            if idx < 0:
                continue
            vals = np.where(mask, blendshapes[:, idx], 0.0)
            total_t += vals[:, None] * dt[None, :] * face_boost

        # Rotation: sum weighted axis-angle vectors. For the small angles
        # typical of face motion (<30 deg) the linear tangent-space sum
        # is a good approximation of true quaternion composition.
        total_aa = np.zeros((n_frames, 3), dtype=np.float64)
        for shape_name, rv in bone_r_drivers.get(name, []):
            idx = bs_idx.get(shape_name, -1)
            if idx < 0:
                continue
            vals = np.where(mask, blendshapes[:, idx], 0.0)
            total_aa += vals[:, None] * rv[None, :] * face_boost

        # Convert per-frame totals to BVH channel values.
        fr[:, :3] = offsets[name] + total_t

        for f in range(n_frames):
            aa = total_aa[f]
            if not mask[f] or np.linalg.norm(aa) < 1e-9:
                fr[f, 3:6] = 0.0
            else:
                R = _axis_angle_to_matrix(aa)
                z, y, x = _matrix_to_euler_zyx_deg(R)
                fr[f, 3] = z
                fr[f, 4] = y
                fr[f, 5] = x

        frames_dict[name] = fr

    n_driven = len(set(bone_t_drivers) | set(bone_r_drivers))
    print(f"       {len(skeleton)} face bones ({n_driven} driven, "
          f"{n_t} trans + {n_r} rot drivers, boost={face_boost:.1f}x)")

    all_rest = np.array([rest_head[n] for n in bone_order])
    if len(all_rest):
        print(f"       Face extent (meters): "
              f"W={all_rest[:,0].max() - all_rest[:,0].min():.3f}  "
              f"H={all_rest[:,1].max() - all_rest[:,1].min():.3f}  "
              f"D={all_rest[:,2].max() - all_rest[:,2].min():.3f}")

    return {
        "offsets": offsets,
        "rest_positions": offsets,
        "frames": frames_dict,
        "types": {n: "6ch" for n in bone_order},
        "order": bone_order,
        "children": children,
        "end_sites": end_sites,
    }
