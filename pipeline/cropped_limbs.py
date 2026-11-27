# pipeline/cropped_limbs.py
# =========================
# Hold Cropped Limbs (--hold-cropped-limbs): body parts that are outside
# the picture are held still instead of animating the model's guess.
#
# The body model always returns a full skeleton. When the legs, an arm
# or the head are cropped by the frame edge it still invents a pose for
# them, and that invented pose moves. For waist-up footage (small rooms,
# social clips) that means legs that walk on their own, and foot
# locking / grounding then pin and level the body on feet nobody filmed.
#
# Detection is geometric and per frame: the model also writes every
# joint's 2D position in the original image (keypoints_2d), so a joint
# whose 2D point lies outside the frame (with a small margin) was not
# seen. There is no confidence output, so OCCLUSION is out of scope: a
# hand behind the back still projects inside the picture and keeps the
# model's guess, exactly as before.
#
# Policy per body part, applied to the part's whole subtree in PARENT-
# RELATIVE space (so held legs still turn with the hips when the body
# spins, and fingers follow a held hand):
#   legs            -> standing pose (the model's T-pose prerotations:
#                      straight legs), blended in from the last seen pose
#   arms, hands,    -> held at the pose they had when they left the
#   head               picture; blended back when they return
# A part has to be out for HOLD_DELAY_SEC before it is held (a hand
# swinging past the edge for a few frames keeps the model's guess, which
# is usually fine), and visible blips shorter than RELEASE_FRAMES inside
# a cropped stretch do not release it. A part that was never seen in the
# whole clip is held in its standing pose (legs) or left to the model
# (arms / head: there is nothing to hold it at).
#
# Feet that are held are reported as "no evidence" to the foot passes
# (foot_grounding.detect_stances_per_foot's valid_feet), so no plant is
# detected on them: no pinning, no floor from guessed feet. When the
# feet are never seen at all, modes.py also flattens the vertical body
# channel (hips at a fixed standing height).
#
# THE HIPS are the root everything hangs from, so they get their own
# rule (steady_hips, run by modes.py after the camera levelling). When
# the pelvis is out of the picture the model guesses its tilt and its
# position from a torso fragment, frame by frame, and that guess rocks
# the whole character (held legs included). So while the hips are
# cropped:
#   position        -> held (no travel, no bobbing, no depth pumping);
#                      modes.py holds the translation and vertical series
#   tilt            -> held upright
#   facing          -> NOT held: it follows the chest's heading, which
#                      comes from the filmed shoulders, low-passed so
#                      slow turns reach the pelvis and quick twists stay
#                      in the spine
# and the spine chain is re-spread between the steady hips and the
# TRACKED chest, so the chest, shoulders, arms and head keep exactly the
# world orientation the model saw: a lean bends the spine over still
# hips instead of tipping the body from the pelvis like a plank.
#
# Everything here is off unless the checkbox is on; with it off the
# pipeline is byte-identical to before.

import numpy as np

from .bvh_math import (
    _rotmats_to_quats_xyzw, _slerp_quats_xyzw, quats_xyzw_to_rotmats,
    compute_tpose_globals, compute_world_positions,
)

MARGIN_FRAC = 0.02       # of the frame's width / height
HOLD_DELAY_SEC = 0.5     # out this long before the part is held
RELEASE_FRAMES = 3       # visible blips shorter than this don't release
BLEND_FRAMES = 8         # slerp ramp at both ends of a held stretch
HEADING_SMOOTH_SEC = 0.35  # low-pass on the facing the held hips follow

# 2D keypoint ids (MHR70 layout, models/.../metadata/mhr70.py).
KP_NOSE, KP_LEYE, KP_REYE = 0, 1, 2
KP_LELBOW, KP_RELBOW = 7, 8
KP_LHIP, KP_RHIP = 9, 10
KP_LKNEE, KP_RKNEE = 11, 12
KP_LANKLE, KP_RANKLE = 13, 14
KP_RWRIST, KP_LWRIST = 41, 62

# (name, keypoints, how many must be inside to count as seen,
#  subtree root in SAM joint ids (body_skeleton.JOINT_NAMES), policy,
#  what to do when the part is NEVER seen in the clip: "keep" the
#  model's pose (an arm has no neutral worth forcing) or "rest" (the
#  T-pose local: a head straight on the neck))
# Order matters: a larger part (arm) before the part it contains (hand),
# so the hand's own hold is applied on top of an already-held arm.
LIMBS = (
    ("left leg",   (KP_LKNEE, KP_LANKLE), "all", 2,   "stand", "rest"),  # LeftUpLeg
    ("right leg",  (KP_RKNEE, KP_RANKLE), "all", 18,  "stand", "rest"),  # RightUpLeg
    ("left arm",   (KP_LELBOW,),          "all", 75,  "last", "keep"),   # LeftArm
    ("right arm",  (KP_RELBOW,),          "all", 39,  "last", "keep"),   # RightArm
    ("left hand",  (KP_LWRIST,),          "all", 78,  "last", "keep"),   # LeftHand
    ("right hand", (KP_RWRIST,),          "all", 42,  "last", "keep"),   # RightHand
    ("head",       (KP_NOSE, KP_LEYE, KP_REYE), "any", 113, "last", "rest"),  # Head
)
_LEFT_LEG, _RIGHT_LEG = "left leg", "right leg"

# Hips / spine joint ids (SAM), for steady_hips.
J_HIPS, J_NECK = 1, 110
J_SPINE_CHAIN = (34, 35, 36, 37)      # Spine .. Spine3; the last is the chest


def keypoints_inside(kp2d, frame_size, margin_frac=MARGIN_FRAC):
    """(n_frames, n_kp) bool: the 2D point lies inside the picture, with a
    margin of `margin_frac` of the frame size on every edge."""
    kp2d = np.asarray(kp2d, dtype=np.float64)
    w, h = float(frame_size[0]), float(frame_size[1])
    mx, my = margin_frac * w, margin_frac * h
    x, y = kp2d[..., 0], kp2d[..., 1]
    inside = (x >= -mx) & (x <= w + mx) & (y >= -my) & (y <= h + my)
    inside &= np.isfinite(x) & np.isfinite(y)
    return inside


def _limb_out(inside, kps, rule):
    """(n_frames,) bool: the part counts as cropped this frame."""
    cols = inside[:, list(kps)]
    seen = cols.all(axis=1) if rule == "all" else cols.any(axis=1)
    return ~seen


def held_runs(out, delay_frames, release_frames=RELEASE_FRAMES):
    """Stretches of a cropped part that get held: runs of `out` at least
    `delay_frames` long, after filling visible blips shorter than
    `release_frames`. Returns [(start, end)] inclusive."""
    out = np.asarray(out, dtype=bool).copy()
    n = len(out)
    if n == 0:
        return []
    # fill short visible gaps between out-frames
    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            if 0 < i and j < n and (j - i) < release_frames:
                out[i:j] = True
            i = j
        else:
            i += 1
    runs = []
    i = 0
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if j - i >= delay_frames:
                runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def _subtree(parents, root):
    """Joint ids of `root` and all its descendants, parents before
    children (SAM ids are topologically ordered, parents[j] < j)."""
    n = len(parents)
    members = [root]
    inset = {root}
    for j in range(n):
        p = int(parents[j])
        if j != root and p >= 0 and p in inset:
            members.append(j)
            inset.add(j)
    return members


def _run_weights(s, e, n, blend=BLEND_FRAMES):
    """Hold weight per frame of a run: 1 inside, ramped over `blend`
    frames at an edge that borders visible frames (no ramp at the
    clip's first / last frame)."""
    L = e - s + 1
    w = np.ones(L, dtype=np.float64)
    b = min(blend, max(1, L // 2))
    if s > 0:
        w[:b] = np.minimum(w[:b], np.linspace(1.0 / b, 1.0, b))
    if e < n - 1:
        w[L - b:] = np.minimum(w[L - b:], np.linspace(1.0, 1.0 / b, b))
    return w


def hold_cropped_limbs(global_rots, kp2d, frame_size, parents,
                       prerot_quats, fps, delay_sec=HOLD_DELAY_SEC):
    """Apply the hold. Returns (global_rots_out, report) where report is
    {'held': {limb name: (n_frames,) bool}, 'valid_feet': (n_frames, 2)
    bool (L, R) for the foot passes, 'feet_never_seen': bool,
    'lines': [console lines]}. global_rots is not modified in place."""
    G = np.array(global_rots, dtype=np.float64, copy=True)
    n = G.shape[0]
    parents = np.asarray(parents)
    inside = keypoints_inside(kp2d, frame_size)
    delay = max(1, int(round(delay_sec * max(float(fps), 1e-6))))
    prerot_mats = (quats_xyzw_to_rotmats(prerot_quats)
                   if prerot_quats is not None else None)
    held_masks = {}
    lines = []

    for name, kps, rule, root, policy, never in LIMBS:
        held = np.zeros(n, dtype=bool)
        held_masks[name] = held
        if root >= len(parents) or int(parents[root]) < 0:
            continue
        out = _limb_out(inside, kps, rule)
        runs = held_runs(out, delay)
        if not runs:
            continue
        if policy == "stand" and prerot_mats is None:
            policy = "last"
        members = _subtree(parents, root)
        # parent-relative locals of the TRACKED data (fixed reference;
        # recomposition below walks parents first, so a held parent's
        # new global is what the child is attached to)
        P = {}
        for j in members:
            p = int(parents[j])
            P[j] = np.einsum('fji,fjk->fik', G[:, p], G[:, j])  # G_p^T G_j
        for (s, e) in runs:
            run_policy = policy
            if policy == "last":
                ref = s - 1 if s > 0 else (e + 1 if e + 1 < n else None)
                if ref is None and never == "rest" and prerot_mats is not None:
                    run_policy = "rest"      # never seen: neutral on its parent
                elif ref is None:
                    lines.append(f"       Hold Cropped Limbs: {name} never "
                                 "in the picture; kept the model's pose")
                    continue
            if run_policy == "last":
                H = {j: P[j][ref] for j in members}
            else:
                H = {j: prerot_mats[j] for j in members}
            w = _run_weights(s, e, n)
            held[s:e + 1] = True
            frames = np.arange(s, e + 1)
            for j in members:
                p = int(parents[j])
                qa = _rotmats_to_quats_xyzw(P[j][frames])          # tracked
                qb = _rotmats_to_quats_xyzw(np.broadcast_to(H[j], (len(frames), 3, 3)))
                local = np.empty((len(frames), 3, 3))
                for k, f in enumerate(frames):
                    if w[k] >= 1.0:
                        local[k] = H[j]
                    else:
                        local[k] = np.asarray(quats_xyzw_to_rotmats(
                            _slerp_quats_xyzw(qa[k], qb[k], w[k]))).reshape(3, 3)
                G[frames, j] = np.einsum('fij,fjk->fik', G[frames, p], local)
        if not held.any():
            continue
        pct = 100.0 * held.sum() / max(n, 1)
        what = ("standing pose" if policy == "stand"
                else "neutral pose" if (runs[0][0] == 0 and runs[0][1] == n - 1
                                        and never == "rest")
                else "last seen pose")
        lines.append(f"       Hold Cropped Limbs: {name} out of the picture "
                     f"in {int(held.sum())}/{n} frames ({pct:.0f}%), held "
                     f"({what}, {len(runs)} stretch{'es' if len(runs) != 1 else ''})")

    # The hips: detected here, applied later (steady_hips needs the
    # levelled frame; the position hold lives in modes.py).
    hips_held = np.zeros(n, dtype=bool)
    for (s, e) in held_runs(_limb_out(inside, (KP_LHIP, KP_RHIP), "all"),
                            delay):
        hips_held[s:e + 1] = True
    if hips_held.any():
        lines.append(f"       Hold Cropped Limbs: hips out of the picture in "
                     f"{int(hips_held.sum())}/{n} frames "
                     f"({100.0 * hips_held.sum() / max(n, 1):.0f}%): position "
                     "held, tilt held upright, facing follows the shoulders")

    valid_feet = np.stack([~held_masks[_LEFT_LEG], ~held_masks[_RIGHT_LEG]],
                          axis=1)
    feet_never_seen = bool(n > 0 and held_masks[_LEFT_LEG].all()
                           and held_masks[_RIGHT_LEG].all())
    if not lines:
        lines.append("       Hold Cropped Limbs: every body part stayed in "
                     "the picture; nothing held")
    return G, {"held": held_masks, "valid_feet": valid_feet,
               "feet_never_seen": feet_never_seen, "hips_held": hips_held,
               "lines": lines}


def _mask_runs(mask):
    """[(start, end)] inclusive runs of True."""
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def mask_weights(mask):
    """(n,) hold weight in [0, 1]: 1 on held frames, ramped over
    BLEND_FRAMES inside each run at an edge that borders visible frames."""
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    w = np.zeros(n, dtype=np.float64)
    for (s, e) in _mask_runs(mask):
        w[s:e + 1] = _run_weights(s, e, n)
    return w


def hold_series(arr, mask):
    """Hold a per-frame series (n,) or (n, k) still on the masked frames:
    each run takes the value just before it (or just after it when the
    run starts the clip; the first frame when it spans the whole clip),
    blended at the edges like the rotation holds. Returns a new array."""
    arr = np.array(arr, dtype=np.float64, copy=True)
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    src = arr.copy()
    for (s, e) in _mask_runs(mask):
        ref = s - 1 if s > 0 else (e + 1 if e + 1 < n else 0)
        w = _run_weights(s, e, n)
        w = w.reshape((-1,) + (1,) * (arr.ndim - 1))
        arr[s:e + 1] = (1.0 - w) * src[s:e + 1] + w * src[ref]
    return arr


def _rot_about(axis, angles):
    """(n, 3, 3) rotations about unit `axis` by `angles` (Rodrigues)."""
    a = np.asarray(axis, dtype=np.float64)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    ang = np.asarray(angles, dtype=np.float64)
    s, c = np.sin(ang)[:, None, None], np.cos(ang)[:, None, None]
    return np.eye(3)[None] + s * K[None] + (1.0 - c) * (K @ K)[None]


def _swing_to(u, target):
    """(n, 3, 3) minimal rotations taking each unit vector u[f] onto the
    unit vector `target`."""
    n = len(u)
    out = np.repeat(np.eye(3)[None], n, axis=0)
    axis = np.cross(u, target[None])
    sn = np.linalg.norm(axis, axis=1)
    cs = np.clip(u @ target, -1.0, 1.0)
    for f in range(n):
        if sn[f] < 1e-12:
            if cs[f] < 0:          # opposite: turn half way round any side axis
                side = np.cross(target, [1.0, 0.0, 0.0])
                if np.linalg.norm(side) < 1e-6:
                    side = np.cross(target, [0.0, 0.0, 1.0])
                out[f] = _rot_about(side / np.linalg.norm(side), [np.pi])[0]
            continue
        out[f] = _rot_about(axis[f] / sn[f], [np.arctan2(sn[f], cs[f])])[0]
    return out


def steady_hips(global_rots, hips_held, parents, prerot_quats, offsets_cm,
                fps, world_up=(0.0, 1.0, 0.0)):
    """Steady the pelvis on frames where it was out of the picture.

    Must run on LEVELLED rotations (after the camera offset): "upright"
    is measured against `world_up`, which is +Y in the frame the model's
    global rotations live in.

    On held frames the hips are set to the T-pose pelvis turned to the
    chest's heading (the chest's world orientation with its tilt removed,
    its turn about the vertical low-passed over HEADING_SMOOTH_SEC); the
    legs and anything else hanging off the pelvis are carried along
    rigidly; the spine chain is spread evenly between the steady hips
    and the TRACKED chest, which keeps its world orientation - and with
    it the shoulders, arms, neck and head. Returns a new array."""
    G = np.array(global_rots, dtype=np.float64, copy=True)
    hips_held = np.asarray(hips_held, dtype=bool)
    n = G.shape[0]
    if prerot_quats is None or not hips_held.any() or n == 0:
        return G
    parents = np.asarray(parents)
    T = compute_tpose_globals(parents, prerot_quats)
    P = compute_world_positions(parents, offsets_cm, T)
    up_m = P[J_NECK] - P[J_HIPS]
    up_m = up_m / max(np.linalg.norm(up_m), 1e-12)
    wup = np.asarray(world_up, dtype=np.float64)
    chest = J_SPINE_CHAIN[-1]

    # Chest heading: its deviation from the T-pose with the tilt removed.
    D_c = G[:, chest] @ T[chest].T                       # (n, 3, 3)
    S = _swing_to(D_c @ up_m, wup)
    H_raw = S @ D_c                                      # upright, facing kept
    rel = H_raw @ H_raw[0].T                             # a turn about wup
    # signed turn angle about wup, unwrapped and low-passed
    sin_a = 0.5 * (np.einsum('i,fi->f', wup,
                             np.stack([rel[:, 2, 1] - rel[:, 1, 2],
                                       rel[:, 0, 2] - rel[:, 2, 0],
                                       rel[:, 1, 0] - rel[:, 0, 1]], axis=1)))
    cos_a = 0.5 * (np.trace(rel, axis1=1, axis2=2) - 1.0)
    phi = np.unwrap(np.arctan2(sin_a, cos_a))
    sigma = HEADING_SMOOTH_SEC * max(float(fps), 1e-6)
    if sigma > 0.5 and n > 2:
        from scipy.ndimage import gaussian_filter1d
        # Odd-reflection padding: a turn still under way at the clip's
        # first or last frame keeps its slope instead of being flattened
        # toward the end value (plain 'nearest' padding lost several
        # degrees of a steady turn at both ends).
        pad = min(n - 1, int(round(4 * sigma)))
        if pad > 0:
            head = 2.0 * phi[0] - phi[pad:0:-1]
            tail = 2.0 * phi[-1] - phi[-2:-pad - 2:-1]
            ext = np.concatenate([head, phi, tail])
            phi = gaussian_filter1d(ext, sigma=sigma, mode='nearest')[pad:pad + n]
        else:
            phi = gaussian_filter1d(phi, sigma=sigma, mode='nearest')
    H = _rot_about(wup, phi) @ H_raw[0][None]

    w = mask_weights(hips_held)
    frames = np.where(w > 0)[0]
    Gh_old = G[frames, J_HIPS].copy()
    Gh_new = H[frames] @ T[J_HIPS][None]
    q_old = _rotmats_to_quats_xyzw(Gh_old)
    q_new = _rotmats_to_quats_xyzw(Gh_new)
    Gh = np.empty_like(Gh_old)
    for k, f in enumerate(frames):
        Gh[k] = (Gh_new[k] if w[f] >= 1.0 else np.asarray(quats_xyzw_to_rotmats(
            _slerp_quats_xyzw(q_old[k], q_new[k], w[f]))).reshape(3, 3))

    # Everything hanging off the pelvis except the spine rides along.
    spine_tree = set(_subtree(parents, J_SPINE_CHAIN[0]))
    carried = [j for j in _subtree(parents, J_HIPS)
               if j != J_HIPS and j not in spine_tree]
    carry = np.einsum('fij,fkj->fik', Gh, Gh_old)        # Gh @ Gh_old^T
    for j in carried:
        G[frames, j] = carry @ G[frames, j]
    G[frames, J_HIPS] = Gh

    # Spine: spread evenly between the steady hips and the tracked chest.
    D_h = Gh @ T[J_HIPS].T[None]
    q_h = _rotmats_to_quats_xyzw(D_h)
    q_c = _rotmats_to_quats_xyzw(D_c[frames])
    m = len(J_SPINE_CHAIN)
    for i, j in enumerate(J_SPINE_CHAIN[:-1], start=1):  # the chest stays tracked
        t = i / float(m)
        q_t = _rotmats_to_quats_xyzw(G[frames, j])
        for k, f in enumerate(frames):
            Dk = np.asarray(quats_xyzw_to_rotmats(
                _slerp_quats_xyzw(q_h[k], q_c[k], t))).reshape(3, 3)
            Gk = Dk @ T[j]
            if w[f] < 1.0:
                Gk = np.asarray(quats_xyzw_to_rotmats(_slerp_quats_xyzw(
                    q_t[k], _rotmats_to_quats_xyzw(Gk[None])[0], w[f]))).reshape(3, 3)
            G[f, j] = Gk
    return G


def upright_held_legs(global_rots, held, parents, prerot_quats, offsets_cm,
                      world_up=(0.0, 1.0, 0.0)):
    """Keep held (standing) legs VERTICAL when the pelvis above them is
    tracked and tilts.

    The leg hold is parent-relative, so under a pelvis that pitches or
    rolls (hips in the picture, legs cropped: knee-up framing) straight
    held legs would swing with it like a pendulum, lifting the feet off
    the floor or pushing them through it. A standing person's pelvis
    tips OVER the legs instead. This removes the pelvis' tilt from the
    held leg subtrees (their turn about the vertical is kept, so they
    still face where the pelvis faces). A no-op where the pelvis is
    already upright, which includes every frame steady_hips handled.
    Must run on levelled rotations, after steady_hips. Returns a new
    array."""
    G = np.array(global_rots, dtype=np.float64, copy=True)
    n = G.shape[0]
    if prerot_quats is None or n == 0:
        return G
    masks = [(name, root) for name, root in ((_LEFT_LEG, 2), (_RIGHT_LEG, 18))
             if name in held and np.asarray(held[name]).any()]
    if not masks:
        return G
    parents = np.asarray(parents)
    T = compute_tpose_globals(parents, prerot_quats)
    P = compute_world_positions(parents, offsets_cm, T)
    up_m = P[J_NECK] - P[J_HIPS]
    up_m = up_m / max(np.linalg.norm(up_m), 1e-12)
    wup = np.asarray(world_up, dtype=np.float64)
    D_h = G[:, J_HIPS] @ T[J_HIPS].T
    C = _swing_to(D_h @ up_m, wup)          # un-tilts the pelvis frame
    q_c = _rotmats_to_quats_xyzw(C)
    q_i = np.array([0.0, 0.0, 0.0, 1.0])
    for name, root in masks:
        w = mask_weights(held[name])
        members = _subtree(parents, root)
        for f in np.where(w > 0)[0]:
            Cf = (C[f] if w[f] >= 1.0 else np.asarray(quats_xyzw_to_rotmats(
                _slerp_quats_xyzw(q_i, q_c[f], w[f]))).reshape(3, 3))
            G[f, members] = Cf[None] @ G[f, members]
    return G


def feet_height_below_hips(global_rots, R0, bvh_offsets, parents,
                           foot_joints):
    """(n_frames,) height of the LOWEST of `foot_joints` relative to the
    hips joint, from the skeleton as it will render: rest offsets
    (bvh_offsets, world-aligned rest deltas) carried by each parent's
    rotation away from the rest pose (G[parent] @ R0[parent]^T). Negative
    numbers: the feet hang below the hips."""
    G = np.asarray(global_rots, dtype=np.float64)
    parents = np.asarray(parents)
    n = G.shape[0]
    low = np.full(n, np.inf)
    for j in foot_joints:
        chain = []
        k = int(j)
        while k != J_HIPS and k >= 0:
            chain.append(k)
            k = int(parents[k])
        if k != J_HIPS:
            continue
        rel = np.zeros((n, 3))
        for c in reversed(chain):               # from the hips outward
            p = int(parents[c])
            D_p = G[:, p] @ R0[p].T[None]
            rel = rel + D_p @ np.asarray(bvh_offsets[c], dtype=np.float64)
        low = np.minimum(low, rel[:, 1])
    return low
