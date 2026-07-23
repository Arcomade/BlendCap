# hand_solver.py
# ==============
# Experimental hand refinement: rebuild finger rotations by fitting an
# anatomically-constrained hand to the model's 2D finger keypoints.
#
# WHY: the body model carries finger detail in two places with very
# different quality. Its 2D keypoint head tracks fingers essentially
# perfectly (the overlay), while its 3D rotation head compresses finger
# flexion to ~10-25 deg where a real curl is ~80+ deg — hands come out
# of the BVH looking relaxed no matter what the video shows, and
# amplifying the rotations (--finger-boost) amplifies their off-axis
# noise into backward-bent fingers long before real articulation
# appears. This module ignores the rotation head for fingers and
# re-derives finger rotations from the 2D keypoints instead:
#
#   1. Carry the hand rigidly on the body (the wrist/hand orientation
#      from global_rots is good — only FINGERS are re-solved).
#   2. Per frame, fit a weak-perspective projection (scale + offset) of
#      the rigid palm (wrist + five knuckles) to the observed 2D palm
#      keypoints. Poor fit = occluded/garbage hand -> keep the model's
#      own rotations for that frame (fallback is per-frame, per-hand).
#   3. Per finger, solve 3 bounded hinge angles (knuckle spread,
#      knuckle curl, mid-joint curl; fingertip joint coupled to the
#      mid joint) so the projected finger lands on its observed 2D
#      keypoints. Bounds encode anatomy: fingers cannot bend backward
#      or sideways, which is exactly the constraint that makes a
#      single camera view recoverable.
#   4. Write the solved chains back into global_rots so every
#      downstream consumer (prune, local rotations, BVH write, the
#      hands-only export) is untouched.
#
# Axis conventions (which way is "curl", whether image Y is flipped
# relative to the camera frame) are auto-calibrated per clip by trying
# both signs on sample frames and keeping the one the observations fit
# — no hardcoded handedness to get wrong.
#
# Runs in the addon venv (scipy available). Mutates global_rots in
# place; frames it can't trust keep the model's rotations, with a short
# quaternion cross-fade at solved/unsolved boundaries so switches don't
# pop.

import numpy as np

from .bvh_math import (
    _rotmats_to_quats_xyzw,
    _slerp_quats_xyzw,
    compute_world_positions,
    quats_xyzw_to_rotmats,
)

# 2D keypoint indices (MHR70 layout, matches overlay_track.py) per hand.
# Finger tuples are chain order: knuckle (third_joint), mid (2), distal
# (3), tip (4).
_KP2D = {
    "right": {"wrist": 41,
              "fingers": {"thumb": (24, 23, 22, 21),
                          "index": (28, 27, 26, 25),
                          "middle": (32, 31, 30, 29),
                          "ring": (36, 35, 34, 33),
                          "pinky": (40, 39, 38, 37)}},
    "left": {"wrist": 62,
             "fingers": {"thumb": (45, 44, 43, 42),
                         "index": (49, 48, 47, 46),
                         "middle": (53, 52, 51, 50),
                         "ring": (57, 56, 55, 54),
                         "pinky": (61, 60, 59, 58)}},
}

# 3D joints (127-joint SAM skeleton): hand root + per-finger articulated
# chain (knuckle, mid, distal) + tip marker joint. Sourced from
# body_skeleton.FINGER_TOPOLOGY so the solver and the finger boost share
# one index table and can never disagree.
from .body_skeleton import FINGER_TOPOLOGY as _J3D

# Distal joint curls at a fixed fraction of the mid joint (standard
# anatomical coupling; removes a depth-ambiguous DOF).
_DIP_COUPLE = 0.7

# Bounds (radians): [spread, knuckle curl, mid curl]. Fingers cannot
# hyperextend much or bend sideways far — this is what makes the
# monocular fit well-posed.
_FINGER_LO = np.array([-0.35, -0.30, -0.17])
_FINGER_HI = np.array([0.35, 1.75, 1.92])
# Thumb gets wider spread (its "flexion plane" is oblique).
_THUMB_LO = np.array([-0.90, -0.50, -0.35])
_THUMB_HI = np.array([0.90, 1.40, 1.60])

# Regularizer weights (residuals are normalized by projected hand size,
# so these are dimensionless): pull toward the previous frame's angles
# and toward zero spread.
_W_TEMPORAL = 0.06
_W_SPREAD = 0.04

# Gates.
_MIN_PALM_SPAN_PX = 15.0     # projected palm smaller than this = no signal
_PALM_RESID_FRAC = 0.30      # palm fit residual vs palm span
_FINGER_RESID_FRAC = 0.22    # mean finger residual vs palm span
_BLEND_SIGMA_FRAMES = 1.2    # solved/unsolved cross-fade width
_STATIC_EPS_FRAC = 0.02      # skip-static: max keypoint move (fraction
                             # of palm span, ~4px on a 180px hand) below
                             # which a frame reuses the last solved
                             # result instead of solving. Anchored to
                             # the last SOLVED frame so drift can't
                             # accumulate. MEASURED NOTE: the 2D head
                             # jitters 5-14% of span per frame on
                             # ordinary footage, so this fires rarely
                             # there - it pays off on stable close-ups
                             # (big hands = less relative jitter), and
                             # costs nothing elsewhere.


def _rot_about(axis, angle):
    """Rotation matrix about a (unit) axis. Rodrigues, scalar angle."""
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    s, c = np.sin(angle), np.cos(angle)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _skew(axis):
    x, y, z = axis
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


# Escape hatch: BLENDCAP_HAND_SOLVER_SCIPY=1 restores the original
# scipy.optimize.least_squares fits (same problem, ~5x slower) in case
# the hand-rolled solver ever misbehaves on someone's footage.
import os as _os
_USE_SCIPY = _os.environ.get("BLENDCAP_HAND_SOLVER_SCIPY", "0") == "1"


def _lm_solve(fn, x0, lo, hi, max_iter=40):
    """Small damped Gauss-Newton (Levenberg-Marquardt) with box
    clamping. Replaces scipy.optimize.least_squares for the tiny
    3-parameter fits here — the SAME residuals and bounds, minus
    scipy's per-call overhead (the measured bottleneck: thousands of
    micro-solves per clip). Forward-difference jacobian, multiplicative
    damping, accept-only-improving steps. Returns (x, cost) with
    cost = sum(r^2).

    Stop criteria matter: only treat a small step as convergence when
    it was an ACCEPTED step taken at LOW damping — a tiny step forced
    by high damping means "blocked", not "converged" (stopping there
    left fits measurably worse than scipy's on ~1/3 of frames)."""
    x = np.clip(np.asarray(x0, dtype=np.float64), lo, hi)
    r = fn(x)
    cost = float(r @ r)
    lam = 1e-3
    eps = 1e-6
    n = x.size
    fails = 0
    for _ in range(max_iter):
        J = np.empty((r.size, n))
        for i in range(n):
            xi = x.copy()
            xi[i] += eps
            J[:, i] = (fn(xi) - r) / eps
        g = J.T @ r
        H = J.T @ J
        dH = np.diag(np.diag(H) + 1e-12)
        stepped = False
        dx = None
        for _try in range(6):
            try:
                dx = np.linalg.solve(H + lam * dH, -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            # Candidate 1: clip to the box. Candidate 2 (only when the
            # clip changed something): scale the WHOLE step to the
            # first bound crossing, preserving its direction — plain
            # clipping distorts the step exactly where fingers live
            # (full curl sits at a bound), which is where scipy's
            # bound-aware TRF was out-fitting this solver.
            xn = np.clip(x + dx, lo, hi)
            rn = fn(xn)
            cn = float(rn @ rn)
            if not np.array_equal(xn, x + dx):
                with np.errstate(divide="ignore", invalid="ignore"):
                    hi_a = np.where(dx > 0, (hi - x) / dx, np.inf)
                    lo_a = np.where(dx < 0, (lo - x) / dx, np.inf)
                alpha = float(min(1.0, np.min(np.minimum(hi_a, lo_a))))
                if 1e-9 < alpha < 1.0:
                    xs = np.clip(x + alpha * dx, lo, hi)
                    rs = fn(xs)
                    cs = float(rs @ rs)
                    if cs < cn:
                        xn, rn, cn = xs, rs, cs
                # Candidate 3 (active-set): pin the coordinates the
                # step drove into their bounds and re-solve for the
                # FREE ones — clipping/scaling otherwise stalls free-
                # coordinate progress whenever one coordinate saturates
                # (full curls sit exactly there; this is what scipy's
                # reflective TRF does natively).
                pinned = ((np.isclose(xn, hi) & (dx > 0))
                          | (np.isclose(xn, lo) & (dx < 0)))
                free = ~pinned
                if pinned.any() and free.any():
                    xa = np.clip(x + dx, lo, hi)
                    Hf = H[np.ix_(free, free)]
                    gf = (g + H @ (xa - x))[free]
                    try:
                        dxf = np.linalg.solve(
                            Hf + lam * np.diag(np.diag(Hf) + 1e-12), -gf)
                        xf = xa.copy()
                        xf[free] += dxf
                        xf = np.clip(xf, lo, hi)
                        rf = fn(xf)
                        cf = float(rf @ rf)
                        if cf < cn:
                            xn, rn, cn = xf, rf, cf
                    except np.linalg.LinAlgError:
                        pass
            if cn < cost - 1e-12:
                improve = (cost - cn) / max(cost, 1e-12)
                x, r, cost = xn, rn, cn
                lam = max(lam * 0.3, 1e-8)
                stepped = True
                break
            lam *= 5.0
        if not stepped:
            fails += 1
            if fails >= 2 or lam > 1e6:
                break
            continue
        fails = 0
        # Converged only if the ACCEPTED step was tiny at low damping,
        # or the relative improvement has petered out.
        if cost < 1e-10:
            break
        if (lam <= 1e-3 and float(np.max(np.abs(dx))) < 1e-5):
            break
        if improve < 1e-8:
            break
    return x, cost


def _build_hand_model(side, parents, rest_pos, R0):
    """Fixed per-hand geometry in the hand's rest frame.

    rest_pos: world rest positions (cm) for all joints; R0: rest global
    rotations. Everything the per-frame solve needs that doesn't change:
    hand-frame joint positions, segment vectors, hinge axes.
    """
    j = _J3D[side]
    hand = j["hand"]
    Rh_T = R0[hand].T
    origin = rest_pos[hand]

    def hframe(idx):
        return Rh_T @ (rest_pos[idx] - origin)

    fingers = {}
    mcps = []
    for name, (mcp, pip, dip, tip) in j["fingers"].items():
        h_mcp, h_pip, h_dip, h_tip = (hframe(mcp), hframe(pip),
                                      hframe(dip), hframe(tip))
        mcps.append(h_mcp)
        fingers[name] = {
            "joints": (mcp, pip, dip, tip),
            "h_mcp": h_mcp,
            "v1": h_pip - h_mcp,
            "v2": h_dip - h_pip,
            "v3": h_tip - h_dip,
        }
    # Palm plane normal from the knuckle row vs wrist (hand origin).
    # Sign is arbitrary here — the flexion-sign calibration absorbs it.
    row = fingers["index"]["h_mcp"] - fingers["pinky"]["h_mcp"]
    mid = fingers["middle"]["h_mcp"]
    n = np.cross(row, mid)
    n = n / max(np.linalg.norm(n), 1e-9)

    for name, f in fingers.items():
        bone = f["v1"] / max(np.linalg.norm(f["v1"]), 1e-9)
        if name == "thumb":
            # The thumb OPPOSES — its curl sweeps the tip across the
            # palm toward the pinky base, in a plane roughly
            # perpendicular to the palm. Finger-style palm-normal axes
            # drive it straight through the finger mass on a fist.
            # Axis construction: rotation about cross(bone, d) moves
            # the tip toward +d, so d = direction to the pinky knuckle
            # gives anatomical opposition with a DETERMINISTIC sign —
            # the thumb ignores the calibrated palm-normal flex sign.
            d_opp = fingers["pinky"]["h_mcp"] - f["h_mcp"]
            d_opp = d_opp - bone * float(d_opp @ bone)
            d_opp = d_opp / max(np.linalg.norm(d_opp), 1e-9)
            flex = np.cross(bone, d_opp)
            flex = flex / max(np.linalg.norm(flex), 1e-9)
            spread = d_opp
            f["sign_locked"] = True
        else:
            flex = np.cross(bone, n)
            flex = flex / max(np.linalg.norm(flex), 1e-9)
            # Spread axis: palm normal component orthogonal to the bone.
            spread = n - bone * float(n @ bone)
            spread = spread / max(np.linalg.norm(spread), 1e-9)
            f["sign_locked"] = False
        f["flex_axis"] = flex
        f["spread_axis"] = spread
        f["lo"] = _THUMB_LO if name == "thumb" else _FINGER_LO
        f["hi"] = _THUMB_HI if name == "thumb" else _FINGER_HI
        # Precomputed Rodrigues terms (K, K@K) per axis: the FK runs
        # thousands of times per clip inside the solver loop, and
        # rebuilding these per call was a measurable cost.
        f["K_flex"] = _skew(flex)
        f["KK_flex"] = f["K_flex"] @ f["K_flex"]
        f["K_spread"] = _skew(spread)
        f["KK_spread"] = f["K_spread"] @ f["K_spread"]

    palm_pts = np.stack([np.zeros(3)] + mcps)   # wrist + 5 knuckles
    palm_extent = float(np.max(np.linalg.norm(palm_pts, axis=1)))
    return {"side": side, "hand": hand, "fingers": fingers,
            "palm_pts": palm_pts, "palm_extent": palm_extent}


def _finger_fk(f, th, flex_sign):
    """Hand-frame FK for one finger. Returns per-joint chain rotations
    (hand frame) and the pip/dip/tip positions. Sign-locked fingers
    (the thumb — deterministic opposition axis) ignore the calibrated
    palm-normal flex sign, which is folded into the ANGLE here
    (rotation about sign*axis by theta == rotation about axis by
    sign*theta) so the precomputed K/KK terms serve both signs."""
    sgn = 1.0 if f.get("sign_locked") else flex_sign
    Kf, KKf = f["K_flex"], f["KK_flex"]
    Ks, KKs = f["K_spread"], f["KK_spread"]
    I = np.eye(3)
    a0, a1, a2 = th[0], sgn * th[1], sgn * th[2]
    R_m = ((I + np.sin(a0) * Ks + (1.0 - np.cos(a0)) * KKs)
           @ (I + np.sin(a1) * Kf + (1.0 - np.cos(a1)) * KKf))
    p_pip = f["h_mcp"] + R_m @ f["v1"]
    R_p = R_m @ (I + np.sin(a2) * Kf + (1.0 - np.cos(a2)) * KKf)
    p_dip = p_pip + R_p @ f["v2"]
    a3 = _DIP_COUPLE * a2
    R_d = R_p @ (I + np.sin(a3) * Kf + (1.0 - np.cos(a3)) * KKf)
    p_tip = p_dip + R_d @ f["v3"]
    return (R_m, R_p, R_d), (p_pip, p_dip, p_tip)


def _fit_palm(M3, palm_pts, obs):
    """Weak-perspective fit: obs ~ s * (M3 @ palm_pts) + c.
    M3: 2x3 (image axes of the current hand frame). Returns
    (s, c, residual) with residual = mean point error in px."""
    q = palm_pts @ M3.T                    # (6, 2)
    qc = q - q.mean(axis=0)
    oc = obs - obs.mean(axis=0)
    denom = float(np.sum(qc * qc))
    if denom < 1e-9:
        return 0.0, np.zeros(2), np.inf
    s = float(np.sum(qc * oc)) / denom
    if s <= 1e-9:
        return 0.0, np.zeros(2), np.inf
    c = obs.mean(axis=0) - s * q.mean(axis=0)
    resid = float(np.mean(np.linalg.norm(s * q + c - obs, axis=1)))
    return s, c, resid


def _refine_palm_pose(A, Gh, palm_pts, obs):
    """Palm-pose refinement: the model's hand ORIENTATION drifts on
    strong poses (a closing fist measured 40-58 deg off), and fingers
    fitted in a wrong palm frame can only partially match the
    observations. Solve a small world-side rotation correction of the
    hand frame that best explains the observed palm keypoints; fingers
    are fitted in the corrected frame AND the correction is written
    back into the hand bone (smoothed + blended by the caller) so the
    curls land in the true palm rather than diagonally beside it.
    Returns (M3, s, c, resid, rv) — rv is the rotation vector of the
    correction (zeros when the unrefined frame fits better)."""
    def sc_resid(rv):
        M3 = A @ (_rot_about_vec(rv) @ Gh)
        s, c, resid = _fit_palm(M3, palm_pts, obs)
        return M3, s, c, resid

    def residuals(rv):
        M3, s, c, _ = sc_resid(rv)
        if s <= 1e-9:
            return np.full(palm_pts.shape[0] * 2, 1e3)
        pred = palm_pts @ M3.T * s + c
        return (pred - obs).ravel()

    lo, hi = -0.7 * np.ones(3), 0.7 * np.ones(3)
    if _USE_SCIPY:
        from scipy.optimize import least_squares
        res = least_squares(residuals, np.zeros(3), bounds=(lo, hi),
                            method="trf", max_nfev=40)
        rv_fit = res.x
    else:
        rv_fit, _ = _lm_solve(residuals, np.zeros(3), lo, hi)
    M3, s, c, resid = sc_resid(rv_fit)
    # Never accept a refinement worse than the unrefined frame.
    M0 = A @ Gh
    s0, c0, r0 = _fit_palm(M0, palm_pts, obs)
    if r0 <= resid:
        return M0, s0, c0, r0, np.zeros(3)
    return M3, s, c, resid, rv_fit


def _rot_about_vec(rv):
    """Rotation matrix from a rotation vector (axis * angle)."""
    ang = float(np.linalg.norm(rv))
    if ang < 1e-12:
        return np.eye(3)
    return _rot_about(np.asarray(rv) / ang, ang)


def _solve_finger(f, M3, s, c, obs3, norm, th0, flex_sign):
    """Bounded least-squares for one finger on one frame.
    obs3: observed 2D for (pip, dip, tip). norm: pixel normalizer
    (projected palm size). Returns (theta, mean_resid_px)."""
    sM3T = M3.T * s

    def residuals(th):
        _, pts = _finger_fk(f, th, flex_sign)
        pred = np.stack(pts) @ sM3T + c
        r = ((pred - obs3) / norm).ravel()
        reg_t = _W_TEMPORAL * (th - th0)
        reg_s = np.array([_W_SPREAD * th[0]])
        return np.concatenate([r, reg_t, reg_s])

    # Restarts cover the fit's local minima: previous frame's answer,
    # half curl, and full curl (a fist from a relaxed init gets stuck
    # at partial curl otherwise). Early-exit thresholds match across
    # both solver paths (scipy cost = 0.5*sum(r^2), LM cost = sum).
    if _USE_SCIPY:
        from scipy.optimize import least_squares
        best = None
        for x0 in (th0, np.array([0.0, 0.6, 0.6]),
                   np.array([0.0, 1.5, 1.7])):
            res = least_squares(residuals, np.clip(x0, f["lo"], f["hi"]),
                                bounds=(f["lo"], f["hi"]), method="trf",
                                max_nfev=60)
            if best is None or res.cost < best.cost:
                best = res
            if best.cost < 5e-3:
                break
        best_x = best.x
    else:
        # Restart policy: extra starts exist to escape HIGH-residual
        # local minima (a fist solved from a relaxed init). When the
        # previous-frame init already lands a good fit (cost ~= a few
        # px of normalized error), further restarts just re-find the
        # same answer - skip them. Poor fits still get all three.
        best_x, best_cost = None, np.inf
        for x0 in (th0, np.array([0.0, 0.6, 0.6]),
                   np.array([0.0, 1.5, 1.7])):
            x, cost = _lm_solve(residuals, x0, f["lo"], f["hi"])
            if cost < best_cost:
                best_x, best_cost = x, cost
            if best_cost < 0.05:
                break
    _, pts = _finger_fk(f, best_x, flex_sign)
    pred = np.stack(pts) @ sM3T + c
    resid = float(np.mean(np.linalg.norm(pred - obs3, axis=1)))
    return best_x, resid


def _valid_2d(pts):
    return np.all(np.isfinite(pts)) and np.all(np.abs(pts) < 1e5)


def _calibrate(hm, kp2d, global_rots, sample_frames):
    """Pick (image y-sign, flexion sign) per hand by trying all four
    combos on sample frames and keeping the best-fitting one."""
    side = hm["side"]
    k = _KP2D[side]
    best = (1.0, 1.0)
    best_cost = np.inf
    for ay in (1.0, -1.0):
        A = np.array([[1.0, 0.0, 0.0], [0.0, ay, 0.0]])
        for fs in (1.0, -1.0):
            cost, used = 0.0, 0
            for fr in sample_frames:
                obs_palm = np.stack(
                    [kp2d[fr, k["wrist"]]]
                    + [kp2d[fr, k["fingers"][n][0]]
                       for n in hm["fingers"]])
                if not _valid_2d(obs_palm):
                    continue
                M3, s, c, resid, _rv = _refine_palm_pose(
                    A, global_rots[fr, hm["hand"]], hm["palm_pts"],
                    obs_palm)
                norm = max(s * hm["palm_extent"], 1e-6)
                if norm < _MIN_PALM_SPAN_PX or resid > 0.6 * norm:
                    continue
                for name, f in hm["fingers"].items():
                    obs3 = kp2d[fr, list(k["fingers"][name][1:])]
                    if not _valid_2d(obs3):
                        continue
                    _, r = _solve_finger(f, M3, s, c, obs3, norm,
                                         np.zeros(3), fs)
                    cost += r / norm
                    used += 1
            if used:
                cost /= used
                if cost < best_cost:
                    best_cost, best = cost, (ay, fs)
    return best, best_cost


def solve_hands(global_rots, keypoints_2d, parents, offsets_cm, R0,
                verbose=True, progress_cb=None):
    """Refit finger rotations to the 2D keypoints. Mutates global_rots
    in place (finger + tip joints only). progress_cb(done, total,
    detail) is called across the per-frame solve (done/total over both
    hands' frames; detail = "hand h/2, frame f/N") so the caller can
    drive a progress bar with a live frame counter - the per-frame fit
    is the long part. Returns a summary dict:
    {'right': solved_frame_count, 'left': ..., 'n_frames': N,
     'any_solved': bool}.

    R0 must be the T-pose rest globals (compute_tpose_globals) — the
    hand-frame geometry is derived from it, matching the convention
    boost_finger_rotations uses.
    """
    n_frames = global_rots.shape[0]
    kp2d = keypoints_2d.astype(np.float64)

    # World rest positions (cm) for hand-frame geometry — offsets are
    # parent-local, so they accumulate through the rest rotations.
    rest = compute_world_positions(parents, offsets_cm, R0)

    summary = {"n_frames": n_frames, "any_solved": False}
    sample = list(range(0, n_frames, max(1, n_frames // 12)))[:12]

    # Progress spans both hands' per-frame solves (the long part).
    total_steps = max(1, 2 * n_frames)
    steps_done = [0]

    for hnum, side in enumerate(("right", "left"), 1):
        hm = _build_hand_model(side, parents, rest, R0)
        k = _KP2D[side]
        (ay, flex_sign), cal_cost = _calibrate(hm, kp2d, global_rots,
                                               sample)
        A = np.array([[1.0, 0.0, 0.0], [0.0, ay, 0.0]])

        solved_mask = np.zeros(n_frames, bool)
        th_arr = {name: np.zeros((n_frames, 3))
                  for name in hm["fingers"]}
        rv_arr = np.zeros((n_frames, 3))
        th_prev = {name: np.zeros(3) for name in hm["fingers"]}
        finger_names = list(hm["fingers"])

        # Skip-static reuse: when this frame's hand keypoints barely
        # moved from the LAST SOLVED frame, reuse that frame's result
        # instead of re-solving. Comparing to the last *solved* frame
        # (not the previous frame) means slow drift still accumulates
        # to a real re-solve once it crosses the threshold, so this is
        # effectively output-identical, not a running approximation.
        last_solved_obs = None   # (K,2) keypoints of the last real solve
        last_solved_norm = 0.0
        last_solved_rv = np.zeros(3)
        reused = 0

        def _gather_obs(fr):
            palm = np.stack(
                [kp2d[fr, k["wrist"]]]
                + [kp2d[fr, k["fingers"][nm][0]] for nm in finger_names])
            fobs = {nm: kp2d[fr, list(k["fingers"][nm][1:])]
                    for nm in finger_names}
            ok = _valid_2d(palm) and all(_valid_2d(o) for o in fobs.values())
            allpts = np.concatenate([palm] + [fobs[nm] for nm in finger_names])
            return palm, fobs, ok, allpts

        for fr in range(n_frames):
            steps_done[0] += 1
            if progress_cb is not None and (fr & 7) == 0:
                progress_cb(steps_done[0], total_steps,
                            f"hand {hnum}/2, frame {fr + 1}/{n_frames}")
            obs_palm, finger_obs, all_ok, all_pts = _gather_obs(fr)
            if not _valid_2d(obs_palm):
                continue

            # Reuse the last solved frame if the hand is effectively
            # static (all keypoints within _STATIC_EPS_FRAC of the palm
            # span). The solver is deterministic + continuous, so near-
            # identical input gives near-identical output; smoothing
            # erases the sub-threshold difference.
            if (all_ok and last_solved_obs is not None
                    and np.max(np.linalg.norm(
                        all_pts - last_solved_obs, axis=1))
                    < _STATIC_EPS_FRAC * last_solved_norm):
                solved_mask[fr] = True
                rv_arr[fr] = last_solved_rv
                for nm in finger_names:
                    th_arr[nm][fr] = th_prev[nm]
                reused += 1
                continue

            M3, s, c, p_resid, rv = _refine_palm_pose(
                A, global_rots[fr, hm["hand"]], hm["palm_pts"], obs_palm)
            norm = max(s * hm["palm_extent"], 1e-6)
            if norm < _MIN_PALM_SPAN_PX or p_resid > _PALM_RESID_FRAC * norm:
                continue

            resids, frame_th = [], {}
            for name, f in hm["fingers"].items():
                obs3 = finger_obs[name]
                if not _valid_2d(obs3):
                    frame_th = None
                    break
                th, r = _solve_finger(f, M3, s, c, obs3, norm,
                                      th_prev[name], flex_sign)
                resids.append(r)
                frame_th[name] = th
                th_prev[name] = th
            if frame_th is None:
                continue
            if np.mean(resids) > _FINGER_RESID_FRAC * norm:
                continue

            solved_mask[fr] = True
            rv_arr[fr] = rv
            for name, th in frame_th.items():
                th_arr[name][fr] = th
            # Anchor for skip-static: this frame's keypoints + result.
            if all_ok:
                last_solved_obs = all_pts
                last_solved_norm = norm
                last_solved_rv = rv

        # Light temporal smoothing of the solved angles AND the palm
        # correction: per-frame keypoint noise otherwise reads as
        # finger/wrist jitter on quiet hands. Window 5 / order 2 keeps
        # real transitions (a counted finger flips in ~4-6 frames)
        # while flattening 1-frame noise.
        idx = np.where(solved_mask)[0]
        grid = np.arange(n_frames)
        if len(idx) >= 5 and n_frames >= 5:
            from scipy.signal import savgol_filter
            for name in hm["fingers"]:
                th = th_arr[name]
                filled = np.stack(
                    [np.interp(grid, idx, th[idx, c3]) for c3 in range(3)],
                    axis=1)
                th_arr[name] = savgol_filter(filled, 5, 2, axis=0)
            rv_filled = np.stack(
                [np.interp(grid, idx, rv_arr[idx, c3]) for c3 in range(3)],
                axis=1)
            rv_arr = savgol_filter(rv_filled, 5, 2, axis=0)
        # Safety cap on the palm correction magnitude (~57 deg).
        mags = np.linalg.norm(rv_arr, axis=1)
        over = mags > 1.0
        if np.any(over):
            rv_arr[over] *= (1.0 / mags[over])[:, None]

        # Chain rotations from the (smoothed) angles.
        chain_R = {name: np.tile(np.eye(3), (n_frames, 3, 1, 1))
                   for name in hm["fingers"]}
        for name, f in hm["fingers"].items():
            th = th_arr[name]
            for fr in idx:
                Rs = _finger_fk(f, th[fr], flex_sign)[0]
                chain_R[name][fr, 0] = Rs[0]
                chain_R[name][fr, 1] = Rs[1]
                chain_R[name][fr, 2] = Rs[2]

        n_solved = int(solved_mask.sum())
        summary[side] = n_solved
        summary[side + "_mask"] = solved_mask
        if verbose:
            print(f"       Hand solver [{side:5s}]: {n_solved}/{n_frames} "
                  f"frames fit ({reused} reused static, axis cal "
                  f"{ay:+.0f}/{flex_sign:+.0f})")
        if not n_solved:
            continue
        summary["any_solved"] = True

        # Cross-fade weight: 1 on solved frames, easing to 0 near
        # unsolved ones so the model/solver switch never pops.
        w = solved_mask.astype(np.float64)
        if 0 < n_solved < n_frames:
            radius = int(np.ceil(3 * _BLEND_SIGMA_FRAMES))
            kern = np.exp(-0.5 * (np.arange(-radius, radius + 1)
                                  / _BLEND_SIGMA_FRAMES) ** 2)
            kern /= kern.sum()
            w = np.convolve(w, kern, mode="same")
            w[~solved_mask] = np.minimum(w[~solved_mask], 0.85)
            w = np.clip(w, 0.0, 1.0)

        hand = hm["hand"]
        # Palm-orientation write-back: rotate the hand bone by the
        # fitted correction (blended by w like the fingers) so the
        # whole hand faces where the video says it faces. Without this
        # the solved curls attach to the model's mis-rotated palm and
        # sweep diagonally toward the knife edge of the hand.
        G_hand_corr = np.stack(
            [_rot_about_vec(rv_arr[f]) @ global_rots[f, hand]
             for f in range(n_frames)])
        q_old_h = _rotmats_to_quats_xyzw(global_rots[:, hand])
        q_new_h = _rotmats_to_quats_xyzw(G_hand_corr)
        global_rots[:, hand] = quats_xyzw_to_rotmats(
            _slerp_quats_xyzw(q_old_h, q_new_h, w[:, None]))

        # Write back fingers: solved world rotation for chain joint j is
        #   G_hand @ R_chain @ R0[hand]^T @ R0[j]
        # (rigid hand carry composed with the solved hand-frame chain,
        # G_hand now palm-corrected), blended toward the model's own
        # rotation by w.
        R0h_T = R0[hand].T
        for name, f in hm["fingers"].items():
            joints = f["joints"]           # (mcp, pip, dip, tip)
            for ci, j in enumerate(joints):
                rig = R0h_T @ R0[j]
                chain_idx = min(ci, 2)      # tip rides the distal chain
                G_new = np.einsum(
                    "nij,njk,kl->nil",
                    global_rots[:, hand],
                    chain_R[name][:, chain_idx],
                    rig)
                q_new = _rotmats_to_quats_xyzw(G_new)
                q_old = _rotmats_to_quats_xyzw(global_rots[:, j])
                q = _slerp_quats_xyzw(q_old, q_new, w[:, None])
                global_rots[:, j] = quats_xyzw_to_rotmats(q)

    return summary
