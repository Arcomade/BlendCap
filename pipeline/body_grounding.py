"""Body-grounding correction for non-upright poses (rolls, lying,
handstands, crawls).

SAM's monocular height estimate drifts hardest when the body is
horizontal or inverted — a front roll or lay-back hovers above the
floor. Foot grounding can't help: the feet aren't the contact point.

RUNS LAST (v3, 2026-07-29): the pass measures the FINAL FK-rendered
skeleton — after rotation smoothing, depth denoise, and footskate
cleanup — and corrects the Hips Y channel as the last word. Earlier
versions measured SAM's raw joint positions early in the pipeline and
were then invalidated by everything downstream: rotation smoothing
straightens the roll's spine curl (rendered spine floated +16cm where
the raw metric said +3), and the foot passes move the feet that define
the visual floor. Measuring what ships makes the correction match the
viewport by construction.

Design (no event detection — per-frame gates):
1. Lowest body surface per frame: min over contact joints of
   (world Y - per-joint skin radius), so "touching" means the SURFACE
   touches, not a spine joint buried inside the floor. Feet/toes are
   in the set with small radii: whenever feet are genuinely lowest the
   hover reads ~0 and the correction self-cancels.
2. Posture gate: only while the torso axis (Hips -> Spine3) is far
   from pointing UP. Upright covers standing, walking, squats, and
   JUMPS (untouchable by construction); non-upright covers rolls,
   lying, crawls, cartwheels, and handstands (inverted counts: angle
   measured from up).
3. Ballistic protection: genuinely airborne frames are recognized by
   their vertical acceleration matching gravity and left alone — that
   protects butterfly kicks, b-twists, and a dive-roll's flight. A
   floated roll just hovers (near-zero vertical acceleration), so it
   stays correctable at ANY altitude. catch_distance is the correction
   magnitude CAP only, not an eligibility band.
4. Per-frame contact grounding: within a roll the contact point walks
   across the body and the hover changes with it, so each gated
   frame's lowest point grounds individually (lightly smoothed), with
   a near-contact clamp guaranteeing the residual lands in a tight
   band around touch. Negative hover lifts (floor penetration).

Floor reference: the FINAL feet — a low percentile of the per-frame
minimum foot surface across the take, i.e. where the feet actually
ended up after the foot passes, not where SAM first guessed them.
"""
import numpy as np

from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.signal import savgol_filter

# BVH joint name -> skin radius (m): distance from joint center to the
# body surface toward the ground for the poses this pass targets (back
# for the spine chain, palm for hands...). Coarse by design — the goal
# is "clearly touching", not skin-perfect contact.
# The back group (hips + spine chain + neck) uses the user-facing skin
# radius instead: MHR places these joints anatomically, almost at the
# back's skin surface, so the offset is small (default 1.5cm) and
# exposed as the Skin Radius slider under Body Grounding.
BACK_JOINTS = ("Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck")

CONTACT_RADII = {
    "Head": 0.10,
    "LeftUpLeg": 0.09, "RightUpLeg": 0.09,
    "LeftLeg": 0.05, "RightLeg": 0.05,
    "LeftFoot": 0.03, "RightFoot": 0.03,
    "LeftToe": 0.02, "RightToe": 0.02,
    "LeftArm": 0.08, "RightArm": 0.08,       # shoulder balls
    "LeftForeArm": 0.04, "RightForeArm": 0.04,
    "LeftHand": 0.02, "RightHand": 0.02,
}
FOOT_NAMES = ("LeftFoot", "RightFoot", "LeftToe", "RightToe")

# Posture gate ramp (degrees of torso tilt from world-up).
UPRIGHT_DEG = 45.0
NONUPRIGHT_DEG = 70.0

CORRECTION_SIGMA_S = 0.10
GATE_SIGMA_S = 0.12

# --- Support-plane tilt alignment (compute_body_tilt_alignment) ---
# Resting gate ramp (deg/s of body angular velocity): full correction
# below LO, none above HI. Measured on a real take: lying is 2-5,
# get-up transitions 30-80, rolls hundreds.
TILT_REST_DEG_S_LO = 40.0
TILT_REST_DEG_S_HI = 90.0
# Underside selection: joints whose windowed-median surface sits
# within this band of the lowest median join the initial plane fit
# (generous — a tilted body's high-side contacts sit 15+ cm up);
# the refit keeps only joints this close to the fitted plane.
TILT_UNDERSIDE_BAND_M = 0.20
TILT_PLANE_BAND_M = 0.07
# Support-area gate: minimum spread along the second in-plane
# principal direction. Two hands (a handstand) span a line, not a
# polygon — no rotation without real area.
TILT_MIN_SPREAD_M = 0.08
# Temporal smoothing of the correction (seconds).
TILT_SMOOTH_S = 0.30
# Resting seating: percentile of the contact-joint surfaces used as
# the "lowest" on resting frames. ~20% of 21 contact joints is the
# 4th-5th lowest — past any single buried outlier, inside the real
# support set (a lying body has 6-10 genuine contacts).
TILT_CONSENSUS_PCT = 20.0

# --- Contact-optimized placement (resting frames) ---
# A resting body should TOUCH the floor the way a settled rigid body
# does: as many support joints connected as possible. After the
# support plane is leveled, a bounded refinement (pitch/roll about
# the contact centroid, plus seat height) minimizes: hover of
# support joints above the floor, plus burial beyond each joint's
# allowance. Burial is free up to what the joint's flesh radius
# hides OR what the capture already buried it by (+ slack) — an
# already-buried outlier stays hidden under the body instead of
# levitating it, but no joint is pushed meaningfully deeper than
# the capture had it (a kneel's load-bearing knees keep their
# depth). Burial beyond allowance costs more than hover
# (TILT_BURY_W) so a single deep outlier always loses the vote —
# the failure mode of scoring candidate planes by raw contact
# count, measured on a real take: the max-count face tilted 10 deg
# and drove a forearm 14 cm under.
TILT_REFINE_DEG = 8.0          # refinement half-range around level
TILT_REFINE_STEP_DEG = 0.5
TILT_SEAT_M = 0.06             # seat-height half-range
TILT_SEAT_STEP_M = 0.005
TILT_NEAR_M = 0.08             # support candidacy band above consensus
TILT_BURY_W = 1.5
TILT_FLESH_FRAC = 0.6          # of contact radius...
TILT_FLESH_CAP_M = 0.02        # ...capped: hideable press depth
TILT_BURY_SLACK_M = 0.01       # beyond the capture's own burial
# Isolation band for the seat blend's deep endpoint. A LONE deep
# joint — no companion within this band above it — is a capture
# depth error, never a seat target: without the filter, the
# resting-gate fade at the end of a lie blends the seat from the
# consensus back to a knee buried 14 cm below the support and hoists
# the body BEFORE the get-up motion justifies it. A DENSE deep group
# passes: a side-lie's hip+forearm+hand press together (measured
# within 5 cm of each other, 9-13 under) and ARE the support — the
# filter must not float the body off them. Measured separation:
# the lone knee's nearest companion sits 13 cm above it.
TILT_ISO_BAND_M = 0.05
TILT_ISO_WIN_S = 0.15          # median window for the walk
TILT_ISO_SMOOTH_S = 0.08       # output smoothing per resting run
# RESTING seat: consensus vs the deep group, decided by their gap.
# A prone lie's support spans a few cm below the consensus (measured
# 4 cm on a real take) — seat the consensus and let the lone buried
# knee hide under the body. A side-lie is different: its hip+hand+
# forearm group presses 11+ cm below the consensus cluster (the
# capture's depth error grows along the body: foot at floor, hip
# 13 under) — the group IS the surface the body rests on, and
# seating the consensus parks the body's CENTERLINE on the floor,
# reading as sunk half a body-width. Ride the group instead — the
# shipped min-seating did this by accident and read correct. The
# same rule seats a handstand on its hands (2-joint dense group far
# below the wrist/arm consensus) instead of pressing it down.
TILT_GROUP_GAP_LO_M = 0.06
TILT_GROUP_GAP_HI_M = 0.12
# Seat handoff sharpening. The iso endpoint exists for TUMBLE safety
# (w ~ 0); a mid-fade resting weight means a grounded TRANSITION
# (get-up / settle, 40-90 deg/s), where the seat must stay at the
# resting value: blending linearly on w slid the seat toward iso
# exactly while a get-up's captured hips dip to load the push —
# the render rose while the body dipped (the reported float).
TILT_SEAT_W_LO = 0.02
TILT_SEAT_W_HI = 0.15

# --- Arm depth trust (deepest-point seating) ---
# Arm depth is the least reliable channel in a monocular capture: a
# planted arm can read 20 cm below the floor the rest of the body
# establishes (measured on an inverted arm-plant move: forearm chain
# at -19..-21 while legs/head sat within a few cm of the floor).
# Seating that arm hoists the WHOLE body by the error, with a lurch
# when the seat hands from one arm to the other. Arms may therefore
# pull the seat only ARM_TRUST_M below the deepest non-arm joint —
# but only when some non-arm joint is itself near the floor. When
# NOTHING but arms is anywhere near the floor (a handstand, an elbow
# stand), the arms keep full authority: real arm support rests AT
# the floor, never under it, so legitimate breakdance holds are
# untouched by construction.
ARM_CHAIN_JOINTS = ("LeftArm", "RightArm", "LeftForeArm",
                    "RightForeArm", "LeftHand", "RightHand")
ARM_TRUST_M = 0.04
ARM_SOLO_M = 0.12


def _smoothstep(x, lo, hi):
    """0 at x<=lo, 1 at x>=hi, smooth in between. Vectorized."""
    t = np.clip((x - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _axis_angle_mat(axis, angle):
    """Rotation matrix from a unit axis and angle (Rodrigues)."""
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * K @ K


def _mat_to_rotvec(R):
    """Rotation matrix -> (rotation vector, angle). Angles here stay
    far below pi (capped at max_tilt_deg), so the sin form is safe."""
    cosa = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosa))
    if angle < 1e-8:
        return np.zeros(3), 0.0
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]])
    axis /= max(2.0 * np.sin(angle), 1e-12)
    return axis * angle, angle


def compute_body_tilt_alignment(world_pos, names, fps,
                                back_radius=0.015, max_tilt_deg=25.0):
    """Support-plane alignment for RESTING grounded poses: per-frame
    rotation (as rotation vectors) + pivots that level the body's
    underside, so the translation pass can seat the whole contact
    polygon instead of pinning the single deepest joint.

    Monocular depth tilts lying bodies (measured on a real kneel-
    collapse-lie take: right knee 15 cm through the floor while the
    left knee and hands hovered ~16 cm — a roll error, not just
    pitch). Translation-only grounding then pins the deepest joint
    and the rest of the body ramps out of the floor. A rigid pitch/
    roll rotation about the contact centroid fixes exactly this class
    of error — the user's manual fix, automated.

    Gates, each against a specific failure mode:
    - Posture gate (shared with the translation pass): upright motion
      and jumps can never be touched.
    - RESTING gate: body angular velocity low. A roll's contact plane
      legitimately rotates — aligning it would fight the tumble. Two
      angular signals are combined: the torso axis direction (misses
      spins about that axis) and the shoulder lateral axis (catches
      b-twists), so tumbling of either kind zeroes the correction.
    - Support-area gate: the contact set must span a real polygon
      (second principal spread). A handstand's two hands define a
      line, not a plane — its tilt is balance, not error. No
      rotation without a plane.
    - Rotation cap: beyond max_tilt_deg it isn't a tilt error, it's a
      different pose. No correction rather than a violent one.

    The underside is found per frame from per-joint MEDIAN surface
    positions over a short window (frame noise can't nominate a
    joint), with one refit pass: generous band from the lowest
    median, fit, keep joints near the plane, fit again. The naive
    "near the lowest point" selection degenerates to a single joint
    on real takes (measured: 1 cm spread; the fit meant nothing).

    Leveling that least-squares plane is only the seed: contact
    points are never coplanar, so the level pose still leaves some
    contacts hovering while others press. A bounded refinement
    (TILT_REFINE_DEG around level, jointly with the seat height)
    then picks the placement that maximizes actual floor contact —
    the settled-rigid-body pose — under the hover/burial costs in
    the constants block above.

    Returns (rotvec (n,3), pivot (n,3), resting_w (n,)): axis-angle
    per frame (zero where no correction applies), contact-centroid
    pivots, and the smoothed resting-gate weight. The translation
    pass consumes resting_w to switch its seating rule on resting
    frames: sit on the support-plane CONSENSUS instead of the single
    deepest joint (a capture that buries one knee 14 cm below the
    rest of the body must not levitate the whole body off the floor
    to un-bury it — the outlier stays under the floor, hidden below
    the body, and everything else actually lies down).
    Apply BEFORE the translation pass, re-FK, then translate.
    """
    n_frames = world_pos.shape[0]
    rotvec = np.zeros((n_frames, 3), dtype=np.float64)
    pivot = np.zeros((n_frames, 3), dtype=np.float64)
    resting_w = np.zeros(n_frames, dtype=np.float64)
    if n_frames < 8:
        return rotvec, pivot, resting_w

    all_radii = dict(CONTACT_RADII)
    all_radii.update({n: back_radius for n in BACK_JOINTS})
    name_to_idx = {n: i for i, n in enumerate(names)}
    cidx = [name_to_idx[n] for n in all_radii if n in name_to_idx]
    if (len(cidx) < 4 or "Hips" not in name_to_idx
            or "Spine3" not in name_to_idx):
        return rotvec, pivot, resting_w
    radii = np.array([all_radii[names[i]] for i in cidx])
    cpos = world_pos[:, cidx, :].copy()
    cpos[:, :, 1] -= radii[None, :]   # surface points
    flesh = np.minimum(TILT_FLESH_FRAC * radii, TILT_FLESH_CAP_M)

    # Refinement search tables: pitch/roll grid (world X/Z about the
    # contact centroid) and seat-height candidates. Only the world-Y
    # row of each grid rotation is needed to score heights; the full
    # matrix of the winner recomposes the applied rotation.
    ref = np.radians(np.arange(-TILT_REFINE_DEG,
                               TILT_REFINE_DEG + 1e-9,
                               TILT_REFINE_STEP_DEG))
    rx, rz = [a.ravel() for a in np.meshgrid(ref, ref, indexing="ij")]
    cx, sx, cz, sz = np.cos(rx), np.sin(rx), np.cos(rz), np.sin(rz)
    zeros = np.zeros_like(cx)
    ref_mats = np.stack([
        np.stack([cz, -sz, zeros], axis=1),
        np.stack([cx * sz, cx * cz, -sx], axis=1),
        np.stack([sx * sz, sx * cz, cx], axis=1)], axis=1)  # (G,3,3)
    yrows = ref_mats[:, 1, :]                               # (G,3)
    deltas = np.arange(-TILT_SEAT_M, TILT_SEAT_M + 1e-9, TILT_SEAT_STEP_M)
    # Tie-break regularizers (loss is meters of hover/burial; a full-
    # range deviation costs 2 mm — far below any real contact gain).
    reg_rot = 0.002 * ((rx ** 2 + rz ** 2)
                       / np.radians(TILT_REFINE_DEG) ** 2)
    reg_seat = 0.0005 * (deltas / TILT_SEAT_M) ** 2

    # Posture gate (same measure as the translation pass).
    torso = (world_pos[:, name_to_idx["Spine3"], :]
             - world_pos[:, name_to_idx["Hips"], :])
    tnorm = np.maximum(np.linalg.norm(torso, axis=1), 1e-9)
    tdir = torso / tnorm[:, None]
    tilt_deg = np.degrees(np.arccos(np.clip(tdir[:, 1], -1.0, 1.0)))
    w_posture = _smoothstep(tilt_deg, UPRIGHT_DEG, NONUPRIGHT_DEG)

    # Resting gate: max of two angular rates (deg/s).
    def _dir_rate(v):
        d = v / np.maximum(np.linalg.norm(v, axis=1), 1e-9)[:, None]
        r = np.degrees(np.arccos(np.clip((d[1:] * d[:-1]).sum(axis=1),
                                         -1.0, 1.0))) * fps
        return np.concatenate([r[:1], r])

    ang = _dir_rate(torso)
    if "LeftArm" in name_to_idx and "RightArm" in name_to_idx:
        lat = (world_pos[:, name_to_idx["LeftArm"], :]
               - world_pos[:, name_to_idx["RightArm"], :])
        ang = np.maximum(ang, _dir_rate(lat))
    ang = gaussian_filter1d(ang, sigma=max(0.15 * fps, 1.0))
    w_rest = 1.0 - _smoothstep(ang, TILT_REST_DEG_S_LO, TILT_REST_DEG_S_HI)
    gate = w_posture * w_rest
    resting_w = gaussian_filter1d(gate, sigma=max(GATE_SIGMA_S * fps, 1.0))
    resting_w = np.clip(resting_w, 0.0, 1.0)

    win = max(int(round(0.35 * fps)), 3)
    up = np.array([0.0, 1.0, 0.0])
    for f in range(n_frames):
        if gate[f] < 0.05:
            continue
        lo, hi = max(0, f - win), min(n_frames, f + win + 1)
        med = np.median(cpos[lo:hi], axis=0)     # (n_contacts, 3)
        base = med[:, 1].min()
        keep = med[:, 1] <= base + TILT_UNDERSIDE_BAND_M
        for _ in range(2):
            pts = med[keep]
            if len(pts) < 3:
                break
            c = pts.mean(axis=0)
            q = pts - c
            evals, evecs = np.linalg.eigh(q.T @ q / len(q))
            normal = evecs[:, 0]
            if normal[1] < 0:
                normal = -normal
            dist = np.abs((med - c) @ normal)
            keep = (dist <= TILT_PLANE_BAND_M) & (
                med[:, 1] <= base + TILT_UNDERSIDE_BAND_M)
        pts = med[keep]
        if len(pts) < 3:
            continue
        c = pts.mean(axis=0)
        q = pts - c
        evals, evecs = np.linalg.eigh(q.T @ q / len(q))
        # Support-area gate: spread along the SECOND principal
        # in-plane direction.
        if np.sqrt(max(evals[1], 0.0)) < TILT_MIN_SPREAD_M:
            continue
        normal = evecs[:, 0]
        if normal[1] < 0:
            normal = -normal
        angle = float(np.arccos(np.clip(normal @ up, -1.0, 1.0)))
        if np.degrees(angle) > max_tilt_deg:
            continue          # a different pose, not a tilt error
        axis = np.cross(normal, up)
        an = np.linalg.norm(axis)
        R_level = (_axis_angle_mat(axis / an, angle)
                   if an > 1e-9 else np.eye(3))

        # Contact refinement: search pitch/roll around the leveled
        # pose jointly with the seat height for the placement that
        # maximizes touching (see the constants block). Support
        # candidacy and burial allowances are judged in the LEVELED
        # frame — on a strongly tilted lie the high-side contacts sit
        # far above the raw low cluster and would otherwise be
        # invisible to the search. The consensus percentile keeps a
        # buried outlier from defining the cluster, and a tiny
        # deviation penalty settles flat-loss ties at level instead
        # of a grid corner.
        R = R_level
        lev = (med - c) @ R_level.T
        lev_y = lev[:, 1] + c[1]
        seat0 = np.percentile(lev_y, TILT_CONSENSUS_PCT)
        sup = lev_y <= seat0 + TILT_NEAR_M
        if sup.sum() >= 3:
            allow = np.maximum(flesh[sup],
                               np.maximum(0.0, seat0 - lev_y[sup])
                               + TILT_BURY_SLACK_M)
            hgt = lev[sup] @ yrows.T + c[1]                # (n_sup, G)
            seat = np.percentile(hgt, TILT_CONSENSUS_PCT, axis=0)
            resid = (hgt[:, :, None]
                     - (seat[None, :, None] + deltas[None, None, :]))
            loss = (np.maximum(resid, 0.0).sum(axis=0)
                    + TILT_BURY_W
                    * np.maximum(-resid - allow[:, None, None],
                                 0.0).sum(axis=0))
            loss += reg_rot[:, None] + reg_seat[None, :]
            R = ref_mats[int(np.argmin(loss.min(axis=1)))] @ R_level
        rv, angle = _mat_to_rotvec(R)
        if np.degrees(angle) < 1.0:
            continue
        if np.degrees(angle) > max_tilt_deg:
            rv *= np.radians(max_tilt_deg) / angle
        rotvec[f] = rv * gate[f]
        pivot[f] = c

    # Heavy temporal smoothing: the support configuration changes
    # slowly (kneel folding into a lie over a second or two); the
    # correction must glide with it, never step. Pivots are smoothed
    # only where a rotation exists, held at edges.
    applied = np.linalg.norm(rotvec, axis=1) > 1e-6
    if not applied.any():
        return rotvec, pivot, resting_w
    sig = max(TILT_SMOOTH_S * fps, 1.0)
    rotvec = gaussian_filter1d(rotvec, sigma=sig, axis=0)
    pv = pivot.copy()
    idxs = np.flatnonzero(applied)
    for k in range(3):
        pv[:, k] = np.interp(np.arange(n_frames), idxs, pivot[idxs, k])
    pivot = gaussian_filter1d(pv, sigma=sig, axis=0)

    deg = np.degrees(np.linalg.norm(rotvec, axis=1))
    print(f"       Body tilt alignment: {int((deg > 0.5).sum())} frames "
          f"leveled, max tilt correction {deg.max():.1f} deg")
    return rotvec, pivot, resting_w


def compute_body_grounding_correction(world_pos, names, fps,
                                      catch_distance=0.30, floor_y=None,
                                      back_radius=0.015, resting_w=None):
    """Per-frame Y delta (n_frames,) to add to the Hips Y channel.

    world_pos: (n_frames, n_joints, 3) FINAL FK world positions of the
        BVH skeleton, Y-up, meters — the joints as they will render.
    names: BVH joint name per index (same order as world_pos).
    catch_distance: maximum correction magnitude (m).
    floor_y: floor level. Pass 0.0 when the pipeline grounded the
        world (Feet Define the Floor: plants sit at clearance above
        y=0 by construction). None derives it from the final feet
        (5th percentile of the per-frame min foot surface) — a
        fallback that self-references on lie-heavy takes, where the
        capture-pressed lying feet ARE the low percentile.
    back_radius: joint-to-skin distance (m) for the back group (hips,
        spine chain, neck) - the Skin Radius slider.
    resting_w: optional (n_frames,) weight from the tilt-alignment
        pass. Where the body is RESTING, the seating rule blends from
        the single lowest surface to a contact-optimized CONSENSUS:
        a low percentile of the contact surfaces (a capture that
        buries one joint far below the rest must not levitate the
        body to un-bury it — the outlier stays under the floor,
        hidden below the body, and the body actually lies down),
        refined per frame so the support cluster actually touches
        instead of straddling a statistic. Rolls keep min-seating
        (their contact point IS the minimum; the resting gate is
        zero while tumbling).
    """
    n_frames = world_pos.shape[0]
    if n_frames < 4:
        return np.zeros(n_frames, dtype=np.float64)

    all_radii = dict(CONTACT_RADII)
    all_radii.update({n: back_radius for n in BACK_JOINTS})
    name_to_idx = {n: i for i, n in enumerate(names)}
    idx = [name_to_idx[n] for n in all_radii if n in name_to_idx]
    radii = np.array([all_radii[names[i]] for i in idx])
    if not idx or "Hips" not in name_to_idx or "Spine3" not in name_to_idx:
        return np.zeros(n_frames, dtype=np.float64)

    surface_y = world_pos[:, idx, 1] - radii[None, :]

    if floor_y is None:
        feet_idx = [name_to_idx[n] for n in FOOT_NAMES if n in name_to_idx]
        feet_surf = (world_pos[:, feet_idx, 1]
                     - np.array([CONTACT_RADII[names[i]]
                                 for i in feet_idx])[None, :])
        floor_y = float(np.percentile(feet_surf.min(axis=1), 5))

    lowest = surface_y.min(axis=1)
    # Arm trust cap (see the constants block): arms only define the
    # seat within ARM_TRUST_M of the body's own deepest evidence,
    # unless nothing but arms is near the floor.
    arm_mask = np.array([names[i] in ARM_CHAIN_JOINTS for i in idx])
    if arm_mask.any() and (~arm_mask).any():
        m_arm = surface_y[:, arm_mask].min(axis=1)
        m_rest = surface_y[:, ~arm_mask].min(axis=1)
        capped = np.minimum(m_rest, np.maximum(m_arm,
                                               m_rest - ARM_TRUST_M))
        lowest = np.where(m_rest <= float(floor_y) + ARM_SOLO_M,
                          capped, lowest)

    if resting_w is not None and np.max(resting_w) > 1e-3:
        consensus = np.percentile(surface_y, TILT_CONSENSUS_PCT, axis=1)
        # Contact-seat refinement (closed loop, measured on the FINAL
        # post-rotation FK): shift the consensus so the support
        # cluster actually TOUCHES — hover costs, burial is free up
        # to flesh or the capture's own burial (+ slack), deeper
        # costs more (see the constants block). The percentile alone
        # seats the body at a statistic; this seats it at contact.
        flesh = np.minimum(TILT_FLESH_FRAC * radii, TILT_FLESH_CAP_M)
        deltas = np.arange(-TILT_SEAT_M, TILT_SEAT_M + 1e-9,
                           TILT_SEAT_STEP_M)
        for f in np.flatnonzero(resting_w > 0.05):
            sy = surface_y[f]
            sup = sy <= consensus[f] + TILT_NEAR_M
            if sup.sum() < 3:
                continue
            allow = np.maximum(flesh[sup],
                               np.maximum(0.0, consensus[f] - sy[sup])
                               + TILT_BURY_SLACK_M)
            resid = sy[sup, None] - (consensus[f] + deltas)[None, :]
            loss = (np.maximum(resid, 0.0).sum(axis=0)
                    + TILT_BURY_W
                    * np.maximum(-resid - allow[:, None],
                                 0.0).sum(axis=0))
            consensus[f] += deltas[int(np.argmin(loss))]
        # Blend floor: the (1-w) side uses the deepest joint that is
        # not a LONE outlier (see TILT_ISO_BAND_M) — the resting-gate
        # fade hands the seat to the real support, never to a lone
        # capture-buried joint (the early-rise hoist). The isolation
        # walk runs on short windowed MEDIANS: raw per-frame
        # companionship flickers during a get-up (measured 5 cm seat
        # steps), while a long window remembers a dissolving lying
        # cluster past the point the body has left it (measured
        # 3.6 cm float at the push-off).
        m = resting_w > 1e-3
        win = max(int(round(TILT_ISO_WIN_S * fps)), 2)
        iso = np.empty(n_frames, dtype=np.float64)
        for f in np.flatnonzero(m):
            lo, hi = max(0, f - win), min(n_frames, f + win + 1)
            v = np.sort(np.median(surface_y[lo:hi], axis=0))
            k = 0
            while k < len(v) - 1 and (v[k + 1] - v[k]) > TILT_ISO_BAND_M:
                k += 1
            iso[f] = v[k]
        # Smooth iso within each resting run (never across gaps).
        sig = max(TILT_ISO_SMOOTH_S * fps, 1.0)
        i = 0
        midx = np.flatnonzero(m)
        while i < len(midx):
            j = i
            while j + 1 < len(midx) and midx[j + 1] == midx[j] + 1:
                j += 1
            run = midx[i:j + 1]
            if len(run) > 2:
                iso[run] = gaussian_filter1d(iso[run], sigma=sig)
            i = j + 1
        # RESTING seat: consensus for a normal lie, the dense deep
        # group where one exists far below it (see the constants
        # block — side-lies rest ON that group).
        grp_w = _smoothstep(consensus[m] - iso[m],
                            TILT_GROUP_GAP_LO_M, TILT_GROUP_GAP_HI_M)
        rest_seat = consensus[m] + grp_w * (iso[m] - consensus[m])
        w_seat = _smoothstep(resting_w[m], TILT_SEAT_W_LO, TILT_SEAT_W_HI)
        lowest[m] = iso[m] * (1.0 - w_seat) + rest_seat * w_seat

    hover = lowest - float(floor_y)

    # Posture gate: torso axis angle from world-up (+Y).
    torso = (world_pos[:, name_to_idx["Spine3"], :]
             - world_pos[:, name_to_idx["Hips"], :])
    norm = np.linalg.norm(torso, axis=1)
    norm[norm < 1e-9] = 1e-9
    cos_up = np.clip(torso[:, 1] / norm, -1.0, 1.0)
    tilt_deg = np.degrees(np.arccos(cos_up))
    w_posture = _smoothstep(tilt_deg, UPRIGHT_DEG, NONUPRIGHT_DEG)

    # Ballistic protection: genuine flight accelerates at g; drift just
    # hovers. Curvature via sliding parabola fit (savgol deriv=2 —
    # exact on a true arc; double-gradient of a smoothed signal
    # underestimates short flights). The fit is only clean where its
    # window sits inside the flight, so the detected core is dilated
    # across the window margin; the contact-zone fade comes after so
    # protection still ends where landing begins (entering a roll is
    # ALSO a genuine fall and must still ground).
    hips_y = world_pos[:, name_to_idx["Hips"], 1]
    win = int(round(0.35 * fps))
    win = max(win + 1 - win % 2, 7)
    if n_frames >= win:
        accel = savgol_filter(hips_y, win, polyorder=2, deriv=2,
                              delta=1.0 / fps)
    else:
        accel = np.zeros(n_frames)
    ballistic = 1.0 - _smoothstep(np.abs(accel + 9.81), 4.0, 8.0)
    ballistic = maximum_filter1d(ballistic, size=win)
    ballistic *= _smoothstep(hover, 0.25, 0.40)
    w_ballistic = gaussian_filter1d(ballistic, sigma=1.5)

    weight = gaussian_filter1d(w_posture * (1.0 - w_ballistic),
                               sigma=max(GATE_SIGMA_S * fps, 1.0))
    weight = np.clip(weight, 0.0, 1.0)

    if weight.max() < 1e-3:
        print("       Body grounding: no non-upright stretches found — "
              "no correction applied")
        return np.zeros(n_frames, dtype=np.float64)

    # Per-frame contact grounding, lightly smoothed...
    correction = -np.clip(hover, -catch_distance, catch_distance)
    correction = gaussian_filter1d(correction,
                                   sigma=max(CORRECTION_SIGMA_S * fps, 1.0))
    correction *= weight
    # ...with a near-contact clamp guaranteeing touch: smoothing
    # dilutes the pull exactly where the hover spikes between contacts
    # (a roll's back-over moment). Only near contact altitude — a high
    # frame with a partially-open gate (a dive's descent) must not be
    # yanked the full distance.
    residual = hover + correction
    adjust = np.clip(residual, -0.02, 0.03) - residual
    w_clamp = weight * (1.0 - _smoothstep(hover, 0.20, 0.30))
    correction = np.clip(correction + adjust * w_clamp,
                         -catch_distance, catch_distance)

    applied = np.abs(correction) > 0.005
    if applied.any():
        n_runs = int(np.diff(applied.astype(int)).clip(min=0).sum()
                     + applied[0])
        print(f"       Body grounding: {int(applied.sum())} frames in "
              f"{n_runs} stretch(es), max pull "
              f"{np.abs(correction).max()*100:.1f} cm, floor at "
              f"{floor_y*100:+.1f} cm (catch distance "
              f"{catch_distance*100:.0f} cm)")
    else:
        print("       Body grounding: no non-upright near-floor "
              "stretches found — no correction applied")
    return correction
