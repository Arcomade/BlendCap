"""Per-foot footskate cleanup: Y-only ankle lock with knee damping.

Pass 2 of the foot-grounding pipeline. Inspired by Kovar 2002
("Footskate Cleanup for Motion Capture Editing") but adapted for
SAM-derived monocular tracking data, where the canonical XYZ ankle
lock over-constrains the foot's natural motion during plants.

What this module does, frame-by-frame, during each detected foot plant:
  - Target the ankle to: (current frame's ankle X, floor_y, current
    frame's ankle Z). Y is locked, X and Z follow the source.
  - Solve analytic 2-bone IK on the leg (UpLeg + Leg) to put the
    ankle there, with Kovar-style knee damping near full extension.
  - Compensate the foot bone's local rotation so its WORLD rotation
    is preserved (heel-toe roll passes through unchanged).

Why Y-only lock instead of full XYZ lock (the previous version):
  When the body shifts forward over a planted foot in monocular video,
  the leg should travel from diagonal → vertical → diagonal. The
  ankle's X/Z position changes during this — that's the natural
  ankle-over-bottom-of-foot arc. A full XYZ lock pins the ankle to
  one place and forces the leg to compensate, distorting the leg
  geometry and causing visible artifacts ("looks like an extra step
  back," "leg stays diagonal when it should be vertical"). Y-only
  lock leaves the ankle's horizontal trajectory alone (= source
  motion) and only enforces the floor contact. Body-over-foot motion
  passes through unchanged.

XZ pinning (ON by default; --no-foot-pin disables):
  Deliberately re-introduces the horizontal constraint, per plant,
  with the safeguards the old always-on XYZ lock lacked. Anchors,
  drift, and release all live in the frame the VIEWER sees: this
  stage runs on the in-place skeleton (Root X/Z locomotion is added
  after cleanup), so with --root-motion the ankle series must first
  add the locomotion back (locomotion_xz) — otherwise a truly
  planted walking foot reads as sliding backward at body speed and
  a pinned foot would be glued to the BODY frame and skate with the
  Root. Each fully-covered region (stance + bridged gaps) gets one
  anchor — the median viewer-frame ankle X/Z over the region — and
  its frames lerp toward it by --foot-pin-strength, ramped through
  the same blend windows the IK weight uses. Before any frame is touched the WHOLE region is
  checked: if the source ankle drifts farther than --foot-pin-release
  from the anchor anywhere in it, the region is released outright
  (retroactive whole-plant verdict — the bake is offline, so a
  misdetected real slide comes out fully intact instead of sticking
  then catching up). Released regions and pin-off keep the Y-only
  behavior above exactly, so the worst case of a misfired pin is
  today's shipped output. If the historical "leg stays diagonal"
  artifact resurfaces on a clip, the strength slider (not code) is
  the first dial.

Why knee damping is still needed:
  Even with Y-only lock, when Pass 1's body grounding pulls the body
  down during a deep stance, the leg can be forced to extend further
  to keep the foot Y at the floor. Without damping, near full
  extension a small Y change requires a large knee-angle change,
  manifesting as a "knee pop." Kovar's f(x) function smoothly limits
  knee rotation as θ → π. Below ρ=160°, f=1 (no damping); above ρ,
  smoothly tapers to 0 at π.

Bridging close-adjacent plants:
  When two plants of the same foot are within 2*BLEND_FRAMES of each
  other (e.g., a brief mis-detected gap mid-stance), the linear-blend
  in/out at the boundaries would partially restore the original FK
  rotations in the middle of the gap — making the foot briefly rise
  back toward its uneven pre-Pass-2 position. To avoid that, we mark
  short gaps as fully covered (weight=1.0) and let the per-frame X/Z
  target carry the foot smoothly across.

Simplifications kept from earlier revs:
  - No leg stretching (BVH OFFSETs are static; we'd need per-frame
    position channels). At max-reach the ankle clamps and the foot
    sits a cm or two above floor — preferred over knee popping.
  - No phase-3 root reachability (Pass 1 handles body Y).
  - No heel/ball distinction; just the ankle is constrained, with
    foot rotation pass-through preserving heel-toe roll.

Floor target Y: rest-pose ankle Y, averaged left/right. Same value
for every plant, every frame — that's what fixes "uneven feet in low
squat" by making both planted ankles target the same height.
"""

import numpy as np

from .foot_grounding import detect_stances_per_foot
from .bvh_math import quats_xyzw_to_rotmats
from .tracking_repair import _rotmats_to_quats_xyzw, _slerp_quats_xyzw


# ============================================================
# TUNABLES
# ============================================================

# Frames over which the IK adjustment fades into adjacent unconstrained
# frames (Kovar's L4). At 24fps, 4 frames ≈ 0.17s — long enough to hide
# the FK→IK transition, short enough not to lag.
BLEND_FRAMES = 4

# Knee-damping threshold (Kovar's ρ). Above this interior knee angle,
# damping kicks in to prevent the abrupt straightening near θ=π that
# manifests as a knee pop. 160° matches Kovar's recommendation.
KNEE_DAMP_RHO_RAD = np.radians(160.0)

# Recover Glitches horizontal-displacement threshold. When Recover
# Glitches is on, a gap is bridged only if the foot didn't move much
# horizontally between the two adjacent plants — distinguishing a
# tracking misread (foot stayed put, looked like it lifted) from a
# real fast step (foot moved to a new position). 15cm catches typical
# misreads while leaving real steps unmerged. Not exposed as a slider
# (yet); add one if real-world clips need per-clip tuning.
RECOVER_HORIZ_THRESHOLD_M = 0.15


# Per-leg bone-name spec. Resolved against the pruned skeleton's
# `names` list at runtime.
_LEG_CHAINS = (
    {'side': 'L', 'upper': 'LeftUpLeg', 'lower': 'LeftLeg',
     'foot': 'LeftFoot', 'foot_y_idx': 0},
    {'side': 'R', 'upper': 'RightUpLeg', 'lower': 'RightLeg',
     'foot': 'RightFoot', 'foot_y_idx': 1},
)


# ============================================================
# FK helpers (BVH frame: rest = identity, globals = cumulative locals)
# ============================================================

def _local_to_global_bvh(parents, local_rots_frame):
    """Compose local rotations into BVH-frame globals for one frame.

    BVH convention has rest = identity (compute_local_rotations applies
    R0 normalization), so the joint's world rotation is just the
    cumulative composition of locals along the parent chain.
    """
    n_joints = local_rots_frame.shape[0]
    g = np.empty_like(local_rots_frame)
    g[0] = local_rots_frame[0]
    for j in range(1, n_joints):
        p = parents[j]
        g[j] = g[p] @ local_rots_frame[j]
    return g


def _fk_world_positions_bvh(parents, bvh_offsets, globals_frame, root_world):
    """Forward kinematics with a runtime override of the root's world
    position (Hips position channel, post-Pass-1).
    """
    n_joints = len(parents)
    pos = np.zeros((n_joints, 3))
    pos[0] = root_world
    for j in range(1, n_joints):
        p = parents[j]
        pos[j] = pos[p] + globals_frame[p] @ bvh_offsets[j]
    return pos


def _world_rest_positions(parents, bvh_offsets):
    """World rest position of each joint = cumulative bvh_offset sum
    along the parent chain. bvh_offsets[0] for the root is its own
    world rest position (per prune_skeleton convention)."""
    n = len(parents)
    pos = np.zeros((n, 3))
    pos[0] = bvh_offsets[0]
    for j in range(1, n):
        pos[j] = pos[parents[j]] + bvh_offsets[j]
    return pos


def _rotation_between(v1, v2):
    """Minimum rotation matrix taking direction v1 to direction v2.

    Returns identity if either input is too short or they're already
    parallel; returns a 180° rotation around an arbitrary perpendicular
    axis if they're antiparallel.
    """
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return np.eye(3)
    a = v1 / n1
    b = v2 / n2
    cos_t = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if cos_t > 1.0 - 1e-9:
        return np.eye(3)
    if cos_t < -1.0 + 1e-9:
        perp = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, np.array([0.0, 1.0, 0.0]))
        perp = perp / np.linalg.norm(perp)
        K = np.array([[0, -perp[2], perp[1]],
                      [perp[2], 0, -perp[0]],
                      [-perp[1], perp[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)
    axis = np.cross(a, b)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(cos_t)
    s = np.sin(angle)
    c = np.cos(angle)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _slerp_rotmat(R_a, R_b, t):
    """Slerp between two rotation matrices via quaternions. t=0 → A, t=1 → B."""
    qa = _rotmats_to_quats_xyzw(R_a[None])[0]
    qb = _rotmats_to_quats_xyzw(R_b[None])[0]
    q = _slerp_quats_xyzw(qa, qb, float(t))
    return quats_xyzw_to_rotmats(q[None])[0]


# ============================================================
# Knee damping (Kovar Section 4.4)
# ============================================================

def _alpha_blend(t):
    """Cubic Hermite blend: α(0)=1, α(1)=0, derivatives zero at endpoints.
    Used both for knee damping and for blending IK adjustments out into
    adjacent unconstrained frames."""
    return 2.0 * t**3 - 3.0 * t**2 + 1.0


def _alpha_integral(u):
    """Antiderivative of α(u) = 2u³−3u²+1. ∫α du = u⁴/2 − u³ + u."""
    return u**4 / 2.0 - u**3 + u


def _knee_damp_integral(a, b, rho=KNEE_DAMP_RHO_RAD, pi_=np.pi):
    """∫[a to b] f(x) dx, where f is Kovar's knee-damping function:
        f(x) = 1                  for x < ρ
        f(x) = α((x-ρ)/(π-ρ))     for ρ ≤ x ≤ π
        f(x) = 0                  for x > π
    Always returns a non-negative value (caller signs it for direction).
    """
    if a >= b:
        return 0.0
    const_part = max(0.0, min(b, rho) - max(a, 0.0))
    damped_lo = max(a, rho)
    damped_hi = min(b, pi_)
    damped_part = 0.0
    if damped_hi > damped_lo:
        u_lo = (damped_lo - rho) / (pi_ - rho)
        u_hi = (damped_hi - rho) / (pi_ - rho)
        damped_part = (pi_ - rho) * (_alpha_integral(u_hi) - _alpha_integral(u_lo))
    return const_part + damped_part


def _damped_knee_angle(theta_0, delta_theta):
    """Apply Kovar damping. Reduces the magnitude of the angle change
    when the leg is near full extension (above ρ), in either direction —
    physically, the joint resists big changes near full extension whether
    straightening or folding.
    """
    if delta_theta == 0:
        return theta_0
    target = theta_0 + delta_theta
    if delta_theta > 0:
        return theta_0 + _knee_damp_integral(theta_0, target)
    return theta_0 - _knee_damp_integral(target, theta_0)


# ============================================================
# 2-bone IK with knee damping
# ============================================================

def _solve_2bone_ik_kovar(hip_world, target_world,
                           rest_to_knee, rest_to_ankle,
                           pole_dir, current_knee_angle,
                           R_upper_old_world, R_lower_old_world):
    """Analytic 2-bone IK with Kovar knee damping.

    Returns (R_upper_world, R_lower_world, actual_ankle_world):
    - R_upper_world / R_lower_world: BVH-frame world rotations for the
      upper (hip→knee) and lower (knee→ankle) bones.
    - actual_ankle_world: where the ankle ends up in world space. May
      differ from target_world if the leg is at full reach (we don't
      stretch bones) or if the knee damping reduced the effective angle.

    Bone twist preservation: instead of rebuilding the leg's world
    rotation from rest with `_rotation_between(rest_dir, new_dir)`,
    which has zero twist about the bone axis by construction and
    therefore strips SAM's tracked bone twist (knee bend plane,
    pronation, etc.), we compute the SWING DELTA — the minimum rotation
    from the bone's CURRENT world direction to the new desired
    direction — and left-multiply it onto SAM's current world rotation.
    Foot still lands at the target (the bone now points the right way),
    but everything else SAM tracked about the bone is preserved. Avoids
    visible "leg twist on plant entry/exit" that the rest-rebuild path
    produced when SAM's bend plane disagreed with the rest-pose plane.
    """
    l1 = float(np.linalg.norm(rest_to_knee))
    l2 = float(np.linalg.norm(rest_to_ankle))

    delta = target_world - hip_world
    d_target = float(np.linalg.norm(delta))

    eps = 1e-4
    d_max = (l1 + l2) - eps
    d_min = abs(l1 - l2) + eps

    # Clamp target distance to the reachable annulus. Without leg
    # stretching, this is the best we can do — visually the foot will
    # sit at max-reach instead of at the unreachable target.
    if d_target > d_max:
        d_target = d_max
    if d_target < d_min:
        d_target = d_min

    cos_theta_target = (l1*l1 + l2*l2 - d_target*d_target) / (2.0 * l1 * l2)
    cos_theta_target = float(np.clip(cos_theta_target, -1.0, 1.0))
    theta_target = float(np.arccos(cos_theta_target))

    delta_theta = theta_target - current_knee_angle
    theta_damped = _damped_knee_angle(current_knee_angle, delta_theta)

    cos_theta_damped = float(np.cos(theta_damped))
    d_actual = float(np.sqrt(max(0.0, l1*l1 + l2*l2 - 2.0*l1*l2*cos_theta_damped)))

    if np.linalg.norm(delta) < 1e-6:
        forward = np.array([0.0, -1.0, 0.0])
    else:
        forward = delta / np.linalg.norm(delta)

    actual_ankle = hip_world + d_actual * forward

    pole_proj = pole_dir - np.dot(pole_dir, forward) * forward
    pole_norm = float(np.linalg.norm(pole_proj))
    if pole_norm < 1e-6:
        for fb in (np.array([0.0, 0.0, -1.0]),
                   np.array([1.0, 0.0, 0.0]),
                   np.array([0.0, 1.0, 0.0])):
            cand = fb - np.dot(fb, forward) * forward
            if np.linalg.norm(cand) > 1e-6:
                pole_proj = cand
                break
    pole_proj = pole_proj / np.linalg.norm(pole_proj)

    cos_hip = (l1*l1 + d_actual*d_actual - l2*l2) / (2.0 * l1 * max(d_actual, 1e-9))
    cos_hip = float(np.clip(cos_hip, -1.0, 1.0))
    sin_hip = float(np.sqrt(max(0.0, 1.0 - cos_hip*cos_hip)))

    knee_world = hip_world + l1 * (cos_hip * forward + sin_hip * pole_proj)

    upper_world_dir = knee_world - hip_world
    lower_world_dir = actual_ankle - knee_world

    # Swing-delta IK: redirect each bone with the minimum rotation
    # taking its CURRENT world direction to the new desired direction,
    # applied to SAM's current world rotation. Preserves SAM's bone
    # twist about the new direction.
    rk_norm = float(np.linalg.norm(rest_to_knee))
    ra_norm = float(np.linalg.norm(rest_to_ankle))
    if rk_norm < 1e-9 or ra_norm < 1e-9:
        # Degenerate rest geometry — fall back to rest-rebuild so we
        # still produce a valid rotation rather than blowing up.
        R_upper_world = _rotation_between(rest_to_knee, upper_world_dir)
        R_lower_world = _rotation_between(rest_to_ankle, lower_world_dir)
        return R_upper_world, R_lower_world, actual_ankle

    upper_old_dir = R_upper_old_world @ (rest_to_knee / rk_norm)
    lower_old_dir = R_lower_old_world @ (rest_to_ankle / ra_norm)

    swing_upper = _rotation_between(upper_old_dir, upper_world_dir)
    swing_lower = _rotation_between(lower_old_dir, lower_world_dir)

    R_upper_world = swing_upper @ R_upper_old_world
    R_lower_world = swing_lower @ R_lower_old_world
    return R_upper_world, R_lower_world, actual_ankle


# ============================================================
# Per-frame target / weight builder (Y-only lock + gap bridging)
# ============================================================

def _build_per_frame_target_weight(stances, ankle_world_seq, foot_y_series,
                                    floor_y, n_frames,
                                    max_gap_frames, max_lift_m,
                                    recover_glitches, recovery_lift_m,
                                    pin_feet=False, pin_strength=1.0,
                                    pin_release_m=0.20,
                                    locomotion_xz=None):
    """For one leg, build (n_frames, 3) target and (n_frames,) weight.

    Y-only lock: target = (current X, floor_y, current Z) per frame.
    With pin_feet, covered regions additionally pull X/Z toward a
    per-region anchor (see step 3 below); pin_feet=False leaves the
    targets byte-identical to the Y-only build.
    Plant frames get weight=1.0; gaps between adjacent plants get
    bridged (also weight=1.0) under either condition:

      Time + height: gap_frames ≤ max_gap_frames  AND
                     max(foot_y in gap) - floor_y ≤ max_lift_m
                     (catches tiny detection blips with low foot lift)

      Recover Glitches: max(foot_y in gap) - floor_y ≤ recovery_lift_m
                        AND foot didn't move much horizontally
                        (catches longer SAM misreads where the foot
                        stayed put but velocity blipped)

    Other near-plant frames get cubic-blended weight in [0, 1].

    Returns (target, weight, n_bridged, pin_info) where pin_info is a
    per-covered-region list of (pinned: bool, max_drift_m) — empty
    when pin_feet is off.
    """
    target = np.zeros((n_frames, 3))
    weight = np.zeros(n_frames)
    if not stances:
        return target, weight, 0, []

    sorted_stances = sorted(stances, key=lambda s: s[0])

    # Mark which frames are "fully covered" (weight=1.0).
    # Plant frames first, then bridge gaps between adjacent plants.
    covered = np.zeros(n_frames, dtype=bool)
    for s_start, s_end in sorted_stances:
        covered[s_start:s_end + 1] = True

    n_bridged = 0
    for i in range(len(sorted_stances) - 1):
        a_start, a_end = sorted_stances[i]
        b_start, b_end = sorted_stances[i + 1]
        gap_size = b_start - a_end - 1
        if gap_size <= 0:
            continue
        # Foot lift in the gap, used by both bridging conditions.
        gap_max_lift = (
            float(np.max(foot_y_series[a_end + 1:b_start])) - floor_y
        )
        # Time+height: short gap with low foot lift.
        if gap_size <= max_gap_frames and gap_max_lift <= max_lift_m:
            covered[a_end + 1:b_start] = True
            n_bridged += 1
            continue
        # Recover Glitches: any gap, low foot lift, low horizontal motion.
        if recover_glitches and gap_max_lift <= recovery_lift_m:
            # Horizontal displacement between A's last frame and B's
            # first frame (X/Z only — Y is floor-locked).
            a_xz = ankle_world_seq[a_end][[0, 2]]
            b_xz = ankle_world_seq[b_start][[0, 2]]
            horiz_disp = float(np.linalg.norm(b_xz - a_xz))
            if horiz_disp <= RECOVER_HORIZ_THRESHOLD_M:
                covered[a_end + 1:b_start] = True
                n_bridged += 1

    # Step 1: full-weight Y-only-lock targets on covered frames.
    for f in range(n_frames):
        if covered[f]:
            target[f] = ankle_world_seq[f].copy()
            target[f, 1] = floor_y
            weight[f] = 1.0

    # Step 2: cubic blends on each side of each covered region.
    # Find covered regions (contiguous runs of True).
    regions = []
    i = 0
    while i < n_frames:
        if covered[i]:
            j = i
            while j < n_frames and covered[j]:
                j += 1
            regions.append((i, j - 1))
            i = j
        else:
            i += 1

    # owner[f] = index of the region whose blend claimed frame f (via
    # the strict `w > weight[f]` precedence below, ties keep the
    # earlier region). Only read by the pin step; unused when pin_feet
    # is off.
    owner = np.full(n_frames, -1, dtype=np.int64)
    for ri, (r_start, r_end) in enumerate(regions):
        # Backward blend (frames before r_start)
        for k in range(1, BLEND_FRAMES + 1):
            f = r_start - k
            if f < 0 or covered[f]:
                break
            w = _alpha_blend(k / (BLEND_FRAMES + 1.0))
            if w > weight[f]:
                target[f] = ankle_world_seq[f].copy()
                target[f, 1] = floor_y
                weight[f] = w
                owner[f] = ri
        # Forward blend (frames after r_end)
        for k in range(1, BLEND_FRAMES + 1):
            f = r_end + k
            if f >= n_frames or covered[f]:
                break
            w = _alpha_blend(k / (BLEND_FRAMES + 1.0))
            if w > weight[f]:
                target[f] = ankle_world_seq[f].copy()
                target[f, 1] = floor_y
                weight[f] = w
                owner[f] = ri

    # Step 3: optional XZ pinning (on by default; --no-foot-pin). One anchor per covered
    # region (the stance plus its bridged gaps, so the foot holds
    # through phantom lifts). The release is a WHOLE-REGION verdict
    # made before any frame is touched — the bake is offline, so a
    # misdetected real slide is left fully intact rather than pinned
    # until it breaches. Released regions keep the Y-only targets
    # built above (today's shipped behavior).
    pin_info = []
    if pin_feet:
        s = float(np.clip(pin_strength, 0.0, 1.0))
        # Viewer-frame ankle XZ: this stage's skeleton is in-place
        # (Root locomotion is added AFTER cleanup), so with
        # --root-motion the locomotion must be added back before
        # anchoring — otherwise a planted walking foot reads as
        # sliding backward at body speed (release fires on every
        # plant) and a pinned foot skates along with the Root.
        # Targets convert back to the cleanup frame per frame below.
        if locomotion_xz is None:
            world_xz = ankle_world_seq[:, (0, 2)]
        else:
            world_xz = ankle_world_seq[:, (0, 2)] + locomotion_xz
        for ri, (r_start, r_end) in enumerate(regions):
            seg_xz = world_xz[r_start:r_end + 1]
            anchor = np.median(seg_xz, axis=0)
            drift = np.hypot(seg_xz[:, 0] - anchor[0],
                             seg_xz[:, 1] - anchor[1])
            max_drift = float(drift.max())
            if max_drift > pin_release_m:
                pin_info.append((False, max_drift))
                continue
            pin_info.append((True, max_drift))
            # Covered frames: full-strength lerp toward the anchor.
            # target currently holds the cleanup-frame ankle, so the
            # viewer-frame delta applied to it lands the ankle at
            # (anchor - locomotion) in cleanup space = the anchor on
            # screen.
            for f in range(r_start, r_end + 1):
                target[f, 0] += s * (anchor[0] - world_xz[f, 0])
                target[f, 2] += s * (anchor[1] - world_xz[f, 1])
            # Blend frames this region owns: scale the pin by the
            # frame's cubic IK weight so the target ramps source ->
            # anchor in step with the rotation blend (a constant-
            # anchor target at low weight would pop on entry/exit).
            for f in range(max(0, r_start - BLEND_FRAMES), r_start):
                if owner[f] == ri:
                    t = s * weight[f]
                    target[f, 0] += t * (anchor[0] - world_xz[f, 0])
                    target[f, 2] += t * (anchor[1] - world_xz[f, 1])
            for f in range(r_end + 1,
                           min(n_frames, r_end + 1 + BLEND_FRAMES)):
                if owner[f] == ri:
                    t = s * weight[f]
                    target[f, 0] += t * (anchor[0] - world_xz[f, 0])
                    target[f, 2] += t * (anchor[1] - world_xz[f, 1])

    return target, weight, n_bridged, pin_info


# ============================================================
# Main entrypoint
# ============================================================

def apply_footskate_cleanup(parents, bvh_offsets, local_rots, hips_pos,
                             names, joint_coords, translation, fps,
                             sensitivity=0.5, max_gap_sec=0.33,
                             max_lift_m=0.10, recover_glitches=False,
                             recovery_lift_m=0.15, pin_feet=False,
                             pin_strength=1.0, pin_release_m=0.20,
                             pin_locomotion_xz=None,
                             robust_floor=False,
                             local_height_base=False,
                             progress_range=None):
    """Modify `local_rots` in place: replace leg rotations during
    detected foot plants with Y-only-locked 2-bone IK + knee damping.

    parents:           (K,) parent indices in the pruned skeleton.
    bvh_offsets:       (K, 3) world-aligned rest deltas (post-prune).
    local_rots:        (n_frames, K, 3, 3) post-Savitzky-Golay rotations.
                       MODIFIED IN PLACE.
    hips_pos:          (n_frames, 3) Hips world position with Pass 1
                       body grounding already applied.
    names:             (K,) bone names (post-prune index order).
    joint_coords:      (n_frames, n_sam_joints, 3) — fed to Pass 1's
                       detect_stances_per_foot for stance lists.
    translation:       (n_frames, 3) — same as Pass 1.
    fps:               float.
    sensitivity:       0..1, mapped to (V, H, MIN) thresholds in
                       foot_grounding.sensitivity_to_thresholds.
    max_gap_sec:       seconds. Adjacent plants closer than this in
                       time are considered for time+height bridging.
    max_lift_m:        meters. Time+height bridging requires the foot
                       to stay below this height above floor in the gap.
    recover_glitches:  if True, also bridge longer gaps where the foot
                       stayed below recovery_lift_m AND didn't move much
                       horizontally (catches longer SAM misreads).
    recovery_lift_m:   meters. Lift cap for the Recover Glitches path.
    pin_feet:          if True, pin each covered region's ankle X/Z to
                       its median anchor (see module docstring / step 3
                       of _build_per_frame_target_weight). False keeps
                       the Y-only targets byte-identical.
    pin_strength:      0..1 lerp toward the anchor on pinned frames.
    pin_release_m:     meters. Whole-region release verdict: a region
                       whose source ankle drifts farther than this from
                       the anchor is left unpinned (real slide).
    pin_locomotion_xz: (n_frames, 2) X/Z the synthetic Root will add
                       AFTER cleanup (i.e. translation[:, (0, 2)] when
                       --root-motion is set), or None when the output
                       is in-place. Pin anchors/release measure drift
                       in the viewer's frame — see module docstring.
    robust_floor:      passed through to detect_stances_per_foot so the
                       stance detector's height cap matches Pass 1's
                       (see foot_grounding.ROBUST_FLOOR_PERCENTILE).
    local_height_base: passed through to detect_stances_per_foot;
                       set by Feet-Define-the-Floor mode so this
                       pass's stances match the ones that built the
                       vertical channel.
    progress_range:    optional (lo, hi) percentage span to emit
                       `[PROGRESS pct] msg` markers into. If None, no
                       progress markers are emitted (back-compat for
                       direct callers / tests).
    """
    def _emit(local_frac, msg):
        if not progress_range:
            return
        lo, hi = progress_range
        pct = lo + max(0.0, min(1.0, local_frac)) * (hi - lo)
        print(f"[PROGRESS {int(pct)}] {msg}", flush=True)

    n_frames = local_rots.shape[0]
    if n_frames < 2:
        return

    left_stances, right_stances, foot_y = detect_stances_per_foot(
        joint_coords, translation, fps, sensitivity=sensitivity,
        robust_floor=robust_floor, local_height_base=local_height_base)
    if not left_stances and not right_stances:
        print("       Footskate cleanup: no plants detected, skipping")
        return

    # Convert max_gap_sec to frames using the take's fps so the
    # threshold is framerate-independent.
    max_gap_frames = max(1, int(round(max_gap_sec * max(fps, 1.0))))

    chains = []
    for ch in _LEG_CHAINS:
        try:
            up_idx = names.index(ch['upper'])
            lo_idx = names.index(ch['lower'])
            ft_idx = names.index(ch['foot'])
        except ValueError:
            print(f"       Footskate cleanup: skipping {ch['side']} chain "
                  f"({ch['upper']}/{ch['lower']}/{ch['foot']} "
                  f"not in skeleton)")
            continue
        chains.append({
            'side': ch['side'],
            'up_idx': up_idx, 'lo_idx': lo_idx, 'ft_idx': ft_idx,
            'foot_y_idx': ch['foot_y_idx'],
            'stances': left_stances if ch['side'] == 'L' else right_stances,
        })
    if not chains:
        return

    # Floor target Y: rest-pose ankle Y in world, averaged L+R. Same
    # for every plant, every frame — fixes uneven feet during simul-
    # taneous plants by giving both ankles the same Y target.
    world_rest = _world_rest_positions(parents, bvh_offsets)
    rest_ankle_ys = [world_rest[c['ft_idx']][1] for c in chains]
    floor_y = float(np.mean(rest_ankle_ys))

    rg_str = "ON" if recover_glitches else "off"
    print(f"       Footskate cleanup: {len(left_stances)} L + "
          f"{len(right_stances)} R plants, "
          f"floor_target_y={floor_y*100:+.1f}cm "
          f"(sens={sensitivity:.2f}, max_gap={max_gap_sec:.2f}s/"
          f"{max_gap_frames}f, max_lift={max_lift_m*100:.0f}cm, "
          f"recover_glitches={rg_str}, "
          f"recovery_lift={recovery_lift_m*100:.0f}cm, "
          f"pin={'on' if pin_feet else 'off'})")

    n_modified = 0
    # Total per-frame passes for progress accounting: each active leg
    # chain runs a pre-compute pass and an IK-apply pass.
    active_chains = [c for c in chains if c['stances']]
    n_passes = max(1, len(active_chains) * 2)
    pass_idx = 0
    # Tick at most ~20 times per per-frame loop to keep stdout chatter low.
    tick_every = max(1, n_frames // 20)

    for ch in chains:
        if not ch['stances']:
            continue

        # Pre-compute ankle world positions for this leg using the
        # CURRENT (pre-IK) local_rots. The X/Z components are used as
        # the per-frame target — Y-only lock means we don't average
        # or fix the horizontal position; we follow the source's
        # natural ankle trajectory and only enforce floor Y.
        ankle_world_seq = np.zeros((n_frames, 3))
        for f in range(n_frames):
            globals_f = _local_to_global_bvh(parents, local_rots[f])
            wpos = _fk_world_positions_bvh(
                parents, bvh_offsets, globals_f, hips_pos[f])
            ankle_world_seq[f] = wpos[ch['ft_idx']]
            if f % tick_every == 0:
                _emit((pass_idx + (f / max(n_frames, 1))) / n_passes,
                      f"Footskate {ch['side']}: FK pre-compute")
        pass_idx += 1

        # Per-leg foot Y series (smoothed bottom-of-foot from Pass 1's
        # detection — used for the gap-lift checks during bridging).
        leg_foot_y = foot_y[:, ch['foot_y_idx']]

        target, weight, n_bridged, pin_info = \
            _build_per_frame_target_weight(
                ch['stances'], ankle_world_seq, leg_foot_y, floor_y,
                n_frames, max_gap_frames, max_lift_m,
                recover_glitches, recovery_lift_m,
                pin_feet=pin_feet, pin_strength=pin_strength,
                pin_release_m=pin_release_m,
                locomotion_xz=pin_locomotion_xz)

        sorted_stances = sorted(ch['stances'], key=lambda s: s[0])
        plant_log = ", ".join(f"f={s}..{e}" for s, e in sorted_stances)
        bridged_log = f"; bridged {n_bridged} gap(s)" if n_bridged else ""
        print(f"         {ch['side']} plants: {plant_log}{bridged_log}")
        if pin_feet:
            n_pinned = sum(1 for p, _ in pin_info if p)
            n_released = len(pin_info) - n_pinned
            detail = ", ".join(
                f"{'pinned' if p else 'released'} {d*100:.1f}cm"
                for p, d in pin_info)
            print(f"         {ch['side']} pin ({n_pinned} pinned, "
                  f"{n_released} released, release="
                  f"{pin_release_m*100:.0f}cm): {detail}")

        # Apply IK + slerp blend per frame.
        for f in range(n_frames):
            if f % tick_every == 0:
                _emit((pass_idx + (f / max(n_frames, 1))) / n_passes,
                      f"Footskate {ch['side']}: IK apply")
            w = float(weight[f])
            if w <= 0.0:
                continue

            globals_f = _local_to_global_bvh(parents, local_rots[f])
            world_pos_f = _fk_world_positions_bvh(
                parents, bvh_offsets, globals_f, hips_pos[f])
            hip_world = world_pos_f[ch['up_idx']]
            current_knee = world_pos_f[ch['lo_idx']]
            current_ankle = world_pos_f[ch['ft_idx']]

            v_knee_to_hip = hip_world - current_knee
            v_knee_to_ankle = current_ankle - current_knee
            l1_curr = float(np.linalg.norm(v_knee_to_hip))
            l2_curr = float(np.linalg.norm(v_knee_to_ankle))
            if l1_curr < 1e-6 or l2_curr < 1e-6:
                continue
            cos_knee = float(np.clip(
                np.dot(v_knee_to_hip, v_knee_to_ankle) / (l1_curr * l2_curr),
                -1.0, 1.0))
            current_knee_angle = float(np.arccos(cos_knee))

            target_ankle = target[f]

            # Pole = current knee position relative to hip. With Y-only
            # lock the bend plane changes smoothly across frames (X/Z
            # of target tracks source per-frame) so the pole projection
            # stays stable.
            pole_dir = current_knee - hip_world

            R_upper_world, R_lower_world, _ = _solve_2bone_ik_kovar(
                hip_world, target_ankle,
                bvh_offsets[ch['lo_idx']], bvh_offsets[ch['ft_idx']],
                pole_dir, current_knee_angle,
                globals_f[ch['up_idx']], globals_f[ch['lo_idx']])

            hip_parent_idx = parents[ch['up_idx']]
            R_hip_global = globals_f[hip_parent_idx]
            R_upper_local_new = R_hip_global.T @ R_upper_world
            R_lower_local_new = R_upper_world.T @ R_lower_world

            # Foot compensation: preserve the foot bone's WORLD rotation
            # across the leg-rotation change (keeps heel-toe roll).
            R_lower_world_old = globals_f[ch['lo_idx']]
            local_foot_old = local_rots[f, ch['ft_idx']]
            R_foot_local_new = (R_lower_world.T @ R_lower_world_old
                                @ local_foot_old)

            if w >= 1.0:
                local_rots[f, ch['up_idx']] = R_upper_local_new
                local_rots[f, ch['lo_idx']] = R_lower_local_new
                local_rots[f, ch['ft_idx']] = R_foot_local_new
            else:
                local_rots[f, ch['up_idx']] = _slerp_rotmat(
                    local_rots[f, ch['up_idx']], R_upper_local_new, w)
                local_rots[f, ch['lo_idx']] = _slerp_rotmat(
                    local_rots[f, ch['lo_idx']], R_lower_local_new, w)
                local_rots[f, ch['ft_idx']] = _slerp_rotmat(
                    local_rots[f, ch['ft_idx']], R_foot_local_new, w)
            n_modified += 1
        pass_idx += 1

    print(f"       Footskate cleanup: modified {n_modified} frame-foot pairs")
