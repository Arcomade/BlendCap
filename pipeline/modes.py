# modes.py
# ========
# Per-export-mode handlers. Each `run_*` function loads its own inputs,
# runs the pipeline, writes the BVH, and returns a small summary dict.
# The entry point (npz_to_bvh.py) only does argparse, dispatch, and a
# trailing summary print.

import json
import os
import sys

import numpy as np


def _progress(pct, msg):
    """Emit a `[PROGRESS pct] msg` marker for the BlendCap addon to parse.

    The Blender addon's run_bvh_thread (blendcap/mocap.py) reads stdout
    line by line and feeds these into the in-panel progress bar.
    Stays useful in plain CLI runs too — just shows up as a print line.
    """
    print(f"[PROGRESS {int(max(0, min(100, pct)))}] {msg}", flush=True)

from .body_skeleton import (
    FINGER_JOINTS,
    KEPT_ORDER,
    assign_names,
    boost_finger_rotations,
    insert_synthetic_root,
    load_skeleton,
    prune_skeleton,
)
from .bvh_math import (
    compute_local_rotations,
    compute_tpose_globals,
    compute_world_positions,
    smooth_positions,
    smooth_rotations,
    seconds_to_savgol_window,
    seconds_to_sigma_frames,
)
from .bvh_writer import write_bvh
from .camera_level import apply_correction as apply_camera_level
from .camera_level import resolve_correction as resolve_camera_level
from .camera_level import apply_head_correction_local as apply_head_local
from .camera_level import resolve_head_correction as resolve_head_level
from .body_grounding import compute_body_grounding_correction
from .foot_grounding import (compute_feet_defined_vertical,
                             compute_per_stance_correction)
from .footskate_cleanup import apply_footskate_cleanup
from .body_depth_denoise import apply_body_depth_denoise
from .head_rotation_denoise import apply_head_rotation_denoise
from .face_export import (
    AVERAGE_IPD_METERS,
    compute_face_bone_data,
    compute_head_local_landmarks,
    face_ipd_for_head_width,
    first_detected,
)
from .skeleton_builders import (
    build_face_only_skeleton,
    build_hands_only_skeleton,
    build_root_only_skeleton,
    compute_hands_local_rots,
)
from .tracking_repair import clean_tracking_failures


def _repair_nonfinite_frames(arrays, what):
    """Replace frames containing NaN/Inf with the nearest earlier finite
    frame (leading bad frames copy the first finite one).

    Runs unconditionally at load — unlike the opt-in --clean-tracking
    glitch repair — because a single non-finite frame from a dropped
    detection otherwise poisons every downstream stage silently: Savgol
    smoothing dies inside np.linalg.svd, and the grounding min() drags
    the whole hips channel (and therefore the written BVH) to literal
    nan values.

    `arrays` share frame indices, so one combined mask keeps them
    consistent. Returns (repaired arrays, bad-frame count); exits with
    an actionable error when every frame is bad."""
    n = arrays[0].shape[0]
    bad = np.zeros(n, dtype=bool)
    for a in arrays:
        bad |= ~np.isfinite(a.reshape(n, -1)).all(axis=1)
    if not bad.any():
        return arrays, 0
    if bad.all():
        print(f"ERROR: every frame of the {what} is non-finite (NaN/Inf).")
        print("       The capture output is unusable - re-run tracking.")
        sys.exit(1)
    good_idx = np.where(~bad)[0]
    prev = np.searchsorted(good_idx, np.arange(n), side='right') - 1
    src = good_idx[np.clip(prev, 0, len(good_idx) - 1)]
    repl = np.where(bad, src, np.arange(n))
    return [a[repl] for a in arrays], int(bad.sum())


# ---- run_body stage helpers ----------------------------------------
# run_body() is a sequence of pipeline stages; each stage below was
# extracted verbatim from the original single-function body. Inputs and
# outputs are explicit so the orchestrator reads as the data flow.

# Foot-grounding constants (shared by _ground_rest_pose and
# _solve_vertical_motion). SAM's natural T-pose has the body floating
# ~5 cm above world origin and the actor's frame-0 pose typically sits
# even higher; Rigify metarigs (and most authored rigs) put the feet
# ~1 cm above the ground plane.
FLOOR_CLEARANCE_M = 0.01
FOOT_SAM_INDICES = [5, 7, 21, 23]  # LeftFoot, LeftToe, RightFoot, RightToe


def _load_body_inputs(args, export_mode, parents, offsets_cm,
                      prerot_quats, n_joints):
    """Load the body NPZ (or synthesize the 1-frame T-pose template).

    Returns (npz, global_rots, joint_coords, fps, n_frames); npz is None
    in template mode."""
    npz = None
    if export_mode == "template":
        # Synthesize a 1-frame T-pose: global_rots = T-pose globals,
        # joint_coords = T-pose world positions in cm. With this input,
        # compute_local_rotations() yields identity for every bone,
        # so the BVH animation channel at frame 0 is exactly the
        # rest pose — perfect for the user to pose in Blender and
        # export back as a custom-rest preset.
        #
        # joint_coords from save_mhr_data.py is stored Y/Z-flipped
        # relative to compute_world_positions output (the existing
        # body path applies the same flip downstream, see "Y/Z flipped
        # to Y-up" in the cam_t branch). We pre-flip here so the
        # downstream flip yields the correct un-flipped values and
        # centering becomes a no-op.
        print("[2/3] Synthesizing 1-frame T-pose template (no NPZ "
              "required)")
        T = compute_tpose_globals(parents, prerot_quats)
        tpose_world = compute_world_positions(parents, offsets_cm, T)
        tpose_world_flipped = tpose_world.copy()
        tpose_world_flipped[:, 1] *= -1
        tpose_world_flipped[:, 2] *= -1
        global_rots = T[None]                            # (1, J, 3, 3)
        joint_coords = tpose_world_flipped[None]         # (1, J, 3)
        fps = args.fps_override if args.fps_override > 0 else 30.0
    else:
        _progress(5, "Loading tracking data")
        print(f"[2/3] Loading {args.input}")
        npz = np.load(args.input)

        global_rots = npz["global_rots"].astype(np.float64)
        joint_coords = npz["joint_coords"].astype(np.float64)
        (global_rots, joint_coords), _n_bad = _repair_nonfinite_frames(
            [global_rots, joint_coords], "body tracking data")
        if _n_bad:
            print(f"[WARNING] Repaired {_n_bad} invalid (NaN) tracking frames")
        fps = float(npz["fps"])
        if args.fps_override > 0:
            fps = args.fps_override

    n_frames = global_rots.shape[0]
    print(f"       {n_frames} frames, {fps:.1f} FPS")
    assert global_rots.shape[1] == n_joints
    return npz, global_rots, joint_coords, fps, n_frames


def _clean_tracking_stage(args, export_mode, parents, prerot_quats,
                          global_rots, joint_coords):
    """Optional --clean-tracking glitch repair. Returns the (possibly
    replaced) arrays plus the flagged frame ranges for the sidecar."""
    flagged_runs = []
    if args.clean_tracking and export_mode != "template":
        if prerot_quats is None:
            print("       Clean tracking: skipping (no prerotations "
                  "available for rest-pose reference)")
        else:
            _progress(8, "Cleaning tracking failures")
            R0_ref = compute_tpose_globals(parents, prerot_quats)
            global_rots, joint_coords, flagged_runs = \
                clean_tracking_failures(global_rots, joint_coords,
                                        parents, R0_ref)
    return global_rots, joint_coords, flagged_runs


def merged_take_cameras(folder, n_frames, cam_dirs=None):
    """For a take merged from several cameras (pipeline/
    two_camera_merge): each camera's own capture, cut to the merged
    take's frames, in the form hand_solver.solve_hands_merged takes.

    folder: the merged take's folder (calibration.npz, and the cameras'
    captures in cam_a, cam_b, ... unless cam_dirs = {'a': dir, 'b':
    dir, ...} says otherwise). Returns (cameras, None) or (None, why
    not)."""
    calib_path = os.path.join(folder, "calibration.npz")
    if not os.path.isfile(calib_path):
        return None, "no calibration next to the merged capture"
    cal = np.load(calib_path)
    GA = np.asarray(cal["GA"], dtype=np.float64)
    flip = np.diag([1.0, -1.0, -1.0])
    level = flip @ GA @ flip          # camera A's frame -> the merged take's
    if "turns" in cal.files:
        turns = np.asarray(cal["turns"], dtype=np.float64)
        starts = [int(s) for s in cal["starts"]]
        shifts = np.asarray(cal["sync_shifts"], dtype=np.float64)
        arms = {"right": np.asarray(cal["arm_right"], dtype=np.float64),
                "left": np.asarray(cal["arm_left"], dtype=np.float64)}
    else:
        # merged before more than two cameras could be: camera B's
        # values under their own names
        o = int(cal["sync_offset"])   # frames camera A leads camera B by
        turns = np.stack([np.eye(3), np.asarray(cal["R"], dtype=np.float64)])
        starts = [max(o, 0), max(-o, 0)]
        shifts = np.zeros((2, n_frames))
        if "sync_shift" in cal.files and len(cal["sync_shift"]) == n_frames:
            shifts[1] = np.asarray(cal["sync_shift"], dtype=np.float64)
        arms = None
        if "arm_w_right" in cal.files:
            arms = {s: np.stack([np.zeros(len(cal[k])), np.asarray(
                cal[k], dtype=np.float64)])
                for s, k in (("right", "arm_w_right"),
                             ("left", "arm_w_left"))}
    # how far the merge trusted each camera for each arm, frame by
    # frame (takes merged before this was recorded: every camera
    # counts alike)
    if arms is not None and arms["right"].shape[1] != n_frames:
        arms = None
    views = "abcd"[:len(turns)]
    cameras = []
    for k, view in enumerate(views):
        sub = os.path.join((cam_dirs or {}).get(view)
                           or os.path.join(folder, f"cam_{view}"),
                           "mhr_frames.npz")
        if not os.path.isfile(sub):
            return None, f"camera {view.upper()}'s capture was not found"
        z = np.load(sub)
        if "keypoints_2d" not in z.files:
            return None, f"camera {view.upper()}'s capture has no 2D keypoints"
        cut = slice(starts[k], None)
        arrs = [z[key][cut][:n_frames].astype(np.float64)
                for key in ("global_rots", "joint_coords", "cam_t")]
        kp2d = z["keypoints_2d"][cut][:n_frames]
        if any(len(a) != n_frames for a in arrs) or len(kp2d) != n_frames:
            return None, (f"camera {view.upper()}'s capture does not cover "
                          f"the merged take's frames")
        (g, jc, ct), _n_bad = _repair_nonfinite_frames(
            arrs, f"camera {view.upper()} tracking data")
        if k > 0 and shifts.shape[1] == n_frames and np.any(shifts[k]):
            # the merge read this camera between its own frames (fine
            # sync); its keypoints have to be read at the same moments
            from .two_camera_merge import resample_frames
            t_k = np.arange(n_frames) - shifts[k]
            g = resample_frames(g, t_k, "rotation")
            jc, ct = resample_frames(jc, t_k), resample_frames(ct, t_k)
            kp2d = resample_frames(kp2d.astype(np.float64), t_k)
        share = None
        if arms is not None:
            # camera A has what the others leave
            share = {s: (w[k] if k > 0 else 1.0 - w[1:].sum(axis=0))
                     for s, w in arms.items()}
        cameras.append({"label": view.upper(), "global_rots": g,
                        "keypoints_2d": kp2d, "joint_coords": jc,
                        "cam_t": ct,
                        "to_merged": (level if k == 0
                                      else level @ flip @ turns[k] @ flip),
                        "share": share})
    return cameras, None


def _solve_hands_stage(args, export_mode, npz, parents, offsets_cm,
                       prerot_quats, global_rots):
    """Optional 2D-keypoint hand refinement (--hand-solver on).

    Runs in the RAW camera frame — before any camera-level/offset
    correction — because the 2D keypoints live there. Mutates
    global_rots finger joints in place. Returns True when any frame
    was solved (the finger boost is skipped then: solved chains are
    absolute fits, amplifying them would break the poses).

    A take merged from several cameras is fitted in EACH camera's picture
    and the fits are merged (hand_solver.solve_hands_merged): the
    merged take's own keypoints are camera A's alone, and its frame is
    no longer the one any picture was taken in."""
    if getattr(args, "hand_solver", "off") != "on":
        return False
    if export_mode == "template" or npz is None:
        return False
    if "keypoints_2d" not in npz:
        print("[WARNING] Hand solver skipped - NPZ has no 2D keypoints")
        return False
    if prerot_quats is None:
        print("[WARNING] Hand solver skipped - model lacks prerotations")
        return False
    kp2d = npz["keypoints_2d"]
    if kp2d.shape[0] != global_rots.shape[0]:
        print("[WARNING] Hand solver skipped - keypoint frame mismatch")
        return False
    _progress(12, "Refitting hands to 2D keypoints")
    print("       Hand solver: fitting fingers to 2D keypoints...")
    from .hand_solver import solve_hands
    R0t = compute_tpose_globals(parents, prerot_quats)

    # Drive the progress bar across the per-frame solve (12->30), the
    # longest part of this step, with a live frame counter in the
    # status text (the bar can hold on the same integer pct while the
    # counter keeps ticking, so the user sees it working). Downstream
    # stages start above 30, so this band never runs the bar backward.
    def _hand_progress(done, total, detail):
        pct = 12 + int(round(max(0.0, min(1.0, done / max(total, 1))) * 18))
        _progress(pct, f"Rebuilding finger motion - {detail}")

    if "gravity_aligned" in getattr(npz, "files", ()):
        cameras, why = merged_take_cameras(
            os.path.dirname(os.path.abspath(args.input)),
            global_rots.shape[0])
        if cameras is not None:
            print("       Hand solver: multi-camera take, fitting each "
                  "camera's picture and merging")
            from .hand_solver import solve_hands_merged
            summary = solve_hands_merged(global_rots, cameras, parents,
                                         offsets_cm, R0t,
                                         progress_cb=_hand_progress)
            return bool(summary.get("any_solved"))
        print(f"       Hand solver: multi-camera take, but {why}; using "
              f"the keypoints kept with the merged take (camera A's)")

    summary = solve_hands(global_rots, kp2d, parents, offsets_cm, R0t,
                          progress_cb=_hand_progress)
    return bool(summary.get("any_solved"))


# Largest camera pitch the detrend below is allowed to believe. The
# regression cannot tell a tilted camera from an actor whose height
# happens to correlate with their depth — walking toward the camera
# while crouching reads identically to a camera pointed down. Measured
# on a close-range two-camera take: both cameras were level to 3 deg by
# their own floor plane, the regression claimed 27 and -35 deg, and
# applying it turned a floor flat to 7 cm into one varying by 22-37 cm.
# A camera framing a standing figure sits within a few degrees of
# level; past this cap the fit is describing the performance, not the
# camera, so trust the footage and leave the vertical alone.
MAX_DETREND_PITCH_DEG = 12.0


def _detrend_camera_pitch(cam_t, npz=None):
    """Remove the depth-correlated component of cam_t.y, in place.

    SAM 3D Body's pred_cam_t is in CAMERA frame, not world frame. For
    any camera pitch (up/down tilt about world X), pred_cam_t.y picks
    up a linear-in-Z component of slope -tan(pitch). Feeding that into
    the BVH as world Y bleeds depth motion into apparent vertical
    motion: as the actor walks toward the camera the body appears to
    rise or sink. Subtracting the regression of cam_t.y on cam_t.z
    removes it. Vertical motion NOT correlated with depth passes
    through unchanged; vertical motion that IS correlated with depth
    gets absorbed, which is why MAX_DETREND_PITCH_DEG exists.

    Returns a status line for the caller to print.
    """
    # A merged take the merge could level. (One it could not, a take
    # with no floor to measure, carries 0 and is in camera A's frame:
    # the same case as a single camera.)
    if npz is not None and "gravity_aligned" in getattr(npz, "files", ()) \
            and int(npz["gravity_aligned"]):
        return ("Camera pitch detrend: skipped (input is already "
                "gravity-aligned)")
    if cam_t.shape[0] < 2 or float(cam_t[:, 2].var()) <= 1e-6:
        return ("Camera pitch detrend: skipped "
                "(insufficient depth motion variance)")
    z_c = cam_t[:, 2] - cam_t[:, 2].mean()
    y_c = cam_t[:, 1] - cam_t[:, 1].mean()
    slope = float((y_c * z_c).sum() / (z_c * z_c).sum())
    pitch = abs(np.degrees(np.arctan(slope)))
    if pitch > MAX_DETREND_PITCH_DEG:
        return (f"Camera pitch detrend: skipped (fit implies {pitch:.0f} deg "
                f"of camera pitch, over the {MAX_DETREND_PITCH_DEG:.0f} deg "
                f"limit, treating it as real vertical motion)")
    cam_t[:, 1] = cam_t[:, 1] - slope * cam_t[:, 2]
    return (f"Camera pitch detrend: removed {slope:+.4f} * cam_t.z from "
            f"cam_t.y ({pitch:.1f} deg of depth-induced Y bleed)")


def _hold_cropped_limbs_stage(args, export_mode, npz, parents,
                              prerot_quats, global_rots, fps):
    """Optional --hold-cropped-limbs (see pipeline/cropped_limbs.py).

    Runs AFTER the hand solver (whose finger fits are overwritten on
    frames where that hand is held) and BEFORE the camera offset (the
    hold works in parent-relative space, which the offset leaves
    alone). Returns (global_rots, report or None); the report carries
    valid_feet, feet_never_seen and hips_held for the later stages."""
    if not getattr(args, 'hold_cropped_limbs', False):
        return global_rots, None
    if export_mode == "template" or npz is None:
        return global_rots, None
    if "keypoints_2d" not in npz.files or "frame_size" not in npz.files:
        print("[WARNING] Hold Cropped Limbs skipped - this capture predates "
              "the feature (no picture size in the tracking data). Run the "
              "capture again to use it.")
        return global_rots, None
    kp2d = npz["keypoints_2d"]
    if kp2d.shape[0] != global_rots.shape[0]:
        print("[WARNING] Hold Cropped Limbs skipped - keypoint frame mismatch")
        return global_rots, None
    _progress(13, "Holding cropped limbs")
    from .cropped_limbs import hold_cropped_limbs
    global_rots, rep = hold_cropped_limbs(global_rots, kp2d, npz["frame_size"],
                                          parents, prerot_quats, fps)
    for line in rep["lines"]:
        print(line)
    if rep["feet_never_seen"]:
        print("       Hold Cropped Limbs: the feet are never in the picture; "
              "foot locking and grounding have nothing to work with and "
              "stand down; the body keeps a fixed standing height")
    return global_rots, rep


def _load_cam_t(npz, args, export_mode):
    """Load + repair cam_t and run the camera-pitch detrend.

    Always attempted (not gated by --root-motion): cam_t carries the
    floor-reference world translation (X/Z locomotion plus the
    camera-Y component of the body root). Vertical body motion is
    sourced from cam_t + jc_hips later; --root-motion only gates the
    X/Z locomotion components. Returns cam_t or None."""
    cam_t = None
    if export_mode != "template":
        if "cam_t" in npz:
            cam_t = npz["cam_t"].astype(np.float64)
            (cam_t,), _n_bad = _repair_nonfinite_frames([cam_t], "cam_t data")
            if _n_bad:
                print(f"[WARNING] Repaired {_n_bad} invalid (NaN) cam_t frames")
            print(f"       cam_t loaded ({cam_t.shape})")

            print(f"       {_detrend_camera_pitch(cam_t, npz)}")
        elif args.root_motion:
            print("WARNING: --root-motion set but 'cam_t' not found in NPZ")
            print("         Available keys:", list(npz.files))
            print("         Root will hold at rest (no vertical motion)")
        else:
            print("       cam_t not in NPZ; Root will hold at rest "
                  "(no vertical motion)")
    return cam_t


def _apply_camera_offset(args, export_mode, global_rots, joint_coords,
                         cam_t):
    """Manual camera-angle offset (UI pitch/roll sliders).

    R left-multiplies global_rots ONLY — joint_coords and cam_t pass
    through apply_correction unchanged BY DESIGN, so foot grounding
    reads the original (un-rotated) positions. That matches the
    live-preview path (which only rewrites Root rotation keys) so baked
    and unbaked offsets agree exactly; see camera_level.apply_correction
    for the full rationale and the floor-reference trade-off.

    Rest-pose invariant: in BOTH frame0 and tpose modes the rest pose
    is sourced from data SAMPLED BEFORE correction (the
    R0_pre_correction snapshot returned here), so R0 (and the rest
    geometry derived from it) stays in the original raw orientation.
    This matches the per-rig live-preview path in
    blendcap.properties._apply_camera_offset_to_rig (which only
    rewrites Root's per-frame rotation keys, never touches rest) so
    baked and unbaked offsets are visually identical and retarget
    bakes see the same rest geometry either way."""
    R0_pre_correction = None
    if export_mode != "template" and args.rest_pose == "frame0":
        R0_pre_correction = global_rots[0].copy()
    if export_mode != "template":
        cam_pitch = getattr(args, 'camera_pitch', 0.0)
        cam_roll = getattr(args, 'camera_roll', 0.0)
        # Roll sign inverted to match the live preview's visible
        # direction. Live preview rotates about Blender world +Y (the
        # forward axis in Z-up; see _apply_camera_offset_to_rig),
        # while the bake here rotates about BVH MODEL +Z (the "forward
        # axis" in Y-up Z-back). Those two forward-axis vectors point
        # OPPOSITE ways (Blender's +Y is character-forward; BVH's +Z
        # is character-BACK), so the same positive roll value lands as
        # a right-tilt in preview but a left-tilt in the bake. Pitch
        # is unaffected because the side axis (+X) matches in both
        # frames. Same class of fix as the head-roll inversion in
        # run_body.
        R_cam = resolve_camera_level(pitch_deg=cam_pitch,
                                     roll_deg=-cam_roll)
        if R_cam is not None:
            print(f"       Camera angle offset: pitch={cam_pitch:+.2f}deg "
                  f"roll={cam_roll:+.2f}deg")
            global_rots, joint_coords, cam_t = apply_camera_level(
                global_rots, joint_coords, cam_t, R_cam)
    return R0_pre_correction, global_rots, joint_coords, cam_t


def _build_rest_pose(args, parents, offsets_cm, prerot_quats,
                     R0_pre_correction, global_rots):
    """Select R0 (T-pose or frame-0) and compute scaled rest positions."""
    # 31 (was 15): the hand solver now owns 12-30 when active; these
    # fast stages sit just above it so the bar never jumps backward.
    _progress(31, "Building rest pose")
    print("[3/3] Building rest pose and animation...")

    if args.rest_pose == "tpose":
        R0 = compute_tpose_globals(parents, prerot_quats)
        print(f"       Rest pose: T-pose (accumulated prerotations)")
    else:
        # Use the pre-correction snapshot so the rest pose stays in
        # the raw (uncorrected) orientation. global_rots[0] at this
        # point would carry the camera-level rotation, which would
        # bake the correction INTO rest geometry and break parity
        # with the per-rig live-preview offset that only touches
        # animation. Falls back to the live array for template /
        # other code paths that don't snapshot.
        R0 = R0_pre_correction if R0_pre_correction is not None \
            else global_rots[0]
        print(f"       Rest pose: frame 0")

    rest_pos_cm = compute_world_positions(parents, offsets_cm, R0)
    rest_pos = rest_pos_cm * args.scale
    if args.rest_pose == "tpose":
        y_range = rest_pos[:, 1].max() - rest_pos[:, 1].min()
        x_range = rest_pos[:, 0].max() - rest_pos[:, 0].min()
        print(f"       T-pose extent: X={x_range:.3f}m  Y={y_range:.3f}m "
              f"(wide X = arms out = good)")
    return R0, rest_pos


def _ground_rest_pose(rest_pos):
    """Rest grounding (first of the two grounding shifts).

    rest_shift_y: vertical translation of rest_pos (in place) so the
    lowest foot in T-pose lands at FLOOR_CLEARANCE_M. Modifies
    bvh_offsets[Hips] downstream (everything below Hips is
    parent-relative and shifts with it). The companion
    animation_shift_y — a constant Y added to Hips.LOC every frame so
    the lowest foot across the entire clip lands at FLOOR_CLEARANCE_M —
    is solved later in _solve_vertical_motion, AFTER translation is
    built, centered, smoothed, and combined with jc_hips, so the
    formula reflects what is actually rendered. The min-over-clip
    metric is robust to mid-jump start, mid-fall start, etc. — as long
    as the actor touches the ground at SOME point in the clip, that
    touchdown is the floor."""
    tpose_feet_y_min = float(rest_pos[FOOT_SAM_INDICES, 1].min())
    rest_shift_y = FLOOR_CLEARANCE_M - tpose_feet_y_min
    rest_pos[:, 1] += rest_shift_y
    print(f"       Rest grounding: T-pose feet "
          f"{tpose_feet_y_min*100:+.1f} cm -> "
          f"{FLOOR_CLEARANCE_M*100:.1f} cm "
          f"(shift {rest_shift_y*100:+.1f} cm)")


def _anchor_face_to_body(face_data, face_calibration, args, parents,
                         offsets_cm, global_rots, R0, rest_pos,
                         n_frames):
    """Face landmarks + bone data, anchored to the SAM eye joints, plus
    the body/face frame-count sync. A shorter face is padded here by
    holding its last frame. A LONGER face never extends the body
    arrays — held body frames poison the clip-wide foot/floor
    statistics and the footskate pin anchors (which rewrite knee IK on
    the real frames), so run_body pads its final output arrays just
    before write_bvh instead. Returns (face_bone_data, target_n);
    face_bone_data is None when no face data, and target_n ==
    n_frames unless the face runs longer."""
    face_bone_data = None
    target_n = n_frames
    if face_data is not None:
        # SAM eye positions in BVH Head-local space (constant).
        HEAD, EYE_L, EYE_R = 113, 122, 124
        LEFT_ARM, RIGHT_ARM = 75, 39   # acromion = "end of shoulder"
        G0 = global_rots[0]
        pos_f0_cm = compute_world_positions(parents, offsets_cm, G0)
        pos_f0 = pos_f0_cm * args.scale
        W_head_inv = R0[HEAD] @ G0[HEAD].T
        sam_eye_L = W_head_inv @ (pos_f0[EYE_L] - pos_f0[HEAD])
        sam_eye_R = W_head_inv @ (pos_f0[EYE_R] - pos_f0[HEAD])

        sam_mid = (sam_eye_L + sam_eye_R) / 2

        # Body-anchored face sizing: head width = HEAD_TO_SHOULDER_RATIO
        # * biacromial span (end-of-shoulder to end-of-shoulder, i.e.
        # the LeftArm <-> RightArm joints which sit at the acromia).
        # 0.45 (instead of the anthropometric ~1/3) compensates for SAM
        # rest poses that draw the shoulders pulled slightly back —
        # the rest-pose acromion span understates "shoulders" as the
        # eye perceives them, so a larger fraction yields a head that
        # reads correctly. face_ipd_for_head_width uses the
        # SOURCE rig's own head-width-to-IPD ratio to convert that
        # target head width back into the face_ipd_target the renderer
        # needs — the source rig's per-bone positions are preserved
        # 1:1 (retarget maps unchanged); only the uniform scale knob
        # is set so the rendered head ends up the right size relative
        # to the body. Falls through to the anatomical IPD if the
        # shoulder span is missing/degenerate.
        # rest_pos (not pos_f0) for shoulder measurement: stable
        # acromion-to-acromion distance regardless of frame-0 pose.
        HEAD_TO_SHOULDER_RATIO = 0.45
        shoulder_span = float(np.linalg.norm(
            rest_pos[LEFT_ARM] - rest_pos[RIGHT_ARM]))
        if shoulder_span > 1e-3 and face_calibration is not None:
            target_head_width = HEAD_TO_SHOULDER_RATIO * shoulder_span
            face_ipd_target = face_ipd_for_head_width(
                target_head_width, face_calibration)
            print(f"       Shoulder span: {shoulder_span*100:.1f} cm "
                  f"-> head width {target_head_width*100:.1f} cm "
                  f"({HEAD_TO_SHOULDER_RATIO:.2f}x) -> face IPD "
                  f"{face_ipd_target*100:.2f} cm")
        else:
            face_ipd_target = AVERAGE_IPD_METERS
            print(f"       Shoulder/calibration unavailable; "
                  f"face IPD target = {face_ipd_target*100:.2f} cm (anatomical)")

        _progress(32, "Computing head-local landmarks")
        print("       Computing head-local landmarks (eye anchors)...")
        head_local_lm, _ = compute_head_local_landmarks(
            face_data, eye_anchors=(sam_eye_L, sam_eye_R))

        # Compute rest face by applying L[0, HEAD] inverse to frame 0's
        # anchor output. This is the same FK rotation the animation
        # applies at frame 0 — inverting it gives the face shape as it
        # appears when the Head bone is at rest (forward-facing).
        rest_face = None
        first_det = first_detected(face_data["detected"])
        # Need this rest-face correction whenever R0 differs from frame-0
        # globals: tpose mode makes rest != frame 0.
        if args.rest_pose == "tpose":
            p_head = parents[HEAD]
            L_head_f0 = (R0[p_head] @ G0[p_head].T
                         @ G0[HEAD] @ R0[HEAD].T)
            face_f0 = head_local_lm[first_det]
            rest_face = (L_head_f0.T @ (face_f0 - sam_mid).T).T + sam_mid
            trace = np.clip(
                (np.trace(L_head_f0) - 1.0) / 2.0, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(trace))
            print(f"       Rest pose: L_head correction "
                  f"{angle_deg:.1f} deg from rest")

        _progress(34, "Computing face bone animation")
        print("       Computing face bone animation...")
        face_bone_data = compute_face_bone_data(
            face_data, head_local_lm,
            rest_face=rest_face, face_boost=args.face_boost,
            face_calibration=face_calibration,
            face_ipd_target=face_ipd_target)

        # Sync face frame count with body. Both streams play out to
        # the longer length by holding their last frame, so neither
        # cuts off early. Only the FACE is padded here. When the face
        # runs longer, the body is padded in run_body just before
        # write_bvh — extending it here fed the held frames into every
        # clip-wide solve (floor percentiles, stance detection,
        # footskate pin anchors, smoothing windows), which changed
        # knee IK on the REAL frames whenever the face NPZ came from
        # a longer clip.
        face_n = len(face_data["landmarks"])
        if face_n != n_frames:
            target_n = max(n_frames, face_n)
            if face_n < n_frames:
                print(f"WARNING: Body has {n_frames} frames, face has "
                      f"{face_n} frames - holding last face frame for "
                      f"the trailing {n_frames - face_n}.")
            else:
                print(f"WARNING: Body has {n_frames} frames, face has "
                      f"{face_n} frames - the body's processed last "
                      f"frame will be held for the trailing "
                      f"{face_n - n_frames}.")
            # Pad face to target by repeating its last frame. Empty-
            # face fallback preserves the prior rest-offset behavior
            # so a corrupt face NPZ doesn't crash; a normal face clip
            # always has at least one frame so the broadcast path
            # runs.
            for bone_name in face_bone_data["order"]:
                old = face_bone_data["frames"][bone_name]
                if old.shape[0] >= target_n:
                    continue
                pad_count = target_n - old.shape[0]
                if old.shape[0] == 0:
                    rest_row = np.zeros(6)
                    rest_row[:3] = face_bone_data["offsets"][bone_name]
                    pad_rows = np.broadcast_to(rest_row, (pad_count, 6))
                else:
                    pad_rows = np.broadcast_to(old[-1], (pad_count, 6))
                face_bone_data["frames"][bone_name] = np.concatenate(
                    [old, pad_rows], axis=0)
    return face_bone_data, target_n


def _build_translation(args, cam_t, n_frames, fps, depth_measured=False):
    """Per-frame translation vector (Y-up, centered, smoothed). Split
    into Root / Hips later: Root takes X/Z locomotion; Hips takes
    vertical body motion (squat / jump / gait bob).

    depth_measured: the take has no depth axis to distrust (a multi-
    camera merge: what was camera A's depth is measured sideways-on by
    another camera, and the merged root sits within a centimetre or
    two of the triangulated one on every axis). Root Smoothing then
    applies as the number on the slider to both floor axes, without
    the doubling a single camera's depth gets. With the doubling, the
    default moved a merged root up to 26 cm on that axis.

    Returns (translation, translation_raw): the smoothed series the
    pipeline renders from, and the pre-smoothing series kept as
    EVIDENCE for Feet Define the Floor's classification votes.
    Smoothing is an aesthetic control; a five-frame landing dip or a
    sharp takeoff rise is exactly what it erases, and detection
    decisions must not change when the user moves an output slider
    (corrections consume the render; classifications consume the
    evidence)."""
    if cam_t is not None:
        translation = cam_t.copy()
        translation[:, 1] *= -1   # SAM Y-down -> BVH Y-up
        translation[:, 2] *= -1   # SAM Z-forward -> BVH Z-back
    else:
        translation = np.zeros((n_frames, 3), dtype=np.float64)

    if not args.no_center:
        # Root rests at world origin; align frame 0 of cam_t to
        # (0, 0, 0) so locomotion starts at the origin.
        delta = translation[0].copy()
        if not np.allclose(delta, 0, atol=1e-6):
            translation -= delta
            print(f"       Centered: frame 0 -> origin")

    translation_raw = translation.copy()

    if args.smooth_root > 0:
        sigma = seconds_to_sigma_frames(args.smooth_root, fps)
        print(f"       Root smoothing: Gaussian sigma={sigma:.1f} frames "
              f"({args.smooth_root:.2f} s at {fps:.1f} fps, X/Z only, "
              + ("depth not doubled: multi-camera take)"
                 if depth_measured else "depth Z 2x)"))
        # Y (axis 1) is deliberately NOT smoothed here: a jump lives
        # almost entirely in cam_t.Y for only ~12-18 airborne frames, so
        # this Gaussian flattens the arc — and the flattened remainder
        # then reads as a floating plant to foot grounding, which pulls
        # it to the floor (net: jumps frozen at standing height). This
        # slider's job is locomotion/depth noise; vertical noise gets its
        # own much smaller sigma below (--smooth-vertical).
        translation = smooth_positions(
            translation, sigma, skip_axes=(1,),
            depth_axis=None if depth_measured else 2)

    if args.smooth_vertical > 0:
        vsigma = seconds_to_sigma_frames(args.smooth_vertical, fps)
        print(f"       Vertical smoothing: Gaussian sigma={vsigma:.1f} frames "
              f"({args.smooth_vertical:.2f} s at {fps:.1f} fps, Y only)")
        translation = smooth_positions(translation, vsigma, skip_axes=(0, 2))
    return translation, translation_raw


def _solve_vertical_motion(args, export_mode, joint_coords, translation,
                           rest_pos, floor_override_y=None, valid_feet=None):
    """Vertical body motion + animation grounding solve.

    floor_override_y: optional floor reference in the same coordinate
    series as the internal foot metric (translation Y + Y-flipped
    joint Y). Feet Define the Floor passes 0.0 here — its rebuilt
    vertical puts planted feet at level 0 — so that level is what
    gets grounded to clearance. None = classic lowest-frame minimum.

    IMPORTANT: cam_t alone is NOT the source of vertical body
    motion. cam_t is the SAM floor-reference world translation —
    for an actor squatting/walking in place it stays roughly
    constant, while the body's actual hip height drops via leg
    rotation. The body-internal vertical motion lives in
    jc_hips_y_yup (= -joint_coords[:, 1, 1]). We add it to
    translation[:, 1] so Hips.Y in the BVH tracks the actor's
    actual hip height. Without this, the body holds rest height
    and feet rise via FK during squats (the legacy "rising feet"
    bug).

    Foot grounding: solve animation_shift_y so the lowest BVH-
    rendered foot lands at FLOOR_CLEARANCE_M.

    With Hips.Y tracking (cam_t + jc_hips), the jc_hips term
    cancels in the foot-from-Hips chain:
      foot_render_Y(f, foot)
        = hips_pos[f, 1] + (jc_feet[f, foot] - jc_hips[f])
        = hips_offset_y + body_y[f] + animation_shift_y
            + jc_feet[f, foot] - jc_hips[f]
        = (hips_offset_y - centering_y + animation_shift_y)
            + (translation[f, 1] + jc_feet_y_yup[f, foot])
    where centering_y = jc_hips_y_yup[0] in the centered case, 0
    otherwise. So the per-frame foot Y is a constant offset from
    (smoothed translation_y + jc_feet_y_yup) — the metric and the
    render use the same series, so grounding is exact (no
    walk-cycle hover from the old metric/render mismatch).

    valid_feet: optional (n_frames, 2) bool (L, R) from Hold Cropped
    Limbs. A foot that was not in the picture is the model's GUESS, and
    the held legs that ship are a different (standing) shape, so a
    guessed foot must not set the floor: only frames where the foot was
    filmed vote. With no filmed foot at all the shift is 0 and run_body
    puts the hips at the rest pose's standing height instead.

    Returns (body_y, hips_offset_y, animation_shift_y)."""
    jc_hips_y_yup = -joint_coords[:, 1, 1]
    if not args.no_center:
        body_y = translation[:, 1] + (jc_hips_y_yup - jc_hips_y_yup[0])
    else:
        body_y = translation[:, 1] + jc_hips_y_yup

    animation_shift_y = 0.0
    hips_offset_y = float(rest_pos[1, 1])  # SAM idx 1 = Hips, post-grounding
    centering_y = float(jc_hips_y_yup[0]) if not args.no_center else 0.0
    if export_mode != "template":
        jc_feet_y_yup = -joint_coords[:, FOOT_SAM_INDICES, 1]
        render_world_feet = translation[:, 1, None] + jc_feet_y_yup
        if valid_feet is not None:
            # FOOT_SAM_INDICES = LeftFoot, LeftToe, RightFoot, RightToe
            vf = np.asarray(valid_feet, dtype=bool)
            seen = vf[:, [0, 0, 1, 1]]
            if not seen.any():
                print("       Animation grounding: no foot was ever in the "
                      "picture; hips sit at the rest pose's standing height")
                return body_y, hips_offset_y, 0.0
            render_world_feet = np.where(seen, render_world_feet, np.inf)
        min_feet = float(render_world_feet.min())
        if floor_override_y is not None:
            # Feet Define the Floor: ground the rebuilt series' plant
            # level (0.0) to clearance instead of the take's single
            # lowest frame, so a dip below the true floor dips briefly
            # instead of floating the whole take.
            min_feet = float(floor_override_y)
        animation_shift_y = (FLOOR_CLEARANCE_M
                             - hips_offset_y
                             + centering_y
                             - min_feet)
        min_pre_shift = hips_offset_y - centering_y + min_feet
        label = ("lowest planted foot" if floor_override_y is not None
                 else "lowest foot")
        print(f"       Animation grounding: {label} "
              f"{min_pre_shift*100:+.1f} cm -> "
              f"{FLOOR_CLEARANCE_M*100:.1f} cm "
              f"(shift {animation_shift_y*100:+.1f} cm)")
    return body_y, hips_offset_y, animation_shift_y


def _write_flagged_sidecar(args, fps, n_frames, flagged_runs):
    """JSON sidecar listing the frame ranges --clean-tracking replaced."""
    if not flagged_runs:
        return
    sidecar = args.output + ".flagged.json"
    payload = {
        "source_npz": args.input,
        "fps": fps,
        "n_frames": n_frames,
        "flagged_ranges": [
            {"start": int(s), "end": int(e), "length": int(e - s)}
            for s, e in flagged_runs
        ],
        "note": ("These NPZ frame indices were detected as tracking "
                 "failures and replaced with bidirectional interpolation. "
                 "Open the BVH in Blender and inspect/touch up these "
                 "frame ranges manually if the interpolated motion is "
                 "unsatisfactory."),
    }
    with open(sidecar, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"       Wrote {sidecar}")


def run_body(args, export_mode, face_data, face_calibration):
    """Body / both / template export.

    Shares the bulk of the pipeline across all three modes. Template
    synthesizes a 1-frame T-pose instead of loading a body NPZ; both
    additionally computes face data anchored to the SAM eye joints.

    Returns a dict with summary fields the entry point needs:
      cam_t_present  bool — whether vertical/locomotion data was found
      depth_measured bool — a multi-camera take: Root Smoothing was not
                     doubled on the depth axis (see _build_translation)
      face_bone_data dict|None — set in 'both' mode; used for the face-bone count
    """
    if not os.path.isfile(args.mhr_model):
        print(f"ERROR: {args.mhr_model} not found")
        sys.exit(1)

    _progress(2, "Loading body skeleton")
    parents, offsets_cm, prerot_quats = load_skeleton(args.mhr_model)
    n_joints = len(parents)

    if export_mode == "template":
        # Force tpose mode for the template — frame0 makes no sense
        # without an input NPZ to read frame 0 from.
        args.rest_pose = "tpose"

    if args.rest_pose == "tpose" and prerot_quats is None:
        print("ERROR: T-pose requested but no prerotations in model")
        print("       Falling back to frame0 rest pose")
        args.rest_pose = "frame0"

    npz, global_rots, joint_coords, fps, n_frames = _load_body_inputs(
        args, export_mode, parents, offsets_cm, prerot_quats, n_joints)

    global_rots, joint_coords, flagged_runs = _clean_tracking_stage(
        args, export_mode, parents, prerot_quats, global_rots, joint_coords)

    hand_solver_ran = _solve_hands_stage(args, export_mode, npz, parents,
                                         offsets_cm, prerot_quats,
                                         global_rots)

    # A person of a Multi-Camera & Multi-Person take is drawn at their
    # own size: the take's cameras, placed once for everybody, give how
    # far each really is from the body against how far it took the body
    # to be, which is the person's size against the body model's
    # (pipeline/people_place.py; the merge has already built the joint
    # positions at that size). After Rebuild Finger Motion on purpose:
    # that goes back to the cameras' own captures, which hold the body
    # at the model's size. The only other capture that carries the
    # number is a person of a one-camera multi-person capture whose
    # height the user typed (people_tracks.typed_size): their joint
    # positions are at that size already too.
    if npz is not None and "body_scale" in getattr(npz, "files", ()):
        body_size = float(npz["body_scale"])
        if abs(body_size - 1.0) > 1e-6:
            offsets_cm = offsets_cm * body_size
            print(f"Body size: {body_size:.3f} of the body model's")

    global_rots, crop = _hold_cropped_limbs_stage(
        args, export_mode, npz, parents, prerot_quats, global_rots, fps)
    valid_feet = crop["valid_feet"] if crop else None
    feet_never_seen = bool(crop and crop["feet_never_seen"])
    hips_held = crop["hips_held"] if crop else None
    if hips_held is not None and not hips_held.any():
        hips_held = None

    cam_t = _load_cam_t(npz, args, export_mode)

    R0_pre_correction, global_rots, joint_coords, cam_t = \
        _apply_camera_offset(args, export_mode, global_rots,
                             joint_coords, cam_t)

    # Hold Cropped Limbs, the pelvis: while the hips were out of the
    # picture, keep them upright and let them follow the chest's facing
    # (see cropped_limbs.steady_hips). After the camera offset on
    # purpose: "upright" has to be measured in the levelled frame.
    if crop is not None and export_mode != "template":
        from .cropped_limbs import (steady_hips, upright_held_legs,
                                    J_SPINE_CHAIN)
        _g0_before = global_rots[0].copy()
        if hips_held is not None:
            global_rots = steady_hips(global_rots, hips_held, parents,
                                      prerot_quats, offsets_cm, fps)
        # Held legs stay vertical under a tracked pelvis that tilts
        # (hips in the picture, legs cropped); a no-op where the pelvis
        # is upright, which covers everything steady_hips handled.
        global_rots = upright_held_legs(global_rots, crop["held"], parents,
                                        prerot_quats, offsets_cm)
        if R0_pre_correction is not None:
            # frame0 rest pose (command line only; the addon always
            # exports with the T-pose rest): keep the rest equal to the
            # PROCESSED frame 0, still in the raw pre-offset orientation
            # the rest-pose invariant asks for. The camera offset is one
            # rotation on every joint, recovered from the chest, which
            # steady_hips leaves alone.
            _chest = J_SPINE_CHAIN[-1]
            _r_cam = _g0_before[_chest] @ R0_pre_correction[_chest].T
            R0_pre_correction = _r_cam.T[None] @ global_rots[0]

    # Head-angle offset is built here (cheap) but applied AFTER
    # prune_skeleton — see "Head-bone angle offset" comment near the
    # apply_head_local call below for the why.
    #
    # Roll sign empirically inverted: applying R as a direct left-
    # multiply on local_rots[Head] (matching the body offset's
    # effective behavior on Hips) produces a roll that visually
    # rotates the head in the OPPOSITE direction from the live-
    # preview slider — likely because Head's bone roll in the
    # imported Blender skeleton is flipped relative to Hips, so the
    # same Z-axis rotation in BVH MODEL frame manifests as a
    # mirrored Y-axis rotation in Blender's pose frame for Head
    # specifically. Negating roll here makes the bake match the
    # live preview's visible direction. Pitch stays uninverted —
    # only the roll axis exhibits the bone-roll-induced flip.
    head_pitch_deg_for_bake = getattr(args, 'head_pitch', 0.0)
    head_roll_deg_for_bake = getattr(args, 'head_roll', 0.0)
    R_head_for_bake = resolve_head_level(
        pitch_deg=head_pitch_deg_for_bake,
        roll_deg=-head_roll_deg_for_bake)

    R0, rest_pos = _build_rest_pose(args, parents, offsets_cm,
                                    prerot_quats, R0_pre_correction,
                                    global_rots)

    _ground_rest_pose(rest_pos)

    # -- Boost finger rotations if requested ----------------------------
    if hand_solver_ran and args.finger_boost > 1.0:
        print("       Finger boost skipped (hand solver active)")
    elif args.finger_boost > 1.0:
        boost_finger_rotations(global_rots, parents, R0, args.finger_boost,
                               offsets_cm)

    face_bone_data, final_n_frames = _anchor_face_to_body(
        face_data, face_calibration, args, parents, offsets_cm,
        global_rots, R0, rest_pos, n_frames)
    # Vectorized BVH offsets
    bvh_offsets = np.zeros((n_joints, 3), dtype=np.float64)
    root_mask = parents < 0
    bvh_offsets[root_mask] = rest_pos[root_mask]
    child_mask = ~root_mask
    bvh_offsets[child_mask] = rest_pos[child_mask] - rest_pos[parents[child_mask]]

    local_rots = compute_local_rotations(global_rots, parents, R0)

    if args.rest_pose == "frame0":
        f0_err = np.max(np.abs(local_rots[0] - np.eye(3)))
        print(f"       Frame 0 identity error: {f0_err:.6f} (should be ~0)")

    translation, translation_raw = _build_translation(
        args, cam_t, n_frames, fps,
        depth_measured="gravity_aligned" in getattr(npz, "files", ()))
    if feet_never_seen and export_mode != "template":
        # No foot was ever filmed: the vertical body channel would come
        # from the model's guessed legs. Hold the hips at one height
        # (X/Z locomotion from the torso is kept).
        translation[:, 1] = translation[0, 1]
        translation_raw[:, 1] = translation_raw[0, 1]
    if hips_held is not None and export_mode != "template":
        # The pelvis was not filmed: its travel, height and depth are
        # all the model's guess. Hold the body where it was.
        from .cropped_limbs import hold_series
        translation = hold_series(translation, hips_held)
        translation_raw = hold_series(translation_raw, hips_held)
    # How far Root Smoothing moved the body on frame 0 (X/Z; zero with
    # the slider at 0, or with root motion off, when the root does not
    # travel at all). The camera record below is pinned on frame 0, and
    # it has to be pinned on where the capture SAW the body there: a
    # smoothed frame 0 is the average of the frames after it, which is
    # somewhere the body never was if it is already moving. Pinned on
    # that, the whole take sat that far from the camera (Add Camera),
    # and two people of one capture, each moved their own way, came out
    # further apart or closer than they were (Keep Captured Position:
    # 12 and 10 cm on the take of 2026-11-24, "when I disable smoothing,
    # the characters are spaced correctly").
    pin_shift = np.zeros(3)
    if args.root_motion and export_mode != "template":
        pin_shift[[0, 2]] = (translation[0] - translation_raw[0])[[0, 2]]

    # Robust floor (--robust-floor / "Recover Floor Level"): estimate
    # the floor from stance frames instead of the take's single lowest
    # frame, so a brief tracking glitch below the true floor can't drag
    # the grounding reference down and float the rest of the take.
    # Stances are detected once here and reused by Pass 1 below. Falls
    # back to the classic lowest-frame reference when no plants are
    # found (all-airborne clip or feet out of frame).
    foot_lock_on = (export_mode != "template"
                    and not getattr(args, 'no_foot_lock', False))

    # Feet Define the Floor (--feet-define-floor): rebuild the vertical
    # channel from plants + ballistic flight and DISCARD the video's
    # vertical translation (which can't tell a jump from camera motion).
    # Supersedes Pass 1's per-stance corrections — plants sit exactly
    # at level 0 by construction, so there is nothing left for Pass 1
    # to pull. A multi-camera take's height is measured, so its flights
    # and held blinks keep it (measured_vertical).
    use_fdf = foot_lock_on and bool(getattr(args, 'feet_define_floor',
                                            False))
    fdf_vertical = None
    fdf_planted = None
    fdf_details = {}
    if use_fdf:
        fdf_result = compute_feet_defined_vertical(
            joint_coords, translation, fps,
            sensitivity=getattr(args, 'foot_sensitivity', 0.5),
            translation_raw=translation_raw, valid_feet=valid_feet,
            measured_vertical=("gravity_aligned" in npz.files
                               and cam_t is not None),
            details=fdf_details)
        if fdf_result is None:
            print("       Feet-defined floor: no plants detected - "
                  "falling back to classic vertical")
            use_fdf = False
        else:
            fdf_vertical, fdf_planted = fdf_result

    floor_override_y = None
    if use_fdf:
        # Vertical channel swap. X/Z locomotion keeps the original
        # translation; only the Y column is rebuilt. Planted feet sit
        # at 0 in the rebuilt series, so ground that level to
        # clearance via the floor override.
        vert_translation = translation.copy()
        vert_translation[:, 1] = fdf_vertical
        floor_override_y = 0.0
    else:
        vert_translation = translation
        if "floor_y" in npz.files and cam_t is not None:
            # A measured floor level (the multi-camera merge writes one) beats
            # the classic lowest-frame rule, which lets a single
            # downward foot glitch define the floor and float the whole
            # take. Convert from NPZ world Y-down to the grounding
            # metric's series: metric = -world_ydown, plus cam_t[0].y
            # when frame 0 was centered to the origin.
            floor_override_y = -float(npz["floor_y"])
            if not args.no_center:
                floor_override_y += float(cam_t[0, 1])
            print(f"       Floor: using measured level from NPZ "
                  f"(multi-camera), not lowest frame")

    body_y, hips_offset_y, animation_shift_y = _solve_vertical_motion(
        args, export_mode, joint_coords, vert_translation, rest_pos,
        floor_override_y=floor_override_y, valid_feet=valid_feet)

    # Split: Root takes X/Z (gated by --root-motion), Hips takes Y
    # always. With --root-motion off, Root sits at (0, 0, 0) every
    # frame and only Hips.Y moves; the body lowers visibly during
    # squats without any horizontal drift. With --root-motion on,
    # Root carries the locomotion in addition.
    #
    # Hips.LOC is ABSOLUTE position relative to parent (Root), not
    # delta-from-OFFSET. Blender's BVH importer (import_bvh.py around
    # line 585) computes pose-location as `bvh_loc - rest_head_local`,
    # i.e. it treats the position channel as the bone's runtime head
    # position relative to its parent and the OFFSET as the rest head.
    # Writing only the delta makes the body drop by the Hips OFFSET
    # height (~92 cm) every frame. So the X/Z values stay at the rest
    # OFFSET (both 0 for Hips) and Y is OFFSET.y + body_y +
    # animation_shift_y.
    # Per-stance foot grounding (Pass 1). Extends animation_shift_y
    # (which only grounds the lowest foot moment globally) by also
    # grounding intermediate plants that float a few cm above due to
    # SAM's monocular jump/squat ambiguity. Returns a per-frame Y delta
    # we add to body_y so the BVH file ships already grounded — every
    # downstream tool (Blender, Maya, MotionBuilder, Unreal) sees the
    # corrected take. Skipped when --no-foot-lock is set.
    per_stance_correction = None
    if foot_lock_on and not use_fdf:
        _progress(35, "Foot grounding (Pass 1)")
        per_stance_correction = compute_per_stance_correction(
            joint_coords, translation, fps,
            sensitivity=getattr(args, 'foot_sensitivity', 0.5),
            valid_feet=valid_feet)
        body_y = body_y + per_stance_correction

    if hips_held is not None and export_mode != "template":
        # The other half of the hips' height (the pelvis' own position
        # inside the model's body frame) is held too.
        from .cropped_limbs import hold_series
        body_y = hold_series(body_y, hips_held)

    root_pos = np.zeros((n_frames, 3), dtype=np.float64)
    hips_pos = np.zeros((n_frames, 3), dtype=np.float64)
    hips_pos[:, 1] = hips_offset_y + body_y + animation_shift_y
    if valid_feet is not None and export_mode != "template":
        # Hold Cropped Limbs: while BOTH legs are held they are the rest
        # pose's standing legs, so the hips belong at the rest pose's
        # standing height - there the held feet sit exactly on the floor.
        # The height the pipeline worked out above was measured on the
        # model's guessed legs (usually bent or shorter than a straight
        # leg), which is what sank the whole body and pushed the held
        # feet under the ground plane. A visible crouch is not shown
        # while the legs are out of the picture: what was not filmed
        # stays still.
        _both_held = ~np.asarray(valid_feet, dtype=bool).any(axis=1)
        if _both_held.any():
            from .cropped_limbs import mask_weights, feet_height_below_hips
            _w = mask_weights(_both_held)
            # Per frame, from the skeleton as it will render: the hips
            # go where the LOWEST held foot touches the floor. Under an
            # upright pelvis this is exactly the rest pose's standing
            # height; under a tracked pelvis that rolls, one hip socket
            # drops and the body rises by that much instead of pushing
            # a foot through the floor.
            _stand_y = FLOOR_CLEARANCE_M - feet_height_below_hips(
                global_rots, R0, bvh_offsets, parents, FOOT_SAM_INDICES)
            _stand_y = np.where(np.isfinite(_stand_y), _stand_y,
                                hips_offset_y)
            hips_pos[:, 1] = (1.0 - _w) * hips_pos[:, 1] + _w * _stand_y
            print(f"       Hold Cropped Limbs: hips at standing height on "
                  f"{int(_both_held.sum())}/{n_frames} frames (both legs held)")
    if args.root_motion:
        root_pos[:, 0] = translation[:, 0]   # X locomotion
        root_pos[:, 2] = translation[:, 2]   # Z locomotion

    if cam_t is not None:
        y_travel = float(body_y.max() - body_y.min())
        if args.root_motion:
            xz_travel = (float(translation[:, 0].max()
                               - translation[:, 0].min()),
                         float(translation[:, 2].max()
                               - translation[:, 2].min()))
            print(f"       Root.X/Z (locomotion): travel "
                  f"X={xz_travel[0]:.3f}m Z={xz_travel[1]:.3f}m")
        else:
            print(f"       Root.X/Z: zeroed (--root-motion off; "
                  f"locomotion suppressed)")
        print(f"       Hips.Y (vertical body, cam_t + jc_hips): "
              f"travel {y_travel:.3f}m (always preserved)")

    names = assign_names(parents, rest_pos)

    # Prune non-skeletal joints and reindex everything to KEPT_ORDER.
    # prune_skeleton recomputes local rotations internally using the
    # kept-ancestor parents, so this must happen BEFORE rotation
    # smoothing (the smoothed rotations in `local_rots` from above
    # are discarded and re-smoothed on the pruned arrays).
    # Build kept_order, optionally excluding finger joints
    ko = KEPT_ORDER
    if args.no_hands:
        finger_set = set(FINGER_JOINTS)
        ko = [j for j in KEPT_ORDER if j not in finger_set]
        print(f"       --no-hands: pruning {len(KEPT_ORDER) - len(ko)} finger joints")

    parents, bvh_offsets, local_rots, names, end_site_offsets = prune_skeleton(
        parents, rest_pos, global_rots, R0, kept_order=ko)

    # ---- Head-bone angle offset --------------------------------------
    # Applied directly to local_rots[Head] AFTER prune_skeleton (NOT
    # via global_rots before compute_local_rotations). Why:
    # compute_local_rotations gives L_new[Head] = M @ R @ M^-1 @
    # L_old[Head] where M = R0[Neck] @ G[Neck].T -- a per-frame
    # conjugation of R through the Neck's pose. That conjugation
    # rotates R's apparent axis, so a pure Z-roll (forward-axis tilt
    # in MODEL frame) can present as a Y-yaw-like rotation in Head's
    # frame depending on Neck's pose. Symptom: head bake reads as
    # "wrong direction" or "ignored" relative to the body offset
    # (which has no such conjugation because Hips becomes the BVH
    # kept-tree root and has no parent).
    #
    # Direct L modification (L_new = R @ L_old) keeps the rotation
    # axis fixed in MODEL frame, matching the body offset's effective
    # behavior on Hips bit-for-bit. Direction convention: same R
    # input now produces the same direction tilt for both body and
    # head sliders.
    if export_mode != "template" and R_head_for_bake is not None:
        print(f"       Head angle offset:   "
              f"pitch={head_pitch_deg_for_bake:+.2f}deg "
              f"roll={head_roll_deg_for_bake:+.2f}deg")
        local_rots = apply_head_local(local_rots, R_head_for_bake, names)

    # Rotation smoothing. Head+Neck are CARVED OUT of the body pass and
    # owned by --smooth-head, so head smoothing tunes independently.
    # The savgol runs per joint with no cross-joint coupling, so at
    # equal window values the split is numerically identical to the
    # old single pass over all joints.
    smooth_head_s = float(getattr(args, 'smooth_head', 0.0) or 0.0)
    _head_neck_idx = [i for i, n in enumerate(names)
                     if n in ("Neck", "Head")]
    if args.smooth > 0:
        _progress(42, "Smoothing rotations")
        window = seconds_to_savgol_window(args.smooth, fps)
        print(f"       Rotation smoothing: Savitzky-Golay window={window} frames "
              f"({args.smooth:.2f} s at {fps:.1f} fps; Head+Neck excluded)")
        _body_idx = [i for i in range(len(names))
                     if i not in _head_neck_idx]
        local_rots[:, _body_idx] = smooth_rotations(
            local_rots[:, _body_idx], window)
    if smooth_head_s > 0 and _head_neck_idx:
        hwindow = seconds_to_savgol_window(smooth_head_s, fps)
        print(f"       Head rotation smoothing: Savitzky-Golay "
              f"window={hwindow} frames ({smooth_head_s:.2f} s at "
              f"{fps:.1f} fps; Head+Neck)")
        local_rots[:, _head_neck_idx] = smooth_rotations(
            local_rots[:, _head_neck_idx], hwindow)

    # Depth-velocity dead-zone on tip world positions (feet, hands,
    # head) and on root translation Z. Mirrors the face Noise Filtering
    # pattern: per-frame depth motion below threshold is suppressed via
    # smoothstep, supra-threshold passes through. Runs AFTER rotation
    # smoothing so it operates on a clean baseline, and BEFORE footskate
    # cleanup so footskate sees the denoised depth and remains the
    # authority on floor contact. See pipeline/body_depth_denoise.py.
    feet_thr = float(getattr(args, 'denoise_depth_feet', 0.0) or 0.0)
    hands_thr = float(getattr(args, 'denoise_depth_hands', 0.0) or 0.0)
    head_thr = float(getattr(args, 'denoise_depth_head', 0.0) or 0.0)
    root_thr = float(getattr(args, 'denoise_depth_root', 0.0) or 0.0)
    if export_mode != "template" and (feet_thr > 0.0 or hands_thr > 0.0
                                       or head_thr > 0.0 or root_thr > 0.0):
        _progress(45, "Depth noise filtering")
        apply_body_depth_denoise(
            parents, bvh_offsets, local_rots, hips_pos, root_pos, names,
            fps,
            feet_thr_m=feet_thr, hands_thr_m=hands_thr,
            head_thr_m=head_thr, root_thr_m=root_thr,
            progress_range=(45, 55))

    # FDF plant reconciliation: re-level the hips so the RENDERED
    # skeleton's planted foot sits at the floor with its captured leg
    # pose. FDF's v placed the hips where the AI's PERCEIVED feet
    # said; the FK skeleton's legs are a different geometry (several
    # cm on real takes), and the footskate IK below would bridge the
    # disagreement by bending knees — a near-straight leg turns 1 cm
    # of reach shortfall into ~6 degrees of visible bend, which is
    # why no amount of smoothing upstream could fix standing knees.
    # Moving the hips WITH the FK foot per frame means the pinning
    # has nothing left to correct on quiet stances: knee angles stay
    # exactly as captured, and crouch/landing dynamics pass through
    # 1:1 — during a fast crouch the planted foot stays put, so the
    # CORRECTION is near-constant and smoothing it never touches the
    # motion. The correction channel itself IS smoothed (short median
    # + Gaussian): it is calibration, not motion, and its only fast
    # content is discrete weight-foot switches and stance-edge steps,
    # which otherwise render as visible cm-scale hips pops. Runs
    # AFTER rotation smoothing and depth denoise so it measures what
    # actually ships — the same lesson body grounding learned. Gap
    # frames (jump arcs) get the correction interpolated between the
    # surrounding plant edges, so arc shapes shift by a near-constant
    # offset and stay intact.
    #
    # Target is the floor plane itself (0), not FLOOR_CLEARANCE_M:
    # holding every plant at exactly +1 cm reads as hovering — the
    # classic path lets plants press to the floor and slightly into
    # it (foot meshes extend below the joints), which is what a
    # firmly planted contact looks like.
    if (use_fdf and fdf_planted is not None and export_mode != "template"
            and fdf_planted.any()):
        _progress(52, "FDF plant reconciliation")
        from .footskate_cleanup import (_fk_world_positions_bvh,
                                        _local_to_global_bvh)
        name_idx = {nm: i for i, nm in enumerate(names)}
        foot_pairs = []
        for side in ("Left", "Right"):
            pair = [name_idx[nm] for nm in (side + "Foot", side + "Toe")
                    if nm in name_idx]
            foot_pairs.append(pair)
        if all(foot_pairs):
            any_plant_m = fdf_planted.any(axis=1)
            delta = np.full(n_frames, np.nan)
            for f in range(n_frames):
                if not any_plant_m[f]:
                    continue
                g = _local_to_global_bvh(parents, local_rots[f])
                w = _fk_world_positions_bvh(parents, bvh_offsets, g,
                                            hips_pos[f])
                # Bottom of each PLANTED foot; weight foot = lowest.
                cand = [w[foot_pairs[c], 1].min()
                        for c in (0, 1) if fdf_planted[f, c]]
                delta[f] = -min(cand)
            # Interpolate across gaps, hold before/after the ends,
            # smooth the correction channel (see the comment above),
            # and clamp for safety (a genuine correction is cm-scale).
            from scipy.ndimage import (gaussian_filter1d as _gauss1d,
                                       median_filter as _med_filt)
            # A gap that kept the take's MEASURED height (multi-
            # camera: a flight, or a squat the detector lost the feet
            # in) gets the take's typical correction, not one
            # interpolated from the plants beside it: those are the
            # takeoff crouch and the landing, the fastest leg changes
            # in the take and so where smoothing moves the rendered
            # feet most, and their corrections shifted the whole
            # flight (6 cm on a 44 cm jump).
            typical = float(np.median(delta[~np.isnan(delta)]))
            blend = int(fdf_details.get("measured_blend_frames", 2))
            for g0, g1 in fdf_details.get("measured_gaps", ()):
                if g1 - g0 + 1 > 2 * blend:
                    delta[g0 + blend:g1 + 1 - blend] = typical
            pl_idx = np.flatnonzero(~np.isnan(delta))
            delta = np.interp(np.arange(n_frames), pl_idx, delta[pl_idx])
            med_w = max(3, int(round(0.2 * fps)) | 1)
            delta = _med_filt(delta, size=med_w, mode='nearest')
            delta = _gauss1d(delta, sigma=max(1.0, 0.08 * fps))
            delta = np.clip(delta, -0.15, 0.15)
            print(f"       FDF plant reconciliation: hips re-leveled "
                  f"{delta.min()*100:+.1f}..{delta.max()*100:+.1f} cm "
                  f"(median {np.median(delta[any_plant_m])*100:+.1f}, "
                  f"max step {np.abs(np.diff(delta)).max()*100:.1f} "
                  f"cm/frame) so planted FK feet touch the floor")
            hips_pos[:, 1] += delta

    # Per-foot FK grounding (Pass 2): Kovar-style footskate cleanup.
    # Runs AFTER rotation smoothing so the smoother provides the "non-
    # stance" baseline; we then overwrite leg rotations during plants
    # with 2-bone-IK + knee-damping solutions targeting a per-PLANT
    # constraint position (averaged from each plant's first frames).
    # Closes the residual that Pass 1 (Hips Y body shift) can't fix —
    # both feet planted at different perceived heights in low squats.
    # See pipeline/footskate_cleanup.py for the algorithm.
    if export_mode != "template" and not getattr(args, 'no_foot_lock', False):
        _progress(55, "Footskate cleanup (Pass 2)")
        apply_footskate_cleanup(
            parents, bvh_offsets, local_rots, hips_pos,
            names, joint_coords, translation, fps,
            sensitivity=getattr(args, 'foot_sensitivity', 0.5),
            max_gap_sec=getattr(args, 'max_gap', 0.33),
            max_lift_m=getattr(args, 'max_lift', 0.10),
            recover_glitches=getattr(args, 'recover_glitches', False),
            recovery_lift_m=getattr(args, 'recovery_lift', 0.15),
            pin_feet=not getattr(args, 'no_foot_pin', False),
            pin_strength=getattr(args, 'foot_pin_strength', 1.0),
            pin_release_m=getattr(args, 'foot_pin_release', 0.20),
            # Pin anchors live in the viewer's frame: when the Root
            # carries X/Z locomotion (added after this pass), the pin
            # must see it — else planted walking feet read as sliding
            # and pinned feet would skate along with the Root.
            pin_locomotion_xz=(
                np.asarray(translation[:, (0, 2)], dtype=np.float64)
                if args.root_motion else None),
            robust_floor=use_fdf,
            local_height_base=use_fdf,
            progress_range=(55, 90), valid_feet=valid_feet)

    # Body grounding (--body-grounding): the LAST grounding pass, run
    # on the FINAL FK-rendered skeleton — after rotation smoothing,
    # depth denoise, and footskate cleanup — so it grounds exactly what
    # ships. Earlier placements measured SAM's raw joint positions and
    # were invalidated downstream (rotation smoothing straightens the
    # roll's spine curl; the foot passes move the feet that define the
    # visual floor). Its posture gate is closed for anything upright,
    # so stances and jumps are untouched; corrections land on the Hips
    # Y channel, moving the whole body per frame.
    if getattr(args, 'body_grounding', False) and export_mode != "template":
        _progress(90, "Body grounding")
        from .footskate_cleanup import (_fk_world_positions_bvh,
                                        _local_to_global_bvh)
        from .body_grounding import compute_body_tilt_alignment

        def _bg_fk():
            w = np.zeros((n_frames, len(parents), 3))
            for f in range(n_frames):
                g = _local_to_global_bvh(parents, local_rots[f])
                w[f] = _fk_world_positions_bvh(parents, bvh_offsets,
                                               g, hips_pos[f])
            return w

        bg_world = _bg_fk()
        # Support-plane tilt alignment first: a resting body whose
        # perceived orientation is rolled/pitched (monocular depth's
        # lying-pose failure) gets leveled by a rigid rotation about
        # its contact centroid, so the translation pass below can
        # seat the whole contact polygon instead of pinning the one
        # deepest joint while everything else ramps out of the floor.
        # Writes the Hips channels directly: at this point Hips is
        # still the FK root (the synthetic Root is inserted after
        # this block), so its local rotation IS its world rotation.
        skin_r = getattr(args, 'body_skin_radius', 0.015)
        rotvec, tilt_pivot, resting_w = compute_body_tilt_alignment(
            bg_world, names, fps, back_radius=skin_r)
        if np.linalg.norm(rotvec, axis=1).max() > 1e-6:
            from scipy.spatial.transform import Rotation as _Rot
            hips_i = names.index("Hips")
            Rm = _Rot.from_rotvec(rotvec).as_matrix()
            for f in range(n_frames):
                if np.abs(rotvec[f]).max() < 1e-9:
                    continue
                local_rots[f, hips_i] = Rm[f] @ local_rots[f, hips_i]
                hips_pos[f] = (tilt_pivot[f]
                               + Rm[f] @ (hips_pos[f] - tilt_pivot[f]))
            bg_world = _bg_fk()
        # With Feet Define the Floor, the floor plane IS y=0 by
        # construction (plants grounded to clearance above it) — pass
        # it. The derive-from-feet fallback self-references on
        # lie-heavy takes: the lying feet, capture-pressed below the
        # floor, become the 5th percentile, and the whole resting
        # body seats several cm under the visible floor.
        hips_pos[:, 1] += compute_body_grounding_correction(
            bg_world, names, fps,
            catch_distance=getattr(args, 'body_catch_distance', 0.30),
            back_radius=skin_r, resting_w=resting_w,
            floor_y=0.0 if floor_override_y is not None else None)

    # Hold Cropped Limbs, final seating. The hips were put at standing
    # height when the vertical channel was assembled, but the stages
    # since then (plant re-levelling between real stances, smoothing,
    # body grounding) are interpolated across the held frames and move
    # them again by a few cm. So, last of all the automatic passes and
    # on the skeleton exactly as it ships: on frames where BOTH legs
    # are held, the hips go where the lowest held foot touches the
    # floor. Frames with a filmed foot are not touched.
    if valid_feet is not None and export_mode != "template":
        _both_held = ~np.asarray(valid_feet, dtype=bool).any(axis=1)
        _foot_idx = [names.index(nm) for nm in
                     ("LeftFoot", "LeftToe", "RightFoot", "RightToe")
                     if nm in names]
        if _both_held.any() and _foot_idx and len(_both_held) == n_frames:
            from .cropped_limbs import mask_weights
            from .footskate_cleanup import (_fk_world_positions_bvh,
                                            _local_to_global_bvh)
            _w = mask_weights(_both_held)
            for f in np.where(_w > 0)[0]:
                g = _local_to_global_bvh(parents, local_rots[f])
                wpos = _fk_world_positions_bvh(parents, bvh_offsets, g,
                                               hips_pos[f])
                low = float(wpos[_foot_idx, 1].min())
                hips_pos[f, 1] += _w[f] * (FLOOR_CLEARANCE_M - low)

    # Manual floor-level correction (--height-offset): a constant Y on
    # the Hips channel, applied AFTER Pass 1/Pass 2 grounding so the
    # pinned plants shift with the body instead of being re-pinned to
    # the uncorrected floor. Mirrors the live Height slider in Blender
    # (properties._apply_height_offset_to_rig writes the same uniform
    # shift into Hips keyframes); the Bake operator passes the
    # cumulative total here so the correction survives in the file.
    height_offset = float(getattr(args, 'height_offset', 0.0))
    if export_mode != "template" and abs(height_offset) > 1e-9:
        print(f"       Height offset: {height_offset:+.3f} m")
        hips_pos[:, 1] += height_offset

    # Head noise filtering: WORLD-SPACE head stabilization (nod/turn/
    # tilt dead-zones in degrees). The wobble is smeared across the
    # whole chain, so the gate works on the head's world orientation
    # and counter-rotates the Head bone — the only joint it writes.
    # MUST RUN LAST among chain-modifying stages: depth denoise, FDF,
    # footskate, and body grounding all rotate/translate joints under
    # the head, and each re-creates world-space head wobble — running
    # the gate right after smoothing measurably did almost nothing on
    # the reference clip once the rest of the pipeline had run
    # (RMS 0.90 vs 0.43 when gated last). Only the synthetic Root
    # insert and tail padding follow this point; neither rotates the
    # chain. See pipeline/head_rotation_denoise.
    _hd_nod = float(getattr(args, 'head_denoise_nod', 0.0) or 0.0)
    _hd_turn = float(getattr(args, 'head_denoise_turn', 0.0) or 0.0)
    _hd_tilt = float(getattr(args, 'head_denoise_tilt', 0.0) or 0.0)
    if export_mode != "template" and (_hd_nod > 0.0 or _hd_turn > 0.0
                                       or _hd_tilt > 0.0):
        _progress(91, "Head noise filtering")
        apply_head_rotation_denoise(local_rots, names, parents, fps,
                                    _hd_nod, _hd_turn, _hd_tilt)

    # Insert synthetic Root (floor projection, X/Z locomotion) and
    # RootEnd (horizontal-forward stub for clean Blender bone-tail
    # visualization). Hips becomes Root's second child and gains a
    # 6-channel slot so its Y position carries vertical body delta.
    parents, bvh_offsets, local_rots, names, end_site_offsets = (
        insert_synthetic_root(parents, bvh_offsets, local_rots,
                              names, end_site_offsets))

    # Face longer than body: hold the body's PROCESSED last frame for
    # the trailing span. Padding here — after grounding, smoothing,
    # depth denoise, and footskate cleanup — keeps every solve stage
    # identical to a no-face export of the same body NPZ; the held
    # frames change no clip-wide statistic and no knee IK anchor.
    if final_n_frames > n_frames:
        pad_n = final_n_frames - n_frames
        local_rots = np.concatenate([
            local_rots,
            np.broadcast_to(local_rots[-1],
                            (pad_n,) + local_rots.shape[1:]),
        ], axis=0)
        hips_pos = np.concatenate([
            hips_pos,
            np.broadcast_to(hips_pos[-1], (pad_n, 3)),
        ], axis=0)
        root_pos = np.concatenate([
            root_pos,
            np.broadcast_to(root_pos[-1], (pad_n, 3)),
        ], axis=0)

    # Add Camera: the real camera's place in this BVH's world, anchored
    # on the hips as they ship on frame 0 (after every pass above), less
    # what Root Smoothing moved them there (pin_shift, above). See
    # pipeline/capture_camera.py.
    if (getattr(args, 'camera_sidecar', '') and export_mode != "template"
            and npz is not None):
        from .capture_camera import write_camera_sidecar
        write_camera_sidecar(
            args.camera_sidecar, npz, args.input,
            hips_world0=root_pos[0] + hips_pos[0] - pin_shift,
            root_world0=root_pos[0],
            R_level=resolve_camera_level(
                pitch_deg=getattr(args, 'camera_pitch', 0.0),
                roll_deg=-getattr(args, 'camera_roll', 0.0)),
            view=getattr(args, 'camera_view', 'a'),
            n_frames=int(local_rots.shape[0]))

    _progress(92, "Writing BVH file")
    hips_idx = names.index("Hips")
    write_bvh(args.output, parents, bvh_offsets, names,
              local_rots, root_pos, fps,
              face_bone_data=face_bone_data,
              end_site_offsets=end_site_offsets,
              extra_pos={hips_idx: hips_pos})
    _progress(99, "Finalizing")

    _write_flagged_sidecar(args, fps, n_frames, flagged_runs)

    return {
        "cam_t_present": cam_t is not None,
        "depth_measured": "gravity_aligned" in getattr(npz, "files", ()),
        "face_bone_data": face_bone_data,
    }


def run_face(args, face_data, face_calibration):
    """Standalone face-only export (Head bone + face bones)."""
    _progress(10, "Computing head-local landmarks")
    print("       Computing head-local landmarks...")
    # Anchor MP eye-mid to (0,0,0) so the rendered face cluster sits
    # centered on the Head bone. Without anchors, compute_head_local_
    # landmarks centers on MP landmark 168 (nose bridge) in X and 20%-
    # up-from-chin in Y, leaving the iris midpoint several cm off origin
    # — that offset becomes the cluster shift downstream.
    half_ipd = AVERAGE_IPD_METERS * 0.5
    eye_anchors = (
        np.array([-half_ipd, 0.0, 0.0]),
        np.array([ half_ipd, 0.0, 0.0]),
    )
    head_local_lm, _ = compute_head_local_landmarks(
        face_data, eye_anchors=eye_anchors)

    print("[1/3] Building minimal skeleton for face-only export")
    parents, offsets, names = build_face_only_skeleton()
    n_joints = len(parents)

    fps = face_data["fps"]
    if args.fps_override > 0:
        fps = args.fps_override

    n_frames = len(face_data["landmarks"])
    print(f"       {n_frames} frames, {fps:.1f} FPS")

    _progress(40, "Computing face bone animation")
    print("       Computing face bone animation (Rigify rig)...")
    # Face-only mode: no body to anchor against, so pin the absolute
    # face size to the anatomical IPD. Same value every face-only
    # export -> identical head/face proportions across captures.
    face_bone_data = compute_face_bone_data(
        face_data, head_local_lm, face_boost=args.face_boost,
        face_calibration=face_calibration,
        face_ipd_target=AVERAGE_IPD_METERS)

    # Center the face cluster's bounding box on the armature origin:
    # X/Z bbox center -> 0 (true visual center, not just eye-mid /
    # Jaw-head which leave small source-rig asymmetries), and the
    # lowest bone resting exactly at Y=0. Shifts both rest offsets and
    # per-frame position channels so rest and animation stay consistent.
    # Head bone itself stays at (0,0,0); only the face bones move.
    all_rest = np.array(list(face_bone_data["offsets"].values()))
    min_xyz = all_rest.min(axis=0)
    max_xyz = all_rest.max(axis=0)
    shift = np.array([
        -0.5 * (min_xyz[0] + max_xyz[0]),
        -min_xyz[1],
        -0.5 * (min_xyz[2] + max_xyz[2]),
    ], dtype=np.float64)
    for name in face_bone_data["order"]:
        face_bone_data["offsets"][name] += shift
        face_bone_data["frames"][name][:, :3] += shift
    print(f"       Face grounding: bbox center shift "
          f"({shift[0]*100:+.1f}, {shift[1]*100:+.1f}, "
          f"{shift[2]*100:+.1f}) cm")

    # Identity rotations: vectorized
    local_rots = np.tile(np.eye(3), (n_frames, n_joints, 1, 1))
    root_pos = np.zeros((n_frames, 3))

    print("[2/3] Face data loaded above")
    _progress(85, "Writing BVH file")
    print("[3/3] Writing face-only BVH...")

    write_bvh(args.output, parents, offsets.copy(), names,
              local_rots, root_pos, fps,
              face_bone_data=face_bone_data)
    _progress(99, "Finalizing")

    return {"face_bone_data": face_bone_data}


def run_hands(args):
    """Standalone hands-only export (Root + 2 hands + fingers)."""
    if not os.path.isfile(args.mhr_model):
        print(f"ERROR: {args.mhr_model} not found")
        sys.exit(1)

    _progress(10, "Loading body skeleton + NPZ")
    print("[1/3] Loading body skeleton + NPZ for hands extraction")
    body_parents, body_offsets_cm, prerot_quats = load_skeleton(
        args.mhr_model)

    if prerot_quats is None:
        print("ERROR: Hands export requires T-pose prerotations in model")
        sys.exit(1)
    body_R0 = compute_tpose_globals(body_parents, prerot_quats)

    npz = np.load(args.input)
    global_rots = npz["global_rots"].astype(np.float64)
    fps = float(npz["fps"])
    if args.fps_override > 0:
        fps = args.fps_override

    n_frames = global_rots.shape[0]
    print(f"       {n_frames} frames, {fps:.1f} FPS")

    hand_solver_ran = _solve_hands_stage(args, "hands", npz, body_parents,
                                         body_offsets_cm, prerot_quats,
                                         global_rots)

    if hand_solver_ran and args.finger_boost > 1.0:
        print("       Finger boost skipped (hand solver active)")
    elif args.finger_boost > 1.0:
        boost_finger_rotations(global_rots, body_parents, body_R0,
                               args.finger_boost, body_offsets_cm)

    _progress(40, "Building hands skeleton")
    print("[2/3] Building hands-only skeleton...")
    (parents, bvh_offsets, names, body_idx,
     end_site_offsets) = build_hands_only_skeleton(
        body_parents, body_offsets_cm, body_R0, scale=args.scale)
    print(f"       {len(names)} bones (Root + 2 hands + {len(names)-3} "
          f"finger joints)")

    local_rots = compute_hands_local_rots(global_rots, body_R0,
                                          body_idx, parents)

    if args.smooth > 0:
        window = seconds_to_savgol_window(args.smooth, fps)
        print(f"       Rotation smoothing: Savitzky-Golay window={window} frames "
              f"({args.smooth:.2f} s at {fps:.1f} fps)")
        local_rots = smooth_rotations(local_rots, window)

    # Hands armature is anchored at origin — no root translation.
    root_pos = np.zeros((n_frames, 3), dtype=np.float64)

    _progress(85, "Writing BVH file")
    print("[3/3] Writing hands-only BVH...")
    write_bvh(args.output, parents, bvh_offsets, names,
              local_rots, root_pos, fps,
              end_site_offsets=end_site_offsets)
    _progress(99, "Finalizing")

    return {}


def run_root(args):
    """Standalone root-only export (single Root bone, cam_t motion)."""
    _progress(20, "Loading body NPZ")
    print("[1/3] Loading body NPZ for root motion")
    npz = np.load(args.input)

    if "cam_t" not in npz:
        print("ERROR: --export root requires 'cam_t' in body NPZ")
        sys.exit(1)

    cam_t = npz["cam_t"].astype(np.float64)
    fps = float(npz["fps"])
    if args.fps_override > 0:
        fps = args.fps_override

    n_frames = cam_t.shape[0]
    print(f"       {n_frames} frames, {fps:.1f} FPS")

    print(f"       {_detrend_camera_pitch(cam_t, npz)}")

    print("[2/3] Building root-only skeleton...")
    parents, bvh_offsets, names, end_site_offsets = (
        build_root_only_skeleton())

    # Same Y/Z flip the body+root path applies to cam_t.
    root_pos = cam_t.copy()
    root_pos[:, 1] *= -1
    root_pos[:, 2] *= -1

    if not args.no_center:
        delta = root_pos[0].copy()
        if not np.allclose(delta, 0, atol=1e-6):
            root_pos -= delta
            print(f"       Centered: frame 0 -> origin")

    if args.smooth_root > 0:
        sigma = seconds_to_sigma_frames(args.smooth_root, fps)
        print(f"       Root smoothing: Gaussian sigma={sigma:.1f} frames "
              f"({args.smooth_root:.2f} s at {fps:.1f} fps, X/Z only)")
        # Same Y exclusion as the body path (_build_translation): in this
        # root-only export ALL vertical motion lives in this one vector,
        # so smoothing Y would flatten jump arcs here too. And the same
        # depth rule: no doubling on a multi-camera take.
        root_pos = smooth_positions(
            root_pos, sigma, skip_axes=(1,),
            depth_axis=None if "gravity_aligned" in npz.files else 2)

    if args.smooth_vertical > 0:
        vsigma = seconds_to_sigma_frames(args.smooth_vertical, fps)
        print(f"       Vertical smoothing: Gaussian sigma={vsigma:.1f} frames "
              f"({args.smooth_vertical:.2f} s at {fps:.1f} fps, Y only)")
        root_pos = smooth_positions(root_pos, vsigma, skip_axes=(0, 2))

    local_rots = np.tile(np.eye(3), (n_frames, 1, 1, 1))

    _progress(85, "Writing BVH file")
    print("[3/3] Writing root-only BVH...")
    write_bvh(args.output, parents, bvh_offsets, names,
              local_rots, root_pos, fps,
              end_site_offsets=end_site_offsets)
    _progress(99, "Finalizing")

    return {}
