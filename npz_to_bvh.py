# npz_to_bvh.py
# ==============
# Converts MHR inference data (NPZ) to BVH motion capture file.
# Supports body, face, body+face, standalone hands, or standalone root export.
#
# This file is the CLI entry point: argparse, mode dispatch, and a final
# summary print. The pipeline itself lives under pipeline/ as topical
# modules:
#
#   pipeline/bvh_math.py          rotation/FK/smoothing primitives
#   pipeline/body_skeleton.py     SAM body skeleton (load/name/prune/finger-boost)
#   pipeline/tracking_repair.py   bad-frame detection + bidirectional interpolation
#   pipeline/face_export.py       Face bone animation from source-rig calibration
#   pipeline/skeleton_builders.py face / root / hands standalone skeletons
#   pipeline/bvh_writer.py        Y-up BVH file writer
#   pipeline/modes.py             per-export-mode handlers (body / face / hands / root)
#
# Verified FK math (error < 0.015 against pred_joint_coords):
#   world_pos[j] = world_pos[parent] + global_rots[parent] @ offsets[j]
#   local_rot[j] = global_rots[parent]^T @ global_rots[j]
#
# Rest pose modes:
#   frame0  - Frame 0 world positions (original behavior)
#   tpose   - T/A-pose from accumulated prerotation quaternions
#
# Root motion (two-bone canonical layout):
#   --root-motion    Include horizontal locomotion (cam_t X/Z) on the
#                    synthetic Root bone. Vertical body motion (squat
#                    depth, jump, gait bob) lives on Hips.Y and is
#                    ALWAYS included regardless of this flag, so the
#                    character's body height stays correct when
#                    locomotion is off.
#   --no-center      Don't recenter the rig to origin at frame 0
#
# Smoothing (all values in SECONDS, converted to frames at runtime
# using the capture fps — see the argparse help strings):
#   --smooth N        Savitzky-Golay on body rotations, window N seconds (e.g. 0.23)
#   --smooth-root N   Gaussian on root position, sigma N seconds (e.g. 0.27)
#                     Depth axis (Z) gets 2x sigma automatically.
#   --smooth-face N   Savitzky-Golay on face data, window N seconds (e.g. 0.23)
#                     Smooths landmarks, blendshapes, eye gaze.
#
# Face bones (from face_frames.npz):
#   --face PATH      Face NPZ from save_face_data.py
#   --export MODE    body | face | both | hands | root
#                    (default: auto based on inputs)
#
# Export modes:
#   body   - Full body skeleton (optionally with face bones under Head
#            when --face is supplied; pass --no-hands to drop fingers).
#   face   - Head bone + face bones only (needs --face NPZ).
#   both   - Body skeleton + face bones under Head.
#   hands  - Standalone hands armature: Root -> LeftHand + RightHand ->
#            finger chains (needs --input body NPZ).
#   root   - Single Root bone animated by cam_t motion
#            (needs --input body NPZ with cam_t).
#
# Face bones are children of the Head joint. They use 6-channel BVH
# (position + rotation) so Blender sees them as position-driven bones.
# Jaw and eyes use rotation channels from blendshape/gaze data.
# All other face bones use position channels from head-local landmarks.
#
# Run:  python npz_to_bvh.py --input mhr_frames.npz
#       python npz_to_bvh.py --input mhr_frames.npz --face face_frames.npz --export both
#       python npz_to_bvh.py --face face_frames.npz --export face
#       python npz_to_bvh.py --input mhr_frames.npz --export hands
#       python npz_to_bvh.py --input mhr_frames.npz --export root
# Out:  output.bvh

import argparse
import os
import sys

from pipeline.bvh_math import HAS_SCIPY, smooth_face_data
from pipeline.face_export import load_face_data, load_face_calibration
from pipeline.modes import run_body, run_face, run_hands, run_root


def _build_argparser():
    parser = argparse.ArgumentParser(
        description="Convert MHR inference NPZ to BVH (body, face, or both)"
    )
    parser.add_argument("--input", default="",
                        help="Body NPZ from save_mhr_data.py (required for body/both)")
    parser.add_argument("--face", default="",
                        help="Face NPZ from save_face_data.py (required for face/both)")
    parser.add_argument("--export",
                        choices=["body", "face", "both", "hands", "root",
                                 "template"],
                        default="",
                        help="Export mode: body | face | both | hands | root "
                             "| template (default: auto from inputs). "
                             "hands/root produce standalone armatures from "
                             "the body NPZ. template produces a 1-frame "
                             "T-pose body+hands armature with no input NPZ "
                             "required — used by the BlendCap addon's "
                             "'Import BlendCap Armature' button so users can "
                             "pose it and save a custom rest-pose preset.")
    parser.add_argument("--output", default="output.bvh",
                        help="Output BVH file")
    parser.add_argument("--mhr-model",
                        default="checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
                        help="Path to mhr_model.pt")
    parser.add_argument("--scale", type=float, default=0.01,
                        help="Scale factor (0.01 = cm to meters)")
    parser.add_argument("--fps-override", type=float, default=0,
                        help="Override FPS (0 = use NPZ value)")
    parser.add_argument("--rest-pose", choices=["frame0", "tpose"],
                        default="frame0",
                        help="Rest pose mode: frame0 (default) or tpose")
    parser.add_argument("--root-motion", action="store_true",
                        help="Include horizontal locomotion (cam_t X/Z) "
                             "on the synthetic Root bone. Vertical body "
                             "motion (squat depth, jump, gait bob) lives "
                             "on Hips.Y and is always included regardless "
                             "of this flag, so body height stays correct "
                             "when locomotion is off.")
    parser.add_argument("--no-center", action="store_true",
                        help="Don't recenter the rig to origin at frame 0")
    parser.add_argument("--smooth", type=float, default=0,
                        help="Savitzky-Golay window for body rotations in "
                             "seconds (e.g. 0.23). FPS-independent: "
                             "converted to an odd frame window at runtime "
                             "using the capture rate. 0 = off.")
    parser.add_argument("--smooth-root", type=float, default=0,
                        help="Gaussian sigma for root position in seconds "
                             "(e.g. 0.27). FPS-independent: converted to "
                             "frames at runtime. Depth axis auto gets 2x. "
                             "0 = off.")
    parser.add_argument("--smooth-face", type=float, default=0.23,
                        help="Face smoothing window in seconds (Savitzky-"
                             "Golay, polyorder=3). FPS-independent: "
                             "converted to a frame window using the "
                             "face NPZ's fps. Zero-phase by construction "
                             "(centered convolution). 0 = off.")
    parser.add_argument("--face-refine",
                        choices=["off", "geometric", "full"],
                        default=os.environ.get("BLENDCAP_FACE_REFINE",
                                               "full"),
                        help="Recompute the blendshape channels "
                             "MediaPipe gets wrong (jawOpen from chin "
                             "drop, teeth-vs-cavity jaw gate, browDown, "
                             "mouthFrown, noseSneer, lip-part routing). "
                             "geometric = landmark+pixel tiers; full "
                             "adds the ICT-FaceKit least-squares solve "
                             "(default; falls back to geometric if the "
                             "baked basis is missing). off = raw "
                             "MediaPipe values. Env default: "
                             "BLENDCAP_FACE_REFINE.")
    parser.add_argument("--face-neutral-frame", type=int, default=-1,
                        help="Frame index (into the face NPZ) to treat "
                             "as the actor's neutral face for --face-"
                             "refine. -1 (default) auto-detects the "
                             "most relaxed frames in the clip. Set this "
                             "when the actor never relaxes on camera.")
    parser.add_argument("--face-refine-report", default="",
                        help="Optional path to write a JSON report of "
                             "what --face-refine changed per channel "
                             "(for tuning/debugging).")
    parser.add_argument("--face-denoise-enabled", action="store_true",
                        help="Apply per-region dead-zone denoising to "
                             "face blendshapes after smoothing. Uses "
                             "the --face-denoise-{region} thresholds.")
    for _r in ("jaw", "gaze", "lids", "brows", "lips", "cheeks", "nose", "tongue"):
        parser.add_argument(f"--face-denoise-{_r}", type=float, default=0.0,
                            help=f"Dead-zone threshold for the '{_r}' "
                                 f"blendshape group, in [0,1] units. "
                                 f"Values below threshold snap to 0. "
                                 f"0 = no filtering for this region.")
    parser.add_argument("--face-amp-enabled", action="store_true",
                        help="Apply per-region amplitude scaling to "
                             "face blendshapes after smoothing+denoise. "
                             "Uses the --face-amp-{region} multipliers.")
    for _r in ("jaw", "gaze", "lids", "brows", "lips", "cheeks", "nose", "tongue"):
        parser.add_argument(f"--face-amp-{_r}", type=float, default=1.0,
                            help=f"Amplitude multiplier for the '{_r}' "
                                 f"blendshape group. 1.0 = 1:1, higher "
                                 f"amplifies, lower attenuates.")
    # --- Body depth noise filtering (sibling of face-denoise but on the
    # camera-depth axis of body tip world positions). Per-tip smoothstep
    # dead-zone on the RESIDUAL against a ~0.3s Savgol baseline. Real
    # slow motion lives in the baseline and passes through; sub-amplitude
    # wobble around the baseline is suppressed. Threshold is in meters
    # of depth-residual amplitude; below threshold the residual is
    # killed, above 2x threshold passes through, smoothstep between.
    # Runs after rotation smoothing, before footskate cleanup.
    for _t, _desc in (
        ("feet",  "feet (back-solved through UpLeg+Leg IK)"),
        ("hands", "hands (back-solved through Arm+ForeArm IK)"),
        ("head",  "head (swing delta on the head bone)"),
        ("root",  "root translation Z"),
    ):
        parser.add_argument(f"--denoise-depth-{_t}", type=float, default=0.0,
                            help=f"Depth-residual amplitude threshold "
                                 f"in METERS for {_desc}. 0 = off.")
    parser.add_argument("--finger-boost", type=float, default=1.0,
                        help="Scale finger rotation angles by this factor. "
                             "SAM's PCA hand model dampens finger curl to ~15 deg "
                             "when a fist should be ~80 deg. Try 5.0 to start. "
                             "1.0 = no change (default).")
    parser.add_argument("--hand-solver",
                        choices=["off", "on"],
                        default=os.environ.get("BLENDCAP_HAND_SOLVER",
                                               "off"),
                        help="Refit finger rotations to the tracked 2D "
                             "finger keypoints (experimental). Recovers "
                             "real articulation the 3D rotation output "
                             "dampens. When on, --finger-boost is "
                             "skipped. Env override: BLENDCAP_HAND_SOLVER.")
    parser.add_argument("--face-boost", type=float, default=1.0,
                        help="Scale face bone blendshape-driven position offsets. "
                             "Higher = more expressive face animation. "
                             "1.0 = raw blendshape scale (default; matches the "
                             "addon's Face Expression Multiplier default).")
    parser.add_argument("--face-calibration", default="",
                        help="Path to face_calibration.json (from extract_face_calibration.py). "
                             "Provides the source rig's bone topology and per-ARKit-shape "
                             "pose deltas (translation + rotation).")
    parser.add_argument("--no-hands", action="store_true",
                        help="Exclude finger bones from BVH. Keeps Hand bones "
                             "as leaf joints; fingers are pruned.")
    parser.add_argument("--clean-tracking", action="store_true",
                        help="Detect tracking-failure frames (limb-length "
                             "anomalies and pose-head regression to the rest "
                             "pose) and bidirectionally interpolate through "
                             "them.  Writes a sidecar <output>.flagged.json "
                             "listing repaired frame ranges for manual cleanup.")
    # --- Foot Locking (Pass 1 + Pass 2) ---
    parser.add_argument("--no-foot-lock", action="store_true",
                        help="Skip foot-locking entirely (both Pass 1 body "
                             "grounding and Pass 2 footskate cleanup).")
    parser.add_argument("--foot-sensitivity", type=float, default=0.5,
                        help="Sensitivity in [0, 1]. Higher catches more "
                             "foot contacts (looser thresholds), lower "
                             "catches fewer (stricter). 0.5 is the "
                             "calibrated default.")
    parser.add_argument("--max-gap", type=float, default=0.33,
                        help="Seconds. Bridge a gap between adjacent "
                             "contacts shorter than this if the foot "
                             "stays low (see --max-lift).")
    parser.add_argument("--max-lift", type=float, default=0.10,
                        help="Meters. Don't bridge a gap if the foot "
                             "rises higher than this between contacts.")
    parser.add_argument("--recover-glitches", action="store_true",
                        help="Bridge longer gaps when the foot stayed "
                             "near the floor and didn't move horizontally — "
                             "catches longer SAM tracking misreads. "
                             "Off by default; can erase small genuine "
                             "steps when on.")
    parser.add_argument("--recovery-lift", type=float, default=0.15,
                        help="Meters. Lift cap for the --recover-glitches "
                             "path. Typically larger than --max-lift.")
    parser.add_argument("--feet-define-floor", action="store_true",
                        help="Rebuild the body's vertical motion from "
                             "foot plants + ballistic flight, discarding "
                             "the video's vertical translation entirely. "
                             "Planted feet are pinned to the floor; "
                             "airborne gaps follow a gravity arc sized "
                             "by hang time.")
    parser.add_argument("--no-foot-pin", action="store_true",
                        help="Skip the planted-foot pin (on by "
                             "default). Pinning holds each planted "
                             "foot in place (X/Z) to remove sliding; "
                             "a contact whose data drifts past "
                             "--foot-pin-release unpins entirely, so "
                             "real slides pass through.")
    parser.add_argument("--foot-pin-strength", type=float, default=1.0,
                        help="0-1. How firmly pinned feet hold their "
                             "planted position; lower values let some "
                             "of the source motion through.")
    parser.add_argument("--foot-pin-release", type=float, default=0.20,
                        help="Meters. Unpin a contact when the foot "
                             "drifts farther than this from its "
                             "planted spot.")
    # --- Camera angle offset (manual pitch/roll) ---
    parser.add_argument("--camera-pitch", type=float, default=0.0,
                        help="Pitch correction in degrees (rotation about "
                             "camera X). Positive = tilts the actor "
                             "backward.")
    parser.add_argument("--camera-roll", type=float, default=0.0,
                        help="Roll correction in degrees (rotation about "
                             "camera Z).")
    parser.add_argument("--height-offset", type=float, default=0.0,
                        help="Constant vertical offset in meters, added to "
                             "the Hips channel after grounding and "
                             "footskate cleanup. Manual floor-level "
                             "correction: positive raises the whole "
                             "performance, negative lowers it.")
    # --- Head-bone-only angle offset (additional manual correction) ---
    # Same axes as --camera-pitch/--camera-roll but applied only to
    # the HEAD bone (and its face-landmark descendants, which ride
    # along via subtree cancellation in compute_local_rotations). Use
    # when the camera offset is already correct for the body but the
    # head has a residual tilt/roll that wasn't captured cleanly.
    parser.add_argument("--head-pitch", type=float, default=0.0,
                        help="Head-only pitch correction in degrees. "
                             "Same axis as --camera-pitch but rotates "
                             "only the Head bone, stacking on top of "
                             "the body camera offset.")
    parser.add_argument("--head-roll", type=float, default=0.0,
                        help="Head-only roll correction in degrees. "
                             "Same axis as --camera-roll but Head-only.")
    return parser


def _resolve_export_mode(args):
    """Pick export mode from --export or auto-detect from --input/--face."""
    has_body = bool(args.input)
    has_face = bool(args.face)

    if args.export:
        export_mode = args.export
    elif has_body and has_face:
        export_mode = "both"
    elif has_body:
        export_mode = "body"
    elif has_face:
        export_mode = "face"
    else:
        print("ERROR: Provide --input (body NPZ), --face (face NPZ), or both.")
        print("       Use --export to choose: body | face | both | hands | root")
        sys.exit(1)

    if export_mode in ("body", "both", "hands", "root") and not has_body:
        print(f"ERROR: --export {export_mode} requires --input (body NPZ)")
        sys.exit(1)
    if export_mode in ("face", "both") and not has_face:
        print(f"ERROR: --export {export_mode} requires --face (face NPZ)")
        sys.exit(1)
    # template mode is self-contained — no NPZ inputs needed.

    return export_mode


def _validate_smoothing_flags(args):
    """In-place: warn when scipy is missing for filters that need it.

    Body/root values are in seconds and get converted to frames
    later (inside modes.py) using the actual capture fps. The
    conversion helpers handle window-oddness and minimum-frame floors,
    so no clamping is needed here.
    """
    if args.smooth > 0 and not HAS_SCIPY:
        print("WARNING: --smooth requires scipy (pip install scipy)")
        print("         Continuing without smoothing.")
        args.smooth = 0

    if args.smooth_root > 0 and not HAS_SCIPY:
        print("WARNING: --smooth-root requires scipy (pip install scipy)")
        args.smooth_root = 0

    # --smooth-face validated at smooth_face_data() call (scipy check
    # + Nyquist check) since it needs face-data fps.


def _print_summary(args, export_mode, info):
    """Trailing summary printed after the export completes."""
    print(f"\nY-up BVH. Blender addon handles import orientation.")
    print(f"Export mode: {export_mode}")

    if export_mode in ("body", "both"):
        print(f"Rest pose: {args.rest_pose}")
        if info.get("cam_t_present"):
            if args.root_motion:
                print(f"Root.X/Z (locomotion): ON; "
                      f"Hips.Y (vertical body): ON")
            else:
                print(f"Root.X/Z (locomotion): OFF "
                      f"(--root-motion flag not set); "
                      f"Hips.Y (vertical body): ON")
        else:
            print(f"Root motion: none (no cam_t)")
        if not args.no_center:
            print(f"Root centered at origin: ON")
        if args.smooth_root > 0:
            print(f"Root smoothing: Gaussian sigma={args.smooth_root:.2f} s (depth 2x)")
        if args.smooth > 0:
            print(f"Rotation smoothing: Savitzky-Golay window={args.smooth:.2f} s")

    if export_mode in ("face", "both"):
        face_bone_data = info.get("face_bone_data")
        if face_bone_data is not None:
            print(f"Face bones: {len(face_bone_data['order'])} (boost={args.face_boost:.1f}x)")

    if export_mode == "hands":
        print(f"Hands armature: standalone (Root + 2 hands + fingers)")
        if args.finger_boost > 1.0:
            print(f"Finger boost: {args.finger_boost:.1f}x")

    if export_mode == "root":
        print(f"Root armature: standalone (single Root bone, cam_t motion)")


def main():
    args = _build_argparser().parse_args()
    export_mode = _resolve_export_mode(args)
    print(f"Export mode: {export_mode}")

    # -- Load face calibration -----------------------------------------
    face_calibration = None
    if export_mode in ("face", "both"):
        cal_path = args.face_calibration
        if not cal_path:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            cal_path = os.path.join(script_dir, "face_calibration.json")
        face_calibration = load_face_calibration(cal_path)
        n_cal_bones = len(face_calibration.get("skeleton", []))
        n_cal_shapes = len(face_calibration.get("calibration", {}))
        print(f"Face calibration: {n_cal_bones} bones, {n_cal_shapes} shapes")

    _validate_smoothing_flags(args)

    # -- Load face data (if needed) ------------------------------------
    face_data = None
    if export_mode in ("face", "both"):
        if not os.path.isfile(args.face):
            print(f"ERROR: Face NPZ not found: {args.face}")
            sys.exit(1)

        face_data = load_face_data(args.face)

        # Face refine: recompute weak MediaPipe channels from landmarks
        # + capture-time mouth pixel stats. Runs on RAW values, before
        # smoothing, so the smoothing window applies to the refined
        # signal. The NPZ itself is never modified.
        if args.face_refine != "off":
            from pipeline.face_refine import apply_face_refine
            apply_face_refine(face_data, mode=args.face_refine,
                              neutral_frame=args.face_neutral_frame,
                              report_path=args.face_refine_report)

        # Face smoothing: centered Savitzky-Golay, window in seconds.
        # 0 disables. See pipeline/bvh_math.py for the filter.
        if args.smooth_face > 0:
            smooth_face_data(face_data, window_s=float(args.smooth_face))

        # Face denoise: per-region dead-zone, applied AFTER smoothing
        # so the threshold operates on already-filtered values.
        # Build the position-ordered name list once; used by both
        # denoise and amp passes if either is enabled.
        bs_names_ordered = None
        if args.face_denoise_enabled or args.face_amp_enabled:
            bs_names_ordered = [None] * len(face_data["bs_name_to_idx"])
            for n, i in face_data["bs_name_to_idx"].items():
                bs_names_ordered[int(i)] = str(n)

        if args.face_denoise_enabled:
            from pipeline.bvh_math import denoise_face_blendshapes
            region_thresholds = {
                r: float(getattr(args, f"face_denoise_{r}", 0.0))
                for r in ("jaw", "gaze", "lids", "brows", "lips", "cheeks", "nose", "tongue")
            }
            denoise_face_blendshapes(face_data, region_thresholds,
                                     bs_names_ordered)

        # Face amp: per-region multipliers, applied AFTER denoise so a
        # dead-zoned value stays at 0 regardless of amplification.
        if args.face_amp_enabled:
            from pipeline.bvh_math import amplify_face_blendshapes
            region_multipliers = {
                r: float(getattr(args, f"face_amp_{r}", 1.0))
                for r in ("jaw", "gaze", "lids", "brows", "lips", "cheeks", "nose", "tongue")
            }
            amplify_face_blendshapes(face_data, region_multipliers,
                                     bs_names_ordered)

    # -- Dispatch ------------------------------------------------------
    if export_mode in ("body", "both", "template"):
        info = run_body(args, export_mode, face_data, face_calibration)
    elif export_mode == "face":
        info = run_face(args, face_data, face_calibration)
    elif export_mode == "hands":
        info = run_hands(args)
    elif export_mode == "root":
        info = run_root(args)
    else:  # unreachable — _resolve_export_mode validates
        print(f"ERROR: unknown export mode: {export_mode}")
        sys.exit(1)

    _print_summary(args, export_mode, info)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"\n[ERROR] BVH generation failed: {exc}", flush=True)
        raise SystemExit(1)
