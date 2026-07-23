# smooth_face_npz.py
# ==================
# Apply face smoothing (Butterworth filtfilt) to a face_frames.npz and
# write a smoothed copy. Invoked by the Blender addon's ARKit Apply
# operator: Blender's bundled Python lacks scipy, so smoothing is
# delegated to the addon venv (which has scipy), the same way BVH
# export does it via npz_to_bvh.py.
#
# Usage:
#   python pipeline/smooth_face_npz.py <input.npz> <output.npz>
#                                       [--smooth-face SECONDS]
#
# The output NPZ is a drop-in replacement for the input: same keys,
# same dtypes, only blendshapes/landmarks/eye_gaze values changed.

import argparse
import os
import sys

import numpy as np

# Make the project root importable (parent of pipeline/) so `pipeline.*`
# resolves when this file is invoked as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.face_export import load_face_data
from pipeline.bvh_math import smooth_face_data


def main():
    p = argparse.ArgumentParser(
        description="Apply face smoothing to a face_frames.npz and write "
                    "a smoothed copy.")
    p.add_argument("input", help="Input face NPZ (raw, from save_face_data.py)")
    p.add_argument("output", help="Output face NPZ (smoothed copy)")
    p.add_argument("--smooth-face", type=float, default=0.23,
                   help="Savitzky-Golay window in seconds (higher = more "
                        "smoothing). Converted to a frame window at "
                        "runtime using the face NPZ's fps. 0 = "
                        "pass-through.")
    p.add_argument("--face-refine",
                   choices=["off", "geometric", "full"],
                   default=os.environ.get("BLENDCAP_FACE_REFINE", "full"),
                   help="Recompute weak MediaPipe channels before "
                        "smoothing (see npz_to_bvh.py --face-refine). "
                        "Same default and env override.")
    p.add_argument("--face-neutral-frame", type=int, default=-1,
                   help="Manual neutral frame index for --face-refine "
                        "(-1 = auto-detect).")
    args = p.parse_args()

    # [PROGRESS N] lines drive the addon-side progress bar (same format
    # the capture/BVH subprocesses use); flush so they arrive live.
    print("[PROGRESS 10] loading", flush=True)
    face_data = load_face_data(args.input)

    # Refine BEFORE smoothing, exactly like the BVH export path, so the
    # ARKit Apply operator sees the same corrected channels.
    if args.face_refine != "off":
        print("[PROGRESS 25] refining", flush=True)
        from pipeline.face_refine import apply_face_refine
        apply_face_refine(face_data, mode=args.face_refine,
                          neutral_frame=args.face_neutral_frame)

    if args.smooth_face > 0:
        print("[PROGRESS 70] smoothing", flush=True)
        smooth_face_data(face_data, window_s=float(args.smooth_face))
    print("[PROGRESS 85] writing", flush=True)

    # Round-trip the original NPZ so unused keys (face_bbox,
    # frame_indices, etc.) pass through unchanged. blendshape names ARE
    # rewritten: the refine stage may have appended channels
    # (tongueOut), and the names array must match the widened values.
    original = np.load(args.input, allow_pickle=True)
    out = {k: original[k] for k in original.files}
    out["face_blendshapes"] = face_data["blendshapes"].astype(np.float32)
    out["face_landmarks"] = face_data["landmarks"].astype(np.float32)
    out["eye_gaze"] = face_data["eye_gaze"].astype(np.float32)
    if "blendshape_names" in face_data:
        out["blendshape_names"] = np.array(face_data["blendshape_names"])
    np.savez(args.output, **out)
    print(f"Smoothed face NPZ written: {args.output}")


if __name__ == "__main__":
    main()
