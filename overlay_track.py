# overlay_track.py — Draws keypoint overlay on video using NPZ tracking data.
# Body only:  python overlay_track.py --video X.mp4 --input mhr_frames.npz
# Face only:  python overlay_track.py --video X.mp4 --face face_frames.npz
# Body+face:  python overlay_track.py --video X.mp4 --input mhr_frames.npz --face face_frames.npz

import os, sys, argparse
import numpy as np
import cv2
from pathlib import Path

# MHR70 keypoint name -> index
KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_hip": 9, "right_hip": 10, "left_knee": 11, "right_knee": 12,
    "left_ankle": 13, "right_ankle": 14,
    "left_big_toe": 15, "left_small_toe": 16, "left_heel": 17,
    "right_big_toe": 18, "right_small_toe": 19, "right_heel": 20,
    "right_thumb4": 21, "right_thumb3": 22, "right_thumb2": 23,
    "right_thumb_third_joint": 24,
    "right_forefinger4": 25, "right_forefinger3": 26, "right_forefinger2": 27,
    "right_forefinger_third_joint": 28,
    "right_middle_finger4": 29, "right_middle_finger3": 30,
    "right_middle_finger2": 31, "right_middle_finger_third_joint": 32,
    "right_ring_finger4": 33, "right_ring_finger3": 34,
    "right_ring_finger2": 35, "right_ring_finger_third_joint": 36,
    "right_pinky_finger4": 37, "right_pinky_finger3": 38,
    "right_pinky_finger2": 39, "right_pinky_finger_third_joint": 40,
    "right_wrist": 41,
    "left_thumb4": 42, "left_thumb3": 43, "left_thumb2": 44,
    "left_thumb_third_joint": 45,
    "left_forefinger4": 46, "left_forefinger3": 47, "left_forefinger2": 48,
    "left_forefinger_third_joint": 49,
    "left_middle_finger4": 50, "left_middle_finger3": 51,
    "left_middle_finger2": 52, "left_middle_finger_third_joint": 53,
    "left_ring_finger4": 54, "left_ring_finger3": 55,
    "left_ring_finger2": 56, "left_ring_finger_third_joint": 57,
    "left_pinky_finger4": 58, "left_pinky_finger3": 59,
    "left_pinky_finger2": 60, "left_pinky_finger_third_joint": 61,
    "left_wrist": 62,
    "left_olecranon": 63, "right_olecranon": 64,
    "left_cubital_fossa": 65, "right_cubital_fossa": 66,
    "left_acromion": 67, "right_acromion": 68,
    "neck": 69,
}

# Bones: (idx_a, idx_b, BGR color)
_G, _O, _Y = (0, 255, 0), (0, 128, 255), (255, 153, 51)
_P, _T, _B = (255, 153, 255), (255, 178, 102), (51, 51, 255)
_k = KP.__getitem__  # shorthand

BONES = [
    # Body
    (_k("left_ankle"), _k("left_knee"), _G), (_k("left_knee"), _k("left_hip"), _G),
    (_k("right_ankle"), _k("right_knee"), _O), (_k("right_knee"), _k("right_hip"), _O),
    (_k("left_hip"), _k("right_hip"), _Y), (_k("left_shoulder"), _k("left_hip"), _Y),
    (_k("right_shoulder"), _k("right_hip"), _Y), (_k("left_shoulder"), _k("right_shoulder"), _Y),
    (_k("left_shoulder"), _k("left_elbow"), _G), (_k("right_shoulder"), _k("right_elbow"), _O),
    (_k("left_elbow"), _k("left_wrist"), _G), (_k("right_elbow"), _k("right_wrist"), _O),
    # Face
    (_k("left_eye"), _k("right_eye"), _Y), (_k("nose"), _k("left_eye"), _Y),
    (_k("nose"), _k("right_eye"), _Y), (_k("left_eye"), _k("left_ear"), _Y),
    (_k("right_eye"), _k("right_ear"), _Y), (_k("left_ear"), _k("left_shoulder"), _Y),
    (_k("right_ear"), _k("right_shoulder"), _Y),
    # Feet
    (_k("left_ankle"), _k("left_big_toe"), _G), (_k("left_ankle"), _k("left_small_toe"), _G),
    (_k("left_ankle"), _k("left_heel"), _G),
    (_k("right_ankle"), _k("right_big_toe"), _O), (_k("right_ankle"), _k("right_small_toe"), _O),
    (_k("right_ankle"), _k("right_heel"), _O),
    # Left hand
    *[(_k(f"left_{f}"), _k(f"left_{t}"), c) for f, t, c in [
        ("wrist", "thumb_third_joint", _O), ("thumb_third_joint", "thumb2", _O),
        ("thumb2", "thumb3", _O), ("thumb3", "thumb4", _O),
        ("wrist", "forefinger_third_joint", _P), ("forefinger_third_joint", "forefinger2", _P),
        ("forefinger2", "forefinger3", _P), ("forefinger3", "forefinger4", _P),
        ("wrist", "middle_finger_third_joint", _T), ("middle_finger_third_joint", "middle_finger2", _T),
        ("middle_finger2", "middle_finger3", _T), ("middle_finger3", "middle_finger4", _T),
        ("wrist", "ring_finger_third_joint", _B), ("ring_finger_third_joint", "ring_finger2", _B),
        ("ring_finger2", "ring_finger3", _B), ("ring_finger3", "ring_finger4", _B),
        ("wrist", "pinky_finger_third_joint", _G), ("pinky_finger_third_joint", "pinky_finger2", _G),
        ("pinky_finger2", "pinky_finger3", _G), ("pinky_finger3", "pinky_finger4", _G),
    ]],
    # Right hand
    *[(_k(f"right_{f}"), _k(f"right_{t}"), c) for f, t, c in [
        ("wrist", "thumb_third_joint", _O), ("thumb_third_joint", "thumb2", _O),
        ("thumb2", "thumb3", _O), ("thumb3", "thumb4", _O),
        ("wrist", "forefinger_third_joint", _P), ("forefinger_third_joint", "forefinger2", _P),
        ("forefinger2", "forefinger3", _P), ("forefinger3", "forefinger4", _P),
        ("wrist", "middle_finger_third_joint", _T), ("middle_finger_third_joint", "middle_finger2", _T),
        ("middle_finger2", "middle_finger3", _T), ("middle_finger3", "middle_finger4", _T),
        ("wrist", "ring_finger_third_joint", _B), ("ring_finger_third_joint", "ring_finger2", _B),
        ("ring_finger2", "ring_finger3", _B), ("ring_finger3", "ring_finger4", _B),
        ("wrist", "pinky_finger_third_joint", _G), ("pinky_finger_third_joint", "pinky_finger2", _G),
        ("pinky_finger2", "pinky_finger3", _G), ("pinky_finger3", "pinky_finger4", _G),
    ]],
]
del _k

FINGER_START = 25  # bones at index >= this are fingers (drawn thinner)

# Pre-extract bone arrays for vectorized coordinate lookup
_BONE_A = np.array([b[0] for b in BONES], dtype=np.intp)
_BONE_B = np.array([b[1] for b in BONES], dtype=np.intp)
_BONE_COLORS = [b[2] for b in BONES]

# Face mesh contours (MediaPipe 478-point)
FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,
             400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10]
LEFT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33]
RIGHT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,362]
LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,61]
LIPS_INNER = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191,78]
LEFT_EYEBROW = [46,53,52,65,55,107,66,105,63,70]
RIGHT_EYEBROW = [276,283,282,295,285,336,296,334,293,300]
NOSE_BRIDGE = [168,6,197,195,5,4]
NOSE_TIP = [48,64,98,97,2,326,327,294,278,440,275,4,45,220,115,48]

# Pre-build contour segment pairs: list of (array_of_pt_a_indices, array_of_pt_b_indices, color)
_CONTOUR_SEGS = []
for _idx_list, _color in [
    (FACE_OVAL, (200,180,130)), (LEFT_EYE, (0,255,200)), (RIGHT_EYE, (0,255,200)),
    (LIPS_OUTER, (180,105,255)), (LIPS_INNER, (180,105,255)),
    (LEFT_EYEBROW, (200,200,200)), (RIGHT_EYEBROW, (200,200,200)),
    (NOSE_BRIDGE, (200,200,200)), (NOSE_TIP, (200,200,200)),
]:
    _a = np.array(_idx_list[:-1], dtype=np.intp)
    _b = np.array(_idx_list[1:], dtype=np.intp)
    _CONTOUR_SEGS.append((_a, _b, _color))

# HUD config
HUD_BLENDSHAPES = [
    "jawOpen", "eyeBlinkLeft", "eyeBlinkRight", "mouthSmileLeft", "mouthSmileRight",
    "browInnerUp", "browDownLeft", "browDownRight", "mouthPucker", "cheekPuff",
]
HUD_LABELS = ["jaw","blinkL","blinkR","smileL","smileR","browUp","browDnL","browDnR","pucker","puff"]

FACE_BOX_OK = (255, 255, 0)
FACE_BOX_MISS = (0, 0, 255)
FACE_HUD_BAR = (255, 255, 0)
FACE_HUD_BG = (30, 30, 30)


# Drawing functions

def draw_skeleton(frame, pts_2d, thickness=2):
    """Draw skeleton with bones and keypoint dots."""
    h, w = frame.shape[:2]
    pts = pts_2d.astype(np.int32)

    # Vectorized coordinate extraction for all bones at once
    ax, ay = pts[_BONE_A, 0], pts[_BONE_A, 1]
    bx, by = pts[_BONE_B, 0], pts[_BONE_B, 1]

    # Vectorized bounds check
    m = w
    valid = ((ax > -m) & (ax < w + m) & (ay > -m) & (ay < h + m) &
             (bx > -m) & (bx < w + m) & (by > -m) & (by < h + m))

    thin = max(1, thickness - 1)
    for i in np.flatnonzero(valid):
        t = thin if i >= FINGER_START else thickness
        cv2.line(frame, (int(ax[i]), int(ay[i])), (int(bx[i]), int(by[i])),
                 _BONE_COLORS[i], t, cv2.LINE_AA)

    # Dots — vectorized bounds check
    px, py = pts[:, 0], pts[:, 1]
    dot_valid = (px > -w) & (px < 2*w) & (py > -h) & (py < 2*h)
    for j in np.flatnonzero(dot_valid):
        cv2.circle(frame, (int(px[j]), int(py[j])), 3, (255,255,255), -1, cv2.LINE_AA)


def draw_face_box(frame, bbox, detected):
    """Draw face crop bounding box."""
    x1, y1, x2, y2 = bbox
    if x2 - x1 < 5 or y2 - y1 < 5:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2),
                  FACE_BOX_OK if detected else FACE_BOX_MISS, 1, cv2.LINE_AA)


def draw_face_landmarks(frame, landmarks_px, thickness=1):
    """Draw face mesh contours from 478 landmark pixel coordinates."""
    n_lm = len(landmarks_px)
    pts = landmarks_px.astype(np.int32)
    for seg_a, seg_b, color in _CONTOUR_SEGS:
        # Skip segments referencing out-of-range landmarks
        mask = (seg_a < n_lm) & (seg_b < n_lm)
        for idx in np.flatnonzero(mask):
            a, b = int(seg_a[idx]), int(seg_b[idx])
            cv2.line(frame, (pts[a, 0], pts[a, 1]), (pts[b, 0], pts[b, 1]),
                     color, thickness, cv2.LINE_AA)


def draw_blendshape_hud(frame, blendshapes, hud_indices, hud_labels):
    """Draw compact blendshape bar chart in top-right corner.
    hud_indices: pre-built list of (label, bs_index) pairs."""
    if not hud_indices:
        return

    h, w = frame.shape[:2]
    n_bars = len(hud_indices)
    bar_h = max(12, int(h * 0.018))
    label_w = max(50, int(w * 0.045))
    bar_max_w = max(80, int(w * 0.10))
    pad = max(4, int(h * 0.005))
    row_h = bar_h + pad

    hud_w = label_w + bar_max_w + pad * 3
    hud_h = row_h * n_bars + pad * 2
    margin = max(10, int(w * 0.01))
    hud_x = w - hud_w - margin
    hud_y = margin

    # ROI-based alpha blend (avoids copying entire frame)
    y1 = max(0, hud_y)
    y2 = min(h, hud_y + hud_h)
    x1 = max(0, hud_x)
    x2 = min(w, hud_x + hud_w)
    if y2 > y1 and x2 > x1:
        roi = frame[y1:y2, x1:x2]
        bg = np.full_like(roi, FACE_HUD_BG)
        cv2.addWeighted(bg, 0.7, roi, 0.3, 0, roi)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.3, bar_h / 30.0)

    for i, (label, bs_idx) in enumerate(hud_indices):
        row_y = hud_y + pad + i * row_h
        text_y = row_y + bar_h - max(2, int(bar_h * 0.15))

        cv2.putText(frame, label, (hud_x + pad, text_y),
                    font, font_scale, (200,200,200), 1, cv2.LINE_AA)

        bar_x = hud_x + label_w + pad
        cv2.rectangle(frame, (bar_x, row_y),
                      (bar_x + bar_max_w, row_y + bar_h), (60,60,60), -1)

        fill_w = max(0, min(bar_max_w, int(float(blendshapes[bs_idx]) * bar_max_w)))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, row_y),
                          (bar_x + fill_w, row_y + bar_h), FACE_HUD_BAR, -1)


def _build_hud_indices(blendshape_names):
    """Build HUD label/index pairs once from blendshape names."""
    name_to_idx = {n: i for i, n in enumerate(blendshape_names)}
    return [(label, name_to_idx[name])
            for name, label in zip(HUD_BLENDSHAPES, HUD_LABELS)
            if name in name_to_idx]


def main():
    parser = argparse.ArgumentParser(description="Overlay keypoint tracking on video")
    parser.add_argument("--video", required=True, help="Original video file")
    parser.add_argument("--input", default="", help="Body NPZ from save_mhr_data.py (optional if --face)")
    parser.add_argument("--face", default="", help="Face NPZ from save_face_data.py (optional if --input)")
    parser.add_argument("--output", default="overlay.mp4", help="Output video path")
    parser.add_argument("--thickness", type=int, default=2, help="Dot size (default: 2)")
    args = parser.parse_args()

    if not args.input and not args.face:
        print("ERROR: Provide at least one of --input (body) or --face (face).")
        sys.exit(1)

    video = Path(args.video)
    if not video.exists():
        print(f"ERROR: Video not found: {video}")
        sys.exit(1)

    # Load body data
    keypoints_2d = None
    frame_indices = None
    fps = None

    if args.input:
        if not os.path.isfile(args.input):
            print(f"ERROR: Body NPZ not found: {args.input}")
            sys.exit(1)
        print(f"Loading body data from {args.input}")
        npz = np.load(args.input)
        if "keypoints_2d" not in npz:
            print("ERROR: keypoints_2d not found in NPZ.")
            print("  Re-run save_mhr_data.py to generate an NPZ with 2D keypoint data.")
            sys.exit(1)
        keypoints_2d = npz["keypoints_2d"]
        fps = float(npz["fps"])
        print(f"  {keypoints_2d.shape[0]} frames, {keypoints_2d.shape[1]} keypoints, {fps:.1f} FPS")
        frame_indices = npz["frame_indices"].astype(int) if "frame_indices" in npz else np.arange(keypoints_2d.shape[0])
        if frame_indices is not None:
            print(f"  Frame indices: {frame_indices[0]} to {frame_indices[-1]}")

    # Load face data
    face_data = None
    hud_indices = []

    if args.face:
        if not os.path.isfile(args.face):
            print(f"ERROR: Face NPZ not found: {args.face}")
            sys.exit(1)
        print(f"Loading face data from {args.face}")
        fnpz = np.load(args.face, allow_pickle=False)

        face_blendshapes = fnpz["face_blendshapes"]
        face_detected = fnpz["face_detected"]
        face_frame_indices = fnpz["frame_indices"].astype(int)
        face_landmarks = fnpz["face_landmarks"] if "face_landmarks" in fnpz else None
        blendshape_names = list(fnpz["blendshape_names"])

        n_face, n_bs = face_blendshapes.shape[0], (face_blendshapes.shape[1] if face_blendshapes.ndim > 1 else 0)
        has_lm = face_landmarks is not None
        print(f"  {n_face} frames, {n_bs} blendshapes, {int(face_detected.sum())} with face, landmarks={'yes' if has_lm else 'no'}")

        face_lookup = {int(v): i for i, v in enumerate(face_frame_indices)}
        hud_indices = _build_hud_indices(blendshape_names)

        face_data = {
            "blendshapes": face_blendshapes, "landmarks": face_landmarks,
            "bbox": fnpz["face_bbox"], "detected": face_detected,
            "lookup": face_lookup, "frame_indices": face_frame_indices,
        }

        if frame_indices is None:
            frame_indices = face_frame_indices
            fps = float(fnpz["fps"])

    # Open video
    print(f"Reading video: {video}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video}")
        print("  This video format is not supported.")
        print("  Please convert to H.264 MP4 and try again.")
        sys.exit(1)

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {vid_total} frames, {width}x{height}, {vid_fps:.1f} FPS")

    # Scale down (cap long edge at 1920, otherwise native)
    if max(width, height) > 1920:
        scale = 1920 / max(width, height)
        out_w, out_h = int(width * scale) // 2 * 2, int(height * scale) // 2 * 2
    else:
        out_w, out_h = width, height

    # Output writer
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), vid_fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"ERROR: Cannot create output video {args.output}")
        sys.exit(1)

    has_body = keypoints_2d is not None
    has_face = face_data is not None
    mode = ("body + face" if has_body and has_face else "body only" if has_body else "face only")
    print(f"  Output: {args.output} ({out_w}x{out_h}) [{mode}]")

    # Build body frame lookup
    body_lookup = {int(v): i for i, v in enumerate(frame_indices)} if has_body and args.input else {}

    # Frame range (union of body and face)
    first_vid_frame = int(frame_indices[0])
    last_vid_frame = int(frame_indices[-1])
    if face_data is not None and len(face_data["frame_indices"]) > 0:
        first_vid_frame = min(first_vid_frame, int(face_data["frame_indices"][0]))
        last_vid_frame = max(last_vid_frame, int(face_data["frame_indices"][-1]))

    render_total = last_vid_frame - first_vid_frame + 1
    n_body = len(body_lookup)
    n_face_frames = len(face_data["lookup"]) if face_data else 0
    print(f"\nRendering {render_total} frames (video frames {first_vid_frame}-{last_vid_frame})"
          f" ({n_body} body, {n_face_frames} face)...")

    if first_vid_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_vid_frame)

    need_resize = (out_w, out_h) != (width, height)
    tracked = 0
    face_drawn = 0
    overlay_done = 0
    overlay_pct = -1
    render_span = max(1, render_total)

    for vid_f in range(first_vid_frame, last_vid_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break

        # Body skeleton
        body_idx = body_lookup.get(vid_f)
        if body_idx is not None:
            draw_skeleton(frame, keypoints_2d[body_idx], args.thickness)
            tracked += 1

        # Face overlay
        face_idx = face_data["lookup"].get(vid_f) if face_data else None
        if face_idx is not None:
            bbox = face_data["bbox"][face_idx]
            detected = bool(face_data["detected"][face_idx])
            draw_face_box(frame, bbox, detected)
            if detected:
                if face_data["landmarks"] is not None:
                    draw_face_landmarks(frame, face_data["landmarks"][face_idx])
                draw_blendshape_hud(frame, face_data["blendshapes"][face_idx], hud_indices, HUD_LABELS)
                face_drawn += 1

        has_track = body_idx is not None or face_idx is not None
        cv2.putText(frame, f"{vid_f}/{last_vid_frame}{'' if has_track else ' [no track]'}",
                    (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200,200,200), 1, cv2.LINE_AA)

        if need_resize:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        writer.write(frame)

        # Emit [PROGRESS N] the capture worker maps into the overlay
        # step's slice of the bar. N is RENDER-relative (0->100 over the
        # frames actually rendered) so a trimmed clip starting at a high
        # frame index still reports from 0 - the old "Frame vid_f/last"
        # print mapped to vid_f/last_vid_frame, which started mid-bar for
        # those. Deduped by integer percent so long clips don't flood
        # stdout.
        overlay_done += 1
        pct = int(overlay_done * 100 / render_span)
        if pct != overlay_pct:
            overlay_pct = pct
            print(f"[PROGRESS {pct}] Overlay {overlay_done}/{render_span}",
                  flush=True)

    cap.release()
    writer.release()

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nDone: {args.output} ({size_mb:.1f} MB)")
    print(f"  {render_total} frames (video {first_vid_frame}-{last_vid_frame}), {vid_fps:.1f} FPS")
    if has_body:
        print(f"  {tracked} frames with body keypoints")
    if has_face:
        print(f"  {face_drawn} frames with face blendshapes")


if __name__ == "__main__":
    main()
