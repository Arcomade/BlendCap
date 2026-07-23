# save_face_data.py
# ==================
# Extracts ARKit face blendshapes from video using MediaPipe Face Mesh.
#
# Two modes:
#   WITH --body: Uses head keypoints from body NPZ to crop face regions.
#                Better quality on full-body shots (tighter crop = better input).
#                Validates that body NPZ matches the video before using crops.
#   WITHOUT --body: Uses MediaPipe Pose for lightweight head localization
#                   (~30 fps on CPU) to get head crops comparable to body
#                   tracking. Falls back to face detector chain if pose fails.
#
# Requires: pip install mediapipe
#
# Always saves raw (unsmoothed) data. Smoothing is done at BVH export
# time via npz_to_bvh.py --smooth-face, so you can try different
# settings without re-running inference.
#
# Run:      python save_face_data.py --video X.mp4
#           python save_face_data.py --video X.mp4 --body mhr_frames.npz
# Preview:  python save_face_data.py --video X.mp4 --preview
#           python save_face_data.py --video X.mp4 --body mhr_frames.npz --preview
# Out:      face_frames.npz (inference) or face_preview.mp4 (preview)

import sys
import os
import argparse
import bisect
import json
import numpy as np
import cv2
from pathlib import Path
import urllib.request

# ── Offline operation ─────────────────────────────────────────────
# Keep offline parity with save_mhr_data.py. The face path uses MediaPipe,
# which loads from local .task/.tflite files the installer pre-fetches into
# checkpoints/mediapipe/, so it has no Hugging Face / Ultralytics dependency
# today — but force offline mode defensively so no library added later phones
# home or stalls on a connectivity probe at capture time.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("YOLO_OFFLINE", "true")

# ── Project root = directory containing this script ──────────
SCRIPT_DIR = Path(__file__).parent.absolute()
CKPT_DIR = SCRIPT_DIR / "checkpoints"
MEDIAPIPE_DIR = CKPT_DIR / "mediapipe"

# Lazy-loaded mediapipe module
_mp = None

def _get_mp():
    """Import mediapipe once, cache globally."""
    global _mp
    if _mp is None:
        try:
            import mediapipe
            _mp = mediapipe
        except ImportError:
            print("ERROR: mediapipe not installed.")
            print("  Run: pip install mediapipe")
            sys.exit(1)
    return _mp


def emit_progress(pct, msg=""):
    """Emit structured progress line for UI parsing. pct is 0-100 within this script's run."""
    print(f"[PROGRESS {int(pct)}] {msg}", flush=True)


def _check_multi_face(count, warned, label=""):
    """Increment multi-face counter; warn once at threshold 3. Returns (count, warned)."""
    count += 1
    if not warned and count >= 3:
        print("\n  Multiple faces detected, or face tracking is flickering\n"
              "  between frames. BlendCap is designed to track a single\n"
              "  person; using IoU bbox continuity to pick one face\n"
              "  and consolidate flickering tracking.")
        print("[WARNING] Multi-face or flickering tracking; using IoU continuity.",
              flush=True)
        emit_progress(20 if label == "preview" else 30,
                      "WARNING: Multiple faces detected")
        warned = True
    return count, warned


def _bbox_iou(a, b):
    """IoU between two [x_min, y_min, x_max, y_max] bboxes. 0 if either is None."""
    if a is None or b is None:
        return 0.0
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


# ── Model URLs and filenames ─────────────────────────────────
_MODELS = {
    "face": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "face_landmarker.task",
    ),
    "detector": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
        "blaze_face_short_range.tflite",
    ),
    "pose": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        "pose_landmarker_lite.task",
    ),
}

# ── Head keypoint indices ─────────────────────────────────────
# MHR70: nose=0, left_eye=1, right_eye=2, left_ear=3, right_ear=4
HEAD_KP_INDICES = [0, 1, 2, 3, 4]
# MediaPipe Pose: nose=0, left_eye=2, right_eye=5, left_ear=7, right_ear=8
POSE_HEAD_INDICES = [0, 2, 5, 7, 8]

# Minimum long edge for detection upscaling
_DETECT_MIN_EDGE = 1080


# ── Model download (consolidated) ────────────────────────────

def _download_model(key, fatal=False):
    """Download a MediaPipe model if not present. Returns path or None."""
    url, filename = _MODELS[key]
    model_path = MEDIAPIPE_DIR / filename
    if model_path.exists():
        size = model_path.stat().st_size
        label = f"{size / (1024*1024):.1f} MB" if size > 1024*1024 else f"{size / 1024:.0f} KB"
        print(f"  {key.title()} model: {model_path} ({label})")
        return model_path

    MEDIAPIPE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {key} model...")
    print(f"  URL: {url}")
    try:
        urllib.request.urlretrieve(url, str(model_path))
    except Exception as e:
        if fatal:
            print(f"ERROR: Failed to download {key} model: {e}")
            print(f"  Download manually and place at: {model_path}")
            sys.exit(1)
        print(f"  WARNING: Failed to download {key} model: {e}")
        return None

    size = model_path.stat().st_size
    label = f"{size / (1024*1024):.1f} MB" if size > 1024*1024 else f"{size / 1024:.0f} KB"
    print(f"  Downloaded: {model_path} ({label})")
    return model_path


# ── Model loading ─────────────────────────────────────────────

def load_face_landmarker(model_path):
    """Load MediaPipe FaceLandmarker in VIDEO mode with blendshape output."""
    mp = _get_mp()
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    print("  FaceLandmarker ready (VIDEO mode, blendshapes enabled)")
    return landmarker


def load_pose_estimator():
    """
    Load MediaPipe PoseLandmarker (Tasks API) in VIDEO mode for
    lightweight head localization with temporal tracking. VIDEO mode
    maintains state between frames, predicting landmark positions
    through brief occlusions (e.g. arm crossing face).
    Runs at 30+ fps on CPU. Returns PoseLandmarker object or None.
    """
    model_path = _download_model("pose")
    if model_path is None:
        return None
    try:
        mp = _get_mp()
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            num_poses=1,
        )
        pose = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        print("  MediaPipe Pose estimator ready (VIDEO mode, CPU)")
        return pose
    except Exception as e:
        print(f"  WARNING: Could not load MediaPipe Pose: {e}")
        return None


def load_face_detector():
    """
    Load face detectors for standalone pre-cropping.
    Loads both MediaPipe BlazeFace (primary) and OpenCV Haar cascade (fallback).
    Returns (detectors_dict, "dual") or (None, None).
    """
    detectors = {}

    detector_path = _download_model("detector")
    if detector_path is not None:
        try:
            mp = _get_mp()
            options = mp.tasks.vision.FaceDetectorOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(detector_path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                min_detection_confidence=0.35,
            )
            detectors["mp"] = mp.tasks.vision.FaceDetector.create_from_options(options)
            print("  BlazeFace detector ready")
        except Exception as e:
            print(f"  WARNING: MediaPipe FaceDetector failed: {e}")

    haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if not haar.empty():
        detectors["haar"] = haar
        print("  Haar cascade detector ready")

    if not detectors:
        print("  WARNING: Could not load any face detector")
        return None, None

    labels = {"mp": "BlazeFace", "haar": "Haar"}
    print(f"  Face detector ready ({' + '.join(labels[k] for k in detectors)}"
          f"{' dual' if len(detectors) > 1 else ''})")
    return detectors, "dual"


# ── Video helpers ─────────────────────────────────────────────

def open_video(video_path):
    """Open video with clear error message if unsupported."""
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        return cap
    print(f"ERROR: Cannot open {video_path}")
    print(f"  This video format is not supported.")
    print(f"  Please convert to H.264 MP4 and try again.")
    sys.exit(1)


def _read_frame(cap, vid_idx, last_read_idx):
    """
    Read a frame, using sequential read when possible (much faster
    than seeking every frame). Returns (frame, new_last_read_idx).
    """
    if last_read_idx is not None and vid_idx == last_read_idx + 1:
        ret, frame = cap.read()
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
        ret, frame = cap.read()
    return (frame if ret else None), vid_idx


# ── Box computation (consolidated padding) ────────────────────

def _pad_box(cx, cy, bw, bh, frame_w, frame_h, pad_factor, min_size=20):
    """Compute padded square-ish bounding box, clamped to frame. Returns tuple or None."""
    size = max(bw, bh)
    half = size * (1.0 + pad_factor) / 2
    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(frame_w, int(cx + half))
    y2 = min(frame_h, int(cy + half))
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return None
    return (x1, y1, x2, y2)


def compute_face_box(head_kps, frame_w, frame_h, pad_factor=0.7,
                     extra_bottom_factor=0.5):
    """
    Compute padded face bounding box from SAM head keypoints.
    head_kps: (5, 2) pixel coords [nose, left_eye, right_eye, left_ear, right_ear]
    Returns (x1, y1, x2, y2) clamped to frame, or None if keypoints invalid.

    The keypoints only span eye-line to nose-tip vertically (no chin, no
    forehead), so symmetric padding leaves the chin sitting near the bottom
    edge -- an open jaw can fall outside the crop. extra_bottom_factor adds
    asymmetric room below to keep an open mouth inside the crop fed to
    FaceLandmarker.
    """
    valid = head_kps[np.any(head_kps != 0, axis=1)]
    if len(valid) < 2:
        return None
    mins, maxs = valid.min(axis=0), valid.max(axis=0)
    bw, bh = maxs[0] - mins[0], maxs[1] - mins[1]
    if bw < 5 or bh < 5:
        return None
    size = max(bw, bh)
    cx = (mins[0] + maxs[0]) / 2
    cy = (mins[1] + maxs[1]) / 2
    half = size * (1.0 + pad_factor) / 2
    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(frame_w, int(cx + half))
    y2 = min(frame_h, int(cy + half + size * extra_bottom_factor))
    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None
    return (x1, y1, x2, y2)


def _upscale_for_detection(frame_bgr):
    """Upscale frame if long edge < _DETECT_MIN_EDGE. Returns (frame, scale)."""
    h, w = frame_bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge >= _DETECT_MIN_EDGE:
        return frame_bgr, 1.0
    scale = _DETECT_MIN_EDGE / long_edge
    return cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR), scale


def _detect_face_box_mp(detector, frame_bgr, pad_factor=0.6, iou_anchor=None):
    """Use MediaPipe BlazeFace for face detection. Returns (box, n_detections) or (None, 0).

    iou_anchor: previous-frame face box (in this detector's frame coords) used
                to pick the most identity-consistent candidate when more than
                one face is detected.  When set, candidates within 0.2 of the
                top score are reranked by IoU vs the anchor; ties in IoU (or
                all-zero IoUs, e.g., subject moved out of the anchor region)
                fall back to score.  When None, behavior is highest-score-wins.
    """
    mp = _get_mp()
    h, w = frame_bgr.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    try:
        result = detector.detect(mp_image)
    except Exception:
        return None, 0
    if not result.detections:
        return None, 0
    n_faces = len(result.detections)

    def _padded(d):
        bb = d.bounding_box
        cx = bb.origin_x + bb.width / 2
        cy = bb.origin_y + bb.height / 2
        return _pad_box(cx, cy, bb.width, bb.height, w, h, pad_factor)

    def _score(d):
        return d.categories[0].score if d.categories else 0.0

    if iou_anchor is not None and n_faces > 1:
        scores = [_score(d) for d in result.detections]
        max_score = max(scores)
        kept = [d for d, s in zip(result.detections, scores)
                if s >= max_score - 0.2]
        ious = [(d, _bbox_iou(_padded(d), iou_anchor)) for d in kept]
        if any(iou > 0 for _, iou in ious):
            best = max(ious, key=lambda di: di[1])[0]
            return _padded(best), n_faces

    best = max(result.detections, key=_score)
    return _padded(best), n_faces


def _detect_face_box_haar(detector, frame_bgr, pad_factor=0.5, iou_anchor=None):
    """Use OpenCV Haar cascade for face detection. Returns box or None.

    iou_anchor: previous-frame face box (in this detector's frame coords).
                When set and multiple faces are detected, picks highest IoU
                vs the anchor; falls back to largest area when all IoUs
                are zero or no anchor is provided.
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None

    def _padded(f):
        bx, by, bw, bh = f
        return _pad_box(bx + bw / 2, by + bh / 2, bw, bh, w, h, pad_factor)

    if iou_anchor is not None and len(faces) > 1:
        ious = [(f, _bbox_iou(_padded(f), iou_anchor)) for f in faces]
        if any(iou > 0 for _, iou in ious):
            best = max(ious, key=lambda fi: fi[1])[0]
            return _padded(best)

    bx, by, bw, bh = max(faces, key=lambda f: f[2] * f[3])
    return _padded((bx, by, bw, bh))


def detect_face_box(detector, detector_type, frame_bgr, pad_factor=0.6,
                    iou_anchor=None):
    """Detect face box via BlazeFace then Haar fallback, with auto-upscale.
    Returns (box, n_faces).

    iou_anchor: optional previous-frame box in ORIGINAL frame coords.  Scaled
                here to match the detection frame before being passed to the
                inner detectors, so they can use it to pick identity-
                consistent faces from multi-result frames.
    """
    h_orig, w_orig = frame_bgr.shape[:2]
    up_frame, scale = _upscale_for_detection(frame_bgr)
    detectors = detector if isinstance(detector, dict) else {"haar": detector}

    # Scale the anchor into detection-frame coords if the frame was upscaled.
    if iou_anchor is not None and scale != 1.0:
        scaled_anchor = (iou_anchor[0] * scale, iou_anchor[1] * scale,
                         iou_anchor[2] * scale, iou_anchor[3] * scale)
    else:
        scaled_anchor = iou_anchor

    box = None
    n_faces = 0
    if "mp" in detectors:
        box, n_faces = _detect_face_box_mp(detectors["mp"], up_frame,
                                           pad_factor, iou_anchor=scaled_anchor)
    if box is None and "haar" in detectors:
        box = _detect_face_box_haar(detectors["haar"], up_frame, pad_factor,
                                    iou_anchor=scaled_anchor)
        if box is not None and n_faces == 0:
            n_faces = 1  # Haar doesn't count reliably, assume 1

    if box is None or scale == 1.0:
        return box, n_faces
    x1, y1, x2, y2 = box
    return (max(0, int(x1 / scale)), max(0, int(y1 / scale)),
            min(w_orig, int(x2 / scale)), min(h_orig, int(y2 / scale))), n_faces


def detect_head_from_pose(pose, frame_bgr, frame_w, frame_h, timestamp_ms):
    """
    Run MediaPipe PoseLandmarker on a frame and extract head keypoints
    in (5, 2) pixel format for compute_face_box().
    Returns (5, 2) numpy array or None.
    """
    mp = _get_mp()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    try:
        result = pose.detect_for_video(mp_image, timestamp_ms)
    except Exception:
        return None
    if not result.pose_landmarks:
        return None

    landmarks = result.pose_landmarks[0]
    head_kps = np.zeros((5, 2), dtype=np.float64)
    for i, pose_idx in enumerate(POSE_HEAD_INDICES):
        lm = landmarks[pose_idx]
        if lm.visibility >= 0.3:
            head_kps[i] = [lm.x * frame_w, lm.y * frame_h]
    return head_kps


# ── Body NPZ validation ──────────────────────────────────────

def validate_body_npz(body_path, video_path):
    """Check body NPZ matches video. Returns (data_dict, warnings) or (None, errors)."""
    body = np.load(str(body_path))
    if "keypoints_2d" not in body:
        return None, ["Body NPZ has no keypoints_2d -- was it made with an older version?"]

    keypoints_2d = body["keypoints_2d"].astype(np.float64)
    frame_indices = body["frame_indices"].astype(int) if "frame_indices" in body else np.arange(len(keypoints_2d))
    body_fps = float(body["fps"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, ["Cannot open video for validation"]
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    errors = []
    if abs(body_fps - vid_fps) > 0.5:
        errors.append(f"FPS mismatch: body NPZ says {body_fps:.1f}, video is {vid_fps:.1f}")
    max_frame = int(frame_indices[-1]) if len(frame_indices) > 0 else 0
    if max_frame >= vid_total:
        errors.append(f"Frame range mismatch: body NPZ references frame {max_frame}, "
                      f"but video only has {vid_total} frames")
    if errors:
        return None, errors

    return {"keypoints_2d": keypoints_2d, "frame_indices": frame_indices,
            "fps": body_fps, "n_tracked": len(frame_indices)}, []


# ── Face detection ────────────────────────────────────────────

def detect_face(landmarker, frame_bgr, timestamp_ms, box=None):
    """
    Run MediaPipe FaceLandmarker on a frame or cropped region.
    Returns (blendshapes, landmarks_3d, eye_gaze, face_matrix) or 4x None.
    """
    mp = _get_mp()
    h, w = frame_bgr.shape[:2]

    if box is not None:
        x1, y1, x2, y2 = box
        region = frame_bgr[y1:y2, x1:x2]
        rw, rh, ox, oy = x2 - x1, y2 - y1, x1, y1
    else:
        region = frame_bgr
        rw, rh, ox, oy = w, h, 0, 0

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
    try:
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
    except Exception:
        return None, None, None, None

    if not result.face_blendshapes:
        return None, None, None, None

    blendshapes = [(bs.category_name, bs.score) for bs in result.face_blendshapes[0]]

    # Vectorized landmark conversion to full-frame coords
    raw = result.face_landmarks[0]
    n_lm = len(raw)
    lm_arr = np.empty((n_lm, 3), dtype=np.float32)
    for i, lm in enumerate(raw):
        lm_arr[i] = (lm.x, lm.y, lm.z)
    lm_arr[:, 0] = lm_arr[:, 0] * rw + ox   # x pixel
    lm_arr[:, 1] = lm_arr[:, 1] * rh + oy   # y pixel
    lm_arr[:, 2] *= rw                        # z depth scaled by face width

    eye_gaze = _compute_eye_gaze(lm_arr)

    face_matrix = None
    if result.facial_transformation_matrixes:
        face_matrix = np.array(result.facial_transformation_matrixes[0], dtype=np.float32)

    return blendshapes, lm_arr, eye_gaze, face_matrix


# ── Eye gaze ──────────────────────────────────────────────────
# Iris centers / Eye corners / Eye lids
_EYE_IDX = {
    "li": 468, "ri": 473,
    "l_in": 133, "l_out": 33, "r_in": 362, "r_out": 263,
    "l_up": 159, "l_lo": 145, "r_up": 386, "r_lo": 374,
}

def _compute_eye_gaze(lm):
    """Compute normalized iris position within each eye. Returns (4,) array."""
    gaze = np.full(4, 0.5, dtype=np.float32)
    E = _EYE_IDX
    pairs = [
        (0, lm[E["li"], :2], lm[E["l_out"], :2], lm[E["l_in"], :2]),   # left horiz
        (1, lm[E["li"], :2], lm[E["l_lo"], :2], lm[E["l_up"], :2]),    # left vert
        (2, lm[E["ri"], :2], lm[E["r_out"], :2], lm[E["r_in"], :2]),   # right horiz
        (3, lm[E["ri"], :2], lm[E["r_lo"], :2], lm[E["r_up"], :2]),    # right vert
    ]
    for idx, iris, a, b in pairs:
        span_sq = float(np.dot(b - a, b - a))
        if span_sq > 1.0:
            gaze[idx] = np.clip(np.dot(iris - a, b - a) / span_sq, 0, 1)
    return gaze


# ── Mouth pixel statistics (for the export-time face refine) ─────────
# Landmarks can't tell teeth-together lips-parted (a teeth smile) from a
# dropped jaw — but pixels can: teeth read as a bright, low-saturation
# band, an open mouth as a dark cavity. Per-frame statistics of the
# pixels inside the inner-lip polygon are stored RAW in the NPZ (like
# everything else this script saves); the classification into teeth /
# cavity happens at export time in pipeline/face_refine.py with
# clip-adaptive thresholds, so re-exports never re-run inference.
#
# Layout (float32, MOUTH_STATS_N per frame) — pipeline/face_refine.py
# indexes into this, keep the two in sync:
#   0  inner_area_px      polygon area in pixels
#   1  interocular_px     eye-center distance (scale reference)
#   2-7   V mean, p10, p25, p50, p75, p90     (inner-mouth pixels)
#   8-9   S mean, p50
#   10 frac dark        V < 64
#   11 frac bright      V > 150
#   12 frac desat-bright V > 140 & S < 70    (fixed-threshold teeth-ish)
#   13 frac red         (H<=12|H>=168) & S>90 & V>60  (tongue-ish)
#   14-15 lip ring   V mean, S mean          (lip-flesh reference)
#   16-17 cheek skin V mean, S mean          (exposure reference)
#   18-81 8x8 normalized 2D histogram over (V//32, S//32), V-major —
#         lets export-time code integrate ANY threshold adaptively.
#   82    central_area_px — pixel count of the CENTRAL band (middle 60%
#         of the mouth polygon's width)
#   83-146 8x8 histogram of the central band only. Why: a wide
#         teeth-bare is bright in the CENTER with dark buccal-corridor
#         corners; an open mouth is dark in the center. Whole-polygon
#         statistics can't tell them apart — the corners fooled the
#         jaw gate on real footage.
#   147-341 (v3) the central band split into vertical THIRDS, each as
#         [area_px, 8x8 hist]: top 147/148-211, middle 212/213-276,
#         bottom 277/278-341. Why: tongue detection. Top-to-bottom, a
#         protruding tongue reads flesh/flesh/flesh, an everted lower
#         lip during speech reads teeth/dark/flesh (it fooled the
#         single-band detector), teeth+tongue reads teeth/flesh/flesh —
#         the MIDDLE third is the deciding band.
MOUTH_STATS_N = 342
MOUTH_STATS_VERSION = 3

# Inner/outer lip rings in polygon (drawing) order and the reference
# points, mirrored from pipeline/face_landmark_topology.py (kept inline
# so this script stays self-contained like save_mhr_data.py).
_MS_LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
                  308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
_MS_LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                  291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
_MS_LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382,
                362, 398, 384, 385, 386, 387, 388, 466]
_MS_RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155,
                 133, 173, 157, 158, 159, 160, 161, 246]
_MS_CHEEKS = [205, 425]


def _mean_vs_at(frame_bgr, cx, cy, half=4):
    """Mean (V, S) of a small patch around a pixel, or (0, 0)."""
    h, w = frame_bgr.shape[:2]
    x1, x2 = max(0, int(cx) - half), min(w, int(cx) + half + 1)
    y1, y2 = max(0, int(cy) - half), min(h, int(cy) + half + 1)
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0
    hsv = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean()), float(hsv[:, :, 1].mean())


def extract_mouth_stats(frame_bgr, lm_arr):
    """Per-frame mouth pixel statistics. Returns (MOUTH_STATS_N,) f32."""
    out = np.zeros(MOUTH_STATS_N, dtype=np.float32)
    try:
        h, w = frame_bgr.shape[:2]
        eye_l = lm_arr[_MS_LEFT_EYE, :2].mean(axis=0)
        eye_r = lm_arr[_MS_RIGHT_EYE, :2].mean(axis=0)
        io_px = float(np.linalg.norm(eye_l - eye_r))
        out[1] = io_px

        # Exposure references first — useful even with a closed mouth.
        skin_v = skin_s = n_skin = 0.0
        for ci in _MS_CHEEKS:
            v, s = _mean_vs_at(frame_bgr, lm_arr[ci, 0], lm_arr[ci, 1])
            if v > 0:
                skin_v += v; skin_s += s; n_skin += 1
        if n_skin:
            out[16] = skin_v / n_skin
            out[17] = skin_s / n_skin

        outer = lm_arr[_MS_LIPS_OUTER, :2]
        inner = lm_arr[_MS_LIPS_INNER, :2]
        x1 = int(np.clip(outer[:, 0].min() - 2, 0, w - 1))
        x2 = int(np.clip(outer[:, 0].max() + 3, 1, w))
        y1 = int(np.clip(outer[:, 1].min() - 2, 0, h - 1))
        y2 = int(np.clip(outer[:, 1].max() + 3, 1, h))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return out
        roi = frame_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        inner_loc = np.round(inner - [x1, y1]).astype(np.int32)
        outer_loc = np.round(outer - [x1, y1]).astype(np.int32)
        inner_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillPoly(inner_mask, [inner_loc], 1)
        outer_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillPoly(outer_mask, [outer_loc], 1)
        lip_mask = (outer_mask == 1) & (inner_mask == 0)

        if lip_mask.any():
            out[14] = float(hsv[:, :, 2][lip_mask].mean())
            out[15] = float(hsv[:, :, 1][lip_mask].mean())

        sel = inner_mask == 1
        area = int(sel.sum())
        out[0] = float(area)
        if area < 4:
            return out

        H = hsv[:, :, 0][sel].astype(np.float32)
        S = hsv[:, :, 1][sel].astype(np.float32)
        V = hsv[:, :, 2][sel].astype(np.float32)

        out[2] = V.mean()
        out[3:8] = np.percentile(V, [10, 25, 50, 75, 90])
        out[8] = S.mean()
        out[9] = np.percentile(S, 50)
        out[10] = float((V < 64).mean())
        out[11] = float((V > 150).mean())
        out[12] = float(((V > 140) & (S < 70)).mean())
        out[13] = float((((H <= 12) | (H >= 168))
                         & (S > 90) & (V > 60)).mean())

        hist, _, _ = np.histogram2d(V, S, bins=8,
                                    range=[[0, 256], [0, 256]])
        out[18:82] = (hist / area).astype(np.float32).ravel()

        # Central band: middle 60% of the inner polygon's width. Teeth
        # bars are bright here; open-mouth cavity is dark here; the
        # buccal-corridor corners are excluded either way.
        ix1, ix2 = inner_loc[:, 0].min(), inner_loc[:, 0].max()
        pad = 0.20 * max(ix2 - ix1, 1)
        col = np.arange(roi.shape[1])[None, :]
        central = sel & (col >= ix1 + pad) & (col <= ix2 - pad)
        c_area = int(central.sum())
        out[82] = float(c_area)
        if c_area >= 4:
            Vc = hsv[:, :, 2][central].astype(np.float32)
            Sc = hsv[:, :, 1][central].astype(np.float32)
            hist_c, _, _ = np.histogram2d(Vc, Sc, bins=8,
                                          range=[[0, 256], [0, 256]])
            out[83:147] = (hist_c / c_area).astype(np.float32).ravel()

            # Vertical thirds of the central band (see layout notes).
            iy1, iy2 = inner_loc[:, 1].min(), inner_loc[:, 1].max()
            row = np.arange(roi.shape[0])[:, None]
            span = max(iy2 - iy1, 3)
            for k in range(3):
                r_lo = iy1 + span * k / 3.0
                r_hi = iy1 + span * (k + 1) / 3.0
                third = central & (row >= r_lo) & (row < r_hi)
                t_area = int(third.sum())
                base = 147 + k * 65
                out[base] = float(t_area)
                if t_area >= 4:
                    Vt = hsv[:, :, 2][third].astype(np.float32)
                    St = hsv[:, :, 1][third].astype(np.float32)
                    ht, _, _ = np.histogram2d(Vt, St, bins=8,
                                              range=[[0, 256], [0, 256]])
                    out[base + 1:base + 65] = \
                        (ht / t_area).astype(np.float32).ravel()
    except Exception:
        # Stats are best-effort; a bad frame must never kill capture.
        return np.zeros(MOUTH_STATS_N, dtype=np.float32)
    return out


# ── Face mesh contours (for preview visualization) ────────────

FACE_CONTOURS = [
    ([10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,
      152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10], (200,180,130)),
    ([33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33], (0,255,200)),
    ([362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,362], (0,255,200)),
    ([61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,61], (180,105,255)),
    ([78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191,78], (180,105,255)),
    ([46,53,52,65,55,107,66,105,63,70], (200,200,200)),
    ([276,283,282,295,285,336,296,334,293,300], (200,200,200)),
    ([168,6,197,195,5,4], (200,200,200)),
    ([48,64,98,97,2,326,327,294,278,440,275,4,45,220,115,48], (200,200,200)),
]


def draw_face_landmarks(frame, landmarks_px, thickness=1):
    """Draw face mesh contours using polylines (faster than per-segment lines)."""
    n_lm = len(landmarks_px)
    for indices, color in FACE_CONTOURS:
        pts = np.array([[int(landmarks_px[i, 0]), int(landmarks_px[i, 1])]
                        for i in indices if i < n_lm], dtype=np.int32)
        if len(pts) > 1:
            cv2.polylines(frame, [pts], isClosed=False, color=color,
                          thickness=thickness, lineType=cv2.LINE_AA)


# ── Shared helpers for preview & inference ────────────────────

def _resolve_box(body_data, pose_estimator, face_detector, detector_type,
                 frame, width, height, timestamp_ms, count=None,
                 source_indices=None, last_known_box=None, frames_since_box=0):
    """
    Determine crop box with priority: body > pose > detector > last-known.
    Returns (box, last_known_box, frames_since_box, no_crop_flag, n_faces).
    n_faces is only meaningful in standalone detector mode.
    """
    box = None
    no_crop = False

    if body_data is not None:
        idx = source_indices[count] if source_indices is not None else count
        head_kps = body_data["keypoints_2d"][idx, HEAD_KP_INDICES, :]
        box = compute_face_box(head_kps, width, height)
        if box is None:
            no_crop = True
        return box, last_known_box, frames_since_box, no_crop, 0

    # Try pose estimator first
    if pose_estimator is not None:
        head_kps = detect_head_from_pose(pose_estimator, frame, width, height, timestamp_ms)
        if head_kps is not None:
            box = compute_face_box(head_kps, width, height)

    # Fall back to detector chain
    n_faces = 0
    if box is None and face_detector is not None:
        box, n_faces = detect_face_box(face_detector, detector_type, frame,
                                       iou_anchor=last_known_box)

    # Track last known box for fallback
    if box is not None:
        return box, box, 0, False, n_faces

    if last_known_box is not None:
        frames_since_box += 1
        lx1, ly1, lx2, ly2 = last_known_box
        expand = min(0.15 + 0.05 * frames_since_box, 0.5)
        pw = int((lx2 - lx1) * expand)
        ph = int((ly2 - ly1) * expand)
        box = (max(0, lx1 - pw), max(0, ly1 - ph),
               min(width, lx2 + pw), min(height, ly2 + ph))

    return box, last_known_box, frames_since_box, False, 0


def _make_miss_data(n_blendshapes, box, width, height, is_no_crop=False):
    """Create placeholder data for a missed/no-crop frame."""
    bs = np.zeros(n_blendshapes) if n_blendshapes is not None else None
    lm = np.zeros((478, 3), dtype=np.float32)
    eg = np.full(4, 0.5, dtype=np.float32)
    fm = np.eye(4, dtype=np.float32)
    bb = (0, 0, 0, 0) if (is_no_crop or box is None) else box
    return bs, lm, eg, fm, bb


def _resolve_mode_label(body_data, pose_estimator, face_detector):
    """Return mode string based on which detectors are available."""
    if body_data is not None:
        return "body-cropped"
    if pose_estimator is not None:
        return "pose-cropped"
    if face_detector is not None:
        return "detector-cropped"
    return "full-frame"


# =====================================================================
#  PREVIEW
# =====================================================================

def _preview_payload_at(idx, sample_indices, payloads):
    """Overlay payload for frame `idx` from sparse per-sample detections.

    Mirror of save_mhr_data._preview_payload_at (these scripts stay
    self-contained: importing save_mhr_data would pull in torch and its
    import-time env setup). sample_indices: sorted video-frame indices
    that were actually checked; payloads[i] is the drawable result at
    sample_indices[i], or None on a detector miss. Between two DETECTED
    samples the payload interpolates; around a missed sample the nearest
    detected payload holds up to the midpoint of the gap, then frames
    read as a miss. Returns (a, b, t): lerp a->b by t when b is not
    None, use `a` as-is when b is None, and a miss when a is None.
    """
    n = len(sample_indices)
    if n == 0:
        return None, None, 0.0
    pos = bisect.bisect_right(sample_indices, idx)
    if pos == 0:
        return payloads[0], None, 0.0
    if pos >= n:
        return payloads[n - 1], None, 0.0
    i0, i1 = pos - 1, pos
    a, b = payloads[i0], payloads[i1]
    f0, f1 = sample_indices[i0], sample_indices[i1]
    if a is not None and b is not None:
        return a, b, (idx - f0) / float(f1 - f0)
    # One side missed: the nearer sample's state wins, held statically.
    if idx - f0 <= f1 - idx:
        return a, None, 0.0
    return b, None, 0.0


def _nearest_sample(idx, sample_indices):
    """Index into sample_indices of the sample nearest to frame `idx`."""
    n = len(sample_indices)
    pos = bisect.bisect_right(sample_indices, idx)
    if pos == 0:
        return 0
    if pos >= n:
        return n - 1
    if idx - sample_indices[pos - 1] <= sample_indices[pos] - idx:
        return pos - 1
    return pos


def run_preview(video_path, landmarker, output_path, body_data=None,
                pose_estimator=None, face_detector=None, detector_type=None,
                step=10, max_frames=0, start_frame=0, bg_video=None,
                body_boxes=None):
    """Preview face detection on video, in two passes. Green = face
    found, red = no face.

    Pass 1 runs detection (crop resolution + MediaPipe) on every Nth
    frame only — the expensive part, and its cost is unchanged. Pass 2
    re-reads the video sequentially and writes EVERY frame at the source
    frame rate, interpolating the face box and mesh contours between
    checked frames, so the preview plays at full speed. Around a missed
    sample the nearest detection holds to the midpoint of the gap, then
    frames read as "NO FACE" / "NO FACE CROP" (see _preview_payload_at),
    with the searched crop box drawn in red when one exists.
    Combined body+face previews render ONCE, here: the addon passes
    `body_boxes` (a JSON sidecar of the body preview's sampled
    detections, written by save_mhr_data --boxes-out) and this render
    draws the interpolated body box underneath the face markers - one
    full-length encode instead of two. The older `bg_video` path
    (compositing over an already-rendered body preview, read one frame
    per source frame) remains supported for CLI use.

    The addon passes the user's Preview Skip (as skip+1), same as the
    body preview. Face CAPTURE is deliberately different - run_inference
    always processes every frame, because blinks and lip-sync events are
    only a few frames long and don't survive interpolation.
    """
    step = max(1, int(step))
    print(f"\nReading video: {video_path}")
    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Open background video for combined preview
    bg_cap = None
    if bg_video is not None and bg_video.exists():
        bg_cap = cv2.VideoCapture(str(bg_video))
        if not bg_cap.isOpened():
            print(f"  WARNING: Cannot open bg video {bg_video}, drawing on original")
            bg_cap = None
        else:
            print(f"  Combined preview: drawing face on body preview")

    # Body detection sidecar for the single-render combined preview
    # (written by save_mhr_data --boxes-out). Failure is non-fatal: the
    # preview still renders, just without the body overlay.
    body_samples_idx = body_samples_boxes = None
    if body_boxes is not None:
        try:
            with open(body_boxes, "r", encoding="utf-8") as f:
                _bs = json.load(f)
            body_samples_idx = [int(i) for i in _bs.get("sample_indices", [])]
            body_samples_boxes = list(_bs.get("boxes", []))
            n_bs = min(len(body_samples_idx), len(body_samples_boxes))
            body_samples_idx = body_samples_idx[:n_bs]
            body_samples_boxes = body_samples_boxes[:n_bs]
            print(f"  Combined preview: drawing body boxes from {body_boxes}")
        except Exception as e:
            print(f"  WARNING: Cannot read body boxes {body_boxes} ({e}); "
                  f"face markers only")
            body_samples_idx = body_samples_boxes = None

    # Scale down for preview (cap long edge at 1920, otherwise native)
    max_edge = 1920
    if max(width, height) > max_edge:
        scale = max_edge / max(width, height)
        out_w = int(width * scale) // 2 * 2
        out_h = int(height * scale) // 2 * 2
    else:
        out_w, out_h = width, height

    # Build the sampled frame list (frames to CHECK) and the full render
    # range (frames to WRITE - every frame between the range bounds).
    source_indices = None
    if body_data is not None:
        frame_indices = body_data["frame_indices"]
        source_indices = list(range(0, len(frame_indices), step))
        if max_frames > 0:
            # max_frames caps the FRAME range (NPZ rows), not the sample
            # count - with step > 1 a plain [:max_frames] slice would
            # silently cover step*max_frames frames.
            source_indices = [i for i in source_indices if i < max_frames]
        vid_frames = [int(frame_indices[i]) for i in source_indices]
        render_start = vid_frames[0] if vid_frames else start_frame
        render_end = (vid_frames[-1] + 1) if vid_frames else start_frame
        mode_label = "body-cropped"
    elif bg_cap is not None:
        effective_total = min(start_frame + max_frames, vid_total) if max_frames > 0 else vid_total
        vid_frames = list(range(start_frame, effective_total, step))
        render_start, render_end = start_frame, effective_total
        mode_label = "combined"
    else:
        # Standalone face preview samples by `step` like every other mode
        # (it used to force step=1 and check every frame - slower, and the
        # overlay jittered per-frame instead of interpolating).
        limit = min(start_frame + max_frames, vid_total) if max_frames > 0 else vid_total
        vid_frames = list(range(start_frame, limit, step))
        render_start, render_end = start_frame, limit
        mode_label = _resolve_mode_label(body_data, pose_estimator, face_detector)

    print(f"  {len(vid_frames)} frames to check ({mode_label}, every {step})")
    print(f"  Preview resolution: {out_w}x{out_h}")

    # ── Pass 1: detection on the sampled frames only ──────────────
    detected = missed = no_crop = 0
    last_known_box = None
    frames_since_box = 0
    last_read_idx = None
    ms_count, ms_warned = 0, False
    sample_payloads = []     # (crop_box, landmarks) per hit, None per miss
    sample_miss_labels = []  # miss text per sample (None on a hit)
    sample_crop_boxes = []   # searched crop box per sample, hit OR miss
                             # (misses draw it in red - see the render pass)

    n_samples = len(vid_frames)
    for count, vid_idx in enumerate(vid_frames):
        frame, last_read_idx = _read_frame(cap, vid_idx, last_read_idx)
        if frame is None:
            break

        timestamp_ms = int(vid_idx * 1000.0 / fps)
        box, last_known_box, frames_since_box, is_no_crop, n_faces = _resolve_box(
            body_data, pose_estimator, face_detector, detector_type,
            frame, width, height, timestamp_ms, count, source_indices,
            last_known_box, frames_since_box)

        if n_faces > 1:
            ms_count, ms_warned = _check_multi_face(ms_count, ms_warned, "preview")

        crop_f = [float(v) for v in box] if box is not None else None
        blendshapes = None
        if body_data is not None and box is None:
            no_crop += 1
            sample_payloads.append(None)
            sample_miss_labels.append("NO FACE CROP")
            sample_crop_boxes.append(None)
        else:
            blendshapes, landmarks_3d, _, _ = detect_face(landmarker, frame, timestamp_ms, box)
            if blendshapes is not None:
                detected += 1
                sample_payloads.append((crop_f, landmarks_3d))
                sample_miss_labels.append(None)
            else:
                missed += 1
                sample_payloads.append(None)
                sample_miss_labels.append("NO FACE")
            sample_crop_boxes.append(crop_f)

        if n_samples > 0:
            emit_progress(15 + (count + 1) / n_samples * 55,
                          f"Face preview {count + 1}/{n_samples}")
        if count % 10 == 0:
            status = "no crop" if (body_data is not None and box is None) else \
                     "face OK" if blendshapes is not None else "no face"
            print(f"  Frame {vid_idx} - {status}")

    sample_indices = vid_frames[:len(sample_payloads)]

    # ── Pass 2: render EVERY frame, overlays lerped between samples ──
    # Cheap next to detection (decode + draw + encode only). Native-fps
    # full-frame output keeps the preview's timing identical to the
    # source and frame-aligned with the body preview background.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"ERROR: Cannot create output video {output_path}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, render_start)
    n_render = max(1, render_end - render_start)
    for idx in range(render_start, render_end):
        ret, frame = cap.read()
        if not ret:
            break

        # Canvas: the body preview frame in combined mode, else the source.
        if bg_cap is not None:
            bg_ret, bg_frame = bg_cap.read()
            draw_frame = cv2.resize(bg_frame, (width, height), interpolation=cv2.INTER_LINEAR) if bg_ret else frame.copy()
        else:
            draw_frame = frame

        # Body overlay first (face markers draw on top): interpolated box
        # from the body sidecar, same visuals as the body preview render
        # (see save_mhr_data.run_preview).
        if body_samples_idx:
            ba, bb, bt = _preview_payload_at(idx, body_samples_idx,
                                             body_samples_boxes)
            bbox = [av + (bv - av) * bt for av, bv in zip(ba, bb)] \
                if (ba is not None and bb is not None) else ba
            if bbox is not None:
                bx1, by1, bx2, by2 = (int(v) for v in bbox[:4])
                cv2.rectangle(draw_frame, (bx1, by1), (bx2, by2),
                              (0, 255, 0), 3, cv2.LINE_AA)
                cv2.putText(draw_frame, "BODY OK", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            else:
                cv2.putText(draw_frame, "NO PERSON", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        a, b, t = _preview_payload_at(idx, sample_indices, sample_payloads)
        if a is not None and b is not None:
            box_a, lm_a = a
            box_b, lm_b = b
            if box_a is not None and box_b is not None:
                box = [av + (bv - av) * t for av, bv in zip(box_a, box_b)]
            else:
                box = box_a if box_a is not None else box_b
            if (lm_a is not None and lm_b is not None
                    and lm_a.shape == lm_b.shape):
                lms = lm_a + (lm_b - lm_a) * t
            else:
                lms = lm_a if lm_a is not None else lm_b
        elif a is not None:
            box, lms = a
        else:
            box, lms = None, None

        if box is not None or lms is not None:
            color = (0, 255, 0)
            if box is not None:
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(draw_frame, f"FACE {x2-x1}x{y2-y1}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                cv2.putText(draw_frame, "FACE OK", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            if lms is not None:
                draw_face_landmarks(draw_frame, lms)
        else:
            # Miss visuals match the pre-interpolation preview: when the
            # governing missed sample HAD a crop box (we looked somewhere
            # and found no face there), draw that box in red with the
            # label beside it - far more visible than corner text, and it
            # shows WHERE detection was looking. Corner text is the
            # fallback when there was no crop box at all.
            label, crop = "NO FACE", None
            if sample_indices:
                near = _nearest_sample(idx, sample_indices)
                label = sample_miss_labels[near] or "NO FACE"
                crop = sample_crop_boxes[near]
            color = (0, 0, 255)
            if crop is not None:
                x1, y1, x2, y2 = (int(v) for v in crop)
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(draw_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                cv2.putText(draw_frame, label, (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        cv2.putText(draw_frame, f"Frame {idx}", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if (out_w, out_h) != (width, height):
            draw_frame = cv2.resize(draw_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(draw_frame)

        done = idx - render_start + 1
        if done % 50 == 0 or done == n_render:
            emit_progress(70 + done / n_render * 30,
                          f"Rendering preview {done}/{n_render}")

    cap.release()
    if bg_cap is not None:
        bg_cap.release()
    writer.release()

    total_checked = detected + missed + no_crop
    size_mb = os.path.getsize(str(output_path)) / (1024 * 1024)
    print(f"\nPreview saved: {output_path} ({size_mb:.1f} MB)")
    print(f"  {total_checked} frames checked: {detected} face OK, {missed} no face"
          + (f", {no_crop} no crop" if no_crop > 0 else ""))

    if total_checked > 0 and (missed + no_crop) > 0:
        miss_pct = (missed + no_crop) / total_checked * 100
        print(f"  WARNING: {miss_pct:.0f}% of frames had no face detection!")
        if no_crop > total_checked * 0.3:
            print("  Many frames have no usable head keypoints from body tracking.")
            print("  Check that body tracking is working first (save_mhr_data.py --preview).")
        if missed > total_checked * 0.3:
            print("  MediaPipe is not finding a face in many frames.")
            print("  The subject may be facing away or the face may be too small.")


# =====================================================================
#  INFERENCE
# =====================================================================

def run_inference(video_path, landmarker, output, body_data=None,
                  pose_estimator=None, face_detector=None, detector_type=None,
                  start_frame=0, max_frames=0):
    """Full face blendshape extraction. Saves raw (unsmoothed) data."""
    print(f"\nReading video: {video_path}")
    cap = open_video(video_path)
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  {vid_total} frames, {width}x{height}, {vid_fps:.1f} FPS")

    if body_data is not None:
        frame_indices = body_data["frame_indices"]
        n_to_process = body_data["n_tracked"]
        fps_out = body_data["fps"]
    else:
        end_frame = min(start_frame + max_frames, vid_total) if max_frames > 0 else vid_total
        frame_indices = np.arange(start_frame, end_frame)
        n_to_process = len(frame_indices)
        fps_out = vid_fps

    mode_label = _resolve_mode_label(body_data, pose_estimator, face_detector)
    print(f"  Mode: {mode_label}, {n_to_process} frames to process"
          + (f" (from frame {start_frame})" if start_frame > 0 and body_data is None else ""))

    # ── Process frames ────────────────────────────────────────
    print("\nProcessing frames...")

    blendshape_names = None
    n_blendshapes = None
    all_blendshapes = []
    all_landmarks = []
    all_eye_gaze = []
    all_face_matrix = []
    all_face_bbox = []
    all_face_detected = []
    all_mouth_stats = []
    detected = missed = no_crop = 0
    last_known_box = None
    frames_since_box = 0
    last_read_idx = None
    ms_count, ms_warned = 0, False

    for count in range(n_to_process):
        vid_idx = int(frame_indices[count])
        frame, last_read_idx = _read_frame(cap, vid_idx, last_read_idx)

        if frame is None:
            print(f"  Frame {vid_idx}: could not read from video")
            bs, lm, eg, fm, bb = _make_miss_data(n_blendshapes, None, width, height)
            all_blendshapes.append(bs)
            all_landmarks.append(lm)
            all_eye_gaze.append(eg)
            all_face_matrix.append(fm)
            all_face_bbox.append(bb)
            all_face_detected.append(False)
            all_mouth_stats.append(np.zeros(MOUTH_STATS_N, dtype=np.float32))
            missed += 1
            continue

        timestamp_ms = int(vid_idx * 1000.0 / vid_fps)
        box, last_known_box, frames_since_box, is_no_crop, n_faces = _resolve_box(
            body_data, pose_estimator, face_detector, detector_type,
            frame, width, height, timestamp_ms, count, None,
            last_known_box, frames_since_box)

        if n_faces > 1:
            ms_count, ms_warned = _check_multi_face(ms_count, ms_warned, "inference")

        if is_no_crop:
            no_crop += 1
            bs, lm, eg, fm, bb = _make_miss_data(n_blendshapes, None, width, height, True)
            all_blendshapes.append(bs)
            all_landmarks.append(lm)
            all_eye_gaze.append(eg)
            all_face_matrix.append(fm)
            all_face_bbox.append(bb)
            all_face_detected.append(False)
            all_mouth_stats.append(np.zeros(MOUTH_STATS_N, dtype=np.float32))
            if count < 5 or count % 100 == 0:
                print(f"  Frame {vid_idx}: no valid face crop")
            continue

        blendshapes, landmarks_3d, eye_gaze, face_matrix = detect_face(
            landmarker, frame, timestamp_ms, box)

        if blendshapes is not None:
            detected += 1
            if blendshape_names is None:
                blendshape_names = [name for name, _ in blendshapes]
                n_blendshapes = len(blendshape_names)
                print(f"  Blendshape count: {n_blendshapes}")
                print(f"  Names: {blendshape_names[:5]}... (showing first 5)")
                for i, entry in enumerate(all_blendshapes):
                    if entry is None:
                        all_blendshapes[i] = np.zeros(n_blendshapes)

            all_blendshapes.append(np.array([s for _, s in blendshapes]))
            all_landmarks.append(landmarks_3d)
            all_eye_gaze.append(eye_gaze)
            all_face_matrix.append(face_matrix if face_matrix is not None
                                   else np.eye(4, dtype=np.float32))
            all_face_bbox.append(box if box is not None else (0, 0, width, height))
            all_face_detected.append(True)
            all_mouth_stats.append(extract_mouth_stats(frame, landmarks_3d))
        else:
            missed += 1
            bs, lm, eg, fm, bb = _make_miss_data(n_blendshapes, box, width, height)
            all_blendshapes.append(bs)
            all_landmarks.append(lm)
            all_eye_gaze.append(eg)
            all_face_matrix.append(fm)
            all_face_bbox.append(bb)
            all_face_detected.append(False)
            all_mouth_stats.append(np.zeros(MOUTH_STATS_N, dtype=np.float32))

        if n_to_process > 0:
            emit_progress(15 + (count + 1) / n_to_process * 80,
                          f"Face frame {count + 1}/{n_to_process}")
        if count % 20 == 0 or count == 0:
            status = "face OK" if blendshapes is not None else "no face"
            print(f"  Frame {vid_idx} [{count}/{n_to_process}] - {status}")

    cap.release()

    # ── Validate ──────────────────────────────────────────────
    if detected == 0:
        print("\nERROR: No frames with detected faces!")
        print(f"  {missed} missed, {no_crop} no crop")
        print("  Check that the subject's face is visible in the video.")
        if body_data is not None:
            print("  Try running without --body to use full-frame detection.")
        sys.exit(1)

    # Fill remaining None placeholders
    for i, entry in enumerate(all_blendshapes):
        if entry is None:
            all_blendshapes[i] = np.zeros(n_blendshapes)

    # Convert to arrays
    blendshapes_arr = np.array(all_blendshapes, dtype=np.float32)
    landmarks_arr = np.array(all_landmarks, dtype=np.float32)
    eye_gaze_arr = np.array(all_eye_gaze, dtype=np.float32)
    face_matrix_arr = np.array(all_face_matrix, dtype=np.float32)
    detected_arr = np.array(all_face_detected, dtype=bool)

    # ── Interpolate short gaps (vectorized) ───────────────────
    MAX_GAP = 15
    interpolated = 0
    n_frames = len(detected_arr)
    i = 0
    while i < n_frames:
        if detected_arr[i]:
            i += 1
            continue
        gap_start = i
        while i < n_frames and not detected_arr[i]:
            i += 1
        gap_end = i
        gap_len = gap_end - gap_start
        if gap_len > MAX_GAP or gap_start == 0 or gap_end >= n_frames:
            continue

        # Vectorized interpolation across the gap
        before, after = gap_start - 1, gap_end
        t = np.linspace(0, 1, gap_len + 2, dtype=np.float32)[1:-1]  # exclude endpoints
        for arr in (blendshapes_arr, landmarks_arr, eye_gaze_arr, face_matrix_arr):
            a, b = arr[before], arr[after]
            arr[gap_start:gap_end] = a + np.outer(t, (b - a).ravel()).reshape(gap_len, *a.shape)
        detected_arr[gap_start:gap_end] = True
        interpolated += gap_len

    if interpolated > 0:
        print(f"  Interpolated {interpolated} frames across short gaps (max {MAX_GAP})")

    # ── Save ──────────────────────────────────────────────────
    total = detected + missed + no_crop
    parts = [f"{detected} face OK", f"{missed} no face"]
    if interpolated > 0:
        parts.append(f"{interpolated} interpolated")
    if no_crop > 0:
        parts.append(f"{no_crop} no crop")
    print(f"\nProcessed {total} frames: {', '.join(parts)}")

    np.savez_compressed(
        output,
        face_blendshapes=blendshapes_arr,
        face_landmarks=landmarks_arr,
        eye_gaze=eye_gaze_arr,
        face_transform=face_matrix_arr,
        blendshape_names=np.array(blendshape_names),
        face_bbox=np.array(all_face_bbox, dtype=np.int32),
        face_detected=detected_arr,
        frame_indices=frame_indices,
        fps=np.array(fps_out),
        # Raw mouth pixel statistics for the export-time face refine
        # (teeth vs open-cavity disambiguation). See extract_mouth_stats.
        mouth_stats=np.array(all_mouth_stats, dtype=np.float32),
        mouth_stats_version=np.array(MOUTH_STATS_VERSION),
        # Schema version stamp — readers must tolerate its absence
        # (older caches lack it). Bump on any field rename/reshape.
        # v2: added mouth_stats.
        blendcap_npz_version=np.array(2),
    )

    print(f"\nSaved to {output}")
    print(f"  face_blendshapes: {blendshapes_arr.shape}")
    print(f"  face_landmarks:   {landmarks_arr.shape}")
    print(f"  eye_gaze:         {eye_gaze_arr.shape}")
    print(f"  face_transform:   {face_matrix_arr.shape}")
    print(f"  blendshape_names: {blendshape_names[:5]}...")
    print(f"  face_detected:    {detected + interpolated}/{total} frames"
          f" ({detected} detected, {interpolated} interpolated)")
    print(f"  fps:              {fps_out}")


# =====================================================================
#  MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract face blendshapes from video using MediaPipe")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--body", default="",
                        help="Body NPZ from save_mhr_data.py (optional, improves crop quality)")
    parser.add_argument("--output", default="",
                        help="Output path (default: face_frames.npz or face_preview.mp4)")
    parser.add_argument("--preview", action="store_true",
                        help="Preview mode: face detection boxes, export MP4")
    parser.add_argument("--preview-step", type=int, default=10,
                        help="Every Nth frame in body-cropped preview (default: 10, ignored without --body)")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Start processing from this frame (default: 0)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames to process (0 = all)")
    parser.add_argument("--bg-video", default="",
                        help="Background video for combined preview (draws face annotations on top)")
    parser.add_argument("--body-boxes", default="",
                        help="JSON sidecar of body-preview detections (from save_mhr_data "
                             "--boxes-out): draws the interpolated body box under the face "
                             "markers, so the combined preview renders in one pass")
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"ERROR: Video not found: {video}")
        sys.exit(1)

    # ── Load face model ───────────────────────────────────────
    emit_progress(2, "Loading face model")
    print("Loading face landmarker...")
    landmarker = load_face_landmarker(_download_model("face", fatal=True))

    # ── Load and validate body data ───────────────────────────
    emit_progress(5, "Loading body data")
    body_data = None
    if args.body:
        body_path = Path(args.body)
        if body_path.exists():
            body_data, issues = validate_body_npz(body_path, video)
            if body_data is not None:
                print(f"  Using body crops: {body_data['n_tracked']} tracked frames")

    # ── Load fallback detectors (no body) ─────────────────────
    face_detector = detector_type = pose_estimator = None
    if body_data is None:
        emit_progress(8, "Loading pose estimator")
        print("Loading pose estimator (head crop)...")
        pose_estimator = load_pose_estimator()
        emit_progress(12, "Loading face detector")
        print("Loading face detector (fallback)...")
        face_detector, detector_type = load_face_detector()

    print(f"  Mode: {_resolve_mode_label(body_data, pose_estimator, face_detector)}")

    # ── Preview or Inference ──────────────────────────────────
    if args.preview:
        output = Path(args.output) if args.output else Path("face_preview.mp4")
        emit_progress(15, "Starting face preview")
        run_preview(video, landmarker, output,
                    body_data=body_data, pose_estimator=pose_estimator,
                    face_detector=face_detector, detector_type=detector_type,
                    step=max(1, args.preview_step), start_frame=args.start_frame,
                    max_frames=args.max_frames,
                    bg_video=Path(args.bg_video) if args.bg_video else None,
                    body_boxes=Path(args.body_boxes) if args.body_boxes else None)
        emit_progress(100, "Face preview complete")
        return

    output = args.output if args.output else "face_frames.npz"
    emit_progress(15, "Starting face inference")
    run_inference(video, landmarker, output,
                  body_data=body_data, pose_estimator=pose_estimator,
                  face_detector=face_detector, detector_type=detector_type,
                  start_frame=args.start_frame, max_frames=args.max_frames)
    emit_progress(100, "Face tracking complete")

    if args.body and body_data is not None:
        print(f"\nOverlay (body + face):")
        print(f"  python overlay_track.py --video {video} --input {args.body} --face {output}")
    else:
        print(f"\nNext: use with overlay_track.py --face {output}")


if __name__ == "__main__":
    main()
