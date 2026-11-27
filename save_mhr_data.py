# save_mhr_data.py
# ================
# Body capture pipeline: video -> body NPZ using Fast-SAM-3D-Body
# (USC PSI, arxiv 2603.15603) + the YOLO11m-pose detector. Produces an
# NPZ with the schema npz_to_bvh.py expects; the Blender addon runs this
# script as a subprocess inside the bundled venv.

import os

# Apple Silicon: allow torch ops not implemented on MPS to fall back to CPU
# silently instead of erroring. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ── Offline operation ─────────────────────────────────────────────
# BlendCap runs fully offline after install. Force the model libraries into
# offline mode so they NEVER make a network round-trip at capture time:
#   - HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE: no Hugging Face HEAD requests
#     (MoGe weights load from the local cache only).
#   - HF_HUB_DISABLE_TELEMETRY: no usage pings.
#   - YOLO_OFFLINE: disables Ultralytics' telemetry, PyPI update check, AND
#     its is_online() DNS probe — that probe alone stalls startup for seconds
#     on a machine with no internet.
# All weights are bundled locally by the installer; the DINOv3 repo loads via
# torch.hub source="local" (see backbones/dinov3.py). Must be set BEFORE torch
# / huggingface_hub / ultralytics import (several read these at import time).
# setdefault so a developer can still override to allow a deliberate download.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("YOLO_OFFLINE", "true")

# ── Fast pipeline env vars (must be set BEFORE importing sam_3d_body) ──
# Conservative first-test settings: pure PyTorch path, torch.compile OFF
# (Windows-safety), all the layer-pruning and small-MoGe wins enabled.
os.environ.setdefault("LAYER_DTYPE", "fp32")            # FP32 (multi-person safe)
os.environ.setdefault("SKIP_KEYPOINT_PROMPT", "1")      # Skip kp-prompt encoding
os.environ.setdefault("USE_COMPILE", "0")               # torch.compile OFF for v1
os.environ.setdefault("USE_COMPILE_BACKBONE", "0")
os.environ.setdefault("DECODER_COMPILE", "0")
os.environ.setdefault("BODY_INTERM_PRED_LAYERS", "0,1,2")
os.environ.setdefault("HAND_INTERM_PRED_LAYERS", "0,1")
os.environ.setdefault("MHR_NO_CORRECTIVES", "1")        # Skip correctives
os.environ.setdefault("FOV_FAST", "1")                  # Fast MoGe mode
os.environ.setdefault("FOV_MODEL", "s")                 # Small MoGe (35M, was 331M)
os.environ.setdefault("FOV_LEVEL", "0")                 # Low MoGe resolution
os.environ.setdefault("GPU_HAND_PREP", "1")             # GPU hand preprocessing

import sys
import types
import argparse
import bisect
import json
import math
from pathlib import Path

import numpy as np
import cv2
import torch

# ── CUDA-stub patches for CPU-only torch builds ───────────────────
# Fast-SAM-3D-Body has ~50 torch.cuda.synchronize() / empty_cache()
# calls scattered through its source for timing/profiling instrumentation.
# On a torch-cpu build these raise "Torch not compiled with CUDA enabled"
# even though they're not doing useful work. Stub them to no-ops on CPU
# so the pipeline runs on machines without NVIDIA hardware. Untouched
# on CUDA — only patches when torch.cuda is actually unavailable.
if not torch.cuda.is_available():
    _cuda_noop = lambda *args, **kwargs: None
    torch.cuda.synchronize = _cuda_noop
    torch.cuda.empty_cache = _cuda_noop

# ── Paths ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.absolute()
# Model files live in the addon's dependency storage location; the
# addon passes it via BLENDCAP_CHECKPOINTS (helpers._subprocess_env).
# Fallback: next to this script - the pre-1.0.6 layout, and what a
# bare manual run of this script should still use.
CKPT_DIR = Path(os.environ.get("BLENDCAP_CHECKPOINTS")
                or (SCRIPT_DIR / "checkpoints"))
os.environ["TORCH_HOME"] = str(CKPT_DIR / "torch")
os.environ["HF_HOME"] = str(CKPT_DIR / "huggingface")

# Point the vendored DINOv3 backbone loader at the repo we ship INSIDE the
# addon (architecture code only, ~1 MB) so it loads offline via torch.hub
# source="local" — see models/fast-sam-3d-body/.../backbones/dinov3.py. And
# hand the MoGe loader the checkpoints dir so it finds the pre-fetched local
# model.pt instead of pulling from Hugging Face. Both setdefault for override.
os.environ.setdefault(
    "DINOV3_HUB_DIR",
    str(SCRIPT_DIR / "models" / "dinov3_hub" / "facebookresearch_dinov3_main"),
)
os.environ.setdefault("BLENDCAP_MOGE_BASE", str(CKPT_DIR))

# Use the vendored Fast-SAM-3D-Body library in models/fast-sam-3d-body/.
# Self-contained — works wherever the script lives, including after the
# user moves the install dir.
FAST_DIR = SCRIPT_DIR / "models" / "fast-sam-3d-body"
if not FAST_DIR.exists():
    print(f"ERROR: Fast-SAM-3D-Body vendored library not found at {FAST_DIR}")
    print("       The addon install looks incomplete - reinstall BlendCap.")
    sys.exit(1)
sys.path.insert(0, str(FAST_DIR))

# Mock pyrender + the visualization.renderer module — we don't render,
# we save NPZ, and importing the real renderer would drag in heavy
# display dependencies the venv doesn't ship.
if "pyrender" not in sys.modules:
    _pr = types.ModuleType("pyrender")
    _pr.Node = type("Node", (), {})
    sys.modules["pyrender"] = _pr

_rkey = "sam_3d_body.visualization.renderer"
if _rkey not in sys.modules:
    _mod = types.ModuleType(_rkey)
    _mod.Renderer = type("Renderer", (), {})
    sys.modules[_rkey] = _mod

# pkg_resources stub for Python 3.12+
try:
    import pkg_resources
except ImportError:
    try:
        __import__("setuptools"); import pkg_resources
    except Exception:
        _pkg = types.ModuleType("pkg_resources")
        _pkg.DistributionNotFound = type("DistributionNotFound", (Exception,), {})
        _pkg.get_distribution = lambda x: None
        sys.modules["pkg_resources"] = _pkg


def to_np(x):
    if x is None:
        return None
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)


def emit_progress(pct, msg=""):
    print(f"[PROGRESS {int(pct)}] {msg}", flush=True)


def open_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        return cap
    print(f"ERROR: Cannot open {video_path}")
    sys.exit(1)


def _get_device():
    """Pick the best torch device for the TORCH side of the pipeline.

    BLENDCAP_FORCE_CPU=1 overrides everything -> "cpu" (the CPU-only capture
    backend; the addon injects it into the subprocess env - see
    run_mocap_thread). Otherwise priority is CUDA > (DirectML detected -> CPU
    hybrid) > MPS > CPU. On a DirectML box we deliberately return "cpu", NOT
    torch_directml.device(): torch-directml lacks the dynamic-shape ops the
    detector/decoder/heads need, so torch runs on CPU while the heavy DINOv3
    backbone stays on the GPU via ONNX Runtime (USE_ORT, which picks
    DmlExecutionProvider on its own). Returns a string in every case.
    """
    if os.environ.get("BLENDCAP_FORCE_CPU", "0") == "1":
        print("[save_mhr_data] BLENDCAP_FORCE_CPU=1 -> all torch on CPU "
              "(CPU-only capture backend).")
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_directml
        if torch_directml.is_available():
            # DirectML hybrid: torch-directml can't run the
            # dynamic-shape ops (boolean-mask index, nonzero, dynamic gather)
            # that pervade YOLO NMS / the SAM3D decoder / the MHR head, so on
            # the raw DML device every frame fails with "The parameter is
            # incorrect." Route torch to CPU and keep ONLY the backbone on
            # DirectML via ONNX Runtime. The ORT backbone selects its EP
            # independently of the torch device, so this does NOT move it
            # off the GPU. (Set BLENDCAP_ORT_CPU=1 to also force the backbone
            # to CPU for a true full-CPU run.)
            print("[save_mhr_data] DirectML present -> torch on CPU; DINOv3 "
                  "backbone runs on DirectML via ONNX Runtime (USE_ORT).")
            return "cpu"
    except ImportError:
        pass
    except Exception as e:
        # torch-directml links DirectML.dll, which initialises against
        # the graphics driver at import; on an old/OEM AMD or Intel
        # driver that raises an OSError-class error, not ImportError.
        # Torch runs on the CPU in that case, same as a DirectML box.
        print(f"[WARNING] torch-directml could not start ({type(e).__name__}); "
              "torch runs on the CPU.")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_estimator_fast():
    """Load Fast-SAM-3D-Body estimator with YOLO11-Pose detector."""
    device = _get_device()
    print(f"  Device: {device}")

    # The ORT path needs the backbone export (.onnx + its external
    # .onnx_data — the .onnx alone is useless) on disk. The ADDON fetches
    # it as a separate pre-flight pipeline step before capture launches
    # (download_models.py --ensure-backbone-onnx); THIS process is forced
    # offline above and must never download. If the export still isn't
    # here (offline first run / fetch failed), demote to the PyTorch
    # backbone so the capture completes instead of crashing. Runs BEFORE
    # every USE_ORT read below, so the backbone, detector, and MoGe blocks
    # all route consistently.
    if os.environ.get("USE_ORT", "0") == "1":
        _bb = Path(os.environ.get(
            "ORT_BACKBONE_PATH",
            str(CKPT_DIR / "onnx" / "backbone_dinov3_fp16.onnx"),
        ))
        if not (_bb.exists() and Path(str(_bb) + "_data").exists()):
            print("[WARNING] ONNX backbone missing - capture falls back to slower path.")
            print(f"  ORT backbone export not on disk ({_bb}). Running the "
                  "PyTorch backbone for this run; the fast backend (~1.7 GB) "
                  "is fetched automatically on the next capture with internet "
                  "access.")
            os.environ["USE_ORT"] = "0"
        else:
            # The DirectML build of ONNX Runtime initialises DirectML.dll
            # against the graphics driver the moment it is imported; on
            # an old/OEM AMD or Intel driver the whole module refuses to
            # import ("DLL load failed while importing
            # onnxruntime_pybind11_state"), even for CPU-only use. Seen
            # on a Radeon 680M laptop 2026-08-24 (support item 9). The
            # PyTorch path needs no ONNX Runtime, so demote instead of
            # dying - every backend mode then works on such an install.
            try:
                import onnxruntime  # noqa: F401
            except Exception as e:
                print("[WARNING] ONNX Runtime can't be loaded on this computer - "
                      "capture falls back to slower path.")
                print(f"  import onnxruntime failed: {type(e).__name__}: {e}")
                print("  Running the PyTorch backbone on the CPU for this run. "
                      "A graphics driver update from the GPU maker's site "
                      "(not the laptop maker's) often fixes the fast path.")
                os.environ["USE_ORT"] = "0"

    print(f"  Pipeline env vars set:")
    for k in ("LAYER_DTYPE", "USE_COMPILE", "FOV_MODEL", "FOV_FAST",
              "BODY_INTERM_PRED_LAYERS", "HAND_INTERM_PRED_LAYERS",
              "MHR_NO_CORRECTIVES", "SKIP_KEYPOINT_PROMPT", "USE_ORT"):
        print(f"    {k}={os.environ.get(k)}")

    ckpt_path = CKPT_DIR / "sam-3d-body-dinov3" / "model.ckpt"
    mhr_path = CKPT_DIR / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt"
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator

    emit_progress(5, "Loading SAM 3D Body checkpoint")
    print(f"  Loading checkpoint: {ckpt_path}")
    model, model_cfg = load_sam_3d_body(
        checkpoint_path=str(ckpt_path), mhr_path=str(mhr_path), device=device)

    # Phase 3 cross-platform GPU path: swap the PyTorch DINOv3 backbone for
    # an ONNX Runtime session. Same .onnx file works on CUDA / DirectML /
    # CoreML / CPU EPs; ORT picks the best available at runtime. Decoder
    # and heads stay PyTorch (smaller, run fine on any backend).
    if os.environ.get("USE_ORT", "0") == "1":
        from pipeline.ort_backend import load_ort_backbone_for_sam3dbody
        onnx_path = os.environ.get(
            "ORT_BACKBONE_PATH",
            str(CKPT_DIR / "onnx" / "backbone_dinov3_fp16.onnx"),
        )
        emit_progress(8, "Wiring ORT backbone")
        active_ep = load_ort_backbone_for_sam3dbody(model, onnx_path)
        print(f"  ORT backbone active: {active_ep}")

    # YOLO11 detector instead of vitdet — kills the detectron2/C++ dep.
    # We use 'yolo' (not 'yolo_pose') so the return value is plain boxes
    # array, drop-in compatible with vitdet's output shape.
    # When USE_ORT=1, point at the .onnx export instead of the .pt; ultralytics
    # auto-detects the format and routes through onnxruntime (which itself
    # uses CUDA / DirectML / CoreML EPs based on what's installed).
    from tools.build_detector import HumanDetector
    use_ort = os.environ.get("USE_ORT", "0") == "1"
    # Prefer the .onnx export on the ORT path, but only when it actually
    # exists — the .pt (installed on every backend) is a working fallback
    # for the detector, which is cheap next to the backbone. Keeps a
    # partial / cuda-type install (ONNX optional there) running instead
    # of hard-exiting. MIRRORED in load_detector_only — keep in sync.
    yolo_onnx = Path(os.environ.get(
        "ORT_YOLO_PATH",
        str(CKPT_DIR / "onnx" / "yolo11m_pose.onnx"),
    ))
    yolo_pt = CKPT_DIR / "yolo" / "yolo11m-pose.pt"
    yolo_path = yolo_onnx if (use_ort and yolo_onnx.exists()) else yolo_pt
    if use_ort and not yolo_onnx.exists():
        print(f"  YOLO .onnx not found at {yolo_onnx}; using the .pt detector.")
    if not yolo_path.exists():
        print(f"ERROR: YOLO weights not found: {yolo_path}")
        sys.exit(1)
    emit_progress(15, "Loading YOLO11 detector")
    print(f"  Loading YOLO11 detector: {yolo_path}")
    human_detector = HumanDetector(name="yolo", device=device, model=str(yolo_path))

    # FOV estimator (Fast variant uses MoGe-s automatically per env vars).
    # When USE_ORT=1, swap the MoGe encoder for an ORT session after load —
    # same pattern as the TRT wrapper in build_fov_estimator.py.
    fov_estimator = None
    try:
        from tools.build_fov_estimator import FOVEstimator
        emit_progress(22, "Loading FOV estimator (MoGe-s)")
        print("  Loading FOV estimator (moge2, model=s, fast=1)...")
        fov_estimator = FOVEstimator(name="moge2", device=device)
        # MoGe ORT path is gated separately. The exported encoder has a
        # fixed 512x512 input (its position-embedding sqrt baked a square
        # token grid into the trace), so forcing the pipeline to pre-resize
        # 1920x1080 -> 512x512 destroys aspect-ratio info that MoGe relies
        # on for accurate focal-length estimation. Empirically this pushes
        # body rotation diff from 0.11 deg (backbone-only) to 4.5 deg mean,
        # with 40% of frames > 5 deg. Off by default; opt in via
        # USE_ORT_MOGE=1 if the quality drop is acceptable.
        if use_ort and os.environ.get("USE_ORT_MOGE", "0") == "1":
            from pipeline.ort_backend import load_ort_moge_encoder
            moge_onnx_path = os.environ.get(
                "ORT_MOGE_PATH",
                str(CKPT_DIR / "onnx" / "moge_s_encoder_fp16.onnx"),
            )
            active_ep = load_ort_moge_encoder(
                fov_estimator.fov_estimator, moge_onnx_path,
            )
            fov_estimator.fixed_size = 512
            print(f"  MoGe encoder swapped to ORT: {active_ep}")
            print(f"  FOV pre-resize forced to 512x512 (ORT export constraint)")
        print("  FOV estimator OK")
    except Exception as e:
        print(f"  FOV estimator failed: {e}")

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model, model_cfg=model_cfg,
        human_detector=human_detector, fov_estimator=fov_estimator)
    emit_progress(28, "Estimator ready")
    print("  Estimator ready.")
    return estimator


# NPZ field schema: (npz_key, estimator output key, per-frame fallback).
# This is the contract npz_to_bvh.py reads — extend, don't rename.
_FIELDS = (
    ("global_rots",     "pred_global_rots",    None),
    ("joint_coords",    "pred_joint_coords",   None),
    ("body_pose_params","body_pose_params",     np.zeros(133)),
    ("global_rot_euler","global_rot",           np.zeros(3)),
    ("hand_pose_params","hand_pose_params",     np.zeros(108)),
    ("cam_t",           "pred_cam_t",           np.zeros(3)),
    ("keypoints_2d",    "pred_keypoints_2d",    np.zeros((70, 2))),
)


# Invalid-output guard (support item 6): how many consecutive junk frames
# at the START of a run abort it. Later junk frames are skipped one by one.
_INVALID_ABORT_AFTER = 5
# Joint positions are metres in camera space; nothing a camera films sits
# a kilometre away or inside the lens.
_JOINT_COORD_LIMIT = 1000.0


def _frame_output_sane(gr, jc):
    """False when the model's per-frame output is unusable: any NaN/Inf,
    joint coordinates beyond any plausible camera-space distance, or an
    all-zero skeleton (what a failed kernel often leaves behind)."""
    try:
        gr = np.asarray(gr, dtype=np.float64)
        jc = np.asarray(jc, dtype=np.float64)
        if not (np.isfinite(gr).all() and np.isfinite(jc).all()):
            return False
        if np.abs(jc).max() > _JOINT_COORD_LIMIT:
            return False
        if not np.any(jc):
            return False
        return True
    except Exception:
        return True   # never let the guard itself kill a capture


def _bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    if a.size < 4 or b.size < 4:
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


def _bbox_area(b):
    if b is None:
        return 0.0
    b = np.asarray(b, dtype=np.float64).flatten()
    if b.size < 4:
        return 0.0
    return max(0.0, (b[2] - b[0]) * (b[3] - b[1]))


def _pick_person(result, prev_bbox):
    if len(result) == 1:
        return result[0]
    if prev_bbox is None:
        return max(result, key=lambda p: _bbox_area(p.get("bbox")))
    return max(result, key=lambda p: _bbox_iou(p.get("bbox"), prev_bbox))


def _person_det(person):
    """One person of a frame the way the people tracker keeps them
    (pipeline/people_tracks.py), or None when the model's output for
    them is unusable."""
    gr = to_np(person.get("pred_global_rots"))
    jc = to_np(person.get("pred_joint_coords"))
    bbox = to_np(person.get("bbox"))
    if gr is None or jc is None or bbox is None:
        return None
    if not _frame_output_sane(gr, jc):
        return None
    det = {"global_rots": gr, "joint_coords": jc}
    for name, key, default in _FIELDS[2:]:
        det[name] = to_np(person.get(key, default))
    det["focal_length"] = float(person.get("focal_length", 0.0))
    det["bbox"] = np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]
    return det


def load_detector_only():
    """Load ONLY the YOLO11 person detector — no SAM 3D Body / MoGe / DINOv3.

    The body preview just needs person bounding boxes, so it skips the
    entire 870M-param pipeline: no multi-GB checkpoint load at startup,
    no per-frame mesh inference. Mirrors the YOLO branch of
    load_estimator_fast (same weights, same USE_ORT opt-in)."""
    device = _get_device()
    print(f"  Device: {device}")
    from tools.build_detector import HumanDetector
    use_ort = os.environ.get("USE_ORT", "0") == "1"
    # MIRROR of load_estimator_fast's detector block (constraint: preview
    # must detect exactly what capture would on every backend): .onnx only
    # when use_ort AND it exists, else the always-installed .pt.
    yolo_onnx = Path(os.environ.get(
        "ORT_YOLO_PATH", str(CKPT_DIR / "onnx" / "yolo11m_pose.onnx")))
    yolo_pt = CKPT_DIR / "yolo" / "yolo11m-pose.pt"
    yolo_path = yolo_onnx if (use_ort and yolo_onnx.exists()) else yolo_pt
    if use_ort and not yolo_onnx.exists():
        print(f"  YOLO .onnx not found at {yolo_onnx}; using the .pt detector.")
    if not yolo_path.exists():
        print(f"ERROR: YOLO weights not found: {yolo_path}")
        sys.exit(1)
    if yolo_path == yolo_onnx:
        # ultralytics starts its own ONNX Runtime session for the .onnx
        # detector. A capture has already imported pipeline.ort_backend
        # by this point (the backbone), which on Linux preloads the
        # venv's NVIDIA libraries; the preview never loads the backbone,
        # so do that preload here or the session aborts on cuDNN
        # (exit -6, support item 8). No-op on Windows and macOS.
        try:
            from pipeline.ort_backend import _preload_nvidia_libs
            _preload_nvidia_libs()
        except Exception as e:
            print(f"  NVIDIA library preload skipped: {type(e).__name__}: {e}")
    emit_progress(15, "Loading YOLO11 detector")
    print(f"  Loading YOLO11 detector: {yolo_path}")
    return HumanDetector(name="yolo", device=device, model=str(yolo_path))


def _pick_box(boxes, prev_bbox):
    """Pick one person box from a YOLO [N,4] result: prefer IoU continuity
    with the previous frame, else fall back to the largest-area box."""
    if boxes is None or len(boxes) == 0:
        return None
    if len(boxes) == 1:
        return boxes[0]
    if prev_bbox is not None:
        best = max(boxes, key=lambda b: _bbox_iou(b, prev_bbox))
        if _bbox_iou(best, prev_bbox) > 0:
            return best
    return max(boxes, key=_bbox_area)


def _preview_payload_at(idx, sample_indices, payloads):
    """Overlay payload for frame `idx` from sparse per-sample detections.

    sample_indices: sorted video-frame indices that were actually checked;
    payloads[i] is the drawable result at sample_indices[i], or None on a
    detector miss. Between two DETECTED samples the payload interpolates;
    around a missed sample the nearest detected payload holds up to the
    midpoint of the gap, then frames read as a miss. Returns (a, b, t):
    lerp a->b by t when b is not None, use `a` as-is when b is None, and
    a miss when a is None.
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


def _match_people(a, b):
    """Which boxes of two checked frames are the same person: [(i, j)],
    nearest centres first, and only while the centres are closer than
    half the larger box's diagonal (further than that it is somebody
    else, or the same person after a move the preview should not guess
    across). A preview only: the capture follows people in 3D
    (pipeline/people_tracks.py)."""
    cand = []
    for i, p in enumerate(a):
        for j, q in enumerate(b):
            d = math.hypot((p[0] + p[2] - q[0] - q[2]) / 2.0,
                           (p[1] + p[3] - q[1] - q[3]) / 2.0)
            reach = 0.5 * max(math.hypot(p[2] - p[0], p[3] - p[1]),
                              math.hypot(q[2] - q[0], q[3] - q[1]))
            if d <= reach:
                cand.append((d, i, j))
    pairs, used_a, used_b = [], set(), set()
    for _d, i, j in sorted(cand):
        if i not in used_a and j not in used_b:
            pairs.append((i, j))
            used_a.add(i)
            used_b.add(j)
    return pairs


def _preview_people_at(idx, sample_indices, people, pairs_cache):
    """Everybody's box on frame `idx` from the checked frames (people[i]
    is the list of boxes found on sample_indices[i]). A person found on
    both checked frames either side moves between the two boxes; one
    found on only one of them is shown for the half of the gap nearer
    to it."""
    n = len(sample_indices)
    if n == 0:
        return []
    pos = bisect.bisect_right(sample_indices, idx)
    if pos == 0:
        return people[0]
    if pos >= n:
        return people[n - 1]
    i0, i1 = pos - 1, pos
    a, b = people[i0], people[i1]
    f0, f1 = sample_indices[i0], sample_indices[i1]
    if i0 not in pairs_cache:
        pairs_cache.clear()          # frames are asked for in order
        pairs_cache[i0] = _match_people(a, b)
    pairs = pairs_cache[i0]
    t = (idx - f0) / float(f1 - f0)
    out = [[av + (bv - av) * t for av, bv in zip(a[i], b[j])] for i, j in pairs]
    if idx - f0 <= f1 - idx:
        seen = {i for i, _j in pairs}
        out += [box for i, box in enumerate(a) if i not in seen]
    else:
        seen = {j for _i, j in pairs}
        out += [box for j, box in enumerate(b) if j not in seen]
    return out


def _status_text(frame, text, colour, right=False):
    """The preview's status line, top left of the picture, or top right
    (a multi-camera preview: the camera letter takes the top left of
    each tile)."""
    x = 20
    if right:
        (tw, _th), _base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        x = max(20, frame.shape[1] - tw - 20)
    cv2.putText(frame, text, (x, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colour, 3)


def run_preview(detector, video_path, output_path, step=10,
                start_frame=0, max_frames=0, boxes_out=None, people=False,
                status_right=False):
    """Stepped body bounding-box preview -> MP4, in two passes.

    With `boxes_out` set (the combined body+face preview), the render pass
    is skipped entirely: pass 1's sampled detections are written to a JSON
    sidecar instead, and save_face_data's preview renders the single
    combined video from it - one full-length encode instead of two.

    Pass 1 runs ONLY the YOLO person detector on every Nth frame — the
    expensive part, and its cost is unchanged. Pass 2 re-reads the video
    sequentially and writes EVERY frame at the source frame rate,
    interpolating the detection box between checked frames, so the
    preview plays at full speed with real motion instead of a held-frame
    slideshow. Around a missed sample the nearest good box holds to the
    midpoint of the gap, then frames read as "NO PERSON" (see
    _preview_payload_at). Green box + "BODY OK" = person detected, red
    "NO PERSON" = detector miss. Detector-only (no mesh inference), so
    it stays fast even on long clips. Same CLI/output contract the
    addon's preview path expects (proxy_path = preview.mp4).

    people=True (the multi-person capture modes): a box on EVERYBODY the
    detector finds, and how many there are in place of "BODY OK". No
    numbers on the boxes: who is who comes from the full model's 3D
    positions, which a preview does not run.
    """
    print(f"\nReading video: {video_path}")
    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    end = min(start_frame + max_frames, vid_total) if max_frames > 0 else vid_total
    vid_frames = list(range(start_frame, end, max(1, step)))

    # Scale down for preview (cap long edge at 1920, otherwise native)
    max_edge = 1920
    if max(width, height) > max_edge:
        scale = max_edge / max(width, height)
        out_w = int(width * scale) // 2 * 2
        out_h = int(height * scale) // 2 * 2
    else:
        out_w, out_h = width, height

    print(f"  {len(vid_frames)} frames to check (every {step})")
    print(f"  Preview resolution: {out_w}x{out_h}")

    # ── Pass 1: detection on the sampled frames only ──────────────
    detected = missed = 0
    prev_bbox = None  # last good box, for IoU continuity across samples
    sample_boxes = []  # one entry per vid_frames item: [x1,y1,x2,y2] or None
    sample_people = []  # people=True: one LIST of boxes per vid_frames item
    n_samples = len(vid_frames)
    # Pass 1 spans 30->75 when we also render (pass 2 takes 75->100), or
    # the whole 30->100 when the render is skipped (boxes_out mode).
    p1_span = 70.0 if boxes_out is not None else 45.0

    for count, vid_idx in enumerate(vid_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO expects BGR — the full pipeline converts RGB->BGR before
        # detection, and cv2 already hands us BGR, so pass frame as-is.
        # default_to_full_image=False so a true miss reads as "no person"
        # rather than a full-frame box.
        try:
            boxes = detector.run_human_detection(
                frame, det_cat_id=0, bbox_thr=0.5, default_to_full_image=False)
        except Exception as e:
            print(f"  Frame {vid_idx}: detection FAILED ({e})")
            boxes = None

        box = _pick_box(boxes, prev_bbox)
        if people:
            found = ([[float(v) for v in bx[:4]] for bx in boxes]
                     if boxes is not None else [])
            sample_people.append(found)
        if box is not None:
            sample_boxes.append([float(v) for v in box[:4]])
            prev_bbox = box
            detected += 1
        else:
            sample_boxes.append(None)
            missed += 1

        if n_samples > 0:
            emit_progress(30 + (count + 1) / n_samples * p1_span,
                          f"Body preview {count + 1}/{n_samples}")
        if count % 10 == 0:
            if people:
                k = len(sample_people[-1])
                print(f"  Frame {vid_idx} - "
                      f"{'no person' if k == 0 else '1 person' if k == 1 else f'{k} people'}")
            else:
                print(f"  Frame {vid_idx} - {'body OK' if box is not None else 'no person'}")

    sample_indices = vid_frames[:len(sample_boxes)]

    if boxes_out is not None:
        # Combined body+face preview: the face preview renders the ONE
        # combined video, so hand it the sampled detections and skip this
        # script's render pass entirely.
        cap.release()
        boxes_path = Path(boxes_out)
        boxes_path.parent.mkdir(parents=True, exist_ok=True)
        with open(boxes_path, "w", encoding="utf-8") as f:
            side = {
                "version": 1,
                "fps": fps,
                "start_frame": int(start_frame),
                "end_frame": int(end),
                "sample_indices": [int(i) for i in sample_indices],
                "boxes": sample_boxes,
            }
            if people:
                # everybody's box per checked frame: the face preview
                # looks for a face on each of them (its run_people_preview)
                side["people"] = sample_people[:len(sample_indices)]
            json.dump(side, f)
        total_checked = detected + missed
        print(f"\nDetection samples saved: {boxes_path}")
        print(f"  {total_checked} frames checked: {detected} body OK, {missed} no person")
        if total_checked > 0 and missed > total_checked * 0.3:
            # The [WARNING] form is parsed by the addon and shown in a popup.
            print(f"[WARNING] {missed * 100 // total_checked}% of preview frames had no person")
            print("  The subject may be partly out of frame or too small.")
        return

    # ── Pass 2: render EVERY frame, box interpolated between samples ──
    # Cheap next to detection (decode + draw + encode only). Writing all
    # frames at the native fps keeps the preview's timing identical to
    # the source and 1:1 frame-aligned with it (the combined face preview
    # relies on that alignment when it draws over this file as its
    # background).
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"ERROR: Cannot create output video {output_path}")
        sys.exit(1)

    # Colour: OpenCV decodes every file as BT.601. The frames drawn here
    # are corrected for a source tagged otherwise, so the preview shows
    # the same colours as the original strip (pipeline/video_colour.py).
    # Render pass only - detection above saw the frames as before. A
    # bare copy of this script without pipeline/ renders uncorrected.
    try:
        from pipeline.video_colour import frame_fix, apply_fix
        colour_fix = frame_fix(video_path, log=print)
    except Exception:
        colour_fix = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    n_render = max(1, end - start_frame)
    people_pairs = {}
    for idx in range(start_frame, end):
        ret, frame = cap.read()
        if not ret:
            break
        if colour_fix is not None:
            frame = apply_fix(frame, colour_fix)

        if people:
            here = _preview_people_at(idx, sample_indices, sample_people,
                                      people_pairs)
            for bx in here:
                x1, y1, x2, y2 = (int(v) for v in bx[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA)
            if here:
                _status_text(frame, "1 PERSON" if len(here) == 1 else f"{len(here)} PEOPLE",
                             (0, 255, 0), status_right)
            else:
                _status_text(frame, "NO PERSON", (0, 0, 255), status_right)
            box = None
        else:
            a, b, t = _preview_payload_at(idx, sample_indices, sample_boxes)
            box = [av + (bv - av) * t for av, bv in zip(a, b)] \
                if (a is not None and b is not None) else a
        if people:
            pass
        elif box is not None:
            x1, y1, x2, y2 = (int(v) for v in box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3, cv2.LINE_AA)
            _status_text(frame, "BODY OK", (0, 255, 0), status_right)
        else:
            _status_text(frame, "NO PERSON", (0, 0, 255), status_right)

        cv2.putText(frame, f"Frame {idx}", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if (out_w, out_h) != (width, height):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(frame)

        done = idx - start_frame + 1
        if done % 50 == 0 or done == n_render:
            emit_progress(75 + done / n_render * 25,
                          f"Rendering preview {done}/{n_render}")

    cap.release()
    writer.release()

    total_checked = detected + missed
    size_mb = os.path.getsize(str(output_path)) / (1024 * 1024)
    print(f"\nPreview saved: {output_path} ({size_mb:.1f} MB)")
    if people and sample_people:
        counts = [len(x) for x in sample_people]
        print(f"  {total_checked} frames checked: {min(counts)} to {max(counts)} "
              f"people found per frame, {missed} with nobody")
    else:
        print(f"  {total_checked} frames checked: {detected} body OK, {missed} no person")
    if total_checked > 0 and missed > total_checked * 0.3:
        # The [WARNING] form is parsed by the addon and shown in a popup.
        print(f"[WARNING] {missed * 100 // total_checked}% of preview frames had no person")
        print("  The subject may be partly out of frame or too small.")


def _build_cam_int_from_focal_mm(focal_mm, width, height, sensor_mm=36.0):
    """Build a (1,3,3) pinhole camera-intrinsics tensor from a 35mm-equivalent
    focal length (mm): fx_px = focal_mm * long_edge / sensor_width, with
    sensor_width = 36mm (full-frame / '35mm equivalent'). The 36mm figure is
    the sensor's LONG edge, so keying off max(width, height) keeps the
    conversion orientation-independent: landscape is unchanged (long edge =
    width), and portrait no longer under-reads the focal length (its width
    spans the sensor's short edge, not 36mm). ORIGINAL image-pixel coords -
    the convention the model uses (matches the dummy cam_int in
    sam_3d_body_estimator.py: fx/fy on the diagonal, cx/cy = image centre).
    process_one_image moves it to the right device/dtype."""
    long_edge = max(float(width), float(height))
    fx = float(focal_mm) * long_edge / float(sensor_mm)
    K = torch.eye(3, dtype=torch.float32)
    K[0, 0] = fx
    K[1, 1] = fx
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    return K.unsqueeze(0)


def _interpolate_subsampled(accum, focal_lengths, frame_indices):
    """Fill the gaps left by --capture-skip so the NPZ stays dense (one entry
    per frame). global_rots are SLERP'd per joint; everything else is linear.
    Needs >=2 sampled frames and scipy; returns the inputs unchanged otherwise."""
    sampled = np.asarray(frame_indices, dtype=np.int64)
    if sampled.size < 2:
        return accum, focal_lengths, frame_indices
    try:
        from scipy.interpolate import interp1d
        from scipy.spatial.transform import Rotation, Slerp
    except Exception as e:
        print(f"  [capture-skip] scipy unavailable ({e}); writing sampled "
              f"frames without interpolation.")
        return accum, focal_lengths, frame_indices
    dense = np.arange(int(sampled[0]), int(sampled[-1]) + 1)
    out = {}
    for name, arr_list in accum.items():
        arr = np.asarray(arr_list)
        if arr.shape[0] != sampled.size:
            out[name] = arr
            continue
        if name == "global_rots":
            dense_rots = np.empty((dense.size,) + arr.shape[1:], dtype=arr.dtype)
            for j in range(arr.shape[1]):
                key_rots = Rotation.from_matrix(arr[:, j, :, :])
                dense_rots[:, j] = Slerp(sampled, key_rots)(dense).as_matrix()
            out[name] = dense_rots
        else:
            out[name] = interp1d(sampled, arr, axis=0,
                                 kind="linear")(dense).astype(arr.dtype)
    dense_focal = np.interp(dense, sampled,
                            np.asarray(focal_lengths, dtype=np.float64))
    print(f"  [capture-skip] interpolated {sampled.size} sampled -> "
          f"{dense.size} dense frames")
    return out, [float(f) for f in dense_focal], [int(i) for i in dense]


def main():
    parser = argparse.ArgumentParser(description="Fast SAM 3D Body inference -> NPZ")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="mhr_frames.npz")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--preview", action="store_true",
                        help="Preview mode: draw person detection box, export MP4")
    parser.add_argument("--preview-step", type=int, default=10,
                        help="Run every Nth frame in preview mode (default: 10)")
    parser.add_argument("--preview-status-right", action="store_true",
                        help="Preview mode: draw the status line (BODY OK / "
                             "NO PERSON / the number of people) top right "
                             "instead of top left. A multi-camera preview "
                             "uses it: the camera letter takes the top left "
                             "of each tile.")
    parser.add_argument("--boxes-out", default="",
                        help="Preview mode: write the sampled detections to this "
                             "JSON sidecar and SKIP rendering the preview video "
                             "(the combined body+face preview renders once, in "
                             "save_face_data.py, from this file)")
    parser.add_argument("--no-fov-cache", action="store_true",
                        help="Re-run MoGe FOV on every frame instead of caching "
                             "the first frame's intrinsics. Use only for clips "
                             "where the camera zooms (focal length changes).")
    parser.add_argument("--no-hands", action="store_true",
                        help="Skip hand tracking (inference_type='body'). Drops the "
                             "second per-frame backbone pass -> ~2-3x faster; hands "
                             "stay at rest in the output.")
    parser.add_argument("--focal-mm", type=float, default=0.0,
                        help="Use this focal length (mm, 35mm/full-frame equiv) for "
                             "every frame and skip MoGe estimation. 0 = estimate via "
                             "MoGe (default).")
    parser.add_argument("--capture-skip", type=int, default=0,
                        help="Infer every (N+1)th frame and interpolate the gaps "
                             "(slerp rotations, lerp the rest). 0 = every frame.")
    parser.add_argument("--people", action="store_true",
                        help="Multi-Person Capture: keep EVERY person on every "
                             "frame instead of the most prominent one. Each is "
                             "followed through the clip and saved under people/ "
                             "next to --output, with people.json for the addon; "
                             "--output itself still gets the most prominent "
                             "person (pipeline/people_tracks.py).")
    args = parser.parse_args()

    # Crash-safe log tee. Blender does not persist the
    # capture subprocess console, and a hard GPU/driver fault (seen on shared-
    # memory iGPUs over-committing the DirectML arena) can reboot the box mid-
    # capture, losing every diagnostic. Mirror stdout+stderr to a debug log
    # next to the output, line-buffered with a flush per write, so the log
    # survives an unclean exit. The real streams still receive every line, so
    # mocap.py's [PROGRESS]/[WARNING] parsing is unaffected. Overwritten each run.
    try:
        _log_dir = Path(args.output).resolve().parent
        _log_dir.mkdir(parents=True, exist_ok=True)
        _log_path = _log_dir / "blendcap_capture_debug.log"
        _log_fh = open(_log_path, "w", buffering=1, encoding="utf-8", errors="replace")

        class _Tee:
            def __init__(self, stream, fh):
                self._stream = stream
                self._fh = fh

            def write(self, data):
                self._stream.write(data)
                try:
                    self._fh.write(data)
                    self._fh.flush()
                except Exception:
                    pass
                return len(data)

            def flush(self):
                self._stream.flush()
                try:
                    self._fh.flush()
                except Exception:
                    pass

            def __getattr__(self, name):
                return getattr(self._stream, name)

        sys.stdout = _Tee(sys.stdout, _log_fh)
        sys.stderr = _Tee(sys.stderr, _log_fh)
        print(f"[save_mhr_data] Logging to {_log_path}")
    except Exception as _e:
        print(f"[save_mhr_data] Could not set up debug log: {_e}")

    video = Path(args.video)
    if not video.exists():
        print(f"ERROR: Video not found: {video}")
        sys.exit(1)

    # ── Preview branch (detector-only, fast) ──────────────────────
    # Loads just YOLO and draws bounding boxes — never touches the full
    # SAM 3D Body pipeline, so it's quick to start and quick per frame.
    if args.preview:
        output = Path(args.output) if args.output else Path("preview.mp4")
        emit_progress(2, "Loading person detector")
        print("Loading person detector...")
        detector = load_detector_only()
        emit_progress(30, "Starting body preview")
        run_preview(detector, video, output,
                    step=max(1, args.preview_step),
                    start_frame=args.start_frame, max_frames=args.max_frames,
                    boxes_out=Path(args.boxes_out) if args.boxes_out else None,
                    people=bool(args.people),
                    status_right=bool(args.preview_status_right))
        emit_progress(100, "Body preview complete")
        return

    emit_progress(2, "Loading SAM 3D Body model")
    print("Loading estimator...")
    estimator = load_estimator_fast()

    emit_progress(30, "Opening video")
    print(f"\nReading video: {video}")
    cap = open_video(video)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Picture size, saved with the keypoints so the BVH stage can tell
    # which joints were outside the frame (Hold Cropped Limbs).
    _frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    _frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.max_frames > 0:
        total = min(args.start_frame + args.max_frames, total)
    start_frame = args.start_frame
    n_frames = total - start_frame
    print(f"  {n_frames} frames at {fps:.1f} FPS (starting at frame {start_frame})")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    emit_progress(30, "Starting body inference")
    print("\nProcessing frames...")

    accum = {name: [] for name, _, _ in _FIELDS}
    focal_lengths, frame_indices = [], []
    skipped = 0
    invalid_frames = 0
    multi_person_frames = 0
    bbox_recovered = 0
    prev_bbox = None  # carries forward across frames for YOLO-miss recovery
    cam_int_cached = None  # FOV cache: reuse first frame's MoGe intrinsics
    if args.focal_mm > 0:
        # Manual focal length: build cam_int once and feed it to
        # every frame so MoGe is never called (it stays loaded but idle).
        _w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        _h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam_int_cached = _build_cam_int_from_focal_mm(args.focal_mm, _w, _h)
        print(f"  Manual focal {args.focal_mm:.1f}mm (35mm-equiv) -> cam_int "
              f"built ({_w}x{_h}); MoGe not used")

    # [BlendCap subsampling] infer every (capture_skip+1)th frame; gaps are
    # interpolated after the loop so the NPZ stays dense. inference_type
    # 'body' (--no-hands) drops the per-frame hand backbone pass.
    step = max(1, args.capture_skip + 1)
    inf_type = "body" if args.no_hands else "full"

    # Multi-Person Capture: everybody on the frame goes to the tracker;
    # the single-person bookkeeping below is not used.
    tracker = None
    people_frames = 0       # frames with somebody on them
    people_last = -1        # last frame handed to the tracker
    if args.people:
        from pipeline.people_tracks import Tracker
        tracker = Tracker(fps, (_frame_w, _frame_h), stride=step)

    for idx in range(start_frame, total):
        ret, frame = cap.read()
        if not ret:
            break

        if step > 1 and (idx - start_frame) % step != 0 and idx != total - 1:
            continue  # subsampled-out frame; filled by interpolation later

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            result = estimator.process_one_image(rgb, cam_int=cam_int_cached,
                                                 inference_type=inf_type)
        except Exception as e:
            print(f"  Frame {idx}: FAILED ({e})")
            skipped += 1
            continue

        if tracker is not None:
            k = idx - start_frame
            found = list(result or [])
            dets = [d for d in (_person_det(p) for p in found) if d is not None]
            if found and not dets:
                # every body on the frame came back as junk (see the
                # single-person branch below for what that means)
                invalid_frames += 1
                if people_frames == 0 and invalid_frames >= _INVALID_ABORT_AFTER:
                    print("")
                    print("ERROR: the tracking model is producing invalid results on this computer.")
                    print("  Every frame so far came back as NaN or nonsense, which means the GPU "
                          "or its driver can't run the model correctly (seen on integrated "
                          "graphics). Switch Capture Backend to CPU Only in BlendCap's "
                          "preferences (Edit > Preferences > Add-ons > BlendCap), or use a "
                          "supported NVIDIA GPU.")
                    sys.exit(3)
            want = tracker.step(k, dets)
            if want:
                # Somebody who was being followed is not among them: ask
                # the model about their last box directly (the YOLO-miss
                # recovery of the single-person capture, per person).
                try:
                    boxes = np.stack([b for _t, b in want]).astype(np.float32).reshape(-1, 4)
                    again = list(estimator.process_one_image(
                        rgb, bboxes=boxes, cam_int=cam_int_cached,
                        inference_type=inf_type) or [])
                    # one answer per box, in order; anything else can't
                    # be told apart and is left to the gap filling
                    if len(again) == len(want):
                        tracker.recover(k, want, [_person_det(p) for p in again])
                except Exception as e:
                    print(f"  Frame {idx}: second look FAILED ({e})")
            people_last = k
            if dets:
                people_frames += 1
            if not args.no_fov_cache and cam_int_cached is None and dets:
                cam_int_cached = getattr(estimator, "last_cam_int", None)
                if cam_int_cached is not None:
                    print(f"  FOV cached from frame {idx}; MoGe skipped on later frames")
            frame_pct = (idx - start_frame + 1) / n_frames
            emit_progress(30 + frame_pct * 70, f"Body frame {idx - start_frame + 1}/{n_frames}")
            if idx % 20 == 0 or idx == start_frame:
                print(f"  Frame {idx}/{total}: {len(dets)} people")
            continue

        # YOLO-miss recovery: if the detector returned nothing this
        # frame but we have a prev_bbox from a recent successful frame,
        # retry inference using that bbox directly. Single-subject
        # mocap clips rarely move the subject far between adjacent
        # frames, so this recovers brief tracker misses without
        # adding any new model. Without this, the baseline run loses
        # 4 consecutive frames on Timeline 1.mp4 (frames 73-76).
        if (not result or len(result) == 0) and prev_bbox is not None:
            try:
                bbox_arr = np.asarray(prev_bbox, dtype=np.float32).reshape(-1, 4)
                result = estimator.process_one_image(rgb, bboxes=bbox_arr,
                                                     cam_int=cam_int_cached,
                                                     inference_type=inf_type)
                if result and len(result) > 0:
                    bbox_recovered += 1
                    if idx % 10 == 0 or bbox_recovered <= 5:
                        print(f"  Frame {idx}: YOLO miss, recovered via prev bbox")
            except Exception as e:
                print(f"  Frame {idx}: bbox-fallback retry FAILED ({e})")

        if not result or len(result) == 0:
            print(f"  Frame {idx}: no person detected (even with prev-bbox fallback)")
            skipped += 1
            continue

        if len(result) > 1:
            multi_person_frames += 1
        person = _pick_person(result, prev_bbox)
        gr = to_np(person.get("pred_global_rots"))
        jc = to_np(person.get("pred_joint_coords"))

        if gr is None or jc is None:
            if idx == start_frame:
                print(f"  Frame {idx} keys: {list(person.keys())}")
            print(f"  Frame {idx}: missing pred_global_rots or pred_joint_coords")
            skipped += 1
            continue

        # Output sanity. On some integrated GPUs the DirectML backbone
        # runs "successfully" and returns NaN / absurd numbers for every
        # frame; the first thing that noticed used to be the BVH export,
        # after the whole capture had run (support item 6, 2026-08-01).
        # A frame of junk is skipped like a failed detection; a run whose
        # first frames are ALL junk stops here with a plain explanation
        # instead of grinding through the clip.
        if not _frame_output_sane(gr, jc):
            invalid_frames += 1
            if invalid_frames <= 3 or idx % 20 == 0:
                print(f"  Frame {idx}: invalid model output (NaN or out of range), skipped")
            skipped += 1
            if len(frame_indices) == 0 and invalid_frames >= _INVALID_ABORT_AFTER:
                print("")
                print("ERROR: the tracking model is producing invalid results on this computer.")
                print("  Every frame so far came back as NaN or nonsense, which means the GPU "
                      "or its driver can't run the model correctly (seen on integrated "
                      "graphics). Switch Capture Backend to CPU Only in BlendCap's "
                      "preferences (Edit > Preferences > Add-ons > BlendCap), or use a "
                      "supported NVIDIA GPU.")
                sys.exit(3)
            continue

        accum["global_rots"].append(gr)
        accum["joint_coords"].append(jc)
        for name, key, default in _FIELDS[2:]:
            accum[name].append(to_np(person.get(key, default)))
        focal_lengths.append(float(person.get("focal_length", 0.0)))
        frame_indices.append(idx)
        prev_bbox = person.get("bbox")

        # FOV cache. MoGe (focal-length estimation) is the
        # per-frame hog off-GPU (~95% of CPU capture time), but focal length is
        # constant within a single clip. After the first good frame, grab the
        # MoGe intrinsics the estimator stashed and feed them back as cam_int=
        # on every later frame, which skips MoGe entirely. --no-fov-cache opts
        # out for zoom clips. Helps every backend; biggest win on CPU/DirectML.
        if not args.no_fov_cache and cam_int_cached is None:
            cam_int_cached = getattr(estimator, "last_cam_int", None)
            if cam_int_cached is not None:
                print(f"  FOV cached from frame {idx}; MoGe skipped on later frames")

        frame_pct = (idx - start_frame + 1) / n_frames
        emit_progress(30 + frame_pct * 70, f"Body frame {idx - start_frame + 1}/{n_frames}")
        if idx % 20 == 0 or idx == start_frame:
            print(f"  Frame {idx}/{total} OK  rots={gr.shape} coords={jc.shape}")

    cap.release()

    if tracker is not None:
        from pipeline.people_tracks import write as write_people
        tracks, unsure = tracker.finish()
        if not tracks:
            print("\nERROR: No frames with valid data!")
            sys.exit(1)
        emit_progress(98, "Saving results")
        print(f"\nFollowed {len(tracks)} people over {people_last + 1} frames "
              f"({tracker.recovered} sightings recovered by a second look)")
        if invalid_frames:
            print(f"  {invalid_frames} frame(s) returned invalid model output and were skipped.")
        if people_frames < (people_last + 1) * 0.7 / step:
            pct_missed = 100 - people_frames * step * 100 // max(1, people_last + 1)
            print(f"[WARNING] {pct_missed}% of frames had no person detected")
            print("  The subjects may be partly out of frame or too small.")
        write_people(args.output, tracks, unsure, people_last + 1, fps,
                     (_frame_w, _frame_h), start_frame, skip=step - 1)
        emit_progress(100, "Body tracking complete")
        print(f"\nSaved to {os.path.dirname(os.path.abspath(args.output))}")
        return

    n_good = len(frame_indices)
    if n_good == 0:
        print("\nERROR: No frames with valid data!")
        sys.exit(1)

    emit_progress(98, "Saving results")
    print(f"\nProcessed {n_good} frames ({skipped} skipped, {bbox_recovered} recovered via prev-bbox)")
    if invalid_frames:
        print(f"  {invalid_frames} frame(s) returned invalid model output and were skipped.")

    # [WARNING] lines are parsed by the addon (blendcap/mocap.py) and
    # surfaced in a popup after the run — single line, ASCII, short.
    total_attempted = n_good + skipped
    if total_attempted > 0 and skipped > total_attempted * 0.3:
        pct_missed = skipped * 100 // total_attempted
        print(f"[WARNING] {pct_missed}% of frames had no person detected")
        print("  The subject may be partly out of frame or too small.")
    if n_good > 0 and multi_person_frames > n_good * 0.05:
        print("[WARNING] Multiple people in frame; tracked the most prominent")

    # [BlendCap subsampling] fill the skipped frames so the NPZ stays dense
    # (one entry per frame); no-op when capture-skip is 0.
    if step > 1:
        accum, focal_lengths, frame_indices = _interpolate_subsampled(
            accum, focal_lengths, frame_indices)

    arrays = {name: np.array(accum[name]) for name, _, _ in _FIELDS}
    arrays["focal_length"] = np.array(focal_lengths)
    arrays["frame_indices"] = np.array(frame_indices)
    arrays["fps"] = np.array(fps)
    arrays["frame_size"] = np.array([_frame_w, _frame_h])
    # Schema version stamp. Readers must tolerate its ABSENCE (caches
    # from older builds don't have it) — never require it. Bump when a
    # field is renamed/reshaped so future readers can say "cache is
    # from an older version, re-capture" instead of raising KeyError.
    arrays["blendcap_npz_version"] = np.array(1)
    np.savez_compressed(args.output, **arrays)
    try:
        from pipeline.people_tracks import clear_people
        clear_people(os.path.dirname(os.path.abspath(args.output)))
    except Exception as _e:
        print(f"  (multi-person data of an earlier capture not cleared: {_e})")

    emit_progress(100, "Body tracking complete")
    print(f"\nSaved to {args.output}")
    print(f"  global_rots:   {arrays['global_rots'].shape}")
    print(f"  joint_coords:  {arrays['joint_coords'].shape}")
    print(f"  fps:           {fps}")


if __name__ == "__main__":
    main()
