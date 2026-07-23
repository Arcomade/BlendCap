#!/usr/bin/env python
"""convert_source.py — Convert image sequences or re-rate video for BlendCap pipeline.

Image sequences:  --image-dir DIR [--elements-file FILE] --output OUT --fps FPS
Video re-rate:    --video PATH --output OUT --fps FPS [--start-frame N] [--max-frames N]

Reports [PROGRESS nn] status for BlendCap UI integration.
"""
import argparse
import os
import sys

# Enable OpenEXR support in OpenCV (disabled by default in pip builds)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np


def progress(pct, msg=""):
    print(f"[PROGRESS {min(int(pct), 100)}] {msg}", flush=True)


def _imread_unicode(path, flags=cv2.IMREAD_UNCHANGED):
    """cv2.imread that survives non-ASCII paths on Windows.

    OpenCV's imread uses the narrow char* API there, so a path with
    accented or non-Latin characters (common in usernames) silently
    returns None. Fall back to reading the bytes ourselves and letting
    imdecode do the parsing. Returns None on genuine failure, same as
    imread."""
    img = cv2.imread(path, flags)
    if img is None and os.path.exists(path):
        try:
            data = np.fromfile(path, dtype=np.uint8)
            if data.size:
                img = cv2.imdecode(data, flags)
        except Exception:
            img = None
    return img


def read_image_robust(path):
    """Read an image file, handling float/HDR/multi-channel formats.
    Falls back to imageio for EXR if OpenCV's codec is unavailable."""
    ext = os.path.splitext(path)[1].lower()

    img = None
    if ext in ('.exr', '.hdr'):
        # Try OpenCV first (works if codec was compiled in)
        try:
            img = _imread_unicode(path, cv2.IMREAD_UNCHANGED)
        except cv2.error:
            img = None
        # Fallback: imageio (if available)
        if img is None:
            try:
                try:
                    import imageio.v3 as iio
                    raw = iio.imread(path)
                except ImportError:
                    import imageio as iio
                    raw = iio.imread(path)
                img = np.asarray(raw, dtype=np.float32)
                # imageio returns RGB; convert to BGR for cv2.VideoWriter
                if len(img.shape) == 3 and img.shape[2] >= 3:
                    img = img[:, :, :3][:, :, ::-1].copy()
            except ImportError:
                print(f"[ERROR] Cannot read {ext.upper()} files. OpenCV EXR codec is disabled "
                      f"and imageio is not installed. Install via: pip install imageio",
                      file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"[ERROR] Failed to read {path}: {e}", file=sys.stderr)
                return None
    else:
        img = _imread_unicode(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        return None
    # Float images (EXR, HDR): clamp to [0,1] and convert to 8-bit
    if img.dtype != np.uint8:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)
    # Ensure 3-channel BGR for VideoWriter
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def compute_output_size(w, h, max_edge=1920):
    """Cap resolution at max_edge, maintain aspect, ensure even dimensions."""
    long = max(w, h)
    if long > max_edge:
        scale = max_edge / long
        w = int(w * scale)
        h = int(h * scale)
    return w // 2 * 2, h // 2 * 2


def convert_image_sequence(args):
    progress(0, "Reading image list...")

    # Load element list from file or auto-detect from directory
    elements = []
    if args.elements_file and os.path.exists(args.elements_file):
        with open(args.elements_file) as f:
            elements = [line.strip() for line in f if line.strip()]

    if not elements and args.image_dir:
        img_exts = {'.png', '.jpg', '.jpeg', '.exr', '.tiff', '.tif', '.bmp', '.webp'}
        elements = sorted(f for f in os.listdir(args.image_dir)
                          if os.path.splitext(f)[1].lower() in img_exts)

    if not elements:
        print("[ERROR] No image files found.", file=sys.stderr)
        sys.exit(1)

    # Apply trim offsets
    start = args.start_frame
    end = len(elements) - args.offset_end if args.offset_end > 0 else len(elements)
    if args.max_frames > 0:
        end = min(start + args.max_frames, end)
    elements = elements[max(0, start):max(0, end)]

    if not elements:
        print("[ERROR] No frames after trim.", file=sys.stderr)
        sys.exit(1)

    n_frames = len(elements)
    progress(2, f"Converting {n_frames} frames...")

    # Read first frame for dimensions
    first_path = os.path.join(args.image_dir, elements[0])
    frame = read_image_robust(first_path)
    if frame is None:
        print(f"[ERROR] Cannot read: {first_path}", file=sys.stderr)
        sys.exit(1)

    h, w = frame.shape[:2]
    out_w, out_h = compute_output_size(w, h)
    needs_resize = (out_w != w or out_h != h)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (out_w, out_h))
    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
    if not writer.isOpened():
        print("[ERROR] Cannot create output video.", file=sys.stderr)
        sys.exit(1)

    report_interval = max(1, n_frames // 50)
    try:
        for i, fname in enumerate(elements):
            img = frame if i == 0 else read_image_robust(os.path.join(args.image_dir, fname))
            if img is None:
                print(f"[WARNING] Skipping unreadable frame: {fname}")
                continue

            if needs_resize or img.shape[:2] != (out_h, out_w):
                img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)

            writer.write(img)

            if (i + 1) % report_interval == 0 or i == n_frames - 1:
                progress(2 + (i + 1) / n_frames * 96, f"Frame {i + 1}/{n_frames}")
    finally:
        writer.release()

    progress(100, "Conversion complete")


def convert_video(args):
    progress(0, "Opening video...")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {args.video}", file=sys.stderr)
        sys.exit(1)

    try:
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_w, out_h = compute_output_size(src_w, src_h)
        needs_resize = (out_w != src_w or out_h != src_h)

        use_match = (args.mode == "match" and args.source_fps > 0
                     and abs(args.source_fps - args.fps) > 0.5)

        if use_match:
            # --- MATCH TIMING MODE ---
            # Use the same source frame range as KEEP mode (start_frame +
            # max_frames), then compute how many output frames that range
            # produces at the target FPS. High->low = fewer output frames
            # (drops), low->high = more output frames (duplicates).
            remaining = max(0, total_frames - args.start_frame)
            source_count = min(args.max_frames, remaining) if args.max_frames > 0 else remaining
            n_output = max(1, round(source_count * args.fps / args.source_fps))

            if source_count <= 0:
                print("[ERROR] No frames to convert.", file=sys.stderr)
                sys.exit(1)

            progress(2, f"Resampling {source_count} source frames -> "
                     f"{n_output} output frames "
                     f"({args.source_fps:.0f}fps -> {args.fps:.0f}fps)...")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(args.output, fourcc, args.fps, (out_w, out_h))
            writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
            if not writer.isOpened():
                print("[ERROR] Cannot create output video.", file=sys.stderr)
                sys.exit(1)

            report_interval = max(1, n_output // 50)
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
                current_src = args.start_frame
                last_frame = None
                written = 0
                source_exhausted = False

                for out_i in range(n_output):
                    target_src = args.start_frame + int(out_i * args.source_fps / args.fps)

                    # Read forward to the target source frame
                    while current_src <= target_src:
                        ret, frame = cap.read()
                        if not ret:
                            source_exhausted = True
                            break
                        if needs_resize:
                            frame = cv2.resize(frame, (out_w, out_h),
                                               interpolation=cv2.INTER_AREA)
                        last_frame = frame
                        current_src += 1

                    if last_frame is None or source_exhausted:
                        break

                    writer.write(last_frame)
                    written += 1

                    if (out_i + 1) % report_interval == 0 or out_i == n_output - 1:
                        progress(2 + (out_i + 1) / n_output * 96,
                                 f"Frame {out_i + 1}/{n_output}")
            finally:
                writer.release()

            if written == 0:
                print("[ERROR] No frames written.", file=sys.stderr)
                sys.exit(1)

        else:
            # --- KEEP ALL FRAMES MODE (default) ---
            # Sequential 1:1 copy, FPS tag changes playback speed.
            if args.start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

            remaining = max(0, total_frames - args.start_frame)
            n_frames = min(args.max_frames, remaining) if args.max_frames > 0 else remaining

            if n_frames <= 0:
                print("[ERROR] No frames to convert.", file=sys.stderr)
                sys.exit(1)

            progress(2, f"Converting {n_frames} frames at {args.fps:.1f}fps...")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(args.output, fourcc, args.fps, (out_w, out_h))
            writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
            if not writer.isOpened():
                print("[ERROR] Cannot create output video.", file=sys.stderr)
                sys.exit(1)

            report_interval = max(1, n_frames // 50)
            try:
                written = 0
                for i in range(n_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if needs_resize:
                        frame = cv2.resize(frame, (out_w, out_h),
                                           interpolation=cv2.INTER_AREA)

                    writer.write(frame)
                    written += 1

                    if (i + 1) % report_interval == 0 or i == n_frames - 1:
                        progress(2 + (i + 1) / n_frames * 96, f"Frame {i + 1}/{n_frames}")
            finally:
                writer.release()

            if written == 0:
                print("[ERROR] No frames written.", file=sys.stderr)
                sys.exit(1)
    finally:
        cap.release()

    progress(100, "Conversion complete")


def main():
    parser = argparse.ArgumentParser(description="Convert source media to MP4 for BlendCap pipeline")
    parser.add_argument("--video", help="Source video path (for video conversion/re-rate)")
    parser.add_argument("--image-dir", help="Image sequence directory")
    parser.add_argument("--elements-file", help="Text file listing image filenames in order (one per line)")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--fps", type=float, required=True, help="Output framerate")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame offset")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to convert (0 = all)")
    parser.add_argument("--offset-end", type=int, default=0, help="End trim offset (image sequences)")
    parser.add_argument("--source-fps", type=float, default=0, help="Source video FPS (for rate-aware conversion)")
    parser.add_argument("--mode", choices=["match", "keep"], default="keep",
                        help="Conversion mode: 'match' = temporal resampling (drop/duplicate frames), "
                             "'keep' = 1:1 sequential copy (default)")
    args = parser.parse_args()

    if args.image_dir:
        convert_image_sequence(args)
    elif args.video:
        convert_video(args)
    else:
        print("[ERROR] Provide either --video or --image-dir", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
