# multicam_overlay.py - One "multiview" video from a multi-camera capture.
#
# Tiles each camera's own overlay (its footage with its own skeleton
# drawn on it, written by overlay_track.py) into a single video, with
# the camera's letter in the corner of its tile. That one video is what
# replaces the component clips in the Sequencer after a Multi-Camera
# Capture (blendcap/multicam.py).
#
#   python multicam_overlay.py --layout layout.json --output overlay.mp4
#
# layout.json is written by the addon, which owns the tile layout (the
# same one the syncing view uses), so this script only places pictures:
#
#   {"width": 1920, "height": 1080, "fps": 24.0,
#    "tiles": [{"label": "A", "video": ".../cam_a/overlay.mp4",
#               "rect": [x, y, w, h]}, ...]}
#
# rect is the picture's place in the output, in pixels from the top
# left, already fitted to the camera's shape. Every camera was captured
# over the same timeline frames, so frame k of each overlay is the same
# moment; a camera that runs out early (it should not) leaves its tile
# black rather than stopping the others.

import argparse
import json
import os
import sys

import cv2
import numpy as np


def letter_metrics(canvas_w):
    """(size, pad, drop) of the camera letters, in pixels of the
    multiview video: the same numbers the syncing view draws them with
    (blendcap/multicam_view.letter_spots), so the video that replaces
    the clips looks like the view the user synced them in."""
    base = max(24, int(canvas_w) // 40)
    size = base * 2.8
    return size, base * 0.9, max(2, int(round(size * 0.07)))


def _glyph(text, size):
    """The letter as a (h, w) array of 0..1, tight around it."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.load_default(size=int(round(size)))
        l, t, r, b = font.getbbox(text)
        img = Image.new("L", (max(1, r - l), max(1, b - t)), 0)
        ImageDraw.Draw(img).text((-l, -t), text, fill=255, font=font)
        return np.asarray(img, dtype=np.float32) / 255.0
    except Exception:
        # No scalable font to hand: OpenCV's own, at the same height.
        face = cv2.FONT_HERSHEY_SIMPLEX
        (_w1, h1), _ = cv2.getTextSize(text, face, 1.0, 2)
        scale = 0.72 * size / max(1, h1)
        thick = max(2, int(round(size * 0.07)))
        (tw, th), base = cv2.getTextSize(text, face, scale, thick)
        img = np.zeros((th + base + thick, tw + thick), np.uint8)
        cv2.putText(img, text, (thick // 2, th + thick // 2), face, scale, 255,
                    thick, cv2.LINE_AA)
        rows = np.where(img.max(axis=1) > 0)[0]
        return img[rows[0]:rows[-1] + 1].astype(np.float32) / 255.0


def letter_sprite(text, canvas_w):
    """(letter, shadow, margin): the white letter and its soft drop
    shadow (down and to the right) as two (h, w) arrays of 0..1 on the
    same patch; the letter's top left corner is `margin` pixels in."""
    size, _pad, drop = letter_metrics(canvas_w)
    g = _glyph(text, size)
    blur = max(1.5, size * 0.04)
    m = int(round(drop + 3 * blur)) + 2
    letter = np.zeros((g.shape[0] + 2 * m, g.shape[1] + 2 * m), np.float32)
    letter[m:m + g.shape[0], m:m + g.shape[1]] = g
    shadow = np.zeros_like(letter)
    shadow[m + drop:m + drop + g.shape[0], m + drop:m + drop + g.shape[1]] = g
    shadow = cv2.GaussianBlur(shadow, (0, 0), blur) * 0.8
    return letter, shadow, m


def draw_label(frame, sprite, x, y, pad):
    """The camera letter, top left of its tile at (x, y): white, with a
    soft drop shadow so it reads on any footage."""
    letter, shadow, m = sprite
    x0, y0 = int(round(x + pad)) - m, int(round(y + pad)) - m
    h, w = letter.shape
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(frame.shape[1], x0 + w), min(frame.shape[0], y0 + h)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    a = letter[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0, None]
    s = shadow[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0, None]
    patch = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    patch = patch * (1.0 - s)
    patch = patch * (1.0 - a) + 255.0 * a
    frame[fy0:fy1, fx0:fx1] = np.clip(patch, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Tile per-camera overlays into one multiview video")
    parser.add_argument("--layout", required=True, help="Layout JSON from the addon")
    parser.add_argument("--output", required=True, help="Output video path")
    args = parser.parse_args()

    with open(args.layout, "r", encoding="utf-8") as f:
        layout = json.load(f)
    out_w = int(layout["width"]) // 2 * 2
    out_h = int(layout["height"]) // 2 * 2
    fps = float(layout.get("fps") or 0.0)
    tiles = layout.get("tiles") or []
    if not tiles or out_w <= 0 or out_h <= 0:
        print("ERROR: The multiview layout is empty.")
        sys.exit(1)

    caps, counts = [], []
    for t in tiles:
        cap = cv2.VideoCapture(str(t["video"]))
        if not cap.isOpened():
            print(f"ERROR: Cannot open camera {t.get('label', '?')}'s overlay: "
                  f"{t['video']}")
            sys.exit(1)
        caps.append(cap)
        counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if fps <= 0:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = max(counts) if counts else 0
    if total <= 0:
        print("ERROR: The camera overlays hold no frames.")
        sys.exit(1)
    print(f"  Multiview: {len(tiles)} cameras, {out_w}x{out_h}, {fps:.1f} FPS, "
          f"{total} frames (per camera: {counts})")

    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"ERROR: Cannot create output video {args.output}")
        sys.exit(1)

    rects = []
    for t in tiles:
        x, y, w, h = [int(round(float(v))) for v in t["rect"]]
        x = max(0, min(x, out_w - 2))
        y = max(0, min(y, out_h - 2))
        w = max(2, min(w, out_w - x))
        h = max(2, min(h, out_h - y))
        rects.append((x, y, w, h))

    _size, label_pad, _drop = letter_metrics(out_w)
    sprites = [letter_sprite(str(t.get("label", "")), out_w) if t.get("label") else None
               for t in tiles]
    alive = [True] * len(caps)
    last_pct = -1
    for k in range(total):
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        for i, cap in enumerate(caps):
            x, y, w, h = rects[i]
            if alive[i]:
                ok, frame = cap.read()
                if not ok or frame is None:
                    alive[i] = False
                else:
                    canvas[y:y + h, x:x + w] = cv2.resize(
                        frame, (w, h), interpolation=cv2.INTER_AREA)
            if sprites[i] is not None:
                draw_label(canvas, sprites[i], x, y, label_pad)
        writer.write(canvas)
        pct = int((k + 1) * 100 / total)
        if pct != last_pct:
            last_pct = pct
            print(f"[PROGRESS {pct}] Multiview {k + 1}/{total}", flush=True)

    for cap in caps:
        cap.release()
    writer.release()
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nDone: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
