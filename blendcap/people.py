"""Who is who in a multi-person capture (no Blender in this file).

A multi-person capture keeps EVERY body it detected on every frame.
Who is who is a small table on top of that, in two files in the
capture's folder:

  people.json        written by the capture (pipeline/people_tracks.py):
                     the automatic result. It is never changed
                     afterwards, so a wrong number is fixed by
                     relabelling, not by capturing again. The motion
                     itself is next to it, one file per person as the
                     capture found them (people/track_NN.npz).
  people_edits.json  written by the addon: the user's corrections and
                     the names typed in. Deleting it is "back to the
                     automatic result".

A capture folder with a people.json in it IS a multi-person capture:
that file is what the panel keys everything off.

people.json (version 1)
    {"version": 1,
     "stamp":   "...",            a new one every time the capture
                                  writes this file; edits made against
                                  another stamp do not apply
     "frames":  412,              frames in the capture
     "start_frame": 0,            the video frame the capture starts on
     "main":    1,                the most prominent person (the folder's
                                  own mhr_frames.npz is theirs)
     "cameras": ["A"],            one entry, or the cameras of a
                                  Multi-Camera & Multi-Person take
     "tracks": [                  one per person the capture found
        {"id": 1,                 the automatic number (1, 2, 3...)
         "present": [[0, 411]],   frame ranges, first and last included
         "filled":  [[150, 158]], the parts of those that were filled
                                  in across a gap (an estimate)
         "boxes": [[[x0, y0, x1, y1] or null, ...]]}],
                                  per camera, one box per frame from
                                  the track's first frame to its last;
                                  0..1 of the picture, from the TOP
                                  LEFT; null where that camera has none
     "unsure": [[340, 2, 3]]}     frame, and the two automatic numbers
                                  the capture could not tell apart

people_edits.json
    {"stamp": "...",
     "edits": [{"op": "join",   "a": 2, "b": 4},
               {"op": "swap",   "a": 1, "b": 2, "frame": 340},
               {"op": "split",  "a": 3, "frame": 120},
               {"op": "remove", "a": 5}],
     "names": {"2": "Kate"},
     "heights": {"2": 1.63}}      metres, only for the people it was
                                  typed for, and the key only when
                                  somebody has one (see HEIGHT_MIN).
                                  A multi-camera take keeps ONE: the
                                  last one typed sets the take's scale

Frames are the capture's own, counted from 0. Numbers are never handed
out twice and never close up: joining 2 and 4 leaves 1, 2, 3, 5 (the
user's call, 2026-11-19).

A MULTI-CAMERA take (Multi-Camera & Multi-Person, written by
pipeline/people_cameras.py) keeps every camera's own tracking apart, in
that camera's folder (cam_a/people/track_NN.npz ...), and its people.json
lists PIECES of those tracks, each with three more fields:

    "camera": 1,      which camera saw it (an index into "cameras")
    "source": 2,      that camera's own track number
    "person": 1       who the capture took it for

and its boxes only under its own camera. One person is then several
pieces, one or more per camera, and a correction can be for ONE camera:

    {"op": "swap", "a": 1, "b": 2, "frame": 0, "cam": 1}

("cam" left out: every camera.) That is how two people one camera has the
wrong way round against the others are put right (the user's
requirement, 2026-11-19), without touching what the other cameras saw.
A single-camera capture has none of these fields: a track there is its
own piece, of camera 0, and the person it was taken for is its own
number.

A join's "a" is the person who was SELECTED, "b" the one picked to join
them. The number kept is the lower of the two either way, but where both
are in the picture on the same frame (one body the capture found twice,
which a messy shot does) it is a's motion that is kept and b's that is
left out (the user's call, 2026-11-19: "selected person wins"). "In the
picture" is per camera: the same person in two cameras at once is what a
multi-camera take is.
"""
import hashlib
import json
import math
import os
import time

# Multi-person capture has no face-only capture: a face is found from
# its body's head. With Body off, the Face checkbox and Run Capture are
# greyed out and say why (the user's wording, 2026-11-19); the capture
# answers the same if it is started some other way.
FACE_ONLY_FACE_TIP = ("Face-only capture is not supported in Multi-Person "
                      "Capture. Turn on Body to track faces")
FACE_ONLY_RUN_TIP = ("Face-only capture is not supported in Multi-Person "
                     "Capture. Turn on Body to run the capture")

AUTO_FILE = "people.json"
EDITS_FILE = "people_edits.json"
VERSION = 1

# One colour per number, picked to stay apart from each other over
# video; past the end they come round again.
PALETTE = (
    (0.96, 0.32, 0.24),
    (0.24, 0.60, 1.00),
    (0.34, 0.84, 0.40),
    (1.00, 0.80, 0.18),
    (0.90, 0.42, 0.92),
    (0.22, 0.86, 0.88),
    (1.00, 0.56, 0.14),
    (0.62, 0.52, 1.00),
    (0.72, 0.90, 0.28),
    (1.00, 0.56, 0.70),
)


def colour(number):
    return PALETTE[(max(1, int(number)) - 1) % len(PALETTE)]


def default_name(number):
    return f"Person {int(number)}"


# A person's real height: the Height column of the People list.
#
# ONE CAMERA (Multi-Person Capture). One camera cannot tell a small
# person nearby from a big one further off, so the capture draws
# everybody at the body model's size; measuring who is taller from one
# camera was tried and came out 3 to 6% wrong, sometimes the wrong way
# round (2026-11-24, support log item 20), which is as much as people
# differ. Typed, it is exact: the export draws the person that tall and
# moves them along the camera's line of sight by the same proportion
# (pipeline/people_tracks.py). Each person's height is their own.
#
# SEVERAL CAMERAS (Multi-Camera & Multi-Person). The cameras measure
# how big the people are AGAINST EACH OTHER (pipeline/people_place.py:
# "sizes", kept in the take's CAMERAS_FILE); how big the whole take is
# they cannot tell, and it is taken to be the size at which the people
# average the body model's. So the column shows an estimate for
# everybody, and a height typed for one person sets the size of the
# whole take: everybody else follows by the measured proportions
# (take_factor). One typed height is kept, the last one (the user,
# 2026-11-24: "go with your recommendation").
#
# Not typed = nothing in "heights": the person stays as captured. Below
# HEIGHT_MIN is not a height anybody has, and typing it (0, say) is how
# a typed height is taken away again.
HEIGHT_MIN = 0.5
HEIGHT_MAX = 2.6
# How tall the body model stands, metres: what the column shows for
# somebody nobody typed a height for. For SHOW only (Blender's Python
# cannot open the model's file); the export reads the model's own
# (pipeline/body_skeleton.model_height, 1.727 for the model that ships).
MODEL_HEIGHT = 1.727


def height_of(state, number):
    """The height typed for a person (metres), or 0.0."""
    try:
        h = float((state.get("heights") or {}).get(str(int(number)), 0.0))
    except Exception:
        return 0.0
    return h if HEIGHT_MIN <= h <= HEIGHT_MAX else 0.0


def take_factor(state, sizes, model=MODEL_HEIGHT):
    """How much bigger than the cameras' own estimate a multi-camera
    take is to be made, from the height(s) typed in it: typed height
    over what the estimate makes that person (their measured size times
    the body model's height). 1.0 when nothing was typed. More than one
    typed height (the panel keeps one) gives the middle of them."""
    logs = []
    for key in (state.get("heights") or {}):
        try:
            number = int(key)
        except Exception:
            continue
        h = height_of(state, number)
        size = float((sizes or {}).get(number, 1.0))
        if h > 0 and size > 0 and model > 0:
            logs.append(math.log(h / (size * model)))
    return math.exp(sum(logs) / len(logs)) if logs else 1.0


def shown_heights(state, numbers, cameras=1, sizes=None):
    """What the Height column shows, metres, per person: {number:
    height}. cameras: how many the capture has. sizes: a multi-camera
    take's measured sizes ({number: size}); None or empty when they are
    not known, and everybody then shows the body model's height."""
    if cameras <= 1:
        return {n: (height_of(state, n) or MODEL_HEIGHT) for n in numbers}
    if not sizes:
        return {n: MODEL_HEIGHT for n in numbers}
    factor = take_factor(state, sizes)
    return {n: float(sizes.get(n, 1.0)) * factor * MODEL_HEIGHT for n in numbers}


# Where the cameras of a multi-camera take stand and how big each
# person in it is, worked out once from everybody (pipeline/
# people_tracks.take_cameras) and kept next to the take's table. It
# goes with one reading of who is who: cameras_sig.
CAMERAS_FILE = "cameras_placed.json"


def cameras_sig(auto, labels):
    """What a take's CAMERAS_FILE was worked out from: the capture and
    who is who on every frame of every camera's pieces, as the People
    list has it (labels: replay's). Any correction changes it."""
    h = hashlib.sha1()
    h.update(str(auto["stamp"]).encode())
    for t in sorted(auto["tracks"], key=lambda t: t["id"]):
        h.update(repr((t["id"], t.get("camera", 0), list(labels[t["id"]]))).encode())
    return "1:" + h.hexdigest()


def load_placed(folder, sig):
    """The people's measured sizes of a multi-camera take, for the
    reading of who is who that `sig` stands for: {number: size}. An
    empty dict when it was worked out and could not be told (the
    cameras cannot be placed); None when it has not been worked out
    for this reading yet."""
    try:
        with open(os.path.join(folder, CAMERAS_FILE), encoding="utf-8") as f:
            kept = json.load(f)
        if kept.get("sig") != sig:
            return None
        if kept.get("none"):
            return {}
        return {int(p): float(s) for p, s in kept["sizes"].items()}
    except Exception:
        return None


# ------------------------------------------------------------
# the files
# ------------------------------------------------------------
_AUTO_CACHE = {}        # path -> (checked at, mtime, size, data or None)

# load_auto is asked on every redraw of every viewport, for every user,
# multi-person or not. The disk is only looked at this often (seconds):
# a cache folder on a slow or network drive must not cost a redraw.
RECHECK = 1.0


def forget():
    """Look at the disk again on the next ask."""
    _AUTO_CACHE.clear()


def _ranges(value):
    out = []
    for r in value or []:
        a, b = int(r[0]), int(r[1])
        if b >= a:
            out.append((a, b))
    return sorted(out)


def _read_auto(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or int(raw.get("version", 0)) != VERSION:
        return None
    frames = int(raw["frames"])
    cameras = [str(c) for c in (raw.get("cameras") or ["A"])]
    tracks = []
    for t in raw.get("tracks") or []:
        present = _ranges(t.get("present"))
        if not present:
            continue
        present = [(max(0, a), min(frames - 1, b)) for a, b in present]
        present = [(a, b) for a, b in present if b >= a]
        if not present:
            continue
        boxes = t.get("boxes") or []
        camera = int(t.get("camera", 0))
        if not 0 <= camera < len(cameras):
            continue
        tracks.append({
            "id": int(t["id"]),
            "camera": camera,
            "source": int(t.get("source", t["id"])),
            "person": int(t.get("person", t["id"])),
            "present": present,
            "filled": _ranges(t.get("filled")),
            "first": present[0][0],
            "last": present[-1][1],
            "boxes": [list(b) if b else [] for b in boxes],
        })
    if frames <= 0 or not tracks:
        return None
    if len({t["id"] for t in tracks}) != len(tracks) or min(t["id"] for t in tracks) < 1:
        return None
    if min(t["person"] for t in tracks) < 1:
        return None
    unsure = []
    for u in raw.get("unsure") or []:
        try:
            unsure.append((int(u[0]), int(u[1]), int(u[2])))
        except Exception:
            pass
    return {"stamp": str(raw.get("stamp", "")), "frames": frames,
            "cameras": cameras, "tracks": tracks, "unsure": unsure}


def load_auto(folder):
    """The automatic result of the capture in `folder`, or None when it
    is not a multi-person capture. Read once per change of the file
    (this is asked at draw time)."""
    path = os.path.join(folder, AUTO_FILE)
    now = time.monotonic()
    hit = _AUTO_CACHE.get(path)
    if hit is not None and now - hit[0] < RECHECK:
        return hit[3]
    if len(_AUTO_CACHE) > 32:
        _AUTO_CACHE.clear()
        hit = None
    try:
        st = os.stat(path)
    except OSError:
        _AUTO_CACHE[path] = (now, None, None, None)
        return None
    if hit is not None and hit[1] == st.st_mtime and hit[2] == st.st_size:
        _AUTO_CACHE[path] = (now, hit[1], hit[2], hit[3])
        return hit[3]
    try:
        data = _read_auto(path)
    except Exception:
        data = None
    _AUTO_CACHE[path] = (now, st.st_mtime, st.st_size, data)
    return data


def empty_state(stamp):
    return {"stamp": str(stamp), "edits": [], "names": {}}


def clean_state(state, stamp):
    """`state` as this file writes it, or an empty one when it was made
    against another automatic result."""
    if not isinstance(state, dict) or str(state.get("stamp", "")) != str(stamp):
        return empty_state(stamp)
    edits = []
    for e in state.get("edits") or []:
        try:
            op = str(e["op"])
            if op == "join":
                one = {"op": op, "a": int(e["a"]), "b": int(e["b"])}
            elif op == "swap":
                one = {"op": op, "a": int(e["a"]), "b": int(e["b"]),
                       "frame": int(e["frame"])}
            elif op == "split":
                one = {"op": op, "a": int(e["a"]), "frame": int(e["frame"])}
            elif op == "remove":
                one = {"op": op, "a": int(e["a"])}
            else:
                continue
            if e.get("cam") is not None and int(e["cam"]) >= 0:
                one["cam"] = int(e["cam"])      # for one camera only
            edits.append(one)
        except Exception:
            pass
    names = {}
    for k, v in (state.get("names") or {}).items():
        try:
            if str(v).strip():
                names[str(int(k))] = str(v).strip()
        except Exception:
            pass
    out = {"stamp": str(stamp), "edits": edits, "names": names}
    heights = {}
    for k, v in (state.get("heights") or {}).items():
        try:
            if HEIGHT_MIN <= float(v) <= HEIGHT_MAX:
                heights[str(int(k))] = round(float(v), 4)
        except Exception:
            pass
    if heights:
        out["heights"] = heights
    return out


def load_edits(folder, stamp):
    """The saved corrections for the automatic result `stamp` (empty
    when there are none, or when they belong to an earlier capture)."""
    try:
        with open(os.path.join(folder, EDITS_FILE), encoding="utf-8") as f:
            return clean_state(json.load(f), stamp)
    except Exception:
        return empty_state(stamp)


def save_edits(folder, state):
    """Write the corrections next to the capture. Nothing to keep means
    no file. Returns False when the folder could not be written."""
    path = os.path.join(folder, EDITS_FILE)
    try:
        if not state.get("edits") and not state.get("names")                 and not state.get("heights"):
            if os.path.exists(path):
                os.remove(path)
            return True
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1, sort_keys=True)
        os.replace(tmp, path)
        return True
    except OSError:
        return False


# ------------------------------------------------------------
# applying the corrections
# ------------------------------------------------------------
def _initial(auto):
    labels = {}
    for t in auto["tracks"]:
        row = [0] * auto["frames"]
        for a, b in t["present"]:
            for f in range(a, b + 1):
                row[f] = t["person"]
        labels[t["id"]] = row
    return labels


def cameras_of(auto):
    """{piece id: the index of the camera that saw it}."""
    return {t["id"]: t.get("camera", 0) for t in auto["tracks"]}


def camera_letter(auto, cam):
    try:
        return str(auto["cameras"][int(cam)])
    except Exception:
        return "?"


def _by_camera(labels, cams, cam=None):
    """The rows of `labels` camera by camera: {camera: [rows]}, only
    camera `cam` when one is given."""
    out = {}
    for tid, row in labels.items():
        c = (cams or {}).get(tid, 0)
        if cam is None or c == cam:
            out.setdefault(c, []).append(row)
    return out


def _apply(labels, top, e, cams=None):
    """One edit onto `labels` in place. Returns the highest number used
    so far. An edit that no longer fits is skipped. cams: cameras_of();
    an edit with "cam" touches that camera's rows only."""
    op = e["op"]
    a = e.get("a", 0)
    groups = _by_camera(labels, cams, e.get("cam"))
    rows_in = [row for rows_c in groups.values() for row in rows_c]
    if op == "join":
        b = e["b"]
        if a == b:
            return top
        # both in ONE camera's picture on a frame: a (the selected one)
        # is kept there, b is nobody on that frame in that camera
        for rows_c in groups.values():
            with_a = set()
            for row in rows_c:
                for f, v in enumerate(row):
                    if v == a:
                        with_a.add(f)
            if with_a:
                for row in rows_c:
                    for f, v in enumerate(row):
                        if v == b and f in with_a:
                            row[f] = 0
        keep, gone = min(a, b), max(a, b)
        for row in rows_in:
            for f, v in enumerate(row):
                if v == gone:
                    row[f] = keep
    elif op == "swap":
        b, start = e["b"], max(0, e["frame"])
        if a == b:
            return top
        for row in rows_in:
            for f in range(start, len(row)):
                v = row[f]
                if v == a:
                    row[f] = b
                elif v == b:
                    row[f] = a
    elif op == "split":
        start = max(0, e["frame"])
        new = top + 1
        used = False
        for row in rows_in:
            for f in range(start, len(row)):
                if row[f] == a:
                    row[f] = new
                    used = True
        if used:
            top = new
    elif op == "remove":
        for row in rows_in:
            for f, v in enumerate(row):
                if v == a:
                    row[f] = 0
    return top


def _both(labels, a, b, cams=None, cam=None):
    """Frames on which a and b are both in the picture of one camera
    (camera `cam`, or any): (count, first)."""
    hits = set()
    for rows_c in _by_camera(labels, cams, cam).values():
        has_a, has_b = set(), set()
        for row in rows_c:
            for f, v in enumerate(row):
                if v == a:
                    has_a.add(f)
                elif v == b:
                    has_b.add(f)
        hits |= has_a & has_b
    return (len(hits), min(hits)) if hits else None


def replay(auto, edits):
    """Who each automatic track (each piece) is on each frame once
    `edits` are applied: ({piece id: [number per frame, 0 = nobody]},
    highest number used)."""
    labels = _initial(auto)
    cams = cameras_of(auto)
    top = max(t["person"] for t in auto["tracks"])
    for e in edits:
        top = _apply(labels, top, e, cams)
    return labels, top


def _frames_of(labels, number, cams=None, cam=None):
    seen = set()
    for rows_c in _by_camera(labels, cams, cam).values():
        for row in rows_c:
            for f, v in enumerate(row):
                if v == number:
                    seen.add(f)
    return seen


def check(auto, edits, edit):
    """Why `edit` can't be made on top of `edits` (a sentence for the
    user), or None when it can."""
    labels, _top = replay(auto, edits)
    cams = cameras_of(auto)
    cam = edit.get("cam")
    gone = ("That person is not in this capture any more." if cam is None else
            f"That person is not in camera {camera_letter(auto, cam)}'s picture.")
    op = edit["op"]
    fa = _frames_of(labels, edit["a"], cams, cam)
    if op in ("join", "swap"):
        if edit["a"] == edit["b"]:
            return "Pick a different person."
        fb = _frames_of(labels, edit["b"], cams, cam)
        # a join needs both of them; a swap in ONE camera also puts
        # right a camera that only ever saw one of the two, under the
        # other's number
        if op == "join" and (not fa or not fb):
            return gone
        if op == "swap" and ((cam is None and (not fa or not fb))
                             or (not fa and not fb)):
            return gone
    elif not fa:
        return gone
    if op == "swap":
        f = edit["frame"]
        if not any(x >= f for x in fa) and not any(x >= f for x in fb):
            return "Neither of them is in the picture from this frame on."
    if op == "split":
        f = edit["frame"]
        if not any(x >= f for x in fa):
            return "That person is not in the picture from this frame on."
        if not any(x < f for x in fa):
            return "That person first appears here: there is nothing before this frame to split from."
    return None


def overlap(auto, edits, a, b, cam=None):
    """The frames on which a and b are both in one camera's picture
    once `edits` are applied (camera `cam`, or any): (how many, the
    first), or None. A join of two such people keeps a on those frames
    (see the top of this file)."""
    labels, _top = replay(auto, edits)
    return _both(labels, a, b, cameras_of(auto), cam)


def join_names(names, a, b):
    """The names after joining a and b: the one that stays (the lower
    number) keeps its name, or takes the other's when it has none.
    (The typed heights go the same way.)"""
    keep, gone = str(min(a, b)), str(max(a, b))
    out = dict(names)
    if gone in out:
        if keep not in out:
            out[keep] = out[gone]
        del out[gone]
    return out


# ------------------------------------------------------------
# what the panel and the preview show
# ------------------------------------------------------------
def rows(auto, labels, names=None):
    """One entry per person, in number order: number, name (typed or
    the default), first and last frame, frames present, and the first
    frame the capture was unsure about (-1 = none; the list does not
    show it, the "?" on a box comes from at_frame)."""
    names = names or {}
    frames = {}                 # number -> the frames they are in (any camera)
    for row in labels.values():
        for f, v in enumerate(row):
            if v:
                frames.setdefault(v, set()).add(f)
    seen = {v: [min(s), max(s), len(s)] for v, s in frames.items()}
    unsure = {}
    for f, ta, tb in auto["unsure"]:
        if not 0 <= f < auto["frames"]:
            continue
        for t in (ta, tb):
            v = labels.get(t, [0] * auto["frames"])[f]
            if v:
                unsure.setdefault(v, []).append(f)
    out = []
    for number in sorted(seen):
        first, last, count = seen[number]
        marks = sorted(set(unsure.get(number, [])))
        out.append({"number": number,
                    "name": names.get(str(number)) or default_name(number),
                    "first": first, "last": last, "count": count,
                    "unsure": marks[0] if marks else -1,
                    "unsure_frames": marks})
    return out


def _in(ranges, f):
    return any(a <= f <= b for a, b in ranges)


def at_frame(auto, labels, frame):
    """Everyone in the picture on `frame`: [{number, boxes (one per
    camera, None where that camera has none), filled, unsure}]."""
    if not 0 <= frame < auto["frames"]:
        return []
    unsure_tracks = {t for f, ta, tb in auto["unsure"] if f == frame for t in (ta, tb)}
    out = []
    for t in auto["tracks"]:
        number = labels[t["id"]][frame]
        if not number:
            continue
        i = frame - t["first"]
        boxes = []
        for cam in range(len(auto["cameras"])):
            per_cam = t["boxes"][cam] if cam < len(t["boxes"]) else []
            b = per_cam[i] if 0 <= i < len(per_cam) else None
            boxes.append(tuple(float(v) for v in b) if b else None)
        out.append({"number": number, "boxes": boxes,
                    "camera": t.get("camera", 0),
                    "filled": _in(t["filled"], frame),
                    "unsure": t["id"] in unsure_tracks})
    return out


# ------------------------------------------------------------
# where a piece's motion is
# ------------------------------------------------------------
def camera_folder(folder, auto, cam):
    """The folder holding camera `cam`'s own tracking: the capture's
    folder itself for a single-camera capture, cam_a / cam_b / ... of a
    multi-camera take."""
    if len(auto["cameras"]) <= 1:
        return folder
    return os.path.join(folder, "cam_" + camera_letter(auto, cam).lower())


def track_file(folder, auto, track, face=False):
    """The saved motion (or face capture) of one piece."""
    return os.path.join(camera_folder(folder, auto, track.get("camera", 0)), "people",
                        f"track_{int(track.get('source', track['id'])):02d}"
                        + ("_face.npz" if face else ".npz"))
