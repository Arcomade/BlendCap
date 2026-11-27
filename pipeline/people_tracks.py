# pipeline/people_tracks.py
# =========================
# Multi-person capture: from "every body the model returned on every
# frame" to one track per person, and back out again as an ordinary
# single-person capture for whoever the BVH is for.
#
# The body model returns a list of people per frame, in no order and
# with no identity. This file:
#
#   link()        follows each person from frame to frame. A person is
#                 matched to where they were (their position in front of
#                 the camera, the size and place of their box), and one
#                 who is lost is picked up again when somebody appears
#                 close to where they would be by then: near, the same
#                 size, moving the same way (the user's rule,
#                 2026-11-18). Too far or too long and it is a new
#                 person; the panel's Join puts two of them together.
#   write()       saves every track, complete, in the capture's folder:
#                   people/track_NN.npz   one person as the capture found
#                                         them, in the ordinary capture
#                                         layout, every frame from their
#                                         first to their last (frames the
#                                         model missed are filled in)
#                   people.json           what the panel shows: who is
#                                         there on which frames, their
#                                         boxes, where the capture was
#                                         not sure (blendcap/people.py)
#                   mhr_frames.npz        the most prominent person, so
#                                         the folder still reads as a
#                                         capture to everything else
#   extract_person()  builds the capture of ONE person as the panel now
#                 has them (after any Swap / Join / Split), into a folder
#                 of its own that Generate BVH and Bake use like any
#                 other capture. Nothing is tracked again: who is who is
#                 only a table on top of the saved tracks.
#   extract_person_cameras()  the same for a person of a Multi-Camera &
#                 Multi-Person take: every camera captured its own
#                 people with the functions above, pipeline/
#                 people_cameras.py matched them across the cameras, and
#                 here the person is taken out of each camera's tracks
#                 and those are merged by the ordinary multi-camera
#                 merge (pipeline/two_camera_merge.py). Their folder is
#                 then a merged take like any other.
#
# Frames: `frame_indices` in the NPZ files are the video's own frame
# numbers, as everywhere else. people.json counts from the capture's
# first frame (0), which is what the panel and its corrections use.

import importlib.util
import json
import os
import time

import numpy as np

FIELDS = ("global_rots", "joint_coords", "body_pose_params",
          "global_rot_euler", "hand_pose_params", "cam_t", "keypoints_2d")
TRACK_DIR = "people"
NPZ_VERSION = 1

# --- following people from frame to frame -------------------------------
# cam_t is the person in front of the camera, metres: x right, y down,
# z away. Depth from one camera is the noisy one, so it counts for less.
DEPTH_WEIGHT = 0.35
# How far a person can be from where they were expected and still be
# them. From one captured frame to the next: this much, plus what a
# running person covers in that time. Once they have been LOST, the
# allowance grows only slowly: whoever turns up has to be about where
# the lost person would be by then, not merely somewhere near.
REACH_M = 0.45
MOVE_M_PER_S = 3.0
LOST_M_PER_S = 1.5
REACH_MAX_M = 2.5
# A lost person is waited for this long (seconds); after that anybody who
# appears is a new person.
MAX_GAP_S = 1.0
# Box height may change by this factor from one captured frame to the
# next (a crouch, a raised arm), and by less across a loss: the same
# person comes back the same size.
SIZE_RATIO = 1.9
SIZE_RATIO_LOST = 1.4
# A person the detector lost is looked for again at their last box for
# up to this long (seconds), frame by frame.
RECOVER_S = 2.0
# Shorter than this (seconds) is a false detection, not a person.
MIN_TRACK_S = 0.25
# Two matches whose costs are this close could as well be the other way
# round: flagged for the user to look at. Wide on purpose: it only ever
# triggers for two people within about half a metre of each other with
# their boxes overlapping, and one camera's depth is not good enough to
# call those.
UNSURE_MARGIN = 0.6
# The model sometimes returns the same person twice on a frame: the same
# spot in front of the camera (true distance, depth in full), boxes on
# top of each other, and the same pose in the picture. Two real people
# never pass all three, however close they stand.
SAME_PERSON_M = 0.25
SAME_PERSON_IOU = 0.45
SAME_PERSON_POSE = 0.12         # mean keypoint distance, in box heights
BIG = 1e6


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _area(b):
    return max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))


def _dist(a, b):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(d[0] ** 2 + d[1] ** 2 + (DEPTH_WEIGHT * d[2]) ** 2))


def same_body(p, q):
    """True when two detections on one frame are the same body."""
    d = np.asarray(p["cam_t"], dtype=np.float64) - np.asarray(q["cam_t"], dtype=np.float64)
    if float(np.sqrt((d * d).sum())) > SAME_PERSON_M:
        return False
    if _iou(p["bbox"], q["bbox"]) < SAME_PERSON_IOU:
        return False
    h = max(1.0, float(min(p["bbox"][3] - p["bbox"][1], q["bbox"][3] - q["bbox"][1])))
    ka = np.asarray(p["keypoints_2d"], dtype=np.float64)
    kb = np.asarray(q["keypoints_2d"], dtype=np.float64)
    apart = float(np.sqrt(((ka - kb) ** 2).sum(axis=1)).mean()) / h
    return apart < SAME_PERSON_POSE


def dedupe(people):
    """One entry per person on a frame: of two detections that are the
    same body the larger box is kept."""
    keep = []
    for p in sorted(people, key=lambda p: -_area(p["bbox"])):
        if not any(same_body(p, q) for q in keep):
            keep.append(p)
    return keep


class _Track:
    def __init__(self, f, det):
        self.frames = [f]
        self.dets = [det]
        self.vel = np.zeros(3)
        self.last_seen = f          # the last frame the DETECTOR found them on

    @property
    def last(self):
        return self.frames[-1]

    def expected(self, f, fps):
        """Where this person would be on frame f: carried on at their
        last speed for at most half a second, then standing."""
        g = min(f - self.last, fps * 0.5)
        return np.asarray(self.dets[-1]["cam_t"], dtype=np.float64) + self.vel * g

    def add(self, f, det, seen=True):
        g = max(1, f - self.last)
        step = (np.asarray(det["cam_t"], dtype=np.float64)
                - np.asarray(self.dets[-1]["cam_t"], dtype=np.float64)) / g
        step[2] *= DEPTH_WEIGHT          # depth jitter is not movement
        self.vel = 0.6 * self.vel + 0.4 * step
        self.frames.append(f)
        self.dets.append(det)
        if seen:
            self.last_seen = f

    def trim(self):
        """Drop what was recovered after the detector's last sighting:
        nothing ever confirmed those frames were still this person."""
        while self.frames and self.frames[-1] > self.last_seen:
            self.frames.pop()
            self.dets.pop()


def _cost(track, det, f, fps, stride=1):
    """How unlike `det` is the person of `track` on frame f (lower is
    more alike), or BIG when it cannot be them. stride: how many frames
    apart the capture looks (Capture Skip + 1); a gap of that size is
    simply the next look, anything beyond it is time the person was
    lost."""
    g = f - track.last
    last = track.dets[-1]
    d = min(_dist(det["cam_t"], track.expected(f, fps)),
            _dist(det["cam_t"], last["cam_t"]))
    lost = max(0, g - stride)
    reach = min(REACH_MAX_M, REACH_M + MOVE_M_PER_S * min(g, stride) / fps
                + LOST_M_PER_S * lost / fps)
    if d > reach:
        return BIG
    h0 = max(1.0, float(last["bbox"][3] - last["bbox"][1]))
    h1 = max(1.0, float(det["bbox"][3] - det["bbox"][1]))
    ratio = max(h0, h1) / min(h0, h1)
    if ratio > (SIZE_RATIO_LOST if lost > 1 else SIZE_RATIO):
        return BIG
    return d + 0.6 * (1.0 - _iou(last["bbox"], det["bbox"])) + 0.5 * float(np.log(ratio))


def keypoint_box(det, frame_size):
    """A box round a person's 2D keypoints, with a little air, inside
    the picture."""
    kp = np.asarray(det["keypoints_2d"], dtype=np.float64)
    x0, y0 = np.nanmin(kp[:, 0]), np.nanmin(kp[:, 1])
    x1, y1 = np.nanmax(kp[:, 0]), np.nanmax(kp[:, 1])
    mx, my = (x1 - x0) * 0.08, (y1 - y0) * 0.06
    w, h = float(frame_size[0]), float(frame_size[1])
    return np.array([max(0.0, x0 - mx), max(0.0, y0 - my * 1.6),
                     min(w, x1 + mx), min(h, y1 + my)], dtype=np.float32)


class Tracker:
    """Follows the people of a capture frame by frame.

    Per frame: step() with the people the model found, which answers
    with the boxes of anybody it was following and did NOT get back.
    The capture runs the model again on exactly those boxes (what the
    single-person capture does when its detector misses a frame) and
    hands the answers to recover(). A person kept alive that way is only
    kept if the detector finds them again later: what was recovered
    after their last real sighting is thrown away at the end, so a
    person who has really gone is not carried on as a guess."""

    def __init__(self, fps, frame_size, stride=1):
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.size = (float(frame_size[0]), float(frame_size[1]))
        self.stride = max(1, int(stride))
        self.max_gap = self.stride + max(1, int(round(MAX_GAP_S * self.fps)))
        self.max_recover = max(1, int(round(RECOVER_S * self.fps)))
        self.live, self.done, self.unsure = [], [], []
        self._now = []
        self.recovered = 0

    def step(self, f, people):
        from scipy.optimize import linear_sum_assignment
        people = dedupe(people)
        for t in [t for t in self.live
                  if f - t.last > self.max_gap or f - t.last_seen > self.max_recover]:
            self.live.remove(t)
            self.done.append(t)
        live = list(self.live)
        taken, matched = set(), set()
        if live and people:
            cost = np.array([[_cost(t, p, f, self.fps, self.stride) for p in people]
                             for t in live])
            rows, cols = linear_sum_assignment(cost)
            pairs = [(r, c) for r, c in zip(rows, cols) if cost[r, c] < BIG]
            for r, c in pairs:
                # could these two have been matched the other way round?
                for r2, c2 in pairs:
                    if r2 <= r:
                        continue
                    if cost[r, c2] < BIG and cost[r2, c] < BIG:
                        other = cost[r, c2] + cost[r2, c]
                        if other - (cost[r, c] + cost[r2, c2]) < UNSURE_MARGIN:
                            self.unsure.append((f, live[r], live[r2]))
            for r, c in pairs:
                live[r].add(f, people[c])
                taken.add(c)
                matched.add(r)
        lost = [t for i, t in enumerate(live) if i not in matched]
        for c, p in enumerate(people):
            if c not in taken:
                self.live.append(_Track(f, p))
        self._now = [t.dets[-1] for t in self.live if t.last == f]
        # ask for another look only where the person was a moment ago and
        # their box is still (mostly) in the picture
        want = []
        w, h = self.size
        for t in lost:
            box = np.asarray(t.dets[-1]["bbox"], dtype=np.float32)[:4]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if 0 <= cx <= w and 0 <= cy <= h and _area(box) > 400:
                want.append((t, box))
        return want

    def recover(self, f, want, results):
        """results: what the model returned for the boxes of `want`, in
        the same order (None where it returned nothing usable)."""
        for (t, box), det in zip(want, results):
            if det is None:
                continue
            det = dict(det)
            det["bbox"] = keypoint_box(det, self.size)
            det["recovered"] = True
            # the same body as somebody already found on this frame: the
            # model has locked onto whoever stands in that box now
            if any(same_body(det, q) for q in self._now):
                continue
            if _cost(t, det, f, self.fps, self.stride) >= BIG:
                continue
            t.add(f, det, seen=False)
            self._now.append(det)
            self.recovered += 1

    def finish(self):
        """([(frame numbers, detections)], [(frame, i, j)]): the people,
        numbered in order of appearance (those there on the same frame,
        left to right), and where two of them could have been mixed up."""
        every = self.done + self.live
        for t in every:
            t.trim()
        min_len = max(2, int(round(MIN_TRACK_S * self.fps)))
        kept = [t for t in every if len(t.frames) >= 2
                and t.last - t.frames[0] + 1 >= min_len]
        kept.sort(key=lambda t: (t.frames[0],
                                 float(t.dets[0]["bbox"][0] + t.dets[0]["bbox"][2])))
        index = {id(t): i for i, t in enumerate(kept)}
        flagged = sorted({(f, index[id(a)], index[id(b)]) for f, a, b in self.unsure
                          if id(a) in index and id(b) in index
                          and f <= a.last and f <= b.last})
        return [(t.frames, t.dets) for t in kept], flagged


def link(frames, fps, frame_size=(1920, 1080)):
    """The people of already-captured frames, with no second look at
    anybody (see Tracker for the capture's own loop). frames[i]: the
    list of people found on frame i, or None for a frame that was not
    looked at (Capture Skip)."""
    looked = [f for f, people in enumerate(frames) if people is not None]
    stride = min(np.diff(looked)) if len(looked) > 1 else 1
    tr = Tracker(fps, frame_size, stride=int(stride))
    for f, people in enumerate(frames):
        if people is not None:
            tr.step(f, people)
    return tr.finish()


# --- filling the frames in between ---------------------------------------
def fill(values, sampled, dense):
    """`values` given on the frames `sampled`, on every frame of `dense`
    (which lies inside sampled's range): rotations turn the short way
    between their neighbours, everything else moves in a straight line."""
    values = np.asarray(values)
    sampled = np.asarray(sampled, dtype=np.int64)
    dense = np.asarray(dense, dtype=np.int64)
    if sampled.size == dense.size and np.array_equal(sampled, dense):
        return values.copy()
    if sampled.size == 1:
        return np.repeat(values[:1], dense.size, axis=0)
    if values.ndim == 4 and values.shape[-2:] == (3, 3):
        from scipy.spatial.transform import Rotation, Slerp
        out = np.empty((dense.size,) + values.shape[1:], dtype=values.dtype)
        for j in range(values.shape[1]):
            out[:, j] = Slerp(sampled, Rotation.from_matrix(values[:, j]))(dense).as_matrix()
        return out
    flat = values.reshape(values.shape[0], -1).astype(np.float64)
    out = np.empty((dense.size, flat.shape[1]))
    for k in range(flat.shape[1]):
        out[:, k] = np.interp(dense, sampled, flat[:, k])
    return out.reshape((dense.size,) + values.shape[1:]).astype(values.dtype)


def _dense_track(frames, dets, skip):
    """One track on every frame from its first to its last. Returns the
    arrays (fields, bbox, focal_length), the frame numbers, and which of
    them were filled in across a gap (not counting Capture Skip's own
    regular gaps)."""
    sampled = np.asarray(frames, dtype=np.int64)
    dense = np.arange(sampled[0], sampled[-1] + 1)
    out = {name: fill(np.stack([np.asarray(d[name]) for d in dets]), sampled, dense)
           for name in FIELDS}
    out["bbox"] = fill(np.stack([np.asarray(d["bbox"], dtype=np.float32)[:4] for d in dets]),
                       sampled, dense)
    out["focal_length"] = np.interp(dense, sampled, [float(d.get("focal_length", 0.0)) for d in dets])
    seen = np.zeros(dense.size, dtype=bool)
    seen[sampled - sampled[0]] = True
    filled = ~seen
    again = np.array([bool(d.get("recovered")) for d in dets])
    if skip > 0:
        # frames Capture Skip left out are ordinary, not a lost person:
        # only the frames of a gap longer than its step count as filled
        gaps = np.diff(sampled)
        filled[:] = False
        for a, g in zip(sampled[:-1], gaps):
            if g > skip + 1:
                filled[a - sampled[0] + 1:a - sampled[0] + g] = True
    # frames the detector missed and the model was asked about again
    filled[sampled[again] - sampled[0]] = True
    return out, dense, filled


def _ranges(mask, first):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append([first + start, first + i - 1])
            start = None
    if start is not None:
        out.append([first + start, first + len(mask) - 1])
    return out


def track_path(folder, track_id):
    return os.path.join(folder, TRACK_DIR, f"track_{int(track_id):02d}.npz")


def face_path(folder, track_id):
    return os.path.join(folder, TRACK_DIR, f"track_{int(track_id):02d}_face.npz")


def clear_people(folder):
    """Take the multi-person data out of a capture's folder (an ordinary
    capture has just been written there)."""
    for name in ("people.json", "people_edits.json"):
        try:
            os.remove(os.path.join(folder, name))
        except OSError:
            pass
    sub = os.path.join(folder, TRACK_DIR)
    if os.path.isdir(sub):
        for name in os.listdir(sub):
            if name.startswith("track_") and name.endswith(".npz"):
                os.remove(os.path.join(sub, name))
        try:
            os.rmdir(sub)
        except OSError:
            pass


def write(body_out, tracks, unsure, n, fps, frame_size, start_frame, skip=0, log=print):
    """Save the people of a capture (see the top of this file): what
    Tracker.finish() returned, for a capture of `n` frames whose first
    is video frame start_frame. `body_out` is the capture's ordinary
    body file (mhr_frames.npz); everything else goes next to it.
    Returns the number of people."""
    if not tracks:
        return 0
    folder = os.path.dirname(os.path.abspath(body_out))
    w, h = float(frame_size[0]), float(frame_size[1])
    os.makedirs(os.path.join(folder, TRACK_DIR), exist_ok=True)
    for name in os.listdir(os.path.join(folder, TRACK_DIR)):
        if name.startswith("track_") and name.endswith(".npz"):
            os.remove(os.path.join(folder, TRACK_DIR, name))

    listed, weight = [], []
    for i, (tf, dets) in enumerate(tracks):
        tid = i + 1
        arrays, dense, filled = _dense_track(tf, dets, skip)
        save = {name: arrays[name] for name in FIELDS}
        save["focal_length"] = arrays["focal_length"]
        save["frame_indices"] = dense + int(start_frame)
        save["fps"] = np.array(fps)
        save["frame_size"] = np.array([int(w), int(h)])
        save["blendcap_npz_version"] = np.array(NPZ_VERSION)
        save["bbox"] = arrays["bbox"]
        save["filled"] = filled
        np.savez_compressed(track_path(folder, tid), **save)
        box = arrays["bbox"] / np.array([w, h, w, h], dtype=np.float64)
        box = np.clip(box, 0.0, 1.0)
        listed.append({"id": tid,
                       "present": [[int(dense[0]), int(dense[-1])]],
                       "filled": _ranges(filled, int(dense[0])),
                       "boxes": [[[round(float(v), 4) for v in b] for b in box]]})
        weight.append(float(((arrays["bbox"][:, 2] - arrays["bbox"][:, 0])
                             * (arrays["bbox"][:, 3] - arrays["bbox"][:, 1])).sum()))
        log(f"  Person {tid}: frames {int(dense[0])} to {int(dense[-1])}"
            + (f", {int(filled.sum())} filled in" if filled.any() else ""))

    # The most prominent person doubles as the folder's ordinary capture.
    main = int(np.argmax(weight)) + 1
    with np.load(track_path(folder, main)) as z:
        np.savez_compressed(body_out,
                            **{k: z[k] for k in z.files if k not in ("bbox", "filled")})

    data = {"version": 1,
            "stamp": f"{int(time.time())}-{len(listed)}-{n}",
            "frames": n,
            "start_frame": int(start_frame),
            "cameras": ["A"],
            "main": main,
            "tracks": listed,
            "unsure": [[int(f), int(a) + 1, int(b) + 1] for f, a, b in unsure]}
    tmp = os.path.join(folder, "people.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    os.replace(tmp, os.path.join(folder, "people.json"))
    # corrections made against an earlier capture of this clip
    try:
        os.remove(os.path.join(folder, "people_edits.json"))
    except OSError:
        pass
    return len(listed)


# --- one person out again ---------------------------------------------------
def _people_module():
    """blendcap/people.py, the table and its corrections (no Blender in
    it). Loaded by path: this file runs outside the addon package."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "blendcap", "people.py")
    spec = importlib.util.spec_from_file_location("blendcap_people_table", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read(folder):
    """The capture's table with the user's corrections applied:
    (the people module, the automatic result, {piece id: who they are
    on each frame}, the capture's first video frame)."""
    people = _people_module()
    auto = people.load_auto(folder)
    if auto is None:
        raise RuntimeError("this is not a multi-person capture (no people.json)")
    state = people.load_edits(folder, auto["stamp"])
    labels, _top = people.replay(auto, state["edits"])
    with open(os.path.join(folder, "people.json"), encoding="utf-8") as fh:
        start = int(json.load(fh).get("start_frame", 0))
    return people, auto, labels, start


def _pieces(auto, labels, number, camera=None):
    """{frame: the piece the person is on that frame}, in every camera
    or in one."""
    src = {}
    for t in auto["tracks"]:
        if camera is not None and t.get("camera", 0) != camera:
            continue
        for f, v in enumerate(labels[t["id"]]):
            if v == number:
                src[f] = t
    return src


def person_sources(folder, number):
    """Which saved track the person is on each frame they are in:
    ({frame: track id}, frames in the capture, its first video frame),
    with the user's corrections applied. Frames count from the capture's
    first."""
    _people, auto, labels, start = _read(folder)
    src = {f: t["id"] for f, t in _pieces(auto, labels, number).items()}
    return src, auto["frames"], start


def _camera_start(people, folder, auto, camera, default):
    """The video frame ONE camera's capture starts on. Each camera of a
    multi-camera take is captured from its own clip, from wherever in
    that clip the take begins once the clips are synced (35 frames in
    for one, the first frame for another), and its saved tracks count
    frames the clip's way. The take as a whole starts on frame 0."""
    try:
        with open(os.path.join(people.camera_folder(folder, auto, camera),
                               "people.json"), encoding="utf-8") as fh:
            return int(json.load(fh).get("start_frame", default))
    except Exception:
        return default


def _assemble(people, folder, auto, src, start, number, whole=0, track_start=None):
    """One person's motion in ONE camera, from the pieces they are made
    of there (src: {frame: piece}). Returns (the arrays to save, the
    frames they cover counted from the capture's first, which of those
    frames are the capture's own sighting of them, how many saved tracks
    it took).

    The result starts on the capture's FIRST frame whoever it is: a
    person who walks in later holds their first pose until then, so
    every person's animation lines up with the clip, and with each
    other, frame for frame. whole: the capture's length when the result
    has to cover ALL of it (one camera of a multi-camera take: the
    cameras are merged frame against frame); the last pose is then held
    to the end the same way.

    track_start: the video frame the saved tracks count from, when that
    is not the capture's own `start` (a camera of a multi-camera take:
    _camera_start). Taking the take's start for a camera whose clip
    begins later read every frame of that camera too late by the
    difference, and the first frames from the END of its track."""
    base = int(start if track_start is None else track_start)
    frames = np.array(sorted(src), dtype=np.int64)
    loaded = {}
    rows = {name: [] for name in FIELDS + ("focal_length",)}
    own = []
    meta = None
    for f in frames:
        piece = src[int(f)]
        key = piece["id"]
        if key not in loaded:
            path = people.track_file(folder, auto, piece)
            if not os.path.isfile(path):
                raise RuntimeError(
                    f"this capture has no motion data for person {number} "
                    f"({os.path.relpath(path, folder)} is missing); run the capture again")
            with np.load(path) as z:
                loaded[key] = ({k: z[k] for k in z.files},
                               int(z["frame_indices"][0]) - base)
                if meta is None:
                    meta = {"fps": z["fps"], "frame_size": z["frame_size"]}
        arrays, first = loaded[key]
        i = int(f) - first
        if not 0 <= i < len(arrays["frame_indices"]):
            # (never read a frame from the other end of the track)
            raise RuntimeError(
                f"person {number}: frame {int(f)} is outside the saved motion "
                f"of the piece they are on there; run the capture again")
        for name in rows:
            rows[name].append(arrays[name][i])
        own.append(not bool(arrays["filled"][i]) if "filled" in arrays else True)
    last = max(int(frames[-1]), int(whole) - 1)
    dense = np.arange(0, last + 1)
    lead = int(frames[0])
    trail = last - int(frames[-1])
    save = {}
    for name in rows:
        body = fill(np.stack(rows[name]), frames, dense[lead:int(frames[-1]) + 1])
        if lead:
            body = np.concatenate([np.repeat(body[:1], lead, axis=0), body], axis=0)
        if trail:
            body = np.concatenate([body, np.repeat(body[-1:], trail, axis=0)], axis=0)
        save[name] = body
    save["frame_indices"] = dense + start
    save["fps"] = meta["fps"]
    save["frame_size"] = meta["frame_size"]
    save["blendcap_npz_version"] = np.array(NPZ_VERSION)
    seen = np.zeros(dense.size, dtype=bool)
    seen[frames[np.array(own, dtype=bool)]] = True
    return save, dense, seen, len(loaded)


# A typed height outside this many of the body model's is not applied
# (0.5 to 2.6 m typed against 1.73 is already inside it; this is for a
# model file that says something odd about itself).
TYPED_SIZE_LIMITS = (0.25, 1.6)


def typed_size(people, folder, auto, number, mhr_model, log=print):
    """A person's size against the body model's, from the height the
    user typed for them in the People list (blendcap/people.py,
    HEIGHT_MIN), or None when none was typed or it cannot be applied.
    One camera only: a multi-camera take measures its people."""
    height = people.height_of(people.load_edits(folder, auto["stamp"]), number)
    if height <= 0:
        return None
    if not mhr_model or not os.path.isfile(mhr_model):
        log(f"Person {number}: the height typed for them was not applied "
            f"(the body model's file was not given)")
        return None
    from .body_skeleton import model_height
    stands = model_height(mhr_model)
    if not stands:
        log(f"Person {number}: the height typed for them was not applied "
            f"(the body model's own height could not be read)")
        return None
    size = height / stands
    if not TYPED_SIZE_LIMITS[0] <= size <= TYPED_SIZE_LIMITS[1]:
        log(f"Person {number}: the height typed for them ({height:.2f} m) "
            f"was not applied")
        return None
    log(f"Person {number}: {height:.2f} m tall as typed, {size:.3f} of the "
        f"body model's {stands:.2f} m")
    return float(size)


def extract_person(folder, number, out_dir, log=print, mhr_model=None):
    """Write person `number` of the multi-person capture in `folder` as
    an ordinary capture in `out_dir` (mhr_frames.npz, and face_frames.npz
    when their face was captured). Returns (body path, face path or None).

    The result starts on the capture's FIRST frame whoever it is: a
    person who walks in later holds their first pose until then, so
    every person's animation lines up with the clip, and with each
    other, frame for frame.

    A multi-camera take: the person is taken out of EVERY camera's
    tracking and those are merged (extract_person_cameras), which is
    what makes out_dir a merged take like any other. mhr_model: the
    body model's file, which the merge needs."""
    people, auto, labels, start = _read(folder)
    if len(auto["cameras"]) > 1:
        return extract_person_cameras(folder, number, out_dir, mhr_model, log=log,
                                      read=(people, auto, labels, start))
    src = _pieces(auto, labels, number)
    if not src:
        raise RuntimeError(f"person {number} is not in this capture")
    save, dense, _seen, n_tracks = _assemble(people, folder, auto, src, start, number)
    # A height typed for this person: they are drawn that tall. The
    # capture took them for the body model's size and put them where
    # the MODEL would fill their place in the picture, so a person of
    # another size is nearer or further along the same line of sight by
    # the same proportion: everything the camera measured scales about
    # the camera, and its picture of them does not change. body_scale
    # tells the export to make the skeleton that size (modes.run_body),
    # as it does for a person of a multi-camera take.
    size = typed_size(people, folder, auto, number, mhr_model, log=log)
    if size is not None:
        for name in ("joint_coords", "cam_t"):
            save[name] = (save[name] * size).astype(save[name].dtype)
        save["body_scale"] = np.float64(size)
    lead = min(src)
    os.makedirs(out_dir, exist_ok=True)
    body_path = os.path.join(out_dir, "mhr_frames.npz")
    np.savez_compressed(body_path, **save)
    with open(os.path.join(out_dir, ".person_of"), "w", encoding="utf-8") as fh:
        json.dump({"capture": os.path.basename(os.path.normpath(folder)),
                   "person": int(number)}, fh)
    log(f"Person {number}: {len(src)} frames of their own from "
        f"{n_tracks} track(s), {len(dense)} in all"
        + (f" (first pose held for {lead})" if lead else ""))
    face = extract_face(people, folder, auto, src, start, dense, out_dir, log=log)
    return body_path, face


# A camera takes part in a person's merge when it saw them on at least
# this share of the frames they are in at all. Less than that and what
# it adds is mostly frames filled in, which the merge has to be told to
# ignore one by one anyway.
CAMERA_MIN_SHARE = 0.3
PERSON_MARK = ".person_of"
PERSON_SIG = ".person_sig"


def _person_sig(auto, per_camera, used, cams=None, number=0, scale=1.0):
    """What a person's merged take was made from: any change to who is
    who in any camera changes it, and so does a change in where the
    take's cameras were placed or how big the person was found to be
    (take_cameras: somebody ELSE's correction can move those), or in
    the size a typed height makes the take (take_scale; nothing typed
    leaves the signature what it was)."""
    import hashlib
    h = hashlib.sha1()
    h.update(str(auto["stamp"]).encode())
    h.update(repr(used).encode())
    if abs(float(scale) - 1.0) > 1e-9:
        h.update(repr(round(float(scale), 5)).encode())
    if cams is not None:
        h.update(np.round(np.asarray(cams["turns"]), 4).tobytes())
        h.update(np.round(np.asarray(cams["places"]), 4).tobytes())
        h.update(repr(round(float(cams["sizes"].get(int(number), 1.0)), 4)).encode())
    for cam in sorted(per_camera):
        src = per_camera[cam]
        runs, prev = [], None
        for f in sorted(src):
            pid = src[f]["id"]
            if prev is not None and prev[0] == pid and prev[2] == f - 1:
                prev[2] = f
            else:
                prev = [pid, f, f]
                runs.append(prev)
        h.update(repr((cam, runs)).encode())
    # (3: each camera's own first frame, see _camera_start; 4: the
    # take's cameras placed once and the person's own size, see
    # take_cameras; 5: the cameras that see a limb ranked by how well
    # each fits the others' pictures, two_camera_merge.fits_others. A
    # person merged before any of these is made again.)
    return "5:" + h.hexdigest()


# Where the take's cameras were placed (take_cameras), kept next to the
# take's table so the next person's export does not work it out again
# (the name is blendcap/people.py's, which reads the sizes out of it).
CAMERAS_FILE = "cameras_placed.json"


def take_cameras(people, folder, auto, labels, start, log=print):
    """Where the cameras of a multi-camera multi-person take stand,
    placed ONCE from everybody in it, and how big each person is
    (pipeline/people_place.py says why): {"turns", "places", "sizes"}
    in the take's camera A's frame, or None when it cannot be told
    (the merge then places each person's cameras on their own, as it
    always did).

    Worked out from who is who as the People list has it now (the
    user's corrections applied), and kept in the take's folder until
    that changes."""
    K = len(auto["cameras"])
    if K < 2:
        return None
    # (the panel works the same signature out, to know whether the
    # sizes it shows in the Height column are still the take's)
    sig = people.cameras_sig(auto, labels)
    path = os.path.join(folder, CAMERAS_FILE)
    try:
        with open(path, encoding="utf-8") as fh:
            kept = json.load(fh)
        if kept.get("sig") == sig:
            if kept.get("none"):
                return None
            return {"turns": np.asarray(kept["turns"], dtype=np.float64),
                    "places": np.asarray(kept["places"], dtype=np.float64),
                    "sizes": {int(p): float(s) for p, s in kept["sizes"].items()}}
    except (OSError, ValueError, KeyError):
        pass
    try:
        with open(os.path.join(folder, "people.json"), encoding="utf-8") as fh:
            first = json.load(fh)["placed"]
        first = [(np.asarray(c["turn"], dtype=np.float64),
                  np.asarray(c["place"], dtype=np.float64)) for c in first]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if len(first) != K:
        return None
    from .two_camera_merge import HIPS, NECK, SHO_R, SHO_L
    joints = [HIPS, NECK, SHO_R, SHO_L]         # the hips first (people_place)
    n = int(auto["frames"])
    numbers = sorted({int(v) for row in labels.values() for v in set(row) if v})
    obs = {}
    for c in range(K):
        own = _camera_start(people, folder, auto, c, start)
        for number in numbers:
            src = _pieces(auto, labels, number, camera=c)
            if not src:
                continue
            try:
                save, _dense, mine, _n = _assemble(people, folder, auto, src, start,
                                                   number, whole=n, track_start=own)
            except RuntimeError:
                continue
            obs[(number, c)] = (
                (save["joint_coords"][:, joints]
                 + save["cam_t"][:, None, :]).astype(np.float64), mine)
    from .people_place import place
    got = place(obs, first, log=log)
    keep = {"sig": sig, "none": got is None}
    if got is not None:
        keep.update(turns=np.asarray(got["turns"]).tolist(),
                    places=np.asarray(got["places"]).tolist(),
                    sizes={str(p): float(s) for p, s in got["sizes"].items()},
                    across=float(got["across"]))
    # (whole or not at all: the panel reads this file while it is made)
    try:
        with open(path + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(keep, fh)
        os.replace(path + ".tmp", path)
    except OSError:
        pass
    return got


def place_take(folder, log=print):
    """Place the cameras of the multi-camera take in `folder` and size
    its people, for who is who as the People list has it now, and keep
    the answer (take_cameras). Run at the end of the capture and by the
    panel after a correction, so the Height column has its estimates
    before the first BVH is asked for. Returns the sizes, or None."""
    people, auto, labels, start = _read(folder)
    got = take_cameras(people, folder, auto, labels, start, log=log)
    return None if got is None else got["sizes"]


def take_scale(people, folder, auto, sizes, mhr_model, log=print):
    """How much bigger than the cameras' own estimate a multi-camera
    take is made, from the height typed for one of its people in the
    People list (blendcap/people.py, take_factor): the whole take is
    scaled by it, the people and the room between them alike, so
    everybody keeps their measured size against everybody else. 1.0
    when nothing was typed or it cannot be applied."""
    state = people.load_edits(folder, auto["stamp"])
    if not state.get("heights"):
        return 1.0
    if not mhr_model or not os.path.isfile(mhr_model):
        log("The height typed in the People list was not applied "
            "(the body model's file was not given)")
        return 1.0
    from .body_skeleton import model_height
    stands = model_height(mhr_model)
    if not stands:
        log("The height typed in the People list was not applied "
            "(the body model's own height could not be read)")
        return 1.0
    factor = float(people.take_factor(state, sizes or {}, model=stands))
    if not TYPED_SIZE_LIMITS[0] <= factor <= TYPED_SIZE_LIMITS[1]:
        log("The height typed in the People list was not applied")
        return 1.0
    typed = ", ".join(f"person {k} {people.height_of(state, int(k)):.2f} m"
                      for k in sorted(state["heights"]))
    log(f"Height typed ({typed}): the take is made {factor:.3f} times the "
        f"size the cameras estimated")
    return factor


def _take_level(folder, camera):
    """Which way is up for the whole take, as the merge wants it: in the
    frame of `camera` (the first camera of the person being merged), or
    None when the matching could not tell (people_cameras.take_level)."""
    try:
        with open(os.path.join(folder, "people.json"), encoding="utf-8") as fh:
            raw = json.load(fh)
        level = np.asarray(raw["level"], dtype=np.float64)
        if level.shape != (3, 3):
            return None
        if camera == 0:
            return level
        turn = np.asarray(raw["placed"][camera]["turn"], dtype=np.float64)
        return level @ turn
    except Exception:
        return None


def extract_person_cameras(folder, number, out_dir, mhr_model, log=print, read=None):
    """Person `number` of a multi-camera multi-person take, as a MERGED
    take in out_dir: the person as each camera saw them goes into
    out_dir/cam_a, cam_b, ... (ordinary captures, as long as the take,
    with `absent` marking the frames that camera did not see them on)
    and pipeline/two_camera_merge merges those into out_dir, exactly as
    it merges the cameras of a one-person take. Returns (body path, face
    path or None).

    The cameras that take part are the ones that saw enough of the
    person (CAMERA_MIN_SHARE), lettered from a in the take's own order;
    .person_of records which of the take's cameras each one is. One
    camera only: its capture is the person's, with nothing to merge.

    Nothing is done again when out_dir already holds this person made
    from the same pieces (a second Generate, or Bake)."""
    people, auto, labels, start = read or _read(folder)
    n = int(auto["frames"])
    per_camera = {c: _pieces(auto, labels, number, camera=c)
                  for c in range(len(auto["cameras"]))}
    every = set()
    for src in per_camera.values():
        every |= set(src)
    if not every:
        raise RuntimeError(f"person {number} is not in this capture")

    def own(c):
        """Frames camera c saw them on itself (not filled in)."""
        got = 0
        files = {}
        for f, piece in per_camera[c].items():
            if piece["id"] not in files:
                files[piece["id"]] = set()
                for a, b in piece.get("filled", []):
                    files[piece["id"]].update(range(a, b + 1))
            got += f not in files[piece["id"]]
        return got
    seen = {c: own(c) for c in per_camera}
    used = [c for c in sorted(per_camera) if seen[c] >= CAMERA_MIN_SHARE * len(every)]
    if not used:
        used = [max(seen, key=lambda c: seen[c])]
    letters = [str(auto["cameras"][c]) for c in used]
    # the take's cameras, placed once from everybody (None: not known)
    typed = bool(people.load_edits(folder, auto["stamp"]).get("heights"))
    took = (take_cameras(people, folder, auto, labels, start, log=log)
            if len(used) > 1 or typed else None)
    cams = took if len(used) > 1 else None
    # A height typed for somebody in this take: the whole take is that
    # much bigger or smaller than the cameras' estimate (take_scale).
    # Scaling a take about its camera A scales where every camera
    # stands and every person's size alike, and what each camera saw of
    # a person, in that camera's own frame, by the same.
    factor = 1.0
    if typed and took is None:
        log("The height typed in the People list was not applied "
            "(the take's cameras could not be placed)")
    elif typed:
        factor = take_scale(people, folder, auto, took["sizes"], mhr_model, log=log)
    sig = _person_sig(auto, per_camera, used, cams, number, scale=factor)
    body_path = os.path.join(out_dir, "mhr_frames.npz")
    face_out = os.path.join(out_dir, "face_frames.npz")
    try:
        with open(os.path.join(out_dir, PERSON_SIG), encoding="utf-8") as fh:
            if fh.read().strip() == sig and os.path.isfile(body_path):
                log(f"Person {number}: already merged from these pieces "
                    f"(cameras {', '.join(letters)}), not merged again")
                return body_path, face_out if os.path.isfile(face_out) else None
    except OSError:
        pass

    os.makedirs(out_dir, exist_ok=True)
    # what an earlier export of this person left: another set of cameras,
    # or a merge whose pieces have changed
    for name in os.listdir(out_dir):
        p = os.path.join(out_dir, name)
        if name.startswith("cam_") and os.path.isdir(p):
            for sub in os.listdir(p):
                try:
                    os.remove(os.path.join(p, sub))
                except OSError:
                    pass
            try:
                os.rmdir(p)
            except OSError:
                pass
    for name in ("calibration.npz", "face_frames.npz", PERSON_SIG):
        try:
            os.remove(os.path.join(out_dir, name))
        except OSError:
            pass

    log(f"Person {number}: seen by camera(s) "
        + ", ".join(f"{auto['cameras'][c]} ({seen[c]} of {len(every)} frames)"
                    for c in sorted(per_camera) if seen[c])
        + (("; left out of the merge: " + ", ".join(
            str(auto["cameras"][c]) for c in sorted(per_camera)
            if seen[c] and c not in used)) if any(
                seen[c] and c not in used for c in per_camera) else ""))
    mark = {"capture": os.path.basename(os.path.normpath(folder)),
            "person": int(number), "cameras": letters}
    # where each camera's own clip starts (not the take's start)
    first = {c: _camera_start(people, folder, auto, c, start) for c in used}
    if len(used) == 1:
        c = used[0]
        save, dense, _seen, _n = _assemble(people, folder, auto, per_camera[c],
                                           start, number, track_start=first[c])
        if factor != 1.0:
            # nobody measured this person (one camera): the take's
            # scale is all there is to apply
            for name in ("joint_coords", "cam_t"):
                save[name] = (save[name] * factor).astype(save[name].dtype)
            save["body_scale"] = np.float64(factor)
        np.savez_compressed(body_path, **save)
        face = extract_face(people, folder, auto, per_camera[c], start, dense,
                            out_dir, log=log, track_start=first[c])
        log(f"Person {number}: only camera {letters[0]} saw them, nothing to merge")
    else:
        dirs = []
        for k, c in enumerate(used):
            sub = os.path.join(out_dir, "cam_" + "abcd"[k])
            os.makedirs(sub, exist_ok=True)
            save, dense, mine, _n = _assemble(people, folder, auto, per_camera[c],
                                              start, number, whole=n,
                                              track_start=first[c])
            save["absent"] = ~mine
            np.savez_compressed(os.path.join(sub, "mhr_frames.npz"), **save)
            extract_face(people, folder, auto, per_camera[c], start, dense, sub,
                         log=lambda *_a: None, track_start=first[c])
            dirs.append(sub)
        if not mhr_model or not os.path.isfile(mhr_model):
            raise RuntimeError("the body model file is needed to merge the cameras "
                               "(--mhr-model)")
        from .two_camera_merge import merge_cameras
        log(f"Person {number}: merging cameras {', '.join(letters)}")
        placed = None
        if cams is not None:
            # in the frame of this person's first camera, as the merge
            # works (the take's camera A unless it hardly saw them)
            Rf, Tf = cams["turns"][used[0]], cams["places"][used[0]]
            placed = {"turns": [Rf.T @ cams["turns"][c] for c in used],
                      "places": [factor * (Rf.T @ (cams["places"][c] - Tf))
                                 for c in used],
                      "scale": factor * float(cams["sizes"].get(int(number), 1.0))}
        merge_cameras(dirs, out_dir, mhr_model,
                      level=_take_level(folder, used[0]), cameras=placed)
        face = face_out if os.path.isfile(face_out) else None
    with open(os.path.join(out_dir, PERSON_MARK), "w", encoding="utf-8") as fh:
        json.dump(mark, fh)
    with open(os.path.join(out_dir, PERSON_SIG), "w", encoding="utf-8") as fh:
        fh.write(sig)
    return body_path, face


def extract_face(people, folder, auto, src, start, dense, out_dir, log=print,
                 track_start=None):
    """The person's face capture in one camera, put together from the
    pieces they are made of there (src: {frame: piece}). Frames with no
    face data are marked as not detected. Returns the file's path, or
    None (and removes a stale one) when none of their pieces has a face
    capture."""
    out = os.path.join(out_dir, "face_frames.npz")
    pieces = {}
    for piece in src.values():
        pieces[piece["id"]] = piece
    faces = {}
    for pid in sorted(pieces):
        p = people.track_file(folder, auto, pieces[pid], face=True)
        if os.path.isfile(p):
            with np.load(p, allow_pickle=False) as z:
                faces[pid] = {k: z[k] for k in z.files}
    if not faces:
        try:
            os.remove(out)
        except OSError:
            pass
        return None
    ref = next(iter(faces.values()))
    n_ref = len(ref["frame_indices"])
    per_frame = [k for k, v in ref.items()
                 if getattr(v, "ndim", 0) >= 1 and v.shape[0] == n_ref
                 and k != "blendshape_names"]
    save = {k: v for k, v in ref.items() if k not in per_frame}
    # (track_start: as in _assemble, the frame the saved tracks count from)
    base = int(start if track_start is None else track_start)
    lookup = {pid: {int(v) - base: i for i, v in enumerate(fz["frame_indices"])}
              for pid, fz in faces.items()}
    got = 0
    for k in per_frame:
        save[k] = np.zeros((dense.size,) + ref[k].shape[1:], dtype=ref[k].dtype)
    # a frame without face data looks the way the face capture itself
    # writes a miss (save_face_data._make_miss_data)
    if "face_transform" in save:
        save["face_transform"][:] = np.eye(4, dtype=save["face_transform"].dtype)
    if "eye_gaze" in save:
        save["eye_gaze"][:] = 0.5
    for row, f in enumerate(dense):
        piece = src.get(int(f))
        pid = piece["id"] if piece is not None else None
        i = lookup.get(pid, {}).get(int(f)) if pid is not None else None
        if i is None:
            continue
        for k in per_frame:
            if k in faces[pid] and faces[pid][k].shape[1:] == save[k].shape[1:]:
                save[k][row] = faces[pid][k][i]
        got += 1
    save["frame_indices"] = (dense + start).astype(ref["frame_indices"].dtype)
    np.savez_compressed(out, **save)
    log(f"       face: {got} of {dense.size} frames")
    return out
