"""Who is who ACROSS the cameras of a Multi-Camera & Multi-Person take.

Run by the addon after every camera's own multi-person capture
(blendcap/multicam.py), and usable on its own:

    python -m pipeline.people_cameras CAM_A_DIR CAM_B_DIR [CAM_C_DIR
                                      [CAM_D_DIR]] -o OUT_DIR

Each CAM_x_DIR is an ordinary Multi-Person capture of one camera
(pipeline/people_tracks.py: people.json and people/track_NN.npz), all
over the same frames. Every camera numbered the people it saw on its
own, in its own order, and may have lost, doubled or exchanged them on
its own. OUT_DIR receives ONE people.json for the take (the layout is at
the top of blendcap/people.py): pieces of the cameras' tracks, each
saying which camera saw it and who it is. Nothing is tracked again and
no motion is copied: the cameras' own files stay where they are, and a
person's merged motion is only built when a BVH is asked for, from the
pieces as the panel has them by then (people_tracks.extract_person).

HOW PEOPLE ARE MATCHED

1. How each camera is turned against camera A, from the BODIES: two
   cameras that see the same person see the same pose, turned
   (two_camera_merge.camera_turn). Every track of camera A is tried
   against every track of the other camera and the pair that fits best
   gives the turn. A wrong pair only fits as well as the right one when
   two people do the same thing, and then it gives the same turn
   anyway, which is why this needs no knowledge of who is who.
2. Where each camera stands against camera A, from where that pair's
   hips are in each.
3. Frame by frame, every body of every camera is put in camera A's
   frame, and bodies of different cameras are grouped when they are in
   the same pose in the same place (never two of one camera in a
   group). Pose counts for more than place: depth from one camera is
   the loose measurement, and it is loose by a different amount for
   each person.
4. A group is the person most of its tracks were a moment ago. So one
   camera exchanging two people half way through is outvoted by the
   others and its track is CUT there, each part going to the right
   person; where the cameras split evenly, who is nearer to where the
   person was decides.
5. A change of person that lasts less than MIN_RUN_S is taken for the
   grouping flickering while people are in contact, not for an
   exchange, and smoothed away.

A body only one camera ever saw becomes a person of its own; the panel's
Join or Remove deals with it. The same goes for a camera whose turn
could not be found at all (it never saw anybody the first camera saw).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.people_tracks import track_path  # noqa: E402
from pipeline.two_camera_merge import (  # noqa: E402
    camera_turn, floor_plane, agreeing_floors, _level_to,
    HIPS, HEAD, TARGETS, LETTERS, ANK_R, ANK_L, FLOOR_MIN_SPREAD)

J = TARGETS + [HEAD]

# Two tracks need this long together (seconds, both really seen) before
# the turn between their cameras is read off them: long enough that a
# glimpse of somebody does not place a camera. A take shorter than twice
# that has no such stretch to ask for, so there it is half the take, and
# never fewer than MIN_COMMON_FRAMES (_need). Before 2026-11-24 it was a
# second whatever the take's length, and a capture shorter than that
# could not match anybody: every camera's people were listed on their
# own (the user's 15-frame four-camera capture of two people: 8).
MIN_COMMON_S = 1.0
MIN_COMMON_FRAMES = 6
# Grouping: how unlike two bodies of different cameras may be and still
# be one person. Pose apart (mean joint distance, metres) plus POS_WEIGHT
# times how far apart they stand (metres).
POS_WEIGHT = 0.3
MAX_COST = 0.45
# Two bodies that were the same person a moment ago stay together a
# little more readily than two that were not (so the grouping does not
# flicker between two near-equal answers).
STAY = 0.05
# A camera's own tracking can hold one body twice (two of its tracks on
# the same person, which a messy moment does). The second one is told
# from a second PERSON by the other cameras: it fits the body they saw
# this well (COPY_COST), the camera's two bodies are this alike
# (TWIN_COST, the same measure between two bodies of ONE camera), and at
# least as many cameras see one body there as see two. Two people side
# by side in the same pose are two bodies to every camera, so they
# never pass. The copy is left out on those frames, not listed as
# somebody.
COPY_COST = 0.30
TWIN_COST = 0.40
# Somebody every camera lost at once (hidden from all of them, or out of
# the picture) and who turns up again in new tracks is the same person
# when they turn up near where they were: within REJOIN_M, plus what
# REJOIN_M_PER_S allows for the time they were gone, for at most
# REJOIN_S. The single-camera capture's own rule (people_tracks), in the
# room and not in one camera's picture.
REJOIN_S = 3.0
REJOIN_M = 0.6
REJOIN_M_PER_S = 1.5
# A change of who a camera's track is has to last this long to count.
MIN_RUN_S = 0.5
# A person seen for less than this (all cameras together) is not one.
MIN_PERSON_S = 0.25


def load_camera(cam_dir):
    """One camera's multi-person capture: its table and, per track, the
    pose and the place on every frame the track covers."""
    try:
        with open(os.path.join(cam_dir, "people.json"), encoding="utf-8") as fh:
            table = json.load(fh)
    except (OSError, ValueError):
        # a camera that found nobody has no table: a camera with no
        # people in it, not a reason to stop
        return {"dir": cam_dir, "frames": 0, "start": 0, "tracks": {},
                "table": {"tracks": [], "unsure": []}}
    start = int(table.get("start_frame", 0))
    tracks = {}
    for t in table.get("tracks", []):
        tid = int(t["id"])
        path = track_path(cam_dir, tid)
        if not os.path.isfile(path):
            continue
        with np.load(path) as z:
            jc = z["joint_coords"].astype(np.float64)
            ct = z["cam_t"].astype(np.float64)
            first = int(z["frame_indices"][0]) - start
            filled = (np.asarray(z["filled"], dtype=bool) if "filled" in z.files
                      else np.zeros(len(jc), dtype=bool))
        tracks[tid] = {"first": first, "last": first + len(jc) - 1,
                       "pose": (jc - jc[:, HIPS:HIPS + 1, :])[:, J],
                       "hips": jc[:, HIPS] + ct,
                       "feet": jc[:, [ANK_R, ANK_L]] + ct[:, None, :],
                       "filled": filled, "table": t}
    return {"dir": cam_dir, "frames": int(table["frames"]), "start": start,
            "tracks": tracks, "table": table}


def _common(ta, tb, real=True):
    """The frames two tracks share (both really seen, when real):
    (index into ta, index into tb)."""
    a, b = max(ta["first"], tb["first"]), min(ta["last"], tb["last"])
    if b < a:
        return np.zeros(0, int), np.zeros(0, int)
    ia = np.arange(a, b + 1) - ta["first"]
    ib = np.arange(a, b + 1) - tb["first"]
    if real:
        ok = ~ta["filled"][ia] & ~tb["filled"][ib]
        ia, ib = ia[ok], ib[ok]
    return ia, ib


def _need(fps, n):
    """How many frames two tracks have to share for the turn between
    their cameras to be read off them, in a take of n frames."""
    return max(MIN_COMMON_FRAMES, min(int(round(MIN_COMMON_S * fps)), n // 2))


def _explains(cam_a, cam_k, R, t, n, step):
    """How well a turn and a place for camera k explain EVERYBODY: the
    mean, over sampled frames, of what it costs to pair camera A's
    bodies with camera k's (a body left without a partner costs
    MAX_COST)."""
    from scipy.optimize import linear_sum_assignment
    total, count = 0.0, 0
    for f in range(0, n, step):
        a = [(0, tid, tr["pose"][f - tr["first"]], tr["hips"][f - tr["first"]])
             for tid, tr in cam_a["tracks"].items() if tr["first"] <= f <= tr["last"]]
        b = [(1, tid, tr["pose"][f - tr["first"]] @ R.T, tr["hips"][f - tr["first"]] @ R.T + t)
             for tid, tr in cam_k["tracks"].items() if tr["first"] <= f <= tr["last"]]
        if not a or not b:
            continue
        cost = np.array([[min(MAX_COST, _cost(x, y)) for y in b] for x in a])
        rows, cols = linear_sum_assignment(cost)
        total += float(cost[rows, cols].sum()) + MAX_COST * abs(len(a) - len(b))
        count += max(len(a), len(b))
    return total / count if count else MAX_COST


def find_turn(cam_a, cam_k, fps):
    """How camera k is turned against camera A and where it stands:
    (R, t, how well the poses agree in metres, the two tracks it was
    read from), or None when no two tracks of the two cameras were in
    the picture together for long enough.

    Every track of one is tried against every track of the other. The
    TURN any well-fitting pair gives is right even when the pair is two
    different people doing the same thing; the PLACE it gives is then
    off by the distance between them. So each candidate is judged on
    how well it explains everybody in both cameras, not on its own
    pair."""
    n = min(cam_a["frames"], cam_k["frames"])
    need = _need(fps, n)
    found = []
    for ida, ta in cam_a["tracks"].items():
        for idb, tb in cam_k["tracks"].items():
            ia, ib = _common(ta, tb)
            if len(ia) < need:
                continue
            R, res = camera_turn(ta["pose"][ia], tb["pose"][ib])
            found.append((len(ia), res, R, ida, idb, ia, ib))
    if not found:
        return None
    # the pairs that were together for a good part of the longest, best
    # fitting first; a handful is plenty
    longest = max(f[0] for f in found)
    short = sorted((f for f in found if f[0] >= 0.5 * longest), key=lambda f: f[1])[:6]
    step = max(1, n // 60)
    best = None
    for _n, res, R, ida, idb, ia, ib in short:
        ta, tb = cam_a["tracks"][ida], cam_k["tracks"][idb]
        t = np.median(ta["hips"][ia] - tb["hips"][ib] @ R.T, axis=0)
        score = _explains(cam_a, cam_k, R, t, n, step)
        if best is None or score < best[0] - 1e-9:
            best = (score, R, t, float(res), (ida, idb))
    return best[1:]


def _bodies(cams, turns, f):
    """Every body on frame f of every placed camera, in camera A's
    frame: [(camera, track id, pose (j, 3), hips (3,))]."""
    out = []
    for k, cam in enumerate(cams):
        if turns[k] is None:
            continue
        R, t = turns[k]
        for tid, tr in cam["tracks"].items():
            if tr["first"] <= f <= tr["last"]:
                i = f - tr["first"]
                out.append((k, tid, tr["pose"][i] @ R.T, tr["hips"][i] @ R.T + t))
    return out


def _cost(a, b):
    pose = float(np.sqrt(((a[2] - b[2]) ** 2).sum(axis=1)).mean())
    place = float(np.sqrt(((a[3] - b[3]) ** 2).sum()))
    return pose + POS_WEIGHT * place


def _groups(bodies, together=None, kept=None):
    """The bodies of one frame, grouped into people: (lists of indices
    into `bodies`, never two of one camera in a list; {index of a body
    that is a second copy of one in a group: the index of the one it
    copies}). together(i, j): were these two the same person a moment
    ago. kept(i): was this body in a group a moment ago (of two copies,
    the one that was stays)."""
    cross, pairs = {}, []
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            if bodies[i][0] != bodies[j][0]:
                c = _cost(bodies[i], bodies[j])
                cross[(i, j)] = cross[(j, i)] = c
                if c < MAX_COST:
                    stay = STAY if together is not None and together(i, j) else 0.0
                    pairs.append((c - stay, c, i, j))
    group = list(range(len(bodies)))
    members = {i: [i] for i in range(len(bodies))}
    copies = {}

    def fit(x, everyone):
        v = [cross[(x, m)] for m in everyone if bodies[m][0] != bodies[x][0]]
        return float(np.mean(v)) if v else 1e9

    for _key, c, i, j in sorted(pairs):
        if i in copies or j in copies:
            continue
        gi, gj = group[i], group[j]
        if gi == gj:
            continue
        ca = {bodies[m][0]: m for m in members[gi]}
        cb = {bodies[m][0]: m for m in members[gj]}
        shared = set(ca) & set(cb)
        drop = []
        if shared:
            # Both groups have a body of the same camera. Two people, as
            # a rule. One person that camera has twice, when: these two
            # fit each other well across cameras, that camera's two
            # bodies are alike, and the cameras that see ONE body here
            # are not outnumbered by those that see two.
            one_body = (set(ca) | set(cb)) - shared
            if c >= COPY_COST or len(one_body) < len(shared):
                continue
            if any(_cost(bodies[ca[k]], bodies[cb[k]]) >= TWIN_COST for k in shared):
                continue
            everyone = members[gi] + members[gj]
            for k in shared:
                x, y = ca[k], cb[k]
                kx = bool(kept(x)) if kept is not None else False
                ky = bool(kept(y)) if kept is not None else False
                if kx != ky:
                    lose = y if kx else x
                else:
                    lose = x if fit(x, everyone) > fit(y, everyone) else y
                drop.append((lose, y if lose == x else x))
        for m in members[gj]:
            group[m] = gi
        members[gi] += members.pop(gj)
        for lose, keep in drop:
            members[gi].remove(lose)
            copies[lose] = keep
    return list(members.values()), copies


def _label_frames(cams, turns, n, fps):
    """Who every placed camera's track is on every frame:
    {(camera, track id): [person per frame, 0 = not there]}, people
    numbered as they turn up."""
    from scipy.optimize import linear_sum_assignment
    labels = {(k, tid): np.zeros(n, dtype=np.int64)
              for k, cam in enumerate(cams) if turns[k] is not None
              for tid in cam["tracks"]}
    current = {}            # (camera, track id) -> who it was a moment ago
    where = {}              # person -> where they were last, camera A's frame
    last = {}               # person -> the last frame they were there
    top = 0
    for f in range(n):
        bodies = _bodies(cams, turns, f)
        if not bodies:
            continue
        keys = [(b[0], b[1]) for b in bodies]

        def together(i, j):
            # (both really in the group a frame ago: a body that was
            # only the second copy of somebody does not push the one
            # that was kept out again)
            return (f > 0 and labels[keys[i]][f - 1] != 0
                    and labels[keys[i]][f - 1] == labels[keys[j]][f - 1])

        def kept(i):
            return f > 0 and labels[keys[i]][f - 1] != 0
        groups, copies = _groups(bodies, together, kept)
        centre = [np.mean([bodies[m][3] for m in g], axis=0) for g in groups]
        known = sorted({current[(bodies[m][0], bodies[m][1])] for g in groups for m in g
                        if (bodies[m][0], bodies[m][1]) in current})
        given = {}
        if known:
            score = np.zeros((len(groups), len(known)))
            for gi, g in enumerate(groups):
                for m in g:
                    p = current.get((bodies[m][0], bodies[m][1]))
                    if p is not None:
                        score[gi, known.index(p)] += 10.0
                for pi, p in enumerate(known):
                    if score[gi, pi] > 0 and p in where:
                        # the nearer of two groups with equal votes
                        score[gi, pi] -= min(5.0, float(np.linalg.norm(
                            centre[gi] - where[p])))
            rows, cols = linear_sum_assignment(-score)
            for gi, pi in zip(rows, cols):
                if score[gi, pi] > 0:
                    given[gi] = known[pi]
        # somebody new to every camera, near where a person was lost a
        # moment ago: that person again
        gone = [p for p in where if p not in given.values()
                and f - last[p] <= REJOIN_S * fps]
        for gi, g in enumerate(groups):
            if gi in given or not gone or any(keys[m] in current for m in g):
                continue
            near = [(float(np.linalg.norm(centre[gi] - where[p])), p) for p in gone]
            d, p = min(near)
            if d <= REJOIN_M + REJOIN_M_PER_S * (f - last[p]) / fps:
                given[gi] = p
                gone.remove(p)
        for gi, g in enumerate(groups):
            if gi not in given:
                if len(g) == 1 and keys[g[0]] in current:
                    # One camera's body that was somebody a moment ago,
                    # while that person is where the other cameras have
                    # them and this body does not fit them: the camera
                    # has lost them (it is looking at a limb, or at the
                    # person in front). Nobody on this frame; the other
                    # cameras have the person to themselves.
                    continue
                top += 1
                given[gi] = top
            p = given[gi]
            where[p] = centre[gi]
            last[p] = f
            for m in g:
                key = (bodies[m][0], bodies[m][1])
                current[key] = p
                labels[key][f] = p
        # a second copy is nobody on this frame, and stays known as the
        # person it copies (it may be the one that is kept a moment later)
        for one, twin in copies.items():
            while twin in copies:
                twin = copies[twin]
            p = current.get(keys[twin])
            if p is not None:
                current[keys[one]] = p
    return labels


def _runs(row):
    """[(first, last, value)] of the non-zero stretches of a row."""
    out, start = [], None
    for f, v in enumerate(row):
        if start is not None and v != row[start]:
            out.append((start, f - 1, int(row[start])))
            start = None
        if v and start is None:
            start = f
    if start is not None:
        out.append((start, len(row) - 1, int(row[start])))
    return out


def _smooth(labels, min_run):
    """Short changes of person inside a camera's track are the grouping
    flickering, not an exchange: each goes to the longer of its
    neighbours. Two tracks of one camera left as the same person on a
    frame by that go back to what they were there."""
    raw = {k: v.copy() for k, v in labels.items()}
    for row in labels.values():
        while True:
            runs = _runs(row)
            short = []
            for i, (a, b, _v) in enumerate(runs):
                if b - a + 1 >= min_run:
                    continue
                before = runs[i - 1] if i > 0 and runs[i - 1][1] == a - 1 else None
                after = (runs[i + 1] if i + 1 < len(runs) and runs[i + 1][0] == b + 1
                         else None)
                if before is not None or after is not None:
                    short.append((b - a, a, b, before, after))
            if not short:
                break
            _len, a, b, before, after = min(short, key=lambda c: c[0])
            pick = max((r for r in (before, after) if r is not None),
                       key=lambda r: r[1] - r[0])
            row[a:b + 1] = pick[2]
    # what is left of a track in snatches shorter than that, with nothing
    # either side, is one camera glimpsing somebody: no use to a merge
    for row in labels.values():
        for a, b, _v in _runs(row):
            if b - a + 1 < max(2, min_run // 2):
                before = a > 0 and row[a - 1] != 0
                after = b + 1 < len(row) and row[b + 1] != 0
                if not before and not after:
                    row[a:b + 1] = 0
    by_cam = {}
    for (k, tid), row in labels.items():
        by_cam.setdefault(k, []).append((tid, row))
    for k, rows in by_cam.items():
        if len(rows) < 2:
            continue
        stack = np.stack([r for _t, r in rows])
        for f in range(stack.shape[1]):
            col = stack[:, f]
            vals = col[col > 0]
            if len(vals) != len(set(vals.tolist())):
                for tid, row in rows:
                    row[f] = raw[(k, tid)][f]
    return labels


def _renumber(labels, cams, turns, n, min_frames):
    """People numbered in order of appearance, those there from the same
    frame left to right as camera A sees the room. Somebody seen for
    less than min_frames is nobody."""
    first, frames = {}, {}
    for (k, tid), row in labels.items():
        tr = cams[k]["tracks"][tid]
        R, t = turns[k]
        for a, b, p in _runs(row):
            frames.setdefault(p, set()).update(range(a, b + 1))
            x = float((tr["hips"][a - tr["first"]] @ R.T + t)[0])
            if p not in first or (a, x) < first[p]:
                first[p] = (a, x)
    keep = [p for p in first if len(frames[p]) >= min_frames]
    order = sorted(keep, key=lambda p: first[p])
    new = {p: i + 1 for i, p in enumerate(order)}
    for row in labels.values():
        for p in np.unique(row):
            if p:
                row[row == p] = -new.get(int(p), 0)     # negative: not renumbered twice
        row *= -1
    return len(order), {new[p]: len(frames[p]) for p in order}


def take_level(cams, turns, labels):
    """Which way is up for the whole take: the rotation that stands
    camera A's frame level, from the floor under EVERYBODY (every
    camera's foot contacts of every person it saw), or None when the
    cameras do not agree on a floor. One person's own contacts often
    cover a patch a foot wide; everybody's cover the room, and one
    answer for the take keeps its people square to each other."""
    planes, R = [], []
    for k, cam in enumerate(cams):
        pts = []
        for tid, tr in cam["tracks"].items():
            row = labels.get((k, tid))
            if row is None:
                continue
            seen = np.flatnonzero(row[tr["first"]:tr["last"] + 1] > 0)
            seen = seen[~tr["filled"][seen]]
            if len(seen):
                pts.append(tr["feet"][seen])
        planes.append(floor_plane(np.concatenate(pts)) if pts
                      else (np.array([0.0, 1.0, 0.0]), np.zeros(3)))
        R.append(turns[k][0] if turns[k] is not None else np.eye(3))
    floors, _apart = agreeing_floors(planes, R)
    if not floors:
        # No two cameras found the same floor. Camera A's own, when its
        # contacts cover some floor: with everybody's feet in it that is
        # the room, not one performer's patch, and ONE answer for the
        # take matters more here than the last degree of it (people
        # levelled apart lean against each other).
        if float(planes[0][1][1]) >= FLOOR_MIN_SPREAD:
            return _level_to(planes[0][0])[0], [0]
        return None, []
    if 0 in floors:
        return _level_to(planes[0][0])[0], floors
    mean = np.sum([R[k] @ planes[k][0] for k in floors], axis=0)
    return _level_to(mean / np.linalg.norm(mean))[0], floors


def match(cam_dirs, log=print):
    """The take's people.json as a dict (see blendcap/people.py), from
    the cameras' own multi-person captures, camera A first."""
    cams = [load_camera(d) for d in cam_dirs]
    K = len(cams)
    lengths = [c["frames"] for c in cams if c["frames"] > 0]
    if not lengths:
        return {"version": 1, "stamp": "", "frames": 0, "start_frame": 0,
                "cameras": list(LETTERS[:K]), "main": 1, "tracks": [], "unsure": []}
    n = min(lengths)
    fps = 24.0
    for c in cams:
        for tid in c["tracks"]:
            with np.load(track_path(c["dir"], tid)) as z:
                fps = float(z["fps"])
            break
    for c in cams:
        for tr in c["tracks"].values():
            tr["last"] = min(tr["last"], n - 1)

    turns = [(np.eye(3), np.zeros(3))] + [None] * (K - 1)
    for k in range(1, K):
        hit = find_turn(cams[0], cams[k], fps)
        if hit is None:
            log(f"Camera {LETTERS[k]}: nobody camera A saw was in its picture for "
                f"long enough; its people are listed on their own")
            continue
        R, t, res, pair = hit
        turns[k] = (R, t)
        yaw = float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))
        log(f"Camera {LETTERS[k]}: {abs(yaw):.0f} degrees round from camera A, "
            f"poses agree to {res * 100:.1f} cm (its person {pair[1]} is "
            f"camera A's {pair[0]})")

    min_run = max(2, int(round(MIN_RUN_S * fps)))
    labels = _label_frames(cams, turns, n, fps)
    # once more with every matched body setting the turn and the place,
    # not the one pair they were first read from
    for k in range(1, K):
        if turns[k] is None:
            continue
        P, Q, HA, HK = [], [], [], []
        for tid, tr in cams[k]["tracks"].items():
            row = labels[(k, tid)]
            for ida, ta in cams[0]["tracks"].items():
                rowa = labels[(0, ida)]
                same = np.flatnonzero((row > 0) & (row == rowa))
                same = same[~tr["filled"][same - tr["first"]]
                            & ~ta["filled"][same - ta["first"]]]
                if len(same):
                    P.append(ta["pose"][same - ta["first"]])
                    Q.append(tr["pose"][same - tr["first"]])
                    HA.append(ta["hips"][same - ta["first"]])
                    HK.append(tr["hips"][same - tr["first"]])
        if P and sum(len(p) for p in P) >= _need(fps, n):
            R, _res = camera_turn(np.concatenate(P), np.concatenate(Q))
            t = np.median(np.concatenate(HA) - np.concatenate(HK) @ R.T, axis=0)
            turns[k] = (R, t)
    labels = _smooth(_label_frames(cams, turns, n, fps), min_run)

    level, level_cams = take_level(cams, turns, labels)
    placed = [None if t is None else {"turn": np.round(t[0], 6).tolist(),
                                       "place": np.round(t[1], 4).tolist()}
              for t in turns]
    if level is not None:
        log("  Floor: from the foot contacts of camera"
            f"{'s' if len(level_cams) != 1 else ''} "
            + ", ".join(LETTERS[k] for k in level_cams))
    else:
        log("  Floor: the cameras do not agree on one; each person is levelled "
            "on their own")

    # a camera that could not be placed: its tracks as people of their own
    top = max([int(r.max()) for r in labels.values()] or [0])
    for k in range(1, K):
        if turns[k] is None:
            turns[k] = (np.eye(3), np.zeros(3))
            for tid, tr in cams[k]["tracks"].items():
                top += 1
                row = np.zeros(n, dtype=np.int64)
                row[tr["first"]:tr["last"] + 1] = top
                labels[(k, tid)] = row
    count, seen = _renumber(labels, cams, turns, n,
                            max(2, int(round(MIN_PERSON_S * fps))))

    # the pieces
    pieces, unsure, piece_at = [], [], {}
    for (k, tid) in sorted(labels):
        tr = cams[k]["tracks"][tid]
        t = tr["table"]
        boxes = (t.get("boxes") or [[]])[0]
        filled = [(int(a), int(b)) for a, b in t.get("filled", [])]
        prev = None
        for a, b, p in _runs(labels[(k, tid)]):
            pid = len(pieces) + 1
            i0 = a - tr["first"]
            box = boxes[i0:i0 + (b - a + 1)]
            pieces.append({
                "id": pid, "camera": k, "source": tid, "person": p,
                "present": [[a, b]],
                "filled": [[max(a, fa), min(b, fb)] for fa, fb in filled
                           if fb >= a and fa <= b],
                "boxes": [box if c == k else [] for c in range(K)]})
            for f in (a, b):
                piece_at[(k, tid, f)] = pid
            piece_at[(k, tid, "runs")] = piece_at.get((k, tid, "runs"), []) + [(a, b, pid)]
            if prev is not None and prev[1] == a - 1:
                # this camera's own track changes person here: worth a look
                unsure.append([a, prev[2], pid])
            prev = (a, b, pid)

    def piece_on(k, tid, f):
        for a, b, pid in piece_at.get((k, tid, "runs"), []):
            if a <= f <= b:
                return pid
        return None
    # where a camera was unsure by itself
    for k, cam in enumerate(cams):
        for u in cam["table"].get("unsure", []):
            try:
                f, ta, tb = int(u[0]), int(u[1]), int(u[2])
            except Exception:
                continue
            pa, pb = piece_on(k, ta, f), piece_on(k, tb, f)
            if pa and pb:
                unsure.append([f, pa, pb])

    for p in sorted(seen):
        cams_of = sorted({pc["camera"] for pc in pieces if pc["person"] == p})
        log(f"  Person {p}: {seen[p]} frames, camera"
            f"{'s' if len(cams_of) != 1 else ''} "
            + ", ".join(LETTERS[c] for c in cams_of))
    cut = sum(1 for (k, tid) in labels if len(_runs(labels[(k, tid)])) > 1)
    if cut:
        log(f"  {cut} of the cameras' own tracks changed person part way and were cut there")
    main = max(seen, key=lambda p: seen[p]) if seen else 1
    return {"version": 1,
            "stamp": f"{int(time.time())}-{count}-{n}-{K}cam",
            "frames": int(n),
            "start_frame": 0,
            "cameras": list(LETTERS[:K]),
            "main": int(main),
            # how the cameras stand against camera A (None: could not
            # be told), and which way is up in camera A's frame
            "placed": placed,
            "level": None if level is None else np.round(level, 6).tolist(),
            "tracks": pieces,
            "unsure": sorted(unsure)}


def write(cam_dirs, out_dir, log=print):
    """Match the cameras' people and save the take's table in out_dir.
    Returns the number of people."""
    data = match(cam_dirs, log=log)
    if not data["tracks"]:
        raise RuntimeError("nobody was found in any camera")
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "people.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    os.replace(tmp, os.path.join(out_dir, "people.json"))
    # corrections made against an earlier capture of this take, and what
    # an ordinary multi-camera capture into this folder left: a merged
    # take of ONE performer that these cameras' captures no longer are
    for name in ("people_edits.json", "mhr_frames.npz", "face_frames.npz",
                 "calibration.npz"):
        try:
            os.remove(os.path.join(out_dir, name))
        except OSError:
            pass
    return len({t["person"] for t in data["tracks"]})


def place(out_dir, log=print):
    """Place the take's cameras from everybody in it and size its
    people (people_tracks.place_take), and say what came of it."""
    from pipeline.people_tracks import place_take
    sizes = place_take(out_dir, log=lambda *_a: None)
    if sizes is None:
        log("Sizes: the take's cameras could not be placed from the people in it")
    else:
        log("Sizes: " + ", ".join(f"person {p} {s:.3f}" for p, s in sorted(sizes.items()))
            + " of the body model's")
    return sizes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cams", nargs="*", metavar="CAM_DIR",
                    help="the cameras' multi-person capture folders, camera A "
                         "first (2 to 4)")
    ap.add_argument("-o", "--out", help="the take's folder")
    ap.add_argument("--sizes", metavar="TAKE_DIR",
                    help="nothing is matched: place the cameras of this take and "
                         "size its people, for who is who as its People list has "
                         "it now (what the panel runs after a correction)")
    a = ap.parse_args()
    if a.sizes:
        try:
            place(a.sizes)
        except Exception as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        return
    if not a.out:
        ap.error("give the take's folder (-o)")
    if not 2 <= len(a.cams) <= len(LETTERS):
        ap.error(f"give 2 to {len(LETTERS)} capture folders")
    print(f"Matching the people of {len(a.cams)} cameras...")
    try:
        count = write(a.cams, a.out)
    except (RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    print(f"Matched: {count} people")
    # Where the cameras stand and how big each person is, now, so the
    # People list has its Height estimates before the first BVH is
    # asked for (Generate BVH works it out itself when this is missing:
    # a failure here costs the capture nothing).
    try:
        place(a.out)
    except Exception as exc:
        print(f"The people's sizes were not worked out yet ({exc}); "
              f"Generate BVH will do it.")


if __name__ == "__main__":
    main()
