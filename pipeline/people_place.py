"""Multi-Camera & Multi-Person: where the cameras stand, worked out ONCE
for the take from everybody in it, and how big each person really is.

Why this exists (2026-11-24, the user's first own four-camera
two-person take, a fight scene: "the characters are making contact, in
the video ... but the bvh is like .5 meters apart at all the contact
points", and "one performer is taller than the other, and the bvh's are
the same").

Each person of such a take is merged on their own (two_camera_merge),
and that merge places a person's cameras from that person alone: every
camera is moved until it agrees with camera A about where the person's
hips are during the T-pose. Camera A's idea of WHERE ACROSS ITS PICTURE
somebody is, is good. Its idea of HOW FAR AWAY they are is the weakest
thing a single camera measures: the capture draws everybody at the body
model's own size, so somebody smaller than that is taken to be further
away, by the same proportion, by every camera. One performer alone can
live with that (the whole take sits a little too far from camera A and
nobody can tell). Two performers each get their own error, and the gap
between them is wrong by the difference: 19 cm typically and up to 26 on
that take, with the shorter performer 12 to 24 cm too far from every
camera.

The cameras do not move between two people. So they are placed here
once, from everybody together, believing each camera only for where it
sees a body across its picture (its line of sight to it) and hardly at
all for the distance along that line. Where the cameras' lines of sight
to one body meet is where the body is. On that take the four cameras
then agree on each body to about a centimetre across their pictures.

And once the cameras are placed, how far each really is from a body is
known, against how far it took the body to be. The ratio is the
person's real size against the size they were drawn at, and every
camera gives the same answer (1.00 for one performer, 0.92 to 0.95 for
the other, on that take).

The cameras are not in step to the frame at this point (the fine sync
is the merge's, per person). A frame or two out moves a body a few
centimetres one way and then the other, and the placement comes out the
same to about 2 cm and 0.3 degrees with or without it (measured).
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

# How much less a camera is believed ALONG its line of sight than across
# it. Not 1.0: with every camera's distance thrown away entirely the
# size of the whole room is left open (two cameras' sightlines fix
# shapes, not scale), and what little say the distances keep is what
# settles it, averaged over every camera and person.
ALONG = 0.95
ROUNDS = 12                 # place the bodies, place the cameras, again
SETTLED = 0.002             # metres a camera may still move and be done
EVERY = 2                   # frames: every other one is plenty to fit a camera to
# A camera is placed from the frames it shares with at least one other
# camera on somebody (counted per person: two people on six frames are
# twelve). Fewer than this and it keeps the place the matching gave it.
#
# 48 until 2026-11-24, which left a take under about two seconds with
# its cameras where the opening frames (or, in a multi-person take, the
# matching) put them. Measured that day on full takes cut to windows of
# 3 to 120 frames, each against what the whole take says. Two people
# (the user's take, one a head shorter): the distance between them came
# out about 19 cm off the old way at every length, and 4 cm off placed
# from a window of 6 to 18 frames (3 cm at 24 to 48, 1 cm at 120); on a
# take of two people the same size both ways were within 1.3 cm. One
# performer, five takes, by the merged limbs seen through each camera
# against that camera's own tracking: from 12 frames up placing from
# the window was the same or better in most windows (0.5 to 1.5 pixels
# of a 1920 picture), under 12 it made no difference either way. The
# worst windows (up to 40 cm between two people) turn up at every
# length up to three seconds: they are not what a short take adds. The
# user's decision on those numbers: 12.
MIN_FRAMES = 12
# The cameras have to agree this well on the bodies, across their
# pictures (metres, the middle one), for the placement to be used at
# all. Past that something else is wrong (a wrong focal length, two
# people taken for one) and the merge falls back to placing each
# person's cameras on their own, as it always did.
MAX_ACROSS = 0.06
# A person's size is read where at least two cameras' lines of sight
# cross at a real angle (this much of a direction is left over when the
# sightlines are laid on each other: two cameras facing each other see
# along the same line and settle nothing about distance).
CROSSING = 0.25
# (counted over the cameras: three frames in four cameras are twelve.
# 48 until 2026-11-24; the two people's sizes against each other came
# out about 2% off from a window of 6 frames and from one of 72 alike)
SIZE_MIN_FRAMES = 12
SIZE_LIMITS = (0.75, 1.25)  # anything past this is not a person's size


def _sight(R, w):
    s = w / np.maximum(np.linalg.norm(w, axis=-1, keepdims=True), 1e-9)
    return np.einsum('ij,...j->...i', R, s)


def place(obs, start, log=print, why=None, keep_turns=False):
    """Place the cameras of one take and size the people in it.

    why: a list that is given the reason when None comes back
    (("nothing", None), or ("apart", metres)), for a caller that says
    it in its own words (two_camera_merge.place_from_take).
    keep_turns: the cameras keep the turns they start with and only
    where they stand is worked out.

    obs: {(person, camera): (points, seen)}: the same few body joints
        of that person as that camera has them, (n, j, 3) metres in the
        camera's OWN frame with the hips first, and (n,) bool, the
        frames the camera really saw them on.
    start: [(turn, place)] for every camera, camera 0 first: the
        rotation that takes that camera's frame to camera 0's and where
        it stands there. Camera 0's is the identity and stays it. (The
        matching step's placement: people_cameras.)

    Returns None when there is nothing to place by or the cameras do
    not come to agree, else {"turns": (K, 3, 3), "places": (K, 3),
    "sizes": {person: size}, "across": metres, "fitted": [bool per
    camera]}. A size of 1.0 is the body model's own."""
    K = len(start)
    if K < 2 or not obs:
        return None
    R = [np.asarray(r, dtype=np.float64) for r, _t in start]
    T = [np.asarray(t, dtype=np.float64) for _r, t in start]
    people = sorted({p for p, _k in obs})
    n = len(next(iter(obs.values()))[1])
    nj = next(iter(obs.values()))[0].shape[1]
    size = {p: 1.0 for p in people}
    eye = np.eye(3)

    def bodies(believe=ALONG):
        """Where every camera's lines of sight to each person meet:
        {person: (points (n, j, 3) in camera 0's frame, cameras seeing
        them (n,), how squarely their sightlines cross (n,))}."""
        out = {}
        for p in people:
            tot = np.zeros((n, nj, 3, 3))
            rhs = np.zeros((n, nj, 3))
            rays = np.zeros((n, 3, 3))
            cnt = np.zeros(n)
            for k in range(K):
                if (p, k) not in obs:
                    continue
                w, ok = obs[(p, k)]
                w = w * size[p]
                s = _sight(R[k], w)
                Q = (eye - believe * s[..., :, None] * s[..., None, :]) \
                    * ok[:, None, None, None]
                tot += Q
                rhs += np.einsum('fkij,fkj->fki', Q,
                                 np.einsum('ij,fkj->fki', R[k], w) + T[k])
                rays += (eye - s[:, 0, :, None] * s[:, 0, None, :]) \
                    * ok[:, None, None]
                cnt += ok
            X = np.linalg.solve(tot + 1e-6 * eye, rhs[..., None])[..., 0]
            out[p] = (X, cnt, np.linalg.eigvalsh(rays)[:, 0])
        return out

    def fit(k, X):
        pts, tgt = [], []
        for p in people:
            if (p, k) not in obs:
                continue
            w, ok = obs[(p, k)]
            use = ok & (X[p][1] >= 2)
            pts.append((w[use][::EVERY] * size[p]).reshape(-1, 3))
            tgt.append(X[p][0][use][::EVERY].reshape(-1, 3))
        if not pts or sum(len(x) for x in pts) < MIN_FRAMES * nj // EVERY:
            return None
        pts, tgt = np.concatenate(pts), np.concatenate(tgt)
        d = pts / np.maximum(np.linalg.norm(pts, axis=1, keepdims=True), 1e-9)

        def res(v):
            # the body as this camera would have to see it, against how
            # it saw it: in full across its line of sight, a twentieth
            # along it
            e = (tgt - v[3:]) @ Rot.from_rotvec(v[:3]).as_matrix() - pts
            along = (e * d).sum(1, keepdims=True) * d
            return ((e - along) + (1.0 - ALONG) * along).ravel()
        if keep_turns:
            rv = Rot.from_matrix(R[k]).as_rotvec()
            s = least_squares(lambda t: res(np.concatenate([rv, t])), T[k],
                              loss="soft_l1", f_scale=0.05)
            return R[k], s.x
        s = least_squares(res, np.concatenate(
            [Rot.from_matrix(R[k]).as_rotvec(), T[k]]),
            loss="soft_l1", f_scale=0.05)
        return Rot.from_rotvec(s.x[:3]).as_matrix(), s.x[3:]

    fitted = [True] + [False] * (K - 1)
    X = None
    for _again in range(2):
        for _round in range(ROUNDS):
            X = bodies()
            moved = 0.0
            for k in range(1, K):
                got = fit(k, X)
                if got is None:
                    continue
                moved = max(moved, float(np.linalg.norm(got[1] - T[k])))
                R[k], T[k] = got
                fitted[k] = True
            if moved < SETTLED:
                break
        X = bodies()
        # how big each person is: the real distance from each camera to
        # their hips over the distance that camera took it to be
        changed = False
        seen_for = {}
        for p in people:
            ratios = []
            for k in range(K):
                if (p, k) not in obs:
                    continue
                w, ok = obs[(p, k)]
                use = ok & (X[p][1] >= 2) & (X[p][2] > CROSSING)
                if not use.any():
                    continue
                true = np.linalg.norm((X[p][0][use, 0] - T[k]) @ R[k], axis=1)
                ratios.append(true / np.maximum(
                    np.linalg.norm(w[use, 0], axis=1), 1e-9))
            if ratios and sum(len(r) for r in ratios) >= SIZE_MIN_FRAMES:
                new = float(np.clip(np.median(np.concatenate(ratios)),
                                    *SIZE_LIMITS))
                changed = changed or abs(new - size[p]) > 0.002
                size[p] = new
                seen_for[p] = sum(len(r) for r in ratios)
        # Sightlines fix the SHAPE of the room, not its size: make
        # everything in it (the cameras' places, the people) one tenth
        # bigger and every camera sees exactly what it saw. What sets
        # the size is the people: on average they are taken to be the
        # body model's size (weighted by how much each was seen), and
        # the cameras' places follow. Who is bigger than whom, and by
        # how much, does not depend on this.
        if seen_for:
            tot = float(sum(seen_for.values()))
            mean = float(np.exp(sum(n_p * np.log(size[p])
                                    for p, n_p in seen_for.items()) / tot))
            for p in seen_for:
                size[p] = float(np.clip(size[p] / mean, *SIZE_LIMITS))
            for k in range(1, K):
                T[k] = T[k] / mean
        if not changed:
            break
    X = bodies()

    # how well the cameras now agree on the bodies, across their pictures
    across = []
    for p in people:
        for k in range(K):
            if (p, k) not in obs:
                continue
            w, ok = obs[(p, k)]
            use = ok & (X[p][1] >= 2)
            if not use.any():
                continue
            w = w[use] * size[p]
            e = np.einsum('ij,fkj->fki', R[k], w) + T[k] - X[p][0][use]
            s = _sight(R[k], w)
            e = e - (e * s).sum(-1, keepdims=True) * s
            across.append(np.linalg.norm(e, axis=-1).ravel())
    if not across or not any(fitted[1:]):
        log("Cameras: nothing to place them by (nobody seen by two cameras "
            "at once for long enough)")
        if why is not None:
            why.append(("nothing", None))
        return None
    mid = float(np.median(np.concatenate(across)))
    if mid > MAX_ACROSS:
        if why is not None:
            why.append(("apart", mid))
        log(f"Cameras: placed once for the take they still disagree on the "
            f"bodies by {mid * 100:.1f} cm across their pictures, so each "
            f"person's cameras are placed on their own")
        return None
    log(f"Cameras: placed once for the take from everybody in it; they agree "
        f"on a body to {mid * 100:.1f} cm across their pictures")
    log("Sizes (the body model's own = 1.00): "
        + ", ".join(f"person {p} {size[p]:.3f}" for p in people))
    return {"turns": np.stack(R), "places": np.stack(T), "sizes": dict(size),
            "across": mid, "fitted": fitted}
