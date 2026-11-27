# pipeline/capture_camera.py
# ==========================
# "Add Camera": where the real camera was, expressed in the BVH's world,
# written as a small JSON next to the BVH for the addon to turn into a
# Blender camera (blendcap/scene_camera.py).
#
# Everything needed is already in the capture:
#   position   the model reports the performer relative to the camera
#              every frame (cam_t + the hips' offset in the body frame).
#              Frame 0 anchors it: the camera goes where it must be for
#              the hips to sit where the capture saw them, measured
#              against the hips as they finally ship in the BVH (after
#              centring, grounding, foot passes, Hold Cropped Limbs),
#              less what Root Smoothing moved them on that frame: the
#              capture saw the body where it was, not where the
#              smoothed path starts (modes.run_body, pin_shift).
#   lens       the focal length the capture used (estimated or manual),
#              in pixels, plus the picture size -> a 36 mm-sensor lens.
#   tilt       the Camera Angle Offset baked into this BVH. That offset
#              levels the performer by rotating the pose; the camera
#              takes the same rotation, so pose and view stay consistent.
#              There is NO tilt estimation here, on purpose.
#   multi-camera  the merge's calibration (gravity from the floor,
#              how each camera is turned against camera A, where it
#              stands) gives every viewpoint in the merged take's
#              world; `view` picks one.
#
# One static camera per clip: a single video cannot separate camera
# motion from performer motion, so a moving camera's motion stays on the
# character, as before. The match is exact at the anchor frame; it drifts
# when the body is deliberately pinned (Root off, hips held) or when a
# tilted camera's offset has not been dialled in.
#
# Frames: capture space is x right, y DOWN, z forward (camera at the
# origin, looking along +z). The BVH world is x right, y UP, z BACK, which
# is also Blender's own camera convention (looks along -z, +y up), so a
# camera's axes convert with the same flip on both sides.

import json
import os

import numpy as np

F_FLIP = np.diag([1.0, -1.0, -1.0])
SENSOR_MM = 36.0
J_HIPS = 1


def _lens_from(npz_like):
    """(focal_px, (width, height)) from a capture NPZ, or None when the
    capture predates the saved picture size."""
    files = getattr(npz_like, "files", ())
    if "frame_size" not in files or "focal_length" not in files:
        return None
    fs = np.asarray(npz_like["frame_size"], dtype=np.float64).reshape(-1)
    f = np.asarray(npz_like["focal_length"], dtype=np.float64).reshape(-1)
    f = f[np.isfinite(f) & (f > 0)]
    if fs.size < 2 or f.size == 0 or fs[0] <= 0 or fs[1] <= 0:
        return None
    return float(np.median(f)), (int(fs[0]), int(fs[1]))


def _first_hips_in_capture(npz):
    """The hips' position in capture space on the first usable frame
    (the BVH stage repairs unusable leading frames by copying the first
    good one, so this is what frame 0 of the BVH was built from)."""
    ct = np.asarray(npz["cam_t"], dtype=np.float64)
    jc = np.asarray(npz["joint_coords"], dtype=np.float64)[:, J_HIPS]
    ok = np.isfinite(ct).all(axis=1) & np.isfinite(jc).all(axis=1)
    if not ok.any():
        return None
    f = int(np.argmax(ok))
    return ct[f] + jc[f]


def camera_from_capture(npz, input_path, hips_world0, root_world0,
                        R_level=None, view="a"):
    """Camera pose + lens for this capture, in the BVH world (metres,
    y up). Returns (data dict, None) or (None, reason).

    hips_world0 / root_world0: the hips' and the root's final world
        position on BVH frame 0.
    R_level: the Camera Angle Offset rotation baked into this BVH (BVH
        world frame), or None.
    view: 'a' to 'd' for a multi-camera take, one of the cameras the
        merge placed; ignored otherwise."""
    p0 = _first_hips_in_capture(npz)
    if p0 is None:
        return None, "no usable frame in the capture"

    folder = os.path.dirname(os.path.abspath(input_path))
    calib_path = os.path.join(folder, "calibration.npz")
    files = getattr(npz, "files", ())
    two_cam = "gravity_aligned" in files and os.path.isfile(calib_path)
    view = str(view).lower()
    view = view if view in ("b", "c", "d") else "a"

    R_m = np.eye(3)          # camera axes in capture space (y down)
    c_m = np.zeros(3)        # camera centre in capture space
    lens_src = npz
    if two_cam:
        cal = np.load(calib_path)
        GA = np.asarray(cal["GA"], dtype=np.float64)
        k = "abcd".index(view)
        if k > 1 and not ("turns" in cal.files and k < len(cal["turns"])):
            return None, (f"camera {view.upper()} is not in this merge "
                          f"(run the capture again to merge every camera)")
        if k == 1 and "turns" not in cal.files:
            # merged before more than two cameras could be
            R_ab = np.asarray(cal["R"], dtype=np.float64)
            off = np.asarray(cal["offset"], dtype=np.float64).reshape(3)
            R_m = GA @ R_ab
            c_m = GA @ off
        elif k > 0:
            # (a person of a multi-person take: the turns the take's
            # cameras were PLACED with, which go with `offsets`;
            # `turns` there is the one read off this performer's poses)
            turns = cal["place_turns"] if "place_turns" in cal.files else cal["turns"]
            R_m = GA @ np.asarray(turns[k], dtype=np.float64)
            c_m = GA @ np.asarray(cal["offsets"][k],
                                  dtype=np.float64).reshape(3)
        else:
            R_m = GA
        sub = os.path.join(folder, f"cam_{view}", "mhr_frames.npz")
        if not os.path.isfile(sub):
            return None, f"cam_{view} capture not found next to the merge"
        lens_src = np.load(sub)
    else:
        view = "a"

    lens = _lens_from(lens_src)
    if lens is None:
        return None, ("this capture predates the feature (no picture size "
                      "in the tracking data); run the capture again")
    focal_px, (w, h) = lens

    R_c = np.eye(3) if R_level is None else np.asarray(R_level, dtype=np.float64)
    hips_world0 = np.asarray(hips_world0, dtype=np.float64).reshape(3)
    location = hips_world0 + R_c @ F_FLIP @ (c_m - p0)
    rotation = R_c @ F_FLIP @ R_m @ F_FLIP          # columns: right, up, back
    return {
        "version": 1,
        "location": [float(v) for v in location],
        "rotation": [[float(v) for v in row] for row in rotation],
        "anchor_root": [float(v) for v in np.asarray(root_world0, dtype=np.float64).reshape(3)],
        "focal_px": focal_px,
        "frame_size": [w, h],
        "sensor_mm": SENSOR_MM,
        "lens_mm": focal_px * SENSOR_MM / float(max(w, h)),
        "view": view,
        "two_camera": bool(two_cam),
    }, None


def _camera_line(data):
    loc = data["location"]
    return (f"       Camera: lens {data['lens_mm']:.1f} mm (36 mm sensor), "
            f"{data['frame_size'][0]}x{data['frame_size'][1]}, at "
            f"({loc[0]:+.2f}, {loc[1]:+.2f}, {loc[2]:+.2f}) m"
            + (f", multi-camera take: camera {data['view'].upper()}"
               if data["two_camera"] else ""))


def write_camera_sidecar(path, npz, input_path, hips_world0, root_world0,
                         R_level=None, view="a", n_frames=0):
    """Write the camera JSON.

    view 'all' (what the addon asks for with every body export, so that
    Add Camera can be clicked afterwards without exporting again): every
    camera of the capture, the one of an ordinary capture or each one a
    multi-camera merge placed, as {"views": {"a": {...}, ...}}. When
    there is none, "views" is empty and "why" says so.

    A single letter writes that one camera at the top level (or removes
    a stale file and says why not)."""
    if str(view).lower() == "all":
        views, why = {}, ""
        for v in "abcd":
            data, reason = camera_from_capture(npz, input_path, hips_world0,
                                               root_world0, R_level=R_level, view=v)
            if data is None:
                why = why or reason
                continue
            if data["view"] != v:
                break           # an ordinary capture: one camera, already taken
            data["frames"] = int(n_frames)
            views[v] = data
            print(_camera_line(data))
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 2, "views": views,
                       "why": "" if views else why}, f, indent=1)
        if not views:
            print(f"       Camera: none for this export ({why})")
        return bool(views)

    data, why = camera_from_capture(npz, input_path, hips_world0,
                                    root_world0, R_level=R_level, view=view)
    if data is None:
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass
        print(f"[WARNING] Add Camera skipped - {why}.")
        return False
    data["frames"] = int(n_frames)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    print(_camera_line(data))
    return True
