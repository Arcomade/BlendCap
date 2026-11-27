# blendcap/scene_camera.py
# ========================
# "Add Camera": turns the camera JSON the BVH export wrote
# (pipeline/capture_camera.py) into a Blender camera that sits where the
# real camera was, with the matching lens and, optionally, the footage
# as its background.
#
# Add Camera is a BUTTON for the selected rig, or for the selected
# clip's rig when that clip has exactly one (target_rig; user decisions
# 2026-11-19; it was a checkbox that only acted together with a Generate
# before). Every body export records where the capture's camera(s)
# stood, and that is kept ON THE RIG at import (RIG_TAG). The button
# builds the camera from it at once, with no new export; a rig from a
# multi-camera take carries every camera the merge placed, and the
# Camera View dropdown under the button picks one. A rig made before
# this has nothing recorded until it is generated or baked again.
#
# The camera is ITS OWN OBJECT, not parented to anything (user decision
# 2026-11-10: it should sit alongside the BVH, not nested under it). It
# is placed in the world relative to wherever the rig is at that moment
# - a fresh import lands at the 3D cursor, a regenerate keeps the rig's
# previous placement - and linked into the rig's collections. After
# that it is an ordinary, independent camera: moving the rig later does
# NOT move it; the Add Camera button again, a regenerate, a Bake, or a
# drag of the rig's Camera Angle Offset / Height sliders places it again
# from the rig's current position. It is never made the scene's active
# camera.
#
# One camera per rig, found again by a tag (the rig's name), so the
# button, a regenerate or a Bake updates the same object instead of
# piling up copies. A regenerate or a Bake never MAKES a camera: it
# only keeps one the rig already has in step with the new BVH.

import json
import math
import os

import bpy
from mathutils import Matrix, Quaternion, Vector

TAG = "blendcap_camera_for"
RIG_TAG = "blendcap_capture_cameras"        # on the rig: see remember()
_FOOTAGE_TAG = "blendcap_footage"
# BVH world (x right, y up, z back) -> Blender (x right, y forward, z up):
# the same conversion the BVH importer applies (axis_forward='-Z',
# axis_up='Y').
_BVH_TO_BLENDER = Matrix(((1.0, 0.0, 0.0),
                          (0.0, 0.0, -1.0),
                          (0.0, 1.0, 0.0)))
FOOTAGE_ALPHA = 0.5


def find_camera(stem):
    """The BlendCap camera that belongs to the rig named `stem`, or None."""
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA' and obj.get(TAG) == stem:
            return obj
    return None


def _read_sidecar(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "location" in data and "rotation" in data and "lens_mm" in data:
            return data
    except Exception:
        pass
    return None


def _place(cam, rig, loc, quat):
    """Put the camera in the WORLD at the pose (loc, quat) given in the
    rig's object space, using the rig's current placement. The camera
    stays unparented (one an earlier build parented to the rig is
    released), keeps unit scale, and its place is written through
    matrix_world so a parent the user gave it themselves is respected."""
    if cam.parent is rig:
        cam.parent = None
        cam.matrix_parent_inverse = Matrix.Identity(4)
    mw = rig.matrix_world
    world_loc = mw @ Vector(loc)
    world_rot = mw.to_quaternion() @ quat
    cam.rotation_mode = 'XYZ'
    cam.matrix_world = Matrix.LocRotScale(world_loc, world_rot,
                                          Vector((1.0, 1.0, 1.0)))


def _remember_pose(cam, data):
    """The export's pose in the rig's object space, kept on the camera
    so it can be placed again from the rig's current position (the live
    sliders, see follow_live_tilt)."""
    loc = _BVH_TO_BLENDER @ Vector(data["location"])
    rot = (_BVH_TO_BLENDER @ Matrix(data["rotation"])).to_quaternion()
    pivot = _BVH_TO_BLENDER @ Vector(data.get("anchor_root", (0.0, 0.0, 0.0)))
    cam["blendcap_cam_base_loc"] = list(loc)
    cam["blendcap_cam_base_rot"] = list(rot)
    cam["blendcap_cam_pivot"] = list(pivot)


def _apply_lens(cam, data):
    cd = cam.data
    cd.type = 'PERSP'
    cd.sensor_fit = 'AUTO'
    cd.sensor_width = float(data.get("sensor_mm", 36.0))
    cd.lens = float(data["lens_mm"])
    fs = data.get("frame_size") or (0, 0)
    cam["blendcap_frame_size"] = [int(fs[0]), int(fs[1])]
    cam["blendcap_cam_view"] = str(data.get("view", "a"))


def _apply_footage(cam, video, start, n_frames, show):
    """The source video as the camera's background: behind everything,
    half transparent, following the clip's trim. Only entries BlendCap
    made are ever removed; a background the user added stays."""
    cd = cam.data
    for bg in list(cd.background_images):
        img = getattr(bg, "image", None)
        if img is not None and img.get(_FOOTAGE_TAG):
            cd.background_images.remove(bg)
    cam["blendcap_footage_path"] = video or ""
    cam["blendcap_footage_start"] = int(start or 0)
    if not show or not video:
        return False
    path = bpy.path.abspath(video)
    if not os.path.isfile(path):
        # An image-sequence source that was never converted, or a clip
        # that has since moved: nothing to show, the camera still works.
        print(f"[BlendCap] Camera footage not shown: no video file at {path}")
        return False
    try:
        img = bpy.data.images.load(path, check_existing=True)
    except Exception as e:
        print(f"[BlendCap] Camera footage not shown: {e}")
        return False
    img[_FOOTAGE_TAG] = 1
    bg = cd.background_images.new()
    bg.source = 'IMAGE'
    bg.image = img
    iu = bg.image_user
    iu.frame_start = 1                      # the BVH imports at frame 1
    iu.frame_offset = int(start or 0)       # the clip's trim
    dur = int(n_frames or 0) or int(getattr(img, "frame_duration", 0) or 0)
    iu.frame_duration = max(1, dur) if dur else 100000
    iu.use_auto_refresh = True
    bg.alpha = FOOTAGE_ALPHA
    bg.display_depth = 'BACK'
    bg.frame_method = 'FIT'
    cd.show_background_images = True
    return True


def _show_footage_pref():
    try:
        from .helpers import addon_prefs
        prefs = addon_prefs()
        return bool(getattr(prefs, "show_footage_in_cameras", True))
    except Exception:
        return True


def rig_stem(rig):
    """The name a rig's camera is tagged with: the BVH it was imported
    from (survives the rig being renamed), else the rig's name. The same
    rule Bake uses."""
    path = str(rig.get("blendcap_bvh_path", "") or "")
    return os.path.splitext(os.path.basename(path))[0] if path else rig.name


def stored(rig):
    """What this rig's last Generate / Bake recorded about the capture's
    camera(s): {'views': {'a': {...}}, 'footage': {'a': {'video',
    'start'}}, 'why': str}, or None when nothing was recorded."""
    try:
        if rig is None or rig.type != 'ARMATURE' or RIG_TAG not in rig:
            return None
        data = json.loads(str(rig[RIG_TAG]))
        if not isinstance(data, dict) or not isinstance(data.get("views"), dict):
            return None
        data.setdefault("footage", {})
        data.setdefault("why", "")
        return data
    except Exception:
        return None


def views_of(rig):
    """The capture cameras this rig carries, as letters: ['a'], or
    ['a', 'b', ...] for a rig from a multi-camera take; [] for none."""
    data = stored(rig)
    return sorted(data["views"]) if data else []


def remember(rig, job):
    """Called right after a generated BVH is imported as `rig`: keep the
    camera(s) the export wrote, on the rig, for Add Camera.

    job: what the Generate / Bake operator recorded for this run
    ({'stem', 'sidecar', 'footage': {view: {'video', 'start'}}}), or
    None for an export without a body. The file is one-shot: a stale
    pose must never be read by a later run."""
    if rig is None or rig.type != 'ARMATURE' or not job or not job.get('sidecar'):
        return False
    path = job['sidecar']
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return False
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    views = data.get("views")
    if not isinstance(views, dict):
        # a single-view file (the export's older form)
        views = {str(data.get("view", "a")): data} if "location" in data else {}
    views = {k: v for k, v in views.items()
             if "location" in v and "rotation" in v and "lens_mm" in v}
    footage = job.get('footage') or {}
    # A person of a Multi-Camera & Multi-Person take is merged from the
    # cameras that saw them, lettered from a in the take's order
    # (people_tracks.extract_person_cameras). When one camera did not,
    # the views here and the take's footage are lettered differently:
    # the export's own note says which of the take's cameras each is.
    try:
        with open(os.path.join(os.path.dirname(path), ".person_of"),
                  encoding="utf-8") as fh:
            took = [str(c).lower() for c in json.load(fh).get("cameras") or []]
        if took and took != sorted(footage)[:len(took)]:
            footage = {"abcd"[i]: footage[c] for i, c in enumerate(took)
                       if c in footage}
    except Exception:
        pass
    rig[RIG_TAG] = json.dumps({"views": views,
                               "footage": footage,
                               "why": str(data.get("why", "") or "")})
    return True


def _apply(cam, rig, data, view):
    v = data["views"][view]
    foot = data["footage"].get(view) or {}
    _remember_pose(cam, v)
    _apply_lens(cam, v)
    cam["blendcap_cam_view"] = str(view)     # the view asked for, whatever the data says of itself
    shown = _apply_footage(cam, foot.get('video', ''), foot.get('start', 0),
                           v.get("frames", 0), _show_footage_pref())
    # The import (or the user) has just set the rig's place; make sure
    # its world matrix says so before the camera is put next to it.
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    follow_live_tilt(rig, cam)   # places it (with any live slider offsets)
    fs = cam["blendcap_frame_size"]
    print(f"[BlendCap] Camera '{cam.name}': lens {cam.data.lens:.1f} mm, "
          f"footage {'shown' if shown else 'not shown'}"
          f" (the view matches the footage when the scene's resolution "
          f"has the video's shape, {fs[0]}x{fs[1]})")


def update_camera(rig, stem):
    """After a regenerate or a Bake: a camera this rig already has takes
    the pose the rig now carries, for the view it shows. A rig without
    a camera gets none (that is the Add Camera button's to make)."""
    if rig is None or rig.type != 'ARMATURE':
        return None
    cam = find_camera(stem)
    if cam is None:
        return None
    data = stored(rig)
    view = str(cam.get("blendcap_cam_view", "a"))
    if data and view in data["views"]:
        _apply(cam, rig, data, view)
    return cam


def add_camera(rig, view):
    """The Add Camera button: create this rig's camera, or move the one
    it has, to capture camera `view`. Returns (camera, created)."""
    data = stored(rig)
    stem = rig_stem(rig)
    cam = find_camera(stem)
    created = cam is None
    if created:
        cam_data = bpy.data.cameras.new(f"{stem}_Camera")
        cam = bpy.data.objects.new(f"{stem}_Camera", cam_data)
        cam[TAG] = stem
        colls = list(rig.users_collection) or [bpy.context.scene.collection]
        for coll in colls:
            coll.objects.link(cam)
    _apply(cam, rig, data, view)
    return cam, created


# Why Add Camera is greyed out (shown under its tooltip).
NEED_BVH = "Generate the BVH first"
NEED_RIG = "Select the rig the camera is for"


def _is_blendcap_rig(obj):
    try:
        return (obj is not None and obj.type == 'ARMATURE'
                and "blendcap_source_video" in obj)
    except Exception:
        return False


def _has_body(rig):
    try:
        return bool(views_of(rig)) or "Hips" in rig.data.bones
    except Exception:
        return False


def target_rig(context):
    """The rig Add Camera acts on, and when there is none, why:
    (rig, "") or (None, reason).

      - a BlendCap rig that is selected (the active object): that rig;
      - else the selected CLIP's rig, when the scene holds exactly one
        body rig generated from it (the user, 2026-11-19: with one
        person in one clip the clip is enough). Several (the same clip
        generated with different options) is not a choice this can
        make, and neither is a multi-person capture, where every person
        has a rig of their own: those need the rig selected."""
    obj = getattr(context, "active_object", None)
    if _is_blendcap_rig(obj):
        return obj, ""
    try:
        from .helpers import _resolve_clip_name
        clip = _resolve_clip_name(context)
    except Exception:
        clip = ""
    if not clip:
        return None, NEED_BVH
    own, people = [], False
    prefix = clip + "_P"
    for o in context.scene.objects:
        if not _is_blendcap_rig(o):
            continue
        src = str(o.get("blendcap_source_video", "") or "")
        if src == clip:
            if _has_body(o):
                own.append(o)
        elif src.startswith(prefix) and src[len(prefix):].isdigit():
            people = True           # a person of this (multi-person) capture
    if len(own) == 1 and not people:
        return own[0], ""
    if own or people:
        return None, NEED_RIG
    return None, NEED_BVH


def target_views(context):
    """The capture cameras of the rig Add Camera would act on."""
    try:
        return views_of(target_rig(context)[0])
    except Exception:
        return []


class BLENDCAP_OT_add_camera(bpy.types.Operator):
    bl_idname = "blendcap.add_camera"
    bl_label = "Add Camera"
    bl_description = ("Add a camera that matches the footage's viewpoint, "
                      "for the selected rig or the selected clip's rig")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        from .helpers import blendcap_state, bvh_state
        if blendcap_state.get('running') or bvh_state.get('running'):
            return False
        rig, why = target_rig(context)
        if rig is None:
            try:
                cls.poll_message_set(why)
            except Exception:
                pass
            return False
        return True

    def execute(self, context):
        rig, why = target_rig(context)
        if rig is None:
            self.report({'ERROR'}, why + ".")
            return {'CANCELLED'}
        data = stored(rig)
        if not data or not data["views"]:
            if data and data.get("why"):
                msg = f"No camera for this rig: {data['why']}."
            elif "Hips" not in rig.data.bones:
                msg = ("Add Camera needs a rig with a body (not a hands, "
                       "root or face one).")
            else:
                msg = ("This rig was generated before cameras could be added "
                       "afterwards. Click Generate/Reload BVH or Bake Settings "
                       "to BVH once, then Add Camera.")
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}
        views = sorted(data["views"])
        want = str(getattr(context.scene.blendcap_props, "multicam_camera_view", "A")
                   or "A").lower()
        view = want if want in views else views[0]
        cam, created = add_camera(rig, view)
        self.report({'INFO'}, f"Camera {'added' if created else 'updated'}: {cam.name}"
                    + (f" (camera {view.upper()})" if len(views) > 1 else ""))
        return {'FINISHED'}


def follow_live_tilt(rig, cam=None):
    """Place the rig's camera from the rig's current position and its
    per-rig sliders, so the view through it stays the footage's view
    while they are dragged:
      - Camera Angle Offset: the live preview turns the body about the
        Root; the camera takes the same turn about the Root's place on
        the first frame.
      - the vertical shifts the preview writes onto the Hips keys (the
        Height slider, and the automatic re-grounding after a tilt):
        the camera rises and sinks by the same amount.
    With every slider at zero this is the pose the export computed,
    relative to the rig. A Bake recomputes the pose from the baked
    result. The camera is not parented, so this is also the moment a
    camera that was left behind by a moved rig catches up with it."""
    if rig is None or rig.type != 'ARMATURE':
        return
    if cam is None:
        cam = find_camera(rig.name)
    if cam is None:
        return
    base_loc = cam.get("blendcap_cam_base_loc")
    base_rot = cam.get("blendcap_cam_base_rot")
    pivot = cam.get("blendcap_cam_pivot")
    if base_loc is None or base_rot is None or pivot is None:
        return
    off = getattr(rig, "blendcap_camera_offset", None)
    pitch = math.radians(getattr(off, "pitch_deg", 0.0)) if off else 0.0
    roll = math.radians(getattr(off, "roll_deg", 0.0)) if off else 0.0
    # Same world-space rotation the live preview writes onto the Root
    # (properties._apply_camera_offset_to_rig), brought into the rig's
    # object space, where the remembered pose lives.
    R_q = Quaternion((0.0, 1.0, 0.0), roll) @ Quaternion((1.0, 0.0, 0.0), pitch)
    arm_q = rig.matrix_world.to_quaternion()
    R_obj = arm_q.inverted() @ R_q @ arm_q
    base_loc = Vector(base_loc)
    pivot = Vector(pivot)
    loc = pivot + R_obj @ (base_loc - pivot)
    quat = R_obj @ Quaternion(base_rot)
    # World-Z shifts the live preview applied to the body (recorded on
    # the rig by properties.py), brought into the rig's object space.
    dz = (float(rig.get("_blendcap_height_offset_applied_m", 0.0))
          + float(rig.get("_blendcap_autoground_applied_m", 0.0)))
    if abs(dz) > 1e-9:
        try:
            loc = loc + rig.matrix_world.to_3x3().inverted() @ Vector((0.0, 0.0, dz))
        except ValueError:
            pass            # a zero-scaled rig has no inverse; skip the shift
    _place(cam, rig, loc, quat)
