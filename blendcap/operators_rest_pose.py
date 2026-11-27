"""Custom rest-pose preset operators.

Three operators back the "Custom Rest Pose" subsection of the Retargeting
panel:

  - BLENDCAP_OT_save_rest_pose: scans the active armature's pose bones
    and writes each posed bone's REST and POSE quaternions (both in
    armature-local Blender frame, no axis conversion) to a JSON preset.

  - BLENDCAP_OT_delete_rest_pose_preset: deletes the currently-selected
    preset. Poll() rejects paths outside the save directory.

  - BLENDCAP_OT_preview_rest_pose_preset: switches focus to the source
    rig, enters Pose Mode, clears every bone's pose, and drives the
    saved preset onto the rig as a *rotation-only* pose. Translation
    and scale are NOT written, so children stay attached to their
    FK-derived parent positions (no scattered-bones artifact). Sets
    pb.rotation_quaternion in bone-local space rather than pb.matrix
    in armature space, computed in topological order so each child
    sees the parent's already-posed state. No keyframes are inserted.
    Scrubbing the timeline restores the animation, so the preview is
    non-destructive.

The retargeting apply path lives in `operators_retarget.py:_prepare()`.
It calls `load_rest_pose_preset_to_override` (defined here) to populate
`source_rest_override` — the same dict the live "Use Current Source
Pose as Rest" toggle populates.

JSON FORMAT
-----------
v5 (current — adds pose translation + scale to v4 so manual rig
    scaling / repositioning at save time survives save→load roundtrips.
    Required for workflows where the user scales shoulder / arm bones
    manually to make BVH retargeting fit a non-standard target rig):
    {
      "name": "...",
      "format_version": 5,
      "include_loc_scale": true,   <-- optional; mirrors the scene's
                                       "Include Location & Scale" checkbox
                                       state at save time. On preset
                                       selection the scene prop is set
                                       from this value, so each preset
                                       carries its own intent and
                                       preview/bake stay consistent.
                                       Missing → defaults to True (matches
                                       legacy v5 behavior where preview
                                       always applied loc/scale).
      "bones": {
        "BoneName": {
          "rest":  [w, x, y, z],   <-- pb.bone.matrix_local quaternion
          "pose":  [w, x, y, z],   <-- pb.matrix quaternion (absolute)
          "loc":   [x, y, z],      <-- pb.matrix translation (arm space)
          "scale": [x, y, z]       <-- pb.matrix scale (arm space)
        }
      }
    }

  v5 readers fall back gracefully when "loc" / "scale" are absent
  (treats as v4 — uses live edit-bone rest translation and identity
  scale).

v4 (legacy — rotation-only, rig-portable):
    {
      "name": "...",
      "format_version": 4,
      "bones": {
        "BoneName": {
          "rest": [w, x, y, z],   <-- pb.bone.matrix_local quaternion
          "pose": [w, x, y, z]    <-- pb.matrix quaternion (absolute pose)
        }
      }
    }

  Loading: per bone, compare the saved `rest` to the live rig's
  pb.bone.matrix_local quaternion.
    - If they match within tolerance → use the saved `pose` directly.
      Same-rig case is bit-identical to absolute-save behavior.
    - If they differ → compute delta = saved_pose @ saved_rest.inv,
      then target = delta @ live_rest. Cross-rig case (face vs.
      no-face head, etc.) gets the saved rotation re-anchored against
      whatever rest the live rig actually has.

v3 (legacy — pose-only delta, broken cross-rig):
    bones: {"BoneName": [w, x, y, z]}  <-- world delta, only loadable
                                            via composition with live
                                            rest (gives a hybrid that
                                            drifts when rests differ)

v2 (legacy — absolute pose, no rest snapshot):
    bones: {"BoneName": {"translation": [...], "quaternion": [...]}}
    Loaded as absolute. Same-rig works, cross-rig doesn't compensate.
"""
import bpy
import os
import json
import math

from mathutils import Matrix, Quaternion, Vector

from .helpers import _filename_error
from .retarget.name_matching import _detect_rig_prefix, _resolve_against_rig
from .properties import (_rest_pose_presets_dir_user,
                         _rest_pose_presets_dir_legacy,
                         _rest_pose_presets_dir_bundled)


def _deletable_rest_pose_path(path):
    """A preset may be deleted from the user-save folder or from the
    legacy addon-folder location (read-only leftovers migration
    couldn't move) - never from bundled/ (shipped presets)."""
    bundled_dir = _rest_pose_presets_dir_bundled()
    try:
        if os.path.commonpath([path, bundled_dir]) == bundled_dir:
            return False
    except ValueError:
        pass
    for d in (_rest_pose_presets_dir_user(),
              _rest_pose_presets_dir_legacy()):
        try:
            if os.path.commonpath([path, d]) == d:
                return True
        except ValueError:
            continue
    return False


# Tolerance for "this bone is at rest" detection at save time. We compare
# pb.matrix to pb.bone.matrix_local; a bone whose pose-mode rotation is
# identity AND whose ancestors are also unposed has pb.matrix ==
# pb.bone.matrix_local exactly. Anything within tolerance is treated as
# unposed and skipped from the JSON to keep the file readable.
_REST_TOL = 1e-6

# Tolerance for "saved rest matches live rest" at load time. Generous
# (1e-4 on the quaternion dot product) because file→json→file round-trips
# can introduce small float drift. Quaternion sign ambiguity is handled
# by taking abs() of the dot product (q and -q represent the same
# rotation).
_REST_MATCH_TOL = 1e-4


def _is_at_rest(pb):
    """True if pb.matrix matches pb.bone.matrix_local within tolerance,
    in rotation, translation, AND scale.

    The scale check is load-bearing: without it, a bone whose only
    pose change is scale (e.g. user selects all bones and scales with
    Individual Origins, or scales the root bone around its own head)
    has a diff with identity rotation and zero translation but
    non-identity scale. The bone would silently skip save -> no scale
    entry in the JSON -> load reconstructs without that scale ->
    visible "bones detached / unscaled" symptom because descendants
    that DID save still expect the parent to carry its scale, and the
    pb.matrix setter at restore time compensates by baking extra
    scale into their own basis."""
    diff = pb.matrix @ pb.bone.matrix_local.inverted()
    q = diff.to_quaternion()
    if (abs(q.w - 1.0) > _REST_TOL
            or abs(q.x) > _REST_TOL
            or abs(q.y) > _REST_TOL
            or abs(q.z) > _REST_TOL):
        return False
    t = diff.to_translation()
    if (abs(t.x) > _REST_TOL
            or abs(t.y) > _REST_TOL
            or abs(t.z) > _REST_TOL):
        return False
    s = diff.to_scale()
    return (abs(s.x - 1.0) <= _REST_TOL
            and abs(s.y - 1.0) <= _REST_TOL
            and abs(s.z - 1.0) <= _REST_TOL)


def _topological_pose_bones(rig):
    """Yield pose bones in parent-before-child order. Required for the
    preview operator: each child's pose_basis is computed against the
    parent's CURRENT pose, so the parent must be set first."""
    def depth(b):
        d = 0
        while b.parent is not None:
            b = b.parent
            d += 1
        return d
    return sorted(rig.pose.bones, key=lambda pb: depth(pb.bone))


def _quats_equivalent(q1, q2, tol=_REST_MATCH_TOL):
    """True if q1 and q2 represent the same rotation (within tol).
    Quaternion sign ambiguity: q and -q are the same rotation, so we
    compare |q1·q2| against 1 rather than q1·q2."""
    return abs(q1.dot(q2)) >= 1.0 - tol


def _resolve_target_quaternion(saved_rest_q, saved_pose_q, live_rest_q,
                               frame_q=None):
    """Pick the right target world rotation for one bone.

    Returns the quaternion that should appear as the bone's world pose
    rotation under the override.

    - Same-rig (saved_rest matches live_rest): return saved_pose
      directly. Bit-identical to v2 absolute-save behavior.
    - Cross-rig (rests differ): compute the world delta the user
      applied at save time and re-apply it relative to the live rest.
    - frame_q (see skeleton_fit): the same skeleton sitting TURNED inside
      its armature (a raw FBX import against one with its transforms
      applied). The delta was measured in the saved rig's armature frame,
      so it is carried into this rig's frame first. None = no turn, and
      the maths above is untouched.
    """
    if _quats_equivalent(saved_rest_q, live_rest_q):
        return saved_pose_q
    delta_q = saved_pose_q @ saved_rest_q.inverted()
    if frame_q is not None:
        delta_q = frame_q @ delta_q @ frame_q.inverted()
    return delta_q @ live_rest_q


# --- Is this preset for this skeleton? (found live 2026-11-15) -------------
# Matching by name alone is not enough: a BlendCap pose on a Mixamo rig
# matches 'Hips', 'Spine', 'LeftArm'... and then applies rotations meant
# for bones that sit completely differently, which looks like a broken rig.
# Every v4+ entry stores the bone's REST orientation on the rig it was saved
# from. On the same skeleton, live_rest @ saved_rest^-1 is the same rotation
# for every bone (identity, or how the two armatures are turned relative to
# each other); on a different skeleton it is all over the place.
# Measured on the user's presets and test rigs:
#   same skeleton (BlendCap on BlendCap, Mixamo on other Mixamo
#   characters and imports): 98 to 100 % of matched bones within 45 deg
#   different skeleton, either way round: 4 to 36 %.
_SKELETON_MIN_BONES = 3          # fewer than this: nothing to judge by
_SKELETON_AGREE_DEG = 45.0
_SKELETON_AGREE_SHARE = 2.0 / 3.0
# A turn this large between the two armature frames is an import
# convention (90 or 180 deg). Different characters of one skeleton differ
# by 10 to 20 deg bone by bone, which is NOT a turn of the whole frame and
# must leave the old maths alone.
_FRAME_TURN_MIN_DEG = 30.0


def _angle_deg(q1, q2):
    a = q1.rotation_difference(q2).angle
    if a > math.pi:
        a = 2.0 * math.pi - a
    return math.degrees(a)


def skeleton_fit(bones, resolved, src_rig, fmt):
    """(share of matched bones whose rest agrees, frame turn quaternion,
    frame turn in degrees). Share is None when it can't be judged (old
    preset formats with no rest data, or under three matched bones); the
    quaternion is None unless the whole armature frame is turned by at
    least _FRAME_TURN_MIN_DEG."""
    if fmt < 4:
        return None, None, 0.0
    turns = []
    for preset_name, bone_name in resolved.items():
        entry = bones.get(preset_name)
        if not isinstance(entry, dict):
            continue
        rest = entry.get("rest")
        if not rest or len(rest) != 4:
            continue
        live = src_rig.pose.bones[bone_name].bone.matrix_local.to_quaternion()
        turns.append(live @ Quaternion(rest).inverted())
    n = len(turns)
    if n < _SKELETON_MIN_BONES:
        return None, None, 0.0
    # The turn most bones share: the one with the smallest median distance
    # to all the others (a medoid, so a few odd bones can't drag it).
    best = None
    for cand in turns:
        d = sorted(_angle_deg(cand, t) for t in turns)
        key = (d[n // 2], sum(d))
        if best is None or key < best[0]:
            best = (key, d, cand)
    _key, dists, frame_q = best
    share = sum(1 for x in dists if x <= _SKELETON_AGREE_DEG) / n
    turn_deg = _angle_deg(frame_q, Quaternion())
    if turn_deg < _FRAME_TURN_MIN_DEG:
        return share, None, turn_deg
    return share, frame_q, turn_deg


WRONG_SKELETON_PREVIEW = ("This rest pose was made for a different skeleton "
                          "than the current Source Rig, so it cannot be "
                          "applied.")
WRONG_SKELETON_APPLY = ("This rest pose was made for a different skeleton "
                        "than the current Source Rig. Pick another rest "
                        "pose or untick Use Custom Rest Pose, then apply "
                        "again.")


# What the most recent load matched: {'total': bones in the preset,
# 'matched': found on the rig, 'renamed': of those, found under another
# namespace, 'rig_prefix': the prefix that resolved them}. Read by the
# Preview operator and the Apply operator for their messages.
LAST_LOAD_STATS = None


def _namespace_of(name):
    """'mixamorig:' for 'mixamorig:Hips', '' for 'Hips'. Only a colon
    counts: it is what Mixamo, Maya-style and Blender-linked rigs use for
    a namespace, and unlike '_' or '.' it never sits inside a plain bone
    name ('Left_Arm')."""
    i = name.rfind(':')
    return name[:i + 1] if i >= 0 else ""


def resolve_preset_bones(names, src_rig):
    """{preset bone name: bone name on src_rig} for every preset bone that
    can be found on the rig, and the rig prefix used ('' when none).

    A name that exists on the rig as written always wins, so a preset used
    on the rig it was saved from behaves exactly as before. Only the names
    left over are looked up without their namespace: the same Mixamo
    skeleton shows up as 'mixamorig:Hips', 'mixamorig1:Hips' (several
    characters in one file) or plain 'Hips', and a preset saved from any
    one of them used to do nothing on the others, without saying so
    (support item 33b). The rig's own prefix comes from the detection the
    retarget maps already use (_detect_rig_prefix)."""
    rig_bones = src_rig.pose.bones
    out = {n: n for n in names if n in rig_bones}
    if len(out) == len(names):
        return out, ""
    shorts = {n: n[len(_namespace_of(n)):] for n in names}
    counts = {}
    for n in names:
        ns = _namespace_of(n)
        if ns:
            counts[ns] = counts.get(ns, 0) + 1
    preset_ns = max(counts, key=lambda k: counts[k]) if counts else ""
    rig_prefix = _detect_rig_prefix(
        src_rig, [s for s in shorts.values() if s], fallback_prefix=preset_ns)
    used = set(out.values())
    used_prefix = ""
    for n in names:
        if n in out or not shorts[n]:
            continue
        found = _resolve_against_rig(shorts[n], src_rig, rig_prefix)
        if found in rig_bones and found not in used:
            out[n] = found
            used.add(found)
            # The prefix the rig really carries for this bone ('' on a
            # rig with plain names), for the console line.
            used_prefix = found[:len(found) - len(shorts[n])]
    return out, used_prefix


def match_summary(stats):
    """A few words for the status bar, or '' when every bone matched under
    its own name (nothing worth saying)."""
    if not stats or not stats.get('total'):
        return ""
    m, t = stats['matched'], stats['total']
    renamed = stats.get('renamed', 0)
    if m == t:
        if not renamed:
            return ""
        return f"all {t} bones matched by name, ignoring the namespace"
    line = f"{m} of {t} preset bones matched the source rig"
    if renamed:
        line += ", ignoring the namespace"
    return line


def load_rest_pose_preset_to_override(preset_path, src_rig):
    """Load a saved preset and convert it to a source_rest_override dict.

    Returns {bone_name: 4x4 mathutils.Matrix in armature-local space},
    matching the dict format that `_matrix_bake_pairs_iter` consumes,
    keyed by the bone's name ON src_rig (see resolve_preset_bones for how
    a preset saved under another namespace is matched). Bones that can't
    be found on src_rig are skipped; LAST_LOAD_STATS says how many.

    Translation / scale handling:
      - v5 entries with "loc" and/or "scale" use those values, so
        manual scaling and repositioning at save time survive the
        round-trip.
      - v4 (and earlier) entries fall back to LIVE edit-bone rest
        translation + identity scale — the historical rotation-only
        behavior. Required because v4 didn't capture loc/scale.
    """
    with open(preset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    bones = data.get("bones", {})
    fmt = int(data.get("format_version", 0))

    global LAST_LOAD_STATS
    resolved, rig_prefix = resolve_preset_bones(list(bones.keys()), src_rig)
    share, frame_q, turn_deg = skeleton_fit(bones, resolved, src_rig, fmt)
    wrong = share is not None and share < _SKELETON_AGREE_SHARE
    LAST_LOAD_STATS = {
        'total': len(bones),
        'matched': len(resolved),
        'renamed': sum(1 for k, v in resolved.items() if k != v),
        'rig_prefix': rig_prefix,
        # Made for another skeleton: nothing is returned, and the two
        # callers say so (WRONG_SKELETON_PREVIEW / _APPLY).
        'wrong_skeleton': wrong,
        'rest_agree_share': share,
        'frame_turn_deg': turn_deg if frame_q is not None else 0.0,
    }
    if wrong:
        print(f"[BlendCap] rest pose preset refused: only "
              f"{share * 100:.0f}% of its {len(resolved)} matched bones sit "
              f"at rest the way they do on '{src_rig.name}' (needs "
              f"{_SKELETON_AGREE_SHARE * 100:.0f}%). It was saved from a "
              f"different skeleton.")
        return {}
    if frame_q is not None:
        print(f"[BlendCap] rest pose preset: '{src_rig.name}' sits turned "
              f"{turn_deg:.0f} deg inside its armature compared with the "
              f"rig the preset was saved from; rotations are carried "
              f"across, saved positions and scales are not used.")

    out = {}
    for preset_name, entry in bones.items():
        bone_name = resolved.get(preset_name)
        if bone_name is None:
            continue
        pb = src_rig.pose.bones[bone_name]
        rest_m = pb.bone.matrix_local
        live_rest_q = rest_m.to_quaternion()

        target_q = None
        # Defaults: live rest translation + identity scale. v5 may
        # override one or both.
        target_loc = rest_m.translation.copy()
        target_scale = Vector((1.0, 1.0, 1.0))

        if fmt >= 4 and isinstance(entry, dict):
            saved_rest = entry.get("rest")
            saved_pose = entry.get("pose")
            if (saved_rest and saved_pose
                    and len(saved_rest) == 4 and len(saved_pose) == 4):
                target_q = _resolve_target_quaternion(
                    Quaternion(saved_rest),
                    Quaternion(saved_pose),
                    live_rest_q,
                    frame_q,
                )
            # v5: pull translation + scale if present. Absent fields
            # silently fall back to defaults above so v4 presets load
            # unchanged.
            saved_loc = entry.get("loc")
            saved_scale = entry.get("scale")
            if frame_q is not None:
                # Positions and scales saved in an armature frame that is
                # turned (and, on a raw Mixamo import, 100x larger) can't
                # be carried over by a rotation; leave the live rest's.
                saved_loc = saved_scale = None
            if saved_loc and len(saved_loc) == 3:
                target_loc = Vector(saved_loc)
            if saved_scale and len(saved_scale) == 3:
                target_scale = Vector(saved_scale)

        elif fmt == 3 and isinstance(entry, list) and len(entry) == 4:
            # v3: world delta only. We don't know the saved rest, so we
            # can only compose blindly with the live rest. This works
            # exactly when live_rest matches save-time rest (same-rig);
            # it drifts otherwise. Preserved for backward compat — the
            # save path no longer writes this format.
            target_q = Quaternion(entry) @ live_rest_q

        elif fmt == 2 and isinstance(entry, dict):
            # v2: absolute pose. Load as-is; no rest snapshot means we
            # can't compensate for cross-rig.
            t = entry.get("translation")
            q = entry.get("quaternion")
            if t and q and len(t) == 3 and len(q) == 4:
                target_q = Quaternion(q)
                target_loc = Vector(t)

        if target_q is None:
            continue

        out[bone_name] = Matrix.LocRotScale(target_loc, target_q, target_scale)
    return out


class BLENDCAP_OT_save_rest_pose(bpy.types.Operator):
    bl_idname = "blendcap.save_rest_pose"
    bl_label = "Save Current Pose as Preset"
    bl_description = ("Capture each posed bone's current rotations and "
                      "save them as a custom preset for retargeting use. "
                      "(position and scale can also be saved if "
                      "\"Include Location & Scale in Rest Pose\" is selected)")
    bl_options = {'REGISTER'}

    preset_name: bpy.props.StringProperty(
        name="Preset Name", default="Untitled Pose",
        description="Filename (without .json) for the saved preset")
    # Hidden flag set by the overwrite-confirm chain to bypass the
    # collision check on the second pass. SKIP_SAVE so it doesn't
    # persist into the Redo panel and surprise the user.
    confirmed: bpy.props.BoolProperty(
        default=False, options={'HIDDEN', 'SKIP_SAVE'})

    def _default_name(self, context):
        """Pre-fill order: currently-selected rest-pose preset's stem
        (so save = overwrite), else '<target rig name> Pose', else
        'Untitled Pose'. Rest-pose presets don't have a dirty-tracking
        dropdown like bone maps, so the selection IS the 'last loaded'."""
        scene = context.scene
        props = scene.blendcap_props
        sel = props.custom_rest_pose_preset
        if sel and not sel.startswith('__') and os.path.isfile(sel):
            stem, _ext = os.path.splitext(os.path.basename(sel))
            if stem:
                return stem
        tgt = scene.blendcap_retarget_target
        if tgt and getattr(tgt, "type", None) == 'ARMATURE' and tgt.name:
            return f"{tgt.name} Pose"
        return "Untitled Pose"

    def invoke(self, context, event):
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            self.report({'ERROR'},
                        "Select a posed armature in the viewport first "
                        "(active object must be the armature).")
            return {'CANCELLED'}
        self.preset_name = self._default_name(context)
        # Red name field in draw() is the sole collision signal. We
        # used to also flip the OK button label to "Overwrite" via
        # confirm_text, but Blender locks confirm_text at dialog-open
        # time -- the label stayed "Overwrite" even after the user
        # typed a non-colliding name, which was actively misleading.
        return context.window_manager.invoke_props_dialog(self, width=320)

    def draw(self, context):
        layout = self.layout
        # Red field whenever the current name maps to an existing
        # preset file or can't be a file name at all; unusable names
        # also get an inline label saying why (execute() refuses them
        # with the same message).
        err = _filename_error(self._sanitized_name())
        row = layout.row()
        row.alert = err is not None or self._collides()
        row.prop(self, "preset_name")
        if err:
            warn = layout.row()
            warn.alert = True
            warn.label(text=err, icon='ERROR')

    def _sanitized_name(self):
        # No character substitution here: names with characters the
        # filesystem can't take are REFUSED with a readable message
        # (_filename_error) instead of silently mangled.
        return (self.preset_name or "").strip() or "Untitled Pose"

    def _collides(self):
        path = os.path.join(_rest_pose_presets_dir_user(),
                            f"{self._sanitized_name()}.json")
        return os.path.exists(path)

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Active object must be an armature")
            return {'CANCELLED'}

        name = self._sanitized_name()
        err = _filename_error(name)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        path = os.path.join(_rest_pose_presets_dir_user(), f"{name}.json")

        # Overwrite confirmation: when the target file already exists
        # and we haven't already been through the confirm step, pop a
        # tiny confirm operator. Its execute() re-invokes this operator
        # with confirmed=True (EXEC_DEFAULT skips the props dialog), so
        # the save proceeds straight through on the second pass.
        if os.path.exists(path) and not self.confirmed:
            bpy.ops.blendcap.save_rest_pose_confirm(
                'INVOKE_DEFAULT', preset_name=name)
            return {'FINISHED'}

        bones_out = {}
        for pb in obj.pose.bones:
            if _is_at_rest(pb):
                continue
            rest_q = pb.bone.matrix_local.to_quaternion()
            # Decompose pb.matrix (full pose in armature space) into
            # rotation, translation, and scale. Translation + scale are
            # captured so manual repositioning / scaling at save time
            # (typical workflow: scaling shoulder + arm bones to fit a
            # taller / shorter target rig before retargeting) survive
            # save→load roundtrips.
            pose_loc, pose_q, pose_scale = pb.matrix.decompose()
            bones_out[pb.name] = {
                "rest": [rest_q.w, rest_q.x, rest_q.y, rest_q.z],
                "pose": [pose_q.w, pose_q.x, pose_q.y, pose_q.z],
                "loc": [pose_loc.x, pose_loc.y, pose_loc.z],
                "scale": [pose_scale.x, pose_scale.y, pose_scale.z],
            }

        if not bones_out:
            self.report({'WARNING'},
                        "No posed bones detected - every bone is at rest. "
                        "Pose at least one bone in Pose Mode before saving.")
            return {'CANCELLED'}

        # Persist the live "Include Location & Scale" checkbox state.
        # On preset selection (see properties.update_custom_rest_pose_
        # preset) the scene prop is restored from this value, so the
        # preset travels with its own intent — preview and bake both
        # read the scene prop and stay consistent.
        include_loc_scale = bool(
            getattr(context.scene,
                    "blendcap_retarget_current_pose_full_matrix", False))
        data = {
            "name": name,
            "format_version": 5,
            "include_loc_scale": include_loc_scale,
            "bones": bones_out,
        }
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.report({'ERROR'}, f"Save failed: {e}")
            return {'CANCELLED'}

        context.scene.blendcap_props.custom_rest_pose_preset = path

        self.report({'INFO'},
                    f"Saved {len(bones_out)} posed bones to "
                    f"rest_pose_presets/{name}.json")
        return {'FINISHED'}


class BLENDCAP_OT_save_rest_pose_confirm(bpy.types.Operator):
    """Tiny "are you sure?" wrapper invoked by BLENDCAP_OT_save_rest_pose
    when the user's chosen filename collides with an existing preset.
    Pops a confirm dialog; on Yes, re-invokes the save operator with
    confirmed=True (EXEC_DEFAULT skips its props dialog and bypasses
    the collision check). On No / dialog dismiss, nothing happens."""
    bl_idname = "blendcap.save_rest_pose_confirm"
    bl_label = "Overwrite Existing Preset?"
    bl_description = ("Replace the existing rest pose preset with the "
                      "same name")
    bl_options = {'REGISTER', 'INTERNAL'}

    preset_name: bpy.props.StringProperty()

    def invoke(self, context, event):
        # Newer Blender (4.1+) accepts message= / title= kwargs on
        # invoke_confirm for a richer prompt; older versions fall back
        # to the operator's bl_label as the question.
        wm = context.window_manager
        msg = f"Overwrite '{self.preset_name}.json'?"
        try:
            return wm.invoke_confirm(self, event, message=msg,
                                     title="Overwrite Existing Preset?")
        except TypeError:
            return wm.invoke_confirm(self, event)

    def execute(self, context):
        bpy.ops.blendcap.save_rest_pose(
            'EXEC_DEFAULT', preset_name=self.preset_name, confirmed=True)
        return {'FINISHED'}


class BLENDCAP_OT_delete_rest_pose_preset(bpy.types.Operator):
    bl_idname = "blendcap.delete_rest_pose_preset"
    bl_label = "Delete Custom Rest Pose Preset"
    bl_description = "Delete the selected custom rest pose preset."
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        props = getattr(context.scene, 'blendcap_props', None)
        if props is None:
            return False
        path = props.custom_rest_pose_preset
        if not path or path.startswith('__'):
            return False
        return _deletable_rest_pose_path(path)

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        props = context.scene.blendcap_props
        path = props.custom_rest_pose_preset
        if not path or path.startswith('__'):
            self.report({'ERROR'}, "No custom preset selected")
            return {'CANCELLED'}
        if not _deletable_rest_pose_path(path):
            self.report({'ERROR'}, "Bundled presets cannot be deleted")
            return {'CANCELLED'}
        if not os.path.isfile(path):
            self.report({'ERROR'}, f"File not found: {path}")
            return {'CANCELLED'}

        try:
            os.remove(path)
        except Exception as e:
            self.report({'ERROR'}, f"Delete failed: {e}")
            return {'CANCELLED'}

        props.custom_rest_pose_preset = '__NONE__'
        self.report({'INFO'},
                    f"Deleted preset: {os.path.basename(path)}")
        return {'FINISHED'}


def _world_rotation_to_pose_basis(pb, target_world_q):
    """Compute the bone-local rotation that, when assigned to
    pb.rotation_quaternion, makes pb.matrix end up with rotation
    target_world_q. Assumes pb.location = (0,0,0), pb.scale = (1,1,1)
    and that the parent has already been driven to its target pose
    (so pb.parent.matrix reflects the parent's posed state).

    Decomposition (in armature-local frame):
        pb.matrix = parent_pose_matrix @ rest_local_offset @ pose_basis
    For rotations:
        target = parent_world_q @ rest_local_offset_q @ pose_basis_q
        pose_basis_q = rest_local_offset_q.inv @ parent_world_q.inv
                       @ target
    """
    if pb.parent is not None:
        parent_world_q = pb.parent.matrix.to_quaternion()
        rest_local_offset_q = (
            pb.parent.bone.matrix_local.inverted() @ pb.bone.matrix_local
        ).to_quaternion()
    else:
        parent_world_q = Quaternion()  # identity
        rest_local_offset_q = pb.bone.matrix_local.to_quaternion()

    return (rest_local_offset_q.inverted()
            @ parent_world_q.inverted()
            @ target_world_q)


# --- Rest pose presets outside retargeting: the library's save dialogs ------
# Save BVH / Save FBX can put a rest pose on an extra first frame. The file
# they write is always BlendCap's own skeleton, so only presets saved from
# that skeleton can be used there.
WRONG_SKELETON_EXPORT = ("This rest pose was made for a different skeleton "
                         "than BlendCap's own, so it cannot be added to the "
                         "file.")
_BLENDCAP_REST = {}      # bone name -> rest orientation of BlendCap's skeleton
_FIT_CACHE = {}          # (path, modified time) -> fits BlendCap's skeleton


def _blendcap_rest_quats():
    """{bone name: rest orientation} of BlendCap's own skeleton, read once
    from the bundled T-POSE preset (every v4+ entry stores the rest it
    was saved over)."""
    if _BLENDCAP_REST:
        return _BLENDCAP_REST
    try:
        path = os.path.join(_rest_pose_presets_dir_bundled(), "T-POSE.json")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for name, entry in data.get("bones", {}).items():
            rest = entry.get("rest") if isinstance(entry, dict) else None
            if rest and len(rest) == 4:
                _BLENDCAP_REST[name] = Quaternion(rest)
    except Exception as e:
        print(f"[BlendCap] bundled T-POSE preset not readable: {e}")
    return _BLENDCAP_REST


def preset_fits_blendcap(path):
    """True when a rest pose preset was saved from BlendCap's own skeleton:
    the same test as skeleton_fit, against the rest orientations the
    bundled T-POSE preset carries, so it needs no rig in the scene. Used
    to build the save dialogs' list; the save judges again on the real
    rig. Presets too old to say (before format 4) do not fit."""
    try:
        key = (os.path.normcase(path), os.path.getmtime(path))
    except OSError:
        return False
    hit = _FIT_CACHE.get(key)
    if hit is not None:
        return hit
    ok = False
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if int(data.get("format_version", 0)) >= 4:
            ref = _blendcap_rest_quats()
            angles = []
            for name, entry in data.get("bones", {}).items():
                short = name[len(_namespace_of(name)):]
                rest = entry.get("rest") if isinstance(entry, dict) else None
                if short in ref and rest and len(rest) == 4:
                    angles.append(_angle_deg(ref[short], Quaternion(rest)))
            if len(angles) >= _SKELETON_MIN_BONES:
                agree = sum(1 for a in angles if a <= _SKELETON_AGREE_DEG)
                ok = agree / len(angles) >= _SKELETON_AGREE_SHARE
    except Exception:
        ok = False
    _FIT_CACHE[key] = ok
    return ok


def pose_rig_from_preset(rig, preset_path):
    """Pose `rig` the way Preview Selected Pose does (rotations only), for
    a caller that keys the result itself. The rig's pose must already be
    cleared. Returns (bones posed, error text or None)."""
    override = load_rest_pose_preset_to_override(preset_path, rig)
    if (LAST_LOAD_STATS or {}).get('wrong_skeleton'):
        return 0, WRONG_SKELETON_EXPORT
    posed = 0
    for pb in _topological_pose_bones(rig):
        if pb.name not in override:
            continue
        q = _world_rotation_to_pose_basis(
            pb, override[pb.name].to_quaternion())
        mode = pb.rotation_mode
        if mode == 'QUATERNION':
            pb.rotation_quaternion = q
        elif mode == 'AXIS_ANGLE':
            axis, angle = q.to_axis_angle()
            pb.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
        else:
            pb.rotation_euler = q.to_euler(mode)
        bpy.context.view_layer.update()
        posed += 1
    return posed, None


class BLENDCAP_OT_preview_rest_pose_preset(bpy.types.Operator):
    bl_idname = "blendcap.preview_rest_pose_preset"
    bl_label = "Preview Selected Pose"
    bl_description = ("Apply the selected preset as a non-destructive "
                      "visible pose on the source rig without keyframing anything.")
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        src = getattr(scene, 'blendcap_retarget_source', None)
        props = getattr(scene, 'blendcap_props', None)
        if not src or src.type != 'ARMATURE' or props is None:
            return False
        path = props.custom_rest_pose_preset
        return bool(path and not path.startswith('__')
                    and os.path.isfile(path))

    def execute(self, context):
        scene = context.scene
        src = scene.blendcap_retarget_source
        props = scene.blendcap_props
        preset_path = props.custom_rest_pose_preset

        if not src or src.type != 'ARMATURE':
            self.report({'ERROR'}, "No source rig selected")
            return {'CANCELLED'}
        if not preset_path or preset_path.startswith('__') \
                or not os.path.isfile(preset_path):
            self.report({'ERROR'}, "No preset selected")
            return {'CANCELLED'}

        # Judge the preset before anything on the rig is touched: the
        # preview below starts by clearing the rig's whole pose.
        try:
            load_rest_pose_preset_to_override(preset_path, src)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load preset: {e}")
            return {'CANCELLED'}
        if (LAST_LOAD_STATS or {}).get('wrong_skeleton'):
            self.report({'ERROR'}, WRONG_SKELETON_PREVIEW)
            return {'CANCELLED'}

        ts = scene.tool_settings
        prev_auto_key = ts.use_keyframe_insert_auto
        ts.use_keyframe_insert_auto = False
        try:
            if context.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            for o in context.view_layer.objects:
                o.select_set(False)
            src.select_set(True)
            context.view_layer.objects.active = src
            bpy.ops.object.mode_set(mode='POSE')

            # Clear every bone's pose (location, rotation, scale) to
            # identity. matrix_basis = Matrix() resets all three at once.
            identity_basis = Matrix()
            for pb in src.pose.bones:
                pb.matrix_basis = identity_basis
            context.view_layer.update()

            # Load the override against this rig (handles cross-rig
            # rest mismatches via the v4 rest-comparison path).
            try:
                override = load_rest_pose_preset_to_override(preset_path, src)
            except Exception as e:
                self.report({'ERROR'}, f"Failed to load preset: {e}")
                return {'CANCELLED'}

            # Preview honors the "Include Location & Scale" scene
            # checkbox uniformly — the same flag the bake reads. On
            # preset selection that scene prop is restored from the
            # JSON's "include_loc_scale" field, so each preset reasserts
            # its intent and preview/bake never disagree.
            use_full_pose = bool(
                getattr(scene,
                        "blendcap_retarget_current_pose_full_matrix",
                        False))

            applied = 0
            for pb in _topological_pose_bones(src):
                if pb.name not in override:
                    continue

                if use_full_pose:
                    # Apply the full saved arm-local matrix. Children
                    # follow naturally — intentional for scale /
                    # reposition presets where the user wants the
                    # downstream chain to reflect those changes.
                    pb.matrix = override[pb.name]
                else:
                    # Rotation-only path (v4 behavior). Setting only
                    # pb.rotation_* (not pb.matrix) leaves pb.location
                    # at (0,0,0) so children stay attached to FK-derived
                    # parent positions instead of getting pulled to a
                    # baked-in translation that doesn't fit their
                    # current FK parent.
                    target_world_q = override[pb.name].to_quaternion()
                    pose_basis_q = _world_rotation_to_pose_basis(
                        pb, target_world_q)
                    # Write to whichever rotation property is active —
                    # DO NOT change rotation_mode (BVH rigs use Euler;
                    # the action's fcurves target rotation_euler, and
                    # flipping to QUATERNION would freeze the preview).
                    mode = pb.rotation_mode
                    if mode == 'QUATERNION':
                        pb.rotation_quaternion = pose_basis_q
                    elif mode == 'AXIS_ANGLE':
                        axis, angle = pose_basis_q.to_axis_angle()
                        pb.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
                    else:
                        pb.rotation_euler = pose_basis_q.to_euler(mode)
                context.view_layer.update()
                applied += 1
        finally:
            ts.use_keyframe_insert_auto = prev_auto_key

        stats = LAST_LOAD_STATS or {}
        total = stats.get('total', 0)
        if applied == 0:
            self.report({'WARNING'},
                        f"0 of {total} bones in this preset match the "
                        f"source rig. Nothing to preview.")
            return {'CANCELLED'}

        # Say so when only part of the preset landed, or when it was
        # matched under another namespace; a full same-name match keeps
        # the short message.
        note = match_summary(stats)
        self.report({'INFO'},
                    f"Previewing {applied} bones from "
                    f"{os.path.basename(preset_path)}"
                    + (f" ({note})" if note else "")
                    + " - scrub the timeline to dismiss.")
        return {'FINISHED'}
