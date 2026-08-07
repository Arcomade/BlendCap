"""PropertyGroups, dynamic enum/update callbacks, BVH library sync.

The BVH library sync timer (_sync_bvh_library + _bvh_library_timer)
lives here too because it's mutating the BlendCapBVHEntry collection
defined here.
"""
import bpy
import json
import math
import os

import mathutils
import numpy as np

from . import preview_ground
from .helpers import _cache_dir, _bvh_library_dir, _strip_capture_prefix, _tag_redraw


# ============================================================
# DYNAMIC ENUM CALLBACKS
# ============================================================
# Dynamic-enum item lists (and their strings) must stay alive across
# draws — Blender reads them by reference, and a garbage-collected list
# shows up as garbled menu text or a crash. Same pattern as
# _REST_POSE_PRESET_ITEMS_CACHE below: per-enum module-level cache,
# rebuilt in place on every call.
_BVH_ITEMS_CACHE = []
_FACE_DATA_ITEMS_CACHE = []
_ARKIT_FACE_DATA_ITEMS_CACHE = []


def _face_cache_folders():
    """(folder, label) pairs for every cache folder holding face data."""
    cd = _cache_dir()
    out = []
    if os.path.exists(cd):
        for folder in os.listdir(cd):
            fp = os.path.join(cd, folder)
            if os.path.isdir(fp) and folder != "bvh_exports" and os.path.exists(os.path.join(fp, "face_frames.npz")):
                out.append((folder, _strip_capture_prefix(folder)))
    return out


def get_bvh_items(self, context):
    # Identify BlendCap rigs by the provenance custom prop set during
    # import (mocap.py / operators_library.py) rather than by name —
    # the rig is freely renameable and the prefix-based naming
    # convention was dropped.
    rigs = [o for o in context.scene.objects
            if o.type == 'ARMATURE' and "blendcap_source_video" in o]
    if not rigs:
        items = [('NONE', "No BVH Data Found", "")]
    else:
        items = [(o.name, o.name, "") for o in rigs]
    _BVH_ITEMS_CACHE.clear()
    _BVH_ITEMS_CACHE.extend(items)
    return _BVH_ITEMS_CACHE


def get_face_data_items(self, context):
    items = [(folder, label, f"Use face data from {label}")
             for folder, label in _face_cache_folders()]
    items = items or [('NONE', "No Face Data Found", "Run Face Tracking first")]
    _FACE_DATA_ITEMS_CACHE.clear()
    _FACE_DATA_ITEMS_CACHE.extend(items)
    return _FACE_DATA_ITEMS_CACHE


def get_arkit_face_data_items(self, context):
    """Same as get_face_data_items, but with an AUTO option at the top
    that resolves to the cache folder for the active video."""
    items = [('AUTO', "Auto (active video)",
              "Use the face cache that matches the active video source")]
    items += [(folder, label, f"Use face data from {label}")
              for folder, label in _face_cache_folders()]
    _ARKIT_FACE_DATA_ITEMS_CACHE.clear()
    _ARKIT_FACE_DATA_ITEMS_CACHE.extend(items)
    return _ARKIT_FACE_DATA_ITEMS_CACHE


# Rest-pose preset dropdown items: every JSON in the addon's
# rest_pose_presets/ user directory. The directory is created on demand by
# the save operator. Items list and its strings must stay alive across draws
# (Blender reads them by reference); cache at module scope.
_REST_POSE_PRESET_ITEMS_CACHE = []

def _rest_pose_presets_dir_user():
    """Rest-pose presets directory inside the addon folder.

    Sits next to retarget_maps/ in the addon directory, matching how
    saved presets were located in pre-refactor versions of the tool.
    The "_user" suffix is a holdover from a brief detour through
    Blender's user config dir; kept here so existing callers and the
    delete-operator's path-protection check don't need to ripple.
    """
    from .helpers import _addon_dir
    d = os.path.join(_addon_dir(), "rest_pose_presets")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def _rest_pose_presets_dir():
    """Back-compat alias used by callers that just need a writable dir."""
    return _rest_pose_presets_dir_user()


def get_rest_pose_preset_items(self, context):
    items = [('__NONE__', "(none — use source rig's edit-bone rest)",
              "Don't apply a custom rest pose. Equivalent to leaving "
              "'Use Custom Rest Pose' unchecked.")]

    user_dir = _rest_pose_presets_dir_user()
    if os.path.isdir(user_dir):
        # Walk subfolders so the user can organize presets into nested
        # folders. Save path is unchanged — presets still write into the
        # top-level folder.
        for root, dirs, files in os.walk(user_dir):
            dirs.sort()
            files.sort()
            for fn in files:
                if not fn.endswith('.json'):
                    continue
                path = os.path.join(root, fn)
                display = os.path.splitext(fn)[0]
                rel = os.path.relpath(path, user_dir).replace("\\", "/")
                items.append((path, display,
                              f"Saved rest-pose preset: {rel}"))

    if len(items) == 1:
        items.append(('__EMPTY_RP__', "(no presets saved yet)",
                      "Pose your source rig and click 'Save Current "
                      "Pose as Preset'."))

    _REST_POSE_PRESET_ITEMS_CACHE.clear()
    _REST_POSE_PRESET_ITEMS_CACHE.extend(items)
    return _REST_POSE_PRESET_ITEMS_CACHE

# ============================================================
# PROPERTY UPDATE CALLBACKS
# ============================================================
def update_export_face(self, context):
    if self.export_face and self.use_separate_face:
        self.use_separate_face = False

def update_use_separate_face(self, context):
    if self.use_separate_face and self.export_face:
        self.export_face = False



# Set True while the "Default Video Source" add-on pref is being applied to a
# fresh scene (registration.py's _blendcap_apply_default_video_source). It lets
# update_video_source skip the NOTIFY warning dialog: a programmatic default is
# not an interactive source switch, and a new scene has render.use_sequencer ON
# by default, so a VSE default would otherwise pop the dialog on every
# new/startup file. AUTO_DISABLE is NOT skipped - it's silent, so the
# no-dialog-spam rationale doesn't apply, and it's how a VSE-default user gets
# new files' Sequencer export turned off without a startup-file edit.
_SETTING_DEFAULT_SOURCE = False


def update_video_source(self, context):
    """When the user switches Source from File Path -> Video Editor Strip
    and Output > Post Processing > Sequencer is currently on, pop a
    warning dialog asking whether to disable it. Reason: with the
    Sequencer checkbox enabled, Blender's render output is whatever the
    VSE is showing — so if the user kicks off a render of the actual
    Blender scene, Blender silently re-encodes their source video clips
    instead. The trap is most likely to bite right after this exact
    source switch.

    Governed by the "Sequencer Render Warning" add-on pref
    (sequencer_export_behavior): NOTIFY (default) pops the dialog once per
    session (blendcap_state['sequencer_warned'], cleared on reload /
    restart); AUTO_DISABLE turns the Sequencer toggle off silently; LEAVE
    does nothing.

    Also fires when the "Default Source" pref applies VSE to a new file
    (_SETTING_DEFAULT_SOURCE): AUTO_DISABLE still runs there — it's silent,
    and a user who sets both prefs expects new files covered — but the
    NOTIFY dialog is skipped, so a VSE default never pops a warning on
    every File > New.
    """
    if self.video_source != 'VSE':
        return
    # The scene that owns this property group — NOT context.scene, which
    # can be a different scene when the default-source handler loops over
    # bpy.data.scenes at load time.
    scene = self.id_data
    if not getattr(scene.render, 'use_sequencer', False):
        return
    from .helpers import blendcap_state, addon_prefs
    prefs = addon_prefs(context)
    behavior = (getattr(prefs, "sequencer_export_behavior", 'NOTIFY')
                if prefs else 'NOTIFY')
    if behavior == 'LEAVE':
        # User opted out: don't warn, don't touch the Sequencer toggle.
        return
    if behavior == 'AUTO_DISABLE':
        # Silently turn it off, no dialog. Runs for interactive switches
        # AND the default-source apply on new files.
        scene.render.use_sequencer = False
        return
    # NOTIFY from here down. A programmatic default apply is not an
    # interactive switch - don't pop the warning dialog for it.
    if _SETTING_DEFAULT_SOURCE:
        return
    # NOTIFY (default): warn once per session. DEFER the dialog out of this
    # update callback. Launching invoke_props_dialog synchronously here fires
    # mid-click on the Source enum (drawn expand=True), so the button's drag
    # interaction never finalizes - after dismissing the dialog, sliding the
    # mouse back over "File Path" silently flips the selection back (the most
    # natural mouse direction, so it bit constantly). A one-shot timer lets
    # the click commit first, then pops the dialog. sequencer_warn_pending
    # coalesces rapid re-toggles into a single dialog.
    if (blendcap_state.get('sequencer_warned')
            or blendcap_state.get('sequencer_warn_pending')):
        return
    blendcap_state['sequencer_warn_pending'] = True

    def _deferred_sequencer_warning():
        blendcap_state['sequencer_warn_pending'] = False
        try:
            bpy.ops.blendcap.sequencer_warning('INVOKE_DEFAULT')
        except RuntimeError:
            # Context can be invalid (e.g. a script-driven change); skip.
            pass
        return None  # one-shot

    bpy.app.timers.register(_deferred_sequencer_warning, first_interval=0.05)


def update_custom_rest_pose_preset(self, context):
    """When the user picks a preset from the dropdown, restore the
    "Include Location & Scale" scene checkbox to whatever the JSON
    stored. Single source of truth at runtime is the scene prop —
    preview and bake both read it, so each preset carries and reasserts
    its own loc/scale intent on selection.

    Legacy v5 files without the field default to True (matches their
    historical preview behavior, which always applied loc/scale)."""
    path = getattr(self, "custom_rest_pose_preset", None)
    if not path or path.startswith('__'):
        return
    if not os.path.isfile(path):
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return
    include = bool(data.get("include_loc_scale", True))
    scene = context.scene
    if getattr(scene, "blendcap_retarget_current_pose_full_matrix",
               None) != include:
        scene.blendcap_retarget_current_pose_full_matrix = include


def update_use_custom_rest_pose(self, context):
    """Mutual exclusion with retargeting's 'Use Current Source Pose as Rest'.

    Both toggles populate `source_rest_override` at retarget time but from
    different sources (live pose vs saved JSON). Only one can be active.
    """
    if self.use_custom_rest_pose:
        scene = context.scene
        if getattr(scene, 'blendcap_retarget_use_current_pose_as_rest', False):
            scene.blendcap_retarget_use_current_pose_as_rest = False


def update_use_current_pose_as_rest(scene, context):
    """Inverse of update_use_custom_rest_pose. Lives at scene level because
    blendcap_retarget_use_current_pose_as_rest is a Scene property; its
    update callback receives the Scene as `self`."""
    if scene.blendcap_retarget_use_current_pose_as_rest:
        props = getattr(scene, 'blendcap_props', None)
        if props is not None and props.use_custom_rest_pose:
            props.use_custom_rest_pose = False


# Set True between undo_pre/undo_post (and redo_pre/redo_post) by the
# handlers in registration.py. Blender re-fires every update callback
# while restoring state during an undo; without this gate, each callback
# pushes a phantom step on top of the restoration and the user has to
# Ctrl+Z 3-4 times to walk back through them before any visible change.
_UNDO_IN_PROGRESS = False

# Set True by operators that bulk-mutate the bone-pair table inside a
# single REGISTER+UNDO action (Auto-Match, Add All Source Bones). The
# operator itself already creates one undo step covering the whole
# action — without this flag, each per-field write inside the operator
# would push its OWN step from _mark_dirty_dispatch, producing N+1
# steps for one user-visible action. Parallel to retarget._LOADING_
# PRESET, which gates the preset-load path for the same reason.
_BULK_EDITING = False

# Tracks the wall-clock time of the most recent undo_push from this
# dispatch. Some Blender property types (notably EnumProperty) fire the
# update callback twice for a single commit — once when the dropdown
# opens, once when the user picks. Without coalescing, that would push
# TWO undo steps per edit (the 2-Ctrl+Z-per-edit symptom). 100ms
# threshold: tight enough that two intentionally separate edits won't
# coalesce (humans can't tab between fields that fast), loose enough to
# absorb double-fires reliably.
_LAST_PUSH_TIME = 0.0
_PUSH_COALESCE_SECONDS = 0.1


def _smart_strip_pair_field(self, context, side):
    """Smart-strip the just-assigned pair field against the relevant
    rig+prefix. `side` is 'source' or 'target'.

    Rule (see name_matching._smart_strip docstring): if the value starts
    with the side's prefix and the stripped form does NOT also exist as
    a distinct bone on the rig, rewrite the field to the stripped form.
    This way the table always stores SHORT names, and editing the prefix
    field re-targets every pair at once (the prefix prepends at bake
    time via _resolve_against_rig).

    The actual re-assignment is DEFERRED to the next event tick via
    bpy.app.timers. Why: pair cells are drawn with `prop_search` in the
    UIList. When the user picks a name from prop_search's dropdown,
    Blender's UI commits the dropdown's selected value into the
    StringProperty AFTER our update callback returns — overwriting any
    same-tick re-assignment we do. Running the strip from a 0-delay
    timer lets prop_search's commit happen first, then we strip cleanly.

    Guards: skipped during preset load (_LOADING_PRESET), undo/redo
    (_UNDO_IN_PROGRESS), and bulk operators (_BULK_EDITING). The
    deferred re-assignment also gates itself with _BULK_EDITING so the
    recursive update doesn't push an extra undo step or re-fire
    mark_dirty.
    """
    from . import retarget
    if retarget._LOADING_PRESET or _UNDO_IN_PROGRESS or _BULK_EDITING:
        return
    _schedule_deferred_strip(self, side)


def _schedule_deferred_strip(pair, side):
    """Register a 0-delay timer that runs the smart-strip after the
    current UI event finishes. Captures `pair` and `side` in a closure
    so the timer can locate the right PropertyGroup to update."""
    import bpy

    def _run():
        from . import retarget
        from .retarget.name_matching import _smart_strip
        global _BULK_EDITING
        try:
            scene = bpy.context.scene
            if scene is None:
                return None
            if retarget._LOADING_PRESET or _UNDO_IN_PROGRESS or _BULK_EDITING:
                return None
            if side == 'source':
                name = pair.source
                rig = scene.blendcap_retarget_source
                prefix = scene.blendcap_retarget_source_prefix or ""
            else:
                name = pair.target
                rig = scene.blendcap_retarget_target
                prefix = scene.blendcap_retarget_namespace_strip or ""
            stripped = _smart_strip(name, rig, prefix)
            if stripped == name:
                return None
            prev = _BULK_EDITING
            _BULK_EDITING = True
            try:
                if side == 'source':
                    pair.source = stripped
                else:
                    pair.target = stripped
            finally:
                _BULK_EDITING = prev
        except (ReferenceError, AttributeError):
            # Pair PropertyGroup was freed between scheduling and now
            # (pairs.clear / preset load / undo). Drop silently.
            pass
        return None  # one-shot

    bpy.app.timers.register(_run, first_interval=0.0)


def _on_pair_source_change(self, context):
    _smart_strip_pair_field(self, context, 'source')
    _mark_dirty_dispatch(self, context)


def _on_pair_target_change(self, context):
    _smart_strip_pair_field(self, context, 'target')
    _mark_dirty_dispatch(self, context)


def _push_field_undo_step(message):
    """Force a discrete undo step for the just-committed field edit.

    Direct PropertyGroup / Scene-prop writes via layout.prop() and
    prop_search() do NOT consistently create undo steps of their own
    on the Blender versions we ship to — Blender batches every edit
    since the last operator-bounded undo entry into one step, so a
    single Ctrl+Z wipes an entire mapping session and Redo is lossy.
    bpy.ops.ed.undo_push() forces per-edit granularity.

    Shared by the pair-table dispatch AND every scene-level
    bone-selection field (head bone, prefix, custom IK source bones)
    so all of them behave the same under Ctrl+Z.

    Guards:
      - retarget._LOADING_PRESET: preset-load writes many fields in one
        shot inside an operator that already creates its own step.
      - _UNDO_IN_PROGRESS: skip while Blender restores state during
        an undo/redo, otherwise we pollute the stack with phantom steps.
      - _BULK_EDITING: skip while an operator (Auto-Match, Add All, the
        prefix-restrip sweep) is bulk-mutating fields under its own
        operator-bounded undo step.
      - _LAST_PUSH_TIME / _PUSH_COALESCE_SECONDS: skip if another push
        happened within the coalesce window. Handles EnumProperty
        dropdowns that fire the update callback twice per commit."""
    global _LAST_PUSH_TIME
    from . import retarget
    if retarget._LOADING_PRESET or _UNDO_IN_PROGRESS or _BULK_EDITING:
        return
    import time
    now = time.monotonic()
    if now - _LAST_PUSH_TIME < _PUSH_COALESCE_SECONDS:
        return
    _LAST_PUSH_TIME = now
    try:
        bpy.ops.ed.undo_push(message=message)
    except Exception:
        # undo_push can refuse during certain contexts (e.g. modal
        # operator running). Swallowing is safe: the worst case is
        # we miss one step, not a crash.
        pass


def _mark_dirty_dispatch(self, context):
    """Pair-field update callback. Dirties the preset and forces a
    per-edit undo step. After the package split mark_dirty lives in
    blendcap.retarget; the lazy import here avoids any module-load
    ordering coupling.

    The _LOADING_PRESET / _UNDO_IN_PROGRESS / _BULK_EDITING guards live
    in _push_field_undo_step (and inside retarget.mark_dirty for the
    dirty mark itself), so a single guarded call covers both halves."""
    from . import retarget
    if retarget._LOADING_PRESET or _UNDO_IN_PROGRESS or _BULK_EDITING:
        return
    retarget.mark_dirty(self, context)
    _push_field_undo_step("Edit bone pair")


# ============================================================
# PER-ARMATURE LIVE CAMERA OFFSET
# ============================================================
# Each imported BlendCap BVH armature carries its own pitch/roll
# offset values via a PropertyGroup attached to the Object. Update
# callbacks rewrite the synthetic Root bone's per-frame rotation
# keyframes so the entire skeleton tilts in world space — without
# touching the rest pose (bone.matrix_local stays unchanged, so
# retarget bakes still see the natural rest geometry).
#
# Why Root and not Hips: the synthetic Root is identity-keyframed at
# every frame (insert_synthetic_root in pipeline/body_skeleton.py).
# Overwriting those keys with a constant value is a clean operation
# that doesn't disturb any other animation. Root inheritance then
# propagates the tilt to Hips and everything below, mirroring exactly
# what the BVH-time bake does to global_rots inside compute_local_
# rotations (the rotation only ends up on the root joint's local
# rotation; non-root locals are unchanged).
#
# Math: Root's pose-world rotation = M @ basis, where M = armature_
# world_rot @ Root.bone.matrix_local_rot. We want pose-world = R @ M
# (R applied as a left-multiply in world). Solve: basis = M.inv @ R
# @ M (a conjugation). Same conjugated quaternion is written into
# every Root rotation keyframe.

def _apply_camera_offset_to_rig(rig):
    """Bake the rig's pitch/roll offset into Root's rotation keys.

    Conjugates a world-space pitch+roll rotation through Root's rest
    matrix and overwrites every keyframe in Root's rotation_quaternion
    fcurves. Idempotent — calling twice with the same offset writes the
    same value, calling with offset=0 restores identity (which is the
    synthetic Root's natural state).

    Gates: rig must be an armature with a "Root" pose bone and a non-
    None action. Silently no-op otherwise.
    """
    if rig is None or rig.type != 'ARMATURE':
        return
    root_pb = rig.pose.bones.get('Root')
    if root_pb is None:
        return
    action = (rig.animation_data.action
              if rig.animation_data else None)
    if action is None:
        return

    offset = rig.blendcap_camera_offset
    pitch_rad = math.radians(offset.pitch_deg)
    roll_rad = math.radians(offset.roll_deg)

    # World-space delta R = R_y(roll) @ R_x(pitch).
    # Axis choice: Blender's scene is Z-up. After BVH import with
    # axis_up='Y', the rig's vertical axis in Blender's world maps to
    # Z. So:
    #   - Pitch about world X (side axis): tilts the body forward/back.
    #   - Roll about world Y (forward axis): tips the body left/right.
    #   - Yaw about world Z (vertical) — not exposed.
    # Earlier iteration used Z for roll, which spun the rig around its
    # vertical axis (yaw) instead of rolling it. The BVH-time bake in
    # pipeline/camera_level.py works in BVH world frame (Y-up, Z-back)
    # where roll is correctly about Z (the forward axis there); only
    # the Blender-side mapping needed updating.
    R_q = (mathutils.Quaternion((0.0, 1.0, 0.0), roll_rad)
           @ mathutils.Quaternion((1.0, 0.0, 0.0), pitch_rad))

    # Conjugation matrix M = armature_world_rot @ Root.matrix_local_rot.
    # Reads the rig's CURRENT object rotation so user-positioned rigs
    # still get a correct world-space tilt.
    arm_q = rig.matrix_world.to_quaternion()
    rest_q = root_pb.bone.matrix_local.to_quaternion()
    M_q = arm_q @ rest_q
    basis_q = M_q.inverted() @ R_q @ M_q

    # Find the four rotation_quaternion fcurves for Root.
    bone_path = 'pose.bones["Root"].rotation_quaternion'
    from .retarget import _iter_action_fcurves
    fcurves_by_index = {}
    for fc in _iter_action_fcurves(action):
        if fc.data_path == bone_path:
            fcurves_by_index[fc.array_index] = fc

    # If Root isn't keyframed yet (rare — synthetic Root is always
    # keyframed by the BVH writer, but be defensive), prime one
    # keyframe per channel at frame_start so the fcurves exist.
    if len(fcurves_by_index) < 4:
        fr_start = int(action.frame_range[0])
        root_pb.rotation_mode = 'QUATERNION'
        root_pb.keyframe_insert('rotation_quaternion', frame=fr_start)
        fcurves_by_index = {}
        for fc in _iter_action_fcurves(action):
            if fc.data_path == bone_path:
                fcurves_by_index[fc.array_index] = fc
        if len(fcurves_by_index) < 4:
            return

    # Use the existing keyframe FRAMES (preserve density / sparseness)
    # — only the VALUES change. Cheaper than rewriting frame indices,
    # and preserves any keys the user may have manually moved.
    components = (basis_q.w, basis_q.x, basis_q.y, basis_q.z)
    for i, val in enumerate(components):
        fc = fcurves_by_index[i]
        n = len(fc.keyframe_points)
        if n == 0:
            continue
        # Read existing co (frame, value pairs), overwrite values.
        flat = [0.0] * (2 * n)
        fc.keyframe_points.foreach_get('co', flat)
        for k in range(n):
            flat[2 * k + 1] = val
        fc.keyframe_points.foreach_set('co', flat)
        fc.update()


def update_camera_offset(self, context):
    """PropertyGroup update callback — fires when pitch/roll changes.

    `self` is the BlendCapCameraOffset PropertyGroup instance; its
    parent Object is the rig (`self.id_data`). Re-applies the offset to
    Root's rotation keys so the viewport reflects the new value live.
    """
    rig = self.id_data
    _apply_camera_offset_to_rig(rig)
    # Keep the preview honest: re-ground the lowest foot sample so the
    # viewport shows what the bake will produce (see the auto-grounding
    # block below _apply_height_offset_to_rig).
    _apply_autoground_to_rig(rig)


def _apply_height_offset_to_rig(rig):
    """Apply the slider's height offset as a world-Z translation on
    Hips, written into Hips' location keyframes.

    Modifies Hips' (X, Y, Z) basis location channels — NOT just Y.
    Reason: Hips' bone-local Y axis isn't necessarily aligned with
    world Z. The bone's tail direction is derived from its first child
    in the BVH (varies by skeleton), and any pitch/roll on Root
    rotates the bone-local frame further. To produce a clean vertical
    translation in world space we compute the basis-frame direction
    that maps to world +Z and write the delta into each channel
    accordingly.

    Math: Hips' world position depends on basis location through
        delta_world = parent_world.to_3x3() @ rel_rest.to_3x3()
                       @ basis_loc
    where parent_world = rig_world @ Root.matrix_local @ Root.basis,
    and rel_rest = Root.matrix_local.inv @ Hips.matrix_local. The
    basis_loc is applied to the bone's ORIGIN (0,0,0), so Hips'
    basis_rotation animation drops out of the translation formula.
    That makes the (parent_world @ rel_rest).to_3x3() product
    constant per frame (Root's basis rotation is constant — we set
    every Root keyframe to the same conjugated quaternion in
    _apply_camera_offset_to_rig). One delta vector, applied to all
    keyframes on all three channels.

    Why bone keyframes and not rig.location.z: the per-rig live
    preview also drives retarget. The retarget bake reads source
    bones via pb.location (basis location = keyframed value) — it
    does NOT see rig.location. So an object-Z translation gets the
    source rig visually right but doesn't propagate to the target on
    Apply Retargeting. Modifying Hips position keyframes makes the
    height offset real animation data, which retargets via the
    standard LOC pair path.

    Bake makes this permanent: the bake operator passes the cumulative
    total (previously baked + slider) to the CLI as --height-offset,
    which modes.py adds to the Hips channel after grounding and
    footskate cleanup. The re-imported rig records the total in
    _blendcap_baked_height_m and its height_offset_m defaults to 0 —
    slider back to neutral, same lifecycle as pitch/roll.

    Tracking via _blendcap_height_offset_applied_m custom prop: each
    slider change applies (new − applied) to all Hips location keys
    and updates applied. Slider 0 → 0.5 → 0 restores the original
    keys exactly.
    """
    if rig is None or rig.type != 'ARMATURE':
        return
    new_value = float(rig.blendcap_camera_offset.height_offset_m)
    applied = float(rig.get("_blendcap_height_offset_applied_m", 0.0))
    delta = new_value - applied
    if abs(delta) < 1e-9:
        return
    if _shift_hips_keys_world_z(rig, delta):
        rig["_blendcap_height_offset_applied_m"] = new_value


def _shift_hips_keys_world_z(rig, delta):
    """Add a world-Z translation to Hips' location keyframes.

    Shared low-level writer for the user Height slider and the
    rotation-preview auto-grounding — each caller does its own
    bookkeeping of what it has applied. Returns True on success.
    """
    hips_pb = rig.pose.bones.get('Hips')
    root_pb = rig.pose.bones.get('Root')
    if hips_pb is None or root_pb is None:
        return False
    action = (rig.animation_data.action
              if rig.animation_data else None)
    if action is None:
        return False

    # Compute the basis-space direction that produces +Z world
    # translation. parent_world rotation = rig_world_rot @
    # Root.matrix_local_rot @ Root.basis_rot. Root's basis rotation
    # mirrors what _apply_camera_offset_to_rig wrote (conjugation of
    # the pitch/roll world rotation through Root's rest), recomputed
    # from the slider values rather than read back from keyframes
    # (avoids depsgraph dependency).
    pitch_rad = math.radians(rig.blendcap_camera_offset.pitch_deg)
    roll_rad = math.radians(rig.blendcap_camera_offset.roll_deg)
    R_q_world = (mathutils.Quaternion((0.0, 1.0, 0.0), roll_rad)
                 @ mathutils.Quaternion((1.0, 0.0, 0.0), pitch_rad))
    arm_q = rig.matrix_world.to_quaternion()
    root_rest_q = root_pb.bone.matrix_local.to_quaternion()
    M_root = arm_q @ root_rest_q
    root_basis_q = M_root.inverted() @ R_q_world @ M_root

    # Hips' rest relative to Root.
    rel_rest_mat = (root_pb.bone.matrix_local.inverted()
                    @ hips_pb.bone.matrix_local)
    rel_rest_q = rel_rest_mat.to_quaternion()

    # Total rotation taking a Hips-basis vector to world space (for
    # translation purposes — Hips' basis rotation drops out because
    # basis_loc is applied to the bone origin).
    hips_to_world_q = arm_q @ root_rest_q @ root_basis_q @ rel_rest_q
    # Direction in Hips basis frame whose world image is +Z:
    basis_dir_per_unit_z = hips_to_world_q.inverted() @ mathutils.Vector(
        (0.0, 0.0, 1.0))

    delta_basis = basis_dir_per_unit_z * delta

    # Find / prime fcurves for all three location channels.
    bone_path = 'pose.bones["Hips"].location'
    from .retarget import _iter_action_fcurves
    fcurves = [None, None, None]
    for fc in _iter_action_fcurves(action):
        if fc.data_path == bone_path and 0 <= fc.array_index < 3:
            fcurves[fc.array_index] = fc

    if any(fc is None for fc in fcurves):
        fr_start = int(action.frame_range[0])
        for i in range(3):
            if fcurves[i] is None:
                hips_pb.keyframe_insert('location', index=i,
                                        frame=fr_start)
        for fc in _iter_action_fcurves(action):
            if fc.data_path == bone_path and 0 <= fc.array_index < 3:
                fcurves[fc.array_index] = fc
        if any(fc is None for fc in fcurves):
            return False

    # Add delta_basis components to keys on each channel. Skip channels
    # whose delta is essentially zero (cheaper, and keeps the action
    # cleaner in the common case where roll=0 and the delta lives
    # mostly on one axis).
    components = (delta_basis.x, delta_basis.y, delta_basis.z)
    for i, comp in enumerate(components):
        if abs(comp) < 1e-9:
            continue
        fc = fcurves[i]
        n = len(fc.keyframe_points)
        if n == 0:
            continue
        flat = [0.0] * (2 * n)
        fc.keyframe_points.foreach_get('co', flat)
        for k in range(n):
            flat[2 * k + 1] += comp
        fc.keyframe_points.foreach_set('co', flat)
        fc.update()
    return True


def update_height_offset(self, context):
    """PropertyGroup update callback for height_offset_m."""
    _apply_height_offset_to_rig(self.id_data)


# ============================================================
# Rotation-preview auto-grounding (feet back on the floor)
# ============================================================
# The pitch/roll live preview rotates the performance about Root,
# which visually lifts or sinks the body — but the BAKED result
# re-grounds itself (grounding + foot locking run after the
# rotation), so the preview would lie about height and bait users
# into "fixing" it with the Height slider, which then double-applies
# after a bake. After every rotation update we compute where the
# lowest foot sample landed and write a compensating world-Z delta
# into the Hips keys, tracked SEPARATELY from the user's Height:
#   _blendcap_height_offset_applied_m — the user's slider
#   _blendcap_autoground_applied_m    — this compensation
# Bake passes only the user's value to the CLI (the baked output
# re-grounds on its own), and a bake's re-import replaces the live
# keys wholesale, so the compensation never leaks into files.
#
# Foot world positions across the clip don't depend on the sliders,
# so they're FK-computed once per (rig, action) from the keyframe
# arrays (BVH actions key every frame — bulk-readable) and cached;
# a slider tick then only rotates the cached points. Math lives in
# preview_ground.py (bpy-free, unit-tested).

_AUTOGROUND_CACHE = {}

_AUTOGROUND_CHAINS = (
    ("Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe"),
    ("Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToe"),
)


def _autoground_read_channel(action, path, index, frames):
    """One fcurve's values sampled on the frame grid, or None."""
    from .retarget import _iter_action_fcurves
    fc = None
    for cand in _iter_action_fcurves(action):
        if cand.data_path == path and cand.array_index == index:
            fc = cand
            break
    if fc is None or len(fc.keyframe_points) == 0:
        return None
    n = len(fc.keyframe_points)
    flat = [0.0] * (2 * n)
    fc.keyframe_points.foreach_get('co', flat)
    co = np.asarray(flat, dtype=np.float64).reshape(n, 2)
    if n == frames.shape[0] and np.array_equal(co[:, 0], frames):
        return co[:, 1]
    # Sparse/offset keys: linear resample (defensive — BVH keys are
    # dense, so this path is rare and the approximation harmless).
    return np.interp(frames, co[:, 0], co[:, 1])


def _autoground_rot_mats(rig, action, bone_name, frames):
    """Per-frame local rotations for one bone, or None for rest."""
    n = frames.shape[0]
    quat_path = f'pose.bones["{bone_name}"].rotation_quaternion'
    chans = [_autoground_read_channel(action, quat_path, i, frames)
             for i in range(4)]
    if any(c is not None for c in chans):
        defaults = list(rig.pose.bones[bone_name].rotation_quaternion)
        cols = [c if c is not None else np.full(n, defaults[i])
                for i, c in enumerate(chans)]
        return preview_ground.quats_to_mats(np.stack(cols, axis=1))
    eul_path = f'pose.bones["{bone_name}"].rotation_euler'
    chans = [_autoground_read_channel(action, eul_path, i, frames)
             for i in range(3)]
    if any(c is not None for c in chans):
        pb = rig.pose.bones[bone_name]
        order = pb.rotation_mode
        if order not in ('XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX'):
            order = 'XYZ'
        defaults = list(pb.rotation_euler)
        cols = [c if c is not None else np.full(n, defaults[i])
                for i, c in enumerate(chans)]
        return preview_ground.euler_to_mats(np.stack(cols, axis=1),
                                            order)
    return None


def _autoground_loc(action, bone_name, frames):
    path = f'pose.bones["{bone_name}"].location'
    chans = [_autoground_read_channel(action, path, i, frames)
             for i in range(3)]
    if not any(c is not None for c in chans):
        return None
    n = frames.shape[0]
    return np.stack([c if c is not None else np.zeros(n)
                     for c in chans], axis=1)


def _autoground_cache_for(rig):
    """Fetch or build the foot-sample cache for this rig's action."""
    action = (rig.animation_data.action if rig.animation_data else None)
    if action is None:
        return None
    fr0, fr1 = (int(action.frame_range[0]), int(action.frame_range[1]))
    key = (rig.name, action.name, fr0, fr1)
    cached = _AUTOGROUND_CACHE.get(key)
    if cached is not None:
        return cached

    bones = rig.pose.bones
    needed = {b for chain in _AUTOGROUND_CHAINS for b in chain}
    needed.add("Root")
    if any(bones.get(b) is None for b in needed):
        return None
    frames = np.arange(fr0, fr1 + 1, dtype=np.float64)

    arm = np.asarray(rig.matrix_world, dtype=np.float64)
    rest = {b: np.asarray(bones[b].bone.matrix_local, dtype=np.float64)
            for b in needed}
    m_world = arm @ rest["Root"]
    c_rot = m_world[:3, :3]

    root_loc = _autoground_loc(action, "Root", frames)
    if root_loc is None:
        root_loc = np.zeros((frames.shape[0], 3))
    pivot = m_world[:3, 3] + np.einsum('ij,nj->ni', c_rot, root_loc)
    pivot_z = np.ascontiguousarray(pivot[:, 2])

    hips_loc = _autoground_loc(action, "Hips", frames)
    pts = []
    for chain in _AUTOGROUND_CHAINS:
        rest_rel = []
        rots = []
        locs = []
        parent_rest = rest["Root"]
        for b in chain:
            rest_rel.append(np.linalg.inv(parent_rest) @ rest[b])
            parent_rest = rest[b]
            rots.append(_autoground_rot_mats(rig, action, b, frames))
            locs.append(hips_loc if b == "Hips" else None)
        try:
            # Foot + Toe heads = the pipeline's ankle/toe ground points
            heads = preview_ground.fk_chain_heads(rest_rel, rots, locs,
                                                  want=[3, 4])
        except ValueError:
            return None
        pts.append(heads)
    q = np.concatenate(pts, axis=1)                       # (n, 4, 3)
    w = np.einsum('ij,npj->npi', c_rot, q)
    cached = {'w': w, 'pivot_z': pivot_z}
    _AUTOGROUND_CACHE.clear()      # one rig at a time is plenty
    _AUTOGROUND_CACHE[key] = cached
    return cached


def _apply_autoground_to_rig(rig):
    """Recompute and apply the preview-grounding compensation."""
    if rig is None or rig.type != 'ARMATURE':
        return
    offset = getattr(rig, "blendcap_camera_offset", None)
    if offset is None:
        return
    applied = float(rig.get("_blendcap_autoground_applied_m", 0.0))
    pitch = math.radians(float(offset.pitch_deg))
    roll = math.radians(float(offset.roll_deg))
    target = 0.0
    if abs(pitch) > 1e-9 or abs(roll) > 1e-9:
        cache = _autoground_cache_for(rig)
        if cache is None:
            # No feet / no action: behave exactly as before this
            # feature existed (rotation preview without grounding).
            target = 0.0
        else:
            rot = preview_ground.preview_rotation(pitch, roll)
            target = float(preview_ground.ground_delta(
                cache['w'], cache['pivot_z'], rot))
    delta = target - applied
    if abs(delta) < 1e-9:
        return
    if _shift_hips_keys_world_z(rig, delta):
        rig["_blendcap_autoground_applied_m"] = target


# ============================================================
# Head-bone angle offset (Blender live preview)
# ============================================================
# Mirrors the Camera Angle Offset but scoped to the Head pose bone.
# Unlike Root (synthetic, unanimated), Head is keyframed by the BVH
# writer with real face-mocap rotations — so the live-preview path
# must COMPOSE the offset onto existing keys, not overwrite them.
#
# Math (Blender Z-up world frame):
#   R_world = R_y(roll) @ R_x(pitch)   # same axes as body offset
#   M       = arm_world_q @ Head.bone.matrix_local.to_quaternion()
#   R_local = M^-1 @ R_world @ M
# Then for each Head rotation_quaternion keyframe:
#   new_basis = R_local @ old_basis
#
# Idempotence: applied pitch/roll tracked in custom props on the rig.
# Each update computes R_local_new and R_local_old (from applied),
# and writes delta_q = R_local_new @ R_local_old^-1 onto each key.
# Slider 0 -> 5 -> 0 restores keys exactly.
#
# Approximation note: M uses Head's REST world rotation (via
# matrix_local), not the per-frame neck pose. For small corrective
# nudges the difference is invisible; the bake side (which writes the
# value through the npz_to_bvh pipeline) is exact in world space.

def _build_head_local_q(rig, pitch_deg, roll_deg):
    """Build the constant local-frame quaternion that, when left-
    multiplied onto Head's basis rotation, produces a world-space
    pitch+roll rotation through Head's rest world orientation.

    Returns None when the rig isn't usable (no Head bone). Axes match
    _apply_camera_offset_to_rig: pitch about world X, roll about
    world Y (Blender Z-up frame).
    """
    if rig is None or rig.type != 'ARMATURE':
        return None
    head_pb = rig.pose.bones.get('Head')
    if head_pb is None:
        return None
    pitch_rad = math.radians(pitch_deg)
    roll_rad = math.radians(roll_deg)
    R_q = (mathutils.Quaternion((0.0, 1.0, 0.0), roll_rad)
           @ mathutils.Quaternion((1.0, 0.0, 0.0), pitch_rad))
    arm_q = rig.matrix_world.to_quaternion()
    rest_q = head_pb.bone.matrix_local.to_quaternion()
    M_q = arm_q @ rest_q
    return M_q.inverted() @ R_q @ M_q


def _apply_head_offset_to_rig(rig):
    """Compose the pitch/roll offset onto Head's rotation keys.

    Reads the slider values and the previously-applied values from
    custom props on the rig, computes the delta quaternion, and
    pre-multiplies it onto every Head rotation_quaternion keyframe.
    Idempotent: re-applies cleanly across edits, slider 0 -> N -> 0
    restores original values.

    Gates: rig must be an armature with a "Head" pose bone and a non-
    None action. Silently no-op otherwise — Head-less rigs (hands-
    only, root-only) just ignore the slider.
    """
    if rig is None or rig.type != 'ARMATURE':
        return
    head_pb = rig.pose.bones.get('Head')
    if head_pb is None:
        return
    action = (rig.animation_data.action
              if rig.animation_data else None)
    if action is None:
        return

    offset = rig.blendcap_camera_offset
    new_pitch = float(offset.head_pitch_deg)
    new_roll = float(offset.head_roll_deg)
    old_pitch = float(rig.get("_blendcap_head_offset_applied_pitch_deg",
                              0.0))
    old_roll = float(rig.get("_blendcap_head_offset_applied_roll_deg",
                             0.0))

    R_new = _build_head_local_q(rig, new_pitch, new_roll)
    R_old = _build_head_local_q(rig, old_pitch, old_roll)
    if R_new is None or R_old is None:
        return
    delta_q = R_new @ R_old.inverted()

    # No-op when the delta is effectively identity (within float
    # noise) — avoids touching every keyframe on spurious updates.
    if (abs(delta_q.w - 1.0) < 1e-9 and abs(delta_q.x) < 1e-9
            and abs(delta_q.y) < 1e-9 and abs(delta_q.z) < 1e-9):
        return

    bone_path = 'pose.bones["Head"].rotation_quaternion'
    from .retarget import _iter_action_fcurves
    fcurves_by_index = {}
    for fc in _iter_action_fcurves(action):
        if fc.data_path == bone_path:
            fcurves_by_index[fc.array_index] = fc

    # If Head isn't keyframed (rare — the BVH writer always keys Head
    # on body+face exports — but defensive), prime one keyframe per
    # channel at frame_start so the fcurves exist.
    if len(fcurves_by_index) < 4:
        fr_start = int(action.frame_range[0])
        head_pb.rotation_mode = 'QUATERNION'
        head_pb.keyframe_insert('rotation_quaternion', frame=fr_start)
        fcurves_by_index = {}
        for fc in _iter_action_fcurves(action):
            if fc.data_path == bone_path:
                fcurves_by_index[fc.array_index] = fc
        if len(fcurves_by_index) < 4:
            return

    # Find the common frame set across all four channels. The BVH
    # writer keys all channels at the same frames so this is normally
    # just every key, but read each channel's frames to handle any
    # edit-time divergence.
    fc_w = fcurves_by_index[0]
    fc_x = fcurves_by_index[1]
    fc_y = fcurves_by_index[2]
    fc_z = fcurves_by_index[3]
    n = len(fc_w.keyframe_points)
    if n == 0:
        rig["_blendcap_head_offset_applied_pitch_deg"] = new_pitch
        rig["_blendcap_head_offset_applied_roll_deg"] = new_roll
        return
    nx = len(fc_x.keyframe_points)
    ny = len(fc_y.keyframe_points)
    nz = len(fc_z.keyframe_points)
    if nx != n or ny != n or nz != n:
        # Channels desynchronized — too risky to compose-multiply per
        # index. Bail rather than silently corrupt.
        return

    flat_w = [0.0] * (2 * n)
    flat_x = [0.0] * (2 * n)
    flat_y = [0.0] * (2 * n)
    flat_z = [0.0] * (2 * n)
    fc_w.keyframe_points.foreach_get('co', flat_w)
    fc_x.keyframe_points.foreach_get('co', flat_x)
    fc_y.keyframe_points.foreach_get('co', flat_y)
    fc_z.keyframe_points.foreach_get('co', flat_z)

    for k in range(n):
        idx = 2 * k + 1
        old_q = mathutils.Quaternion((flat_w[idx], flat_x[idx],
                                      flat_y[idx], flat_z[idx]))
        new_q = delta_q @ old_q
        flat_w[idx] = new_q.w
        flat_x[idx] = new_q.x
        flat_y[idx] = new_q.y
        flat_z[idx] = new_q.z

    fc_w.keyframe_points.foreach_set('co', flat_w)
    fc_x.keyframe_points.foreach_set('co', flat_x)
    fc_y.keyframe_points.foreach_set('co', flat_y)
    fc_z.keyframe_points.foreach_set('co', flat_z)
    fc_w.update()
    fc_x.update()
    fc_y.update()
    fc_z.update()

    rig["_blendcap_head_offset_applied_pitch_deg"] = new_pitch
    rig["_blendcap_head_offset_applied_roll_deg"] = new_roll


def update_head_offset(self, context):
    """PropertyGroup update callback for head_pitch_deg / head_roll_deg."""
    _apply_head_offset_to_rig(self.id_data)


class BlendCapCameraOffset(bpy.types.PropertyGroup):
    """Per-armature pitch/roll offset, baked into Root rotation keys
    on edit. Lives on bpy.types.Object via a PointerProperty so each
    BlendCap-imported rig has its own state — the panel binds directly
    to whichever armature is the active object.
    """
    pitch_deg: bpy.props.FloatProperty(
        name="Pitch (deg)",
        default=0.0, min=-45.0, max=45.0, step=10, precision=2,
        update=update_camera_offset,
        description="Manual forward/back tilt of the body, in degrees. "
                    "Live-updates the rig in the viewport. Click Bake to "
                    "write the value into the BVH on disk")
    roll_deg: bpy.props.FloatProperty(
        name="Roll (deg)",
        default=0.0, min=-45.0, max=45.0, step=10, precision=2,
        update=update_camera_offset,
        description="Manual side-to-side tilt of the body, in degrees. "
                    "Live-updates the rig in the viewport")
    height_offset_m: bpy.props.FloatProperty(
        name="Height (m)",
        default=0.0, min=-1.0, max=1.0, step=1, precision=3,
        unit='LENGTH',
        update=update_height_offset,
        description="Manual floor-level correction that shifts the whole "
                    "performance up or down. Live-updates the rig in the "
                    "viewport. Click Bake to write the value into the "
                    "BVH on disk")
    # Head-only pitch/roll, additive on top of the body pitch/roll.
    # Same world-frame axis convention; live-preview composes onto
    # Head's existing keys (see _apply_head_offset_to_rig). Baked into
    # the BVH alongside the body offset by Bake Settings to BVH.
    head_pitch_deg: bpy.props.FloatProperty(
        name="Head Pitch (deg)",
        default=0.0, min=-45.0, max=45.0, step=10, precision=2,
        update=update_head_offset,
        description="Extra pitch applied only to the Head bone, on top "
                    "of the body pitch above. Use when the head has a "
                    "residual forward/back tilt the body offset can't fix")
    head_roll_deg: bpy.props.FloatProperty(
        name="Head Roll (deg)",
        default=0.0, min=-45.0, max=45.0, step=10, precision=2,
        update=update_head_offset,
        description="Extra roll applied only to the Head bone, on top of "
                    "the body roll above")


# ============================================================
# PROPERTIES
# ============================================================
class BlendCapBVHEntry(bpy.types.PropertyGroup):
    filename: bpy.props.StringProperty()

# Static items list for the channels UI dropdown. Kept at module level
# (not inside the property declaration) because Blender requires the
# items list and its strings to stay alive across draws — it reads the
# enum items by reference. None entries render as visual separators in
# the dropdown.
_CHANNELS_UI_ITEMS = (
    ('LOC',     "Location",            "Copy location only (all axes)",         0),
    ('ROT',     "Rotation",            "Copy rotation only",                    1),
    ('LOC_ROT', "Location & Rotation", "Copy both location and rotation",       2),
    None,
    ('LOC_XY',  "Location (XY)",       "Copy location, X and Y axes only",      4),
    ('LOC_Z',   "Location (Z)",        "Copy location, Z axis only",            5),
)


def _channels_ui_items(self, context):
    return _CHANNELS_UI_ITEMS


def _channels_ui_get(self):
    """Map the underlying (channels, axes) state onto one of the 5 UI
    dropdown entries. Non-matching combinations (e.g. a preset with
    axes='YZ') fall back to the closest match by channels alone — user
    can re-pick from the dropdown to clean up."""
    axes = (self.axes or 'XYZ').upper()
    axes_norm = ''.join(sorted(set(c for c in axes if c in 'XYZ')))
    ch = self.channels
    if ch == 'ROT':
        return 1
    if ch == 'LOC_ROT':
        return 2
    if ch == 'LOC':
        if axes_norm in ('', 'XYZ'):
            return 0
        if axes_norm == 'XY':
            return 4
        if axes_norm == 'Z':
            return 5
        return 0
    return 2


def _channels_ui_set(self, value):
    """Translate a UI dropdown selection into (channels, axes) writes."""
    table = {
        0: ('LOC',     'XYZ'),
        1: ('ROT',     'XYZ'),
        2: ('LOC_ROT', 'XYZ'),
        4: ('LOC',     'XY'),
        5: ('LOC',     'Z'),
    }
    spec = table.get(value)
    if spec is None:
        return
    self.channels, self.axes = spec


class BlendCapBonePair(bpy.types.PropertyGroup):
    """One row in the retargeting bone-mapping table.

    `channels` decides what gets copied:
        ROT     -> COPY_ROTATION (axes ignored)
        LOC     -> COPY_LOCATION (axes filters X/Y/Z)
        LOC_ROT -> both

    Every field has update=_mark_dirty_dispatch so user edits flip the preset
    dropdown to __UNSAVED__ (see retarget.mark_dirty docstring).
    """
    source: bpy.props.StringProperty(name="Source Bone",
        description="Source Bone",
        update=_on_pair_source_change)
    target: bpy.props.StringProperty(name="Target Bone",
        description="Target Bone",
        update=_on_pair_target_change)
    influence: bpy.props.FloatProperty(name="Influence", default=1.0, min=0.0, max=1.0,
        update=_mark_dirty_dispatch)
    channels: bpy.props.EnumProperty(
        name="Channels",
        items=[('ROT',     "Rotation", "Copy rotation only"),
               ('LOC',     "Location", "Copy location only"),
               ('LOC_ROT', "Both",     "Copy rotation and location")],
        default='ROT',
        update=_mark_dirty_dispatch,
    )
    axes: bpy.props.StringProperty(name="Axes", default="XYZ",
        description="Location axes to copy (any subset of XYZ); ignored for rotation-only pairs",
        update=_mark_dirty_dispatch)
    loc_scale: bpy.props.FloatProperty(name="LOC Scale", default=1.0, min=0.0, soft_max=20.0,
        description="Per-pair LOC multiplier override. Default 1.0 means "
                    "'no explicit override' — the pair takes the Face Capture "
                    "Compensation region default when the scene checkbox is on "
                    "(jaw, eyes, brows, lips/cheeks each get a calibrated value), "
                    "or 1:1 transfer when the checkbox is off. Any other value "
                    "here wins regardless of the checkbox. Set to ~1.0001 if you "
                    "want to opt one pair out of compensation while keeping it on "
                    "globally.",
        update=_mark_dirty_dispatch)
    loc_scale_residual: bpy.props.BoolProperty(
        name="Scale Residual", default=False,
        description="HEAD_LOCAL pairs only: apply the LOC multiplier to the "
                    "bone's deviation from what its parent/carry already does, "
                    "instead of to the raw captured motion (the default). For "
                    "handles whose parent already follows the jaw — scaling "
                    "the raw delta would re-scale that built-in follow and "
                    "warp speech. CloudRig's graded lip handles use this; "
                    "ARP/Rigify lips keep the default",
        update=_mark_dirty_dispatch)
    loc_space: bpy.props.EnumProperty(
        name="LOC Space",
        items=[('BASIS',      "Basis (parent-local)",
                "Copy source basis location directly into the target bone's basis. "
                "Correct when source and target share an equivalent parent chain."),
               ('HEAD_LOCAL', "Head-Local (world delta)",
                "Compute the source bone's motion in the source head bone's local "
                "frame, apply it on top of the target bone's rest position in the "
                "target head bone's local frame, then back-solve the basis via "
                "pb.matrix. Use when source and target parent chains have "
                "different semantics — e.g. one rotates while the other "
                "translates — so basis-to-basis transfer would land the bone at "
                "the wrong head-relative position. Requires face_head_source / "
                "face_head_target on the preset; falls back to BASIS if either is "
                "missing or unresolved.")],
        default='BASIS',
        update=_mark_dirty_dispatch,
    )

    # UI-only wrappers. `anchor_to_source` proxies `loc_space` (so the
    # row gets a clean checkbox instead of an enum dropdown), and
    # `channels_ui` collapses `channels` + `axes` into a single 5-entry
    # dropdown for the common cases. Storage stays on the underlying
    # fields, so preset JSON format is unchanged — these are pure view
    # models. Setting the wrappers triggers the underlying fields' own
    # update callbacks (_mark_dirty_dispatch), so editing through the
    # wrapper still flips the preset dropdown to __UNSAVED__.
    anchor_to_source: bpy.props.BoolProperty(
        name="Anchor to Source",
        description="Anchor the target bone to the source's head-relative "
                    "position to compensate for constraint effects (only "
                    "available for face bones).",
        get=lambda self: self.loc_space == 'HEAD_LOCAL',
        set=lambda self, v: setattr(self, 'loc_space',
                                    'HEAD_LOCAL' if v else 'BASIS'),
    )
    channels_ui: bpy.props.EnumProperty(
        name="Channels",
        description="What data gets copied from source to target: "
                    "Location, Rotation, or both",
        items=_channels_ui_items,
        get=_channels_ui_get,
        set=_channels_ui_set,
    )


class BlendCapIKChain(bpy.types.PropertyGroup):
    """One row in the FK->IK chain mapping table.

    `owner` is read-only — set during chain discovery to the name of the bone
    hosting the IK constraint (e.g. 'MCH-shin_ik.L'). It identifies the chain
    topologically and stays stable across discovery runs, so user edits to
    the other fields survive a target-rig swap or a refresh.

    The user fills `ik_control` and (optionally) `pole_control`. Both are
    bone names on the target rig — picked via prop_search in the panel. An
    empty `ik_control` means "skip this chain entirely"; an empty
    `pole_control` (or `bake_pole=False`) means "bake the IK control but
    leave the pole alone".

    Pole baking is positional only: the pole control is keyed to a world
    position perpendicular to the chain root→end axis on the bend side,
    using the canonical Rigify formula (project lower segment onto chain
    axis, scale to chain length, place at elbow + perp). This is rig-
    agnostic — works on any rig with a positional pole target (Rigify
    `*_target.L` bones, custom rigs, etc.).
    Rotational pole controls (Rigify rotation_axis driver setups) are
    not supported — point pole_control at the rig's positional pole bone
    and flip the rig's pole_vector toggle to "Pole" mode in the rig UI.

    update=_mark_dirty_dispatch on every editable field so user edits
    flip the preset dropdown to __UNSAVED__ — same convention as
    BlendCapBonePair.
    """
    owner: bpy.props.StringProperty(
        name="IK Constraint Owner",
        description="Bone hosting the IK constraint on the target rig — "
                    "auto-populated by chain discovery, informational only. "
                    "Used by the pole-vector flip to find rig property "
                    "bones; doesn't drive the bake math itself.")
    limb_kind: bpy.props.EnumProperty(
        name="Limb Kind",
        items=[('ARM',   "Arm",   "Arm chain"),
               ('LEG',   "Leg",   "Leg chain"),
               ('OTHER', "Other", "Generic chain (legacy / migration only)")],
        default='ARM',
        description="Fixed identity for this row in the locked 4-row "
                    "FK → IK Mapping layout. Set by the operator; not "
                    "user-editable.")
    side: bpy.props.EnumProperty(
        name="Side",
        items=[('L', "Left",  "Left side"),
               ('R', "Right", "Right side")],
        default='L',
        description="Anatomical side. Fixed identity for this row in "
                    "the locked 4-row FK → IK Mapping layout. Set by "
                    "the operator; not user-editable.")
    ik_control: bpy.props.StringProperty(
        name="IK Control",
        description="The IK controller for the limb's hand/foot.",
        update=_mark_dirty_dispatch)
    pole_control: bpy.props.StringProperty(
        name="Pole Control",
        description="The positional pole bone that controls the limb's "
                    "elbow/knee direction.",
        update=_mark_dirty_dispatch)

def _sync_bvh_library(scene):
    bvh_dir = _bvh_library_dir()
    disk = []
    if os.path.isdir(bvh_dir):
        disk = sorted(f for f in os.listdir(bvh_dir) if f.endswith('.bvh'))
    current = [e.filename for e in scene.blendcap_bvh_entries]
    if disk == current:
        return False
    scene.blendcap_bvh_entries.clear()
    for name in disk:
        scene.blendcap_bvh_entries.add().filename = name
    if scene.blendcap_bvh_index >= len(disk):
        scene.blendcap_bvh_index = max(0, len(disk) - 1)
    return True

def _bvh_library_timer():
    try:
        changed = False
        for scene in bpy.data.scenes:
            if _sync_bvh_library(scene):
                changed = True
        if changed:
            _tag_redraw('VIEW_3D', 'PROPERTIES')
    except Exception:
        pass
    return 1.0

class BLENDCAP_UL_bvh(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.label(text=item.filename)
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="")

class BlendCapProperties(bpy.types.PropertyGroup):
    video_source: bpy.props.EnumProperty(
        name="Source",
        items=[('FILE', "File Path", "Run capture on a specific video file picked from disk"),
               ('VSE', "Video Editor Strip", "Run capture on the active clip in Blender's Video Sequencer")],
        default='FILE',
        update=update_video_source,
    )
    video_path: bpy.props.StringProperty(name="Video", description="Select the video to track", subtype='FILE_PATH')
    track_body: bpy.props.BoolProperty(name="Body Tracking", default=True,
        description="Track the body in the source video (pose, hands and root motion)")
    track_face: bpy.props.BoolProperty(name="Face Tracking", default=False,
        description="Track the face in the source video (blendshapes, landmarks, gaze)")
    preview_step: bpy.props.IntProperty(name="Preview Skip",
        description="How many frames to skip between checked preview frames (0 = check every frame; higher is faster but less thorough)",
        default=10, min=0, max=100)
    capture_skip: bpy.props.IntProperty(name="Capture Skip", default=0, min=0, max=30,
        description="How many frames to skip during capture, interpolated afterward (0 = every frame; higher is faster but rougher)")
    fov_mode: bpy.props.EnumProperty(name="FOV Mode",
        description="How the camera field of view is set during capture",
        items=[
            ('FIRST', "Sample first frame only",
             "Estimate the field of view once and reuse it for the clip"),
            ('EVERY', "Sample every frame",
             "Re-estimate every frame, for clips where the camera zooms"),
            ('MANUAL', "Enter focal length",
             "Skip estimation and use the focal length set below (fastest)"),
        ],
        default='FIRST')
    fov_focal_mm: bpy.props.FloatProperty(name="Focal Length (mm)",
        description="Camera focal length in millimetres (35mm / full-frame equivalent)",
        default=35.0, min=1.0, max=500.0)
    export_body: bpy.props.BoolProperty(name="Body", default=True,
        description="Include selected clip's body tracking data in the bvh armature")
    export_hands: bpy.props.BoolProperty(name="Hands", default=True,
        description="Include selected clip's hand tracking data in the bvh armature")
    export_root: bpy.props.BoolProperty(
        name="Root Motion",
        default=True,
        description="Include horizontal locomotion so the armature travels "
                    "through world space as the actor did.")
    export_face: bpy.props.BoolProperty(name="Face", default=False, update=update_export_face,
        description="Include selected clip's face tracking data in the bvh armature")
    use_separate_face: bpy.props.BoolProperty(name="Use Separate Face Data", default=False, update=update_use_separate_face,
        description="Combine the selected clip's body data with another clip's face data.")
    separate_face_data: bpy.props.EnumProperty(name="Face Data", description="Select cached face tracking data", items=get_face_data_items)
    smoothing_enabled: bpy.props.BoolProperty(
        name="Enable Smoothing",
        default=True,
        description="Apply the smoothing values below when generating "
                    "the BVH")
    smooth_body: bpy.props.FloatProperty(
        name="Body Smoothing",
        description="Smoothing window for body rotations, measured in seconds. 0 = off",
        default=0.23, min=0.0, max=2.0, step=5, precision=2)
    smooth_root: bpy.props.FloatProperty(
        name="Root Smoothing",
        description="Smoothing window for root position, measured in seconds. 0 = off",
        default=0.27, min=0.0, max=2.0, step=5, precision=2)
    smooth_vertical: bpy.props.FloatProperty(
        name="Vertical Smoothing",
        description="Smoothing window for vertical root motion (jumps, squats), measured in seconds. 0 = off",
        default=0.08, min=0.0, max=0.3, step=5, precision=2)
    smooth_face: bpy.props.FloatProperty(
        name="Face Smoothing",
        description="Smoothing window for face data, measured in seconds. 0 = off",
        default=0.23, min=0.0, max=2.0, step=5, precision=2)

    # ARKit shape-key application: independent clip selector and face
    # smoothing cutoff. Filter is Butterworth filtfilt (zero-phase,
    # FPS-independent); the slider value is the cutoff frequency in Hz.
    arkit_face_data: bpy.props.EnumProperty(
        name="ARKit Face Data",
        description="Select cached face tracking data",
        items=get_arkit_face_data_items)
    arkit_face_smooth: bpy.props.FloatProperty(
        name="ARKit Face Smoothing",
        description="Smoothing window for face data, measured in seconds. 0 = off",
        default=0.23, min=0.0, max=2.0, step=5, precision=2)
    arkit_section_expanded: bpy.props.BoolProperty(
        name="Show ARKit Shape Keys",
        description="Expand the ARKit Shape Keys section.",
        default=False)

    # ----- Face noise filtering (per-region dead-zone) -----
    # Applied to blendshape values AFTER smoothing, baking the filter
    # into the output (BVH/ARKit) so the data carries cleanly to other
    # programs. Thresholds are in [0,1] blendshape units. Master
    # toggle gates the per-region thresholds; expand toggle controls
    # whether the slider block is visible. Two independent sets so
    # BVH face and ARKit face can be tuned separately.

    bvh_face_denoise_enabled: bpy.props.BoolProperty(
        name="Use Face Noise Filtering",
        description="Snap small blendshape values to zero to mute jitter "
                    "that survives smoothing. Bakes into the exported BVH. "
                    "0 = no filtering.",
        default=False)
    bvh_face_denoise_expanded: bpy.props.BoolProperty(
        name="Show BVH Face Noise Filtering",
        description="Expand the BVH face noise filtering controls.",
        default=False)
    bvh_face_denoise_jaw: bpy.props.FloatProperty(
        name="Jaw",
        default=0.0, min=0.0, max=1.0, step=1, precision=3)
    bvh_face_denoise_gaze: bpy.props.FloatProperty(
        name="Gaze",
        default=0.0, min=0.0, max=1.0, step=1, precision=3)
    bvh_face_denoise_lids: bpy.props.FloatProperty(
        name="Eyelids",
        default=0.0, min=0.0, max=1.0, step=1, precision=3)
    bvh_face_denoise_brows: bpy.props.FloatProperty(
        name="Brows",
        default=0.0, min=0.0, max=1.0, step=1, precision=3)
    bvh_face_denoise_lips: bpy.props.FloatProperty(
        name="Lips",
        default=0.0, min=0.0, max=1.0, step=1, precision=3)
    bvh_face_denoise_cheeks: bpy.props.FloatProperty(
        name="Cheeks",
        default=0.0, min=0.0, max=1.0, step=1, precision=3)
    bvh_face_denoise_nose: bpy.props.FloatProperty(
        name="Nose",
        default=0.0, min=0.0, max=1.0, step=1, precision=3)
    bvh_face_denoise_tongue: bpy.props.FloatProperty(
        name="Tongue",
        default=0.0, min=0.0, max=1.0, step=1, precision=3)

    # --- BVH body depth noise filtering ---
    # Sibling of face noise filtering, but on the camera-depth axis of
    # body tip world positions. Per-tip smoothstep dead-zone on per-
    # frame depth velocity. Threshold sliders are in cm/frame; the
    # pipeline converts to meters before passing to npz_to_bvh.py.
    bvh_depth_denoise_enabled: bpy.props.BoolProperty(
        name="Use Depth Noise Filtering",
        description="Mute small depth-axis wobble that survives "
                    "smoothing on feet/hands/head/root. Bakes into "
                    "the exported BVH. 0 = no filtering.",
        default=True)
    bvh_depth_denoise_expanded: bpy.props.BoolProperty(
        name="Show BVH Depth Noise Filtering",
        description="Expand the BVH depth noise filtering controls.",
        default=False)
    bvh_depth_denoise_feet: bpy.props.FloatProperty(
        name="Feet (cm)",
        default=4.0, min=0.0, max=15.0, step=10, precision=2)
    bvh_depth_denoise_hands: bpy.props.FloatProperty(
        name="Hands (cm)",
        default=2.0, min=0.0, max=15.0, step=10, precision=2)
    bvh_depth_denoise_head: bpy.props.FloatProperty(
        name="Head (cm)",
        default=1.0, min=0.0, max=15.0, step=10, precision=2)
    bvh_depth_denoise_root: bpy.props.FloatProperty(
        name="Root (cm)",
        default=0.0, min=0.0, max=15.0, step=10, precision=2)

    arkit_face_denoise_enabled: bpy.props.BoolProperty(
        name="Use Face Noise Filtering",
        description="Snap small blendshape values to zero to mute jitter "
                    "that survives smoothing.",
        default=False)
    arkit_face_denoise_expanded: bpy.props.BoolProperty(
        name="Show ARKit Face Noise Filtering",
        description="Expand the ARKit face noise filtering controls.",
        default=False)
    arkit_face_denoise_jaw: bpy.props.FloatProperty(
        name="Jaw",
        default=0.0, min=0.0, max=1.0, step=1, precision=3)
    arkit_face_denoise_gaze: bpy.props.FloatProperty(
        name="Gaze",
        default=0.0, min=0.0, max=1.0, step=1, precision=3)
    arkit_face_denoise_lids: bpy.props.FloatProperty(
        name="Eyelids",
        default=0.0, min=0.0, max=1.0, step=1, precision=3)
    arkit_face_denoise_brows: bpy.props.FloatProperty(
        name="Brows",
        default=0.0, min=0.0, max=1.0, step=1, precision=3)
    arkit_face_denoise_lips: bpy.props.FloatProperty(
        name="Lips",
        default=0.0, min=0.0, max=1.0, step=1, precision=3)
    arkit_face_denoise_cheeks: bpy.props.FloatProperty(
        name="Cheeks",
        default=0.0, min=0.0, max=1.0, step=1, precision=3)
    arkit_face_denoise_nose: bpy.props.FloatProperty(
        name="Nose",
        default=0.0, min=0.0, max=1.0, step=1, precision=3)
    arkit_face_denoise_tongue: bpy.props.FloatProperty(
        name="Tongue",
        default=0.0, min=0.0, max=1.0, step=1, precision=3)

    # ----- BVH-side collapsible section states -----
    bvh_smoothing_expanded: bpy.props.BoolProperty(
        name="Show BVH Smoothing",
        description="Expand the BVH smoothing controls.",
        default=False)
    bvh_motion_scaling_expanded: bpy.props.BoolProperty(
        name="Show Motion Scaling",
        description="Expand the BVH Motion Scaling section.",
        default=False)

    # ----- BVH per-region face motion scaling -----
    # Mirrors the retarget face_amp concept (per-region multipliers
    # baked into the exported BVH face data so the scaling travels
    # with the export). Master toggle gates the per-region values;
    # expand toggle controls slider visibility. Per-region is ON by
    # default with tuned region values (2026-07 pass); 1.0 = 1:1
    # pass-through.
    bvh_face_amp_enabled: bpy.props.BoolProperty(
        name="Use Per-Region Face Scaling",
        description="Apply per-region face multipliers below instead of the global Strength",
        default=True)
    bvh_face_amp_jaw: bpy.props.FloatProperty(
        name="Jaw",
        default=0.7, min=0.0, max=20.0, step=10, precision=2)
    bvh_face_amp_gaze: bpy.props.FloatProperty(
        name="Gaze",
        default=2.0, min=0.0, max=20.0, step=10, precision=2)
    bvh_face_amp_lids: bpy.props.FloatProperty(
        name="Eyelids",
        default=1.4, min=0.0, max=20.0, step=10, precision=2)
    bvh_face_amp_brows: bpy.props.FloatProperty(
        name="Brows",
        default=1.0, min=0.0, max=20.0, step=10, precision=2)
    bvh_face_amp_lips: bpy.props.FloatProperty(
        name="Lips",
        default=0.5, min=0.0, max=20.0, step=10, precision=2)
    bvh_face_amp_cheeks: bpy.props.FloatProperty(
        name="Cheeks",
        default=0.7, min=0.0, max=20.0, step=10, precision=2)
    bvh_face_amp_nose: bpy.props.FloatProperty(
        name="Nose",
        default=1.0, min=0.0, max=20.0, step=10, precision=2)
    bvh_face_amp_tongue: bpy.props.FloatProperty(
        name="Tongue",
        default=1.2, min=0.0, max=20.0, step=10, precision=2)

    # ----- ARKit per-region face motion scaling -----
    # Mirrors the BVH bvh_face_amp_* set. When enabled, takes
    # precedence over the global face_expression_multiplier (which is
    # otherwise applied as a uniform multiplier in apply_arkit) to
    # avoid compounding.
    arkit_face_amp_enabled: bpy.props.BoolProperty(
        name="Use Per-Region Face Scaling",
        description="Apply the per-region face multipliers below instead of the global Strength",
        default=False)
    arkit_face_amp_expanded: bpy.props.BoolProperty(
        name="Show ARKit Face Motion Scaling",
        description="Expand the per-region ARKit face motion scaling sliders.",
        default=False)
    arkit_face_amp_jaw: bpy.props.FloatProperty(
        name="Jaw",
        default=1.0, min=0.0, max=20.0, step=10, precision=2)
    arkit_face_amp_gaze: bpy.props.FloatProperty(
        name="Gaze",
        default=1.0, min=0.0, max=20.0, step=10, precision=2)
    arkit_face_amp_lids: bpy.props.FloatProperty(
        name="Eyelids",
        default=1.0, min=0.0, max=20.0, step=10, precision=2)
    arkit_face_amp_brows: bpy.props.FloatProperty(
        name="Brows",
        default=1.0, min=0.0, max=20.0, step=10, precision=2)
    arkit_face_amp_lips: bpy.props.FloatProperty(
        name="Lips",
        default=1.0, min=0.0, max=20.0, step=10, precision=2)
    arkit_face_amp_cheeks: bpy.props.FloatProperty(
        name="Cheeks",
        default=1.0, min=0.0, max=20.0, step=10, precision=2)
    arkit_face_amp_nose: bpy.props.FloatProperty(
        name="Nose",
        default=1.0, min=0.0, max=20.0, step=10, precision=2)
    arkit_face_amp_tongue: bpy.props.FloatProperty(
        name="Tongue",
        default=1.0, min=0.0, max=20.0, step=10, precision=2)
    finger_curl_multiplier: bpy.props.FloatProperty(
        name="Finger Curl Strength",
        description="Amplifies finger curl to compensate for body trackers "
                    "that underestimate hand motion. Higher = more expressive "
                    "fingers (not compatible with Rebuild Finger Motion)",
        default=1.0, min=1.0, max=15.0, step=50, precision=1)
    bvh_hand_solver: bpy.props.BoolProperty(
        name="Rebuild Finger Motion (Experimental)",
        description="Rebuild finger poses from the tracked fingertips for "
                    "real per-finger articulation. Replaces Finger Curl "
                    "Strength; slow on long clips",
        default=False)
    face_expression_multiplier: bpy.props.FloatProperty(
        name="Face Expression Multiplier",
        description="Global multiplier for face bone movement from blendshape data. "
                    "Higher = more expressive",
        default=1.0, min=0.0, max=20.0, step=25, precision=1)
    custom_rest_pose_expanded: bpy.props.BoolProperty(
        name="Show Custom Rest Pose",
        default=False,
        description="Expand the custom rest-pose section.")
    use_custom_rest_pose: bpy.props.BoolProperty(
        name="Use Custom Rest Pose",
        default=False,
        description="Apply a saved pose preset as the source rig's rest "
                    "baseline during retargeting.",
        update=update_use_custom_rest_pose)
    custom_rest_pose_preset: bpy.props.EnumProperty(
        name="Rest Pose Preset",
        items=get_rest_pose_preset_items,
        description="Load a specific pre-built rest pose to a compatible "
                    "rig during retargeting.",
        update=update_custom_rest_pose_preset)

    # --- Foot Locking (Pass 1 + Pass 2 of the foot-grounding pipeline) ---
    foot_lock_enabled: bpy.props.BoolProperty(
        name="Enable Foot Locking",
        default=True,
        description="Snap feet to the floor during ground contacts")
    foot_lock_settings_expanded: bpy.props.BoolProperty(
        name="Foot Locking Settings",
        default=False,
        description="Expand the foot-locking settings sub-section.")
    foot_sensitivity: bpy.props.FloatProperty(
        name="Sensitivity",
        default=0.5, min=0.0, max=1.0, step=5, precision=2,
        description="Higher values catch more foot contacts. "
                    "Lower values catch fewer. Also sets how Feet "
                    "Define the Floor finds the contacts it builds "
                    "the floor from")
    foot_max_gap_sec: bpy.props.FloatProperty(
        name="Max Gap",
        default=0.33, min=0.0, max=2.0, step=5, precision=2,
        subtype='TIME_ABSOLUTE', unit='TIME',
        description="Merge contacts that occur within this time of each other")
    foot_max_lift_cm: bpy.props.FloatProperty(
        name="Max Lift (cm)",
        default=10.0, min=0.0, max=50.0, step=50, precision=1,
        description="Don't merge contacts if the foot rises higher "
                    "than this between them")
    foot_recover_glitches: bpy.props.BoolProperty(
        name="Bridge Phantom Lifts",
        default=True,
        description="Bridge contacts when the tracker reports a lift "
                    "but the foot didn't actually move horizontally")
    foot_recovery_lift_cm: bpy.props.FloatProperty(
        name="Recovery Lift (cm)",
        default=15.0, min=0.0, max=50.0, step=50, precision=1,
        description="Max foot lift treated as a phantom lift "
                    "when Bridge Phantom Lifts is on. Typically "
                    "larger than Max Lift")
    foot_pin_enabled: bpy.props.BoolProperty(
        name="Pin Planted Feet",
        default=True,
        description="Hold each foot in place during ground contacts "
                    "to remove sliding. Feet stay where they plant "
                    "until they lift")
    foot_pin_strength: bpy.props.FloatProperty(
        name="Pinning Strength",
        default=1.0, min=0.0, max=1.0, step=5, precision=2,
        description="How firmly planted feet hold their position. "
                    "Lower values let some of the original motion "
                    "through")
    foot_pin_release_cm: bpy.props.FloatProperty(
        name="Release Distance (cm)",
        default=20.0, min=0.0, max=50.0, step=50, precision=1,
        description="Unpin a contact when the foot drifts farther "
                    "than this from its planted spot. Frees real "
                    "slides that were mistaken for plants")
    foot_define_floor: bpy.props.BoolProperty(
        name="Feet Define the Floor",
        default=True,
        description="Uses a mixture of foot activity and video "
                    "information to determine the armature's height "
                    "above the ground, keeping the character on the "
                    "ground plane without destroying jumps. Works "
                    "best with Capture Skip at 0")

    # --- Body Grounding (BVH generation time) ---
    body_grounding_expanded: bpy.props.BoolProperty(
        name="Body Grounding",
        default=False,
        description="Expand the body-grounding settings sub-section.")
    body_grounding_enabled: bpy.props.BoolProperty(
        name="Enable Body Grounding",
        default=True,
        description="Pull rolls, lying, handstands, and crawls down so "
                    "the body actually touches the floor. Upright "
                    "motion and jumps are never affected")
    body_catch_distance_cm: bpy.props.FloatProperty(
        name="Catch Distance (cm)",
        default=30.0, min=10.0, max=100.0, step=100, precision=0,
        description="The farthest the body can be pulled toward the "
                    "floor. Raise it if a badly floating roll only "
                    "comes partway down; lower it to limit how much "
                    "this setting is allowed to move the body")
    body_skin_radius_cm: bpy.props.FloatProperty(
        name="Skin Radius (cm)",
        default=1.5, min=0.0, max=15.0, step=10, precision=1,
        description="Distance between the spine bones and the "
                    "character's back surface. The back counts as "
                    "touching when it is this far from the floor")

    # --- Camera Angle Offset (per-armature, BVH generation time) ---
    # Manual pitch / roll offset lives per-armature on the imported rig
    # (see Object.blendcap_camera_offset) so each take can be nudged
    # independently after import.
    camera_offset_expanded: bpy.props.BoolProperty(
        name="Camera Angle Offset",
        default=False,
        description="Expand the per-armature camera-angle offset section.")
    bvh_post_processing_expanded: bpy.props.BoolProperty(
        name="Show BVH Post Processing",
        default=True,
        description="Expand the BVH Post Processing category (camera "
                    "angle offset and other tools that act on already-"
                    "generated BVH data).")

    # --- Top-level section expansion toggles ---
    capture_expanded: bpy.props.BoolProperty(
        name="Show Capture",
        default=True,
        description="Expand the Capture section.")
    video_source_expanded: bpy.props.BoolProperty(
        name="Show Video Source",
        default=True,
        description="Expand the Video Source section.")
    bvh_settings_expanded: bpy.props.BoolProperty(
        name="Show BVH Settings",
        default=False,
        description="Expand the BVH Settings section.")
    bvh_library_expanded: bpy.props.BoolProperty(
        name="Show BVH Library",
        default=True,
        description="Expand the BVH Library section.")
