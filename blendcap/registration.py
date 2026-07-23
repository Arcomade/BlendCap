"""Class tuple, register(), unregister(), scene PropertyGroup wiring.

The order in `classes` matters:
PropertyGroups before operators that reference them via PointerProperty,
panels last (they reference operator IDs by string, so technically the
ordering between panels and operators doesn't matter for Blender, but
keeping operators-first matches Blender addon convention).
"""
import bpy

from .properties import (
    BlendCapBVHEntry, BlendCapBonePair, BlendCapIKChain,
    BLENDCAP_UL_bvh, BlendCapProperties, BlendCapCameraOffset,
    _bvh_library_timer,
)
from .operators_retarget import (
    BLENDCAP_UL_retarget_pairs,
    BLENDCAP_OT_retarget_save_preset,
    BLENDCAP_OT_retarget_save_preset_confirm,
    BLENDCAP_OT_retarget_reload_preset,
    BLENDCAP_OT_retarget_delete_preset,
    BLENDCAP_OT_retarget_auto_match,
    BLENDCAP_OT_retarget_clear_pairs,
    BLENDCAP_OT_retarget_add_pair,
    BLENDCAP_OT_retarget_add_all_source_bones,
    BLENDCAP_OT_retarget_sort_pairs,
    BLENDCAP_OT_retarget_remove_pair,
    BLENDCAP_OT_apply_retarget,
    BLENDCAP_OT_cancel_retarget,
)
from .operators_ik import (
    BLENDCAP_OT_convert_fk_to_ik,
    BLENDCAP_OT_cancel_fk_to_ik,
)
from .operators_rest_pose import (
    BLENDCAP_OT_save_rest_pose,
    BLENDCAP_OT_save_rest_pose_confirm,
    BLENDCAP_OT_delete_rest_pose_preset,
    BLENDCAP_OT_preview_rest_pose_preset,
)
from .properties import update_use_current_pose_as_rest
from .operators_pipeline import (
    BLENDCAP_LicenseLine,
    BLENDCAP_UL_license_text,
    BLENDCAP_OT_install_dependencies,
    BLENDCAP_OT_install_cancel,
    BLENDCAP_OT_install_run,
    BLENDCAP_OT_install_open_prefs,
    BLENDCAP_OT_copy_grant_cmd,
    BLENDCAP_OT_uninstall_dependencies,
    BLENDCAP_OT_deps_missing,
    BLENDCAP_OT_open_license,
    BLENDCAP_OT_append_workspace,
    BLENDCAP_OT_generate_mocap,
    BLENDCAP_OT_cancel,
    BLENDCAP_OT_cancel_cleanup,
    BLENDCAP_OT_generate_bvh,
    BLENDCAP_OT_bake_camera_offset,
)
from .operators_library import (
    BLENDCAP_OT_revert_strip,
    BLENDCAP_OT_show_warnings,
    BLENDCAP_OT_sequencer_warning,
    BLENDCAP_OT_vse_default_prompt,
    BLENDCAP_OT_import_lib_bvh,
    BLENDCAP_OT_export_lib_bvh,
    BLENDCAP_OT_export_lib_fbx,
    BLENDCAP_OT_delete_lib_bvh,
    BLENDCAP_OT_clear_current_cache,
    BLENDCAP_OT_clear_all_caches,
    BLENDCAP_OT_apply_arkit,
    BLENDCAP_OT_cancel_arkit_apply,
    BLENDCAP_OT_save_arkit_csv,
    BLENDCAP_OT_donate,
)
from .panels import (
    VIEW3D_PT_blendcap_ui,
    OUTPUT_PT_blendcap_ui,
)
from . import retarget


# ============================================================
# UPDATE CALLBACKS
# ============================================================
def _reload_current_preset(scene, context):
    """Re-fire the currently-selected preset's load. Delegates to
    retarget.update_preset_selection, which gates on the dropdown
    value: '__UNSAVED__' (user has edited the table since loading),
    non-selectable header / empty items, and missing-file paths all
    short-circuit as quiet no-ops. So calling this after a rig swap
    only re-loads when a real preset is still selected — user edits
    are preserved when the dropdown reads '(Unsaved changes…)'.

    The point is to re-run _populate_pairs_from_map's per-row
    existence filter against the freshly-assigned rig so rows that
    were filtered out under the old rig come back if the new rig
    has those bones (and vice versa).
    """
    retarget.update_preset_selection(scene, context)


def _on_source_rig_change(scene, context):
    """Auto-detect the source head bone when the user picks a source rig,
    then re-fire the current preset load against the new rig.

    Head autodetect only populates `blendcap_retarget_face_head_source`
    if it's currently blank — preserves any value already set by preset
    load or manual override. Passes the Source Prefix so a Mixamo rig
    (where the bone is literally 'mixamorig:Head') is matched too; the
    field's smart-strip update callback then writes back the SHORT form
    'Head'. The preset reload runs LAST so its `face_head_source` value
    (if the preset specifies one) wins over the autodetect.
    """
    from .helpers import autodetect_head_bone_name
    rig = scene.blendcap_retarget_source
    if rig is not None and not scene.blendcap_retarget_face_head_source:
        src_prefix = scene.blendcap_retarget_source_prefix or ""
        detected = autodetect_head_bone_name(rig, src_prefix)
        if detected:
            scene.blendcap_retarget_face_head_source = detected
    _reload_current_preset(scene, context)


def _on_target_rig_change(scene, context):
    """Sync the IK chain collection when the user picks/changes the
    target rig, then re-fire the current preset load against the new
    rig. Discovers IK constraints on the new rig and adds rows for any
    owners not already in the collection. Existing rows are preserved
    so user-edited fields (and preset entries) survive a rig swap.

    Also auto-detects the target head bone for Anchor pairs. Three-stage
    fallback (target side only — the source side stays simple):
      1. Common head bone name list, prefix-aware (handles Mixamo etc.).
      2. Walk the pair table for the row whose source matches Head
         (Source); use its target. Catches preset-defined pairings where
         the head name isn't in the common-name list (e.g. ARP's
         c_head.x → c_head_lol_grab.x or similar non-standard mappings).
      3. Give up; user must fill manually.

    The preset reload runs LAST so its IK chains / head / pair table
    overwrite the sync + autodetect output where the preset has values.
    """
    from .operators_ik import sync_ik_chain_collection
    from .helpers import autodetect_head_bone_name, target_head_from_pair_table
    sync_ik_chain_collection(scene.blendcap_retarget_target,
                             scene.blendcap_retarget_ik_chains)
    rig = scene.blendcap_retarget_target
    if rig is not None and not scene.blendcap_retarget_face_head_target:
        tgt_prefix = scene.blendcap_retarget_namespace_strip or ""
        detected = autodetect_head_bone_name(rig, tgt_prefix)
        if not detected:
            detected = target_head_from_pair_table(scene)
        if detected:
            scene.blendcap_retarget_face_head_target = detected
    _reload_current_preset(scene, context)


# ============================================================
# REGISTRATION
# ============================================================
classes = (
    BlendCapBVHEntry,
    BlendCapBonePair,
    BlendCapIKChain,
    BlendCapCameraOffset,
    BLENDCAP_LicenseLine,
    BLENDCAP_UL_bvh,
    BLENDCAP_UL_retarget_pairs,
    BLENDCAP_UL_license_text,
    BlendCapProperties,
    BLENDCAP_OT_install_dependencies,
    BLENDCAP_OT_install_cancel,
    BLENDCAP_OT_install_run,
    BLENDCAP_OT_install_open_prefs,
    BLENDCAP_OT_copy_grant_cmd,
    BLENDCAP_OT_uninstall_dependencies,
    BLENDCAP_OT_deps_missing,
    BLENDCAP_OT_open_license,
    BLENDCAP_OT_append_workspace,
    BLENDCAP_OT_generate_mocap,
    BLENDCAP_OT_cancel,
    BLENDCAP_OT_cancel_cleanup,
    BLENDCAP_OT_generate_bvh,
    BLENDCAP_OT_bake_camera_offset,
    BLENDCAP_OT_revert_strip,
    BLENDCAP_OT_show_warnings,
    BLENDCAP_OT_sequencer_warning,
    BLENDCAP_OT_vse_default_prompt,
    BLENDCAP_OT_import_lib_bvh,
    BLENDCAP_OT_export_lib_bvh,
    BLENDCAP_OT_export_lib_fbx,
    BLENDCAP_OT_delete_lib_bvh,
    BLENDCAP_OT_clear_current_cache,
    BLENDCAP_OT_clear_all_caches,
    BLENDCAP_OT_retarget_save_preset,
    BLENDCAP_OT_retarget_save_preset_confirm,
    BLENDCAP_OT_retarget_reload_preset,
    BLENDCAP_OT_retarget_delete_preset,
    BLENDCAP_OT_retarget_auto_match,
    BLENDCAP_OT_retarget_clear_pairs,
    BLENDCAP_OT_retarget_add_pair,
    BLENDCAP_OT_retarget_add_all_source_bones,
    BLENDCAP_OT_retarget_sort_pairs,
    BLENDCAP_OT_retarget_remove_pair,
    BLENDCAP_OT_apply_retarget,
    BLENDCAP_OT_cancel_retarget,
    BLENDCAP_OT_convert_fk_to_ik,
    BLENDCAP_OT_cancel_fk_to_ik,
    BLENDCAP_OT_save_rest_pose,
    BLENDCAP_OT_save_rest_pose_confirm,
    BLENDCAP_OT_delete_rest_pose_preset,
    BLENDCAP_OT_preview_rest_pose_preset,
    BLENDCAP_OT_apply_arkit,
    BLENDCAP_OT_cancel_arkit_apply,
    BLENDCAP_OT_save_arkit_csv,
    BLENDCAP_OT_donate,
    VIEW3D_PT_blendcap_ui,
    OUTPUT_PT_blendcap_ui,
)

@bpy.app.handlers.persistent
def _undo_redo_pre(scene, depsgraph=None):
    """Flip properties._UNDO_IN_PROGRESS True during undo/redo so the
    bone-pair dispatch skips its manual undo_push while Blender is
    restoring previous values. Otherwise each restored field fires
    its update callback, each pushes a phantom step, and the user
    has to Ctrl+Z 3-4 times before anything visibly changes."""
    from . import properties
    properties._UNDO_IN_PROGRESS = True


@bpy.app.handlers.persistent
def _undo_redo_post(scene, depsgraph=None):
    """Clear the flag after the restoration storm so genuine user
    edits resume pushing steps."""
    from . import properties
    properties._UNDO_IN_PROGRESS = False


@bpy.app.handlers.persistent
def _clear_dangling_retarget_rigs(scene, depsgraph=None):
    """Null out blendcap_retarget_source/target when their objects are
    no longer in the scene. Blender doesn't auto-clear PointerProperty
    references when the user deletes an object — the Python reference
    survives until garbage collection, and the property keeps showing
    the now-orphan datablock. The validator in operators_retarget.py
    catches this and reports a clean error, but the field still
    visibly holds a "ghost" rig that the user can't usefully interact
    with. This handler clears it so the UI reflects reality.

    Fires on depsgraph_update_post (covers user-driven deletes, undo,
    redo, collection exclusion changes) AND on load_post (covers
    .blend reopen). Cheap: two membership checks against scene.objects
    per registered scene. We only WRITE when there's a dangling
    reference, so the handler is a no-op on normal scene updates and
    doesn't feed back into the depsgraph loop.

    @persistent so the handler survives file-load and stays attached
    across sessions."""
    # load_post passes the FILEPATH STRING (not a Scene) as the first arg, so
    # guard on the actual type - a bare `is not None` check let the str through
    # and the loop below raised AttributeError on s.blendcap_retarget_source.
    scenes = (scene,) if isinstance(scene, bpy.types.Scene) else bpy.data.scenes
    for s in scenes:
        src = s.blendcap_retarget_source
        if src is not None and src.name not in s.objects:
            s.blendcap_retarget_source = None
        tgt = s.blendcap_retarget_target
        if tgt is not None and tgt.name not in s.objects:
            s.blendcap_retarget_target = None


# Reentrancy guard for the re-save the cache-migration handler triggers.
# The re-save fires save_post again; this flag makes that second pass skip
# straight through (the migration would no-op anyway — the temp cache is
# already emptied — but the flag avoids a wasted scan and any save loop).
_blendcap_migration_resaving = False


@bpy.app.handlers.persistent
def _blendcap_migrate_cache_on_save(*args):
    """After a project is saved, move an unsaved-session temp cache into
    the project's own folder so it persists, then re-point the loaded
    capture previews at the new location. Opt-in via the 'Move cache into
    project folder on save' add-on pref (default on). The actual move +
    repath lives in helpers.migrate_temp_cache_into.

    @persistent so it survives file loads. save_post runs on the main
    thread, so resolving prefs / touching bpy.data here is safe."""
    global _blendcap_migration_resaving
    if _blendcap_migration_resaving:
        _blendcap_migration_resaving = False
        return

    # Local import (matches the other handlers): keeps registration cheap
    # and means a stale/partially-updated helpers.py fails here at save
    # time — caught below — rather than as a registration-time ImportError
    # that would take down the whole add-on.
    from .helpers import addon_prefs, _cache_dir, migrate_temp_cache_into

    # Only after a real save (filepath now set). Moving the temp cache into
    # the project on first save is now default behavior (no longer a pref).
    if not bpy.data.filepath:
        return
    prefs = addon_prefs()
    if not prefs:
        return
    # A pinned custom cache folder ('Always use this folder') is the
    # intended permanent home, so never migrate into the project for it.
    # (The temp cache wouldn't exist in that mode anyway; explicit guard
    # for clarity.)
    if getattr(prefs, "cache_use_custom", False) and \
       getattr(prefs, "cache_always_use_custom", False):
        return

    try:
        new_cache = _cache_dir()
        moved = migrate_temp_cache_into(new_cache)
    except Exception as e:
        print(f"[BlendCap] Cache migration error: {e}")
        return
    if not moved:
        return

    # The save already wrote the .blend with the OLD (temp) paths; the
    # repath happened in memory afterward. Re-save once so the file on
    # disk records the new in-project locations. Deferred to a one-shot
    # timer (saving from inside save_post is reentrant) and guarded by
    # _blendcap_migration_resaving so it can't recurse.
    def _resave():
        global _blendcap_migration_resaving
        _blendcap_migration_resaving = True
        try:
            if bpy.data.filepath:
                bpy.ops.wm.save_mainfile()
        except Exception as e:
            _blendcap_migration_resaving = False
            print(f"[BlendCap] Re-save after cache move failed: {e}")
        return None
    bpy.app.timers.register(_resave, first_interval=0.05)


@bpy.app.handlers.persistent
def _blendcap_apply_default_video_source(*args):
    """Apply the 'Default Source' add-on pref to a fresh session so a user
    who normally captures from the Video Editor can make VSE the source a
    new scene starts on (the per-scene property's own default is FILE).

    Fires on load_post, and is also called once shortly after registration
    via a one-shot timer - the startup .blend loads BEFORE the add-on
    registers, so load_post never sees that first scene.

    ONLY applies when there is no saved file on disk (File > New, the startup
    file, a brand-new session). Opening a SAVED .blend keeps whatever source
    was stored with it, so this never overrides a real project's choice.

    @persistent so it survives file loads. Runs on the main thread (load_post
    / timer), so touching prefs + bpy.data is safe.
    """
    # Local import (matches the other handlers): keeps registration cheap and
    # turns a stale/partial helpers.py into a runtime error here rather than a
    # registration-time ImportError that would take down the whole add-on.
    from . import properties
    from .helpers import addon_prefs

    # Saved project -> respect the stored per-scene value.
    if bpy.data.filepath:
        return
    prefs = addon_prefs()
    if not prefs:
        return
    default_src = getattr(prefs, "default_video_source", 'FILE')

    # Mark this as a programmatic apply: update_video_source skips the NOTIFY
    # warning dialog for it (a VSE default would otherwise pop it on every new
    # file - new scenes have render.use_sequencer on) but still honors the
    # silent AUTO_DISABLE behavior, so "Turn off automatically" covers new
    # files too.
    prev = properties._SETTING_DEFAULT_SOURCE
    properties._SETTING_DEFAULT_SOURCE = True
    try:
        for s in bpy.data.scenes:
            props = getattr(s, "blendcap_props", None)
            if props is not None and props.video_source != default_src:
                props.video_source = default_src
    finally:
        properties._SETTING_DEFAULT_SOURCE = prev

    # Enforce AUTO_DISABLE directly instead of riding on the property
    # update above: the update only fires when the value actually CHANGES,
    # and a startup file saved with the source already on VSE (likely for
    # exactly the users this pref exists for) changes nothing - the
    # callback never runs and the Sequencer checkbox stays on.
    if (default_src == 'VSE'
            and getattr(prefs, "sequencer_export_behavior", 'NOTIFY')
                == 'AUTO_DISABLE'):
        for s in bpy.data.scenes:
            props = getattr(s, "blendcap_props", None)
            if (props is not None and props.video_source == 'VSE'
                    and getattr(s.render, 'use_sequencer', False)):
                s.render.use_sequencer = False


def _blendcap_default_source_startup_timer():
    """One-shot: apply the default-source pref to the startup session, which
    load_post can't catch (it loaded before the add-on registered)."""
    try:
        _blendcap_apply_default_video_source()
    except Exception as e:
        print(f"[BlendCap] default video source apply failed: {e}")
    return None  # returning None unregisters this one-shot timer


def register():
    # Reload trap: if a module wasn't reloaded but its sibling registration
    # module was, Blender may have a stale class registered under the same
    # bl_idname that our top-level unregister() couldn't address (because
    # its `classes` tuple references the freshly-imported class object).
    # When register_class hits that case it raises ValueError. Catch, kick
    # the previous occupant via the same bl_idname, retry once. Avoids
    # forcing a Blender restart after every file rename in the package.
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            try:
                bpy.utils.unregister_class(cls)
            except Exception:
                pass
            bpy.utils.register_class(cls)
    bpy.types.Scene.blendcap_props = bpy.props.PointerProperty(type=BlendCapProperties)
    bpy.types.Scene.blendcap_bvh_entries = bpy.props.CollectionProperty(type=BlendCapBVHEntry)
    bpy.types.Scene.blendcap_bvh_index = bpy.props.IntProperty(default=0)

    # Transient backing store for the install wizard's scrollable license
    # viewer. Lives on WindowManager (session-scoped, never saved to file);
    # repopulated each time the wizard starts.
    bpy.types.WindowManager.blendcap_license_lines = bpy.props.CollectionProperty(
        type=BLENDCAP_LicenseLine)
    bpy.types.WindowManager.blendcap_license_index = bpy.props.IntProperty(default=0)
    # Inline install-wizard state (drawn in the Add-on Preferences panel).
    # WindowManager-scoped so it resets each session and is never saved.
    bpy.types.WindowManager.blendcap_install_active = bpy.props.BoolProperty(
        default=False)
    bpy.types.WindowManager.blendcap_install_page = bpy.props.EnumProperty(
        items=[('INTRO', "Intro", ""),
               ('LICENSE', "License", ""),
               ('CONFIRM', "Confirm", "")],
        default='INTRO')
    bpy.types.WindowManager.blendcap_install_agreed = bpy.props.BoolProperty(
        name="I have read and agree to the Meta SAM License", default=False)
    bpy.types.WindowManager.blendcap_install_force = bpy.props.BoolProperty(
        default=False)
    # Inline uninstall confirm: True while the red Confirm/Cancel block is
    # showing in the prefs panel. WindowManager-scoped, session-only.
    bpy.types.WindowManager.blendcap_uninstall_armed = bpy.props.BoolProperty(
        default=False)

    # --- Retargeting v2 scene state ---
    bpy.types.Scene.blendcap_retarget_pairs = bpy.props.CollectionProperty(type=BlendCapBonePair)
    bpy.types.Scene.blendcap_retarget_pair_index = bpy.props.IntProperty(default=0)
    # PointerProperty + poll restricts the picker to armatures only. The
    # picker UI (layout.prop with an Object pointer) auto-generates a
    # browse-icon dropdown that honors the poll filter, so non-armature
    # objects (meshes, empties, cameras, lights) never appear.
    bpy.types.Scene.blendcap_retarget_source = bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Source Rig",
        description="Armature carrying the animation.",
        poll=lambda self, obj: obj.type == 'ARMATURE',
        update=_on_source_rig_change)
    bpy.types.Scene.blendcap_retarget_target = bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Target Rig",
        description="Armature to receive the animation.",
        poll=lambda self, obj: obj.type == 'ARMATURE',
        update=_on_target_rig_change)
    bpy.types.Scene.blendcap_retarget_preset = bpy.props.EnumProperty(
        name="Preset",
        items=retarget.get_combined_preset_items,
        update=retarget.update_preset_selection,
        description="Load a pre-built mapping of bone pairs for two specific rigs.")
    bpy.types.Scene.blendcap_retarget_map_hint = bpy.props.StringProperty(
        name="Map Hint", default="", options={'HIDDEN'},
        description="Apply-path hint carried from the loaded preset's target_kind field "
                    "(e.g. 'rigify_control'). Not user-editable — set by preset load.")
    bpy.types.Scene.blendcap_retarget_last_loaded_preset = bpy.props.StringProperty(
        name="Last Loaded Preset", default="", options={'HIDDEN'},
        description="Path of the most recently loaded bone-map preset. "
                    "Survives the dropdown's flip to '(Unsaved changes…)' "
                    "after edits, so Save Map As... can default its name "
                    "to the loaded preset (i.e. save = overwrite).")
    # Target Prefix — kept under the legacy property name "blendcap_retarget_
    # namespace_strip" so old presets (which save this value under the
    # "namespace_strip" JSON key) populate the right field on load with
    # zero migration. UI label says "Target Prefix" everywhere. The update
    # callback walks existing pairs and smart-strips their target names
    # so editing the prefix retroactively cleans rows added before the
    # prefix was set.
    bpy.types.Scene.blendcap_retarget_namespace_strip = bpy.props.StringProperty(
        name="Target Prefix",
        description="Prefix to be ignored on the TARGET rig's bones "
                    "(e.g. 'samplerig1:'). Leave blank if none",
        update=retarget._on_target_prefix_change)
    bpy.types.Scene.blendcap_retarget_source_prefix = bpy.props.StringProperty(
        name="Source Prefix",
        description="Prefix to be ignored on the SOURCE rig's bones "
                    "(e.g. 'samplerig1:'). Leave blank if none",
        update=retarget._on_source_prefix_change)
    # Head bone references. Update callbacks smart-strip against the
    # matching prefix so storage stays in SHORT form (same convention
    # as the pair source/target cells). Auto-detect at rig assignment
    # writes the literal rig bone name, which then auto-strips on the
    # callback's first fire.
    bpy.types.Scene.blendcap_retarget_face_head_source = bpy.props.StringProperty(
        name="Head (Source)", default="",
        description="Source rig's head bone, used as the reference for "
                    "Anchor pairs. (Auto-filled from common names when "
                    "you pick a source rig).",
        update=retarget._on_face_head_source_change)
    bpy.types.Scene.blendcap_retarget_face_head_target = bpy.props.StringProperty(
        name="Head (Target)", default="",
        description="Target rig's head bone, used as the reference for "
                    "Anchor pairs. (Auto-filled from common names).",
        update=retarget._on_face_head_target_change)
    bpy.types.Scene.blendcap_retarget_expanded = bpy.props.BoolProperty(
        name="Show Retargeting", default=False,
        description="Expand the retargeting bone-mapping panel")
    bpy.types.Scene.blendcap_retarget_advanced_expanded = bpy.props.BoolProperty(
        name="Show Advanced", default=False,
        description="Expand the Advanced sub-section (rarely-needed escape hatches)")
    bpy.types.Scene.blendcap_retarget_auto_scale = bpy.props.BoolProperty(
        name="Auto-scale to Target",
        default=True,
        description="Scale the source vertically so its height matches the "
                    "target before baking. Keeps feet on the floor regardless "
                    "of character proportions")
    bpy.types.Scene.blendcap_face_amp_global = bpy.props.FloatProperty(
        name="Strength",
        description="Global multiplier for face bone movement. Higher = more expressive",
        default=1.0, soft_min=0.0, soft_max=20.0)
    bpy.types.Scene.blendcap_face_capture_compensation = bpy.props.BoolProperty(
        name="Use Per-Region",
        default=False,
        description="Apply per-region face multipliers below instead of the global Strength")
    bpy.types.Scene.blendcap_face_amp_expanded = bpy.props.BoolProperty(
        name="Show Amplification Values", default=False,
        description="Expand the per-region amplification sliders (jaw, gaze, lids, brows, "
                    "lips, cheeks, nose, tongue).")
    # Per-region amplification multipliers. Engine reads via
    # _resolve_loc_scale; presets can override on load via the JSON
    # face_amplification block. Default: 1.0 for every region (pure
    # 1:1 transfer). Increase gaze for rigs whose look-at target sits
    # far in front of the face — at long lever arms, 1:1 source eye
    # displacement produces a very small visible gaze swing.
    _face_amp_defaults = {
        "jaw":    (1.0,  ""),
        "gaze":   (1.0,  ""),
        "lids":   (1.0,  ""),
        "brows":  (1.0,  ""),
        "lips":   (1.0,  ""),
        "cheeks": (1.0,  ""),
        "nose":   (1.0,  ""),
        "tongue": (1.0,  ""),
    }
    _face_amp_display_names = {
        "jaw":    "Jaw",
        "gaze":   "Gaze",
        "lids":   "Eyelids",
        "brows":  "Brows",
        "lips":   "Lips",
        "cheeks": "Cheeks",
        "nose":   "Nose",
        "tongue": "Tongue",
    }
    for _region, (_default, _desc) in _face_amp_defaults.items():
        setattr(bpy.types.Scene, f"blendcap_face_amp_{_region}",
                bpy.props.FloatProperty(
                    name=_face_amp_display_names[_region],
                    default=_default,
                    soft_min=0.0, soft_max=20.0,
                    description=_desc))
    # Face Denoising — dead-zone filter applied per-region at retarget time.
    # Zero out per-pair motion whose magnitude is below the region's
    # threshold, so source-side noise floor (e.g. MediaPipe + calibration
    # cross-pollution leaving jawOpen at 3% on a closed-mouth video, which
    # bakes into a couple degrees of jaw rotation noise) doesn't leak to
    # the target. Slider value is a single number per region, interpreted
    # as mm for LOC channels and degrees for ROT channels — same scalar
    # reads sensibly for both since face motion's "small" range is ~1mm /
    # ~1° in both. Per-pair LOC Scale override is NOT considered here;
    # denoise gates the raw signal, scaling happens after.
    bpy.types.Scene.blendcap_face_denoise = bpy.props.BoolProperty(
        name="Use Face Denoising",
        default=False,
        description="Apply per-region dead-zone thresholds to face bone pairs. Any "
                    "per-frame motion below the threshold reads as rest (zero), so "
                    "source noise (MediaPipe / calibration cross-pollution producing "
                    "small non-zero deltas on bones that should be at rest) doesn't "
                    "bleed onto the target. Thresholds are in mm for LOC pairs and "
                    "degrees for ROT pairs (same slider value, unit depends on the "
                    "pair's channel). Tunable in the Denoising Values dropdown below; "
                    "engine defaults are 0 for every region (no filtering). Face "
                    "amplitude scaling runs AFTER denoising — so a dead-zoned value "
                    "stays zero regardless of amplification.")
    bpy.types.Scene.blendcap_face_denoise_expanded = bpy.props.BoolProperty(
        name="Show Denoising Values", default=False,
        description="Expand the per-region denoising threshold sliders (jaw, gaze, "
                    "lids, brows, lips, cheeks, nose, tongue).")
    _face_denoise_defaults = {
        "jaw":    (0.0, "Jaw denoise threshold. mm for LOC channels, degrees for ROT "
                        "channels (Jaw → jaw_master is ROT-dominant on Rigify; raise "
                        "this to mute small jaw rotations like the mouth opening "
                        "slightly when source is at rest). Typical: 1-3 degrees."),
        "gaze":   (0.0, "Gaze denoise threshold. mm for LOC, degrees for ROT. Raise "
                        "to quiet small eye darting on a fixed gaze. Typical: 0.5-2."),
        "lids":   (0.0, "Eyelid denoise threshold. mm / degrees. Raise to suppress "
                        "small lid jitter on a steady open eye. Typical: 0.5-1.5."),
        "brows":  (0.0, "Brow denoise threshold. mm / degrees. Raise to flatten brow "
                        "twitch on a neutral expression. Typical: 0.5-1.5."),
        "lips":   (0.0, "Lip denoise threshold. mm / degrees. Raise to mute small lip "
                        "tremor on a closed mouth. Typical: 0.5-2."),
        "cheeks": (0.0, "Cheek denoise threshold. mm / degrees. Raise to suppress "
                        "cheek micro-puff at rest. Typical: 0.5-2."),
        "nose":   (0.0, "Nose denoise threshold. mm / degrees. Raise to flatten "
                        "nostril flare on neutral expression. Typical: 0.5-1.5."),
        "tongue": (0.0, "Tongue denoise threshold. mm for LOC, degrees for ROT. "
                        "Raise to mute stray tongue flicks while the mouth "
                        "talks; real tongue-outs move far past it. Typical: "
                        "0.5-2."),
    }
    for _region, (_default, _desc) in _face_denoise_defaults.items():
        setattr(bpy.types.Scene, f"blendcap_face_denoise_{_region}",
                bpy.props.FloatProperty(
                    name=_face_amp_display_names[_region],
                    default=_default,
                    soft_min=0.0, soft_max=20.0,
                    description=_desc))
    bpy.types.Scene.blendcap_retarget_use_current_pose_as_rest = bpy.props.BoolProperty(
        name="Use Current Source Pose as Rest",
        default=False,
        description="Treat the source rig's currently visible pose as its "
                    "rest baseline. Pose the source to roughly match the "
                    "target's rest pose before retargeting.",
        update=update_use_current_pose_as_rest)
    bpy.types.Scene.blendcap_retarget_current_pose_full_matrix = bpy.props.BoolProperty(
        name="Include Location & Scale in Rest Pose",
        default=False,
        description="Only meaningful when using a rest-pose override option. "
                    "By default only ROTATION is taken from the current pose; "
                    "enable this to use location and scale as well.")
    bpy.types.Scene.blendcap_retarget_use_world_location = bpy.props.BoolProperty(
        name="Read Source Location in World Space",
        default=False,
        description="Sample source location in world space instead of basis "
                    "(fixes Mixamo and other sources with a non-identity object transforms).")

    # --- FK->IK chain mapping (separate panel section) ---
    # Owner-keyed collection of per-chain mappings. Discovery scans the
    # target rig's IK constraints and populates `owner` for each row;
    # ik_control / pole_control / pole_mode / pole_axis / bake_pole are
    # filled by the user or a custom preset. Auto-bake-IK is the opt-in
    # that runs FK->IK conversion automatically at the end of an Apply
    # Retargeting run; default OFF, presets may flip it ON.
    bpy.types.Scene.blendcap_retarget_ik_chains = bpy.props.CollectionProperty(
        type=BlendCapIKChain)
    bpy.types.Scene.blendcap_retarget_ik_chains_index = bpy.props.IntProperty(default=0)
    bpy.types.Scene.blendcap_retarget_ik_expanded = bpy.props.BoolProperty(
        name="Show FK → IK Mapping", default=False,
        description="Expand the FK → IK chain mapping sub-section "
                    "(per-chain IK control + pole control assignments).")
    bpy.types.Scene.blendcap_retarget_auto_bake_ik = bpy.props.BoolProperty(
        name="Auto-bake IK on Apply", default=False,
        description="After retargeting finishes, automatically run "
                    "Convert FK → IK using the chain mappings below.")

    # --- Custom IK source-bone fields (Advanced section) ---
    # Eight per-side end-effector + pole source bone fields. Empty by
    # default; only consulted when `blendcap_use_custom_ik_sources` is
    # True. Otherwise the FK->IK bake uses the silent BlendCap BVH
    # defaults (LeftHand / RightHand / LeftFoot / RightFoot for end-
    # effectors; LeftForeArm / RightForeArm / LeftLeg / RightLeg as
    # pole chain mids).
    bpy.types.Scene.blendcap_use_custom_ik_sources = bpy.props.BoolProperty(
        name="Use Custom IK Source Bones", default=False,
        description="Use custom source bone names below when baking FK → IK "
                    "from third-party armatures.")
    # Property `name=` fields below describe the BONE TO TYPE on the
    # source rig (Hand / Foot / Forearm / Shin), not the downstream IK
    # control role the value drives. Labeling them "Pole" used to send
    # users hunting for explicit elbow/knee indicator bones their rig
    # didn't have; the default FORMULA path actually wants the chain
    # mid bone (forearm or shin) whose head sits at the bend joint.
    # Custom IK source bone fields. Each gets a smart-strip update
    # callback so values stored here mirror the pair-source convention:
    # SHORT bone name on disk, Source Prefix prepends at bake time. Edit
    # Source Prefix and these fields restrip in the same sweep as the
    # pair table (see retarget._on_source_prefix_change).
    bpy.types.Scene.blendcap_ik_src_left_arm_end = bpy.props.StringProperty(
        name="Left Hand Source", default="",
        description="Source armature bone name for the left hand.",
        update=retarget._on_custom_ik_source_field_change("blendcap_ik_src_left_arm_end"))
    bpy.types.Scene.blendcap_ik_src_right_arm_end = bpy.props.StringProperty(
        name="Right Hand Source", default="",
        description="Source armature bone name for the right hand.",
        update=retarget._on_custom_ik_source_field_change("blendcap_ik_src_right_arm_end"))
    bpy.types.Scene.blendcap_ik_src_left_leg_end = bpy.props.StringProperty(
        name="Left Foot Source", default="",
        description="Source armature bone name for the left foot.",
        update=retarget._on_custom_ik_source_field_change("blendcap_ik_src_left_leg_end"))
    bpy.types.Scene.blendcap_ik_src_right_leg_end = bpy.props.StringProperty(
        name="Right Foot Source", default="",
        description="Source armature bone name for the right foot.",
        update=retarget._on_custom_ik_source_field_change("blendcap_ik_src_right_leg_end"))
    bpy.types.Scene.blendcap_ik_src_left_arm_pole = bpy.props.StringProperty(
        name="Left Forearm Source", default="",
        description="Source armature bone name for the left forearm.",
        update=retarget._on_custom_ik_source_field_change("blendcap_ik_src_left_arm_pole"))
    bpy.types.Scene.blendcap_ik_src_right_arm_pole = bpy.props.StringProperty(
        name="Right Forearm Source", default="",
        description="Source armature bone name for the right forearm.",
        update=retarget._on_custom_ik_source_field_change("blendcap_ik_src_right_arm_pole"))
    bpy.types.Scene.blendcap_ik_src_left_leg_pole = bpy.props.StringProperty(
        name="Left Shin Source", default="",
        description="Source armature bone name for the left shin.",
        update=retarget._on_custom_ik_source_field_change("blendcap_ik_src_left_leg_pole"))
    bpy.types.Scene.blendcap_ik_src_right_leg_pole = bpy.props.StringProperty(
        name="Right Shin Source", default="",
        description="Source armature bone name for the right shin.",
        update=retarget._on_custom_ik_source_field_change("blendcap_ik_src_right_leg_pole"))
    bpy.types.Scene.blendcap_retarget_ik_src_expanded = bpy.props.BoolProperty(
        name="Show Custom IK Source Bones", default=False,
        description="Expand the Custom IK Source Bones sub-section "
                    "(per-side end-effector + pole source bone names "
                    "used by the FK → IK retarget when the 'Use Custom "
                    "IK Source Bones' checkbox is on).")

    # --- Per-armature live camera offset ---
    # PropertyGroup attached to every Object so each imported BlendCap
    # armature has its own pitch/roll state. Updates rewrite Root's
    # rotation keyframes via the conjugation math in properties.
    # _apply_camera_offset_to_rig. Panel binds directly to whichever
    # armature is the active object (no scene-level proxy state for
    # the live data).
    bpy.types.Object.blendcap_camera_offset = bpy.props.PointerProperty(
        type=BlendCapCameraOffset)
    # Scene-level placeholder used purely so the Camera Angle Offset
    # sliders can always render (greyed) even when no valid BlendCap
    # body rig is active. Values are inert — nothing reads from this.
    bpy.types.Scene.blendcap_camera_offset_placeholder = bpy.props.PointerProperty(
        type=BlendCapCameraOffset)

    if not bpy.app.timers.is_registered(_bvh_library_timer):
        bpy.app.timers.register(_bvh_library_timer, first_interval=0.2, persistent=True)

    if _clear_dangling_retarget_rigs not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_clear_dangling_retarget_rigs)
    if _clear_dangling_retarget_rigs not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_clear_dangling_retarget_rigs)
    if _blendcap_migrate_cache_on_save not in bpy.app.handlers.save_post:
        bpy.app.handlers.save_post.append(_blendcap_migrate_cache_on_save)
    if _blendcap_apply_default_video_source not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_blendcap_apply_default_video_source)
    # Catch the startup scene, which loaded before this register() ran (so
    # load_post never fired for it). One-shot - returns None and unregisters.
    bpy.app.timers.register(_blendcap_default_source_startup_timer,
                            first_interval=0.1)
    if _undo_redo_pre not in bpy.app.handlers.undo_pre:
        bpy.app.handlers.undo_pre.append(_undo_redo_pre)
    if _undo_redo_post not in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(_undo_redo_post)
    if _undo_redo_pre not in bpy.app.handlers.redo_pre:
        bpy.app.handlers.redo_pre.append(_undo_redo_pre)
    if _undo_redo_post not in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.append(_undo_redo_post)


def unregister():
    if _clear_dangling_retarget_rigs in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_clear_dangling_retarget_rigs)
    if _clear_dangling_retarget_rigs in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_clear_dangling_retarget_rigs)
    if _blendcap_migrate_cache_on_save in bpy.app.handlers.save_post:
        bpy.app.handlers.save_post.remove(_blendcap_migrate_cache_on_save)
    if _blendcap_apply_default_video_source in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_blendcap_apply_default_video_source)
    if bpy.app.timers.is_registered(_blendcap_default_source_startup_timer):
        bpy.app.timers.unregister(_blendcap_default_source_startup_timer)
    if _undo_redo_pre in bpy.app.handlers.undo_pre:
        bpy.app.handlers.undo_pre.remove(_undo_redo_pre)
    if _undo_redo_post in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.remove(_undo_redo_post)
    if _undo_redo_pre in bpy.app.handlers.redo_pre:
        bpy.app.handlers.redo_pre.remove(_undo_redo_pre)
    if _undo_redo_post in bpy.app.handlers.redo_post:
        bpy.app.handlers.redo_post.remove(_undo_redo_post)
    if bpy.app.timers.is_registered(_bvh_library_timer):
        bpy.app.timers.unregister(_bvh_library_timer)
    for cls in reversed(classes): bpy.utils.unregister_class(cls)
    del bpy.types.Scene.blendcap_props
    del bpy.types.Scene.blendcap_bvh_entries
    del bpy.types.Scene.blendcap_bvh_index
    del bpy.types.WindowManager.blendcap_license_lines
    del bpy.types.WindowManager.blendcap_license_index
    del bpy.types.WindowManager.blendcap_install_active
    del bpy.types.WindowManager.blendcap_install_page
    del bpy.types.WindowManager.blendcap_install_agreed
    del bpy.types.WindowManager.blendcap_install_force
    del bpy.types.WindowManager.blendcap_uninstall_armed
    del bpy.types.Scene.blendcap_retarget_pairs
    del bpy.types.Scene.blendcap_retarget_pair_index
    del bpy.types.Scene.blendcap_retarget_source
    del bpy.types.Scene.blendcap_retarget_target
    del bpy.types.Scene.blendcap_retarget_preset
    del bpy.types.Scene.blendcap_retarget_last_loaded_preset
    del bpy.types.Scene.blendcap_retarget_map_hint
    del bpy.types.Scene.blendcap_retarget_namespace_strip
    del bpy.types.Scene.blendcap_retarget_source_prefix
    del bpy.types.Scene.blendcap_retarget_face_head_source
    del bpy.types.Scene.blendcap_retarget_face_head_target
    del bpy.types.Scene.blendcap_retarget_expanded
    del bpy.types.Scene.blendcap_retarget_advanced_expanded
    del bpy.types.Scene.blendcap_retarget_auto_scale
    del bpy.types.Scene.blendcap_face_amp_global
    del bpy.types.Scene.blendcap_face_capture_compensation
    del bpy.types.Scene.blendcap_face_amp_expanded
    for _region in ("jaw", "gaze", "lids", "brows", "lips", "cheeks", "nose", "tongue"):
        delattr(bpy.types.Scene, f"blendcap_face_amp_{_region}")
    del bpy.types.Scene.blendcap_face_denoise
    del bpy.types.Scene.blendcap_face_denoise_expanded
    for _region in ("jaw", "gaze", "lids", "brows", "lips", "cheeks", "nose", "tongue"):
        delattr(bpy.types.Scene, f"blendcap_face_denoise_{_region}")
    del bpy.types.Scene.blendcap_retarget_use_current_pose_as_rest
    del bpy.types.Scene.blendcap_retarget_current_pose_full_matrix
    del bpy.types.Scene.blendcap_retarget_use_world_location
    del bpy.types.Scene.blendcap_retarget_ik_chains
    del bpy.types.Scene.blendcap_retarget_ik_chains_index
    del bpy.types.Scene.blendcap_retarget_ik_expanded
    del bpy.types.Scene.blendcap_retarget_auto_bake_ik
    del bpy.types.Scene.blendcap_ik_src_left_arm_end
    del bpy.types.Scene.blendcap_ik_src_right_arm_end
    del bpy.types.Scene.blendcap_ik_src_left_leg_end
    del bpy.types.Scene.blendcap_ik_src_right_leg_end
    del bpy.types.Scene.blendcap_ik_src_left_arm_pole
    del bpy.types.Scene.blendcap_ik_src_right_arm_pole
    del bpy.types.Scene.blendcap_ik_src_left_leg_pole
    del bpy.types.Scene.blendcap_ik_src_right_leg_pole
    del bpy.types.Scene.blendcap_retarget_ik_src_expanded
    del bpy.types.Scene.blendcap_use_custom_ik_sources
    del bpy.types.Object.blendcap_camera_offset
    del bpy.types.Scene.blendcap_camera_offset_placeholder
