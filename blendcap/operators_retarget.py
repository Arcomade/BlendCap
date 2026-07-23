"""Retarget operators + the UIList for the bone-pair table.

Operators that previously
did `global _LOADING_PRESET; _LOADING_PRESET = True` now write the flag
through the module attribute (`retarget._LOADING_PRESET = True`) — the
loading guard MUST stay in retarget.py so mark_dirty / update_preset_selection
read the same value the operators set.
"""
import bpy
import os
import json

from . import retarget
from .helpers import get_3d_override


# Module-level flags for the Apply Retargeting button → Cancel button
# swap. `_BAKE_RUNNING` is set by _prepare on success and cleared by
# _finalize, so the panel can pick which operator to draw.
# `_BAKE_CANCEL_REQUESTED` is the cross-thread (well, cross-event)
# signal the cancel button raises; the modal checks it each tick and
# treats it like an ESC press. Both are simple booleans because a
# retarget is single-instance — there's no second bake to disambiguate.
_BAKE_RUNNING = False
_BAKE_CANCEL_REQUESTED = False
from .helpers import retarget_progress_state, format_workspace_status
from .retarget import (
    SHIPPED_PRESET_FILENAMES,
    _sanitize_preset_filename,
    _retarget_maps_dir_shipped,
    _retarget_maps_dir_user,
    _save_retarget_map,
    _populate_pairs_from_map,
    _auto_match_pairs,
    _ordered_bones_for_add_all,
    _resolve_against_rig,
    _resolve_pair_names_for_bake,
    _restore_pair_names_after_bake,
    _compute_rig_height,
    _matrix_bake_pairs_iter,
    _frame_range_for,
)


def _validate_rig_slot(scene, obj, label):
    """Return a user-facing error string when the rig slot is unusable,
    else None. `label` is "Source" / "Target" — folded into the message.

    Four failure modes, in priority order:
      1. Slot empty — user never picked a rig.
      2. Object datablock exists (the PointerProperty still holds a
         reference) but the rig has been deleted from the scene. This
         is the trap that motivated the proper-error work: `obj.type`
         is still 'ARMATURE' so a naive check passes, but the bake
         eventually crashes when bpy.ops.object.mode_set can't find
         the object in the active view layer.
      3. Object is in the scene but not in the active view layer (its
         collection is excluded). mode_set rejects this too.
      4. Object is the wrong type. Shouldn't happen via the UI because
         the PointerProperty has poll=type=='ARMATURE', but we cover
         it for scripted callers."""
    if obj is None:
        return f"No {label.lower()} rig assigned — pick one in the Retarget panel."
    name = getattr(obj, "name", "?")
    if name not in scene.objects:
        return (f"{label} rig '{name}' was deleted from the scene. "
                f"Pick a new rig in the Retarget panel.")
    try:
        view_layer = bpy.context.view_layer
        in_view_layer = (name in view_layer.objects)
    except Exception:
        in_view_layer = True
    if not in_view_layer:
        return (f"{label} rig '{name}' is in a collection excluded from "
                f"the current view layer. Enable its collection before baking.")
    if getattr(obj, "type", None) != 'ARMATURE':
        return f"{label} rig '{name}' isn't an armature."
    return None


def _force_cloudrig_fk(rig):
    """CloudRig counterpart of the Rigify/ARP FK-switch flip (same
    Protocols carve-out, set-only, never keyed). CloudRig differs from
    both in two load-bearing ways:

      1. INVERTED semantics: per-limb switches are custom props named
         ik_<side>_<limb> (e.g. ik_left_arm) on a central 'Properties'
         bone, where 1.0 = IK and 0.0 = FK — so the generic
         _FK_SWITCH_KEYS loop (which writes 1.0) must NOT see them.
      2. The FK control chain itself (UpperArm.L / Forearm.L / Hand.L)
         carries 'Copy Transforms IK' constraints whose influence is
         driven by those switches. In IK mode the FK controls are
         constraint-slaved to IK mechanism bones the fast engines
         can't simulate (real IK solver), so BOTH the FK bake's
         visibility AND the FK->IK convert's chain read require the
         switches at 0.0.

    Gated on the CloudRig marker custom property (object or armature
    data) so no other rig family's 'ik_*' props get touched. Excluded
    prefixes are CloudRig's non-switch ik_ prop families (stretch /
    parent-switch index / pole-follow). Numeric props only; type-
    preserving assignment. Returns True when anything changed (caller
    should let drivers re-evaluate before scanning influences)."""
    if rig is None or getattr(rig, 'type', None) != 'ARMATURE':
        return False
    try:
        is_cloudrig = ('cloudrig' in rig.keys()
                       or 'cloudrig' in rig.data.keys())
    except Exception:
        is_cloudrig = False
    if not is_cloudrig:
        return False
    non_switch = ('ik_stretch', 'ik_parents', 'ik_pole_follow', 'ik_hinge')
    changed = False
    for pb in rig.pose.bones:
        for key in list(pb.keys()):
            if not key.startswith('ik_') or key.startswith(non_switch):
                continue
            try:
                value = pb[key]
                if not isinstance(value, (int, float)):
                    continue
                if value != 0:
                    pb[key] = type(value)(0)
                    changed = True
            except Exception:
                pass
    return changed


# ============================================================
# RETARGET OPERATORS
# ============================================================
class BLENDCAP_OT_retarget_save_preset(bpy.types.Operator):
    bl_idname = "blendcap.retarget_save_preset"
    bl_label = "Save Map As..."
    bl_description = "Save the current bone-mapping table as a preset"
    bl_options = {'REGISTER'}

    map_name: bpy.props.StringProperty(name="Preset Name", default="Untitled Map")
    # Hidden flag set by the overwrite-confirm chain to bypass the
    # collision check on the second pass. SKIP_SAVE so it doesn't
    # persist into the Redo panel.
    confirmed: bpy.props.BoolProperty(
        default=False, options={'HIDDEN', 'SKIP_SAVE'})

    def _default_name(self, context):
        """Pre-fill order: loaded-preset stem (so save = overwrite),
        else '<target rig name> Map', else 'Untitled Map'.

        The "loaded preset" check has two sources because they cover
        different cases:
          (a) blendcap_retarget_last_loaded_preset survives the dropdown
              flip to __UNSAVED__ after edits — covers the load-then-
              edit-then-save flow.
          (b) blendcap_retarget_preset (the dropdown itself) covers the
              case where a preset is the dropdown's restored value from
              a previous session: Blender's EnumProperty update doesn't
              fire on initial-value restore, so last_loaded stays blank.
        Either source is valid; (a) wins when both are set because it's
        more specific (it records the LAST explicit load action).
        Standard / shipped presets resolve here too — letting the field
        show the Standard name (red, with the "Can't overwrite" hint)
        signals that the user already has an exact copy on disk."""
        scene = context.scene
        for candidate in (
            scene.blendcap_retarget_last_loaded_preset,
            scene.blendcap_retarget_preset,
        ):
            if candidate and not candidate.startswith('__') \
                    and os.path.isfile(candidate):
                stem, _ext = os.path.splitext(os.path.basename(candidate))
                if stem:
                    return stem
        tgt = scene.blendcap_retarget_target
        if tgt and getattr(tgt, "type", None) == 'ARMATURE' and tgt.name:
            return f"{tgt.name} Map"
        return "Untitled Map"

    def _resolved(self):
        """Return (filename, path, collision_kind) for the current name.
        collision_kind is 'shipped' / 'custom' / None. shipped wins over
        custom when the sanitized filename matches a SHIPPED entry, since
        we refuse to overwrite shipped presets regardless of which folder
        the on-disk file happens to live in."""
        filename = _sanitize_preset_filename(self.map_name)
        if not filename:
            return "", "", None
        path = os.path.join(_retarget_maps_dir_shipped(), filename)
        if filename in SHIPPED_PRESET_FILENAMES:
            return filename, path, 'shipped'
        if os.path.exists(path):
            return filename, path, 'custom'
        return filename, path, None

    def invoke(self, context, event):
        # Show a small props dialog with just the name field. No file
        # picker — saves auto-route to the shipped maps folder so user
        # presets sit next to the standard ones.
        # Pre-fill the name with the loaded preset (so Save = Overwrite
        # in the common edit-then-save workflow) or "<target rig> Map"
        # when nothing is loaded. See _default_name.
        self.map_name = self._default_name(context)
        # Red name field in draw() is the sole collision signal. We
        # used to also flip the OK button label to "Overwrite" via
        # confirm_text, but Blender locks confirm_text at dialog-open
        # time -- the label stayed "Overwrite" even after the user
        # typed a non-colliding name, which was actively misleading.
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        _, _, kind = self._resolved()
        row = layout.row()
        row.alert = kind is not None
        row.prop(self, "map_name")
        if kind == 'shipped':
            warn = layout.row()
            warn.alert = True
            warn.label(text="Can't overwrite Standard presets.",
                       icon='ERROR')

    def execute(self, context):
        scene = context.scene
        filename, path, kind = self._resolved()
        if not filename:
            self.report({'ERROR'}, "Preset name is required")
            return {'CANCELLED'}

        # Refuse to overwrite shipped presets. SHIPPED_PRESET_FILENAMES
        # is filename-based (not folder-based), so a custom preset
        # using a shipped filename would also be blocked — by design,
        # to keep shipped names reserved.
        if kind == 'shipped':
            self.report({'ERROR'},
                        "Can't overwrite Standard presets.")
            return {'CANCELLED'}

        # Overwrite confirmation for custom collisions. Pops the
        # confirm operator, which re-invokes this one with
        # confirmed=True via EXEC_DEFAULT on Yes.
        if kind == 'custom' and not self.confirmed:
            bpy.ops.blendcap.retarget_save_preset_confirm(
                'INVOKE_DEFAULT', map_name=self.map_name)
            return {'FINISHED'}

        # Custom maps save into the running addon's retarget_maps/ folder
        # (alongside the shipped presets), via the `path` returned by
        # _resolved(). The combined preset dropdown reads both this folder
        # and Blender's user config dir, so customs saved by either path
        # appear in the list.
        try:
            _save_retarget_map(
                path, self.map_name or "Unnamed",
                scene.blendcap_retarget_map_hint,
                scene.blendcap_retarget_namespace_strip,
                scene.blendcap_retarget_pairs,
                auto_bake_ik=scene.blendcap_retarget_auto_bake_ik,
                ik_chains_collection=scene.blendcap_retarget_ik_chains,
                face_head_source=scene.blendcap_retarget_face_head_source,
                face_head_target=scene.blendcap_retarget_face_head_target,
                source_prefix=scene.blendcap_retarget_source_prefix,
                face_capture_compensation=scene.blendcap_face_capture_compensation,
                face_amplification={
                    r: getattr(scene, f"blendcap_face_amp_{r}")
                    for r in ("jaw", "gaze", "lids", "brows", "lips", "cheeks", "nose", "tongue")
                },
                face_amp_global=scene.blendcap_face_amp_global,
                use_world_location=scene.blendcap_retarget_use_world_location,
                custom_ik_sources={
                    "enabled":       scene.blendcap_use_custom_ik_sources,
                    "left_hand":     scene.blendcap_ik_src_left_arm_end,
                    "right_hand":    scene.blendcap_ik_src_right_arm_end,
                    "left_foot":     scene.blendcap_ik_src_left_leg_end,
                    "right_foot":    scene.blendcap_ik_src_right_leg_end,
                    "left_forearm":  scene.blendcap_ik_src_left_arm_pole,
                    "right_forearm": scene.blendcap_ik_src_right_arm_pole,
                    "left_shin":     scene.blendcap_ik_src_left_leg_pole,
                    "right_shin":    scene.blendcap_ik_src_right_leg_pole,
                },
            )
        except Exception as e:
            self.report({'ERROR'}, f"Save failed: {e}")
            return {'CANCELLED'}

        # Point the dropdown at the freshly-saved preset. Guard so the
        # update callback doesn't re-load a file we just wrote from
        # current state (wasteful no-op).
        retarget._LOADING_PRESET = True
        try:
            scene.blendcap_retarget_preset = path
        finally:
            retarget._LOADING_PRESET = False
        scene.blendcap_retarget_last_loaded_preset = path

        self.report({'INFO'}, f"Saved map to {path}")
        return {'FINISHED'}


class BLENDCAP_OT_retarget_save_preset_confirm(bpy.types.Operator):
    """Tiny "are you sure?" wrapper invoked by retarget_save_preset
    when the user's chosen filename collides with an existing CUSTOM
    preset. Shipped collisions never reach this — they're rejected
    upstream by retarget_save_preset.execute(). On Yes, re-invokes
    the save operator with confirmed=True via EXEC_DEFAULT so the
    second pass bypasses the collision check."""
    bl_idname = "blendcap.retarget_save_preset_confirm"
    bl_label = "Overwrite Existing Preset?"
    bl_description = "Replace the existing bone map preset with the same name"
    bl_options = {'REGISTER', 'INTERNAL'}

    map_name: bpy.props.StringProperty()

    def invoke(self, context, event):
        wm = context.window_manager
        msg = f"Overwrite '{self.map_name}'?"
        try:
            return wm.invoke_confirm(self, event, message=msg,
                                     title="Overwrite Existing Preset?")
        except TypeError:
            return wm.invoke_confirm(self, event)

    def execute(self, context):
        bpy.ops.blendcap.retarget_save_preset(
            'EXEC_DEFAULT', map_name=self.map_name, confirmed=True)
        return {'FINISHED'}


class BLENDCAP_OT_retarget_reload_preset(bpy.types.Operator):
    """Re-read the currently-selected preset from disk. Useful if the
    preset JSON has been edited externally, or after swapping rigs to
    refresh the panel's "N / M pairs valid" count against the new bones."""
    bl_idname = "blendcap.retarget_reload_preset"
    bl_label = "Reload Preset"
    bl_description = "Re-read the selected preset from disk."
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        sel = scene.blendcap_retarget_preset
        # The dropdown uses file paths as item IDs; sentinels (__UNSAVED__,
        # __HDR_*__, __EMPTY_*__) all start with "__".
        if not sel or sel.startswith('__') or not os.path.isfile(sel):
            self.report({'WARNING'},
                        "No preset selected — pick one from the dropdown first")
            return {'CANCELLED'}
        try:
            with open(sel, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to read preset: {e}")
            return {'CANCELLED'}
        _populate_pairs_from_map(scene, data)
        scene.blendcap_retarget_last_loaded_preset = sel
        pairs = scene.blendcap_retarget_pairs
        src_rig = scene.blendcap_retarget_source
        tgt_rig = scene.blendcap_retarget_target
        valid = 0
        for p in pairs:
            sv = (p.source and src_rig and src_rig.type == 'ARMATURE'
                  and p.source in src_rig.data.bones)
            tv = (p.target and tgt_rig and tgt_rig.type == 'ARMATURE'
                  and p.target in tgt_rig.data.bones)
            if sv and tv:
                valid += 1
        self.report({'INFO'},
                    f"Reloaded preset: {valid} / {len(pairs)} pairs valid "
                    f"against current rigs")
        return {'FINISHED'}


class BLENDCAP_OT_retarget_delete_preset(bpy.types.Operator):
    """Delete the currently-selected retarget preset. Shipped (Standard)
    presets are protected and cannot be deleted."""
    bl_idname = "blendcap.retarget_delete_preset"
    bl_label = "Delete Retarget Preset"
    bl_description = "Delete the selected preset"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        sel = getattr(context.scene, 'blendcap_retarget_preset', '')
        if not sel or sel.startswith('__'):
            return False
        # Gray out the button on shipped presets so users see the
        # protection up-front instead of after clicking + confirm.
        if os.path.basename(sel) in SHIPPED_PRESET_FILENAMES:
            return False
        return True

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        scene = context.scene
        sel = scene.blendcap_retarget_preset
        if not sel or sel.startswith('__'):
            self.report({'ERROR'}, "No preset selected")
            return {'CANCELLED'}
        # Defense in depth: poll() blocks the UI path, but a script
        # could call the operator directly. Refuse here too.
        if os.path.basename(sel) in SHIPPED_PRESET_FILENAMES:
            self.report({'ERROR'},
                        f"'{os.path.basename(sel)}' is a Standard preset "
                        f"and cannot be deleted")
            return {'CANCELLED'}
        if not os.path.isfile(sel):
            self.report({'ERROR'}, f"File not found: {sel}")
            return {'CANCELLED'}

        try:
            os.remove(sel)
        except Exception as e:
            self.report({'ERROR'}, f"Delete failed: {e}")
            return {'CANCELLED'}

        # Reset dropdown selection so the next draw rebuilds the items
        # list without the stale entry. Use the LOADING_PRESET flag so
        # the update callback doesn't try to load a path we just deleted.
        retarget._LOADING_PRESET = True
        try:
            scene.blendcap_retarget_preset = '__UNSAVED__'
        finally:
            retarget._LOADING_PRESET = False

        self.report({'INFO'},
                    f"Deleted preset: {os.path.basename(sel)}")
        return {'FINISHED'}


class BLENDCAP_OT_retarget_auto_match(bpy.types.Operator):
    bl_idname = "blendcap.retarget_auto_match"
    bl_label = "Auto-Match Names"
    bl_description = ("Automatically fill Source/Target pairs wherever "
                      "matching names are found.")
    bl_options = {'REGISTER', 'UNDO'}

    replace: bpy.props.BoolProperty(name="Replace existing", default=False)

    def execute(self, context):
        scene = context.scene
        src = scene.blendcap_retarget_source
        tgt = scene.blendcap_retarget_target
        err = _validate_rig_slot(scene, src, "Source")
        if err is not None:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        err = _validate_rig_slot(scene, tgt, "Target")
        if err is not None:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}

        # Bulk-edit gate: this operator already produces one
        # REGISTER+UNDO step covering the whole match action. Without
        # the gate, every per-field write below would push its own
        # undo step via _mark_dirty_dispatch and one Auto-Match would
        # need N+1 Ctrl+Z presses to walk back. See properties._BULK_
        # EDITING docstring.
        from . import properties as _props
        _props._BULK_EDITING = True
        try:
            pairs = scene.blendcap_retarget_pairs
            if self.replace:
                pairs.clear()

            src_strip = scene.blendcap_retarget_source_prefix or ""
            tgt_strip = scene.blendcap_retarget_namespace_strip or ""
            suggested = dict(_auto_match_pairs(src, tgt, src_strip, tgt_strip))

            # Pass 1: fill empty target fields on existing pairs.
            filled = 0
            for p in pairs:
                if not p.target and p.source in suggested:
                    p.target = suggested[p.source]
                    filled += 1

            # Pass 2: append pairs for source bones not yet represented at all.
            sources_present = {p.source for p in pairs}
            appended = 0
            for s, t in suggested.items():
                if s in sources_present:
                    continue
                p = pairs.add()
                p.source = s
                p.target = t
                p.channels = 'ROT'
                p.influence = 1.0
                appended += 1
        finally:
            _props._BULK_EDITING = False

        msg_parts = []
        if filled:   msg_parts.append(f"filled {filled} empty target(s)")
        if appended: msg_parts.append(f"added {appended} new pair(s)")
        self.report({'INFO'}, "Auto-match: " + (", ".join(msg_parts) or "no matches found"))
        return {'FINISHED'}


class BLENDCAP_OT_retarget_clear_pairs(bpy.types.Operator):
    bl_idname = "blendcap.retarget_clear_pairs"
    bl_label = "Clear Map"
    bl_description = "Remove every pair from the bone-mapping table"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        retarget._LOADING_PRESET = True
        try:
            scene = context.scene
            scene.blendcap_retarget_pairs.clear()
            scene.blendcap_retarget_pair_index = 0
            scene.blendcap_retarget_map_hint = ""
            scene.blendcap_retarget_preset = '__UNSAVED__'
        finally:
            retarget._LOADING_PRESET = False
        return {'FINISHED'}


class BLENDCAP_OT_retarget_add_pair(bpy.types.Operator):
    bl_idname = "blendcap.retarget_add_pair"
    bl_label = "Add Pair"
    bl_description = "Append an empty bone pair to the bottom of the table"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        pairs = context.scene.blendcap_retarget_pairs
        pairs.add()
        context.scene.blendcap_retarget_pair_index = len(pairs) - 1
        return {'FINISHED'}


class BLENDCAP_OT_retarget_add_all_source_bones(bpy.types.Operator):
    bl_idname = "blendcap.retarget_add_all_source_bones"
    bl_label = "Add All Source Bones"
    bl_description = ("Add one pair per source-rig bone with empty target "
                      "fields, skipping any source bone already in the table.")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        src = scene.blendcap_retarget_source
        err = _validate_rig_slot(scene, src, "Source")
        if err is not None:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        # Bulk-edit gate (same reason as Auto-Match): keep this whole
        # action as a single REGISTER+UNDO step rather than N+1.
        from . import properties as _props
        _props._BULK_EDITING = True
        try:
            pairs = scene.blendcap_retarget_pairs
            sources_present = {p.source for p in pairs}

            # Bones come off the rig in their literal (possibly prefixed)
            # form. The bulk-edit gate above suppresses the per-field
            # smart-strip update callback, so do the strip inline here
            # to keep pair storage consistent with manual-entry rows.
            from .retarget.name_matching import _smart_strip
            src_prefix = scene.blendcap_retarget_source_prefix or ""

            bones_sorted = _ordered_bones_for_add_all(src)

            added = 0
            for bone in bones_sorted:
                stored_name = _smart_strip(bone.name, src, src_prefix)
                if stored_name in sources_present:
                    continue
                p = pairs.add()
                p.source = stored_name
                # target intentionally left empty for manual mapping
                added += 1
        finally:
            _props._BULK_EDITING = False
        self.report({'INFO'}, f"Added {added} source bone(s) with empty target")
        return {'FINISHED'}


class BLENDCAP_OT_retarget_remove_pair(bpy.types.Operator):
    bl_idname = "blendcap.retarget_remove_pair"
    bl_label = "Remove Pair"
    bl_description = "Remove this bone pair from the table"
    bl_options = {'REGISTER', 'UNDO'}

    index: bpy.props.IntProperty(default=-1)

    def execute(self, context):
        scene = context.scene
        pairs = scene.blendcap_retarget_pairs
        idx = self.index if self.index >= 0 else scene.blendcap_retarget_pair_index
        if 0 <= idx < len(pairs):
            pairs.remove(idx)
            scene.blendcap_retarget_pair_index = max(0, min(idx, len(pairs) - 1))
        return {'FINISHED'}


class BLENDCAP_OT_retarget_sort_pairs(bpy.types.Operator):
    bl_idname = "blendcap.retarget_sort_pairs"
    bl_label = "Sort Pairs"
    bl_description = ("Reorder the bone map into the standard reading order "
                      "(face, head/neck, left arm, right arm, spine, legs) — "
                      "the same layout 'Add All' produces. Pairs whose source "
                      "bone isn't on the source rig sink to the bottom in "
                      "their current order")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        src = scene.blendcap_retarget_source
        err = _validate_rig_slot(scene, src, "Source")
        if err is not None:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}
        pairs = scene.blendcap_retarget_pairs
        n = len(pairs)
        if n < 2:
            self.report({'INFO'}, "Nothing to sort.")
            return {'CANCELLED'}

        from .retarget.name_matching import _resolve_against_rig
        src_prefix = scene.blendcap_retarget_source_prefix or ""
        # Rank = position in the standardized order (the exact list Add
        # All walks). Pairs store SHORT names; resolve each one against
        # the rig (exact -> prefix+name) before the rank lookup so
        # prefixed rigs sort identically to bare ones.
        rank = {b.name: i
                for i, b in enumerate(_ordered_bones_for_add_all(src))}

        def _key(orig_idx):
            name = _resolve_against_rig(pairs[orig_idx].source, src,
                                        src_prefix)
            # (rank, original index): unknown sources sink below every
            # ranked bone; the index tiebreak keeps the sort stable for
            # duplicates (one source mapped to several targets) and for
            # the unranked tail.
            return (rank.get(name, len(rank) + 1), orig_idx)

        desired = sorted(range(n), key=_key)
        if desired == list(range(n)):
            self.report({'INFO'}, "Bone map is already in standard order.")
            return {'CANCELLED'}

        # Follow the active row to its new slot after the sort.
        active_idx = scene.blendcap_retarget_pair_index

        # Bulk-edit gate (same reason as Auto-Match / Add All): one
        # REGISTER+UNDO step for the whole reorder. collection.move()
        # itself doesn't fire field update callbacks, but the gate keeps
        # this operator consistent with every other bulk mutator.
        from . import properties as _props
        _props._BULK_EDITING = True
        try:
            # Selection-sort by moves: walk the destination slots and
            # pull each slot's intended pair up from wherever it
            # currently sits. `current` mirrors the collection so we can
            # find items by ORIGINAL index across moves.
            current = list(range(n))
            for dest, orig in enumerate(desired):
                cur = current.index(orig)
                if cur != dest:
                    pairs.move(cur, dest)
                    current.insert(dest, current.pop(cur))
        finally:
            _props._BULK_EDITING = False

        if 0 <= active_idx < n:
            scene.blendcap_retarget_pair_index = desired.index(active_idx)
        self.report({'INFO'}, "Bone map sorted into standard order.")
        return {'FINISHED'}


class BLENDCAP_OT_apply_retarget(bpy.types.Operator):
    """Apply the current bone-mapping table from source rig to target rig.

    Uses world-space delta-from-rest math (see _matrix_bake_pairs_iter) so
    cross-skeleton rest mismatches (different rolls, T-pose vs A-pose,
    short source stubs vs tall target bones) don't produce flipped arms
    or inverted spines.

    Modal/timer-driven: invoke() does all upfront setup synchronously
    (mode changes, auto-scale), then hands off to a per-frame coroutine
    that yields once per baked frame. The TIMER tick advances N frames
    before returning to Blender's main loop, so the UI stays responsive
    and the user sees a "frame x/y" status line + progress bar instead
    of a "Not Responding" window. ESC cancels mid-bake; cleanup (restore
    source scale) runs on every exit path.

    Auto-scale (visible checkbox) measures each rig's vertical extent
    via pose-bone heads and scales the source so motion lands at the
    target's scale — feet on floor regardless of character height.

    "Use current source pose as rest" samples the source's live pose at
    Apply time as its rest baseline instead of its edit-bone rest.
    Useful when auto-alignment produces bad results on odd rigs: user
    poses source to match target's rest, clicks Apply.
    """
    bl_idname = "blendcap.apply_retarget"
    bl_label = "Apply Retargeting"
    bl_description = "Bake the source rig's animation onto the target rig using the bone-mapping table."
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        # Grey the button when either rig slot is unusable, mirroring
        # FK->IK. Reuses _validate_rig_slot so the gate matches the
        # error path exactly — no risk of a clickable button that
        # immediately errors. Cheap: two attribute reads + a dict-style
        # lookup against scene.objects per redraw.
        scene = context.scene
        if _validate_rig_slot(scene, scene.blendcap_retarget_source, "Source") is not None:
            return False
        if _validate_rig_slot(scene, scene.blendcap_retarget_target, "Target") is not None:
            return False
        return True

    # ----- modal state (instance attrs, set in invoke) -----
    _timer = None
    _coroutine = None
    _src = None
    _tgt = None
    _src_orig_scale = None
    _scene = None
    _wm = None
    _frame_start = None
    _frames_done = 0
    _frames_total = 0
    # Captured at _prepare entry, restored in _finalize. Lets the user
    # come back to whatever mode + active object they had at click time
    # instead of being dumped into POSE on the target rig.
    _prev_mode = None
    _prev_active = None
    _prev_selected = None
    # Hide-state snapshot for src/tgt. mode_set.poll() rejects hidden
    # active objects, so we temporarily un-hide both rigs for the bake
    # and restore on _finalize.
    _hide_snapshot = None  # list of (obj, hide_viewport, hide_get())
    # Snapshot of pre-resolution pair source/target names. The bake
    # engines iterate pair fields directly, so we resolve each name to
    # its rig bone (prefix-prepended) for the bake's duration and
    # restore the SHORT stored form in _finalize. See
    # retarget._resolve_pair_names_for_bake.
    _pair_name_snapshot = None

    # Frames advanced per timer tick. Higher = faster bake but coarser
    # responsiveness. 4 keeps UI snappy on a ~60-frame clip while still
    # finishing quickly on a 300-frame clip.
    FRAMES_PER_TICK = 4

    def execute(self, context):
        # Synchronous fallback (e.g. when called from a Python script via
        # bpy.ops.blendcap.apply_retarget() with no event). Iterates the
        # coroutine inline. UI will block during this — use invoke from UI.
        try:
            ok, err = self._prepare(context)
        except Exception as e:
            self.report({'ERROR'}, f"Retargeting setup failed: {e}")
            self._finalize(context)
            return {'CANCELLED'}
        if not ok:
            self.report({'ERROR'}, err)
            self._finalize(context)
            return {'CANCELLED'}
        try:
            for _ in self._coroutine:
                pass
        except Exception as e:
            self._finalize(context)
            self.report({'ERROR'}, f"Retargeting failed: {e}")
            return {'CANCELLED'}
        self._finalize(context)
        self.report({'INFO'}, "Retargeting complete")
        return {'FINISHED'}

    def invoke(self, context, event):
        try:
            ok, err = self._prepare(context)
        except Exception as e:
            self.report({'ERROR'}, f"Retargeting setup failed: {e}")
            self._finalize(context)
            return {'CANCELLED'}
        if not ok:
            self.report({'ERROR'}, err)
            self._finalize(context)
            return {'CANCELLED'}
        wm = context.window_manager
        self._wm = wm
        self._timer = wm.event_timer_add(0.03, window=context.window)
        wm.modal_handler_add(self)
        # No wm.progress_begin/update/end calls — those cause Blender to
        # show a numeric percentage in the cursor during the modal bake,
        # which the user finds distracting. Status text + percentage in
        # status bar (set elsewhere via status_text_set) is enough.
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        global _BAKE_CANCEL_REQUESTED
        if event.type == 'ESC' or _BAKE_CANCEL_REQUESTED:
            self.report({'WARNING'}, "Retargeting cancelled")
            self._finalize(context)
            return {'CANCELLED'}
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        try:
            for _ in range(self.FRAMES_PER_TICK):
                # The bake coroutine yields a frame number per baked frame
                # during the per-frame phase, then yields None per chunk
                # during the keyframe-flush tail. Only count frame-number
                # yields toward _frames_done so the status text doesn't
                # blow past _frames_total during the flush.
                yielded = next(self._coroutine)
                if yielded is not None:
                    self._frames_done += 1
        except StopIteration:
            self._finalize(context)
            self.report({'INFO'}, "Retargeting complete")
            # Auto-bake IK on apply: opt-in via panel checkbox or
            # preset. Fires AFTER _finalize so the user's mode/active/
            # selection have been restored and any intermediate rig
            # cleaned up. Convert FK->IK switches into POSE mode itself
            # for the duration of the bake. Failure here doesn't roll
            # back the retarget — surface as a WARNING so the FK keys
            # still land cleanly even if FK->IK has a configuration
            # issue (missing IK control mapping, etc.).
            if context.scene.blendcap_retarget_auto_bake_ik:
                try:
                    # INVOKE_DEFAULT so the FK->IK conversion runs as
                    # a modal too — preserves UI responsiveness, the
                    # progress bar, and the ESC/Cancel button while
                    # auto-bake is mid-flight. (EXEC_DEFAULT used to
                    # block the UI here.)
                    bpy.ops.blendcap.convert_fk_to_ik('INVOKE_DEFAULT')
                except Exception as e:
                    self.report({'WARNING'},
                                f"Auto-bake IK skipped: {e}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Retargeting failed: {e}")
            self._finalize(context)
            return {'CANCELLED'}
        try:
            if self._frames_done >= self._frames_total:
                # Per-frame phase finished; we're in the keyframe-flush tail.
                pct = 99.0
                short = "Writing keyframes…  (ESC to cancel)"
            else:
                pct = (100.0 * self._frames_done
                       / max(self._frames_total, 1))
                short = (f"Frame {self._frames_done}/"
                         f"{self._frames_total}  (ESC to cancel)")
            context.workspace.status_text_set(
                format_workspace_status(
                    "BlendCap Retarget", pct, short))
            retarget_progress_state.update({
                'running': True,
                'phase': "Retargeting",
                'progress': min(pct, 99.9),
                'status_text': short,
                'frames_done': self._frames_done,
                'frames_total': self._frames_total,
            })
            # Nudge the N-panel so its progress label refreshes between
            # ticks — the modal timer alone doesn't trigger a UI redraw.
            try:
                for window in context.window_manager.windows:
                    for area in window.screen.areas:
                        if area.type == 'VIEW_3D':
                            for region in area.regions:
                                if region.type == 'UI':
                                    region.tag_redraw()
            except Exception:
                pass
        except Exception:
            pass
        return {'PASS_THROUGH'}

    # ----- internals -----

    def _prepare(self, context):
        scene = context.scene
        self._scene = scene
        src = scene.blendcap_retarget_source
        tgt = scene.blendcap_retarget_target
        err = _validate_rig_slot(scene, src, "Source")
        if err is not None:
            return False, err
        err = _validate_rig_slot(scene, tgt, "Target")
        if err is not None:
            return False, err
        pairs = list(scene.blendcap_retarget_pairs)
        if not pairs:
            return False, ("Bone map is empty. Load a preset from the "
                           "dropdown or click Auto-Match first.")

        # Resolve stored SHORT names ("RightLeg") to actual rig bone
        # names ("mixamorig:RightLeg") for the duration of the bake.
        # Snapshot is restored in _finalize. Done BEFORE source_rest_
        # override below so its `src.pose.bones[p.source]` lookups see
        # resolved names.
        self._pair_name_snapshot = _resolve_pair_names_for_bake(scene)

        self._src = src
        self._tgt = tgt
        self._src_orig_scale = tuple(src.scale)

        # Capture pre-bake context so _finalize can put the user back where
        # they started. context.mode returns names like 'EDIT_ARMATURE' /
        # 'EDIT_MESH' / 'POSE' / 'OBJECT' — bpy.ops.object.mode_set wants
        # the short form ('EDIT'/'POSE'/'OBJECT'), so collapse EDIT_*.
        try:
            mode = context.mode
            self._prev_mode = 'EDIT' if mode.startswith('EDIT_') else mode
        except Exception:
            self._prev_mode = 'OBJECT'
        try:
            self._prev_active = context.view_layer.objects.active
        except Exception:
            self._prev_active = None
        try:
            self._prev_selected = [o for o in context.view_layer.objects if o.select_get()]
        except Exception:
            self._prev_selected = None

        auto_scale = bool(scene.blendcap_retarget_auto_scale)
        use_current_pose = bool(scene.blendcap_retarget_use_current_pose_as_rest)
        current_pose_full_matrix = bool(scene.blendcap_retarget_current_pose_full_matrix)
        props = scene.blendcap_props
        use_custom_rest_pose = bool(getattr(props, 'use_custom_rest_pose', False))
        custom_rest_pose_path = getattr(props, 'custom_rest_pose_preset', '__NONE__')

        # Snapshot source pose BEFORE any frame_set / scale change, so the
        # user's manual pose at click-time isn't wiped by curve evaluation.
        # The two paths feeding `source_rest_override` are mutually exclusive
        # at the UI level (toggling one auto-clears the other) — we still
        # check both defensively and prefer the live snapshot if anything
        # ever bypasses the UI guard.
        source_rest_override = None
        if use_current_pose:
            source_rest_override = {}
            for p in pairs:
                if p.source and p.source in src.pose.bones:
                    source_rest_override[p.source] = src.pose.bones[p.source].matrix.copy()
        elif use_custom_rest_pose and custom_rest_pose_path \
                and not custom_rest_pose_path.startswith('__') \
                and os.path.isfile(custom_rest_pose_path):
            from .operators_rest_pose import load_rest_pose_preset_to_override
            try:
                source_rest_override = load_rest_pose_preset_to_override(
                    custom_rest_pose_path, src)
                if not source_rest_override:
                    source_rest_override = None
                    self.report({'WARNING'},
                                "Custom rest pose preset has no entries "
                                "matching the source rig's bones — skipping.")
            except Exception as e:
                source_rest_override = None
                self.report({'WARNING'},
                            f"Failed to load rest pose preset: {e}")

        with get_3d_override(context):
            frame_start, frame_end = _frame_range_for(src, scene)
            self._frame_start = frame_start
            scene.frame_set(frame_start)
            context.view_layer.update()

            # bpy.ops.object.mode_set.poll() requires context.active_object
            # to be non-None AND visible in the viewport. Two failure modes
            # we have to defend against:
            #   1) Active slot empty (previously-active object deleted /
            #      excluded between sessions, common after re-importing the
            #      BVH). Fix: promote tgt to active up-front.
            #   2) tgt or src is hidden (hide_viewport / hide_get()) — the
            #      poll rejects mode_set on hidden objects, with the same
            #      "Context missing active object" surface error. Fix:
            #      snapshot hide state, un-hide for the bake, restore in
            #      _finalize. Also un-hide src so the per-frame source
            #      pose-bone reads work.
            self._hide_snapshot = [
                (o, o.hide_viewport, o.hide_get()) for o in (src, tgt)
            ]
            for o in (src, tgt):
                if o.hide_viewport:
                    o.hide_viewport = False
                if o.hide_get():
                    o.hide_set(False)

            if context.view_layer.objects.active is None:
                context.view_layer.objects.active = tgt
            if context.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.select_all(action='DESELECT')
            tgt.select_set(True)
            context.view_layer.objects.active = tgt

            bpy.ops.object.mode_set(mode='POSE')
            # Force any FK/IK switch custom properties to FK so the
            # bake's FK keys are visible in the viewport (otherwise, an
            # animator who left the rig in IK mode from a prior session
            # would see deform bones following IK while we write FK).
            # We SET the value only — don't key it. Animators add their
            # own switch keys when they want IK takeover; we shouldn't
            # pollute the action with a frame-start key they didn't ask
            # for.
            #
            # Two known conventions, both with FK_value = 1.0:
            #   "IK_FK"          — Rigify
            #   "ik_fk_switch"   — ARP (set on every IK control AND on
            #                     each finger IK control, which is why
            #                     finger FK retarget on ARP requires
            #                     this flip)
            _FK_SWITCH_KEYS = ("IK_FK", "ik_fk_switch")
            for pb in tgt.pose.bones:
                for key in _FK_SWITCH_KEYS:
                    if key in pb.keys():
                        try:
                            pb[key] = 1.0
                        except Exception:
                            pass
            # CloudRig convention: ik_<side>_<limb> props with INVERTED
            # semantics (0.0 = FK) — see _force_cloudrig_fk. Required
            # there, not just cosmetic: CloudRig's FK controls are
            # constraint-slaved to un-simulatable IK mech bones while
            # the switch is 1.0. The engines re-evaluate drivers before
            # scanning constraint influences, so setting the props here
            # is sufficient.
            _force_cloudrig_fk(tgt)

            if auto_scale:
                src_h = _compute_rig_height(src)
                tgt_h = _compute_rig_height(tgt)
                if src_h > 1e-6 and tgt_h > 1e-6:
                    ratio = tgt_h / src_h
                    bpy.ops.object.mode_set(mode='OBJECT')
                    # Multiply, don't overwrite. Mixamo sources sit at
                    # ~(0.01, 0.01, 0.01) to compensate for a 100x rest
                    # pose; clobbering that scale exposes the giant rest
                    # to world-space sampling and breaks the world-LOC
                    # toggle.
                    sx, sy, sz = self._src_orig_scale
                    src.scale = (sx * ratio, sy * ratio, sz * ratio)
                    context.view_layer.update()
                    bpy.ops.object.select_all(action='DESELECT')
                    tgt.select_set(True)
                    context.view_layer.objects.active = tgt
                    bpy.ops.object.mode_set(mode='POSE')

            # No source-rig Z alignment shift. The bake transfers per-bone
            # BASIS values, which are independent of source-object world
            # position — so shifting source.location.z is purely cosmetic
            # (visual overlay of source on target during the bake) and
            # never reaches the target keyframes. The earlier alignment
            # produced a confusing visible "jump" on the source rig at
            # the start of every retarget. If a future use case needs
            # side-by-side overlay, do it as an opt-in toggle.

            phase_frames = (frame_end - frame_start + 1)

            self._coroutine = _matrix_bake_pairs_iter(
                pairs, src, tgt, frame_start, frame_end,
                mapped_bones=set(),
                source_rest_override=source_rest_override,
                override_rotation_only=not current_pose_full_matrix,
            )
            self._frames_total = phase_frames

        global _BAKE_RUNNING, _BAKE_CANCEL_REQUESTED
        _BAKE_RUNNING = True
        _BAKE_CANCEL_REQUESTED = False
        return True, None

    def _finalize(self, context):
        """Always-safe cleanup. Idempotent — called from FINISHED, CANCELLED,
        and exception paths. Restores source scale, returns scene to start
        frame, removes timer + status text."""
        wm = self._wm or context.window_manager
        if self._timer is not None:
            try:
                wm.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None
        try:
            context.workspace.status_text_set(None)
        except Exception:
            pass

        if self._src is not None and self._src_orig_scale is not None:
            try:
                self._src.scale = self._src_orig_scale
            except Exception:
                pass

        if self._scene is not None and self._frame_start is not None:
            try:
                self._scene.frame_set(self._frame_start)
            except Exception:
                pass

        # Restore the user's pre-bake mode + active object + selection.
        # The mode change has to happen with the captured active object
        # active (mode_set operates on context's active), so set active
        # first, then mode_set, then re-apply the selection set.
        try:
            with get_3d_override(context):
                if self._prev_active is not None:
                    try:
                        context.view_layer.objects.active = self._prev_active
                    except Exception:
                        pass
                if self._prev_mode:
                    try:
                        bpy.ops.object.mode_set(mode=self._prev_mode)
                    except Exception:
                        pass
                if self._prev_selected is not None:
                    try:
                        bpy.ops.object.select_all(action='DESELECT')
                        for o in self._prev_selected:
                            try:
                                o.select_set(True)
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            pass

        # Restore hide state on src/tgt that we un-hid in _prepare so the
        # mode_set polls would pass. Done AFTER the mode/active restore
        # above — hiding before restoring active would re-trigger the
        # poll-fail dance on the next operator the user runs.
        if self._hide_snapshot:
            for o, prev_hide_viewport, prev_hide_get in self._hide_snapshot:
                try:
                    if o.hide_viewport != prev_hide_viewport:
                        o.hide_viewport = prev_hide_viewport
                except Exception:
                    pass
                try:
                    if o.hide_get() != prev_hide_get:
                        o.hide_set(prev_hide_get)
                except Exception:
                    pass

        # Restore pair source/target fields to their stored SHORT names.
        # Done after the bake completes (or after cancellation/error) so
        # the bake engines saw resolved names throughout, but the UI
        # returns to showing the short stored form. See
        # retarget._resolve_pair_names_for_bake.
        try:
            _restore_pair_names_after_bake(self._pair_name_snapshot)
        except Exception:
            pass
        self._pair_name_snapshot = None

        # Drop refs so the operator instance can be GC'd cleanly.
        self._coroutine = None
        self._wm = None
        self._prev_active = None
        self._prev_selected = None
        self._prev_mode = None
        self._hide_snapshot = None

        # Clear the bake-active flags so the panel swaps the Cancel
        # button back to Apply Retargeting on the next redraw. Tag
        # VIEW_3D UI regions explicitly — the modal exits cleanly
        # without a click event and otherwise the N-panel would keep
        # showing Cancel until the user moves the mouse over it.
        global _BAKE_RUNNING, _BAKE_CANCEL_REQUESTED
        _BAKE_RUNNING = False
        _BAKE_CANCEL_REQUESTED = False
        # Clear the shared progress state — but only if WE own it (it
        # could already have been re-tagged by the auto-bake FK->IK
        # follow-on, in which case we leave its 'Converting FK->IK'
        # state alone).
        if retarget_progress_state.get('phase') == "Retargeting":
            retarget_progress_state.update({
                'running': False, 'phase': "", 'progress': 0.0,
                'status_text': "", 'frames_done': 0, 'frames_total': 0,
            })
        try:
            for window in context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        for region in area.regions:
                            if region.type == 'UI':
                                region.tag_redraw()
        except Exception:
            pass


class BLENDCAP_OT_cancel_retarget(bpy.types.Operator):
    """Set the cancel flag the apply_retarget modal checks each tick.
    The actual mode/rig/timer cleanup happens inside _finalize on the
    apply operator — we just trip the signal here, identical to a user
    pressing ESC over the viewport. No-op when no bake is running."""
    bl_idname = "blendcap.cancel_retarget"
    bl_label = "Cancel Retargeting"
    bl_description = "Stop the current process"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _BAKE_RUNNING

    def execute(self, context):
        global _BAKE_CANCEL_REQUESTED
        _BAKE_CANCEL_REQUESTED = True
        return {'FINISHED'}


# ============================================================
# RETARGET UI LIST
# ============================================================
def _is_pair_anchor_eligible(item, src_rig, head_src_name, src_prefix=""):
    """True if the Anchor checkbox should be enabled for this pair.

    Anchor (HEAD_LOCAL) only makes sense for face bones, because the
    math measures source's motion relative to the source head bone. A
    body bone like LeftHand would get its position computed relative
    to the head, which gives a hand that follows head movement —
    clearly wrong.

    Check: pair's source bone walks up its parent chain and reaches
    the source head bone. Target side is intentionally NOT checked
    here — target face bones may parent through complex MCH chains
    that don't reach the auto-detected head directly (some ARP
    controls parent to c_head.x rather than head.x), and target-side
    walking would produce false negatives on legitimate anchor
    candidates. The handful of users who anchor a face source to a
    non-face target get an immediately-visible wrong result, which
    is its own feedback signal.

    `head_src_name` and `item.source` are typically stored SHORT (the
    bone's short name without the rig's namespace prefix). When the
    rig carries a prefix, we resolve both names through `src_prefix`
    so the lookup and the parent-chain comparison still match.
    """
    if src_rig is None or src_rig.type != 'ARMATURE':
        return False
    if not head_src_name:
        return False
    from .retarget import _resolve_against_rig
    head_resolved = _resolve_against_rig(head_src_name, src_rig, src_prefix)
    if head_resolved not in src_rig.data.bones:
        return False
    if not item.source:
        return False
    source_resolved = _resolve_against_rig(item.source, src_rig, src_prefix)
    src_bone = src_rig.data.bones.get(source_resolved)
    if src_bone is None:
        return False
    p = src_bone.parent
    while p is not None:
        if p.name == head_resolved:
            return True
        p = p.parent
    return False


class BLENDCAP_UL_retarget_pairs(bpy.types.UIList):
    """Side-by-side bone mapper row: source | -> | target | channels | anchor | x.

    `prop_search` gives a typeable dropdown bound to each rig's bone
    collection. Rows whose source or target is missing/invalid render
    in alert (red) state.

    Channels and Anchor are UI-only wrappers (channels_ui, anchor_to_
    source) — they read/write the underlying channels+axes / loc_space
    storage so preset JSON format stays unchanged. The Anchor checkbox
    greys out for non-face pairs (pair's source bone doesn't parent
    under the source head bone, per _is_pair_anchor_eligible).
    """
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        scene = context.scene
        src_rig = scene.blendcap_retarget_source
        tgt_rig = scene.blendcap_retarget_target
        # Prefix-aware row validity: a pair stored SHORT ("RightLeg")
        # still counts as valid when the rig carries a namespace
        # ("mixamorig:RightLeg") because the resolver tries exact match
        # first and falls back to prefix+name. Without this the table
        # rendered everything alert-red after the prefix refactor.
        src_prefix = scene.blendcap_retarget_source_prefix or ""
        tgt_prefix = scene.blendcap_retarget_namespace_strip or ""

        def _row_resolves(rig, name, prefix):
            if not (name and rig and rig.type == 'ARMATURE'):
                return False
            bones = rig.data.bones
            return (name in bones) or (bool(prefix) and (prefix + name) in bones)

        src_valid = _row_resolves(src_rig, item.source, src_prefix)
        tgt_valid = _row_resolves(tgt_rig, item.target, tgt_prefix)

        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            # Spreadsheet-style 5-column layout. Uses `row.column()`
            # with matching scale_x ratios on both this row and the
            # header row in panels.py so widgets align in true columns
            # (chained split() factors drifted in practice; column +
            # scale_x is what produces deterministic alignment).
            # IMPORTANT: if you edit scale_x or column count here, edit
            # the matching block in panels.py to keep header alignment.
            # Scale ratios:  Source 12 | Target 12 | Channels 9 | Anchor 3 | X 1
            row = layout.row(align=True)

            c = row.column(align=True); c.scale_x = 12.0
            c.alert = not src_valid
            if src_rig and src_rig.type == 'ARMATURE':
                c.prop_search(item, "source", src_rig.data, "bones", text="")
            else:
                c.prop(item, "source", text="")

            c = row.column(align=True); c.scale_x = 12.0
            c.alert = not tgt_valid
            if tgt_rig and tgt_rig.type == 'ARMATURE':
                c.prop_search(item, "target", tgt_rig.data, "bones", text="")
            else:
                c.prop(item, "target", text="")

            c = row.column(align=True); c.scale_x = 9.0
            c.prop(item, "channels_ui", text="")

            c = row.column(align=True); c.scale_x = 1.1
            head_src_name = getattr(scene, 'blendcap_retarget_face_head_source', '') or ''
            c.enabled = _is_pair_anchor_eligible(item, src_rig, head_src_name, src_prefix)
            c.prop(item, "anchor_to_source", text="")

            c = row.column(align=True); c.scale_x = 1.0
            op = c.operator("blendcap.retarget_remove_pair", text="", icon='X')
            op.index = index

        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="")

    def filter_items(self, context, data, propname):
        """Make the list's built-in search box useful. Blender's default
        implementation matches the typed text against each item's `name`
        property — which bone pairs never set, so every search came back
        empty (and the A-Z sort toggle compared empty strings). Match the
        source AND target bone names instead, case-insensitive substring;
        the A-Z toggle sorts by source name. Blender applies the invert /
        reverse toggles itself on top of what we return."""
        pairs = getattr(data, propname)
        flt_flags = []
        flt_neworder = []
        if self.filter_name:
            needle = self.filter_name.lower()
            flt_flags = [
                self.bitflag_filter_item
                if (needle in (p.source or "").lower()
                    or needle in (p.target or "").lower())
                else 0
                for p in pairs
            ]
        if self.use_filter_sort_alpha:
            flt_neworder = bpy.types.UI_UL_list.sort_items_by_name(
                pairs, "source")
        return flt_flags, flt_neworder
