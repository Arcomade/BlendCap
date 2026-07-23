"""Retargeting package — preset I/O, name matching, per-frame matrix bake
(world-space delta-from-rest math).

Refactored from the original single-file blendcap/retarget.py into focused
submodules. The package re-exports the full public surface here, so every
external caller (operators_retarget.py, operators_ik.py, properties.py,
registration.py) continues to import from `.retarget` unchanged.

Module layout:

    preset_io.py     — preset filenames, sanitize, dir helpers, JSON
                       load/save, face region classification, amplitude /
                       denoise resolvers.
    name_matching.py — namespace strip, normalize, auto-match, effective
                       strip prefixes.
    bone_ordering.py — _ordered_bones_for_add_all (rig bones sorted for
                       the 'Add All Source Bones' table).
    fcurves.py       — _iter_action_fcurves (Blender 4.3/4.4/5 layout
                       compatibility), _bulk_write_keys (priming +
                       foreach_set bulk fill), _accumulate_basis_key
                       (per-frame per-bone key accumulator shared by
                       three of the four bake sites).
    bake_fk_common.py — _compute_scale_ratio, _compute_pair_rest_invariants
                       (shared FK setup math used by both engines).
    bake_fk.py       — _matrix_bake_pairs_iter — the per-frame FK
                       retarget bake (world-space delta-from-rest math),
                       plus _compute_rig_height and _frame_range_for.
    bake_fk_fast.py  — _matrix_bake_pairs_iter_fast — the fcurve-direct
                       FK engine that bypasses scene.frame_set on dense
                       rigs.
    fcurve_eval.py   — _extract_source_fcurves, _eval_basis — pure-math
                       source-bone evaluator (used by both fast engines).
    constraint_sim.py — _SUPPORTED_TYPES, _scan_supported_constraints,
                       _apply_constraint_chain, _compensate_basis_for_
                       constraints, _topo_sort_target_bones — pose-bone
                       constraint simulator for the fast engines.
    bake_ik.py       — _bake_world_pairs_iter — IK control + pole bake
                       driven by the target's retargeted FK chain.
                       Includes Child Of compensation and the canonical
                       pole-position formula.
    bake_ik_fast.py  — _bake_world_pairs_iter_fast — fast FK->IK engine.

The matrix-bake architecture (_matrix_bake_pairs_iter and friends) is the
working in-Blender retargeter. An earlier "v34 attempt 3" replaced this
with a constraint+nla.bake approach that was never tested in Blender; the
v35 refactor accidentally inherited that broken version. v36 restored the
matrix-bake code from BACKUP.py.

The loading-flag state machine (_LOADING_PRESET, _COMBINED_PRESET_ITEMS_CACHE
plus mark_dirty / update_preset_selection / _populate_pairs_from_map) lives
in THIS module because operators outside the package write to
_LOADING_PRESET via attribute assignment:
    from . import retarget
    retarget._LOADING_PRESET = True
They MUST NOT do `from .retarget import _LOADING_PRESET` — that imports
the value, not a binding, and any reassignment goes to the importer's
local name without affecting the module attribute. Keeping the flag and
the functions that mutate it in the same module preserves the
`global _LOADING_PRESET` semantics intact.

Blender's EnumProperty items callback (registration.py wires
get_combined_preset_items as the `items` param) requires the returned
items list + its strings to stay alive across draws — GC'ing them
corrupts dropdown labels. _COMBINED_PRESET_ITEMS_CACHE is the module-
level cache that keeps the list alive.
"""
import os

import bpy

# ----- public re-exports from submodules -----
# Every name below is in the public surface of this package: external
# callers import these via `from .retarget import X` or access them as
# `retarget.X`. Don't remove without checking operators_retarget.py,
# operators_ik.py, properties.py, registration.py.

from .preset_io import (
    SHIPPED_PRESET_FILENAMES,
    FACE_AMPLIFICATION_REGIONS,
    _face_region_from_source,
    _resolve_loc_scale,
    _resolve_denoise,
    _sanitize_preset_filename,
    _retarget_maps_dir_shipped,
    _retarget_maps_dir_user,
    _list_presets_in,
    _load_retarget_map,
    _save_retarget_map,
)

from .name_matching import (
    _strip_namespace,
    _resolve_against_rig,
    _detect_rig_prefix,
    _present_under_any_namespace,
    _smart_strip,
    _normalize_bone_name,
    _auto_match_pairs,
)

from .bone_ordering import _ordered_bones_for_add_all

from .fcurves import (
    _iter_action_fcurves,
    _bulk_write_keys,
)

from .bake_fk import (
    _compute_rig_height,
    _matrix_bake_pairs_iter,
    _frame_range_for,
)

from .bake_ik import (
    _detect_child_of_constraints,
    _child_of_compensation,
    _bake_world_pairs_iter,
    _compute_pole_position,
)


# ============================================================
# Combined preset dropdown (single enum, sections inside)
# ============================================================
# Special item identifiers (non-selectable; update callback intercepts):
#   __UNSAVED__       sentinel, "Unsaved / no preset" — shown when dirty
#   __HDR_STD__       "── Standard Presets ──" header line
#   __HDR_CUS__       "── Custom Presets ──" header line
#   __EMPTY_*__       "(none)" placeholder when a section is empty
#
# Real items use their JSON file path as the identifier.
#
# Blender requires the items list + its strings to stay alive across draws
# (GC'ing them corrupts dropdown labels). Keep a module-level cache.

_COMBINED_PRESET_ITEMS_CACHE = []


def get_combined_preset_items(self, context):
    # Read presets from both folders, categorize each by FILENAME (not
    # by folder) against SHIPPED_PRESET_FILENAMES. Custom maps now save
    # alongside the shipped ones in the addon's retarget_maps/ folder,
    # but the user dir is still read so legacy custom maps saved there
    # before the change continue to show up.
    standard = []
    custom = []
    seen_paths = set()
    for directory, kind in ((_retarget_maps_dir_shipped(), "Standard"),
                            (_retarget_maps_dir_user(), "Custom")):
        for path, name, tip in _list_presets_in(directory, kind):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            if os.path.basename(path) in SHIPPED_PRESET_FILENAMES:
                standard.append((path, name, tip))
            else:
                custom.append((path, name, tip))
    standard.sort(key=lambda t: t[1].lower())
    custom.sort(key=lambda t: t[1].lower())

    # Partition Standard: the BlendCap-capture-source presets up top, then
    # a separator line, then the Mixamo-source presets (used for the niche
    # case of transferring an already-animated Mixamo skeleton onto a
    # different rig — filename convention `mixamo_to_*.json`).
    standard_main = [t for t in standard
                     if not os.path.basename(t[0]).startswith("mixamo_to_")]
    standard_mixamo_src = [t for t in standard
                           if os.path.basename(t[0]).startswith("mixamo_to_")]

    items = [
        ('__UNSAVED__', "(Unsaved changes…)", "Map has been modified since last preset load/save"),
        None,
        ('__HDR_STD__', "── Standard Presets ──", "Built-in presets shipped with the addon (read-only)"),
    ]
    if standard_main or standard_mixamo_src:
        for path, name, tip in standard_main:
            items.append((path, "   " + name, tip))
        if standard_main and standard_mixamo_src:
            items.append(None)
        for path, name, tip in standard_mixamo_src:
            items.append((path, "   " + name, tip))
    else:
        items.append(('__EMPTY_STD__', "   (none)", ""))

    items.append(None)
    items.append(('__HDR_CUS__', "── Custom Presets ──", "User-saved presets"))
    if custom:
        for path, name, tip in custom:
            items.append((path, "   " + name, tip))
    else:
        items.append(('__EMPTY_CUS__', "   (none saved yet)", ""))

    _COMBINED_PRESET_ITEMS_CACHE.clear()
    _COMBINED_PRESET_ITEMS_CACHE.extend(items)
    return _COMBINED_PRESET_ITEMS_CACHE


# Module-level guard — suppresses all update callbacks during preset load /
# programmatic state changes, so populating pairs doesn't mark the result
# as "dirty" via mark_dirty, and doesn't fire update_preset_selection when
# we programmatically set the selected preset ID.
#
# Operators outside this module flip this flag via attribute assignment:
#   from . import retarget
#   retarget._LOADING_PRESET = True
# They MUST NOT do `from .retarget import _LOADING_PRESET` — that imports
# the value, not a binding, and any reassignment goes to the importer's
# local name without affecting the module attribute.
_LOADING_PRESET = False


def _resolve_pair_names_for_bake(scene):
    """Temporarily rewrite each pair's source/target field to its resolved
    rig bone name (prefix-prepended where needed), so bake engines that
    iterate `pair.source` / `pair.target` directly can dict-lookup the
    pose bones without per-site prefix handling.

    Returns a snapshot list of (pair, orig_source, orig_target) tuples.
    Caller MUST pass this to `_restore_pair_names_after_bake` from a
    finally block — the temporary rewrite is observable in the UI while
    in flight (the pair table briefly shows the prefixed names during
    bake), and a missed restore would leave the table in the prefixed
    form permanently.

    Guards: writes happen with _LOADING_PRESET + properties._BULK_EDITING
    set to True so the pair-field update callbacks (smart-strip,
    mark_dirty, undo push) don't fire. Smart-strip in particular would
    UNDO the resolve immediately if it ran.
    """
    global _LOADING_PRESET
    from .. import properties as _props
    src_rig = scene.blendcap_retarget_source
    tgt_rig = scene.blendcap_retarget_target
    src_prefix = scene.blendcap_retarget_source_prefix or ""
    tgt_prefix = scene.blendcap_retarget_namespace_strip or ""

    snapshot = []
    prev_loading = _LOADING_PRESET
    prev_bulk = _props._BULK_EDITING
    _LOADING_PRESET = True
    _props._BULK_EDITING = True
    try:
        for p in scene.blendcap_retarget_pairs:
            orig_source = p.source
            orig_target = p.target
            resolved_source = _resolve_against_rig(orig_source, src_rig, src_prefix)
            resolved_target = _resolve_against_rig(orig_target, tgt_rig, tgt_prefix)
            snapshot.append((p, orig_source, orig_target))
            if resolved_source != orig_source:
                p.source = resolved_source
            if resolved_target != orig_target:
                p.target = resolved_target
    finally:
        _LOADING_PRESET = prev_loading
        _props._BULK_EDITING = prev_bulk
    return snapshot


_CUSTOM_IK_SOURCE_BONE_PROPS = (
    "blendcap_ik_src_left_arm_end",
    "blendcap_ik_src_right_arm_end",
    "blendcap_ik_src_left_leg_end",
    "blendcap_ik_src_right_leg_end",
    "blendcap_ik_src_left_arm_pole",
    "blendcap_ik_src_right_arm_pole",
    "blendcap_ik_src_left_leg_pole",
    "blendcap_ik_src_right_leg_pole",
)


def _smart_strip_custom_ik_source_field(scene, prop_name):
    """Smart-strip a single custom IK source bone field against the
    Source Prefix. Used by the per-field update callbacks and by the
    Source Prefix sweep — keeps the eight blendcap_ik_src_* fields in
    SHORT form, matching the pair source field convention.

    Guards via properties._BULK_EDITING so the same field's update
    callback doesn't recurse when we re-assign the stripped value.
    """
    from .. import properties as _props
    name = getattr(scene, prop_name, "") or ""
    if not name:
        return False
    rig = scene.blendcap_retarget_source
    prefix = scene.blendcap_retarget_source_prefix or ""
    stripped = _smart_strip(name, rig, prefix)
    if stripped == name:
        return False
    prev_bulk = _props._BULK_EDITING
    _props._BULK_EDITING = True
    try:
        setattr(scene, prop_name, stripped)
    finally:
        _props._BULK_EDITING = prev_bulk
    return True


def _on_custom_ik_source_field_change(prop_name):
    """Factory: returns an update callback bound to `prop_name` so each
    of the 8 fields gets its own self-stripping update without sharing
    a single closure-capturing variable.

    Pushes a per-edit undo step so Ctrl+Z reverts one IK source field
    at a time instead of batching the whole panel's edits into one step.
    No mark_dirty: these 8 fields are not preset-saved yet (the
    eventual save/load path is mechanical SHORT-form addition per the
    handoff)."""
    def _cb(self, context):
        from .. import properties as _props
        if _LOADING_PRESET or _props._BULK_EDITING or _props._UNDO_IN_PROGRESS:
            return
        _smart_strip_custom_ik_source_field(context.scene, prop_name)
        _props._push_field_undo_step(f"Edit {prop_name}")
    return _cb


def _restrip_pairs_after_prefix_change(scene, side):
    """Walk all pairs and smart-strip the given side's stored name against
    the current Source/Target Prefix. Called from the prefix fields'
    update callbacks so editing a prefix retroactively cleans pairs
    that still hold the prefixed form — typical case is the user added
    pairs (via eyedropper or Add All) before setting the prefix, or
    switched to a re-namespaced rig and updated the prefix to match.

    Empty prefix is a no-op: leaving any already-stripped names in
    place is correct, and the resolver's exact-match-first rule lets
    unprefixed bones resolve fine.

    Guards: writes happen with _LOADING_PRESET + properties._BULK_EDITING
    set so the per-pair smart-strip and mark_dirty callbacks don't fire
    once per modified pair. We mark the preset dirty exactly once at the
    end if anything actually changed.
    """
    global _LOADING_PRESET
    from .. import properties as _props
    if side == 'source':
        rig = scene.blendcap_retarget_source
        prefix = scene.blendcap_retarget_source_prefix or ""
    else:
        rig = scene.blendcap_retarget_target
        prefix = scene.blendcap_retarget_namespace_strip or ""
    if not prefix:
        return

    changed_any = False
    prev_loading = _LOADING_PRESET
    prev_bulk = _props._BULK_EDITING
    _LOADING_PRESET = True
    _props._BULK_EDITING = True
    try:
        for p in scene.blendcap_retarget_pairs:
            if side == 'source':
                name = p.source
            else:
                name = p.target
            stripped = _smart_strip(name, rig, prefix)
            if stripped != name:
                if side == 'source':
                    p.source = stripped
                else:
                    p.target = stripped
                changed_any = True
    finally:
        _LOADING_PRESET = prev_loading
        _props._BULK_EDITING = prev_bulk

    if changed_any and scene.blendcap_retarget_preset != '__UNSAVED__':
        # Side-effect: the prefix edit changed pair storage, which the
        # user should see reflected in the preset dropdown.
        _LOADING_PRESET = True
        try:
            scene.blendcap_retarget_preset = '__UNSAVED__'
        finally:
            _LOADING_PRESET = prev_loading


def _smart_strip_head_field(scene, side):
    """Smart-strip blendcap_retarget_face_head_source / face_head_target
    against the matching prefix. Used by both the per-field update
    callbacks and the Source/Target Prefix sweeps so the Head fields
    track the same SHORT-form convention as the pair table.

    Guards via _BULK_EDITING so the field's own update callback doesn't
    recurse when we re-assign.
    """
    from .. import properties as _props
    if side == 'source':
        prop_name = "blendcap_retarget_face_head_source"
        rig = scene.blendcap_retarget_source
        prefix = scene.blendcap_retarget_source_prefix or ""
    else:
        prop_name = "blendcap_retarget_face_head_target"
        rig = scene.blendcap_retarget_target
        prefix = scene.blendcap_retarget_namespace_strip or ""
    name = getattr(scene, prop_name, "") or ""
    if not name:
        return
    stripped = _smart_strip(name, rig, prefix)
    if stripped == name:
        return
    prev_bulk = _props._BULK_EDITING
    _props._BULK_EDITING = True
    try:
        setattr(scene, prop_name, stripped)
    finally:
        _props._BULK_EDITING = prev_bulk


def _on_face_head_source_change(self, context):
    """Update callback for blendcap_retarget_face_head_source — smart-
    strips against the Source Prefix so values written here (manual
    edit, eyedropper, or the rig-picker auto-detect) land in SHORT form.

    Also marks the preset dirty + pushes a per-edit undo step. Head
    (Source) is a preset-saved field (face_head_source in JSON); without
    these the user's edit wouldn't flip the dropdown to __UNSAVED__ and
    Ctrl+Z would batch-undo it together with adjacent edits."""
    from .. import properties as _props
    if _LOADING_PRESET or _props._BULK_EDITING or _props._UNDO_IN_PROGRESS:
        return
    _smart_strip_head_field(context.scene, 'source')
    mark_dirty(self, context)
    _props._push_field_undo_step("Edit Head (Source)")


def _on_face_head_target_change(self, context):
    """Update callback for blendcap_retarget_face_head_target — smart-
    strips against the Target Prefix. Marks preset dirty + pushes undo
    step (see _on_face_head_source_change docstring)."""
    from .. import properties as _props
    if _LOADING_PRESET or _props._BULK_EDITING or _props._UNDO_IN_PROGRESS:
        return
    _smart_strip_head_field(context.scene, 'target')
    mark_dirty(self, context)
    _props._push_field_undo_step("Edit Head (Target)")


def _on_source_prefix_change(self, context):
    """Source Prefix scene-prop update callback. Restrips all pair source
    fields, the custom IK source bone fields, AND the Head (Source)
    field against the new prefix.

    Always marks the preset dirty + pushes a per-edit undo step. The
    prefix is a preset-saved field (source_prefix in JSON v72+), so
    changing it changes preset content even when no pair restrip was
    needed (e.g. all pairs already SHORT). Guard checks happen inside
    _push_field_undo_step / mark_dirty so the bulk-restrip sub-calls
    don't double-fire."""
    from .. import properties as _props
    scene = context.scene
    _restrip_pairs_after_prefix_change(scene, 'source')
    # Custom IK source bones use the SAME prefix as pair sources, so a
    # source-prefix edit cleans them in the same sweep. Each call has
    # its own _BULK_EDITING guard internally.
    for prop_name in _CUSTOM_IK_SOURCE_BONE_PROPS:
        _smart_strip_custom_ik_source_field(scene, prop_name)
    # Head (Source) follows the same rule.
    _smart_strip_head_field(scene, 'source')
    mark_dirty(self, context)
    _props._push_field_undo_step("Edit Source Prefix")


def _on_target_prefix_change(self, context):
    """Target Prefix scene-prop update callback. Restrips all pair target
    fields AND the Head (Target) field against the new prefix.

    The underlying scene prop is `blendcap_retarget_namespace_strip` (kept
    under the legacy name for back-compat preset loading). Always marks
    dirty + pushes undo (see _on_source_prefix_change docstring)."""
    from .. import properties as _props
    scene = context.scene
    _restrip_pairs_after_prefix_change(scene, 'target')
    _smart_strip_head_field(scene, 'target')
    mark_dirty(self, context)
    _props._push_field_undo_step("Edit Target Prefix")


def _restore_pair_names_after_bake(snapshot):
    """Undo the temporary mutation done by `_resolve_pair_names_for_bake`.
    Idempotent — safe to call with an empty/None snapshot."""
    if not snapshot:
        return
    global _LOADING_PRESET
    from .. import properties as _props
    prev_loading = _LOADING_PRESET
    prev_bulk = _props._BULK_EDITING
    _LOADING_PRESET = True
    _props._BULK_EDITING = True
    try:
        for p, orig_source, orig_target in snapshot:
            try:
                if p.source != orig_source:
                    p.source = orig_source
                if p.target != orig_target:
                    p.target = orig_target
            except ReferenceError:
                # Pair PropertyGroup got freed (e.g. pairs.clear during
                # a teardown path) — nothing to restore.
                pass
    finally:
        _LOADING_PRESET = prev_loading
        _props._BULK_EDITING = prev_bulk


def mark_dirty(self, context):
    """Pair-field update callback. When any pair field changes via the UI,
    flip the preset dropdown to __UNSAVED__ so the user sees the map no
    longer matches the loaded preset."""
    global _LOADING_PRESET
    if _LOADING_PRESET:
        return
    scene = context.scene
    if scene.blendcap_retarget_preset != '__UNSAVED__':
        _LOADING_PRESET = True
        try:
            scene.blendcap_retarget_preset = '__UNSAVED__'
        finally:
            _LOADING_PRESET = False


def update_preset_selection(self, context):
    """Preset EnumProperty update callback. Auto-loads the chosen preset,
    with special handling for header/sentinel/empty items (revert to sentinel)."""
    global _LOADING_PRESET
    if _LOADING_PRESET:
        return
    value = self.blendcap_retarget_preset
    # Non-selectable rows: revert to sentinel so dropdown shows "(Unsaved…)".
    if value.startswith('__HDR_') or value.startswith('__EMPTY_'):
        _LOADING_PRESET = True
        try:
            self.blendcap_retarget_preset = '__UNSAVED__'
        finally:
            _LOADING_PRESET = False
        return
    if value == '__UNSAVED__':
        return
    # Real preset path
    if not os.path.isfile(value):
        return
    try:
        data = _load_retarget_map(value)
    except Exception:
        return
    _populate_pairs_from_map(context.scene, data)
    # Remember the path so Save Map As... can default to the loaded
    # preset's name even after the dropdown flips to __UNSAVED__.
    context.scene.blendcap_retarget_last_loaded_preset = value


def _populate_pairs_from_map(scene, map_dict):
    """Populate scene state from a loaded JSON map. Guards dirty-tracking
    (_LOADING_PRESET) so pair-field update callbacks don't flip the
    preset enum to __UNSAVED__ while we're writing the values that
    came FROM the preset.

    A side the preset DECLARES a prefix on (e.g. Mixamo's source_prefix
    'mixamorig:') is auto-corrected to the rig's real namespace via
    _detect_rig_prefix, so a Mixamo rig re-namespaced as 'mixamorig1:' /
    'mixamorig2:' resolves with no user edit. Detection runs ONLY for a
    side the preset declared a prefix on — a no-prefix preset is for
    plain rigs and is left exactly as stored, so it never invents a
    namespace even on a namespaced rig — and even then overrides only
    when the detected prefix resolves strictly more names.

    Rows are then filtered. The empty placeholder (both sides blank) is
    dropped. A fully-mapped pair is dropped only when a side's bone is
    genuinely ABSENT from its rig — not present under ANY namespace — so
    the old declutter behavior is preserved: a full body+hands+face
    preset on a rig that lacks face / finger bones still drops those
    rows. But a bone that exists under a different prefix (mixamorig1:)
    is KEPT, not dropped: auto-detect usually resolves it, and otherwise
    it renders red (the UIList alerts invalid rows) and re-resolves live
    once the user edits the prefix — so a namespace mismatch never
    silently empties the table the way it used to. Half-filled rows (one
    side named) are kept as in-progress.

    The rig-change callbacks (registration.py) re-fire this load once a
    rig lands, which re-runs detection and re-marks validity against the
    real rig."""
    global _LOADING_PRESET
    _LOADING_PRESET = True
    try:
        # Prefixes: start from what the preset stored. ONLY a side the
        # preset DECLARES a prefix on gets auto-corrected to the rig's
        # actual namespace (Mixamo etc.) — a preset with no prefix field
        # is for plain rigs and is left exactly as stored, so loading one
        # never invents a namespace, even on a namespaced rig.
        src_rig = scene.blendcap_retarget_source
        tgt_rig = scene.blendcap_retarget_target
        pre_namespace = map_dict.get("namespace_strip", "")
        pre_source = map_dict.get("source_prefix", "")
        detected_source = pre_source
        detected_namespace = pre_namespace
        if pre_source or pre_namespace:
            # Gather the preset's short source/target names (legacy-
            # stripped) to feed detection. New presets store SHORT names;
            # old ones stored the prefixed form, so strip first.
            _ls = pre_source or pre_namespace or ""
            _lt = pre_namespace or ""
            det_srcs, det_tgts = [], []
            for entry in map_dict.get("pairs", []):
                s = entry.get("source", "") or ""
                t = entry.get("target", "") or ""
                if _ls and s.startswith(_ls):
                    s = s[len(_ls):]
                if _lt and t.startswith(_lt):
                    t = t[len(_lt):]
                if s:
                    det_srcs.append(s)
                if t:
                    det_tgts.append(t)
            # _detect_rig_prefix returns the stored prefix unless a
            # different one resolves strictly more names (the mixamorig1: /
            # mixamorig2: case). No rig assigned -> stored prefix kept; the
            # rig-change callback re-fires this load and re-detects later.
            if pre_source:
                detected_source = _detect_rig_prefix(src_rig, det_srcs, pre_source)
            if pre_namespace:
                detected_namespace = _detect_rig_prefix(tgt_rig, det_tgts, pre_namespace)
        scene.blendcap_retarget_namespace_strip = detected_namespace
        scene.blendcap_retarget_source_prefix = detected_source
        legacy_src_prefix = scene.blendcap_retarget_source_prefix or scene.blendcap_retarget_namespace_strip or ""
        legacy_tgt_prefix = scene.blendcap_retarget_namespace_strip or ""

        pairs = scene.blendcap_retarget_pairs
        pairs.clear()
        # Precomputed bone-name sets for the keep/drop existence check in
        # the loop. None when a rig isn't assigned -> that side isn't
        # checked, so a preset loaded before the rigs are picked still
        # populates fully (the rig-change callback re-fires this load).
        src_bone_names = (set(src_rig.data.bones.keys())
                          if src_rig and src_rig.type == 'ARMATURE' else None)
        tgt_bone_names = (set(tgt_rig.data.bones.keys())
                          if tgt_rig and tgt_rig.type == 'ARMATURE' else None)

        for entry in map_dict.get("pairs", []):
            src = entry.get("source", "")
            tgt = entry.get("target", "")
            # Legacy-strip: if an older preset saved the full prefixed
            # name, peel the saved prefix so we land in SHORT form. New
            # presets save SHORT to begin with, so this is a no-op for
            # them.
            if legacy_src_prefix and src.startswith(legacy_src_prefix):
                src = src[len(legacy_src_prefix):]
            if legacy_tgt_prefix and tgt.startswith(legacy_tgt_prefix):
                tgt = tgt[len(legacy_tgt_prefix):]
            # Per-row filter. Drop the empty placeholder, and drop a
            # fully-mapped pair only when a side's bone is genuinely
            # ABSENT from its rig (not present under ANY namespace). A
            # bone that exists under a different prefix (mixamorig1:) is
            # KEPT: auto-detect above usually resolves it, and if not the
            # row stays red and editing the prefix lights it up. A bone
            # the rig simply lacks (face / finger bones on a body rig) is
            # dropped, preserving the table-declutter behavior. Half-
            # filled rows (one side) are kept as in-progress. See docstring.
            if not src and not tgt:
                continue
            if src and tgt:
                if src_bone_names is not None and not _present_under_any_namespace(src, src_bone_names):
                    continue
                if tgt_bone_names is not None and not _present_under_any_namespace(tgt, tgt_bone_names):
                    continue
            p = pairs.add()
            p.source = src
            p.target = tgt
            p.channels = entry.get("channels", "ROT")
            p.influence = float(entry.get("influence", 1.0))
            p.axes = entry.get("axes", "XYZ")
            loc_space_raw = str(entry.get("loc_space", "basis")).upper()
            p.loc_space = loc_space_raw if loc_space_raw in ('BASIS', 'HEAD_LOCAL') else 'BASIS'
            p.loc_scale = float(entry.get("loc_scale", 1.0))
            p.loc_scale_residual = bool(entry.get("loc_scale_residual", False))
        scene.blendcap_retarget_pair_index = 0
        scene.blendcap_retarget_map_hint = map_dict.get("target_kind", "generic").lower()
        scene.blendcap_retarget_face_head_source = map_dict.get("face_head_source", "")
        scene.blendcap_retarget_face_head_target = map_dict.get("face_head_target", "")
        # auto_bake_ik — reset to engine default (False) first, then
        # apply the preset's value if present. Same explicit-load
        # convention as the face state below.
        scene.blendcap_retarget_auto_bake_ik = False
        if "auto_bake_ik" in map_dict:
            scene.blendcap_retarget_auto_bake_ik = bool(map_dict["auto_bake_ik"])
        # Retarget-owned scene state is EXPLICITLY RESET to engine
        # defaults on every preset load, then preset fields overwrite
        # where present. Result: loading a preset always produces a
        # deterministic scene state — fields missing from the JSON revert
        # to engine defaults rather than carrying over from a previously-
        # loaded preset. The save path always writes the full set, so a
        # normally-saved preset round-trips losslessly; this only matters
        # for partial / hand-edited / older-version presets that omit
        # some fields.
        #
        # NOTE: face_denoise / face_denoising are intentionally NOT reset
        # or read here. Those controls now live in the BVH and ARKit
        # sections — bone-map presets don't own them. Old presets that
        # still carry the keys are silently ignored on load.
        scene.blendcap_face_capture_compensation = False
        scene.blendcap_face_amp_global = 1.0
        for region in FACE_AMPLIFICATION_REGIONS:
            setattr(scene, f"blendcap_face_amp_{region}", 1.0)
        scene.blendcap_retarget_use_world_location = False
        # Custom IK source bones block — master toggle off + every
        # bone field blank is the engine default.
        scene.blendcap_use_custom_ik_sources = False
        for prop_name in (
            "blendcap_ik_src_left_arm_end",
            "blendcap_ik_src_right_arm_end",
            "blendcap_ik_src_left_leg_end",
            "blendcap_ik_src_right_leg_end",
            "blendcap_ik_src_left_arm_pole",
            "blendcap_ik_src_right_arm_pole",
            "blendcap_ik_src_left_leg_pole",
            "blendcap_ik_src_right_leg_pole",
        ):
            setattr(scene, prop_name, "")
        # Now apply whatever the preset specifies.
        if "face_capture_compensation" in map_dict:
            scene.blendcap_face_capture_compensation = bool(map_dict["face_capture_compensation"])
        if "face_amplification" in map_dict:
            amp_block = map_dict["face_amplification"]
            # Global rides under the reserved key "global"; per-region
            # values use the FACE_AMPLIFICATION_REGIONS names.
            if "global" in amp_block:
                scene.blendcap_face_amp_global = float(amp_block["global"])
            for region in FACE_AMPLIFICATION_REGIONS:
                if region in amp_block:
                    setattr(scene, f"blendcap_face_amp_{region}", float(amp_block[region]))
        if "use_world_location" in map_dict:
            scene.blendcap_retarget_use_world_location = bool(map_dict["use_world_location"])
        if "custom_ik_sources" in map_dict:
            cis = map_dict["custom_ik_sources"]
            scene.blendcap_use_custom_ik_sources = bool(cis.get("enabled", False))
            _cis_map = {
                "left_hand":     "blendcap_ik_src_left_arm_end",
                "right_hand":    "blendcap_ik_src_right_arm_end",
                "left_foot":     "blendcap_ik_src_left_leg_end",
                "right_foot":    "blendcap_ik_src_right_leg_end",
                "left_forearm":  "blendcap_ik_src_left_arm_pole",
                "right_forearm": "blendcap_ik_src_right_arm_pole",
                "left_shin":     "blendcap_ik_src_left_leg_pole",
                "right_shin":    "blendcap_ik_src_right_leg_pole",
            }
            for json_key, prop_name in _cis_map.items():
                if json_key in cis:
                    setattr(scene, prop_name, str(cis[json_key] or ""))
        # IK chains — delegated to operators_ik to avoid a cross-package
        # circular import (operators_ik imports from this package). The
        # lazy import here is the seam.
        from ..operators_ik import apply_ik_chains_from_map
        apply_ik_chains_from_map(scene, map_dict)
        # NOTE: deliberately NO sync_ik_chain_collection here.
        # apply_ik_chains_from_map blanks the four rows then writes exactly
        # what the preset specifies, so the FK->IK panel reflects the
        # loaded preset and nothing else — a preset with no ik_chains
        # leaves the rows blank (cleared) instead of silently auto-filling
        # from the target rig. Rig discovery still populates the panel
        # VISIBLY when you assign a target rig with no preset loaded
        # (registration._on_target_rig_change), so this only removes the
        # silent override of a chosen preset.
    finally:
        _LOADING_PRESET = False
