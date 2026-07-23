"""Alt retarget engine — evaluates source poses by reading fcurves directly,
bypassing scene.frame_set and Blender's depsgraph for the source rig.

Why this exists
---------------
The classic engine (_matrix_bake_pairs_iter in bake_fk.py) calls
scene.frame_set(frame) per frame, which forces Blender to evaluate the
ENTIRE source rig's pose graph: every constraint, every driver, every
IK chain, every modifier. On a 1618-bone ARP target, that per-frame
evaluation is the dominant cost — we only care about ~92 mapped bones
but Blender evaluates all 1618 every frame just so we can read the
.matrix of the 92 we touch.

This engine never calls scene.frame_set. Instead it:

  1. Extracts the source rig's fcurves once at setup (fc.evaluate is
     Blender's public C-level fcurve sampler — same code that the
     depsgraph uses internally, but called directly without the rest
     of the eval pipeline).

  2. Walks the source skeleton depth-first per frame, multiplying each
     bone's basis-pose (loc + rot from fcurves) into its parent's
     arm-local matrix to produce arm-local world poses we read from.

  3. Computes target basis values using the same delta-from-rest math
     the classic engine uses, then maintains its own
     `computed_tgt_arm_local` dict so child-bone target reads see fresh
     parent matrices without touching Blender's pose state.

  4. Bulk-writes accumulated keyframes at the end via the same
     `_bulk_write_keys` machinery the classic engine uses — the only
     point at which Blender's depsgraph re-evaluates anything.

Expected gain: 30–100×, depending on rig size. Larger rigs win bigger
because the classic engine's depsgraph cost scales with the entire pose
hierarchy, while this engine's per-frame cost scales only with the
mapped bones.

Current limitations
-------------------
- Source rigs with drivers / constraints / IK on the SOURCE side
  produce wrong poses: we evaluate only raw fcurves, not the
  constraint/driver stack. BVH-style sources are the canonical fit.
- Target constraints OUTSIDE the supported set (see _SUPPORTED_TYPES)
  silently no-op — bones with IK, Spline IK, Action, Armature, Stretch
  To, Shrinkwrap, Floor, Follow Path, Clamp To, Pivot, Transformation,
  Transform Cache produce incorrect poses.
- Pre-existing animation on the target rig is IGNORED when computing
  child-parent world matrices — we assume a clean target. Mid-air bake
  augmentation will produce wrong child poses for un-mapped parents.

Now supported (matches classic engine):
- HEAD_LOCAL face pairs (algebraic basis back-solve, including Child Of
  compensation)
- Leg-anchor compensation
- source_rest_override ("use current pose as rest" + custom rest pose
  presets)
- Quaternion-sign continuity on basis rotations
- ~13 common pose constraint types in pure math (Copy Transforms /
  Location / Rotation / Scale, Child Of, Damped / Track To / Locked
  Track, Limit Rotation / Location / Scale, Maintain Volume)
"""
import bpy
# math.radians is used by the rotation-denoise threshold checks (ported
# from bake_fk.py, which imports math; the import didn't ride along, so
# any ROT pair with a denoise threshold > 0 raised NameError here).
import math

from mathutils import Matrix, Quaternion, Vector

from .bake_fk_common import _compute_scale_ratio, _compute_pair_rest_invariants
from .fcurves import _bulk_write_keys, _accumulate_basis_key
from .fcurve_eval import _extract_source_fcurves, _eval_basis
from .constraint_sim import (
    _SUPPORTED_TYPES,
    _scan_supported_constraints,
    _apply_constraint_chain,
    _armature_delta_from_scan,
    _compensate_basis_for_constraints,
    _topo_sort_target_bones,
)


def _settle_target_drivers_and_eval(scene, tgt_rig):
    """Setup stage, extracted verbatim from _matrix_bake_pairs_iter_fast:
    one-shot driver settle on the target rig + evaluated-influence reader.
    Returns (_eval_pbs, _eff_influence)."""
    # One-shot full depsgraph refresh at setup. The retarget operator
    # sets FK/IK switch properties (e.g. `IK_FK = 1.0`) immediately
    # before calling the bake to force FK mode. ARP / Rigify drive
    # constraint influences from these switch properties via drivers
    # (verified in the rig analysis: every rotFK / scaleFK / locFK
    # influence has a driver). `con.influence` reads stale values
    # until those drivers re-evaluate.
    #
    # `view_layer.update()` alone is unreliable for driver re-eval
    # in complex dependency graphs — it flushes pending tags but
    # doesn't always trigger full driver evaluation. `scene.frame_set`
    # at the current frame is the strongest trigger we can use without
    # advancing time. Cost: ~50-200ms one-time; vital for correctness
    # on ARP/Rigify FK retargets.
    # Force the target's drivers to re-evaluate before we read evaluated
    # constraint influences below. The operator writes IK_FK/ik_fk_switch
    # custom properties in _prepare to force FK mode, but a custom-property
    # write does NOT reliably tag the depsgraph to re-run the drivers that
    # read it. Without an explicit tag, evaluated_get() returns a driver
    # graph that was settled on SOME launches and stale (IK-mode) on others
    # — the intermittent "fingers bake right only sometimes" race. update_
    # tag() forces the re-eval deterministically; frame_set + view_layer
    # .update then settle it. Generalized: fixes any rig with driver-driven
    # constraint influences (Rigify FK/IK blend, ARP, neck/head_follow).
    try:
        tgt_rig.update_tag()
    except Exception:
        pass
    scene.frame_set(scene.frame_current)
    bpy.context.view_layer.update()

    # Evaluated target pose bones, for reading driver-driven constraint
    # influence/mute (FK-IK blend switches, neck_follow/head_follow).
    # Drivers write to the evaluated copy, not the original datablock the
    # engine otherwise reads — so on a cold rig the original con.influence
    # is stale (IK-mode) even after the frame_set above settles the
    # evaluated graph. This is why a single (or repeated) frame_set never
    # fixed the cold-rig first-bake-wrong: frame_set updates the evaluated
    # copy, which is exactly what we now read from. Mirrors bake_ik_fast.
    try:
        _depsgraph = bpy.context.evaluated_depsgraph_get()
        _eval_tgt = tgt_rig.evaluated_get(_depsgraph)
        _eval_pbs = _eval_tgt.pose.bones
    except Exception:
        _eval_pbs = None

    def _eff_influence(con, bone_name):
        """(mute, influence) from the evaluated copy when available, so
        driver-driven FK/IK-blend + neck-follow influences are settled."""
        if _eval_pbs is not None and bone_name in _eval_pbs:
            ec = _eval_pbs[bone_name].constraints.get(con.name)
            if ec is not None:
                return ec.mute, ec.influence
        return con.mute, con.influence
    return _eval_pbs, _eff_influence


def _build_rest_data(pairs, src_rig, tgt_rig, src_mw, tgt_mw,
                     source_rest_override, override_rotation_only,
                     compensation_on, denoise_on, scene, mapped_bones):
    """Setup stage, extracted verbatim from _matrix_bake_pairs_iter_fast.
    Fills `mapped_bones` in place when given. Returns (rest_data,
    src_bone_names)."""
    # ------------------------------------------------------------------
    # Setup: build rest_data with per-pair invariants. Mirrors the
    # classic engine's setup so the math downstream is identical apart
    # from where we read source poses from.
    # ------------------------------------------------------------------
    rest_data = []
    src_bone_names = set()
    for p in pairs:
        if not p.source or not p.target:
            continue
        if p.source not in src_rig.pose.bones:
            continue
        if p.target not in tgt_rig.pose.bones:
            continue

        s_pb = src_rig.pose.bones[p.source]
        t_bone = tgt_rig.data.bones[p.target]

        rd = _compute_pair_rest_invariants(
            p, s_pb, t_bone, src_mw, tgt_mw,
            source_rest_override, override_rotation_only,
            compensation_on, denoise_on, scene)
        rd['pair'] = p
        rd['source'] = p.source
        rd['target'] = p.target
        rest_data.append(rd)
        src_bone_names.add(p.source)
        if mapped_bones is not None:
            mapped_bones.add(p.target)
    return rest_data, src_bone_names


def _setup_source_skeleton(src_rig, src_bone_names):
    """Setup stage, extracted verbatim from _matrix_bake_pairs_iter_fast.
    Returns (src_walk_set, _depth_src, src_sorted, src_rel_rest,
    src_parent, src_fcurves)."""
    # ------------------------------------------------------------------
    # Source skeleton walk set — we need every mapped source bone PLUS
    # all its ancestors, so each bone's arm-local matrix can be composed
    # from `parent_arm @ rel_rest @ basis`.
    # ------------------------------------------------------------------
    src_walk_set = set()
    for name in src_bone_names:
        b = src_rig.data.bones.get(name)
        while b is not None:
            src_walk_set.add(b.name)
            b = b.parent

    def _depth_src(name):
        d, b = 0, src_rig.data.bones[name].parent
        while b is not None:
            d += 1
            b = b.parent
        return d

    src_sorted = sorted(src_walk_set, key=_depth_src)

    # Precompute source rel_rest (parent_local^-1 @ self_local) and the
    # parent-name lookup. Both are rest-pose invariant.
    src_rel_rest = {}
    src_parent = {}
    for name in src_walk_set:
        bone = src_rig.data.bones[name]
        if bone.parent is None:
            src_rel_rest[name] = bone.matrix_local.copy()
            src_parent[name] = None
        else:
            src_rel_rest[name] = bone.parent.matrix_local.inverted() @ bone.matrix_local
            src_parent[name] = bone.parent.name

    # Single one-time extraction of source fcurves. After this we never
    # touch Blender's pose state for the source rig.
    src_fcurves = _extract_source_fcurves(src_rig, src_walk_set)
    return (src_walk_set, _depth_src, src_sorted, src_rel_rest,
            src_parent, src_fcurves)


def _setup_target_side(rest_data, tgt_rig, _eval_pbs, _eff_influence):
    """Setup stage, extracted verbatim from _matrix_bake_pairs_iter_fast:
    target walk set, constraint scan, topo order, per-target metadata and
    armature-carry detection. Returns (by_target, constraint_meta,
    target_names_sorted, target_meta)."""
    # ------------------------------------------------------------------
    # Group rest_data entries by target. Build per-target metadata.
    # ------------------------------------------------------------------
    by_target = {}
    for rd in rest_data:
        by_target.setdefault(rd['target'], []).append(rd)

    # Walk set on the target side — every mapped target, plus all
    # data ancestors, plus all constraint subtargets (and THEIR
    # ancestors). Constraint subtargets must be in the set so the
    # per-frame loop can read their computed pose when applying
    # constraint effects. We iterate the expansion until stable —
    # subtargets may themselves have constraints to other bones.
    tgt_walk_set = set()
    for tgt_name in by_target.keys():
        b = tgt_rig.data.bones.get(tgt_name)
        while b is not None:
            tgt_walk_set.add(b.name)
            b = b.parent
    # Expansion: add constraint subtargets + their ancestors. Iterates
    # until stable — a subtarget may itself have constraints to other
    # bones, transitively pulling more into the walk set. Also pulls
    # in space_subtarget for CUSTOM-space constraints (Rigify face uses
    # this for the jaw_master chain).
    _SUBTARGET_TYPES = {
        'COPY_TRANSFORMS', 'COPY_LOCATION', 'COPY_ROTATION', 'COPY_SCALE',
        'CHILD_OF', 'DAMPED_TRACK', 'TRACK_TO', 'LOCKED_TRACK',
        'STRETCH_TO', 'LIMIT_DISTANCE', 'FLOOR',
    }
    while True:
        new_bones = set()
        for name in tgt_walk_set:
            pb = tgt_rig.pose.bones.get(name)
            if pb is None:
                continue
            for con in pb.constraints:
                c_mute, c_infl = _eff_influence(con, name)
                if c_mute or c_infl <= 0.0:
                    continue
                # ARMATURE has multiple targets in a `targets` list,
                # not a single subtarget. Handle it explicitly.
                if con.type == 'ARMATURE':
                    for t in con.targets:
                        sub = getattr(t, 'subtarget', None)
                        if not sub or sub in tgt_walk_set:
                            continue
                        b = tgt_rig.data.bones.get(sub)
                        while b is not None and b.name not in tgt_walk_set:
                            new_bones.add(b.name)
                            b = b.parent
                    continue
                if con.type not in _SUBTARGET_TYPES:
                    continue
                # Primary subtarget + custom-space subtarget
                for attr_name in ('subtarget', 'space_subtarget'):
                    sub = getattr(con, attr_name, None)
                    if not sub or sub in tgt_walk_set:
                        continue
                    b = tgt_rig.data.bones.get(sub)
                    while b is not None and b.name not in tgt_walk_set:
                        new_bones.add(b.name)
                        b = b.parent
        if not new_bones:
            break
        tgt_walk_set |= new_bones

    # Scan supported constraints per bone. Empty list for bones with
    # no supported constraints.
    constraint_meta = {}
    for name in tgt_walk_set:
        pb = tgt_rig.pose.bones.get(name)
        if pb is None:
            continue
        eval_pb = _eval_pbs.get(name) if _eval_pbs is not None else None
        cons = _scan_supported_constraints(pb, tgt_walk_set, eval_pb=eval_pb)
        if cons:
            constraint_meta[name] = cons

    def _depth_tgt(name):
        d, b = 0, tgt_rig.data.bones[name].parent
        while b is not None:
            d += 1
            b = b.parent
        return d

    # Topological sort: respect both data hierarchy AND constraint
    # subtargets. Pinky's MCH parent might be at shallow data depth
    # but constraint-depend on c_hand_fk.l at deep data depth; topo
    # sort ensures c_hand_fk.l is processed first.
    target_names_sorted = _topo_sort_target_bones(
        tgt_walk_set, tgt_rig, constraint_meta)

    # Target rest metadata — invariants only. Unmapped ancestors get
    # empty rd_list / any_rot=False / any_loc=False; the per-frame loop
    # falls through to `arm_local = parent_arm @ rel_rest @ identity`
    # for those bones, which is correct (rest pose in parent's frame).
    target_meta = {}
    for tgt_name in target_names_sorted:
        t_bone = tgt_rig.data.bones[tgt_name]
        rd_list = by_target.get(tgt_name, [])
        parent_bone = t_bone.parent
        parent_name = parent_bone.name if parent_bone else None

        if parent_bone is None:
            rel_rest = t_bone.matrix_local.copy()
        else:
            rel_rest = parent_bone.matrix_local.inverted() @ t_bone.matrix_local
        rel_rest_rot_q_inv = rel_rest.to_quaternion().inverted()

        bone_key = tgt_name.replace('"', '\\"')
        bone_path = f'pose.bones["{bone_key}"]'

        target_meta[tgt_name] = {
            'parent_name': parent_name,
            'rel_rest': rel_rest,
            'rel_rest_rot_q_inv': rel_rest_rot_q_inv,
            'rotation_mode': tgt_rig.pose.bones[tgt_name].rotation_mode,
            'any_rot': any(rd['in_rot'] for rd in rd_list),
            'any_loc': any(rd['in_loc'] for rd in rd_list),
            'rd_list': rd_list,
            'bone_path': bone_path,
            'is_mapped': tgt_name in by_target,
            # Filled below: the scanned ARMATURE constraint that fully
            # carries this PARENTLESS mapped bone, if any.
            'carry_con': None,
        }

    # ------------------------------------------------------------------
    # Armature-carry detection. A mapped bone with NO data parent whose
    # pose is fully carried by a single-target ARMATURE constraint
    # (CloudRig: FK-Toes via "Armature (Toe FK/IK)", the STR-I face weld
    # handles via "Armature@Jaw...") is effectively PARENTED to that
    # constraint's target. Every absolute/head-anchored write path
    # (world-delta ROT back-solve, HEAD_LOCAL LOC's P matrix) must use
    # the carry frame as the bone's parent — computing against "no
    # parent" bakes the carried motion into the basis and playback then
    # applies the carry AGAIN on top (toes pointed sideways; cheek weld
    # handles flew with every head/jaw move). The per-frame loop reads
    # carry_con and substitutes the live carry delta D for the parent
    # in those two computations ONLY; the walk/compose path keeps the
    # real (absent) parent so the constraint sim applies D exactly once.
    # Qualification: parentless + mapped + exactly one ARMATURE
    # constraint with exactly one surviving scanned target + influence
    # ~1. Partial influence = genuine blend -> hands off. Parented
    # bones (deform-blend riders like CloudRig's other face bones) are
    # never touched.
    for tgt_name, meta in target_meta.items():
        if not meta['is_mapped'] or meta['parent_name'] is not None:
            continue
        arm_cons = [c for c in constraint_meta.get(tgt_name, [])
                    if c['type'] == 'ARMATURE']
        if len(arm_cons) != 1:
            continue
        con = arm_cons[0]
        if len(con['targets']) != 1 or con['influence'] < 0.999:
            continue
        meta['carry_con'] = con
        print(f"[BlendCap] FK bake: treating armature carry as the "
              f"parent of {tgt_name!r} (target "
              f"{con['targets'][0]['subtarget']!r}, weight "
              f"{con['targets'][0]['weight']:.3f})")
    return by_target, constraint_meta, target_names_sorted, target_meta


def _setup_leg_anchor_compensation(pairs, src_rig, tgt_rig, src_mw, tgt_mw,
                                   by_target, src_walk_set, target_meta):
    """Setup stage, extracted verbatim from _matrix_bake_pairs_iter_fast.
    Returns (comp_active, comp_target_name, comp_baseline, comp_legs)."""
    # ------------------------------------------------------------------
    # Leg-anchor compensation setup. Mirrors the classic engine's block.
    # For rigs where the leg chain parents through the spine (Rigify),
    # spine rotation drags the legs in world space relative to the
    # source's sibling-of-pelvis convention. We measure the per-frame
    # drift and add it to a root-ish LOC pair's basis translation.
    # ------------------------------------------------------------------
    _LEG_ROOT_KW = ('thigh', 'upleg', 'upper_leg')
    comp_loc_pair = None
    _comp_candidates = []
    for p in pairs:
        if p.channels not in ('LOC', 'LOC_ROT'):
            continue
        if 'Z' not in (p.axes or 'XYZ').upper():
            continue
        tname = p.target.lower()
        if tname == 'root' or tname.startswith('root.'):
            continue
        if p.target not in tgt_rig.pose.bones:
            continue
        if p.target not in by_target:
            continue
        _comp_candidates.append(p)
    # Prefer the pair actually FED BY THE HIPS. Face pairs are LOC with
    # axes XYZ ('Z' matches!), so plain first-match grabs whichever LOC
    # pair happens to sit higher in the table — in hand-built maps that
    # can be a face pair, which (a) leaves the body drift uncorrected
    # (rig rises on lean/kick) and (b) dumps body-sized corrections
    # onto a face bone every frame. Source hips are reliably named
    # ('Hips' in our BVH and Mixamo; 'pelvis' on some customs);
    # first-match stays as the fallback for exotic sources.
    for p in _comp_candidates:
        sname = p.source.lower()
        if 'hip' in sname or 'pelvis' in sname:
            comp_loc_pair = p
            break
    if comp_loc_pair is None and _comp_candidates:
        comp_loc_pair = _comp_candidates[0]
    comp_legs = []  # list of (source_bone_name, target_bone_name)
    _seen = set()
    for p in pairs:
        if p.target in _seen:
            continue
        if not any(kw in p.target.lower() for kw in _LEG_ROOT_KW):
            continue
        if p.source not in src_rig.pose.bones or p.target not in tgt_rig.pose.bones:
            continue
        if p.source not in src_walk_set or p.target not in target_meta:
            continue
        comp_legs.append((p.source, p.target))
        _seen.add(p.target)
    comp_active = comp_loc_pair is not None and len(comp_legs) > 0
    comp_target_name = comp_loc_pair.target if comp_active else None
    comp_baseline = Vector((0.0, 0.0, 0.0))
    if comp_active:
        # Rest baseline: world avg of (source leg roots) - (target leg roots).
        # Bone.head_local is the rest head position in armature space; same as
        # the translation of bone.matrix_local. Identical to the classic engine.
        n_legs = len(comp_legs)
        src_rest = Vector((0.0, 0.0, 0.0))
        tgt_rest = Vector((0.0, 0.0, 0.0))
        for s_name, t_name in comp_legs:
            src_rest += src_mw @ src_rig.data.bones[s_name].head_local
            tgt_rest += tgt_mw @ tgt_rig.data.bones[t_name].head_local
        src_rest /= n_legs
        tgt_rest /= n_legs
        comp_baseline = src_rest - tgt_rest
    return comp_active, comp_target_name, comp_baseline, comp_legs


def _setup_head_local(scene, src_rig, tgt_rig, rest_data, src_walk_set,
                      src_rel_rest, src_parent, src_sorted, src_fcurves,
                      _depth_src, target_names_sorted, target_meta):
    """Setup stage, extracted verbatim from _matrix_bake_pairs_iter_fast:
    HEAD_LOCAL LOC setup, head-local ROT flagging, and the per-target
    hl_rds / basis_loc_rds split. Mutates rest_data / target_meta /
    src_walk_set / src_rel_rest / src_parent in place; src_sorted and
    src_fcurves may be REBOUND (head bone added to the walk set), so the
    caller must take them back from the return value. Returns
    (hl_src_name, hl_tgt_name, hl_motion_correction_q, src_sorted,
    src_fcurves)."""
    # ------------------------------------------------------------------
    # HEAD_LOCAL setup. Pairs flagged loc_space=HEAD_LOCAL apply source
    # motion in the source head's local frame onto the target bone in
    # the target head's local frame, bypassing parent-chain semantic
    # mismatches.
    #
    # Fast engine difference from classic: no pb.matrix back-solve through
    # target constraints — we solve the basis algebraically. Works
    # correctly when the target face bone has no constraints affecting
    # it (the typical case for FK-driven face controls). For face bones
    # with Child Of / Copy constraints, fall back to the classic engine.
    # ------------------------------------------------------------------
    hl_src_name = getattr(scene, 'blendcap_retarget_face_head_source', '') or ''
    hl_tgt_name = getattr(scene, 'blendcap_retarget_face_head_target', '') or ''
    from .name_matching import _resolve_against_rig
    src_prefix = scene.blendcap_retarget_source_prefix or ''
    tgt_prefix = scene.blendcap_retarget_namespace_strip or ''
    if not hl_src_name or not hl_tgt_name:
        from ..helpers import autodetect_head_bone_name, target_head_from_pair_table
        if not hl_src_name:
            hl_src_name = autodetect_head_bone_name(src_rig, src_prefix)
        if not hl_tgt_name:
            # Same three-stage fallback as the rig-picker callback;
            # see bake_fk.py for the rationale.
            hl_tgt_name = autodetect_head_bone_name(tgt_rig, tgt_prefix)
            if not hl_tgt_name:
                hl_tgt_name = target_head_from_pair_table(scene)
    # Resolve via the Source/Target Prefix — see comment in bake_fk.py.
    hl_src_name = _resolve_against_rig(hl_src_name, src_rig, src_prefix) if hl_src_name else hl_src_name
    hl_tgt_name = _resolve_against_rig(hl_tgt_name, tgt_rig, tgt_prefix) if hl_tgt_name else hl_tgt_name
    hl_src_data_bone = src_rig.data.bones.get(hl_src_name) if hl_src_name else None
    hl_tgt_data_bone = tgt_rig.data.bones.get(hl_tgt_name) if hl_tgt_name else None
    hl_available = hl_src_data_bone is not None and hl_tgt_data_bone is not None
    # Extraction plumbing only: bind the name so it can be returned when
    # hl_available is False. Never read in that case — the per-frame
    # HEAD_LOCAL branch is gated on meta['hl_rds'], which stays empty
    # unless hl_available (every rd falls back to BASIS below).
    hl_motion_correction_q = None
    if hl_available:
        # Make sure the source head is in our walk set so its arm_local
        # gets evaluated. It's usually already there (face source bones
        # are descendants of Head), but defend against the edge case.
        if hl_src_name not in src_walk_set:
            src_walk_set.add(hl_src_name)
            # Re-add ancestors too.
            b = hl_src_data_bone.parent
            while b is not None:
                src_walk_set.add(b.name)
                b = b.parent
            src_sorted = sorted(src_walk_set, key=_depth_src)
            # Re-extract fcurves to include the head bone.
            src_fcurves = _extract_source_fcurves(src_rig, src_walk_set)
            # Re-populate rel_rest / parent maps for any newly added bones.
            for name in src_walk_set:
                if name in src_rel_rest:
                    continue
                bone = src_rig.data.bones[name]
                if bone.parent is None:
                    src_rel_rest[name] = bone.matrix_local.copy()
                    src_parent[name] = None
                else:
                    src_rel_rest[name] = bone.parent.matrix_local.inverted() @ bone.matrix_local
                    src_parent[name] = bone.parent.name
        hl_src_head_rest_inv = hl_src_data_bone.matrix_local.inverted()
        hl_tgt_head_rest_inv = hl_tgt_data_bone.matrix_local.inverted()
        # Frame-conversion quaternion: motion_h is computed in source
        # head's REST local frame; tgt_rest_h is in target head's REST
        # local frame. Adding them as-is is geometrically incoherent
        # when source and target heads have different rest orientations
        # (always true for cross-rig retargeting). The mismatch is small
        # in face-only mode (small enough to go unnoticed) but gets
        # amplified by target head's body-driven pose at playback,
        # producing the "ARP chin protrudes forward when retargeting
        # body+face" bug. Rotating motion_h into target's frame fixes it.
        hl_motion_correction_q = (
            hl_tgt_data_bone.matrix_local.to_quaternion().inverted()
            @ hl_src_data_bone.matrix_local.to_quaternion())

    # Resolve loc_space per rd (mirrors the classic setup loop).
    hl_pair_count = 0
    for rd in rest_data:
        p = rd['pair']
        space = (getattr(p, 'loc_space', 'BASIS') or 'BASIS').upper()
        if space != 'HEAD_LOCAL' or not hl_available:
            rd['loc_space'] = 'BASIS'
            continue
        s_data = src_rig.data.bones.get(p.source)
        t_data = tgt_rig.data.bones.get(p.target)
        if s_data is None or t_data is None:
            rd['loc_space'] = 'BASIS'
            continue
        rd['loc_space'] = 'HEAD_LOCAL'
        # Source/target bone rest in head's REST local frame.
        rd['src_rest_h'] = hl_src_head_rest_inv @ s_data.matrix_local.translation
        rd['tgt_rest_h'] = hl_tgt_head_rest_inv @ t_data.matrix_local.translation
        hl_pair_count += 1

    # ------------------------------------------------------------------
    # Head-local ROTATION transfer (auto-applied to face bones).
    #
    # For face bones (source = direct child of source head), the world-
    # delta rotation transfer doesn't cleanly cancel parent chain
    # rotation when source and target chain rests don't match exactly
    # (which is the normal case — Rigify/ARP rest pose ≠ BVH rest pose).
    # The result is a small residual rotation on face bones that's
    # proportional to the body's chain rotation (e.g. Hips yaw), which
    # the user observes as "chin protrudes forward when jaw opens" on
    # ARP and "jaw shifts sideways" on Rigify when retargeting body+face
    # together vs face-only.
    #
    # Fix: for face bones, read the source basis rotation directly (not
    # the world matrix — basis is chain-pose invariant by definition)
    # and apply it to target via a setup-time rest correction. This
    # bypasses the residual entirely because the head bone's pose
    # factors out by construction.
    #
    # Auto-applied to any pair whose source is a direct child of the
    # source head bone (matches BlendCap's flat-parented face BVH). For
    # body bones (not children of head), the existing world-delta path
    # is preserved unchanged.
    hl_src_children = set()
    if hl_available:
        hl_src_children = {b.name for b in hl_src_data_bone.children}
    if not hl_available:
        # Silent degradation trap: with no resolvable head bones, every
        # face pair's ROT falls back to WORLD-DELTA (head rotation baked
        # in absolute terms) and HEAD_LOCAL LOC pairs fall back to
        # BASIS. Parented face targets mostly absorb it; free-floating
        # ones visibly fly. Make the state loud so a bad bake is
        # diagnosable from the console.
        from .preset_io import _face_region_from_source
        _face_pair_count = sum(
            1 for rd in rest_data
            if _face_region_from_source(getattr(rd['pair'], 'source', '')))
        if _face_pair_count:
            print(f"[BlendCap] WARNING: {_face_pair_count} face pair(s) "
                  f"mapped but head bones could not be resolved "
                  f"(source={hl_src_name!r}, target={hl_tgt_name!r}) — "
                  f"head-local transfer is OFF for this bake. Set the "
                  f"Head (Source) / Head (Target) fields in Advanced.")
    for rd in rest_data:
        rd['hl_rot'] = False
        if not rd['in_rot']:
            continue
        if not hl_available:
            continue
        p = rd['pair']
        if p.source not in hl_src_children:
            continue
        rd['hl_rot'] = True
        # No new fields needed — the per-frame branch reuses rd['offset_q']
        # for the conjugation. Setup-time math is the same as world-delta,
        # but the per-frame input is src_basis (chain-pose invariant)
        # instead of src_world (chain-pose contaminated), which is the
        # whole point of the fix.

    # Build per-target meta extension: which rds are HEAD_LOCAL? Bare BASIS
    # rds vs HEAD_LOCAL rds split mirrors the classic engine.
    for tgt_name in target_names_sorted:
        meta = target_meta[tgt_name]
        rd_list = meta['rd_list']
        # HEAD_LOCAL path is loc-only and bypasses BASIS; classic skips
        # it when a rotation pair also feeds this bone (any_rot==True)
        # because pb.matrix would overwrite rotation. We mirror that.
        if meta['any_rot']:
            meta['hl_rds'] = []
            meta['basis_loc_rds'] = [rd for rd in rd_list if rd['in_loc']]
        else:
            meta['hl_rds'] = [rd for rd in rd_list if rd.get('loc_space') == 'HEAD_LOCAL']
            meta['basis_loc_rds'] = [rd for rd in rd_list
                                     if rd['in_loc'] and rd.get('loc_space') != 'HEAD_LOCAL']
    return (hl_src_name, hl_tgt_name, hl_motion_correction_q, src_sorted,
            src_fcurves)


def _eval_source_frame(frame, src_sorted, src_fcurves, src_parent,
                       src_rel_rest):
    """Per-frame stage, extracted verbatim from the per-frame loop of
    _matrix_bake_pairs_iter_fast. Returns (src_arm_local,
    src_basis_loc_cache, src_basis_rot_cache)."""
    # Step 1: source skeleton evaluation. Depth-sorted so parent
    # arm-locals exist before children read them.
    src_arm_local = {}
    src_basis_loc_cache = {}
    src_basis_rot_cache = {}
    for sname in src_sorted:
        loc, rot = _eval_basis(src_fcurves[sname], frame)
        src_basis_loc_cache[sname] = loc
        src_basis_rot_cache[sname] = rot
        basis_mat = Matrix.LocRotScale(loc, rot, (1.0, 1.0, 1.0))
        parent_name = src_parent[sname]
        if parent_name is None:
            src_arm_local[sname] = src_rel_rest[sname] @ basis_mat
        else:
            src_arm_local[sname] = (src_arm_local[parent_name]
                                    @ src_rel_rest[sname]
                                    @ basis_mat)
    return src_arm_local, src_basis_loc_cache, src_basis_rot_cache


def _compute_rotation_basis(rd_list, src_basis_rot_cache, src_arm_local,
                            src_mw, tgt_mw_rot_inv, rel_rest_rot_q_inv,
                            parent_arm_rot_q, last_q, tgt_name):
    """Per-frame stage, extracted verbatim from the `if any_rot:` branch of
    _matrix_bake_pairs_iter_fast. Mutates last_q in place. Returns
    basis_rot_q (None when no rotation pair produced one)."""
    basis_rot_q = None
    out_q = None
    hl_basis_rot_q = None
    for rd in rd_list:
        if not rd['in_rot']:
            continue
        if rd.get('hl_rot'):
            # Head-local: source basis is chain-pose
            # invariant by definition (basis is relative to
            # bone's own rest). Apply denoise/scale to the
            # basis directly, then map into target's
            # basis frame via the setup-time rest
            # correction.
            src_basis_q = src_basis_rot_cache[rd['source']]
            if rd['needs_delta']:
                thresh_deg = rd['thresh_value']
                pair_scale = rd['pair_scale']
                axis, angle = src_basis_q.to_axis_angle()
                if thresh_deg > 0.0 and abs(angle) < math.radians(thresh_deg):
                    angle = 0.0
                if abs(pair_scale - 1.0) > 1e-6:
                    angle = angle * pair_scale
                src_basis_q = Quaternion(axis, angle)
            # Conjugate src basis by offset_q⁻¹ to map
            # source jaw axes into target jaw axes. Same
            # net rotation the world-delta formula would
            # produce, but the input is chain-pose
            # invariant so chain rotation can't contaminate.
            offset_q = rd['offset_q']
            hl_basis_rot_q = (offset_q.inverted()
                              @ src_basis_q
                              @ offset_q)
        else:
            s_arm = src_arm_local[rd['source']]
            S_pw_q = (src_mw @ s_arm).to_quaternion()
            if rd['needs_delta']:
                thresh_deg = rd['thresh_value']
                pair_scale = rd['pair_scale']
                S_rest_q = rd['S_rest_q']
                delta_q = S_pw_q @ rd['S_rest_q_inv']
                axis, angle = delta_q.to_axis_angle()
                if thresh_deg > 0.0 and abs(angle) < math.radians(thresh_deg):
                    angle = 0.0
                if abs(pair_scale - 1.0) > 1e-6:
                    angle = angle * pair_scale
                S_pw_q = Quaternion(axis, angle) @ S_rest_q
            out_q = S_pw_q @ rd['offset_q']
    if hl_basis_rot_q is not None:
        basis_rot_q = hl_basis_rot_q
    elif out_q is not None:
        # parent_arm_rot_q is the CARRY frame's rotation for
        # armature-carried parentless bones (CloudRig toes)
        # — that's what divides the carry out of the
        # absolute world-delta target so playback's
        # re-applied constraint lands it back on it.
        desired_arm_rot_q = tgt_mw_rot_inv @ out_q
        basis_rot_q = (rel_rest_rot_q_inv
                       @ parent_arm_rot_q.inverted()
                       @ desired_arm_rot_q)
    if basis_rot_q is not None:
        prev_q = last_q.get(tgt_name)
        if prev_q is not None and prev_q.dot(basis_rot_q) < 0.0:
            basis_rot_q = Quaternion((-basis_rot_q.w, -basis_rot_q.x,
                                      -basis_rot_q.y, -basis_rot_q.z))
        last_q[tgt_name] = basis_rot_q.copy()
    return basis_rot_q


def _compute_location_basis(meta, basis_loc, wrote_loc, use_world_loc,
                            src_arm_local, src_basis_loc_cache, src_mw,
                            scale_ratio, hl_src_name, hl_tgt_name,
                            hl_motion_correction_q, computed_tgt_arm_local,
                            basis_parent_arm, rel_rest):
    """Per-frame stage, extracted verbatim from the `if any_loc:` branch of
    _matrix_bake_pairs_iter_fast (BASIS world-frame accumulation +
    HEAD_LOCAL algebraic back-solve). Returns (basis_loc, wrote_loc)."""
    basis_loc_rds = meta['basis_loc_rds']
    # BASIS-path world-frame accumulator. Partial-axes
    # filters select WORLD components, not target-local
    # ones — the shipped presets pair "Root -> XY" (ground
    # travel) with "Hips -> Z" (height), which only means
    # what it says in world space. Target-local filtering
    # happened to work while every preset's hips/root bone
    # had local Z ~ world Z (Rigify root/torso, ARP c_pos/
    # c_root_master.x, Mixamo Hips); CloudRig's TORSO-Spine
    # points straight up with roll 0 (local Z = world -Y),
    # so a Z-only pair kept forward/back motion and DROPPED
    # the squat. Accumulating per WORLD component also keeps
    # split-axis pairs on the SAME bone composable
    # (blendcap_to_mixamo: Root->Hips XY + Hips->Hips Z).
    # Full-XYZ pairs round-trip world-and-back exactly, so
    # face presets are unaffected. The world-LOC mode below
    # already filters in world; HEAD_LOCAL keeps its
    # head-frame filter by design.
    world_loc_acc = None
    t_rest_w_rot_inv = None
    for rd in basis_loc_rds:
        pair_scale = rd['pair_scale']
        thresh_mm = rd['thresh_value']
        if use_world_loc:
            # Axes filter in WORLD frame before projection
            # (mirrors the classic engine — see bake_fk.py
            # for the rationale). Denoise threshold runs on
            # the unfiltered world magnitude.
            src_arm_mat = src_arm_local[rd['source']]
            src_world_head = (src_mw @ src_arm_mat).translation
            world_delta = src_world_head - rd['S_rest_w'].translation
            if thresh_mm > 0.0 and world_delta.length < thresh_mm / 1000.0:
                world_delta = Vector((0.0, 0.0, 0.0))
            wd = Vector((
                world_delta.x if rd['axes_x'] else 0.0,
                world_delta.y if rd['axes_y'] else 0.0,
                world_delta.z if rd['axes_z'] else 0.0,
            ))
            contribution = rd['T_rest_w_rot_inv'] @ (wd * pair_scale)
            basis_loc.x = contribution.x
            basis_loc.y = contribution.y
            basis_loc.z = contribution.z
            if rd['axes_x'] or rd['axes_y'] or rd['axes_z']:
                wrote_loc = True
        else:
            src_basis_loc = src_basis_loc_cache[rd['source']]
            if thresh_mm > 0.0 and src_basis_loc.length < thresh_mm / 1000.0:
                src_basis_loc = Vector((0.0, 0.0, 0.0))
            contribution = (rd['loc_frame_q'] @ src_basis_loc
                            * scale_ratio * pair_scale)
            world_vec = rd['T_rest_w_rot'] @ contribution
            if world_loc_acc is None:
                world_loc_acc = Vector((0.0, 0.0, 0.0))
                t_rest_w_rot_inv = rd['T_rest_w_rot_inv']
            if rd['axes_x']:
                world_loc_acc.x = world_vec.x
                wrote_loc = True
            if rd['axes_y']:
                world_loc_acc.y = world_vec.y
                wrote_loc = True
            if rd['axes_z']:
                world_loc_acc.z = world_vec.z
                wrote_loc = True

    if world_loc_acc is not None and wrote_loc:
        # One rotation back into the bone's basis frame for
        # the combined world-filtered vector. All rds in
        # basis_loc_rds share this target bone, so any rd's
        # T_rest_w_rot_inv is THE bone's.
        filtered = t_rest_w_rot_inv @ world_loc_acc
        basis_loc.x = filtered.x
        basis_loc.y = filtered.y
        basis_loc.z = filtered.z

    # HEAD_LOCAL: compute desired position in target head's
    # current local frame, then convert to target arm-local,
    # then algebraically back-solve basis_loc.
    hl_rds = meta['hl_rds']
    if hl_rds:
        # Source/target head arm-locals at this frame.
        src_head_arm = src_arm_local[hl_src_name]
        tgt_head_arm = computed_tgt_arm_local[hl_tgt_name]
        src_head_arm_inv = src_head_arm.inverted()
        # basis_parent_arm (not parent_arm): for armature-
        # carried parentless bones P must be the carry
        # frame, or the head-anchored target position gets
        # baked in absolute terms and the live constraint
        # carries it AGAIN at playback (head motion applied
        # twice — CloudRig STR-I face weld handles flew).
        P = (basis_parent_arm @ rel_rest
             if basis_parent_arm is not None else rel_rest)
        P_rot_inv = P.to_3x3().inverted()
        P_translation = P.translation
        for rd in hl_rds:
            influence = float(getattr(rd['pair'], 'influence', 1.0))
            s_arm = src_arm_local[rd['source']]
            pair_scale = rd['pair_scale']
            thresh_mm = rd['thresh_value']

            # Amplitude pivot (rd['scale_residual'], per-pair opt-in):
            #   False (DEFAULT) — multiplier scales the RAW captured
            #     delta, head-anchored. The long-standing behavior every
            #     ARP/Rigify face tuning was calibrated against; pivoting
            #     it elsewhere made amplified ARP lower lips overshoot
            #     during jaw motion (regression, 2026-06-11).
            #   True — multiplier scales the back-solved RESIDUAL over
            #     the parent/carry. For handles whose parent already
            #     follows the jaw, scaling the raw delta re-scales that
            #     built-in follow and warps speech (CloudRig graded lip
            #     rows wave everywhere except smiles); the residual is
            #     the expression on top, which is what grading means.
            # For bones whose parent doesn't move and scale 1.0, the two
            # are identical.
            scale_residual = rd.get('scale_residual', False)

            # Source position in source head's current local frame.
            src_cur_h = (src_head_arm_inv @ s_arm).translation
            motion_h = src_cur_h - rd['src_rest_h']
            if thresh_mm > 0.0 and motion_h.length < thresh_mm / 1000.0:
                motion_h = Vector((0.0, 0.0, 0.0))
            if scale_residual:
                # scale_ratio only (cross-rig unit conversion); the
                # multiplier applies to the residual after back-solve.
                motion_h = motion_h * scale_ratio
            else:
                motion_h = motion_h * scale_ratio * pair_scale
            if not rd['axes_x']:
                motion_h.x = 0.0
            if not rd['axes_y']:
                motion_h.y = 0.0
            if not rd['axes_z']:
                motion_h.z = 0.0
            # Rotate motion_h from source head rest frame to
            # target head rest frame so the addition with
            # tgt_rest_h is geometrically coherent. See setup
            # block for the full rationale.
            motion_h = hl_motion_correction_q @ motion_h

            # Target desired position in target head's local
            # frame, then to target arm-local.
            tgt_desired_h = rd['tgt_rest_h'] + motion_h
            hl_target_arm = tgt_head_arm @ tgt_desired_h

            # Optional lerp with the BASIS-path arm position
            # so influence < 1.0 dials back the head-local
            # correction (matches classic behavior).
            if influence < 1.0:
                src_basis_l = src_basis_loc_cache[rd['source']]
                basis_contribution = (rd['loc_frame_q'] @ src_basis_l
                                      * scale_ratio)
                if not scale_residual:
                    basis_contribution = basis_contribution * pair_scale
                basis_loc_vec = Vector((0.0, 0.0, 0.0))
                if rd['axes_x']:
                    basis_loc_vec.x = basis_contribution.x
                if rd['axes_y']:
                    basis_loc_vec.y = basis_contribution.y
                if rd['axes_z']:
                    basis_loc_vec.z = basis_contribution.z
                # P @ vec applies full rotation + translation
                # (treats vec as a point), producing the
                # BASIS-path arm-local position directly.
                basis_target_arm = P @ basis_loc_vec
                hl_target_arm = basis_target_arm.lerp(hl_target_arm, influence)

            # Algebraic back-solve: basis_loc such that
            # (P @ Translation(basis_loc)).translation = hl_target_arm.
            # Solves P.rot @ basis_loc + P.translation = hl_target_arm.
            basis_loc = P_rot_inv @ (hl_target_arm - P_translation)
            if scale_residual and abs(pair_scale - 1.0) > 1e-9:
                # basis_loc IS the residual over the parent/carry,
                # expressed in that frame — scaling it here is the
                # opt-in semantics described above.
                basis_loc = basis_loc * pair_scale
            wrote_loc = True
    return basis_loc, wrote_loc


def _compose_and_constrain(tgt_name, meta, constraint_meta, basis_rot_q,
                           basis_loc, wrote_loc, parent_arm, rel_rest,
                           last_q, tgt_mw, tgt_mw_inv,
                           computed_tgt_arm_local):
    """Per-frame stage, extracted verbatim from _matrix_bake_pairs_iter_fast:
    compose arm-local + constraint compensation/application. Writes
    computed_tgt_arm_local[tgt_name] and mutates last_q in place. Returns
    (basis_loc, basis_rot_q, wrote_loc) for the keyframe accumulation."""
    # ----------------------------------------------------------
    # Compose arm-local. Three paths based on whether the bone
    # is mapped and whether it has supported constraints.
    # ----------------------------------------------------------
    cons = constraint_meta.get(tgt_name)
    is_mapped = meta['is_mapped']

    if basis_rot_q is None:
        basis_rot_mat = Matrix.Identity(4)
    else:
        basis_rot_mat = basis_rot_q.to_matrix().to_4x4()
    basis_mat = Matrix.Translation(basis_loc) @ basis_rot_mat
    if parent_arm is not None:
        arm_local_pre = parent_arm @ rel_rest @ basis_mat
    else:
        arm_local_pre = rel_rest @ basis_mat

    if not cons:
        # No constraints — pre == post. Common path.
        arm_local_post = arm_local_pre

    elif is_mapped and meta['hl_rds']:
        # HEAD_LOCAL bone with constraints. The classic engine
        # uses `pb.matrix = new_mat` for HEAD_LOCAL, which back-
        # solves the basis through constraints — that's the
        # behavior we replicate here via algebraic compensation.
        # Non-HEAD_LOCAL mapped bones with constraints (the
        # branch below) intentionally do NOT compensate, matching
        # classic's direct-write behavior for rotation_* / location
        # properties: the constraint is allowed to displace the
        # bone, which is usually what the rig is designed to do.
        desired_arm = arm_local_pre
        new_basis_mat = _compensate_basis_for_constraints(
            desired_arm, cons, parent_arm, rel_rest,
            tgt_mw, tgt_mw_inv, computed_tgt_arm_local)
        if new_basis_mat is None:
            # Bone is fully slaved by COPY_TRANSFORMS REPLACE
            # (influence 1). Basis values are overridden at
            # playback — leave the fast-pass values in place
            # (harmless), don't bother compensating.
            pass
        else:
            # Use compensated basis. Extract loc + rot from the
            # new basis matrix, ensure BOTH get keyed since
            # compensation can introduce non-zero values on
            # channels the source didn't directly drive (e.g.
            # Child Of on a HEAD_LOCAL-only bone requires
            # rotation compensation too).
            basis_loc = new_basis_mat.to_translation()
            basis_rot_q = new_basis_mat.to_quaternion()
            # Quaternion-sign continuity for compensated rotations
            # (the rotation branch's continuity check ran on the
            # pre-compensation value and gets overwritten here).
            prev_q = last_q.get(tgt_name)
            if prev_q is not None and prev_q.dot(basis_rot_q) < 0.0:
                basis_rot_q = Quaternion((-basis_rot_q.w, -basis_rot_q.x,
                                          -basis_rot_q.y, -basis_rot_q.z))
            last_q[tgt_name] = basis_rot_q.copy()
            wrote_loc = True
            basis_mat = (Matrix.Translation(basis_loc)
                         @ basis_rot_q.to_matrix().to_4x4())
            if parent_arm is not None:
                arm_local_pre = parent_arm @ rel_rest @ basis_mat
            else:
                arm_local_pre = rel_rest @ basis_mat
        # Apply constraints to compose the actual post arm_local
        # children will read.
        arm_local_post = _apply_constraint_chain(
            arm_local_pre, cons, tgt_mw, tgt_mw_inv,
            computed_tgt_arm_local, parent_arm, rel_rest)
    else:
        # Unmapped bone with constraints (or mapped bone with no
        # animation pair feeding it). Basis stays at the pre-
        # computed value (typically zero/rest). Apply constraints
        # to derive post arm_local.
        #
        # Mapped bones whose pose is fully carried by an
        # ARMATURE constraint (CloudRig FK-Toes, the STR-I face
        # weld handles) need NO special handling here anymore:
        # the basis math above already computed against the
        # carry frame (basis_parent_arm), so applying the
        # constraint below — and Blender applying it live at
        # playback — lands the bone exactly where the pair
        # aimed. Copy-pose constraints and Child Of stay
        # un-compensated by design (see branch comment above).
        arm_local_post = _apply_constraint_chain(
            arm_local_pre, cons, tgt_mw, tgt_mw_inv,
            computed_tgt_arm_local, parent_arm, rel_rest)

    computed_tgt_arm_local[tgt_name] = arm_local_post
    return basis_loc, basis_rot_q, wrote_loc


def _apply_leg_anchor_compensation(frame, comp_legs, comp_baseline,
                                   comp_target_name, src_mw, tgt_mw,
                                   src_arm_local, computed_tgt_arm_local,
                                   target_meta, key_accum, primed_channels):
    """Per-frame stage, extracted verbatim from the `if comp_active:` block
    of _matrix_bake_pairs_iter_fast. Mutates key_accum, primed_channels and
    computed_tgt_arm_local in place."""
    n_legs = len(comp_legs)
    src_pose = Vector((0.0, 0.0, 0.0))
    tgt_pose = Vector((0.0, 0.0, 0.0))
    for s_name, t_name in comp_legs:
        src_pose += (src_mw @ src_arm_local[s_name]).translation
        tgt_pose += (tgt_mw @ computed_tgt_arm_local[t_name]).translation
    src_pose /= n_legs
    tgt_pose /= n_legs
    delta_world = (src_pose - tgt_pose) - comp_baseline
    if delta_world.length > 1e-5:
        # World delta → comp_target arm-space delta → basis-space
        # delta (rotated into comp_target's parent frame).
        arm_delta = tgt_mw.inverted().to_3x3() @ delta_world
        ct_meta = target_meta[comp_target_name]
        ct_parent_name = ct_meta['parent_name']
        if ct_parent_name is None:
            P_ct = ct_meta['rel_rest']
        else:
            P_ct = computed_tgt_arm_local[ct_parent_name] @ ct_meta['rel_rest']
        P_ct_rot_inv = P_ct.to_3x3().inverted()
        basis_delta = P_ct_rot_inv @ arm_delta

        # Patch / create accumulator entries for all 3 axes.
        # If the BASIS pair only filtered Z (typical torso-only-
        # Z pair), X and Y won't have an entry for this frame —
        # create one at basis_delta.x / .y. If an entry exists
        # for this frame, add delta to the existing value.
        ct_bone_path = ct_meta['bone_path']
        primed_channels.add((comp_target_name, 'location'))
        for axis_idx in range(3):
            key = (ct_bone_path + '.location', axis_idx)
            lst = key_accum.get(key)
            if lst is None:
                key_accum[key] = [frame, basis_delta[axis_idx]]
            elif len(lst) >= 2 and lst[-2] == frame:
                lst[-1] = lst[-1] + basis_delta[axis_idx]
            else:
                lst.extend((frame, basis_delta[axis_idx]))

        # Keep computed_tgt_arm_local in sync so any further
        # per-frame reads (none right now, but defensive)
        # see the corrected pose. Children's basis values are
        # unchanged — they'll inherit the comp_target shift
        # at playback when Blender composes the chain.
        ct_arm = computed_tgt_arm_local[comp_target_name].copy()
        ct_arm.translation = ct_arm.translation + arm_delta
        computed_tgt_arm_local[comp_target_name] = ct_arm


def _matrix_bake_pairs_iter_fast(pairs, src_rig, tgt_rig, frame_start, frame_end,
                                 mapped_bones=None, source_rest_override=None,
                                 override_rotation_only=True):
    """Fcurve-direct FK bake. GENERATOR — same protocol as the classic
    engine: yields once per frame during the per-frame phase, then
    yields from `_bulk_write_keys` to drive the keyframe flush.

    `source_rest_override` (per-bone source pose snapshots from the
    operator's "Use current pose as rest" / custom rest pose preset
    paths) is honored — modifies S_rest_w in the same way the classic
    engine does. `override_rotation_only` (default True) keeps the
    edit-bone translation but uses the snapshot's rotation.
    """
    scene = bpy.context.scene
    compensation_on = bool(getattr(scene, 'blendcap_face_capture_compensation', False))
    denoise_on = bool(getattr(scene, 'blendcap_face_denoise', False))
    # World-space LOC toggle. Mirrors the classic engine. With the
    # fast engine the per-frame source arm-local matrix already comes
    # from src_arm_local[sname] (built without depsgraph), so
    # converting to world is just src_mw @ that matrix — no extra
    # cost beyond a 4x4 multiply.
    use_world_loc = bool(getattr(scene, 'blendcap_retarget_use_world_location', False))

    _eval_pbs, _eff_influence = _settle_target_drivers_and_eval(scene, tgt_rig)

    src_mw = src_rig.matrix_world
    tgt_mw = tgt_rig.matrix_world
    scale_ratio = _compute_scale_ratio(src_mw, tgt_mw)

    rest_data, src_bone_names = _build_rest_data(
        pairs, src_rig, tgt_rig, src_mw, tgt_mw, source_rest_override,
        override_rotation_only, compensation_on, denoise_on, scene,
        mapped_bones)

    if not rest_data:
        return

    (src_walk_set, _depth_src, src_sorted, src_rel_rest, src_parent,
     src_fcurves) = _setup_source_skeleton(src_rig, src_bone_names)

    (by_target, constraint_meta, target_names_sorted,
     target_meta) = _setup_target_side(rest_data, tgt_rig, _eval_pbs,
                                       _eff_influence)

    tgt_mw_rot_inv = tgt_mw.to_quaternion().inverted()
    tgt_mw_inv = tgt_mw.inverted()

    (comp_active, comp_target_name, comp_baseline,
     comp_legs) = _setup_leg_anchor_compensation(
         pairs, src_rig, tgt_rig, src_mw, tgt_mw, by_target, src_walk_set,
         target_meta)

    (hl_src_name, hl_tgt_name, hl_motion_correction_q, src_sorted,
     src_fcurves) = _setup_head_local(
         scene, src_rig, tgt_rig, rest_data, src_walk_set, src_rel_rest,
         src_parent, src_sorted, src_fcurves, _depth_src,
         target_names_sorted, target_meta)

    # ------------------------------------------------------------------
    # Per-frame loop. Maintains computed_tgt_arm_local (per frame) so
    # depth-sorted child reads see fresh parent matrices without ever
    # touching Blender's depsgraph.
    # ------------------------------------------------------------------
    last_q = {}
    key_accum = {}
    primed_channels = set()

    for frame in range(frame_start, frame_end + 1):
        src_arm_local, src_basis_loc_cache, src_basis_rot_cache = (
            _eval_source_frame(frame, src_sorted, src_fcurves, src_parent,
                               src_rel_rest))

        # Step 2: target side. Depth-sorted; compute basis + arm-local
        # per bone, accumulate keyframes.
        computed_tgt_arm_local = {}

        for tgt_name in target_names_sorted:
            meta = target_meta[tgt_name]
            rd_list = meta['rd_list']
            any_rot = meta['any_rot']
            any_loc = meta['any_loc']
            rel_rest = meta['rel_rest']
            rel_rest_rot_q_inv = meta['rel_rest_rot_q_inv']
            parent_name = meta['parent_name']

            # Parent arm-local — always from computed_tgt_arm_local
            # because depth-sorted iteration over tgt_walk_set covers
            # every ancestor of every mapped target before we reach it.
            if parent_name is None:
                parent_arm = None
                parent_arm_rot_q = Quaternion()
            else:
                parent_arm = computed_tgt_arm_local[parent_name]
                parent_arm_rot_q = (tgt_mw @ parent_arm).to_quaternion()

            # Armature-carry-as-parent (see the setup comment at
            # carry_con detection): the basis math below — world-delta
            # ROT back-solve and HEAD_LOCAL LOC's P matrix — sees the
            # live carry frame as this bone's parent. The compose path
            # further down keeps the REAL (absent) parent, so the
            # constraint sim / Blender applies the carry exactly once.
            basis_parent_arm = parent_arm
            carry_con = meta['carry_con']
            if carry_con is not None:
                D_carry = _armature_delta_from_scan(
                    carry_con, computed_tgt_arm_local)
                if D_carry is not None:
                    basis_parent_arm = D_carry
                    parent_arm_rot_q = (tgt_mw @ D_carry).to_quaternion()

            basis_rot_q = None
            basis_loc = Vector((0.0, 0.0, 0.0))
            wrote_loc = False

            # --- Rotation: world-delta transfer for body bones,
            # head-local transfer for face bones (rd['hl_rot']==True).
            # Head-local avoids the chain-rotation residual that
            # contaminates face when body chain rests don't match
            # exactly between source and target (the canonical case).
            if any_rot:
                basis_rot_q = _compute_rotation_basis(
                    rd_list, src_basis_rot_cache, src_arm_local, src_mw,
                    tgt_mw_rot_inv, rel_rest_rot_q_inv, parent_arm_rot_q,
                    last_q, tgt_name)

            # --- Location: BASIS path first, then HEAD_LOCAL on top.
            # HEAD_LOCAL solves the basis_loc algebraically (no pb.matrix
            # back-solve through target constraints — works when the
            # target face bone has no Child Of / Copy constraints, which
            # is the typical case for FK face controls).
            if any_loc:
                basis_loc, wrote_loc = _compute_location_basis(
                    meta, basis_loc, wrote_loc, use_world_loc, src_arm_local,
                    src_basis_loc_cache, src_mw, scale_ratio, hl_src_name,
                    hl_tgt_name, hl_motion_correction_q,
                    computed_tgt_arm_local, basis_parent_arm, rel_rest)

            basis_loc, basis_rot_q, wrote_loc = _compose_and_constrain(
                tgt_name, meta, constraint_meta, basis_rot_q, basis_loc,
                wrote_loc, parent_arm, rel_rest, last_q, tgt_mw, tgt_mw_inv,
                computed_tgt_arm_local)

            # Accumulate keyframes for the channels we wrote. Continuity
            # tracking (last_q sign flip) happened upstream in the rotation
            # / HEAD_LOCAL branches above — pass last_q=None to skip it.
            bone_path = meta['bone_path']
            _accumulate_basis_key(
                key_accum, primed_channels, bone_path, tgt_name,
                frame, meta['rotation_mode'],
                basis_loc if wrote_loc else None,
                basis_rot_q)

        # ------------------------------------------------------------
        # Leg-anchor compensation post-step.
        # ------------------------------------------------------------
        # Measures the per-frame drift between source leg-roots and
        # target leg-roots (in WORLD space) and adds the diff to the
        # comp_target's arm-local translation. Self-zeroing on rigs
        # whose leg chain doesn't inherit from the spine. Identical
        # math to the classic engine, but reads world positions from
        # our computed arm_locals instead of pb.head.
        if comp_active:
            _apply_leg_anchor_compensation(
                frame, comp_legs, comp_baseline, comp_target_name, src_mw,
                tgt_mw, src_arm_local, computed_tgt_arm_local, target_meta,
                key_accum, primed_channels)

        yield frame

    # Same bulk-write tail as the classic engine. ESC during this phase
    # can leave partial fcurves; the per-frame phase above is cancellable
    # cleanly because no Blender state has been mutated yet.
    yield from _bulk_write_keys(tgt_rig, key_accum, primed_channels, frame_start)
