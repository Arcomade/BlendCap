"""Per-frame FK matrix bake — world-space delta-from-rest math.

Why not raw COPY_ROTATION WORLD REPLACE (prior implementation)?
  COPY_ROTATION REPLACE in world space forces the target bone's world
  orientation to equal the source bone's world orientation — INCLUDING
  roll. When source and target rest poses differ (different bone rolls,
  different A/T-pose conventions, or — worst case — a source stub like
  SAM3D's Hips pointing forward-down while the corresponding target bone
  points up) the delta from target's rest is wrong on every frame, so
  arms/hands flip ~180° around their length and spines point the wrong
  way. This is the #1 retargeting complaint across most addons in the
  wild.

What we do instead — world-space delta transfer:
  For each pair we precompute:
    S_rest_w = source bone's rest world matrix
               (edit-bone rest by default, or the source's *live* pose
                sampled at frame_start if "use current pose as rest"
                is on — user aligns source to target by posing the
                source once, no edit-mode bone roll work needed)
    T_rest_w = target bone's edit-bone rest world matrix
    offset_q = S_rest_q.inverted() @ T_rest_q
  Per frame:
    out_q = S_pose_w_q @ offset_q
          = S_pose_w @ S_rest_w.inverted() @ T_rest_w
          = (source's world delta-from-rest) applied to target's rest
  This makes the target's delta-from-ITS-rest match the source's
  delta-from-ITS-rest in world space, which is what "motion transfer"
  means visually. Works across arbitrary rest-pose mismatches.
"""
import bpy
import math

from mathutils import Matrix, Quaternion, Vector

from .fcurves import _bulk_write_keys
from .bake_fk_common import _compute_scale_ratio, _compute_pair_rest_invariants


# --- Engine switch ----------------------------------------------------
# When True, _matrix_bake_pairs_iter delegates to the fcurve-direct
# engine in bake_fk_fast.py — bypasses scene.frame_set + view_layer
# .update() per frame by evaluating source rig fcurves directly. Big
# speedup on rigs where Blender's per-frame depsgraph eval of the full
# pose graph dominates (1000+ bone control rigs like ARP Tifa).
#
# V1 limitations: source rigs with drivers/constraints/IK won't
# evaluate correctly (raw-fcurve-only path), HEAD_LOCAL face math
# falls back to BASIS, leg-anchor compensation disabled, source_rest
# _override / use-current-pose-as-rest not supported. See bake_fk_fast
# .py module docstring for the full list.
#
# The fast engine is the shipped default. To A/B against the classic
# per-frame depsgraph engine when debugging a rig, flip from a Python
# console:
#     import blendcap.retarget.bake_fk as bf
#     bf.USE_FAST_ENGINE = False
USE_FAST_ENGINE = True


def _compute_rig_height(rig):
    """Vertical extent (world Z) of a rig measured across all pose-bone
    head positions. Used for auto-scale. 0 if the rig has no bones."""
    if not rig or rig.type != 'ARMATURE' or not rig.pose.bones:
        return 0.0
    mw = rig.matrix_world
    zs = [(mw @ pb.head).z for pb in rig.pose.bones]
    if not zs:
        return 0.0
    return max(zs) - min(zs)


def _matrix_bake_pairs_iter(pairs, src_rig, tgt_rig, frame_start, frame_end,
                            mapped_bones=None, source_rest_override=None,
                            override_rotation_only=True):
    """Per-frame matrix bake transferring source motion onto target using
    world-space delta-from-rest math. GENERATOR: yields once per frame
    after keyframes are inserted (caller may render/redraw between yields).

    Rotation transfer is world-delta — handles cross-skeleton rest-pose
    differences (different rolls, T-pose vs A-pose).

    LOCATION transfer uses source's basis location (the per-bone channel
    value) frame-converted into the target bone's basis frame, NOT the
    bone's world position. This is the difference from earlier revisions:
        OLD: target_world = T_rest_world + (source_world - source_rest_world)
        NEW: target.location = scale * tgt_rest_rot.inv @ src_rest_rot @ source.location
    The old form bakes a child bone's parent-rotation swing into the
    location channel — when the source's parent (e.g. Head) rotates, the
    child's world position picks up the rotational sweep, which the math
    then has to cancel via a non-zero target basis_loc, dragging the
    skinned mesh visibly. Worse: with axes-filtered LOC pairs (e.g. Hips→
    torso Z), the OLD math pinned non-listed axes to T_rest_world,
    fighting any natural inheritance (root XY motion never reached the
    spine, so the upper body lagged behind the pelvis).
    The NEW form transfers the source's *local channel* — which is
    parent-rotation-free — and lets natural parent inheritance fill in
    non-listed axes.

    Non-listed axes default to 0 in basis space (target's natural rest
    relative to its parent's pose), so a child bone follows its parent
    chain naturally.

    Quaternion sign continuity: pb.matrix.to_quaternion() can return
    ±q for the same physical rotation; adjacent frames keying q then
    -q produce wobble (interpolation takes the long way). We track the
    last-keyed quaternion per bone and negate the new one if their dot
    is negative.

    Parents are updated before children each frame so child basis
    back-solves see fresh parent world matrices (and any driven MCH
    bones in Rigify chains).

    `source_rest_override` (rotation-only by default) provides a
    pre-captured per-bone armature-local matrix to use as the source's
    rotational rest baseline, instead of the source's edit-bone rest.
    Used by "use current source pose as rest" — captured BEFORE any
    scene.frame_set/auto-scale so the user's manual pose isn't wiped.
    LOC transfer ignores this override (always reads s_pb.location); set
    `override_rotation_only=False` only if the source basis location at
    snapshot time should be used as the LOC baseline (rare).

    Mutates `mapped_bones` in place (set of target bone names touched).

    Keyframes are accumulated per-channel during the frame loop, then
    bulk-written at the end via fcurve.keyframe_points.foreach_set('co').
    This avoids ~N_bones * N_channels * N_frames per-call overhead from
    pb.keyframe_insert() on every frame — the dominant cost on dense face
    bakes. Per channel, we prime the fcurve with a single keyframe_insert
    (Blender 5's slotted action API forbids action.fcurves.new), look up
    the resulting fcurve through the layers/strips/channelbags chain, then
    foreach_set the rest of the points in one C call. See _bulk_write_keys
    for the priming + slotted lookup path.
    """
    # Engine switch — fast path delegates to fcurve-direct bake. See
    # USE_FAST_ENGINE comment at module top for limitations.
    if USE_FAST_ENGINE:
        from .bake_fk_fast import _matrix_bake_pairs_iter_fast
        yield from _matrix_bake_pairs_iter_fast(
            pairs, src_rig, tgt_rig, frame_start, frame_end,
            mapped_bones=mapped_bones,
            source_rest_override=source_rest_override,
            override_rotation_only=override_rotation_only,
        )
        return

    scene = bpy.context.scene
    compensation_on = bool(getattr(scene, 'blendcap_face_capture_compensation', False))
    denoise_on = bool(getattr(scene, 'blendcap_face_denoise', False))
    # World-space LOC toggle. When ON, BASIS-mode LOC pairs are read
    # as (src_mw @ pose_matrix).translation - S_rest_w.translation so
    # source rigs with non-identity object transforms (Mixamo etc.)
    # transfer correctly. Doesn't affect HEAD_LOCAL pairs.
    use_world_loc = bool(getattr(scene, 'blendcap_retarget_use_world_location', False))

    src_mw = src_rig.matrix_world
    tgt_mw = tgt_rig.matrix_world
    # Uniform-scale assumption; per-axis non-uniform scale is rare and
    # correct retargeting under it is undefined anyway.
    scale_ratio = _compute_scale_ratio(src_mw, tgt_mw)

    rest_data = []
    for p in pairs:
        if not p.source or not p.target:
            continue
        if p.source not in src_rig.pose.bones:
            continue
        if p.target not in tgt_rig.pose.bones:
            continue

        s_pb = src_rig.pose.bones[p.source]
        t_bone = tgt_rig.data.bones[p.target]

        # Cache per-pair invariants used by the hot frame loop. pair_scale /
        # thresh_value depend on scene state captured above (compensation_on,
        # denoise_on) plus the pair's source name + loc_scale field — none
        # change during the bake. Caching here saves ~3-5 RNA reads per pair
        # per frame plus two function-call hops into _resolve_* (which each
        # do further RNA lookups). On dense face rigs that's the dominant
        # python overhead.
        rd = _compute_pair_rest_invariants(
            p, s_pb, t_bone, src_mw, tgt_mw,
            source_rest_override, override_rotation_only,
            compensation_on, denoise_on, scene)
        rd['pair'] = p
        rd['s_pb'] = s_pb
        rd['channels'] = p.channels
        rd['influence'] = float(getattr(p, 'influence', 1.0))
        rest_data.append(rd)
        if mapped_bones is not None:
            mapped_bones.add(p.target)

    if not rest_data:
        return

    by_target = {}
    for rd in rest_data:
        by_target.setdefault(rd['pair'].target, []).append(rd)

    def _depth(bone):
        d, b = 0, bone.parent
        while b:
            d += 1
            b = b.parent
        return d
    depth_by_name = {n: _depth(tgt_rig.data.bones[n]) for n in by_target.keys()}
    sorted_targets = sorted(by_target.keys(), key=lambda n: depth_by_name[n])

    tgt_mw_rot_inv = tgt_mw.to_quaternion().inverted()

    # Quaternion-sign continuity: remember last-keyed q per target bone so
    # we can flip the sign of the new q when it's on the opposite hemisphere
    # (dot < 0). Without this, fcurve interpolation between q and -q goes
    # the long way and visible wobble appears.
    last_q = {}

    # Per-fcurve accumulator: keys are (data_path, array_index), values are
    # flat [frame, value, frame, value, ...] lists. Filled below; flushed
    # once after the frame loop completes via _bulk_write_keys.
    key_accum = {}
    # Set of (bone_name, channel_name) pairs that need an fcurve primed
    # before bulk-write. channel_name is the data_path SUFFIX
    # ('location' / 'rotation_quaternion' / 'rotation_axis_angle' /
    # 'rotation_euler') — bone path is derived from bone_name at flush time.
    primed_channels = set()

    # Leg-anchor compensation. Some rigs (Rigify is the example we hit) parent
    # legs through the spine chain instead of as siblings of the spine, so
    # spine rotation drags the legs in world space. Source skeletons typically
    # have legs as siblings of the pelvis, so source motion was authored on
    # that assumption. Without correction, target feet drift by however much
    # the chain rotation displaced the leg-root.
    #
    # Per frame we measure target leg-root world position vs source leg-root
    # world position and add the difference to the LOC target bone. Two
    # subtleties:
    #
    #  - Baseline subtraction: at REST pose (no animation), source and target
    #    leg-roots are at fixed positions determined by their respective rest
    #    poses. These can differ by a few cm even on similarly-proportioned
    #    rigs, and that rest-pose mismatch isn't something the chain caused —
    #    it's just rig setup. We sample (src_rest - tgt_rest) once before the
    #    frame loop and subtract it from per-frame delta, so compensation
    #    only fires for DYNAMIC drift (chain rotation) and zeros out at rest.
    #
    #  - XYZ, not just Z: spine bending pushes the leg chain forward/back AND
    #    up/down, so X/Y compensation prevents foot moonwalking on the floor.
    #    Source pelvis XY motion is unaffected (it's already routed through
    #    the root LOC XY pair); compensation only corrects rig-induced drift.
    #
    # Auto-detect:
    #   comp_loc_pair  — first LOC/LOC_ROT pair with Z in axes whose target
    #                    isn't a root-named bone (root LOC pair carries
    #                    in-place gait XY, not pelvis lift).
    #   comp_legs      — pairs whose target name contains thigh/upleg.
    # Self-zeroing: rigs whose leg chain is independent of the spine produce
    # delta=0 every frame.
    _LEG_ROOT_KW = ('thigh', 'upleg', 'upper_leg')
    comp_loc_pair = None
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
        comp_loc_pair = p
        break
    comp_legs = []
    _seen = set()
    for p in pairs:
        if p.target in _seen:
            continue
        if not any(kw in p.target.lower() for kw in _LEG_ROOT_KW):
            continue
        if p.source not in src_rig.pose.bones or p.target not in tgt_rig.pose.bones:
            continue
        comp_legs.append((src_rig.pose.bones[p.source], tgt_rig.pose.bones[p.target]))
        _seen.add(p.target)
    comp_active = comp_loc_pair is not None and len(comp_legs) > 0
    comp_target_pb = tgt_rig.pose.bones[comp_loc_pair.target] if comp_active else None
    comp_baseline = Vector((0.0, 0.0, 0.0))
    if comp_active:
        # Baseline = (src_rest_avg - tgt_rest_avg) in world space.
        # Uses bone.head_local (edit-bone rest); source_rest_override is
        # rotation-only by default so it doesn't shift LOC, and the rare
        # full-override case gracefully degrades to a small constant offset.
        n_legs = len(comp_legs)
        src_rest = Vector((0.0, 0.0, 0.0))
        tgt_rest = Vector((0.0, 0.0, 0.0))
        for s_pb, t_pb in comp_legs:
            src_rest += src_mw @ s_pb.bone.head_local
            tgt_rest += tgt_mw @ t_pb.bone.head_local
        src_rest /= n_legs
        tgt_rest /= n_legs
        comp_baseline = src_rest - tgt_rest

    # Head-local LOC support. Pairs flagged loc_space=HEAD_LOCAL compute
    # the source bone's motion in the source head bone's local frame and
    # apply it on top of the target bone's rest in the target head bone's
    # local frame, then back-solve the basis via pb.matrix. Bypasses
    # parent-chain semantic mismatches — e.g. ARP's c_lips_bot.l rides on
    # jawbone.x ROTATION while source's lip.B.L rides on jaw_master
    # TRANSLATION, so plain basis-to-basis transfer overshoots downward.
    # Per-rd state is resolved once here so the per-frame branch is just
    # two matrix multiplies + a pb.matrix back-solve.
    hl_src_name = getattr(scene, 'blendcap_retarget_face_head_source', '') or ''
    hl_tgt_name = getattr(scene, 'blendcap_retarget_face_head_target', '') or ''
    # Bake-time safety net: if the user built the table from scratch
    # (no preset) and never touched the Advanced overrides, the Scene
    # props can still be blank when a pair has loc_space=HEAD_LOCAL.
    # The rig-picker callbacks already auto-detect at rig assignment,
    # but a preset that didn't ship face_head_* keys (or a rig swap
    # that happened before the callbacks were wired) can leave them
    # empty. Try one more auto-detect against the actual rigs here so
    # the bake doesn't silently fall back to BASIS just because the
    # Scene prop wasn't populated.
    from .name_matching import _resolve_against_rig
    src_prefix = scene.blendcap_retarget_source_prefix or ''
    tgt_prefix = scene.blendcap_retarget_namespace_strip or ''
    if not hl_src_name or not hl_tgt_name:
        from ..helpers import autodetect_head_bone_name, target_head_from_pair_table
        if not hl_src_name:
            hl_src_name = autodetect_head_bone_name(src_rig, src_prefix)
        if not hl_tgt_name:
            # Same three-stage fallback as the rig-picker callback:
            # autodetect, then pair-table lookup, then give up.
            hl_tgt_name = autodetect_head_bone_name(tgt_rig, tgt_prefix)
            if not hl_tgt_name:
                hl_tgt_name = target_head_from_pair_table(scene)
    # Resolve via the Source/Target Prefix in case the field stores the
    # SHORT bone name ("Head") and the rig has a namespace ("mixamorig:
    # Head"). _resolve_against_rig is exact-match-first so unprefixed
    # rigs are unaffected.
    hl_src_name = _resolve_against_rig(hl_src_name, src_rig, src_prefix) if hl_src_name else hl_src_name
    hl_tgt_name = _resolve_against_rig(hl_tgt_name, tgt_rig, tgt_prefix) if hl_tgt_name else hl_tgt_name
    hl_src_pb = src_rig.pose.bones.get(hl_src_name) if hl_src_name else None
    hl_tgt_pb = tgt_rig.pose.bones.get(hl_tgt_name) if hl_tgt_name else None
    hl_available = hl_src_pb is not None and hl_tgt_pb is not None
    if hl_available:
        hl_src_head_rest_inv = hl_src_pb.bone.matrix_local.inverted()
        hl_tgt_head_rest_inv = hl_tgt_pb.bone.matrix_local.inverted()
        # Frame-conversion quaternion: motion_h is computed in source
        # head's REST local frame, but it has to be applied at target
        # via tgt_head_pb.matrix (target head's POSE matrix). For the
        # geometric direction of "jaw moved forward" to survive across
        # rigs with different head rest orientations, we rotate motion_h
        # from source head rest frame into target head rest frame before
        # adding to tgt_rest_h. Without this correction, the small
        # head-rest mismatch (always present in cross-rig retargeting)
        # gets amplified by target head's body-driven pose at playback,
        # producing the "ARP chin protrudes forward with body rotation"
        # bug. In face-only mode the correction is also strictly needed,
        # but the visible discrepancy was small enough to go unnoticed
        # before the body-rotation case forced it.
        hl_motion_correction_q = (
            hl_tgt_pb.bone.matrix_local.to_quaternion().inverted()
            @ hl_src_pb.bone.matrix_local.to_quaternion())
    hl_pair_count = 0
    for rd in rest_data:
        p = rd['pair']
        space = (getattr(p, 'loc_space', 'BASIS') or 'BASIS').upper()
        if space != 'HEAD_LOCAL':
            rd['loc_space'] = 'BASIS'
            continue
        if not hl_available:
            rd['loc_space'] = 'BASIS'
            continue
        s_pb_setup = src_rig.pose.bones.get(p.source)
        t_pb_setup = tgt_rig.pose.bones.get(p.target)
        if s_pb_setup is None or t_pb_setup is None:
            rd['loc_space'] = 'BASIS'
            continue
        rd['loc_space'] = 'HEAD_LOCAL'
        rd['src_head_pb'] = hl_src_pb
        rd['tgt_head_pb'] = hl_tgt_pb
        # Bone rest position expressed in the head bone's REST local frame.
        # Per-frame we transform back into the head bone's CURRENT pose
        # frame to handle head rotation/translation correctly.
        rd['src_rest_h'] = hl_src_head_rest_inv @ s_pb_setup.bone.matrix_local.translation
        rd['tgt_rest_h'] = hl_tgt_head_rest_inv @ t_pb_setup.bone.matrix_local.translation
        hl_pair_count += 1

    # Head-local ROTATION transfer (auto-applied to face bones). See
    # bake_fk_fast.py for the full rationale — short version: world-delta
    # rotation transfer leaves a residual on face bones whose source has
    # chain rotation above the head (e.g. Hips body yaw) when source and
    # target chain rests don't match. Reading source basis directly
    # (chain-pose invariant) and applying via a setup-time rest
    # correction bypasses the residual cleanly. Auto-applied to any
    # pair whose source is a direct child of the source head bone.
    hl_src_children = set()
    if hl_available:
        hl_src_children = {b.name for b in hl_src_pb.bone.children}
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
        # No new fields — per-frame branch reuses rd['offset_q'] for
        # the conjugation. See bake_fk_fast.py for the full derivation.

    # Per-target metadata cache — everything below is rest-pose invariant
    # but was being recomputed every frame for every target. On a 140-bone
    # face rig × 300 frames that's ~42k redundant dict-lookup chains, matrix
    # builds, and `any(...)` walks. Built once after HEAD_LOCAL setup
    # (which mutates rd['loc_space']) so hl_rds / basis_rds resolve correctly.
    target_meta = {}
    for tgt_name in sorted_targets:
        t_pb = tgt_rig.pose.bones[tgt_name]
        t_bone = t_pb.bone
        rd_list = by_target[tgt_name]
        parent_bone = t_bone.parent
        parent_pb = tgt_rig.pose.bones.get(parent_bone.name) if parent_bone else None

        if parent_bone is None:
            rel_rest = t_bone.matrix_local
        else:
            rel_rest = parent_bone.matrix_local.inverted() @ t_bone.matrix_local
        rel_rest_rot_q_inv = rel_rest.to_quaternion().inverted()

        any_rot_t = any(rd['in_rot'] for rd in rd_list)
        any_loc_t = any(rd['in_loc'] for rd in rd_list)
        # hl_rds / basis_rds invariance: rd['loc_space'] is set during the
        # HEAD_LOCAL setup block above and never mutated afterward; any_rot_t
        # is also invariant. Both filters are pure-derived from invariants.
        hl_rds = ([rd for rd in rd_list if rd.get('loc_space') == 'HEAD_LOCAL']
                  if not any_rot_t else [])
        basis_rds = [rd for rd in rd_list if rd not in hl_rds and rd['in_loc']]

        bone_key = tgt_name.replace('"', '\\"')
        bone_path = f'pose.bones["{bone_key}"]'

        target_meta[tgt_name] = {
            't_pb': t_pb,
            't_bone': t_bone,
            'parent_pb': parent_pb,
            'rel_rest_rot_q_inv': rel_rest_rot_q_inv,
            'any_rot': any_rot_t,
            'any_loc': any_loc_t,
            'hl_rds': hl_rds,
            'basis_rds': basis_rds,
            'bone_path': bone_path,
        }

    # Two-pass warm-up on frame_start. The per-bone processing below
    # writes pose values for each target in depth-sorted order, reading
    # parent_arm_rot_q from the target's parent at write time. This
    # assumes target parent-chain depth matches dependency order, which
    # holds on simple rigs but NOT on rigs where some target bones
    # parent through deform/MCH chains that take a shorter path to root
    # than other control bones do (Tifa ARP: c_pinky1.l data-depth=6
    # vs c_shoulder.l depth=11, so fingers are sorted ahead of the arm
    # chain even though they semantically depend on it via deform-bone
    # constraints). On a fresh bake (no prior fcurves), that misorder
    # made finger bones read still-at-rest arm matrices on frame_start
    # and bake the wrong basis into the keys — visible as mangled-curl
    # finger frames at the start of every retarget. Frame 2+ wasn't
    # affected because the previous frame's end-of-frame view_layer.
    # update at line 602 left the arm chain at posed state, which
    # scene.frame_set on frame N+1 then evaluated correctly against.
    # Re-baking the same animation fixed it the same way: the second
    # bake's scene.frame_set(frame_start) loaded the first bake's
    # already-correct fcurves for the arm chain.
    #
    # Fix without changing sort order or adding source/target hierarchy
    # assumptions: on frame_start, run the per-bone processing TWICE.
    # The first pass writes pose values (some wrong, given out-of-
    # order parent reads). End-of-pass view_layer.update settles
    # everything. The second pass re-runs in the same order, but its
    # parent reads now see pass-1's writes (correct), so it computes
    # and writes the correct basis. Only pass 2 captures into
    # key_accum. Mirrors what "re-bake fixes it" does naturally —
    # primes in-memory pose so reads are against settled state.
    #
    # Implemented via a generator that yields (frame, capture_flag)
    # — warm-up pass yields (frame_start, False), then normal frames
    # yield (frame, True). Avoids wrapping the existing per-frame body
    # in another nested loop.
    def _frame_passes():
        yield (frame_start, False)
        for f in range(frame_start, frame_end + 1):
            yield (f, True)

    for frame, _capture_keys in _frame_passes():
        scene.frame_set(frame)
        # scene.frame_set tags the depsgraph but doesn't always flush
        # driver/constraint/matrix chains synchronously. Belt-and-
        # suspenders flush so per-bone matrix reads below see settled
        # state from the start. One eval per frame, cheap vs the per-
        # bone evals we already collapsed at depth boundaries.
        bpy.context.view_layer.update()

        # Track which bones got rot/loc this frame, for the keyframe-insert
        # pass. Built fresh each frame because partial pair sets per bone
        # can vary depending on which source bones exist.
        rot_set = set()
        loc_set = set()

        # Depsgraph re-evaluation (view_layer.update) used to fire per
        # bone here — that meant 139 evals per frame for the new-face
        # preset (vs 24 for body-only), making face retargeting ~6×
        # slower than body. Same-depth bones don't reference each other
        # in the hierarchy, so we only need to flush at depth boundaries:
        # children at depth D+1 read parent matrices from depth D, but a
        # face bone at depth D doesn't read another face bone at depth D.
        # Cuts the eval count from O(bones) to O(distinct_depths) per
        # frame — typically 8-10 evals instead of 100+.
        # NOTE: if you have a Rigify rig with sibling-driven MCH bones
        # at the same depth, this can produce stale-parent-matrix wobble
        # — fall back to per-bone updates by setting prev_depth = -1
        # outside the loop and updating every iteration.
        prev_depth = None

        for tgt_name in sorted_targets:
            cur_depth = depth_by_name[tgt_name]
            if prev_depth is not None and cur_depth != prev_depth:
                bpy.context.view_layer.update()
            prev_depth = cur_depth

            meta = target_meta[tgt_name]
            t_pb = meta['t_pb']
            rd_list = by_target[tgt_name]
            any_rot = meta['any_rot']
            any_loc = meta['any_loc']
            parent_pb = meta['parent_pb']
            rel_rest_rot_q_inv = meta['rel_rest_rot_q_inv']

            # parent_arm_rot_q is the only per-frame value at the target level
            # — depends on the parent's current pose matrix.
            if parent_pb is None:
                parent_arm_rot_q = Quaternion()  # identity
            else:
                parent_arm_rot_q = parent_pb.matrix.to_quaternion()

            # --- Rotation: world-delta transfer for body bones,
            # head-local transfer for face bones (rd['hl_rot']==True).
            # Head-local avoids the residual chain-rotation contamination
            # that the world-delta formula leaves on face bones when
            # source/target chain rests don't match exactly.
            if any_rot:
                out_q = None
                hl_basis_rot_q = None
                for rd in rd_list:
                    if not rd['in_rot']:
                        continue
                    if rd.get('hl_rot'):
                        # Head-local: source basis is chain-pose
                        # invariant. Apply denoise/scale to basis
                        # directly, then map into target's basis frame
                        # via the setup-time rest correction.
                        src_pb = rd['s_pb']
                        mode = src_pb.rotation_mode
                        if mode == 'QUATERNION':
                            src_basis_q = src_pb.rotation_quaternion.copy()
                        elif mode == 'AXIS_ANGLE':
                            aa = src_pb.rotation_axis_angle
                            src_basis_q = Quaternion(
                                (aa[1], aa[2], aa[3]), aa[0])
                        else:
                            src_basis_q = src_pb.rotation_euler.to_quaternion()
                        if rd['needs_delta']:
                            thresh_deg = rd['thresh_value']
                            pair_scale = rd['pair_scale']
                            axis, angle = src_basis_q.to_axis_angle()
                            if thresh_deg > 0.0 and \
                                    abs(angle) < math.radians(thresh_deg):
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
                        S_pw_q = (src_mw @ rd['s_pb'].matrix).to_quaternion()
                        # Denoise dead-zone (units: degrees of rotation
                        # delta from rest). Then amplitude scaling. Both
                        # extract the delta-from-rest, so we compute it
                        # once and chain. needs_delta is False when both
                        # short-circuit (denoise=0 AND pair_scale=1) —
                        # computed once per pair at setup.
                        if rd['needs_delta']:
                            thresh_deg = rd['thresh_value']
                            pair_scale = rd['pair_scale']
                            S_rest_q = rd['S_rest_q']
                            delta_q = S_pw_q @ rd['S_rest_q_inv']
                            axis, angle = delta_q.to_axis_angle()
                            # Denoise: if rotation magnitude below
                            # threshold, snap to rest.
                            if thresh_deg > 0.0 and \
                                    abs(angle) < math.radians(thresh_deg):
                                angle = 0.0
                            if abs(pair_scale - 1.0) > 1e-6:
                                angle = angle * pair_scale
                            S_pw_q = Quaternion(axis, angle) @ S_rest_q
                        out_q = S_pw_q @ rd['offset_q']
                basis_rot_q = None
                if hl_basis_rot_q is not None:
                    basis_rot_q = hl_basis_rot_q
                elif out_q is not None:
                    desired_arm_rot_q = tgt_mw_rot_inv @ out_q
                    basis_rot_q = (rel_rest_rot_q_inv @
                                   parent_arm_rot_q.inverted() @
                                   desired_arm_rot_q)
                if basis_rot_q is not None:
                    prev_q = last_q.get(tgt_name)
                    if prev_q is not None and prev_q.dot(basis_rot_q) < 0.0:
                        basis_rot_q = Quaternion((-basis_rot_q.w, -basis_rot_q.x,
                                                  -basis_rot_q.y, -basis_rot_q.z))
                    last_q[tgt_name] = basis_rot_q.copy()

                    mode = t_pb.rotation_mode
                    if mode == 'QUATERNION':
                        t_pb.rotation_quaternion = basis_rot_q
                    elif mode == 'AXIS_ANGLE':
                        axis, angle = basis_rot_q.to_axis_angle()
                        t_pb.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
                    else:
                        t_pb.rotation_euler = basis_rot_q.to_euler(mode)
                    rot_set.add(tgt_name)

            # --- Location: per-pair transfer.
            #
            # BASIS path (default): write source basis-local LOC into target
            # basis-local LOC via a pre-computed frame quaternion. Correct
            # when source and target share equivalent parent chain semantics.
            # Each pair overrides only its listed axes; default basis_loc
            # components are 0 (target's natural rest). Cross-rig unit
            # conversion (scale_ratio) and the per-pair LOC multiplier are
            # applied (see _resolve_loc_scale — Face Capture Compensation
            # region defaults or explicit preset loc_scale override).
            #
            # HEAD_LOCAL path (opt-in via preset loc_space): compute source
            # bone's motion in source head's local frame, apply on top of
            # target bone's rest position in target head's local frame, then
            # use pb.matrix setter to back-solve basis through whatever
            # ancestor + constraint stack is in place. Used when source and
            # target parent chains have different semantics — e.g. one side
            # of the family has a rotation-driven parent and the other has
            # a translation-driven parent — so basis-to-basis transfer
            # would land at the wrong head-relative position.
            #
            # The pair's `influence` field acts as a HEAD_LOCAL ↔ BASIS
            # blend weight (default 1.0 = full head-local back-solve).
            # influence=0.0 degrades to pure BASIS; influence=0.5 lands
            # the bone at the midpoint between the two outcomes. Useful
            # when neither extreme is right — e.g. target parent partially
            # produces the right motion via its own rotation, so the full
            # head-local correction over-compensates.
            #
            # The two paths can coexist on the same bone in principle, but
            # HEAD_LOCAL uses pb.matrix which overwrites rotation as well,
            # so we only take that path when no rotation pair is feeding
            # this bone (any_rot==False). Mixed rotation + HEAD_LOCAL
            # silently falls back to BASIS for safety; presets should keep
            # HEAD_LOCAL pairs LOC-only.
            if any_loc:
                hl_rds = meta['hl_rds']
                basis_rds = meta['basis_rds']
                basis_loc = Vector((0.0, 0.0, 0.0))
                wrote_any = False
                for rd in basis_rds:
                    # Source basis location at this frame (BVH key value),
                    # rotated to target's basis frame. Denoise first (snap
                    # to zero if magnitude below the region's threshold in
                    # mm), then apply cross-rig unit conversion ratio
                    # (scale_ratio) and the LOC multiplier resolved per
                    # pair. The multiplier is the preset's explicit
                    # loc_scale if non-default, otherwise the region
                    # default when the scene checkbox is on, else 1.0 for
                    # pure 1:1 transfer. See _resolve_loc_scale and
                    # _resolve_denoise for precedence rules.
                    pair_scale = rd['pair_scale']
                    thresh_mm = rd['thresh_value']
                    if use_world_loc:
                        # World-space delta-from-rest, projected into
                        # target basis. scale_ratio is NOT applied — the
                        # world delta already carries source's object
                        # scale, and T_rest_w_rot_inv removes target's.
                        # Axes filter runs in WORLD frame (before
                        # projection) so the X/Y/Z choice on the pair
                        # refers to world axes, matching the user mental
                        # model when they opted into world-space LOC.
                        # Denoise threshold runs on the unfiltered
                        # magnitude so filter behavior doesn't change
                        # denoise semantics.
                        s_pb = rd['s_pb']
                        src_world_head = (src_mw @ s_pb.matrix).translation
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
                            wrote_any = True
                    else:
                        src_basis_loc = rd['s_pb'].location
                        if thresh_mm > 0.0 and src_basis_loc.length < thresh_mm / 1000.0:
                            src_basis_loc = Vector((0.0, 0.0, 0.0))
                        contribution = (rd['loc_frame_q'] @ src_basis_loc
                                        * scale_ratio * pair_scale)
                        if rd['axes_x']:
                            basis_loc.x = contribution.x
                            wrote_any = True
                        if rd['axes_y']:
                            basis_loc.y = contribution.y
                            wrote_any = True
                        if rd['axes_z']:
                            basis_loc.z = contribution.z
                            wrote_any = True
                if wrote_any:
                    t_pb.location = basis_loc
                    loc_set.add(tgt_name)
                for rd in hl_rds:
                    s_pb = rd['s_pb']
                    src_head_pb = rd['src_head_pb']
                    tgt_head_pb = rd['tgt_head_pb']
                    influence = rd['influence']
                    # Apply the same LOC multiplier the BASIS path uses,
                    # so face_capture_compensation and per-pair overrides
                    # behave consistently regardless of loc_space. Denoise
                    # similarly mirrors the BASIS path — snap motion_h to
                    # zero below the region's threshold (in mm).
                    pair_scale = rd['pair_scale']
                    thresh_mm = rd['thresh_value']

                    # HEAD_LOCAL desired arm-local position.
                    src_cur_h = src_head_pb.matrix.inverted() @ s_pb.matrix.translation
                    motion_h = src_cur_h - rd['src_rest_h']
                    if thresh_mm > 0.0 and motion_h.length < thresh_mm / 1000.0:
                        motion_h = Vector((0.0, 0.0, 0.0))
                    motion_h = motion_h * scale_ratio * pair_scale
                    if not rd['axes_x']:
                        motion_h.x = 0.0
                    if not rd['axes_y']:
                        motion_h.y = 0.0
                    if not rd['axes_z']:
                        motion_h.z = 0.0
                    # Rotate motion_h from source head's rest frame into
                    # target head's rest frame, so the addition with
                    # tgt_rest_h (which is in target head's rest frame) is
                    # geometrically coherent. Without this, head-rest
                    # mismatch between rigs gets amplified by target head's
                    # body pose at playback (ARP chin protrusion bug).
                    motion_h = hl_motion_correction_q @ motion_h
                    tgt_desired_h = rd['tgt_rest_h'] + motion_h
                    hl_target_arm = tgt_head_pb.matrix @ tgt_desired_h

                    # BASIS-path arm-local position (where the bone would
                    # land with plain basis-LOC transfer instead of the
                    # head-local back-solve). Computed analytically from
                    # the bone hierarchy — no extra depsgraph eval needed.
                    # Used when influence < 1.0 to lerp toward the BASIS
                    # outcome (lets the user dial the head-local correction
                    # back when the target rig's parent chain already
                    # produces some of the motion source's chain provides).
                    if influence < 1.0:
                        basis_contribution = (rd['loc_frame_q'] @ s_pb.location
                                              * scale_ratio * pair_scale)
                        basis_loc_vec = Vector((0.0, 0.0, 0.0))
                        if rd['axes_x']:
                            basis_loc_vec.x = basis_contribution.x
                        if rd['axes_y']:
                            basis_loc_vec.y = basis_contribution.y
                        if rd['axes_z']:
                            basis_loc_vec.z = basis_contribution.z
                        parent_pb = t_pb.parent
                        if parent_pb is not None:
                            rel_rest_local = (parent_pb.bone.matrix_local.inverted()
                                              @ t_pb.bone.matrix_local)
                            basis_target_arm = (parent_pb.matrix
                                                @ rel_rest_local
                                                @ Matrix.Translation(basis_loc_vec)
                                                ).translation
                        else:
                            basis_target_arm = (t_pb.bone.matrix_local
                                                @ Matrix.Translation(basis_loc_vec)
                                                ).translation
                        final_arm = basis_target_arm.lerp(hl_target_arm, influence)
                    else:
                        final_arm = hl_target_arm

                    new_mat = t_pb.matrix.copy()
                    new_mat.translation = final_arm
                    # Back-solve basis through ancestor chain + any
                    # constraints. The captured pb.location in the keyframe
                    # capture pass below reflects this back-solved value.
                    t_pb.matrix = new_mat
                    loc_set.add(tgt_name)

        # Final flush so the value-capture pass below sees fully-settled
        # pose values (last depth's writes need to propagate).
        bpy.context.view_layer.update()

        # Leg-anchor compensation (see setup block above the frame loop).
        # Runs after rotations but before keyframe capture so the captured
        # LOC values include the correction.
        if comp_active:
            n_legs = len(comp_legs)
            src_pose = Vector((0.0, 0.0, 0.0))
            tgt_pose = Vector((0.0, 0.0, 0.0))
            for s_pb, t_pb in comp_legs:
                src_pose += src_mw @ s_pb.head
                tgt_pose += tgt_mw @ t_pb.head
            src_pose /= n_legs
            tgt_pose /= n_legs
            # Subtract baseline so rest-pose mismatch doesn't bake in a
            # constant offset. delta_world is dynamic chain-induced drift.
            delta_world = (src_pose - tgt_pose) - comp_baseline
            if delta_world.length > 1e-5:
                # World delta in the target armature's local frame
                # (handles rotated/scaled target rigs).
                arm_delta = tgt_mw.inverted().to_3x3() @ delta_world
                new_matrix = comp_target_pb.matrix.copy()
                new_matrix.translation = comp_target_pb.matrix.translation + arm_delta
                # Setting pb.matrix back-solves the basis through any
                # constraints on comp_target_pb. Captured pb.location below
                # reflects the corrected basis.
                comp_target_pb.matrix = new_matrix
                bpy.context.view_layer.update()
                loc_set.add(comp_target_pb.name)

        # Capture settled values into per-fcurve accumulator. Bulk-written
        # once after the frame loop. We escape any quotes in bone names
        # because the data_path embeds the name directly between brackets.
        # Skipped on the warm-up pass (frame_start, capture_keys=False)
        # — that pass exists only to settle the in-memory pose so the
        # following capturing pass reads parent matrices correctly.
        if _capture_keys:
            for tgt_name in sorted_targets:
                meta = target_meta[tgt_name]
                pb = meta['t_pb']
                bone_path = meta['bone_path']
                if tgt_name in loc_set:
                    primed_channels.add((tgt_name, 'location'))
                    loc = pb.location
                    key_accum.setdefault((bone_path + '.location', 0), []).extend((frame, loc.x))
                    key_accum.setdefault((bone_path + '.location', 1), []).extend((frame, loc.y))
                    key_accum.setdefault((bone_path + '.location', 2), []).extend((frame, loc.z))
                if tgt_name in rot_set:
                    mode = pb.rotation_mode
                    if mode == 'QUATERNION':
                        primed_channels.add((tgt_name, 'rotation_quaternion'))
                        q = pb.rotation_quaternion
                        key_accum.setdefault((bone_path + '.rotation_quaternion', 0), []).extend((frame, q.w))
                        key_accum.setdefault((bone_path + '.rotation_quaternion', 1), []).extend((frame, q.x))
                        key_accum.setdefault((bone_path + '.rotation_quaternion', 2), []).extend((frame, q.y))
                        key_accum.setdefault((bone_path + '.rotation_quaternion', 3), []).extend((frame, q.z))
                    elif mode == 'AXIS_ANGLE':
                        primed_channels.add((tgt_name, 'rotation_axis_angle'))
                        aa = pb.rotation_axis_angle
                        for i in range(4):
                            key_accum.setdefault((bone_path + '.rotation_axis_angle', i), []).extend((frame, aa[i]))
                    else:
                        primed_channels.add((tgt_name, 'rotation_euler'))
                        e = pb.rotation_euler
                        key_accum.setdefault((bone_path + '.rotation_euler', 0), []).extend((frame, e.x))
                        key_accum.setdefault((bone_path + '.rotation_euler', 1), []).extend((frame, e.y))
                        key_accum.setdefault((bone_path + '.rotation_euler', 2), []).extend((frame, e.z))

            yield frame

    # All frames done — flush the accumulator. The flush itself yields
    # None periodically (see _bulk_write_keys) so the modal timer keeps
    # ticking and Windows doesn't mark the window as "Not Responding"
    # while a fresh action allocates ~700 fcurves on a face rig.
    #
    # ESC mid-bake during the per-frame loop above closes the generator
    # before reaching this point — no keys land, action stays untouched.
    # ESC during the flush itself can leave a partial set of fcurves
    # filled (whatever batches landed before the cancel); the user can
    # re-bake to overwrite. Soft regression vs. v36's "ESC leaves NO
    # keys" invariant — accepted for the responsiveness win on face rigs.
    yield from _bulk_write_keys(tgt_rig, key_accum, primed_channels, frame_start)


def _frame_range_for(source_rig, scene):
    if source_rig.animation_data and source_rig.animation_data.action:
        fr = source_rig.animation_data.action.frame_range
        return int(fr[0]), int(fr[1])
    return scene.frame_start, scene.frame_end
