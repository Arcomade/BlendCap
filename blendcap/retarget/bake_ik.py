"""Per-frame IK bake driven by the target's retargeted FK chain.

Most pair shapes read from the TARGET rig only (FK retarget has already
populated target's FK chain, so sampling those bones gives correct IK
target poses at target's exact proportions / placement). The POLE pair
has a `pole_mode` field that selects between FORMULA (compute pole
perpendicular to chain) and SAMPLE (sample a source rig bone directly,
for users with custom source rigs that have real pole bones).

Includes Child Of constraint compensation (ARP rigs put Child Of on
every IK control + pole that displaces the bone after our basis write —
we pre-multiply desired_arm by the inverse of the constraint effect)
and the canonical pole-position formula (mirrors Rigify's
`compute_elbow_vector`).
"""
import bpy

from mathutils import Matrix, Quaternion, Vector

from .fcurves import _bulk_write_keys, _accumulate_basis_key


# Opt-in dispatcher for the fast FK->IK engine. Flip True after
# validating against classic on the target rig — see bake_ik_fast.py's
# module docstring for known limitations (the algebraic back-solve does
# NOT cover every flag combo; Child Of subtargets are snapshotted as
# static). Falls back to classic automatically when any pair is POLE
# SAMPLE mode (source-rig sampling isn't implemented in fast yet).
USE_FAST_FK_TO_IK_ENGINE = True


def _detect_child_of_constraints(t_pb, tgt_rig):
    """Scan the bone for Child Of constraints we need to compensate
    for. ARP rigs put these on every IK control to make them follow a
    trajectory bone (c_traj) and on poles to make them follow the
    matching IK control. Returns a list of {target_pb, inverse_matrix}
    dicts — only constraints with mute=False, influence>0, and a
    resolvable subtarget. Influence blending for partial-influence
    constraints isn't supported (uncommon in practice; ARP uses 0/1
    via driven properties)."""
    out = []
    pbs = tgt_rig.pose.bones
    for con in t_pb.constraints:
        if con.type != 'CHILD_OF':
            continue
        if con.mute or con.influence <= 0.0:
            continue
        if not con.subtarget or con.subtarget not in pbs:
            continue
        out.append({
            'target_pb': pbs[con.subtarget],
            'inverse_matrix': con.inverse_matrix.copy(),
            'name': con.name,
        })
    return out


def _child_of_compensation(child_of_list, tgt_mw, tgt_mw_inv, baked_arm=None):
    """Compose the inverse of Child Of effects so that, when the
    constraints run after our basis write, the bone lands at desired
    pose. Per Blender's evaluateConstraint(CHILDOF) with target_space=
    WORLD, owner_space=POSE:

        bone_post_arm = tgt_mw_inv @ target_world @ inverse_matrix
                        @ tgt_mw @ bone_pre_arm

    To make bone_post_arm = desired_arm:
        bone_pre_arm = X @ desired_arm
        X = tgt_mw_inv @ inverse_matrix^-1 @ target_world^-1 @ tgt_mw

    For multiple constraints, compose by composing their X matrices
    in reverse evaluation order (last constraint to run is the first
    we undo).

    `baked_arm` (optional): {bone_name -> playback arm-local matrix}
    for pairs we've already processed THIS FRAME. When a Child Of
    subtarget is the name of one of those, we use the baked playback
    pose instead of pb.matrix (which still shows rest). Critical for
    ARP's c_leg_pole.l → Child Of from c_foot_ik.l setup, where the
    foot IK will be displaced to the FK foot position at playback and
    the pole's compensation must account for it.
    """
    X = Matrix.Identity(4)
    for cof in child_of_list:
        sub_name = cof['target_pb'].name
        if baked_arm is not None and sub_name in baked_arm:
            target_world = tgt_mw @ baked_arm[sub_name]
        else:
            target_world = tgt_mw @ cof['target_pb'].matrix
        inverse = cof['inverse_matrix']
        Xi = (tgt_mw_inv
              @ inverse.inverted()
              @ target_world.inverted()
              @ tgt_mw)
        X = Xi @ X
    return X


def _bake_world_pairs_iter(pairs, src_rig, tgt_rig, frame_start, frame_end):
    """Per-frame IK bake driven by the target's retargeted FK chain.

    Most pair shapes read from the TARGET rig only (FK retarget has
    already populated target's FK chain, so sampling those bones gives
    correct IK target poses at target's exact proportions / placement).
    The POLE pair has a `pole_mode` field that selects between two
    behaviors:

      pole_mode='FORMULA' (default): canonical perpendicular-to-chain
        pole position computed from target FK chain root + mid bones.
        Used when the source rig is BlendCap BVH (no real pole bones).

      pole_mode='SAMPLE': sample a SOURCE rig bone's arm-local position
        directly — for users with custom source rigs that already have
        real pole indicator bones. Assumes source and target share
        canonical orientation; cross-rig world-position differences are
        absorbed because we transfer arm-local coords (NOT world).

    BONE pair (IK end-effector) always reads from target FK chain.

    Iterative basis solve (3 iters) per pair per frame because
    positional pole targets / IK controls can parent through parent-
    switch widgets that depend on the IK chain's own solution.

    Generator: yields once per frame, then drains _bulk_write_keys.
    """
    # Fast-engine dispatcher. The fast path bypasses scene.frame_set per
    # frame by reading target FK fcurves directly — same trick as
    # bake_fk_fast. POLE SAMPLE pairs need source-rig sampling (not yet
    # ported) so fall back to classic if ANY pair is SAMPLE.
    if USE_FAST_FK_TO_IK_ENGINE:
        has_sample = any(
            getattr(p, 'kind', 'BONE') == 'POLE'
            and getattr(p, 'pole_mode', 'FORMULA') == 'SAMPLE'
            for p in pairs
        )
        if not has_sample:
            from .bake_ik_fast import _bake_world_pairs_iter_fast
            yield from _bake_world_pairs_iter_fast(
                pairs, src_rig, tgt_rig, frame_start, frame_end)
            return
        print("[BlendCap] FK->IK fast engine: SAMPLE-mode pole detected "
              "— falling back to classic engine for the whole bake")

    scene = bpy.context.scene
    tgt_mw = tgt_rig.matrix_world
    tgt_mw_inv = tgt_mw.inverted()
    have_src = (src_rig is not None and src_rig.type == 'ARMATURE')
    # World-space LOC toggle. Mirrors the FK engine's flag. Only affects
    # POLE pole_mode='SAMPLE' pairs (the only IK code path that samples
    # the source rig). With this ON we convert source bone position via
    # world before handing it to the target as arm-local — needed for
    # Mixamo and other sources with a non-identity object transform.
    use_world_loc = bool(getattr(scene, 'blendcap_retarget_use_world_location', False))
    src_mw = src_rig.matrix_world if have_src else None

    rest_data = []
    for p in pairs:
        kind = getattr(p, 'kind', 'BONE')
        if not p.target or p.target not in tgt_rig.pose.bones:
            continue
        t_pb = tgt_rig.pose.bones[p.target]
        T_rest_arm_q = t_pb.bone.matrix_local.to_quaternion()
        # Detect Child Of constraints on the target bone — ARP rigs
        # put these on every IK control + pole and they displace the
        # bone post-basis-write. We pre-multiply our desired_arm by
        # the inverse of the constraint's effect so the final pose
        # lands at the right place.
        child_of = _detect_child_of_constraints(t_pb, tgt_rig)

        if kind == 'BONE':
            chain_end_name = getattr(p, 'target_chain_end', None)
            if not chain_end_name or chain_end_name not in tgt_rig.pose.bones:
                continue
            chain_end_pb = tgt_rig.pose.bones[chain_end_name]
            rest_data.append({
                'pair': p,
                'kind': 'BONE',
                't_pb': t_pb,
                'chain_end_pb': chain_end_pb,
                # Pre-compute (chain_end_rest_arm)^-1 @ ik_rest_arm so
                # the per-frame BONE branch is a single matrix multiply.
                'chain_end_rest_arm_inv': chain_end_pb.bone.matrix_local.inverted(),
                'ik_rest_arm': t_pb.bone.matrix_local.copy(),
                'T_rest_arm_q': T_rest_arm_q,
                'child_of': child_of,
            })
        elif kind == 'POLE':
            pole_mode = getattr(p, 'pole_mode', 'FORMULA')
            if pole_mode == 'SAMPLE':
                src_name = getattr(p, 'source_pb_name', '')
                if not have_src or not src_name or src_name not in src_rig.pose.bones:
                    continue
                rest_data.append({
                    'pair': p,
                    'kind': 'POLE',
                    'pole_mode': 'SAMPLE',
                    't_pb': t_pb,
                    'src_pb': src_rig.pose.bones[src_name],
                    'T_rest_arm_q': T_rest_arm_q,
                    'child_of': child_of,
                })
            else:  # FORMULA
                mid_name = getattr(p, 'target_chain_mid', None)
                root_name = getattr(p, 'target_chain_root', None)
                if not mid_name or not root_name:
                    continue
                if mid_name not in tgt_rig.pose.bones or root_name not in tgt_rig.pose.bones:
                    continue
                rest_data.append({
                    'pair': p,
                    'kind': 'POLE',
                    'pole_mode': 'FORMULA',
                    't_pb': t_pb,
                    'mid_pb': tgt_rig.pose.bones[mid_name],
                    'root_pb': tgt_rig.pose.bones[root_name],
                    'T_rest_arm_q': T_rest_arm_q,
                    'child_of': child_of,
                    # Bend axis (upper × lower) expressed in mid_pb's
                    # LOCAL space — invariant to limb rotation, so a
                    # single anchor seeded from the most-bent frame
                    # works across the entire animation regardless of
                    # how the upper bone twists / sweeps. Populated
                    # by the anchor pre-scan below.
                    'bend_axis_local': None,
                })

    if not rest_data:
        return

    # Sort by target depth so parents are written before children — the
    # iterative pb.matrix solve below depends on parent_pose being current
    # when we write a child's basis.
    def _depth(bone):
        d, b = 0, bone.parent
        while b:
            d += 1
            b = b.parent
        return d
    rest_data.sort(key=lambda rd: _depth(tgt_rig.data.bones[rd['pair'].target]))

    # Anchor each POLE FORMULA pair's bend direction before the main
    # loop. On nearly-straight chains (walking arms ~5-15° bend), the
    # per-frame perp direction is the difference of two near-parallel
    # vectors and amplifies FK noise; without an anchor the first
    # frame's noisy direction locks the pole on the wrong side for
    # the whole bake. Dynamic-martial-arts animations historically
    # worked because deep bends give perp a clear, large magnitude
    # every frame — but they also rotate the limb through huge
    # armature-space arcs, which is why we store the anchor in
    # mid_pb's LOCAL space (invariant to limb rotation) rather than
    # armature space. See _scan_anchor_bend_axis_local_classic.
    _scan_total = frame_end - frame_start + 1
    _scan_step = max(1, _scan_total // 60)
    for rd in rest_data:
        if rd['kind'] != 'POLE' or rd['pole_mode'] != 'FORMULA':
            continue
        anchor = _scan_anchor_bend_axis_local_classic(
            rd, scene, frame_start, frame_end, _scan_step)
        source = 'scan'
        if anchor is None:
            anchor = _rest_pose_bend_axis_local(rd['root_pb'], rd['mid_pb'])
            source = 'rest-pose'
        if anchor is None:
            source = 'none'
        rd['bend_axis_local'] = anchor
        print(f"[BlendCap] Pole anchor for {rd['pair'].target!r}: "
              f"source={source} bend_axis_local={anchor}")

    last_q = {}
    key_accum = {}
    primed_channels = set()

    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        # `baked_arm` tracks the playback arm-local pose of each pair's
        # target. Subsequent pairs whose Child Of subtargets are
        # previously-processed targets (e.g. ARP's c_leg_pole.l Child Of
        # from c_foot_ik.l) read from baked_arm INSTEAD of pbs[X].matrix
        # — because pbs[X].matrix still reflects the rest pose; the IK
        # control's eventual baked position isn't in the depsgraph yet.
        # Pre-fix, the pole's compensation pulled the foot's rest pose,
        # so at playback the Child Of constraint displaced the pole by
        # the foot's full IK travel (knees-bend-backwards on ARP). An
        # older diagnostic block at frame_start happened to leave the
        # right basis on pb.loc/rot/scale, which gave a close-enough
        # cached pose that masked the bug; removing the diagnostic
        # exposed it.
        baked_arm = {}

        for rd in rest_data:
            p = rd['pair']
            t_pb = rd['t_pb']
            influence = float(getattr(p, 'influence', 1.0))

            if rd['kind'] == 'BONE':
                # IK control receives the FK chain end's delta-from-rest,
                # applied to the IK control's own rest. For bones whose
                # FK/IK rest orientations match (Rigify hand), this is
                # equivalent to copying chain_end's matrix. For bones
                # whose rest orientations DIFFER (Rigify foot — foot_fk
                # points along the foot, foot_ik is oriented for the
                # heel-pivot mechanism), this preserves IK's rest
                # orientation while transferring FK's pose change, so
                # the IK mechanism doesn't read a 180° flip from rest.
                chain_end_pb = rd['chain_end_pb']
                desired_arm = (chain_end_pb.matrix
                               @ rd['chain_end_rest_arm_inv']
                               @ rd['ik_rest_arm'])
                do_loc = True
                do_rot = True
            elif rd['pole_mode'] == 'SAMPLE':
                # Direct sample: source rig has a real pole indicator
                # bone (e.g. a custom-rig user with permanent pole
                # bones). Take its arm-local head position and use it
                # as the target pole's arm-local position. Assumes
                # both rigs share canonical orientation; cross-rig
                # world-position differences are absorbed because we
                # transfer arm-local coords (NOT world).
                src_pb = rd['src_pb']
                if use_world_loc and src_mw is not None:
                    # World-space sample, then convert to target arm-
                    # local. Handles non-identity source object scale /
                    # rotation (Mixamo) — the default branch's comment
                    # "assumes both rigs share canonical orientation"
                    # doesn't hold there.
                    src_world_head = src_mw @ src_pb.matrix.translation
                    pole_arm_local = tgt_mw_inv @ src_world_head
                else:
                    pole_arm_local = src_pb.matrix.translation.copy()
                desired_arm = Matrix.LocRotScale(
                    pole_arm_local, rd['T_rest_arm_q'], (1.0, 1.0, 1.0))
                do_loc = True
                do_rot = False
            else:
                # Pole at canonical position computed from target chain
                # bones in armature-local space. Rotation stays at the
                # bone's rest (pole solver only reads world position).
                # `prev_perp` is derived FRESH each frame from this
                # pair's bend-axis anchor (fixed in mid_pb's local
                # space) rotated into armature space by the current
                # mid_pb pose. That keeps the sign-flip guard inside
                # _compute_pole_position aligned with the rig's
                # natural bend direction even when the limb sweeps
                # through huge armature-space rotations (kick poses).
                # We don't propagate current_perp back into rd — the
                # local-space anchor is the single source of truth.
                mid_pb = rd['mid_pb']
                root_pb = rd['root_pb']
                root_head = root_pb.head.copy()
                mid_head = mid_pb.head.copy()
                mid_tail = mid_pb.tail.copy()
                perp_target = _bend_axis_local_to_perp_target(
                    mid_pb.matrix, root_head, mid_tail,
                    rd.get('bend_axis_local'))
                pole_arm_local, _current_perp = _compute_pole_position(
                    root_head, mid_head, mid_tail,
                    fallback_axis=Vector((0.0, 1.0, 0.0)),
                    prev_perp=perp_target,
                )
                desired_arm = Matrix.LocRotScale(
                    pole_arm_local, rd['T_rest_arm_q'], (1.0, 1.0, 1.0))
                do_loc = True
                do_rot = False

            if influence < 1.0:
                cur = t_pb.matrix.copy()
                desired_arm = cur.lerp(desired_arm, influence)

            # Snapshot the pre-compensation desired_arm. This is the
            # arm-local pose the IK control / pole will land at after
            # the bake — exactly what Child Of compensation is designed
            # to produce post-constraint. Subsequent pairs in the same
            # frame whose Child Of subtargets are THIS bone need this
            # value (see `baked_arm` comment above).
            playback_arm = desired_arm.copy()

            # Child Of compensation. ARP rigs have Child Of constraints
            # on every IK control + pole that displace the bone after
            # our basis runs. Pre-multiply desired_arm by the inverse
            # of the constraint stack's effect so the FINAL post-
            # constraint pose lands at desired_arm. For rigs without
            # such constraints this is a no-op (compensation = identity).
            # When the subtarget is another IK control we already baked
            # this frame, we substitute baked_arm[sub] for the
            # depsgraph-cached pb.matrix (which still shows rest).
            if rd['child_of']:
                X = _child_of_compensation(
                    rd['child_of'], tgt_mw, tgt_mw_inv, baked_arm)
                desired_arm = X @ desired_arm

            baked_arm[p.target] = playback_arm

            # Direct basis solve (single pass). Bypasses pb.matrix
            # setter (which can be intercepted by constraints or
            # depsgraph quirks). Compute basis = (parent_pose @
            # rel_rest)^-1 @ desired_arm and record for bulk-write.
            # Subsequent pairs in the frame don't depend on this
            # bone's pose (BONE / POLE pairs sample target FK chain,
            # not IK controls), so we don't have to write basis here
            # — just compute and accumulate.
            #
            # An iterative variant was tried (3 iters of compute /
            # write / view_layer.update) to converge through Rigify's
            # parent-switch widgets that depend on the IK chain
            # solution. It worked on Rigify but broke ARP — caused
            # leg-pole rubberbanding and arm distortion (the iteration
            # oscillated through ARP's deeper constraint stack).
            # Reverted; the Rigify hand-position regression remains
            # open (PENDING).
            parent_pb = t_pb.parent
            # Delegate basis decomposition to Blender's pb.matrix
            # setter. It calls BKE_armature_mat_pose_to_bone, which
            # is the authoritative implementation and accounts for
            # every per-bone flag (use_local_location, use_inherit_
            # rotation, inherit_scale) plus parent's current pose
            # rotation and rel-rest rotation — all of which can
            # combine in non-trivial ways across rig families.
            #
            # Bug history (PENDING A in v52): pre-v51 used this
            # setter path. v51 replaced it with a hand-rolled
            # algebraic solve `basis = (parent_pose @ rel_rest)^-1
            # @ desired_arm`, which was flag-blind — correct only
            # for the default flag combination. Rigify hand_ik
            # (use_local_location=False, non-identity rest rotation)
            # landed ~0.5m off; pole controls (use_local_location=
            # False, parent constraint-driven to follow FK chain)
            # landed off by a parent-pose rotation factor. A
            # per-flag manual fix was attempted but the parent-pose-
            # rotation case has no clean closed form — Blender's
            # setter is the correct path.
            #
            # The bone's pose state (loc/rot/scale) gets mutated by
            # this assignment. That's harmless: subsequent pairs
            # sample the target FK chain (not IK controls), and the
            # bake re-writes pb.matrix at every frame, so per-frame
            # state never compounds. Final keys are written via
            # _bulk_write_keys after the bake loop.
            t_pb.matrix = desired_arm
            basis_t = t_pb.location.copy()
            if t_pb.rotation_mode == 'QUATERNION':
                basis_q = t_pb.rotation_quaternion.copy()
            elif t_pb.rotation_mode == 'AXIS_ANGLE':
                aa = t_pb.rotation_axis_angle
                basis_q = Quaternion((aa[1], aa[2], aa[3]), aa[0])
            else:
                basis_q = t_pb.rotation_euler.to_quaternion()

            bone_key = p.target.replace('"', '\\"')
            bone_path = f'pose.bones["{bone_key}"]'

            _accumulate_basis_key(
                key_accum, primed_channels, bone_path, p.target,
                frame, t_pb.rotation_mode,
                basis_t if do_loc else None,
                basis_q if do_rot else None,
                last_q=last_q if do_rot else None)

        yield frame

    yield from _bulk_write_keys(tgt_rig, key_accum, primed_channels, frame_start)


# How far from the bend joint (mid_head) to place the pole, as a
# fraction of the chain length. Rigify's canonical formula uses 1.0,
# which gives poles at ~85cm from the knee on adult-scale legs —
# visually distracting in the viewport. The IK solver only cares
# about pole DIRECTION (not distance), so a smaller fraction keeps
# the visual closer to the joint while leaving solve behavior
# identical.
_POLE_DISTANCE_FRACTION = 0.4


def _compute_perp_raw(root_head, mid_head, mid_tail):
    """Raw (unnormalized) perpendicular-to-chain vector pointing toward
    the elbow/knee side, in whatever space the inputs live in (we use
    armature-local everywhere).

    Returns (perp_vec, chain_vec). When the chain is degenerate (root
    and end coincide), returns (None, chain_vec). When the chain is
    nearly straight, `perp_vec` will have small magnitude — callers
    decide how to interpret (the pole formula falls back to
    fallback_axis; the anchor pre-scan tracks the max-magnitude frame
    and ignores low-confidence ones).
    """
    chain_vec = mid_tail - root_head
    if chain_vec.length < 1e-6:
        return None, chain_vec
    lo_vec = mid_tail - mid_head
    perp = lo_vec.project(chain_vec) - lo_vec
    return perp, chain_vec


def _rest_pose_bend_axis_local(root_pb, mid_pb):
    """Fallback anchor: bend axis (upper × lower) at the chain's rest
    pose, expressed in mid_pb's LOCAL space. ARP / Rigify rigs put a
    slight bend in the rest pose so the IK solver has a defined bend
    direction — that rest bend tells us which side of the chain plane
    the joint naturally bends toward. Storing it in mid_pb's local
    space (rather than armature space) makes the anchor invariant to
    limb rotation: when the upper bone twists or sweeps during an
    animation, mid_pb's pose carries the local-space anchor along
    with it. Returns None when the rest chain is straight (rig
    authored without bend; rare).
    """
    root_bone = root_pb.bone
    mid_bone = mid_pb.bone
    mid_rest = mid_bone.matrix_local
    root_head = root_bone.head_local.copy()
    mid_head = mid_bone.head_local.copy()
    mid_tail = (mid_rest @ Vector((0.0, mid_bone.length, 0.0)))
    upper = mid_head - root_head
    lower = mid_tail - mid_head
    bend_axis_arm = upper.cross(lower)
    if bend_axis_arm.length < 1e-6:
        return None
    axis_local = mid_rest.to_3x3().inverted() @ bend_axis_arm
    if axis_local.length < 1e-6:
        return None
    return axis_local.normalized()


def _scan_anchor_bend_axis_local_classic(rd, scene, frame_start, frame_end, step):
    """Pre-scan the animation to find the frame with the strongest
    elbow/knee bend, and return its bend-axis direction in mid_pb's
    LOCAL space.

    Bend axis = (upper_segment) × (lower_segment), which for a hinge
    joint is parallel to the rig's hinge axis (typically mid_pb local
    X). Its sign in mid_pb's local frame is CONSISTENT across every
    frame where the joint bends its natural way, regardless of how the
    upper bone is rotated in armature space — that's why we anchor in
    local space instead of armature space.

    First-pass attempt anchored perp directly in armature space; that
    worked for walking-style animations where the limb stayed in
    roughly its rest orientation, but broke on martial-arts animations
    where the leg sweeps through huge rotations. The most-bent frame
    in a kicking animation has armature-space perp pointing in a
    direction nearly orthogonal to where the natural perp points
    during the rest of the take, and the dot-product sign check ends
    up flipping in the wrong direction.

    `step` caps the number of frame_set calls (~60 samples is plenty).
    Returns None if every sampled frame is degenerate.
    """
    best_len = 0.0
    best_axis_local = None
    mid_pb = rd['mid_pb']
    root_pb = rd['root_pb']
    for f in range(frame_start, frame_end + 1, step):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        upper = mid_pb.head - root_pb.head
        lower = mid_pb.tail - mid_pb.head
        bend_axis_arm = upper.cross(lower)
        if bend_axis_arm.length < 1e-6:
            continue
        if bend_axis_arm.length > best_len:
            best_len = bend_axis_arm.length
            axis_local = mid_pb.matrix.to_3x3().inverted() @ bend_axis_arm
            if axis_local.length < 1e-6:
                continue
            best_axis_local = axis_local.normalized()
    return best_axis_local


def _bend_axis_local_to_perp_target(mid_pb_matrix, root_head, mid_tail,
                                    bend_axis_local):
    """Translate a local-space bend-axis anchor into the per-frame
    'expected perp direction' that `_compute_pole_position` consumes
    via its `prev_perp` argument.

    Given bend_axis (perp ⊥ chain ⊥ bend_axis), the canonical pole-side
    perp is parallel to (chain × bend_axis) — derivation:

        perp × chain = bend_axis        (definition)
        => (chain × bend_axis) × chain = bend_axis * |chain|²
        => perp ∥ (chain × bend_axis)   (same sign)

    Per-frame: rotate the anchor's local-space bend axis back into
    armature space via the current mid_pb.matrix, then take chain ×
    bend_axis_arm. Caller passes the result as prev_perp; the existing
    sign-flip guard inside _compute_pole_position aligns the
    freshly-computed perp to it.

    Returns None when the chain is degenerate or chain ∥ bend_axis
    (the cross product collapses) — caller falls back to
    _compute_pole_position's normal behavior.
    """
    if bend_axis_local is None:
        return None
    bend_axis_arm = mid_pb_matrix.to_3x3() @ bend_axis_local
    chain_vec = mid_tail - root_head
    perp_target = chain_vec.cross(bend_axis_arm)
    if perp_target.length < 1e-6:
        return None
    return perp_target.normalized()


def _compute_pole_position(root_head, mid_head, mid_tail, fallback_axis,
                           prev_perp=None):
    """Canonical IK pole position from three armature-local chain points.

    Mirrors Rigify's `compute_elbow_vector` (limb_rigs.py): the pole
    sits in the chain's bend plane, perpendicular to the root→end
    axis. Twist-immune (uses positions, not rotations). Distance from
    the bend joint = chain_length * _POLE_DISTANCE_FRACTION.

        chain_vec = mid_tail - root_head     # shoulder → wrist
        lo_vec    = mid_tail - mid_head      # lower segment vector
        elbow_v   = (lo_vec.project(chain_vec) - lo_vec)
        pole = mid_head + elbow_v.normalized() * chain_length * frac

    `fallback_axis` is used when the chain is fully extended (no
    defined bend plane).

    `prev_perp` (optional) is the previous frame's perpendicular
    direction for this same chain. If provided and the freshly
    computed perp would flip to the opposite side (dot < 0), we
    negate to maintain a consistent side across frames. The caller
    seeds this from a pre-bake scan of the most-bent frame in the
    animation (see `_scan_anchor_perp_classic`) so the FIRST frame
    has an anchor too — without that, a noisy frame_start on a
    nearly-straight chain (walking arms, ~5-15° bend) locks the pole
    on the wrong side for the entire bake.

    Returns (pole_position, current_perp_normalized). The caller
    propagates current_perp_normalized as next frame's prev_perp.
    When the current frame had to fall back to `fallback_axis`
    (chain too straight to determine the bend side from positions
    alone) we return None for current_perp_normalized so the caller
    keeps the stronger anchor from earlier frames instead of letting
    a noisy fallback overwrite it.
    """
    perp, chain_vec = _compute_perp_raw(root_head, mid_head, mid_tail)
    if perp is None:
        # Degenerate: root and end coincide. Pick a fallback offset
        # and don't update the caller's anchor.
        return (mid_head + fallback_axis * 0.1, None)
    used_fallback = False
    if perp.length < 1e-4:
        # Straight chain — build a perpendicular to chain_vec from
        # fallback_axis (Gram-Schmidt).
        f = fallback_axis.copy()
        f -= f.project(chain_vec)
        if f.length < 1e-6:
            f = Vector((0.0, 0.0, 1.0))
            f -= f.project(chain_vec)
            if f.length < 1e-6:
                return (mid_head + fallback_axis * chain_vec.length, None)
        perp = f
        used_fallback = True
    perp_norm = perp.normalized()
    if prev_perp is not None and perp_norm.dot(prev_perp) < 0.0:
        # Sign flipped relative to the anchor — keep the anchored side.
        perp_norm = -perp_norm
    pole_pos = mid_head + perp_norm * chain_vec.length * _POLE_DISTANCE_FRACTION
    # Don't propagate a fallback-axis perp as the new anchor: it's not
    # tied to the actual chain bend, just a fixed reference plane, so
    # letting it overwrite prev_perp would erase the high-confidence
    # anchor from the pre-scan.
    return (pole_pos, None if used_fallback else perp_norm)
