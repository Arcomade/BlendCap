"""Fast FK->IK bake — bypasses scene.frame_set by evaluating target FK
chain matrices directly from fcurves.

Mirrors bake_ik._bake_world_pairs_iter (same pair shapes, same Child Of
compensation, same pole formula) but never calls scene.frame_set or
view_layer.update during the per-frame loop. Per-frame cost scales with
~4 ancestors per IK pair (forward kinematic walk), not with the ENTIRE
target rig the depsgraph re-evaluates.

Expected speedup: 30-100x (same ratio as the FK fast engine). FK->IK
only has ~6-8 mapped bones so wall-clock difference is modest in
absolute terms — Tifa's ~18s FK->IK becomes ~0.5-2s.

KNOWN LIMITATIONS (opt in via USE_FAST_FK_TO_IK_ENGINE in bake_ik.py):
  - Algebraic back-solve is FLAG-AWARE for the common case
    use_local_location=False (Rigify hand_ik) but does NOT cover all
    flag combinations. use_inherit_rotation=False, hinge bones, and
    exotic inherit_scale interactions are not handled.
  - Constraints on the TARGET FK chain bones are IGNORED — we read
    raw fcurve values. Standard FK retargets are fine (the FK chain
    receives basis-only keys). Rigs that drive FK chain pose through
    bone constraints will produce wrong samples.
  - Walked bones with NO fcurves at all are SNAPSHOTTED at setup-time
    pose (current scrub frame) rather than computed from basis. This
    is what makes ARP work — master controllers like c_traj /
    c_root_master are constraint-driven, have no fcurves, and would
    otherwise resolve to rest and offset every chain that passes
    through them. Assumption: snapshotted bones are static during
    the bake. If a user animates a master controller, the snapshot
    will freeze it at the setup-time pose.
  - Child Of subtarget matrices are snapshotted at setup (assumed
    static — ARP's c_traj convention). Animated subtargets won't track.
  - POLE pair pole_mode='SAMPLE' falls back to classic for the whole
    bake (we'd need source-rig fcurve reads otherwise — adds surface
    area without freeing up real per-frame cost on the BVH source case
    that's our default).

If you hit any of these on a rig, flip the dispatcher off in
bake_ik.py and the classic engine handles it.
"""
import bpy
from mathutils import Matrix, Quaternion, Vector

from .fcurve_eval import _extract_source_fcurves, _eval_basis
from .constraint_sim import (
    _SUPPORTED_TYPES,
    _scan_supported_constraints,
    _apply_constraint_chain,
    _topo_sort_target_bones,
)
from .bake_ik import (
    _detect_child_of_constraints,
    _compute_pole_position,
    _compute_perp_raw,
    _rest_pose_bend_axis_local,
    _bend_axis_local_to_perp_target,
)
from .fcurves import _bulk_write_keys, _accumulate_basis_key


# Diagnostic print toggle, off in production. Flip True when debugging
# fast-IK bakes (originally used for the ARP "static hands/feet" issue).
_DEBUG_FAST_IK = False

# Pole back-solve diagnostic. Setup audit + per-frame trace + post-bake
# landing validation. Used to diagnose the v75 use_local_location=False
# back-solve bug. Kept gated for the next time pole positions look
# wrong — the post-bake MATCH/OFFSET check is the fastest way to
# distinguish "bake landed where we aimed" vs "intent itself was wrong".
_DEBUG_POLE_COMP = False
# Frames to trace per pole. None -> first 6 frames. Set to a list of
# frame numbers (e.g. [42, 60, 91]) once a bad frame is identified.
_DEBUG_POLE_FRAMES = None

# Filled per-frame by _pole_dbg_trace_frame; drained after the bake
# completes by _pole_dbg_post_bake_validate. Tuples of
# (target_bone_name, frame, desired_arm_local_translation).
_pole_dbg_pending_validation = None


def _dbg(msg):
    if _DEBUG_FAST_IK:
        print(f"[BlendCap fast IK] {msg}")


def _pole_dbg_audit_constraints(rest_data, tgt_rig, constraint_meta=None,
                                snapshot_arm=None):
    """Setup-time: dump every Child Of constraint on every POLE bone
    with both static and driver-evaluated influence values, plus what
    `_detect_child_of_constraints` actually captured for our
    compensation. Also dumps the pole's ctrl_parent bone: its captured
    constraint_meta, every live ARMATURE/SWITCH_PARENT target's static
    vs depsgraph-evaluated weight, and its snapshot pose. Read-only."""
    if not _DEBUG_POLE_COMP:
        return
    pbs = tgt_rig.pose.bones
    # Driver-evaluated influence requires depsgraph-evaluated copies.
    try:
        import bpy
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_rig = tgt_rig.evaluated_get(depsgraph)
        eval_pbs = eval_rig.pose.bones
    except Exception:
        eval_pbs = None
    for rd in rest_data:
        if rd['kind'] != 'POLE':
            continue
        t_name = rd['t_name']
        t_pb = pbs[t_name]
        captured = {c['name'] for c in rd['child_of']}
        all_cofs = [c for c in t_pb.constraints if c.type == 'CHILD_OF']
        print(f"[BlendCap pole-comp] {t_name!r}: "
              f"{len(all_cofs)} Child Of constraint(s), "
              f"{len(rd['child_of'])} captured for compensation, "
              f"pole_parent={t_pb.get('pole_parent', '<unset>')}")
        for con in all_cofs:
            eval_inf = None
            if eval_pbs is not None and t_name in eval_pbs:
                ec = eval_pbs[t_name].constraints.get(con.name)
                if ec is not None:
                    eval_inf = ec.influence
            has_driver = False
            try:
                ad = tgt_rig.animation_data
                if ad is not None:
                    for d in ad.drivers:
                        if con.name in d.data_path and 'influence' in d.data_path:
                            has_driver = True
                            break
            except Exception:
                pass
            inv_t = con.inverse_matrix.to_translation()
            print(f"[BlendCap pole-comp]   - {con.name!r} "
                  f"sub={con.subtarget!r} mute={con.mute} "
                  f"inf_static={con.influence:.3f} "
                  f"inf_driver={eval_inf if eval_inf is None else f'{eval_inf:.3f}'} "
                  f"has_driver={has_driver} "
                  f"inv_t=({inv_t.x:+.4f},{inv_t.y:+.4f},{inv_t.z:+.4f}) "
                  f"captured={'YES' if con.name in captured else 'NO'}")

        # ---- Pole's ctrl_parent (the bone our back-solve compensates
        # against). For stock Rigify this is MCH-<chain>_ik_target.parent.<side>
        # which has no bone parent and carries a single ARMATURE
        # SWITCH_PARENT constraint driven by `pole_parent`.
        cp_name = rd.get('ctrl_parent_name')
        if cp_name is None or cp_name not in pbs:
            print(f"[BlendCap pole-comp]   ctrl_parent=<none>")
            continue
        cp_pb = pbs[cp_name]
        cp_data = cp_pb.bone
        cp_bone_parent = cp_data.parent.name if cp_data.parent else None
        cap = (constraint_meta or {}).get(cp_name)
        cap_summary = '<none>'
        if cap:
            type_counts = {}
            arm_target_count = 0
            for c in cap:
                type_counts[c['type']] = type_counts.get(c['type'], 0) + 1
                if c['type'] == 'ARMATURE':
                    arm_target_count += len(c.get('targets', []))
            cap_summary = (f"{len(cap)} constraint(s) "
                           f"types={type_counts} arm_targets={arm_target_count}")
        snap_t = None
        if snapshot_arm is not None and cp_name in snapshot_arm:
            snap_t = snapshot_arm[cp_name].to_translation()
        print(f"[BlendCap pole-comp]   ctrl_parent={cp_name!r} "
              f"bone_parent={cp_bone_parent!r} "
              f"constraint_meta={cap_summary} "
              f"snapshot_t={None if snap_t is None else f'({snap_t.x:+.4f},{snap_t.y:+.4f},{snap_t.z:+.4f})'}")

        # Dump every ARMATURE/SWITCH_PARENT constraint on ctrl_parent
        # with per-target static vs depsgraph-evaluated weights.
        for con in cp_pb.constraints:
            if con.type != 'ARMATURE':
                continue
            ev_con = None
            if eval_pbs is not None and cp_name in eval_pbs:
                ev_con = eval_pbs[cp_name].constraints.get(con.name)
            print(f"[BlendCap pole-comp]   ARMATURE {con.name!r} "
                  f"infl={con.influence:.3f} "
                  f"target_count={len(con.targets)}")
            for i, t in enumerate(con.targets):
                sub = getattr(t, 'subtarget', '') or '<none>'
                w_static = float(getattr(t, 'weight', 0.0))
                w_eval = None
                if ev_con is not None and i < len(ev_con.targets):
                    w_eval = float(ev_con.targets[i].weight)
                # Detect driver on weight specifically.
                has_w_driver = False
                try:
                    ad = tgt_rig.animation_data
                    if ad is not None:
                        target_path = (f"pose.bones[\"{cp_name}\"]"
                                       f".constraints[\"{con.name}\"]"
                                       f".targets[{i}].weight")
                        for d in ad.drivers:
                            if d.data_path == target_path:
                                has_w_driver = True
                                break
                except Exception:
                    pass
                print(f"[BlendCap pole-comp]     t[{i}] sub={sub!r} "
                      f"w_static={w_static:.3f} "
                      f"w_eval={w_eval if w_eval is None else f'{w_eval:.3f}'} "
                      f"driver={has_w_driver}")


def _pole_dbg_trace_frame(rd, frame, pre_comp_arm, post_comp_arm,
                          basis_t, arm_local, baked_arm,
                          child_of_subtarget_arm, tgt_rig,
                          root_head=None, mid_head=None, mid_tail=None,
                          perp_target=None):
    """Per-frame: dump the Child Of compensation arithmetic for one
    POLE pair on one frame, plus chain geometry + anchor-side sanity
    check. Records (t_name, frame, desired arm_local translation) into
    `_pole_dbg_pending_validation` for post-bake landing comparison.
    Read-only."""
    if not _DEBUG_POLE_COMP:
        return
    if _DEBUG_POLE_FRAMES is None:
        if not hasattr(_pole_dbg_trace_frame, '_seen_first'):
            _pole_dbg_trace_frame._seen_first = {}
        seen = _pole_dbg_trace_frame._seen_first.setdefault(rd['t_name'], 0)
        if seen >= 6:
            return
        _pole_dbg_trace_frame._seen_first[rd['t_name']] = seen + 1
    elif frame not in _DEBUG_POLE_FRAMES:
        return
    # Record the desired post-comp arm-local translation for post-bake
    # validation, plus this frame's simulated ctrl_parent translation
    # and the active SWITCH_PARENT subtargets (so post-bake can compare
    # sim-vs-live for both the parent and its driving source bones).
    # Module-level container, drained in _pole_dbg_post_bake_validate
    # after the bake's _bulk_write_keys.
    global _pole_dbg_pending_validation
    if _pole_dbg_pending_validation is None:
        _pole_dbg_pending_validation = []
    cp_name = rd.get('ctrl_parent_name')
    sim_parent_t = None
    if cp_name and cp_name in arm_local:
        sim_parent_t = arm_local[cp_name].to_translation().copy()
    # Active SWITCH_PARENT subtargets on the ctrl_parent (those whose
    # live weight > 0). Captured from the live constraint at trace
    # time; if drivers change between trace and post-bake validation
    # the captured list is stale, but for stock Rigify pole_parent is
    # rig-level and stable.
    sw_subs = []
    if cp_name and cp_name in tgt_rig.pose.bones:
        for con in tgt_rig.pose.bones[cp_name].constraints:
            if con.type != 'ARMATURE':
                continue
            for t in con.targets:
                w = float(getattr(t, 'weight', 0.0))
                sub = getattr(t, 'subtarget', '')
                if w > 0.0 and sub:
                    sw_subs.append(sub)
            break  # one ARMATURE per parent in Rigify
        # Also capture sim arm-local of each active sub for comparison.
    sim_sub_t = {s: (arm_local[s].to_translation().copy() if s in arm_local else None)
                 for s in sw_subs}
    _pole_dbg_pending_validation.append({
        't_name': rd['t_name'],
        'frame': frame,
        'baked_intent_t': post_comp_arm.to_translation().copy(),
        'ctrl_parent_name': cp_name,
        'sim_parent_t': sim_parent_t,
        'sw_subs': sw_subs,
        'sim_sub_t': sim_sub_t,
    })
    pre_t = pre_comp_arm.to_translation()
    post_t = post_comp_arm.to_translation()
    delta = post_t - pre_t
    cp_name = rd.get('ctrl_parent_name')
    cp_t_str = '<none>'
    if cp_name and cp_name in arm_local:
        cp_t = arm_local[cp_name].to_translation()
        cp_t_str = f"({cp_t.x:+.4f},{cp_t.y:+.4f},{cp_t.z:+.4f})"
    print(f"[BlendCap pole-comp] f={frame} {rd['t_name']!r} "
          f"pre=({pre_t.x:+.4f},{pre_t.y:+.4f},{pre_t.z:+.4f}) "
          f"post=({post_t.x:+.4f},{post_t.y:+.4f},{post_t.z:+.4f}) "
          f"delta=({delta.x:+.4f},{delta.y:+.4f},{delta.z:+.4f}) "
          f"basis_t=({basis_t.x:+.4f},{basis_t.y:+.4f},{basis_t.z:+.4f}) "
          f"parent_t={cp_t_str}")
    # Chain geometry + anchor-side sanity. perp_target is the v70
    # bend-axis anchor projected into per-frame armature space (None
    # when chain near-straight). Sign of (pre - mid_head).dot(perp_target)
    # tells whether the pole sat on the anchored side.
    if root_head is not None and mid_head is not None and mid_tail is not None:
        chain = mid_tail - root_head
        elbow_to_pole = post_comp_arm.to_translation() - mid_head
        if perp_target is not None:
            side_dot = elbow_to_pole.dot(perp_target)
            side = '+ (anchored)' if side_dot > 0 else '- (FLIPPED)'
            print(f"[BlendCap pole-comp]   chain_len={chain.length:.4f} "
                  f"upper_len={(mid_head - root_head).length:.4f} "
                  f"lower_len={(mid_tail - mid_head).length:.4f} "
                  f"perp_t=({perp_target.x:+.4f},{perp_target.y:+.4f},{perp_target.z:+.4f}) "
                  f"side_dot={side_dot:+.4f} {side}")
        else:
            print(f"[BlendCap pole-comp]   chain_len={chain.length:.4f} "
                  f"upper_len={(mid_head - root_head).length:.4f} "
                  f"lower_len={(mid_tail - mid_head).length:.4f} "
                  f"perp_t=<none — straight chain fallback>")
    t_pb = tgt_rig.pose.bones[rd['t_name']]
    for cof in rd['child_of']:
        sub_name = cof['target_pb'].name
        if sub_name in baked_arm:
            sub_arm = baked_arm[sub_name]
            sub_src = 'baked'
        elif sub_name in arm_local:
            sub_arm = arm_local[sub_name]
            sub_src = 'arm_local'
        elif sub_name in child_of_subtarget_arm:
            sub_arm = child_of_subtarget_arm[sub_name]
            sub_src = 'snapshot'
        else:
            print(f"[BlendCap pole-comp]   cof sub={sub_name!r} NOT FOUND")
            continue
        sub_t = sub_arm.to_translation()
        inv_t = cof['inverse_matrix'].to_translation()
        live_con = t_pb.constraints.get(cof['name'])
        live_inf = live_con.influence if live_con is not None else None
        print(f"[BlendCap pole-comp]   cof={cof['name']!r} sub={sub_name!r} "
              f"src={sub_src} "
              f"sub_t=({sub_t.x:+.4f},{sub_t.y:+.4f},{sub_t.z:+.4f}) "
              f"inv_t=({inv_t.x:+.4f},{inv_t.y:+.4f},{inv_t.z:+.4f}) "
              f"live_inf={live_inf if live_inf is None else f'{live_inf:.3f}'}")


def _pole_dbg_post_bake_validate(tgt_rig):
    """Post-bake: for each (bone, frame) the per-frame trace recorded,
    frame_set to that frame, force a depsgraph update, and compare the
    live pole bone's arm-local translation to the desired_arm we baked
    toward. Also compares sim vs live for the pole's ctrl_parent bone
    and each active SWITCH_PARENT subtarget, so we can pinpoint which
    bone first diverges. Read-only — no key writes."""
    global _pole_dbg_pending_validation
    if not _DEBUG_POLE_COMP:
        _pole_dbg_pending_validation = None
        return
    if not _pole_dbg_pending_validation:
        _pole_dbg_pending_validation = None
        return
    scene = bpy.context.scene
    saved_frame = scene.frame_current
    pbs = tgt_rig.pose.bones
    by_frame = {}
    for rec in _pole_dbg_pending_validation:
        by_frame.setdefault(rec['frame'], []).append(rec)
    print(f"[BlendCap pole-comp] === post-bake landing validation "
          f"({len(by_frame)} frame(s)) ===")

    def _fmt(v):
        if v is None:
            return '<none>'
        return f"({v.x:+.4f},{v.y:+.4f},{v.z:+.4f})"

    for frame in sorted(by_frame.keys()):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        for rec in by_frame[frame]:
            t_name = rec['t_name']
            pre_t = rec['baked_intent_t']
            if t_name not in pbs:
                continue
            live_t = pbs[t_name].matrix.to_translation()
            delta = live_t - pre_t
            verdict = 'MATCH' if delta.length < 1e-3 else f'OFFSET ({delta.length*100:.2f}cm)'
            print(f"[BlendCap pole-comp] f={frame} {t_name!r} "
                  f"baked_intent={_fmt(pre_t)} live={_fmt(live_t)} "
                  f"delta={_fmt(delta)} {verdict}")
            cp_name = rec['ctrl_parent_name']
            if cp_name and cp_name in pbs:
                sim_p = rec['sim_parent_t']
                live_p = pbs[cp_name].matrix.to_translation()
                if sim_p is not None:
                    p_delta = live_p - sim_p
                    p_verdict = 'MATCH' if p_delta.length < 1e-3 else f'DIVERGES ({p_delta.length*100:.2f}cm)'
                else:
                    p_delta = None
                    p_verdict = '<no sim>'
                print(f"[BlendCap pole-comp]   parent={cp_name!r} "
                      f"sim={_fmt(sim_p)} live={_fmt(live_p)} "
                      f"delta={_fmt(p_delta)} {p_verdict}")
            for sub in rec.get('sw_subs', []):
                if sub not in pbs:
                    continue
                sim_s = rec['sim_sub_t'].get(sub)
                live_s = pbs[sub].matrix.to_translation()
                if sim_s is not None:
                    s_delta = live_s - sim_s
                    s_verdict = 'MATCH' if s_delta.length < 1e-3 else f'DIVERGES ({s_delta.length*100:.2f}cm)'
                else:
                    s_delta = None
                    s_verdict = '<no sim>'
                print(f"[BlendCap pole-comp]   sw_sub={sub!r} "
                      f"sim={_fmt(sim_s)} live={_fmt(live_s)} "
                      f"delta={_fmt(s_delta)} {s_verdict}")
    scene.frame_set(saved_frame)
    bpy.context.view_layer.update()
    _pole_dbg_pending_validation = None


def _eval_basis_with_scale(fcurve_info, frame):
    """Like _eval_basis but also returns scale, AND falls back to
    whichever rotation channel actually has fcurves when the pose-
    bone's rotation_mode doesn't match.

    Why the channel-aware fallback exists: ARP's FK retarget writes
    rotation_euler keys onto bones whose pb.rotation_mode is still
    'QUATERNION'. _eval_basis selects by mode and finds no quaternion
    fcurves, returning identity — which makes every FK chain bone
    evaluate to rest, every frame. We patch that by checking what
    channels are actually present and reading those instead.
    """
    # Channel-aware rotation read. Pick whichever channel has fcurves;
    # if multiple do (rare but possible — bones can carry both euler
    # and quaternion fcurves from a mode switch), prefer in this
    # priority order to match Blender's own active-channel preference.
    mode = fcurve_info.get('rotation_mode', 'QUATERNION')
    q_fcs = fcurve_info.get('rotation_quaternion')
    e_fcs = fcurve_info.get('rotation_euler')
    aa_fcs = fcurve_info.get('rotation_axis_angle')

    effective_mode = mode
    if mode == 'QUATERNION' and q_fcs is None:
        if e_fcs is not None:
            # Default Euler order when mode says quaternion but only
            # euler fcurves exist. XYZ is Blender's default; if the
            # original Euler keys were written in a different order
            # the per-frame quaternions will be subtly off, but at
            # least the chain stops being static.
            effective_mode = 'XYZ'
        elif aa_fcs is not None:
            effective_mode = 'AXIS_ANGLE'
    elif mode == 'AXIS_ANGLE' and aa_fcs is None:
        if q_fcs is not None:
            effective_mode = 'QUATERNION'
        elif e_fcs is not None:
            effective_mode = 'XYZ'
    elif mode not in ('QUATERNION', 'AXIS_ANGLE') and e_fcs is None:
        if q_fcs is not None:
            effective_mode = 'QUATERNION'
        elif aa_fcs is not None:
            effective_mode = 'AXIS_ANGLE'

    # Swap rotation_mode in a shallow copy so _eval_basis takes the
    # right branch without us reimplementing it.
    if effective_mode != mode:
        patched_info = dict(fcurve_info)
        patched_info['rotation_mode'] = effective_mode
        loc, q = _eval_basis(patched_info, frame)
    else:
        loc, q = _eval_basis(fcurve_info, frame)
    # Scale fcurves aren't in the bake_fk_fast extraction; do an inline
    # lookup on the same pose-bone path. Cheap because we only call this
    # for the small set of pairs' chain bones — not the whole rig.
    s_fcs = fcurve_info.get('scale')
    if s_fcs is None:
        scale = Vector((1.0, 1.0, 1.0))
    else:
        scale = Vector((
            s_fcs[0].evaluate(frame) if s_fcs[0] is not None else 1.0,
            s_fcs[1].evaluate(frame) if s_fcs[1] is not None else 1.0,
            s_fcs[2].evaluate(frame) if s_fcs[2] is not None else 1.0,
        ))
    return loc, q, scale


def _expand_walk_for_constraints(tgt_rig, walk_names_initial):
    """Add bones referenced by supported constraints on walked bones,
    plus their ancestors. Iterates until stable so multi-hop chains
    (mech_a copies from mech_b copies from control) are fully captured.

    Why: ARP / Rigify insert mech bones in the FK chain (e.g. ARP's
    'forearm_fk.l' between 'c_forearm_fk.l' and 'c_hand_fk.l') whose
    poses are driven by Copy Rotation / Copy Transforms constraints
    from the user-facing control bones. Without including the
    constraint subtargets in the walk + simulating those constraints
    per frame, the mech bones stay frozen at their setup-time pose
    and break the chain end's position propagation.
    """
    pbs = tgt_rig.pose.bones
    walk_set = set(walk_names_initial)

    def _add_ancestry(bone_name):
        pb = pbs.get(bone_name)
        while pb is not None:
            if pb.name in walk_set:
                break
            walk_set.add(pb.name)
            pb = pb.parent

    changed = True
    iterations = 0
    while changed and iterations < 16:
        changed = False
        iterations += 1
        for name in list(walk_set):
            pb = pbs.get(name)
            if pb is None:
                continue
            for con in pb.constraints:
                if con.mute or con.influence <= 0.0:
                    continue
                if con.type not in _SUPPORTED_TYPES:
                    continue
                sub = getattr(con, 'subtarget', None)
                if sub and sub in pbs and sub not in walk_set:
                    _add_ancestry(sub)
                    changed = True
                if con.type == 'ARMATURE':
                    for t in getattr(con, 'targets', []) or []:
                        sub_t = getattr(t, 'subtarget', None)
                        if sub_t and sub_t in pbs and sub_t not in walk_set:
                            _add_ancestry(sub_t)
                            changed = True
    return walk_set


def _collect_child_of_subtargets(rest_data):
    """Return the set of bone names every IK pair's Child Of constraints
    reference. Added to the walk set so their matrices are computed per
    frame instead of snapshotted (ARP's c_traj is animated when the user
    has root motion on c_pos)."""
    out = set()
    for rd in rest_data:
        for cof in rd['child_of']:
            out.add(cof['target_pb'].name)
    return out


def _collect_walk_set(tgt_rig, pairs):
    """Collect every target bone we need a per-frame arm-local matrix
    for. That's: each IK control's PARENT chain (for back-solve), and
    each FK chain bone we sample (chain_end / mid / root) plus all
    their ancestors. Returns a depth-sorted list of bone names so the
    per-frame walk processes parents before children.
    """
    pbs = tgt_rig.pose.bones
    need = set()

    def _add_ancestry(bone_name):
        pb = pbs.get(bone_name)
        while pb is not None:
            if pb.name in need:
                break
            need.add(pb.name)
            pb = pb.parent

    for p in pairs:
        if p.target in pbs:
            # IK control: we don't need ITS arm-local, we need its
            # parent's (for back-solve). Add parent + ancestors.
            ctrl_parent = pbs[p.target].parent
            if ctrl_parent is not None:
                _add_ancestry(ctrl_parent.name)
        kind = getattr(p, 'kind', 'BONE')
        if kind == 'BONE':
            ce = getattr(p, 'target_chain_end', None)
            if ce:
                _add_ancestry(ce)
        elif kind == 'POLE':
            mid = getattr(p, 'target_chain_mid', None)
            root = getattr(p, 'target_chain_root', None)
            if mid:
                _add_ancestry(mid)
            if root:
                _add_ancestry(root)

    def _depth(name):
        d, b = 0, pbs[name].parent
        while b is not None:
            d += 1
            b = b.parent
        return d
    return sorted(need, key=_depth)


class _OverlayLookup:
    """Read-only mapping view for _apply_constraint_chain: resolves a
    bone name against several dicts in priority order (typically baked
    pair poses first — the arm-local each already-processed IK control
    / pole will be at on playback — then per-frame re-evaluated bones,
    then the walk's arm_local). The constraint sim only calls .get()."""
    __slots__ = ('_layers',)

    def __init__(self, *layers):
        self._layers = layers

    def get(self, key, default=None):
        for layer in self._layers:
            v = layer.get(key)
            if v is not None:
                return v
        return default


def _detect_armature_constraints(t_pb, tgt_rig, eval_pb=None):
    """Scan a pair target for ARMATURE constraints we must compensate
    for — the parent-switching pattern placed DIRECTLY on a control
    (CloudRig's P-IK-* / P-POLE-* bones; custom rigs that skip the MCH
    indirection). Mirrors _detect_child_of_constraints' role: at
    playback the live constraint re-applies on top of our baked basis,
    so the back-solve must aim at the pre-constraint pose.

    Influence and per-target weights come from the evaluated copy when
    available (they're driver-driven on parent-switch setups). Returns
    a list in constraint-stack order of
    {influence, targets: [{subtarget, weight, rest_inv}]}.
    """
    out = []
    pbs = tgt_rig.pose.bones
    data_bones = tgt_rig.data.bones
    for con in t_pb.constraints:
        if con.type != 'ARMATURE':
            continue
        e_con = (eval_pb.constraints.get(con.name)
                 if eval_pb is not None else None)
        mute = e_con.mute if e_con is not None else con.mute
        infl = float(e_con.influence if e_con is not None else con.influence)
        if mute or infl <= 0.0:
            continue
        e_targets = None
        if e_con is not None:
            try:
                e_targets = list(e_con.targets)
            except Exception:
                e_targets = None
        targets = []
        for t_idx, t in enumerate(con.targets):
            sub = getattr(t, 'subtarget', None)
            if e_targets is not None and t_idx < len(e_targets):
                weight = float(getattr(e_targets[t_idx], 'weight', 0.0))
            else:
                weight = float(getattr(t, 'weight', 0.0))
            if not sub or weight <= 0.0 or sub not in pbs:
                continue
            # Same-rig targets only (matches the constraint sim's
            # assumption — cross-armature parent switching is out of
            # scope for compensation too).
            if getattr(t, 'target', None) is not tgt_rig:
                continue
            target_bone = data_bones.get(sub)
            if target_bone is None:
                continue
            targets.append({
                'subtarget': sub,
                'weight': weight,
                'rest_inv': target_bone.matrix_local.inverted(),
            })
        if targets:
            out.append({'influence': infl, 'targets': targets,
                        'name': con.name})
    return out


def _armature_delta(ac, *lookups):
    """Blended ARMATURE-constraint deform delta D for one captured
    constraint, from the targets' current arm-local poses:

        D_j = target_pose_arm @ target_rest_arm^-1
        D   = weighted blend of D_j  (normalized weights)

    Blender applies bone_post = D @ bone_pre (armdef is parent-like),
    so compensation feeds bone_pre = D^-1 @ desired into the back-
    solve. `lookups` are tried in order per subtarget (baked_arm first,
    then the walk's arm_local, then the setup snapshot) — the baked-
    first order is what makes a pole whose parent-switch follows the
    just-baked foot IK land correctly (same idea as the Child Of
    baked_arm fix, one constraint type over).

    Weight normalization matches Blender (total-weight divide), which
    coincides with the sim's single-target path for the one-hot 0/1
    weights parent-switch drivers produce. Fractional single-target
    weights diverge from the sim there — acceptable; switches are
    one-hot in practice. Returns None when no target resolves.
    """
    collected = []
    for t in ac['targets']:
        sub_arm = None
        for lk in lookups:
            sub_arm = lk.get(t['subtarget'])
            if sub_arm is not None:
                break
        if sub_arm is None:
            continue
        collected.append((sub_arm @ t['rest_inv'], t['weight']))
    if not collected:
        return None
    if len(collected) == 1:
        return collected[0][0]
    total_w = sum(w for _, w in collected)
    if total_w <= 1e-9:
        return None
    sum_loc = Vector((0.0, 0.0, 0.0))
    sum_scale = Vector((0.0, 0.0, 0.0))
    sum_q = [0.0, 0.0, 0.0, 0.0]
    ref_q = None
    for delta, w in collected:
        nw = w / total_w
        d_loc, d_rot, d_scale = delta.decompose()
        sum_loc += d_loc * nw
        sum_scale += d_scale * nw
        q = d_rot
        if ref_q is None:
            ref_q = q
        elif ref_q.dot(q) < 0.0:
            q = Quaternion((-q.w, -q.x, -q.y, -q.z))
        sum_q[0] += q.w * nw
        sum_q[1] += q.x * nw
        sum_q[2] += q.y * nw
        sum_q[3] += q.z * nw
    avg_q = Quaternion(sum_q)
    if avg_q.magnitude > 1e-9:
        avg_q.normalize()
    else:
        avg_q = Quaternion()
    return Matrix.LocRotScale(sum_loc, avg_q, sum_scale)


def _cons_referenced_names(cons_list):
    """All bone names a scanned constraint list reads from — subtarget,
    custom-space bone, LOCAL/LOCAL target parent, and ARMATURE target
    entries. Used to decide whether a control's ancestor depends on a
    pair target (and therefore needs baked-aware re-evaluation)."""
    names = set()
    for con in cons_list or ():
        sub = con.get('subtarget')
        if sub:
            names.add(sub)
        space_sub = con.get('space_subtarget')
        if space_sub:
            names.add(space_sub)
        target_parent = con.get('target_parent')
        if target_parent:
            names.add(target_parent)
        for t in con.get('targets', ()) or ():
            sub2 = t.get('subtarget')
            if sub2:
                names.add(sub2)
    return names


def _back_solve_basis(desired_arm, parent_arm, ctrl_rel_rest,
                      parent_rest_rot, no_local_location):
    """Algebraic back-solve: given the bone's target arm-local matrix
    and the pieces of its rest hierarchy, recover basis (loc, quat,
    scale).

    Rotation/scale always come from inv(parent_arm @ ctrl_rel_rest) @
    desired_arm.

    Location depends on use_local_location:
      - True (default): basis_loc is in BONE LOCAL frame; full inverse-
        pose-mat extraction works.
      - False (BONE_NO_LOCAL_LOCATION; Rigify hand_ik / pole controls):
        Blender applies basis_loc in (parent_pose.rot @ parent_rest.rot^-1)
        frame, skipping the bone's own rest rotation. Derivation:
            pose.translation = parent_pose @ ctrl_rel_rest.translation
                             + (parent_pose.rot @ parent_rest.rot^-1) @ basis_loc
        Solving for basis_loc:
            basis_loc = (parent_pose.rot @ parent_rest.rot^-1)^-1
                        @ (desired.translation - parent_pose @ ctrl_rel_rest.translation)
                      = parent_rest.rot @ parent_pose.rot^-1
                        @ (desired.translation - parent_pose @ ctrl_rel_rest.translation)
        Prior v51-style `basis_t = desired - pose_mat.translation`
        was correct only when parent_pose.rot == parent_rest.rot
        (parent at rest, or both at identity). When parent has any
        pose rotation, the live pole lands `parent.rot @ basis_t`
        away from intent. See bake_ik.py:412-440 for prior bug.

        For root bones (parent_arm is None), parent_pose.rot ==
        parent_rest.rot == identity, both branches give the same
        answer; we take the simple armature-space delta.

    Returns (loc_vec, quat, scale_vec).
    """
    if parent_arm is None:
        pose_mat = ctrl_rel_rest
    else:
        pose_mat = parent_arm @ ctrl_rel_rest
    inv_pose = pose_mat.inverted()
    basis_full = inv_pose @ desired_arm
    basis_q = basis_full.to_quaternion()
    basis_s = basis_full.to_scale()
    if no_local_location:
        if parent_arm is None:
            basis_t = desired_arm.translation - pose_mat.translation
        else:
            rest_offset_t = ctrl_rel_rest.to_translation()
            bone_rest_head_at_pose = parent_arm @ rest_offset_t
            delta_arm = desired_arm.translation - bone_rest_head_at_pose
            parent_pose_rot = parent_arm.to_3x3()
            if parent_rest_rot is not None:
                frame_rot = parent_pose_rot @ parent_rest_rot.inverted()
            else:
                frame_rot = parent_pose_rot
            basis_t = frame_rot.inverted() @ delta_arm
    else:
        basis_t = basis_full.to_translation()
    return basis_t, basis_q, basis_s


def _settle_driver_state(tgt_rig):
    """Setup stage: settle driver-driven constraint state, then return
    an evaluated pose-bones handle for influence/weight reads (None
    when unavailable)."""
    # Settle driver-driven constraint state before ANY scan below reads
    # influences / armature-target weights / snapshots. The convert
    # operator may have just written FK-switch custom properties
    # (Rigify IK_FK, ARP ik_fk_switch, CloudRig ik_*) and a custom-
    # property write does not reliably tag the depsgraph to re-run the
    # drivers that read it. Same update_tag + frame_set + view_layer
    # .update sequence bake_fk_fast uses (see its cold-rig comment),
    # followed by an evaluated-copy handle for influence/weight reads.
    try:
        tgt_rig.update_tag()
    except Exception:
        pass
    try:
        _scene = bpy.context.scene
        _scene.frame_set(_scene.frame_current)
        bpy.context.view_layer.update()
    except Exception:
        pass
    try:
        _depsgraph = bpy.context.evaluated_depsgraph_get()
        _eval_pbs = tgt_rig.evaluated_get(_depsgraph).pose.bones
    except Exception:
        _eval_pbs = None
    return _eval_pbs


def _build_pair_rest_data(pairs, pbs, tgt_rig, _eval_pbs):
    """Setup stage: build per-pair rest data (returns the rest_data
    list, possibly empty)."""
    # ---- Setup: per-pair rest data + Child Of detection ----
    rest_data = []
    for p in pairs:
        kind = getattr(p, 'kind', 'BONE')
        if not p.target or p.target not in pbs:
            continue
        t_pb = pbs[p.target]
        T_rest_arm_q = t_pb.bone.matrix_local.to_quaternion()
        child_of = _detect_child_of_constraints(t_pb, tgt_rig)
        armature_cons = _detect_armature_constraints(
            t_pb, tgt_rig,
            eval_pb=_eval_pbs.get(p.target) if _eval_pbs is not None else None)

        if kind == 'BONE':
            chain_end_name = getattr(p, 'target_chain_end', None)
            if not chain_end_name or chain_end_name not in pbs:
                continue
            rest_data.append({
                'pair': p,
                'kind': 'BONE',
                't_pb': t_pb,
                't_name': p.target,
                'chain_end_name': chain_end_name,
                'chain_end_rest_arm_inv': pbs[chain_end_name].bone.matrix_local.inverted(),
                'ik_rest_arm': t_pb.bone.matrix_local.copy(),
                'T_rest_arm_q': T_rest_arm_q,
                'child_of': child_of,
                'armature_cons': armature_cons,
            })
        elif kind == 'POLE':
            pole_mode = getattr(p, 'pole_mode', 'FORMULA')
            if pole_mode == 'SAMPLE':
                # Should have been filtered by dispatcher. Defensive skip.
                continue
            mid_name = getattr(p, 'target_chain_mid', None)
            root_name = getattr(p, 'target_chain_root', None)
            if not mid_name or not root_name:
                continue
            if mid_name not in pbs or root_name not in pbs:
                continue
            rest_data.append({
                'pair': p,
                'kind': 'POLE',
                'pole_mode': 'FORMULA',
                't_pb': t_pb,
                't_name': p.target,
                'mid_name': mid_name,
                'root_name': root_name,
                'T_rest_arm_q': T_rest_arm_q,
                'child_of': child_of,
                'armature_cons': armature_cons,
                # See classic engine for rationale on local-space
                # anchoring. Populated by the pre-scan below.
                'bend_axis_local': None,
            })
    return rest_data


def _sort_rest_data(rest_data, pbs):
    """Setup stage: order rest_data in place — see the comment below
    for why BONE-before-POLE is load-bearing."""
    # Sort pairs: all BONE (IK control) pairs before all POLE pairs,
    # depth within each group. NOT cosmetic — per-frame processing
    # populates `baked_arm` in this order, and pole compensation reads
    # the baked pose of its limb's IK control from it (ARP's pole Child
    # Of from the foot IK, CloudRig's pole parent-switch following the
    # foot IK). Poles never feed back into IK controls, so BONE-first
    # is always safe.
    def _depth_pb(name):
        d, b = 0, pbs[name].parent
        while b is not None:
            d += 1
            b = b.parent
        return d
    rest_data.sort(key=lambda rd: (0 if rd['kind'] == 'BONE' else 1,
                                   _depth_pb(rd['t_name'])))


def _snapshot_subtarget_arms(rest_data, pbs):
    """Setup stage: snapshot Child Of / Armature subtarget arm-locals
    (static fallback dict)."""
    # ---- Snapshot Child Of / Armature subtarget arm-locals as a static
    # fallback ----
    # Used only when a subtarget isn't in our walk_set (shouldn't happen
    # with the expansion below but kept as a defensive fallback).
    child_of_subtarget_arm = {}
    for rd in rest_data:
        for cof in rd['child_of']:
            sub_pb = cof['target_pb']
            if sub_pb.name in child_of_subtarget_arm:
                continue
            child_of_subtarget_arm[sub_pb.name] = sub_pb.matrix.copy()
        for ac in rd['armature_cons']:
            for t in ac['targets']:
                sub_pb = pbs.get(t['subtarget'])
                if sub_pb is None or sub_pb.name in child_of_subtarget_arm:
                    continue
                child_of_subtarget_arm[sub_pb.name] = sub_pb.matrix.copy()
    return child_of_subtarget_arm


def _collect_walk_and_constraint_meta(tgt_rig, pairs, rest_data, pbs,
                                      _eval_pbs):
    """Setup stage: build + expand the walk set, scan per-bone
    constraint metadata, topo-sort. Returns (walk_names,
    constraint_meta)."""
    # ---- Collect & cache fcurves for every bone we need arm-local on ----
    walk_names_initial = set(_collect_walk_set(tgt_rig, pairs))
    # Also pull in Child Of subtargets + their ancestry — ARP's c_traj
    # inherits from animated c_pos so it varies per frame; computing it
    # in the walk is what makes Child Of compensation track root motion.
    for sub_name in _collect_child_of_subtargets(rest_data):
        pb_walk = pbs.get(sub_name)
        while pb_walk is not None and pb_walk.name not in walk_names_initial:
            walk_names_initial.add(pb_walk.name)
            pb_walk = pb_walk.parent
    # Same for ARMATURE parent-switch targets on pair controls (CloudRig
    # P-IK-* / P-POLE-* plugged directly as controls): their per-frame
    # deform delta reads the targets' arm-locals, so those must be
    # walked too — the setup snapshot above is only a last resort.
    for rd in rest_data:
        for ac in rd['armature_cons']:
            for t in ac['targets']:
                pb_walk = pbs.get(t['subtarget'])
                while (pb_walk is not None
                       and pb_walk.name not in walk_names_initial):
                    walk_names_initial.add(pb_walk.name)
                    pb_walk = pb_walk.parent

    # Expand walk_set to include constraint subtargets first, BEFORE
    # we topo-sort: a bone's constraint subtarget must be in the set
    # so the sort treats it as a real dependency.
    # Expand to include constraint subtargets so ARP/Rigify mech bones
    # in the FK chain can be driven by their constraints during the
    # per-frame walk.
    walk_set = _expand_walk_for_constraints(tgt_rig, walk_names_initial)

    # Per-bone constraint metadata. Built BEFORE the topo sort so the
    # sort can use constraint dependencies as ordering edges. Without
    # this, sibling bones with constraint dependencies (e.g. ARP's
    # c_arm_fk.l COPY_LOCATION from c_shoulder.l — same parent, same
    # depth) get arbitrary order and the COPY_LOCATION reads None
    # when c_shoulder.l comes later → arm position is wrong.
    constraint_meta = {}
    for name in walk_set:
        pb = pbs[name]
        constraint_meta[name] = _scan_supported_constraints(
            pb, walk_set,
            eval_pb=_eval_pbs.get(name) if _eval_pbs is not None else None)

    # Topo sort: each bone processed after its parent AND after every
    # constraint subtarget it depends on. Mirrors what bake_fk_fast
    # does; reusing the same helper for parity.
    walk_names = _topo_sort_target_bones(walk_set, tgt_rig, constraint_meta)
    return walk_names, constraint_meta


def _extract_fcurves_and_snapshots(tgt_rig, walk_names, pbs,
                                   constraint_meta):
    """Setup stage: cache fcurves for walked bones, snapshot setup-time
    pose, flag which bones have fcurves. Returns (fcurve_info,
    snapshot_arm, has_fcurves)."""
    action = tgt_rig.animation_data and tgt_rig.animation_data.action
    fcurve_info = _extract_source_fcurves(tgt_rig, walk_names, include_scale=True)

    # Snapshot current pose-arm matrix for every walked bone. Used for
    # bones with NO fcurves at all — those are constraint-driven
    # controllers (ARP's c_traj, c_pos_*, c_root_master; Rigify's MCH-*
    # parent-switch widgets) whose pose isn't keyed but isn't at rest
    # either. Reading raw fcurves on them gives rest, which propagates
    # wrong matrices down through every descendant chain and offsets
    # hands / feet / poles by the controller's actual displacement.
    #
    # Assumption: snapshotted bones are STATIC during the bake. True for
    # the master controllers on standard rigs. Animated controllers
    # would need fcurves keyed on them; FK retarget doesn't write to
    # those, so any user-keyed motion on a controller would freeze at
    # the setup-time pose. Edge case — document and move on.
    snapshot_arm = {}
    has_fcurves = {}
    for name in walk_names:
        snapshot_arm[name] = pbs[name].matrix.copy()
        info = fcurve_info.get(name)
        if info is None:
            has_fcurves[name] = False
            continue
        has_fcurves[name] = any(
            info.get(ch) is not None
            for ch in ('location', 'rotation_quaternion',
                       'rotation_euler', 'rotation_axis_angle', 'scale')
        )

    _dbg(f"setup: action={action.name if action else None!r}, "
         f"walk_names={len(walk_names)}, "
         f"with_fcurves={sum(1 for v in has_fcurves.values() if v)}, "
         f"snapshot_only={sum(1 for v in has_fcurves.values() if not v)}")
    # List ALL walk names with their fcurve status, data parent, and
    # constraint stack. Want to see exactly which constraints are on
    # the FK chain mech bones (arms have an extra forearm_fk.l vs legs
    # — the "arms more wrong than legs" symptom points at it).
    for name in walk_names:
        pb = pbs[name]
        info = fcurve_info.get(name)
        channels = []
        if info:
            for ch in ('location', 'rotation_quaternion',
                      'rotation_euler', 'rotation_axis_angle', 'scale'):
                if info.get(ch) is not None:
                    channels.append(ch)
        parent_name = pb.parent.name if pb.parent else None
        con_descs = []
        for con in pb.constraints:
            if con.mute:
                continue
            sub = getattr(con, 'subtarget', None) or ''
            ts = getattr(con, 'target_space', '')
            os_ = getattr(con, 'owner_space', '')
            supported = 'OK' if con.type in _SUPPORTED_TYPES else 'SKIP'
            con_descs.append(
                f"{con.type}[{supported}](sub={sub!r},"
                f"ts={ts},os={os_},infl={con.influence:.2f})")
        scanned = constraint_meta.get(name, [])
        _dbg(f"  walk {name!r} parent={parent_name!r} "
             f"has_fcurves={has_fcurves[name]} channels={channels} "
             f"scanned_constraints={len(scanned)} all_constraints={con_descs}")
    return fcurve_info, snapshot_arm, has_fcurves


def _build_rest_tables(walk_names, pbs):
    """Setup stage: per-bone rest lookups for the walk. Returns
    (rest_arm, rel_rest, parent_name_of)."""
    # Pre-compute per-bone rest data needed by the walk:
    #   rest_arm:  bone.matrix_local
    #   rel_rest:  parent.matrix_local^-1 @ bone.matrix_local  (or rest_arm if root)
    #   parent_name: parent bone name or None (root)
    rest_arm = {}
    rel_rest = {}
    parent_name_of = {}
    for name in walk_names:
        pb = pbs[name]
        b = pb.bone
        rest_arm[name] = b.matrix_local.copy()
        parent_pb = pb.parent
        if parent_pb is None:
            parent_name_of[name] = None
            rel_rest[name] = b.matrix_local.copy()
        else:
            parent_name_of[name] = parent_pb.name
            rel_rest[name] = (parent_pb.bone.matrix_local.inverted()
                              @ b.matrix_local)
    return rest_arm, rel_rest, parent_name_of


def _attach_ctrl_parent_data(rest_data):
    """Setup stage: attach control-parent rest data to each pair
    (mutates rest_data entries in place)."""
    # Per-pair: parent of the IK control + rel_rest (for back-solve),
    # plus the bone's use_local_location flag. Also store the parent's
    # rest rotation; the use_local_location=False back-solve needs it
    # to express basis_loc in (parent_pose @ parent_rest^-1) frame
    # (see _back_solve_basis).
    for rd in rest_data:
        t_pb = rd['t_pb']
        parent_pb = t_pb.parent
        rd['ctrl_parent_name'] = parent_pb.name if parent_pb else None
        if parent_pb is None:
            rd['ctrl_rel_rest'] = t_pb.bone.matrix_local.copy()
            rd['ctrl_parent_rest_rot'] = None
        else:
            rd['ctrl_rel_rest'] = (parent_pb.bone.matrix_local.inverted()
                                   @ t_pb.bone.matrix_local)
            rd['ctrl_parent_rest_rot'] = parent_pb.bone.matrix_local.to_3x3().copy()
        rd['no_local_location'] = bool(getattr(t_pb.bone,
                                               'use_local_location', True) is False)


def _compute_reval_names(rest_data, walk_names, parent_name_of,
                         constraint_meta):
    """Setup stage: tainted-set forward pass. Sets each pair's
    rd['parent_needs_reval'] and returns the topo-ordered list of
    bones to re-evaluate per frame."""
    # ---- Walk bones that depend on pair targets need baked-aware
    # re-evaluation ----
    # The walk computes every bone's arm-local from fcurves + constraint
    # sim — correct for FK-chain ancestry, but STALE for any bone whose
    # pose depends on a bone we're baking in this same pass: pair
    # targets have no fcurves yet, so the walk sees them at rest while
    # playback sees them at the baked pose. Canonical case: CloudRig's
    # POLE-Leg.L is parented to P-POLE-Leg.L, whose ARMATURE parent-
    # switch follows IK-Foot.L when ik_pole_follow=1 — the back-solve
    # compensated against a rest-pose parent and the pole landed offset
    # by the foot's whole baked displacement ("poles fly off to one
    # side"). Compute the TAINTED set: a bone is tainted if it IS a
    # pair target, its data parent is tainted, or any constraint it
    # carries references a tainted bone. One forward pass suffices —
    # walk_names is topo-sorted with constraint deps as edges. Pairs
    # whose control parent is tainted re-evaluate the tainted bones per
    # frame with baked poses overlaid (see _reval_parent_chain). Rigify
    # / ARP parent-switch targets are ORG-/c_traj-style bones (never
    # pair targets), so nothing taints for them — zero behavior change.
    pair_target_names = {rd['t_name'] for rd in rest_data}
    _tainted = set()
    for name in walk_names:
        if name in pair_target_names:
            _tainted.add(name)
            continue
        parent = parent_name_of.get(name)
        if parent is not None and parent in _tainted:
            _tainted.add(name)
            continue
        if _cons_referenced_names(constraint_meta.get(name)) & _tainted:
            _tainted.add(name)
    # Topo-ordered re-eval list: tainted bones that are NOT pair targets
    # (pair targets resolve from baked_arm — recomputing them from walk
    # rules would just give rest back).
    _reval_names = [n for n in walk_names
                    if n in _tainted and n not in pair_target_names]
    for rd in rest_data:
        ctrl_parent = rd['ctrl_parent_name']
        needs = bool(ctrl_parent is not None and ctrl_parent in _tainted)
        rd['parent_needs_reval'] = needs
        if needs:
            print(f"[BlendCap] FK->IK: {rd['t_name']!r} parent "
                  f"{ctrl_parent!r} depends on baked pair target(s) — "
                  f"re-evaluating per frame with baked poses "
                  f"(chain: {_reval_names})")
    return _reval_names


def _anchor_pole_bend_axes(rest_data, frame_start, frame_end,
                           walk_names, parent_name_of, constraint_meta,
                           has_fcurves, fcurve_info, rel_rest, rest_arm,
                           snapshot_arm, tgt_mw, tgt_mw_inv, pbs):
    """Setup stage: pre-scan the animation and anchor each FORMULA POLE
    pair's bend axis (writes rd['bend_axis_local'] in place).

    The coarse-scan walk inside is a deliberate near-duplicate of
    _walk_arm_locals_for_frame (identical per-bone rules, coarser
    frame sampling) — see the docstring there before unifying."""
    # ---- Anchor prev_perp for POLE FORMULA pairs ----
    # See bake_ik._compute_pole_position docstring. Pre-scans the
    # animation to find the most-bent frame for each pair and seeds
    # prev_perp from that frame's perp direction. Walking-style arms
    # (~5-15° bend) have very small perp magnitudes and amplify FK
    # noise into pole-side flips; without an anchor the first frame's
    # noisy direction locks the wrong side in for the whole bake.
    # Reuses the per-frame walk's logic with the same `walk_names`
    # and constraint stack so the anchor sees exactly what the main
    # loop will see — just sampled at a coarser rate (~60 samples).
    _pole_formula_rd = [rd for rd in rest_data
                        if rd['kind'] == 'POLE'
                        and rd['pole_mode'] == 'FORMULA']
    if _pole_formula_rd:
        _scan_total = frame_end - frame_start + 1
        _scan_step = max(1, _scan_total // 60)
        _best = {id(rd): (0.0, None) for rd in _pole_formula_rd}
        for _scan_f in range(frame_start, frame_end + 1, _scan_step):
            _scan_arm = {}
            for name in walk_names:
                parent = parent_name_of[name]
                cons = constraint_meta.get(name)
                if has_fcurves.get(name, False):
                    info = fcurve_info[name]
                    loc, q, scale = _eval_basis_with_scale(info, _scan_f)
                    basis = Matrix.LocRotScale(loc, q, scale)
                    if parent is None:
                        pre_constraint = rel_rest[name] @ basis
                    else:
                        pre_constraint = _scan_arm[parent] @ rel_rest[name] @ basis
                elif parent is None:
                    if cons:
                        pre_constraint = rest_arm[name]
                    else:
                        pre_constraint = snapshot_arm[name]
                else:
                    pre_constraint = _scan_arm[parent] @ rel_rest[name]
                if cons:
                    parent_arm_v = _scan_arm.get(parent) if parent else None
                    _scan_arm[name] = _apply_constraint_chain(
                        pre_constraint, cons, tgt_mw, tgt_mw_inv,
                        _scan_arm, parent_arm=parent_arm_v,
                        rel_rest=rel_rest.get(name))
                else:
                    _scan_arm[name] = pre_constraint
            for rd in _pole_formula_rd:
                mid_arm = _scan_arm[rd['mid_name']]
                root_arm = _scan_arm[rd['root_name']]
                mid_pb = pbs[rd['mid_name']]
                mid_head = mid_arm.translation.copy()
                mid_tail = (mid_arm @ Vector((0.0, mid_pb.bone.length, 0.0)))
                root_head = root_arm.translation.copy()
                upper = mid_head - root_head
                lower = mid_tail - mid_head
                bend_axis_arm = upper.cross(lower)
                if bend_axis_arm.length < 1e-6:
                    continue
                if bend_axis_arm.length > _best[id(rd)][0]:
                    axis_local = mid_arm.to_3x3().inverted() @ bend_axis_arm
                    if axis_local.length < 1e-6:
                        continue
                    _best[id(rd)] = (bend_axis_arm.length,
                                     axis_local.normalized())
        for rd in _pole_formula_rd:
            anchor = _best[id(rd)][1]
            source = 'scan'
            if anchor is None:
                anchor = _rest_pose_bend_axis_local(
                    pbs[rd['root_name']], pbs[rd['mid_name']])
                source = 'rest-pose'
            if anchor is None:
                source = 'none'
            rd['bend_axis_local'] = anchor
            print(f"[BlendCap] Pole anchor for {rd['t_name']!r}: "
                  f"source={source} bend_axis_local={anchor}")


def _walk_arm_locals_for_frame(frame, walk_names, parent_name_of,
                               constraint_meta, has_fcurves, fcurve_info,
                               rel_rest, rest_arm, snapshot_arm, tgt_mw,
                               tgt_mw_inv):
    """Per-frame Step 1: compute the arm-local matrix for every walked
    bone at `frame`. Returns this frame's arm_local dict.

    NOTE: the per-bone rules below exist in three near-copies — this
    walk, the coarse pre-scan in _anchor_pole_bend_axes (identical
    rules), and _reval_parent_chain inside _bake_world_pairs_iter_fast
    (NOT identical: it resolves parents and the constraint-sim lookup
    through a baked_arm-first _OverlayLookup, and roots on the resolved
    parent matrix being absent rather than the parent name being None).
    Deliberately NOT unified; do not fold them together."""
    # Step 1: compute arm-local matrix for every walked target bone
    # at this frame. Depth order guarantees parents-first.
    #
    # Rules:
    #   - Bones with fcurves: arm_local = parent_arm @ rel_rest @ basis
    #     (basis from fcurves). For roots, parent_arm folds into rel_rest.
    #   - Bones without fcurves but WITH a parent in walk_set:
    #     arm_local = parent_arm @ rel_rest (identity basis). This is
    #     hierarchy inheritance — exactly what Blender does for mech
    #     bones that just inherit pose from their parent control.
    #   - True root bones with no fcurves: snapshot (captures whatever
    #     constraint-driven pose the bone has at setup; only place
    #     where snapshot is correct, since we can't derive from parent).
    #   - After arm_local is set, supported constraints run on top —
    #     same machinery FK retarget uses. This is what makes ARP mech
    #     bones with Copy Rotation/Transforms from controls land
    #     correctly per frame.
    arm_local = {}
    for name in walk_names:
        parent = parent_name_of[name]
        cons = constraint_meta.get(name)
        if has_fcurves.get(name, False):
            info = fcurve_info[name]
            loc, q, scale = _eval_basis_with_scale(info, frame)
            basis = Matrix.LocRotScale(loc, q, scale)
            if parent is None:
                pre_constraint = rel_rest[name] @ basis
            else:
                pre_constraint = arm_local[parent] @ rel_rest[name] @ basis
        elif parent is None:
            if cons:
                # Root bone with supported constraints — start from
                # rest and let the constraint sim produce the pose.
                # Snapshot would already include the constraint
                # effect (Blender's depsgraph evaluated it at setup
                # time), so feeding it in and reapplying the
                # constraints would double-apply them. ARP's
                # c_foot_ik.l / c_hand_ik.l (Child Of from c_traj,
                # parented to null) are the canonical case — the
                # snapshot+reapply path was producing a constant
                # ~26mm offset that propagated into the leg pole's
                # Child Of compensation (pole sits too far from the
                # knee, knee a hair off).
                pre_constraint = rest_arm[name]
            else:
                # No supported constraints — bone is either static
                # at rest or driven by unsupported constraints
                # (drivers, IK, etc.). Snapshot captures whatever
                # Blender computed at setup and treats it as static
                # for the duration of the bake.
                pre_constraint = snapshot_arm[name]
        else:
            pre_constraint = arm_local[parent] @ rel_rest[name]
        if cons:
            parent_arm = arm_local.get(parent) if parent else None
            arm_local[name] = _apply_constraint_chain(
                pre_constraint, cons, tgt_mw, tgt_mw_inv,
                arm_local, parent_arm=parent_arm,
                rel_rest=rel_rest.get(name))
        else:
            arm_local[name] = pre_constraint
    return arm_local


def _bake_pairs_for_frame(frame, rest_data, arm_local,
                          child_of_subtarget_arm, pbs, fcurve_info,
                          has_fcurves, tgt_rig, tgt_mw, tgt_mw_inv,
                          _reval_parent_chain, key_accum, primed_channels,
                          last_q, last_e, frame_start, frame_end):
    """Per-frame Step 2: per-pair desired_arm, Child Of / ARMATURE
    compensation, back-solve, and key accumulation. Mutates key_accum,
    primed_channels, last_q and last_e in place. `_reval_parent_chain`
    is the generator's closure, passed in so its call site stays
    verbatim."""
    # Step 2: for each pair, compute desired_arm and back-solve basis.
    # `baked_arm` tracks each pair target's playback arm-local — i.e.
    # the position the IK control / pole will be at AFTER the bake
    # lands and Blender re-evaluates with the new keys. We seed it
    # with each pair's pre-Child-Of-compensation desired_arm (= the
    # arm-local the back-solve aims for). Subsequent pairs whose
    # Child Of subtargets are previously-processed targets read from
    # baked_arm INSTEAD of the walk's `arm_local` (which still shows
    # the rest pose for IK controls). ARP's c_leg_pole.l has Child
    # Of from c_foot_ik.l; before this fix, the pole's compensation
    # used c_foot_ik.l's REST pose, so at playback (when c_foot_ik.l
    # is at the baked IK position) Child Of shifted the pole by the
    # foot's full IK displacement. Visible as knees-bend-backwards
    # and a pole that sits too far from the knee on ARP. Rigify
    # rigs don't Child-Of the leg pole from the foot IK, so they
    # were unaffected by the bug AND by this fix.
    baked_arm = {}

    for rd in rest_data:
        p = rd['pair']
        influence = float(getattr(p, 'influence', 1.0))

        if rd['kind'] == 'BONE':
            chain_end_arm = arm_local[rd['chain_end_name']]
            desired_arm = (chain_end_arm
                           @ rd['chain_end_rest_arm_inv']
                           @ rd['ik_rest_arm'])
            if _DEBUG_FAST_IK and frame in (frame_start, frame_start + 5,
                                             frame_start + 20,
                                             (frame_start + frame_end) // 2):
                t = chain_end_arm.translation
                # Also report the chain_end's evaluated basis
                # rotation directly, so we can tell whether basis
                # eval itself is constant or whether the chain
                # math is dropping the variation.
                info_ce = fcurve_info.get(rd['chain_end_name'])
                if info_ce is not None:
                    ce_loc, ce_q, ce_s = _eval_basis_with_scale(info_ce, frame)
                else:
                    ce_loc, ce_q, ce_s = (None, None, None)
                # And the chain ROOT's basis (the bone whose
                # rotation actually moves chain_end's position).
                chain_root_name = rd['chain_end_name']
                walk_pb = pbs[chain_root_name]
                while walk_pb.parent is not None and walk_pb.parent.name in has_fcurves and has_fcurves[walk_pb.parent.name]:
                    walk_pb = walk_pb.parent
                info_cr = fcurve_info.get(walk_pb.name)
                if info_cr is not None:
                    cr_loc, cr_q, cr_s = _eval_basis_with_scale(info_cr, frame)
                else:
                    cr_loc, cr_q, cr_s = (None, None, None)
                _dbg(f"frame {frame} pair t={rd['t_name']!r} "
                     f"chain_end={rd['chain_end_name']!r} "
                     f"chain_end_arm.t=({t.x:.4f}, {t.y:.4f}, {t.z:.4f}) "
                     f"chain_end_basis_q={ce_q} "
                     f"chain_root={walk_pb.name!r} chain_root_basis_q={cr_q}")
            do_loc = True
            do_rot = True
        else:
            # POLE FORMULA — compute pole position from chain arm-
            # locals in armature-local space (matches classic, which
            # read mid_pb.head / mid_pb.tail in arm-local).
            #
            # `prev_perp` is derived fresh each frame from this
            # pair's bend-axis anchor (fixed in mid_pb's local
            # space) rotated by the current mid_arm orientation,
            # so the sign-flip guard inside _compute_pole_position
            # tracks the rig's natural bend direction even through
            # large limb rotations. See classic engine for the
            # full rationale.
            mid_arm = arm_local[rd['mid_name']]
            root_arm = arm_local[rd['root_name']]
            # Pose-bone head/tail in arm-local: head = matrix.translation,
            # tail = matrix @ Vector((0, length, 0)).
            mid_pb = pbs[rd['mid_name']]
            mid_head = mid_arm.translation.copy()
            mid_tail = (mid_arm @ Vector((0.0, mid_pb.bone.length, 0.0)))
            root_head = root_arm.translation.copy()
            perp_target = _bend_axis_local_to_perp_target(
                mid_arm, root_head, mid_tail,
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
            # Need current arm-local of the IK control for the lerp.
            # Cached at setup is wrong (current scrub-state, not the
            # IK control's pose this frame). Best approximation: read
            # the bone's current pb.matrix once per frame. That's a
            # tiny cost (we don't drive depsgraph from this read).
            # Practical reality: influences are nearly always 1.0 on
            # FK->IK; users wanting partial-influence should stay on
            # classic.
            cur = rd['t_pb'].matrix.copy()
            desired_arm = cur.lerp(desired_arm, influence)

        # Snapshot the pre-compensation desired_arm. This IS the
        # arm-local pose the IK control / pole will land at after
        # the bake (the whole point of Child Of compensation is
        # making the post-constraint pose equal this). Subsequent
        # pairs in the same frame whose Child Of subtargets are
        # THIS bone should read this value, not the walk's arm_local
        # (which has the rest pose). See `baked_arm` comment above.
        playback_arm = desired_arm.copy()

        if rd['child_of']:
            # Read each Child Of subtarget's arm-local FRESH per
            # frame. Lookup order: baked_arm (other IK controls
            # baked earlier this frame) → arm_local (walk result
            # for chain bones / c_traj-style controllers) →
            # child_of_subtarget_arm (setup-time snapshot, last
            # resort). This is what makes ARP root motion (c_pos
            # animated -> c_traj inherits -> IK control Child Of
            # from c_traj) actually compensate correctly per
            # frame, AND what makes c_leg_pole.l's Child Of from
            # c_foot_ik.l see the BAKED foot position instead of
            # the foot's rest pose.
            X = Matrix.Identity(4)
            for cof in rd['child_of']:
                sub_name = cof['target_pb'].name
                sub_arm = baked_arm.get(sub_name)
                if sub_arm is None:
                    sub_arm = arm_local.get(sub_name)
                if sub_arm is None:
                    sub_arm = child_of_subtarget_arm.get(sub_name)
                if sub_arm is None:
                    continue
                target_world = tgt_mw @ sub_arm
                inverse = cof['inverse_matrix']
                Xi = (tgt_mw_inv
                      @ inverse.inverted()
                      @ target_world.inverted()
                      @ tgt_mw)
                X = Xi @ X
            desired_arm = X @ desired_arm

        if rd['armature_cons']:
            # ARMATURE parent-switch directly on the control
            # (CloudRig P-IK-* / P-POLE-* when plugged as the
            # control). Blender re-applies it at playback as
            # post = D @ pre, so aim the back-solve at
            # pre = D^-1 @ desired. Undo in reverse stack order;
            # baked_arm-first lookup for targets that are pairs we
            # baked earlier this frame. If a control ever mixes
            # Child Of AND Armature in one stack, this treats the
            # Armature as evaluating first (approximate) — not a
            # configuration any supported rig family ships.
            for ac in reversed(rd['armature_cons']):
                D = _armature_delta(ac, baked_arm, arm_local,
                                    child_of_subtarget_arm)
                if D is None:
                    continue
                # NOTE: armature weights are NORMALIZED by Blender
                # (single target = full carry whatever its weight);
                # only influence blends toward the unconstrained
                # pose.
                if ac['influence'] >= 1.0:
                    desired_arm = D.inverted() @ desired_arm
                else:
                    # Approximate partial-influence undo (mirrors
                    # the Child Of partial path; rare in practice).
                    desired_arm = desired_arm.lerp(
                        D.inverted() @ desired_arm, ac['influence'])

        # Record this pair's playback arm for downstream Child Of
        # lookups. Done after Child Of comp so any sibling-comp
        # lookup of THIS bone in the same loop iteration would still
        # see arm_local (correct — at the moment of comp we haven't
        # determined the final pose yet); downstream pairs see the
        # finalized playback arm.
        baked_arm[rd['t_name']] = playback_arm

        # Back-solve basis. _back_solve_basis composes pose_mat from
        # parent_arm + ctrl_rel_rest itself; for use_local_location=False
        # bones it ALSO needs parent_arm and parent_rest_rot separately
        # to produce the correct basis_loc frame (see its docstring).
        ctrl_parent = rd['ctrl_parent_name']
        if ctrl_parent is None:
            parent_arm = None
        else:
            if rd['parent_needs_reval']:
                # Parent depends on pair targets (parent-switch
                # following an IK control we baked this frame) —
                # recompute the tainted bones with baked poses so
                # the back-solve compensates against the PLAYBACK
                # parent pose, not the walk's rest-state one. See
                # the setup comment.
                parent_arm = _reval_parent_chain(
                    ctrl_parent, arm_local, baked_arm, frame)
            else:
                parent_arm = arm_local.get(ctrl_parent)
            if parent_arm is None:
                # Defensive — parent missing from walk (shouldn't
                # happen because _collect_walk_set adds it).
                parent_arm = pbs[ctrl_parent].matrix

        basis_t, basis_q, _basis_s = _back_solve_basis(
            desired_arm, parent_arm, rd['ctrl_rel_rest'],
            rd['ctrl_parent_rest_rot'], rd['no_local_location'])

        if rd['kind'] == 'POLE':
            _pole_dbg_trace_frame(
                rd, frame, playback_arm, desired_arm, basis_t,
                arm_local, baked_arm, child_of_subtarget_arm,
                tgt_rig,
                root_head=root_head, mid_head=mid_head,
                mid_tail=mid_tail, perp_target=perp_target)

        # ---- Accumulate keys (same layout as classic) ----
        t_pb = rd['t_pb']
        target_name = rd['t_name']
        bone_key = target_name.replace('"', '\\"')
        bone_path = f'pose.bones["{bone_key}"]'

        _accumulate_basis_key(
            key_accum, primed_channels, bone_path, target_name,
            frame, t_pb.rotation_mode,
            basis_t if do_loc else None,
            basis_q if do_rot else None,
            last_q=last_q if do_rot else None,
            last_e=last_e if do_rot else None)


def _bake_world_pairs_iter_fast(pairs, src_rig, tgt_rig, frame_start, frame_end):
    """Fast FK->IK bake — see module docstring for the architecture.

    Generator: yields once per frame, then drains _bulk_write_keys.

    SAMPLE-mode pole pairs aren't supported here; the dispatcher in
    bake_ik.py falls back to classic when any pair is SAMPLE.
    """
    pbs = tgt_rig.pose.bones
    tgt_mw = tgt_rig.matrix_world
    tgt_mw_inv = tgt_mw.inverted()

    _eval_pbs = _settle_driver_state(tgt_rig)

    rest_data = _build_pair_rest_data(pairs, pbs, tgt_rig, _eval_pbs)

    if not rest_data:
        return

    _sort_rest_data(rest_data, pbs)

    child_of_subtarget_arm = _snapshot_subtarget_arms(rest_data, pbs)

    walk_names, constraint_meta = _collect_walk_and_constraint_meta(
        tgt_rig, pairs, rest_data, pbs, _eval_pbs)

    fcurve_info, snapshot_arm, has_fcurves = _extract_fcurves_and_snapshots(
        tgt_rig, walk_names, pbs, constraint_meta)

    rest_arm, rel_rest, parent_name_of = _build_rest_tables(walk_names, pbs)

    _attach_ctrl_parent_data(rest_data)

    _reval_names = _compute_reval_names(rest_data, walk_names,
                                        parent_name_of, constraint_meta)

    _anchor_pole_bend_axes(rest_data, frame_start, frame_end,
                           walk_names, parent_name_of, constraint_meta,
                           has_fcurves, fcurve_info, rel_rest, rest_arm,
                           snapshot_arm, tgt_mw, tgt_mw_inv, pbs)

    _pole_dbg_audit_constraints(rest_data, tgt_rig,
                                constraint_meta=constraint_meta,
                                snapshot_arm=snapshot_arm)

    # ---- Per-frame loop ----
    key_accum = {}
    primed_channels = set()
    last_q = {}
    # Euler-mode bones need previous-frame compat to avoid the
    # representation-flip wobble — for a given rotation there are
    # multiple euler triplets and to_euler picks one arbitrarily;
    # passing the previous frame's euler as `compat` keeps the
    # representation continuous across frames. Blender's pb.matrix=
    # setter does this internally, which is part of why the classic
    # engine doesn't have to.
    last_e = {}

    def _reval_parent_chain(ctrl_parent, arm_local, baked_arm, frame):
        """Re-evaluate the tainted walk bones (topo order) with pair
        targets resolved from this frame's baked poses, so the back-
        solve sees the parent pose PLAYBACK will produce. Mirrors the
        walk's per-bone rules exactly (fcurves / constrained-root-from-
        rest / hierarchy inheritance) — only the constraint/parent
        lookup differs. Returns the control parent's arm-local (which
        may itself be a baked pair target — child-of-a-baked-control
        setups resolve through baked_arm)."""
        reval_out = {}
        lookup = _OverlayLookup(baked_arm, reval_out, arm_local)
        for name in _reval_names:
            parent = parent_name_of.get(name)
            cur_parent = lookup.get(parent) if parent is not None else None
            cons = constraint_meta.get(name)
            if has_fcurves.get(name, False):
                info = fcurve_info[name]
                loc, q, scale = _eval_basis_with_scale(info, frame)
                basis = Matrix.LocRotScale(loc, q, scale)
                if cur_parent is None:
                    pre_constraint = rel_rest[name] @ basis
                else:
                    pre_constraint = cur_parent @ rel_rest[name] @ basis
            elif cur_parent is None:
                if cons:
                    pre_constraint = rest_arm[name]
                else:
                    pre_constraint = snapshot_arm[name]
            else:
                pre_constraint = cur_parent @ rel_rest[name]
            if cons:
                reval_out[name] = _apply_constraint_chain(
                    pre_constraint, cons, tgt_mw, tgt_mw_inv,
                    lookup, parent_arm=cur_parent,
                    rel_rest=rel_rest.get(name))
            else:
                reval_out[name] = pre_constraint
        return lookup.get(ctrl_parent)

    for frame in range(frame_start, frame_end + 1):
        arm_local = _walk_arm_locals_for_frame(
            frame, walk_names, parent_name_of, constraint_meta,
            has_fcurves, fcurve_info, rel_rest, rest_arm, snapshot_arm,
            tgt_mw, tgt_mw_inv)
        _bake_pairs_for_frame(
            frame, rest_data, arm_local, child_of_subtarget_arm, pbs,
            fcurve_info, has_fcurves, tgt_rig, tgt_mw, tgt_mw_inv,
            _reval_parent_chain, key_accum, primed_channels, last_q,
            last_e, frame_start, frame_end)

        yield frame

    yield from _bulk_write_keys(tgt_rig, key_accum, primed_channels, frame_start)

    _pole_dbg_post_bake_validate(tgt_rig)
