"""Pure-math simulator for Blender's pose-bone constraint evaluation.

The fast FK engine (bake_fk_fast.py) and the fast FK->IK engine
(bake_ik_fast.py) both need to derive per-frame arm-local matrices
without round-tripping through scene.frame_set + Blender's depsgraph
(the dominant cost on rigs with thousands of bones). They both walk
the target skeleton, multiplying parent_arm @ rel_rest @ basis to
get a pre-constraint arm-local, then apply the supported constraint
subset on top.

The simulator covers POSE/POSE COPY_*, CHILD_OF, the *_TRACK family,
STRETCH_TO, LIMIT_DISTANCE, FLOOR, LIMIT_* (rotation / location /
scale), MAINTAIN_VOLUME, ARMATURE multi-target blending, and the
local-offset COPY_LOCATION variants (owner LOCAL, target LOCAL or
LOCAL_OWNER_ORIENT, use_offset) that Rigify's face rigs use for
"secondary control follows primaries" stacks — lips / lids / brows /
cheeks / nose on both the new (MCH-*_offset) and old (on-control)
face. Constraints outside _SUPPORTED_TYPES are silently skipped —
bones that depend on them (IK, drivers, Transform, Action, etc.)
produce wrong results in the fast engines.

`_compensate_basis_for_constraints` is the inverse operation: given
a desired arm-local for a mapped bone with constraints, derives the
basis that would land at it after constraints fire.

`_topo_sort_target_bones` orders the per-frame walk so each bone is
visited after its data parent AND after every constraint subtarget
it depends on.
"""
import math

from mathutils import Matrix, Quaternion, Vector, Euler


# Constraint types we simulate in pure math. Anything outside this set
# is either silently skipped (unary constraints with subtle interactions
# we don't model — bone behaves as if the constraint were muted) or, in
# the case of subtarget-based constraints we don't simulate, the bone
# will produce wrong results. Add types here as needs arise; each entry
# needs matching cases in _apply_constraint_chain and (optionally) in
# _compensate_basis_for_constraints.
_SUPPORTED_TYPES = frozenset({
    # Subtarget-based — affect arm-local using another bone's pose.
    'COPY_TRANSFORMS', 'COPY_LOCATION', 'COPY_ROTATION', 'COPY_SCALE',
    'CHILD_OF',
    'DAMPED_TRACK', 'TRACK_TO', 'LOCKED_TRACK',
    'STRETCH_TO',
    'LIMIT_DISTANCE',
    'FLOOR',
    'ARMATURE',  # Multi-target weighted blending (Rigify SWITCH_PARENT)
    # Unary — affect arm-local based on the bone's own pose only.
    'LIMIT_ROTATION', 'LIMIT_LOCATION', 'LIMIT_SCALE',
    'MAINTAIN_VOLUME',
})

# Constraint types we deliberately don't simulate. The scan skips them
# but prints a per-bone console warning so a wrong-looking bake is
# self-explanatory instead of a mystery. Add to _SUPPORTED_TYPES +
# matching scan/apply code when needed.
_UNSUPPORTED_TYPES = frozenset({
    'IK',              # Needs FABRIK or similar inverse solver
    'SPLINE_IK',       # Needs curve evaluation + IK solver
    'TRANSFORM',       # Input remapping with min/max channels — complex
    'ACTION',          # Reads + evaluates a sub-Action at a mapped frame
    'FOLLOW_PATH',     # Needs curve evaluation
    'CLAMP_TO',        # Needs curve evaluation
    'PIVOT',           # Reference-frame setup, niche
    'SHRINKWRAP',      # Needs mesh queries
    'TRANSFORM_CACHE', # Alembic/USD playback
    'CAMERA_SOLVER',   # Camera-tracking
    'FOLLOW_TRACK',    # Camera-tracking
    'OBJECT_SOLVER',   # Camera-tracking
})


def _scan_supported_constraints(pb, walk_set, eval_pb=None):
    """Scan a target pose-bone for constraints we can simulate in pure
    math. Returns list of metadata dicts in evaluation order.

    For subtarget-based constraints (COPY_*, CHILD_OF, *_TRACK), only
    POSE/POSE spaces are simulated — other spaces (LOCAL,
    LOCAL_WITH_PARENT, WORLD on owner side, etc.) are silently skipped.
    Most retarget-target rigs use POSE/POSE; rigs that use other
    spaces will produce slightly-off results on those bones.
    Exception: COPY_LOCATION with owner LOCAL / target LOCAL or
    LOCAL_OWNER_ORIENT / use_offset IS simulated (exactly) — that's
    the Rigify face follow-stack variant; see the capture block below.

    Unary constraints (LIMIT_*, MAINTAIN_VOLUME) don't need a subtarget;
    they clamp/modify the bone's own arm-local.

    Constraints OUTSIDE _SUPPORTED_TYPES (IK, Spline IK, Action, Follow
    Path, Transformation, etc. — see _UNSUPPORTED_TYPES) are skipped
    with a console warning: bones carrying them can bake wrong in the
    fast engine. A depsgraph fallback for those is the next layer of
    robustness (not implemented in this version).
    """
    out = []
    for con in pb.constraints:
        # Driver-driven influence/mute (Rigify/ARP FK-IK blend switches,
        # Rigify neck_follow / head_follow) are written by the depsgraph
        # to the EVALUATED datablock, NOT the original `con` we iterate
        # here. On a cold rig the original con.influence reads its
        # pre-settle value, so the engine reconstructs blended parent
        # chains (arm FK/IK, neck-follow) in the wrong state for the whole
        # clip — the "first bake wrong, re-bake fixes it" report. Read
        # influence/mute from the evaluated pose bone when the caller
        # supplies it (mirrors bake_ik_fast's evaluated-influence reads).
        # Static fallback keeps old behavior when eval_pb is None. Other
        # con.* props (spaces, subtarget, rest_length, head_tail) are
        # eval-invariant — keep reading them from the original con below.
        e_con = eval_pb.constraints.get(con.name) if eval_pb is not None else None
        c_mute = e_con.mute if e_con is not None else con.mute
        c_infl = e_con.influence if e_con is not None else con.influence
        if c_mute or c_infl <= 0.0:
            continue
        ctype = con.type
        if ctype not in _SUPPORTED_TYPES:
            # Live (unmuted, influence > 0) constraint we can't simulate:
            # the bone may bake wrong. Say so in the console instead of
            # skipping silently — turns "retarget mangled my rig" reports
            # into self-diagnosing ones.
            if ctype in _UNSUPPORTED_TYPES:
                print(f"[BlendCap] Warning: bone '{pb.name}' has a {ctype} "
                      "constraint the fast retarget engine can't simulate - "
                      "this bone may bake wrong. To compare against the "
                      "classic engine, set blendcap.retarget.bake_fk."
                      "USE_FAST_ENGINE = False in the Python console.")
            else:
                print(f"[BlendCap] Warning: bone '{pb.name}' has an "
                      f"unrecognized {ctype} constraint; the fast retarget "
                      "engine is skipping it.")
            continue
        infl = float(c_infl)
        owner_space = getattr(con, 'owner_space', 'POSE')

        if ctype in ('COPY_TRANSFORMS', 'COPY_LOCATION',
                     'COPY_ROTATION', 'COPY_SCALE'):
            sub = getattr(con, 'subtarget', None)
            if not sub or sub not in walk_set or sub == pb.name:
                continue
            # Space gating: accept POSE/POSE, WORLD/WORLD, and LOCAL/
            # LOCAL — all treated as POSE/POSE math in _apply_constraint
            # _chain. Why each is safe:
            #   POSE/POSE   — direct armature-local copy. Exact.
            #   WORLD/WORLD — same-rig matrix_world cancels. Exact for
            #                 single-armature constraints (the typical
            #                 case). ARP/Rigify FK/IK switching uses
            #                 this heavily on deform bones.
            #   LOCAL/LOCAL — for same-parent bones, local_rot/loc/scale
            #                 reduces to arm-space transfer because the
            #                 parent rotation cancels in the local-space
            #                 conversion. The ARP jawbone.x ← jawbone
            #                 _track.x pattern (both parent head.x) is
            #                 the canonical case. ONLY safe under
            #                 "pure replace" semantics — see the
            #                 LOCAL-specific gates below for the cases
            #                 that REQUIRE proper LOCAL math (those are
            #                 skipped to avoid wrong-offset bugs).
            # Anything else (mixed spaces, LOCAL_WITH_PARENT, CUSTOM,
            # cross-armature WORLD) — skip; needs proper math.
            target_space = getattr(con, 'target_space', 'POSE')
            # Special case: COPY_TRANSFORMS with target=CUSTOM and
            # owner=LOCAL is supported (Rigify jaw mechanism uses this
            # for the "remaining-half-jaw" propagation through
            # MCH-jaw_master_bottom_out). The math reads the target's
            # arm-local pose in the custom-space bone's frame and
            # assigns it to owner's basis (LOCAL = basis-equivalent).
            is_custom_local_copy_transforms = (
                ctype == 'COPY_TRANSFORMS'
                and target_space == 'CUSTOM'
                and owner_space == 'LOCAL'
            )
            # Local-offset Copy Location — Rigify's face "secondary
            # follows primaries" stacks (new face: the MCH-lip_offset /
            # lid_offset / brow_offset / cheek_offset / nose_offset
            # parents + glue bones; old face: the same follows sitting
            # directly on the controls). Semantics measured against
            # Blender 5.1 ground truth (probe: per-axis displacements,
            # posed parents, stacked constraints — all exact):
            #   LOCAL_OWNER_ORIENT — subtarget's channel translation
            #     re-oriented through the two bones' REST orientations:
            #     d = R_owner_rest^-1 @ R_sub_rest @ t. Pose-INDEPENDENT,
            #     so the reorient matrix is captured once here.
            #   LOCAL — raw channel-value copy (d = t, component-wise).
            # Both add the influence-scaled d onto the owner's channel-
            # space translation, stack linearly, and have an exact
            # closed-form inverse in _compensate_basis_for_constraints.
            is_local_offset_copy_loc = (
                ctype == 'COPY_LOCATION'
                and owner_space == 'LOCAL'
                and target_space in ('LOCAL', 'LOCAL_OWNER_ORIENT')
                and getattr(con, 'use_offset', False)
                and float(getattr(con, 'head_tail', 0.0)) == 0.0
            )
            if is_local_offset_copy_loc:
                pass  # dedicated capture below; generic gates don't apply
            elif is_custom_local_copy_transforms:
                space_sub = getattr(con, 'space_subtarget', '')
                if not space_sub or space_sub not in walk_set:
                    continue
            else:
                if owner_space not in ('POSE', 'WORLD', 'LOCAL'):
                    continue
                if target_space not in ('POSE', 'WORLD', 'LOCAL'):
                    continue
                if owner_space != target_space:
                    continue  # Mixed spaces — not approximately POSE/POSE

            # LOCAL-only restrictions: the POSE/POSE approximation is
            # only valid for "pure REPLACE" semantics. OFFSET / ADD /
            # use_offset / partial-axis variants need real LOCAL math
            # (subtract sub's parent rotation before applying) — skip
            # those rather than apply wrong math. ARP's finger curl-all
            # mechanism uses Copy Rotation LOCAL/LOCAL OFFSET; correctly
            # skipping means fingers default to the rig's IK/curl-all
            # being inactive, which is what we want for FK retargets.
            if owner_space == 'LOCAL' and not is_local_offset_copy_loc:
                if ctype == 'COPY_ROTATION':
                    if getattr(con, 'mix_mode', 'REPLACE') != 'REPLACE':
                        continue
                    if not (con.use_x and con.use_y and con.use_z):
                        continue
                    if con.invert_x or con.invert_y or con.invert_z:
                        continue
                elif ctype == 'COPY_TRANSFORMS':
                    if getattr(con, 'mix_mode', 'REPLACE') != 'REPLACE':
                        continue
                elif ctype == 'COPY_LOCATION':
                    if con.use_offset:
                        continue
                    if not (con.use_x and con.use_y and con.use_z):
                        continue
                    if con.invert_x or con.invert_y or con.invert_z:
                        continue
                elif ctype == 'COPY_SCALE':
                    if con.use_offset:
                        continue
                    if not (con.use_x and con.use_y and con.use_z):
                        continue
            entry = {
                'type': ctype,
                'subtarget': sub,
                'influence': infl,
            }
            if ctype == 'COPY_TRANSFORMS':
                entry['mix_mode'] = getattr(con, 'mix_mode', 'REPLACE')
                # Mark CUSTOM/LOCAL with the custom-space bone so apply
                # can read its arm_local at runtime.
                if is_custom_local_copy_transforms:
                    entry['custom_space_target'] = True
                    entry['space_subtarget'] = getattr(con, 'space_subtarget', '')
                # LOCAL/LOCAL: when target and owner have DIFFERENT data
                # parents, the POSE/POSE approximation is wrong. Need to
                # compute target.local relative to target's actual parent,
                # then transform into owner's frame. Capture the target's
                # parent and rel_rest at setup so apply can do the right
                # math without needing tgt_rig at runtime. Rigify spine's
                # MCH-spine ← hips infl=0.5 is the canonical case.
                if owner_space == 'LOCAL' and target_space == 'LOCAL':
                    entry['local_local'] = True
                    tgt_data_bones = pb.id_data.data.bones
                    sub_data = tgt_data_bones.get(sub)
                    if sub_data is not None and sub_data.parent is not None:
                        entry['target_parent'] = sub_data.parent.name
                        entry['target_rel_rest'] = (
                            sub_data.parent.matrix_local.inverted()
                            @ sub_data.matrix_local
                        )
                    else:
                        entry['target_parent'] = None
                        entry['target_rel_rest'] = (
                            sub_data.matrix_local.copy()
                            if sub_data is not None
                            else Matrix.Identity(4)
                        )
            elif ctype == 'COPY_LOCATION':
                entry['use_x'] = con.use_x
                entry['use_y'] = con.use_y
                entry['use_z'] = con.use_z
                entry['invert_x'] = con.invert_x
                entry['invert_y'] = con.invert_y
                entry['invert_z'] = con.invert_z
                entry['use_offset'] = con.use_offset
                # head_tail: 0 = subtarget head, 1 = tail, lerp between.
                # ARP rigs use head_tail=1 on c_arm_fk's Copy Location
                # from c_shoulder so the arm sits at the shoulder TAIL,
                # not the head — without this the FK chain root is
                # offset by a shoulder length. Capture rest length too;
                # arm-local point = sub_arm @ (0, length * head_tail, 0).
                entry['head_tail'] = float(getattr(con, 'head_tail', 0.0))
                sub_bone = pb.id_data.data.bones.get(sub)
                entry['sub_length'] = float(sub_bone.length) if sub_bone else 0.0
                if is_local_offset_copy_loc:
                    if sub_bone is None:
                        continue
                    # Rest-capture for the local-offset variant. Reuses
                    # the 'target_parent'/'target_rel_rest' key names so
                    # _topo_sort_target_bones picks up the dependency
                    # edge without changes.
                    entry['local_offset'] = True
                    entry['target_rest_local'] = sub_bone.matrix_local.copy()
                    if sub_bone.parent is not None:
                        entry['target_parent'] = sub_bone.parent.name
                        entry['target_rel_rest'] = (
                            sub_bone.parent.matrix_local.inverted()
                            @ sub_bone.matrix_local)
                    else:
                        entry['target_parent'] = None
                        entry['target_rel_rest'] = sub_bone.matrix_local.copy()
                    if target_space == 'LOCAL_OWNER_ORIENT':
                        # Rest 3x3s are orthonormal — transposed ==
                        # inverted, cheaper.
                        entry['reorient'] = (
                            pb.bone.matrix_local.to_3x3().transposed()
                            @ sub_bone.matrix_local.to_3x3())
                    else:
                        entry['reorient'] = None
            elif ctype == 'COPY_ROTATION':
                entry['use_x'] = con.use_x
                entry['use_y'] = con.use_y
                entry['use_z'] = con.use_z
                entry['invert_x'] = con.invert_x
                entry['invert_y'] = con.invert_y
                entry['invert_z'] = con.invert_z
                entry['mix_mode'] = getattr(con, 'mix_mode', 'REPLACE')
                entry['euler_order'] = getattr(con, 'euler_order', 'AUTO')
            elif ctype == 'COPY_SCALE':
                entry['use_x'] = con.use_x
                entry['use_y'] = con.use_y
                entry['use_z'] = con.use_z
                entry['use_offset'] = con.use_offset
                entry['use_make_uniform'] = getattr(con, 'use_make_uniform', False)
            out.append(entry)

        elif ctype == 'CHILD_OF':
            sub = getattr(con, 'subtarget', None)
            if not sub or sub not in walk_set or sub == pb.name:
                continue
            out.append({
                'type': 'CHILD_OF',
                'subtarget': sub,
                'inverse_matrix': con.inverse_matrix.copy(),
                'influence': infl,
                'use_location_x': getattr(con, 'use_location_x', True),
                'use_location_y': getattr(con, 'use_location_y', True),
                'use_location_z': getattr(con, 'use_location_z', True),
                'use_rotation_x': getattr(con, 'use_rotation_x', True),
                'use_rotation_y': getattr(con, 'use_rotation_y', True),
                'use_rotation_z': getattr(con, 'use_rotation_z', True),
                'use_scale_x': getattr(con, 'use_scale_x', True),
                'use_scale_y': getattr(con, 'use_scale_y', True),
                'use_scale_z': getattr(con, 'use_scale_z', True),
            })

        elif ctype in ('DAMPED_TRACK', 'TRACK_TO', 'LOCKED_TRACK'):
            sub = getattr(con, 'subtarget', None)
            if not sub or sub not in walk_set or sub == pb.name:
                continue
            sub_bone = pb.id_data.data.bones.get(sub)
            entry = {
                'type': ctype,
                'subtarget': sub,
                'influence': infl,
                'track_axis': con.track_axis,
                'head_tail': float(getattr(con, 'head_tail', 0.0)),
                'sub_length': float(sub_bone.length) if sub_bone else 0.0,
            }
            if ctype == 'TRACK_TO':
                entry['up_axis'] = con.up_axis
                entry['use_target_z'] = getattr(con, 'use_target_z', False)
            elif ctype == 'LOCKED_TRACK':
                entry['lock_axis'] = con.lock_axis
            out.append(entry)

        elif ctype == 'STRETCH_TO':
            sub = getattr(con, 'subtarget', None)
            if not sub or sub not in walk_set or sub == pb.name:
                continue
            rest_length = float(con.rest_length)
            if rest_length < 1e-9:
                # Fall back to bone's data length when the constraint's
                # rest_length is 0 (Blender uses bone.length in that case).
                rest_length = float(pb.bone.length)
            if rest_length < 1e-9:
                continue
            sub_bone = pb.id_data.data.bones.get(sub)
            out.append({
                'type': 'STRETCH_TO',
                'subtarget': sub,
                'influence': infl,
                'rest_length': rest_length,
                'volume': getattr(con, 'volume', 'VOLUME_XZX'),
                'keep_axis': getattr(con, 'keep_axis', 'PLANE_X'),
                'bulge': getattr(con, 'bulge', 1.0),
                'head_tail': float(getattr(con, 'head_tail', 0.0)),
                'sub_length': float(sub_bone.length) if sub_bone else 0.0,
            })

        elif ctype == 'LIMIT_DISTANCE':
            sub = getattr(con, 'subtarget', None)
            if not sub or sub not in walk_set or sub == pb.name:
                continue
            sub_bone = pb.id_data.data.bones.get(sub)
            out.append({
                'type': 'LIMIT_DISTANCE',
                'subtarget': sub,
                'influence': infl,
                'distance': float(con.distance),
                'limit_mode': con.limit_mode,
                'head_tail': float(getattr(con, 'head_tail', 0.0)),
                'sub_length': float(sub_bone.length) if sub_bone else 0.0,
            })

        elif ctype == 'FLOOR':
            sub = getattr(con, 'subtarget', None)
            if not sub or sub not in walk_set or sub == pb.name:
                continue
            sub_bone = pb.id_data.data.bones.get(sub)
            out.append({
                'type': 'FLOOR',
                'subtarget': sub,
                'influence': infl,
                'floor_location': con.floor_location,
                'offset': float(getattr(con, 'offset', 0.0)),
                'use_rotation': getattr(con, 'use_rotation', False),
                'head_tail': float(getattr(con, 'head_tail', 0.0)),
                'sub_length': float(sub_bone.length) if sub_bone else 0.0,
            })

        elif ctype == 'LIMIT_ROTATION':
            if owner_space not in ('POSE', 'LOCAL'):
                continue
            out.append({
                'type': 'LIMIT_ROTATION',
                'influence': infl,
                'owner_space': owner_space,
                'use_limit_x': con.use_limit_x,
                'use_limit_y': con.use_limit_y,
                'use_limit_z': con.use_limit_z,
                'min_x': con.min_x, 'max_x': con.max_x,
                'min_y': con.min_y, 'max_y': con.max_y,
                'min_z': con.min_z, 'max_z': con.max_z,
            })

        elif ctype == 'LIMIT_LOCATION':
            if owner_space not in ('POSE', 'LOCAL'):
                continue
            out.append({
                'type': 'LIMIT_LOCATION',
                'influence': infl,
                'owner_space': owner_space,
                'use_min_x': con.use_min_x, 'min_x': con.min_x,
                'use_max_x': con.use_max_x, 'max_x': con.max_x,
                'use_min_y': con.use_min_y, 'min_y': con.min_y,
                'use_max_y': con.use_max_y, 'max_y': con.max_y,
                'use_min_z': con.use_min_z, 'min_z': con.min_z,
                'use_max_z': con.use_max_z, 'max_z': con.max_z,
            })

        elif ctype == 'LIMIT_SCALE':
            if owner_space not in ('POSE', 'LOCAL'):
                continue
            out.append({
                'type': 'LIMIT_SCALE',
                'influence': infl,
                'owner_space': owner_space,
                'use_min_x': con.use_min_x, 'min_x': con.min_x,
                'use_max_x': con.use_max_x, 'max_x': con.max_x,
                'use_min_y': con.use_min_y, 'min_y': con.min_y,
                'use_max_y': con.use_max_y, 'max_y': con.max_y,
                'use_min_z': con.use_min_z, 'min_z': con.min_z,
                'use_max_z': con.use_max_z, 'max_z': con.max_z,
            })

        elif ctype == 'MAINTAIN_VOLUME':
            out.append({
                'type': 'MAINTAIN_VOLUME',
                'influence': infl,
                'free_axis': con.free_axis,
                'volume': con.volume,
            })

        elif ctype == 'ARMATURE':
            # Multi-target weighted "parent-like" blend. For each target,
            # the owner inherits its REST OFFSET relative to that target
            # — NOT the target's pose directly. Rigify's SWITCH_PARENT
            # uses this with one active target at full weight (functionally
            # equivalent to data-parenting to that bone). Capture the
            # rest_offset = target.rest_local.inverted() @ owner.rest_local
            # at setup so apply can compose contribution = target.arm @
            # rest_offset.
            owner_rest = pb.bone.matrix_local
            tgt_data_bones = pb.id_data.data.bones
            # Per-target weights are driver-driven on parent-switch
            # setups (CloudRig ik_pole_follow / ik_parents, Rigify
            # SWITCH_PARENT) — read them from the evaluated copy when
            # available, same as influence/mute above. The original
            # datablock's weight can be stale on a cold rig.
            e_targets = None
            if e_con is not None:
                try:
                    e_targets = list(e_con.targets)
                except Exception:
                    e_targets = None
            tgt_entries = []
            for t_idx, t in enumerate(con.targets):
                sub = getattr(t, 'subtarget', None)
                if e_targets is not None and t_idx < len(e_targets):
                    weight = float(getattr(e_targets[t_idx], 'weight', 0.0))
                else:
                    weight = float(getattr(t, 'weight', 0.0))
                if not sub or sub not in walk_set or sub == pb.name:
                    continue
                if weight <= 0.0:
                    continue
                target_bone = tgt_data_bones.get(sub)
                if target_bone is None:
                    continue
                rest_offset = target_bone.matrix_local.inverted() @ owner_rest
                tgt_entries.append({
                    'subtarget': sub,
                    'weight': weight,
                    'rest_offset': rest_offset,
                })
            if not tgt_entries:
                continue
            out.append({
                'type': 'ARMATURE',
                'targets': tgt_entries,
                'influence': infl,
                # Lets apply/compensate recover the pure deform delta
                # D_j = sub_arm @ target_rest^-1 from the stored
                # rest_offset (= target_rest^-1 @ owner_rest):
                # D_j = sub_arm @ rest_offset @ owner_rest_inv.
                'owner_rest_inv': owner_rest.inverted(),
            })
    return out


def _armature_delta_from_scan(con, computed_tgt_arm_local):
    """Blended ARMATURE deform delta D for one SCANNED constraint dict,
    from the targets' current arm-local poses:

        D_j = sub_arm @ target_rest^-1
            = sub_arm @ rest_offset @ owner_rest_inv
        D   = weighted blend of D_j (normalized weights)

    Blender's armdef applies bone_post = D @ bone_pre, so the apply
    path multiplies D onto the pre-constraint pose and the compensate
    path feeds D^-1 @ desired into the basis back-solve. Single target
    is exact. Multi-target accumulates the weighted deltas as a LINEAR
    MATRIX SUM with normalized weights — exactly what armdef does when
    use_deform_preserve_volume is off (CloudRig's face Armature@Jaw@
    DEF-Head blends), including its characteristic slight scale shrink
    between diverging rotations. preserve_volume=True (dual quaternion)
    setups get this as an approximation — fine for the one-hot parent
    switches that actually use that flag. An earlier revision
    decompose-averaged loc/rot/scale instead, which diverges from
    Blender as the targets' rotations separate (visible as bake-vs-
    playback drift on jaw-blended lip bones during jaw motion).
    Returns None when no target resolves."""
    owner_rest_inv = con.get('owner_rest_inv')
    if owner_rest_inv is None:
        return None
    collected = []
    for t in con['targets']:
        sub_arm = computed_tgt_arm_local.get(t['subtarget'])
        if sub_arm is None:
            continue
        collected.append((sub_arm @ t['rest_offset'] @ owner_rest_inv,
                          t['weight']))
    if not collected:
        return None
    if len(collected) == 1:
        return collected[0][0]
    total_w = sum(w for _, w in collected)
    if total_w <= 1e-9:
        return None
    acc = Matrix(((0.0, 0.0, 0.0, 0.0),
                  (0.0, 0.0, 0.0, 0.0),
                  (0.0, 0.0, 0.0, 0.0),
                  (0.0, 0.0, 0.0, 0.0)))
    for delta, w in collected:
        nw = w / total_w
        for i in range(4):
            acc[i][0] += nw * delta[i][0]
            acc[i][1] += nw * delta[i][1]
            acc[i][2] += nw * delta[i][2]
            acc[i][3] += nw * delta[i][3]
    return acc


_TRACK_AXIS_VECTORS = {
    'TRACK_X': Vector((1.0, 0.0, 0.0)),
    'TRACK_Y': Vector((0.0, 1.0, 0.0)),
    'TRACK_Z': Vector((0.0, 0.0, 1.0)),
    'TRACK_NEGATIVE_X': Vector((-1.0, 0.0, 0.0)),
    'TRACK_NEGATIVE_Y': Vector((0.0, -1.0, 0.0)),
    'TRACK_NEGATIVE_Z': Vector((0.0, 0.0, -1.0)),
}

_FREE_AXIS_INDEX = {'SAMPLE_X': 0, 'SAMPLE_Y': 1, 'SAMPLE_Z': 2}


def _sub_target_pos(sub_arm, con):
    """Point on the subtarget bone at `head_tail` fraction (0=head, 1=tail).
    Returns arm-local position. ARP rigs lean on this for FK chain roots
    (e.g. c_arm_fk's Copy Location from c_shoulder uses head_tail=1) —
    falling back to sub_arm.translation alone offsets the chain by a bone
    length. Bones whose `con` predates the head_tail capture (or where the
    constraint type doesn't have head_tail) still get the head."""
    h = con.get('head_tail', 0.0)
    if h == 0.0:
        return sub_arm.translation.copy()
    length = con.get('sub_length', 0.0)
    if length == 0.0:
        return sub_arm.translation.copy()
    return sub_arm @ Vector((0.0, length * h, 0.0))


def _local_offset_copy_location_delta(con, computed_tgt_arm_local):
    """Influence-scaled channel-space translation delta contributed by a
    local-offset COPY_LOCATION entry (see the scan-side capture). Shared
    by _apply_constraint_chain (adds it) and _compensate_basis_for_
    constraints (subtracts it) so the two stay exact mirrors. Returns
    None when the subtarget isn't resolvable this frame."""
    sub_arm = computed_tgt_arm_local.get(con['subtarget'])
    if sub_arm is None:
        return None
    # Subtarget's channel-equivalent local translation from its arm
    # matrix: (parent_arm @ rel_rest)^-1 @ sub_arm. When the parent
    # isn't in the walk it holds rest pose, so the frame collapses to
    # the subtarget's own rest matrix_local.
    tp = con.get('target_parent')
    tp_arm = computed_tgt_arm_local.get(tp) if tp else None
    if tp_arm is not None:
        P_sub = tp_arm @ con['target_rel_rest']
    else:
        P_sub = con['target_rest_local']
    t = (P_sub.inverted() @ sub_arm).translation
    reo = con.get('reorient')
    d = (reo @ t) if reo is not None else t.copy()
    if con['invert_x']:
        d.x = -d.x
    if con['invert_y']:
        d.y = -d.y
    if con['invert_z']:
        d.z = -d.z
    # Disabled axes contribute nothing in offset mode. (All shipped
    # Rigify follow stacks are all-axes / no-invert — the masked and
    # inverted paths follow the same post-reorient convention.)
    if not con['use_x']:
        d.x = 0.0
    if not con['use_y']:
        d.y = 0.0
    if not con['use_z']:
        d.z = 0.0
    return d * con['influence']


def _apply_constraint_chain(arm_local, cons, tgt_mw, tgt_mw_inv,
                            computed_tgt_arm_local, parent_arm=None,
                            rel_rest=None):
    """Apply a chain of constraints (in evaluation order) to an
    arm-local pose matrix. Returns the post-constraint arm-local.
    Pure-math simulation of Blender's depsgraph for the supported
    constraint subset (see _SUPPORTED_TYPES at module level)."""
    out = arm_local
    for con in cons:
        ctype = con['type']
        infl = con['influence']

        # --- Subtarget-based constraints
        if ctype == 'COPY_TRANSFORMS':
            sub_arm = computed_tgt_arm_local.get(con['subtarget'])
            if sub_arm is None:
                continue
            # CUSTOM/LOCAL special case: read target's arm pose in the
            # custom space's frame, set owner.basis = that. Used heavily
            # in Rigify face for jaw chain "remaining motion" propagation
            # (MCH-jaw_master_bottom_out). Requires parent_arm + rel_rest
            # so we can compose arm_local = parent @ rel_rest @ basis.
            if con.get('custom_space_target'):
                custom_arm = computed_tgt_arm_local.get(con['space_subtarget'])
                if custom_arm is None:
                    continue
                if parent_arm is None and rel_rest is None:
                    continue  # can't compose without P
                # target_in_custom = custom.inverted() @ target (same-rig
                # CUSTOM math reduces to this since matrix_world cancels).
                target_in_custom = custom_arm.inverted() @ sub_arm
                P = parent_arm @ rel_rest if parent_arm is not None else rel_rest
                new_arm = P @ target_in_custom
                if infl >= 1.0:
                    out = new_arm
                else:
                    out_loc, out_rot, out_scale = out.decompose()
                    new_loc, new_rot, new_scale = new_arm.decompose()
                    out = Matrix.LocRotScale(
                        out_loc.lerp(new_loc, infl),
                        out_rot.slerp(new_rot, infl),
                        out_scale.lerp(new_scale, infl),
                    )
                continue
            # LOCAL/LOCAL special handling: transfer target's local
            # transform (relative to target's data parent) onto owner's
            # local frame (relative to owner's data parent). For SAME
            # parents this reduces to POSE/POSE, but for DIFFERENT
            # parents (Rigify MCH-spine ← hips infl=0.5) the math is
            # non-trivial. We need both target.local and owner's frame
            # (parent_arm @ rel_rest) to compose.
            if con.get('local_local'):
                if parent_arm is None and rel_rest is None:
                    continue  # need P to compose
                # target.local = (parent_target_arm @ rel_rest_target)^-1 @ target_arm
                target_parent_name = con.get('target_parent')
                if target_parent_name:
                    target_parent_arm = computed_tgt_arm_local.get(target_parent_name)
                    if target_parent_arm is not None:
                        P_target = target_parent_arm @ con['target_rel_rest']
                    else:
                        # Target parent not in walk set — use rest-pose
                        # fallback (its arm_local equals its matrix_local).
                        P_target = con['target_rel_rest']
                else:
                    P_target = con['target_rel_rest']
                target_local = P_target.inverted() @ sub_arm
                # Compose new owner.arm = P_owner @ target_local
                P_owner = parent_arm @ rel_rest if parent_arm is not None else rel_rest
                new_arm = P_owner @ target_local
                if infl >= 1.0:
                    out = new_arm
                else:
                    out_loc, out_rot, out_scale = out.decompose()
                    new_loc, new_rot, new_scale = new_arm.decompose()
                    out = Matrix.LocRotScale(
                        out_loc.lerp(new_loc, infl),
                        out_rot.slerp(new_rot, infl),
                        out_scale.lerp(new_scale, infl),
                    )
                continue
            # Standard same-space (POSE/POSE, WORLD/WORLD).
            # v3: only REPLACE mix is fully modeled. Other mix modes
            # (BEFORE/AFTER/SPLIT) approximated as REPLACE — typically
            # only on hand-tuned rigs; common case is REPLACE.
            if infl >= 1.0:
                out = sub_arm.copy()
            else:
                # Partial-influence: decompose + lerp loc/scale, slerp
                # rot. Matrix.lerp() is element-wise and produces non-
                # orthonormal results on rotation matrices — silent
                # corruption that propagates through dependent bones.
                out_loc, out_rot, out_scale = out.decompose()
                sub_loc, sub_rot, sub_scale = sub_arm.decompose()
                out = Matrix.LocRotScale(
                    out_loc.lerp(sub_loc, infl),
                    out_rot.slerp(sub_rot, infl),
                    out_scale.lerp(sub_scale, infl),
                )

        elif ctype == 'COPY_LOCATION':
            sub_arm = computed_tgt_arm_local.get(con['subtarget'])
            if sub_arm is None:
                continue
            if con.get('local_offset'):
                # Channel-space additive follow (Rigify face secondary
                # stacks). Needs the owner's frame to convert between
                # arm and channel space; a call site that doesn't pass
                # parent_arm/rel_rest keeps the pre-fix skip behavior.
                if parent_arm is None and rel_rest is None:
                    continue
                d = _local_offset_copy_location_delta(
                    con, computed_tgt_arm_local)
                if d is None:
                    continue
                P_owner = (parent_arm @ rel_rest
                           if parent_arm is not None else rel_rest)
                local = P_owner.inverted() @ out
                local.translation = local.translation + d
                out = P_owner @ local
                continue
            sub_t = _sub_target_pos(sub_arm, con)
            if con['invert_x']:
                sub_t.x = -sub_t.x
            if con['invert_y']:
                sub_t.y = -sub_t.y
            if con['invert_z']:
                sub_t.z = -sub_t.z
            out_loc, out_rot, out_scale = out.decompose()
            new_t = out_loc.copy()
            if con['use_offset']:
                if con['use_x']:
                    new_t.x = out_loc.x + sub_t.x
                if con['use_y']:
                    new_t.y = out_loc.y + sub_t.y
                if con['use_z']:
                    new_t.z = out_loc.z + sub_t.z
            else:
                if con['use_x']:
                    new_t.x = sub_t.x
                if con['use_y']:
                    new_t.y = sub_t.y
                if con['use_z']:
                    new_t.z = sub_t.z
            if infl < 1.0:
                new_t = out_loc.lerp(new_t, infl)
            out = Matrix.LocRotScale(new_t, out_rot, out_scale)

        elif ctype == 'COPY_ROTATION':
            sub_arm = computed_tgt_arm_local.get(con['subtarget'])
            if sub_arm is None:
                continue
            sub_rot = sub_arm.to_quaternion()
            out_loc, out_rot, out_scale = out.decompose()
            # Axis filtering / inverts via euler if needed.
            if (not (con['use_x'] and con['use_y'] and con['use_z'])
                    or con['invert_x'] or con['invert_y'] or con['invert_z']):
                order = con['euler_order']
                if order == 'AUTO':
                    order = 'XYZ'
                sub_e = sub_rot.to_euler(order)
                out_e = out_rot.to_euler(order)
                new_e = Euler((out_e.x, out_e.y, out_e.z), order)
                if con['use_x']:
                    new_e.x = -sub_e.x if con['invert_x'] else sub_e.x
                if con['use_y']:
                    new_e.y = -sub_e.y if con['invert_y'] else sub_e.y
                if con['use_z']:
                    new_e.z = -sub_e.z if con['invert_z'] else sub_e.z
                base_rot = new_e.to_quaternion()
            else:
                base_rot = sub_rot
            # Mix mode. v3 implements REPLACE plus the common combine
            # modes; some Blender 3.x specific modes fall through to
            # REPLACE.
            mix = con['mix_mode']
            if mix == 'REPLACE':
                new_rot = base_rot
            elif mix == 'ADD' or mix == 'BEFORE':
                new_rot = base_rot @ out_rot
            elif mix == 'AFTER':
                new_rot = out_rot @ base_rot
            elif mix == 'OFFSET':
                new_rot = base_rot @ out_rot
            else:
                new_rot = base_rot
            if infl < 1.0:
                new_rot = out_rot.slerp(new_rot, infl)
            out = Matrix.LocRotScale(out_loc, new_rot, out_scale)

        elif ctype == 'COPY_SCALE':
            sub_arm = computed_tgt_arm_local.get(con['subtarget'])
            if sub_arm is None:
                continue
            sub_s = sub_arm.to_scale()
            if con['use_make_uniform']:
                avg = (sub_s.x + sub_s.y + sub_s.z) / 3.0
                sub_s = Vector((avg, avg, avg))
            out_loc, out_rot, out_scale = out.decompose()
            new_s = Vector(out_scale)
            if con['use_offset']:
                if con['use_x']:
                    new_s.x = out_scale.x * sub_s.x
                if con['use_y']:
                    new_s.y = out_scale.y * sub_s.y
                if con['use_z']:
                    new_s.z = out_scale.z * sub_s.z
            else:
                if con['use_x']:
                    new_s.x = sub_s.x
                if con['use_y']:
                    new_s.y = sub_s.y
                if con['use_z']:
                    new_s.z = sub_s.z
            if infl < 1.0:
                new_s = out_scale.lerp(new_s, infl)
            out = Matrix.LocRotScale(out_loc, out_rot, new_s)

        elif ctype == 'CHILD_OF':
            sub_arm = computed_tgt_arm_local.get(con['subtarget'])
            if sub_arm is None:
                continue
            # v3 ignores per-component flags (use_location_*/etc.);
            # default settings (all on) are by far the common case.
            sub_world = tgt_mw @ sub_arm
            new_arm = (tgt_mw_inv @ sub_world @ con['inverse_matrix']
                       @ tgt_mw @ out)
            if infl >= 1.0:
                out = new_arm
            else:
                # Partial-influence — decompose + slerp rotation rather
                # than element-wise Matrix.lerp (which corrupts rotation).
                out_loc, out_rot, out_scale = out.decompose()
                new_loc, new_rot, new_scale = new_arm.decompose()
                out = Matrix.LocRotScale(
                    out_loc.lerp(new_loc, infl),
                    out_rot.slerp(new_rot, infl),
                    out_scale.lerp(new_scale, infl),
                )

        elif ctype in ('DAMPED_TRACK', 'TRACK_TO', 'LOCKED_TRACK'):
            sub_arm = computed_tgt_arm_local.get(con['subtarget'])
            if sub_arm is None:
                continue
            target_pos = _sub_target_pos(sub_arm, con)
            bone_pos = out.translation
            direction = target_pos - bone_pos
            if direction.length < 1e-9:
                continue
            direction = direction.normalized()
            out_loc, out_rot, out_scale = out.decompose()
            local_track = _TRACK_AXIS_VECTORS.get(
                con['track_axis'], Vector((0.0, 1.0, 0.0)))
            world_track = out_rot @ local_track
            track_quat = world_track.rotation_difference(direction)
            new_rot = track_quat @ out_rot
            # TRACK_TO additionally orients an up axis; LOCKED_TRACK
            # constrains rotation to one axis. v3 treats all three as
            # damped-track (rotation that points track_axis at target).
            # For rigs that depend on the up-axis / lock-axis behavior,
            # this is an approximation.
            if infl < 1.0:
                new_rot = out_rot.slerp(new_rot, infl)
            out = Matrix.LocRotScale(out_loc, new_rot, out_scale)

        elif ctype == 'STRETCH_TO':
            sub_arm = computed_tgt_arm_local.get(con['subtarget'])
            if sub_arm is None:
                continue
            target_pos = _sub_target_pos(sub_arm, con)
            bone_pos = out.translation
            direction = target_pos - bone_pos
            dist = direction.length
            rest_length = con['rest_length']
            if dist < 1e-9 or rest_length < 1e-9:
                continue
            direction_n = direction / dist
            out_loc, out_rot, out_scale = out.decompose()
            # Align bone's Y axis with the direction (Damped Track math).
            world_y = out_rot @ Vector((0.0, 1.0, 0.0))
            track_quat = world_y.rotation_difference(direction_n)
            new_rot = track_quat @ out_rot
            # Scale Y by stretch ratio. Volume preservation for X/Z
            # depends on the volume mode: NO_VOLUME keeps original
            # scales; VOLUME_XZX / VOLUME_X / VOLUME_Z apply 1/sqrt
            # inverse along the volume-preserving axes.
            stretch = dist / rest_length
            volume = con['volume']
            if volume == 'NO_VOLUME':
                new_scale = Vector((out_scale.x, stretch, out_scale.z))
            else:
                inv = 1.0 / math.sqrt(max(1e-9, stretch))
                if volume == 'VOLUME_X':
                    new_scale = Vector((inv, stretch, out_scale.z))
                elif volume == 'VOLUME_Z':
                    new_scale = Vector((out_scale.x, stretch, inv))
                else:
                    # VOLUME_XZX (default) — both X and Z preserve volume
                    new_scale = Vector((inv, stretch, inv))
            if infl < 1.0:
                new_rot = out_rot.slerp(new_rot, infl)
                new_scale = out_scale.lerp(new_scale, infl)
            out = Matrix.LocRotScale(out_loc, new_rot, new_scale)

        elif ctype == 'LIMIT_DISTANCE':
            sub_arm = computed_tgt_arm_local.get(con['subtarget'])
            if sub_arm is None:
                continue
            target_pos = _sub_target_pos(sub_arm, con)
            bone_pos = out.translation
            diff = bone_pos - target_pos
            dist = diff.length
            if dist < 1e-9:
                continue
            target_dist = con['distance']
            mode = con['limit_mode']
            new_dist = dist
            if mode == 'LIMITDIST_INSIDE' and dist > target_dist:
                new_dist = target_dist
            elif mode == 'LIMITDIST_OUTSIDE' and dist < target_dist:
                new_dist = target_dist
            elif mode == 'LIMITDIST_ONSURFACE':
                new_dist = target_dist
            if new_dist == dist:
                continue  # already within bounds
            new_pos = target_pos + diff * (new_dist / dist)
            out_loc, out_rot, out_scale = out.decompose()
            if infl < 1.0:
                new_pos = bone_pos.lerp(new_pos, infl)
            out = Matrix.LocRotScale(new_pos, out_rot, out_scale)

        elif ctype == 'FLOOR':
            sub_arm = computed_tgt_arm_local.get(con['subtarget'])
            if sub_arm is None:
                continue
            axis_map = {
                'FLOOR_X':  (0, +1), 'FLOOR_NEG_X': (0, -1),
                'FLOOR_Y':  (1, +1), 'FLOOR_NEG_Y': (1, -1),
                'FLOOR_Z':  (2, +1), 'FLOOR_NEG_Z': (2, -1),
            }
            idx, sign = axis_map.get(con['floor_location'], (2, +1))
            target_pos = _sub_target_pos(sub_arm, con)
            offset = con['offset']
            floor_val = target_pos[idx] + offset * sign
            bone_pos = out.translation
            new_pos = bone_pos.copy()
            # sign +1: bone must be >= floor_val. sign -1: <= floor_val.
            if sign > 0 and new_pos[idx] < floor_val:
                new_pos[idx] = floor_val
            elif sign < 0 and new_pos[idx] > floor_val:
                new_pos[idx] = floor_val
            else:
                continue  # already above/below floor
            if infl < 1.0:
                new_pos = bone_pos.lerp(new_pos, infl)
            out_loc, out_rot, out_scale = out.decompose()
            out = Matrix.LocRotScale(new_pos, out_rot, out_scale)

        # --- Unary constraints
        elif ctype == 'LIMIT_ROTATION':
            out_loc, out_rot, out_scale = out.decompose()
            e = out_rot.to_euler('XYZ')
            new_e = Euler((e.x, e.y, e.z), 'XYZ')
            if con['use_limit_x']:
                new_e.x = max(con['min_x'], min(con['max_x'], new_e.x))
            if con['use_limit_y']:
                new_e.y = max(con['min_y'], min(con['max_y'], new_e.y))
            if con['use_limit_z']:
                new_e.z = max(con['min_z'], min(con['max_z'], new_e.z))
            new_rot = new_e.to_quaternion()
            if infl < 1.0:
                new_rot = out_rot.slerp(new_rot, infl)
            out = Matrix.LocRotScale(out_loc, new_rot, out_scale)

        elif ctype == 'LIMIT_LOCATION':
            out_loc, out_rot, out_scale = out.decompose()
            new_t = out_loc.copy()
            if con['use_min_x']:
                new_t.x = max(con['min_x'], new_t.x)
            if con['use_max_x']:
                new_t.x = min(con['max_x'], new_t.x)
            if con['use_min_y']:
                new_t.y = max(con['min_y'], new_t.y)
            if con['use_max_y']:
                new_t.y = min(con['max_y'], new_t.y)
            if con['use_min_z']:
                new_t.z = max(con['min_z'], new_t.z)
            if con['use_max_z']:
                new_t.z = min(con['max_z'], new_t.z)
            if infl < 1.0:
                new_t = out_loc.lerp(new_t, infl)
            out = Matrix.LocRotScale(new_t, out_rot, out_scale)

        elif ctype == 'LIMIT_SCALE':
            out_loc, out_rot, out_scale = out.decompose()
            new_s = Vector(out_scale)
            if con['use_min_x']:
                new_s.x = max(con['min_x'], new_s.x)
            if con['use_max_x']:
                new_s.x = min(con['max_x'], new_s.x)
            if con['use_min_y']:
                new_s.y = max(con['min_y'], new_s.y)
            if con['use_max_y']:
                new_s.y = min(con['max_y'], new_s.y)
            if con['use_min_z']:
                new_s.z = max(con['min_z'], new_s.z)
            if con['use_max_z']:
                new_s.z = min(con['max_z'], new_s.z)
            if infl < 1.0:
                new_s = out_scale.lerp(new_s, infl)
            out = Matrix.LocRotScale(out_loc, out_rot, new_s)

        elif ctype == 'MAINTAIN_VOLUME':
            out_loc, out_rot, out_scale = out.decompose()
            idx = _FREE_AXIS_INDEX.get(con['free_axis'], 1)
            s_free = max(1e-9, abs(out_scale[idx]))
            s_other = math.sqrt(max(0.0, con['volume'] / s_free))
            new_s = Vector(out_scale)
            for i in range(3):
                if i != idx:
                    new_s[i] = s_other
            if infl < 1.0:
                new_s = out_scale.lerp(new_s, infl)
            out = Matrix.LocRotScale(out_loc, out_rot, new_s)

        elif ctype == 'ARMATURE':
            # ARMATURE is "parent-like" not "copy-pose": Blender's
            # armdef computes post = D @ pre, where D is the weighted
            # blend of per-target deform deltas (target.arm @
            # target_rest^-1) and pre is the owner's pre-constraint
            # pose. Earlier revisions replaced the pose with
            # D @ owner_REST instead — identical while the owner has no
            # basis animation (Rigify SWITCH_PARENT receptacles,
            # CloudRig P-* bones, all rest-parked), but wrong for
            # owners that DO carry basis keys (CloudRig FK-Toes after
            # the mapped-bone armature compensation writes them).
            # D @ pre matches Blender for both. Falls back to skipping
            # when no target resolves (matches old behavior).
            D = _armature_delta_from_scan(con, computed_tgt_arm_local)
            if D is None:
                continue
            new_arm = D @ out
            # Blender NORMALIZES armature weights (they're relative,
            # like vertex weights) — a single target is a FULL carry
            # whatever its weight value (CloudRig's STR-I face weld
            # handles sit at fractional weights and Blender still
            # fully carries them). Only the constraint's influence
            # blends toward the un-constrained pose. An earlier
            # revision lerped by weight here — wrong vs Blender for
            # fractional single-target setups, identical for the
            # one-hot parent switches everything else uses.
            effective = infl
            if effective >= 1.0:
                out = new_arm
            else:
                out_loc, out_rot, out_scale = out.decompose()
                c_loc, c_rot, c_scale = new_arm.decompose()
                out = Matrix.LocRotScale(
                    out_loc.lerp(c_loc, effective),
                    out_rot.slerp(c_rot, effective),
                    out_scale.lerp(c_scale, effective),
                )

    return out


def _compensate_basis_for_constraints(desired_arm, cons, parent_arm, rel_rest,
                                      tgt_mw, tgt_mw_inv, computed_tgt_arm_local):
    """For a mapped bone with constraints, derive the basis matrix such
    that AFTER constraints fire, arm_local_post == desired_arm.

    Walks constraints in reverse order, undoing each effect to derive
    the pre-constraint arm_local, then back-solves basis from the
    pre-constraint pose given the bone's parent + rest.

    Returns None if compensation isn't possible (COPY_TRANSFORMS REPLACE
    with influence=1 forces bone pose entirely from subtarget regardless
    of basis — caller writes a zero/identity basis in that case).
    """
    arm_pre = desired_arm
    for con in reversed(cons):
        ctype = con['type']
        # Local-offset COPY_LOCATION has an EXACT inverse: its delta
        # depends only on subtarget state, so undoing it is a plain
        # subtraction in the owner's channel space. This is the old-
        # face quarter-lip / lid-glue case — the follow constraints sit
        # on the mapped control itself, and without this inverse the
        # live constraint re-adds the follow on top of the baked keys
        # (motion-correlated drift).
        if ctype == 'COPY_LOCATION' and con.get('local_offset'):
            d = _local_offset_copy_location_delta(
                con, computed_tgt_arm_local)
            if d is not None:
                P = (parent_arm @ rel_rest
                     if parent_arm is not None else rel_rest)
                local = P.inverted() @ arm_pre
                local.translation = local.translation - d
                arm_pre = P @ local
            continue
        # Besides that, only COPY_TRANSFORMS and CHILD_OF participate.
        # Other supported types (LIMIT_*, MAINTAIN_VOLUME, ARMATURE, TRACK_*,
        # STRETCH_TO, etc.) don't have a closed-form inverse here and are
        # silently skipped — basis lands at an approximate pose; the live
        # constraint re-applies on playback. Dispatch on type BEFORE reading
        # 'subtarget' since unary/multi-target entries don't carry that key.
        if ctype != 'COPY_TRANSFORMS' and ctype != 'CHILD_OF':
            continue
        sub_arm = computed_tgt_arm_local.get(con['subtarget'])
        if sub_arm is None:
            continue
        infl = con['influence']
        if ctype == 'COPY_TRANSFORMS':
            # post = lerp(pre, sub_arm, infl). At infl=1 post is forced
            # to sub_arm — bone is fully slaved, basis has no effect.
            if infl >= 1.0:
                return None
            # Partial influence: closed-form back-solve through matrix
            # lerp is messy. Skip compensation for this constraint —
            # bone may be slightly off (acceptable for partial-infl,
            # rare in practice).
            continue
        else:  # CHILD_OF
            sub_world = tgt_mw @ sub_arm
            X = (tgt_mw_inv @ sub_world @ con['inverse_matrix'] @ tgt_mw)
            if infl >= 1.0:
                arm_pre = X.inverted() @ arm_pre
            else:
                # Approximate partial-influence back-solve.
                arm_pre = arm_pre.lerp(X.inverted() @ arm_pre, infl)
    P = parent_arm @ rel_rest if parent_arm is not None else rel_rest
    return P.inverted() @ arm_pre


def _topo_sort_target_bones(walk_set, tgt_rig, constraint_meta):
    """Topological sort of target bones: each bone is processed after
    its data parent AND after every constraint subtarget it depends on.
    Falls back to data-depth sort if a cycle is detected (shouldn't
    happen on a well-formed rig but defensive)."""
    from collections import deque
    deps = {n: set() for n in walk_set}
    for name in walk_set:
        bone = tgt_rig.data.bones.get(name)
        if bone and bone.parent and bone.parent.name in walk_set:
            deps[name].add(bone.parent.name)
        for con in constraint_meta.get(name, []):
            sub = con.get('subtarget')
            if sub and sub in walk_set:
                deps[name].add(sub)
            # CUSTOM space references another bone whose pose defines
            # the reference frame — also a dependency.
            space_sub = con.get('space_subtarget')
            if space_sub and space_sub in walk_set:
                deps[name].add(space_sub)
            # ARMATURE constraint has multiple targets in a list.
            for t in con.get('targets', []):
                sub2 = t.get('subtarget')
                if sub2 and sub2 in walk_set:
                    deps[name].add(sub2)
            # LOCAL/LOCAL Copy Transforms needs target's data parent
            # to compute target.local at runtime.
            target_parent = con.get('target_parent')
            if target_parent and target_parent in walk_set:
                deps[name].add(target_parent)
    in_degree = {n: len(deps[n]) for n in walk_set}
    reverse_deps = {n: set() for n in walk_set}
    for name in walk_set:
        for dep in deps[name]:
            reverse_deps[dep].add(name)
    queue = deque(sorted(n for n, d in in_degree.items() if d == 0))
    out = []
    while queue:
        n = queue.popleft()
        out.append(n)
        for dependent in sorted(reverse_deps[n]):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
    if len(out) != len(walk_set):
        def _depth(name):
            d, b = 0, tgt_rig.data.bones[name].parent
            while b is not None:
                d += 1
                b = b.parent
            return d
        out = sorted(walk_set, key=_depth)
    return out
