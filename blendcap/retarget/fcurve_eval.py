"""Pure-math source-bone fcurve evaluator used by the fast FK and
fast FK->IK engines. Reads animation fcurves directly via
fc.evaluate (Blender's public C-level sampler) — no scene.frame_set
required, no depsgraph eval. Together with _walk_arm_local in
the bake engines, this is the path that bypasses Blender's per-
frame full-rig evaluation cost on dense rigs.
"""
from mathutils import Quaternion, Vector, Euler

from .fcurves import _iter_action_fcurves


def _extract_source_fcurves(src_rig, bone_names, include_scale=False):
    """Pull rotation_*/location (and optionally scale) fcurves for each
    requested source bone.

    Returns: {bone_name: {
        'rotation_quaternion': [fc_w, fc_x, fc_y, fc_z] or None,
        'rotation_euler':      [fc_x, fc_y, fc_z]       or None,
        'rotation_axis_angle': [fc_w, fc_x, fc_y, fc_z] or None,
        'location':            [fc_x, fc_y, fc_z]       or None,
        'scale':               [fc_x, fc_y, fc_z]       or None,   # only when include_scale
        'rotation_mode':       pose-bone rotation mode string,
    }}

    Channels with no fcurves: value is None (use rest default).
    Channels with PARTIAL fcurves (e.g. only Y keyed on a euler):
    list is present but missing slots are None (use rest default for
    those indices).

    `include_scale` is opt-in because FK delta-from-rest math doesn't
    need scale; the IK walk does (chain bones with non-unit scale need
    correct arm-local matrices).

    Uses _iter_action_fcurves so we cover Blender 4.3 (flat fcurves),
    4.4 early (strip.fcurves), and 4.4+/5.x slotted layouts.
    """
    result = {}
    action = src_rig.animation_data and src_rig.animation_data.action

    pb_map = src_rig.pose.bones
    if action is None:
        # No animation — every requested bone reports "no fcurves" and
        # the eval path falls through to rest defaults (identity rot,
        # zero loc) per frame.
        for name in bone_names:
            pb = pb_map.get(name)
            if pb is None:
                continue
            info = {
                'rotation_quaternion': None,
                'rotation_euler': None,
                'rotation_axis_angle': None,
                'location': None,
                'rotation_mode': pb.rotation_mode,
            }
            if include_scale:
                info['scale'] = None
            result[name] = info
        return result

    # Single pass over the action's fcurves, bucketed by data_path. On a
    # face rig (~700 fcurves) this is ~700 inserts vs N*M lookup walks.
    fcurves_by_path = {}
    for fc in _iter_action_fcurves(action):
        fcurves_by_path.setdefault(fc.data_path, []).append(fc)

    channels = [('rotation_quaternion', 4),
                ('rotation_euler', 3),
                ('rotation_axis_angle', 4),
                ('location', 3)]
    if include_scale:
        channels.append(('scale', 3))

    for name in bone_names:
        pb = pb_map.get(name)
        if pb is None:
            continue
        bone_key = name.replace('"', '\\"')
        bone_path = f'pose.bones["{bone_key}"]'

        info = {
            'rotation_mode': pb.rotation_mode,
            'rotation_quaternion': None,
            'rotation_euler': None,
            'rotation_axis_angle': None,
            'location': None,
        }
        if include_scale:
            info['scale'] = None
        for channel, n in channels:
            fcs = fcurves_by_path.get(bone_path + '.' + channel)
            if not fcs:
                continue
            slots = [None] * n
            for fc in fcs:
                if 0 <= fc.array_index < n:
                    slots[fc.array_index] = fc
            info[channel] = slots

        result[name] = info

    return result


def _eval_basis(fcurve_info, frame):
    """Evaluate one source bone's basis at `frame`.

    Returns (loc_vec, rot_quat). fc.evaluate handles bezier / linear /
    constant interpolation, extrapolation modes, and fcurve modifiers
    — same code Blender uses internally during depsgraph eval, minus
    the constraint/driver pass.

    Rotation mode handling:
      QUATERNION  -> read 4 channels into Quaternion(w, x, y, z)
      AXIS_ANGLE  -> read 4 channels: idx 0 = angle, idx 1-3 = axis
                     (matches Blender's storage order for the property)
      Euler ('XYZ', 'ZYX', etc.) -> read 3 channels into Euler(...) and
                     convert to quaternion. The mode string IS the order.

    If no fcurves exist for the rotation channel (e.g. a bone has only
    location keys), returns identity quaternion. Same for location.
    """
    # --- Location
    loc_fcs = fcurve_info['location']
    if loc_fcs is None:
        loc = Vector((0.0, 0.0, 0.0))
    else:
        loc = Vector((
            loc_fcs[0].evaluate(frame) if loc_fcs[0] is not None else 0.0,
            loc_fcs[1].evaluate(frame) if loc_fcs[1] is not None else 0.0,
            loc_fcs[2].evaluate(frame) if loc_fcs[2] is not None else 0.0,
        ))

    # --- Rotation
    mode = fcurve_info['rotation_mode']
    if mode == 'QUATERNION':
        q_fcs = fcurve_info['rotation_quaternion']
        if q_fcs is None:
            q = Quaternion()
        else:
            q = Quaternion((
                q_fcs[0].evaluate(frame) if q_fcs[0] is not None else 1.0,
                q_fcs[1].evaluate(frame) if q_fcs[1] is not None else 0.0,
                q_fcs[2].evaluate(frame) if q_fcs[2] is not None else 0.0,
                q_fcs[3].evaluate(frame) if q_fcs[3] is not None else 0.0,
            ))
    elif mode == 'AXIS_ANGLE':
        aa_fcs = fcurve_info['rotation_axis_angle']
        if aa_fcs is None:
            q = Quaternion()
        else:
            angle = aa_fcs[0].evaluate(frame) if aa_fcs[0] is not None else 0.0
            ax = Vector((
                aa_fcs[1].evaluate(frame) if aa_fcs[1] is not None else 0.0,
                aa_fcs[2].evaluate(frame) if aa_fcs[2] is not None else 1.0,
                aa_fcs[3].evaluate(frame) if aa_fcs[3] is not None else 0.0,
            ))
            if ax.length < 1e-9:
                ax = Vector((0.0, 1.0, 0.0))
            q = Quaternion(ax, angle)
    else:
        # Euler — the rotation_mode string IS the rotation order.
        e_fcs = fcurve_info['rotation_euler']
        if e_fcs is None:
            q = Quaternion()
        else:
            e = Euler((
                e_fcs[0].evaluate(frame) if e_fcs[0] is not None else 0.0,
                e_fcs[1].evaluate(frame) if e_fcs[1] is not None else 0.0,
                e_fcs[2].evaluate(frame) if e_fcs[2] is not None else 0.0,
            ), mode)
            q = e.to_quaternion()

    return loc, q
