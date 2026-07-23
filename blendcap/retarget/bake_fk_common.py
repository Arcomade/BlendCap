"""Shared FK-bake setup helpers used by both the classic engine
(bake_fk.py) and the fcurve-direct fast engine (bake_fk_fast.py).

The two engines share the world-space delta-from-rest math and the
per-pair invariant set (S_rest_q, offset_q, loc_frame_q, axes flags,
pair_scale, thresh_value, needs_delta). Only the per-frame source
evaluation differs. These helpers extract the common setup so both
engines compute identical invariants from identical inputs.

Functions here are pure (modulo reading scene props that the
amplitude/denoise resolvers already require).
"""
from mathutils import Matrix

from .preset_io import _resolve_loc_scale, _resolve_denoise


def _compute_scale_ratio(src_mw, tgt_mw):
    """Source/target world-scale ratio under the uniform-scale assumption.

    Both rigs' object-space scale are averaged (per-axis non-uniformity
    is rare and correct retargeting under it is undefined anyway).
    Guarded against zero target scale.
    """
    src_obj_scale = src_mw.to_scale()
    tgt_obj_scale = tgt_mw.to_scale()
    src_avg = (src_obj_scale.x + src_obj_scale.y + src_obj_scale.z) / 3.0
    tgt_avg = (tgt_obj_scale.x + tgt_obj_scale.y + tgt_obj_scale.z) / 3.0
    return src_avg / max(1e-9, tgt_avg)


def _compute_pair_rest_invariants(p, src_pb, tgt_data_bone, src_mw, tgt_mw,
                                  source_rest_override, override_rotation_only,
                                  compensation_on, denoise_on, scene):
    """Compute the rest-pose invariants both engines need for one pair.

    Returns a dict with these keys (callers extend with engine-specific
    fields like 's_pb' / 'source' / 'target' / 'pair' / 'S_rest_w' / etc):

        S_rest_w, T_rest_w   — world rest matrices (used by classic;
                              fast doesn't keep these but they're cheap
                              to compute and useful for callers that
                              want them).
        S_rest_q, S_rest_q_inv, offset_q, loc_frame_q
        in_rot, in_loc       — channel flags derived from p.channels.
        axes_x, axes_y, axes_z — axes mask derived from p.axes.
        pair_scale, thresh_value, needs_delta — amplitude / denoise
                              resolvers, captured once at setup.

    `source_rest_override` is the same per-bone map the operator
    passes for "Use current pose as rest" / custom rest pose presets.
    `override_rotation_only=True` (default) keeps the edit-bone
    translation and uses only the snapshot's rotation.
    """
    edit_rest_w = src_mw @ src_pb.bone.matrix_local
    if source_rest_override is not None and p.source in source_rest_override:
        snap_w = src_mw @ source_rest_override[p.source]
        if override_rotation_only:
            S_rest_w = Matrix.LocRotScale(
                edit_rest_w.translation,
                snap_w.to_quaternion(),
                (1.0, 1.0, 1.0),
            )
        else:
            S_rest_w = snap_w
    else:
        S_rest_w = edit_rest_w
    T_rest_w = tgt_mw @ tgt_data_bone.matrix_local

    S_rest_q = S_rest_w.to_quaternion()
    T_rest_q = T_rest_w.to_quaternion()
    S_rest_q_inv = S_rest_q.inverted()
    offset_q = S_rest_q_inv @ T_rest_q
    # LOC frame conversion: rotate src-basis loc into tgt-basis frame.
    loc_frame_q = T_rest_q.inverted() @ S_rest_q

    channels = p.channels
    axes_str = (p.axes or "XYZ").upper()
    pair_scale = _resolve_loc_scale(p, compensation_on, scene)
    thresh_value = _resolve_denoise(p, denoise_on, scene)

    # World->target-basis rotation/scale inverse — used by the
    # world-space LOC path to convert a world-space delta-from-rest
    # into the target bone's basis frame. Computed once per pair. The
    # forward rotation is kept too: the fast engine's BASIS LOC path
    # filters partial axes in the WORLD frame (basis -> world -> filter
    # -> basis round trip).
    T_rest_w_rot = T_rest_w.to_3x3()
    T_rest_w_rot_inv = T_rest_w_rot.inverted()

    return {
        'S_rest_w': S_rest_w,
        'T_rest_w': T_rest_w,
        'S_rest_q': S_rest_q,
        'S_rest_q_inv': S_rest_q_inv,
        'offset_q': offset_q,
        'loc_frame_q': loc_frame_q,
        'T_rest_w_rot': T_rest_w_rot,
        'T_rest_w_rot_inv': T_rest_w_rot_inv,
        'in_rot': channels in ('ROT', 'LOC_ROT'),
        'in_loc': channels in ('LOC', 'LOC_ROT'),
        'axes_x': 'X' in axes_str,
        'axes_y': 'Y' in axes_str,
        'axes_z': 'Z' in axes_str,
        'pair_scale': pair_scale,
        # HEAD_LOCAL amplitude pivot: False (default) scales the raw
        # captured delta (head-anchored — the long-standing behavior
        # ARP/Rigify presets were tuned against); True scales the
        # back-solved residual over the parent/carry (for handles whose
        # parent already follows the jaw — CloudRig graded lip rows).
        'scale_residual': bool(getattr(p, 'loc_scale_residual', False)),
        'thresh_value': thresh_value,
        'needs_delta': (thresh_value > 0.0 or abs(pair_scale - 1.0) > 1e-6),
    }
