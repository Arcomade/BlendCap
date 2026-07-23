"""Self-grounding math for the Camera Angle Offset live preview.

Problem this solves: the pitch/roll preview rotates the performance
about the Root bone, which visually lifts or sinks the character —
but the BAKED result re-grounds itself (grounding + foot locking run
after the rotation), so the preview lies about height and users
"fix" it with the Height slider, which then double-applies after a
bake. The fix: after each rotation-preview update, compute where the
lowest foot sample ended up and shift the rig so it sits back at its
pre-rotation level — the preview then lands where the bake lands.

Everything here is bpy-free numpy so the FK math can be unit-tested
outside Blender. properties.py owns the bpy side: reading fcurves,
building the cache, and writing the compensating delta into Hips keys.

Coordinate story (see _apply_camera_offset_to_rig): the preview's
world rotation is R = Ry(roll) @ Rx(pitch) in Blender world space,
conjugated onto Root. For a point q expressed relative to Root's
post-location frame, its world position is

    world = M @ T(root_loc) @ R_basis @ q        (M = arm @ root_rest)

and since R_basis = M_rot^-1 @ R @ M_rot, the world-frame offset from
the per-frame pivot is simply R @ (M_rot @ q). So the cache stores
w(f) = M_rot @ q(f) (world-frame foot offsets from the pivot) plus
pivot_z(f), and a slider tick only computes [R @ w]_z + pivot_z.
"""
import numpy as np


# ------------------------------------------------------------
# Rotation helpers
# ------------------------------------------------------------

def _axis_mats(angles, axis):
    """(n,) angles -> (n, 3, 3) rotation matrices about a world axis."""
    n = angles.shape[0]
    c = np.cos(angles)
    s = np.sin(angles)
    m = np.zeros((n, 3, 3), dtype=np.float64)
    if axis == 'X':
        m[:, 0, 0] = 1
        m[:, 1, 1] = c
        m[:, 1, 2] = -s
        m[:, 2, 1] = s
        m[:, 2, 2] = c
    elif axis == 'Y':
        m[:, 0, 0] = c
        m[:, 0, 2] = s
        m[:, 1, 1] = 1
        m[:, 2, 0] = -s
        m[:, 2, 2] = c
    else:
        m[:, 0, 0] = c
        m[:, 0, 1] = -s
        m[:, 1, 0] = s
        m[:, 1, 1] = c
        m[:, 2, 2] = 1
    return m


def euler_to_mats(eulers, order):
    """(n, 3) XYZ-component euler angles + Blender order string ->
    (n, 3, 3). Blender semantics: order 'XYZ' rotates about X first,
    then Y, then Z, in fixed (world/static) axes — matrix = Rz@Ry@Rx.
    """
    axis_index = {'X': 0, 'Y': 1, 'Z': 2}
    out = None
    for axis in order:          # apply left-to-right = pre-multiply
        m = _axis_mats(eulers[:, axis_index[axis]], axis)
        out = m if out is None else np.einsum('nij,njk->nik', m, out)
    return out


def quats_to_mats(quats):
    """(n, 4) WXYZ quaternions -> (n, 3, 3). Normalizes defensively."""
    q = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True),
                           1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y - z * w)
    m[:, 0, 2] = 2 * (x * z + y * w)
    m[:, 1, 0] = 2 * (x * y + z * w)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z - x * w)
    m[:, 2, 0] = 2 * (x * z - y * w)
    m[:, 2, 1] = 2 * (y * z + x * w)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def preview_rotation(pitch_rad, roll_rad):
    """The preview's world rotation: R = Ry(roll) @ Rx(pitch).
    Matches _apply_camera_offset_to_rig's axis choice."""
    ang = np.array([pitch_rad], dtype=np.float64)
    rx = _axis_mats(ang, 'X')[0]
    ang = np.array([roll_rad], dtype=np.float64)
    ry = _axis_mats(ang, 'Y')[0]
    return ry @ rx


# ------------------------------------------------------------
# FK over a bone chain (batched across frames)
# ------------------------------------------------------------

def fk_chain_heads(rest_rel_mats, rot_mats, locs, want):
    """Walk one chain, returning each wanted bone's HEAD position per
    frame, in the space of the chain root's PARENT (i.e. Root's
    post-location frame when the chain starts at Hips).

    rest_rel_mats: list of (4, 4) rest matrices, each relative to its
        parent (parent_rest^-1 @ bone_rest). First entry is the chain
        root (e.g. Hips relative to Root).
    rot_mats: list of (n, 3, 3) per-frame local rotations (or None for
        identity), same order.
    locs: list of (n, 3) per-frame local location keys (or None for
        zeros), same order. Only Root/Hips carry these in BVH rigs.
    want: list of chain indices whose head positions to return.

    Returns (n, len(want), 3).
    """
    n = None
    for r in rot_mats:
        if r is not None:
            n = r.shape[0]
            break
    if n is None:
        for l in locs:
            if l is not None:
                n = l.shape[0]
                break
    if n is None:
        raise ValueError("no animated channels in chain")

    eye_rot = np.broadcast_to(np.eye(3), (n, 3, 3))
    cur_rot = None            # (n, 3, 3) accumulated rotation
    cur_pos = None            # (n, 3) accumulated translation
    heads = {}
    for i, rest in enumerate(rest_rel_mats):
        rest_r = rest[:3, :3]
        rest_t = rest[:3, 3]
        loc = locs[i] if locs[i] is not None else None
        rot = rot_mats[i] if rot_mats[i] is not None else eye_rot

        # basis = T(loc) @ R(rot); combined with rest:
        #   pose_rel = rest @ T(loc) @ R(rot)
        # translation contribution: rest_t + rest_r @ loc
        step_t = (np.broadcast_to(rest_t, (n, 3)).copy() if loc is None
                  else rest_t + np.einsum('ij,nj->ni', rest_r, loc))
        step_r = np.einsum('ij,njk->nik', rest_r, rot)

        if cur_rot is None:
            cur_pos = step_t
            cur_rot = step_r
        else:
            cur_pos = cur_pos + np.einsum('nij,nj->ni', cur_rot, step_t)
            cur_rot = np.einsum('nij,njk->nik', cur_rot, step_r)

        if i in want:
            heads[i] = cur_pos
    return np.stack([heads[i] for i in want], axis=1)


# ------------------------------------------------------------
# The per-tick computation
# ------------------------------------------------------------

def ground_delta(world_offsets, pivot_z, rot_world):
    """Vertical compensation for the current preview rotation.

    world_offsets: (n, p, 3) world-frame foot offsets from the pivot
        (cache: M_rot @ q).
    pivot_z: (n,) pivot world height per frame.
    rot_world: (3, 3) current preview rotation (world space).

    Returns the delta to ADD to the rig's height so the lowest foot
    sample returns to its unrotated level. Zero when rot is identity.
    """
    base_min = float((world_offsets[:, :, 2] + pivot_z[:, None]).min())
    rot_z = np.einsum('j,npj->np', rot_world[2], world_offsets)
    rot_min = float((rot_z + pivot_z[:, None]).min())
    return base_min - rot_min
