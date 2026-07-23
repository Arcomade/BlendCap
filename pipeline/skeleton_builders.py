# skeleton_builders.py
# ====================
# Skeletons for the standalone export modes (face / root / hands).
# These are NOT the body skeleton — they're slimmer purpose-built rigs
# emitted when the user wants only the face, only the root motion, or
# only the hands as a separate armature.

import numpy as np

from .bvh_math import compute_world_positions
from .body_skeleton import HARDCODED_NAMES, _ROOT_TAIL_FORWARD


def build_face_only_skeleton():
    """Single-joint Head skeleton for face-only export.
    Face bones attach as children of Head (handled by write_bvh)."""
    parents = np.array([-1], dtype=np.int32)
    offsets = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    return parents, offsets, ["Head"]


def build_root_only_skeleton():
    """Single-joint Root skeleton (for root-motion-only export).

    Returns parents/offsets/names plus an End Site dict so Blender
    draws the bone as a short horizontal stub on the ground plane.
    Standalone Root has no children, so its End Site is honored as
    the tail direction (no post-import processing needed for this
    export mode — the body export's Root has Hips as a child and
    must be fixed up by the addon's BVH importer instead).
    """
    parents = np.array([-1], dtype=np.int32)
    offsets = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    end_sites = {0: _ROOT_TAIL_FORWARD.copy()}
    return parents, offsets, ["Root"], end_sites


def build_hands_only_skeleton(body_parents, body_offsets_cm, body_R0,
                              scale=0.01):
    """Standalone hands armature: Root -> [LeftHand, RightHand] -> fingers.

    Hands are positioned at their body T-pose world locations, recentered so
    the midpoint between wrists sits at the Root (origin). Finger offsets
    preserve the body's rest geometry.

    Returns:
        parents:  np.int32 (n_new,)
        offsets:  np.float64 (n_new, 3)  - in BVH units (scale applied)
        names:    list[str]
        body_idx: list[int]   - body joint index for each new joint;
                                -1 indicates the synthetic Root.
        end_site_offsets: dict {new_idx: np.array(3,)} - end-site offsets
                          for leaf bones (finger tips), matching the body
                          export's convention so Blender doesn't draw
                          default +Y stubs.
    """
    L_HAND = 78
    R_HAND = 42
    # KEPT_ORDER convention: Middle first under each Hand so the Hand bone
    # visually runs along the arm.
    L_FINGERS = [88, 89, 90, 91,   # Middle
                 92, 93, 94, 95,   # Index
                 84, 85, 86, 87,   # Ring
                 79, 80, 81, 82,   # Pinky
                 96, 97, 98, 99]   # Thumb
    R_FINGERS = [52, 53, 54, 55,   # Middle
                 56, 57, 58, 59,   # Index
                 48, 49, 50, 51,   # Ring
                 43, 44, 45, 46,   # Pinky
                 60, 61, 62, 63]   # Thumb

    rest_world_cm = compute_world_positions(body_parents, body_offsets_cm,
                                            body_R0)
    rest_world = rest_world_cm * scale
    midpoint = (rest_world[L_HAND] + rest_world[R_HAND]) / 2.0

    body_idx = [-1]
    parents = [-1]
    offsets = [np.zeros(3, dtype=np.float64)]
    names = ["Root"]
    body_to_new = {}

    def _add(b_idx, name, parent_new_idx, offset):
        body_to_new[b_idx] = len(body_idx)
        body_idx.append(b_idx)
        parents.append(parent_new_idx)
        offsets.append(offset)
        names.append(name)

    # Left side
    _add(L_HAND, "LeftHand", 0, rest_world[L_HAND] - midpoint)
    for f in L_FINGERS:
        body_p = int(body_parents[f])
        new_p = body_to_new.get(body_p, body_to_new[L_HAND])
        offset = rest_world[f] - rest_world[body_p]
        _add(f, HARDCODED_NAMES.get(f, f"L_{f}"), new_p, offset)

    # Right side
    _add(R_HAND, "RightHand", 0, rest_world[R_HAND] - midpoint)
    for f in R_FINGERS:
        body_p = int(body_parents[f])
        new_p = body_to_new.get(body_p, body_to_new[R_HAND])
        offset = rest_world[f] - rest_world[body_p]
        _add(f, HARDCODED_NAMES.get(f, f"R_{f}"), new_p, offset)

    # End-site offsets for leaf bones (finger tips). Matches prune_skeleton's
    # rule: if the body has descendants past this joint, point at the
    # farthest descendant; otherwise continue the bone's own direction by
    # half its length. Without this, Blender draws a default +Y stub and
    # every finger tip sticks straight up.
    n_body = len(body_parents)
    body_kids = [[] for _ in range(n_body)]
    for j in range(n_body):
        p = int(body_parents[j])
        if p >= 0:
            body_kids[p].append(j)

    def _farthest_descendant(j):
        cur = j
        while body_kids[cur]:
            cur = body_kids[cur][0]
        return cur

    n_new = len(parents)
    new_kids = [[] for _ in range(n_new)]
    for i in range(n_new):
        p = int(parents[i])
        if p >= 0:
            new_kids[p].append(i)

    end_site_offsets = {}
    for i in range(n_new):
        if new_kids[i]:
            continue  # not a leaf
        b_i = body_idx[i]
        if b_i < 0:
            continue  # synthetic Root never a leaf here, skip defensively
        if body_kids[b_i]:
            tip = _farthest_descendant(b_i)
            end_site_offsets[i] = rest_world[tip] - rest_world[b_i]
        else:
            par = int(parents[i])
            if par >= 0:
                end_site_offsets[i] = 0.5 * offsets[i]
            else:
                end_site_offsets[i] = np.array([0.0, 0.05, 0.0])

    return (np.array(parents, dtype=np.int32),
            np.array(offsets, dtype=np.float64),
            names,
            body_idx,
            end_site_offsets)


def compute_hands_local_rots(global_rots, body_R0, body_idx, new_parents):
    """Compute local rotations for the hands-only skeleton.

    Root (new index 0) has identity rest and identity rotations. Wrists have
    Root as parent in the new skeleton, so their local rotation is the
    wrist's world rotation aligned to its body rest. Fingers retain the same
    body parent in the new skeleton, so their local rotations match the
    body's local rotations for those joints.
    """
    n_frames = global_rots.shape[0]
    n_new = len(body_idx)
    local = np.empty((n_frames, n_new, 3, 3), dtype=np.float64)
    eye = np.eye(3)
    local[:, 0] = eye  # Root: identity all frames

    for j in range(1, n_new):
        b_j = body_idx[j]
        new_p = int(new_parents[j])
        b_p = body_idx[new_p]
        Rj_T = body_R0[b_j].T
        if b_p < 0:
            # New parent is the synthetic Root (R0 = identity)
            local[:, j] = global_rots[:, b_j] @ Rj_T
        else:
            GpT = global_rots[:, b_p].transpose(0, 2, 1)
            local[:, j] = body_R0[b_p] @ GpT @ global_rots[:, b_j] @ Rj_T

    return local
