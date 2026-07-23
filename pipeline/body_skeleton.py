# body_skeleton.py
# ================
# SAM 3D Body skeleton: loading from the TorchScript model, joint-name
# map, kept-joint pruning, synthetic Root insertion, and finger-rotation
# boost. Anything specific to the SAM body topology lives here.

import sys

import numpy as np
import torch

from .bvh_math import (
    compute_local_rotations,
    compute_world_positions,
    _rotmats_to_quats_xyzw,
    quats_xyzw_to_rotmats,
    _slerp_quats_xyzw,
)


# =====================================================================
#  LOAD SKELETON
# =====================================================================

def load_skeleton(mhr_model_path):
    """Extract joint parents, rest-pose offsets, and prerotations from TorchScript model."""
    print(f"[1/3] Loading skeleton from {mhr_model_path}")
    model = torch.jit.load(mhr_model_path, map_location="cpu")

    parents = offsets = prerotations = None
    for name, buf in model.named_buffers():
        if "skeleton.joint_parents" in name:
            parents = buf.numpy().astype(np.int32).copy()
        elif "skeleton.joint_translation_offsets" in name:
            offsets = buf.numpy().astype(np.float64).copy()
        elif "skeleton.joint_prerotations" in name:
            prerotations = buf.numpy().astype(np.float64).copy()

    if parents is None or offsets is None:
        print("ERROR: skeleton data not found")
        sys.exit(1)

    n = len(parents)
    print(f"       {n} joints")

    if prerotations is not None:
        # Vectorized identity check
        id_mask = (np.allclose(prerotations, [0, 0, 0, 1], atol=1e-8)
                   | np.all(np.isclose(prerotations, [0, 0, 0, -1], atol=1e-8), axis=-1))
        # Per-joint check
        id_count = sum(1 for i in range(n)
                       if np.allclose(prerotations[i], [0, 0, 0, 1])
                       or np.allclose(prerotations[i], [0, 0, 0, -1]))
        print(f"       Prerotations: {n - id_count} non-identity, {id_count} identity")
    else:
        print("       WARNING: No prerotations found, T-pose mode unavailable")

    return parents, offsets, prerotations


# NOTE: a previous version of this file shipped an `apply_rest_pose_preset`
# function that consumed a JSON of saved Blender pose deltas and rewrote
# R0 / offsets_cm to bake a "custom rest pose" into the BVH file. It was
# removed: the preset feature now applies its data inside Blender at
# retarget time (via `source_rest_override`, the same dict the live
# 'Use Current Source Pose as Rest' toggle populates). Going through a
# Blender↔BVH frame conversion was an avoidable source of bugs — see
# `blendcap.operators_rest_pose` for the new save/apply path. Preset
# JSON files saved by the legacy code are NOT compatible with the new
# pipeline; users must re-pose and re-save once.


# =====================================================================
#  JOINT NAMING AND PRUNING
# =====================================================================
#
# Hardcoded joint-name map and kept-joint list, verified by the user
# against skeleton_naming.xlsx. Side convention matches the spreadsheet:
# +X = character's Left, -X = character's Right (NOT the topology code's
# original heuristic, which had L/R reversed).
#
# Anything not in HARDCODED_NAMES is treated as non-skeletal detail and
# gets pruned by prune_skeleton(); its descendants are reparented to the
# closest kept ancestor and local rotations are recomputed so kept joints
# retain their original world motion (FK-safe, modulo the assumption that
# pruned intermediates have identity rotation at every frame, which holds
# for SAM 3D Body's zero-offset markers and linear-interpolation detail
# joints).

HARDCODED_NAMES = {
    # [0] is SAM's world-origin joint; dropped from the rig entirely so
    # Blender doesn't draw a 92cm octahedron from the floor up to the
    # pelvis. Hips [1] is the spine-chain root after pruning; a synthetic
    # "Root" bone is then inserted above it (see _insert_synthetic_root).
    # Hips becomes a 6-channel non-root JOINT whose Y position channel
    # carries vertical body delta (squat depth / jump / gait bob), while
    # Root carries X/Z locomotion. Hips's rest world position
    # (0, 0.924, 0) becomes its offset relative to Root.
    1:   "Hips",
    34:  "Spine",
    35:  "Spine1",
    36:  "Spine2",
    37:  "Spine3",
    110: "Neck",
    113: "Head",                # leaf; HeadTop [126] is dropped, becomes End Site

    # SAM's skull-area joints [114-125] are intentionally dropped from the
    # rig. [122]/[124] (eye joints) are still used internally as anchor
    # points for MediaPipe face alignment, but they aren't written to the
    # BVH. [117]-[120] (jaw/tongue) and [114]-[116] (nose/forehead) are
    # not exposed either — face motion is driven entirely by MediaPipe
    # face bones attached as children of "Head" in write_bvh.
    # SAM [126] (HeadTop) is also dropped — its rotation is always identity
    # and the End Site that prune_skeleton emits gives the same visual result.

    # Left leg (+X side). SAM has a zero-offset marker [4] at the ankle
    # and a heel-pad marker [6] sitting ~3.4cm below it; both are
    # absorbed into [5] so LeftFoot is the single foot bone running
    # ankle -> ball-of-foot. SAM [8] (toe tip) is dropped — its rotation
    # carries a few degrees of SMPL-X noise but no real articulation, so
    # we promote it to an End Site for consistency with the finger leaves.
    2:   "LeftUpLeg",
    3:   "LeftLeg",
    5:   "LeftFoot",       # absorbs [4] (ankle zero-offset marker) and [6] (heel pad)
    7:   "LeftToe",

    # Right leg (-X side)
    18:  "RightUpLeg",
    19:  "RightLeg",
    21:  "RightFoot",      # absorbs [20] (ankle zero-offset marker) and [22] (heel pad)
    23:  "RightToe",

    # Right arm and hand ([38]-[42] are on -X side).
    # Pinky chain starts at SAM joint [44] because [43] is a non-
    # articulating root marker (always identity rotation); dropping it
    # reparents Pinky1 directly to RightHand. Same on the left.
    # Middle/Index/Ring stop at *3 — the *4 slots in those chains are
    # tip markers with no rotation animation. Thumb stops at *3 too:
    # SAM exposes 4 thumb joints but anatomy + MANO only define 3
    # articulating thumb joints, so [63]/[99] are dropped as the
    # redundant tip slot.
    38:  "RightShoulder",
    39:  "RightArm",
    40:  "RightForeArm",
    42:  "RightHand",      # absorbs [41] (zero-offset marker)
    44:  "RightHandPinky1",
    45:  "RightHandPinky2",
    46:  "RightHandPinky3",
    48:  "RightHandRing1",
    49:  "RightHandRing2",
    50:  "RightHandRing3",
    52:  "RightHandMiddle1",
    53:  "RightHandMiddle2",
    54:  "RightHandMiddle3",
    56:  "RightHandIndex1",
    57:  "RightHandIndex2",
    58:  "RightHandIndex3",
    60:  "RightHandThumb1",
    61:  "RightHandThumb2",
    62:  "RightHandThumb3",

    # Left arm and hand ([74]-[78] are on +X side)
    74:  "LeftShoulder",
    75:  "LeftArm",
    76:  "LeftForeArm",
    78:  "LeftHand",       # absorbs [77]
    80:  "LeftHandPinky1",
    81:  "LeftHandPinky2",
    82:  "LeftHandPinky3",
    84:  "LeftHandRing1",
    85:  "LeftHandRing2",
    86:  "LeftHandRing3",
    88:  "LeftHandMiddle1",
    89:  "LeftHandMiddle2",
    90:  "LeftHandMiddle3",
    92:  "LeftHandIndex1",
    93:  "LeftHandIndex2",
    94:  "LeftHandIndex3",
    96:  "LeftHandThumb1",
    97:  "LeftHandThumb2",
    98:  "LeftHandThumb3",
}


# Explicit kept-joint order. This controls the order children appear
# under each parent in the output BVH, which in turn controls which
# child Blender treats as the parent's "main bone direction".
#
# Important orderings:
#   Hips     -> Spine, LeftUpLeg, RightUpLeg          (Spine first -> points up)
#   Spine3   -> Neck, RightShoulder, LeftShoulder     (Neck first  -> points up)
#   Head     -> HeadTop                               (only child; trivially main bone)
#   Hand     -> Middle, Index, Ring, Pinky, Thumb     (Middle first -> along arm)
KEPT_ORDER = [
    # Spine chain (Hips -> ... -> Head). KEPT_ORDER[0] must be the
    # spine-chain root: prune_skeleton enforces parent < 0 for it, and
    # a synthetic "Root" bone is inserted above it post-prune by
    # _insert_synthetic_root. Everything else must follow its parent
    # in this list (parents must appear before children).
    1,                                  # Hips ([0] is dropped; Hips becomes a 6-channel child of synthetic Root)
    34, 35, 36, 37,                     # Spine, Spine1, Spine2, Spine3
    110,                                # Neck
    113,                                # Head (leaf; HeadTop pruned, becomes End Site)
    # Hips children (come after Spine chain so Spine is "main bone")
    2, 3, 5, 7,                         # Left leg: UpLeg, Leg, Foot, Toe (ToeEnd pruned, becomes End Site)
    18, 19, 21, 23,                     # Right leg: UpLeg, Leg, Foot, Toe (ToeEnd pruned, becomes End Site)
    # Spine3 children (come after Neck so Neck is "main bone")
    # Right side
    38, 39, 40, 42,                     # RightShoulder, RightArm, RightForeArm, RightHand
    # RightHand children (Middle first so Hand bone runs along arm)
    52, 53, 54,                         # Middle1-3
    56, 57, 58,                         # Index1-3
    48, 49, 50,                         # Ring1-3
    44, 45, 46,                         # Pinky1-3 (SAM [43] is a non-articulating root marker; pruned)
    60, 61, 62,                         # Thumb1-3 (Thumb4 [63] is the redundant tip slot; pruned)
    # Left side
    74, 75, 76, 78,                     # LeftShoulder, LeftArm, LeftForeArm, LeftHand
    88, 89, 90,                         # Middle1-3
    92, 93, 94,                         # Index1-3
    84, 85, 86,                         # Ring1-3
    80, 81, 82,                         # Pinky1-3 (SAM [79] is a non-articulating root marker; pruned)
    96, 97, 98,                         # Thumb1-3 (Thumb4 [99] is the redundant tip slot; pruned)
]


def assign_names(parents, rest_pos):
    """Return a full-length name list using HARDCODED_NAMES; J### fallback.

    Names every joint in the ORIGINAL 127-joint SAM skeleton. Unnamed
    joints get J### placeholders so that downstream code can still
    reference them by index before prune_skeleton() drops them.
    """
    n = len(parents)
    names = [HARDCODED_NAMES.get(i, f"J{i:03d}") for i in range(n)]
    named = sum(1 for nm in names if not nm.startswith("J"))
    print(f"       Named {named}/{n} joints (from HARDCODED_NAMES)")
    return names


def prune_skeleton(parents, rest_pos, global_rots, R0, kept_order=None):
    """Prune non-skeletal joints, preserving kept joints' world motion.

    kept_order: optional list of joint indices to keep (overrides KEPT_ORDER).
                Use to exclude finger joints when --no-hands is set.

    Reparents each kept joint to its closest kept ancestor in the original
    tree, then recomputes local rotations from global_rots using the new
    parent relationships. For kept joints whose original parent is already
    kept, this is a no-op (same local rotation as before). For kept joints
    whose parent chain contains pruned intermediates, the new local
    rotation is measured relative to the kept ancestor, which absorbs the
    intermediate chain's rest-pose rotation into the offset.

    Returns (new_parents, new_offsets, new_local_rots, new_names) where
    all arrays are reindexed to len(KEPT_ORDER) entries in KEPT_ORDER
    order.
    """
    n = len(parents)
    ko = kept_order if kept_order is not None else KEPT_ORDER
    kept = set(ko)

    # Sanity checks
    if len(ko) != len(set(ko)):
        raise ValueError("kept_order contains duplicates")
    # Only validate against HARDCODED_NAMES when using the default order
    if kept_order is None and set(ko) != set(HARDCODED_NAMES.keys()):
        missing = set(HARDCODED_NAMES) - set(ko)
        extra = set(ko) - set(HARDCODED_NAMES)
        raise ValueError(
            f"KEPT_ORDER / HARDCODED_NAMES mismatch: "
            f"missing={sorted(missing)} extra={sorted(extra)}")
    for j in ko:
        if not (0 <= j < n):
            raise ValueError(f"kept_order index {j} out of range for {n} joints")

    # Sanity check: every kept joint must have all of its ancestors in
    # kept_order appearing BEFORE it (so parents exist when we reindex).
    kept_before = set()
    for k in ko:
        kept_before.add(k)

    # Closest kept ancestor in the original tree. May be -1 (root of the
    # new tree) even if the joint's original parent was the SAM root [0],
    # when [0] is not in KEPT_ORDER.
    kept_ancestor = [-1] * n
    for j in range(n):
        p = parents[j]
        while p >= 0 and p not in kept:
            p = parents[p]
        kept_ancestor[j] = p

    # Exactly one kept joint becomes the new root (kept_ancestor == -1)
    new_root_candidates = [k for k in ko if kept_ancestor[k] < 0]
    if len(new_root_candidates) != 1:
        raise ValueError(
            f"Expected exactly one root in kept_order, got "
            f"{len(new_root_candidates)}: {new_root_candidates}")
    new_root_old_idx = new_root_candidates[0]
    if ko[0] != new_root_old_idx:
        raise ValueError(
            f"kept_order[0] must be the new root ({new_root_old_idx}), "
            f"got {ko[0]}")

    # Modified 127-length parents array: each kept joint points to its
    # kept ancestor (skipping pruned intermediates). Pruned entries still
    # point to their original parent, but we will discard their rows.
    new_parents_old = np.array(parents).copy()
    for K_old in ko:
        new_parents_old[K_old] = kept_ancestor[K_old]

    # Recompute local rotations in the pruned tree. compute_local_rotations
    # only looks at parents[j] for each joint, so passing new_parents_old
    # gives kept joints their new local rotations directly. For the new
    # root (parent < 0), compute_local_rotations returns
    # L[f, root] = G[f, root] @ R0[root]^T, which is correct for a BVH
    # root joint regardless of whether the old SAM root [0] was kept.
    pruned_local_rots = compute_local_rotations(global_rots, new_parents_old, R0)

    # Reindex to new kept_order positions
    old_to_new = {old: new for new, old in enumerate(ko)}
    K = len(ko)

    new_parents = np.empty(K, dtype=np.int32)
    for new_idx, old_idx in enumerate(ko):
        par_old = kept_ancestor[old_idx]
        new_parents[new_idx] = old_to_new[par_old] if par_old >= 0 else -1

    # New BVH offsets: world-space deltas from the new parent, same
    # convention as the un-pruned code (cumulative sum from root = world
    # rest position). For the root joint, use its world rest position so
    # the cumulative sum still reconstructs world-space positions.
    new_offsets = np.zeros((K, 3), dtype=np.float64)
    for new_idx, old_idx in enumerate(ko):
        par_new = new_parents[new_idx]
        if par_new < 0:
            new_offsets[new_idx] = rest_pos[old_idx]
        else:
            par_old = ko[par_new]
            new_offsets[new_idx] = rest_pos[old_idx] - rest_pos[par_old]

    new_local_rots = pruned_local_rots[:, ko, :, :]
    new_names = [HARDCODED_NAMES[i] for i in ko]

    # Compute End Site offsets for leaves in the new tree. A BVH leaf
    # with OFFSET 0 0 0 makes Blender default to a tiny +Y stub that
    # doesn't align with the joint chain, so every fingertip / toe tip /
    # HeadTop / eye looks "pulled up". Fix: for each kept leaf K whose
    # original tree had children (all now pruned), use the world-space
    # offset to K's FIRST pruned descendant-leaf as the End Site. This
    # continues the bone along its natural direction. For truly-leaf
    # joints (no children in the original tree either), fall back to a
    # small extension of K's own bone direction.
    kids_orig = [[] for _ in range(n)]
    for i in range(n):
        p = parents[i]
        if p >= 0:
            kids_orig[p].append(i)

    def _farthest_descendant(j):
        """Walk down the original tree to the deepest reachable descendant.
        Used to pick an end-site anchor for a leaf whose children were all
        pruned. Returns the old index of the leaf of the subtree."""
        cur = j
        while kids_orig[cur]:
            cur = kids_orig[cur][0]
        return cur

    new_kids = [[] for _ in range(K)]
    for i in range(K):
        p = new_parents[i]
        if p >= 0:
            new_kids[p].append(i)

    end_site_offsets = {}
    for new_idx in range(K):
        if new_kids[new_idx]:
            continue  # not a leaf, no end site
        old_idx = ko[new_idx]
        if kids_orig[old_idx]:
            tip_old = _farthest_descendant(old_idx)
            end_site_offsets[new_idx] = rest_pos[tip_old] - rest_pos[old_idx]
        else:
            # True leaf in both trees (only HeadTop in the default rig).
            # Continue the bone's own direction by half the parent->child
            # offset, so Blender draws a short stub aligned with the bone.
            par_new = new_parents[new_idx]
            if par_new >= 0:
                bone_dir = new_offsets[new_idx]
                end_site_offsets[new_idx] = 0.5 * bone_dir
            else:
                end_site_offsets[new_idx] = np.array([0.0, 0.05, 0.0])

    # Explicit End Site overrides for kept leaves whose default
    # _farthest_descendant walk overshoots the natural bone end. This
    # happens when the SAM chain past the kept leaf is longer than one
    # step — e.g. RightHandThumb3 [62] has children [63, 64] in SAM, so
    # the default walk lands at [64] and the Thumb3 bone ends up spanning
    # the entire Thumb4-plus-tip extent. The override re-anchors the End
    # Site to the immediate next named joint that we pruned, restoring
    # the bone's pre-prune size.
    #
    # Each entry: parent_old_idx -> tip_old_idx
    _END_SITE_OVERRIDES = {
        62: 63,    # RightHandThumb3 -> position where Thumb4 [63] was
        98: 99,    # LeftHandThumb3  -> position where Thumb4 [99] was
        113: 126,  # Head            -> position where HeadTop [126] was
    }
    for new_idx in range(K):
        old_idx = ko[new_idx]
        if old_idx in _END_SITE_OVERRIDES and new_idx in end_site_offsets:
            tip_old = _END_SITE_OVERRIDES[old_idx]
            end_site_offsets[new_idx] = rest_pos[tip_old] - rest_pos[old_idx]

    # SAM's rest pose puts the toe tip ~3cm below the ball of foot, which
    # makes the BVH toe bone slope downward at rest. That doesn't match
    # Rigify's metarig (and most other mocap conventions) where the toe
    # bone runs roughly parallel to the floor. Override the Y component of
    # the End Site offset for LeftToe / RightToe so the bone draws flat.
    # Rest-pose-only; per-frame rotation math is unaffected.
    _FLAT_TOE_NAMES = ("LeftToe", "RightToe")
    for new_idx in range(K):
        if new_names[new_idx] in _FLAT_TOE_NAMES and new_idx in end_site_offsets:
            v = end_site_offsets[new_idx]
            end_site_offsets[new_idx] = np.array(
                [v[0], 0.0, v[2]], dtype=np.float64)

    print(f"       Pruned {n - K} non-skeletal joints, {K} kept")
    return new_parents, new_offsets, new_local_rots, new_names, end_site_offsets


# Tail length used to draw a leaf Root bone horizontally on the floor
# in the standalone root-only export (build_root_only_skeleton). With
# axis_forward='-Z' / axis_up='Y' (the import flags the addon uses),
# -Z is forward in source space. 0.30 m matches typical mocap-rig
# proportions. This End Site is honored by Blender's BVH importer
# because Root has no children in that mode, so the bone direction is
# set BEFORE position channels are baked and pose-bone .location curves
# end up in the correct frame.
#
# The body export's Root has Hips as a child, so its tail is derived
# from Hips's offset (vertical, ~92 cm) and this End Site is ignored.
# Do NOT try to override Root's edit_bone.tail after import in that
# case: the importer has already baked .location curves in the
# vertical-Y bone-local frame, and rotating the bone post-hoc rotates
# those curves out of the ground plane (locomotion lands on Blender Z
# instead of X/Y). See blendcap/helpers.py for the history.
_ROOT_TAIL_FORWARD = np.array([0.0, 0.0, -0.30], dtype=np.float64)


def insert_synthetic_root(parents, bvh_offsets, local_rots, names,
                          end_site_offsets):
    """Insert a single synthetic "Root" bone at world origin above the
    spine-chain root.

    Resulting BVH structure::

        ROOT Root              CHANNELS 6 (LOC X/Y/Z + ROT identity)
            OFFSET 0 0 0
            JOINT Hips         CHANNELS 6 (LOC absolute parent-rel + ROT)
                OFFSET 0 0.924 0
                ...

    Translation channel split (Option 2 — canonical pro-mocap layout):
      Root.LOC.X / Z    locomotion (gated by --root-motion); Y stays 0
      Hips.LOC          Hips's absolute position relative to Root, with
                        rest = OFFSET = (0, hip_height, 0). The caller
                        is responsible for writing rest + Y-delta.
                        X / Z stay at the rest values (0).
      Root.ROT          identity (rotation lives on Hips and below)

    Visual rendering:
      Blender's BVH importer derives a non-leaf joint's tail from its
      children, so a synthetic Root with Hips as its only child renders
      as a tall vertical octahedron (~92 cm) pointing at the pelvis.
      The addon's _import_bvh post-process shrinks Root's tail to a
      small vertical stub (~10 cm) by moving it along the SAME axis,
      not rotating it. This is critical: rotating the bone direction
      post-import rotates the bone-local frame, and the BVH importer
      has already baked Root's .location curves in the original frame
      — so the curves are re-interpreted in the new frame and
      locomotion lands on the wrong axes. A matrix-based curve-
      compensation attempt also produced bad motion in practice.
      Shrink-only along the existing axis preserves the local frame.
      Logical orientation must stay vertical-Y in BVH source space.
    """
    n_frames = local_rots.shape[0]
    K = len(parents)
    new_K = K + 1

    # Parents: prepend [Root=-1]. The old spine-chain root (parent -1)
    # becomes a child of Root (parent=0); other parent indices bump +1.
    new_parents = np.empty(new_K, dtype=parents.dtype)
    new_parents[0] = -1
    new_parents[1:] = np.where(parents < 0, 0, parents + 1)

    # Offsets: Root at world origin, all existing offsets preserved
    # (the old root's offset already encoded its world rest position).
    new_offsets = np.empty((new_K, 3), dtype=bvh_offsets.dtype)
    new_offsets[0] = 0.0
    new_offsets[1:] = bvh_offsets

    # Local rotations: Root is identity every frame.
    eye = np.eye(3, dtype=local_rots.dtype)
    root_rots = np.broadcast_to(eye, (n_frames, 1, 3, 3)).copy()
    new_local_rots = np.concatenate([root_rots, local_rots], axis=1)

    new_names = ["Root"] + list(names)

    # End sites just bump by +1; Root has Hips as a child so it's not a
    # leaf and gets no End Site of its own.
    new_end_sites = {k + 1: v for k, v in end_site_offsets.items()}

    return new_parents, new_offsets, new_local_rots, new_names, new_end_sites


# =====================================================================
#  FINGER ROTATION RECONSTRUCTION (Path B)
# =====================================================================

# All finger joint indices in SAM's 127-joint skeleton.
# Right hand [42]=wrist, [43-47]=Pinky, [48-51]=Ring, [52-55]=Middle,
#            [56-59]=Index, [60-64]=Thumb
# Left hand  [78]=wrist, [79-83]=Pinky, [84-87]=Ring, [88-91]=Middle,
#            [92-95]=Index, [96-100]=Thumb
FINGER_JOINTS = list(range(43, 65)) + list(range(79, 101))

# Per-hand finger topology (SAM 127-joint skeleton): hand root + the
# per-finger chain [knuckle(MCP), mid(PIP), distal(DIP), tip]. The first
# three are the articulated joints that carry a flexion hinge; the tip
# is a leaf marker. This is the SINGLE source of the finger index tables
# — pipeline/hand_solver.py imports it as _J3D so the boost and the
# solver can never disagree about which joint is which.
FINGER_TOPOLOGY = {
    "right": {"hand": 42,
              "fingers": {"thumb": [60, 61, 62, 63],
                          "index": [56, 57, 58, 59],
                          "middle": [52, 53, 54, 55],
                          "ring": [48, 49, 50, 51],
                          "pinky": [44, 45, 46, 47]}},
    "left": {"hand": 78,
             "fingers": {"thumb": [96, 97, 98, 99],
                         "index": [92, 93, 94, 95],
                         "middle": [88, 89, 90, 91],
                         "ring": [84, 85, 86, 87],
                         "pinky": [80, 81, 82, 83]}},
}

# Off-axis (non-flexion) amount kept when boosting: the flexion twist is
# scaled by the user's Finger Curl Strength, the sideways/backward swing
# is scaled by this instead so amplifying curl doesn't also amplify the
# tracker's off-axis noise into hyperextended, splayed "octopus" fingers.
# 1.0 = leave off-axis untouched; lower = actively calm it. Tunable.
_BOOST_SWING_SCALE = 0.6

# How far a finger may bend BACKWARD past straight after boosting, in
# degrees. Amplifying a finger's extension (when frame 0 wasn't fully
# open) would otherwise drive it into hyperextension; curl is never
# clamped, only reverse-bend. ~10 deg is natural slack.
_BOOST_HYPEREXT_LIMIT = np.radians(10.0)


def _quat_mul_xyzw(a, b):
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz], axis=1)


def _quat_conj_xyzw(q):
    out = q.copy()
    out[:, :3] *= -1.0
    return out


def _finger_hinge_axes(parents, offsets_cm, R0, n_joints):
    """Flexion hinge axis (unit, in the delta-rotation frame) for every
    articulated finger joint. For a pure flexion the boost's delta
    rotation L_delta = Rot(h_world, theta) with h_world the world-rest
    hinge axis, so the axis to decompose about is simply
    cross(bone, palm_normal) in world-rest coordinates (the thumb
    opposes, so it uses the direction across the palm instead of the
    palm normal). Sign is irrelevant — swing-twist about +/-h is the
    same decomposition. Returns {joint_index: axis}."""
    rest = compute_world_positions(parents, offsets_cm, R0)
    axes = {}
    for side, cfg in FINGER_TOPOLOGY.items():
        fings = cfg["fingers"]
        hand = cfg["hand"]
        knuck = {nm: ch[0] for nm, ch in fings.items()}
        if max(knuck["index"], knuck["pinky"], knuck["middle"],
               hand) >= n_joints:
            continue
        row = rest[knuck["index"]] - rest[knuck["pinky"]]
        midv = rest[knuck["middle"]] - rest[hand]
        n = np.cross(row, midv)
        n = n / max(np.linalg.norm(n), 1e-9)
        for nm, chain in fings.items():
            for k in range(3):                 # knuckle, mid, distal
                j, child = chain[k], chain[k + 1]
                if child >= n_joints:
                    continue
                bone = rest[child] - rest[j]
                bone = bone / max(np.linalg.norm(bone), 1e-9)
                if nm == "thumb":
                    d = rest[knuck["pinky"]] - rest[knuck["thumb"]]
                    d = d - bone * float(d @ bone)
                    ref = d / max(np.linalg.norm(d), 1e-9)
                else:
                    ref = n
                axis = np.cross(bone, ref)
                na = np.linalg.norm(axis)
                if na > 1e-9:
                    axes[j] = axis / na
    return axes


def _twist_angle_about(L, axis):
    """Signed twist angle (rad) of rotations L (N,3,3) about `axis`."""
    q = _rotmats_to_quats_xyzw(L)
    d = q[:, :3] @ axis
    return 2.0 * np.arctan2(d, q[:, 3])


def _swing_twist_boost(L_all, axis, twist_scale, swing_scale, max_twist,
                       hyperext_limit):
    """Boost the FLEXION of a finger joint's local rotation.

    L_all: (N,3,3) ABSOLUTE local rotations (rest = identity). Scales
    the twist about `axis` of the delta-from-frame-0 by twist_scale and
    the off-axis swing by swing_scale, then clamps the resulting
    ABSOLUTE flexion so the joint can't bend backward past straight by
    more than hyperext_limit (curl is never clamped). Returns the new
    (N,3,3) absolute local rotations."""
    L0 = L_all[0].copy()
    L_delta = np.einsum("nij,jk->nik", L_all, L0.T)

    q = _rotmats_to_quats_xyzw(L_delta)          # (N,4) xyzw
    v, w = q[:, :3], q[:, 3]
    d = v @ axis
    proj = d[:, None] * axis[None, :]
    tw = np.concatenate([proj, w[:, None]], axis=1)
    q_tw = tw / np.clip(np.linalg.norm(tw, axis=1, keepdims=True), 1e-9, None)
    theta = 2.0 * np.arctan2(d, w)
    theta_new = np.clip(theta * twist_scale, -max_twist, max_twist)
    half = 0.5 * theta_new
    q_tw_new = np.concatenate(
        [np.sin(half)[:, None] * axis[None, :], np.cos(half)[:, None]],
        axis=1)
    q_sw = _quat_mul_xyzw(q, _quat_conj_xyzw(q_tw))
    ident = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(q), 1))
    q_sw_d = (q_sw if swing_scale >= 0.999
              else _slerp_quats_xyzw(ident, q_sw, swing_scale))
    q_new = _quat_mul_xyzw(q_sw_d, q_tw_new)
    L_new = np.einsum("nij,jk->nik", quats_xyzw_to_rotmats(q_new), L0)

    # Hyperextension clamp on ABSOLUTE flexion. Curl direction is where
    # the joint reaches its DEEPEST flexion (the fist), not the average
    # — a knuckle that spends time extended has a small mean but still
    # curls hard at its peak. Only clamp joints that actually move (a
    # near-static joint can't hyperextend).
    theta_abs = _twist_angle_about(L_new, axis)
    peak = theta_abs[np.argmax(np.abs(theta_abs))]
    if abs(peak) > np.radians(15.0):
        curl = 1.0 if peak > 0 else -1.0
        over = theta_abs * curl < -hyperext_limit
        if np.any(over):
            corr = (-hyperext_limit * curl) - theta_abs[over]
            ch = 0.5 * corr
            q_corr = np.concatenate(
                [np.sin(ch)[:, None] * axis[None, :], np.cos(ch)[:, None]],
                axis=1)
            L_corr = quats_xyzw_to_rotmats(q_corr)
            L_new[over] = np.einsum("nij,njk->nik", L_corr, L_new[over])
    return L_new


def _boost_whole_angle(L_all, boost, max_angle):
    """Legacy whole-angle boost (axis preserved) for finger joints
    with no defined hinge — metacarpals and tip markers, whose motion
    is negligible. Kept so those joints behave as before. Takes and
    returns ABSOLUTE local rotations (delta-from-frame-0 boosted, then
    re-composed on the frame-0 baseline)."""
    L0 = L_all[0].copy()
    L_delta = np.einsum("nij,jk->nik", L_all, L0.T)
    out = np.empty_like(L_delta)
    for f in range(len(L_delta)):
        D = L_delta[f]
        angle = np.arccos(np.clip((np.trace(D) - 1.0) / 2.0, -1.0, 1.0))
        if angle < 1e-6:
            out[f] = np.eye(3)
            continue
        r = np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0],
                      D[1, 0] - D[0, 1]])
        rn = np.linalg.norm(r)
        if rn < 1e-10:
            out[f] = np.eye(3)
            continue
        ax = r / rn
        na = min(angle * boost, max_angle)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]],
                      [-ax[1], ax[0], 0]])
        out[f] = np.eye(3) + np.sin(na) * K + (1.0 - np.cos(na)) * (K @ K)
    return np.einsum("nij,jk->nik", out, L0)


def boost_finger_rotations(global_rots, parents, R0, boost, offsets_cm):
    """Amplify finger curl in global_rots by `boost`.

    SAM's MANO-based hand model uses PCA compression which dampens finger
    rotations to ~15 deg when a closed fist should be ~80. This scales
    each finger joint's LOCAL rotation (delta from frame 0, so the
    resting pose is preserved) — but only the FLEXION component, about
    each joint's anatomical hinge axis. The off-axis sideways/backward
    part is scaled by _BOOST_SWING_SCALE instead, so cranking the curl
    doesn't also amplify the tracker's off-axis noise into hyperextended,
    splayed fingers (the earlier whole-angle boost's "octopus" failure).
    Metacarpals and tip markers have no hinge and keep the legacy
    whole-angle boost (their motion is negligible).

    Only touches joints in FINGER_JOINTS; body joints are untouched.
    Globals are rebuilt parent-first so each knuckle inherits the
    boosted parent (= cumulative curl along the finger chain).
    """
    n_frames = global_rots.shape[0]
    n_joints = global_rots.shape[1]
    max_angle = np.radians(150)  # hard clamp to prevent over-curl

    hinge_axes = _finger_hinge_axes(parents, offsets_cm, R0, n_joints)

    # Step 1: LOCAL rotations for all finger joints,
    # L[f,j] = R0[p] @ G[f,p]^T @ G[f,j] @ R0[j]^T  (vectorized over f).
    finger_local = {}
    for j in sorted(FINGER_JOINTS):
        if j >= n_joints:
            continue
        p = parents[j]
        if p < 0:
            continue
        L_all = np.einsum("ij,njk,nkl,lm->nim",
                          R0[p], np.transpose(global_rots[:, p], (0, 2, 1)),
                          global_rots[:, j],
                          R0[j].T)
        finger_local[j] = L_all

    # Step 2: boost the delta-from-frame-0, swing-twist where a hinge is
    # defined (flexion amplified, off-axis calmed, hyperextension
    # clamped), legacy whole-angle otherwise.
    for j, L_all in finger_local.items():
        if j in hinge_axes:
            finger_local[j] = _swing_twist_boost(
                L_all, hinge_axes[j], boost, _BOOST_SWING_SCALE, max_angle,
                _BOOST_HYPEREXT_LIMIT)
        else:
            finger_local[j] = _boost_whole_angle(L_all, boost, max_angle)

    # Step 3: reconstruct globals, parent-first (SAM indices: parent <
    # child), G[f,j] = G[f,p] @ R0[p]^T @ L_boosted[f,j] @ R0[j].
    for j in sorted(finger_local.keys()):
        p = parents[j]
        global_rots[:, j] = np.einsum(
            "nij,jk,nkl,lm->nim",
            global_rots[:, p], R0[p].T, finger_local[j], R0[j])

    print(f"       Finger boost: amplified {len(finger_local)} joints "
          f"by {boost:.1f}x flexion "
          f"(swing x{_BOOST_SWING_SCALE:.1f}, {len(hinge_axes)} hinged)")
