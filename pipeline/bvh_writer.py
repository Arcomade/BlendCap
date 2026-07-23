# bvh_writer.py
# =============
# Standard Y-up BVH writer with optional face-bone attachment under Head.
# Vectorizes the per-frame Euler conversion and frame-formatting pass.

import os

import numpy as np

from .bvh_math import rotmats_to_euler_zyx_deg


def write_bvh(filepath, parents, bvh_offsets, names, local_rots,
              root_pos, fps, face_bone_data=None, head_joint_name="Head",
              end_site_offsets=None, extra_pos=None):
    """Write standard Y-up BVH file, optionally with face bones.

    end_site_offsets: optional dict {joint_idx: (dx, dy, dz)} giving the
    End Site offset for leaf joints. Leaves not in the dict default to
    (0, 0, 0), which Blender draws as a default +Y stub — usually wrong.

    extra_pos: optional dict {joint_idx: (n_frames, 3) array} of bones
    that should carry POSITION channels in addition to rotation, even
    though they are not the BVH ROOT. The body export uses this to give
    Hips a Y-position channel for vertical body delta (squat depth /
    jump / gait bob) while the synthetic Root carries X/Z locomotion
    only. Such joints are written as JOINT with 6 channels (3 LOC + 3
    ROT) instead of the default 3-channel (ROT only).
    """
    if end_site_offsets is None:
        end_site_offsets = {}
    if extra_pos is None:
        extra_pos = {}
    n_joints = len(parents)
    n_frames = local_rots.shape[0]
    frame_time = 1.0 / fps

    kids = [[] for _ in range(n_joints)]
    for i in range(n_joints):
        if parents[i] >= 0:
            kids[parents[i]].append(i)

    leaves = set(i for i in range(n_joints) if not kids[i])

    head_idx = -1
    if face_bone_data is not None:
        for i, name in enumerate(names):
            if name == head_joint_name:
                head_idx = i
                break
        if head_idx < 0:
            print(f"WARNING: Head joint '{head_joint_name}' not found, "
                  f"face bones will not be attached")
            face_bone_data = None

    if face_bone_data is not None and head_idx in leaves:
        leaves.discard(head_idx)

    lines = ["HIERARCHY"]
    channel_order = []  # list of (joint_idx_or_bone_name, n_channels)

    def write_joint(j, depth):
        pad = "  " * depth
        off = bvh_offsets[j]
        tag = "ROOT" if depth == 0 else "JOINT"
        lines.append(f"{pad}{tag} {names[j]}")
        lines.append(f"{pad}{{")
        lines.append(f"{pad}  OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}")

        if depth == 0 or j in extra_pos:
            lines.append(f"{pad}  CHANNELS 6 Xposition Yposition Zposition "
                         "Zrotation Yrotation Xrotation")
            channel_order.append((j, 6))
        else:
            lines.append(f"{pad}  CHANNELS 3 Zrotation Yrotation Xrotation")
            channel_order.append((j, 3))

        has_children = bool(kids[j]) or (face_bone_data is not None and j == head_idx)

        if not has_children:
            es = end_site_offsets.get(j, (0.0, 0.0, 0.0))
            lines.append(f"{pad}  End Site")
            lines.append(f"{pad}  {{")
            lines.append(f"{pad}    OFFSET {es[0]:.6f} {es[1]:.6f} {es[2]:.6f}")
            lines.append(f"{pad}  }}")
        else:
            for child in kids[j]:
                write_joint(child, depth + 1)
            if face_bone_data is not None and j == head_idx:
                write_face_bones("Head", depth + 1)

        lines.append(f"{pad}}}")

    def write_face_bones(parent_name, depth):
        """Recursively write face bone hierarchy under parent_name."""
        children_map = face_bone_data["children"]
        for bone_name in children_map.get(parent_name, []):
            pad = "  " * depth
            off = face_bone_data["offsets"][bone_name]
            lines.append(f"{pad}JOINT {bone_name}")
            lines.append(f"{pad}{{")
            lines.append(f"{pad}  OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}")
            lines.append(f"{pad}  CHANNELS 6 Xposition Yposition Zposition "
                         "Zrotation Yrotation Xrotation")
            channel_order.append((bone_name, 6))

            bone_children = children_map.get(bone_name, [])
            if bone_children:
                write_face_bones(bone_name, depth + 1)
            else:
                es = face_bone_data.get("end_sites", {}).get(
                    bone_name, np.array([0.0, -0.005, 0.0]))
                lines.append(f"{pad}  End Site")
                lines.append(f"{pad}  {{")
                lines.append(f"{pad}    OFFSET {es[0]:.6f} {es[1]:.6f} {es[2]:.6f}")
                lines.append(f"{pad}  }}")

            lines.append(f"{pad}}}")

    root_idx = int(np.where(np.array(parents) == -1)[0][0])
    write_joint(root_idx, 0)

    lines.append("MOTION")
    lines.append(f"Frames: {n_frames}")
    lines.append(f"Frame Time: {frame_time:.6f}")

    # Batch Euler conversion: all frames × all joints at once
    print("       Converting to Euler angles...")
    flat_rots = local_rots.reshape(-1, 3, 3)
    flat_euler = rotmats_to_euler_zyx_deg(flat_rots)
    euler = flat_euler.reshape(n_frames, n_joints, 3)

    f0_max = np.abs(euler[0]).max()
    print(f"       Frame 0 max Euler: {f0_max:.2f} deg (should be ~0 for frame0 rest)")

    # Pre-build face frame data as contiguous array for fast access
    face_frames_lookup = face_bone_data["frames"] if face_bone_data else {}

    # Build full frame data array for vectorized output
    print("       Writing frames...")

    # Count total channels
    total_ch = sum(nch for _, nch in channel_order)

    # Pre-allocate frame buffer
    frame_buf = np.empty((n_frames, total_ch), dtype=np.float64)
    col = 0
    for entry_id, n_ch in channel_order:
        if isinstance(entry_id, int):
            j = entry_id
            if n_ch == 6:
                # Position source: depth-0 (BVH ROOT) uses root_pos;
                # any other 6-channel joint pulls from extra_pos[j].
                pos_src = root_pos if parents[j] < 0 else extra_pos[j]
                frame_buf[:, col:col+3] = pos_src
                frame_buf[:, col+3:col+6] = euler[:, j]
                col += 6
            else:
                frame_buf[:, col:col+3] = euler[:, j]
                col += 3
        else:
            bone_name = entry_id
            fd = face_frames_lookup[bone_name]  # (N, 6)
            if n_ch == 6:
                frame_buf[:, col:col+6] = fd
            else:
                frame_buf[:, col:col+3] = fd[:, 3:6]
            col += n_ch

    # Format all frames at once using numpy string formatting
    fmt = " ".join(["%.4f"] * total_ch)
    frame_lines = [fmt % tuple(row) for row in frame_buf]
    lines.extend(frame_lines)

    with open(filepath, "w", encoding="ascii") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")

    n_face = len(face_bone_data["order"]) if face_bone_data else 0
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"       Wrote {filepath} ({size_mb:.1f} MB)")
    print(f"       {n_frames} frames, {n_joints} body joints"
          + (f" + {n_face} face bones" if n_face > 0 else "")
          + f", {fps:.1f} FPS")
