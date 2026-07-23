# apply_face_blendshapes.py
# ==========================
# Blender script: keyframes ARKit blendshape weights from face_frames.npz
# onto a mesh that has matching Shape Keys.
#
# Usage:
#   1. Select your mesh object with ARKit Shape Keys
#   2. Set NPZ_PATH below to your face_frames.npz
#   3. Run the script in Blender's text editor or via:
#      blender yourfile.blend --python apply_face_blendshapes.py
#
# Requires: numpy (Blender's bundled Python has it)

import bpy
import numpy as np
import os

# =====================================================================
#  CONFIGURATION
# =====================================================================
NPZ_PATH = "face_frames.npz"
SKIP_UNDETECTED = False
BLENDSHAPE_SCALE = 1.0
BLENDSHAPE_CLAMP = 1.0

# =====================================================================
#  MAIN
# =====================================================================

def main():
    # -- Resolve path --------------------------------------------------
    npz_path = NPZ_PATH
    if not os.path.isabs(npz_path):
        blend_dir = os.path.dirname(bpy.data.filepath)
        if blend_dir:
            candidate = os.path.join(blend_dir, npz_path)
            if os.path.isfile(candidate):
                npz_path = candidate

    if not os.path.isfile(npz_path):
        print(f"ERROR: Face NPZ not found: {npz_path}")
        return

    # -- Load face data ------------------------------------------------
    print(f"Loading: {npz_path}")
    face = np.load(npz_path, allow_pickle=True)

    blendshapes = face["face_blendshapes"]       # (N, 52) float32
    bs_names = list(face["blendshape_names"])     # (52,) strings
    detected = face["face_detected"]             # (N,) bool
    fps = float(face["fps"])
    n_frames = len(blendshapes)

    print(f"  {n_frames} frames, {len(bs_names)} blendshapes, {fps:.1f} FPS")
    print(f"  {detected.sum()} frames with face detected")

    # -- Get active mesh -----------------------------------------------
    obj = bpy.context.active_object
    if obj is None or obj.type != 'MESH':
        print("ERROR: No active mesh object selected.")
        return
    if obj.data.shape_keys is None:
        print(f"ERROR: '{obj.name}' has no Shape Keys.")
        return

    sk_names = {sk.name: sk for sk in obj.data.shape_keys.key_blocks}
    print(f"  Mesh '{obj.name}' has {len(sk_names)} Shape Keys")

    # -- Match blendshape names to Shape Keys --------------------------
    sk_lower = {name.lower(): sk for name, sk in sk_names.items()}
    matched = {}
    missing_sk = []

    for i, bs_name in enumerate(bs_names):
        sk = (sk_names.get(bs_name)
              or sk_names.get(f"_{bs_name}")
              or sk_lower.get(bs_name.lower()))
        if sk:
            matched[i] = sk
        else:
            missing_sk.append(bs_name)

    matched_sk_names = {sk.name for sk in matched.values()}
    missing_bs = [n for n in sk_names if n != "Basis" and n not in matched_sk_names]

    print(f"\n  Matched: {len(matched)}/{len(bs_names)} blendshapes to Shape Keys")
    if missing_sk:
        print(f"  No Shape Key for: {missing_sk[:10]}"
              + (f"... (+{len(missing_sk)-10} more)" if len(missing_sk) > 10 else ""))
    if missing_bs:
        print(f"  Unused Shape Keys: {missing_bs[:10]}"
              + (f"... (+{len(missing_bs)-10} more)" if len(missing_bs) > 10 else ""))

    if not matched:
        print(f"\nERROR: No blendshapes matched! Expected names like: {bs_names[:5]}")
        return

    # -- Pre-process values with numpy (vectorized) --------------------
    vals = blendshapes[:, list(matched.keys())] * BLENDSHAPE_SCALE
    np.clip(vals, 0.0, BLENDSHAPE_CLAMP, out=vals)

    # Build frame mask
    if SKIP_UNDETECTED:
        frame_mask = np.where(detected)[0]
    else:
        frame_mask = np.arange(n_frames)

    n_active = len(frame_mask)

    # -- Set scene FPS -------------------------------------------------
    scene = bpy.context.scene
    scene.render.fps = int(round(fps))
    scene.frame_start = 1
    scene.frame_end = n_frames
    print(f"\n  Scene: {n_frames} frames at {int(round(fps))} FPS")

    # -- Ensure animation data exists ----------------------------------
    sk_data = obj.data.shape_keys
    if sk_data.animation_data is None:
        sk_data.animation_data_create()
    if sk_data.animation_data.action is None:
        sk_data.animation_data.action = bpy.data.actions.new(
            name=f"{obj.name}_ShapeKeyAction"
        )
    action = sk_data.animation_data.action

    # -- Batch-insert keyframes via F-Curves ---------------------------
    print(f"  Keyframing {len(matched)} Shape Keys across {n_active} frames...")

    # Pre-compute frame numbers (1-indexed) as floats for Blender
    frame_nums = (frame_mask + 1).astype(np.float32)

    for col, (bs_idx, sk) in enumerate(matched.items()):
        data_path = f'key_blocks["{sk.name}"].value'
        fc = action.fcurves.find(data_path)
        if fc is None:
            fc = action.fcurves.new(data_path)

        # Build flat co array: [frame, value, frame, value, ...]
        channel_vals = vals[frame_mask, col].astype(np.float32)
        co = np.empty(n_active * 2, dtype=np.float32)
        co[0::2] = frame_nums
        co[1::2] = channel_vals

        # Batch add and set all keyframe points at once
        fc.keyframe_points.add(n_active)
        fc.keyframe_points.foreach_set("co", co)

        # Set all interpolation to LINEAR (prevents bezier overshoot)
        fc.keyframe_points.foreach_set(
            "interpolation",
            [1] * n_active  # 1 = LINEAR in Blender's enum
        )

        fc.update()

    print(f"\nDone! {len(matched)} Shape Keys keyframed across {n_active} frames.")
    print("Press Space to play animation.")


if __name__ == "__main__":
    main()
