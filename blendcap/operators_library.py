"""Library + utility operators: clear caches, revert proxy strip, show
warnings, BVH library import/export/delete, FBX export, ARKit shape-keys.

Grouped here because they're all "side dishes" to the main pipeline
operators — none of them spawn the capture thread, they just operate
on cache/files/strips already on disk.
"""
import bpy
import os
import shutil
import subprocess
import threading

from bpy_extras.io_utils import orientation_helper

from .helpers import (
    blendcap_state, arkit_state,
    _cache_dir, _bvh_library_dir, _addon_dir, _python_exe,
    _host_spawn_prefix, _subprocess_env, _terminate_state_process,
    _resolve_clip_name, _file_mode_source, _progress_re,
    format_workspace_status,
    _seq_accessors, _tag_redraw, _import_bvh, _active_video_strip,
    bvh_importer_state,
    _bake_root_into_hips_for_export,
)
from .properties import _sync_bvh_library
from .mocap import _revert_proxy_strip


# Region -> ARKit blendshape names. Mirrors the master copy in
# pipeline/bvh_math.py (ARKIT_REGION_BLENDSHAPES); the two must stay
# in sync. Duplicated here because the BVH path runs in the venv
# subprocess (imports pipeline.*) while the ARKit path runs in
# Blender's bundled Python, which doesn't import pipeline/.
_ARKIT_REGION_BLENDSHAPES = {
    "jaw":    ["jawOpen", "jawForward", "jawLeft", "jawRight", "mouthClose"],
    "gaze":   ["eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
               "eyeLookUpLeft", "eyeLookDownRight", "eyeLookInRight",
               "eyeLookOutRight", "eyeLookUpRight"],
    "lids":   ["eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft",
               "eyeSquintRight", "eyeWideLeft", "eyeWideRight"],
    "brows":  ["browDownLeft", "browDownRight", "browInnerUp",
               "browOuterUpLeft", "browOuterUpRight"],
    "lips":   ["mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
               "mouthSmileLeft", "mouthSmileRight",
               "mouthFrownLeft", "mouthFrownRight",
               "mouthDimpleLeft", "mouthDimpleRight",
               "mouthStretchLeft", "mouthStretchRight",
               "mouthRollLower", "mouthRollUpper",
               "mouthShrugLower", "mouthShrugUpper",
               "mouthPressLeft", "mouthPressRight",
               "mouthLowerDownLeft", "mouthLowerDownRight",
               "mouthUpperUpLeft", "mouthUpperUpRight"],
    "cheeks": ["cheekPuff", "cheekSquintLeft", "cheekSquintRight"],
    "nose":   ["noseSneerLeft", "noseSneerRight"],
    "tongue": ["tongueOut"],
}


def _matched_name_to_col(matched_list, bs_names):
    """Build {ARKit blendshape name -> column index in vals} for the
    matched-and-keyframed subset. Shared helper for denoise + amp."""
    return {bs_names[bs_idx]: col
            for col, (bs_idx, _sk) in enumerate(matched_list)}


def _apply_arkit_denoise(vals, matched_list, bs_names, props):
    """In-place per-region SOFT dead-zone on the matched blendshape
    columns. Smooth ramp from 0 (at |x|=T) to 1x (at |x|=2T), so
    values straddling the threshold no longer snap on/off frame-to-
    frame. Below T -> 0; above 2T -> unchanged; in between -> scaled
    by smoothstep((|x|-T)/T).

    vals: (N, K) float -- N frames, K matched-and-keyframed blendshapes
    matched_list: list of (bs_idx, sk) pairs; column order of vals
                  corresponds to this list
    bs_names: list of all 52 ARKit blendshape names indexed by bs_idx
    props: BlendCapProps, source of the per-region thresholds
    """
    import numpy as np
    if not getattr(props, "arkit_face_denoise_enabled", False):
        return
    name_to_col = _matched_name_to_col(matched_list, bs_names)
    affected = []
    for region in ("jaw", "gaze", "lids", "brows", "lips", "cheeks", "nose", "tongue"):
        thr = float(getattr(props, f"arkit_face_denoise_{region}", 0.0))
        if thr <= 0:
            continue
        cols = [name_to_col[n] for n in _ARKIT_REGION_BLENDSHAPES[region]
                if n in name_to_col]
        if not cols:
            continue
        sub = vals[:, cols]
        # Smoothstep ramp on |x| / thr in [0,1]:
        #   t = (|x| - thr) / thr   clipped to [0,1]
        #   scale = 3t^2 - 2t^3
        # scale is 0 at |x|<=thr, 1 at |x|>=2*thr, smooth between.
        t = np.clip((np.abs(sub) - thr) / thr, 0.0, 1.0)
        scale = t * t * (3.0 - 2.0 * t)
        vals[:, cols] = sub * scale
        affected.append(f"{region}={thr:.3f}")
    if affected:
        print(f"[ARKit] Face denoise applied (soft): " + ", ".join(affected))


def _apply_arkit_face_amp(vals, matched_list, bs_names, props):
    """In-place per-region amplitude scaling on the matched blendshape
    columns. Skipped entirely when the master toggle is off. When on,
    apply_arkit bypasses the global face_expression_multiplier so the
    two never compound.
    """
    if not getattr(props, "arkit_face_amp_enabled", False):
        return
    name_to_col = _matched_name_to_col(matched_list, bs_names)
    affected = []
    for region in ("jaw", "gaze", "lids", "brows", "lips", "cheeks", "nose", "tongue"):
        mult = float(getattr(props, f"arkit_face_amp_{region}", 1.0))
        if mult == 1.0:
            continue
        cols = [name_to_col[n] for n in _ARKIT_REGION_BLENDSHAPES[region]
                if n in name_to_col]
        if not cols:
            continue
        vals[:, cols] *= mult
        affected.append(f"{region}={mult:.2f}")
    if affected:
        print(f"[ARKit] Face amp applied: " + ", ".join(affected))


def _smooth_face_npz_via_venv(input_npz, output_npz, smooth_s):
    """Run smooth_face_npz.py in the addon venv to apply smoothing to a
    face NPZ. Returns (ok, message) so the caller can surface failures
    to the Blender UI.

    Blender's bundled Python lacks scipy, so the smoother runs in the
    same venv subprocess the BVH export pipeline uses. On any failure
    (script missing, venv missing, subprocess error) returns (False,
    error_string); the caller can fall back to the unsmoothed NPZ.
    """
    script = os.path.join(_addon_dir(), "pipeline", "smooth_face_npz.py")
    python = _python_exe()
    if not os.path.isfile(script):
        return False, f"smooth_face_npz.py not found at {script}"
    if not os.path.isfile(python):
        return False, f"venv python not found at {python}"
    cmd = [python, script, input_npz, output_npz,
           "--smooth-face", f"{smooth_s:.4f}"]
    # Flatpak: run the host venv python via flatpak-spawn ([] elsewhere).
    cmd = _host_spawn_prefix() + cmd
    print(f"[ARKit] Running smoothing subprocess: {' '.join(cmd)}")
    try:
        # utf-8 on BOTH sides: _subprocess_env pins the child's
        # PYTHONIOENCODING, so decoding with the system code page here
        # would just move the UnicodeError from the writer to the reader.
        result = subprocess.run(cmd, capture_output=True, text=True,
                                encoding='utf-8', errors='replace',
                                check=False, env=_subprocess_env(),
                                creationflags=getattr(
                                    subprocess, 'CREATE_NO_WINDOW', 0))
    except Exception as e:
        return False, f"subprocess failed to launch: {e}"
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    if result.returncode != 0:
        return False, (f"subprocess exited {result.returncode}"
                       f" (see console for stderr)")
    if not os.path.isfile(output_npz):
        return False, f"subprocess succeeded but output not found at {output_npz}"
    return True, "ok"


# Live Link Face CSV columns: the 52 ARKit blendshapes in canonical
# ARKit order (UpperCamelCase, TongueOut included), then the 9 head /
# eye rotation columns the Live Link Face app records. Importers
# (Unreal/MetaHuman and friends) expect this exact header set, so the
# rotations are written as 0.0 rather than omitted — BlendCap's face
# path tracks expression only (head rotation comes from the body).
# TongueOut fills by name like every other column, so the pixel-based
# tongueOut channel flows through when the NPZ carries it.
_LLF_BLENDSHAPE_COLUMNS = [
    "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft",
    "EyeLookUpLeft", "EyeSquintLeft", "EyeWideLeft",
    "EyeBlinkRight", "EyeLookDownRight", "EyeLookInRight",
    "EyeLookOutRight", "EyeLookUpRight", "EyeSquintRight", "EyeWideRight",
    "JawForward", "JawLeft", "JawRight", "JawOpen",
    "MouthClose", "MouthFunnel", "MouthPucker", "MouthLeft", "MouthRight",
    "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft", "MouthFrownRight",
    "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft",
    "MouthStretchRight", "MouthRollLower", "MouthRollUpper",
    "MouthShrugLower", "MouthShrugUpper", "MouthPressLeft",
    "MouthPressRight", "MouthLowerDownLeft", "MouthLowerDownRight",
    "MouthUpperUpLeft", "MouthUpperUpRight",
    "BrowDownLeft", "BrowDownRight", "BrowInnerUp",
    "BrowOuterUpLeft", "BrowOuterUpRight",
    "CheekPuff", "CheekSquintLeft", "CheekSquintRight",
    "NoseSneerLeft", "NoseSneerRight",
    "TongueOut",
]
_LLF_ROTATION_COLUMNS = [
    "HeadYaw", "HeadPitch", "HeadRoll",
    "LeftEyeYaw", "LeftEyePitch", "LeftEyeRoll",
    "RightEyeYaw", "RightEyePitch", "RightEyeRoll",
]


def _arkit_csv_text(vals, bs_names, fps):
    """Build Live Link Face-style CSV text from processed blendshape
    values.

    vals: (N, K) float array, one column per bs_names entry, [0, 1].
    bs_names: the NPZ's blendshape name list (camelCase; MediaPipe's
              extra "_neutral" entry is simply never matched).
    fps: frame rate for the Timecode column.

    Columns are matched by name case-insensitively, so the NPZ's
    column order doesn't matter; LLF columns with no NPZ counterpart
    (the rotation columns, or TongueOut on older captures) are
    written as 0.0.
    """
    name_to_col = {str(n).lower(): i for i, n in enumerate(bs_names)}
    src_cols = [name_to_col.get(n.lower()) for n in _LLF_BLENDSHAPE_COLUMNS]
    header = (["Timecode", "BlendShapeCount"]
              + _LLF_BLENDSHAPE_COLUMNS + _LLF_ROTATION_COLUMNS)
    count = len(_LLF_BLENDSHAPE_COLUMNS) + len(_LLF_ROTATION_COLUMNS)
    fps_i = max(1, int(round(fps)))
    zeros_rot = ["0.000000"] * len(_LLF_ROTATION_COLUMNS)
    lines = [",".join(header)]
    for f_idx in range(len(vals)):
        total_s = f_idx // fps_i
        hh = total_s // 3600
        mm = (total_s % 3600) // 60
        ss = total_s % 60
        ff = f_idx % fps_i
        tc = f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}.000"
        row = [tc, str(count)]
        for c in src_cols:
            row.append("0.000000" if c is None
                       else f"{vals[f_idx, c]:.6f}")
        row.extend(zeros_rot)
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def _smooth_face_npz_worker(input_npz, output_npz, smooth_s, state):
    """Worker-thread body for the ARKit Apply modal operator: runs
    smooth_face_npz.py in the addon venv, streaming its [PROGRESS N]
    lines into `state` so the status bar can show real progress.
    No bpy access here. Sets state['done'] last; state['ok'] only on a
    verified output file."""
    script = os.path.join(_addon_dir(), "pipeline", "smooth_face_npz.py")
    python = _python_exe()
    if not os.path.isfile(script):
        state['error_msg'] = f"smooth_face_npz.py not found at {script}"
        state['done'] = True
        return
    if not os.path.isfile(python):
        state['error_msg'] = f"venv python not found at {python}"
        state['done'] = True
        return
    cmd = [python, script, input_npz, output_npz,
           "--smooth-face", f"{smooth_s:.4f}"]
    # Flatpak: run the host venv python via flatpak-spawn ([] elsewhere).
    cmd = _host_spawn_prefix() + cmd
    print(f"[ARKit] Running face-processing subprocess: {' '.join(cmd)}")
    try:
        # utf-8 on BOTH sides - see the note in the sibling call above.
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding='utf-8', errors='replace',
                                env=_subprocess_env(),
                                creationflags=getattr(
                                    subprocess, 'CREATE_NO_WINDOW', 0))
    except Exception as e:
        state['error_msg'] = f"subprocess failed to launch: {e}"
        state['done'] = True
        return
    state['active_process'] = proc
    # A cancel raised before the handle above was published couldn't
    # terminate anything — honor it now instead of orphaning the child.
    if state['cancel_requested']:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            print(line)
            m = _progress_re.search(line)
            if m:
                state['progress'] = float(m.group(1))
                state['status_text'] = m.group(2)
    except Exception:
        pass
    rc = proc.wait()
    state['active_process'] = None
    if state['cancel_requested']:
        state['error_msg'] = "cancelled"
    elif rc != 0:
        state['error_msg'] = f"subprocess exited {rc} (see console for stderr)"
    elif not os.path.isfile(output_npz):
        state['error_msg'] = f"output not found at {output_npz}"
    else:
        state['ok'] = True
    state['done'] = True


def _arkit_apply_npz(op, context, obj, load_path, smooth_status):
    """Load a processed face NPZ and keyframe it onto obj's Shape Keys.
    Main thread only; shared by the modal finish and the synchronous
    execute() fallback. Reports through `op`; returns True on success.

    Keyframes are written by building each shape key's fcurve directly
    (keyframe_points.add + foreach_set) instead of per-frame
    keyframe_insert calls — the insert path re-sorts the curve on every
    call, which read as "Not Responding" for minutes on long clips.
    """
    import numpy as np

    props = context.scene.blendcap_props
    print(f"[ARKit] Loading NPZ: {load_path}  [{smooth_status}]")

    # --- Load face data ---
    face = np.load(load_path, allow_pickle=True)
    blendshapes = face["face_blendshapes"]        # (N, 52/53)
    bs_names = list(face["blendshape_names"])
    fps = float(face["fps"])
    n_frames = len(blendshapes)

    # --- Match blendshape names to Shape Keys ---
    sk_names = {sk.name: sk for sk in obj.data.shape_keys.key_blocks}
    sk_lower = {name.lower(): sk for name, sk in sk_names.items()}
    matched = {}
    missing = []

    for i, bs_name in enumerate(bs_names):
        sk = (sk_names.get(bs_name)
              or sk_names.get(f"_{bs_name}")
              or sk_lower.get(bs_name.lower()))
        if sk:
            matched[i] = sk
        else:
            missing.append(bs_name)

    if not matched:
        op.report({'ERROR'},
            f"No Shape Keys matched. Mesh needs ARKit-named keys "
            f"(e.g. jawOpen, eyeBlinkLeft). Found: {list(sk_names.keys())[:5]}")
        return False

    # --- Set scene FPS ---
    # FPS only, so the keyframes (written at the capture's frame
    # rate) play at real-time speed. The scene frame range is the
    # user's own; never touch it here.
    scene = context.scene
    scene.render.fps = int(round(fps))

    # --- Smooth and scale ---
    sk_data = obj.data.shape_keys
    vals = blendshapes[:, list(matched.keys())].astype(np.float64)

    # Processing order (matches BVH face pipeline):
    # 1. Refine + smoothing -- already applied upstream via the venv
    #    subprocess at the NPZ load (runs at every smoothing value,
    #    incl. 0).
    # 2. Per-region dead-zone denoise.
    # 3. Amplitude scaling: per-region (if enabled) OR global
    #    face_expression_multiplier -- mutually exclusive to avoid
    #    compounding.
    matched_list = list(matched.items())

    _apply_arkit_denoise(vals, matched_list, bs_names, props)

    if getattr(props, "arkit_face_amp_enabled", False):
        _apply_arkit_face_amp(vals, matched_list, bs_names, props)
    else:
        multiplier = float(props.face_expression_multiplier)
        if multiplier != 1.0:
            vals *= multiplier

    np.clip(vals, 0.0, 1.0, out=vals)

    # --- Bulk-write the fcurves ---
    # Materialize one keyframe per matched key so the action + fcurves
    # exist regardless of Blender's action layout (flat fcurves vs the
    # 4.4+/5.x layered/slotted layout), then find each curve by its
    # exact data path and rebuild it in one foreach_set.
    for _bs_idx, sk in matched_list:
        sk.keyframe_insert(data_path="value", frame=1)

    adt = sk_data.animation_data
    action = adt.action if adt else None
    fcurves = []
    if action:
        # Slotted actions (4.4+/5.x): collect ONLY this datablock's
        # slot. A shared action carries other datablocks' curves under
        # identical shape-key data paths — a cross-slot sweep would
        # rebuild another character's face animation.
        slot = getattr(adt, 'action_slot', None)
        if slot is not None and hasattr(action, 'layers'):
            for layer in action.layers:
                for strip in layer.strips:
                    cb = None
                    if hasattr(strip, 'channelbag'):
                        try:
                            cb = strip.channelbag(slot)
                        except Exception:
                            cb = None
                    if cb is not None:
                        fcurves.extend(cb.fcurves)
        if not fcurves:
            if hasattr(action, 'fcurves'):
                fcurves = list(action.fcurves)
            elif hasattr(action, 'layers'):
                for layer in action.layers:
                    for strip in layer.strips:
                        if hasattr(strip, 'channelbags'):
                            for cb in strip.channelbags:
                                fcurves.extend(cb.fcurves)
                        elif hasattr(strip, 'fcurves'):
                            fcurves.extend(strip.fcurves)
    fc_by_path = {fc.data_path: fc for fc in fcurves}

    linear = bpy.types.Keyframe.bl_rna.properties[
        'interpolation'].enum_items['LINEAR'].value
    frames = np.arange(1, n_frames + 1, dtype=np.float64)
    co = np.empty(n_frames * 2, dtype=np.float64)
    # LINEAR everywhere to prevent bezier overshoot.
    interp = np.full(n_frames, linear, dtype=np.int32)
    unwritten = 0
    for col, (_bs_idx, sk) in enumerate(matched_list):
        fc = fc_by_path.get(sk.path_from_id("value"))
        if fc is None:
            unwritten += 1
            continue
        pts = fc.keyframe_points
        pts.clear()
        pts.add(n_frames)
        co[0::2] = frames
        co[1::2] = vals[:, col]
        pts.foreach_set('co', co)
        pts.foreach_set('interpolation', interp)
        fc.update()
    if unwritten:
        print(f"[ARKit] WARNING: {unwritten} matched shape keys had no "
              f"fcurve after keyframe_insert; skipped.")

    n_matched = len(matched)
    n_missing = len(missing)
    msg = (f"Applied {n_matched}/{len(bs_names)} ARKit Shape Keys "
           f"across {n_frames} frames")
    if n_missing:
        msg += f" ({n_missing} unmatched)"
    op.report({'INFO'}, msg)
    return True


class BLENDCAP_OT_apply_arkit(bpy.types.Operator):
    bl_idname = "blendcap.apply_arkit"
    bl_label = "Apply ARKit Shape Keys"
    bl_description = "Keyframe ARKit blendshape weights from the face capture data " \
                     "onto the active mesh's Shape Keys"
    bl_options = {'REGISTER', 'UNDO'}

    _timer = None
    _obj_name = ""
    _npz_path = ""
    _load_path = ""
    _smooth_s = 0.0

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj is not None and obj.type == 'MESH'
                and obj.data.shape_keys is not None)

    def _resolve_npz(self, context):
        """(npz_path, error). The ARKit clip dropdown is the primary
        control: an explicit clip choice wins outright and needs no
        video selection — only the mesh matters then. AUTO reads the
        active video source to locate its cache (same resolution
        save_arkit_csv uses)."""
        props = context.scene.blendcap_props
        cache_dir = _cache_dir()
        arkit_clip = getattr(props, "arkit_face_data", 'AUTO')
        if arkit_clip and arkit_clip not in ('AUTO', 'NONE'):
            npz_path = os.path.join(cache_dir, arkit_clip,
                                    "face_frames.npz")
        else:
            clip_name = _resolve_clip_name(context)
            if not clip_name:
                return "", ("No video source found. Select the tracked "
                            "video, or pick a clip in the Face Data "
                            "dropdown.")
            npz_path = os.path.join(cache_dir, f"Capture_{clip_name}",
                                    "face_frames.npz")
        if not os.path.isfile(npz_path):
            return "", f"Face data not found: {npz_path}"
        return npz_path, ""

    def invoke(self, context, event):
        # Responsive path for the button: the venv refine/smoothing
        # subprocess runs in a worker thread with the standard BlendCap
        # status-bar progress, then the (fast) keyframe write happens
        # on the main thread when it lands. Esc cancels.
        if arkit_state['running']:
            self.report({'WARNING'}, "An ARKit apply is already running.")
            return {'CANCELLED'}
        # A cancelled run's worker can outlive its modal briefly; it
        # shares the state dict and the *_smoothed.npz output path, so
        # don't start over it.
        prev = arkit_state.get('thread')
        if prev is not None and prev.is_alive():
            self.report({'WARNING'},
                        "Previous ARKit run is still winding down - "
                        "try again in a moment.")
            return {'CANCELLED'}

        npz_path, err = self._resolve_npz(context)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}

        props = context.scene.blendcap_props
        self._npz_path = npz_path
        self._smooth_s = max(0.0, float(getattr(props, "arkit_face_smooth",
                                                0.0)))
        self._load_path = os.path.splitext(npz_path)[0] + "_smoothed.npz"
        self._obj_name = context.active_object.name

        # Pre-set the whole state before thread.start so a stale prior
        # run can't be read by this modal's first timer tick.
        arkit_state.update(running=True, done=False, ok=False,
                           error_msg="", cancel_requested=False,
                           active_process=None, progress=2.0,
                           status_text="starting")
        thread = threading.Thread(
            target=_smooth_face_npz_worker,
            args=(npz_path, self._load_path, self._smooth_s, arkit_state),
            daemon=True)
        arkit_state['thread'] = thread
        thread.start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.25, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _set_status(self, context, text):
        try:
            context.workspace.status_text_set(text)
        except Exception:
            pass

    def _finish(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        self._set_status(context, None)
        arkit_state['running'] = False
        # Swap the panel's progress bar back to the action buttons.
        _tag_redraw('VIEW_3D')

    def cancel(self, context):
        # Blender force-terminates the modal without a final modal()
        # call on File > New / File > Open / window close. Same
        # wind-down as ESC (no report — the context is going away).
        arkit_state['cancel_requested'] = True
        _terminate_state_process(arkit_state)
        self._finish(context)

    def modal(self, context, event):
        if event.type == 'ESC':
            arkit_state['cancel_requested'] = True
            _terminate_state_process(arkit_state)
            self._finish(context)
            self.report({'WARNING'}, "ARKit apply cancelled.")
            return {'CANCELLED'}
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        # The panel Cancel button (cancel_arkit_apply) kills the
        # subprocess and raises this flag from its own operator; never
        # fall through to the raw-NPZ fallback on a cancel.
        if arkit_state['cancel_requested']:
            self._finish(context)
            self.report({'WARNING'}, "ARKit apply cancelled.")
            return {'CANCELLED'}

        if not arkit_state['done']:
            # Subprocess [PROGRESS] lines own 0-85; the keyframe write
            # is the final stretch (95). Panel bar reads the same raw
            # value so the two displays always agree.
            pct = min(95.0, arkit_state['progress'])
            self._set_status(context, format_workspace_status(
                "BlendCap ARKit", pct,
                f"{arkit_state['status_text']}  (Esc to cancel)"))
            # Keep the in-panel progress bar ticking.
            _tag_redraw('VIEW_3D')
            return {'RUNNING_MODAL'}

        # --- Subprocess landed: apply on the main thread ---
        arkit_state['progress'] = 95.0
        arkit_state['status_text'] = "applying keys"
        obj = bpy.data.objects.get(self._obj_name)
        if (obj is None or obj.type != 'MESH'
                or obj.data.shape_keys is None):
            self._finish(context)
            self.report({'ERROR'},
                        "Target mesh disappeared during face processing.")
            return {'CANCELLED'}

        if arkit_state['ok']:
            load_path = self._load_path
            smooth_status = (f"refined + smoothed @ {self._smooth_s:.2f}s"
                             if self._smooth_s > 0
                             else "refined (unsmoothed)")
        else:
            # Same fallback the old synchronous path had: surface the
            # failure and apply the raw NPZ rather than aborting.
            self.report({'WARNING'},
                        f"ARKit face processing skipped "
                        f"({arkit_state['error_msg']}); "
                        f"applying RAW blendshapes")
            load_path = self._npz_path
            smooth_status = "FAILED (raw)"

        self._set_status(context, format_workspace_status(
            "BlendCap ARKit", 95, "applying keys"))
        # try/finally: an exception here (unreadable NPZ, non-editable
        # linked shape keys, ...) must still tear down the timer and
        # drop arkit_state['running'], or the panel is stuck on the
        # progress bar until Blender restarts.
        applied = False
        try:
            applied = _arkit_apply_npz(self, context, obj, load_path,
                                       smooth_status)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"ARKit apply failed: {e}")
        finally:
            self._finish(context)
        return {'FINISHED'} if applied else {'CANCELLED'}

    def execute(self, context):
        # Synchronous fallback for script/batch calls (the button goes
        # through invoke's responsive modal path).
        npz_path, err = self._resolve_npz(context)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}

        props = context.scene.blendcap_props
        smooth_s = max(0.0, float(getattr(props, "arkit_face_smooth", 0.0)))
        smoothed_path = os.path.splitext(npz_path)[0] + "_smoothed.npz"
        ok, msg = _smooth_face_npz_via_venv(npz_path, smoothed_path,
                                            smooth_s)
        if ok:
            load_path = smoothed_path
            smooth_status = (f"refined + smoothed @ {smooth_s:.2f}s"
                             if smooth_s > 0 else "refined (unsmoothed)")
        else:
            self.report({'WARNING'},
                        f"ARKit face processing skipped ({msg}); "
                        f"applying RAW blendshapes")
            load_path = npz_path
            smooth_status = "FAILED (raw)"

        applied = _arkit_apply_npz(self, context, context.active_object,
                                   load_path, smooth_status)
        return {'FINISHED'} if applied else {'CANCELLED'}


class BLENDCAP_OT_cancel_arkit_apply(bpy.types.Operator):
    bl_idname = "blendcap.cancel_arkit_apply"
    bl_label = "Cancel ARKit Apply"
    bl_description = "Stop the running ARKit face processing"
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return arkit_state['running']

    def execute(self, context):
        # The modal operator owns the wind-down: this just raises the
        # flag and kills the subprocess so its blocked reader returns.
        arkit_state['cancel_requested'] = True
        _terminate_state_process(arkit_state)
        return {'FINISHED'}


class BLENDCAP_OT_save_arkit_csv(bpy.types.Operator):
    """Export the face capture's ARKit weights as a Live Link Face-
    style CSV (Timecode, BlendShapeCount, 52 blendshapes, head/eye
    rotation columns) for use outside Blender — Unreal/MetaHuman,
    iClone and similar tools read this format directly.

    Runs the same processing chain apply_arkit runs (venv smoothing
    subprocess, per-region denoise, per-region or global amplitude)
    so the CSV matches what Apply would have keyframed. Unlike Apply
    it needs no mesh: all 52 tracked columns export, not just the
    matched subset.
    """
    bl_idname = "blendcap.save_arkit_csv"
    bl_label = "Save ARKit CSV"
    bl_description = ("Export the face capture's ARKit blendshape "
                      "weights as a CSV file (Live Link Face format) "
                      "for use in other programs. Applies the "
                      "smoothing, noise filtering and expression "
                      "settings above")
    bl_options = {'REGISTER'}

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(default="*.csv",
                                          options={'HIDDEN'})

    def _resolve_npz(self, context):
        """(npz_path, clip_label) for the ARKit dropdown selection —
        the same resolution apply_arkit uses: an explicit clip choice
        wins, AUTO falls back to the active video's cache."""
        props = context.scene.blendcap_props
        cache_dir = _cache_dir()
        arkit_clip = getattr(props, "arkit_face_data", 'AUTO')
        if arkit_clip and arkit_clip not in ('AUTO', 'NONE'):
            label = (arkit_clip[len("Capture_"):]
                     if arkit_clip.startswith("Capture_") else arkit_clip)
            return (os.path.join(cache_dir, arkit_clip,
                                 "face_frames.npz"), label)
        clip_name = _resolve_clip_name(context) or "sequence"
        return (os.path.join(cache_dir, f"Capture_{clip_name}",
                             "face_frames.npz"), clip_name)

    def invoke(self, context, event):
        # The Apply ARKit modal's subprocess writes the same
        # *_smoothed.npz this export would — never run both at once.
        if arkit_state['running']:
            self.report({'WARNING'},
                        "Wait for the running ARKit apply to finish first.")
            return {'CANCELLED'}
        npz_path, label = self._resolve_npz(context)
        if not os.path.isfile(npz_path):
            self.report({'ERROR'}, f"Face data not found: {npz_path}")
            return {'CANCELLED'}
        if not self.filepath:
            self.filepath = f"{label}_arkit.csv"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        import numpy as np

        # Re-check here too: the file-select dialog can sit open while
        # the user starts an Apply, and execute() runs after it closes.
        if arkit_state['running']:
            self.report({'WARNING'},
                        "Wait for the running ARKit apply to finish first.")
            return {'CANCELLED'}

        props = context.scene.blendcap_props
        npz_path, _label = self._resolve_npz(context)
        if not os.path.isfile(npz_path):
            self.report({'ERROR'}, f"Face data not found: {npz_path}")
            return {'CANCELLED'}

        # Refine + smoothing first, same as apply_arkit (scipy lives
        # in the venv, not Blender's Python). Runs even at smoothing 0
        # so the refine pass + tongueOut are never skipped; the helper
        # treats smoothing 0 as pass-through. Failure falls back to
        # raw.
        load_path = npz_path
        smooth_s = max(0.0, float(getattr(props, "arkit_face_smooth", 0.0)))
        smoothed_path = os.path.splitext(npz_path)[0] + "_smoothed.npz"
        ok, msg = _smooth_face_npz_via_venv(npz_path, smoothed_path,
                                            smooth_s)
        if ok:
            load_path = smoothed_path
        else:
            self.report({'WARNING'},
                        f"ARKit face processing skipped ({msg}); "
                        f"exporting RAW blendshapes")
        print(f"[ARKit] CSV export loading NPZ: {load_path}")

        face = np.load(load_path, allow_pickle=True)
        blendshapes = face["face_blendshapes"]        # (N, 52)
        bs_names = [str(n) for n in face["blendshape_names"]]
        fps = float(face["fps"])
        n_frames = len(blendshapes)
        if n_frames == 0:
            self.report({'ERROR'}, "Face data is empty, nothing to export.")
            return {'CANCELLED'}

        # Denoise + amplitude over ALL columns (no mesh matching).
        # The helpers only read the blendshape index out of each
        # (bs_idx, sk) pair, so a full-range list with sk=None works.
        vals = blendshapes.astype(np.float64)
        all_cols = [(i, None) for i in range(len(bs_names))]
        _apply_arkit_denoise(vals, all_cols, bs_names, props)
        if getattr(props, "arkit_face_amp_enabled", False):
            _apply_arkit_face_amp(vals, all_cols, bs_names, props)
        else:
            multiplier = float(props.face_expression_multiplier)
            if multiplier != 1.0:
                vals *= multiplier
        np.clip(vals, 0.0, 1.0, out=vals)

        filepath = self.filepath
        if not filepath.lower().endswith(".csv"):
            filepath += ".csv"
        try:
            with open(filepath, "w", encoding="utf-8", newline="") as f:
                f.write(_arkit_csv_text(vals, bs_names, fps))
        except OSError as e:
            self.report({'ERROR'}, f"Could not write CSV: {e}")
            return {'CANCELLED'}

        self.report({'INFO'},
                    f"Saved ARKit CSV ({n_frames} frames) to {filepath}")
        return {'FINISHED'}


def _clear_person_folders(cache_dir, vid_name):
    """A multi-person capture's people each get a capture folder of
    their own at export (Capture_<clip>_P<n>, marked with .person_of).
    They go with the capture. Returns the number of files removed."""
    count = 0
    try:
        names = os.listdir(cache_dir)
    except OSError:
        return 0
    for name in names:
        tail = name[len(vid_name):]
        if not (name.startswith(vid_name) and tail.startswith("_P") and tail[2:].isdigit()):
            continue
        folder = os.path.join(cache_dir, name)
        if not os.path.isfile(os.path.join(folder, ".person_of")):
            continue        # a clip of the user's that only happens to be named like one
        # (a person of a multi-camera take: the person as each camera
        # saw them is in cam_a, cam_b, ... inside it)
        subs = [os.path.join(folder, s) for s in os.listdir(folder)
                if s.startswith("cam_") and os.path.isdir(os.path.join(folder, s))]
        for where in subs + [folder]:
            for fname in os.listdir(where):
                if fname in (".person_of", ".person_sig", ".cache_params") or fname.endswith(
                        (".npz", ".npy", ".json", ".bvh")):
                    try:
                        os.remove(os.path.join(where, fname))
                        count += 1
                    except Exception:
                        pass
            try:
                os.rmdir(where)
            except OSError:
                pass
    return count


class BLENDCAP_OT_clear_current_cache(bpy.types.Operator):
    bl_idname = "blendcap.clear_current_cache"
    bl_label = "Clear Cache?"
    bl_description = "Delete the cached tracking data for the selected clip"
    bl_options = {'REGISTER'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=350)

    def draw(self, context):
        layout = self.layout
        layout.label(text="Delete previews and tracking data for this clip.", icon='ERROR')
        layout.label(text="Restores original video. BVH library not affected.")

    def execute(self, context):
        props = context.scene.blendcap_props

        # Resolve video path (same winner as capture — _file_mode_source —
        # so Clear Cache always targets the clip capture would run on)
        if props.video_source == 'FILE':
            video_path, _empty = _file_mode_source(context)
        else:
            seq_ed = context.scene.sequence_editor
            strip = _active_video_strip(seq_ed)
            if strip and strip.type in ('MOVIE', 'IMAGE'):
                video_path = bpy.path.abspath(str(strip["blendcap_source"])) if "blendcap_source" in strip else bpy.path.abspath(getattr(strip, 'filepath', getattr(strip, 'directory', '')))
            else:
                video_path = ""

        if not video_path or not os.path.exists(video_path):
            self.report({'ERROR'}, "Please select a valid video to clear its cache.")
            return {'CANCELLED'}

        clip_name = _resolve_clip_name(context) or "sequence"
        vid_name = f"Capture_{clip_name}"
        output_dir = os.path.join(_cache_dir(), vid_name)
        output_dir_n = os.path.normpath(output_dir)

        # Revert only the proxies stamped with THIS clip_name, then
        # remove any cache-only strips (movies imported directly from
        # the cache folder). Sibling proxies that share the same
        # source file but were captured from a different VSE split
        # (different clip_name) are left alone.
        seq_ed = context.scene.sequence_editor
        if seq_ed:
            seq_all, seq_coll = _seq_accessors(seq_ed)
            if seq_coll:
                reverted = 0
                last_reverted = None
                for s in list(seq_all):
                    if s.type not in ('MOVIE', 'IMAGE'):
                        continue
                    if "blendcap_source" in s:
                        stamped = str(s.get("blendcap_source_strip", "") or "")
                        if stamped and stamped == clip_name:
                            from . import multicam
                            if multicam.is_multicam_strip(s):
                                back, why = multicam.revert_multicam(s, seq_ed)
                                if back:
                                    last_reverted = back[0]
                                    reverted += 1
                                elif why:
                                    print(f"[BlendCap] Clear Cache: {why}")
                                continue
                            new_strip = _revert_proxy_strip(s, seq_coll, seq_all)
                            if new_strip:
                                last_reverted = new_strip
                                reverted += 1
                    elif s.type == 'MOVIE':
                        sp = os.path.normpath(bpy.path.abspath(s.filepath))
                        if sp.startswith(output_dir_n + os.sep):
                            seq_coll.remove(s)
                if last_reverted:
                    seq_ed.active_strip = last_reverted
                if reverted:
                    _tag_redraw('SEQUENCE_EDITOR')

        # Remove viewport Empty only when it represents THIS clip
        # (matched by stamped clip name, not the source file path —
        # multiple file-mode captures of split clips can coexist).
        for obj in list(bpy.data.objects):
            if obj.type == 'EMPTY' and "blendcap_source" in obj:
                stamped = str(obj.get("blendcap_source_strip", "") or "")
                if stamped and stamped == clip_name:
                    if obj.data and isinstance(obj.data, bpy.types.Image):
                        bpy.data.images.remove(obj.data)
                    bpy.data.objects.remove(obj, do_unlink=True)
                    _tag_redraw('VIEW_3D')

        # Delete cache files
        if os.path.exists(output_dir):
            # Remove cache params marker first
            cp = os.path.join(output_dir, ".cache_params")
            if os.path.exists(cp):
                try: os.remove(cp)
                except Exception: pass
            count = 0
            for f in os.listdir(output_dir):
                if f.endswith((".npz", ".npy", ".json", ".mp4", ".bvh")):
                    try:
                        os.remove(os.path.join(output_dir, f))
                        count += 1
                    except Exception: pass
            # A multi-camera capture keeps each camera's own capture in
            # a subfolder.
            from .multicam import cache_subdirs
            for sub in cache_subdirs(output_dir):
                if not os.path.isdir(sub):
                    continue
                for f in os.listdir(sub):
                    if f in (".cache_params", ".no_face") or f.endswith(
                            (".npz", ".npy", ".json", ".mp4", ".bvh", ".log")):
                        try:
                            os.remove(os.path.join(sub, f))
                            count += 1
                        except Exception: pass
                try: os.rmdir(sub)
                except OSError: pass
            count += _clear_person_folders(_cache_dir(), vid_name)
            try:
                os.rmdir(output_dir)
                self.report({'INFO'}, f"Cleared {count} cache files and deleted folder.")
            except OSError:
                self.report({'INFO'}, f"Cleared {count} cache files. (Folder in use).")
        else:
            self.report({'INFO'}, "Proxy objects removed. (No cache files found on disk).")
        return {'FINISHED'}


class BLENDCAP_OT_clear_all_caches(bpy.types.Operator):
    bl_idname = "blendcap.clear_all_caches"
    bl_label = "Clear All?"
    bl_description = "Delete the cached tracking data for every clip in the project directory"
    bl_options = {'REGISTER'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=350)

    def draw(self, context):
        layout = self.layout
        layout.label(text="Delete previews and tracking data for all clips.", icon='ERROR')
        layout.label(text="Restores all originals. BVH library not affected.")

    def execute(self, context):
        cache_dir = _cache_dir()

        # Revert proxy strips across EVERY scene, not just the
        # active one — a proxy hiding in another scene still ties
        # up its source file.
        touched_scenes = []
        for scene in bpy.data.scenes:
            seq_ed = scene.sequence_editor
            if not seq_ed:
                continue
            seq_all, seq_coll = _seq_accessors(seq_ed)
            if not seq_coll:
                continue
            had_proxy = False
            from . import multicam
            for s in list(seq_all):
                if s.type in ('MOVIE', 'IMAGE') and "blendcap_source" in s:
                    if multicam.is_multicam_strip(s):
                        # a multi-camera capture: all its clips come back
                        _back, why = multicam.revert_multicam(s, seq_ed)
                        if why:
                            print(f"[BlendCap] Clear All: {why}")
                    else:
                        _revert_proxy_strip(s, seq_coll, seq_all)
                    had_proxy = True
            if had_proxy:
                seq_ed.active_strip = None
                touched_scenes.append(scene)
        if touched_scenes:
            _tag_redraw('SEQUENCE_EDITOR')

        # Remove all viewport empties with blendcap_source
        for obj in list(bpy.data.objects):
            if obj.type == 'EMPTY' and "blendcap_source" in obj:
                if obj.data and isinstance(obj.data, bpy.types.Image):
                    bpy.data.images.remove(obj.data)
                bpy.data.objects.remove(obj, do_unlink=True)
        _tag_redraw('VIEW_3D')

        # Flush Blender's references to the just-removed strips so
        # FFmpeg movie handles drop before we try to delete the
        # files. Without this nudge overlay.mp4 / source_export.mp4
        # frequently stay locked on Windows after scrubbing.
        try:
            context.view_layer.update()
        except Exception:
            pass

        # Drop the per-strip frame cache on every scene we touched.
        # cache_clear drops the cache without reopening files (which
        # refresh_all would do). It needs a SEQUENCE_EDITOR area in
        # the override; if none is open we temporarily retype one
        # of the user's visible areas to SEQUENCE_EDITOR, run the
        # operator, then put it back. Single-frame flicker, no
        # data touched.
        seq_window = seq_area = seq_region = None
        for win in context.window_manager.windows:
            for area in win.screen.areas:
                if area.type == 'SEQUENCE_EDITOR':
                    for region in area.regions:
                        if region.type == 'WINDOW':
                            seq_window, seq_area, seq_region = win, area, region
                            break
                    if seq_area is not None:
                        break
            if seq_area is not None:
                break

        retyped_area = None
        retyped_orig = None
        if seq_area is None:
            for win in context.window_manager.windows:
                for area in win.screen.areas:
                    if area.type in ('VIEW_3D', 'PROPERTIES', 'OUTLINER',
                                     'IMAGE_EDITOR', 'NODE_EDITOR',
                                     'TEXT_EDITOR', 'CONSOLE'):
                        retyped_area = area
                        retyped_orig = area.type
                        try:
                            area.type = 'SEQUENCE_EDITOR'
                        except Exception:
                            retyped_area = None
                            retyped_orig = None
                            continue
                        for region in area.regions:
                            if region.type == 'WINDOW':
                                seq_window, seq_area, seq_region = win, area, region
                                break
                        break
                if seq_area is not None:
                    break

        for scene in touched_scenes:
            se = scene.sequence_editor
            if se is None:
                continue
            try:
                prev_show = bool(getattr(se, 'show_cache', False))
                se.show_cache = False
                se.show_cache = prev_show
            except Exception:
                pass
            if seq_area is not None and hasattr(bpy.ops.sequencer, 'cache_clear'):
                try:
                    with context.temp_override(
                            window=seq_window, area=seq_area,
                            region=seq_region, scene=scene):
                        bpy.ops.sequencer.cache_clear()
                except Exception:
                    pass

        if retyped_area is not None and retyped_orig is not None:
            try:
                retyped_area.type = retyped_orig
            except Exception:
                pass

        # Bump the depsgraph to evict any stale C-side references
        # cache_clear didn't catch.
        try:
            f = context.scene.frame_current
            context.scene.frame_set(f)
        except Exception:
            pass

        # Delete cache folders. No retry loop — if a file is still
        # locked after the flush above (rare, Windows), the folder
        # survives, so its tracking-identity files are removed
        # individually: a half-deleted cache must never satisfy a
        # cache-params match and silently pass off stale data as a
        # fresh capture. Locked leftovers (overlay/source mp4s) get
        # overwritten by the next capture.
        if not os.path.exists(cache_dir):
            self.report({'INFO'}, "No cache directory found. Proxy strips reverted.")
            return {'FINISHED'}

        cleared = 0
        failed = 0
        for folder in os.listdir(cache_dir):
            fp = os.path.join(cache_dir, folder)
            if os.path.isdir(fp) and folder != "bvh_exports":
                try:
                    shutil.rmtree(fp)
                    cleared += 1
                except Exception as e:
                    print(f"[BlendCap] Failed to delete {fp}: {e}")
                    failed += 1
                    from .multicam import cache_subdirs
                    for sub in [fp] + cache_subdirs(fp):
                        for leftover in (".cache_params", "mhr_frames.npz",
                                         "face_frames.npz"):
                            try:
                                os.remove(os.path.join(sub, leftover))
                            except OSError:
                                pass

        if failed:
            self.report({'WARNING'},
                        f"Emptied {cleared} cache folders; {failed} could not "
                        f"be fully removed (files in use). Tracking data in "
                        f"them was invalidated.")
        else:
            self.report({'INFO'}, f"Emptied {cleared} cache folders. Mocap Library preserved.")
        return {'FINISHED'}


class BLENDCAP_OT_revert_strip(bpy.types.Operator):
    bl_idname = "blendcap.revert_strip"
    bl_label = "Revert to Original Video"
    bl_description = "Replace the BlendCap preview/overlay strip with the original source video"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        seq_ed = context.scene.sequence_editor
        if not seq_ed:
            self.report({'ERROR'}, "No sequence editor found.")
            return {'CANCELLED'}

        strip = getattr(seq_ed, "active_strip", None)
        if not strip or "blendcap_source" not in strip:
            self.report({'ERROR'}, "No BlendCap proxy strip selected.")
            return {'CANCELLED'}

        from . import multicam
        if multicam.is_multicam_strip(strip):
            back, why = multicam.revert_multicam(strip, seq_ed)
            if not back:
                self.report({'ERROR'}, why or "Failed to revert strip.")
                return {'CANCELLED'}
            self.report({'INFO'},
                        f"Reverted to the {len(back)} original clips, still synced.")
            return {'FINISHED'}

        original_path = bpy.path.abspath(str(strip["blendcap_source"]))
        original_type = str(strip.get("blendcap_original_type", "MOVIE"))
        if original_type == 'IMAGE':
            orig_dir = str(strip.get("blendcap_original_directory", original_path))
            if not os.path.isdir(orig_dir):
                self.report({'ERROR'}, f"Original image directory not found: {os.path.basename(orig_dir)}")
                return {'CANCELLED'}
        elif not os.path.exists(original_path):
            self.report({'ERROR'}, f"Original video not found: {os.path.basename(original_path)}")
            return {'CANCELLED'}

        seq_all, seq_coll = _seq_accessors(seq_ed)
        new_strip = _revert_proxy_strip(strip, seq_coll, seq_all)
        if new_strip:
            seq_ed.active_strip = new_strip
            _tag_redraw('SEQUENCE_EDITOR')
            self.report({'INFO'}, f"Reverted to original: {os.path.basename(original_path)}")
        else:
            self.report({'ERROR'}, "Failed to revert strip.")
            return {'CANCELLED'}
        return {'FINISHED'}


class BLENDCAP_OT_show_warnings(bpy.types.Operator):
    bl_idname = "blendcap.show_warnings"
    bl_label = "BlendCap Warning"
    bl_options = {'REGISTER'}

    def invoke(self, context, event):
        # Centered modal dialog matching the framerate-mismatch warning
        # in operators_pipeline.py. invoke_props_dialog ships with OK +
        # Cancel buttons — Blender's API has no flag to hide just Cancel.
        # For a post-pipeline informational dialog both buttons just
        # dismiss; not ideal but the dialog placement and visibility
        # outweigh the extra button.
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        layout = self.layout
        warnings = blendcap_state.get('warnings', [])
        if not warnings:
            layout.label(text="No warnings.", icon='CHECKMARK')
            return

        # Multi-subject / multi-face / flickering tracking gets a curated
        # two-line layout (icon line + follow-up explanation), matching
        # the framerate-mismatch dialog. Everything else (custom messages
        # passed via _show_warning_popup, e.g. "Body tracking data is
        # missing") gets rendered line-by-line as-is.
        def _is_multi_subject(text):
            t = text.lower()
            return (("multi-subject" in t) or ("multi-face" in t)
                    or ("multiple subject" in t) or ("multiple face" in t)
                    or ("flickering tracking" in t))

        if any(_is_multi_subject(w) for w in warnings):
            # A capture that finished body-only after its face step
            # failed (mocap._face_fallback_lines) still says so here.
            # Only lines that are in THIS warning list count, so an
            # earlier capture's notice can't show up in a later dialog.
            fallback = blendcap_state.get('face_fallback_lines') or []
            shown = [w for w in warnings if w in fallback]
            for i, w in enumerate(shown):
                layout.label(text=w, icon=('ERROR' if i == 0 else 'NONE'))
            if shown:
                layout.separator()
            layout.label(text="Multiple people detected, or flickering tracking.",
                         icon='ERROR')
            layout.label(text="Continuity compensation has been applied.")
            return

        for i, w in enumerate(warnings):
            layout.label(text=w, icon=('ERROR' if i == 0 else 'NONE'))

    def execute(self, context):
        return {'FINISHED'}


def _parse_vid_from_stem(stem):
    """Inverse of the BlendCap BVH naming convention. Returns the bare
    clip identifier embedded in the stem, or None if the stem doesn't
    follow the convention.

    Conventions (from BLENDCAP_OT_generate_bvh / BLENDCAP_OT_bake_camera_offset):
      Body w/ own face / no face : {vid}_{codes}        codes ⊆ "BHRF"
      Body w/ separate face       : {vid}_{codes}-{face_source}_F
      Standalone face from clip   : {face_source}_F     (treated as vid="face_source", codes="F")

    Returned vid lets the Bake button rebuild the cache folder path
    (Capture_{vid}) and locate the source NPZs.
    """
    from .helpers import stem_clip, stem_parts
    s = stem
    # Strip optional separate-face suffix '-{face_source}_F' so what's
    # left is the body part. Use rpartition so the last '-' wins (vid
    # names are allowed to contain '-'), and only when both halves are
    # names of their own: a clip called 'cam-a' has a face-only BVH
    # 'cam-a_F', which is one name, not two.
    if "-" in s:
        body_part, _, face_part = s.rpartition("-")
        face = stem_parts(face_part)
        if face is not None and face[1] == "F" and stem_parts(body_part) is not None:
            s = body_part
    # (helpers.stem_clip: a multi-camera take's number of cameras sits
    # in front of the letters, 'Demo_4cBHR' is the capture 'Demo_4cam')
    return stem_clip(s, _cache_dir())


class BLENDCAP_OT_import_lib_bvh(bpy.types.Operator):
    bl_idname = "blendcap.import_lib_bvh"
    bl_label = "Import BVH"
    bl_description = "Import the selected BVH from the library into the current scene"
    filename: bpy.props.StringProperty()

    def execute(self, context):
        full_path = os.path.join(_bvh_library_dir(), self.filename)
        # Windows path limit (item 10): Blender can't see the file past it.
        from .helpers import block_if_path_too_long
        if block_if_path_too_long([(full_path, False, 'library')]):
            return {'CANCELLED'}
        if os.path.exists(full_path):
            # The library file is already on disk, so the dialog carries
            # its path either way - a user whose Blender has no BVH
            # importer at all can still go and fetch it.
            if bvh_importer_state() != 'OK':
                bpy.ops.blendcap.bvh_importer_missing('INVOKE_DEFAULT',
                                                      bvh_path=full_path)
                return {'CANCELLED'}
            stem = os.path.splitext(self.filename)[0]
            _import_bvh(full_path, stem)
            # Stamp the same provenance + baked-totals custom props the
            # bake / generate import path sets, so the BVH Post
            # Processing sliders (Camera Angle Offset, Head Angle
            # Offset) recognize the rig as a BlendCap armature and
            # the Bake button can rebuild the cache folder path from
            # the stem. Baked totals default to 0 — there's no
            # metadata stored in the BVH file telling us what was
            # previously baked, so the next Bake will write whatever
            # the slider says (instead of adding on top of any prior
            # bake). Acceptable: an extra bake recovers parity.
            imported = bpy.context.active_object
            if imported is not None and imported.type == 'ARMATURE':
                vid = _parse_vid_from_stem(stem)
                if vid is not None:
                    imported["blendcap_source_video"] = vid
                imported["blendcap_bvh_path"] = full_path
                imported["_blendcap_baked_pitch_deg"] = 0.0
                imported["_blendcap_baked_roll_deg"] = 0.0
                imported["_blendcap_baked_head_pitch_deg"] = 0.0
                imported["_blendcap_baked_head_roll_deg"] = 0.0
                # Flag drives the panel hint that warns the user the
                # baked-totals counters are unknown for this rig (we
                # don't store them in the BVH file). Cleared by the
                # Bake operator on its first successful run so the
                # warning fades after the user resyncs the totals.
                imported["_blendcap_library_reimport"] = True
            self.report({'INFO'}, f"Imported {self.filename}")
        return {'FINISHED'}


# --- Save BVH / Save FBX: what the two dialogs offer ------------------------
EXPORT_POSE_REST = '__REST__'
_EXPORT_POSE_ITEMS = []     # keeps the dynamic enum's strings alive


def _export_pose_items(self, context):
    """BlendCap's own rest pose, then every rest pose preset that was saved
    from BlendCap's skeleton (the file written is always that skeleton)."""
    global _EXPORT_POSE_ITEMS
    items = [(EXPORT_POSE_REST, "BlendCap Rest Pose",
              "The pose BlendCap's own skeleton rests in")]
    try:
        from .properties import get_rest_pose_preset_items
        from .operators_rest_pose import preset_fits_blendcap
        for item in get_rest_pose_preset_items(None, context):
            if not item:            # (a separator line of that list)
                continue
            ident, label = item[0], item[1]
            if ident.startswith('__') or not os.path.isfile(ident):
                continue
            if not preset_fits_blendcap(ident):
                continue
            label = label.strip()
            if label.startswith("BlendCap"):
                label = os.path.splitext(os.path.basename(ident))[0]
            items.append((ident, label, "Rest pose preset"))
    except Exception as e:
        print(f"[BlendCap] rest pose presets not listed: {e}")
    _EXPORT_POSE_ITEMS = items
    return items


def _action_span(obj):
    """(first, last) frame of the motion on `obj`, or None."""
    action = obj.animation_data.action if obj.animation_data else None
    if action is None:
        return None
    a, b = action.frame_range
    return int(round(a)), int(round(b))


def _add_pose_frame(context, obj, preset):
    """One extra frame in front of the motion on `obj`, holding a rest
    pose: BlendCap's own (every bone at rest) or a rest pose preset. The
    body stands where the motion starts, at the pose's own height, not
    turned. The motion keeps its frame numbers and the pose goes on the
    frame before it, except when that would be a negative frame: then the
    motion moves one frame later. Returns an error text, or None."""
    from mathutils import Matrix
    from .retarget.fcurves import _iter_action_fcurves
    from .operators_rest_pose import pose_rig_from_preset
    span = _action_span(obj)
    if span is None:
        return "The motion has no keyframes."
    action = obj.animation_data.action
    fcurves = list(_iter_action_fcurves(action))
    first = span[0]
    pose_f = first - 1
    if pose_f < 0:
        shift = -pose_f
        for fc in fcurves:
            for kp in fc.keyframe_points:
                kp.co.x += shift
                kp.handle_left.x += shift
                kp.handle_right.x += shift
            fc.update()
        first += shift
        pose_f = 0
    scene = context.scene
    saved_frame = scene.frame_current
    ts = scene.tool_settings
    prev_auto = ts.use_keyframe_insert_auto
    ts.use_keyframe_insert_auto = False
    try:
        # where the motion starts
        scene.frame_set(first)
        context.view_layer.update()
        mw = obj.matrix_world.copy()
        roots = [pb for pb in obj.pose.bones if pb.parent is None]
        starts = {pb.name: (mw @ pb.matrix).translation.copy() for pb in roots}
        # the pose
        scene.frame_set(pose_f)
        for pb in obj.pose.bones:
            pb.matrix_basis = Matrix()
        context.view_layer.update()
        if preset != EXPORT_POSE_REST:
            _n, err = pose_rig_from_preset(obj, preset)
            if err:
                return err
            context.view_layer.update()
        # stand it where the motion starts (across the floor only)
        turn = mw.to_3x3().inverted()
        for pb in roots:
            d = starts[pb.name] - (mw @ pb.matrix).translation
            d.z = 0.0
            pb.location = pb.location + (
                pb.bone.matrix_local.to_3x3().inverted() @ (turn @ d))
        context.view_layer.update()
        located = {fc.data_path for fc in fcurves
                   if fc.data_path.endswith('.location')}
        for pb in obj.pose.bones:
            mode = pb.rotation_mode
            path = ('rotation_quaternion' if mode == 'QUATERNION'
                    else 'rotation_axis_angle' if mode == 'AXIS_ANGLE'
                    else 'rotation_euler')
            pb.keyframe_insert(path, frame=pose_f)
            if (pb.parent is None
                    or f'pose.bones["{pb.name}"].location' in located):
                pb.keyframe_insert('location', frame=pose_f)
    finally:
        ts.use_keyframe_insert_auto = prev_auto
        scene.frame_set(saved_frame)
    return None


def _save_name(source_bvh):
    """The name a saved file carries: the take's, with "_e1" on the end.

    An FBX keeps two names somebody opening it elsewhere gets to see:
    the rig's (the name of the object it was saved from) and the
    animation's (the name of the scene it was saved from). Both get
    this one. The take's bare name cannot be used: a take loaded in the
    project already has it, and Blender would hand out ".001".

    The number is the first one nothing in the project has, and that is
    not cosmetic. _import_bvh REPLACES an armature that already has the
    name it is given, and gives the new rig the old one's place,
    rotation and scale (what a re-import of a take wants). A rig read
    back from one of our FBX files has the name the next save would
    use: with a fixed name, saving again deleted that rig from the
    project and wrote its rotation into the new file (found 2026-11-20
    in a check: a take saved again after its FBX had been read back and
    turned came out lying down). So "_e2" with "_e1" in the project."""
    base = os.path.splitext(source_bvh)[0]
    n = 1
    while True:
        name = f"{base}_e{n}"
        # (objects for the rig, scenes for the animation; data nobody is
        # using any more does not count, or the number would climb with
        # every rig read back and deleted again)
        if name not in bpy.data.objects and name not in bpy.data.scenes:
            return name
        n += 1


def _scene_frames(scene):
    return scene.frame_start, scene.frame_end, scene.frame_current


def _scene_frames_restore(scene, saved):
    """The save dialogs import the BVH into a throwaway rig, which lengthens
    the scene to fit it, and the FBX exporter reads its range from the
    scene: put the user's own range back."""
    try:
        start, end, current = saved
        scene.frame_end = max(end, scene.frame_start)
        scene.frame_start = start
        scene.frame_end = end
        if scene.frame_current != current:
            scene.frame_set(current)
    except Exception as e:
        print(f"[BlendCap] scene frame range not put back: {e}")


def _save_scene_make(scene, name):
    """A throwaway scene for a save to work in.

    The save imports the BVH into a temporary rig, bakes Root into Hips,
    may add the rest pose frame, and hands the rig to Blender's exporter.
    Three of those four step through every frame of the motion, and in
    the user's own scene every step re-evaluates the whole project: with
    a heavy project open (character rigs, constraints, drivers) a save
    that takes seconds took minutes, Blender not responding meanwhile
    (found live 2026-11-20). In a scene that holds nothing but the
    temporary rig a step costs the same whatever is open. It also means
    the user's scene, selection and frame range are never touched.
    Frame rate and units are copied: both exporters read them. The
    scene's name is what the FBX exporter calls the animation in the
    file, hence the take's (see _save_name)."""
    work = bpy.data.scenes.new(name)
    try:
        work.render.fps = scene.render.fps
        work.render.fps_base = scene.render.fps_base
        work.unit_settings.system = scene.unit_settings.system
        work.unit_settings.scale_length = scene.unit_settings.scale_length
    except Exception as e:
        print(f"[BlendCap] save scene: settings not copied: {e}")
    return work


def _save_scene_drop(work):
    try:
        for o in list(work.collection.all_objects):
            data = o.data
            bpy.data.objects.remove(o, do_unlink=True)
            if (data is not None and data.users == 0
                    and isinstance(data, bpy.types.Armature)):
                bpy.data.armatures.remove(data)
        bpy.data.scenes.remove(work)
    except Exception as e:
        print(f"[BlendCap] save scene not removed: {e}")


def _save_in_own_scene(op, context):
    """Run op._save in a throwaway scene (see _save_scene_make)."""
    src = os.path.join(_bvh_library_dir(), op.source_bvh)
    if os.path.exists(src) and bvh_importer_state() != 'OK':
        # (asked here, in the user's own scene: the window it opens
        # must not belong to a scene that is about to be removed)
        bpy.ops.blendcap.bvh_importer_missing('INVOKE_DEFAULT',
                                              bvh_path=src)
        return {'CANCELLED'}
    own = context.scene
    saved = _scene_frames(own)
    name = _save_name(op.source_bvh)
    win = context.window
    if win is None:
        wins = context.window_manager.windows
        win = wins[0] if wins else None
    if win is None or win.scene is not own:
        # No window to show another scene in: work in the user's own
        # scene, as the save always did.
        try:
            return op._save(context, name)
        finally:
            _scene_frames_restore(own, saved)
    # The window shows the throwaway scene for the length of the save.
    # (Overriding the scene alone is not enough: Blender's BVH importer
    # switches the rig into edit mode, which only works in the scene the
    # window is really showing.) Nothing is drawn meanwhile: the save
    # does not hand control back until the window is on its own scene.
    layer = win.view_layer.name
    work = _save_scene_make(own, name)
    try:
        win.scene = work
        with context.temp_override(window=win):
            return op._save(bpy.context, name)
    finally:
        try:
            win.scene = own
            if layer in own.view_layers:
                win.view_layer = own.view_layers[layer]
        except Exception as e:
            print(f"[BlendCap] the window was not put back on its scene: {e}")
        _save_scene_drop(work)
        _scene_frames_restore(own, saved)


class _LibSaveOptions:
    """What both save dialogs offer."""
    add_pose_frame: bpy.props.BoolProperty(
        name="Add Rest Pose Frame",
        default=False,
        description=("Put one extra frame at the start of the file holding "
                     "a rest pose, for software that sets up retargeting "
                     "from the first frame"))
    pose_preset: bpy.props.EnumProperty(
        name="Pose",
        items=_export_pose_items,
        description="The pose the extra first frame holds")
    export_scale: bpy.props.FloatProperty(
        name="Scale",
        default=1.0, min=0.001, max=1000.0,
        description=("Size of the saved skeleton and its motion. BlendCap "
                     "works in metres; 100 gives centimetres"))

    def _draw_shared(self, layout):
        """Draws what both dialogs share. Returns (the column for a
        checkbox, the column for a labelled field): the dialog's side
        panel is narrow, and a checkbox's name only fits at full width."""
        layout.use_property_decorate = False
        col = layout.column()
        col.prop(self, "add_pose_frame")
        sub = col.column()
        sub.use_property_split = True
        sub.enabled = self.add_pose_frame
        sub.prop(self, "pose_preset")
        col.separator()
        fields = col.column()
        fields.use_property_split = True
        fields.prop(self, "export_scale")
        return col, fields


@orientation_helper(axis_forward='-Z', axis_up='Y')
class BLENDCAP_OT_export_lib_bvh(_LibSaveOptions, bpy.types.Operator):
    bl_idname = "blendcap.export_lib_bvh"
    bl_label = "Save BVH"
    bl_description = ("Save the selected BVH from the library to a chosen "
                      "location (Root locomotion well baked into the hip bone).")
    source_bvh: bpy.props.StringProperty(options={'HIDDEN'})
    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(default="*.bvh", options={'HIDDEN'})

    def invoke(self, context, event):
        self.filepath = self.source_bvh
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def draw(self, context):
        _col, fields = self._draw_shared(self.layout)
        fields.prop(self, "axis_forward")
        fields.prop(self, "axis_up")

    def execute(self, context):
        if self.axis_forward[-1] == self.axis_up[-1]:
            # (the dialog cannot get here: picking one moves the other)
            self.report({'ERROR'},
                        "Forward and Up cannot be the same axis.")
            return {'CANCELLED'}
        return _save_in_own_scene(self, context)

    def _save(self, context, tmp_name):
        src = os.path.join(_bvh_library_dir(), self.source_bvh)
        if not os.path.exists(src):
            self.report({'ERROR'}, f"Source BVH not found: {self.source_bvh}")
            return {'CANCELLED'}
        dst = self.filepath
        if not dst.lower().endswith('.bvh'):
            dst += '.bvh'

        # Import the source BVH into a hidden temp armature so the
        # addon's post-import surgery runs, then bake Root into Hips
        # and re-export via Blender's stock BVH exporter. The library
        # file on disk is untouched.
        prev_active = context.view_layer.objects.active
        prev_selected = [o for o in context.view_layer.objects
                         if o.select_get()]
        if bvh_importer_state() != 'OK':
            bpy.ops.blendcap.bvh_importer_missing('INVOKE_DEFAULT',
                                                  bvh_path=src)
            return {'CANCELLED'}
        try:
            _import_bvh(src, tmp_name)
        except Exception as e:
            self.report({'ERROR'}, f"BVH import failed: {e}")
            return {'CANCELLED'}

        tmp_obj = bpy.data.objects.get(tmp_name)
        if not tmp_obj or tmp_obj.type != 'ARMATURE':
            self.report({'ERROR'},
                        "Imported BVH did not yield an armature.")
            return {'CANCELLED'}

        restructure_err = None
        try:
            _bake_root_into_hips_for_export(tmp_obj)
        except Exception as e:
            restructure_err = str(e)

        pose_err = None
        if restructure_err is None and self.add_pose_frame:
            try:
                pose_err = _add_pose_frame(context, tmp_obj, self.pose_preset)
            except Exception as e:
                pose_err = f"The rest pose frame could not be added ({e})"
        # The whole motion, whatever the scene's frame range is.
        span = _action_span(tmp_obj)

        tmp_action = (tmp_obj.animation_data.action
                      if tmp_obj.animation_data else None)
        tmp_data = tmp_obj.data

        try:
            bpy.ops.object.select_all(action='DESELECT')
        except Exception:
            for o in context.view_layer.objects:
                o.select_set(False)
        tmp_obj.select_set(True)
        context.view_layer.objects.active = tmp_obj

        export_ok = restructure_err is None and pose_err is None
        export_err = (f"Restructure failed: {restructure_err}"
                      if restructure_err else pose_err)
        if export_ok:
            # Blender's BVH exporter writes bone.head_local verbatim with
            # no axis conversion (export_bvh.py around line 81), so a
            # Blender Z-up armature ships out as a Z-up BVH inside a file
            # other tools and Blender's own importer read as Y-up. The
            # result is a skeleton that re-imports lying on its side.
            # Rotate the temp armature into BVH Y-up frame first --
            # axis_conversion(-Z/Y -> -Y/Z) is the same matrix the BVH
            # importer used to convert in the opposite direction and is
            # self-inverse, so applying it here undoes the import-side
            # rotation. transform_apply on an animated armature is the
            # same operation the BVH importer itself performs (see
            # import_bvh.py line 654-655) so the action keys stay
            # visually consistent.
            try:
                from bpy_extras.io_utils import axis_conversion
                # Match the BVH importer exactly: it calls
                # axis_conversion(from_forward=axis_forward,
                # from_up=axis_up) with the to_forward/to_up params
                # left at their library defaults ('Y' and 'Z'),
                # NOT Blender's '-Y'/'Z' world convention. Picking
                # different to_* values yields a matrix that differs
                # by a 180-degree Z rotation -- exactly the symptom
                # that turns the exported skeleton around. Invert to
                # undo the import's rotation; this matrix is not
                # self-inverse.
                # The dialog's Forward and Up; -Z and Y (the default)
                # is the frame described above. Y and Z is Blender's
                # own, where this rotation is no rotation at all.
                M = axis_conversion(
                    from_forward=self.axis_forward, from_up=self.axis_up
                ).to_4x4()
                tmp_obj.matrix_world = M.inverted()
                context.view_layer.update()
                # context.object / selected_objects inside a running
                # operator are FROZEN at button-click time — setting
                # view_layer.objects.active above updates the scene but
                # not this operator's context members. The BVH exporter's
                # poll and save() both read context.object, so without an
                # explicit override the export fails ("poll() failed,
                # context is incorrect") whenever the user clicked with
                # no armature active. Hand both ops the temp armature
                # directly, regardless of click-time state.
                with context.temp_override(
                        object=tmp_obj, active_object=tmp_obj,
                        selected_objects=[tmp_obj],
                        selected_editable_objects=[tmp_obj]):
                    bpy.ops.object.transform_apply(
                        location=False, rotation=True, scale=False)
                    bpy.ops.export_anim.bvh(
                        filepath=dst,
                        global_scale=self.export_scale,
                        frame_start=(span[0] if span
                                     else context.scene.frame_start),
                        frame_end=(span[1] if span
                                   else context.scene.frame_end),
                        rotate_mode='NATIVE',
                        root_transform_only=False,
                    )
            except Exception as e:
                export_ok = False
                export_err = str(e)

        try:
            bpy.data.objects.remove(tmp_obj, do_unlink=True)
        except Exception:
            pass
        if tmp_data and tmp_data.users == 0:
            try:
                bpy.data.armatures.remove(tmp_data)
            except Exception:
                pass
        if tmp_action and tmp_action.users == 0:
            try:
                bpy.data.actions.remove(tmp_action)
            except Exception:
                pass

        try:
            bpy.ops.object.select_all(action='DESELECT')
        except Exception:
            for o in context.view_layer.objects:
                o.select_set(False)
        for o in prev_selected:
            try:
                o.select_set(True)
            except Exception:
                pass
        if prev_active is not None:
            try:
                context.view_layer.objects.active = prev_active
            except Exception:
                pass

        if not export_ok:
            self.report({'ERROR'}, f"BVH export failed: {export_err}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Exported BVH to {dst}")
        return {'FINISHED'}


@orientation_helper(axis_forward='-Z', axis_up='Y')
class BLENDCAP_OT_export_lib_fbx(_LibSaveOptions, bpy.types.Operator):
    bl_idname = "blendcap.export_lib_fbx"
    bl_label = "Save FBX"
    bl_description = ("Convert the selected BVH to an FBX file and save it to "
                      "a chosen location (Root locomotion well baked into the hip bone).")
    source_bvh: bpy.props.StringProperty(options={'HIDDEN'})
    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(default="*.fbx", options={'HIDDEN'})
    add_leaf_bones: bpy.props.BoolProperty(
        name="Add Leaf Bones",
        default=True,
        description=("Add an end bone to every chain (the ends of the "
                     "fingers, the toes, the head). Some software needs "
                     "them to draw the last bone at its real length; "
                     "others show them as extra bones"))

    def invoke(self, context, event):
        self.filepath = os.path.splitext(self.source_bvh)[0] + ".fbx"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def draw(self, context):
        col, fields = self._draw_shared(self.layout)
        fields.prop(self, "axis_forward")
        fields.prop(self, "axis_up")
        col.separator()
        col.prop(self, "add_leaf_bones")

    def execute(self, context):
        if self.axis_forward[-1] == self.axis_up[-1]:
            # (the dialog cannot get here: picking one moves the other)
            self.report({'ERROR'},
                        "Forward and Up cannot be the same axis.")
            return {'CANCELLED'}
        return _save_in_own_scene(self, context)

    def _save(self, context, tmp_name):
        src = os.path.join(_bvh_library_dir(), self.source_bvh)
        if not os.path.exists(src):
            self.report({'ERROR'}, f"Source BVH not found: {self.source_bvh}")
            return {'CANCELLED'}
        dst = self.filepath
        if not dst.lower().endswith('.fbx'):
            dst += '.fbx'

        # Remember selection/active so we can restore it.
        prev_active = context.view_layer.objects.active
        prev_selected = [o for o in context.view_layer.objects if o.select_get()]

        # Quietly import the BVH.
        if bvh_importer_state() != 'OK':
            bpy.ops.blendcap.bvh_importer_missing('INVOKE_DEFAULT',
                                                  bvh_path=src)
            return {'CANCELLED'}
        try:
            _import_bvh(src, tmp_name)
        except Exception as e:
            self.report({'ERROR'}, f"BVH import failed: {e}")
            return {'CANCELLED'}

        tmp_obj = bpy.data.objects.get(tmp_name)
        if not tmp_obj or tmp_obj.type != 'ARMATURE':
            self.report({'ERROR'}, "Imported BVH did not yield an armature.")
            return {'CANCELLED'}

        # Bake Root's locomotion into Hips and drop Root so the FBX
        # encodes the same skeleton other tools will draw. No-op for
        # face-only exports.
        restructure_err = None
        try:
            _bake_root_into_hips_for_export(tmp_obj)
        except Exception as e:
            restructure_err = str(e)

        pose_err = None
        if restructure_err is None and self.add_pose_frame:
            try:
                pose_err = _add_pose_frame(context, tmp_obj, self.pose_preset)
            except Exception as e:
                pose_err = f"The rest pose frame could not be added ({e})"
        # The whole motion, whatever the scene's frame range is.
        span = _action_span(tmp_obj)

        tmp_action = None
        if tmp_obj.animation_data and tmp_obj.animation_data.action:
            tmp_action = tmp_obj.animation_data.action
        tmp_data = tmp_obj.data

        # Select only the temp armature for export.
        try:
            bpy.ops.object.select_all(action='DESELECT')
        except Exception:
            for o in context.view_layer.objects:
                o.select_set(False)
        tmp_obj.select_set(True)
        context.view_layer.objects.active = tmp_obj

        export_ok = restructure_err is None and pose_err is None
        export_err = (f"Restructure failed: {restructure_err}"
                      if restructure_err else pose_err)
        if export_ok:
            try:
                # Same frozen-context issue as the BVH export above, but
                # worse: use_selection=True reads the FROZEN click-time
                # selection, so this could silently export the user's
                # selected objects instead of the temp armature. Override
                # pins both to the temp armature.
                if span:
                    # The exporter bakes the scene's frame range.
                    context.scene.frame_end = max(span[1],
                                                  context.scene.frame_start)
                    context.scene.frame_start = span[0]
                    context.scene.frame_end = span[1]
                with context.temp_override(
                        object=tmp_obj, active_object=tmp_obj,
                        selected_objects=[tmp_obj],
                        selected_editable_objects=[tmp_obj]):
                    bpy.ops.export_scene.fbx(
                        filepath=dst,
                        use_selection=True,
                        object_types={'ARMATURE'},
                        bake_anim=True,
                        bake_anim_use_all_bones=True,
                        bake_anim_use_nla_strips=False,
                        bake_anim_use_all_actions=False,
                        add_leaf_bones=self.add_leaf_bones,
                        global_scale=self.export_scale,
                        axis_forward=self.axis_forward,
                        axis_up=self.axis_up,
                    )
            except Exception as e:
                export_ok = False
                export_err = str(e)

        # Clean up the throwaway armature + its action + armature data.
        try:
            bpy.data.objects.remove(tmp_obj, do_unlink=True)
        except Exception:
            pass
        if tmp_data and tmp_data.users == 0:
            try:
                bpy.data.armatures.remove(tmp_data)
            except Exception:
                pass
        if tmp_action and tmp_action.users == 0:
            try:
                bpy.data.actions.remove(tmp_action)
            except Exception:
                pass

        # Restore selection.
        try:
            bpy.ops.object.select_all(action='DESELECT')
        except Exception:
            for o in context.view_layer.objects:
                o.select_set(False)
        for o in prev_selected:
            try:
                o.select_set(True)
            except Exception:
                pass
        if prev_active is not None:
            try:
                context.view_layer.objects.active = prev_active
            except Exception:
                pass

        if not export_ok:
            self.report({'ERROR'}, f"FBX export failed: {export_err}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Exported FBX to {dst}")
        return {'FINISHED'}


class BLENDCAP_OT_delete_lib_bvh(bpy.types.Operator):
    bl_idname = "blendcap.delete_lib_bvh"
    bl_label = "Delete BVH"
    bl_description = "Delete this BVH file from the project's library folder."
    filename: bpy.props.StringProperty()

    def invoke(self, context, event):
        # Deleting a library file is destructive and has no undo — ask
        # first, like every other destructive button in the addon.
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        full_path = os.path.join(_bvh_library_dir(), self.filename)
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
            except OSError as e:
                self.report({'ERROR'},
                            f"Couldn't delete {self.filename} - it may be "
                            f"open in another program. ({e})")
                return {'CANCELLED'}
            self.report({'INFO'}, f"Deleted {self.filename} from library.")
        _sync_bvh_library(context.scene)
        _tag_redraw('VIEW_3D', 'PROPERTIES', 'SEQUENCE_EDITOR')
        return {'FINISHED'}


class BLENDCAP_OT_donate(bpy.types.Operator):
    """Thin wrapper around wm.url_open so the Donate / Support button
    carries a BlendCap-specific tooltip (wm.url_open's built-in
    description can't be overridden per-call)."""
    bl_idname = "blendcap.donate"
    bl_label = "Donate / Support"
    bl_description = "Open the BlendCap donation page in your browser"
    bl_options = {'REGISTER'}

    def execute(self, context):
        bpy.ops.wm.url_open(url="https://ko-fi.com/arcomade")
        return {'FINISHED'}
