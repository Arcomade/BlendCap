"""Background capture runner, VSE / viewport proxy loaders, and BVH-export
thread machinery.

Operators in operators_pipeline.py spawn `run_mocap_thread` /
`run_bvh_thread` and register `monitor_progress_timer` / `monitor_bvh_timer`.

The two big mutable state dicts (`blendcap_state`, `bvh_state`) live in
helpers.py — every function here mutates them via the imported reference,
which works because Python imports the dict object, not a copy.
"""
import bpy
import os
import math
import shutil
import subprocess
import threading
import time

from .helpers import (
    blendcap_state, bvh_state,
    _addon_dir, _python_exe, _tag_redraw, addon_prefs,
    _host_spawn_prefix, _subprocess_env,
    _seq_accessors, _strip_visible_start, _strip_visible_end,
    _import_bvh, _write_cache_params, bvh_importer_state,
    _ansi_re, _progress_re, _warning_re, _frame_re, _tqdm_re,
    _capture_noise_re,
    format_workspace_status,
)


# ============================================================
# BACKGROUND MOCAP RUNNER & VSE LOADER
# ============================================================
def load_proxy_to_viewport(proxy_path, source_path, strip_name="BlendCap_Preview",
                           source_strip=""):
    if not os.path.exists(proxy_path):
        return

    # Find existing BlendCap viewport empty by property
    empty = None
    for obj in bpy.data.objects:
        if obj.type == 'EMPTY' and "blendcap_source" in obj:
            empty = obj
            break

    if empty is None:
        empty = bpy.data.objects.new(name=strip_name, object_data=None)
        bpy.context.collection.objects.link(empty)
        empty.empty_display_type = 'IMAGE'
        empty.rotation_euler = (math.radians(90), 0, 0)
    else:
        empty.name = strip_name
    # The plane is reused for every File Path capture: whatever it shows
    # next is not the multi-camera capture it may have shown before
    # (multicam.load_to_viewport stamps it again when it is one).
    if "blendcap_multicam" in empty:
        del empty["blendcap_multicam"]

    img_name = os.path.basename(proxy_path)
    if img_name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[img_name])

    img = bpy.data.images.load(proxy_path)
    img.source = 'MOVIE'
    empty.data = img

    movie_frames = getattr(img, 'frame_duration', 0) or 250

    if hasattr(empty, 'image_user'):
        iu = empty.image_user
        iu.use_auto_refresh = True
        iu.use_cyclic = True
        iu.frame_start = 1
        iu.frame_duration = movie_frames

    empty.scale = (5, 5, 5)
    empty.location = (0, 0, 0)

    empty["blendcap_source"] = source_path
    if source_strip:
        empty["blendcap_source_strip"] = source_strip
    _tag_redraw('VIEW_3D')


def _strip_colorspace(strip):
    """Name of a strip's input colour space, '' when it has none."""
    try:
        cs = getattr(strip, "colorspace_settings", None)
        return str(cs.name) if cs is not None else ""
    except Exception:
        return ""


def _set_strip_colorspace(strip, name):
    """Set a strip's input colour space. True only when it changed;
    a name this colour configuration doesn't have is left alone."""
    if not name:
        return False
    try:
        cs = getattr(strip, "colorspace_settings", None)
        if cs is None or cs.name == name:
            return False
        cs.name = name
        return cs.name == name
    except Exception:
        return False


def _replaced_colorspace(strip):
    """Colour spaces to carry over from a strip an overlay is about to
    replace: (space for the overlay, space of the original clip).

    The sequencer guard only acts when the strip count goes up, and a
    replace leaves it unchanged, so without this the overlay comes in
    as plain sRGB and shows tone-mapped under AgX / Filmic while the
    clip it replaced showed as-is.

    The overlay is an 8-bit movie holding the source's own pixels, so
    it takes the replaced strip's space when that strip is a movie
    (an earlier overlay included). An image sequence only passes on a
    round-trip space ('AgX Base sRGB' / 'Filmic sRGB'): a linear EXR
    space would be wrong on the overlay movie.

    The original's space is what Revert puts back. An earlier overlay
    hands on the one it was stamped with."""
    space = _strip_colorspace(strip)
    try:
        is_overlay = "blendcap_source" in strip
    except Exception:
        is_overlay = False
    if is_overlay:
        try:
            original = str(strip.get("blendcap_original_colorspace", "") or "")
        except Exception:
            original = ""
        return space, original
    if getattr(strip, "type", "") == 'IMAGE':
        try:
            from .sequencer_guard import _ROUNDTRIP
            roundtrip = set(_ROUNDTRIP.values())
        except Exception:
            roundtrip = {'AgX Base sRGB', 'Filmic sRGB'}
        return (space if space in roundtrip else ""), space
    return space, space


def load_proxy_to_vse(proxy_path, source_path, source_start, source_max,
                      strip_name="Mocap_Overlay", vse_replace=None,
                      source_strip=""):
    """Replace the original/proxy strip with the new proxy output.
    vse_replace: dict with original strip properties (from blendcap_state).
    """
    if not os.path.exists(proxy_path):
        return

    scene = bpy.context.scene
    if not scene.sequence_editor:
        scene.sequence_editor_create()

    seq = scene.sequence_editor
    seq_all, seq_coll = _seq_accessors(seq)

    # Determine placement from vse_replace info or fallback to playhead
    if vse_replace:
        insert_frame = int(vse_replace.get('frame_final_start', scene.frame_current))
        insert_channel = int(vse_replace.get('channel', 1))
        replace_name = vse_replace.get('strip_name', '')
    else:
        insert_frame = scene.frame_current
        insert_channel = 1
        replace_name = ''

    # Colour space carried over from the strip being replaced, read
    # before it is removed (see _replaced_colorspace).
    overlay_space = original_space = ""

    # Remove the strip we're replacing (by name, use its visible position)
    if replace_name:
        for s in list(seq_all):
            if s.name == replace_name:
                insert_frame = _strip_visible_start(s)
                insert_channel = s.channel
                overlay_space, original_space = _replaced_colorspace(s)
                seq_coll.remove(s)
                break
    else:
        # Fallback: remove any existing strip with same proxy filepath or same source
        for s in list(seq_all):
            if s.type == 'MOVIE':
                sp = os.path.normpath(bpy.path.abspath(s.filepath))
                if sp == os.path.normpath(bpy.path.abspath(proxy_path)):
                    insert_channel = s.channel
                    insert_frame = _strip_visible_start(s)
                    overlay_space, original_space = _replaced_colorspace(s)
                    seq_coll.remove(s)
                    break
                if "blendcap_source" in s and os.path.normpath(bpy.path.abspath(str(s["blendcap_source"]))) == os.path.normpath(bpy.path.abspath(source_path)):
                    insert_channel = s.channel
                    insert_frame = _strip_visible_start(s)
                    overlay_space, original_space = _replaced_colorspace(s)
                    seq_coll.remove(s)
                    break

    strip = seq_coll.new_movie(strip_name, proxy_path, insert_channel, int(insert_frame))

    # Force channel in case Blender auto-placed on a higher one
    if strip.channel != insert_channel:
        strip.channel = insert_channel

    # Show the overlay the way the clip it replaced was shown.
    _set_strip_colorspace(strip, overlay_space)

    # Scale strip to fit project frame (letterbox/pillarbox, never over-expand)
    scene_w = scene.render.resolution_x
    scene_h = scene.render.resolution_y
    if hasattr(strip, 'transform') and strip.transform is not None:
        strip_w = strip_h = 0
        if hasattr(strip, 'orig_width'):
            strip_w, strip_h = strip.orig_width, strip.orig_height
        if (strip_w <= 0 or strip_h <= 0) and hasattr(strip, 'elements') and len(strip.elements) > 0:
            strip_w, strip_h = strip.elements[0].orig_width, strip.elements[0].orig_height
        if strip_w > 0 and strip_h > 0 and scene_w > 0 and scene_h > 0:
            fit = min(scene_w / strip_w, scene_h / strip_h)
            strip.transform.scale_x = fit
            strip.transform.scale_y = fit

    # Store original video info on the proxy for revert and re-run
    strip["blendcap_source"] = source_path
    strip["blendcap_type"] = blendcap_state.get('_strip_type', 'capture')
    # Stamp the bare (un-prefixed) source-strip identifier so a later
    # re-capture of this proxy still maps to the original cache folder
    # rather than forking based on the proxy's current name.
    if source_strip:
        strip["blendcap_source_strip"] = source_strip
    # The original clip's colour space, for Revert.
    if original_space:
        strip["blendcap_original_colorspace"] = original_space

    if vse_replace:
        orig_type = vse_replace.get('strip_type', 'MOVIE')
        strip["blendcap_original_type"] = orig_type
        strip["blendcap_original_channel"] = int(vse_replace.get('channel', insert_channel))

        if orig_type == 'IMAGE':
            strip["blendcap_original_directory"] = vse_replace.get('directory', '')
            strip["blendcap_original_elements"] = "|".join(vse_replace.get('elements', []))
            strip["blendcap_original_frame_final_end"] = int(vse_replace.get('frame_final_end', 0))
            strip["blendcap_original_image_frame_start"] = int(vse_replace.get('image_frame_start', 0))
            strip["blendcap_original_image_offset_start"] = int(vse_replace.get('image_offset_start', 0))
            strip["blendcap_original_image_offset_end"] = int(vse_replace.get('image_offset_end', 0))
        else:
            strip["blendcap_original_offset_start"] = int(vse_replace.get('offset_start', source_start))
            strip["blendcap_original_offset_end"] = int(vse_replace.get('offset_end', 0))
            strip["blendcap_original_duration"] = int(vse_replace.get('frame_final_duration', source_max))
            strip["blendcap_original_vis_start"] = int(vse_replace.get('frame_final_start', insert_frame))
            strip["blendcap_original_vis_end"] = int(vse_replace.get('frame_final_end', insert_frame + source_max))
            strip["blendcap_original_frame_start"] = int(vse_replace.get('frame_start',
                                                         insert_frame - vse_replace.get('offset_start', source_start)))
            strip["blendcap_original_frame_duration"] = int(vse_replace.get('frame_duration', 0))
    else:
        strip["blendcap_original_type"] = 'MOVIE'
        strip["blendcap_original_offset_start"] = source_start
        strip["blendcap_original_offset_end"] = 0
        strip["blendcap_original_duration"] = source_max
        strip["blendcap_original_vis_start"] = int(insert_frame)
        strip["blendcap_original_vis_end"] = int(insert_frame + source_max) if source_max > 0 else 0
        strip["blendcap_original_frame_start"] = int(insert_frame - source_start)
        strip["blendcap_original_frame_duration"] = 0
        strip["blendcap_original_channel"] = insert_channel

    # Snap playhead to start of new strip - unless the user turned it off
    # (Disable playhead auto-snap add-on pref).
    _prefs = addon_prefs()
    if not (_prefs and _prefs.disable_timeline_autosnap):
        scene.frame_current = int(insert_frame)

    # Make the new strip the active selection
    seq.active_strip = strip

    _tag_redraw('SEQUENCE_EDITOR')

    if "Motion Capture" in bpy.data.workspaces:
        bpy.context.window.workspace = bpy.data.workspaces["Motion Capture"]


def _revert_proxy_strip(strip, seq_coll, seq_all):
    """Revert a proxy strip back to its original video/image sequence.
    Returns new strip on success, None on failure."""
    if "blendcap_source" not in strip:
        return None

    original_path = bpy.path.abspath(str(strip["blendcap_source"]))
    original_type = str(strip.get("blendcap_original_type", "MOVIE"))

    # Clip identity carried by the proxy — re-stamped onto the restored
    # strip at the end so a later re-capture maps back to this clip's
    # cache folder (and skips the capture collision guard). blendcap_
    # source is deliberately NOT restored: the strip is the original
    # video again, so capture must read trim from the strip itself,
    # not from stored proxy values.
    source_strip_id = str(strip.get("blendcap_source_strip", "") or "")

    # Colour space to put back on the restored strip: the original's
    # own, stamped at capture. An overlay from before that stamp hands
    # a movie original the space the overlay carries now (an image
    # sequence keeps whatever Blender gives it). Read before the
    # overlay is removed.
    restore_space = str(strip.get("blendcap_original_colorspace", "") or "")
    if not restore_space and original_type != 'IMAGE':
        restore_space = _strip_colorspace(strip)

    # Use proxy's current visible position (user may have moved it)
    proxy_position = _strip_visible_start(strip)
    insert_channel = strip.channel

    # Use stored original channel (proxy may be on a different channel)
    orig_channel = int(strip.get("blendcap_original_channel", strip.channel))

    if original_type == 'IMAGE':
        # --- IMAGE STRIP REVERT ---
        insert_channel = orig_channel
        orig_dir = str(strip.get("blendcap_original_directory", original_path))
        elem_str = str(strip.get("blendcap_original_elements", ""))
        elements = elem_str.split("|") if elem_str else []
        orig_frame_start = int(strip.get("blendcap_original_image_frame_start", proxy_position))
        orig_offset_start = int(strip.get("blendcap_original_image_offset_start", 0))
        orig_offset_end = int(strip.get("blendcap_original_image_offset_end", 0))

        if not orig_dir or not os.path.isdir(orig_dir):
            return None
        if not elements:
            # Fallback: list directory for common image extensions
            img_exts = {'.png', '.jpg', '.jpeg', '.exr', '.tiff', '.tif', '.bmp', '.dpx', '.hdr'}
            elements = sorted(f for f in os.listdir(orig_dir)
                              if os.path.splitext(f)[1].lower() in img_exts)
        if not elements:
            return None

        # Compute frame_start so visible start lands at proxy_position
        insert_frame = proxy_position - orig_offset_start

        # Prefer the source-side strip name we stamped at capture time
        # so revert restores the user's original strip identity (e.g.
        # "myclip_001" from a split). Falls back to the directory name
        # for proxies created before that prop existed.
        strip_name = str(strip.get("blendcap_source_strip", "") or "") \
            or os.path.basename(orig_dir.rstrip(os.sep))
        seq_coll.remove(strip)

        first_file = os.path.join(orig_dir, elements[0])
        new_strip = seq_coll.new_image(strip_name, first_file, insert_channel, insert_frame)
        new_strip.directory = orig_dir
        for fname in elements[1:]:
            new_strip.elements.append(fname)

        # Restore trim (deprecated on ImageStrip, suppressed at module level)
        try:
            if orig_offset_start > 0:
                new_strip.frame_offset_start = orig_offset_start
            if orig_offset_end > 0:
                new_strip.frame_offset_end = orig_offset_end
        except Exception:
            pass

        # Force channel after trim (Blender may auto-place on higher channel)
        if new_strip.channel != insert_channel:
            new_strip.channel = insert_channel

    else:
        # --- MOVIE STRIP REVERT ---
        if not os.path.exists(original_path):
            return None

        orig_vis_start = int(strip.get("blendcap_original_vis_start", proxy_position))
        orig_vis_end = int(strip.get("blendcap_original_vis_end", 0))
        orig_offset_start = int(strip.get("blendcap_original_offset_start", 0))
        orig_frame_duration = int(strip.get("blendcap_original_frame_duration", 0))
        insert_channel = orig_channel

        # Prefer the stamped source-strip identifier (preserves the
        # user's original strip name across capture/revert cycles).
        vid_name = str(strip.get("blendcap_source_strip", "") or "") \
            or os.path.splitext(os.path.basename(original_path))[0]
        seq_coll.remove(strip)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            # Probe frame_duration on a temp strip (high channel to avoid conflicts)
            probe = seq_coll.new_movie("_blendcap_probe", original_path, 128, 1)
            new_frame_dur = int(probe.frame_duration)
            seq_coll.remove(probe)

            # Scale offset to match the new frame space
            if orig_frame_duration > 0 and new_frame_dur != orig_frame_duration:
                ratio = new_frame_dur / orig_frame_duration
                scaled_offset = int(round(orig_offset_start * ratio))
            else:
                scaled_offset = orig_offset_start

            # Create final strip at correct position and channel
            new_frame_start = orig_vis_start - scaled_offset
            new_strip = seq_coll.new_movie(vid_name, original_path, insert_channel, new_frame_start)

            if orig_vis_end > orig_vis_start:
                new_strip.frame_final_end = orig_vis_end
                new_strip.frame_final_start = orig_vis_start

            # Blender may auto-place on a higher channel if the full
            # (untrimmed) range overlapped with adjacent strips. After
            # trimming, force the strip to the correct channel.
            if new_strip.channel != insert_channel:
                new_strip.channel = insert_channel

    if source_strip_id:
        new_strip["blendcap_source_strip"] = source_strip_id

    _set_strip_colorspace(new_strip, restore_space)

    # Scale to fit project frame
    scene = bpy.context.scene
    scene_w = scene.render.resolution_x
    scene_h = scene.render.resolution_y
    if hasattr(new_strip, 'transform') and new_strip.transform is not None:
        strip_w = strip_h = 0
        if hasattr(new_strip, 'orig_width'):
            strip_w, strip_h = new_strip.orig_width, new_strip.orig_height
        if (strip_w <= 0 or strip_h <= 0) and hasattr(new_strip, 'elements') and len(new_strip.elements) > 0:
            strip_w, strip_h = new_strip.elements[0].orig_width, new_strip.elements[0].orig_height
        if strip_w > 0 and strip_h > 0 and scene_w > 0 and scene_h > 0:
            fit = min(scene_w / strip_w, scene_h / strip_h)
            new_strip.transform.scale_x = fit
            new_strip.transform.scale_y = fit

    return new_strip


def monitor_progress_timer():
    if blendcap_state['running']:
        pct = blendcap_state['progress']
        status = blendcap_state.get('status_text', '')
        # Warn-only watchdog: a healthy capture prints constantly; long
        # silence usually means a wedged GPU/driver or a stalled model
        # load. Never auto-kill (first runs on slow disks ARE silent for
        # minutes) — just tell the user Cancel is available. The worker
        # overwrites status_text on the next real output line.
        last = blendcap_state.get('last_output_time')
        if (last and (time.time() - last) > 600
                and "no output" not in status):
            status += "  (no output for 10+ min - Cancel if stuck)"
            blendcap_state['status_text'] = status
        txt = format_workspace_status("BlendCap Capture", pct, status)
        try:
            for win in bpy.context.window_manager.windows:
                if win.workspace:
                    win.workspace.status_text_set(txt)
        except Exception: pass
        _tag_redraw('VIEW_3D', 'PROPERTIES')
        return 0.25

    # Finished — clear status bar
    try:
        for win in bpy.context.window_manager.windows:
            if win.workspace:
                win.workspace.status_text_set(None)
    except Exception: pass

    # Face data may have landed in an existing capture folder: let the
    # face-data dropdowns rebuild on their next draw.
    try:
        from .properties import face_cache_changed
        face_cache_changed()
    except Exception:
        pass

    # Compute strip name from the resolved source-strip identifier
    # (not the file basename) so Blender-split source strips get
    # separate proxy names like Capture_myclip_001. Falls back to
    # the file stem only if state wasn't plumbed (legacy call path).
    clip_name = blendcap_state.get('source_strip_name', '')
    if not clip_name:
        src = blendcap_state.get('source_path', '')
        clip_name = os.path.splitext(os.path.basename(src))[0] if src else "clip"
    is_preview = blendcap_state.get('is_preview', False)
    if is_preview:
        parts = []
        if blendcap_state.get('track_body'): parts.append("Body")
        if blendcap_state.get('track_face'): parts.append("Face")
        track_type = "+".join(parts) or "Track"
        strip_name = f"Preview_{clip_name}_{track_type}"
    else:
        strip_name = f"Capture_{clip_name}"
    blendcap_state['_strip_type'] = 'preview' if is_preview else 'capture'

    # A multi-camera capture (multicam.py): its own loader, per-camera
    # cache records, and none of the single-camera cancel dialogs.
    multicam_run = blendcap_state.get('multicam')
    if multicam_run:
        from . import multicam as _mc
        # Cameras that finished are reusable whatever happened after.
        _mc.write_camera_params(blendcap_state)

    if blendcap_state['return_code'] == 0 and multicam_run:
        _forget_sac_block()
        print("\n[BlendCap] Complete. Loading result...")
        _mc.load_result(blendcap_state['proxy_path'], multicam_run['record'])
        lines = _mc.result_lines(blendcap_state)
        blendcap_state['multicam'] = None
        blendcap_state['warnings'] = lines + [
            w for w in blendcap_state.get('warnings', []) if w not in lines]
        if blendcap_state['warnings']:
            def _deferred_multicam_popup():
                try:
                    bpy.ops.blendcap.show_warnings('INVOKE_DEFAULT')
                except Exception as e:
                    print(f"[BlendCap] Could not show warning popup: {e}")
                return None
            bpy.app.timers.register(_deferred_multicam_popup, first_interval=0.3)

    elif blendcap_state['return_code'] == 0:
        _forget_sac_block()   # the engine ran: nothing is blocked now
        proxy = blendcap_state['proxy_path']
        if proxy and os.path.exists(proxy):
            print("\n[BlendCap] Complete. Loading result...")
            if blendcap_state['mode'] == 'VSE':
                load_proxy_to_vse(
                    proxy, blendcap_state['source_path'],
                    blendcap_state['source_start'], blendcap_state['source_max'],
                    strip_name=strip_name,
                    vse_replace=blendcap_state.get('vse_replace'),
                    source_strip=clip_name)
            else:
                load_proxy_to_viewport(proxy, blendcap_state['source_path'],
                                       strip_name=strip_name,
                                       source_strip=clip_name)
        else:
            print("\n[BlendCap] Capture complete.")

        # Record the frame parameters so future runs can validate cache.
        # Only update when tracking actually ran (body/face step), not when
        # just the overlay regenerated — otherwise a shorter re-run would
        # overwrite the wider params and invalidate the still-valid NPZ cache.
        # PREVIEW runs use the same scripts (same step labels) but write no
        # NPZs — their params must never be recorded, or a preview after an
        # FPS/trim/settings change stamps the CURRENT params over the OLD
        # NPZ and the next capture silently reuses stale tracking data.
        output_dir = blendcap_state.get('output_dir', '')
        ran_labels = blendcap_state.get('step_labels', [])
        if (output_dir and not blendcap_state.get('is_preview')
                and ('body' in ran_labels or 'face' in ran_labels)):
            _write_cache_params(output_dir,
                                blendcap_state.get('cache_start_frame', 0),
                                blendcap_state.get('cache_max_frames', 0),
                                blendcap_state.get('cache_img_offset_start', 0),
                                blendcap_state.get('cache_img_offset_end', 0),
                                blendcap_state.get('cache_project_fps', 0.0),
                                blendcap_state.get('cache_capture_sig', ''),
                                blendcap_state.get('cache_video_path', ''))

        # A face step that failed and was dropped: say so first, above
        # any tracking warnings (the dialog draws these lines even when
        # it collapses the rest into the multi-subject notice).
        fallback = blendcap_state.get('face_fallback')
        if fallback:
            fb_lines = _face_fallback_lines(fallback, output_dir)
            blendcap_state['face_fallback_lines'] = fb_lines
            blendcap_state['warnings'] = fb_lines + [
                w for w in blendcap_state.get('warnings', [])
                if w not in fb_lines]

        # Show multi-subject warning popup if any warnings accumulated
        warnings = blendcap_state.get('warnings', [])
        if warnings:
            def _deferred_warning_popup():
                try:
                    bpy.ops.blendcap.show_warnings('INVOKE_DEFAULT')
                except Exception as e:
                    print(f"[BlendCap] Could not show warning popup: {e}")
                return None
            bpy.app.timers.register(_deferred_warning_popup, first_interval=0.3)

    elif blendcap_state['return_code'] == -2:
        # Cancel cleanup
        output_dir = blendcap_state.get('output_dir', '')
        completed = blendcap_state.get('steps_completed', [])
        labels = blendcap_state.get('step_labels', [])
        outputs = blendcap_state.get('step_outputs', [])
        completed_labels = [labels[i] for i in completed if i < len(labels)]
        body_done = 'body' in completed_labels
        face_done = 'face' in completed_labels
        n_total_steps = len(labels)

        if multicam_run and len(completed) >= n_total_steps \
                and os.path.exists(blendcap_state.get('proxy_path') or ""):
            print("\n[BlendCap] Cancel arrived after completion. Loading result...")
            _mc.load_result(blendcap_state['proxy_path'], multicam_run['record'])
            blendcap_state['multicam'] = None

        # Cancel race condition: if all pipeline steps completed, treat as success
        elif not multicam_run and (len(completed) >= n_total_steps or (len(completed) == n_total_steps - 1 and labels[-1] == 'overlay')):
            # All real work done, overlay might have been interrupted
            # Try to load whatever we have
            proxy = blendcap_state['proxy_path']
            body_npz = os.path.join(output_dir, "mhr_frames.npz") if output_dir else ""
            face_npz = os.path.join(output_dir, "face_frames.npz") if output_dir else ""

            # If overlay exists, use it; otherwise generate body-only overlay
            if proxy and os.path.exists(proxy) and os.path.getsize(proxy) > 1000:
                print("\n[BlendCap] Cancel arrived after completion. Loading result...")
                if blendcap_state['mode'] == 'VSE':
                    load_proxy_to_vse(
                        proxy, blendcap_state['source_path'],
                        blendcap_state['source_start'], blendcap_state['source_max'],
                        strip_name=strip_name,
                        vse_replace=blendcap_state.get('vse_replace'),
                        source_strip=clip_name)
                else:
                    load_proxy_to_viewport(proxy, blendcap_state['source_path'],
                                           strip_name=strip_name,
                                           source_strip=clip_name)
            elif body_npz and os.path.exists(body_npz) and blendcap_state.get('source_path'):
                # Generate body-only overlay as fallback
                print("\n[BlendCap] Overlay interrupted. Generating body-only overlay...")
                _run_fallback_overlay(output_dir, body_npz, face_npz, strip_name)
            else:
                print("\n[BlendCap] Cancelled. No usable output.")

        elif (not multicam_run
                and blendcap_state.get('track_body') and blendcap_state.get('track_face')
                and not blendcap_state.get('is_preview') and body_done and not face_done):
            print("\n[BlendCap] Body complete, face cancelled. Asking user...")
            def _deferred_cancel_dialog():
                try:
                    bpy.ops.blendcap.cancel_cleanup('INVOKE_DEFAULT')
                except Exception as e:
                    print(f"[BlendCap] Could not show dialog: {e}")
                    print("[BlendCap] Body data kept by default.")
                return None
            bpy.app.timers.register(_deferred_cancel_dialog, first_interval=0.3)
        else:
            # Delete only what THIS run wrote (file newer than launch).
            # On a re-capture, the not-yet-reached outputs on disk are
            # the PREVIOUS capture's good files — a cancel must leave
            # those exactly as they were.
            run_started = blendcap_state.get('run_started', 0.0)
            for i, out_path in enumerate(outputs):
                if i not in completed and out_path and os.path.exists(out_path):
                    try:
                        if run_started and os.path.getmtime(out_path) < run_started:
                            print(f"  Kept previous capture's: "
                                  f"{os.path.basename(out_path)}")
                            continue
                    except Exception:
                        pass
                    try:
                        os.remove(out_path)
                        print(f"  Removed partial: {os.path.basename(out_path)}")
                    except Exception: pass
            print("\n[BlendCap] Cancelled. Partial data removed.")
    else:
        rc = blendcap_state['return_code']
        print(f"\n[BlendCap] Failed (code {rc}). Check System Console.")
        # Surface the failure in a dialog instead of console-only: reuse
        # the warnings popup with the subprocess's last meaningful lines
        # so the user sees WHY without hunting for the System Console.
        tail = [t for t in blendcap_state.get('error_tail', []) if t][-5:]
        tail = _tail_with_exit_code(tail, rc)
        _remember_sac_block(tail)
        out_dir = blendcap_state.get('output_dir', '')
        log_path = _ensure_debug_log(out_dir, blendcap_state.get('last_cmd'),
                                     rc, tail)
        # A recognised failure gets the plain explanation ALONE (user
        # decision 2026-08-18: no raw clutter under it - the raw text
        # already went to the console and the debug log, so nothing is
        # lost if a translation ever mislabels). Unrecognised failures
        # keep the raw tail: for those it IS the useful content.
        lines = _plain_failure_explanation(tail)
        if not lines:
            lines = [f"Capture failed (exit code {rc})."]
            if tail:
                lines.append("Last output:")
                lines.extend(f"  {t[:90]}" for t in tail)
            if log_path:
                lines.append("Full log: blendcap_capture_debug.log in the "
                             "capture's cache folder.")
        blendcap_state['warnings'] = lines

        def _deferred_failure_popup():
            try:
                bpy.ops.blendcap.show_warnings('INVOKE_DEFAULT')
            except Exception as e:
                print(f"[BlendCap] Could not show failure dialog: {e}")
            return None
        bpy.app.timers.register(_deferred_failure_popup, first_interval=0.3)
    return None


# Plain-English translations for helper-process failures the user can
# actually do something about. Each entry is (needles, lines): if ANY
# needle appears in the child's captured output, those lines REPLACE
# the raw output in the failure dialog (the raw text still reaches the
# console and the debug log). lines=None is a FORCED-RAW guard: match
# means "never translate this", protecting against a broader entry
# further down claiming it.
#
# FIRST MATCH WINS, so these are ordered most-specific first - a missing
# VC++ runtime reports ImportError, which must not be read as a missing
# package, and an out-of-memory during model load must not be read as a
# model-file problem just because a checkpoint path is in the tail.
#
# Two whole FAMILIES are covered generically (user decision 2026-08-18:
# any venv-import or model-file problem should get the reinstall
# advice, not just enumerated instances):
#   - import failures: DLL load failed / ModuleNotFoundError / any
#     other ImportError. If the child died importing, the install is
#     broken by definition.
#   - model files missing or damaged: one merged message, because the
#     user action is identical either way. Missing-side needles are
#     HARVESTED from the real scripts: "Checkpoint not found" and
#     "YOLO weights not found" (save_mhr_data), "mhr_model.pt"
#     (modes.py's '<path> not found'), "Unable to open file at"
#     (mediapipe's exact FileNotFoundError text for a missing .task,
#     reproduced against the venv).
#
# Everything else falls through to the raw tail ON PURPOSE: prescribing
# a 13 GB reinstall for an out-of-memory, an unreadable video, or a
# genuine pipeline bug is confidently-wrong advice, and the raw text is
# how new failure classes become new entries here (one per ticket).
_FAILURE_HINTS = (
    (
        # FORCED RAW: out-of-memory tracebacks routinely contain
        # checkpoint paths and torch module names - nothing below may
        # claim them. Reinstalling does not add VRAM.
        ("out of memory", "OutOfMemoryError", "MemoryError"),
        None,
    ),
    (
        # Windows Smart App Control (Windows 11) refused to load one of
        # the engine's unsigned open-source binaries. Must sit ABOVE the
        # DLL row: its text also reads "DLL load failed", and the fix is
        # a Windows setting, not a reinstall (helpers.smart_app_control_*).
        ("Application Control policy",),
        [
            "Windows Smart App Control blocked a file BlendCap needs.",
            "You will need to turn it off to continue: Go to Windows",
            "Security > App & browser control > Smart App Control",
            "settings > Off. This is separate from your antivirus,",
            "which will keep running as usual.",
        ],
    ),
    (
        # The model ran but returned NaN / nonsense on every frame
        # (save_mhr_data's invalid-output guard, support item 6). A
        # hardware / driver limit, not an install problem: a reinstall
        # changes nothing, the CPU backend does.
        ("producing invalid results on this computer",),
        [
            "The tracking model is producing invalid results on this",
            "computer, which usually means the graphics card or its",
            "driver can't run it correctly (common on integrated",
            "graphics). Open BlendCap's preferences (Edit > Preferences",
            "> Add-ons > BlendCap) and set Capture Backend to CPU Only,",
            "or use a supported NVIDIA GPU.",
        ],
    ),
    (
        # The DLL a package needs is absent, not the package itself.
        # Overwhelmingly the Microsoft VC++ runtime on a clean machine.
        # The installer bootstraps it, so Reinstall is the easy fix and
        # is safer to recommend than sending anyone to a download page.
        ("DLL load failed",),
        [
            "A component Windows needs to run BlendCap is missing",
            "(the Microsoft Visual C++ Redistributable).",
            "To fix this issue, open BlendCap's preferences",
            "(Edit > Preferences > Add-ons > BlendCap) and run",
            "\"Reinstall Dependencies\", which installs it for you.",
        ],
    ),
    (
        # The import-failure family. ModuleNotFoundError is the
        # missing-package case; bare ImportError catches every other
        # way an import can die (incl. lazy imports mid-run, e.g.
        # scipy inside the grounding passes). If the child died
        # importing, the code never ran - that is an install problem
        # by definition. is_dependencies_installed() cannot catch this
        # and is deliberately not changed to try (constraint 40).
        ("ModuleNotFoundError", "No module named", "ImportError"),
        [
            "The dependency install looks incomplete.",
            "To fix this issue, open BlendCap's preferences",
            "(Edit > Preferences > Add-ons > BlendCap) and run",
            "\"Reinstall Dependencies\" - or uninstall the",
            "dependencies and install them again.",
        ],
    ),
    (
        # The model-file family, missing and damaged merged: the user
        # action is identical, so distinguishing them in the dialog is
        # precision nobody benefits from. Missing-side needles are the
        # helpers' own printed lines (harvested, see header comment);
        # damaged-side needles are the loader exceptions for truncated
        # or corrupt files, where the existence check passes and the
        # contents choke the loader.
        ("Checkpoint not found", "YOLO weights not found", "mhr_model.pt",
         "Unable to open file at",
         "UnpicklingError", "pickle data was truncated", "invalid load key",
         "PytorchStreamReader", "central directory", "SafetensorError",
         "HeaderTooLarge"),
        [
            "Some of BlendCap's tracking model files are missing",
            "or damaged.",
            "To fix this issue, open BlendCap's preferences",
            "(Edit > Preferences > Add-ons > BlendCap) and run",
            "\"Reinstall Dependencies\" - it re-downloads only",
            "what is missing or broken.",
        ],
    ),
    (
        # The card is real and the driver works, but the installed
        # wheels carry no kernels for its architecture - the GTX 10 /
        # Pascal case the cu128 pin dropped. is_available() says yes,
        # which is why this only surfaces at run time.
        ("no kernel image is available", "CUDA capability", "kernel image"),
        [
            "Your graphics card isn't supported by the AI",
            "components BlendCap installed.",
            "To fix this issue, set the Capture Backend to",
            "\"CPU Only\" in BlendCap's preferences - it is much",
            "slower, but it runs on any machine.",
        ],
    ),
)


# Windows exit status when the helper process itself was refused by
# Smart App Control / App Control: "An Application Control policy has
# blocked this file" (0xC0E90002). The process never ran, so there is
# no output and no child-written log; the code is the only evidence.
_SAC_EXIT_CODES = {0xC0E90002}
_SAC_EXIT_LINE = ("An Application Control policy has blocked this file "
                  "(exit code 0xC0E90002: the helper process was not "
                  "allowed to start)")


def _tail_with_exit_code(tail, rc):
    """The hint table matches on text; a blocked-at-launch process
    leaves none, so add a line that carries the exit code's meaning."""
    try:
        if int(rc) & 0xFFFFFFFF in _SAC_EXIT_CODES:
            return list(tail) + [_SAC_EXIT_LINE]
    except (TypeError, ValueError):
        pass
    return list(tail)


def _ensure_debug_log(out_dir, cmd, rc, tail):
    """The failure dialog promises blendcap_capture_debug.log in the
    cache folder. The helper (save_mhr_data.py) writes it once it is
    running; a process that never started leaves nothing, so write one
    from what the addon knows. Returns the path when a log exists."""
    if not out_dir:
        return ""
    p = os.path.join(out_dir, "blendcap_capture_debug.log")
    if os.path.isfile(p):
        # The helper wrote its log while it ran, but a hard crash (a
        # SIGABRT from a native library, exit -6) never reaches its
        # own log; the addon's captured tail is the only record of the
        # last lines. Append them so the file the dialog points at
        # really holds the failure (support item 8: a customer went to
        # the log and found nothing about the crash).
        try:
            with open(p, "a", encoding="utf-8") as f:
                f.write("\n--- added by the addon after the helper exited ---\n")
                try:
                    f.write(f"exit code: {rc} (0x{int(rc) & 0xFFFFFFFF:08X})\n")
                except (TypeError, ValueError):
                    f.write(f"exit code: {rc}\n")
                if tail:
                    f.write("last output seen by the addon:\n")
                    for t in tail:
                        f.write(f"  {t}\n")
        except Exception:
            pass
        return p
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write("BlendCap capture failure (written by the addon: the "
                    "helper process left no log of its own)\n")
            f.write(f"command: {cmd or '?'}\n")
            try:
                f.write(f"exit code: {rc} (0x{int(rc) & 0xFFFFFFFF:08X})\n")
            except (TypeError, ValueError):
                f.write(f"exit code: {rc}\n")
            if tail:
                f.write("last output:\n")
                for t in tail:
                    f.write(f"  {t}\n")
            else:
                f.write("last output: (none - the process produced nothing, "
                        "most likely it was not allowed to start)\n")
        return p
    except Exception:
        return ""


def _remember_sac_block(tail):
    """An observed Smart App Control block turns the preferences note
    red (helpers.set_sac_block_seen); a later successful capture clears
    it (_forget_sac_block)."""
    if any("Application Control" in t for t in tail):
        try:
            from .helpers import set_sac_block_seen
            set_sac_block_seen(True)
        except Exception:
            pass


def _forget_sac_block():
    try:
        from .helpers import set_sac_block_seen, sac_block_seen
        if sac_block_seen():
            set_sac_block_seen(False)
    except Exception:
        pass


def _plain_failure_explanation(tail):
    """Plain-English REPLACEMENT for the failure dialog's content, or []
    when the failure is not one we recognise (caller then shows the raw
    tail - the honest answer for an unknown failure).

    Matching is on the child's own error text because that is the only
    evidence available after the fact - the addon cannot inspect a venv
    it did not build. A forced-raw entry (lines=None) returns [] so
    dangerous lookalikes are never translated."""
    blob = "\n".join(tail)
    for needles, lines in _FAILURE_HINTS:
        if any(n in blob for n in needles):
            return list(lines) if lines is not None else []
    return []


# ------------------------------------------------------------
# Body + face capture whose face step failed: finish as body-only
# ------------------------------------------------------------
# The body step is the long one. A face step that then finds no face
# in any frame (or dies for its own reasons) used to fail the whole
# capture: no overlay, an error dialog, and - because the capture was
# never recorded as finished - a body NPZ the next run tracked again
# from scratch. Now the run carries on without the face.

# save_face_data.py's own line when zero frames had a face.
_NO_FACE_NEEDLE = "No frames with detected faces"


def _script_name(cmd):
    return os.path.basename(cmd[1]) if len(cmd) > 1 else ""


def _cmd_value(cmd, flag):
    try:
        return cmd[cmd.index(flag) + 1]
    except (ValueError, IndexError):
        return ""


def _face_step_fallback(cmds, step_idx, st, rc):
    """True when the failed step is a capture's face step and the run
    can finish as body-only. Decided from the commands themselves:

      - the failed step is save_face_data.py, not a preview;
      - body tracking was asked for (a face-only capture has nothing to
        fall back to, even when an old body NPZ is lying in the folder);
      - it is the only face step (single-camera), or the run is a
        multi-camera capture, where each camera's face step stands on
        its own: a camera that sees no face (one behind the performer)
        must not fail the whole take;
      - a later overlay step draws from a body NPZ that exists.

    On True the remaining overlay steps lose their --face input, and a
    face NPZ already in the folder is removed: the run re-tracked the
    face because that file did not match this capture, and the capture
    is about to be recorded as finished, which would make a later run
    (and BVH export) take the old face data as this capture's."""
    cmd = cmds[step_idx]
    if _script_name(cmd) != "save_face_data.py" or "--preview" in cmd:
        return False
    if not st.get('track_body'):
        return False
    names = [_script_name(c) for c in cmds]
    # (a Multi-Camera & Multi-Person capture matches its cameras' people
    # where an ordinary one merges them)
    multicam = "two_camera_merge.py" in names or "people_cameras.py" in names
    if not multicam and names.count("save_face_data.py") != 1:
        return False
    overlays = [c for c in cmds[step_idx + 1:]
                if _script_name(c) == "overlay_track.py"]
    if multicam:
        # only THIS camera's overlay: the one drawing this face file
        this_face = _cmd_value(cmd, "--output")
        overlays = [c for c in overlays
                    if this_face and _cmd_value(c, "--face") == this_face]
    if not any(os.path.isfile(_cmd_value(c, "--input")) for c in overlays):
        return False

    for c in overlays:
        while "--face" in c:
            i = c.index("--face")
            del c[i:i + 2]

    face_npz = _cmd_value(cmd, "--output")
    if face_npz and os.path.isfile(face_npz):
        try:
            os.remove(face_npz)
            print(f"[BlendCap] Removed face data from an earlier capture: "
                  f"{os.path.basename(face_npz)}")
        except Exception as e:
            print(f"[BlendCap] Could not remove old face data ({e}).")

    tail = [t for t in st.get('error_tail', []) if t][-5:]
    info = {
        'rc': rc,
        'tail': tail,
        'cmd': ' '.join(cmd),
        'no_face': any(_NO_FACE_NEEDLE in t for t in st.get('error_tail', [])),
    }
    if multicam:
        # Reported per camera by multicam.result_lines; the error tail
        # is cleared so the next camera's failure is judged on its own.
        st.setdefault('multicam_face_skipped', []).append(
            dict(info, step=step_idx))
        st['error_tail'] = []
    else:
        st['face_fallback'] = info
    return True


def _face_fallback_lines(info, out_dir):
    """Dialog lines for a capture that finished body-only. No face in
    the clip gets the plain statement alone; any other face-step
    failure also says why, the same way the failure dialog would."""
    if info.get('no_face'):
        return ["No face was found in this clip.",
                "The body capture was kept and finished without it."]
    rc = info.get('rc')
    tail = _tail_with_exit_code(info.get('tail') or [], rc)
    log_path = _ensure_debug_log(out_dir, info.get('cmd'), rc, tail)
    lines = ["Face capture failed.",
             "The body capture was kept and finished without it."]
    plain = _plain_failure_explanation(tail)
    if plain:
        return lines + plain
    if tail:
        lines.append(f"Last output (exit code {rc}):")
        lines.extend(f"  {t[:90]}" for t in tail)
    if log_path:
        lines.append("Full log: blendcap_capture_debug.log in the "
                     "capture's cache folder.")
    return lines


def _run_fallback_overlay(output_dir, body_npz, face_npz, strip_name):
    """Generate an overlay from available NPZ data after a cancel race condition."""
    source_path = blendcap_state.get('actual_video', '') or blendcap_state.get('source_path', '')
    python_exe = blendcap_state.get('python_exe') or _python_exe()
    addon_dir = blendcap_state.get('addon_dir') or _addon_dir()
    overlay_script = os.path.join(addon_dir, "overlay_track.py")
    overlay_mp4 = os.path.join(output_dir, "overlay.mp4")

    if not (os.path.exists(python_exe) and os.path.exists(overlay_script) and source_path):
        print("[BlendCap] Cannot generate fallback overlay.")
        return

    cmd = [python_exe, overlay_script, "--video", source_path, "--output", overlay_mp4]
    if os.path.exists(body_npz):
        cmd.extend(["--input", body_npz])
    if face_npz and os.path.exists(face_npz):
        cmd.extend(["--face", face_npz])

    blendcap_state['step_labels'] = ['overlay']
    blendcap_state['step_outputs'] = [overlay_mp4]
    blendcap_state['proxy_path'] = overlay_mp4
    blendcap_state['_strip_type'] = blendcap_state.get('_strip_type', 'capture')

    # Pre-set state before thread starts to prevent race with monitor_progress_timer
    # (timer fires with first_interval=0 and could see stale running=False/return_code=-2)
    blendcap_state['running'] = True
    blendcap_state['return_code'] = 0
    blendcap_state['cancel_requested'] = False

    thread = threading.Thread(
        target=run_mocap_thread,
        args=([cmd], overlay_mp4, blendcap_state['insert_frame'],
              blendcap_state['insert_channel'], source_path, 0, 0,
              blendcap_state['mode'], [100]),
        daemon=True)
    thread.start()
    bpy.app.timers.register(monitor_progress_timer)


def run_mocap_thread(cmds, proxy_path, insert_frame, insert_channel,
                     source_path, source_start, source_max, mode, step_weights=None,
                     use_ort=None, force_cpu=False, step_prefixes=None):
    """step_prefixes: optional text put in front of each step's status
    line ("Camera B: "), one per command (multi-camera captures)."""
    n = len(cmds)
    if step_weights and len(step_weights) == n:
        total_w = sum(step_weights)
        norm = [w / total_w * 100.0 for w in step_weights]
    else:
        norm = [100.0 / n] * n

    offsets = []
    cum = 0.0
    for w in norm:
        offsets.append(cum)
        cum += w

    st = blendcap_state
    st.update({
        'running': True, 'total_steps': n, 'current_step': 0,
        'step_progress': 0.0, 'step_weights': norm, 'step_offsets': offsets,
        'progress': 0.0, 'status_text': "Starting...",
        'proxy_path': proxy_path, 'insert_frame': insert_frame,
        'insert_channel': insert_channel, 'source_path': source_path,
        'source_start': source_start, 'source_max': source_max,
        'mode': mode, 'return_code': 0,
        'cancel_requested': False, 'active_process': None,
        'steps_completed': [], 'warnings': [],
        # Watchdog + failure-dialog support: when the last subprocess
        # output arrived, and the last few non-noise lines it printed.
        'last_output_time': time.time(), 'error_tail': [],
        # Set when a failed face step was dropped and the run carried
        # on as body-only (_face_step_fallback).
        'face_fallback': None, 'face_fallback_lines': [],
        # Multi-camera: face steps dropped because that camera's face
        # capture failed (a camera behind the performer sees no face).
        'multicam_face_skipped': [],
    })
    prefixes = list(step_prefixes or [])

    def update_progress(idx, pct, status=""):
        overall = offsets[idx] + (pct / 100.0) * norm[idx]
        st['step_progress'] = pct
        st['progress'] = min(overall, 99.9)
        if status:
            pre = prefixes[idx] if idx < len(prefixes) else ""
            st['status_text'] = pre + status

    # Capture backend override: force USE_ORT for the subprocess when the
    # add-on pref selects a specific backend (NVIDIA off / non-NVIDIA on).
    # force_cpu (CPU-only backend) additionally pins BOTH torch and ONNX
    # Runtime to the CPU via env vars save_mhr_data._get_device and
    # ort_backend.select_providers read - injected straight into the subprocess
    # env so they propagate reliably (a user-set OS env var only reaches us if
    # Blender was relaunched after setting it). None/False = inherit unchanged.
    env_overrides = {}
    if use_ort is not None:
        env_overrides['USE_ORT'] = use_ort
    if force_cpu:
        env_overrides['BLENDCAP_FORCE_CPU'] = '1'
        env_overrides['BLENDCAP_ORT_CPU'] = '1'
    # _subprocess_env also strips the Steam runtime's library overrides
    # when Blender was launched through Steam (they break torch /
    # onnxruntime native imports in the venv python). None = inherit.
    capture_env = _subprocess_env(env_overrides)

    # Under Flatpak Blender the venv lives on the host and must be run
    # there (see helpers._host_spawn_prefix); [] on every other platform.
    # The injected capture env rides flatpak-spawn --env= because it is
    # not reliably forwarded across the spawn.
    spawn_prefix = _host_spawn_prefix(extra_env=env_overrides)

    try:
        for step_idx, cmd in enumerate(cmds):
            if st['cancel_requested']:
                st['return_code'] = -2
                st['status_text'] = "Cancelled"
                print("[BlendCap] Cancelled by user.")
                return

            st['current_step'] = step_idx
            update_progress(step_idx, 0.0, "Starting step...")
            print(f"[BlendCap] Running: {' '.join(cmd)}")
            st['last_cmd'] = ' '.join(cmd)

            # CREATE_NO_WINDOW (Windows): output is piped, so a console
            # window for the child is never useful — without the flag,
            # launchers that give Blender no console can flash one per step.
            proc = subprocess.Popen(spawn_prefix + cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, encoding='utf-8', errors='replace',
                                    env=capture_env,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            st['active_process'] = proc
            st['last_output_time'] = time.time()

            buf = ""
            while True:
                if st['cancel_requested']:
                    proc.terminate()
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired: proc.kill()
                    st['return_code'] = -2
                    st['status_text'] = "Cancelled"
                    print("[BlendCap] Cancelled by user.")
                    return

                ch = proc.stdout.read(1)
                if not ch and proc.poll() is not None:
                    break
                if ch in ('\r', '\n'):
                    st['last_output_time'] = time.time()
                    line = _ansi_re.sub('', buf).strip()
                    if line:
                        if not _capture_noise_re.match(line):
                            print(line)
                            # Keep the last few meaningful lines so a
                            # failure dialog can show WHY, not just a code.
                            tail = st.setdefault('error_tail', [])
                            tail.append(line[:120])
                            del tail[:-12]
                        wm = _warning_re.search(line)
                        if wm:
                            warn_text = wm.group(1).strip()
                            if warn_text and warn_text not in st.get('warnings', []):
                                st.setdefault('warnings', []).append(warn_text)
                        m = _progress_re.search(line)
                        if m:
                            update_progress(step_idx, min(int(m.group(1)), 100), m.group(2).strip())
                        else:
                            m = _frame_re.search(line)
                            if m:
                                cur, tot = int(m.group(1)), int(m.group(2))
                                if tot > 0:
                                    update_progress(step_idx, (cur / tot) * 100.0)
                            elif '%' in line and '|' in line:
                                m = _tqdm_re.search(line)
                                if m:
                                    update_progress(step_idx, float(m.group(1)))
                    buf = ""
                else:
                    buf += ch

            proc.wait()
            st['active_process'] = None

            # The Cancel operator terminates the child directly (the read
            # above can block for minutes on a silent subprocess, so the
            # loop's own cancel check may never run). A non-zero exit with
            # the flag set is a cancel, not a failure.
            if st['cancel_requested']:
                st['return_code'] = -2
                st['status_text'] = "Cancelled"
                print("[BlendCap] Cancelled by user.")
                return

            if proc.returncode != 0:
                # A body + face capture keeps its body when the face
                # step fails: drop the face and run the rest.
                if _face_step_fallback(cmds, step_idx, st, proc.returncode):
                    print(f"[BlendCap] Face step failed with code "
                          f"{proc.returncode}; finishing as a body-only "
                          f"capture.")
                    update_progress(step_idx, 100.0,
                                    "No face data, continuing with body")
                    continue
                st['return_code'] = proc.returncode
                st['status_text'] = f"Step {step_idx + 1} failed"
                print(f"[BlendCap] Step {step_idx + 1} failed with code {proc.returncode}")
                return

            st['steps_completed'].append(step_idx)
            update_progress(step_idx, 100.0, "Step complete")

        st['progress'] = 100.0
        st['status_text'] = "Complete"
        st['return_code'] = 0
    except Exception as e:
        st['return_code'] = -1
        st['status_text'] = f"Error: {e}"
        print(f"[BlendCap] Thread error: {e}")
    finally:
        st['active_process'] = None
        st['running'] = False


# ============================================================
# BVH EXPORT BACKGROUND LOGIC
# ============================================================
def _place_person(rig):
    """Keep Captured Position (multi-person captures): move the freshly
    imported rig so that the camera that filmed it stands on the 3D
    cursor's spot. Every person of a capture exported this way ends up
    where they stood relative to each other, each on the cursor's
    floor. The camera's place is what the export recorded and the
    import has just put on the rig (scene_camera.remember); only the
    rig's location changes."""
    from mathutils import Vector
    from .scene_camera import _BVH_TO_BLENDER, stored
    data = stored(rig)
    view = (data or {}).get("views", {}).get("a")
    if not view:
        why = (data or {}).get("why") or "the export recorded no camera"
        print(f"[BlendCap] Keep Captured Position skipped: {why}")
        return
    cam = Vector(view["location"])
    bpy.context.view_layer.update()
    off = rig.matrix_world.to_3x3() @ (_BVH_TO_BLENDER @ cam)
    cursor = bpy.context.scene.cursor.location
    rig.location = (cursor.x - off.x, cursor.y - off.y, cursor.z)
    print(f"[BlendCap] '{rig.name}' placed {off.length:.2f} m from the camera's "
          f"spot at the 3D cursor")


def monitor_bvh_timer():
    if bvh_state['running']:
        # Mirror monitor_progress_timer: paint the workspace status bar
        # and redraw the panel so the in-panel progress bar ticks.
        pct = bvh_state.get('progress', 0.0)
        status = bvh_state.get('status_text', '')
        txt = format_workspace_status("BlendCap BVH", pct, status)
        try:
            for win in bpy.context.window_manager.windows:
                if win.workspace:
                    win.workspace.status_text_set(txt)
        except Exception:
            pass
        _tag_redraw('VIEW_3D', 'PROPERTIES')
        return 0.25

    # Finished — clear the workspace status bar.
    try:
        for win in bpy.context.window_manager.windows:
            if win.workspace:
                win.workspace.status_text_set(None)
    except Exception:
        pass

    # New BVH files may have landed in the library folder: refresh the
    # panel's list now instead of waiting for the next draw to notice.
    try:
        from .properties import _bvh_library_resync
        _bvh_library_resync()
    except Exception:
        pass

    # one-shot, so a later Bake can never be placed by this Generate
    placement = bvh_state.pop('placement', None)

    if bvh_state['return_code'] == 0:
        # bvh_paths is a list of (final_path, stem) for all completed jobs.
        # `stem` is the final filename without extension and already encodes
        # the vid_name + component codes (e.g., "clip1_BHRF", "clip1_H",
        # "clip2_F", "clip1_BHR-clip2_F").
        paths = bvh_state.get('bvh_paths', [])
        if not paths and bvh_state.get('bvh_path'):
            stem = os.path.splitext(os.path.basename(
                bvh_state['bvh_path']))[0]
            paths = [(bvh_state['bvh_path'], stem)]
        for final_path, stem in paths:
            if not os.path.exists(final_path):
                print(f"[BlendCap] BVH file not found at {final_path}")
                continue
            try:
                _import_bvh(final_path, stem)
                # Tag the freshly-imported armature with provenance so
                # the per-rig camera-offset Bake button can re-run
                # NPZ→BVH for THIS specific clip without the user
                # having to re-pick anything in the Capture section.
                # `bvh_state['vid_name']` is the BARE clip identifier
                # (no Capture_ prefix). Stamped on the rig so the
                # camera-offset Bake operator can rebuild the cache
                # folder path (Capture_ + vid_name) and parse the BVH
                # stem (which also uses the bare name). Set after
                # import so bpy.context.active_object is the new rig.
                imported = bpy.context.active_object
                if imported is not None and imported.type == 'ARMATURE':
                    imported["blendcap_source_video"] = bvh_state.get(
                        'vid_name', '')
                    imported["blendcap_bvh_path"] = final_path
                    # Cumulative camera offset baked into THIS BVH on
                    # disk. Read by the Bake Camera Offset operator
                    # so a follow-up bake adds on top instead of
                    # treating the previous bake as 0. Set by the
                    # Generate / Bake operators BEFORE thread.start;
                    # default 0 for plain Generate, total for Bake.
                    imported["_blendcap_baked_pitch_deg"] = float(
                        bvh_state.get('baked_pitch_deg', 0.0))
                    imported["_blendcap_baked_roll_deg"] = float(
                        bvh_state.get('baked_roll_deg', 0.0))
                    imported["_blendcap_baked_head_pitch_deg"] = float(
                        bvh_state.get('baked_head_pitch_deg', 0.0))
                    imported["_blendcap_baked_head_roll_deg"] = float(
                        bvh_state.get('baked_head_roll_deg', 0.0))
                    imported["_blendcap_baked_height_m"] = float(
                        bvh_state.get('baked_height_m', 0.0))
                    # Where the capture's camera(s) stood goes onto
                    # the rig (for the Add Camera button, and for Keep
                    # Captured Position just below); a camera this rig
                    # already has is then put back next to it.
                    try:
                        from . import scene_camera
                        cam_job = bvh_state.get('camera')
                        if cam_job and cam_job.get('stem') == stem:
                            scene_camera.remember(imported, cam_job)
                        if placement and placement.get('stem') == stem:
                            _place_person(imported)
                        scene_camera.update_camera(imported, stem)
                    except Exception as cam_err:
                        print(f"[BlendCap] Camera step failed: {cam_err}")
                print(f"[BlendCap] BVH saved to library and imported: "
                      f"{final_path}")
            except Exception as e:
                print(f"[BlendCap] Failed to import {final_path}: {e}")
                # Used to stop at that print. The export had already run
                # to 100%, so the user saw a completed progress bar, no
                # armature, and no error - "nothing happens", the
                # single most reported ticket. Surface it instead.
                #
                # A missing/disabled BVH importer gets its own dialog
                # (it names the add-on and can switch it back on);
                # anything else gets the generic one, which always ends
                # by saying where the file is, because the capture data
                # survives every one of these failures.
                if bvh_importer_state() != 'OK':
                    def _deferred_importer_popup(path=final_path):
                        try:
                            bpy.ops.blendcap.bvh_importer_missing(
                                'INVOKE_DEFAULT', bvh_path=path)
                        except Exception as err:
                            print("[BlendCap] Could not show the BVH "
                                  f"importer dialog: {err}")
                        return None
                    bpy.app.timers.register(_deferred_importer_popup,
                                            first_interval=0.3)
                else:
                    folder, name = os.path.split(final_path)
                    blendcap_state['warnings'] = [
                        "The capture finished, but the animation couldn't "
                        "be imported.",
                        f"  {str(e)[:88]}",
                        "The BVH file was still saved to:",
                        f"  {name}",
                        f"  in {folder}",
                    ]

                    def _deferred_import_failure_popup():
                        try:
                            bpy.ops.blendcap.show_warnings('INVOKE_DEFAULT')
                        except Exception as err:
                            print("[BlendCap] Could not show the import "
                                  f"failure dialog: {err}")
                        return None
                    bpy.app.timers.register(_deferred_import_failure_popup,
                                            first_interval=0.3)
    elif bvh_state['return_code'] == -2:
        print("[BlendCap] BVH export cancelled.")
    else:
        print(f"[BlendCap] BVH Export failed: {bvh_state['error_msg']}")
        # Same failure dialog as the capture path: show the last lines
        # the export subprocess printed instead of console-only output.
        tail = [t for t in bvh_state.get('error_tail', []) if t][-5:]
        tail = _tail_with_exit_code(tail, bvh_state.get('return_code'))
        _remember_sac_block(tail)
        # Same rule as the capture branch: recognised -> the plain
        # explanation alone; unrecognised -> the raw details.
        lines = _plain_failure_explanation(tail)
        if not lines:
            lines = [
                f"BVH export failed (exit code {bvh_state['return_code']})."]
            if bvh_state.get('error_msg'):
                lines.append(str(bvh_state['error_msg'])[:90])
            if tail:
                lines.append("Last output:")
                lines.extend(f"  {t[:90]}" for t in tail)
        blendcap_state['warnings'] = lines

        def _deferred_bvh_failure_popup():
            try:
                bpy.ops.blendcap.show_warnings('INVOKE_DEFAULT')
            except Exception as e:
                print(f"[BlendCap] Could not show failure dialog: {e}")
            return None
        bpy.app.timers.register(_deferred_bvh_failure_popup,
                                first_interval=0.3)
    # Generate/Reload All (multi-person captures): on to the next person
    try:
        from . import people_ui
        people_ui.queue_advance(bvh_state['return_code'])
    except Exception as e:
        print(f"[BlendCap] Generate/Reload All could not continue: {e}")
    return None


def run_bvh_thread(jobs, vid_name, cache_dir, bvh_library_dir=None):
    """Run a list of BVH-generation subprocesses sequentially.

    jobs: list of (cmd, intermediate_bvh_path, output_stem). Each command
    writes to its intermediate path; on success the file is moved to
    cache/bvh_exports/{output_stem}.bvh. The stem is the full final
    filename (without extension) using the naming convention described
    in BLENDCAP_OT_generate_bvh.execute().
    """
    n_jobs = len(jobs)
    bvh_state.update({
        'running': True, 'return_code': 0,
        'cancel_requested': False, 'active_process': None,
        'error_tail': [],
        'bvh_paths': [],
        # Clear the legacy single-path field at the start of every
        # run. Previously it was only assigned on a successful move-
        # to-library, so a Generate that crashed mid-run could leave
        # a stale path from a previous Generate, and monitor_bvh_
        # timer's fallback (`if not paths and bvh_state.get(
        # 'bvh_path')`) would re-import the OLD library file
        # silently — the "regenerate sometimes pulls from library"
        # bug. Setting it to "" up front means a failed run has no
        # path to fall back to and the monitor correctly does
        # nothing.
        'bvh_path': "",
        # NOTE: baked_pitch_deg / baked_roll_deg / baked_height_m are
        # deliberately NOT included here. They're set by the calling
        # operator BEFORE thread.start() (Generate sets zeros, Bake
        # sets cumulative totals) so the worker can't race-overwrite
        # them. Read by monitor_bvh_timer to set the _blendcap_baked_*
        # custom props on the imported armature so the next Bake can
        # add ON TOP.
        'progress': 0.0, 'status_text': "Starting...",
        'current_job': 0, 'total_jobs': n_jobs,
    })
    bvh_state['vid_name'] = vid_name
    bvh_exports_dir = bvh_library_dir or os.path.join(cache_dir, "bvh_exports")
    os.makedirs(bvh_exports_dir, exist_ok=True)

    def _set_progress(job_idx_zero, local_pct, status):
        # Map a per-job 0-100 into the global 0-100 across all jobs.
        slice_size = 100.0 / max(n_jobs, 1)
        overall = (job_idx_zero * slice_size
                   + (max(0.0, min(local_pct, 100.0)) / 100.0) * slice_size)
        bvh_state['progress'] = min(overall, 99.9)
        if status:
            if n_jobs > 1:
                bvh_state['status_text'] = (
                    f"Job {job_idx_zero + 1}/{n_jobs}: {status}")
            else:
                bvh_state['status_text'] = status

    try:
        for idx, (cmd, bvh_path, stem) in enumerate(jobs, start=1):
            if bvh_state['cancel_requested']:
                bvh_state['return_code'] = -2
                print("[BlendCap] BVH export cancelled by user.")
                return
            bvh_state['current_job'] = idx
            _set_progress(idx - 1, 0.0, "Starting...")
            if n_jobs > 1:
                print(f"[BlendCap] BVH job {idx}/{n_jobs} ({stem})")

            # Flatpak: npz_to_bvh runs the host venv python too (see
            # helpers._host_spawn_prefix); [] on every other platform.
            proc = subprocess.Popen(_host_spawn_prefix() + cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    bufsize=1, encoding='utf-8',
                                    errors='replace',
                                    env=_subprocess_env(),
                                    creationflags=getattr(
                                        subprocess, 'CREATE_NO_WINDOW', 0))
            bvh_state['active_process'] = proc
            for line in proc.stdout:
                if bvh_state['cancel_requested']:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    bvh_state['return_code'] = -2
                    print("[BlendCap] BVH export cancelled by user.")
                    return
                clean = _ansi_re.sub('', line).rstrip()
                # Parse `[PROGRESS pct] message` markers emitted by the
                # BVH pipeline (pipeline/modes.py, footskate_cleanup.py,
                # bvh_writer.py). Suppress the marker line from console
                # output to keep the log readable.
                m = _progress_re.search(clean)
                if m:
                    _set_progress(idx - 1,
                                  float(int(m.group(1))),
                                  m.group(2).strip())
                else:
                    print(clean)
                    if clean:
                        tail = bvh_state.setdefault('error_tail', [])
                        tail.append(clean[:120])
                        del tail[:-12]
            proc.wait()
            bvh_state['active_process'] = None
            # Cancel operator may have terminated the child directly while
            # the line iterator was blocked — treat that exit as a cancel.
            if bvh_state['cancel_requested']:
                bvh_state['return_code'] = -2
                print("[BlendCap] BVH export cancelled by user.")
                return
            if proc.returncode != 0:
                bvh_state['return_code'] = proc.returncode
                return

            final = os.path.join(bvh_exports_dir, f"{stem}.bvh")
            if os.path.exists(bvh_path):
                shutil.move(bvh_path, final)
                bvh_state['bvh_paths'].append((final, stem))
                # Keep single-path field populated for legacy callers
                bvh_state['bvh_path'] = final

            _set_progress(idx - 1, 100.0, "Job complete")

        bvh_state['progress'] = 100.0
        bvh_state['status_text'] = "Complete"
        bvh_state['return_code'] = 0
    except Exception as e:
        bvh_state['return_code'] = -1
        bvh_state['error_msg'] = str(e)
    finally:
        bvh_state['active_process'] = None
        bvh_state['running'] = False
