"""Multi-Camera Capture: 2 to 4 clips of one performance, captured
separately and merged into one take.

The workflow, start to finish:

  1. The user selects 2 to 4 video strips and presses Sync Clips. The
     clips are shown tiled and lettered A to D (syncing mode), each with
     its focal length and, for B to D, a Sync Slide against A. While
     syncing, the Sequencer shows ONLY those clips and the sound of ONE
     of them (camera A's, or the first after it that has real sound):
     they are nested in one strip and the Sequencer is stepped inside
     it ("the nest" below). That one sound is all that is heard; every
     other sound strip is muted until syncing mode ends. Everything
     else in the timeline is untouched and comes back on leaving. Sync
     by Sound lines the clips up from their sound tracks.
  2. Run Capture (only reachable from syncing mode, so nobody skips the
     look at the sync) captures every camera over the frames the clips
     SHARE on the timeline, merges them, and builds one "multiview"
     video: every camera's own skeleton overlay, tiled and lettered.
  3. That one strip replaces the component clips. It carries the whole
     capture's identity (blendcap_source_strip = the cache folder's clip
     id), so Generate BVH, Clear Cache and Add Camera work from it alone
     and nothing depends on which clips are selected or which mode the
     dropdown is on. It also carries everything needed to put the
     original clips back (STRIP_TAG, a JSON record).
  4. Revert is the capture's mirror: the one strip opens back up, in
     place, into a tight block of the clips (the strip's channel and
     the ones above it, A on top), still synced, never on top of other
     strips. The one sound the capture kept stays where it is; the
     other cameras' sound strips went at the capture and do not come
     back (Sync by Sound reads the clips' files, not those strips).

Cache layout (the merge sits where a single capture's files would, which
is what lets Generate BVH read it unchanged):

    Capture_<A>_<N>cam/   mhr_frames.npz, face_frames.npz   <- merged
                          calibration.npz
                          overlay.mp4                        <- multiview
                          cam_a/  mhr_frames.npz, face_frames.npz,
                                  overlay.mp4, .cache_params
                          cam_b/ ...

MULTI-CAMERA & MULTI-PERSON is the same workflow with more than one
performer. What differs is what is captured and when the cameras are
merged:

    Capture_<A>_<N>cam/   people.json            <- who is who, across cameras
                          people_edits.json      <- the user's corrections
                          overlay.mp4            <- multiview
                          cam_a/  people.json, people/track_NN.npz, ...
                                  (a Multi-Person capture of that camera)
                          cam_b/ ...
    Capture_<A>_<N>cam_P1/  mhr_frames.npz, calibration.npz   <- person 1,
                          cam_a/ cam_b/ ...       merged, made at Generate BVH

Every camera is captured as a Multi-Person capture of its own, then
pipeline/people_cameras.py matches the cameras' people to each other
(no merge at capture). The People list (people_ui.py) shows the result
and takes corrections, for every camera or for one. A person's cameras
are merged when their BVH is asked for, from the pieces as corrected by
then (people_tracks.extract_person_cameras), into a folder of their own
that is a merged take like the one above.

Which clip is A: the top channel, then B, C, D going down, the way the
panel lists them (A first); clips sharing a channel go left to right.
The syncing view stacks them the same way round, A on top (until
2026-11-25 it had A at the bottom, the opposite of the timeline and of
the panel), with the one sound that is heard on the bottom channel,
under them all. The captured strip goes DOWN: onto the lowest channel the
clips were on, not onto camera A's, and the one sound that stays (the
one heard while syncing: camera A's, or the next camera's that has
real sound) comes down directly under it (the user, 2026-11-25 and 26). "Were
on" counts the channel a clip had BEFORE Sync Clips re-stacked it: clips
that shared a channel or sat end to end are moved above everything
else for tiling, out of sight inside the syncing view, and the result
must not end up there.

MERGED_CAMERAS is how many cameras pipeline/two_camera_merge.py can
combine: all four (the script keeps its name from when it took two). A
take captured while it took two has "merged": 2 in its record, and Add
Camera offers only the cameras its merge placed until it is captured
again.

FILE PATH (Video Source = File Path) is a deliberately limited second
way in, for a studio running a batch of takes that are ALREADY synced:
Select Clips (2 to 4 files, or a take's folder), one focal length for
the group, Run Capture. No syncing step, because there is no timeline
to sync on. It assumes identical cameras with identical settings whose
clips start and stop together; a set that does not look like that gets
a warning pointing at the Sequencer route, not a refusal (a warning can
be wrong). The result is shown on the viewport plane, the way the
single-camera File Path capture shows its overlay. One set of selected
clips is one take, which is the unit a capture queue would list later.
"""
import json
import os
import threading
import time

import bpy

from .helpers import (
    blendcap_state,
    _cache_dir, _addon_dir, _python_exe, _tag_redraw, addon_prefs,
    _seq_accessors, _strip_visible_start, _strip_visible_end,
    _linked_sound_strip, _strip_slip,
    _strip_frame_start, _strip_set_frame_start,
    _strip_media_extension, _sanitize_clip_name,
    check_npz_coverage, _write_cache_params, _cache_params_match,
    is_dependencies_installed, checkpoints_dir, _installer_dir,
    _show_warning_popup,
)

LETTERS = "ABCD"
MIN_CAMERAS, MAX_CAMERAS = 2, 4
MERGED_CAMERAS = 4

STRIP_TAG = "blendcap_multicam"          # JSON record on the multiview strip

# The capture modes of the Capture dropdown. Multi-Camera & Multi-Person
# selects and syncs its clips exactly as Multi-Camera Capture does, so
# everything here asks is_multi_mode, never the mode's name. What it
# does differently is in _build_plan: every camera is captured as a
# Multi-Person capture of its own, and in place of the merge the
# cameras' people are matched to each other (pipeline/
# people_cameras.py). The cameras are merged later, per person, when a
# BVH is asked for (people_tracks.extract_person_cameras): by then the
# user has had the chance to put right who is who.
MULTI_MODES = ('MULTI', 'PEOPLE_MULTI')
PEOPLE_MODES = ('PEOPLE', 'PEOPLE_MULTI')


def is_multi_mode(props):
    """More than one camera: Multi-Camera Capture or Multi-Camera & Multi-Person."""
    return getattr(props, "capture_mode", 'SINGLE') in MULTI_MODES


def is_people_mode(props):
    """More than one person: Multi-Person Capture or Multi-Camera & Multi-Person."""
    return getattr(props, "capture_mode", 'SINGLE') in PEOPLE_MODES


# Multi-Camera & Multi-Person: the shortest take the matching can do
# anything with (pipeline/people_cameras.py: somebody seen for less than
# MIN_PERSON_S there is nobody, and two cameras need MIN_COMMON_FRAMES
# on the same person). Under it the capture would run and then list
# nobody, so it is refused before it starts (the user, 2026-11-24).
PEOPLE_MIN_S = 0.25
PEOPLE_MIN_FRAMES = 6
# What the panel says then: the user's wording (2026-11-24), broken
# over lines the sidebar has room for. {n}: the frames it needs. Run
# Capture reports it as the one sentence it is (first_sentence).
TOO_SHORT_LINES = ("'Multi-Camera & Multi-Person'",
                   "capture mode requires at least {n}",
                   "frames to capture")


def people_min_frames(scene):
    """The fewest timeline frames a Multi-Camera & Multi-Person capture
    may have: a quarter of a second, and never fewer than 6."""
    try:
        fps = float(scene.render.fps) / float(scene.render.fps_base)
    except Exception:
        fps = 24.0
    return max(PEOPLE_MIN_FRAMES, int(round(PEOPLE_MIN_S * fps)))


def too_short_lines(scene, n_frames, preview=False):
    """The panel's line when a Multi-Camera & Multi-Person capture of
    n_frames timeline frames is too short to run (not for a preview,
    which matches nobody; not when the length is not known, 0)."""
    props = scene.blendcap_props
    if preview or not (is_multi_mode(props) and is_people_mode(props)):
        return []
    need = people_min_frames(scene)
    if 0 < int(n_frames) < need:
        return [line.format(n=need) for line in TOO_SHORT_LINES]
    return []


def continues(line):
    """A panel line that carries on the message of the line above it
    (too_short_lines' second and third): no warning icon of its own."""
    head = TOO_SHORT_LINES[1].split("{n}")[0]
    return line.startswith(head) or line == TOO_SHORT_LINES[2]


SYNC_FLAG ="blendcap_multicam_sync"     # scene: syncing mode is on
SYNC_CLIPS = "blendcap_multicam_clips"   # scene: the pinned clip names, A first
SLIP_APPLIED = "blendcap_multicam_slip_applied"
TEMP_TAG = "blendcap_sync_temp"          # text strips of builds before the nest
SYNC_NEST = "blendcap_multicam_nest"     # scene: the nest's strip name
NEST_NAME = "BlendCap Syncing"
NEST_TAG = "blendcap_sync_nest"          # on the nest itself
HOME_TAG = "blendcap_sync_home"          # on a nested strip: its channel outside
SOUND_TAG = "blendcap_sync_sound"        # on a clip: how its sound sat against it
WAVE_TAG = "blendcap_sync_waveform"      # on the heard sound: its waveform switch before
MUTE_TAG = "blendcap_sync_mute"          # on a sound strip: whether it was muted before
NO_FACE_MARK = ".no_face"                # in a camera's folder: it saw no face
TAKE_TAG = "blendcap_multicam_take"      # on a clip: the take it was captured as
# on a clip Sync Clips re-stacked: [the channel it was on, the one it was put on]
FROM_TAG = "blendcap_multicam_from"
MAX_CHANNEL = 128                        # Blender's sequencer channel limit

# One line, and no word on HOW to select: box select works as well as
# shift-click (the user's wording, 2026-11-16).
SELECT_LINES = ["Select 2 to 4 clips in the Sequencer."]

# File Path: what the panel says, above Select Clips, before any clip
# is selected (the user's wording, 2026-11-18). These and the warnings
# below are broken so that no line is wider than the sidebar at its
# default width, where Blender cuts a label off with "...": about 188
# px of text at the default font, which is 31 to 34 characters.
FILE_INTRO_LINES = ["Clips must be synced and filmed",
                    "on identical cameras.",
                    "For anything else, change Video",
                    "Source to 'Video Editor Strip'."]
# ...and when the selected clips do not look like one synced take from
# identical cameras. Warnings: Run Capture stays available.
FILE_LENGTH_LINES = ["These clips are different lengths.",
                     "To sync them, change Video",
                     "Source to 'Video Editor Strip'."]
FILE_RESOLUTION_LINES = ["These clips are different",
                         "resolutions.",
                         "For mixed cameras, change Video",
                         "Source to 'Video Editor Strip'."]
FILE_RATE_LINES = ["These clips run at different",
                   "frame rates.",
                   "Re-export them to match",
                   "each other."]


def first_sentence(lines):
    """The first sentence of a message that is broken over panel lines,
    as one line: what Run Capture reports when it refuses."""
    out = []
    for line in lines:
        out.append(line)
        if line.rstrip().endswith("."):
            break
    return " ".join(out)
# Lengths this close count as the same: cameras started and stopped
# together can still end a frame or two apart, or drop a frame.
FILE_LENGTH_SLACK = 3


# ============================================================
# SMALL READERS
# ============================================================
def _quiet(fn):
    """Call fn with the 5.1+ strip-timing deprecation warnings off."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return fn()


def is_multicam_strip(strip):
    try:
        return strip is not None and STRIP_TAG in strip
    except Exception:
        return False


def strip_record(strip):
    """The capture record stamped on a multiview strip, or None."""
    if not is_multicam_strip(strip):
        return None
    try:
        rec = json.loads(str(strip[STRIP_TAG]))
        return rec if isinstance(rec, dict) and rec.get("cams") else None
    except Exception:
        return None


def active_multicam_strip(context):
    """The active strip when it is a multiview capture, else None."""
    seq_ed = getattr(context.scene, "sequence_editor", None)
    strip = getattr(seq_ed, "active_strip", None) if seq_ed else None
    return strip if is_multicam_strip(strip) else None


def is_multicam_empty(obj):
    """A viewport plane showing a File Path multi-camera capture."""
    try:
        return (obj is not None and obj.type == 'EMPTY'
                and "blendcap_source" in obj and STRIP_TAG in obj)
    except Exception:
        return False


def _record_of(holder):
    try:
        rec = json.loads(str(holder[STRIP_TAG]))
        return rec if isinstance(rec, dict) and rec.get("cams") else None
    except Exception:
        return None


def active_record(context):
    """The record of the multi-camera capture the user is pointing at,
    or None: the active multiview strip (Sequencer source), or with
    File Path the active multiview plane, else the plane of the take
    whose clips are selected in the panel."""
    try:
        props = context.scene.blendcap_props
        if props.video_source != 'FILE':
            return strip_record(active_multicam_strip(context))
        obj = context.active_object
        if is_multicam_empty(obj):
            return _record_of(obj)
        if is_multi_mode(props):
            cid = files_clip_id(file_infos(props))
            for o in bpy.data.objects:
                if is_multicam_empty(o) and str(o.get("blendcap_source_strip", "")) == cid:
                    return _record_of(o)
    except Exception:
        pass
    return None


def focal_mm(props, index):
    return float(getattr(props, f"multicam_focal_{LETTERS[index].lower()}_mm", 0.0))


def sync_active(scene):
    return bool(scene.get(SYNC_FLAG))


def _order_key(strip):
    return (-int(strip.channel), _strip_visible_start(strip), strip.name)


def selection(context):
    """(strips, lines): the selected clips in camera order A.., or None
    with the lines that say what to select instead."""
    seq_ed = getattr(context.scene, "sequence_editor", None)
    if not seq_ed:
        return None, list(SELECT_LINES)
    seq_all, _ = _seq_accessors(seq_ed)
    picked = [s for s in seq_all
              if getattr(s, "select", False)
              and getattr(s, "type", "") in ('MOVIE', 'IMAGE')]
    if any(is_multicam_strip(s) for s in picked):
        return None, ["This is already a multi-camera capture.",
                      "Revert it to sync or capture again."]
    for s in picked:
        if "blendcap_source" in s:
            return None, [f"'{s.name}' is a capture overlay.",
                          "Revert it to its original video first."]
    if any(s.type == 'IMAGE' for s in picked):
        return None, ["Image sequences can't be used here."] + SELECT_LINES[:1]
    if not (MIN_CAMERAS <= len(picked) <= MAX_CAMERAS):
        return None, list(SELECT_LINES)
    return sorted(picked, key=_order_key), []


def pinned_strips(scene):
    """The clips pinned when syncing mode was entered, A first, or None
    when one of them no longer exists."""
    seq_ed = getattr(scene, "sequence_editor", None)
    names = [str(n) for n in (scene.get(SYNC_CLIPS, []) or [])]
    if not seq_ed or not (MIN_CAMERAS <= len(names) <= MAX_CAMERAS):
        return None
    seq_all, _ = _seq_accessors(seq_ed)
    by_name = {s.name: s for s in seq_all if getattr(s, "type", "") == 'MOVIE'}
    if not all(n in by_name for n in names):
        return None
    return [by_name[n] for n in names]


def strips(context):
    """The clips Multi-Camera Capture is working on: the pinned ones
    while syncing (selection changes with every click in the Sequencer,
    and the user is in there the whole time), else the selection."""
    if sync_active(context.scene):
        return pinned_strips(context.scene)
    return selection(context)[0]


def overlap(clips):
    """(lo, hi): the timeline frames every clip covers. lo >= hi means
    there is no moment all of them show."""
    lo = max(_strip_visible_start(s) for s in clips)
    hi = min(_strip_visible_end(s) for s in clips)
    return lo, hi


def _strip_size(strip):
    """A movie strip's picture size, (0, 0) when Blender has not read it."""
    try:
        el = strip.elements[0]
        return int(el.orig_width), int(el.orig_height)
    except Exception:
        return 0, 0


def single_capture_block(context):
    """A multiview strip or plane stands for a whole multi-camera
    capture. The single-camera capture must never run on it: it would
    track camera A alone into that capture's folder, over the merged
    data. Returns the two lines to show, or None."""
    try:
        props = context.scene.blendcap_props
        if is_multi_mode(props):
            return None
        if props.video_source == 'VSE':
            if active_multicam_strip(context) is not None:
                return ["This is a multi-camera capture.",
                        "Revert it to sync or capture again."]
        else:
            from .helpers import _file_mode_source
            if is_multicam_empty(_file_mode_source(context)[1]):
                return ["This is a multi-camera capture.",
                        "To capture a single clip, pick its video under Video Source."]
    except Exception:
        pass
    return None


# ============================================================
# FILE PATH: THE SELECTED CLIPS
# ============================================================
def file_infos(props):
    """The clips selected for a File Path multi-camera capture, A
    first: [{'path', 'fps', 'frames', 'w', 'h'}]. What each file holds
    is read once, when it is selected (the panel must not open video
    files at every redraw)."""
    try:
        data = json.loads(str(props.multicam_files or "[]"))
        return [d for d in data if isinstance(d, dict) and d.get("path")]
    except Exception:
        return []


def _probe_file(path):
    info = {"path": path, "fps": 0.0, "frames": 0, "w": 0, "h": 0}
    try:
        clip = bpy.data.movieclips.load(path, check_existing=False)
        info["fps"] = float(clip.fps) if clip.fps > 0 else 0.0
        info["frames"] = int(getattr(clip, 'frame_duration', 0) or 0)
        info["w"], info["h"] = int(clip.size[0]), int(clip.size[1])
        bpy.data.movieclips.remove(clip)
    except Exception:
        pass
    return info


def files_clip_id(infos):
    """The take's identity: the first clip's name plus the camera
    count, like the Sequencer route."""
    if not infos:
        return ""
    stem = os.path.splitext(os.path.basename(infos[0]["path"]))[0]
    return _sanitize_clip_name(f"{_sanitize_clip_name(stem)}_{len(infos)}cam")


def file_warnings(infos):
    """(warning lines, refusal lines) for the selected clips. Different
    lengths (beyond FILE_LENGTH_SLACK) and different resolutions are
    warnings: the clips may still be what the user says they are.
    Different frame rates are a refusal: clips at 25 and 30 fps cannot
    be lined up frame by frame at all."""
    warn, refuse = [], []
    rates = [i["fps"] for i in infos if i.get("fps", 0) > 0]
    if rates and not all(rates_match(r, rates[0]) for r in rates):
        refuse += FILE_RATE_LINES
    frames = [i["frames"] for i in infos if i.get("frames", 0) > 0]
    if frames and max(frames) - min(frames) > FILE_LENGTH_SLACK:
        warn += FILE_LENGTH_LINES
    sizes = {(i["w"], i["h"]) for i in infos if i.get("w", 0) > 0}
    if len(sizes) > 1:
        warn += FILE_RESOLUTION_LINES
    return warn, refuse


def file_blockers(context, preview=False):
    """Why Run Capture can't start with the File Path clips, as lines
    for the panel. Empty when it can. preview: the same for Run
    Preview, which needs no focal length."""
    props = context.scene.blendcap_props
    infos = file_infos(props)
    if len(infos) < MIN_CAMERAS:
        return ["Select at least 2 clips."]
    lines = list(file_warnings(infos)[1])
    if not preview and float(props.multicam_focal_mm) <= 0.0:
        lines.append("Enter the cameras' focal length.")
    if not props.track_body:
        lines.append("Body tracking must be on.")
    frames = [i["frames"] for i in infos if i.get("frames", 0) > 0]
    if frames:
        lines += too_short_lines(context.scene, min(frames), preview)
    return lines


class BLENDCAP_OT_multicam_select_clips(bpy.types.Operator):
    """Pick 2 to 4 clips of one take"""
    bl_idname = "blendcap.multicam_select_clips"
    bl_label = "Select Clips"

    directory: bpy.props.StringProperty(subtype='DIR_PATH', options={'HIDDEN', 'SKIP_SAVE'})
    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement,
                                        options={'HIDDEN', 'SKIP_SAVE'})
    filter_folder: bpy.props.BoolProperty(default=True, options={'HIDDEN'})
    filter_movie: bpy.props.BoolProperty(default=True, options={'HIDDEN'})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        from .helpers import _SUPPORTED_VIDEO_EXTS, _MEDIA_EXTS_FOR_STRIP
        props = context.scene.blendcap_props
        folder = bpy.path.abspath(self.directory)
        names = sorted(f.name for f in self.files if f.name)
        video_exts = set(_SUPPORTED_VIDEO_EXTS) | {'.mpg', '.mpeg', '.ts', '.m2ts'}
        if names:
            picked = [os.path.join(folder, n) for n in names]
            picked = [p for p in picked if os.path.isfile(p)]
        else:
            # nothing selected: the folder IS the take
            try:
                picked = sorted(os.path.join(folder, n) for n in os.listdir(folder)
                                if os.path.splitext(n)[1].lower() in video_exts
                                and os.path.isfile(os.path.join(folder, n)))
            except OSError:
                picked = []
            if not picked:
                self.report({'ERROR'}, "No video clips in this folder.")
                return {'CANCELLED'}
        have = file_infos(props)
        known = {os.path.normcase(os.path.normpath(i["path"])) for i in have}
        new = [p for p in picked
               if os.path.normcase(os.path.normpath(p)) not in known]
        if len(have) + len(new) > MAX_CAMERAS:
            if not names and not have:
                self.report({'ERROR'},
                            f"This folder has {len(picked)} video clips. Select "
                            f"the 2 to 4 that belong to one take.")
            else:
                self.report({'ERROR'},
                            "A take is 2 to 4 clips. Clear Clips to start again.")
            return {'CANCELLED'}
        have += [_probe_file(p) for p in new]
        props.multicam_files = json.dumps(have)
        _tag_redraw()
        return {'FINISHED'}


class BLENDCAP_OT_multicam_clear_clips(bpy.types.Operator):
    """Empty the list of selected clips"""
    bl_idname = "blendcap.multicam_clear_clips"
    bl_label = "Clear Clips"

    def execute(self, context):
        context.scene.blendcap_props.multicam_files = "[]"
        _tag_redraw()
        return {'FINISHED'}


# ============================================================
# TILE LAYOUT (shared by the syncing view and the multiview video)
# ============================================================
def tile_layout(sizes, canvas_w, canvas_h):
    """Where each camera's picture goes, in pixels from the TOP LEFT.

    sizes: [(width, height)] per camera, A first. Returns one
    (x, y, w, h) per camera: the picture fitted into its cell, shape
    kept.

      - every clip vertical: one row, left to right;
      - otherwise a 2 x 2 grid: A top left, B top right, C bottom left,
        D bottom right (two clips fill the top half).

    A size that is not known takes the canvas's shape."""
    n = len(sizes)
    cw, ch = float(canvas_w), float(canvas_h)
    sizes = [((w, h) if w > 0 and h > 0 else (cw, ch)) for w, h in sizes]
    if n and all(h > w for w, h in sizes):
        cells = [(i * cw / n, 0.0, cw / n, ch) for i in range(n)]
    else:
        grid = [(0, 0), (1, 0), (0, 1), (1, 1)]
        cells = [(gx * cw / 2, gy * ch / 2, cw / 2, ch / 2)
                 for gx, gy in grid[:n]]
    out = []
    for (w, h), (cx, cy, cellw, cellh) in zip(sizes, cells):
        k = min(cellw / w, cellh / h)
        fw, fh = w * k, h * k
        out.append((cx + (cellw - fw) / 2, cy + (cellh - fh) / 2, fw, fh))
    return out


def multiview_canvas(scene):
    """The multiview video's size: the project's shape, long edge capped
    at 1920 like every other overlay, even dimensions."""
    w = max(2, int(scene.render.resolution_x))
    h = max(2, int(scene.render.resolution_y))
    if max(w, h) > 1920:
        k = 1920.0 / max(w, h)
        w, h = int(w * k), int(h * k)
    return max(2, w // 2 * 2), max(2, h // 2 * 2)


# ============================================================
# SYNCING MODE
# ============================================================
def _remove_temp_strips(scene):
    """Builds before the nest put the letters and a hint line in the
    timeline as text strips. A file saved in syncing mode by one of
    those still holds them: remove them."""
    seq_ed = getattr(scene, "sequence_editor", None)
    if not seq_ed:
        return
    seq_all, seq_coll = _seq_accessors(seq_ed)
    for s in list(seq_all):
        try:
            ours = TEMP_TAG in s or s.name == "BlendCap Sync Hint"
        except Exception:
            ours = False
        if ours and seq_coll is not None:
            seq_coll.remove(s)


def _came_from(strip):
    """The channel a clip was on before Sync Clips re-stacked it, while
    it is still on the channel it was put on then; else None."""
    try:
        was, put = [int(v) for v in strip[FROM_TAG]]
    except Exception:
        return None
    return was if put == int(strip.get(HOME_TAG, strip.channel)) else None


def _stack_if_needed(scene, clips):
    """Clips that share a channel, or that never all show at the same
    time (placed end to end), cannot be tiled as they are. Move them
    onto channels of their own above everything else, A on top, all
    starting where A starts. Their sound comes along. Returns
    (moved, error)."""
    lo, hi = overlap(clips)
    shared = len({int(s.channel) for s in clips}) < len(clips)
    if not shared and lo < hi:
        return False, ""
    seq_ed = scene.sequence_editor
    seq_all, _ = _seq_accessors(seq_ed)
    top = max((int(getattr(s, "channel", 1)) for s in seq_all), default=1)
    n = len(clips)
    if top + 2 * n + n + 1 > MAX_CHANNEL:
        return False, "Not enough free channels above the clips to stack them."
    start = _strip_visible_start(clips[0])
    sounds = [_linked_sound_strip(seq_ed, s) for s in clips]
    for i, s in enumerate(clips):
        dt = start - _strip_visible_start(s)
        was = _came_from(s) or int(s.channel)
        s.channel = top + (n - i)                    # A highest
        s[FROM_TAG] = [int(was), int(s.channel)]
        if dt:
            _strip_set_frame_start(s, _strip_frame_start(s) + dt)
            snd = sounds[i]
            if snd is not None:
                snd.channel = top + n + 1 + i        # a free one: it moves in time
                _strip_set_frame_start(snd, _strip_frame_start(snd) + dt)
    return True, ""


def _apply_tile_view(scene, clips):
    """Show the clips tiled, purely through each strip's transform. The
    view each strip had is saved on it first (once), for sync_restore."""
    res_x = int(scene.render.resolution_x)
    res_y = int(scene.render.resolution_y)
    rects = tile_layout([_strip_size(s) for s in clips], res_x, res_y)
    for s, (x, y, w, h) in zip(clips, rects):
        t = s.transform
        if "blendcap_sync_saved" not in s:
            s["blendcap_sync_saved"] = 1
            s["blendcap_sync_blend"] = s.blend_type
            s["blendcap_sync_xform"] = [t.scale_x, t.scale_y,
                                        t.offset_x, t.offset_y]
        sw, sh = _strip_size(s)
        if sw <= 0 or sh <= 0:
            sw, sh = res_x, res_y
        t.scale_x = w / sw
        t.scale_y = h / sh
        t.offset_x = (x + w / 2.0) - res_x / 2.0
        t.offset_y = res_y / 2.0 - (y + h / 2.0)
        # The strips composite over one another: each must let the
        # ones below show through around its own tile.
        s.blend_type = 'ALPHA_OVER'
    return rects


# ------------------------------------------------------------------
# The nest: while syncing, the Sequencer shows only the synced clips
# ------------------------------------------------------------------
# Blender has no way to hide a strip in the timeline. What it does have
# is the meta strip: step inside one and the timeline shows its contents
# and nothing else. So Sync Clips nests the clips, and the one sound
# that is heard, in one strip and steps the Sequencer inside it; leaving
# syncing mode steps out and un-nests them onto the channels they came
# from. Strips keep their place in TIME through all of it, so
# nothing else in this module needs to know whether the clips are
# nested: the pin, the slides, the shared frames and the capture plan
# all read the strips themselves.
#
# What the nest costs, and how it is paid:
#   - a nested strip's channel is its channel INSIDE the nest. Its
#     channel outside is kept on it (HOME_TAG) and is what the capture
#     record stores;
#   - the other clips' sound stays outside, hidden. The slides move it
#     with its clip; a clip dragged or slipped BY HAND leaves it behind,
#     so the way each sound sat against its clip is noted on entry
#     (SOUND_TAG) and put back on leaving and before a capture;
#   - Tab is Blender's key for stepping out of a nested strip. Stepping
#     out ends syncing mode (stepped_out; the panel and the Sequencer
#     drawing watch for it);
#   - the nest has a range of its own, which Blender does not update
#     while its clips are moved, and View All frames that range.
#     fit_nest keeps it on the clips (after a slide that moved a strip;
#     and from the drawing's watch, for a clip moved by hand).
# If anything here cannot be done (the user is inside a nested strip of
# their own, an API call fails), syncing carries on in the open
# timeline, which is how it worked before the nest.
def _children(meta):
    kids = getattr(meta, "strips", None)
    return kids if kids is not None else meta.sequences


def _nest(scene):
    """The strip the synced clips are nested in, or None."""
    seq_ed = getattr(scene, "sequence_editor", None)
    name = str(scene.get(SYNC_NEST, "") or "")
    if not seq_ed or not name:
        return None
    seq_all = _seq_accessors(seq_ed)[0]
    try:
        s = seq_all.get(name)
    except Exception:
        s = next((x for x in seq_all if x.name == name), None)
    try:
        return s if (s is not None and s.type == 'META' and NEST_TAG in s) else None
    except Exception:
        return None


def nest_open(scene):
    """True while the Sequencer is showing the inside of the nest."""
    nest = _nest(scene)
    if nest is None:
        return False
    try:
        stack = scene.sequence_editor.meta_stack
        return len(stack) > 0 and stack[len(stack) - 1].name == nest.name
    except Exception:
        return False


def stepped_out(scene):
    """Syncing mode is on, its clips are nested, and the Sequencer is
    not showing the inside of the nest any more: the user pressed Tab.
    Syncing mode is over then, and the caller ends it."""
    return (sync_active(scene) and _nest(scene) is not None
            and not nest_open(scene))


def _nest_span(nest):
    kids = list(_children(nest))
    if not kids:
        return None
    return (min(_strip_visible_start(s) for s in kids),
            max(_strip_visible_end(s) for s in kids))


def nest_stale(scene):
    """True when the nest's own range is no longer its clips' range."""
    nest = _nest(scene)
    if nest is None:
        return False
    try:
        span = _nest_span(nest)
        return span is not None and span != (_strip_visible_start(nest),
                                             _strip_visible_end(nest))
    except Exception:
        return False


def fit_nest(scene):
    """Make the nest's range its clips' range. The Sequencer reads that
    range when it steps inside, so it is stepped in again."""
    nest = _nest(scene)
    if nest is None or not nest_stale(scene):
        return
    try:
        lo, hi = _nest_span(nest)
        end = _strip_visible_end(nest)

        def _set():
            # the end first, far enough out that the start always fits
            nest.frame_final_end = max(end, hi)
            nest.frame_final_start = lo
            nest.frame_final_end = hi
        _quiet(_set)
        if nest_open(scene):
            seq_ed = scene.sequence_editor
            active = seq_ed.active_strip        # stepping in clears it
            seq_ed.display_stack(nest)
            seq_ed.active_strip = active
    except Exception as exc:
        print(f"[BlendCap] Syncing view not refitted: {exc}")


_fit_pending = False


def _fit_fire():
    global _fit_pending
    _fit_pending = False
    try:
        fit_nest(bpy.context.scene)
        _tag_redraw('SEQUENCE_EDITOR')
    except Exception as exc:
        print(f"[BlendCap] Syncing view not refitted: {exc}")
    return None


def schedule_nest_fit():
    """fit_nest on the next timer tick, for draw code (which may not
    write scene data). Registered once however many redraws ask."""
    global _fit_pending
    if _fit_pending:
        return
    _fit_pending = True
    bpy.app.timers.register(_fit_fire, first_interval=0.0)


def _set_channels(strips, targets, siblings):
    """Put each strip on its target channel. Blender pushes a strip that
    lands on another one onto a different channel by itself, so they go
    to channels nobody has or wants first, then to their own."""
    busy = {int(s.channel) for s in siblings} | {int(c) for c in targets}
    spare = [c for c in range(MAX_CHANNEL, 0, -1) if c not in busy]
    for s, c in zip(strips, spare):
        s.channel = c
    for s, c in zip(strips, targets):
        s.channel = int(c)


# ------------------------------------------------------------------
# What is heard while syncing: one camera
# ------------------------------------------------------------------
# Blender plays every sound strip in the timeline, whether or not the
# Sequencer is stepped inside the nest (measured: the scene's sound
# mixed down with each strip on its own pitch). Four phones on a loud
# set, all at once and a little out of step, is noise. So one camera's
# sound is heard and every other sound strip is muted until syncing
# mode ends: camera A's, or when A's clip has no sound strip or a dead
# track (a camera with its microphone off still writes one), the first
# of B, C, D that has something on it. That strip is shown in the nest
# directly under its own camera, waveform on. When no camera has sound
# there is nothing to hear and the nest holds the video strips alone.
# (The user's design, 2026-11-17.)
def _pick_heard(scene, clips):
    """(camera index, its sound strip) for the camera heard while
    syncing, or None when no camera's clip has real sound."""
    from . import multicam_sound
    seq_ed = scene.sequence_editor
    fps = scene.render.fps / scene.render.fps_base
    for i, s in enumerate(clips):
        snd = _linked_sound_strip(seq_ed, s)
        if snd is None:
            continue
        try:
            path = bpy.path.abspath(snd.sound.filepath)
            secs = _quiet(lambda: float(snd.frame_duration)) / fps
        except Exception:
            continue
        if multicam_sound.has_real_sound(path, secs):
            return i, snd
    return None


def _mute_for_sync(scene, heard):
    """Mute every sound strip but the heard camera's (which is unmuted
    if the user had it muted), remembering how each one was."""
    seq_ed = scene.sequence_editor
    for s in _seq_accessors(seq_ed)[0]:
        try:
            if s.type != 'SOUND':
                continue
            if MUTE_TAG not in s:
                s[MUTE_TAG] = int(bool(s.mute))
            s.mute = heard is None or s.name != heard.name
        except Exception as exc:
            print(f"[BlendCap] Sound strip '{getattr(s, 'name', '?')}' not muted: {exc}")
    if heard is not None and hasattr(heard, "show_waveform"):
        if WAVE_TAG not in heard:
            heard[WAVE_TAG] = int(bool(heard.show_waveform))
        heard.show_waveform = True


def _enter_nest(scene, clips, heard=None):
    """Nest the clips and step the Sequencer inside. In there, from the
    TOP: A, B, C, D (the way the timeline and the panel have them), and
    under them all, on the bottom channel, the heard camera's sound
    (`heard`, from _pick_heard; camera A's unless A has none). Returns
    False, with the timeline as it was, when it can't be done."""
    seq_ed = scene.sequence_editor
    seq_all, seq_coll = _seq_accessors(seq_ed)
    try:
        # from the bottom channel up: the sound, then D, C, B, A
        members = [heard[1]] if heard is not None else []
        members.extend(reversed(clips))
        if len(seq_ed.meta_stack) or any(s.parent_meta() is not None
                                         for s in members):
            return False
        top = max(int(s.channel) for s in seq_all)
        lo = min(_strip_visible_start(s) for s in members)
        nest = seq_coll.new_meta(NEST_NAME, min(top + 1, MAX_CHANNEL), lo)
    except Exception as exc:
        print(f"[BlendCap] Syncing in the open timeline (no nest): {exc}")
        return False
    try:
        nest[NEST_TAG] = 1
        nest.select = False
        scene[SYNC_NEST] = nest.name
        for s in clips:
            snd = _linked_sound_strip(seq_ed, s)
            if snd is not None:
                s[SOUND_TAG] = [
                    _strip_frame_start(s) - _strip_frame_start(snd),
                    _strip_visible_start(s) - _strip_visible_start(snd)]
        for s in members:
            s[HOME_TAG] = int(s.channel)
            s.move_to_meta(nest)
        _set_channels(members, range(1, len(members) + 1), members)
        fit_nest(scene)
        seq_ed.display_stack(nest)
        return True
    except Exception as exc:
        print(f"[BlendCap] Syncing in the open timeline (nest failed): {exc}")
        _leave_nest(scene)
        return False


def _leave_nest(scene):
    """Step out of the nest and un-nest its strips, each onto the
    channel it came from, or onto a free one above everything when
    something of the user's is there now."""
    seq_ed = getattr(scene, "sequence_editor", None)
    nest = _nest(scene)
    if not seq_ed or nest is None:
        return
    try:
        _, seq_coll = _seq_accessors(seq_ed)
        inner = list(_children(nest))
        outside = [s for s in seq_coll if s.name != nest.name]
        top = max((int(s.channel) for s in outside), default=0)
        homed = [s for s in inner if HOME_TAG in s]
        # First every strip whose own channel is free goes home; only
        # then are the ones that can't given channels above everything
        # (the homes included), so a displaced strip never takes the
        # place another one is going back to.
        spans = [(_strip_visible_start(s), _strip_visible_end(s)) for s in homed]
        targets, given = [None] * len(homed), []
        for k, s in enumerate(homed):
            lo, hi = spans[k]
            ch = int(s[HOME_TAG])
            if (_region_free(outside, ch, lo, hi)
                    and not any(c == ch and a < hi and b > lo for c, a, b in given)):
                targets[k] = ch
                given.append((ch, lo, hi))
        top = max([top] + [c for c, _a, _b in given])
        for k, s in enumerate(homed):
            if targets[k] is None:
                top = min(top + 1, MAX_CHANNEL)
                targets[k] = top
                given.append((top,) + spans[k])
        _set_channels(homed, targets, inner)
        # A strip coming out onto the nest's own channel would be pushed
        # off it (the nest covers them all): move the nest to a channel
        # nothing is on or going to.
        used = {int(s.channel) for s in outside} | set(targets)
        spare = next((c for c in range(MAX_CHANNEL, 0, -1) if c not in used), None)
        if spare is not None:
            nest.channel = spare
        if len(seq_ed.meta_stack):
            seq_ed.display_stack(None)
        nest.separate()
        for s in homed:
            del s[HOME_TAG]
    except Exception as exc:
        print(f"[BlendCap] Could not un-nest the synced clips ({exc}). They "
              f"are inside the strip '{NEST_NAME}': select it and use "
              f"Strip > Un-Meta Strip.")


def _shift_strip(seq_ed, strip, frames):
    """Move a strip along the timeline, onto a free channel above its
    neighbours when its own is taken there."""
    parent = strip.parent_meta()
    sibs = _children(parent) if parent is not None else _seq_accessors(seq_ed)[1]
    others = [s for s in sibs if s.name != strip.name]
    lo = _strip_visible_start(strip) + frames
    hi = _strip_visible_end(strip) + frames
    if not _region_free(others, strip.channel, lo, hi):
        top = max((int(s.channel) for s in others), default=0)
        if top < MAX_CHANNEL:
            strip.channel = top + 1
    _strip_set_frame_start(strip, _strip_frame_start(strip) + frames)


def restep_sounds(scene, forget=False):
    """Put every synced clip's sound back the way it sat against its
    clip when syncing started. The Sync Slides keep them together by
    themselves; this is for a clip dragged or slipped by hand while its
    sound was hidden outside the nest (or, for camera A, not linked to
    it in Blender's own sense). A dragged clip's sound is moved the same
    way; a slipped clip's sound is slipped."""
    seq_ed = getattr(scene, "sequence_editor", None)
    if not seq_ed:
        return
    for s in list(_seq_accessors(seq_ed)[0]):
        try:
            if SOUND_TAG not in s:
                continue
            gap = [int(v) for v in s[SOUND_TAG]]
            if forget:
                del s[SOUND_TAG]
            snd = _linked_sound_strip(seq_ed, s)
            if snd is None:
                continue
            # how many frames later the sound's footage has to sit
            d = (_strip_frame_start(s) - _strip_frame_start(snd)) - gap[0]
            if not d:
                continue
            # the part of that the clip's place on the timeline moved by
            dv = (_strip_visible_start(s) - _strip_visible_start(snd)) - gap[1]
            move = dv if (dv * d > 0 and abs(dv) <= abs(d)) else 0
            if move:
                _shift_strip(seq_ed, snd, move)
            if d - move:
                left = _strip_slip(snd, -(d - move))
                if left:
                    _shift_strip(seq_ed, snd, -left)
            print(f"[BlendCap] Sound of '{s.name}' brought back in step "
                  f"with its clip ({d:+d} frames).")
        except Exception as exc:
            print(f"[BlendCap] Sound of '{getattr(s, 'name', '?')}' not "
                  f"re-stepped: {exc}")


def _sync_members(scene):
    """(the pinned clips that still exist, A first; the sound strip that
    is kept when syncing mode ends without a capture). The kept sound is
    the one heard while syncing (it carries WAVE_TAG), or camera A's
    when no camera's sound was heard: the same choice load_to_vse makes
    at a capture. Read BEFORE the restore, which clears the tags."""
    seq_ed = getattr(scene, "sequence_editor", None)
    if not seq_ed:
        return [], None
    seq_all = _seq_accessors(seq_ed)[0]
    by_name = {x.name: x for x in seq_all if getattr(x, "type", "") == 'MOVIE'}
    clips = [by_name[n] for n in (str(n) for n in (scene.get(SYNC_CLIPS, []) or []))
             if n in by_name]
    kept = None
    try:
        kept = next((x for x in seq_all if x.type == 'SOUND' and WAVE_TAG in x), None)
        if kept is None and clips:
            kept = _linked_sound_strip(seq_ed, clips[0])
    except Exception as exc:
        print(f"[BlendCap] Which sound to keep could not be worked out: {exc}")
    return clips, kept


def _tidy_after_sync(scene, clips, kept):
    """Leave the synced clips the way a capture followed by Revert to
    Original Clips leaves them (the user, 2026-11-26): the videos as one
    tight block on neighbouring channels, A on top; the ONE sound that
    was heard while syncing directly under the block; the other cameras'
    sound strips removed.

    Until then leaving syncing mode without a capture showed what Sync
    Clips had done to clips that shared a channel or sat end to end
    (_stack_if_needed): the videos above everything, the sound of each
    clip it had moved in time above those, camera A's sound left below.

    Nothing of the user's is overwritten and Blender never has to
    shuffle a strip aside: the block's bottom goes on the lowest channel
    the clips are or were on (before Sync Clips re-stacked them), or on
    the nearest channel above it, where every clip's place is free over
    its own time span. The sound goes under the block only when that
    place is free, and the block goes at most ONE channel higher than it
    otherwise would to make that place (a block that could sit on
    channel 1 has nothing under it); failing that the block is put where
    it fits and the sound stays where it is. The clips do not move in
    time. Clips inside a strip of the user's own (syncing in the open
    timeline, or an un-nest that failed) are left as they are."""
    seq_ed = getattr(scene, "sequence_editor", None)
    if not seq_ed or not clips:
        return
    try:
        if any(x.parent_meta() is not None for x in clips):
            return
        if kept is not None and kept.parent_meta() is not None:
            kept = None
        _, seq_coll = _seq_accessors(seq_ed)
        kept_name = kept.name if kept is not None else ""
        # the other cameras' sound
        for x in clips:
            snd = _linked_sound_strip(seq_ed, x)
            if snd is None or snd.name == kept_name or snd.parent_meta() is not None:
                continue
            seq_coll.remove(snd)

        _, seq_coll = _seq_accessors(seq_ed)
        top_level = list(seq_coll)
        n = len(clips)
        mine = {x.name for x in clips}
        spans = [(_strip_visible_start(x), _strip_visible_end(x)) for x in clips]
        was = [_came_from(x) for x in clips]
        places = {int(x.channel) for x in clips} | {int(c) for c in was if c}
        with_sound = [x for x in top_level if x.name not in mine]
        without = [x for x in with_sound if x.name != kept_name]

        def fits(base, pool):
            return all(_region_free(pool, base + (n - 1 - i), a, b)
                       for i, (a, b) in enumerate(spans))

        bases = range(min(places), MAX_CHANNEL - n + 2)
        # where the block fits with the sound left where it is
        base = next((c for c in bases if fits(c, with_sound)), None)
        move_sound = False
        if kept is not None:
            # where it fits with the sound directly under it
            ks, ke = _strip_visible_start(kept), _strip_visible_end(kept)
            seated = next((c for c in bases
                           if c >= 2 and fits(c, without)
                           and _region_free(without, c - 1, ks, ke)), None)
            if seated is not None and (base is None or seated <= base + 1):
                base, move_sound = seated, True
        if base is None:
            print("[BlendCap] No free channels to tidy the synced clips onto: "
                  "they stay where they are.")
            return
        moving = list(clips)
        targets = [base + (n - 1 - i) for i in range(n)]          # A highest
        if move_sound:
            moving.append(kept)
            targets.append(base - 1)
        _set_channels(moving, targets, top_level)
        for x, w in zip(clips, was):
            if w:
                # still "the channel it had before Sync Clips re-stacked
                # it", for where a later capture's strip goes
                x[FROM_TAG] = [int(w), int(x.channel)]
            elif FROM_TAG in x:
                del x[FROM_TAG]
    except Exception as exc:
        print(f"[BlendCap] The synced clips were not tidied: {exc}")


def sync_restore(scene, tidy=True):
    """End syncing mode: step out of the nest and un-nest the clips,
    give every strip its own view back (blend mode + transform), put
    each sound back in step with its clip, clear the pin and the flag.
    The alignment the user made (slipped footage, moved strips) is the
    sync itself and stays.

    tidy: the clips are then put together as one block with the one
    heard sound under them and the other cameras' sound removed
    (_tidy_after_sync). That is every way of leaving that is not a
    capture (the Exit button, Tab, another capture mode, a clip that was
    deleted: the user, 2026-11-26). A capture passes False: it replaces
    the clips with its one strip and places that itself."""
    _remove_temp_strips(scene)
    members = _sync_members(scene) if tidy else ([], None)
    _leave_nest(scene)
    restep_sounds(scene, forget=True)
    seq_ed = getattr(scene, "sequence_editor", None)
    seq_all = _seq_accessors(seq_ed)[0] if seq_ed else []
    for s in list(seq_all):
        try:
            if MUTE_TAG in s:
                s.mute = bool(s[MUTE_TAG])
                del s[MUTE_TAG]
            if WAVE_TAG in s:
                s.show_waveform = bool(s[WAVE_TAG])
                del s[WAVE_TAG]
            if HOME_TAG in s:          # left behind by a failed un-nest
                del s[HOME_TAG]
            if "blendcap_sync_saved" not in s:
                continue
        except Exception:
            continue
        try:
            s.blend_type = str(s["blendcap_sync_blend"])
            sx, sy, ox, oy = list(s["blendcap_sync_xform"])
            t = s.transform
            t.scale_x, t.scale_y, t.offset_x, t.offset_y = sx, sy, ox, oy
        except Exception as exc:
            print(f"[BlendCap] Could not restore '{s.name}': {exc}")
        for k in ("blendcap_sync_saved", "blendcap_sync_blend",
                  "blendcap_sync_xform",
                  # written by builds before the slip rework
                  "blendcap_sync_channel", "blendcap_sync_frame_start"):
            if k in s:
                del s[k]
    if tidy:
        _tidy_after_sync(scene, *members)
    scene[SYNC_FLAG] = 0
    blendcap_state.pop('_multicam_sound', None)
    for k in (SYNC_CLIPS, SLIP_APPLIED, SYNC_NEST,
              # the two-camera build's scene keys
              "blendcap_dual_sync_pair", "blendcap_dual_sync_active",
              "blendcap_dual_slip_applied"):
        if k in scene:
            del scene[k]
    _tag_redraw()


def apply_slide(context, index, value):
    """Sync Slide for camera `index` (1 = B ...): move that clip's STRIP
    along the timeline, its linked sound with it. One frame on the
    slider is one frame in the timeline, and +N starts the footage N
    frames EARLIER (the strip moves left).

    Until 2026-11-17 the slide was a slip edit: the footage slid under
    a strip that stayed put, for as long as the strip had trimmed-off
    footage to slide into, and only then did the strip move. An
    untrimmed clip moved at once, a trimmed one seemed not to respond
    for as many frames as had been trimmed off, and the slide quietly
    changed which footage the user's trim showed. It was built that way
    so a slide could not push a strip into its neighbours; in the
    syncing view every clip has a channel to itself.

    Capture does not read these numbers. It reads the timeline: however
    the clips end up lined up (these sliders, Blender's own slip tool,
    dragging strips by hand), what gets captured is the frames the
    clips share when Run Capture is pressed."""
    scene = context.scene
    if not sync_active(scene):
        return
    clips = pinned_strips(scene)
    if not clips or index >= len(clips):
        return
    applied = [int(v) for v in (scene.get(SLIP_APPLIED, []) or [])]
    applied += [0] * (MAX_CAMERAS - len(applied))
    delta = int(value) - applied[index]
    if not delta:
        return
    s = clips[index]
    seq_ed = scene.sequence_editor
    snd = _linked_sound_strip(seq_ed, s)
    _shift_strip(seq_ed, s, -delta)
    if snd is not None:
        _shift_strip(seq_ed, snd, -delta)
    fit_nest(scene)
    applied[index] = int(value)
    scene[SLIP_APPLIED] = applied


def run_blockers(context, preview=False):
    """Why Run Capture can't start right now, as lines for the panel
    (cheap: runs at draw time). Empty when it can. preview: the same
    for Run Preview, which needs no focal lengths (only the person
    detector runs)."""
    scene = context.scene
    props = scene.blendcap_props
    clips = pinned_strips(scene)
    if not clips:
        return ["A synced clip was removed."]
    lines = []
    if not preview and any(focal_mm(props, i) <= 0.0 for i in range(len(clips))):
        lines.append("Enter each camera's focal length.")
    if not props.track_body:
        lines.append("Body tracking must be on.")
    lo, hi = overlap(clips)
    if lo >= hi:
        lines.append("The clips don't overlap in time.")
    else:
        lines += too_short_lines(scene, hi - lo, preview)
    return lines


_autoclear_pending = False


def _autoclear_fire():
    global _autoclear_pending
    _autoclear_pending = False
    try:
        sync_restore(bpy.context.scene)
    except Exception as exc:
        print(f"[BlendCap] Sync auto-clear failed: {exc}")
    return None


def schedule_sync_autoclear():
    """End a syncing session whose pinned clip stopped existing. Draw
    code may not write scene data, so the restore runs on the next
    timer tick; registered once however many redraws ask."""
    global _autoclear_pending
    if _autoclear_pending:
        return
    _autoclear_pending = True
    bpy.app.timers.register(_autoclear_fire, first_interval=0.0)


class BLENDCAP_OT_multicam_sync(bpy.types.Operator):
    """Show the selected clips together, lettered A to D, so they can be lined up before capturing"""
    bl_idname = "blendcap.multicam_sync"
    bl_label = "Sync Clips"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        if blendcap_state['running']:
            self.report({'WARNING'}, "A capture is running.")
            return {'CANCELLED'}
        if scene.blendcap_props.video_source != 'VSE':
            self.report({'ERROR'}, "Multi-Camera Capture needs the Video Sequencer source.")
            return {'CANCELLED'}
        clips, lines = selection(context)
        if not clips:
            self.report({'ERROR'}, " ".join(lines))
            return {'CANCELLED'}

        moved, err = _stack_if_needed(scene, clips)
        if err:
            self.report({'ERROR'}, err)
            return {'CANCELLED'}

        _remove_temp_strips(scene)
        _apply_tile_view(scene, clips)

        # The sliders are relative to wherever the clips are right now:
        # zero them without letting the updates fire a phantom slide
        # (apply_slide does nothing while the flag is off).
        scene[SYNC_FLAG] = 0
        props = scene.blendcap_props
        for letter in LETTERS[1:].lower():
            setattr(props, f"multicam_slide_{letter}", 0)
        scene[SLIP_APPLIED] = [0] * MAX_CAMERAS
        # Pin THESE clips, in this order, for the whole session.
        scene[SYNC_CLIPS] = [s.name for s in clips]
        blendcap_state.pop(SOUND_RESULT, None)
        # One camera's sound is heard while syncing, the rest is muted.
        heard = _pick_heard(scene, clips)
        _mute_for_sync(scene, heard[1] if heard else None)
        if heard is None:
            print("[BlendCap] Syncing mode: none of the clips has sound; "
                  "all sound is muted until syncing mode ends.")
        else:
            print(f"[BlendCap] Syncing mode: hearing camera {LETTERS[heard[0]]}"
                  + ("" if heard[0] == 0 else " (the cameras before it have no sound)")
                  + "; other sound is muted until syncing mode ends.")
        # Last, so that everything above saw the clips' own channels:
        # from here the Sequencer shows only these clips.
        _enter_nest(scene, clips, heard)
        scene[SYNC_FLAG] = 1
        _tag_redraw()
        if moved:
            self.report({'INFO'},
                        "The clips were stacked on their own channels so "
                        "they can be shown together.")
        else:
            self.report({'INFO'},
                        "Scrub to a moment you can see in every clip, then "
                        "slide until they match.")
        return {'FINISHED'}


# ------------------------------------------------------------------
# Sync by Sound (the measuring is in multicam_sound.py)
# ------------------------------------------------------------------
SOUND_RESULT = '_multicam_sound'         # blendcap_state: (lines, slide values then)


def _slide_values(props):
    return tuple(int(getattr(props, f"multicam_slide_{l}"))
                 for l in LETTERS[1:].lower())


def sound_gaps(clips):
    """Why Sync by Sound has nothing to work with, as a line for the
    panel, or "" when it can run: camera A's clip has no sound (there
    is nothing to line the others up on), or none of the others has.
    Cheap enough for draw code: each file is asked once."""
    from . import multicam_sound
    has = [multicam_sound.has_sound(bpy.path.abspath(s.filepath)) for s in clips]
    if not has[0]:
        return "No sound in clip A."
    if not any(has[1:]):
        rest = [LETTERS[i] for i in range(1, len(clips))]
        return (f"No sound in clip {rest[0]}." if len(rest) == 1
                else f"No sound in clips {', '.join(rest)}.")
    return ""


# What the pop-up says above the clips Sync by Sound left alone.
SOUND_POPUP_HEAD = "Sync by Sound could not line up every clip."
SOUND_POPUP_TAIL = "Line those up by hand with their Sync Slide."


def sound_failures(lines):
    """The lines of a Sync by Sound result that say a clip was NOT
    lined up (could not tell, or no sound)."""
    return [l for l in lines if "could not tell" in l or "no sound" in l]


def sound_result_lines(scene):
    """What Sync by Sound last did, for the panel: shown while the
    slides still hold what it set (a slide moved by hand, or an undo,
    makes the lines untrue, and they go)."""
    held = blendcap_state.get(SOUND_RESULT)
    if not held or not sync_active(scene):
        return []
    lines, values = held
    return list(lines) if values == _slide_values(scene.blendcap_props) else []


def sync_by_sound(context):
    """Line cameras B to D up on camera A from their sound tracks, by
    setting their Sync Slides (so the footage, and its sound, move the
    way a slide moved by hand moves them). A clip it cannot be sure of
    is left as it is. Returns one line per clip saying what happened.

    The measurement says how many seconds camera B started recording
    after camera A. In step means the same moment of both recordings
    sits on the same timeline frame, which is B's footage starting that
    many frames after A's footage starts."""
    from . import multicam_sound
    scene = context.scene
    props = scene.blendcap_props
    clips = pinned_strips(scene)
    if not clips:
        return ["A synced clip was removed."]
    paths = [bpy.path.abspath(s.filepath) for s in clips]
    fps = float(rate_check(context, clips)["fps"] or 24.0)
    on_a = multicam_sound.clip_onsets(paths[0])
    if on_a is None:
        return ["A: no sound in this clip."]
    lines = []
    for i in range(1, len(clips)):
        letter = LETTERS[i]
        on_b = multicam_sound.clip_onsets(paths[i])
        if on_b is None:
            lines.append(f"{letter}: no sound in this clip.")
            continue
        r = multicam_sound.measure_lag(on_a, on_b, fps)
        print(f"[BlendCap] Sync by Sound, {letter}: started {r['lag']:+.3f} s after A "
              f"({r['lag'] * fps:+.2f} frames); stands {r['z']:.1f} above the rest, "
              f"{r['ratio']:.2f}x the runner-up, {r['witnesses']} of {r['pieces']} parts "
              f"agree, {r['against']} against: "
              + ("believed" if r["ok"] else f"not believed ({r['why']})"))
        if not r["ok"]:
            lines.append(f"{letter}: could not tell from the sound. Left as is.")
            continue
        want = _strip_frame_start(clips[0]) + r["lag"] * fps
        # a Sync Slide of +N starts the footage N frames EARLIER
        step = int(round(_strip_frame_start(clips[i]) - want))
        name = f"multicam_slide_{letter.lower()}"
        if not step:
            lines.append(f"{letter}: already lined up.")
            continue
        value = int(getattr(props, name)) + step
        setattr(props, name, value)          # its update slides footage and sound
        lines.append(f"{letter}: Sync Slide set to {value:+d}.")
    return lines


class BLENDCAP_OT_multicam_sync_sound(bpy.types.Operator):
    """Line the clips up based on their matching audio"""
    bl_idname = "blendcap.multicam_sync_sound"
    bl_label = "Sync by Sound"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        if blendcap_state['running']:
            self.report({'WARNING'}, "A capture is running.")
            return {'CANCELLED'}
        if not sync_active(scene):
            self.report({'ERROR'}, "Press Sync Clips first.")
            return {'CANCELLED'}
        win = getattr(context, "window", None)
        try:
            if win is not None:
                win.cursor_set('WAIT')
            lines = sync_by_sound(context)
        finally:
            if win is not None:
                win.cursor_set('DEFAULT')
        blendcap_state[SOUND_RESULT] = (lines, _slide_values(scene.blendcap_props))
        fit_nest(scene)
        _tag_redraw()
        self.report({'INFO'}, "Sync by Sound: " + " ".join(lines))
        # A clip it left alone gets a pop-up: a line under the button is
        # easy to miss, and the user would capture an unsynced take.
        failed = sound_failures(lines)
        if failed:
            _show_warning_popup([SOUND_POPUP_HEAD] + failed + [SOUND_POPUP_TAIL])
        return {'FINISHED'}


class BLENDCAP_OT_multicam_sync_exit(bpy.types.Operator):
    """Leave syncing mode. The clips stay lined up the way you left them"""
    bl_idname = "blendcap.multicam_sync_exit"
    bl_label = "Exit Syncing Mode"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if blendcap_state['running']:
            self.report({'WARNING'}, "A capture is running.")
            return {'CANCELLED'}
        sync_restore(context.scene)
        return {'FINISHED'}


# ============================================================
# CAPTURE
# ============================================================
def _take_id(name, count):
    return _sanitize_clip_name(
        f"{_sanitize_clip_name(_strip_media_extension(name))}_{count}cam")


def clip_id_for(clips):
    """The capture's identity: clip A's name plus the camera count, not
    every name joined (four joined names run into the Windows path
    limit). 'A0314' with four cameras -> 'A0314_4cam'. From the strip's
    NAME, so a renamed clip A is a take of its own (Blender's '.001' on
    a duplicate gives 'A0314_001_4cam')."""
    return _take_id(clips[0].name, len(clips))


def _folder_has_capture(folder):
    if not os.path.isdir(folder):
        return False
    names = ["mhr_frames.npz", "face_frames.npz"]
    return any(os.path.exists(os.path.join(d, n))
               for d in [folder] + cache_subdirs(folder) for n in names)


def take_collision(context, clips):
    """The single-camera capture's cache-collision guard, for a take.

    A capture into a folder that already holds ANOTHER capture would
    replace it without a word: after a capture the original clips are
    gone from the timeline (the multiview strip stands in for them), so
    the same file dropped in again takes the plain name, resolves to
    the same take name, and would overwrite the capture the multiview
    strip still points at.

    Clips that were themselves captured as this take carry its name
    (TAKE_TAG: stamped when Run Capture starts, restored by Revert), so
    running the same take again, after Revert or after a cancelled run,
    is not a collision: that is what lets a re-run reuse the earlier
    work. Returns (take id, a free name for clip A) or None."""
    cid = clip_id_for(clips)
    if not _folder_has_capture(os.path.join(_cache_dir(), f"Capture_{cid}")):
        return None
    if all(str(s.get(TAKE_TAG, "") or "") == cid for s in clips):
        return None
    return cid, _next_free_take_name(context, clips)


def _next_free_take_name(context, clips):
    """Blender-style '.NNN' name for clip A that is free in the
    Sequencer AND whose take folder does not exist yet."""
    stem = clips[0].name
    if '.' in stem:
        head, tail = stem.rsplit('.', 1)
        if tail.isdigit() and head:
            stem = head
    seq_ed = context.scene.sequence_editor
    taken = {s.name for s in _seq_accessors(seq_ed)[0]} if seq_ed else set()
    base = _cache_dir()
    for i in range(1, 1000):
        cand = f"{stem}.{i:03d}"
        if cand in taken:
            continue
        if os.path.exists(os.path.join(
                base, f"Capture_{_take_id(cand, len(clips))}")):
            continue
        return cand
    return f"{stem}.001"


def _sound_record(snd):
    if snd is None:
        return None
    try:
        return {
            "name": snd.name,
            "path": str(getattr(getattr(snd, "sound", None), "filepath", "") or ""),
            "channel": int(snd.get(HOME_TAG, snd.channel)),
            "frame_start": _strip_frame_start(snd),
            "vis_start": _strip_visible_start(snd),
            "vis_end": _strip_visible_end(snd),
            "volume": float(getattr(snd, "volume", 1.0)),
        }
    except Exception:
        return None


def _camera_record(seq_ed, strip, index, lo, focal):
    t = strip.transform
    try:
        xform = [float(v) for v in strip["blendcap_sync_xform"]]
        blend = str(strip["blendcap_sync_blend"])
    except Exception:
        xform = [t.scale_x, t.scale_y, t.offset_x, t.offset_y]
        blend = strip.blend_type
    off_s = _quiet(lambda: int(strip.frame_offset_start))
    vis_s = _strip_visible_start(strip)
    try:
        space = str(strip.colorspace_settings.name)
    except Exception:
        space = ""
    return {
        "letter": LETTERS[index],
        "name": strip.name,
        "path": str(strip.filepath),
        # while syncing the clip is nested: its channel outside the nest
        "channel": int(strip.get(HOME_TAG, strip.channel)),
        # where it was before Sync Clips re-stacked it (None: it was not)
        "from": _came_from(strip),
        "frame_start": _strip_frame_start(strip),
        "offset_start": off_s,
        "vis_start": vis_s,
        "vis_end": _strip_visible_end(strip),
        "frame_duration": _quiet(lambda: int(strip.frame_duration)),
        "colorspace": space,
        "xform": xform,
        "blend": blend,
        "stamp": str(strip.get("blendcap_source_strip", "") or ""),
        # the take this clip was last CAPTURED as ("" = never)
        "take": str(strip.get(TAKE_TAG, "") or ""),
        "size": list(_strip_size(strip)),
        # the first source frame this capture reads from this camera
        "cap_start": off_s + (lo - vis_s),
        "focal": float(focal),
        "sound": _sound_record(_linked_sound_strip(seq_ed, strip)),
    }


FPS_MODES = [
    ('MATCH', "Match Timing", "Drop or duplicate frames to match project speed"),
    ('KEEP', "Keep All Frames", "1:1 frame copy \u2014 changes playback speed"),
    ('SKIP', "Skip Conversion", "No conversion \u2014 track original video directly"),
]


def clip_rate(path):
    """(frames per second, frame count) of a video file, (0, 0) when
    Blender can't read it."""
    try:
        clip = bpy.data.movieclips.load(path, check_existing=False)
        fps = float(clip.fps) if clip.fps > 0 else 0.0
        frames = int(getattr(clip, 'frame_duration', 0) or 0)
        bpy.data.movieclips.remove(clip)
        return fps, frames
    except Exception:
        return 0.0, 0


# Two rates count as "the same" within this fraction. It is wide enough
# for the broadcast pairs (29.97 / 30, 23.976 / 24, 59.94 / 60 are all
# 0.1% apart): some cameras cannot shoot the round number and some
# cannot shoot the fractional one, and a shoot scraped together from
# whatever cameras are to hand will mix them.
RATE_SAME = 0.0015
# ...and above this fraction two rates are different clocks: 0.1% is a
# frame of drift every 1001 frames (33 s at 30 fps), which over a long
# take the merge would read as the cameras disagreeing. Such a camera
# is resampled onto the common clock (build_capture).
RATE_EXACT = 0.0001


def rates_match(a, b, tol=RATE_SAME):
    return abs(a - b) <= tol * max(a, b, 1e-9)


def rate_check(context, clips):
    """How the clips' frame rates sit against each other and against
    the project: {'rates': [(fps, frames)...], 'same': bool, 'fps':
    camera A's rate, 'project': the project's, 'mismatch': bool,
    'mixed': bool}.

    The clips have to match EACH OTHER, to within the broadcast
    fraction (RATE_SAME). 'mixed' says they are within it but not
    identical (a 29.97 camera next to a 30 one). Against the PROJECT
    they may differ: that is what the conversion choice is for, the
    same one a single clip gets."""
    return rate_check_paths(
        context, [bpy.path.abspath(s.filepath) for s in clips])


def rate_check_paths(context, paths):
    """rate_check for a list of video files."""
    scene = context.scene
    project = scene.render.fps / scene.render.fps_base
    rates = [clip_rate(p) for p in paths]
    known = [r[0] for r in rates if r[0] > 0]
    fps = rates[0][0] if (rates and rates[0][0] > 0) else (known[0] if known else project)
    same = all(rates_match(a, known[0]) for a in known) if known else True
    mixed = same and any(not rates_match(a, known[0], RATE_EXACT) for a in known)
    return {"rates": rates, "same": same, "fps": fps, "project": project,
            "mismatch": same and abs(fps - project) > 0.5, "mixed": mixed}


def rates_differ_text(rates):
    parts = ", ".join(f"{LETTERS[i]} {r[0]:g}" for i, r in enumerate(rates))
    return (f"The clips run at different frame rates ({parts} fps). "
            f"Re-export them to match each other.")


def frame_ratio(strip_duration, source_frames, source_fps, project_fps):
    """Source frames per timeline frame for one strip whose file runs at
    another rate than the project. Blender either keeps such a strip at
    one timeline frame per source frame (it then plays at the wrong
    speed), or retimes it to play at the right speed, and then a
    timeline frame is source_fps / project_fps source frames. Told
    apart the way the single-camera capture does: by whether the
    strip's length is the file's frame count."""
    if source_frames > 0 and strip_duration > 0 \
            and abs(strip_duration - source_frames) <= 1:
        return 1.0
    if project_fps <= 0:
        return 1.0
    return source_fps / project_fps


def _face_cache_ok(face_npz):
    """A cached face capture is only reusable with mouth-stats v3 (the
    same rule the single-camera capture applies)."""
    try:
        import numpy as np
        with np.load(face_npz, allow_pickle=True) as f:
            v = int(f["mouth_stats_version"]) if "mouth_stats_version" in f else 1
            return "mouth_stats" in f and v >= 3
    except Exception:
        return True


def _people_faces(folder):
    """The face captures of a multi-person capture's people (one file
    per person whose face was found)."""
    sub = os.path.join(folder, "people")
    try:
        return [os.path.join(sub, n) for n in sorted(os.listdir(sub))
                if n.startswith("track_") and n.endswith("_face.npz")]
    except OSError:
        return []


def preview_strip(context):
    """(strip, record) when the active Sequencer strip is a multi-camera
    PREVIEW, else (None, None). Run Preview and Run Capture work straight
    from it: its record holds every clip, trim and slide."""
    try:
        if context.scene.blendcap_props.video_source != 'VSE':
            return None, None
    except Exception:
        return None, None
    strip = active_multicam_strip(context)
    rec = strip_record(strip)
    if rec is not None and rec.get("preview"):
        return strip, rec
    return None, None


def record_collision(record):
    """True when a capture made from this preview's record would go into
    a folder that holds ANOTHER capture (take_collision's test, from the
    record: the clips themselves are not in the timeline). There is no
    renaming clip A from here, so the capture is refused and the user
    is sent back to the clips."""
    cid = str(record.get("clip_id", ""))
    if not _folder_has_capture(os.path.join(_cache_dir(), f"Capture_{cid}")):
        return False
    return not all(str(c.get("take", "") or "") == cid
                   for c in record.get("cams", []))


COLLISION_LINES = ["Another capture already has this take's name.",
                   "Revert to Original Clips and capture from there."]


def strip_blockers(context, record, preview=False):
    """Why Run Capture (or, with preview, Run Preview) can't start from
    a preview strip, as lines for the panel. Empty when it can."""
    props = context.scene.blendcap_props
    lines = []
    n = len(record.get("cams", []))
    if not preview and any(focal_mm(props, i) <= 0.0 for i in range(n)):
        lines.append("Enter each camera's focal length.")
    if not props.track_body:
        lines.append("Body tracking must be on.")
    if not preview and record_collision(record):
        lines += COLLISION_LINES
    # (last: it is one sentence over several lines, with no full stop)
    try:
        lines += too_short_lines(context.scene,
                                 int(record["hi"]) - int(record["lo"]), preview)
    except (KeyError, TypeError, ValueError):
        pass
    return lines


def build_capture_record(context, record, fps_mode='MATCH', preview=False):
    """build_capture from a preview strip's record instead of the clips:
    the same cameras, frames and take name, with the focal lengths as
    the panel has them now."""
    props = context.scene.blendcap_props
    cams = [dict(c, focal=float(focal_mm(props, i)))
            for i, c in enumerate(record.get("cams", []))]
    return _build_plan(context, cams, int(record["lo"]), int(record["hi"]),
                       str(record["clip_id"]), fps_mode, 'VSE', preview=preview)


def build_capture(context, clips, fps_mode='MATCH', preview=False):
    """Everything Run Capture is about to do with the synced strips,
    without doing it: (plan, "") or (None, why not). The plan holds the
    commands, the per-step bookkeeping the progress monitor needs, and
    the record that ends up on the multiview strip.

    fps_mode: what to do when the clips run at another frame rate than
    the project (the single-camera capture's three choices).
    preview: the plan of Run Preview instead (see _build_plan)."""
    scene = context.scene
    props = scene.blendcap_props
    lo, hi = overlap(clips)
    if lo >= hi:
        return None, "The clips don't overlap in time."
    cams = [_camera_record(scene.sequence_editor, s, i, lo, focal_mm(props, i))
            for i, s in enumerate(clips)]
    return _build_plan(context, cams, lo, hi, clip_id_for(clips), fps_mode, 'VSE',
                       preview=preview)


def build_capture_files(context, fps_mode='MATCH', preview=False):
    """The same plan for the File Path clips: every clip from its first
    frame, for as long as the shortest runs, one focal length for all
    (they are taken to be synced and from identical cameras)."""
    props = context.scene.blendcap_props
    infos = file_infos(props)
    if len(infos) < MIN_CAMERAS:
        return None, "Select at least 2 clips."
    frames = [i["frames"] for i in infos if i.get("frames", 0) > 0]
    n_frames = min(frames) if frames else 0
    cams = [{
        "letter": LETTERS[k],
        "name": os.path.basename(i["path"]),
        "path": i["path"],
        "channel": 1,
        # one timeline frame is one frame of the file
        "frame_duration": int(i.get("frames", 0)),
        "size": [int(i.get("w", 0)), int(i.get("h", 0))],
        "cap_start": 0,
        "focal": float(props.multicam_focal_mm),
        "sound": None,
    } for k, i in enumerate(infos)]
    return _build_plan(context, cams, 1, 1 + n_frames, files_clip_id(infos),
                       fps_mode, 'FILE', preview=preview)


def _build_plan(context, cams, lo, hi, clip_id, fps_mode, source, preview=False):
    """The plan both sources share. `cams`: one record per camera, A
    first, saying where its footage is and which frame of it the first
    captured frame is. lo / hi: the frames captured, in timeline frames
    (hi == lo means "the whole clips", File Path with unknown lengths).

    preview=True is Run Preview's plan: per camera, the person detector
    alone on every Nth frame of the SAME frames a capture would read
    (the same frame-rate handling), a box on whoever it finds, then the
    same multiview video. It reads and writes nothing of a capture:
    its files are preview.mp4 / preview_source.mp4 next to the
    capture's own, no cache record is made, nothing is cleared, and
    there is no merge. Focal lengths are not needed."""
    from .operators_pipeline import (
        _append_frame_args, _classify_step, _is_supported_video,
        _capture_use_ort_override, _capture_force_cpu, refine_hands_effective)
    from .helpers import block_if_path_too_long, capture_folder_checks

    scene = context.scene
    props = scene.blendcap_props
    n = len(cams)
    n_frames = hi - lo
    for cam in cams:
        if cam["focal"] <= 0.0 and not preview:
            return None, ("Enter each camera's focal length." if source == 'VSE'
                          else "Enter the cameras' focal length.")
    paths = [bpy.path.abspath(c["path"]) for c in cams]

    out_dir = os.path.join(_cache_dir(), f"Capture_{clip_id}")
    cam_dirs = [os.path.join(out_dir, f"cam_{c['letter'].lower()}") for c in cams]

    # Windows path limit (item 10), before the "not found" test: Blender
    # can't see a file whose path is too long.
    checks = [(p, False, 'video') for p in paths if p]
    checks += capture_folder_checks(out_dir)
    for d in cam_dirs:
        checks += capture_folder_checks(d)
    if block_if_path_too_long(checks):
        return None, ""
    for cam, p in zip(cams, paths):
        if not p or not os.path.exists(p):
            return None, f"Video for '{cam['name']}' not found on disk."

    # The clips must run at ONE frame rate (to the broadcast fraction):
    # the cameras are lined up frame by frame, and a clip at another
    # rate drifts against the others for the whole take. A rate that
    # differs from the PROJECT's is handled per fps_mode, as for a
    # single clip.
    rc = rate_check_paths(context, paths)
    if not rc["same"]:
        return None, rates_differ_text(rc["rates"])
    src_fps, mismatch = rc["fps"], rc["mismatch"]
    convert = mismatch and fps_mode in ('MATCH', 'KEEP')
    # The clock every camera is put on. Converting to the project's
    # rate with Match Timing, that is the project's (each camera is
    # resampled from ITS OWN rate, so a 29.97 one and a 30 one land
    # together). Otherwise camera A's: a camera a fraction off A's rate
    # is resampled onto it (one frame repeated or dropped every 1001).
    # Keep All Frames and Skip Conversion promise to leave the frames
    # alone, so there a mixed set drifts, and the result says so.
    to_project = mismatch and fps_mode == 'MATCH'
    clock = project_clock = rc["project"]
    if not to_project:
        clock = src_fps
    drift_left = rc["mixed"] and mismatch and fps_mode in ('KEEP', 'SKIP')
    if mismatch:
        print(f"[BlendCap] FPS mismatch: clips={src_fps:.2f}fps, "
              f"project={rc['project']:.2f}fps. Mode: {fps_mode}.")

    addon_dir = _addon_dir()
    python_exe = _python_exe()
    if not os.path.exists(python_exe):
        return None, "DEPS"
    project_fps = scene.render.fps / scene.render.fps_base
    cap_ort = _capture_use_ort_override(context)
    cap_cpu = _capture_force_cpu(context)
    no_hands = not refine_hands_effective(context)
    # Multi-Camera & Multi-Person: every camera's capture is a
    # Multi-Person capture (everybody it sees, followed and saved)
    people = is_people_mode(props)

    cmds, prefixes = [], []
    cam_params, cam_steps, stale_faces = [], [], []
    body_steps = []          # per camera: its body step's index, or None
    face_steps = {}          # step index -> that camera's folder
    any_body = False
    for cam, path, cam_dir, rate in zip(cams, paths, cam_dirs, rc["rates"]):
        letter = cam["letter"]
        prefix = f"Camera {letter}: "
        start = int(cam["cap_start"])
        body_npz = os.path.join(cam_dir, "mhr_frames.npz")
        face_npz = os.path.join(cam_dir, "face_frames.npz")
        overlay_mp4 = os.path.join(cam_dir, "overlay.mp4")
        export = os.path.join(cam_dir, "preview_source.mp4" if preview
                              else "source_export.mp4")
        sig = (f"nh={int(no_hands)};fov=MANUAL;focal={cam['focal']:.4f};"
               f"skip={props.capture_skip};ort={cap_ort};"
               f"cpu={int(bool(cap_cpu))}")
        cam_fps = rate[0] if rate[0] > 0 else src_fps
        # this camera is a fraction off the common clock: resample it
        resample = (not mismatch and rc["mixed"]
                    and not rates_match(cam_fps, clock, RATE_EXACT))
        if mismatch:
            sig += f";fps={fps_mode};src={cam_fps:.3f}"
        elif resample:
            sig += f";resample={cam_fps:.3f}>{clock:.3f}"
        if people:
            # only ever added: an ordinary multi-camera capture's
            # signature is as it was. The two kinds never pass for each
            # other: a camera's files hold one performer or everybody.
            sig += ";people=1"
        # Reuse a camera's capture only when it was made from the SAME
        # frames of the SAME file with the same settings (the exact rule
        # the single-camera capture uses). A camera whose clip was slid
        # is captured again; the ones left alone are not.
        # (a preview never reuses anything: its converted copy is made
        # again, and the capture's own record is not consulted)
        params_ok = (not preview) and _cache_params_match(
            cam_dir, start, n_frames, 0, 0, project_fps, sig, source_path=path)
        steps = []

        # What the tracking reads. `start` / `n_frames` are in TIMELINE
        # frames; at another rate than the project's they become source
        # frames first (frame_ratio). With a conversion the camera's
        # frames are re-rated into its own folder and tracked from
        # there, whole (the single-camera capture's first step).
        video, t_start, t_max = path, start, n_frames
        if mismatch or resample:
            k = frame_ratio(int(cam.get("frame_duration", 0)), rate[1],
                            cam_fps, rc["project"])
            t_start, t_max = int(round(start * k)), int(round(n_frames * k))
        # (a format OpenCV does not read is copied into one it does, the
        # way the single-camera capture does it)
        reformat = not _is_supported_video(path)
        if convert or resample or reformat:
            if not params_ok or not os.path.exists(export):
                cmd = [python_exe, os.path.join(addon_dir, "convert_source.py"),
                       "--video", path, "--output", export,
                       "--fps", str(clock if (resample and not convert) else project_fps),
                       "--start-frame", str(t_start),
                       "--max-frames", str(t_max),
                       "--source-fps", str(cam_fps)]
                if resample or (convert and fps_mode == 'MATCH'):
                    # the threshold lets a 0.1% difference through
                    cmd.extend(["--mode", "match",
                                "--match-threshold", "0.0005"])
                cmds.append(cmd)
                prefixes.append(prefix)
            video, t_start, t_max = export, 0, 0
        elif os.path.exists(export) and not preview:
            # a converted copy from an earlier run with another choice:
            # it is no longer what this capture reads
            stale_faces.append(export)

        if preview:
            # The person detector alone, a box on whoever it finds
            # (everybody, in Multi-Camera & Multi-Person). Preview Skip
            # counts SKIPPED frames; the script's flag is "every Nth".
            pv_out = os.path.join(cam_dir, "preview.mp4")
            pv_step = str(int(props.preview_step) + 1)
            cmd = [python_exe, os.path.join(addon_dir, "save_mhr_data.py"),
                   "--video", video, "--preview",
                   "--preview-step", pv_step,
                   # (the camera letter takes the top left of its tile)
                   "--preview-status-right",
                   "--output", pv_out]
            _append_frame_args(cmd, t_start, t_max)
            if people:
                cmd.append("--people")
            if props.track_face:
                # With Face on the camera's preview is rendered ONCE, by
                # the face step, from the body step's detections (the
                # single-camera combined preview). A camera that sees
                # no face says NO FACE in its tile; it is not a failure.
                pv_boxes = os.path.join(cam_dir, "preview_boxes.json")
                cmd.extend(["--boxes-out", pv_boxes])
            cmds.append(cmd)
            prefixes.append(prefix)
            if props.track_face:
                fcmd = [python_exe, os.path.join(addon_dir, "save_face_data.py"),
                        "--video", video, "--preview", "--output", pv_out,
                        "--preview-step", pv_step,
                        "--body-boxes", pv_boxes, "--preview-status-right"]
                _append_frame_args(fcmd, t_start, t_max)
                cmds.append(fcmd)
                prefixes.append(prefix)
            cam_steps.append([])
            body_steps.append(None)
            continue

        # A face file from a capture this camera's record no longer
        # matches cannot belong to what is captured now. It goes before
        # the run starts, not when the new face step ends: a camera is
        # recorded as captured as soon as its BODY is done, and if the
        # run is then cancelled during its face, the old face file must
        # not be there to pass for the new capture's.
        if not params_ok and os.path.exists(face_npz):
            stale_faces.append(face_npz)

        body_step = None
        body_cached = (params_ok and check_npz_coverage(
            body_npz, t_start, t_max)[0] == 'covers')
        if people:
            # the camera's own body file is one of its people, who may
            # not be there for the whole clip: its table of people is
            # what says the capture is complete
            body_cached = (params_ok and os.path.isfile(body_npz) and os.path.isfile(
                os.path.join(cam_dir, "people.json")))
        if body_cached:
            print(f"[BlendCap] Camera {letter}: body data already captured, reusing it.")
        else:
            body_step = len(cmds)
            cmd = [python_exe, os.path.join(addon_dir, "save_mhr_data.py"),
                   "--video", video, "--output", body_npz,
                   "--focal-mm", f"{cam['focal']:.4f}"]
            _append_frame_args(cmd, t_start, t_max)
            if no_hands:
                cmd.append("--no-hands")
            if props.capture_skip > 0:
                cmd.extend(["--capture-skip", str(props.capture_skip)])
            if people:
                cmd.append("--people")
            steps.append(len(cmds))
            cmds.append(cmd)
            prefixes.append(prefix)
            any_body = True

        # A camera that saw no face in exactly these frames (a camera
        # behind the performer) is not asked again on a re-run. The mark
        # only counts next to a capture it belongs to.
        no_face_mark = os.path.join(cam_dir, NO_FACE_MARK)
        if not params_ok and os.path.exists(no_face_mark):
            stale_faces.append(no_face_mark)
        if props.track_face:
            face_cached = (params_ok
                           and check_npz_coverage(face_npz, t_start, t_max)[0] == 'covers'
                           and _face_cache_ok(face_npz))
            if people:
                # the camera's own face file is its most prominent
                # person's, and is only there when THEIR face was found:
                # the faces that were found are next to the tracks
                found = _people_faces(cam_dir)
                face_cached = (params_ok and body_cached and bool(found)
                               and all(_face_cache_ok(p) for p in found))
            if face_cached:
                print(f"[BlendCap] Camera {letter}: face data already captured, reusing it.")
            elif params_ok and body_cached and os.path.exists(no_face_mark):
                print(f"[BlendCap] Camera {letter}: no face was found in this clip "
                      f"last time, not tracking its face again.")
            else:
                cmd = [python_exe, os.path.join(addon_dir, "save_face_data.py"),
                       "--video", video, "--output", face_npz]
                if people:
                    # one face pass per person this camera saw, each
                    # cropped to their own head from their own track
                    cmd.extend(["--people", cam_dir])
                else:
                    cmd.extend(["--body", body_npz])
                    if body_cached:
                        # (a body step in this run hands its frame range
                        # over through the NPZ; a reused one needs it
                        # spelled out)
                        _append_frame_args(cmd, t_start, t_max)
                face_steps[len(cmds)] = cam_dir
                steps.append(len(cmds))
                cmds.append(cmd)
                prefixes.append(prefix)
        elif (not body_cached and os.path.exists(face_npz)
              and face_npz not in stale_faces):
            # Face is off and this camera's body is being captured anew:
            # a face file left from an earlier capture would no longer
            # belong to it.
            stale_faces.append(face_npz)

        overlay_cmd = [python_exe, os.path.join(addon_dir, "overlay_track.py"),
                       "--video", video, "--output", overlay_mp4,
                       "--input", body_npz]
        # Drawn with the face when a face file will be there: one this
        # run makes, or one already captured that still belongs.
        face_this_run = bool(steps) and any(i in face_steps for i in steps)
        if face_this_run or (os.path.exists(face_npz)
                             and face_npz not in stale_faces):
            overlay_cmd.extend(["--face", face_npz])
        if people:
            overlay_cmd.extend(["--people", cam_dir])
        cmds.append(overlay_cmd)
        prefixes.append(prefix)

        cam_steps.append(steps)
        body_steps.append(body_step)
        cam_params.append({"dir": cam_dir, "start": start, "max": n_frames,
                           "fps": project_fps, "sig": sig, "path": path})

    # Same one-time pre-flight as the single-camera capture: the ORT
    # backbone is an optional download the capture itself never fetches.
    shift = 0
    if any_body and cap_ort == "1":
        ckpt = checkpoints_dir()
        bb = os.path.join(ckpt, "onnx", "backbone_dinov3_fp16.onnx")
        dl = os.path.join(_installer_dir(), "download_models.py")
        if os.path.isfile(dl) and not (os.path.isfile(bb)
                                       and os.path.isfile(bb + "_data")):
            cmds.insert(0, [python_exe, dl, "--project-root", addon_dir,
                            "--checkpoints-dir", ckpt, "--ensure-backbone-onnx"])
            prefixes.insert(0, "")
            shift = 1
    if shift:
        cam_steps = [[i + shift for i in steps] for steps in cam_steps]
        body_steps = [None if b is None else b + shift for b in body_steps]
        face_steps = {i + shift: d for i, d in face_steps.items()}

    if preview:
        pass          # nothing to match or merge: no capture was made
    elif people:
        # Who is who across the cameras, in place of the merge: the
        # take's one table of people. A camera captured again means the
        # old table (and the corrections made against it) is about
        # somebody else's tracks from the moment the run starts.
        if any(b is not None for b in body_steps):
            stale_faces += [p for p in (os.path.join(out_dir, "people.json"),
                                        os.path.join(out_dir, "people_edits.json"))
                            if os.path.exists(p)]
        cmds.append([python_exe,
                     os.path.join(addon_dir, "pipeline", "people_cameras.py"),
                     *cam_dirs[:MERGED_CAMERAS], "-o", out_dir])
        prefixes.append("Matching people: ")
    else:
        # An ordinary multi-camera capture into a folder that held a
        # Multi-Camera & Multi-Person take: the take's table of people
        # is what makes it one, and its cameras are all captured again
        # (their signatures differ).
        stale_faces += [p for p in (os.path.join(out_dir, "people.json"),
                                    os.path.join(out_dir, "people_edits.json"))
                        if os.path.exists(p)]
        # The merge. Every camera was captured over the same timeline
        # frames, so the sync offset between the files is zero by
        # construction.
        cmds.append([python_exe,
                     os.path.join(addon_dir, "pipeline", "two_camera_merge.py"),
                     *cam_dirs[:MERGED_CAMERAS], "-o", out_dir,
                     "--sync-offset", "0"])
        prefixes.append("Merging: ")

    # The multiview video. The addon owns the layout (the syncing view
    # uses the same one); the script only places pictures.
    canvas_w, canvas_h = multiview_canvas(scene)
    rects = tile_layout([tuple(c["size"]) for c in cams], canvas_w, canvas_h)
    layout = {"width": canvas_w, "height": canvas_h, "fps": 0,
              "tiles": [{"label": c["letter"],
                         "video": os.path.join(d, "preview.mp4" if preview
                                               else "overlay.mp4"),
                         "rect": [round(v, 2) for v in r]}
                        for c, d, r in zip(cams, cam_dirs, rects)]}
    layout_path = os.path.join(out_dir, "preview_layout.json" if preview
                               else "multiview_layout.json")
    multiview = os.path.join(out_dir, "preview.mp4" if preview else "overlay.mp4")
    cmds.append([python_exe, os.path.join(addon_dir, "multicam_overlay.py"),
                 "--layout", layout_path, "--output", multiview])
    prefixes.append("")

    wmap = {"convert_source": (5, "convert"),
            "download_models": (10, "download"), "save_mhr_data": (70, "body"),
            "save_face_data": (20, "face"), "overlay_track": (10, "overlay"),
            "two_camera_merge": (6, "merge"), "people_cameras": (3, "match"),
            "multicam_overlay": (8, "multiview")}
    weights, labels, outputs = [], [], []
    for cmd in cmds:
        w, lbl, out = _classify_step(os.path.basename(cmd[1]), cmd, wmap)
        weights.append(w)
        labels.append(lbl)
        outputs.append(out)

    notes = []
    if drift_left:
        fr = 1.0 / max(1e-9, max(abs(r[0] - src_fps) for r in rc["rates"] if r[0] > 0))
        choice = "Keep All Frames" if fps_mode == 'KEEP' else "Skip Conversion"
        notes += [f"The clips' frame rates differ slightly "
                  f"({', '.join(format(r[0], 'g') for r in rc['rates'])} fps).",
                  f"With {choice} the cameras drift apart by about one "
                  f"frame every {fr:.0f} seconds. Match Timing keeps them together."]

    record = {
        "version": 1,
        "source": source,
        "clip_id": clip_id,
        "n": n,
        "lo": int(lo),
        "hi": int(hi),
        "merged": min(n, MERGED_CAMERAS),
        "fps_mode": fps_mode if mismatch else "",
        "cams": cams,
    }
    if people:
        record["people"] = True
    if preview:
        record["preview"] = True
    return {
        "record": record,
        "clip_id": clip_id,
        "out_dir": out_dir,
        "cam_dirs": cam_dirs,
        "cmds": cmds,
        "prefixes": prefixes,
        "weights": weights,
        "labels": labels,
        "outputs": outputs,
        "cam_steps": cam_steps,
        "cam_params": cam_params,
        "body_steps": body_steps,
        "face_steps": face_steps,
        "stale_faces": stale_faces,
        "layout": layout,
        "layout_path": layout_path,
        "multiview": multiview,
        "notes": notes,
        "use_ort": cap_ort,
        "force_cpu": cap_cpu,
        "paths": paths,
    }, ""


class BLENDCAP_OT_multicam_capture(bpy.types.Operator):
    """Capture every camera angle and combine them into one result"""
    bl_idname = "blendcap.multicam_capture"
    bl_label = "Run Capture"

    # Run Preview is this operator with the flag on, as in the
    # single-camera capture (blendcap.generate_mocap).
    is_update_only: bpy.props.BoolProperty(default=False)

    @classmethod
    def description(cls, context, properties):
        if properties.is_update_only:
            return "Run a stepped, lower-quality preview of every camera angle"
        return cls.__doc__

    fps_mode: bpy.props.EnumProperty(name="Conversion", items=FPS_MODES,
                                     default='MATCH')
    collision_mode: bpy.props.EnumProperty(
        name="Existing Data",
        items=[
            ('NEW', "Capture as New Take",
             "Rename clip A and capture into its own cache folder, "
             "keeping the existing capture"),
            ('OVERWRITE', "Overwrite Existing",
             "Replace the capture data already cached under this name"),
        ],
        default='NEW')
    # Set by invoke() when the collision warning fired, so execute()
    # only acts on collision_mode after the user actually saw the dialog.
    collision_detected: bpy.props.BoolProperty(
        default=False, options={'HIDDEN', 'SKIP_SAVE'})
    collision_new_name: bpy.props.StringProperty(
        default="", options={'HIDDEN', 'SKIP_SAVE'})

    def invoke(self, context, event):
        # Two things worth asking about first, exactly as the
        # single-camera capture asks: the clips match each other but not
        # the project's frame rate, and the take's name already holds
        # another capture.
        scene = context.scene
        blendcap_state.pop('_multicam_fps', None)
        blendcap_state.pop('_multicam_collision', None)
        if not blendcap_state['running'] and is_dependencies_installed():
            rc = None
            preview = bool(self.is_update_only)
            if scene.blendcap_props.video_source == 'FILE':
                if not file_blockers(context, preview):
                    rc = rate_check_paths(context, [
                        bpy.path.abspath(i["path"])
                        for i in file_infos(scene.blendcap_props)])
            elif not sync_active(scene):
                # straight from a preview strip
                _pv, pv_rec = preview_strip(context)
                if pv_rec is not None and not strip_blockers(context, pv_rec, preview):
                    rc = rate_check_paths(context, [
                        bpy.path.abspath(str(c["path"])) for c in pv_rec["cams"]])
            elif not run_blockers(context, preview):
                clips = pinned_strips(scene)
                rc = rate_check(context, clips) if clips else None
                # (a preview replaces no capture: nothing to ask about)
                hit = (take_collision(context, clips)
                       if clips and not preview else None)
                if hit:
                    self.collision_detected = True
                    self.collision_new_name = hit[1]
                    blendcap_state['_multicam_collision'] = hit
            if rc and rc["mismatch"]:
                blendcap_state['_multicam_fps'] = (rc["fps"], rc["project"])
            if (blendcap_state.get('_multicam_fps')
                    or blendcap_state.get('_multicam_collision')):
                return context.window_manager.invoke_props_dialog(self, width=420)
        return self.execute(context)

    def draw(self, context):
        layout = self.layout
        fps = blendcap_state.get('_multicam_fps')
        hit = blendcap_state.get('_multicam_collision')
        if fps:
            clip_fps, proj_fps = fps
            layout.label(text=f"Framerate mismatch: clips {clip_fps:.0f}fps, "
                              f"project {proj_fps:.0f}fps.", icon='INFO')
            layout.prop(self, "fps_mode")
            if self.fps_mode == 'MATCH':
                layout.label(text="Drops frames to match project timing."
                             if clip_fps > proj_fps else
                             "Duplicates frames to match project timing.")
            elif self.fps_mode == 'KEEP':
                layout.label(text="Plays in slow motion (all frames preserved)."
                             if clip_fps > proj_fps else
                             "Plays in fast motion (all frames preserved).")
            else:
                layout.label(text="Original videos passed directly to tracking.")
            layout.separator()
        if hit:
            take, new_name = hit
            layout.label(text=f"Capture data for '{take}' already exists.",
                         icon='ERROR')
            layout.label(text="Choose an action:")
            layout.prop(self, "collision_mode", text="")
            if self.collision_mode == 'NEW':
                layout.label(text=f"Clip A will be renamed to '{new_name}'.")
            else:
                layout.label(text=f"The cached data for '{take}' will be replaced.")
            layout.separator()
        layout.label(text="Continue?")

    def execute(self, context):
        from .mocap import run_mocap_thread, monitor_progress_timer
        blendcap_state.pop('_multicam_fps', None)
        blendcap_state.pop('_multicam_collision', None)

        if blendcap_state['running']:
            self.report({'WARNING'}, "A capture is already running.")
            return {'CANCELLED'}
        if not is_dependencies_installed():
            bpy.ops.blendcap.deps_missing('INVOKE_DEFAULT')
            return {'CANCELLED'}
        scene = context.scene
        preview = bool(self.is_update_only)
        from_files = scene.blendcap_props.video_source == 'FILE'
        if from_files:
            blocked = file_blockers(context, preview)
            if blocked:
                self.report({'ERROR'}, first_sentence(blocked))
                return {'CANCELLED'}
            plan, why = build_capture_files(context, self.fps_mode, preview=preview)
        elif not sync_active(scene) and preview_strip(context)[1] is not None:
            # Straight from a preview strip: the same take, from its
            # record. The result replaces the preview strip.
            pv_rec = preview_strip(context)[1]
            blocked = strip_blockers(context, pv_rec, preview)
            if blocked:
                self.report({'ERROR'}, " ".join(blocked) if blocked == COLLISION_LINES
                            else first_sentence(blocked))
                return {'CANCELLED'}
            plan, why = build_capture_record(context, pv_rec, self.fps_mode,
                                             preview=preview)
        else:
            if not sync_active(scene):
                self.report({'ERROR'}, "Press Sync Clips first.")
                return {'CANCELLED'}
            clips = pinned_strips(scene)
            if not clips:
                self.report({'ERROR'}, "A synced clip was removed.")
                return {'CANCELLED'}
            blocked = run_blockers(context, preview)
            if blocked:
                self.report({'ERROR'}, first_sentence(blocked))
                return {'CANCELLED'}
            # The collision dialog's "Capture as New Take": give clip A
            # the free name BEFORE the take is named after it, which is
            # what routes the run into a folder of its own. The pin is
            # by name, so it follows.
            if (self.collision_detected and self.collision_mode == 'NEW'
                    and self.collision_new_name and not preview):
                clips[0].name = self.collision_new_name
                scene[SYNC_CLIPS] = [s.name for s in clips]
                print(f"[BlendCap] Clip A renamed to '{clips[0].name}' to "
                      f"protect the existing capture.")
            # The record stores where each clip's sound is, for Revert:
            # a sound left behind by a hand-dragged clip goes back first.
            restep_sounds(scene)
            plan, why = build_capture(context, clips, self.fps_mode, preview=preview)
            if plan is not None and not preview:
                # These clips ARE this take from here on: running it
                # again (after a cancel, or after Revert) is no collision.
                for s in clips:
                    s[TAKE_TAG] = plan["clip_id"]
        if plan is None:
            if why == "DEPS":
                bpy.ops.blendcap.deps_missing('INVOKE_DEFAULT')
            elif why:
                self.report({'ERROR'}, why)
            return {'CANCELLED'}

        for d in (plan["out_dir"], *plan["cam_dirs"]):
            os.makedirs(d, exist_ok=True)
        with open(plan["layout_path"], "w", encoding="utf-8") as f:
            json.dump(plan["layout"], f, indent=1)
        for p in plan["stale_faces"]:
            try:
                os.remove(p)
            except OSError:
                pass

        rec = plan["record"]
        cam_a = rec["cams"][0]
        props = scene.blendcap_props
        st = blendcap_state
        st['addon_dir'] = _addon_dir()
        st['python_exe'] = _python_exe()
        st['track_body'] = True
        st['track_face'] = bool(props.track_face)
        st['output_dir'] = plan["out_dir"]
        st['vid_name'] = f"Capture_{plan['clip_id']}"
        st['clip_name'] = plan["clip_id"]
        st['source_strip_name'] = plan["clip_id"]
        st['is_preview'] = preview
        st['actual_video'] = plan["paths"][0]
        st['vse_replace'] = None
        st['step_labels'] = plan["labels"]
        st['step_outputs'] = plan["outputs"]
        st['multicam'] = {
            "record": rec,
            "cam_steps": plan["cam_steps"],
            "cam_params": plan["cam_params"],
            "body_steps": plan["body_steps"],
            "face_steps": plan["face_steps"],
            "notes": plan["notes"],
            "out_dir": plan["out_dir"],
            "preview": preview,
        }
        # Pre-set before the thread starts so the monitor (registered
        # with first_interval=0) can't read a previous run's leftovers.
        st['running'] = True
        st['return_code'] = 0
        st['cancel_requested'] = False
        st['run_started'] = time.time()

        print(f"[BlendCap] {'Multi-Camera & Multi-Person' if rec.get('people') else 'Multi-Camera'} "
              f"{'Preview' if preview else 'Capture'} ({'File Path' if from_files else 'Sequencer'}): "
              f"{', '.join(c['letter'] + '=' + c['name'] for c in rec['cams'])} "
              f"-> {plan['out_dir']} (frames {rec['lo']}-{rec['hi']}, "
              f"focals {'/'.join(format(c['focal'], '.1f') for c in rec['cams'])} mm)")

        thread = threading.Thread(
            target=run_mocap_thread,
            args=(plan["cmds"], plan["multiview"], rec["lo"], cam_a["channel"],
                  plan["paths"][0], cam_a["cap_start"], rec["hi"] - rec["lo"],
                  'FILE' if from_files else 'VSE', plan["weights"]),
            kwargs={'use_ort': plan["use_ort"], 'force_cpu': plan["force_cpu"],
                    'step_prefixes': plan["prefixes"]},
            daemon=True)
        thread.start()
        bpy.app.timers.register(monitor_progress_timer)
        self.report({'INFO'}, f"Previewing {rec['n']} cameras..." if preview
                    else f"Capturing {rec['n']} cameras...")
        return {'FINISHED'}


# ============================================================
# AFTER THE RUN (called by mocap.monitor_progress_timer)
# ============================================================
def write_camera_params(state):
    """Record, per camera, what its capture was made from, so a later
    run can reuse it. A camera is recorded as soon as its BODY capture
    ran to the end in this run: a run cancelled (or failed) during that
    camera's face capture keeps the body, as the single-camera capture
    does, and the next run tracks only the face. The face file itself
    says whether the face is there; an old one was cleared before the
    run started (build: stale_faces)."""
    mc = state.get('multicam') or {}
    done = set(state.get('steps_completed', []) or [])
    done |= {int(e.get("step", -1))
             for e in (state.get('multicam_face_skipped') or [])}
    cam_steps = mc.get("cam_steps", [])
    body_steps = list(mc.get("body_steps") or []) + [None] * len(cam_steps)
    for steps, body, p in zip(cam_steps, body_steps, mc.get("cam_params", [])):
        if (body is not None and body in done) \
                or (steps and all(i in done for i in steps)):
            _write_cache_params(p["dir"], p["start"], p["max"], 0, 0,
                                p["fps"], p["sig"], p["path"])
    # A camera whose clip holds no face: mark it, so a re-run with the
    # same frames does not track its face again. (Only "no face found",
    # never a face step that failed for another reason.)
    face_steps = {int(k): v for k, v in (mc.get("face_steps") or {}).items()}
    for e in (state.get('multicam_face_skipped') or []):
        d = face_steps.get(int(e.get("step", -1)))
        if d and e.get("no_face"):
            try:
                with open(os.path.join(d, NO_FACE_MARK), "w", encoding="utf-8") as f:
                    f.write("No face was found in this camera's clip.\n")
            except OSError:
                pass


def result_lines(state):
    """What the user should be told after a multi-camera run: cameras
    whose face capture was dropped, and what the merge could not settle
    by itself."""
    mc = state.get('multicam') or {}
    if mc.get("preview"):
        # no capture and no merge: only what the plan itself noted
        return list(mc.get("notes", []))
    lines = []
    steps = mc.get("cam_steps", [])
    cams = (mc.get("record") or {}).get("cams", [])
    for e in (state.get('multicam_face_skipped') or []):
        letter = "?"
        for cam, ss in zip(cams, steps):
            if int(e.get("step", -1)) in ss:
                letter = cam["letter"]
        if e.get("no_face"):
            lines.append(f"Camera {letter}: no face was found. "
                         f"Its body capture was kept.")
        else:
            lines.append(f"Camera {letter}: face capture failed "
                         f"(exit code {e.get('rc')}). Its body capture was kept.")
    return lines + list(mc.get("notes", [])) + merge_lines(mc.get("out_dir"))


def merge_lines(out_dir):
    """What the merge found that the user has something to do about:
    its fine sync, when the clips are too far out of step for it to
    correct. What it settled by itself is in the console and is not
    repeated here: the result popup is a warning dialog. A take it
    could not level (calibration's level_act) is in the console only
    too, by the user's call: the merge prints it."""
    try:
        import numpy as np
        with np.load(os.path.join(out_dir, "calibration.npz")) as cal:
            if "sync_act" in cal.files and int(cal["sync_act"]):
                # (sync_act_text: only the cameras that need it; a take
                # merged before that was written has the one line)
                text = str(cal["sync_act_text"] if "sync_act_text"
                           in cal.files else cal["sync_text"])
                return [t for t in text.splitlines() if t]
    except Exception:
        pass
    return []


def load_result(proxy_path, record):
    """Show a finished capture: the multiview strip in the Sequencer,
    or for a File Path capture the multiview on the viewport plane."""
    if record.get("source") == 'FILE':
        return load_to_viewport(proxy_path, record)
    return load_to_vse(proxy_path, record)


def load_to_viewport(proxy_path, record):
    """File Path: show the multiview on the BlendCap viewport plane (the
    single-camera File Path capture's own loader), and stamp the plane
    with the capture's identity and record, so Generate BVH, Clear
    Cache and Add Camera find the capture from it."""
    from .mocap import load_proxy_to_viewport
    if not proxy_path or not os.path.exists(proxy_path):
        return None
    cid = record["clip_id"]
    load_proxy_to_viewport(proxy_path,
                           bpy.path.abspath(str(record["cams"][0]["path"])),
                           strip_name=f"{'Preview' if record.get('preview') else 'Capture'}_{cid}",
                           source_strip=cid)
    for obj in bpy.data.objects:
        if (obj.type == 'EMPTY' and "blendcap_source" in obj
                and str(obj.get("blendcap_source_strip", "")) == cid):
            obj[STRIP_TAG] = json.dumps(record)
            return obj
    return None


def load_to_vse(proxy_path, record):
    """Replace the component clips with the one multiview strip.

    The strip carries the capture's identity the same way a single
    capture's overlay does (blendcap_source / blendcap_source_strip), so
    Generate BVH and Clear Cache find the capture from it, plus the
    record Revert rebuilds the clips from. Sound: ONE sound strip stays,
    the one heard while syncing (camera A's, or the first camera's after
    it that has real sound), directly under the new strip when that
    place is free. The other cameras' sound strips go with their clips
    and do not come back. Returns the new strip or None."""
    from .mocap import _set_strip_colorspace
    if not proxy_path or not os.path.exists(proxy_path):
        return None
    scene = bpy.context.scene
    if not scene.sequence_editor:
        scene.sequence_editor_create()
    # ends syncing mode, removes our texts; no tidying: the clips are
    # about to be replaced by the one strip, which is placed below
    sync_restore(scene, tidy=False)
    seq_ed = scene.sequence_editor
    seq_all, seq_coll = _seq_accessors(seq_ed)

    cams = record["cams"]
    cam_a = cams[0]
    # Where the one strip goes: DOWN, onto the lowest channel the clips
    # were on (camera A is the top one), so the result sits at the bottom
    # of the group rather than on top of it. Decided below, once the
    # clips are out of the way.
    channel = min(int(cam["channel"]) for cam in cams)
    found = []
    # A PREVIEW strip of this take is what the run started from (Run
    # Capture or Run Preview straight from it): the result takes its
    # place. The clips themselves are not in the timeline then.
    from_preview = False
    kept = ""               # the name of the one sound strip that stays
    for s in list(seq_all):
        old = strip_record(s)
        if old is not None and old.get("preview") \
                and old.get("clip_id") == record.get("clip_id"):
            channel = int(s.channel)
            kept = str(old.get("kept") or kept)
            seq_coll.remove(s)
            from_preview = True
    seq_all, seq_coll = _seq_accessors(seq_ed)
    by_name = {s.name: s for s in seq_all}
    if not from_preview:
        # the sound heard while syncing is the one that stays
        try:
            clips_now = [by_name.get(cam["name"]) for cam in cams]
            heard = (_pick_heard(scene, clips_now)
                     if all(c is not None and getattr(c, "type", "") == 'MOVIE'
                            for c in clips_now) else None)
            if heard is not None:
                kept = heard[1].name
        except Exception as exc:
            print(f"[BlendCap] Which sound to keep could not be worked out: {exc}")
        if not kept and cam_a.get("sound"):
            kept = str(cam_a["sound"]["name"])
    # (from a preview strip there are no clips to take out, and a strip
    # that merely has a clip's name is somebody else's)
    for i, cam in enumerate([] if from_preview else cams):
        s = by_name.get(cam["name"])
        if s is not None and getattr(s, "type", "") == 'MOVIE' \
                and not is_multicam_strip(s):
            found.append(int(s.channel))
            seq_coll.remove(s)
        snd = cam.get("sound")
        if snd and snd["name"] != kept:
            ss = by_name.get(snd["name"])
            if ss is not None and getattr(ss, "type", "") == 'SOUND':
                seq_coll.remove(ss)

    if found:
        # The lowest place that is free now: where the clips really were
        # just now, and where each was before Sync Clips re-stacked it.
        seq_all, seq_coll = _seq_accessors(seq_ed)
        top_level = list(seq_coll)
        places = set(found)
        places.update(int(cam["from"]) for cam in cams if cam.get("from"))
        lo_, hi_ = int(record["lo"]), int(record["hi"])
        channel = next((c for c in sorted(places)
                        if _region_free(top_level, c, lo_, hi_)), min(found))
    # A preview's multiview stands in for the clips exactly as a
    # capture's does (same record, so Revert to Original Clips puts
    # them back, still synced); only its name and type say what it is.
    is_preview = bool(record.get("preview"))
    name = f"{'Preview' if is_preview else 'Capture'}_{record['clip_id']}"
    strip = seq_coll.new_movie(name, proxy_path, channel, int(record["lo"]))
    if strip.channel != channel:
        strip.channel = channel
    _set_strip_colorspace(strip, cam_a.get("colorspace", ""))

    # The one sound that stays comes down with the picture: directly
    # under the strip, when that place is free (the other cameras'
    # sounds are gone by now). Revert leaves it there.
    record.pop("sound_moved", None)
    record["kept"] = kept
    if kept:
        try:
            sa = next((x for x in _seq_accessors(seq_ed)[1]
                       if x.name == kept and x.type == 'SOUND'), None)
            under = int(strip.channel) - 1
            if sa is not None and under >= 1 and int(sa.channel) != under \
                    and _region_free([x for x in _seq_accessors(seq_ed)[1] if x != sa], under,
                                     _strip_visible_start(sa), _strip_visible_end(sa)):
                sa.channel = under
        except Exception as exc:
            print(f"[BlendCap] The kept sound was left where it was: {exc}")

    # Fit to the project frame (letterbox, never over-expand).
    scene_w, scene_h = scene.render.resolution_x, scene.render.resolution_y
    sw, sh = _strip_size(strip)
    if sw > 0 and sh > 0 and scene_w > 0 and scene_h > 0:
        fit = min(scene_w / sw, scene_h / sh)
        strip.transform.scale_x = fit
        strip.transform.scale_y = fit

    n_frames = int(record["hi"]) - int(record["lo"])
    strip["blendcap_source"] = cam_a["path"]
    strip["blendcap_type"] = 'preview' if is_preview else 'capture'
    strip["blendcap_source_strip"] = record["clip_id"]
    if cam_a.get("colorspace"):
        strip["blendcap_original_colorspace"] = cam_a["colorspace"]
    # Camera A's captured window, in the fields a single capture's
    # overlay carries, for the code that reads a clip's trim from them.
    strip["blendcap_original_type"] = 'MOVIE'
    strip["blendcap_original_channel"] = int(cam_a["channel"])
    strip["blendcap_original_offset_start"] = int(cam_a["cap_start"])
    strip["blendcap_original_offset_end"] = 0
    strip["blendcap_original_duration"] = n_frames
    strip["blendcap_original_vis_start"] = int(record["lo"])
    strip["blendcap_original_vis_end"] = int(record["hi"])
    strip["blendcap_original_frame_start"] = int(record["lo"]) - int(cam_a["cap_start"])
    strip["blendcap_original_frame_duration"] = int(cam_a.get("frame_duration", 0))
    strip[STRIP_TAG] = json.dumps(record)

    prefs = addon_prefs()
    if not (prefs and prefs.disable_timeline_autosnap):
        scene.frame_current = int(record["lo"])
    for s in seq_all:
        try:
            s.select = False
        except Exception:
            pass
    strip.select = True
    seq_ed.active_strip = strip
    _tag_redraw('SEQUENCE_EDITOR')
    if "Motion Capture" in bpy.data.workspaces:
        try:
            bpy.context.window.workspace = bpy.data.workspaces["Motion Capture"]
        except Exception:
            pass
    return strip


# ============================================================
# REVERT (the whole group, still synced)
# ============================================================
def _region_free(seq_all, channel, start, end, ignore=()):
    for s in seq_all:
        if s in ignore or int(getattr(s, "channel", -1)) != int(channel):
            continue
        if _strip_visible_start(s) < end and _strip_visible_end(s) > start:
            return False
    return True


def _trim(strip, vis_start, vis_end):
    """Give a freshly made strip its visible range (same calls, same
    order, as the single-camera revert)."""
    def _do():
        if vis_end > vis_start:
            strip.frame_final_end = int(vis_end)
            strip.frame_final_start = int(vis_start)
    _quiet(_do)


def revert_multicam(strip, seq_ed):
    """Open a multiview strip back up into its clips: the mirror of what
    the capture did. They come back as one tight block, in place: the
    last camera on the strip's own channel and the others on the
    channels above it, A on top, with the same slips and trims and the
    same spacing in time (if the strip was moved along the timeline, the
    group moves with it). If any of those places is taken, the whole
    block goes to the nearest run of free channels above instead, kept
    together, so nothing is overwritten and Blender never shuffles the
    clips apart one by one.

    Sound: the one strip the capture kept is not touched (it goes along
    in time when the multiview strip was moved and it is still where the
    capture left it). The other cameras' sound strips went at the
    capture and do NOT come back: the timeline stays as tidy as the
    capture left it, and a sound the user deleted since stays deleted.
    Returns (restored video strips A first, error text)."""
    from .mocap import _set_strip_colorspace
    rec = strip_record(strip)
    if rec is None:
        return [], "This strip's capture record can't be read."
    cams = rec["cams"]
    for cam in cams:
        p = bpy.path.abspath(str(cam["path"]))
        if not p or not os.path.exists(p):
            return [], f"Original video not found: {os.path.basename(p)}"

    seq_all, seq_coll = _seq_accessors(seq_ed)
    delta = _strip_visible_start(strip) - int(rec["lo"])
    by_name = {s.name: s for s in seq_all}
    n = len(cams)
    others = [s for s in seq_coll if s != strip]

    # The kept sound (camera A's in a record made before 2026-11-26).
    # If the multiview strip was moved, it goes along with its picture
    # when it is still where the capture left it and the new place is
    # free.
    kept = str(rec.get("kept") or ((cams[0].get("sound") or {}).get("name") or ""))
    kept_rec = next((cam["sound"] for cam in cams
                     if cam.get("sound") and cam["sound"].get("name") == kept), None)
    if delta and kept_rec:
        sa = by_name.get(kept)
        if (sa is not None and getattr(sa, "type", "") == 'SOUND'
                and _strip_visible_start(sa) == int(kept_rec["vis_start"])
                and _region_free([s for s in others if s != sa], sa.channel,
                                 _strip_visible_start(sa) + delta,
                                 _strip_visible_end(sa) + delta)):
            _strip_set_frame_start(sa, _strip_frame_start(sa) + delta)

    # Where the block goes: from the strip's own channel up, or from the
    # nearest channel above it that has room for all of them.
    spans = [(int(cam["vis_start"]) + delta, int(cam["vis_end"]) + delta) for cam in cams]

    def fits(base):
        return all(_region_free(others, base + (n - 1 - i), a_, b_)
                   for i, (a_, b_) in enumerate(spans))
    base = next((c for c in range(int(strip.channel), MAX_CHANNEL - n + 2) if fits(c)), None)
    if base is None:
        return [], "Not enough free channels to bring the clips back."

    seq_coll.remove(strip)

    restored = []
    for i, cam in enumerate(cams):
        ch = base + (n - 1 - i)                  # A highest
        vis_s, vis_e = spans[i]
        new = seq_coll.new_movie(cam["name"], bpy.path.abspath(cam["path"]),
                                 ch, int(cam["frame_start"]) + delta)
        _trim(new, vis_s, vis_e)
        if new.channel != ch:
            new.channel = ch
        try:
            sx, sy, ox, oy = [float(v) for v in cam["xform"]]
            t = new.transform
            t.scale_x, t.scale_y, t.offset_x, t.offset_y = sx, sy, ox, oy
            new.blend_type = str(cam.get("blend") or new.blend_type)
        except Exception as exc:
            print(f"[BlendCap] View of '{cam['name']}' not restored: {exc}")
        _set_strip_colorspace(new, cam.get("colorspace", ""))
        if cam.get("stamp"):
            new["blendcap_source_strip"] = cam["stamp"]
        if rec.get("preview"):
            # A preview captured nothing: the clip is what it was
            # before (tagged only if it had been captured earlier),
            # or the collision guard would take a clip that merely
            # went through a preview for one of that take's own.
            if cam.get("take"):
                new[TAKE_TAG] = str(cam["take"])
        else:
            new[TAKE_TAG] = str(rec.get("clip_id", ""))
        restored.append(new)

    seq_all, _ = _seq_accessors(seq_ed)
    for s in seq_all:
        try:
            s.select = s in restored
        except Exception:
            pass
    if restored:
        seq_ed.active_strip = restored[0]
    _tag_redraw('SEQUENCE_EDITOR')
    return restored, ""


def cache_subdirs(output_dir):
    """The folders inside a capture's folder: one per camera of a
    multi-camera capture, and the people of a multi-person one (each
    camera's own, in a Multi-Camera & Multi-Person take). A folder
    comes after the folders inside it, so emptying them in this order
    leaves each one free to be removed."""
    out = []
    for l in LETTERS:
        cam = os.path.join(output_dir, f"cam_{l.lower()}")
        out += [os.path.join(cam, "people"), cam]
    return out + [os.path.join(output_dir, "people")]


# ============================================================
# ADD CAMERA: which camera's view
# ============================================================
_VIEW_ITEMS = [
    ('A', "Camera A", "Place a camera where camera A stood"),
    ('B', "Camera B", "Place a camera where camera B stood"),
    ('C', "Camera C", "Place a camera where camera C stood"),
    ('D', "Camera D", "Place a camera where camera D stood"),
]


def camera_view_items(self, context):
    """The cameras the rig Add Camera would act on carries, for the
    Camera View dropdown above it (module-level tuples: Blender keeps
    no reference to strings built inside an items callback)."""
    letters = []
    try:
        from .scene_camera import target_views
        letters = target_views(context)
    except Exception:
        pass
    return [it for it in _VIEW_ITEMS if it[0].lower() in letters] or _VIEW_ITEMS[:1]


def camera_sources(context, output_dir):
    """For a multi-camera capture: each merged camera's video and the
    first frame its capture read, by view letter ({'a': {'video',
    'start'}, ...}), recorded on the rig at Generate for Add Camera.
    None when the user is not pointing at one (active_record)."""
    rec = active_record(context)
    if rec is None:
        return None
    cams = rec["cams"]
    merged = max(1, min(int(rec.get("merged", MERGED_CAMERAS)), len(cams)))
    out = {}
    for cam in cams[:merged]:
        letter = cam["letter"].lower()
        # A capture that converted its footage tracked that copy: it is
        # frame-for-frame what was captured and needs no trim offset.
        exported = os.path.join(output_dir, f"cam_{letter}", "source_export.mp4")
        if os.path.isfile(exported):
            out[letter] = {'video': exported, 'start': 0}
        else:
            out[letter] = {'video': bpy.path.abspath(str(cam["path"])),
                           'start': int(cam["cap_start"])}
    return out
