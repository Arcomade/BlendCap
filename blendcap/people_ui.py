"""Multi-person capture in Blender: the People list, its edits, and the
boxes drawn over the video (people.py holds the table itself).

WHERE THINGS LIVE
  The capture's folder holds the truth: people.json (the automatic
  result) and people_edits.json (the corrections, written the moment
  each one is made). Nothing of it is saved in the .blend, so closing
  without saving loses nothing and the corrections follow the capture
  into any project.

  Blender's undo only knows what is in the .blend, so the corrections
  are ALSO mirrored on the scene (blendcap_people_states, per capture
  folder). Ctrl+Z restores the mirror and _after_undo writes the file
  from it. Opening a project throws the mirror away: the file wins.

  The first correction after a capture is selected needs care. The
  mirror is filled without an undo step (a step there would be a Ctrl+Z
  that visibly does nothing), so undoing that first correction restores
  a scene that has no entry for the capture at all. _LOADED_AS
  remembers what the file held when the entry was filled, and that is
  what goes back to the file.

  The list rows (blendcap_people) are a picture of the mirror, rebuilt
  whenever blendcap_people_shown stops matching it. Draw code may not
  write scene data, so that runs on the next timer tick.

THE HEIGHT COLUMN
  One camera: each person's height is their own, the body model's until
  one is typed. Several cameras: the cameras measured the people against
  each other, so the column shows an estimate for everybody and a height
  typed for one sets the size of the whole take (people.py says why).
  Those measured sizes are worked out by the pipeline, which Blender's
  own Python cannot run: the capture leaves them next to the take, and
  after a correction that changes who is who a process is started for
  them here (take_sizes) and the column fills in when it ends.

A MULTI-CAMERA TAKE
  The same list, one row per person however many cameras saw them. A
  dropdown over the buttons says which cameras a correction is for: all
  of them, or one (two people one camera has the wrong way round
  against the others). Every camera's boxes are drawn in its own tile
  of the multiview, in the person's colour, so a camera that disagrees
  with the rest shows at a glance.

THE BOXES
  Drawn over the Sequencer preview (and over the File Path plane in the
  3D viewport) every time it redraws, straight from the mirror. They are
  not part of the overlay video, which is why a correction shows on
  every frame at once with nothing rendered again.
"""
import copy
import json
import os

import bpy
import bpy.utils.previews      # here, never inside a function: there it makes `bpy` a local name

from . import people
from .helpers import blendcap_state

_SYNCING = False        # rows are being rebuilt: their update callbacks stay quiet
_LOADED_AS = {}         # capture key -> (folder, state as read from disk, json)
_LABELS = {}            # signature -> (labels, highest number): one entry
_pending = False
_gave_up = None
_HANDLES = {}
_PCOLL = None
_ITEMS = []             # keeps a dynamic enum's strings alive (Blender needs that)
_DRAW_FIRST = 1         # the capture's first frame on the timeline, for the rows being drawn
_DRAW_HEIGHT = 'ONE'    # the Height cell of the rows being drawn: 'ONE' camera's, a
                        # 'TAKE' of several, or 'OFF' (a take whose sizes cannot be told)
_SIZES = {}             # _ekey -> a multi-camera take's measured sizes: None (being
                        # worked out), {} (cannot be told) or {number: size}. One entry
_CSIG = {}              # _ekey -> people.cameras_sig. One entry
_SIZES_JOB = {}         # the process working sizes out: 'proc', 'csig', 'folder'
_SIZES_TRIES = {}       # cameras_sig -> how often that was started
_SIZES_FAILED = set()   # cameras_sigs it gave no answer for


# ------------------------------------------------------------
# which capture, and what it holds
# ------------------------------------------------------------
def _key(folder):
    return os.path.normcase(os.path.normpath(folder))


def active_folder(context):
    """The selected capture's folder when it is a multi-person capture,
    else None. The same resolution Generate BVH uses."""
    try:
        from .helpers import _cache_dir, _resolve_clip_name
        clip = _resolve_clip_name(context)
        if not clip:
            return None
        folder = os.path.join(_cache_dir(), f"Capture_{clip}")
        return folder if people.load_auto(folder) is not None else None
    except Exception:
        return None


def _states(scene):
    try:
        data = json.loads(scene.blendcap_people_states or "{}")
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _set_states(scene, states):
    scene.blendcap_people_states = json.dumps(states, sort_keys=True)


def _sig(key, auto, state):
    return key + "|" + json.dumps(state, sort_keys=True)


def view(context):
    """What the selected capture shows right now, or None when it is
    not a multi-person capture. Reads only: safe at draw time."""
    folder = active_folder(context)
    if folder is None:
        return None
    try:
        auto = people.load_auto(folder)
        key = _key(folder)
        state = _states(context.scene).get(key)
        if not isinstance(state, dict) or state.get("stamp") != auto["stamp"]:
            # not mirrored yet (sync does that): show what the file holds
            state = people.load_edits(folder, auto["stamp"])
        sig = _sig(key, auto, state)
        hit = _LABELS.get(sig)
        if hit is None:
            hit = people.replay(auto, state["edits"])
            _LABELS.clear()
            _LABELS[sig] = hit
        return {"folder": folder, "key": key, "auto": auto, "state": state,
                "labels": hit[0], "top": hit[1], "sig": sig}
    except Exception:
        return None


def _fill_rows(scene, v):
    coll = scene.blendcap_people
    idx = int(scene.blendcap_people_index)
    old = coll[idx].number if 0 <= idx < len(coll) else None
    data = people.rows(v["auto"], v["labels"], v["state"]["names"])
    heights = people.shown_heights(v["state"], [r["number"] for r in data],
                                   len(v["auto"]["cameras"]), take_sizes(v))
    per = _metres_per_unit(scene)
    coll.clear()
    for r in data:
        item = coll.add()
        item.number = r["number"]
        item.name = r["name"]
        item.height = item.height_take = item.height_off = heights[r["number"]] / per
        item.first = r["first"]
        item.last = r["last"]
        item.count = r["count"]
    new = next((i for i, r in enumerate(data) if r["number"] == old),
               max(0, min(idx, len(data) - 1)))
    if scene.blendcap_people_index != new:
        scene.blendcap_people_index = new


def sync(context):
    """Bring the scene's mirror and the list rows in line with the
    selected capture. Writes scene data: never from draw code."""
    global _SYNCING
    scene = context.scene
    v = view(context)
    _SYNCING = True
    try:
        if v is None:
            if scene.blendcap_people_shown or len(scene.blendcap_people):
                scene.blendcap_people.clear()
                scene.blendcap_people_shown = ""
            return None
        states = _states(scene)
        if states.get(v["key"]) != v["state"]:
            states[v["key"]] = v["state"]
            _set_states(scene, states)
            _LOADED_AS[(scene.name, v["key"])] = (
                v["folder"], json.dumps(v["state"], sort_keys=True))
        if scene.blendcap_people_shown != v["sig"]:
            _fill_rows(scene, v)
            scene.blendcap_people_shown = v["sig"]
        return v
    finally:
        _SYNCING = False


def _sync_fire():
    global _pending, _gave_up
    _pending = False
    try:
        context = bpy.context
        v = sync(context)
        want = v["sig"] if v is not None else ""
        # a sync that did not take must not be asked for on every redraw
        _gave_up = want if context.scene.blendcap_people_shown != want else None
        _redraw()
    except Exception as exc:
        print(f"[BlendCap] People list not refreshed: {exc}")
    return None


def want_sync(scene, v):
    """Called from draw code: ask for a sync when the rows do not show
    what `v` holds. Registered once however many redraws ask."""
    global _pending
    want = v["sig"] if v is not None else ""
    if scene.blendcap_people_shown == want and (v is not None or not len(scene.blendcap_people)):
        return
    if _pending or _gave_up == want:
        return
    _pending = True
    bpy.app.timers.register(_sync_fire, first_interval=0.0)


def _redraw():
    try:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type in ('VIEW_3D', 'SEQUENCE_EDITOR'):
                    area.tag_redraw()
    except Exception:
        pass


def _store(context, v, state, rows=True):
    """A change made by the user: into the mirror, into the file, and
    onto the list. rows=False when the list already shows it (a name
    typed into a row: the rows must not be rebuilt under that edit)."""
    scene = context.scene
    states = _states(scene)
    states[v["key"]] = state
    _set_states(scene, states)
    saved = people.save_edits(v["folder"], state)
    if rows:
        sync(context)
    else:
        scene.blendcap_people_shown = _sig(v["key"], v["auto"], state)
    _redraw()
    return saved


# ------------------------------------------------------------
# undo, redo, and opening a project
# ------------------------------------------------------------
@bpy.app.handlers.persistent
def _after_undo(scene=None, depsgraph=None):
    """The mirror just changed under Ctrl+Z (or redo): the files follow."""
    try:
        scene = bpy.context.scene or scene
        states = _states(scene)
        people.forget()
        for (owner, key), (folder, loaded) in list(_LOADED_AS.items()):
            if owner != scene.name:
                continue        # another scene's mirror says nothing about this one
            auto = people.load_auto(folder)
            if auto is None:
                continue
            state = states.get(key)
            if isinstance(state, dict):
                state = people.clean_state(state, auto["stamp"])
            else:
                # undone past the point the capture was first shown
                state = people.clean_state(json.loads(loaded), auto["stamp"])
            on_disk = people.load_edits(folder, auto["stamp"])
            if on_disk != state:
                people.save_edits(folder, state)
        _redraw()
    except Exception as exc:
        print(f"[BlendCap] People edits not written after undo: {exc}")


@bpy.app.handlers.persistent
def _after_load(_filepath=None, _depsgraph=None):
    """A project was opened: its saved mirror is dropped, the capture's
    own file is what counts."""
    global _SYNCING, _gave_up
    _LOADED_AS.clear()
    _LABELS.clear()
    _SIZES.clear()
    _CSIG.clear()
    _QUEUE.clear()
    people.forget()
    _gave_up = None
    _SYNCING = True
    try:
        for scene in bpy.data.scenes:
            try:
                scene.blendcap_people_states = "{}"
                scene.blendcap_people_shown = ""
                scene.blendcap_people.clear()
            except Exception:
                pass
    finally:
        _SYNCING = False


# ------------------------------------------------------------
# the list
# ------------------------------------------------------------
def _name_update(self, context):
    global _SYNCING
    if _SYNCING:
        return
    from . import properties
    if properties._UNDO_IN_PROGRESS:
        return
    try:
        v = view(context)
        if v is None:
            return
        typed = (self.name or "").strip()
        default = people.default_name(self.number)
        state = copy.deepcopy(v["state"])
        if not typed or typed == default:
            state["names"].pop(str(self.number), None)
            typed = default
        else:
            state["names"][str(self.number)] = typed
        if self.name != typed:
            _SYNCING = True
            try:
                self.name = typed
            finally:
                _SYNCING = False
        if state != v["state"]:
            _store(context, v, state, rows=False)
    except Exception as exc:
        print(f"[BlendCap] Name not saved: {exc}")


def _metres_per_unit(scene):
    """The Height column is a length, shown in the scene's units; the
    file keeps metres. One Blender unit is this many of them."""
    try:
        us = scene.unit_settings
        return float(us.scale_length) if us.system != 'NONE' and us.scale_length > 0 else 1.0
    except Exception:
        return 1.0


def _ekey(v):
    """What a multi-camera take's sizes go with: the capture and the
    corrections (not the names or heights typed)."""
    return (v["key"] + "|" + str(v["auto"]["stamp"]) + "|"
            + json.dumps(v["state"]["edits"], sort_keys=True))


def take_sizes(v, start=True, watch=True):
    """The measured sizes of the people of a multi-camera take, for who
    is who as `v` has it: {number: size}, {} when they cannot be told,
    None while they are not known. Not known and `start`: a process is
    started to work them out and the rows are filled in when it ends.
    Always None for one camera. Never from draw code (a file is read
    and a process may be started): draw code reads _SIZES."""
    if v is None or len(v["auto"]["cameras"]) <= 1:
        return None
    ek = _ekey(v)
    csig = _CSIG.get(ek)
    if csig is None:
        csig = people.cameras_sig(v["auto"], v["labels"])
        _CSIG.clear()
        _CSIG[ek] = csig
    got = people.load_placed(v["folder"], csig)
    if got is None and csig in _SIZES_FAILED:
        got = {}
    _SIZES.clear()
    _SIZES[ek] = got
    if got is None and start:
        _sizes_start(v["folder"], csig)
        if watch:
            _sizes_watch()
    return got


def _sizes_start(folder, csig):
    """Start the process that places the take's cameras and sizes its
    people (pipeline/people_cameras.py --sizes, a few seconds). One at
    a time, and never beside a capture or an export: the capture is
    rewriting the take, and the export works the same thing out
    itself. _sizes_tick asks again."""
    if _SIZES_JOB.get('proc') is not None:
        return
    from .helpers import bvh_state
    if blendcap_state['running'] or bvh_state['running']:
        return
    if _SIZES_TRIES.get(csig, 0) >= 2:
        _SIZES_FAILED.add(csig)     # it ran and left no answer, twice
        return
    try:
        import subprocess
        from .helpers import (_addon_dir, _python_exe, _subprocess_env,
                              _host_spawn_prefix)
        script = os.path.join(_addon_dir(), "pipeline", "people_cameras.py")
        if not os.path.isfile(script):
            raise OSError("pipeline/people_cameras.py is missing")
        proc = subprocess.Popen(
            _host_spawn_prefix() + [_python_exe(), script, "--sizes", folder],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=_subprocess_env(),
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except Exception as exc:
        print(f"[BlendCap] People sizes not worked out: {exc}")
        _SIZES_FAILED.add(csig)
        return
    if len(_SIZES_TRIES) > 64:
        _SIZES_TRIES.clear()
    _SIZES_TRIES[csig] = _SIZES_TRIES.get(csig, 0) + 1
    _SIZES_JOB.update(proc=proc, csig=csig, folder=folder)


def _sizes_watch():
    try:
        if not bpy.app.timers.is_registered(_sizes_tick):
            bpy.app.timers.register(_sizes_tick, first_interval=0.3)
    except Exception:
        pass


def _sizes_tick():
    """While a multi-camera take's sizes are not known: wait for the
    process, then show what it found (or start it, when it could not
    be started before)."""
    try:
        proc = _SIZES_JOB.get('proc')
        if proc is not None:
            if proc.poll() is None:
                return 0.3
            if proc.returncode != 0:
                _SIZES_FAILED.add(_SIZES_JOB.get('csig'))
            _SIZES_JOB.clear()
        context = bpy.context
        v = view(context)
        if v is None or len(v["auto"]["cameras"]) <= 1:
            return None
        # (no watch: a timer that registers itself while it runs is lost
        # when it returns; what it returns keeps it going)
        got = take_sizes(v, watch=False)
        _show_heights(context.scene, v, got)
        _redraw()
        if got is not None:
            return None
        return 0.3 if _SIZES_JOB.get('proc') is not None else 1.0
    except Exception as exc:
        print(f"[BlendCap] People sizes not shown: {exc}")
        return None


def _show_heights(scene, v, sizes=None):
    """The Height column's numbers onto the rows as they are (a typed
    height changed them, or a take's sizes arrived). Writes scene data:
    never from draw code."""
    global _SYNCING
    if scene.blendcap_people_shown != v["sig"]:
        return          # the rows are not this capture's: the sync fills them
    shown = people.shown_heights(v["state"], [p.number for p in scene.blendcap_people],
                                 len(v["auto"]["cameras"]), sizes)
    per = _metres_per_unit(scene)
    was = _SYNCING
    _SYNCING = True
    try:
        for p in scene.blendcap_people:
            h = shown[p.number] / per
            if abs(p.height - h) > 1e-7:
                p.height = h
            if abs(p.height_take - h) > 1e-7:
                p.height_take = h
            if abs(p.height_off - h) > 1e-7:
                p.height_off = h
    finally:
        _SYNCING = was


def _height_typed(self, context, value, take):
    """A height typed into a row: into the capture's corrections, as a
    name is. take: the capture has several cameras, and the height is
    the one that sets the size of the whole take (people.py). Anything
    under people.HEIGHT_MIN takes a typed height away again."""
    if _SYNCING:
        return
    from . import properties
    if properties._UNDO_IN_PROGRESS:
        return
    try:
        v = view(context)
        if v is None:
            return
        scene = context.scene
        typed = min(float(value) * _metres_per_unit(scene), people.HEIGHT_MAX)
        state = copy.deepcopy(v["state"])
        heights = {} if take else dict(state.get("heights") or {})
        heights.pop(str(self.number), None)
        # (one camera: the body model's own height typed back in is
        # "as captured", which is what Backspace over the cell gives)
        if typed >= people.HEIGHT_MIN and (
                take or abs(typed - people.MODEL_HEIGHT) >= 0.0005):
            heights[str(self.number)] = round(typed, 4)
        state.pop("heights", None)
        if heights:
            state["heights"] = heights
        if state != v["state"]:
            _store(context, v, state, rows=False)
            v = view(context)
        # the cell shows what was kept (held to the most a height can
        # be, or back to the estimate), and in a take everybody follows
        if v is not None:
            _show_heights(scene, v, take_sizes(v) if take else None)
    except Exception as exc:
        print(f"[BlendCap] Height not saved: {exc}")


def _height_update(self, context):
    _height_typed(self, context, self.height, take=False)


def _height_take_update(self, context):
    _height_typed(self, context, self.height_take, take=True)


def _index_update(self, context):
    _redraw()       # the selected person's box is the thick one


class BlendCapPerson(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(
        name="Name",
        description="A name for this person (double-click to change it)",
        update=_name_update)
    # The Height column. Three properties for one number because a
    # tooltip belongs to a property: `height` is drawn for a one-camera
    # capture, `height_take` for a take of several cameras, and
    # `height_off` for one whose cameras could not be placed (the cell
    # is switched off there, and says why). Tooltips: the user's
    # wording (2026-11-24). No soft limits: they count in scene units,
    # and a scene whose unit is a centimetre would be held to 2 of them.
    height: bpy.props.FloatProperty(
        name="Height",
        description="This person's height (double-click to change it)",
        default=people.MODEL_HEIGHT, min=0.0,
        step=1, precision=3,
        subtype='DISTANCE', unit='LENGTH',
        update=_height_update)
    height_take: bpy.props.FloatProperty(
        name="Height",
        description="This person's height (double-click to change it). "
                    "Heights are linked in multi-cam mode, so changing one "
                    "changes them all",
        default=people.MODEL_HEIGHT, min=0.0,
        step=1, precision=3,
        subtype='DISTANCE', unit='LENGTH',
        update=_height_take_update)
    height_off: bpy.props.FloatProperty(
        name="Height",
        description="Height cannot be changed because BlendCap could not "
                    "estimate the camera placement for this capture",
        default=people.MODEL_HEIGHT, min=0.0,
        step=1, precision=3,
        subtype='DISTANCE', unit='LENGTH')
    number: bpy.props.IntProperty()
    first: bpy.props.IntProperty()
    last: bpy.props.IntProperty()
    count: bpy.props.IntProperty()


def first_scene_frame(context):
    """The scene frame the capture's first frame sits on, or None when
    nothing on screen says."""
    try:
        scene = context.scene
        if scene.blendcap_props.video_source == 'FILE':
            plane = _plane(context)
            if plane is not None:
                iu = plane.image_user
                return int(iu.frame_start) - int(iu.frame_offset)
            return 1
        from .helpers import _active_video_strip, _strip_visible_start
        strip = _active_video_strip(scene.sequence_editor)
        if strip is None:
            return None
        if "blendcap_source" in strip:
            return int(round(strip.frame_start))      # an overlay starts where its data does
        return _strip_visible_start(strip)
    except Exception:
        return None


def capture_frame(context):
    """The capture frame under the playhead, or None."""
    first = first_scene_frame(context)
    return None if first is None else int(context.scene.frame_current) - first


def chip(number):
    """Icon id of a person's colour chip (0 = no icon)."""
    try:
        return _PCOLL[f"p{(max(1, int(number)) - 1) % len(people.PALETTE)}"].icon_id
    except Exception:
        return 0


# The list is a table: number | Name | Height | Frames. Its header row
# (draw_panel) and every row (draw_item) are built by _cells: the
# number's cell has a width of its own and the other three take fixed
# shares of what is left, so a cell sits where its header does whatever
# is in it. (Columns with a scale_x, the retarget pair table's way, did
# not line up here: these cells hold labels and icons, whose own width
# moves the column.)
#
# The shares are worked out per redraw from the sidebar's width
# (_table_shares): the height and the frames get the room their text
# needs, the name the rest up to _NAME_WIDE, and what is over is shared
# out between all three. In a narrow sidebar it is the name that is cut
# short first, as it always was.
_NUMBER_UNITS = 2.0             # the colour chip and a number, in widget units
_TABLE_MARGIN = 68.0            # sidebar width that is not the rows' (tabs, panel and list edges)
_NAME_WIDE, _NAME_MIN = 100.0, 50.0
_HEIGHT_WIDE = 72.0             # "1.73 m" in a number field, with the field's own padding
_DRAW_SHARES = (0.4, 0.47)      # the name's share, then the height's of the rest (see _cells)
_HEADERS = ("", "Name", "Height", "Frames")     # the user's (2026-11-24)


def _table_shares(context, first):
    """The shares _cells splits a line by, for the rows as they are and
    the sidebar as wide as it is: (the name's share of the line after
    the number, the height's share of what the name leaves)."""
    name, height, frames = _NAME_WIDE, _HEIGHT_WIDE, 80.0
    try:
        rows = context.scene.blendcap_people
        chars = max((len(f"{p.first + first}-{p.last + first}") for p in rows), default=7)
        frames = 14.0 + 6.6 * chars
        ui = float(context.preferences.system.ui_scale) or 1.0
        room = context.region.width / ui - _TABLE_MARGIN - _NUMBER_UNITS * 20.0
        name = min(name, max(_NAME_MIN, room - height - frames))
    except Exception:
        pass
    return name / (name + height + frames), height / (height + frames)


def _cells(layout):
    """The four cells of one line of the table, left to right."""
    row = layout.row(align=True)
    number = row.row(align=True)
    number.ui_units_x = _NUMBER_UNITS
    rest = row.split(factor=_DRAW_SHARES[0], align=True)
    name = rest.row(align=True)
    rest = rest.split(factor=_DRAW_SHARES[1], align=True)
    height = rest.row(align=True)
    frames = rest.row(align=True)
    return number, name, height, frames


class BLENDCAP_UL_people(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=str(item.number), icon_value=chip(item.number))
            return
        first = _DRAW_FIRST
        number, name, height, frames = _cells(layout)
        number.label(text=str(item.number), icon_value=chip(item.number))
        name.prop(item, "name", text="", emboss=False)
        if _DRAW_HEIGHT == 'ONE':
            height.prop(item, "height", text="", emboss=False)
        elif _DRAW_HEIGHT == 'TAKE':
            height.prop(item, "height_take", text="", emboss=False)
        else:
            # a take whose cameras could not be placed has no sizes to
            # scale: the cell shows the body model's height and is off,
            # with a tooltip of its own saying why
            height.enabled = False
            height.prop(item, "height_off", text="", emboss=False)
        frames.label(text=f"{item.first + first}-{item.last + first}")


def _selected(scene):
    coll = scene.blendcap_people
    idx = int(scene.blendcap_people_index)
    return coll[idx].number if 0 <= idx < len(coll) else None


def _select(scene, number):
    for i, item in enumerate(scene.blendcap_people):
        if item.number == number:
            scene.blendcap_people_index = i
            return


def selected_person(context, folder=None):
    """For Generate BVH: who the BVH is for when the selected capture is
    a multi-person capture ({'number', 'name', 'has_face', 'folder'}),
    else None. `folder`, when given, must be that capture's folder. Not
    for draw code: it brings the list up to date first."""
    v = sync(context)
    if v is None:
        return None
    if folder is not None and _key(folder) != v["key"]:
        return None
    scene = context.scene
    number = _selected(scene)
    if number is None:
        return None
    by_id = {t["id"]: t for t in v["auto"]["tracks"]}
    has_face = any(os.path.isfile(people.track_file(v["folder"], v["auto"],
                                                    by_id[t], face=True))
                   for t, row in v["labels"].items() if number in row)
    idx = int(scene.blendcap_people_index)
    return {"number": int(number), "name": scene.blendcap_people[idx].name,
            "has_face": has_face, "folder": v["folder"]}


def person_clip(clip_name, number):
    """The clip name a person of a multi-person capture exports under.
    Their own copy of the capture lives in the cache as Capture_<this>,
    an ordinary capture folder, which is what lets Generate BVH, Bake
    and the BVH library treat a person like any other clip."""
    return f"{clip_name}_P{int(number)}"


def _other_items(self, context):
    global _ITEMS
    scene = context.scene
    sel = _selected(scene)
    items = []
    for i, item in enumerate(scene.blendcap_people):
        if item.number == sel:
            continue
        default = people.default_name(item.number)
        label = default if item.name == default else f"{item.number}: {item.name}"
        items.append((str(item.number), label, "", chip(item.number), i))
    _ITEMS = items or [('0', "Nobody else", "", 0, 0)]
    return _ITEMS


# A multi-camera take: a correction is for every camera, or for ONE
# (two people the wrong way round in one camera against the others; the
# user's requirement, 2026-11-19). The choice is a dropdown over the
# buttons, shown only for such a take.
_SCOPE_ITEMS = []
_SCOPE_ALL = ('ALL', "All Cameras", "The correction applies to every camera")


def scope_items(self, context):
    global _SCOPE_ITEMS
    items = [_SCOPE_ALL]
    try:
        v = view(context)
        cams = v["auto"]["cameras"] if v is not None else []
        if len(cams) > 1:
            items += [(str(i), f"Camera {c}",
                       f"The correction applies to camera {c} only", i + 1)
                      for i, c in enumerate(cams)]
            items[0] = _SCOPE_ALL + (0,)
    except Exception:
        pass
    _SCOPE_ITEMS = items
    return _SCOPE_ITEMS


def edit_camera(context, v):
    """The one camera corrections are being made for (its index), or
    None: every camera, and always for a single-camera capture."""
    try:
        if v is None or len(v["auto"]["cameras"]) <= 1:
            return None
        value = str(context.scene.blendcap_people_camera or 'ALL')
        if value == 'ALL':
            return None
        cam = int(value)
        return cam if 0 <= cam < len(v["auto"]["cameras"]) else None
    except Exception:
        return None


def _edit(op, context, make):
    """Run one correction. `make(v, selected)` returns the edit, or a
    sentence saying why there is none."""
    if blendcap_state['running']:
        op.report({'WARNING'}, "A capture is running.")
        return {'CANCELLED'}
    v = sync(context)
    scene = context.scene
    sel = _selected(scene)
    if v is None or sel is None:
        op.report({'ERROR'}, "Select a multi-person capture first.")
        return {'CANCELLED'}
    edit = make(v, sel)
    if isinstance(edit, str):
        op.report({'ERROR'}, edit)
        return {'CANCELLED'}
    cam = edit_camera(context, v)
    if cam is not None:
        edit["cam"] = cam
    err = people.check(v["auto"], v["state"]["edits"], edit)
    if err:
        op.report({'ERROR'}, err)
        return {'CANCELLED'}
    state = copy.deepcopy(v["state"])
    state["edits"].append(edit)
    keep = sel
    # (in one camera only, the person may still be there in the others:
    # a name moves on, or goes, with the LAST of them)
    left = {x for row in people.replay(v["auto"], state["edits"])[0].values()
            for x in set(row)}
    if edit["op"] == "join":
        gone = max(edit["a"], edit["b"])
        if gone not in left:
            state["names"] = people.join_names(state["names"], edit["a"], edit["b"])
            if state.get("heights"):
                state["heights"] = people.join_names(state["heights"], edit["a"], edit["b"])
        if sel not in left:
            keep = min(edit["a"], edit["b"])
    elif edit["op"] == "remove":
        if edit["a"] not in left:
            state["names"].pop(str(edit["a"]), None)
            if state.get("heights"):
                state["heights"].pop(str(edit["a"]), None)
                if not state["heights"]:
                    del state["heights"]
    if not _store(context, v, state):
        op.report({'WARNING'}, "The change could not be saved to the capture's folder.")
    _select(scene, keep)
    return {'FINISHED'}


def _playhead(context, v):
    f = capture_frame(context)
    if f is None or not 0 <= f < v["auto"]["frames"]:
        return None
    return f


NO_PLAYHEAD = "Move the playhead onto the capture first."


# The user's wording (2026-11-19). It does not say whose motion is kept on
# the frames they share (the selected person's): his call, the result is
# the selected person either way.
JOIN_ASK = ("Join Detections",
            "These people are both on screen at the same time. "
            "Would you like to join them anyway?")


def join_question(context, other):
    """What Join asks before it joins the selected person with `other`
    when the two are in the picture together: (title, message). None
    when there is nothing to ask (they never are, or the join would be
    refused anyway, which execute() reports)."""
    v = sync(context)
    sel = _selected(context.scene)
    other = int(other or 0)
    if v is None or sel is None or not other or other == sel:
        return None
    edits = v["state"]["edits"]
    cam = edit_camera(context, v)
    edit = {"op": "join", "a": sel, "b": other}
    if cam is not None:
        edit["cam"] = cam
    if people.check(v["auto"], edits, edit):
        return None
    if not people.overlap(v["auto"], edits, sel, other, cam):
        return None
    return JOIN_ASK


class BLENDCAP_OT_people_join(bpy.types.Operator):
    """Make the selected person and another one the same person. The lower number is kept"""
    bl_idname = "blendcap.people_join"
    bl_label = "Join"
    bl_options = {'REGISTER', 'UNDO'}

    other: bpy.props.EnumProperty(name="With", items=_other_items)

    @classmethod
    def poll(cls, context):
        return len(context.scene.blendcap_people) >= 2

    def invoke(self, context, event):
        # Two people who are in the picture together are usually two
        # people. They can also be one body the capture found twice
        # (a messy shot), and then joining them is the fix: ask, so a
        # slip of the mouse does not fold two real people into one
        # (the user, 2026-11-19; this was a refusal before).
        ask = join_question(context, self.other)
        if ask is None:
            return self.execute(context)
        wm = context.window_manager
        try:
            return wm.invoke_confirm(self, event, title=ask[0], message=ask[1],
                                     confirm_text="Join", icon='QUESTION')
        except TypeError:
            return wm.invoke_confirm(self, event)

    def execute(self, context):
        other = int(self.other or 0)
        return _edit(self, context,
                     lambda v, sel: {"op": "join", "a": sel, "b": other})


class BLENDCAP_OT_people_swap(bpy.types.Operator):
    """Exchange the selected person and another one from the current frame to the end"""
    bl_idname = "blendcap.people_swap"
    bl_label = "Swap"
    bl_options = {'REGISTER', 'UNDO'}

    other: bpy.props.EnumProperty(name="With", items=_other_items)

    @classmethod
    def poll(cls, context):
        return len(context.scene.blendcap_people) >= 2

    def execute(self, context):
        other = int(self.other or 0)

        def make(v, sel):
            f = _playhead(context, v)
            if f is None:
                return NO_PLAYHEAD
            return {"op": "swap", "a": sel, "b": other, "frame": f}
        return _edit(self, context, make)


class BLENDCAP_OT_people_split(bpy.types.Operator):
    """Make the selected person a new person from the current frame to the end"""
    bl_idname = "blendcap.people_split"
    bl_label = "Split"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return len(context.scene.blendcap_people) >= 1

    def execute(self, context):
        def make(v, sel):
            f = _playhead(context, v)
            if f is None:
                return NO_PLAYHEAD
            return {"op": "split", "a": sel, "frame": f}
        return _edit(self, context, make)


class BLENDCAP_OT_people_remove(bpy.types.Operator):
    """Take the selected person out of this capture (a false detection, or someone not wanted)"""
    bl_idname = "blendcap.people_remove"
    bl_label = "Remove"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return len(context.scene.blendcap_people) >= 1

    def execute(self, context):
        return _edit(self, context, lambda v, sel: {"op": "remove", "a": sel})


class BLENDCAP_OT_people_reset(bpy.types.Operator):
    """Drop every change made here and go back to what the capture found"""
    bl_idname = "blendcap.people_reset"
    bl_label = "Reset"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if blendcap_state['running']:
            self.report({'WARNING'}, "A capture is running.")
            return {'CANCELLED'}
        v = sync(context)
        if v is None:
            self.report({'ERROR'}, "Select a multi-person capture first.")
            return {'CANCELLED'}
        ids = {str(t["id"]) for t in v["auto"]["tracks"]}
        state = people.empty_state(v["auto"]["stamp"])
        state["names"] = {k: n for k, n in v["state"]["names"].items() if k in ids}
        heights = {k: h for k, h in (v["state"].get("heights") or {}).items() if k in ids}
        if heights:
            state["heights"] = heights
        if not _store(context, v, state):
            self.report({'WARNING'}, "The change could not be saved to the capture's folder.")
        return {'FINISHED'}


# ------------------------------------------------------------
# Generate/Reload All
# ------------------------------------------------------------
# One Generate/Reload BVH per person of the list, one after the other.
# Each is the ordinary export of that person, started exactly as the
# button starts it (the person is selected in the list first), so a rig
# made this way is no different from one made by hand. They cannot be
# one run: every person exports under a clip name of their own, and the
# rig's tag and Keep Captured Position are per run. No camera is made:
# Add Camera is a button of its own, for whichever rig is selected.
#
# The BVH monitor (mocap.monitor_bvh_timer) reports each run's end to
# queue_advance. The next person is started from a timer of its own,
# never from inside the monitor: the export registers that same monitor
# again, and a timer that registers itself while it is running is lost
# when it returns.
_QUEUE = {}


def queue_active():
    return bool(_QUEUE)


def queue_status(pct, status):
    """The progress of the whole queue, and its label, from the running
    export's own: (percent, text)."""
    total = max(1, int(_QUEUE.get('total', 1)))
    done = int(_QUEUE.get('done', 0)) + len(_QUEUE.get('skipped', []))
    overall = (min(done, total - 1) + max(0.0, min(float(pct), 100.0)) / 100.0) / total * 100.0
    text = f"Person {min(done + 1, total)} of {total}"
    return overall, (f"{text}: {status}" if status else text)


def queue_cancel():
    """Stop after the export that is running (Cancel stops that one)."""
    if _QUEUE:
        _queue_end(stopped=True)


def _queue_end(stopped=False):
    q = dict(_QUEUE)
    _QUEUE.clear()
    try:
        scene = bpy.context.scene
        if q.get('restore') is not None:
            _select(scene, q['restore'])
    except Exception:
        pass
    made, skipped = q.get('done', 0), q.get('skipped', [])
    print(f"[BlendCap] Generate/Reload All: {made} of {q.get('total', 0)} people exported"
          + (", stopped before the rest" if stopped else "")
          + (f"; not exported: {', '.join(str(n) for n in skipped)}" if skipped else ""))
    if skipped and not stopped:
        try:
            from .helpers import _show_warning_popup
            _show_warning_popup([
                f"{made} of {q.get('total', 0)} people were exported.",
                "Nothing to export for: " + ", ".join(q['names'].get(n, str(n)) for n in skipped),
            ])
        except Exception:
            pass
    _redraw()


def _queue_step():
    """Start the next person's export. Anybody the export has nothing
    for is skipped; the queue ends when nobody is left, or when the
    capture it was started on is no longer the selected one. True when
    an export is now running."""
    from .helpers import bvh_state
    context = bpy.context
    while _QUEUE.get('numbers'):
        v = sync(context)
        if v is None or v["key"] != _QUEUE['key']:
            print("[BlendCap] Generate/Reload All stopped: another capture is selected now.")
            _queue_end(stopped=True)
            return False
        number = _QUEUE['numbers'].pop(0)
        scene = context.scene
        if not any(p.number == number for p in scene.blendcap_people):
            continue                    # removed from the list meanwhile
        _select(scene, number)
        try:
            res = bpy.ops.blendcap.generate_bvh('EXEC_DEFAULT')
        except RuntimeError as exc:     # the export refused, and said why
            print(f"[BlendCap] Person {number}: {str(exc).strip()}")
            res = {'CANCELLED'}
        # FINISHED means the export's thread was started (every refusal
        # is a CANCELLED); its running flag is not asked, a run that
        # fails at once may already have dropped it
        if 'FINISHED' in res:
            return True
        _QUEUE['skipped'].append(number)
    _queue_end()
    return False


def _queue_fire():
    if not _QUEUE.pop('pending', False):
        return None
    try:
        _queue_step()
    except Exception as exc:
        print(f"[BlendCap] Generate/Reload All stopped: {exc}")
        _queue_end(stopped=True)
    return None


def queue_advance(return_code):
    """A BVH export has just ended (called by its monitor)."""
    if not _QUEUE:
        return
    if return_code != 0:
        # failed or cancelled: the rest is not attempted (the monitor
        # has already said what happened to this one)
        _queue_end(stopped=True)
        return
    _QUEUE['done'] = _QUEUE.get('done', 0) + 1
    _QUEUE['pending'] = True
    bpy.app.timers.register(_queue_fire, first_interval=0.05)
    _redraw()


class BLENDCAP_OT_generate_bvh_all(bpy.types.Operator):
    bl_idname = "blendcap.generate_bvh_all"
    bl_label = "Generate/Reload All"
    bl_description = ("Generate a BVH for every person in the list, and "
                      "import each into the scene")

    def execute(self, context):
        from .helpers import bvh_state, bvh_importer_state
        if blendcap_state['running']:
            self.report({'WARNING'}, "A capture is running. Wait for it to finish before generating BVH.")
            return {'CANCELLED'}
        if bvh_state['running'] or _QUEUE:
            self.report({'WARNING'}, "A BVH export is already running.")
            return {'CANCELLED'}
        v = sync(context)
        scene = context.scene
        people_now = [(p.number, p.name) for p in scene.blendcap_people]
        if v is None or not people_now:
            self.report({'ERROR'}, "Select a multi-person capture first.")
            return {'CANCELLED'}
        props = scene.blendcap_props
        if not (props.export_body or props.export_hands or props.export_root
                or props.export_face or props.use_separate_face):
            self.report({'ERROR'},
                        "Nothing selected to export. Enable at least one of: "
                        "Body, Hands, Root, Face, or Use Separate Face Data.")
            return {'CANCELLED'}
        if bvh_importer_state() == 'DISABLED':
            # one dialog, not one per person
            bpy.ops.blendcap.bvh_importer_missing('INVOKE_DEFAULT')
            return {'CANCELLED'}
        _QUEUE.update({'key': v["key"], 'numbers': [n for n, _name in people_now],
                       'names': dict(people_now), 'total': len(people_now),
                       'done': 0, 'skipped': [], 'restore': _selected(scene)})
        if not _queue_step():
            self.report({'ERROR'}, "There was nothing to export for anybody in this capture.")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Generating BVHs for {len(people_now)} people... "
                              f"Each will auto-import when it finishes.")
        return {'FINISHED'}


def draw_panel(layout, context):
    """The People block of BVH Settings. Draws nothing, and returns
    None, unless the selected capture is a multi-person capture."""
    global _DRAW_FIRST, _DRAW_HEIGHT, _DRAW_SHARES
    scene = context.scene
    v = view(context)
    want_sync(scene, v)
    if v is None:
        return None
    if len(v["auto"]["cameras"]) <= 1:
        _DRAW_HEIGHT = 'ONE'
    else:
        # {} = worked out, and the cameras could not be placed
        _DRAW_HEIGHT = 'OFF' if _SIZES.get(_ekey(v)) == {} else 'TAKE'
    # frames are shown as the timeline counts them; with nothing on
    # screen to say where the capture sits, from 1
    first = first_scene_frame(context)
    _DRAW_FIRST = first = 1 if first is None else first
    col = layout.column(align=True)
    col.label(text="People", icon='COMMUNITY')
    _DRAW_SHARES = _table_shares(context, first)
    number, name, height, frames = _cells(col)
    # (an empty cell has no width, a text field starts its text a
    # little in from its edge, and a number field puts its number in
    # the middle)
    number.label(text="")
    name.separator(factor=0.9)
    name.label(text=_HEADERS[1])
    height.alignment = 'CENTER'
    height.label(text=_HEADERS[2])
    frames.label(text=_HEADERS[3])
    col.template_list("BLENDCAP_UL_people", "",
                      scene, "blendcap_people",
                      scene, "blendcap_people_index", rows=4)
    if len(v["auto"]["cameras"]) > 1:
        col.prop(scene, "blendcap_people_camera", text="")
    row = col.row(align=True)
    row.operator_menu_enum("blendcap.people_swap", "other", text="Swap")
    row.operator_menu_enum("blendcap.people_join", "other", text="Join")
    row.operator("blendcap.people_split", text="Split")
    row.operator("blendcap.people_remove", text="Remove")
    reset = col.row(align=True)
    reset.enabled = bool(v["state"]["edits"])
    reset.operator("blendcap.people_reset", icon='LOOP_BACK', text="Reset to Automatic")
    # (where the capture was not sure who is who, the person's box says
    # so with a "?" on those frames, _draw_boxes. The list had a warning
    # for it too, a triangle in the Frames column and a "Check frame N"
    # line here; taken out 2026-11-25, the user: a warning that cannot
    # explain itself)
    layout.separator()
    return v


# ------------------------------------------------------------
# the boxes over the video
# ------------------------------------------------------------
_TILES = {}             # layout path -> (mtime, [(x, y, w, h) as fractions])


def _tiles(folder, cameras):
    """Where each camera's picture sits inside the capture's video, as
    fractions from the top left. One camera fills it; a multi-camera
    take is tiled the way its multiview video was made."""
    if cameras <= 1:
        return [(0.0, 0.0, 1.0, 1.0)]
    path = os.path.join(folder, "multiview_layout.json")
    try:
        mtime = os.stat(path).st_mtime
    except OSError:
        return []
    hit = _TILES.get(path)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            lay = json.load(f)
        w, h = float(lay["width"]), float(lay["height"])
        out = [(t["rect"][0] / w, t["rect"][1] / h, t["rect"][2] / w, t["rect"][3] / h)
               for t in lay["tiles"]]
    except Exception:
        out = []
    _TILES.clear()
    _TILES[path] = (mtime, out)
    return out


def _plane(context, v=None):
    """The viewport plane showing the selected File Path capture."""
    try:
        from .helpers import _sanitize_clip_name, _strip_media_extension
        folder = v["folder"] if v is not None else active_folder(context)
        if folder is None:
            return None
        want = os.path.basename(folder)
        for obj in context.scene.objects:
            if obj.type != 'EMPTY' or "blendcap_source" not in obj:
                continue
            if obj.empty_display_type != 'IMAGE' or obj.data is None:
                continue
            stamp = str(obj.get("blendcap_source_strip", "") or "") \
                or _strip_media_extension(obj.name)
            if f"Capture_{_sanitize_clip_name(stamp)}" == want:
                return obj
    except Exception:
        pass
    return None


def _quad(p, q, half):
    """Two triangles covering the segment p-q, `half` wide each side."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-6:
        return []
    nx, ny = -dy / length * half, dx / length * half
    a = (p[0] + nx, p[1] + ny)
    b = (p[0] - nx, p[1] - ny)
    c = (q[0] - nx, q[1] - ny)
    d = (q[0] + nx, q[1] + ny)
    return [a, b, c, a, c, d]


def outline(corners, thick, dashed, dash=10.0, gap=7.0):
    """Triangles for a closed outline through `corners`, `thick` pixels
    wide; in dashes when `dashed`."""
    half = thick / 2.0
    tris = []
    n = len(corners)
    for i in range(n):
        p, q = corners[i], corners[(i + 1) % n]
        dx, dy = q[0] - p[0], q[1] - p[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-6:
            continue
        ux, uy = dx / length, dy / length
        if not dashed:
            # run past both ends so the corners close
            tris += _quad((p[0] - ux * half, p[1] - uy * half),
                          (q[0] + ux * half, q[1] + uy * half), half)
            continue
        at = 0.0
        while at < length:
            end = min(length, at + dash)
            tris += _quad((p[0] + ux * at, p[1] + uy * at),
                          (p[0] + ux * end, p[1] + uy * end), half)
            at = end + gap
    return tris


def _fill(tris, rgba):
    if not tris:
        return
    import gpu
    from gpu_extras.batch import batch_for_shader
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'TRIS', {"pos": tris})
    shader.bind()
    shader.uniform_float("color", rgba)
    batch.draw(shader)


def _draw_boxes(context, v, frame, mappers):
    """Everyone on `frame`, a box and a number each. mappers: per
    camera, a function taking a point of that camera's picture (0..1
    from the top left) to region pixels, or None."""
    import blf
    import gpu
    now = people.at_frame(v["auto"], v["labels"], frame)
    if not now:
        return
    try:
        ui = float(context.preferences.system.ui_scale)
    except Exception:
        ui = 1.0
    sel = _selected(context.scene)
    region = context.region
    font = 0
    gpu.state.blend_set('ALPHA')
    try:
        # the selected person last, so their box is on top
        for person in sorted(now, key=lambda p: p["number"] == sel):
            rgb = people.colour(person["number"])
            chosen = person["number"] == sel
            for cam, box in enumerate(person["boxes"]):
                to = mappers[cam] if cam < len(mappers) else None
                if box is None or to is None:
                    continue
                x0, y0, x1, y1 = box
                corners = [to(x0, y0), to(x1, y0), to(x1, y1), to(x0, y1)]
                if any(c is None for c in corners):
                    continue
                top, bottom = to(0.0, 0.0), to(0.0, 1.0)
                if top is None or bottom is None:
                    continue
                pic_h = ((top[0] - bottom[0]) ** 2 + (top[1] - bottom[1]) ** 2) ** 0.5
                thick = (4.5 if chosen else 2.5) * ui
                dashed = person["filled"]
                _fill(outline(corners, thick + 2.0 * ui, dashed, 10.0 * ui, 7.0 * ui),
                      (0.0, 0.0, 0.0, 0.55))
                _fill(outline(corners, thick, dashed, 10.0 * ui, 7.0 * ui),
                      (rgb[0], rgb[1], rgb[2], 1.0))

                # the number, on a tag of the person's colour sitting on
                # the box's top left corner
                text = str(person["number"]) + (" ?" if person["unsure"] else "")
                size = max(15.0 * ui, min(40.0 * ui, pic_h * 0.055))
                blf.size(font, size)
                tw, _th = blf.dimensions(font, text)
                th = blf.dimensions(font, "0")[1]
                pad = size * 0.28
                w, h = tw + 2 * pad, th + 2 * pad
                tl = max(corners, key=lambda c: c[1] - c[0])     # upper left on screen
                x = tl[0] - thick / 2.0
                y = tl[1] + thick / 2.0
                if y + h > region.height:
                    y = tl[1] - h          # no room above: inside the box
                x = max(0.0, min(x, region.width - w))
                tag = [(x, y), (x + w, y), (x + w, y + h), (x, y), (x + w, y + h), (x, y + h)]
                edge = 1.5 * ui
                _fill([(x - edge, y - edge), (x + w + edge, y - edge),
                       (x + w + edge, y + h + edge), (x - edge, y - edge),
                       (x + w + edge, y + h + edge), (x - edge, y + h + edge)],
                      (0.0, 0.0, 0.0, 0.55))
                _fill(tag, (rgb[0], rgb[1], rgb[2], 1.0))
                dark = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] > 0.5
                blf.color(font, *((0.0, 0.0, 0.0, 1.0) if dark else (1.0, 1.0, 1.0, 1.0)))
                blf.position(font, x + pad, y + pad, 0)
                blf.draw(font, text)
                gpu.state.blend_set('ALPHA')
    finally:
        gpu.state.blend_set('NONE')


def strip_mappers(scene, strip, v2d, tiles):
    """One mapper per camera for a capture shown as a Sequencer strip."""
    from .multicam import _strip_size
    import math
    w, h = _strip_size(strip)
    if w <= 0 or h <= 0:
        return []
    tr = strip.transform
    sw, sh = w * float(tr.scale_x), h * float(tr.scale_y)
    cx, cy = float(tr.offset_x), float(tr.offset_y)
    rot = float(getattr(tr, "rotation", 0.0))
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    flip_x = bool(getattr(strip, "use_flip_x", False))
    flip_y = bool(getattr(strip, "use_flip_y", False))

    def make(tile):
        tx, ty, tw, th = tile

        def to(nx, ny):
            u, t = tx + nx * tw, ty + ny * th
            if flip_x:
                u = 1.0 - u
            if flip_y:
                t = 1.0 - t
            x, y = (u - 0.5) * sw, (0.5 - t) * sh
            x, y = x * cos_r - y * sin_r, x * sin_r + y * cos_r
            px, py = v2d.view_to_region(x + cx, y + cy, clip=False)
            return (float(px), float(py))
        return to
    return [make(t) for t in tiles]


def plane_mappers(context, plane, tiles):
    """One mapper per camera for a capture shown on the viewport plane."""
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    from mathutils import Vector
    img = plane.data
    iw, ih = float(img.size[0]), float(img.size[1])
    if iw <= 0 or ih <= 0:
        return []
    size = float(plane.empty_display_size)
    w, h = (size, size * ih / iw) if iw >= ih else (size * iw / ih, size)
    ox, oy = plane.empty_image_offset
    mw = plane.matrix_world.copy()
    region, rv3d = context.region, context.region_data

    def make(tile):
        tx, ty, tw, th = tile

        def to(nx, ny):
            u, t = tx + nx * tw, ty + ny * th
            p = location_3d_to_region_2d(
                region, rv3d, mw @ Vector(((ox + u) * w, (oy + 1.0 - t) * h, 0.0)))
            return None if p is None else (float(p[0]), float(p[1]))
        return to
    return [make(t) for t in tiles]


# The boxes belong to the SELECTED capture, and the preview shows
# whatever is on top at the playhead. When that is not the selected
# capture's own video (another strip lies over it, or it is muted) the
# boxes would be drawn over somebody else's picture, so they are not
# drawn at all (the user, 2026-11-24).
#
# Strips that put a picture of their own over what is below them. Text,
# adjustment layers and effects are left out: what is below shows
# through them, or is what they are made of.
_PICTURE_TYPES = frozenset(('MOVIE', 'IMAGE', 'SCENE', 'META', 'MOVIECLIP', 'COLOR', 'MASK'))
# ...and only when laid over it whole: the other blend modes mix with it.
_OPAQUE_BLENDS = frozenset(('REPLACE', 'ALPHA_OVER', 'CROSS', 'GAMMA_CROSS'))


def _clip_of(strip):
    """The capture a strip belongs to, named as _resolve_clip_name does."""
    from .helpers import _sanitize_clip_name, _strip_media_extension
    stored = str(strip.get("blendcap_source_strip", "") or "")
    return _sanitize_clip_name(stored or _strip_media_extension(strip.name))


def _reaches(scene, strip, x, y):
    """Whether a strip's picture lies over the point (x, y) of the
    preview (project pixels from its centre, y up). A strip whose size
    cannot be told counts as lying over it."""
    import math
    from .multicam import _strip_size
    try:
        w, h = _strip_size(strip)
        if w <= 0 or h <= 0:
            if strip.type in ('MOVIE', 'IMAGE'):
                return True
            # a colour, a scene, a meta: the project's own picture
            w, h = int(scene.render.resolution_x), int(scene.render.resolution_y)
        tr = strip.transform
        sx, sy = float(tr.scale_x), float(tr.scale_y)
        if abs(sx) < 1e-9 or abs(sy) < 1e-9:
            return False
        dx, dy = x - float(tr.offset_x), y - float(tr.offset_y)
        rot = float(getattr(tr, "rotation", 0.0))
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        # the point in the strip's own picture, pixels from its centre
        lx = (dx * cos_r + dy * sin_r) / sx
        ly = (-dx * sin_r + dy * cos_r) / sy
        crop = getattr(strip, "crop", None)
        left = -w / 2.0 + float(getattr(crop, "min_x", 0))
        right = w / 2.0 - float(getattr(crop, "max_x", 0))
        bottom = -h / 2.0 + float(getattr(crop, "min_y", 0))
        top = h / 2.0 - float(getattr(crop, "max_y", 0))
        return left <= lx <= right and bottom <= ly <= top
    except Exception:
        return True


def strip_shows(scene, strip, frame):
    """Whether the preview shows `strip`'s own picture on scene frame
    `frame`: it is not muted (nor its channel), and no other picture
    lies over the middle of it from a channel above. A strip of the
    same capture above it (a source clip under its own overlay) is the
    same picture and does not count; neither does anything see-through
    (under full opacity, a mixing blend mode, a meta of sounds only) or
    anything small or off to a side (a logo, a letterbox bar)."""
    from .helpers import _seq_accessors, _strip_visible_start, _strip_visible_end
    try:
        seq_ed = scene.sequence_editor
        parent = strip.parent_meta()
        if parent is not None:
            strips, channels = parent.strips, parent.channels
        else:
            strips, channels = _seq_accessors(seq_ed)[1], seq_ed.channels

        def off(s):
            if s.mute:
                return True
            try:
                return bool(channels[int(s.channel)].mute)
            except Exception:
                return False

        if off(strip):
            return False
        mine = _clip_of(strip)
        tr = strip.transform
        x, y = float(tr.offset_x), float(tr.offset_y)
        for s in strips:
            if s.channel <= strip.channel or s.type not in _PICTURE_TYPES:
                continue
            if off(s) or float(s.blend_alpha) < 0.999 or s.blend_type not in _OPAQUE_BLENDS:
                continue
            if not _strip_visible_start(s) <= frame < _strip_visible_end(s):
                continue
            if s.type == 'META' and all(c.type == 'SOUND' for c in s.strips):
                continue
            if _clip_of(s) == mine:
                continue
            if _reaches(scene, s, x, y):
                return False
        return True
    except Exception:
        return True


def _draw_preview():
    try:
        context = bpy.context
        scene = context.scene
        if scene.blendcap_props.video_source != 'VSE':
            return
        v = view(context)
        if v is None:
            return
        want_sync(scene, v)
        from . import multicam
        if multicam.sync_active(scene):
            return
        from .helpers import (_active_video_strip, _strip_visible_start,
                              _strip_visible_end)
        strip = _active_video_strip(scene.sequence_editor)
        if strip is None or strip.type not in ('MOVIE', 'IMAGE'):
            return
        now = int(scene.frame_current)
        if not _strip_visible_start(strip) <= now < _strip_visible_end(strip):
            return
        if not strip_shows(scene, strip, now):
            return
        frame = capture_frame(context)
        if frame is None:
            return
        tiles = _tiles(v["folder"], len(v["auto"]["cameras"]))
        _draw_boxes(context, v, frame,
                    strip_mappers(scene, strip, context.region.view2d, tiles))
    except Exception:
        pass


def _draw_viewport():
    try:
        context = bpy.context
        scene = context.scene
        if scene.blendcap_props.video_source != 'FILE':
            return
        v = view(context)
        if v is None or context.region_data is None:
            return
        want_sync(scene, v)
        plane = _plane(context, v)
        if plane is None or not plane.visible_get():
            return
        iu = plane.image_user
        frame = int(scene.frame_current) - int(iu.frame_start) + int(iu.frame_offset)
        n = v["auto"]["frames"]
        if iu.use_cyclic and n > 0:
            frame %= n
        tiles = _tiles(v["folder"], len(v["auto"]["cameras"]))
        _draw_boxes(context, v, frame, plane_mappers(context, plane, tiles))
    except Exception:
        pass


def _make_chips():
    """A small rounded square of each person's colour, as icons."""
    global _PCOLL
    try:
        _PCOLL = bpy.utils.previews.new()
        s = 32
        r = 7.0
        for i, (cr, cg, cb) in enumerate(people.PALETTE):
            p = _PCOLL.new(f"p{i}")
            px = []
            for y in range(s):
                for x in range(s):
                    # inside the square, 3 pixels in, with rounded corners
                    dx = max(3 + r - x, x - (s - 4 - r), 0.0)
                    dy = max(3 + r - y, y - (s - 4 - r), 0.0)
                    inside = (3 <= x <= s - 4 and 3 <= y <= s - 4
                              and dx * dx + dy * dy <= r * r)
                    px += [cr, cg, cb, 1.0 if inside else 0.0]
            p.icon_size = (s, s)
            p.icon_pixels_float = px
    except Exception as exc:
        print(f"[BlendCap] People colour chips not made: {exc}")
        _PCOLL = None


def arm():
    h = bpy.app.handlers
    for group in (h.undo_post, h.redo_post):
        if _after_undo not in group:
            group.append(_after_undo)
    if _after_load not in h.load_post:
        h.load_post.append(_after_load)
    if bpy.app.background or _HANDLES:
        return
    _make_chips()
    for name, space, region, fn in (
            ('seq', bpy.types.SpaceSequenceEditor, 'PREVIEW', _draw_preview),
            ('view', bpy.types.SpaceView3D, 'WINDOW', _draw_viewport)):
        try:
            _HANDLES[name] = (space, region,
                              space.draw_handler_add(fn, (), region, 'POST_PIXEL'))
        except Exception as exc:
            print(f"[BlendCap] People boxes ({name}) not started: {exc}")


def disarm():
    global _PCOLL, _pending
    h = bpy.app.handlers
    for group in (h.undo_post, h.redo_post):
        if _after_undo in group:
            group.remove(_after_undo)
    if _after_load in h.load_post:
        h.load_post.remove(_after_load)
    for space, region, handle in list(_HANDLES.values()):
        try:
            space.draw_handler_remove(handle, region)
        except Exception:
            pass
    _HANDLES.clear()
    try:
        if bpy.app.timers.is_registered(_sync_fire):
            bpy.app.timers.unregister(_sync_fire)
        if bpy.app.timers.is_registered(_sizes_tick):
            bpy.app.timers.unregister(_sizes_tick)
    except Exception:
        pass
    _pending = False
    proc = _SIZES_JOB.get('proc')
    _SIZES_JOB.clear()
    try:
        if proc is not None and proc.poll() is None:
            proc.kill()
    except Exception:
        pass
    if _PCOLL is not None:
        try:
            bpy.utils.previews.remove(_PCOLL)
        except Exception:
            pass
        _PCOLL = None
