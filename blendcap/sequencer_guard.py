"""Keep project settings when video lands in the sequencer.

Blender's Add Movie operator (drag-and-drop and Add > Movie both use
it) carries two options that are on by default and have no preference
to turn them off: "Set View Transform" switches the scene's view
transform to Standard for sRGB media, and "Use Movie Framerate" sets
the scene frame rate to the clip's. Add Image carries the first one
too. Nothing in the addon can intercept the drop, and the operator
writes the scene data directly (no property update, so msgbus never
hears it). What CAN be seen is the aftermath: on the scene update
that follows the drop, the strip count went up and one of those two
settings changed in the same step. This handler notices exactly that
pairing and puts the setting back.

Two more things ride on the same "a clip just appeared" signal:

  * With the view transform kept on AgX or Filmic, the new clip would
    display tone-mapped (flatter than the file). Its input colour
    space is switched to the matching round-trip space ('AgX Base
    sRGB' / 'Filmic sRGB'), which the view transform then undoes
    exactly, so the preview shows the clip as-is (measured: mean
    0.19/255 off under AgX, 0.17 under Filmic). Only strips Blender
    labelled sRGB get this; linear strips (EXR) are correct as they
    are under AgX and are left alone.
  * Output > Post Processing > Sequencer is on in every new file and
    makes renders output the strips instead of the scene. It's turned
    off when a clip appears (and when the capture source switches to
    Video Editor Strip, see on_source_switched_to_vse).

All of it is gated to the BlendCap workspace tab being the active tab
in some window (preference, default on), matched against the same
accepted tab names the Load Mocap Workspace button uses. A workspace
that merely exists in the file (startup-file users) does not count;
the tab has to be the one on screen.

Per scene update the work is two setting reads and a strip walk, a
few microseconds. With every preference off the handler is not
registered at all (see sync_handler), so the cost is then zero.

Restores never fire for a setting the user changes on their own (no
strip appeared in the same update) or for a setting that already
matched. Each intervention prints one console line, nothing in the UI.

Better still, the common case never gets that far: while the gate is
open, Blender's own Add Movie / Add Image options that change the
project ("Use Movie Framerate", "Set View Transform") are pre-set to
off through the operator's remembered last-used options, so a drop
changes nothing and Blender shows no warning banner. See apply_seed.
"""
import bpy

from .helpers import addon_prefs

# scene name -> {'vt': str, 'fps': (int, float), 'movies': int,
#                'images': int, 'names': set}
_STATE = {}

# View transform -> input colour space whose inverse it is.
_ROUNDTRIP = {'AgX': 'AgX Base sRGB', 'Filmic': 'Filmic sRGB'}

_PREFS_HINT = "Preferences > BlendCap > Video Editor."


# ---------------------------------------------------------------------------
# Preferences / gating
# ---------------------------------------------------------------------------

def _flags():
    prefs = addon_prefs()
    if prefs is None:
        return None
    return {
        'vt': bool(getattr(prefs, "keep_view_transform_on_video_add", False)),
        'fps': bool(getattr(prefs, "keep_frame_rate_on_video_add", False)),
        'seq': bool(getattr(prefs, "disable_sequencer_render_on_video_add",
                            False)),
        'ws_only': bool(getattr(prefs, "sequencer_guard_workspace_only",
                                True)),
    }


def _accepted_workspace_names():
    try:
        from .operators_pipeline import BLENDCAP_OT_append_workspace
        return tuple(BLENDCAP_OT_append_workspace._ACCEPTED_NAMES)
    except Exception:
        return ("Performance Capture",)


def in_blendcap_workspace():
    """True when any window's ACTIVE workspace tab is a BlendCap tab.
    Existing-but-not-selected tabs don't count."""
    try:
        names = _accepted_workspace_names()
        for win in bpy.context.window_manager.windows:
            ws = win.workspace
            if ws is not None and ws.name in names:
                return True
    except Exception:
        return False
    return False


def _gate_open(flags):
    return (not flags['ws_only']) or in_blendcap_workspace()


def _migrate_legacy(prefs):
    """One-time: the old three-way 'Sequencer Render Warning' becomes
    the on/off checkbox. 'Leave it alone' -> off, the other two -> on.
    Runs only while the new checkbox has never been written."""
    try:
        # AddonPreferences has no keys(); is_property_set is the RNA
        # way to ask "has this ever been written".
        if prefs.is_property_set("disable_sequencer_render_on_video_add"):
            return
        if not prefs.is_property_set("sequencer_export_behavior"):
            return
        legacy = getattr(prefs, "sequencer_export_behavior", 'NOTIFY')
        prefs.disable_sequencer_render_on_video_add = (legacy != 'LEAVE')
        print(f"[BlendCap] Sequencer Render Warning setting '{legacy}' "
              f"carried over to 'Turn off Sequencer rendering' = "
              f"{legacy != 'LEAVE'}.")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Notices: console only
# ---------------------------------------------------------------------------
# Tried and rejected: operator reports (a report from an operator called
# through bpy.ops is printed by Python and never reaches the status bar)
# and workspace status text (works, but sat next to Blender's own yellow
# banner and read as noise). With the pre-set below the common case
# changes nothing, so there is nothing to announce; the console keeps
# the record.

def notify(message):
    print(f"[BlendCap] {message}")


# ---------------------------------------------------------------------------
# Pre-set Blender's own Add Movie / Add Image options
# ---------------------------------------------------------------------------
# Blender remembers an operator's last-used options for the session
# (neither option is skip-save), and Python can write them through
# WindowManager.operator_properties_last. While the gate is open the
# options that change the project are set to off, so a drop changes
# nothing and Blender has nothing to warn about; when the gate closes
# the remembered values go back. The guard below stays as the safety
# net (a drop before the pre-set landed, a user re-ticking the option
# in the redo panel) and still does the colour round-trip and the
# Sequencer-rendering switch, which the options don't cover.

_SEED = {'applied': False, 'saved': None}
_MSGBUS_OWNER = object()


def _last_props(idname):
    try:
        return bpy.context.window_manager.operator_properties_last(idname)
    except Exception:
        return None


def _seed_targets(flags):
    targets = []
    if flags['fps']:
        targets.append(("SEQUENCER_OT_movie_strip_add", "use_framerate"))
    if flags['vt']:
        targets.append(("SEQUENCER_OT_movie_strip_add",
                        "set_view_transform"))
        targets.append(("SEQUENCER_OT_image_strip_add",
                        "set_view_transform"))
    return targets


def _unseed():
    for (idname, prop), value in (_SEED['saved'] or {}).items():
        lp = _last_props(idname)
        if lp is None:
            continue
        try:
            setattr(lp, prop, value)
        except Exception:
            pass
    _SEED['applied'] = False
    _SEED['saved'] = None


def apply_seed(flags=None):
    """Pre-set the options while the gate is open, restore them when it
    closes. Idempotent; cheap enough to call from the scene-update
    handler."""
    if flags is None:
        flags = _flags()
    want = (bool(flags) and (flags['vt'] or flags['fps'])
            and _gate_open(flags))
    if want and not _SEED['applied']:
        saved = {}
        for idname, prop in _seed_targets(flags):
            lp = _last_props(idname)
            if lp is None:
                continue
            try:
                saved[(idname, prop)] = bool(getattr(lp, prop))
                setattr(lp, prop, False)
            except Exception:
                pass
        _SEED['applied'] = True
        _SEED['saved'] = saved
    elif not want and _SEED['applied']:
        _unseed()


def _on_workspace_change(*_args):
    apply_seed()


_DRAW_HANDLE = {'h': None}


def _draw_check():
    """Sequencer-editor draw callback. A switch by keyboard
    (workspace_cycle) changes the workspace without touching the RNA
    property, so the msgbus never hears it (verified) - but a clip can
    only be dropped into a sequencer editor that is on screen, and one
    that is on screen redraws. Re-check the gate here. Runs only while
    a sequencer is visible; a few attribute reads per redraw and no
    writes unless the gate state actually changed."""
    try:
        apply_seed()
    except Exception:
        pass


def _start_seed_watch():
    if bpy.app.background or _DRAW_HANDLE['h'] is not None:
        return
    try:
        _DRAW_HANDLE['h'] = bpy.types.SpaceSequenceEditor.draw_handler_add(
            _draw_check, (), 'WINDOW', 'POST_PIXEL')
    except Exception:
        _DRAW_HANDLE['h'] = None


def _stop_seed_watch():
    h = _DRAW_HANDLE['h']
    if h is None:
        return
    try:
        bpy.types.SpaceSequenceEditor.draw_handler_remove(h, 'WINDOW')
    except Exception:
        pass
    _DRAW_HANDLE['h'] = None


def _subscribe_workspace():
    """Re-evaluate the pre-set whenever a window's workspace tab
    changes. Persistent across file loads; re-subscribed on load
    anyway for safety."""
    try:
        bpy.msgbus.clear_by_owner(_MSGBUS_OWNER)
        bpy.msgbus.subscribe_rna(
            key=(bpy.types.Window, "workspace"),
            owner=_MSGBUS_OWNER, args=(),
            notify=_on_workspace_change, options={'PERSISTENT'})
    except Exception:
        pass


def _unsubscribe_workspace():
    try:
        bpy.msgbus.clear_by_owner(_MSGBUS_OWNER)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------

def _strips(scene):
    se = getattr(scene, "sequence_editor", None)
    if se is None:
        return ()
    strips = getattr(se, "strips_all", None)
    if strips is None:
        strips = getattr(se, "sequences_all", ())
    return strips


def _snapshot(scene):
    movies = images = 0
    names = set()
    for s in _strips(scene):
        t = s.type
        if t == 'MOVIE':
            movies += 1
            names.add(s.name)
        elif t == 'IMAGE':
            images += 1
            names.add(s.name)
    return {
        'vt': scene.view_settings.view_transform,
        'fps': (scene.render.fps, scene.render.fps_base),
        'movies': movies,
        'images': images,
        'names': names,
    }


def _fps_label(pair):
    fps, base = pair
    value = fps / base if base else fps
    return f"{value:g}"


def _roundtrip_new_strips(scene, new_names, view_transform):
    """Give freshly added sRGB strips the colour space that round-trips
    through the current view transform. Returns (count, space)."""
    space = _ROUNDTRIP.get(view_transform)
    if not space or not new_names:
        return 0, None
    n = 0
    for s in _strips(scene):
        if s.name not in new_names or s.type not in ('MOVIE', 'IMAGE'):
            continue
        cs = getattr(s, "colorspace_settings", None)
        if cs is None or cs.name != 'sRGB':
            continue
        try:
            cs.name = space
            if cs.name == space:
                n += 1
        except Exception:
            pass
    return n, space


def check_scene(scene, flags, gate_open=None):
    """One scene's guard step. Split out from the handler so it can be
    exercised headless without a depsgraph. Returns the list of things
    it did: 'view_transform', 'colorspace', 'frame_rate',
    'sequencer_render'."""
    now = _snapshot(scene)
    prev = _STATE.get(scene.name)
    done = []
    parts = []
    if prev is not None:
        if gate_open is None:
            gate_open = _gate_open(flags)
        movie_added = now['movies'] > prev['movies']
        image_added = now['images'] > prev['images']
        new_names = now['names'] - prev['names']
        if gate_open and (movie_added or image_added):
            if flags['vt'] and now['vt'] != prev['vt']:
                try:
                    blender_set = now['vt']
                    scene.view_settings.view_transform = prev['vt']
                    now['vt'] = prev['vt']
                    done.append('view_transform')
                    parts.append(f"kept the {prev['vt']} view transform "
                                 f"(Blender set {blender_set})")
                except Exception:
                    pass
            if flags['vt']:
                n, space = _roundtrip_new_strips(scene, new_names, now['vt'])
                if n:
                    done.append('colorspace')
                    parts.append(f"set the clip to display as-is "
                                 f"({space})")
            if flags['fps'] and movie_added and now['fps'] != prev['fps']:
                try:
                    blender_set = now['fps']
                    scene.render.fps, scene.render.fps_base = prev['fps']
                    now['fps'] = prev['fps']
                    done.append('frame_rate')
                    parts.append(f"kept {_fps_label(prev['fps'])} fps "
                                 f"(Blender set "
                                 f"{_fps_label(blender_set)})")
                except Exception:
                    pass
            if flags['seq'] and getattr(scene.render, 'use_sequencer',
                                        False):
                try:
                    scene.render.use_sequencer = False
                    done.append('sequencer_render')
                    parts.append("turned off Sequencer rendering so "
                                 "renders show your scene, not the video")
                except Exception:
                    pass
    _STATE[scene.name] = now
    if parts:
        notify("BlendCap: " + ", ".join(parts) + ". " + _PREFS_HINT)
    return done


def on_source_switched_to_vse(scene):
    """Second trigger for the Sequencer-rendering option: the capture
    source switched to Video Editor Strip (interactive switch only;
    properties.update_video_source calls this). Returns True when the
    option was turned off."""
    flags = _flags()
    if not flags or not flags['seq']:
        return False
    if not _gate_open(flags):
        return False
    if not getattr(scene.render, 'use_sequencer', False):
        return False
    try:
        scene.render.use_sequencer = False
    except Exception:
        return False
    notify("BlendCap: turned off Sequencer rendering so renders show "
           "your scene, not the video. " + _PREFS_HINT)
    return True


@bpy.app.handlers.persistent
def _sequencer_guard(scene, depsgraph=None):
    """depsgraph_update_post: guard the updated scene. load_post passes
    a filepath string instead of a scene; then just resync every scene
    so the first update after a load can't 'restore' anything."""
    if not isinstance(scene, bpy.types.Scene):
        _STATE.clear()
        for s in bpy.data.scenes:
            try:
                _STATE[s.name] = _snapshot(s)
            except Exception:
                pass
        return
    flags = _flags()
    if not flags or not (flags['vt'] or flags['fps'] or flags['seq']):
        _STATE[scene.name] = _snapshot(scene)
        return
    try:
        # Tab switches that bypass msgbus (workspace_cycle etc.) still
        # produce scene updates; keep the pre-set in step here too.
        apply_seed(flags)
        check_scene(scene, flags)
    except Exception:
        # Visible, once: a silent failure here looks exactly like
        # "the guard does nothing".
        if not _STATE.get('__reported__'):
            _STATE['__reported__'] = True
            import traceback
            print("[BlendCap] sequencer guard error:")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Attach / detach
# ---------------------------------------------------------------------------

def _deferred_sync():
    """Timer: the preferences entry can be missing while register()
    runs (Blender adds it around the enable call), so a sync at that
    moment reads every flag as off and attaches nothing. Re-sync once
    the first UI tick has passed, when the entry exists."""
    sync_handler()
    return None


@bpy.app.handlers.persistent
def _sync_on_load(_filepath=None, _depsgraph=None):
    """load_post: re-sync after a file load as well - cheap, and it
    covers a session where the timer fired before prefs existed."""
    sync_handler(quiet=True)


def arm():
    """Called from register(): sync now (harmless if prefs aren't there
    yet), and again shortly after via a one-shot timer + on every file
    load."""
    sync_handler(quiet=True)
    try:
        if not bpy.app.timers.is_registered(_deferred_sync):
            bpy.app.timers.register(_deferred_sync, first_interval=0.5,
                                    persistent=True)
    except Exception:
        pass
    if _sync_on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_sync_on_load)


def sync_handler(quiet=False):
    """Attach the handler when any of the three options is on, detach
    it when all are off. Called at register (directly, by timer, on
    load) and from the preference update callbacks."""
    prefs = addon_prefs()
    if prefs is not None:
        _migrate_legacy(prefs)
    flags = _flags()
    want = bool(flags) and (flags['vt'] or flags['fps'] or flags['seq'])
    dg = bpy.app.handlers.depsgraph_update_post
    lp = bpy.app.handlers.load_post
    if want:
        newly = _sequencer_guard not in dg
        if newly:
            _STATE.clear()
            for s in bpy.data.scenes:
                try:
                    _STATE[s.name] = _snapshot(s)
                except Exception:
                    pass
            dg.append(_sequencer_guard)
        if _sequencer_guard not in lp:
            lp.append(_sequencer_guard)
        # Re-evaluate the pre-set against the current flags/gate.
        _unseed()
        apply_seed(flags)
        _subscribe_workspace()
        _start_seed_watch()
        if newly or not quiet:
            what = ", ".join(n for n, on in (
                ("view transform", flags['vt']),
                ("frame rate", flags['fps']),
                ("Sequencer rendering off", flags['seq'])) if on)
            where = ("in the BlendCap workspace" if flags['ws_only']
                     else "everywhere")
            print(f"[BlendCap] sequencer guard active ({where}): "
                  f"{what} when video is added.")
    else:
        if _sequencer_guard in dg:
            print("[BlendCap] sequencer guard off (all options "
                  "unticked).")
        remove_handler()
        _unseed()
        _unsubscribe_workspace()
        _stop_seed_watch()
        if not quiet and prefs is None:
            print("[BlendCap] sequencer guard: preferences not found, "
                  "guard not attached.")


def remove_handler():
    dg = bpy.app.handlers.depsgraph_update_post
    lp = bpy.app.handlers.load_post
    if _sequencer_guard in dg:
        dg.remove(_sequencer_guard)
    if _sequencer_guard in lp:
        lp.remove(_sequencer_guard)
    _STATE.clear()


def disarm():
    """Called from unregister(): everything arm() and sync_handler()
    installed."""
    remove_handler()
    _unseed()
    _unsubscribe_workspace()
    _stop_seed_watch()
    if _sync_on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_sync_on_load)
    try:
        if bpy.app.timers.is_registered(_deferred_sync):
            bpy.app.timers.unregister(_deferred_sync)
    except Exception:
        pass
