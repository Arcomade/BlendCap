"""Preset backup/transfer and dependency-storage operators.

Export/Import bundle the user's custom presets (retarget maps + rest
poses) into a single zip - backup, machine-to-machine transfer, and
sharing maps for rigs BlendCap doesn't ship. Move relocates the
installed dependency files into the configured storage location (the
cross-volume case migration deliberately leaves to the user - see
helpers.migrate_user_data).
"""
import os
import sys
import json
import shutil
import threading
import time
import zipfile

import bpy
from bpy_extras.io_utils import ExportHelper, ImportHelper

from .helpers import (
    _addon_dir, _deps_custom_dir, _deps_default_dir, _dir_has_venv,
    _filename_error, _install_marker_path_legacy, _invalidate_deps_cache,
    _tag_redraw, blendcap_state, bvh_state, deps_root, deps_target,
    pending_user_data_moves, presets_root, move_user_presets,
    storage_speed_warning, _lp, MOVE_SENTINEL,
)
from .properties import (_rest_pose_presets_dir_user,
                         _rest_pose_presets_dir_legacy)
from .retarget.preset_io import (SHIPPED_PRESET_FILENAMES,
                                 _retarget_maps_dir_user,
                                 _retarget_maps_dir_legacy_config)


# ============================================================
# Preset Export / Import
# ============================================================

def _iter_custom_presets():
    """Yield (arc_prefix, filename, absolute_path) for every custom
    preset the dropdowns can see - the new user-save folders plus the
    read-only legacy locations, deduped by filename with the new home
    winning (same rule the dropdowns use). Shipped retarget filenames
    and the bundled/ rest poses never yield."""
    from .helpers import preset_read_dirs
    sources = (
        ("retarget_maps",
         (*preset_read_dirs("retarget_maps"),
          _retarget_maps_dir_legacy_config()),
         SHIPPED_PRESET_FILENAMES),
        ("rest_pose_presets",
         (*preset_read_dirs("rest_pose_presets"),
          _rest_pose_presets_dir_legacy()),
         frozenset()),
    )
    for arc_prefix, dirs_in_priority, skip in sources:
        seen = set()
        for i, d in enumerate(dirs_in_priority):
            if not os.path.isdir(d):
                continue
            for root, dirs, files in os.walk(d):
                # The legacy rest-pose folder holds the shipped
                # bundled/ subtree - never export shipped content.
                dirs[:] = sorted(x for x in dirs if x != "bundled")
                files.sort()
                for fn in files:
                    if (not fn.endswith(".json") or fn in skip
                            or fn in seen):
                        continue
                    seen.add(fn)
                    yield arc_prefix, fn, os.path.join(root, fn)


class BLENDCAP_OT_export_presets(bpy.types.Operator, ExportHelper):
    bl_idname = "blendcap.export_presets"
    bl_label = "Export Presets"
    bl_description = ("Save all your custom retarget maps and rest "
                      "pose presets into one zip file - for backup, "
                      "moving to another computer, or sharing")
    bl_options = {'REGISTER'}

    filename_ext = ".zip"
    filter_glob: bpy.props.StringProperty(  # type: ignore
        default="*.zip", options={'HIDDEN'})

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = "blendcap_presets.zip"
        return super().invoke(context, event)

    def execute(self, context):
        count = 0
        try:
            with zipfile.ZipFile(self.filepath, "w",
                                 zipfile.ZIP_DEFLATED) as zf:
                for arc_prefix, fn, path in _iter_custom_presets():
                    zf.write(path, arcname=arc_prefix + "/" + fn)
                    count += 1
        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {e}")
            return {'CANCELLED'}
        if not count:
            try:
                os.remove(self.filepath)
            except Exception:
                pass
            self.report({'WARNING'}, "No custom presets to export yet.")
            return {'CANCELLED'}
        self.report({'INFO'},
                    f"Exported {count} preset(s) to "
                    f"{os.path.basename(self.filepath)}")
        return {'FINISHED'}


# Import size guard: presets are small JSON files; anything this large
# inside a preset zip is not a preset.
_IMPORT_MAX_ENTRY_BYTES = 20 * 1024 * 1024


class BLENDCAP_OT_import_presets(bpy.types.Operator, ImportHelper):
    bl_idname = "blendcap.import_presets"
    bl_label = "Import Presets"
    bl_description = ("Add presets from a zip made with Export Presets. "
                      "Presets you already have are kept unless "
                      "'Replace existing' is ticked")
    bl_options = {'REGISTER'}

    filename_ext = ".zip"
    filter_glob: bpy.props.StringProperty(  # type: ignore
        default="*.zip", options={'HIDDEN'})
    overwrite: bpy.props.BoolProperty(  # type: ignore
        name="Replace existing",
        description="When the file contains a preset with the same "
                    "name as one you already have, replace yours with "
                    "the imported one. Unticked, yours are kept and "
                    "those entries are skipped",
        default=False)

    def execute(self, context):
        try:
            zf = zipfile.ZipFile(self.filepath, "r")
        except Exception as e:
            self.report({'ERROR'}, f"Couldn't open the file: {e}")
            return {'CANCELLED'}

        added = replaced = kept = not_presets = 0
        with zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                # Basename ONLY - paths inside a zip are untrusted and
                # must never steer where a file lands.
                base = os.path.basename(
                    info.filename.replace("\\", "/"))
                if (not base.endswith(".json")
                        or _filename_error(base)
                        or base in SHIPPED_PRESET_FILENAMES
                        or info.file_size > _IMPORT_MAX_ENTRY_BYTES):
                    not_presets += 1
                    continue
                # Only real preset files get written: a retarget map
                # has a "pairs" list, a rest pose a "bones" table.
                # Anything else in the zip is skipped, not extracted.
                try:
                    raw = zf.read(info)
                    data = json.loads(raw.decode("utf-8"))
                except Exception:
                    not_presets += 1
                    continue
                if isinstance(data, dict) and "pairs" in data:
                    dest_dir = _retarget_maps_dir_user()
                elif isinstance(data, dict) and "bones" in data:
                    dest_dir = _rest_pose_presets_dir_user()
                else:
                    not_presets += 1
                    continue
                dest = os.path.join(dest_dir, base)
                existed = os.path.isfile(dest)
                if existed and not self.overwrite:
                    kept += 1
                    continue
                try:
                    with open(dest, "wb") as f:
                        f.write(raw)
                except Exception as e:
                    self.report({'ERROR'}, f"Import failed at "
                                           f"{base}: {e}")
                    return {'CANCELLED'}
                if existed:
                    replaced += 1
                else:
                    added += 1

        parts = []
        if added:
            parts.append(f"added {added}")
        if replaced:
            parts.append(f"replaced {replaced}")
        if kept:
            parts.append(f"kept {kept} you already had (tick "
                         f"'Replace existing' to overwrite)")
        if not parts and not_presets:
            self.report({'WARNING'},
                        "Nothing imported - the file doesn't contain "
                        "any BlendCap presets.")
            return {'CANCELLED'}
        if not parts:
            self.report({'WARNING'}, "The file was empty.")
            return {'CANCELLED'}
        msg = "Import: " + ", ".join(parts)
        if not_presets:
            msg += f" ({not_presets} non-preset file(s) ignored)"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


# ============================================================
# Storage consent dialog
# ============================================================
# 1.0.5 kept everything inside the addon folder, which updates
# replace. NOTHING moves automatically: this dialog appears shortly
# after startup whenever custom presets or the dependency files still
# sit in the addon folder, explains the risk, and offers the move -
# to the default locations in one click, or via preferences for
# custom ones. Dismissing changes nothing and the dialog returns next
# launch (deliberate: the at-risk state should not be silent).
# Fired by a timer from registration - register() itself runs before
# any window exists, so a dialog can't open there.

# Enum-items cache: Blender requires the returned strings to stay
# referenced or the dropdown labels corrupt (standard dynamic-items
# rule, same as the preset dropdowns).
_ACTION_ITEMS_CACHE = []


def _consent_action_items():
    custom = bool(_deps_custom_dir()) or bool(
        os.path.normcase(presets_root())
        != os.path.normcase(_presets_default_dir_safe()))
    items = [
        ('MOVE',
         ("Move files to the locations shown above" if custom
          else "Move files to the new default location"),
         "Move your custom presets and the dependency files to the "
         "folders shown above. Progress shows in BlendCap's "
         "preferences"),
        ('CHOOSE',
         "Open the preferences and let me choose different folders",
         "Nothing moves yet. Pick your own folders in BlendCap's "
         "preferences, then move the files there. The locations can "
         "be changed again at any time"),
    ]
    _ACTION_ITEMS_CACHE.clear()
    _ACTION_ITEMS_CACHE.extend(items)
    return _ACTION_ITEMS_CACHE


def _short_path(p):
    """Readable display form of a long storage path: keep the start
    (drive + user, which the reader recognizes) and the identifying
    tail, elide the AppData noise in the middle. Short paths - custom
    folders are usually 'D:\\BlendCap'-ish - come back untouched. The
    real path stays visible in Preferences (the dialog says so)."""
    p = os.path.normpath(p or "")
    if len(p) <= 50:
        return p
    parts = p.split(os.sep)
    if len(parts) <= 6:
        return p
    return os.sep.join(parts[:3]) + os.sep + "..." + os.sep \
        + os.sep.join(parts[-3:])


class BLENDCAP_OT_storage_consent(bpy.types.Operator):
    bl_idname = "blendcap.storage_consent"
    bl_label = "BlendCap Update"
    bl_options = {'REGISTER', 'INTERNAL'}

    # Radio choice + one Continue button: buttons drawn in a dialog
    # BODY run their operator but leave the dialog open, in BOTH
    # dialog flavors (props dialog AND plain popup - live-observed
    # 2026-08-27), so the actions must ride the footer confirm. A
    # radio picks WHICH action; Continue applies it and closes.
    # Items via callback: the MOVE label says "default location" only
    # when the targets ARE the defaults (custom folders configured ->
    # "the locations shown above"). First item = default selection.
    action: bpy.props.EnumProperty(  # type: ignore
        items=lambda self, context: _consent_action_items(),
        options={'SKIP_SAVE'})

    def invoke(self, context, event):
        # Dismissed is dismissed - Cancel, Esc, or a click outside all
        # postpone until next launch, no same-session retries (a
        # reopen-if-dismissed-quickly heuristic was rejected as spammy;
        # the quiet-fire watcher already keeps stray clicks from
        # landing on a freshly opened dialog).
        return context.window_manager.invoke_props_dialog(
            self, width=470, confirm_text="Continue")

    def draw(self, context):
        pending = pending_user_data_moves()
        col = self.layout.column(align=True)
        col.label(text="Your BlendCap files should be moved out of "
                       "the addon folder", icon='ERROR')
        col.separator()
        col.label(text="BlendCap now stores its files in a new "
                       "location, so")
        col.label(text="updating the addon can no longer delete your "
                       "dependencies")
        col.label(text="and custom presets or force you to reinstall "
                       "them.")
        col.separator()

        box = col.box()
        bcol = box.column(align=True)
        bcol.label(text="Where the files will go:")
        if pending["presets"]:
            bcol.label(text=f"Custom presets ({pending['presets']})",
                       icon='PRESET')
            bcol.label(text="     → " + _short_path(presets_root()))
        if pending["deps"]:
            bcol.label(text="Dependency files (about 11 GB)",
                       icon='DISK_DRIVE')
            bcol.label(text="     → " + _short_path(deps_target()))
            warn = storage_speed_warning(deps_target())
            if warn:
                wcol = bcol.column(align=True)
                wcol.alert = warn[0] == 'slow_fs'
                for i, line in enumerate(warn[1]):
                    wcol.label(text=line,
                               icon=('ERROR' if i == 0 else 'NONE'))
        bcol.label(text="(full paths are shown in Preferences)")
        col.separator()

        radio = col.column(align=True)
        radio.prop(self, "action", expand=True)
        col.separator()
        col.label(text="No changes have been made yet. If you cancel, "
                       "this notice")
        col.label(text="will pop up again next launch (it can be "
                       "turned off in")
        col.label(text="the preferences).")

    def execute(self, context):
        if self.action == 'CHOOSE':
            try:
                bpy.ops.blendcap.install_open_prefs()
            except Exception:
                pass
            self.report({'INFO'},
                        "Nothing moved yet - pick your folders under "
                        "File & Cache Locations, then use the Move "
                        "buttons there.")
            return {'FINISHED'}
        return _do_move_all(self, context)


def _presets_default_dir_safe():
    from .helpers import _presets_default_dir
    return _presets_default_dir()


def _do_move_all(op, context):
    """Move presets (sync) then dependencies (worker thread), with
    prefs opened so the progress bar is visible. Shared by the consent
    dialog's confirm button and blendcap.storage_move_all so reports
    surface from whichever operator the user actually clicked."""
    if blendcap_state.get('running') or bvh_state.get('running'):
        op.report({'ERROR'}, "Finish or cancel the running job first.")
        return {'CANCELLED'}
    # Presets first: tiny files, instant, synchronous.
    log = []
    n = move_user_presets(log=log)
    for line in log:
        print("[BlendCap] presets: " + line)
    # Then the heavyweight move, on its worker thread. Preferences are
    # NOT opened here - the user chose "move", not "show me settings"
    # (live feedback 2026-08-29). move_deps opens them by itself only
    # when the move turns out to be slow, so the progress bar appears
    # exactly when there is something to watch.
    pending = pending_user_data_moves()
    if pending["deps"]:
        bpy.ops.blendcap.move_deps('INVOKE_DEFAULT', skip_warning=True)
    if n and not pending["deps"]:
        op.report({'INFO'}, f"Moved {n} custom preset(s) to safety.")
    elif n:
        op.report({'INFO'},
                  f"Moved {n} custom preset(s); dependency files are "
                  f"moving now.")
    elif not pending["deps"]:
        op.report({'INFO'}, "Nothing needed moving.")
    return {'FINISHED'}


class BLENDCAP_OT_storage_move_all(bpy.types.Operator):
    bl_idname = "blendcap.storage_move_all"
    bl_label = "Move BlendCap Files"
    bl_description = ("Move your custom presets and the dependency "
                      "files out of the addon folder to the locations "
                      "shown, so addon updates can't delete them")
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, context):
        return _do_move_all(self, context)


class BLENDCAP_OT_move_presets(bpy.types.Operator):
    bl_idname = "blendcap.move_presets"
    bl_label = "Move Presets to Safety"
    bl_description = ("Move your custom presets out of the addon "
                      "folder to the presets folder shown above. "
                      "Small files - this is instant")
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, context):
        log = []
        n = move_user_presets(log=log)
        for line in log:
            print("[BlendCap] presets: " + line)
        _tag_redraw('PREFERENCES')
        if not n:
            self.report({'INFO'}, "No presets needed moving.")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Moved {n} preset(s).")
        return {'FINISHED'}


# Watcher session state. event_seen proves event delivery reaches the
# modal (the watchdog falls back to a direct open when it doesn't);
# fired stops duplicates; shutdown lets unregister stop a live
# watcher cleanly.
_WATCH = {"event_seen": False, "fired": False, "shutdown": False}


def _warning_hidden():
    """The user opted out of the startup notice in preferences."""
    try:
        from .helpers import addon_prefs
        prefs = addon_prefs()
        return bool(prefs and getattr(prefs, "hide_storage_warning",
                                      False))
    except Exception:
        return False


def rearm_consent_if_needed():
    """Loading a .blend that carries its own UI replaces the window
    manager, which silently kills the click-watcher modal (reported
    2026-09-21: opening a saved project directly didn't reliably show
    the window). Re-arm from load_post when the window is still owed."""
    try:
        if bpy.app.background or _WATCH["fired"] or _WATCH["shutdown"]:
            return
        if _warning_hidden():
            return
        p = pending_user_data_moves()
        if not (p["presets"] or p["deps"]):
            return
        if not bpy.app.timers.is_registered(_storage_consent_timer):
            bpy.app.timers.register(_storage_consent_timer,
                                    first_interval=0.1, persistent=True)
    except Exception:
        pass


@bpy.app.handlers.persistent
def _consent_on_load(_filepath=None, _extra=None):
    rearm_consent_if_needed()


class BLENDCAP_OT_storage_consent_watcher(bpy.types.Operator):
    """Invisible pass-through modal that shows the consent dialog
    right after the user's first click of the session (the click that
    dismisses the splash screen). The watcher arms fractions of a
    second after launch - before any human can click - so the trigger
    click can't be missed (live-reported miss 2026-08-27: a click
    BEFORE the old 1.5 s arming left the watcher waiting forever).
    The dialog opens on a 0.2 s timer rather than inside the press
    event itself: the button RELEASE lands a few ms after the press,
    and a release outside a freshly opened dialog would cancel it.
    Every event passes through untouched; only its own TIMER events
    are consumed."""
    bl_idname = "blendcap.storage_consent_watcher"
    bl_label = "BlendCap Storage Watcher"
    bl_options = {'INTERNAL'}

    def invoke(self, context, event):
        self._timer = context.window_manager.event_timer_add(
            0.25, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _stop(self, context):
        try:
            context.window_manager.event_timer_remove(self._timer)
        except Exception:
            pass
        return {'FINISHED'}

    def modal(self, context, event):
        _WATCH["event_seen"] = True
        if _WATCH["shutdown"] or _WATCH["fired"]:
            # Wind down - but only ever CONSUME one of our own TIMER
            # ticks. Ending on the user's press (or its release) would
            # swallow the very click that dismisses the splash.
            if event.type == 'TIMER':
                return self._stop(context)
            return {'PASS_THROUGH'}
        if event.value == 'PRESS':
            _WATCH["fired"] = True
            try:
                bpy.app.timers.register(_open_consent_dialog,
                                        first_interval=0.2)
            except Exception:
                pass
        return {'PASS_THROUGH'}


def _open_consent_dialog():
    """Direct fallback: show the consent dialog (re-checks pending and
    the opt-out first)."""
    try:
        wm = bpy.context.window_manager
        if wm and wm.windows and not _warning_hidden():
            p = pending_user_data_moves()
            if p["presets"] or p["deps"]:
                _WATCH["fired"] = True
                with bpy.context.temp_override(window=wm.windows[0]):
                    bpy.ops.blendcap.storage_consent('INVOKE_DEFAULT')
    except Exception:
        pass
    return None


def _watcher_watchdog():
    """If no event ever reached the watcher modal (event delivery from
    a timer-started modal is the one link that can't be proven in
    advance), show the dialog directly - late and possibly over the
    splash beats never."""
    if not _WATCH["event_seen"] and not _WATCH["fired"]:
        _open_consent_dialog()
    return None


def _storage_consent_timer():
    """Startup: poll FAST for the first window (0.1 s steps - the
    watcher must be listening before any human can click), then arm
    the watcher + its watchdog. Registered by registration.register()
    whenever something is pending, and re-registered by the consent
    dialog's accidental-dismiss retry."""
    try:
        if _WATCH["shutdown"] or _warning_hidden():
            return None
        wm = bpy.context.window_manager
        if not wm or not wm.windows:
            return 0.1  # no window yet - poll fast
        p = pending_user_data_moves()
        if not (p["presets"] or p["deps"]):
            return None
        _WATCH["fired"] = False
        _WATCH["event_seen"] = False
        with bpy.context.temp_override(window=wm.windows[0]):
            result = bpy.ops.blendcap.storage_consent_watcher(
                'INVOKE_DEFAULT')
        if 'RUNNING_MODAL' in result:
            bpy.app.timers.register(_watcher_watchdog,
                                    first_interval=4.0)
        else:
            _open_consent_dialog()
    except Exception:
        try:
            _open_consent_dialog()
        except Exception:
            pass
    return None


# ============================================================
# Dependency storage move
# ============================================================
# State shared with the preferences panel (drawn while a move runs).
# A cross-drive move copies ~13 GB / tens of thousands of files, so it
# runs on a worker thread with a modal timer relaying status; same-
# drive moves are a couple of os.rename calls and finish instantly.
# done/total drive a real progress bar (layout.progress) in prefs.
# cancel: set by the Cancel button, honoured between files during the
# copy phase only. phase: 'copy' (cancellable - sources untouched) or
# 'finish' (sources being deleted - too late to unwind, and quick).
MOVE_STATE = {"running": False, "msg": "", "error": "",
              "done": 0, "total": 0, "cancel": False,
              "cancelled": False, "phase": "copy"}


class _MoveCancelled(Exception):
    pass


def _rmtree_quiet(path):
    try:
        if os.path.isdir(path):
            shutil.rmtree(_lp(path))
    except Exception:
        pass


def _copy_tree_with_progress(src, dst, state, label):
    """Copy src -> dst file by file (so the progress bar can update and
    Cancel can land between files). Does NOT touch src - deletion is a
    separate, later phase so a cancel can always restore. Raises
    _MoveCancelled when the cancel flag is seen."""
    lsrc, ldst = _lp(src), _lp(dst)
    total = 0
    for root, _dirs, files in os.walk(lsrc):
        total += len(files)
    state["done"], state["total"] = 0, total
    done = 0
    for root, dirs, files in os.walk(lsrc):
        rel = os.path.relpath(root, lsrc)
        troot = os.path.join(ldst, rel) if rel != "." else ldst
        os.makedirs(troot, exist_ok=True)
        for fn in files:
            if state.get("cancel"):
                raise _MoveCancelled()
            shutil.copy2(os.path.join(root, fn),
                         os.path.join(troot, fn))
            done += 1
            if done % 25 == 0 or done == total:
                state["done"] = done
                state["msg"] = (f"Moving {label}... "
                                f"{done:,} of {total:,} files")
    state["done"], state["total"] = 0, 0


def _run_deps_move(src, dst, state):
    """Two phases. COPY: every tree lands at dst (instant rename when
    the OS allows, file-by-file copy otherwise) while the sources stay
    untouched, so Cancel - or a killed Blender - can always leave the
    original install intact. FINISH: sources deleted, marker moved,
    sentinel removed. The sentinel (helpers.MOVE_SENTINEL) sits at dst
    from before the first byte until after the last delete: while it's
    there the resolver refuses to treat dst as an environment, and the
    next attempt wipes the half-copy instead of skipping it."""
    sentinel = os.path.join(dst, MOVE_SENTINEL)
    done_ops = []  # (name, 'renamed' | 'copied') for cancel-unwind
    started = []
    try:
        state["phase"] = "copy"
        state["cancelled"] = False
        os.makedirs(dst, exist_ok=True)
        if os.path.isfile(sentinel):
            # Leftover of an earlier interrupted move: clear the partial
            # copy before starting over.
            for name in ("venv", "checkpoints"):
                _rmtree_quiet(os.path.join(dst, name))
        with open(sentinel, "w") as f:
            f.write("move in progress")

        started = []  # trees we began on - the only ones cancel may delete
        for name in ("venv", "checkpoints"):
            s = os.path.join(src, name)
            d = os.path.join(dst, name)
            if not os.path.isdir(s) or os.path.exists(d):
                continue
            if state.get("cancel"):
                raise _MoveCancelled()
            state["msg"] = f"Moving {name}..."
            started.append(name)
            try:
                os.rename(s, d)  # same drive: instant
                done_ops.append((name, 'renamed'))
            except OSError as e:
                # Cross-drive, or a transient handle inside the folder
                # (Explorer/AV after recent activity there) - the copy
                # fallback succeeds either way, just slower. Name the
                # reason so a surprise progress bar is explainable.
                print(f"[BlendCap] rename of {name} refused "
                      f"({e}); copying instead")
                _copy_tree_with_progress(s, d, state, name)
                done_ops.append((name, 'copied'))

        # FINISH: point of no return - quick, and cancel is disabled.
        state["phase"] = "finish"
        state["msg"] = "Finishing up..."
        for name, how in done_ops:
            if how == 'copied':
                _rmtree_quiet(os.path.join(src, name))
        for marker in (os.path.join(src, "install_marker.json"),
                       _install_marker_path_legacy()):
            if os.path.isfile(marker):
                d = os.path.join(dst, "install_marker.json")
                if not os.path.exists(d):
                    shutil.move(marker, d)
        try:
            os.remove(sentinel)
        except Exception:
            pass
        state["msg"] = "Done."
    except _MoveCancelled:
        # Unwind: renamed trees go back by name (instant), copied
        # partials at dst are deleted. Sources were never touched.
        state["msg"] = "Cancelling - restoring..."
        state["cancel"] = False  # the restore itself must not be cut short
        for name, how in done_ops:
            d = os.path.join(dst, name)
            if how == 'renamed':
                back = os.path.join(src, name)
                try:
                    os.rename(d, back)
                except OSError:
                    # The rename back was refused too (same transient
                    # lock class): copy it home, then drop the copy at
                    # dst. Slower, but the original location is what
                    # the user asked to keep.
                    try:
                        _copy_tree_with_progress(d, back, state,
                                                 f"{name} (restoring)")
                        _rmtree_quiet(d)
                    except Exception as e2:
                        state["error"] = (f"couldn't restore {name} to "
                                          f"{back}: {e2}")
            else:
                _rmtree_quiet(d)
        for name in started:
            # The tree that was mid-copy when cancel landed (never a
            # pre-existing folder at dst we didn't create).
            if not any(n == name for n, _ in done_ops):
                _rmtree_quiet(os.path.join(dst, name))
        try:
            os.remove(sentinel)
        except Exception:
            pass
        state["cancelled"] = True
        state["msg"] = "Cancelled."
    except Exception as e:
        state["error"] = str(e)
    finally:
        state["done"], state["total"] = 0, 0
        state["running"] = False
        _invalidate_deps_cache()


class BLENDCAP_OT_move_deps(bpy.types.Operator):
    bl_idname = "blendcap.move_deps"
    bl_label = "Move Dependency Files"
    bl_description = ("Move the installed dependency files (the Python "
                      "environment and tracking models, about 11 GB) to the "
                      "storage location shown above. Usually instant when "
                      "both are on the same drive; between drives the files "
                      "are copied, which can take a while")
    bl_options = {'REGISTER', 'INTERNAL'}

    _timer = None

    # True when the caller already showed the slow-drive warning (the
    # startup consent dialog does) - skips the confirm step here.
    skip_warning: bpy.props.BoolProperty(  # type: ignore
        default=False, options={'SKIP_SAVE', 'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return not MOVE_STATE["running"]

    def invoke(self, context, event):
        # Pre-operation heads-up for slow/removable targets - the user
        # decides BEFORE the multi-hour case starts, not during.
        self._warn = (None if self.skip_warning
                      else storage_speed_warning(deps_target()))
        if self._warn:
            return context.window_manager.invoke_props_dialog(
                self, width=440)
        return self.execute(context)

    def draw(self, context):
        warn = getattr(self, "_warn", None)
        if not warn:
            return
        col = self.layout.column(align=True)
        col.alert = warn[0] == 'slow_fs'
        for i, line in enumerate(warn[1]):
            col.label(text=line, icon=('ERROR' if i == 0 else 'NONE'))
        col2 = self.layout.column(align=True)
        col2.label(text="OK starts the move anyway; progress shows "
                        "below in")
        col2.label(text="this panel. Esc cancels.")

    def execute(self, context):
        if blendcap_state.get('running') or bvh_state.get('running'):
            self.report({'ERROR'},
                        "Finish or cancel the running job before "
                        "moving the dependency files.")
            return {'CANCELLED'}

        # Source: wherever the deps actually are. The addon folder is
        # checked first - clearing the legacy layout is this button's
        # whole reason to exist.
        src = None
        for cand in (_addon_dir(), deps_root()):
            if _dir_has_venv(cand):
                src = cand
                break
        if src is None:
            self.report({'ERROR'}, "No installed dependencies found "
                                   "to move.")
            return {'CANCELLED'}
        dst = deps_target()
        if (os.path.normcase(os.path.abspath(src))
                == os.path.normcase(os.path.abspath(dst))):
            self.report({'INFO'}, "The dependency files are already "
                                  "in the storage location.")
            return {'CANCELLED'}
        if _dir_has_venv(dst):
            self.report({'ERROR'},
                        "The storage location already contains an "
                        "installed environment. Uninstall one of the "
                        "two first (Uninstall button above).")
            return {'CANCELLED'}

        # Sweep: the dependency move also rescues any presets still in
        # the addon folder (kilobytes, instant) - so the FIRST move a
        # user makes secures everything, and a partial move can't be
        # mistaken for a complete one. The reverse never happens: a
        # preset move never drags 11 GB along.
        self._swept = 0
        try:
            slog = []
            self._swept = move_user_presets(log=slog)
            for line in slog:
                print("[BlendCap] presets: " + line)
        except Exception:
            self._swept = 0

        MOVE_STATE["running"] = True
        MOVE_STATE["msg"] = "Starting..."
        MOVE_STATE["error"] = ""
        MOVE_STATE["done"], MOVE_STATE["total"] = 0, 0
        MOVE_STATE["cancel"] = False
        MOVE_STATE["cancelled"] = False
        MOVE_STATE["phase"] = "copy"
        self._src = src
        # For the slow-move case: preferences open automatically once
        # the move has run for a moment, so the progress bar is in
        # front of the user - but an instant same-drive rename never
        # shuffles windows at all.
        self._t0 = time.monotonic()
        self._prefs_opened = False
        self._thread = threading.Thread(
            target=_run_deps_move, args=(src, dst, MOVE_STATE),
            daemon=True)
        self._thread.start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.25, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        _tag_redraw('PREFERENCES')
        if MOVE_STATE["running"]:
            if (not self._prefs_opened
                    and time.monotonic() - self._t0 > 0.8):
                self._prefs_opened = True
                try:
                    bpy.ops.blendcap.install_open_prefs()
                except Exception:
                    pass
            return {'PASS_THROUGH'}
        context.window_manager.event_timer_remove(self._timer)
        _invalidate_deps_cache()
        _tag_redraw('PREFERENCES')
        if MOVE_STATE["cancelled"]:
            self.report({'INFO'},
                        "Move cancelled - your files are still at "
                        + getattr(self, "_src", "their previous location")
                        + ".")
            return {'CANCELLED'}
        if MOVE_STATE["error"]:
            self.report({'ERROR'},
                        "Move failed: " + MOVE_STATE["error"]
                        + " - nothing was deleted; the files still "
                          "work from where they are.")
            return {'CANCELLED'}
        msg = "Dependency files moved."
        swept = getattr(self, "_swept", 0)
        if swept:
            msg += (f" Your custom presets were moved to safety too "
                    f"({swept} file(s)).")
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class BLENDCAP_OT_cancel_move_deps(bpy.types.Operator):
    bl_idname = "blendcap.cancel_move_deps"
    bl_label = "Cancel Move"
    bl_description = ("Stop the move and put everything back where it "
                      "was. Available while files are still being "
                      "copied; the brief final cleanup can't be "
                      "interrupted")
    bl_options = {'REGISTER', 'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return (MOVE_STATE["running"] and MOVE_STATE["phase"] == "copy"
                and not MOVE_STATE["cancel"])

    def execute(self, context):
        MOVE_STATE["cancel"] = True
        MOVE_STATE["msg"] = "Cancelling..."
        _tag_redraw('PREFERENCES')
        return {'FINISHED'}
