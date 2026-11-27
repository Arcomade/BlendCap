"""Update check and one-click update.

How it works
------------
* At most once a day, shortly after startup, a background thread
  fetches a small JSON file (latest.json on the public repo, served
  raw by GitHub). Five-second timeout; any failure is one console line
  and nothing else. Offline machines behave exactly as before.
* If the file names a newer version than the running one, a small
  window appears after the user's first click of the session (same
  pattern as the storage consent window, which has priority: if that
  one is pending, the update notice waits for the next launch). It
  offers Update now / Remind me tomorrow / Skip this version. The
  Updates box in preferences offers the same three choices; if that
  box is drawn while the notice is still pending, the box IS the
  notice and no window opens on top of it.
* Update now downloads the public zip (checksum verified) and installs
  it with Blender's own Install from Disk operator over the existing
  copy - the exact path the storage batch was live-tested on, so
  dependencies, presets, settings and a relocated installer all
  survive. If Blender refuses (portable-build rename failure), the
  zip is merged over the addon folder file by file instead, and the
  user is asked to restart.
* Never active on development builds (manifest version with a
  pre-release suffix). A testing override - the environment variable
  BLENDCAP_UPDATE_INDEX_URL - enables it on a dev build and points it
  at a local index; release builds ignore the variable.
* Nothing here runs unless Blender's own online access is allowed and
  the "Check for updates" preference is on.

latest.json shape (written by the release script):
    {"version": "1.1.0",
     "url": "https://github.com/.../releases/download/v1.1.0/BlendCap-1.1.0.zip",
     "sha256": "...",
     "notes": "https://github.com/.../releases/tag/v1.1.0",
     "required_installer": "1.1.0"}
"""
import bpy
import os
import sys
import json
import time
import hashlib
import shutil
import tempfile
import threading
import zipfile
import urllib.request

from .helpers import (addon_prefs, _addon_dir, _root_package, parse_version,
                      installer_files_present, installer_version,
                      pending_user_data_moves)

UPDATE_INDEX_URL = ("https://raw.githubusercontent.com/Arcomade/BlendCap/"
                    "main/latest.json")
INDEX_URL_ENV = "BLENDCAP_UPDATE_INDEX_URL"
STORE_URL = "https://superhivemarket.com/products/blendcap"
CHECK_INTERVAL = 24 * 3600.0
FETCH_TIMEOUT = 5.0
NOTICE_GIVE_UP_SECONDS = 120.0

# ---------------------------------------------------------------------------
# Version of the running addon
# ---------------------------------------------------------------------------

_MANIFEST = {'version': None}


def current_version():
    """Version string from blender_manifest.toml (read once)."""
    if _MANIFEST['version'] is None:
        ver = ""
        try:
            import tomllib
            with open(os.path.join(_addon_dir(), "blender_manifest.toml"),
                      "rb") as f:
                ver = str(tomllib.load(f).get("version", "") or "")
        except Exception:
            ver = ""
        _MANIFEST['version'] = ver
    return _MANIFEST['version']


def is_dev_build():
    """True for any pre-release manifest version ('1.1.0-dev'). The
    release script strips the suffix, so shipped builds are never dev."""
    v = current_version()
    return (not v) or ('-' in v) or ('.dev' in v)


def index_url():
    """Where to look, or None when the updater must stay off."""
    if is_dev_build():
        return os.environ.get(INDEX_URL_ENV) or None
    return UPDATE_INDEX_URL


def pref_time(prefs, name):
    """Timestamp stored as text in preferences (0.0 when unset)."""
    try:
        return float(getattr(prefs, name, "") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def set_pref_time(prefs, name, value):
    try:
        setattr(prefs, name, repr(float(value)) if value else "")
    except Exception:
        pass


def managed_repo():
    """Name of the REMOTE repository this copy is installed in (a store
    or studio repository added under Get Extensions), or '' for a
    normal local install. Blender updates remote-repository packages
    itself, and that channel delivers the store's own zip (installer
    included), so this updater stands down there: one update channel
    per install."""
    try:
        module = _repo_module()
        for repo in bpy.context.preferences.extensions.repos:
            if repo.module == module:
                if getattr(repo, "use_remote_url", False):
                    return repo.name or module
                return ""
    except Exception:
        pass
    return ""


def updater_enabled():
    prefs = addon_prefs()
    if prefs is None or not getattr(prefs, "check_for_updates", True):
        return False
    if not index_url():
        return False
    if managed_repo():
        return False
    try:
        if not bpy.app.online_access:
            return False
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

_STATE = {
    'checking': False,
    'pending': None,      # result dict or {'error': str}, set by the thread
    'latest': None,       # last good result dict (this session)
    'shutdown': False,
}


def _fetch_index(url):
    req = urllib.request.Request(url, headers={"User-Agent": "BlendCap-updater"})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as r:
        data = r.read(64 * 1024)
    obj = json.loads(data.decode("utf-8"))
    if not isinstance(obj, dict) or not obj.get("version"):
        raise ValueError("index has no version")
    return {
        'version': str(obj.get("version", "")).strip(),
        'url': str(obj.get("url", "") or ""),
        'sha256': str(obj.get("sha256", "") or "").lower(),
        'notes': str(obj.get("notes", "") or ""),
        'required_installer': str(obj.get("required_installer", "") or ""),
        'deps_revision': _as_int(obj.get("deps_revision", 0)),
    }


def _as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _check_thread(url):
    try:
        _STATE['pending'] = _fetch_index(url)
    except Exception as e:
        _STATE['pending'] = {'error': f"{type(e).__name__}: {e}"}
    finally:
        _STATE['checking'] = False


def start_check(force=False):
    """Kick off a background fetch. Returns True if one was started."""
    if _STATE['checking']:
        return True
    url = index_url()
    if url is None:
        return False
    if not force and not updater_enabled():
        return False
    prefs = addon_prefs()
    # Dev builds pointed at a test index check every launch, so a bumped
    # fake release shows up without waiting a day or clearing prefs.
    if not force and not is_dev_build() and _throttled(prefs, url):
        return False
    _STATE['checking'] = True
    _STATE['pending'] = None
    threading.Thread(target=_check_thread, args=(url,), daemon=True).start()
    try:
        bpy.app.timers.register(_poll_check, first_interval=0.25,
                                persistent=True)
    except Exception:
        pass
    return True


def _throttled(prefs, url):
    """True while the last check of THIS index is less than a day old."""
    return (prefs is not None
            and getattr(prefs, "update_latest_source", "") == url
            and time.time() - pref_time(prefs, "update_last_check")
            < CHECK_INTERVAL)


def _poll_check():
    """Main-thread side of the fetch: consume the thread's result."""
    if _STATE['shutdown']:
        return None
    if _STATE['checking']:
        return 0.25
    res = _STATE['pending']
    _STATE['pending'] = None
    if res is None:
        return None
    prefs = addon_prefs()
    if prefs is not None:
        set_pref_time(prefs, "update_last_check", time.time())
    if 'error' in res:
        print(f"[BlendCap] update check skipped: {res['error']}")
        return None
    _STATE['latest'] = res
    if prefs is not None:
        try:
            prefs.update_latest_version = res['version']
            prefs.update_latest_url = res['url']
            prefs.update_latest_sha256 = res['sha256']
            prefs.update_latest_notes = res['notes']
            prefs.update_latest_installer = res['required_installer']
            prefs.update_latest_deps_revision = res.get('deps_revision', 0)
            prefs.update_latest_source = index_url() or ""
        except Exception:
            pass
    if update_available():
        print(f"[BlendCap] update available: {res['version']} "
              f"(running {current_version()}).")
        _maybe_show_notice()
    else:
        print(f"[BlendCap] update check: {current_version()} is current.")
    try:
        from .helpers import _tag_redraw
        # preferences box + the sidebar strip (both panel hosts)
        _tag_redraw('PREFERENCES', 'VIEW_3D', 'SEQUENCE_EDITOR')
    except Exception:
        pass
    return None


def known_latest():
    """The newest version we know about: this session's fetch, else
    what preferences remembered from an earlier one - but only if that
    came from the index this launch is pointed at (a result from a
    local test index must not surface on a normal launch)."""
    if _STATE['latest']:
        return _STATE['latest']
    prefs = addon_prefs()
    if prefs is None:
        return None
    v = getattr(prefs, "update_latest_version", "") or ""
    if not v:
        return None
    url = index_url()
    if not url or getattr(prefs, "update_latest_source", "") != url:
        return None
    return {
        'version': v,
        'url': getattr(prefs, "update_latest_url", "") or "",
        'sha256': (getattr(prefs, "update_latest_sha256", "") or "").lower(),
        'notes': getattr(prefs, "update_latest_notes", "") or "",
        'required_installer': getattr(prefs, "update_latest_installer", "") or "",
        'deps_revision': _as_int(getattr(prefs, "update_latest_deps_revision", 0)),
    }


def update_available():
    latest = known_latest()
    if not latest or not latest.get('url'):
        return False
    return parse_version(latest['version']) > parse_version(current_version())


def installer_update_needed(latest=None):
    """True when the paid installer is older than the newest release
    needs. Public/source copies have no installer and return False."""
    latest = latest or known_latest()
    if not latest or not installer_files_present():
        return False
    req = latest.get('required_installer') or ""
    cur = installer_version()
    if not req or not cur:
        return False
    return parse_version(cur) < parse_version(req)


# ---------------------------------------------------------------------------
# Stale extension-repository lock
# ---------------------------------------------------------------------------
# Blender's extension installer takes a file lock on the repository
# (<repo dir>/.blender_ext/index.lock) and removes it when done. An
# install that crashes mid-way (Smart App Control blocking its ctypes
# DLL, 2026-10-20) leaves the file behind; every later install, ours
# included, then fails with Blender's "lock exists" message, and
# Blender's own fix is the Force Unlock button, which deletes the file.
# We check for the file at a few explicit points (startup, before
# Update Now, after an install attempt, the Unlock button) and keep the
# answer; nothing is checked at draw time.
_LOCK = {'stale': False}


def repo_lock_path():
    """The lock file of the repository this copy is installed in, or ''."""
    try:
        module = _repo_module()
        for repo in bpy.context.preferences.extensions.repos:
            if repo.module == module:
                d = repo.directory
                return (os.path.join(d, ".blender_ext", "index.lock")
                        if d else "")
    except Exception:
        pass
    return ""


def refresh_repo_lock():
    """Re-check for a leftover lock file. Any lock present while no
    install of ours is running is stale: Blender holds it only for the
    duration of an install, and ours run through this module."""
    p = repo_lock_path()
    _LOCK['stale'] = bool(p) and os.path.isfile(p)
    return _LOCK['stale']


def repo_lock_stale():
    return _LOCK['stale']


def update_block_reason(latest=None):
    """Why Update Now must not run right now, or '' when it may.

    'legacy'    - the dependency files (venv/models) are still INSIDE
                  the addon folder. Blender's install replaces that
                  folder wholesale, so the update would delete them.
                  The startup window / Move Dependency Files fixes it.
    'presets'   - no venv in the addon folder, but custom presets are
                  still in there (saved by 1.0.5 or older, and the user
                  put the startup window off or switched it off). The
                  same wholesale replace would delete them, and they
                  can't be downloaded again. Move Presets to Safety in
                  File & Cache Locations fixes it. (With a venv in there
                  too the reason is 'legacy': that move takes the
                  presets along.)
    'installer' - the release needs a newer dependency installer than
                  this paid copy has: the update would install code
                  the stored installer can't build an environment for.
                  Route: store package (Install from Disk replaces the
                  installer), then Reinstall Dependencies. Never for
                  source copies (no installer to be behind).
    'repo'      - this copy lives in a remote repository (a store or
                  studio repository under Get Extensions). Blender
                  updates it, and that channel carries the store's own
                  zip, installer included; this updater stands down.
    Checked repo first (nothing else applies there), then legacy and
    presets (the destructive ones), then installer."""
    if managed_repo():
        return 'repo'
    # Windows Smart App Control blocks the ctypes DLL Blender's own
    # extension installer loads, so the install step would crash
    # mid-way and leave the repository locked (seen 2026-10-20).
    try:
        from .helpers import sac_blocks_this_process
        if sac_blocks_this_process():
            return 'sac'
    except Exception:
        pass
    if repo_lock_stale():
        return 'lock'
    try:
        pend = pending_user_data_moves()
        if pend.get("deps"):
            return 'legacy'
        if pend.get("presets"):
            return 'presets'
    except Exception:
        pass
    if installer_update_needed(latest):
        return 'installer'
    return ''


def rebuild_after_update(latest=None):
    """True when the known release was built against a newer dependency
    revision than the installed environment carries, so the user can be
    told BEFORE updating that a rebuild follows. Compared against the
    stamp in the venv (what is actually installed), falling back to
    this code's own revision when there is no stamp. False with nothing
    installed, or when the index carries no revision."""
    from .helpers import (is_dependencies_installed, read_deps_stamp,
                          DEPS_REVISION)
    latest = latest or known_latest()
    if not latest:
        return False
    new_rev = _as_int(latest.get('deps_revision', 0))
    if new_rev <= 0 or not is_dependencies_installed():
        return False
    have = read_deps_stamp()
    if have is None:
        have = DEPS_REVISION
    return new_rev > have


def update_blocked(latest=None):
    return bool(update_block_reason(latest))


def update_block_short(reason):
    """One short line for the greyed button's tooltip (poll message)."""
    if reason == 'repo':
        return (f"This copy is updated through the '{managed_repo()}' "
                f"repository: use Get Extensions")
    if reason == 'lock':
        return ("Blender's extensions are locked by an install that didn't "
                "finish: click Unlock Extensions in Preferences > Updates")
    if reason == 'sac':
        return ("Windows Smart App Control is blocking BlendCap: turn it "
                "off first (see Preferences > Dependencies)")
    if reason == 'legacy':
        return ("Your dependency files are still inside the addon folder: "
                "move them first (Preferences > Dependencies > Move "
                "Dependency Files)")
    if reason == 'presets':
        return ("Some of your custom presets are still inside the addon "
                "folder: move them first (Preferences > File & Cache "
                "Locations > Move Presets to Safety)")
    if reason == 'installer':
        return ("This release needs a newer dependency installer: install "
                "the new package from your store account first")
    return ""


def update_block_message(reason):
    """One plain sentence for the preferences box / operator report."""
    if reason == 'repo':
        return (f"This copy is updated through the '{managed_repo()}' "
                f"repository. Use Get Extensions to update it.")
    if reason == 'lock':
        return ("Blender's extension system is locked from an addon "
                "install that didn't finish, so Update Now can't run. "
                "Click Unlock Extensions to clear it. Only do this if no "
                "other Blender window is installing an addon right now.")
    if reason == 'sac':
        return ("BlendCap can't update while Smart App Control is on: "
                "Windows blocks a file Blender's extension installer "
                "needs. You will need to turn it off in order to update: "
                "Go to Windows Security > App & browser control > Smart "
                "App Control settings > Off. Then click Update Now again. "
                "Nothing was changed.")
    if reason == 'legacy':
        return ("Update Now is disabled: your dependency files are still "
                "inside the addon folder and an update would delete them. "
                "Move them out first (Dependencies > Move Dependency Files).")
    if reason == 'presets':
        return ("Update Now is disabled: some of your custom presets are "
                "still inside the addon folder and an update would delete "
                "them. Move them out first (File & Cache Locations > Move "
                "Presets to Safety).")
    if reason == 'installer':
        return ("This release needs a newer dependency installer than this "
                "copy has, so Update Now is disabled. Download the new "
                "BlendCap package from your store account and install it "
                "over this one (Install from Disk); that replaces the "
                "installer. Then click Reinstall Dependencies.")
    return ""


def update_block_lines(reason):
    """The same message pre-broken for the fixed-width popup."""
    if reason == 'repo':
        return [f"This copy is updated through the '{managed_repo()}' "
                f"repository.",
                "Use Get Extensions to update it."]
    if reason == 'lock':
        return ["Blender's extension system is locked from an addon "
                "install that didn't finish,",
                "so Update Now can't run. Clear it with Unlock Extensions "
                "in Preferences >",
                "BlendCap > Updates (only if no other Blender window is "
                "installing an addon)."]
    if reason == 'sac':
        return ["BlendCap can't update while Smart App Control is on: "
                "Windows blocks a file",
                "Blender's extension installer needs. You will need to "
                "turn it off in order",
                "to update: Go to Windows Security > App & browser "
                "control > Smart App Control",
                "settings > Off. Then click Update Now again. Nothing "
                "was changed."]
    if reason == 'legacy':
        return ["Your dependency files are still inside the addon folder, "
                "and an update",
                "would delete them. Move them out first: Preferences > "
                "BlendCap >",
                "Dependencies > Move Dependency Files. Then update."]
    if reason == 'presets':
        return ["Some of your custom presets are still inside the addon "
                "folder, and an",
                "update would delete them. Move them out first: "
                "Preferences > BlendCap >",
                "File & Cache Locations > Move Presets to Safety. Then "
                "update."]
    if reason == 'installer':
        return ["This release needs a newer dependency installer than this "
                "copy has.",
                "Download the new package from your store account, install "
                "it over this one",
                "(Install from Disk), then click Reinstall Dependencies in "
                "Preferences."]
    return []


# ---------------------------------------------------------------------------
# Notice window, shown after the first click
# ---------------------------------------------------------------------------

_WATCH = {'fired': False, 'shutdown': False, 'armed_at': 0.0,
          'claimed': False}


def notice_state():
    """'SKIPPED' when this version was skipped, 'SNOOZED' while a
    remind-me-tomorrow is running, else ''."""
    prefs = addon_prefs()
    latest = known_latest()
    if prefs is None or not latest:
        return ''
    if getattr(prefs, "update_skip_version", "") == latest['version']:
        return 'SKIPPED'
    if time.time() < pref_time(prefs, "update_remind_after"):
        return 'SNOOZED'
    return ''


def _notice_suppressed():
    """Skip this version, or remind-later still running."""
    if addon_prefs() is None or not known_latest():
        return True
    return bool(notice_state())


def strip_wanted(prefs=None):
    """Should the sidebar strip draw: preference on, a newer version
    known, and neither Remind nor Skip in effect."""
    prefs = prefs if prefs is not None else addon_prefs()
    if prefs is None or not getattr(prefs, "show_update_notice_in_sidebar",
                                    True):
        return False
    if not getattr(prefs, "check_for_updates", True):
        return False   # checks turned off: no unasked-for notice
    if managed_repo():
        return False
    return update_available() and not notice_state()


def notice_shown_in_prefs():
    """Called by the Updates box in preferences whenever it draws an
    available update. The user is looking at the same information and
    the same choices, so a pending popup is dropped: the box is the
    notice for this launch."""
    if not _WATCH['claimed']:
        _WATCH['claimed'] = True
        _WATCH['fired'] = True


def _helper_process_alive():
    """A capture / BVH helper process that is still running, even
    after its job was reported finished or cancelled. Such a process
    holds the addon folder (its scripts live there), which is exactly
    what makes Blender's folder replacement fail on Windows."""
    try:
        from .helpers import blendcap_state, bvh_state, arkit_state
        for st in (blendcap_state, bvh_state, arkit_state):
            proc = st.get('active_process')
            if proc is not None and proc.poll() is None:
                return True
    except Exception:
        pass
    return False


def _jobs_running():
    try:
        from .helpers import (blendcap_state, bvh_state,
                              retarget_progress_state, arkit_state)
        if (blendcap_state.get('running') or bvh_state.get('running')
                or retarget_progress_state.get('running')
                or arkit_state.get('running')):   # face-data import
            return True
    except Exception:
        return False
    return _helper_process_alive()


def _storage_consent_pending():
    try:
        prefs = addon_prefs()
        if prefs is not None and getattr(prefs, "hide_storage_warning", False):
            return False
        p = pending_user_data_moves()
        return bool(p.get("presets") or p.get("deps"))
    except Exception:
        return False


def _maybe_show_notice():
    if bpy.app.background or _WATCH['fired']:
        return
    # The one gate every path goes through (startup timer, a fetch
    # result, the load_post re-arm): "Automatically check for updates"
    # off, online access off, dev build, repository install = no
    # window, even for a result remembered from an earlier launch.
    # (The load_post path used to skip this: popup with the box
    # unticked, seen live 2026-11-05.)
    if not updater_enabled():
        return
    if repo_lock_stale():
        # The popup exists to offer Update now, which can't run; the
        # sidebar strip and the preferences note say so WITH the Unlock
        # button, which a dialog can't carry. Next launch is normal.
        print("[BlendCap] update notice skipped: the extension system is "
              "locked (see the BlendCap panel).")
        return
    if not update_available() or _notice_suppressed():
        return
    if _storage_consent_pending():
        print("[BlendCap] update notice deferred: the storage notice "
              "comes first this launch.")
        return
    try:
        wm = bpy.context.window_manager
        if not wm or not wm.windows:
            return
        _WATCH['armed_at'] = time.time()
        with bpy.context.temp_override(window=wm.windows[0]):
            bpy.ops.blendcap.update_watcher('INVOKE_DEFAULT')
    except Exception as e:
        print(f"[BlendCap] update notice not shown: {e}")


def _open_notice():
    if _WATCH['shutdown'] or _WATCH['claimed'] or bpy.app.background:
        return None
    if _jobs_running() or not update_available() or _notice_suppressed():
        return None
    try:
        wm = bpy.context.window_manager
        if wm and wm.windows:
            with bpy.context.temp_override(window=wm.windows[0]):
                bpy.ops.blendcap.update_notice('INVOKE_DEFAULT')
    except Exception as e:
        print(f"[BlendCap] update notice not shown: {e}")
    return None


class BLENDCAP_OT_update_watcher(bpy.types.Operator):
    """Invisible pass-through modal: opens the update notice 0.2 s
    after the user's first click (never inside the press itself - the
    release would land on the fresh dialog and cancel it). Consumes
    only its own TIMER ticks. Gives up quietly after a couple of
    minutes; the notice then waits for the next launch."""
    bl_idname = "blendcap.update_watcher"
    bl_label = "BlendCap Update Watcher"
    bl_options = {'INTERNAL'}

    def invoke(self, context, event):
        self._timer = context.window_manager.event_timer_add(
            0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _stop(self, context):
        try:
            context.window_manager.event_timer_remove(self._timer)
        except Exception:
            pass
        return {'FINISHED'}

    def modal(self, context, event):
        if _WATCH['shutdown'] or _WATCH['fired']:
            if event.type == 'TIMER':
                return self._stop(context)
            return {'PASS_THROUGH'}
        if (event.type == 'TIMER'
                and time.time() - _WATCH['armed_at'] > NOTICE_GIVE_UP_SECONDS):
            return self._stop(context)
        if event.value == 'PRESS' and not _jobs_running():
            _WATCH['fired'] = True
            try:
                bpy.app.timers.register(_open_notice, first_interval=0.2)
            except Exception:
                pass
        return {'PASS_THROUGH'}


_NOTICE_ITEMS = [
    ('UPDATE', "Update now", "Download and install the new version, "
                             "then restart Blender"),
    ('LATER', "Remind me tomorrow", "Ask again on the next launch after "
                                    "24 hours"),
    ('SKIP', "Skip this version", "Don't mention this version again; the "
                                  "next release will"),
]


class BLENDCAP_OT_update_notice(bpy.types.Operator):
    """The update window."""
    bl_idname = "blendcap.update_notice"
    bl_label = "BlendCap Update"
    bl_options = {'INTERNAL'}

    # The property needs a real name: with name="" the expanded radio
    # buttons draw with no labels at all (Blender 5.2, seen live
    # 2026-09-13). The name itself is never drawn.
    action: bpy.props.EnumProperty(  # type: ignore
        name="Action", items=_NOTICE_ITEMS, default='UPDATE')

    def invoke(self, context, event):
        self.action = 'LATER' if update_blocked() else 'UPDATE'
        return context.window_manager.invoke_props_dialog(
            self, width=440, confirm_text="Continue")

    def draw(self, context):
        latest = known_latest() or {}
        col = self.layout.column(align=True)
        col.label(text=f"BlendCap {latest.get('version', '?')} is available. "
                       f"You have {current_version()}.", icon='INFO')
        if latest.get('notes'):
            # Opens the release's changelog in the browser; the popup
            # stays open (operator buttons don't close a props dialog).
            row = col.row()
            row.alignment = 'LEFT'
            row.operator("blendcap.update_whats_new", icon='URL')
        col.separator()
        reason = update_block_reason(latest)
        if reason:
            # No "Update now" while the update would destroy the
            # dependencies (legacy layout) or install code the stored
            # installer can't support. The lines say what to do.
            warn = col.column(align=True)
            warn.alert = True
            for i, line in enumerate(update_block_lines(reason)):
                warn.label(text=line, icon='ERROR' if i == 0 else 'NONE')
            col.separator()
            col.prop_enum(self, "action", 'LATER')
            col.prop_enum(self, "action", 'SKIP')
        else:
            if rebuild_after_update(latest):
                col.label(text="This update needs a dependency rebuild "
                               "afterwards.", icon='ERROR')
                col.separator()
            col.prop(self, "action", expand=True)
        col.separator()
        col.label(text="You can turn these checks off any time in "
                       "Preferences > BlendCap > Updates.")

    def execute(self, context):
        prefs = addon_prefs()
        latest = known_latest() or {}
        if self.action == 'UPDATE':
            reason = update_block_reason(latest)
            if reason:
                self.report({'ERROR'}, update_block_message(reason))
                return {'CANCELLED'}
            try:
                bpy.ops.blendcap.update_now('INVOKE_DEFAULT')
            except Exception as e:
                self.report({'ERROR'}, f"Couldn't start the update: {e}")
                return {'CANCELLED'}
        elif self.action == 'LATER' and prefs is not None:
            set_pref_time(prefs, "update_remind_after",
                          time.time() + CHECK_INTERVAL)
        elif self.action == 'SKIP' and prefs is not None:
            prefs.update_skip_version = latest.get('version', '')
        return {'FINISHED'}


class BLENDCAP_OT_update_whats_new(bpy.types.Operator):
    """Open the known release's changelog (the `notes` URL from the
    index) in the browser. Used by the update popup and the Updates box;
    the generic wm.url_open can't carry our tooltip."""
    bl_idname = "blendcap.update_whats_new"
    bl_label = "What's new"
    bl_description = ("Open the changelog for this release on GitHub in "
                      "your browser. Needs an internet connection")
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return bool((known_latest() or {}).get('notes'))

    def execute(self, context):
        url = (known_latest() or {}).get('notes', '')
        if not url:
            self.report({'WARNING'}, "This release has no changelog link.")
            return {'CANCELLED'}
        try:
            bpy.ops.wm.url_open(url=url)
        except Exception as e:
            self.report({'ERROR'}, f"Couldn't open the browser: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}


_DISMISS_ITEMS = [
    ('LATER', "Remind Me Tomorrow", "Don't show the update window again "
                                    "for 24 hours"),
    ('SKIP', "Skip This Version", "Don't show the update window for this "
                                  "version; the next release will"),
]


class BLENDCAP_OT_update_dismiss(bpy.types.Operator):
    """Preferences box: the notice window's other two choices."""
    bl_idname = "blendcap.update_dismiss"
    bl_label = "Update Notice"
    bl_options = {'INTERNAL'}

    action: bpy.props.EnumProperty(  # type: ignore
        name="Action", items=_DISMISS_ITEMS, default='LATER')

    @classmethod
    def description(cls, context, props):
        for ident, _name, desc in _DISMISS_ITEMS:
            if ident == props.action:
                return desc
        return ""

    def execute(self, context):
        prefs = addon_prefs()
        latest = known_latest() or {}
        if prefs is None:
            return {'CANCELLED'}
        if self.action == 'LATER':
            set_pref_time(prefs, "update_remind_after",
                          time.time() + CHECK_INTERVAL)
        else:
            prefs.update_skip_version = latest.get('version', '')
        _WATCH['fired'] = True
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Download + install
# ---------------------------------------------------------------------------

_DL = {'running': False, 'done': 0, 'total': 0, 'path': '', 'error': '',
       'cancel': False, 'version': ''}


def _download_thread(url, path, expected_sha):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BlendCap-updater"})
        h = hashlib.sha256()
        with urllib.request.urlopen(req, timeout=30) as r, open(path, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            _DL['total'] = total
            while True:
                if _DL['cancel']:
                    raise RuntimeError("cancelled")
                chunk = r.read(256 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                h.update(chunk)
                _DL['done'] += len(chunk)
        if expected_sha and h.hexdigest().lower() != expected_sha:
            raise RuntimeError("the downloaded file's checksum doesn't match")
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:
                raise RuntimeError("the downloaded zip is damaged")
    except Exception as e:
        _DL['error'] = str(e)
        try:
            os.remove(path)
        except OSError:
            pass
    finally:
        _DL['running'] = False


def _repo_module():
    """'user_default' from 'bl_ext.user_default.blendcap'."""
    parts = (_root_package() or "").split(".")
    return parts[1] if len(parts) >= 3 else "user_default"


def _clean_stale_temp():
    """Blender renames the old addon folder to '.<id>.~temp~' while
    installing; a leftover from a failed attempt blocks the next one
    (portable-build report 2026-08-29)."""
    try:
        ext_dir = bpy.utils.user_resource('EXTENSIONS')
        d = os.path.join(ext_dir, _repo_module(), ".blendcap.~temp~")
        if os.path.isdir(d):
            from .operators_pipeline import _rm_readonly
            shutil.rmtree(d, onerror=_rm_readonly)
            print(f"[BlendCap] removed stale {d}")
    except Exception as e:
        print(f"[BlendCap] stale temp cleanup skipped: {e}")


def _try_install(path):
    wm = bpy.context.window_manager
    win = wm.windows[0] if wm and len(wm.windows) else None
    kwargs = dict(filepath=path, enable_on_install=True, overwrite=True)
    for attempt in (dict(repo=_repo_module()), {}):
        try:
            if win is not None:
                with bpy.context.temp_override(window=win):
                    res = bpy.ops.extensions.package_install_files(
                        **kwargs, **attempt)
            else:
                res = bpy.ops.extensions.package_install_files(
                    **kwargs, **attempt)
            if 'FINISHED' in res:
                return True
            print(f"[BlendCap] package_install_files returned {res}")
        except TypeError as e:
            # unknown repo identifier -> retry letting Blender pick
            print(f"[BlendCap] install retry without repo: {e}")
            continue
        except Exception as e:
            print(f"[BlendCap] package_install_files failed: {e}")
            return False
    return False


# Never touched by the in-place sync below: user data and anything
# that isn't ours to delete. Top-level folder names and file names,
# matched case-insensitively against the first path component.
_SYNC_PROTECTED = frozenset({
    "venv", "checkpoints", "installer", "_offline_stash", ".git",
    "retarget_maps", "rest_pose_presets",
    "install_marker.json", "install.log",
})


def _sync_protected(rel):
    head = rel.replace("\\", "/").split("/", 1)[0].lower()
    return head in _SYNC_PROTECTED


def _merge_install(path):
    """Fallback when Blender can't replace the addon folder (on
    Windows the rename fails while anything has the folder open - an
    Explorer window, a terminal, a helper process): unpack the zip
    over the folder file by file, then delete the files the previous
    version left behind that the new one no longer ships, so the
    result matches a clean install. Deletion is limited to what was
    already there AND isn't protected (_SYNC_PROTECTED: dependencies,
    installer, presets, markers - none of which should be in the addon
    folder any more, but a legacy layout must survive this). Needs a
    Blender restart."""
    tmp = tempfile.mkdtemp(prefix="blendcap_update_")
    try:
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        src_root = None
        for root, _dirs, files in os.walk(tmp):
            if "blender_manifest.toml" in files:
                src_root = root
                break
        if src_root is None:
            raise RuntimeError("no blender_manifest.toml in the zip")
        dst_root = _addon_dir()

        # What is there now (relative paths), before anything changes.
        before = set()
        for root, dirs, files in os.walk(dst_root):
            rel_dir = os.path.relpath(root, dst_root)
            if rel_dir != "." and _sync_protected(rel_dir):
                dirs[:] = []
                continue
            for fn in files:
                rel = fn if rel_dir == "." else os.path.join(rel_dir, fn)
                if not _sync_protected(rel):
                    before.add(os.path.normcase(rel))

        shipped = set()
        for root, _dirs, files in os.walk(src_root):
            rel_dir = os.path.relpath(root, src_root)
            out_dir = dst_root if rel_dir == "." else os.path.join(dst_root, rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            for fn in files:
                rel = fn if rel_dir == "." else os.path.join(rel_dir, fn)
                shipped.add(os.path.normcase(rel))
                shutil.copy2(os.path.join(root, fn), os.path.join(out_dir, fn))

        # Stale files: were there, not shipped any more, not protected.
        removed = 0
        for rel in sorted(before - shipped):
            try:
                os.remove(os.path.join(dst_root, rel))
                removed += 1
            except OSError as e:
                print(f"[BlendCap] couldn't remove stale {rel}: {e}")
        # Folders emptied by that (deepest first); protected ones never
        # were walked, so they can't be here.
        for root, dirs, files in os.walk(dst_root, topdown=False):
            if root == dst_root or files or dirs:
                continue
            rel_dir = os.path.relpath(root, dst_root)
            if _sync_protected(rel_dir):
                continue
            try:
                os.rmdir(root)
            except OSError:
                pass
        print(f"[BlendCap] files updated in place: {len(shipped)} written, "
              f"{removed} stale removed")
        return True
    except Exception as e:
        print(f"[BlendCap] merge install failed: {e}")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _popup(title, lines, icon='INFO'):
    def draw(self, _context):
        for line in lines:
            self.layout.label(text=line)
    try:
        wm = bpy.context.window_manager
        win = wm.windows[0] if wm and len(wm.windows) else None
        if win is not None:
            with bpy.context.temp_override(window=win):
                wm.popup_menu(draw, title=title, icon=icon)
        else:
            wm.popup_menu(draw, title=title, icon=icon)
    except Exception:
        pass


def _install_downloaded():
    """Timer: runs AFTER the download operator has finished, so the
    install (which replaces this very module) never happens inside one
    of our own operators."""
    path, version = _DL['path'], _DL['version']
    _clean_stale_temp()
    ok = _try_install(path)
    refresh_repo_lock()   # an attempt that died mid-way leaves the lock
    if not ok:
        time.sleep(0.5)
        _clean_stale_temp()
        ok = _try_install(path)
    merged = False
    if not ok:
        merged = _merge_install(path)
        ok = merged
    try:
        os.remove(path)
    except OSError:
        pass
    if not ok:
        print("[BlendCap] update failed; nothing was changed.")
        _popup("BlendCap update failed", [
            "The new version couldn't be installed. Nothing was changed.",
            "Download the zip from the release page and use Install from "
            "Disk."], icon='ERROR')
        return None
    print(f"[BlendCap] updated to {version}"
          + (" (files updated in place; restart Blender)" if merged else ""))
    lines = ["Restart Blender to finish the update."]
    if merged:
        lines = [
            "Blender couldn't replace the addon folder because something "
            "has it open",
            "(an Explorer window or a terminal in that folder, for "
            "example).",
            "The files were updated in place instead.",
            "Restart Blender to finish the update.",
        ]
    _popup(f"BlendCap {version} installed", lines)
    return None


class BLENDCAP_OT_update_now(bpy.types.Operator):
    """Download the newest version and install it over this one."""
    bl_idname = "blendcap.update_now"
    bl_label = "Update BlendCap"
    bl_description = ("Download the newest BlendCap and install it over "
                      "this copy. Dependencies, presets and settings are "
                      "kept. Restart Blender afterwards")
    bl_options = {'INTERNAL'}

    _timer = None

    @classmethod
    def poll(cls, context):
        # A greyed button explains itself: Blender shows the poll
        # message in the tooltip as "Disabled: ...", wherever the
        # operator is drawn (Updates box, sidebar strip).
        reason = update_block_reason()
        if reason:
            cls.poll_message_set(update_block_short(reason))
            return False
        return True

    def invoke(self, context, event):
        if _DL['running']:
            self.report({'WARNING'}, "An update download is already running.")
            return {'CANCELLED'}
        if _jobs_running():
            self.report({'ERROR'}, "Finish or cancel the running job first.")
            return {'CANCELLED'}
        latest = known_latest()
        if not latest or not latest.get('url'):
            self.report({'ERROR'}, "No update download is known. Click "
                                   "Check now in the preferences first.")
            return {'CANCELLED'}
        refresh_repo_lock()
        reason = update_block_reason(latest)
        if reason:
            self.report({'ERROR'}, update_block_message(reason))
            return {'CANCELLED'}
        if _helper_process_alive():
            self.report({'ERROR'}, "A capture helper process from an earlier "
                                   "job is still running and holds the addon "
                                   "folder. Wait for it to finish, or "
                                   "restart Blender, then update.")
            return {'CANCELLED'}
        path = os.path.join(tempfile.gettempdir(),
                            f"BlendCap-{latest['version']}.zip")
        _DL.update({'running': True, 'done': 0, 'total': 0, 'path': path,
                    'error': '', 'cancel': False,
                    'version': latest['version']})
        threading.Thread(target=_download_thread,
                         args=(latest['url'], path, latest.get('sha256', '')),
                         daemon=True).start()
        self._timer = context.window_manager.event_timer_add(
            0.2, window=context.window)
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, f"Downloading BlendCap {latest['version']}...")
        return {'RUNNING_MODAL'}

    def _status(self, context, text):
        try:
            context.workspace.status_text_set(text)
        except Exception:
            pass

    def _finish(self, context):
        try:
            context.window_manager.event_timer_remove(self._timer)
        except Exception:
            pass
        self._status(context, None)

    def modal(self, context, event):
        if event.type == 'ESC':
            _DL['cancel'] = True
            self._finish(context)
            self.report({'WARNING'}, "Update cancelled.")
            return {'CANCELLED'}
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        if _DL['running']:
            done, total = _DL['done'], _DL['total']
            if total:
                self._status(context, f"BlendCap update: downloading "
                                      f"{done // 1024} / {total // 1024} KB "
                                      f"(ESC to cancel)")
            else:
                self._status(context, f"BlendCap update: downloading "
                                      f"{done // 1024} KB (ESC to cancel)")
            return {'PASS_THROUGH'}
        self._finish(context)
        if _DL['error']:
            self.report({'ERROR'}, f"Download failed: {_DL['error']}. "
                                   f"Nothing was changed.")
            return {'CANCELLED'}
        self.report({'INFO'}, "Download complete, installing...")
        try:
            bpy.app.timers.register(_install_downloaded, first_interval=0.1,
                                    persistent=True)
        except Exception as e:
            self.report({'ERROR'}, f"Couldn't start the install: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}


class BLENDCAP_OT_repo_unlock(bpy.types.Operator):
    """The Unlock Extensions button: remove the leftover lock file, the
    same thing Blender's own Force Unlock Repository does."""
    bl_idname = "blendcap.repo_unlock"
    bl_label = "Unlock Extensions"
    bl_description = ("Clear the lock an earlier install left behind so "
                      "updates can run again. Only use this when no other "
                      "Blender window is installing something")
    bl_options = {'INTERNAL'}

    def execute(self, context):
        p = repo_lock_path()
        if p and os.path.isfile(p):
            try:
                os.remove(p)
            except OSError as e:
                self.report({'ERROR'}, f"Couldn't remove the lock: {e}")
                return {'CANCELLED'}
        refresh_repo_lock()
        try:
            from .helpers import _tag_redraw
            _tag_redraw('PREFERENCES', 'VIEW_3D', 'SEQUENCE_EDITOR')
        except Exception:
            pass
        self.report({'INFO'}, "Extension system unlocked.")
        return {'FINISHED'}


class BLENDCAP_OT_update_check_now(bpy.types.Operator):
    """Preferences button: fetch the index right now."""
    bl_idname = "blendcap.update_check_now"
    bl_label = "Check now"
    bl_description = "Look for a newer BlendCap release right now"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        refresh_repo_lock()   # a lock left by a crashed install mid-session
        if index_url() is None:
            self.report({'WARNING'}, "Update checks are off on development "
                                     "builds.")
            return {'CANCELLED'}
        repo_name = managed_repo()
        if repo_name:
            self.report({'WARNING'}, f"This copy is updated through the "
                                     f"'{repo_name}' repository.")
            return {'CANCELLED'}
        try:
            if not bpy.app.online_access:
                self.report({'WARNING'}, "Blender's online access is off "
                                         "(Preferences > System).")
                return {'CANCELLED'}
        except Exception:
            pass
        if start_check(force=True):
            self.report({'INFO'}, "Checking for updates...")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _startup_timer():
    """A couple of seconds after launch: fetch if a day has passed;
    otherwise, if an update is already known, show the notice."""
    if _STATE['shutdown'] or bpy.app.background:
        return None
    try:
        refresh_repo_lock()
        if not updater_enabled():
            return None
        if not start_check():
            if update_available():
                _maybe_show_notice()
    except Exception as e:
        print(f"[BlendCap] update check skipped: {e}")
    return None


@bpy.app.handlers.persistent
def _notice_on_load(_filepath=None, _extra=None):
    """Loading a .blend that carries its own UI replaces the window
    manager and kills the click-watcher modal. If a notice is still
    owed (update known, nothing chosen yet), arm the watcher again."""
    try:
        if _STATE['shutdown'] or _WATCH['fired'] or bpy.app.background:
            return
        if update_available() and not _notice_suppressed():
            _maybe_show_notice()
    except Exception:
        pass


def arm():
    _STATE['shutdown'] = False
    _WATCH['shutdown'] = False
    _WATCH['fired'] = False
    _WATCH['claimed'] = False
    # persistent=True: Blender drops ordinary timers when a file loads,
    # and a .blend opened by double-click loads right after register,
    # which used to cancel the check for that session.
    try:
        bpy.app.timers.register(_startup_timer, first_interval=2.0,
                                persistent=True)
    except Exception:
        pass
    if _notice_on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_notice_on_load)


def disarm():
    _STATE['shutdown'] = True
    _WATCH['shutdown'] = True
    if _notice_on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_notice_on_load)
    for fn in (_startup_timer, _poll_check, _open_notice,
               _install_downloaded):
        try:
            if bpy.app.timers.is_registered(fn):
                bpy.app.timers.unregister(fn)
        except Exception:
            pass


classes = (
    BLENDCAP_OT_update_watcher,
    BLENDCAP_OT_update_notice,
    BLENDCAP_OT_update_dismiss,
    BLENDCAP_OT_update_whats_new,
    BLENDCAP_OT_update_now,
    BLENDCAP_OT_repo_unlock,
    BLENDCAP_OT_update_check_now,
)
