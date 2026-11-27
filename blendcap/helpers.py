"""Shared helpers, constants, and module-level state.

Cross-cutting utilities (path resolution, VSE strip access, NPZ probing,
cache params, warning popup, the two big mutable state dicts) live here
so every operator module can import them without dragging in retarget /
capture code.
"""
import bpy
import os
import sys
import re
import json
import shutil
import subprocess
import tempfile
import contextlib
import warnings
import atexit

# Suppress Blender 5.x ImageStrip deprecation warnings — these properties still
# work but have no replacement API yet. Remove this filter when Blender 6.0
# provides new ImageStrip APIs.
warnings.filterwarnings("ignore", message=r"'ImageStrip\.", category=DeprecationWarning)

# ============================================================
# SHARED HELPERS
# ============================================================
def _flatpak_shared_cache_base():
    """A base folder whose absolute path is IDENTICAL inside the Flatpak
    sandbox and on the host, so Blender (in the sandbox) and the capture
    process (run on the host via flatpak-spawn) read/write the same files.

    Blender's per-session temp (bpy.app.tempdir, e.g. /tmp/blender_XXXX) must
    NOT be used under Flatpak: the sandbox has a PRIVATE /tmp, so a host-spawned
    capture writes its npz/overlay into the HOST's /tmp, which Blender inside
    the sandbox never sees - the monitor then waits forever and the capture
    appears to stall at the final step with no output (KEY BUG E). The app's
    own cache dir (XDG_CACHE_HOME = ~/.var/app/org.blender.Blender/cache) is
    exposed at the same path on both sides - the same place the venv and model
    checkpoints already live - so both sides agree."""
    xdg = (os.environ.get("XDG_CACHE_HOME") or "").strip()
    if xdg and os.path.isabs(xdg):
        return xdg
    return os.path.join(os.path.expanduser("~"),
                        ".var", "app", "org.blender.Blender", "cache")


def _cache_dir():
    """Folder BlendCap writes tracking data into (a "Blendcap_Cache"
    subfolder of the resolved base). Resolution order:
      1. Custom cache folder (add-on pref) - when "Set default cache folder"
         is on AND either the project is unsaved or "Always use this folder"
         is also on.
      2. Saved project: alongside the .blend (persists with the project).
      3. Unsaved project: Blender's per-session temp dir, auto-removed on
         close (so a throwaway test doesn't litter the Desktop). EXCEPTION:
         under Flatpak that temp dir is sandbox-private and host-spawned
         capture output would be invisible to Blender, so a host-shared
         base is used instead (see _flatpak_shared_cache_base / KEY BUG E).
    A "Blendcap_Cache" subfolder is always appended (even to a custom base),
    so the destructive Clear All only ever operates inside a folder BlendCap
    owns - never loose in whatever folder the user pointed at."""
    prefs = addon_prefs()
    custom = ""
    if prefs and getattr(prefs, "cache_use_custom", False):
        custom = (prefs.cache_custom_path or "").strip()
        if custom:
            custom = bpy.path.abspath(custom)
    saved = bool(bpy.data.filepath)
    always = bool(prefs and getattr(prefs, "cache_always_use_custom", False))
    if custom and (always or not saved):
        base = custom
    elif saved:
        base = os.path.dirname(bpy.data.filepath)
    elif _in_flatpak():
        # Unsaved project under Flatpak: bpy.app.tempdir is sandbox-private,
        # so host-spawned capture output would be invisible to Blender. Use a
        # base both sides share (see _flatpak_shared_cache_base / KEY BUG E).
        base = _flatpak_shared_cache_base()
    else:
        base = bpy.app.tempdir or tempfile.gettempdir()
    return os.path.join(base, "Blendcap_Cache")


def dir_tree_signature(dirs, suffix):
    """Cheap change detector for dropdowns that list files on disk.

    Blender calls a dynamic enum's items callback on every redraw of
    the panel that shows it, so anything that walks folders or opens
    files there runs dozens of times per second while the mouse is
    over the panel. Callers compute this signature first and rebuild
    their item list only when it changed. It walks each directory with
    os.scandir and records (path, mtime_ns, size) for files with
    `suffix` plus the subfolder names; on Windows DirEntry.stat() comes
    from the listing itself, so this is about the cost of one listing
    per folder and no file is opened. Missing dirs contribute nothing.
    """
    sig = []
    for root in dirs:
        if not root:
            continue
        stack = [root]
        while stack:
            d = stack.pop()
            try:
                with os.scandir(d) as it:
                    for e in it:
                        try:
                            if e.is_dir(follow_symlinks=False):
                                stack.append(e.path)
                                sig.append((e.path, -1, -1))
                            elif e.name.endswith(suffix):
                                st = e.stat(follow_symlinks=False)
                                sig.append((e.path, st.st_mtime_ns,
                                            st.st_size))
                        except OSError:
                            continue
            except OSError:
                continue
    sig.sort()
    return tuple(sig)


def _bvh_library_dir():
    """Folder holding the BVH library (the exports managed in the BVH Library
    panel). Custom location via the add-on pref (used as-is - the user names
    the library folder directly), else the bvh_exports subfolder of the cache
    dir so it follows the cache.

    NOTE: reads add-on prefs / bpy.context, so callers on a background thread
    (run_bvh_thread) must resolve this on the main thread and pass it in."""
    prefs = addon_prefs()
    if prefs and getattr(prefs, "bvh_library_use_custom", False):
        p = (prefs.bvh_library_path or "").strip()
        if p:
            return bpy.path.abspath(p)
    return os.path.join(_cache_dir(), "bvh_exports")


# ---------------------------------------------------------------------------
# Windows path length (support item 10)
# ---------------------------------------------------------------------------
# Blender's own Python cannot open, see or create anything past the classic
# Windows limits: 259 characters for a file path, 247 for a folder. That holds
# even on a PC with Windows' long-path setting switched on (measured on
# Blender 5.3 with LongPathsEnabled = 1: the capture helper's Python wrote a
# 300-character path happily, Blender's could not even see the file). So a
# project on a deep server path used to capture fine and then have Generate
# BVH do nothing at all, with no message: the BVH was written, or not, and
# Blender simply could not find it. These checks run before a job starts and
# say which path is too long. They are a wall, not a fix: nothing in the
# chain can be made to work past the limit (Blender's BVH importer and its
# video reader have the same limit).
WIN_MAX_FILE_PATH = 259
WIN_MAX_DIR_PATH = 247
# The longest file name BlendCap writes straight into a capture folder.
CAPTURE_LONGEST_FILE = "blendcap_capture_debug.log"


def path_over_limit(path, is_dir=False):
    """The path's length when Blender can't use it on Windows, else 0."""
    if sys.platform != 'win32' or not path:
        return 0
    try:
        full = os.path.abspath(path)
    except Exception:
        full = str(path)
    n = len(full)
    return n if n > (WIN_MAX_DIR_PATH if is_dir else WIN_MAX_FILE_PATH) else 0


def _wrap_path(path, width=60):
    """Split a path over several dialog lines, breaking after a separator
    where possible (dialog labels don't wrap and clip at the edge)."""
    parts = re.split(r'(?<=[\\/])', path)
    lines, cur = [], ""
    for part in parts:
        while len(part) > width:
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(part[:width])
            part = part[width:]
        if cur and len(cur) + len(part) > width:
            lines.append(cur)
            cur = part
        else:
            cur += part
    if cur:
        lines.append(cur)
    return lines


_LONG_PATH_ADVICE = {
    'video': ["Move or rename the video so its path is shorter, then try again."],
    'cache': ["Save the project in a folder with a shorter path, or shorten",
              "the clip's name. A custom cache folder with a short path also",
              "works (Preferences > File & Cache Locations, with Always use",
              "this folder ticked)."],
    'library': ["Use a BVH library folder with a shorter path, or shorten the",
                "clip's name."],
}


def path_too_long_lines(path, length, is_dir=False, kind='cache'):
    """The warning dialog's lines for one path that is over the limit."""
    limit = WIN_MAX_DIR_PATH if is_dir else WIN_MAX_FILE_PATH
    try:
        full = os.path.abspath(path)
    except Exception:
        full = str(path)
    lines = [f"Path too long for Blender on Windows: {length} characters "
             f"(limit {limit})."]
    lines += ["    " + ln for ln in _wrap_path(full)]
    lines += _LONG_PATH_ADVICE.get(kind, _LONG_PATH_ADVICE['cache'])
    return lines


def first_path_too_long(checks):
    """checks: (path, is_dir, kind) tuples. The first one over the limit as
    (path, length, is_dir, kind), or None."""
    for path, is_dir, kind in checks:
        n = path_over_limit(path, is_dir)
        if n:
            return path, n, is_dir, kind
    return None


def block_if_path_too_long(checks):
    """Show the warning dialog for the first path over the limit and
    return True (the caller cancels its job); False when all are fine."""
    hit = first_path_too_long(checks)
    if hit is None:
        return False
    path, n, is_dir, kind = hit
    print(f"[BlendCap] path too long for Blender on Windows "
          f"({n} characters): {path}")
    _show_warning_popup(path_too_long_lines(path, n, is_dir, kind))
    return True


def capture_folder_checks(output_dir):
    """The checks for a capture folder: the folder itself, and the longest
    file BlendCap writes into it."""
    return [(output_dir, True, 'cache'),
            (os.path.join(output_dir, CAPTURE_LONGEST_FILE), False, 'cache')]


def bvh_job_checks(jobs, library_dir):
    """The checks for a list of BVH jobs (cmd, intermediate path, stem):
    each intermediate file and each final file in the library folder."""
    custom = (os.path.normcase(os.path.abspath(library_dir))
              != os.path.normcase(os.path.abspath(
                  os.path.join(_cache_dir(), "bvh_exports"))))
    kind = 'library' if custom else 'cache'
    checks = [(library_dir, True, kind)]
    for _cmd, intermediate, stem in jobs:
        checks.append((intermediate, False, 'cache'))
        checks.append((os.path.join(library_dir, f"{stem}.bvh"), False, kind))
    return checks


def _temp_cache_dir():
    """The Blendcap_Cache folder inside Blender's per-session temp dir —
    where an UNSAVED project's tracking data lives (mirrors the unsaved
    branch of _cache_dir()). Stable for the whole Blender session, so it's
    still the right 'old' location at save time even though _cache_dir()
    has by then switched to pointing next to the saved .blend."""
    # Must mirror _cache_dir's unsaved branch, incl. the Flatpak-shared base
    # (KEY BUG E) - otherwise save-time migration looks in the wrong 'old' dir.
    base = _flatpak_shared_cache_base() if _in_flatpak() else \
        (bpy.app.tempdir or tempfile.gettempdir())
    return os.path.join(base, "Blendcap_Cache")


def _norm_path(p):
    """Absolute, normalized, case-folded path for reliable comparison
    (Windows paths are case-insensitive)."""
    return os.path.normcase(os.path.normpath(bpy.path.abspath(p)))


# Legacy DOS device names Windows refuses as file names regardless of
# extension ("CON.json" fails the same way "CON" does).
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _filename_error(name):
    """Plain-English reason a user-typed file name can't be used, or
    None when it's fine. Catches what open() would otherwise reject
    with an unreadable OSError: the characters Windows forbids in file
    names, and the legacy device names it reserves. Checked on every
    platform so presets stay shareable across OSes."""
    bad = {c for c in (name or "") if c in '<>:"/\\|?*' or ord(c) < 32}
    if bad:
        shown = " ".join(sorted(c for c in bad if ord(c) >= 32))
        if shown:
            return f"Names can't contain these characters: {shown}"
        return "Names can't contain control characters"
    stem = (name or "").split(".")[0].strip().upper()
    if stem in _WINDOWS_RESERVED_NAMES:
        return (f'"{stem}" is a name reserved by Windows, '
                f"pick another name")
    return None


def _path_is_under(path, root):
    """True when `path` is `root` itself or lives inside it."""
    try:
        p = _norm_path(path)
        r = _norm_path(root)
    except Exception:
        return False
    return p == r or p.startswith(r + os.sep)


def _reroot_path(path, old_root, new_root):
    """Rewrite a path that lives under old_root so it lives under new_root
    instead, preserving the sub-path. Returns an absolute OS path. Callers
    confirm containment with _path_is_under first, so relpath stays on one
    drive and can't raise."""
    p = os.path.normpath(bpy.path.abspath(path))
    o = os.path.normpath(bpy.path.abspath(old_root))
    rel = os.path.relpath(p, o)
    return os.path.normpath(os.path.join(new_root, rel))


def _merge_move_dir(src, dst):
    """Move every entry from src into dst, merging into existing sub-
    folders and overwriting colliding files, then remove src. Used when
    the destination cache already exists (the project folder had a prior
    Blendcap_Cache) — os.rename / shutil.move can't merge into an existing
    directory, so we recurse entry-by-entry."""
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        if os.path.isdir(s):
            _merge_move_dir(s, d)
        else:
            if os.path.exists(d):
                try:
                    os.remove(d)
                except OSError:
                    pass
            try:
                shutil.move(s, d)
            except Exception as e:
                print(f"[BlendCap] Could not move {s}: {e}")
    try:
        os.rmdir(src)
    except OSError:
        pass


def _repath_cache_references(old_root, new_root):
    """Re-point in-memory datablocks whose files were just moved from
    old_root to new_root. The capture proxy strips (VSE), the viewport
    preview image, and the per-rig stored BVH path all hold ABSOLUTE paths
    into the cache, so after the folder is physically moved they would read
    as 'file missing' without this.

    Each reference is only repointed when the new file actually exists, so
    a partial move (e.g. one locked file) leaves the unmoved references
    pointing at the copy that's still on disk.

    MAIN-THREAD ONLY (touches bpy.data) — called from the save_post
    handler, which is main-thread."""
    # VSE movie proxies + any image strips, across every scene.
    for scene in bpy.data.scenes:
        se = getattr(scene, "sequence_editor", None)
        if not se:
            continue
        seq_all = getattr(se, "strips_all",
                          getattr(se, "sequences_all", []))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for s in seq_all:
                try:
                    if s.type == 'MOVIE' and _path_is_under(s.filepath, old_root):
                        new_p = _reroot_path(s.filepath, old_root, new_root)
                        if os.path.exists(new_p):
                            s.filepath = new_p
                    elif s.type == 'IMAGE' and _path_is_under(s.directory, old_root):
                        new_d = _reroot_path(s.directory, old_root, new_root)
                        if os.path.isdir(new_d):
                            s.directory = new_d
                except Exception as e:
                    print(f"[BlendCap] Could not re-point strip "
                          f"{getattr(s, 'name', '?')!r}: {e}")

    # Images — the viewport preview empty loads the proxy as a MOVIE image.
    for img in list(bpy.data.images):
        try:
            if img.filepath and _path_is_under(img.filepath, old_root):
                new_p = _reroot_path(img.filepath, old_root, new_root)
                if os.path.exists(new_p):
                    img.filepath = new_p
                    img.reload()
        except Exception as e:
            print(f"[BlendCap] Could not re-point image {img.name!r}: {e}")

    # Per-rig stored BVH path (the camera-offset re-bake button reads it).
    for obj in bpy.data.objects:
        if obj.type != 'ARMATURE':
            continue
        stored = obj.get("blendcap_bvh_path", "")
        if stored and _path_is_under(stored, old_root):
            new_p = _reroot_path(stored, old_root, new_root)
            if os.path.exists(new_p):
                try:
                    obj["blendcap_bvh_path"] = new_p
                except Exception:
                    pass


def migrate_temp_cache_into(new_cache_dir):
    """Move an unsaved session's temp cache into new_cache_dir (the just-
    saved project's cache folder) and re-point references at the new
    location.

    Returns True only when data was actually moved, so the caller can
    trigger a one-off re-save to record the new paths in the .blend.

    Scope is deliberately narrow — ONLY the temp -> project transition. A
    custom cache folder is never touched. No-ops when there's no temp
    cache, when it's empty, or when it already resolves to new_cache_dir.
    MAIN-THREAD ONLY (resolves prefs + touches bpy.data)."""
    old = _temp_cache_dir()
    if _norm_path(old) == _norm_path(new_cache_dir):
        return False
    if not os.path.isdir(old):
        return False
    try:
        if not os.listdir(old):
            # Empty throwaway cache — tidy it up, nothing to move.
            try:
                os.rmdir(old)
            except OSError:
                pass
            return False
    except OSError:
        return False

    print(f"[BlendCap] Moving temp cache into the project folder:\n"
          f"           from {old}\n           to   {new_cache_dir}")
    try:
        if not os.path.exists(new_cache_dir):
            parent = os.path.dirname(new_cache_dir)
            if parent:
                os.makedirs(parent, exist_ok=True)
            shutil.move(old, new_cache_dir)  # fast rename when same drive
        else:
            _merge_move_dir(old, new_cache_dir)
    except Exception as e:
        print(f"[BlendCap] Cache move failed (left in place): {e}")
        return False

    _repath_cache_references(old, new_cache_dir)
    return True


def _project_cache_entries(old_cache, with_library):
    """What the open project uses in old_cache: (capture folder names,
    BVH library file names). old_cache is a folder SHARED by every
    unsaved project (the custom cache folder), so only this project's
    part of it may follow the project when it is saved:

      - the folder of every path the project holds into the cache (a
        capture's overlay strip, the viewport preview image, a rig's
        stored BVH path);
      - the capture folder of every clip in the project: stamped
        overlays and Empties (blendcap_source_strip), the video strips
        in the sequencer by name, and the clip the panel is on now (a
        capture made in an earlier session is found by its clip's name
        alone, with nothing loaded from it);
      - the per-person folders of those captures (Capture_<clip>_P<n>,
        marked .person_of);
      - with_library: the BVHs in bvh_exports named after one of those
        clips, or that a rig in the project was imported from, with any
        file sharing their name.

    MAIN-THREAD ONLY (touches bpy.data)."""
    try:
        present = {n for n in os.listdir(old_cache)
                   if os.path.isdir(os.path.join(old_cache, n))}
    except OSError:
        return set(), set()
    by_fold = {os.path.normcase(n): n for n in present}
    wanted, bvh_refs = set(), set()

    def under(path):
        """Note the top-level cache entry a stored path sits in."""
        try:
            if not path or not _path_is_under(path, old_cache):
                return
            rel = os.path.relpath(
                os.path.normpath(bpy.path.abspath(path)),
                os.path.normpath(bpy.path.abspath(old_cache)))
        except Exception:
            return
        parts = rel.split(os.sep)
        if not parts or parts[0] in ("", ".", ".."):
            return
        if os.path.normcase(parts[0]) == "bvh_exports":
            if len(parts) > 1:
                bvh_refs.add(os.path.normcase(os.path.splitext(parts[1])[0]))
        else:
            wanted.add(os.path.normcase(parts[0]))

    def clip(raw):
        name = _sanitize_clip_name(str(raw or ""))
        if name:
            wanted.add(os.path.normcase(f"Capture_{name}"))

    for scene in bpy.data.scenes:
        se = getattr(scene, "sequence_editor", None)
        if not se:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            strips = list(getattr(se, "strips_all",
                                  getattr(se, "sequences_all", [])))
        for s in strips:
            try:
                if s.type not in ('MOVIE', 'IMAGE'):
                    continue
                clip(s.get("blendcap_source_strip", ""))
                where = s.filepath if s.type == 'MOVIE' else s.directory
                if _path_is_under(where, old_cache):
                    under(where)
                else:
                    clip(_strip_media_extension(s.name))
            except Exception:
                pass
    for img in bpy.data.images:
        under(img.filepath)
    for obj in bpy.data.objects:
        try:
            clip(obj.get("blendcap_source_strip", ""))
            if obj.type == 'ARMATURE':
                under(str(obj.get("blendcap_bvh_path", "") or ""))
        except Exception:
            pass
    try:
        clip(_resolve_clip_name(bpy.context))
    except Exception:
        pass

    folders = {by_fold[n] for n in wanted if n in by_fold}
    for name in present:
        for f in list(folders):
            tail = name[len(f):]
            if (os.path.normcase(name).startswith(os.path.normcase(f))
                    and tail.startswith("_P") and tail[2:].isdigit()
                    and os.path.isfile(os.path.join(old_cache, name, ".person_of"))):
                folders.add(name)

    files = set()
    if with_library:
        lib = os.path.join(old_cache, "bvh_exports")
        try:
            names = [n for n in os.listdir(lib)
                     if os.path.isfile(os.path.join(lib, n))]
        except OSError:
            names = []
        folded = {os.path.normcase(f) for f in folders}
        stems = set()
        for n in names:
            stem, ext = os.path.splitext(n)
            if ext.lower() != ".bvh":
                continue
            if os.path.normcase(stem) in bvh_refs:
                stems.add(os.path.normcase(stem))
                continue
            try:
                c = stem_clip(stem, old_cache)
            except Exception:
                c = None
            if c and os.path.normcase(f"Capture_{c}") in folded:
                stems.add(os.path.normcase(stem))
        files = {n for n in names
                 if os.path.normcase(os.path.splitext(n)[0]) in stems}
    return folders, files


def migrate_project_cache_into(old_cache, new_cache_dir):
    """Move the open project's captures out of old_cache (the custom
    cache folder it used while it was unsaved) into new_cache_dir (the
    cache next to the .blend it was just saved as), and re-point the
    project at them.

    Why: with 'Set default cache folder' on and 'Always use this
    folder' off, an unsaved project's cache is the custom folder and a
    saved project's is next to its .blend. Without this the captures
    made before the first save stayed behind in the custom folder and
    the saved project no longer found them (the user, 2026-11-26).

    Unlike the temp cache (migrate_temp_cache_into, the whole folder),
    the custom folder is shared by every unsaved project, so only what
    this project uses is moved (_project_cache_entries). Returns True
    only when something was moved. MAIN-THREAD ONLY."""
    if not old_cache or not new_cache_dir:
        return False
    if _norm_path(old_cache) == _norm_path(new_cache_dir):
        return False
    if _norm_path(old_cache) == _norm_path(_temp_cache_dir()):
        return False                # the whole-folder move covers it
    if not os.path.isdir(old_cache):
        return False
    # the library follows the cache unless it has a folder of its own
    with_library = (_norm_path(_bvh_library_dir())
                    == _norm_path(os.path.join(new_cache_dir, "bvh_exports")))
    folders, files = _project_cache_entries(old_cache, with_library)
    if not folders and not files:
        return False

    print(f"[BlendCap] Moving this project's captures into its folder:\n"
          f"           from {old_cache}\n           to   {new_cache_dir}")
    moved = False
    for name in sorted(folders):
        src = os.path.join(old_cache, name)
        dst = os.path.join(new_cache_dir, name)
        if _path_is_under(new_cache_dir, src):
            continue                # saved inside its own capture folder
        try:
            if not os.path.exists(dst):
                os.makedirs(new_cache_dir, exist_ok=True)
                shutil.move(src, dst)
            else:
                _merge_move_dir(src, dst)
            moved = True
        except Exception as e:
            print(f"[BlendCap] Could not move {name} (left in place): {e}")
    if files:
        lib_new = os.path.join(new_cache_dir, "bvh_exports")
        try:
            os.makedirs(lib_new, exist_ok=True)
        except OSError as e:
            print(f"[BlendCap] Could not create {lib_new}: {e}")
            files = set()
        for name in sorted(files):
            src = os.path.join(old_cache, "bvh_exports", name)
            dst = os.path.join(lib_new, name)
            try:
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)
                moved = True
            except Exception as e:
                print(f"[BlendCap] Could not move {name} (left in place): {e}")
    if moved:
        _repath_cache_references(old_cache, new_cache_dir)
    return moved


def _addon_dir():
    # helpers.py lives at <addon_root>/blendcap/helpers.py, so the addon
    # root — where the helper scripts (save_mhr_data.py, npz_to_bvh.py),
    # the installer/, the venv/, and models/ all live — is two levels up.
    #
    # Use abspath (NOT realpath) so symlinks aren't resolved: if the
    # installed addon symlinks into a developer's working tree, realpath
    # would land in the dev tree and saves would pollute that folder
    # instead of the install location. abspath keeps resolution at the
    # running install's literal path.
    #
    # As an installed Blender extension this module always has a real
    # on-disk __file__, so this resolves the live install folder directly.
    # (The old C:\...\BlendCap_v0 fallback only ever applied to a script
    # pasted into a .blend — which can't happen for an installed addon.)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _root_package():
    """The extension's root package name ('bl_ext.<repo>.blendcap') -
    the key Blender's addons collection and extension_path_user are
    keyed by. This module is <root>.blendcap.helpers, so drop the last
    segment (same computation addon_prefs uses)."""
    return (__package__ or "").rpartition('.')[0]


def _extension_user_root():
    """Per-extension USER data folder - Blender's official home for
    data an extension must keep across updates (extension_path_user
    survives update, reinstall, and repository changes). Everything
    BlendCap must not lose on update lives under here: saved presets
    (presets_root) and, by default, the installed dependencies
    (_deps_default_dir). Falls back to a folder in Blender's user
    config when the extension API is unavailable (running from source
    as a plain legacy addon) - same survives-updates property."""
    try:
        d = bpy.utils.extension_path_user(_root_package(), create=True)
        if d:
            return d
    except Exception:
        pass
    try:
        base = bpy.utils.user_resource('CONFIG')
    except Exception:
        base = os.path.join(os.path.expanduser("~"), ".config")
    d = os.path.join(base, "blendcap", "user_data")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def _inside_addon_dir(path):
    """True when `path` is the addon folder or lives inside it - the
    one place user data must never be configured to live, because
    updates replace that folder. The safeguard every location field
    and move target runs through."""
    try:
        p = os.path.abspath(bpy.path.abspath(path or ""))
        a = os.path.abspath(_addon_dir())
        return os.path.commonpath([os.path.normcase(p),
                                   os.path.normcase(a)]) \
            == os.path.normcase(a)
    except Exception:
        return False


def _presets_default_dir():
    """Default home for saved presets when no custom location is set."""
    return os.path.join(_extension_user_root(), "presets")


def _presets_custom_dir():
    """The custom preset location from preferences, or ''. A value
    inside the addon folder is IGNORED (treated as unset) - the prefs
    panel shows why in red next to the field. Presets live in a
    'BlendCap Presets' subfolder of whatever the user picked, so the
    folder is self-identifying instead of scattering retarget_maps/
    rest_pose_presets loose into someone's directory (no double
    nesting when they picked a folder already named that)."""
    try:
        prefs = addon_prefs()
        if prefs and getattr(prefs, "presets_use_custom", False):
            p = (getattr(prefs, "presets_custom_path", "") or "").strip()
            if p and not _inside_addon_dir(p):
                p = os.path.abspath(bpy.path.abspath(p))
                base = os.path.basename(os.path.normpath(p))
                if base.lower() != "blendcap presets":
                    p = os.path.join(p, "BlendCap Presets")
                return p
    except Exception:
        pass
    return ""


# Preset locations whose folders were already created this session
# (see presets_root). A location that gets pruned after presets move out
# of it is forgotten again (_prune_vacated_preset_root).
_PRESET_DIRS_READY = set()


def presets_root():
    """Folder holding the user's saved presets (retarget_maps/ and
    rest_pose_presets/ subfolders): the custom location from
    preferences if set, else the extension user folder - both survive
    addon updates. Custom presets used to save into the addon folder
    itself, and replacing that folder on update deleted every one.
    Configurable SEPARATELY from the dependency location on purpose."""
    d = _presets_custom_dir() or _presets_default_dir()
    # The root AND its two category folders exist from the first time
    # the location is resolved (first start after install, or the
    # moment a custom location is picked), so presets can be copied in
    # by hand without saving a dummy one first to get the folders.
    # Once per location per session: this runs from panel draws.
    key = os.path.normcase(d)
    if key not in _PRESET_DIRS_READY:
        try:
            os.makedirs(d, exist_ok=True)
            for cat in ("retarget_maps", "rest_pose_presets"):
                os.makedirs(os.path.join(d, cat), exist_ok=True)
            _PRESET_DIRS_READY.add(key)
        except Exception:
            pass
    return d


def preset_read_dirs(sub):
    """Ordered list of preset folders to READ for category `sub`
    ('retarget_maps' / 'rest_pose_presets'): the resolved save home
    first, then the default location when a custom one is set - so
    switching the preset location never makes anything vanish from the
    dropdowns. Callers append their own legacy locations after these
    and dedupe by filename (first hit wins)."""
    dirs = [os.path.join(presets_root(), sub)]
    default = os.path.join(_presets_default_dir(), sub)
    if os.path.normcase(default) != os.path.normcase(dirs[0]):
        dirs.append(default)
    return dirs


def _deps_default_dir():
    """Default home for the installed dependencies (venv/, checkpoints/,
    install_marker.json) when no custom location is set."""
    return os.path.join(_extension_user_root(), "deps")


def _deps_custom_dir():
    """The custom dependency location from preferences, or ''. Read
    defensively - preferences may be unavailable during early
    registration or headless use. A value inside the addon folder is
    IGNORED (treated as unset) - the prefs panel shows why in red
    next to the field."""
    try:
        prefs = addon_prefs()
        if prefs and getattr(prefs, "deps_use_custom", False):
            p = (getattr(prefs, "deps_custom_path", "") or "").strip()
            if p and not _inside_addon_dir(p):
                return os.path.abspath(bpy.path.abspath(p))
    except Exception:
        pass
    return ""


def deps_uninstall_roots():
    """Every location an installed environment could plausibly sit,
    for Uninstall to sweep: the resolved location, the default, the
    addon folder (legacy), and the stored custom path EVEN WHEN the
    custom checkbox is off - unticking leaves the path in the field,
    and an environment left behind there would otherwise survive an
    uninstall. Deduped. (A custom path RETYPED to something else is
    genuinely unknowable and stays uncovered - accepted.)"""
    roots = [deps_root(), _deps_default_dir(), _addon_dir()]
    p = _deps_stored_custom_dir()
    if p:
        roots.append(p)
    out, seen = [], set()
    for r in roots:
        k = os.path.normcase(os.path.abspath(r))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def deps_target():
    """Where dependencies SHOULD live per configuration (custom pref
    or default) - as opposed to deps_root(), which is where they
    actually are. The two differ on an unmoved legacy install or
    after the user changes the location; the Move buttons close the
    gap."""
    return _deps_custom_dir() or _deps_default_dir()


def _deps_stored_custom_dir():
    """The custom dependency path as STORED in preferences, whether or
    not the checkbox is ticked. Unticking the checkbox keeps the path
    in the field, and an environment living there must stay findable -
    otherwise unticking abandons the venv and the addon reports
    'not installed' with everything present (live report 2026-08-30).
    Same inside-the-addon guard as the active resolver."""
    try:
        prefs = addon_prefs()
        p = ((getattr(prefs, "deps_custom_path", "") or "").strip()
             if prefs else "")
        if p and not _inside_addon_dir(p):
            return os.path.abspath(bpy.path.abspath(p))
    except Exception:
        pass
    return ""


def _lp(p):
    """Extended-length path prefix on Windows. The venv holds package
    test-data files whose full path can cross 260 characters (measured
    262 at the default storage location; deeper custom folders make it
    worse) - uv writes them fine (Rust is long-path aware), but plain
    Python file ops fail on them: an uninstall's rmtree left exactly
    those files behind (live report 2026-08-30). Use for pure-Python
    file IO on dependency trees; never hand these paths to Blender."""
    if sys.platform == "win32":
        p = os.path.abspath(p)
        if not p.startswith("\\\\?\\"):
            return "\\\\?\\" + p
    return p


# Written into a move DESTINATION before the first byte is copied and
# removed after the last source is deleted. A folder carrying it is a
# half-finished move (cancelled, or Blender killed mid-copy): the
# resolver must never mistake it for an installed environment just
# because python.exe happened to be copied early, and the next move
# attempt wipes it and starts clean.
MOVE_SENTINEL = ".blendcap_move_in_progress"


def _dir_has_venv(d):
    if os.path.isfile(os.path.join(d, MOVE_SENTINEL)):
        return False
    if sys.platform == "win32":
        return os.path.isfile(os.path.join(d, "venv", "Scripts",
                                           "python.exe"))
    return os.path.isfile(os.path.join(d, "venv", "bin", "python"))


# deps_root() resolution cache. The uncached path touches bpy
# (preferences + extension_path_user), which only the main thread may
# do - but capture/BVH monitor threads can land here through the
# _python_exe fallback in mocap. The cache is primed at register and
# refreshed by every operation that can change the answer (pref
# update, install, uninstall, migration, storage move) via
# _invalidate_deps_cache, so thread-side calls always hit the cache.
_DEPS_ROOT_CACHE = None


def _invalidate_deps_cache():
    global _DEPS_ROOT_CACHE
    _DEPS_ROOT_CACHE = None
    try:
        _MARKER_CACHE['key'] = None
        _MARKER_CACHE['data'] = None
    except NameError:
        pass
    try:
        _DEPS_REV_CACHE['key'] = None
        _DEPS_REV_CACHE['value'] = None
    except NameError:
        pass


def deps_root():
    """Folder holding the installed dependencies: venv/, checkpoints/,
    and install_marker.json.

    Resolution prefers wherever a venv ACTUALLY exists, so every
    layout keeps working without a flag day:
      1. the custom location from preferences (if enabled),
      2. the default (extension user folder / deps),
      3. the addon folder itself - the pre-1.0.6 layout, still live
         when migration couldn't move the files (cross-volume install).
    With no venv anywhere (fresh install) the custom location (or
    default) is returned as the INSTALL TARGET. The addon folder is
    deliberately last and only ever wins when a legacy venv is present:
    new installs must never land inside the addon folder, because
    updates replace it."""
    global _DEPS_ROOT_CACHE
    if _DEPS_ROOT_CACHE:
        return _DEPS_ROOT_CACHE
    # Preferences are not readable while register() runs (and for a
    # moment after), so a custom folder can't be seen yet. A result
    # computed then must NOT be cached: it would pin the default
    # folder for the whole session, showing "Install" instead of
    # "Reinstall" and telling capture nothing is installed, until a
    # preference toggle happened to clear it (seen live 2026-11-05
    # with the dependencies on an external drive).
    prefs_readable = addon_prefs() is not None
    custom = _deps_custom_dir()
    candidates = ([custom] if custom else []) + [_deps_default_dir(),
                                                 _addon_dir()]
    # A custom path left in the UNTICKED field is still a place the
    # files may actually be (unticking must not abandon them) - last
    # in line, like the legacy addon folder.
    stored = _deps_stored_custom_dir()
    if stored and not any(os.path.normcase(stored) == os.path.normcase(c)
                          for c in candidates):
        candidates.append(stored)
    resolved = None
    for c in candidates:
        if _dir_has_venv(c):
            resolved = c
            break
    if resolved is None:
        resolved = custom or _deps_default_dir()
    if prefs_readable:
        _DEPS_ROOT_CACHE = resolved
    return resolved


def checkpoints_dir():
    """The downloaded model files folder."""
    return os.path.join(deps_root(), "checkpoints")


# Per-drive volume info cache for storage_speed_warning - the prefs
# panel calls it every redraw, and the underlying OS query shouldn't
# run that often.
_VOLUME_WARNING_CACHE = {}


def storage_speed_warning(path):
    """Pre-operation heads-up for a storage target, or None.

    Returns (kind, [lines]): kind 'slow_fs' for exFAT/FAT-family
    volumes (pathological with the venv's ~47k small files - the
    2026-07-31 two-hour-crawl ticket), 'removable' for removable
    drives (fine, but it has to be present whenever BlendCap runs).
    Best-effort: any detection failure returns None rather than
    blocking anything."""
    try:
        p = os.path.abspath(bpy.path.abspath(path or ""))
        if not p:
            return None
        if sys.platform == "win32":
            drive = os.path.splitdrive(p)[0]
            if not drive:
                return None
            key = os.path.normcase(drive)
            if key in _VOLUME_WARNING_CACHE:
                return _VOLUME_WARNING_CACHE[key]
            import ctypes
            root = drive + "\\"
            fsbuf = ctypes.create_unicode_buffer(64)
            ok = ctypes.windll.kernel32.GetVolumeInformationW(
                ctypes.c_wchar_p(root), None, 0, None, None, None,
                fsbuf, 64)
            fs = fsbuf.value.upper() if ok else ""
            dtype = ctypes.windll.kernel32.GetDriveTypeW(
                ctypes.c_wchar_p(root))
            result = None
            if fs in ("EXFAT", "FAT32", "FAT"):
                result = ("slow_fs", [
                    f"This drive is formatted {fs}, which is very slow "
                    f"with the",
                    "tens of thousands of small files in the Python "
                    "environment.",
                    "It will work, but can take a very long time. An "
                    "NTFS drive",
                    "is much faster."])
            elif dtype == 2:  # DRIVE_REMOVABLE
                result = ("removable", [
                    "This looks like a removable drive. It has to be "
                    "plugged in,",
                    "with the same drive letter, whenever BlendCap "
                    "runs."])
            _VOLUME_WARNING_CACHE[key] = result
            return result
        if sys.platform.startswith("linux"):
            # Same per-session cache as the Windows branch (keyed on the
            # path): this runs from the preferences draw.
            key = p
            if key in _VOLUME_WARNING_CACHE:
                return _VOLUME_WARNING_CACHE[key]
            best, fstype = "", ""
            with open("/proc/mounts", "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    mnt = parts[1].replace("\\040", " ")
                    if p.startswith(mnt) and len(mnt) > len(best):
                        best, fstype = mnt, parts[2]
            result = None
            if fstype.lower() in ("vfat", "exfat", "msdos", "fat32"):
                result = ("slow_fs", [
                    f"This drive uses a {fstype} filesystem, which is "
                    f"very slow",
                    "with the tens of thousands of small files in the "
                    "Python",
                    "environment. It will work, but can take a very "
                    "long time."])
            _VOLUME_WARNING_CACHE[key] = result
            return result
    except Exception:
        pass
    return None


def _python_exe():
    d = deps_root()
    if sys.platform == "win32":
        return os.path.join(d, "venv", "Scripts", "python.exe")
    return os.path.join(d, "venv", "bin", "python")


def _steam_launched():
    """True when Blender was launched through the Steam client on Linux
    (the Steam version of Blender). Steam exports its runtime's
    LD_LIBRARY_PATH (old bundled libs) and an LD_PRELOAD overlay hook
    into Blender's environment, and every subprocess inherits them -
    which breaks host tools (rocminfo / nvidia-smi during install, so
    the GPU is misdetected as absent) and can break the venv python's
    native imports (torch / onnxruntime) at capture time. Detected via
    the env Steam sets rather than any file, so it also covers a user
    who exported the same vars by hand."""
    if not sys.platform.startswith("linux"):
        return False
    return (bool(os.environ.get("STEAM_RUNTIME"))
            or "steam-runtime" in os.environ.get("LD_LIBRARY_PATH", "")
            or "gameoverlayrenderer" in os.environ.get("LD_PRELOAD", ""))


_VGL_FAKERS = ("libvglfaker.so", "libdlfaker.so", "libgefaker.so")


def _subprocess_env(overrides=None):
    """Environment for installer / venv-python subprocesses.

    ALWAYS returns a dict - never None. The one setting every child
    needs is PYTHONIOENCODING (see below), so there is no such thing as
    "nothing to change" any more; the old early return that inherited
    Blender's env untouched is exactly what let the child pick the
    system code page.
      - PYTHONIOENCODING=utf-8: pins the child's stdout/stderr encoding.
        Both readers here already decode as utf-8, but the child used to
        default its stdout to the system code page, so ANY non-ASCII
        character reaching a print raised UnicodeEncodeError and killed
        the run. Two triggers, both reported by customers: a non-UTF-8
        console locale (cp932), and - on an ordinary English install -
        a non-ASCII character anywhere in the video path, which the
        cache folder name carries into every downstream path.
      - overrides: {VAR: VALUE} injected on top (capture backend pins).
      - Steam-launched Blender: the Steam runtime's library overrides
        are stripped (see _steam_launched). LD_LIBRARY_PATH is restored
        to the pre-Steam value Steam saves in SYSTEM_LD_LIBRARY_PATH
        (usually empty -> dropped); only the gameoverlayrenderer entries
        are removed from LD_PRELOAD, anything user-set is kept.
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    # Where the model files live. save_mhr_data.py / save_face_data.py
    # default to <script dir>/checkpoints, which was correct when the
    # models lived inside the addon folder; now that dependencies live
    # in their own storage location (deps_root) the children need to be
    # told. Guarded: if resolution fails (exotic headless use) the var
    # is simply absent and the children fall back to the legacy layout.
    try:
        env["BLENDCAP_CHECKPOINTS"] = checkpoints_dir()
    except Exception:
        pass
    if _steam_launched():
        orig = env.get("SYSTEM_LD_LIBRARY_PATH")
        if orig:
            env["LD_LIBRARY_PATH"] = orig
        else:
            env.pop("LD_LIBRARY_PATH", None)
        pre = env.get("LD_PRELOAD")
        if pre:
            # LD_PRELOAD entries may be separated by colons OR spaces.
            kept = [p for p in re.split(r"[:\s]+", pre)
                    if p and "gameoverlayrenderer" not in p]
            if kept:
                env["LD_PRELOAD"] = ":".join(kept)
            else:
                env.pop("LD_PRELOAD", None)
        env.pop("STEAM_RUNTIME", None)
    # VirtualGL (vglrun / TurboVNC remote-desktop GPU setups) injects its
    # fakers through LD_PRELOAD; libdlfaker.so interposes dlopen(), and a
    # capture child that inherits it can't load the bundled cuDNN (ORT
    # aborts with "Invalid handle. Cannot load symbol cudnnCreate",
    # exit -6). The children do no OpenGL, so the fakers serve no purpose
    # there; Blender's own viewport keeps them, only the child env
    # changes. Reported with a fix by ptrthomas 2026-07-29 (item 8).
    pre = env.get("LD_PRELOAD")
    if pre and any(f in pre for f in _VGL_FAKERS):
        kept = [p for p in re.split(r"[:\s]+", pre)
                if p and os.path.basename(p) not in _VGL_FAKERS]
        if kept:
            env["LD_PRELOAD"] = ":".join(kept)
        else:
            env.pop("LD_PRELOAD", None)
        print("[BlendCap] VirtualGL detected: removed its dlopen fakers from "
              "LD_PRELOAD for the helper process.")
    if overrides:
        env.update(overrides)
    return env


def _in_flatpak():
    """True when Blender (and thus this addon) is running inside a Flatpak
    sandbox. Detected via /.flatpak-info, which the runtime always mounts.

    Flatpak Blender is one of the most common Linux installs, but the
    sandbox has no git / C toolchain / package manager, so the dependency
    venv cannot be BUILT inside it - and a host-built venv is most reliably
    RUN on the host too (the sandbox's /app/lib would otherwise shadow
    system libraries). So when this is True the installer and every venv-
    python subprocess are launched on the HOST via flatpak-spawn (see
    _host_spawn_prefix)."""
    return os.path.exists("/.flatpak-info")


def _host_spawn_prefix(directory=None, extra_env=None):
    """Argv prefix that runs a command on the HOST from inside a Flatpak
    sandbox. Returns [] on every non-Flatpak platform (native Linux,
    Windows, macOS), so callers can unconditionally prepend it.

    - Empties LD_LIBRARY_PATH on the host side so the host-built venv does
      not inherit the sandbox's /app/lib search path.
    - directory: host working directory. Absolute paths under ~/.var/app
      are identical inside and outside the sandbox, so the addon's own
      paths need no translation.
    - extra_env: {VAR: VALUE} injected on the host side. flatpak-spawn does
      not reliably forward the env set on the Popen, so capture's USE_ORT /
      BLENDCAP_FORCE_CPU / BLENDCAP_ORT_CPU must ride here.
    - PYTHONIOENCODING=utf-8 is injected unconditionally for the same
      reason: _subprocess_env sets it on the Popen env, which does not
      survive the spawn, so under Flatpak it has to be re-stated here or
      the host-side child falls back to the system code page (see
      _subprocess_env). Caller-supplied extra_env can still override it.

    Requires a one-time host grant (an addon cannot edit Blender's own
    Flatpak manifest):
        flatpak override --user --talk-name=org.freedesktop.Flatpak \\
            org.blender.Blender
    """
    if not _in_flatpak():
        return []
    pre = ["flatpak-spawn", "--host", "--env=LD_LIBRARY_PATH="]
    # flatpak-spawn --host inherits the SANDBOX cwd, which for Flatpak
    # Blender is /app/blender - a path that does not exist on the host, so
    # the portal aborts with "failed to change directory '/app/blender'"
    # before the command runs. Always pin a directory that exists on the
    # host; every command we launch uses absolute paths, so the working
    # directory only needs to be valid, not meaningful. $HOME always is.
    pre.append("--directory=" + (directory or os.path.expanduser("~")))
    host_env = {"PYTHONIOENCODING": "utf-8"}
    host_env.update(extra_env or {})
    for k, v in host_env.items():
        pre.append("--env=%s=%s" % (k, v))
    return pre


def _host_spawn_grant_cmd():
    """The one-time host grant a user must run (in a host terminal) so
    flatpak-spawn can reach the host. An addon cannot apply this itself:
    running `flatpak override` needs flatpak-spawn --host, which needs
    exactly this permission - so the wizard can only show the command."""
    return ("flatpak override --user "
            "--talk-name=org.freedesktop.Flatpak org.blender.Blender")


# Cached so the panel-draw path never re-probes (panels redraw constantly).
# The grant only takes effect on a fresh Blender start, so within a session
# the answer can't change under us - a cached False stays False until the
# user restarts Blender (which clears the cache anyway).
_host_spawn_ok_cache = None


def _host_spawn_ok(force=False):
    """Under Flatpak, True iff `flatpak-spawn --host` actually works - i.e.
    the --talk-name=org.freedesktop.Flatpak grant (see _host_spawn_grant_cmd)
    is present. Without it every host spawn fails with a portal error
    ("Portal call failed"), so the installer and all capture/BVH subprocesses
    silently do nothing. Returns True on every non-Flatpak platform (there is
    nothing to grant there).

    Probes once by running `true` on the host through the SAME prefix every
    real spawn uses (_host_spawn_prefix) and CACHES the result; a short timeout
    guards against the portal ever blocking the caller (this can be reached
    from a panel draw).

    Must go through _host_spawn_prefix, NOT a bare `flatpak-spawn --host true`:
    when Blender launches normally its cwd is /app/blender, which does not
    exist on the host, so a bare spawn fails with "Failed to change to
    directory '/app/blender'" - a false negative that makes the grant warning
    show forever even when the grant IS present (KEY BUG D, in the probe).
    The prefix always pins --directory=$HOME, so this tests the real path."""
    global _host_spawn_ok_cache
    if not _in_flatpak():
        return True
    if _host_spawn_ok_cache is not None and not force:
        return _host_spawn_ok_cache
    try:
        r = subprocess.run(_host_spawn_prefix() + ["true"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=10)
        ok = (r.returncode == 0)
    except Exception:
        # flatpak-spawn missing, portal denied/timed out -> treat as not OK.
        ok = False
    _host_spawn_ok_cache = ok
    return ok


def _installer_candidates():
    """Where a relocated installer/ may live. Since 2026-11-05 it is
    always the DEFAULT dependency storage on the addon's drive, even
    when the dependencies themselves sit in a custom folder (user
    decision: the installer is small, must still be there when an
    external drive is unplugged, and must survive the custom folder
    being unticked). A copy next to a custom folder, placed by earlier
    builds, is still found second."""
    roots = [_deps_default_dir()]
    custom = _deps_stored_custom_dir()
    if custom:
        roots.append(custom)
    return [os.path.join(r, "installer") for r in roots]


def _installer_dir():
    """Folder holding install.ps1 / install.bat / install.sh.

    Only present in PAID artifacts - the public GitHub source
    distribution ships the addon without it (see
    installer_files_present below). Since 1.1.0 the shipped installer/
    is moved out of the addon folder into the dependency storage at
    register (relocate_installer), so addon updates - which replace
    the addon folder - can't delete it and the public zip served by
    the updater leaves it alone. The addon folder is the fallback for
    the legacy in-addon layout and for the moment between install and
    relocation."""
    for d in _installer_candidates():
        if os.path.isdir(d):
            return d
    return os.path.join(_addon_dir(), "installer")


def relocate_installer():
    """Move an installer/ shipped inside the addon folder into the
    dependency storage, REPLACING whatever installer was there. The
    copy that ships with the addon being enabled is the right one for
    that addon version, so a paid zip installed over an existing
    install (the normal way to update the installer) lands its new
    scripts here on the spot. The public zip has no installer/ and
    therefore never touches the stored one: paid stays paid through
    public updates.

    Skipped on the legacy layout (dependencies still inside the addon
    folder): there the installer already sits where it belongs. An
    _offline_stash/ in the old stored copy is carried over when the
    shipped copy has none (offline-bundle users updating with an
    online zip). Returns the destination, or None when nothing moved."""
    src = os.path.join(_addon_dir(), "installer")
    if not os.path.isdir(src):
        return None
    if os.path.isdir(os.path.join(_addon_dir(), ".git")):
        # A developer's checkout: leave the tree alone.
        return None
    if _inside_addon_dir(deps_root() or ""):
        # Legacy layout: the installer already sits where it belongs.
        return None
    # Always the default storage on the addon's drive, never a custom
    # dependency folder (see _installer_candidates). Needs no
    # preference to be readable, so the move is right at register.
    root = _deps_default_dir()
    dst = os.path.join(root, "installer")
    try:
        from .operators_pipeline import _rm_readonly
    except Exception:
        _rm_readonly = None
    keep = None
    try:
        os.makedirs(root, exist_ok=True)
        if os.path.isdir(dst):
            old_stash = os.path.join(dst, "_offline_stash")
            if (os.path.isdir(old_stash)
                    and not os.path.isdir(os.path.join(src, "_offline_stash"))):
                keep = os.path.join(root, "_offline_stash.keep")
                if os.path.isdir(keep):
                    shutil.rmtree(_lp(keep), onerror=_rm_readonly)
                shutil.move(old_stash, keep)
            shutil.rmtree(_lp(dst), onerror=_rm_readonly)
            if os.path.isdir(dst):
                raise OSError("the old installer folder could not be removed")
        shutil.move(src, dst)
        if keep and os.path.isdir(keep):
            shutil.move(keep, os.path.join(dst, "_offline_stash"))
    except Exception as e:
        print(f"[BlendCap] installer relocation skipped: {e}")
        return None
    _invalidate_deps_cache()
    _INSTALLER_VER_CACHE['key'] = None
    print(f"[BlendCap] Installer moved out of the addon folder to {dst}")
    return dst


# The installer version this addon needs. Bumped with the installer
# (installer/pyproject.toml) at release; the updater's index carries
# the same number so the notice can say "your installer also needs
# updating". The paid zip is the delivery route: installing it over
# the existing copy replaces the stored installer via
# relocate_installer.
REQUIRED_INSTALLER_VERSION = "1.1.0"

_INSTALLER_VER_CACHE = {'key': None, 'version': ""}


def installer_version():
    """Version from the (relocated or shipped) installer's
    pyproject.toml, '' when absent. Stat-keyed cache: read from the
    preferences draw."""
    p = os.path.join(_installer_dir(), "pyproject.toml")
    try:
        st = os.stat(p)
    except OSError:
        return ""
    key = (p, st.st_mtime_ns, st.st_size)
    if _INSTALLER_VER_CACHE['key'] == key:
        return _INSTALLER_VER_CACHE['version']
    ver = ""
    try:
        import tomllib
        with open(p, "rb") as f:
            ver = str(tomllib.load(f).get("project", {}).get("version", "") or "")
    except Exception:
        ver = ""
    _INSTALLER_VER_CACHE['key'] = key
    _INSTALLER_VER_CACHE['version'] = ver
    return ver


def parse_version(s):
    """Comparable tuple for '1.1.0', '1.1.0-dev', '1.0.6.dev0', 'v1.2':
    (major, minor, patch, is_final). A final release outranks any
    pre-release of the same number. Unparseable -> (0, 0, 0, 0)."""
    import re
    s = (s or "").strip().lstrip("vV")
    m = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?(.*)$", s)
    if not m:
        return (0, 0, 0, 0)
    nums = tuple(int(m.group(i) or 0) for i in (1, 2, 3))
    rest = (m.group(4) or "").strip().lower()
    final = 0 if rest and any(t in rest for t in ("dev", "rc", "alpha",
                                                     "beta", "a", "b",
                                                     "-")) else 1
    return nums + (final,)


def installer_outdated():
    """(installed version, required version) when the paid installer is
    older than this addon needs; None otherwise (including when no
    installer is present)."""
    cur = installer_version()
    if not cur:
        return None
    if parse_version(cur) < parse_version(REQUIRED_INSTALLER_VERSION):
        return (cur, REQUIRED_INSTALLER_VERSION)
    return None


# ---------------------------------------------------------------------------
# Dependency revision: "were these dependencies built for this BlendCap?"
# ---------------------------------------------------------------------------
# Two copies of one number. DEPS_REVISION below is what this code needs;
# bump it at release time ONLY when requirements.txt, the installer lock
# or the model list changes (see the note at the top of requirements.txt).
# The other copy sits INSIDE the venv (blendcap_deps_revision.txt) and
# says what the environment was built for. Updates replace the addon
# folder and never touch the venv, so after an update the two can
# differ - that is the whole signal, no runtime checking beyond reading
# one small file. Who writes the venv copy: the installer at the end of
# a successful run (it gets the number in BLENDCAP_DEPS_REVISION), the
# addon the first time it sees a working environment with no stamp
# (hand-built venvs, and every environment on the first release that
# ships this), and the "Dependencies Rebuilt" button. Deleting and
# recreating the venv drops the stamp with it, so a fresh hand rebuild
# clears the banner on its own.
DEPS_REVISION = 1
_DEPS_STAMP_NAME = "blendcap_deps_revision.txt"
_DEPS_REV_CACHE = {'key': None, 'value': None}


def deps_stamp_path():
    """The stamp file inside the venv (next to Scripts/ or bin/)."""
    return os.path.join(os.path.dirname(os.path.dirname(_python_exe())),
                        _DEPS_STAMP_NAME)


def read_deps_stamp():
    """Revision the environment was stamped with, or None when there is
    no stamp (or it is unreadable). Stat-keyed cache: read from panel
    draws."""
    p = deps_stamp_path()
    try:
        st = os.stat(p)
    except OSError:
        return None
    key = (p, st.st_mtime_ns, st.st_size)
    if _DEPS_REV_CACHE['key'] == key:
        return _DEPS_REV_CACHE['value']
    val = None
    try:
        with open(p, encoding="utf-8") as f:
            val = int(f.read().strip())
    except Exception:
        val = None
    _DEPS_REV_CACHE['key'] = key
    _DEPS_REV_CACHE['value'] = val
    return val


def write_deps_stamp(revision=None):
    """Stamp the venv with DEPS_REVISION (or the given number). False
    when nothing is installed or the file can't be written."""
    if not is_dependencies_installed():
        return False
    try:
        with open(deps_stamp_path(), "w", encoding="utf-8") as f:
            f.write(f"{DEPS_REVISION if revision is None else int(revision)}\n")
    except OSError:
        return False
    _DEPS_REV_CACHE['key'] = None
    return True


def deps_outdated():
    """(stamped revision, required revision) when the installed
    environment was built for an earlier BlendCap; None when it is
    current, unstamped, or not installed."""
    if not is_dependencies_installed():
        return None
    stamped = read_deps_stamp()
    if stamped is None or stamped >= DEPS_REVISION:
        return None
    return (stamped, DEPS_REVISION)


def install_incomplete():
    """True when a paid copy has a venv but the installer never reached
    its final phase: the marker next to the venv says "running" (written
    at the start of every run since 1.1.0), or there is no marker at all.
    A half-built venv passes is_dependencies_installed (deliberately, see
    constraint 40 there), so this is what tells the UI to say "the last
    install didn't finish" instead of "Dependencies installed". Source
    copies (no installer) are never flagged: their venv is hand-built and
    has no marker by design."""
    try:
        if not installer_files_present() or not is_dependencies_installed():
            return False
        marker = read_install_marker()
        if marker is None:
            return True
        return str(marker.get("status", "")).lower() != "success"
    except Exception:
        return False


INSTALL_INCOMPLETE_NOTE = (
    "The last dependency install didn't finish. If the installer window "
    "is still running, wait for it; otherwise click Reinstall Dependencies "
    "to continue where it stopped (finished downloads are kept). Capture "
    "will fail until the install completes.")


def stamp_deps_if_unstamped():
    """Startup: a working environment with no stamp gets the current
    revision. One write per environment, ever."""
    try:
        if is_dependencies_installed() and read_deps_stamp() is None:
            write_deps_stamp()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Windows Smart App Control
# ---------------------------------------------------------------------------
# Smart App Control (Windows 11) only allows files Microsoft's cloud
# recognises; the capture engine's open-source binaries (Python, torch,
# onnxruntime...) are not signed by Microsoft, so when it is ENFORCING
# it blocks them, and Blender's own extension installer too (its ctypes
# DLL). Windows currently offers no per-app exception. Windows records the state in
# the registry: 0 off, 1 on (enforcing), 2 evaluation (not blocking).
# One registry read, cached for the session; refreshed at startup, when
# the user clicks Install/Reinstall, and from the notice's Check Again
# button. It only ever switches itself on after a Windows update, i.e.
# a reboot, so a startup read is current for the session.
_SAC_CACHE = {'state': None}
SAC_FIX = ("Go to Windows Security > App & browser control > Smart App "
           "Control settings > Off")
SAC_ANTIVIRUS = ("This is separate from your antivirus, which will keep "
                 "running as usual.")
SAC_INFO_NOTE = ("Smart App Control is on. It checks each file against "
                 "Microsoft's cloud and can block parts of BlendCap's capture "
                 "engine, which is built from open-source components that "
                 "Microsoft hasn't signed. It can also slow captures down and "
                 "cause hard to predict behaviour. Turning it off is "
                 "recommended: " + SAC_FIX + ". " + SAC_ANTIVIRUS)
SAC_ALERT_NOTE = ("Smart App Control has blocked part of BlendCap's capture "
                  "engine. You will need to turn it off in order to run "
                  "BlendCap: " + SAC_FIX + ". " + SAC_ANTIVIRUS)
SAC_EVAL_NOTE = ("Windows is currently deciding whether to turn on Smart App "
                 "Control. If it turns on, it will block BlendCap's capture "
                 "engine, and BlendCap will tell you. To avoid the "
                 "interruption, you can turn it off now: " + SAC_FIX + ".")


def smart_app_control_state(refresh=False):
    """'ON' (enforcing), 'EVAL', 'OFF', or '' (not Windows / unreadable)."""
    if _SAC_CACHE['state'] is not None and not refresh:
        return _SAC_CACHE['state']
    # Testing override, development builds only (same pattern as the
    # update index override): a user can never put Smart App Control
    # back into evaluation mode, so the notices for ON / EVAL can only
    # be seen this way once it has been turned off.
    forced = os.environ.get("BLENDCAP_SAC_STATE", "").strip().upper()
    if forced in ('ON', 'EVAL', 'OFF'):
        try:
            from .updater import is_dev_build
            if is_dev_build():
                _SAC_CACHE['state'] = forced
                return forced
        except Exception:
            pass
    state = ''
    if sys.platform == "win32":
        try:
            import winreg   # built into python's DLL, not a .pyd
            with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SYSTEM\CurrentControlSet\Control\CI\Policy") as k:
                v, _t = winreg.QueryValueEx(
                    k, "VerifiedAndReputablePolicyState")
            state = {0: 'OFF', 1: 'ON', 2: 'EVAL'}.get(int(v), '')
        except Exception:
            state = ''
    _SAC_CACHE['state'] = state
    return state


def smart_app_control_on(refresh=False):
    """True when Smart App Control is enforcing. Being on is a RISK, not
    a certainty: it decides per file by reputation, and files it has
    seen enough of get through (a capture ran fine on an enforcing
    machine 2026-10-30, two days after the same python was refused)."""
    return smart_app_control_state(refresh) == 'ON'


_SAC_SEEN_FALLBACK = {'seen': False}   # when preferences aren't reachable


def sac_block_seen():
    """Has this machine actually had a file blocked (a capture / BVH run
    dying with 0xC0E90002 or the 'Application Control policy' text, or
    Blender's installer unable to load its ctypes DLL)? Remembered in
    preferences so the alert survives a restart; forgotten when a
    capture succeeds or Check Again finds the setting off."""
    try:
        prefs = addon_prefs()
        if prefs is not None:
            return bool(getattr(prefs, "sac_block_seen", False))
    except Exception:
        pass
    return _SAC_SEEN_FALLBACK['seen']


def set_sac_block_seen(flag):
    _SAC_SEEN_FALLBACK['seen'] = bool(flag)
    try:
        prefs = addon_prefs()
        if prefs is not None and bool(getattr(prefs, "sac_block_seen", False)) != bool(flag):
            prefs.sac_block_seen = bool(flag)
    except Exception:
        pass


def smart_app_control_blocking(refresh=False):
    """The ALERT: Smart App Control is enforcing AND it has actually
    blocked something here. Drives the red note, the sidebar strip and
    the greyed install buttons. Off (any state) clears the memory."""
    if smart_app_control_state(refresh) != 'ON':
        if sac_block_seen():
            set_sac_block_seen(False)
        return False
    return sac_block_seen()


def sac_blocks_this_process():
    """Ground truth for Update Now: can Blender's extension installer
    load the ctypes DLL it needs? Tries the import (cached by Python
    once it succeeds, so this costs nothing after the first call)."""
    try:
        import ctypes  # noqa: F401
        return False
    except ImportError as e:
        blocked = "Application Control" in str(e) or smart_app_control_on()
        if blocked:
            set_sac_block_seen(True)   # an observed block
        return blocked
    except Exception:
        return False


def _license_path(filename):
    """Absolute path to a bundled license file. LICENSE and NOTICE.txt
    live at the addon root; the third-party texts (SAM_LICENSE.txt,
    DINOV3_LICENSE.txt, LICENSE_AGPL.txt, THIRD_PARTY.txt) under
    <root>/licenses/. Root is checked first so either name resolves."""
    root_file = os.path.join(_addon_dir(), filename)
    if os.path.isfile(root_file):
        return root_file
    return os.path.join(_addon_dir(), "licenses", filename)


def installer_files_present():
    """True when the dependency-installer scripts ship with this copy.

    The public GitHub repo is a source-availability artifact that
    deliberately excludes installer/ — the prefs panel swaps the
    Install/Reinstall buttons for a "source distribution" note when
    this returns False. A hand-built venv still makes the addon fully
    functional either way (is_dependencies_installed only checks the
    venv interpreter)."""
    d = _installer_dir()
    return (os.path.isfile(os.path.join(d, "install.ps1"))
            or os.path.isfile(os.path.join(d, "install.sh")))


def is_paid_build():
    """True for PAID artifacts, False for the public GitHub source.
    Drives the Donate / Support button —
    paying customers shouldn't be asked to donate; self-builders running
    from the repo are exactly the donate audience.

    Detection keys off a difference that already MUST exist between the
    channels, so packaging needs no remember-to-flip step: paid zips
    ship the installer/ folder (install scripts + download_models.py,
    and _offline_stash/ on offline bundles), while the public GitHub
    repo deliberately has no installer/ at all -> False -> button
    shows. No token sniffing — the model mirrors are public and no
    credential ships anywhere. Any unexpected read failure likewise
    defaults to False (the harmless direction). Note this makes the
    dev working tree read as paid (it has installer/), which just
    hides the donate button there — harmless."""
    try:
        inst = _installer_dir()
        return (os.path.isfile(os.path.join(inst, "download_models.py"))
                or os.path.isdir(os.path.join(inst, "_offline_stash")))
    except Exception:
        return False


def _install_marker_path():
    """install_marker.json, written by the installer's final phase.
    Lives next to the venv it describes (deps_root); pre-1.0.6
    installers wrote it into installer/ inside the addon folder -
    read_install_marker still falls back there so an unmigrated
    install keeps its recorded GPU/version details."""
    return os.path.join(deps_root(), "install_marker.json")


def _install_marker_path_legacy():
    return os.path.join(_installer_dir(), "install_marker.json")


# read_install_marker() is called from the preferences draw (twice per
# redraw) - keep the parsed file keyed on (path, mtime, size) so a
# redraw costs a stat, not an open + json.load. Self-invalidating when
# the installer rewrites the file; _invalidate_deps_cache clears it too.
_MARKER_CACHE = {'key': None, 'data': None}


def read_install_marker():
    """Parsed install_marker.json as a dict, or None if absent/unreadable.

    The installer writes this from PowerShell, which on 5.1 can emit a
    UTF-8 BOM (or UTF-16); try a few encodings before giving up so a
    perfectly valid marker isn't missed over an encoding quirk."""
    for p in (_install_marker_path(), _install_marker_path_legacy()):
        try:
            st = os.stat(p)
        except OSError:
            continue
        key = (p, st.st_mtime_ns, st.st_size)
        if _MARKER_CACHE['key'] == key:
            data = _MARKER_CACHE['data']
            return dict(data) if data is not None else None
        data = None
        for enc in ("utf-8-sig", "utf-16", "utf-8"):
            try:
                with open(p, "r", encoding=enc) as f:
                    data = json.load(f)
                break
            except (UnicodeError, ValueError):
                continue
            except Exception:
                data = None
                break
        _MARKER_CACHE['key'] = key
        _MARKER_CACHE['data'] = data
        return dict(data) if data is not None else None
    return None


def deps_in_addon_folder():
    """True when dependency files still sit inside the addon folder -
    the pre-1.0.6 layout, which the next update would wipe. Drives the
    warning + Move button in preferences. True both when the addon
    folder IS the live deps location (migration couldn't move, e.g.
    cross-volume) and when a stale copy lingers there next to a live
    one elsewhere."""
    try:
        return _dir_has_venv(_addon_dir())
    except Exception:
        return False


def _files_identical(a, b):
    """Byte-compare two small files; False on any read problem."""
    try:
        if os.path.getsize(a) != os.path.getsize(b):
            return False
        with open(a, "rb") as fa, open(b, "rb") as fb:
            return fa.read() == fb.read()
    except Exception:
        return False


def _migrate_preset_tree(src_dir, dst_dir, skip_filenames=(),
                         skip_subdir=None, bundled_dir=None, log=None):
    """Move every custom .json under src_dir into dst_dir, preserving
    subfolder structure. Shipped filenames and the shipped subtree are
    skipped. A file whose basename also exists in bundled_dir and is
    byte-identical to it is a drag-merge leftover of a shipped preset
    that moved - deleted, not migrated. Destination collisions: an
    identical file drops the source; a differing one moves in under a
    '_migrated' suffix so nothing is silently lost."""
    if not os.path.isdir(src_dir):
        return 0
    moved = 0
    for root, dirs, files in os.walk(src_dir):
        if skip_subdir:
            dirs[:] = [d for d in dirs
                       if os.path.join(root, d) != skip_subdir]
        for fn in files:
            if not fn.endswith(".json") or fn in skip_filenames:
                continue
            src = os.path.join(root, fn)
            if bundled_dir:
                shipped_twin = os.path.join(bundled_dir, fn)
                if (os.path.isfile(shipped_twin)
                        and _files_identical(src, shipped_twin)):
                    try:
                        os.remove(src)
                        if log is not None:
                            log.append(f"removed stale shipped copy {fn}")
                    except Exception:
                        pass
                    continue
            rel = os.path.relpath(src, src_dir)
            dst = os.path.join(dst_dir, rel)
            try:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.exists(dst):
                    if _files_identical(src, dst):
                        os.remove(src)
                        continue
                    # Differing twin: keep both. _migrated, then
                    # _migrated2, 3... for repeated moves; a copy
                    # identical to one already moved is dropped instead.
                    stem, ext = os.path.splitext(dst)
                    dst = None
                    for i in range(1, 100):
                        cand = stem + ("_migrated" if i == 1
                                       else f"_migrated{i}") + ext
                        if not os.path.exists(cand):
                            dst = cand
                            break
                        if _files_identical(src, cand):
                            break
                    if dst is None:
                        os.remove(src)
                        continue
                shutil.move(src, dst)
                moved += 1
                if log is not None:
                    log.append(f"moved {rel}")
            except Exception as e:
                if log is not None:
                    log.append(f"FAILED to move {rel}: {e}")
    return moved


def _legacy_preset_sources():
    """(src_dir, category, skip_filenames, skip_subdir, bundled_dir)
    for every LEGACY location that can hold custom presets an update
    would delete or that should consolidate into presets_root():
    the addon's own two folders, and the ancient CONFIG-dir maps."""
    from .retarget.preset_io import (SHIPPED_PRESET_FILENAMES,
                                     _retarget_maps_dir_legacy_config)
    a = _addon_dir()
    rm = os.path.join(a, "retarget_maps")
    rp = os.path.join(a, "rest_pose_presets")
    return [
        (rm, "retarget_maps", SHIPPED_PRESET_FILENAMES,
         os.path.join(rm, "blendcap_standard"), None),
        (rp, "rest_pose_presets", frozenset(),
         os.path.join(rp, "bundled"), os.path.join(rp, "bundled")),
        (_retarget_maps_dir_legacy_config(), "retarget_maps",
         SHIPPED_PRESET_FILENAMES, None, None),
    ]


def _count_presets_in(src_dir, skip_filenames=(), skip_subdir=None):
    if not os.path.isdir(src_dir):
        return 0
    n = 0
    for root, dirs, files in os.walk(src_dir):
        if skip_subdir:
            dirs[:] = [d for d in dirs
                       if os.path.join(root, d) != skip_subdir]
        n += sum(1 for fn in files
                 if fn.endswith(".json") and fn not in skip_filenames)
    return n


def pending_user_data_moves():
    """{'presets': n, 'deps': bool} - what still sits inside the addon
    folder, where an update can delete it. Drives the startup consent
    dialog (fires every launch until this is empty; NOTHING moves
    without the user choosing) and the prefs warnings. Only the
    ADDON-folder locations count as at-risk; the old CONFIG-dir maps
    are safe from updates and merely consolidate on a Move."""
    n = 0
    for src, _cat, skip, skip_subdir, _b in _legacy_preset_sources():
        if not _inside_addon_dir(src):
            continue
        n += _count_presets_in(src, skip, skip_subdir)
    return {"presets": n, "deps": _dir_has_venv(_addon_dir())}


def presets_need_move():
    """True when any readable preset location other than the current
    save home still holds custom presets - the addon folder (at risk)
    or, after a location change, the default/CONFIG folders
    (consolidation). Shows the Move Presets button."""
    target = presets_root()
    for src, _cat, skip, skip_subdir, _b in _legacy_preset_sources():
        if os.path.normcase(src) == os.path.normcase(target):
            continue
        if _count_presets_in(src, skip, skip_subdir):
            return True
    default = _presets_default_dir()
    if os.path.normcase(default) != os.path.normcase(presets_root()):
        for sub in ("retarget_maps", "rest_pose_presets"):
            if _count_presets_in(os.path.join(default, sub)):
                return True
    return False


def move_user_presets(log=None):
    """Move every custom preset into presets_root() - out of the
    addon folder (the at-risk case) and out of any other readable
    location (consolidation after a location change). Returns the
    number of files moved. Runs ONLY on user consent (the startup
    dialog's move buttons or Move Presets in prefs) - never
    automatically. Small files; synchronous is fine."""
    target = presets_root()
    moved = 0
    for src, cat, skip, skip_subdir, bundled in _legacy_preset_sources():
        if os.path.normcase(src) == os.path.normcase(target):
            continue
        moved += _migrate_preset_tree(
            src, os.path.join(target, cat),
            skip_filenames=skip, skip_subdir=skip_subdir,
            bundled_dir=bundled, log=log)
    default = _presets_default_dir()
    if os.path.normcase(default) != os.path.normcase(target):
        for cat in ("retarget_maps", "rest_pose_presets"):
            moved += _migrate_preset_tree(
                os.path.join(default, cat),
                os.path.join(target, cat), log=log)
    return moved


# Last-known preset root, so a location change knows where to move
# FROM. Primed at register; refreshed by every relocation.
_CURRENT_PRESETS_ROOT = None


def _track_presets_root():
    global _CURRENT_PRESETS_ROOT
    _CURRENT_PRESETS_ROOT = presets_root()


def _prune_vacated_preset_root(old_root):
    """After presets move OUT of a location, remove the folders they
    leave behind - but ONLY empty ones, and only folders that are
    ours: empty dirs under the old root's two category folders, the
    category folders themselves, and finally the root - that last one
    only when it's literally named 'BlendCap Presets' (never the
    user's own parent folder, and never the default root, which is
    BlendCap's home inside Blender's config). Anything the user put
    there makes a folder non-empty and everything above it stays."""
    try:
        if not old_root or not os.path.isdir(old_root):
            return
        if (os.path.normcase(os.path.abspath(old_root))
                == os.path.normcase(os.path.abspath(
                    _presets_default_dir()))):
            return
        for cat in ("retarget_maps", "rest_pose_presets"):
            d = os.path.join(old_root, cat)
            if not os.path.isdir(d):
                continue
            # Bottom-up: empty subfolders first, then the category dir.
            for root, dirs, files in os.walk(d, topdown=False):
                if not os.listdir(root):
                    os.rmdir(root)
        if (os.path.basename(os.path.normpath(old_root)).lower()
                == "blendcap presets" and not os.listdir(old_root)):
            os.rmdir(old_root)
    except Exception:
        pass  # cleanup is a courtesy - never let it break the move
    # Its folders may be gone now: create them again if this location is
    # ever chosen again.
    try:
        _PRESET_DIRS_READY.discard(os.path.normcase(old_root))
    except Exception:
        pass


def on_presets_location_changed():
    """Choosing a preset location IS the move (presets are tiny and
    instant - unlike the dependencies, which get an explicit button).
    Called by the preset-location preference updates: consolidates
    from the previous root, the default, and every legacy location
    into the new root. No-op when the resolved root didn't change
    (e.g. an invalid path that resolution ignores)."""
    global _CURRENT_PRESETS_ROOT
    old = _CURRENT_PRESETS_ROOT
    new = presets_root()
    if old and os.path.normcase(old) == os.path.normcase(new):
        print(f"[BlendCap] preset location unchanged ({new})")
        return 0
    log = []
    moved = 0
    if old and os.path.isdir(old):
        for cat in ("retarget_maps", "rest_pose_presets"):
            moved += _migrate_preset_tree(
                os.path.join(old, cat), os.path.join(new, cat),
                log=log)
    moved += move_user_presets(log=log)
    _CURRENT_PRESETS_ROOT = new
    _prune_vacated_preset_root(old)
    print(f"[BlendCap] preset location changed:")
    print(f"[BlendCap]   from: {old}")
    print(f"[BlendCap]   to:   {new}")
    print(f"[BlendCap]   moved {moved} file(s)")
    for line in log:
        print("[BlendCap]   " + line)
    return moved


def scan_user_data():
    """Register-time scan. Deliberately moves NO user data - moving is
    consent-only (the startup dialog / prefs Move buttons). The single
    automatic action here
    is deleting drag-merge leftovers of the SHIPPED rest poses:
    top-level copies of files that moved into bundled/, deleted only
    when byte-identical to the bundled original (shipped content
    cleanup, not user data). Returns pending_user_data_moves() so
    register can decide whether to arm the consent dialog."""
    try:
        rp = os.path.join(_addon_dir(), "rest_pose_presets")
        bundled = os.path.join(rp, "bundled")
        if os.path.isdir(rp) and os.path.isdir(bundled):
            for fn in os.listdir(rp):
                if not fn.endswith(".json"):
                    continue
                stale = os.path.join(rp, fn)
                twin = os.path.join(bundled, fn)
                if (os.path.isfile(stale) and os.path.isfile(twin)
                        and _files_identical(stale, twin)):
                    os.remove(stale)
                    print(f"[BlendCap] removed stale copy of shipped "
                          f"preset {fn} (now in bundled/)")
    except Exception:
        pass
    return pending_user_data_moves()


def deps_unreachable_path():
    """The configured custom dependency folder when it CANNOT BE
    REACHED (typically an unplugged external drive) and no environment
    exists anywhere else - else ''. Discriminates the two not-installed
    stories: a reachable-but-empty folder is a normal fresh install,
    while an unreachable one means the files probably exist and the
    drive is simply absent - telling that user to 'Install
    Dependencies' would send them into an 11 GB re-download instead of
    plugging a cable back in (support-cost decision 2026-08-30)."""
    if is_dependencies_installed():
        return ""
    p = _deps_stored_custom_dir()
    if p and not os.path.isdir(p):
        return p
    return ""


def is_dependencies_installed():
    """True only when a USABLE environment is present.

    The venv interpreter is the ground truth — without it nothing can run, so
    we require it. A leftover install_marker.json (status 'success') from an
    environment the user has since deleted must NOT read as installed: that
    stale marker used to win here and kept the UI showing 'Reinstall' (and
    skipping the installer) even after the venv was gone. The marker is still
    read elsewhere for display details (GPU / library versions); it just no
    longer decides this flag."""
    return os.path.isfile(_python_exe())


def addon_prefs(context=None):
    """The BlendCap AddonPreferences instance, or None if unavailable.

    Preferences live on the ROOT package (one level up from this blendcap/
    subpackage) and are registered under that package's name. From here
    __package__ is '<root>.blendcap', so dropping the trailing segment gives
    the root key the addons collection is keyed by - the same computation
    BLENDCAP_OT_install_open_prefs uses for addon_show(). Returns None when
    the addon isn't found (e.g. very early during registration), so callers
    should treat None as 'use defaults'."""
    if context is None:
        context = bpy.context
    root_pkg = (__package__ or "").rpartition('.')[0]
    addon = context.preferences.addons.get(root_pkg)
    return addon.preferences if addon else None


def _tag_redraw(*area_types):
    try:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type in area_types:
                    area.tag_redraw()
    except Exception:
        pass

# Common head bone names checked in order when auto-detecting the head
# reference bone on source/target rigs. Used by the Anchor (HEAD_LOCAL)
# pipeline to know which bone face animation should be measured against.
# Order matters: rigs that have both `Head` and `head.x` (mixed naming)
# get the convention-matching one chosen first.
HEAD_BONE_NAMES = ('Head', 'head', 'head.x', 'Head_M', 'J_Head', 'Bip01_Head')


def autodetect_head_bone_name(rig, prefix=""):
    """Return the first HEAD_BONE_NAMES entry that exists on the rig, or "".

    Tries the literal candidate first, then `prefix + candidate` if a
    prefix was passed — so an unprefixed rig returns 'Head', a Mixamo
    rig with prefix='mixamorig:' returns 'mixamorig:Head'. Callers that
    don't pass a prefix get the legacy (literal-only) behavior.

    Pure name lookup against rig.data.bones — fast enough to call from
    draw() (UIList eligibility check) without caching."""
    if rig is None or rig.type != 'ARMATURE':
        return ""
    bones = rig.data.bones
    for name in HEAD_BONE_NAMES:
        if name in bones:
            return name
        if prefix and (prefix + name) in bones:
            return prefix + name
    return ""


def target_head_from_pair_table(scene):
    """Last-resort fallback for the TARGET head bone: look at the bone-
    pair table and return the target field of any pair whose source
    matches the current Head (Source) value.

    Rationale: if the user explicitly paired the source's head bone to a
    specific target bone, that target bone is overwhelmingly likely to
    be the target head. Used only AFTER prefix-aware autodetect fails,
    and ONLY for the target side — the source head is the upstream
    anchor the user picks first, so we don't try to derive it
    indirectly from a table the user might not have built yet.

    Returns "" if Head (Source) is blank, the pair table is empty, or
    no matching pair has a non-empty target.
    """
    src_head = (getattr(scene, 'blendcap_retarget_face_head_source', '') or '').strip()
    if not src_head:
        return ""
    for p in scene.blendcap_retarget_pairs:
        if p.source == src_head and p.target:
            return p.target
    return ""


def _seq_accessors(seq_editor):
    """Return (all_strips, mutable_collection) from the sequence editor."""
    seq_all = getattr(seq_editor, "strips_all", getattr(seq_editor, "sequences_all", []))
    seq_coll = getattr(seq_editor, "strips", getattr(seq_editor, "sequences", None))
    return seq_all, seq_coll

def _strip_visible_start(strip):
    """Read a strip's visible start frame, suppressing deprecation warnings (ImageStrip)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return int(getattr(strip, 'frame_final_start', 1))

def _strip_visible_end(strip):
    """Read a strip's visible end frame, suppressing deprecation warnings (ImageStrip)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return int(getattr(strip, 'frame_final_end', _strip_visible_start(strip) + 1))

def _linked_sound_strip(seq_ed, strip):
    """SOUND strip that belongs to a MOVIE strip — the inverse of
    _linked_video_strip, same resolution order: the 5.2+ connection
    made at import time first, then a same-source-file match with ties
    broken by identical visible start. Ambiguous = None, never guess."""
    if strip is None or getattr(strip, "type", "") != 'MOVIE':
        return None
    for other in (getattr(strip, "connections", None) or ()):
        if getattr(other, "type", "") == 'SOUND':
            return other
    src = getattr(strip, "filepath", "") or str(strip.get("blendcap_source", ""))
    if not src:
        return None
    src = os.path.normpath(bpy.path.abspath(src))
    seq_all, _ = _seq_accessors(seq_ed)
    matches = []
    for s in seq_all:
        if getattr(s, "type", "") != 'SOUND':
            continue
        snd = getattr(s, "sound", None)
        sp = getattr(snd, "filepath", "") if snd is not None else ""
        if sp and os.path.normpath(bpy.path.abspath(sp)) == src:
            matches.append(s)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        vis = _strip_visible_start(strip)
        tied = [s for s in matches if _strip_visible_start(s) == vis]
        if len(tied) == 1:
            return tied[0]
    return None


def _strip_slip(strip, delta):
    """Slide a strip's CONTENT under its fixed timeline window (a slip
    edit): +N shows N source-frames later at every timeline frame. The
    visible range does not move, so channels, cuts, and neighbours are
    untouched. Returns the leftover frames it could NOT slip because
    the strip ran out of trimmed-off footage in that direction — the
    caller decides whether to fall back to physically moving the strip.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        off_s = int(getattr(strip, "frame_offset_start", 0))
        off_e = int(getattr(strip, "frame_offset_end", 0))
        applied = max(-off_s, min(int(delta), off_e))
        if applied:
            strip.frame_offset_start = off_s + applied
            strip.frame_offset_end = off_e - applied
            strip.frame_start = int(strip.frame_start) - applied
    return int(delta) - applied


def _strip_frame_start(strip):
    """Read a strip's frame_start, suppressing the 5.1+ deprecation warning."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return int(getattr(strip, 'frame_start', 0))

def _strip_set_frame_start(strip, value):
    """Slide a strip along the timeline. Same deprecation suppression as
    the readers above — frame_start is deprecated from Blender 5.1 and
    slated for removal in 6.0, and there is no replacement that moves a
    strip without also changing its trim."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        strip.frame_start = int(value)

def _delete_armature_in_place(obj):
    """Delete an armature object plus its data block and any solo-owned
    action. Idempotent — no-op if obj is None or not an armature.

    Used by _import_bvh to clear the previous import of the same source
    BVH before re-importing, so re-imports REPLACE the existing armature
    instead of accumulating .001 / .002 suffixes."""
    if obj is None or obj.type != 'ARMATURE':
        return
    armature_data = obj.data
    action = (obj.animation_data.action
              if obj.animation_data else None)
    for coll in list(obj.users_collection):
        coll.objects.unlink(obj)
    bpy.data.objects.remove(obj, do_unlink=True)
    if armature_data is not None and armature_data.users == 0:
        bpy.data.armatures.remove(armature_data)
    if action is not None and action.users == 0:
        bpy.data.actions.remove(action)


_BVH_IMPORTER_SUFFIX = "io_anim_bvh"


def _bvh_importer_module():
    """Module name of Blender's bundled BVH importer, or None if this
    Blender build doesn't have it at all.

    Matched by SUFFIX, never hardcoded: it ships as a core add-on named
    "io_anim_bvh" in every version BlendCap supports (verified on disk
    for 4.2 / 4.3 / 4.4 / 4.5 / 5.2, all under scripts/addons_core), but
    extension-style installs prefix the module path, so an exact-name
    lookup would silently read as "absent" there.
    """
    try:
        import addon_utils
        for mod in addon_utils.modules():
            name = getattr(mod, "__name__", "")
            if name == _BVH_IMPORTER_SUFFIX or name.endswith(
                    "." + _BVH_IMPORTER_SUFFIX):
                return name
    except Exception:
        pass
    return None


def bvh_importer_state():
    """Whether Blender can import a BVH right now. Three states, because
    they need three different answers:

      'OK'       - registered, nothing to do.
      'DISABLED' - the add-on is present but switched off. This is the
                   common one by a wide margin (five tickets in sixteen
                   days, one refund) and we can fix it in one click.
      'ABSENT'   - not in this Blender build at all. We cannot fix it,
                   so we must NOT block the export: the BVH file itself
                   is still worth producing so the user can import it
                   elsewhere.

    'OK' keys off the registered operator type rather than the add-on's
    enabled flag - the operator is what _import_bvh actually calls, and
    it is what goes missing (bpy.ops raises AttributeError: "Calling
    operator bpy.ops.import_anim.bvh error, could not be found", which
    is verbatim what customers pasted).
    """
    if hasattr(bpy.types, "IMPORT_ANIM_OT_bvh"):
        return 'OK'
    return 'DISABLED' if _bvh_importer_module() else 'ABSENT'


def _import_bvh(filepath, armature_name=None):
    # Replace-on-reimport: if a previous BVH import created an armature
    # with the same target name, delete it before importing so the new
    # one takes the original name (without Blender's .001/.002 suffix).
    # Different source BVHs map to different armature_names so unrelated
    # imports aren't affected.
    #
    # Preserve the prior armature's object transform across re-import
    # (location / rotation / scale + rotation_mode). Live retarget-time
    # tilt sliders write to rig.rotation_euler — without this carry-
    # over, re-importing the BVH would snap the new rig to (0,0,0) and
    # the slider value would be visually disconnected from the rig
    # until the user nudged it. Same logic for any manual location/
    # scale tweaks the user made on the prior import.
    prior_transform = None
    if armature_name:
        existing = bpy.data.objects.get(armature_name)
        if existing is not None and existing.type == 'ARMATURE':
            prior_transform = {
                'location': existing.location.copy(),
                'rotation_mode': existing.rotation_mode,
                'rotation_euler': existing.rotation_euler.copy(),
                'rotation_quaternion': existing.rotation_quaternion.copy(),
                'rotation_axis_angle': tuple(existing.rotation_axis_angle),
                'scale': existing.scale.copy(),
            }
            # Note: the per-rig height offset (BlendCapCameraOffset.
            # height_offset_m) writes to Hips.Y keyframes, not the
            # object's location.z, so no subtraction needed here. The
            # new BVH from a Bake re-run has its own freshly-grounded
            # Hips.Y written by pipeline/modes.py — slider on the new
            # rig defaults to 0, no double-application possible.
        _delete_armature_in_place(existing)
    bpy.ops.import_anim.bvh(
        filepath=filepath, axis_forward='-Z', axis_up='Y',
        global_scale=1.0, use_fps_scale=False,
        update_scene_fps=False, update_scene_duration=True
    )
    obj = bpy.context.active_object
    if obj and obj.type == 'ARMATURE':
        # Visualization shrink for the Root bone. With Hips as Root's
        # only child at (0, hip_height, 0) in BVH source space,
        # Blender's BVH importer derives Root's tail from Hips and
        # draws Root as a tall vertical octahedron (~92 cm) pointing
        # at the pelvis. We shorten it to a small vertical stub at
        # the world origin so it's less visually distracting.
        #
        # CRITICAL: keep the tail strictly along the bone's existing
        # axis (world +Z after axis_forward='-Z' / axis_up='Y'). DO
        # NOT swing it to a horizontal-forward direction. The BVH
        # importer has already baked Root's pose-bone .location
        # curves in the bone-local frame defined by Hips-up rest, so
        # rotating the bone post-hoc rotates those curves out of the
        # ground plane and breaks locomotion. A previous attempt to
        # rotate the curves to compensate (with R = M_new^-1 @ M_old)
        # also produced bad motion in practice — reverted. Shrink-
        # only along the existing axis preserves the local frame;
        # only the magnitude changes.
        #
        # Hips's edit-bone head is unaffected: Blender creates non-
        # connected child bones from BVH OFFSETs, so Hips stays at
        # its hip-height position regardless of where Root's tail
        # ends up.
        # Bones whose imported edit-mode geometry we want to override:
        #   Root  — shrunken to a vertical stub at world origin (the
        #           existing locomotion-axis-preserving shrink, see comment
        #           above).
        #   Hips  — extended to the neighboring Spine bone's length,
        #           strictly along its importer-derived axis (which is
        #           the near-zero average of Spine up + two UpLegs down,
        #           hence the sub-centimeter nub users missed during
        #           retarget setup). Length-only, direction untouched:
        #           the rest axis anchors the baked rotation + Hips.Y
        #           curves. Wherever the axis happens to point is
        #           accepted — user's call: "I don't care what direction
        #           it's pointing, I just want it to be a bone".
        #   Head  — in a face-only BVH (Head is the armature root, no
        #           body to give it context) the importer averages the
        #           ~70 face children's heads to derive Head's tail,
        #           which lands inside the face area and draws Head as
        #           a long bone poking through the front of the face.
        #           Shrink it to a 2 cm stub along its existing axis so
        #           it doesn't dominate the viewport. Direction is
        #           preserved -> Head's local frame and any rotation
        #           channels still match what was baked at import time.
        #           Skipped for body+face exports — there Head is a
        #           normal joint between Neck and the face bones, and
        #           leaving it at full length keeps the body skeleton
        #           visually balanced.
        _FACE_MARKER_BONES = ('Jaw', 'LeftEye', 'RightEye',
                              'UpperLip', 'LowerLip', 'LeftMouthCorner')
        head_bone = obj.data.bones.get('Head')
        face_only = (
            head_bone is not None
            and head_bone.parent is None
            and any(n in obj.data.bones for n in _FACE_MARKER_BONES)
        )

        prev_active = bpy.context.view_layer.objects.active
        prev_mode = obj.mode
        try:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            eb_root = obj.data.edit_bones.get('Root')
            if eb_root is not None:
                eb_root.tail = (eb_root.head.x, eb_root.head.y,
                                eb_root.head.z + 0.10)
            if face_only:
                eb_head = obj.data.edit_bones.get('Head')
                eb_jaw = obj.data.edit_bones.get('Jaw')
                if eb_head is not None:
                    # In face-only mode the Head bone is a static root
                    # (local_rots=identity, root_pos=zeros — see
                    # pipeline/modes.py run_face), so we can move its
                    # entire edit-bone (head AND tail) anywhere in
                    # armature-local space without breaking animation.
                    # Face bone children store their own world-space
                    # edit-bone heads independently and their pose
                    # location keyframes are deltas in Head's REST
                    # frame (axes unchanged), so moving Head's
                    # edit-bone leaves the face cluster's positions
                    # and motion unchanged.
                    #
                    # Anatomical placement: Head represents the skull,
                    # rooted at the jaw hinge (= Jaw's edit-bone head)
                    # and extending UPWARD (+Z in Blender world after
                    # BVH Y-up -> Blender Z-up import) to the top of
                    # the face cluster. Sharing a joint with Jaw lines
                    # up the bones' "fat parts" — Blender draws each
                    # octahedron's widest segment ~1/8 from the head
                    # end, so when Head.head == Jaw.head and they
                    # point in opposite directions (Head up, Jaw down
                    # to chin), both fat parts cluster around the
                    # jaw hinge.
                    if eb_jaw is not None:
                        jaw_pos = eb_jaw.head.copy()
                        face_top_zs = [b.head.z for b in obj.data.edit_bones
                                       if b.name not in ('Head', 'Jaw')]
                        if face_top_zs:
                            top_z = max(face_top_zs)
                            head_len = max(top_z - jaw_pos.z, 0.05)
                        else:
                            # No face cluster to measure against — pick a
                            # length proportional to Jaw's own length so
                            # the skull stays in scale with the rig.
                            head_len = max(eb_jaw.length * 1.6, 0.05)
                        eb_head.head = (jaw_pos.x, jaw_pos.y, jaw_pos.z)
                        eb_head.tail = (jaw_pos.x, jaw_pos.y,
                                        jaw_pos.z + head_len)
                    else:
                        # Fallback if Jaw is missing: root Head at the
                        # lowest face point (chin region) and extend it
                        # up to the top of the face cluster, giving the
                        # same skull-from-jaw-to-crown anatomy without
                        # a Jaw reference.
                        face_xs = [b.head.x for b in obj.data.edit_bones
                                   if b.name != 'Head']
                        face_ys = [b.head.y for b in obj.data.edit_bones
                                   if b.name != 'Head']
                        face_zs = [b.head.z for b in obj.data.edit_bones
                                   if b.name != 'Head']
                        if face_zs:
                            mid_x = (max(face_xs) + min(face_xs)) * 0.5
                            mid_y = (max(face_ys) + min(face_ys)) * 0.5
                            bottom_z = min(face_zs)
                            top_z = max(face_zs)
                            eb_head.head = (mid_x, mid_y, bottom_z)
                            eb_head.tail = (mid_x, mid_y,
                                            top_z + 0.02)
            # Extend Hips to a visible length. Same safety rule as the
            # Root stub above: scale the tail along the bone's EXISTING
            # axis only — direction defines the rest frame the baked
            # rotation + Hips.Y curves live in and must never change;
            # length is free. Target = the neighboring Spine bone's
            # length (its true chain segment: single child, tail on
            # Spine1's head), floored by the pelvis->Spine gap. Children
            # are imported non-connected, so moving Hips' tail moves
            # nothing else.
            eb_hips = obj.data.edit_bones.get('Hips')
            eb_spine = obj.data.edit_bones.get('Spine')
            if (eb_hips is not None and eb_spine is not None
                    and eb_hips.length > 1e-6):
                target_len = max(eb_spine.length,
                                 (eb_spine.head - eb_hips.head).length)
                if target_len > eb_hips.length:
                    eb_hips.tail = (eb_hips.head
                                    + (eb_hips.tail - eb_hips.head)
                                    * (target_len / eb_hips.length))
            bpy.ops.object.mode_set(mode='OBJECT')
            # The custom-shape disguise this replaced is retired; remove
            # the shape object an older addon version may have saved
            # into this .blend (it had a fake user, so it would linger
            # forever otherwise).
            stale = bpy.data.objects.get("BlendCapBoneShape")
            if stale is not None:
                stale_mesh = stale.data if stale.type == 'MESH' else None
                bpy.data.objects.remove(stale, do_unlink=True)
                if stale_mesh is not None and stale_mesh.users == 0:
                    bpy.data.meshes.remove(stale_mesh)
        finally:
            if prev_active is not None:
                bpy.context.view_layer.objects.active = prev_active
            if obj.mode != prev_mode:
                try:
                    bpy.ops.object.mode_set(mode=prev_mode)
                except Exception:
                    pass
        if armature_name:
            obj.name = armature_name
        # Restore the prior import's object transform so live tilt
        # sliders (and any user location/scale tweaks) survive a
        # re-import. Done AFTER the rename so the active rig is
        # definitely the new one. On a FRESH import (no prior
        # armature with this name) fall back to the 3D cursor for
        # location — matches standard Blender "add object" behavior so
        # imports land where the user is looking, not at world origin.
        # Rotation / scale stay at the importer's defaults (identity /
        # unit) so the first import is predictable.
        if prior_transform is not None:
            obj.location = prior_transform['location']
            obj.rotation_mode = prior_transform['rotation_mode']
            obj.rotation_euler = prior_transform['rotation_euler']
            obj.rotation_quaternion = prior_transform['rotation_quaternion']
            obj.rotation_axis_angle = prior_transform['rotation_axis_angle']
            obj.scale = prior_transform['scale']
        else:
            try:
                obj.location = bpy.context.scene.cursor.location.copy()
            except AttributeError:
                pass


def _bake_root_into_hips_for_export(obj):
    """Restructure a body-mocap armature so the addon-view's small Root
    stub is encoded in the *joint topology* — not just the edit-bone
    visual — so external tools (Maya, MotionBuilder, Unreal, Unity)
    draw the skeleton the same way Blender does after the addon import.

    Pipeline: bake Root's per-frame basis location into Hips's basis
    location (so the locomotion that Hips used to inherit through the
    parent chain is now carried by Hips itself), unparent Hips so it
    becomes the top-level bone, then delete Root entirely. Result:
    receiving software sees Hips as the armature root, no oversized
    Root bone reaching from origin to the pelvis.

    No-op for face-only exports and for any armature that doesn't have
    the body Root -> Hips structure. Intended for use by the BVH and
    FBX library-save operators only — the in-blender library imports
    keep the original Root + Hips parent chain (the addon's retarget
    pair-table presets name Root for locomotion).
    """
    if obj is None or obj.type != 'ARMATURE':
        return
    bones = obj.data.bones
    if 'Root' not in bones or 'Hips' not in bones:
        return
    hips_parent = bones['Hips'].parent
    if hips_parent is None or hips_parent.name != 'Root':
        return

    from .retarget.fcurves import _iter_action_fcurves

    action = (obj.animation_data.action
              if obj.animation_data else None)
    scene = bpy.context.scene
    prev_active = bpy.context.view_layer.objects.active
    saved_frame = scene.frame_current

    # Determine which frames are keyed on either bone's loc/rot. Read
    # only integer frames -- mocap BVH is dense per-integer-frame and
    # scene.frame_set wants ints. We snapshot at exactly those frames
    # so the post-restructure action keys the same frames as before.
    relevant_prefixes = ('pose.bones["Root"].',
                         'pose.bones["Hips"].')
    relevant_suffixes = ('location', 'rotation_quaternion',
                         'rotation_euler', 'rotation_axis_angle')
    frames = set()
    if action is not None:
        for fc in _iter_action_fcurves(action):
            dp = fc.data_path
            if not any(dp.startswith(p) for p in relevant_prefixes):
                continue
            if not any(dp.endswith(s) for s in relevant_suffixes):
                continue
            for kp in fc.keyframe_points:
                frames.add(int(round(kp.co.x)))
    frames = sorted(frames)

    # Snapshot Hips's armature-space matrix per frame. pb.matrix is in
    # armature space (= world-space when obj is at identity); it folds
    # in the parent chain, rest offsets, and basis transforms — i.e.
    # exactly what the viewer renders. Restoring via pb.matrix = ...
    # after unparenting lets Blender decompose into the new basis for
    # us, sidestepping the per-axis math that fails when Root and Hips
    # have non-aligned bone-local frames (Hips's tail is averaged
    # across Spine + leg children, so its local X/Z aren't exactly
    # parallel to Root's).
    pb_hips = obj.pose.bones.get('Hips')
    snapshots = {}
    if pb_hips is not None and frames:
        try:
            bpy.context.view_layer.objects.active = obj
            for f in frames:
                scene.frame_set(f)
                snapshots[f] = pb_hips.matrix.copy()
        except Exception:
            scene.frame_set(saved_frame)
            raise

    # Unparent Hips and delete Root in edit mode. Setting parent=None
    # on the edit-bone leaves head/tail unchanged in armature space, so
    # Hips's rest_matrix is preserved -- the only thing that changes is
    # that Hips no longer inherits Root's pose.
    try:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        eb = obj.data.edit_bones
        if 'Hips' in eb:
            eb['Hips'].parent = None
        if 'Root' in eb:
            eb.remove(eb['Root'])
        bpy.ops.object.mode_set(mode='OBJECT')
    finally:
        if prev_active is not None:
            try:
                bpy.context.view_layer.objects.active = prev_active
            except Exception:
                pass

    if not snapshots:
        scene.frame_set(saved_frame)
        return

    # Re-key Hips by re-applying its snapshot pose-matrix per frame.
    # Setting pb.matrix recomputes pb.location + pb.rotation_* such
    # that the world (armature-space) result matches; keyframe_insert
    # then captures those decomposed values onto fcurves. The keys
    # we write here REPLACE any prior Hips loc/rot keys at these
    # frames (keyframe_insert overwrites at the same frame), which is
    # what we want — the new values fold in Root's contribution.
    try:
        bpy.context.view_layer.objects.active = obj
        pb_hips = obj.pose.bones['Hips']
        rmode = pb_hips.rotation_mode
        if rmode == 'QUATERNION':
            rot_path = 'rotation_quaternion'
        elif rmode == 'AXIS_ANGLE':
            rot_path = 'rotation_axis_angle'
        else:
            rot_path = 'rotation_euler'
        for f in frames:
            scene.frame_set(f)
            pb_hips.matrix = snapshots[f]
            pb_hips.keyframe_insert('location', frame=f)
            pb_hips.keyframe_insert(rot_path, frame=f)
    finally:
        scene.frame_set(saved_frame)
        if prev_active is not None:
            try:
                bpy.context.view_layer.objects.active = prev_active
            except Exception:
                pass


def _linked_video_strip(seq_ed, strip):
    """MOVIE strip that belongs to a SOUND strip imported alongside it.

    Blender's drag-and-drop import selects both halves of a movie file
    but makes the AUDIO strip the active one, so capturing right after
    dropping a clip in used to fail with the no-active-video error.
    Resolution order:
      1. Blender 5.2+ strip connections — the exact link made at import
         time (read via getattr; older Blenders skip straight to 2).
      2. Any version: the MOVIE strip reading the same source file as
         the sound. A BlendCap overlay proxy matches through its stamped
         blendcap_source and a reverted original through its filepath,
         so the hop keeps working after capture/revert cycles (both
         create fresh strips, which silently severs connections).
    Same-file ties (split or re-added clips) are broken by identical
    visible start frame; anything still ambiguous returns None so the
    caller keeps the old behavior — never guess.
    """
    if strip is None or getattr(strip, "type", "") != 'SOUND':
        return None
    # 1) Authoritative link (Strip.connections, Blender 5.2+).
    for other in (getattr(strip, "connections", None) or ()):
        if getattr(other, "type", "") == 'MOVIE':
            return other
    # 2) Same-source-file match.
    snd = getattr(strip, "sound", None)
    src = getattr(snd, "filepath", "") if snd is not None else ""
    if not src:
        return None
    src = os.path.normcase(os.path.normpath(bpy.path.abspath(src)))
    seq_all, _unused = _seq_accessors(seq_ed)
    candidates = []
    for s in seq_all:
        if getattr(s, "type", "") != 'MOVIE':
            continue
        for p in (getattr(s, "filepath", ""),
                  str(s.get("blendcap_source", "") or "")):
            if p and os.path.normcase(
                    os.path.normpath(bpy.path.abspath(p))) == src:
                candidates.append(s)
                break
    if len(candidates) > 1:
        start = _strip_visible_start(strip)
        candidates = [s for s in candidates
                      if _strip_visible_start(s) == start]
    return candidates[0] if len(candidates) == 1 else None


def _active_video_strip(seq_ed):
    """seq_ed.active_strip, hopping from an active SOUND strip to the
    MOVIE strip it was imported with (see _linked_video_strip). Every
    "which clip is the user pointing at" read goes through here so
    capture, BVH generation, cache identity, and the panel can't
    resolve the answer differently. Returns the sound strip unchanged
    when no linked video is found — callers' MOVIE/IMAGE type checks
    then reject it exactly like before."""
    strip = getattr(seq_ed, "active_strip", None) if seq_ed else None
    if strip is not None and getattr(strip, "type", "") == 'SOUND':
        video = _linked_video_strip(seq_ed, strip)
        if video is not None:
            return video
    return strip


def _file_mode_source(context):
    """Resolve File-Path mode's video source. Returns (video_path, empty).

    The active BlendCap preview Empty normally carries the clip identity —
    clicking the preview plane and re-running capture must keep pointing
    at that clip. But when the Video field holds a DIFFERENT existing file
    than the Empty's stored source, the user has just picked a new video
    and the field wins; otherwise the selected Empty silently hijacks
    every capture (and Clear Cache) until it is deselected or removed.

    The field only wins when its file exists on disk: a stale/moved field
    path keeps the old Empty-first behavior, so re-capture and BVH
    generation from a still-valid cache keep working after the source
    file is gone.

    `empty` is the object whose stored identity should drive clip naming
    (None = the field is authoritative, name from the file stem). Every
    File-mode consumer (capture, BVH generation, Clear Cache, clip
    naming) resolves through here so they can't disagree.
    """
    props = context.scene.blendcap_props
    obj = context.active_object
    empty = (obj if obj and obj.type == 'EMPTY' and "blendcap_source" in obj
             else None)
    from . import multicam
    if multicam.is_multi_mode(props):
        # Multi-Camera Capture from File Path: the take is the clips
        # selected in the panel (multicam.py), not the single Video
        # field. Same rule as below: the active plane carries the
        # identity unless the panel now holds a DIFFERENT take.
        infos = multicam.file_infos(props)
        take = multicam.files_clip_id(infos) if len(infos) >= 2 else ""
        if empty is not None and (
                not take
                or str(empty.get("blendcap_source_strip", "") or "") == take):
            return bpy.path.abspath(str(empty["blendcap_source"])), empty
        if take:
            return bpy.path.abspath(str(infos[0]["path"])), None
        return "", None
    field = bpy.path.abspath(props.video_path) if props.video_path else ""
    if empty is None:
        return field, None
    stored = bpy.path.abspath(str(empty["blendcap_source"]))
    if field and os.path.exists(field):
        try:
            differs = (os.path.normcase(os.path.normpath(field))
                       != os.path.normcase(os.path.normpath(stored)))
        except Exception:
            differs = False
        if differs:
            return field, None
    return stored, empty


def _resolve_video_path(context):
    """Resolve the active video path and mode from UI state.
    Returns (video_path, start_frame, max_frames, insert_frame, insert_channel, mode) or None on error.
    """
    props = context.scene.blendcap_props
    start_frame = 0; max_frames = 0
    insert_frame = 1; insert_channel = 1

    if props.video_source == 'FILE':
        video_path, _empty = _file_mode_source(context)
        return (video_path, start_frame, max_frames, insert_frame, insert_channel, 'FILE')

    # VSE mode
    seq_ed = context.scene.sequence_editor
    strip = _active_video_strip(seq_ed)
    if not strip or strip.type not in ('MOVIE', 'IMAGE'):
        return None

    if "blendcap_source" in strip:
        # Proxy strip — trace back to original video, use stored trim
        video_path = bpy.path.abspath(str(strip["blendcap_source"]))
        start_frame = int(strip.get("blendcap_original_offset_start", 0))
        max_frames = int(strip.get("blendcap_original_duration", 0))
        insert_frame = _strip_visible_start(strip)
        insert_channel = strip.channel
    elif strip.type == 'IMAGE':
        # Image sequence — converted via convert_source.py subprocess.
        # Trim offsets are read from the strip in execute() and passed to the
        # conversion script, so no deprecated properties needed here.
        img_dir = bpy.path.abspath(strip.directory)
        video_path = img_dir
        start_frame = 0
        max_frames = 0
        insert_frame = _strip_visible_start(strip)
        insert_channel = strip.channel
    else:
        # Original movie strip — properties deprecated in Blender 5.1+
        video_path = bpy.path.abspath(strip.filepath)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            start_frame = int(strip.frame_offset_start)
            max_frames = int(strip.frame_final_duration)
        insert_frame = _strip_visible_start(strip)
        insert_channel = strip.channel

    return (video_path, start_frame, max_frames, insert_frame, insert_channel, 'VSE')


def _has_active_vse_strip(context):
    """True when there's an active movie/image strip in the sequencer.
    Doesn't check the strip's file resolves on disk — used only to decide
    whether to suggest the user switch Source over to it."""
    seq_ed = context.scene.sequence_editor
    strip = _active_video_strip(seq_ed)
    return bool(strip and strip.type in ('MOVIE', 'IMAGE'))


def _has_valid_file_path(context):
    """True when File-Path mode has a video that exists on disk (either the
    active BlendCap Empty's stored source or the Video field). Used to
    suggest switching Source back to File Path from an empty VSE selection."""
    props = context.scene.blendcap_props
    obj = context.active_object
    if obj and obj.type == 'EMPTY' and "blendcap_source" in obj:
        p = bpy.path.abspath(str(obj["blendcap_source"]))
        if p and os.path.exists(p):
            return True
    p = bpy.path.abspath(props.video_path) if props.video_path else ""
    return bool(p and os.path.exists(p))


def source_mode_error(context):
    """Build a mode-aware 'no valid source' error for the status bar.

    Names the source mode the addon is actually reading from, and — when
    the OTHER mode has something valid selected — points the user at the
    switch they most likely forgot. The classic trap: a strip is selected
    in the sequencer but Source is still on 'File Path', so the old generic
    "select a valid clip" read as nonsense ("I have one selected!").

    ASCII only and short enough for Blender's status-bar report().
    """
    props = context.scene.blendcap_props
    if props.video_source == 'FILE':
        if _has_active_vse_strip(context):
            return ("Source is set to 'File Path' but no file is selected. "
                    "A sequencer strip is active: switch Source to "
                    "'Video Editor Strip' to use it.")
        return ("No video file selected. Choose a file under Video Source, "
                "or switch Source to 'Video Editor Strip' for the sequencer.")
    # VSE mode
    if _has_active_vse_strip(context):
        # A strip IS active but its file couldn't be resolved on disk.
        return ("Original video not found. Relink the source or select a "
                "different clip.")
    if _has_valid_file_path(context):
        return ("Source is set to 'Video Editor Strip' but no strip is active. "
                "A video file is set under Video Source: switch Source to "
                "'File Path' to use it.")
    return ("No active sequencer strip. Select a movie or image strip in the "
            "Video Sequencer, or switch Source to 'File Path'.")


# ------------------------------------------------------------
# BVH names
# ------------------------------------------------------------
# A BVH (and the armature imported from it) is named after its capture
# plus letters for what is in it: B body, H hands, R root motion, F
# face. A multi-camera take puts its number of cameras in front of the
# letters ("4c"), and a person of a multi-person capture comes before
# that (the user's scheme, 2026-11-25: "So the filenames don't get any
# longer than they need to be"):
#
#   the capture (clip id)     the name
#   Demo                      Demo_BHR
#   Demo_4cam                 Demo_4cBHR
#   Demo_P2                   Demo_P2_BHR
#   Demo_4cam_P2              Demo_P2_4cBHR
#
# Until then the name was the clip id plus the letters
# (Demo_4cam_BHR, Demo_4cam_P2_BHR); names of that shape are still
# read. The CAPTURE keeps its id (its cache folder, the rig's
# blendcap_source_video): only the name is shorter. bvh_stem makes a
# name; stem_parts and stem_clip read one back.
_TAKE_CLIP = re.compile(r"^(?P<base>.+)_(?P<cams>[1-9]\d*)cam(?P<person>_P\d+)?$")
_STEM_TAGS = re.compile(r"^(?:(?P<cams>[1-9]\d*)c)?(?P<codes>[BHRF]+)$")
_PERSON_TAIL = re.compile(r"^(?P<base>.+)(?P<person>_P\d+)$")


def bvh_stem(clip, codes):
    """The name of a capture's BVH holding `codes` (letters of BHRF;
    "" gives what all of its names start with)."""
    m = _TAKE_CLIP.match(clip or "")
    if m:
        return f"{m.group('base')}{m.group('person') or ''}_{m.group('cams')}c{codes}"
    return f"{clip}_{codes}"


def stem_parts(stem):
    """One clip's part of a BVH name read back: (the clip ids it can
    stand for, the likeliest first; its letters), or None when it does
    not follow the naming. Two clip ids come back for a name like
    Demo_P2_4cBHR, which is person 2 of the take Demo_4cam and would
    also be the take of a clip that is itself called Demo_P2."""
    head, sep, tags = (stem or "").rpartition("_")
    m = _STEM_TAGS.match(tags) if (sep and head) else None
    if m is None:
        return None
    cams = m.group("cams")
    if not cams:
        return [head], m.group("codes")
    clips = []
    p = _PERSON_TAIL.match(head)
    if p:
        clips.append(f"{p.group('base')}_{cams}cam{p.group('person')}")
    clips.append(f"{head}_{cams}cam")
    return clips, m.group("codes")


def stem_clip(stem, cache_dir=None):
    """The clip id one clip's part of a BVH name stands for, or None.
    Where it can be read two ways (stem_parts), the one whose capture
    is in the cache."""
    got = stem_parts(stem)
    if got is None:
        return None
    clips = got[0]
    if len(clips) > 1 and cache_dir:
        for clip in clips:
            if os.path.isdir(os.path.join(cache_dir, f"Capture_{clip}")):
                return clip
    return clips[0]


def _strip_capture_prefix(folder):
    """Strip the leading 'Capture_' prefix from a cache folder name to
    get the bare clip identifier. Used both for user-visible dropdown
    labels and for BVH stem naming (which intentionally omits the
    prefix — it only exists to keep VSE strip names recognizable)."""
    return folder[len("Capture_"):] if folder.startswith("Capture_") else folder


_MEDIA_EXTS_FOR_STRIP = {
    # video — kept in sync with _SUPPORTED_VIDEO_EXTS below plus a
    # few we don't natively support but Blender will name a strip
    # after (mpg, m4v, ts).
    'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv', 'm4v',
    'mpg', 'mpeg', 'ts', 'm2ts',
    # image — covers IMAGE-strip names too.
    'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp', 'exr', 'dpx', 'hdr',
    'tga', 'webp', 'gif',
}


def _strip_media_extension(name):
    """Drop a trailing media file extension from a clip identifier
    while preserving Blender's duplicate-counter suffix. Only strips
    recognized media extensions — arbitrary trailing tokens like
    'xyz' or 'v2' that look like extensions but aren't media types
    are left alone, since those are usually part of the user's name.
      myclip                  -> myclip
      myclip.mp4              -> myclip
      myclip.001              -> myclip.001         (counter kept)
      myclip.mp4.001          -> myclip.001         (ext dropped)
      myclip.xyz.mp4.001      -> myclip.xyz.001     (ext dropped, .xyz kept)
      myclip.xyz.001          -> myclip.xyz.001     (.xyz not a media ext)
      myclip.v2.mov           -> myclip.v2          (.mov stripped)
    The sanitizer collapses the remaining dots to underscores
    afterward, so without this pass "myclip.mp4" would leak into the
    cache folder name as "myclip_mp4"."""
    if not name:
        return name
    base = name
    counter = ""
    # Peel a trailing .NNN counter first (Blender duplicate suffix).
    if '.' in base:
        head, tail = base.rsplit('.', 1)
        if tail.isdigit() and head:
            base = head
            counter = '.' + tail
    # Then peel a trailing extension, but ONLY if it's a recognized
    # media type.
    if '.' in base:
        head, tail = base.rsplit('.', 1)
        if tail.lower() in _MEDIA_EXTS_FOR_STRIP and head:
            base = head
    return base + counter


def _sanitize_clip_name(name):
    """Make a strip / Empty / filename safe to use as a directory name.

    Replaces characters that break paths on Windows (and confuse the
    bake-flag parser, which splits stems on '.' and '-'). Periods are
    the main concern: Blender auto-numbers duplicated strips with
    '.001', '.002', etc., so a strip named 'myclip.001' becomes the
    folder 'myclip_001'.
    """
    if not name:
        return ""
    out = name
    for ch in '.\\/:*?"<>|':
        out = out.replace(ch, '_')
    return out.strip().strip('_')


def _is_multi_mode(props):
    from . import multicam
    return multicam.is_multi_mode(props)


def _resolve_clip_name(context):
    """Bare clip identifier (sanitized) used to build cache folder and
    proxy strip names.

    A Multi-Camera Capture needs nothing special here: its multiview
    strip is stamped with the capture's clip id (blendcap/multicam.py)
    like any capture overlay, and the merged NPZ sits at the top level
    of that folder exactly where a single capture's would.

    The Capture_ / Preview_ prefix is applied at the call site so this
    function returns just the source-side identifier.

    VSE mode: prefers the proxy's stored `blendcap_source_strip`
    custom prop (so re-capture stays in the original cache folder);
    otherwise the active strip's current name.

    FILE / viewport mode: resolves through _file_mode_source (so naming
    can't disagree with the path capture reads): the winning Empty's
    stored `blendcap_source_strip`, otherwise the file's basename stem.

    Returns "" if nothing can be resolved.
    """
    props = context.scene.blendcap_props
    raw = ""

    if props.video_source == 'FILE':
        vp, empty = _file_mode_source(context)
        if empty is not None:
            stored = str(empty.get("blendcap_source_strip", "") or "")
            # Stored stamps are already extension-stripped at write time.
            raw = stored or _strip_media_extension(empty.name)
        elif vp and _is_multi_mode(props):
            # the take selected in the panel (see _file_mode_source)
            from . import multicam
            raw = multicam.files_clip_id(multicam.file_infos(props))
        elif vp:
            base = os.path.basename(vp.rstrip(os.sep))
            stem = os.path.splitext(base)[0]
            raw = stem if stem and not stem.startswith('.') else base
    else:
        seq_ed = context.scene.sequence_editor
        strip = _active_video_strip(seq_ed)
        if strip and strip.type in ('MOVIE', 'IMAGE'):
            stored = str(strip.get("blendcap_source_strip", "") or "")
            # strip.name can include the file extension on some
            # import paths (drag-and-drop, duplicates of a renamed
            # strip); peel it before sanitizing so the cache folder
            # name doesn't carry .mp4 / .mov / .png etc.
            raw = stored or _strip_media_extension(strip.name)

    return _sanitize_clip_name(raw)


def _next_free_clip_name(context, base):
    """Next Blender-style '.NNN'-suffixed strip name that is free in the
    sequence editor AND whose cache folder doesn't already exist.

    Used by the capture collision guard's "Capture as New Clip" choice:
    a strip re-added from the same file silently reuses a name whose
    cache folder still holds an earlier capture, so the fork needs a
    name that is unique on BOTH sides. Main thread only (_cache_dir).
    """
    stem = base
    if '.' in stem:
        head, tail = stem.rsplit('.', 1)
        if tail.isdigit() and head:
            stem = head
    taken = set()
    seq_ed = context.scene.sequence_editor
    if seq_ed:
        seq_all, _ = _seq_accessors(seq_ed)
        taken = {s.name for s in seq_all}
    cache_base = _cache_dir()
    for i in range(1, 1000):
        cand = f"{stem}.{i:03d}"
        if cand in taken:
            continue
        # Resolve the candidate's folder EXACTLY the way capture does
        # (extension peel included): a strip named "dance.mp4.001"
        # resolves to Capture_dance_001, so THAT folder must be free —
        # checking Capture_dance_mp4_001 would green-light a name that
        # lands on an existing capture.
        folder = os.path.join(
            cache_base,
            f"Capture_{_sanitize_clip_name(_strip_media_extension(cand))}")
        if os.path.exists(folder):
            continue
        return cand
    return f"{stem}.001"


# ============================================================
# CONTEXT OVERRIDE HELPER (CRITICAL FOR UI PANELS/POPUPS)
# ============================================================
@contextlib.contextmanager
def get_3d_override(context):
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        with context.temp_override(window=window, area=area, region=region):
                            yield
                        return
    with context.temp_override():
        yield

# ============================================================
# SHARED STATE
# ============================================================
blendcap_state = {
    'running': False, 'progress': 0.0, 'total_steps': 1,
    'current_step': 0, 'step_progress': 0.0,
    'step_weights': [], 'step_offsets': [],
    'status_text': "", 'proxy_path': "",
    'insert_frame': 1, 'insert_channel': 1,
    'source_path': "", 'source_start': 0, 'source_max': 0,
    'mode': 'FILE', 'return_code': 0,
    'cancel_requested': False, 'active_process': None,
    'steps_completed': [], 'step_labels': [], 'step_outputs': [],
    'track_body': False, 'track_face': False,
    'output_dir': "", 'is_preview': False,
    'warnings': [],
    'addon_dir': "", 'python_exe': "",
    'actual_video': "",  # converted source (source_export.mp4) or same as source_path
    'cache_start_frame': 0, 'cache_max_frames': 0,  # original frame params for cache validation
    'cache_img_offset_start': 0, 'cache_img_offset_end': 0,  # image sequence trim offsets
    'vse_replace': None,  # dict of original strip properties for replacement
    # Set the first time the user switches Source from FILE -> VSE in a
    # session and we've shown the "Sequencer is on" warning popup. Stops
    # the dialog from re-firing on every subsequent toggle.
    'sequencer_warned': False,
}

bvh_state = {
    'running': False, 'bvh_path': "", 'vid_name': "",
    'cache_dir': "", 'return_code': 0, 'error_msg': "",
    'cancel_requested': False, 'active_process': None,
    # Progress display (mirrors blendcap_state's progress fields). The
    # subprocess emits `[PROGRESS pct] message` lines; run_bvh_thread
    # maps each job's local 0-100 into the global 0-100 across all jobs.
    'progress': 0.0, 'status_text': "",
    'current_job': 0, 'total_jobs': 0,
}

# ARKit Apply's face-processing subprocess (modal operator in
# operators_library.py). Same active_process contract as the two dicts
# above so _terminate_state_process / the exit hook cover it.
arkit_state = {
    'running': False, 'done': False, 'ok': False, 'error_msg': "",
    'cancel_requested': False, 'active_process': None,
    'progress': 0.0, 'status_text': "",
}

def _terminate_state_process(state):
    """Terminate the live capture/BVH subprocess recorded in `state`, if any.

    Called from the Cancel operator (main thread) and the exit hook below.
    The worker's reader loop can sit blocked on the child's stdout for
    minutes while it loads checkpoints in silence, so a cancel flag alone
    is not enough — killing the child closes the pipe, the blocked read
    returns EOF, and the worker then sees cancel_requested and winds down."""
    proc = state.get('active_process')
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
    except Exception:
        pass


def _kill_subprocesses_at_exit():
    # Blender quitting mid-capture: the worker threads are daemons (nothing
    # joins them at shutdown), so without this hook the multi-GB inference
    # child would outlive Blender and keep the GPU busy until the clip
    # finishes. Subprocess handles only — no bpy access at exit time.
    for state in (blendcap_state, bvh_state, arkit_state):
        _terminate_state_process(state)


atexit.register(_kill_subprocesses_at_exit)


# Shared progress dict for the in-Blender bake operators (apply_retarget
# and convert_fk_to_ik). The retarget operator is modal so it ticks this
# from its TIMER handler; the FK->IK operator is synchronous and ticks it
# inline + pumps redraw_timer so the panel still updates. The panel reads
# `phase` + `progress` + `status_text` to render a single progress line
# under whichever operator is active.
retarget_progress_state = {
    'running': False,
    'phase': "",            # 'Retargeting' / 'Converting FK->IK'
    'progress': 0.0,        # 0..100
    'status_text': "",
    'frames_done': 0,
    'frames_total': 0,
}


def format_workspace_status(label, pct, status=""):
    """Format the ASCII progress string painted into Blender's
    bottom-left workspace status bar (`workspace.status_text_set`).

    Single source of truth so every BlendCap subsystem (Capture, BVH,
    Retarget, FK->IK) renders an identical corner widget — same bar
    width, same fill chars, same trailing-status convention. Format:

        BlendCap <Subsystem>  [████░░░░░░░░░░░░░░░░]  47%  <status>
    """
    pct_i = int(max(0, min(100, pct)))
    filled = pct_i // 5
    bar = "█" * filled + "░" * (20 - filled)
    if status:
        return f"{label}  [{bar}]  {pct_i}%  {status}"
    return f"{label}  [{bar}]  {pct_i}%"

_ansi_re = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
_progress_re = re.compile(r'\[PROGRESS (\d+)\]\s*(.*)')
_warning_re = re.compile(r'\[WARNING\]\s*(.*)')
_frame_re = re.compile(r'Frame (\d+)/(\d+)')
_tqdm_re = re.compile(r'(\d+)%')
# Per-frame debug/timing spam from the vendored Fast-SAM-3D-Body model
# (unconditional [run_inference]/[forward_pose_branch]/[process_one_image]
# timing prints, [DEBUG]/[DEBUG NaN] lines, the postprocess_ik box, and the
# per-frame detector/FOV notices). Dropped from the Blender console by
# mocap.py's reader; the full <output>/blendcap_capture_debug.log keeps it all.
_capture_noise_re = re.compile(
    r'^(?:\[run_inference\]|\[forward_pose_branch\]|\[process_one_image\]'
    r'|\[DEBUG\]|\[DEBUG NaN\]|\[postprocess_ik|\[SAM3DBodyEstimator\]'
    r'|#######|Found boxes:|Running object detector|Running FOV estimator'
    r'|Mask-condition inference|[┌└│|─])'
)

# Video formats OpenCV typically handles (pass through, no conversion)
_SUPPORTED_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}


def check_npz_coverage(npz_path, start_frame, max_frames):
    import numpy as np
    if not os.path.exists(npz_path):
        return ('none', 0, 0)
    try:
        data = np.load(npz_path)
        if 'frame_indices' not in data:
            return ('none', 0, 0)
        indices = data['frame_indices'].astype(int)
        if len(indices) == 0:
            return ('none', 0, 0)
        first, last = int(indices[0]), int(indices[-1])
        clip_end = (start_frame + max_frames - 1) if max_frames > 0 else last
        if first <= start_frame and last >= clip_end:
            return ('covers', first, last)
        return ('partial', first, last)
    except Exception:
        return ('none', 0, 0)


def _show_warning_popup(messages):
    """Display a modal warning dialog via BLENDCAP_OT_show_warnings.
    messages: list[str] - shown as labels with ERROR icons.
    Caller is responsible for returning {'CANCELLED'} after invoking."""
    blendcap_state['warnings'] = list(messages) if messages else []
    try:
        bpy.ops.blendcap.show_warnings('INVOKE_DEFAULT')
    except Exception as e:
        print(f"[BlendCap] Failed to show warning popup: {e}")
        for m in blendcap_state['warnings']:
            print(f"[BlendCap] WARNING: {m}")


def _write_cache_params(output_dir, start_frame, max_frames,
                        img_offset_start=0, img_offset_end=0,
                        project_fps=0.0, capture_sig="", source_path=""):
    """Write the frame parameters used for this run so cache validity
    can be checked later. Includes image-sequence offsets since IMAGE
    strips always have start_frame=0, max_frames=0. project_fps is
    written so re-running a project at a different framerate
    invalidates the NPZ (the data is sampled per project frame; a
    different FPS would misalign it). capture_sig (6th line) records the
    capture-affecting settings (FOV mode/focal, hand refinement, capture
    skip, backend) so changing any of them re-captures instead of reusing
    a mismatched NPZ. source_path (7th line) records the resolved source
    video (or image directory) so a DIFFERENT file that happens to
    resolve to the same clip name (same basename in another folder, a
    relinked strip) can never pass as a cache hit and silently reuse
    another video's tracking data. utf-8: paths can carry non-ASCII."""
    try:
        with open(os.path.join(output_dir, ".cache_params"), 'w',
                  encoding='utf-8') as f:
            f.write(f"{start_frame}\n{max_frames}\n{img_offset_start}\n{img_offset_end}\n{project_fps:.6f}\n{capture_sig}\n{source_path}\n")
    except Exception:
        pass


def _cache_params_match(output_dir, start_frame, max_frames,
                        img_offset_start=0, img_offset_end=0,
                        project_fps=0.0, capture_sig="", source_path=""):
    """Exact match: cache is valid only when it was generated with the
    SAME trim, SAME project FPS, SAME capture settings, and SAME source
    file as the current request. A wider OR narrower cache forces
    re-tracking: an earlier cache-covers-request shortcut produced
    overlay/NPZ misalignment when the current trim was much smaller than
    the cached range.

    Legacy caches without a recorded project FPS (4-line), without a
    capture-settings signature (5-line), or without a recorded source
    path (6-line) are treated as mismatched so they get regenerated and
    pick up the new metadata on the next run."""
    path = os.path.join(output_dir, ".cache_params")
    if not os.path.exists(path):
        return False
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
        if len(lines) < 5:
            return False  # legacy or truncated
        c_start = int(lines[0])
        c_max = int(lines[1])
        c_img_start = int(lines[2])
        c_img_end = int(lines[3])
        c_fps = float(lines[4])

        if (c_start, c_max, c_img_start, c_img_end) != \
           (start_frame, max_frames, img_offset_start, img_offset_end):
            return False
        # FPS comparison with float tolerance — project fps comes from
        # render.fps/render.fps_base and can carry tiny rounding noise.
        if abs(c_fps - project_fps) > 0.01:
            return False
        # Capture-settings signature (6th line). A cache without it (5-line,
        # pre-this-feature) reads as "" and mismatches any real signature, so
        # it regenerates once and records the line.
        c_sig = lines[5] if len(lines) >= 6 else ""
        if c_sig != capture_sig:
            return False
        # Source path (7th line). Same regenerate-once rule for 6-line
        # caches. Compared case/separator-insensitively (Windows paths);
        # a different file under the same clip name must never reuse
        # another video's tracking data.
        if source_path:
            c_src = lines[6] if len(lines) >= 7 else ""
            if not c_src:
                return False
            if (os.path.normcase(os.path.normpath(c_src))
                    != os.path.normcase(os.path.normpath(source_path))):
                return False
        return True
    except Exception:
        return False


