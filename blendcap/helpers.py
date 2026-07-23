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

def _python_exe():
    d = _addon_dir()
    if sys.platform == "win32":
        return os.path.join(d, "venv", "Scripts", "python.exe")
    return os.path.join(d, "venv", "bin", "python")


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
    for k, v in (extra_env or {}).items():
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


def _installer_dir():
    """Folder holding install.ps1 / install.bat / install.sh.

    Only present in PAID artifacts — the public GitHub source
    distribution ships the addon without it (see
    installer_files_present below)."""
    return os.path.join(_addon_dir(), "installer")


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
    """install_marker.json, written by the installer's final phase."""
    return os.path.join(_installer_dir(), "install_marker.json")


def read_install_marker():
    """Parsed install_marker.json as a dict, or None if absent/unreadable.

    The installer writes this from PowerShell, which on 5.1 can emit a
    UTF-8 BOM (or UTF-16); try a few encodings before giving up so a
    perfectly valid marker isn't missed over an encoding quirk."""
    p = _install_marker_path()
    if not os.path.isfile(p):
        return None
    for enc in ("utf-8-sig", "utf-16", "utf-8"):
        try:
            with open(p, "r", encoding=enc) as f:
                return json.load(f)
        except (UnicodeError, ValueError):
            continue
        except Exception:
            return None
    return None


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
            bpy.ops.object.mode_set(mode='OBJECT')
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


def _resolve_video_path(context):
    """Resolve the active video path and mode from UI state.
    Returns (video_path, start_frame, max_frames, insert_frame, insert_channel, mode) or None on error.
    """
    props = context.scene.blendcap_props
    start_frame = 0; max_frames = 0
    insert_frame = 1; insert_channel = 1

    if props.video_source == 'FILE':
        obj = context.active_object
        if obj and obj.type == 'EMPTY' and "blendcap_source" in obj:
            video_path = bpy.path.abspath(str(obj["blendcap_source"]))
        else:
            video_path = bpy.path.abspath(props.video_path)
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


def _resolve_clip_name(context):
    """Bare clip identifier (sanitized) used to build cache folder and
    proxy strip names.

    The Capture_ / Preview_ prefix is applied at the call site so this
    function returns just the source-side identifier.

    VSE mode: prefers the proxy's stored `blendcap_source_strip`
    custom prop (so re-capture stays in the original cache folder);
    otherwise the active strip's current name.

    FILE / viewport mode: prefers the active Empty's stored
    `blendcap_source_strip`; otherwise the file's basename stem.

    Returns "" if nothing can be resolved.
    """
    props = context.scene.blendcap_props
    raw = ""

    if props.video_source == 'FILE':
        obj = context.active_object
        if obj and obj.type == 'EMPTY' and "blendcap_source" in obj:
            stored = str(obj.get("blendcap_source_strip", "") or "")
            # Stored stamps are already extension-stripped at write time.
            raw = stored or _strip_media_extension(obj.name)
        else:
            vp = bpy.path.abspath(props.video_path) if props.video_path else ""
            if vp:
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
                        project_fps=0.0, capture_sig=""):
    """Write the frame parameters used for this run so cache validity
    can be checked later. Includes image-sequence offsets since IMAGE
    strips always have start_frame=0, max_frames=0. project_fps is
    written so re-running a project at a different framerate
    invalidates the NPZ (the data is sampled per project frame; a
    different FPS would misalign it). capture_sig (6th line) records the
    capture-affecting settings (FOV mode/focal, hand refinement, capture
    skip, backend) so changing any of them re-captures instead of reusing
    a mismatched NPZ."""
    try:
        with open(os.path.join(output_dir, ".cache_params"), 'w') as f:
            f.write(f"{start_frame}\n{max_frames}\n{img_offset_start}\n{img_offset_end}\n{project_fps:.6f}\n{capture_sig}\n")
    except Exception:
        pass


def _cache_params_match(output_dir, start_frame, max_frames,
                        img_offset_start=0, img_offset_end=0,
                        project_fps=0.0, capture_sig=""):
    """Exact match: cache is valid only when it was generated with the
    SAME trim, SAME project FPS, and SAME capture settings as the current
    request. A wider OR narrower cache forces re-tracking: an earlier
    cache-covers-request shortcut produced overlay/NPZ misalignment when
    the current trim was much smaller than the cached range.

    Legacy caches without a recorded project FPS (4-line) or without a
    capture-settings signature (5-line) are treated as mismatched so they
    get regenerated and pick up the new metadata on the next run."""
    path = os.path.join(output_dir, ".cache_params")
    if not os.path.exists(path):
        return False
    try:
        with open(path, 'r') as f:
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
        return True
    except Exception:
        return False


