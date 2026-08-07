"""Main pipeline operators: install dependencies, append workspace,
generate mocap (capture + preview), cancel, generate BVH.

The two big "command-builder" operators (BLENDCAP_OT_generate_mocap,
BLENDCAP_OT_generate_bvh) are the heart of the pipeline — they spawn
the threads in mocap.py.
"""
import bpy
import os
import sys
import shlex
import shutil
import subprocess
import threading
import textwrap
import time

from .helpers import (
    blendcap_state, bvh_state,
    _cache_dir, _bvh_library_dir, _addon_dir, _python_exe,
    _resolve_video_path, _resolve_clip_name, _strip_capture_prefix,
    _next_free_clip_name, _active_video_strip, _file_mode_source,
    source_mode_error,
    _strip_visible_start, _strip_visible_end,
    check_npz_coverage, _show_warning_popup,
    _cache_params_match, _SUPPORTED_VIDEO_EXTS,
    _installer_dir, _install_marker_path, _license_path,
    _host_spawn_prefix, _host_spawn_ok, _host_spawn_grant_cmd, _in_flatpak,
    _subprocess_env,
    read_install_marker, is_dependencies_installed,
    addon_prefs, _tag_redraw, _terminate_state_process,
)
from .mocap import (
    run_mocap_thread, run_bvh_thread,
    monitor_progress_timer, monitor_bvh_timer,
)


def _capture_use_ort_override(context):
    """USE_ORT value ('1'/'0') to force for the capture subprocess, or None
    to leave the inherited environment alone.

    Maps the Capture Backend add-on pref to the existing USE_ORT lever that
    save_mhr_data.py reads: the NVIDIA/CUDA path runs PyTorch with USE_ORT
    off; the non-NVIDIA path runs ONNX Runtime + DirectML with it on. AUTO
    matches whatever the installer set up (recorded in install_marker.json),
    so an AMD/Intel install works without the manual env-var step the README
    used to require."""
    prefs = addon_prefs(context)
    backend = getattr(prefs, "capture_backend", 'AUTO') if prefs else 'AUTO'
    if backend == 'GPU':
        return "0"
    if backend == 'COMPAT':
        return "1"
    if backend == 'CPU':
        # CPU-only: prefer the ORT backbone on the CPU execution provider —
        # measured several times faster than the PyTorch backbone on CPU
        # (Steam Deck test, 2026-06-11). The CPU pinning itself rides the
        # separate force_cpu flag (_capture_force_cpu), whose BLENDCAP_ORT_CPU
        # keeps the ORT session off the GPU. The ~1.7 GB backbone export is
        # fetched once by a pre-flight pipeline step when missing (see the
        # ensure-backbone block in invoke); if the fetch can't happen
        # (offline), save_mhr_data demotes itself back to the PyTorch
        # backbone — so this backend still has no HARD dependency on the
        # ONNX exports being installed.
        return "1"
    # AUTO: follow the backend the installer recorded at install time.
    marker = read_install_marker() or {}
    extra = (marker.get("extra") or "").lower()
    gtype = (marker.get("gpu_type") or "").lower()
    if extra == "directml" or gtype.startswith("amd"):
        return "1"
    if gtype in ("cpu-linux", "cpu-win"):
        # Box with no supported GPU: the installer warned and installed
        # the plain CPU package set (recorded extra is "mps" — the package
        # set, not the platform; gpu_type carries the truth). The fast CPU
        # path is the ORT backbone on the CPU EP, same engine the CPU Only
        # backend uses, so AUTO requests ORT here. The ONNX exports were
        # required at install for this gpu_type; if absent anyway, the
        # pre-flight fetch / PyTorch demote chain still covers it.
        return "1"
    if extra == "cuda" or gtype == "nvidia" or extra == "rocm" or gtype == "rocm":
        # rocm = AMD GPU via a ROCm build of PyTorch, which exposes the GPU
        # through the torch.cuda API -> the native PyTorch backbone path
        # (USE_ORT=0), same as NVIDIA. onnxruntime-rocm is NOT used (that EP is
        # discontinued upstream).
        return "0"
    return None


def _capture_force_cpu(context):
    """True when the Capture Backend pref selects CPU-only, so the capture
    subprocess pins BOTH torch and ONNX Runtime to the CPU. run_mocap_thread
    injects BLENDCAP_FORCE_CPU / BLENDCAP_ORT_CPU into the subprocess env when
    this is True. Reliable by construction - the addon writes the subprocess
    env directly, so it never depends on the user setting an OS env var before
    launching Blender (the route that silently failed in testing)."""
    prefs = addon_prefs(context)
    backend = getattr(prefs, "capture_backend", 'AUTO') if prefs else 'AUTO'
    return backend == 'CPU'


def _bvh_face_denoise_args(props):
    """Build the CLI args for BVH face denoising from the addon props.

    Property values and CLI flag both use [0,1] blendshape units; no
    conversion needed. Returns an empty list when the master toggle
    is off so the subprocess command stays uncluttered.
    """
    if not getattr(props, "bvh_face_denoise_enabled", False):
        return []
    args = ["--face-denoise-enabled"]
    for region in ("jaw", "gaze", "lids", "brows", "lips", "cheeks", "nose", "tongue"):
        val = float(getattr(props, f"bvh_face_denoise_{region}", 0.0))
        if val > 0:
            args += [f"--face-denoise-{region}", f"{val:.4f}"]
    return args


def _bvh_face_boost_value(props):
    """Effective face-boost value for the BVH export CLI.

    Forces 1.0 (no boost) when per-region Face Motion Scaling is on,
    since the two would compound. The UI greys the global multiplier
    in that case; this enforces the same behavior in the actual
    subprocess command.
    """
    if getattr(props, "bvh_face_amp_enabled", False):
        return 1.0
    return float(props.face_expression_multiplier)


def _bvh_face_amp_args(props):
    """Build the CLI args for BVH per-region face amplitude scaling.

    Returns empty list when the master toggle is off, or when every
    region is at 1.0 (1:1 pass-through, nothing to bake).
    """
    if not getattr(props, "bvh_face_amp_enabled", False):
        return []
    args = ["--face-amp-enabled"]
    any_non_unit = False
    for region in ("jaw", "gaze", "lids", "brows", "lips", "cheeks", "nose", "tongue"):
        val = float(getattr(props, f"bvh_face_amp_{region}", 1.0))
        if val != 1.0:
            args += [f"--face-amp-{region}", f"{val:.4f}"]
            any_non_unit = True
    return args if any_non_unit else []


def _bvh_depth_denoise_args(props):
    """Build the CLI args for BVH body depth-axis noise filtering.

    Sliders are in cm/frame; the CLI takes meters/frame. Returns an
    empty list when the master toggle is off or every region is 0
    (nothing to suppress).
    """
    if not getattr(props, "bvh_depth_denoise_enabled", False):
        return []
    args = []
    any_on = False
    for region in ("feet", "hands", "head", "root"):
        val_cm = float(getattr(props, f"bvh_depth_denoise_{region}", 0.0))
        if val_cm > 0.0:
            args += [f"--denoise-depth-{region}", f"{val_cm / 100.0:.4f}"]
            any_on = True
    return args if any_on else []


# ============================================================
# MAIN OPERATORS
# ============================================================
# Plain-English digest of the Meta SAM License key terms. The full text
# ships in licenses/SAM_LICENSE.txt and is one click away via the
# "View Full SAM License" button — this digest is the at-a-glance version,
# not a substitute for the agreement the user is accepting.
_SAM_LICENSE_SUMMARY = (
    "BlendCap's body tracking uses Meta's SAM 3D Body model, which is",
    "covered by the Meta SAM License. By installing, you agree to it:",
    "",
    "    - Commercial use is allowed, royalty-free.",
    "    - You may copy, modify and redistribute the model, but must",
    "      include a copy of the SAM License when you do.",
    "    - Prohibited uses: military / warfare, weapons, nuclear,",
    "      espionage, and anything barred by U.S. trade-control law.",
    "    - Provided \"as is\" - no warranty and no Meta support.",
)


class BLENDCAP_OT_open_license(bpy.types.Operator):
    bl_idname = "blendcap.open_license"
    bl_label = "View Full License"
    bl_description = "Open the full license text in your system's default viewer"
    bl_options = {'INTERNAL'}

    # Bundled license filename, resolved by helpers._license_path (root
    # for LICENSE/NOTICE.txt, licenses/ for the rest). Defaults to the
    # SAM License since that's the one the user is being asked to agree to.
    filename: bpy.props.StringProperty(default="SAM_LICENSE.txt")  # type: ignore

    def execute(self, context):
        path = _license_path(self.filename)
        if not os.path.exists(path):
            self.report({'ERROR'}, f"License file not found: {path}")
            return {'CANCELLED'}
        try:
            if sys.platform == "win32":
                os.startfile(path)  # noqa: Windows-only, guarded above
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            self.report({'ERROR'}, f"Couldn't open license file: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}


# ------------------------------------------------------------
# Install wizard (INLINE in the Add-on Preferences panel)
# ------------------------------------------------------------
# NOT a popup. Blender's popups can't give all of: a greyable button, control
# over button order, no clipping, and a working Cancel — invoke_popup has no
# button that closes it, and invoke_props_dialog won't let you grey or reorder
# its native footer buttons. So the wizard is drawn INLINE in the Add-on
# Preferences draw() (see draw_install_wizard below), where every button is an
# ordinary panel button: greyable via row.enabled, positioned by layout order,
# and never clipped (it just scrolls with the panel).
#
# Session state lives on WindowManager (registration.py): blendcap_install_
# active (wizard vs idle Install button), _page (INTRO/LICENSE/CONFIRM),
# _agreed (license checkbox), _force (reinstall vs first install). Forward/back
# are prop_enum on _page (in-place redraw); Cancel + the final Install are
# operators. The license gate is the greyed Continue button — it can't be
# clicked until _agreed is True, so the Confirm/Install page is unreachable
# without ticking the box. The viewport Setup button (panels.py) just jumps to
# this prefs section via blendcap.install_open_prefs.
class BLENDCAP_LicenseLine(bpy.types.PropertyGroup):
    """One display line of license text for the scrollable viewer."""
    text: bpy.props.StringProperty(default="")  # type: ignore


class BLENDCAP_UL_license_text(bpy.types.UIList):
    """Read-only scrollable viewer for a license file. Each item is one
    pre-wrapped line; the UIList supplies the scrollbar a column of labels
    in a popup can't."""
    bl_idname = "BLENDCAP_UL_license_text"

    def draw_item(self, context, layout, data, item, icon,
                  active_data, active_propname, index):
        # A blank string renders as a zero-height row; a single space keeps
        # paragraph breaks visible.
        layout.label(text=item.text if item.text else " ")

    def filter_items(self, context, data, propname):
        # Preserve file order, no name filter. Empty lists => show all,
        # don't reorder.
        return [], []


def _load_license_lines(context, filename="SAM_LICENSE.txt", width=58):
    """Read a bundled license file, soft-wrap it to `width` columns, and push
    the lines into the WindowManager collection the scrollable viewer binds
    to. Called when the wizard starts (BLENDCAP_OT_install_dependencies)."""
    wm = context.window_manager
    wm.blendcap_license_lines.clear()
    path = _license_path(filename)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except Exception:
        for msg in ("(Could not read the bundled license file at:",
                    " " + path + ")"):
            wm.blendcap_license_lines.add().text = msg
        return
    for line in raw.splitlines():
        if not line.strip():
            wm.blendcap_license_lines.add().text = ""
            continue
        for piece in (textwrap.wrap(line, width=width,
                                    break_long_words=False,
                                    break_on_hyphens=False) or [""]):
            wm.blendcap_license_lines.add().text = piece


# Approximate model + environment download size, and the free-disk figure we
# show users - the SAME 32 GB the documentation asks for, so the two sources
# never disagree. The install scripts enforce a lower silent floor (hard-fail
# below 12 GB, where the install genuinely can't fit; warn below 32) but all
# user-facing text speaks only in the documented 32 GB.
_INSTALL_DOWNLOAD_GB = 11
_INSTALL_FREE_SPACE_GB = 32


# --- inline wizard page bodies ---------------------------------------------
def _wizard_intro(layout):
    layout.label(text="Install AI Tracking Dependencies", icon='CONSOLE')
    layout.separator()
    col = layout.column(align=True)
    for line in (
        "BlendCap's tracking runs on a bundled, isolated Python",
        "environment plus a set of AI models (SAM 3D Body and",
        "supporting networks). This one-time setup will:",
        "",
        "    1. Build the bundled Python environment.",
        f"    2. Download the AI models (about {_INSTALL_DOWNLOAD_GB} GB).",
        "",
        "You'll need a working internet connection, a compatible GPU,",
        f"and at least {_INSTALL_FREE_SPACE_GB} GB of free disk space.",
    ):
        col.label(text=line)


def _wizard_license(context, layout):
    wm = context.window_manager
    layout.label(text="Meta SAM License Agreement", icon='TEXT')
    # License on the left (wider), summary digest on the right.
    split = layout.split(factor=0.62)
    left = split.column()
    left.label(text="Full license (scroll to read):")
    left.template_list("BLENDCAP_UL_license_text", "", wm,
                       "blendcap_license_lines", wm,
                       "blendcap_license_index", rows=14)
    right = split.column()
    right.label(text="Summary", icon='INFO')
    sbox = right.box().column(align=True)
    for line in _SAM_LICENSE_SUMMARY:
        sbox.label(text=line)

    layout.separator()
    layout.prop(wm, "blendcap_install_agreed")


def _wizard_confirm(layout, verb):
    layout.label(text=f"Ready to {verb}", icon='IMPORT')
    layout.separator()
    col = layout.column(align=True)
    lines = [
        f"Press {verb} to open a separate console window and:",
        "",
        "    - Build the bundled Python environment.",
        "    - Download the SAM 3D Body model and supporting",
        f"      AI models (about {_INSTALL_DOWNLOAD_GB} GB - needs internet).",
        "",
        "If they aren't already on your system, it also fetches",
        "these:",
        "    - uv (the Python environment manager)",
        "    - Git (used to fetch some components)",
    ]
    # The VC++ Redistributable is a Windows-only system component, and
    # the ONLY tool whose install needs permissions (uv and Git arrive
    # as portable app-local copies).
    if sys.platform == "win32":
        lines.append("    - Microsoft Visual C++ Redistributable")
        lines.append("      (if missing, Windows asks for permissions")
        lines.append("      in order to install it)")
    lines += [
        "",
        f"You'll need at least {_INSTALL_FREE_SPACE_GB} GB of free disk space.",
        "The first run can take a while. You can keep using Blender;",
        "BlendCap reports when it's ready.",
    ]
    for line in lines:
        col.label(text=line)


def _draw_flatpak_grant_warning(layout):
    """If running under Flatpak WITHOUT the host-spawn grant, draw a red
    alert with the one-time command the user must run on the host. Drawn at
    the top of the install wizard so it's seen before Install is pressed
    (the installer itself, and later every capture, run via flatpak-spawn
    and would fail without the grant). No-op on every other platform and
    once the grant is present (_host_spawn_ok is cached, so cheap to call
    from this draw path)."""
    if not _in_flatpak() or _host_spawn_ok():
        return
    box = layout.box()
    box.alert = True
    col = box.column(align=True)
    col.label(text="Flatpak: one-time permission needed", icon='ERROR')
    col.label(text="Blender is sandboxed. Run this in a terminal, then")
    col.label(text="restart Blender, before installing:")
    col.separator()
    cmd = box.box()
    cmd.label(text=_host_spawn_grant_cmd())
    box.operator("blendcap.copy_grant_cmd", text="Copy Command", icon='COPYDOWN')


class BLENDCAP_OT_copy_grant_cmd(bpy.types.Operator):
    bl_idname = "blendcap.copy_grant_cmd"
    bl_label = "Copy Flatpak Permission Command"
    bl_description = ("Copy the one-time 'flatpak override' command to the "
                      "clipboard so you can paste it into a host terminal")
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, context):
        context.window_manager.clipboard = _host_spawn_grant_cmd()
        self.report({'INFO'}, "Command copied - paste it in a terminal, then "
                              "restart Blender.")
        return {'FINISHED'}


def draw_install_wizard(context, layout):
    """Render the inline install wizard inside the Add-on Preferences panel.

    Real panel buttons throughout: greyable via row.enabled, ordered by layout,
    never clipped. Forward/back are prop_enum on the WindowManager page property
    (in-place redraw); Cancel and the final Install are operators. The greyed
    Continue button on the license page is the license gate.
    """
    wm = context.window_manager
    page = wm.blendcap_install_page
    # "Reinstall" only when an environment is actually present right now;
    # "Install" otherwise. Checked live (not via the entry button's flag) so
    # the label always reflects the real state.
    verb = "Reinstall" if is_dependencies_installed() else "Install"

    # Flatpak host-spawn grant must be in place before Install (the installer
    # runs on the host via flatpak-spawn). Shown above the wizard box.
    _draw_flatpak_grant_warning(layout)

    box = layout.box()
    if page == 'LICENSE':
        _wizard_license(context, box)
    elif page == 'CONFIRM':
        _wizard_confirm(box, verb)
    else:
        _wizard_intro(box)

    box.separator()
    footer = box.row(align=True)
    footer.scale_y = 1.3
    # Cancel — far left.
    footer.operator("blendcap.install_cancel", text="Cancel", icon='X')
    # Back — present past the first page.
    if page == 'LICENSE':
        footer.prop_enum(wm, "blendcap_install_page", 'INTRO',
                         text="Back", icon='BACK')
    elif page == 'CONFIRM':
        footer.prop_enum(wm, "blendcap_install_page", 'LICENSE',
                         text="Back", icon='BACK')
    # Forward — far right.
    fwd = footer.row(align=True)
    if page == 'INTRO':
        fwd.prop_enum(wm, "blendcap_install_page", 'LICENSE',
                      text="Continue", icon='FORWARD')
    elif page == 'LICENSE':
        # Greyed until the agree box is ticked — the hard license gate.
        fwd.enabled = wm.blendcap_install_agreed
        fwd.prop_enum(wm, "blendcap_install_page", 'CONFIRM',
                      text="Continue", icon='FORWARD')
    else:  # CONFIRM
        fwd.operator("blendcap.install_run", text=verb, icon='IMPORT')


class BLENDCAP_OT_install_dependencies(bpy.types.Operator):
    bl_idname = "blendcap.install_dependencies"
    bl_label = "Install AI Dependencies"
    bl_description = ("Set up BlendCap's AI tracking dependencies (a bundled "
                      "Python environment + the AI models). Opens an inline "
                      "wizard: intro, the Meta SAM License, then a "
                      "confirmation before running the installer. Skips if "
                      "already installed.")
    bl_options = {'REGISTER'}

    # True for the reinstall path (bypasses the already-installed skip).
    force: bpy.props.BoolProperty(  # type: ignore
        default=False, options={'HIDDEN', 'SKIP_SAVE'})

    def execute(self, context):
        # Start (or restart) the inline wizard at the intro page.
        if not self.force and is_dependencies_installed():
            self.report({'INFO'},
                        "AI dependencies already installed - nothing to do.")
            return {'CANCELLED'}
        # macOS can't be driven from here: Gatekeeper blocks launching the
        # installer from Blender, so point the user at the manual path.
        if sys.platform == "darwin":
            self.report({'ERROR'},
                        "On macOS, run installer/install.sh from a terminal "
                        "yourself (Gatekeeper blocks launching it from Blender).")
            return {'CANCELLED'}
        wm = context.window_manager
        wm.blendcap_install_force = self.force
        wm.blendcap_install_agreed = False
        wm.blendcap_install_page = 'INTRO'
        _load_license_lines(context, "SAM_LICENSE.txt")
        wm.blendcap_install_active = True
        # The Preferences window doesn't auto-redraw when an operator changes a
        # WindowManager prop, so the panel would keep showing the idle view
        # ("nothing happens") without this nudge.
        _tag_redraw('PREFERENCES')
        return {'FINISHED'}


class BLENDCAP_OT_install_cancel(bpy.types.Operator):
    bl_idname = "blendcap.install_cancel"
    bl_label = "Cancel"
    bl_description = "Close the installer without installing anything"
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, context):
        wm = context.window_manager
        wm.blendcap_install_active = False
        wm.blendcap_install_page = 'INTRO'
        wm.blendcap_install_agreed = False
        _tag_redraw('PREFERENCES')
        return {'FINISHED'}


class BLENDCAP_OT_install_run(bpy.types.Operator):
    bl_idname = "blendcap.install_run"
    bl_label = "Run Installer"
    bl_description = "Launch the BlendCap dependency installer"
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, context):
        installer_dir = _installer_dir()
        launcher = os.path.join(
            installer_dir, "install.bat" if sys.platform == "win32" else "install.sh")
        if not os.path.exists(launcher):
            self.report({'ERROR'}, f"Installer not found: {launcher}")
            return {'CANCELLED'}

        # Under Flatpak the installer runs on the host via flatpak-spawn;
        # without the one-time grant it fails silently (portal error). Block
        # here with the exact fix rather than launching into a no-op.
        if _in_flatpak() and not _host_spawn_ok():
            self.report({'ERROR'},
                        "Flatpak needs a one-time permission first. Run this "
                        "in a terminal, then restart Blender:  "
                        + _host_spawn_grant_cmd())
            return {'CANCELLED'}

        try:
            if sys.platform == "win32":
                # Launch in its own console so the user can watch progress;
                # install.bat runs as the invoking user and pauses when
                # finished (only a missing VC++ runtime triggers a Windows
                # permission prompt, from inside install.ps1).
                os.startfile(launcher)  # noqa: Windows-only, guarded at intro
            else:
                # Linux: launch in a VISIBLE terminal so install.sh's own
                # git/sudo bootstrap can prompt for a password and the user
                # can watch the multi-GB download. A blind detached process
                # (the old behaviour) can't prompt for anything, which is why
                # the git step silently failed. Under Flatpak the sandbox has
                # no git/toolchain, so the whole thing runs on the HOST via
                # flatpak-spawn (prefix is [] on native Linux); the host
                # terminal needs the GUI env (DISPLAY etc.) forwarded.
                # IMPORTANT: bash, not sh. install.sh uses bash features
                # (set -o pipefail, [[ ]], process substitution); on
                # Debian/Ubuntu/Mint /bin/sh is dash, which aborts at line 1
                # of pipefail and writes nothing - the old Popen(["/bin/sh",
                # launcher]) silently died there.
                run = 'cd %s; bash ./install.sh' % shlex.quote(installer_dir)
                # Try common terminals in order; fall back to a headless
                # detached run (servers / minimal images with no terminal).
                pick = (
                    'SUF="; echo; echo \\"[BlendCap] Installer finished. '
                    'Press Enter to close.\\"; read _"; '
                    'if command -v gnome-terminal >/dev/null 2>&1; then '
                    'exec gnome-terminal -- bash -lc "$BLENDCAP_RUN$SUF"; '
                    'elif command -v x-terminal-emulator >/dev/null 2>&1; then '
                    'exec x-terminal-emulator -e bash -lc "$BLENDCAP_RUN$SUF"; '
                    'elif command -v konsole >/dev/null 2>&1; then '
                    'exec konsole -e bash -lc "$BLENDCAP_RUN$SUF"; '
                    'elif command -v xfce4-terminal >/dev/null 2>&1; then '
                    'exec xfce4-terminal -x bash -lc "$BLENDCAP_RUN$SUF"; '
                    'elif command -v xterm >/dev/null 2>&1; then '
                    'exec xterm -e bash -lc "$BLENDCAP_RUN$SUF"; '
                    'else bash -lc "$BLENDCAP_RUN"; fi'
                )
                launch_env = {"BLENDCAP_RUN": run}
                # Forward the display env so a host terminal can open under
                # Flatpak (flatpak-spawn gives the host process a bare env).
                # Deliberately NOT forwarding XAUTHORITY: in the sandbox it
                # points at /run/flatpak/Xauthority, which doesn't exist on
                # the host. A host login shell (bash -lc) resolves the host's
                # own ~/.Xauthority, which is the valid cookie for :0.
                for var in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR"):
                    val = os.environ.get(var)
                    if val:
                        launch_env[var] = val
                prefix = _host_spawn_prefix(directory=installer_dir,
                                            extra_env=launch_env)
                if prefix:
                    subprocess.Popen(prefix + ["bash", "-lc", pick],
                                     cwd=installer_dir)
                else:
                    # Native Linux: pass BLENDCAP_RUN through the Popen env.
                    # _subprocess_env strips the Steam runtime's library
                    # overrides when Blender was launched through Steam -
                    # without that, the terminal emulator itself (and every
                    # host tool install.sh runs) starts against the Steam
                    # runtime's old libs and can fail or misdetect the GPU.
                    native_env = _subprocess_env({"BLENDCAP_RUN": run})
                    subprocess.Popen(["bash", "-lc", pick],
                                     cwd=installer_dir, env=native_env)
        except Exception as e:
            self.report({'ERROR'}, f"Couldn't launch the installer: {e}")
            return {'CANCELLED'}

        # Collapse the wizard back to the idle prefs view.
        wm = context.window_manager
        wm.blendcap_install_active = False
        wm.blendcap_install_page = 'INTRO'
        _tag_redraw('PREFERENCES')
        self.report({'INFO'},
                    "Installer launched in a new window. First run downloads "
                    "several GB and can take a while.")
        return {'FINISHED'}


class BLENDCAP_OT_install_open_prefs(bpy.types.Operator):
    bl_idname = "blendcap.install_open_prefs"
    bl_label = "Open BlendCap Preferences"
    bl_description = ("Open the Add-on Preferences, where the AI dependency "
                      "installer lives")
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, context):
        # The AddonPreferences (and the wizard) live in the ROOT package, one
        # level up from this blendcap/ subpackage.
        root_pkg = (__package__ or "").rpartition('.')[0]
        try:
            bpy.ops.preferences.addon_show(module=root_pkg)
            return {'FINISHED'}
        except Exception:
            pass
        # Fallback: open the Preferences window at the Add-ons section.
        try:
            bpy.ops.screen.userpref_show('INVOKE_DEFAULT')
            context.preferences.active_section = 'ADDONS'
        except Exception:
            self.report({'INFO'},
                        "Open Edit > Preferences > Add-ons > BlendCap to "
                        "install dependencies.")
        return {'FINISHED'}


def _rm_readonly(func, path, _exc):
    """shutil.rmtree onerror hook: clear the read-only bit and retry the
    delete. venvs and bundled git checkouts contain read-only files that
    rmtree otherwise can't remove on Windows. Swallows anything it still
    can't delete - the caller re-checks whether the directory survived."""
    import stat
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


class BLENDCAP_OT_uninstall_dependencies(bpy.types.Operator):
    bl_idname = "blendcap.uninstall_dependencies"
    bl_label = "Uninstall AI Dependencies"
    bl_description = ("Remove BlendCap's installed AI dependencies - the "
                      "bundled Python environment and the downloaded AI "
                      "models - to free up disk space. You can reinstall "
                      "them later from this panel")
    bl_options = {'REGISTER', 'INTERNAL'}

    # ARM shows the inline confirm block, CANCEL hides it, RUN performs the
    # delete. Inline (driven via the prefs panel) rather than a popup because
    # dialogs launched from the Preferences window get clipped.
    mode: bpy.props.EnumProperty(  # type: ignore
        items=[('ARM', "Arm", ""), ('RUN', "Run", ""), ('CANCEL', "Cancel", "")],
        default='ARM', options={'HIDDEN', 'SKIP_SAVE'})

    def execute(self, context):
        wm = context.window_manager
        if self.mode == 'ARM':
            wm.blendcap_uninstall_armed = True
            _tag_redraw('PREFERENCES')
            return {'FINISHED'}
        if self.mode == 'CANCEL':
            wm.blendcap_uninstall_armed = False
            _tag_redraw('PREFERENCES')
            return {'FINISHED'}

        # mode == 'RUN': perform the delete.
        wm.blendcap_uninstall_armed = False
        if blendcap_state.get('running') or bvh_state.get('running'):
            self.report({'ERROR'},
                        "Finish or cancel the running job before uninstalling.")
            _tag_redraw('PREFERENCES')
            return {'CANCELLED'}

        addon_dir = _addon_dir()
        targets = [os.path.join(addon_dir, "venv"),
                   os.path.join(addon_dir, "checkpoints")]
        removed, failed = [], []
        for path in targets:
            if not os.path.isdir(path):
                continue
            try:
                shutil.rmtree(path, onerror=_rm_readonly)
            except Exception as e:
                failed.append(f"{os.path.basename(path)} ({e})")
                continue
            # onerror swallows per-file failures, so confirm the dir is gone.
            if os.path.isdir(path):
                failed.append(f"{os.path.basename(path)} (some files remain)")
            else:
                removed.append(os.path.basename(path))

        # Drop the marker so the panel resets to "Install" and stale GPU /
        # version details clear (is_dependencies_installed keys off the venv,
        # but the marker still drives the display text).
        try:
            marker = _install_marker_path()
            if os.path.isfile(marker):
                os.remove(marker)
        except Exception:
            pass

        _tag_redraw('PREFERENCES')

        if failed:
            self.report({'ERROR'},
                        "Uninstall incomplete - couldn't remove "
                        + "; ".join(failed)
                        + ". Close anything using those files and retry.")
            return {'CANCELLED'}
        if not removed:
            self.report({'INFO'}, "Nothing to uninstall - no files found.")
            return {'CANCELLED'}
        self.report({'INFO'},
                    "Uninstalled " + " + ".join(removed)
                    + ". Reinstall from this panel when you need capture again.")
        return {'FINISHED'}


class BLENDCAP_OT_deps_missing(bpy.types.Operator):
    """Friendly blocking dialog shown when a capture is started before the AI
    dependencies are installed. Beats a cryptic 'Cannot find Python' report —
    it explains what's missing and offers a one-click jump to the installer."""
    bl_idname = "blendcap.deps_missing"
    bl_label = "AI Dependencies Required"
    bl_options = {'REGISTER', 'INTERNAL'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="BlendCap's AI tracking isn't installed yet.",
                  icon='ERROR')
        col.separator()
        col.label(text="Capturing needs a one-time setup: a bundled Python")
        col.label(text="environment and the AI models (about "
                       f"{_INSTALL_DOWNLOAD_GB} GB).")
        col.separator()
        col.label(text="Open the Add-on Preferences and click")
        col.label(text="\"Install Dependencies\", then try again.")
        layout.separator()
        layout.operator("blendcap.install_open_prefs",
                        text="Open Preferences", icon='PREFERENCES')

    def execute(self, context):
        return {'FINISHED'}


class BLENDCAP_OT_append_workspace(bpy.types.Operator):
    bl_idname = "blendcap.append_workspace"
    bl_label = "Load Mocap Workspace"
    bl_description = ("Load the bundled BlendCap workspace, or switch "
                      "to it if it's already present")

    # Accepted workspace tab names. Users sometimes rename their
    # BlendCap workspace tab to taste; the operator activates whichever
    # match exists in the current file, and falls through to loading
    # whichever match exists in the shipped blendcap_workspace.blend.
    # Order matters only for tie-breaking when multiple are present.
    _ACCEPTED_NAMES = (
        "Performance Capture",
        "Motion Capture",
        "MoCap",
        "Mocap",
        "BlendCap",
        "Blendcap",
        "Capture",
    )

    def execute(self, context):
        # 1. If any accepted workspace is already in this file, activate it.
        for name in self._ACCEPTED_NAMES:
            if name in bpy.data.workspaces:
                context.window.workspace = bpy.data.workspaces[name]
                self.report({'INFO'},
                            f"Workspace '{name}' already loaded.")
                return {'FINISHED'}

        # 2. Otherwise load from the shipped workspace .blend.
        ws_file = os.path.join(_addon_dir(), "blendcap_workspace.blend")
        if not os.path.exists(ws_file):
            self.report({'ERROR'}, f"Workspace file not found at: {ws_file}")
            return {'CANCELLED'}

        matched = None
        try:
            with bpy.data.libraries.load(ws_file, link=False) as (data_from, data_to):
                for name in self._ACCEPTED_NAMES:
                    if name in data_from.workspaces:
                        matched = name
                        break
                if matched:
                    data_to.workspaces = [matched]
        except Exception as e:
            self.report({'ERROR'}, f"Failed to append workspace: {e}")
            return {'CANCELLED'}

        if matched is None:
            self.report({'ERROR'},
                        f"No matching workspace found in {ws_file}. "
                        f"Expected one of: "
                        f"{', '.join(self._ACCEPTED_NAMES)}")
            return {'CANCELLED'}

        if matched in bpy.data.workspaces:
            context.window.workspace = bpy.data.workspaces[matched]
            self.report({'INFO'}, f"Workspace '{matched}' loaded.")
        return {'FINISHED'}


def _append_frame_args(cmd, start_frame, max_frames):
    if start_frame > 0:
        cmd.extend(["--start-frame", str(start_frame)])
    if max_frames > 0:
        cmd.extend(["--max-frames", str(max_frames)])


def _classify_step(script_name, cmd, weight_map):
    """Return (weight, label, output_path) for a pipeline step."""
    out_idx = cmd.index("--output") + 1 if "--output" in cmd else -1
    out_path = cmd[out_idx] if 0 < out_idx < len(cmd) else ""
    for key, (w, lbl) in weight_map.items():
        if key in script_name:
            return w, lbl, out_path
    return 10, "other", out_path


def _probe_video_frames(video_path, max_frames=0):
    """Probe video frame count using Blender's movieclip loader. Returns 0 on failure."""
    try:
        clip = bpy.data.movieclips.load(video_path, check_existing=False)
        frames = max_frames if max_frames > 0 else clip.frame_duration
        bpy.data.movieclips.remove(clip)
        return frames
    except Exception:
        return 0


def _is_supported_video(video_path):
    """Check if the video format is likely supported by OpenCV (by extension)."""
    ext = os.path.splitext(video_path)[1].lower()
    return ext in _SUPPORTED_VIDEO_EXTS


def _detect_fps_mismatch(context, video_path):
    """Check if a source video's FPS differs from project FPS.
    Returns (source_fps, project_fps, source_frame_count) if mismatch > 0.5, else None.
    """
    scene = context.scene
    project_fps = scene.render.fps / scene.render.fps_base
    source_fps = 0
    source_frames = 0
    if video_path and os.path.exists(video_path):
        try:
            clip = bpy.data.movieclips.load(video_path, check_existing=False)
            source_fps = clip.fps if clip.fps > 0 else 0
            source_frames = getattr(clip, 'frame_duration', 0) or 0
            bpy.data.movieclips.remove(clip)
        except Exception:
            pass
    if source_fps > 0 and abs(source_fps - project_fps) > 0.5:
        return (source_fps, project_fps, source_frames)
    return None




def _snapshot_strip_props(strip):
    """Capture strip properties for replacement. Returns dict or None."""
    if not strip:
        return None
    final_start = _strip_visible_start(strip)
    final_end = _strip_visible_end(strip)
    props = {
        'strip_name': strip.name,
        'strip_type': strip.type,
        'frame_final_start': final_start,
        'frame_final_end': final_end,
        'channel': strip.channel,
    }

    if strip.type == 'IMAGE':
        props['directory'] = bpy.path.abspath(strip.directory)
        props['elements'] = [e.filename for e in strip.elements] if hasattr(strip, 'elements') else []
        try:
            props['image_frame_start'] = int(strip.frame_start)
            props['image_offset_start'] = int(strip.frame_offset_start)
            props['image_offset_end'] = int(getattr(strip, 'frame_offset_end', 0))
        except Exception:
            props['image_frame_start'] = final_start
            props['image_offset_start'] = 0
            props['image_offset_end'] = 0
    else:
        # MOVIE strip — all properties deprecated in Blender 5.1+
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            props['offset_start'] = int(strip.frame_offset_start)
            props['offset_end'] = int(getattr(strip, 'frame_offset_end', 0))
            props['frame_final_duration'] = int(strip.frame_final_duration)
            props['frame_start'] = int(strip.frame_start)
            props['frame_duration'] = int(strip.frame_duration)

    # If this is already a proxy, carry forward the original properties
    if "blendcap_source" in strip:
        orig_type = str(strip.get("blendcap_original_type", "MOVIE"))
        props['strip_type'] = orig_type
        # Carry forward the stored original channel (proxy may be on a
        # different channel if Blender auto-placed it due to overlap)
        stored_channel = strip.get("blendcap_original_channel")
        if stored_channel is not None:
            props['channel'] = int(stored_channel)
        if orig_type == 'IMAGE':
            props['directory'] = str(strip.get("blendcap_original_directory", ""))
            elem_str = str(strip.get("blendcap_original_elements", ""))
            props['elements'] = elem_str.split("|") if elem_str else []
            props['frame_final_end'] = int(strip.get("blendcap_original_frame_final_end", final_end))
            props['image_frame_start'] = int(strip.get("blendcap_original_image_frame_start", final_start))
            props['image_offset_start'] = int(strip.get("blendcap_original_image_offset_start", 0))
            props['image_offset_end'] = int(strip.get("blendcap_original_image_offset_end", 0))
        else:
            props['offset_start'] = int(strip.get("blendcap_original_offset_start", 0))
            props['offset_end'] = int(strip.get("blendcap_original_offset_end", 0))
            props['frame_final_duration'] = int(strip.get("blendcap_original_duration", 0))
            props['frame_final_start'] = int(strip.get("blendcap_original_vis_start", final_start))
            props['frame_final_end'] = int(strip.get("blendcap_original_vis_end", final_end))
            props['frame_start'] = int(strip.get("blendcap_original_frame_start",
                                                  final_start - props['offset_start']))
            props['frame_duration'] = int(strip.get("blendcap_original_frame_duration", 0))
    return props


class BLENDCAP_OT_generate_mocap(bpy.types.Operator):
    bl_idname = "blendcap.generate_mocap"
    bl_label = "Generate Mocap & Preview"
    is_update_only: bpy.props.BoolProperty(default=False)

    @classmethod
    def description(cls, context, properties):
        if properties.is_update_only:
            return ("Run a stepped, lower-quality preview of the "
                    "selected tracking methods")
        return ("Run the full motion-capture process with the "
                "selected tracking methods")
    fps_mode: bpy.props.EnumProperty(
        name="Conversion",
        items=[
            ('MATCH', "Match Timing", "Drop or duplicate frames to match project speed"),
            ('KEEP', "Keep All Frames", "1:1 frame copy — changes playback speed"),
            ('SKIP', "Skip Conversion", "No conversion — track original video directly"),
        ],
        default='MATCH'
    )
    collision_mode: bpy.props.EnumProperty(
        name="Existing Data",
        items=[
            ('NEW', "Capture as New Clip",
             "Rename this strip and capture into its own cache folder, "
             "keeping the existing capture"),
            ('OVERWRITE', "Overwrite Existing",
             "Replace the capture data already cached under this name"),
        ],
        default='NEW'
    )
    # Set by invoke() when the cache-collision warning fired, so execute()
    # only acts on collision_mode after the user actually saw the dialog.
    collision_detected: bpy.props.BoolProperty(
        default=False, options={'HIDDEN', 'SKIP_SAVE'})
    collision_new_name: bpy.props.StringProperty(
        default="", options={'HIDDEN', 'SKIP_SAVE'})

    def invoke(self, context, event):
        # Block early with a clear, actionable dialog if the AI dependencies
        # aren't installed — otherwise the run fails deep in the pipeline with
        # a cryptic "Cannot find Python" message.
        if not is_dependencies_installed():
            bpy.ops.blendcap.deps_missing('INVOKE_DEFAULT')
            return {'CANCELLED'}
        props = context.scene.blendcap_props
        if props.track_body or props.track_face:
            resolved = _resolve_video_path(context)
            if resolved and resolved[0]:
                video_path = resolved[0]
                max_frames = resolved[2]

                # Check frame count
                warnings = []
                if os.path.exists(video_path):
                    n_frames = _probe_video_frames(video_path, max_frames)
                elif max_frames > 0:
                    n_frames = max_frames
                else:
                    n_frames = 0
                if n_frames > 1000:
                    warnings.append(('frames', n_frames))

                # Check FPS mismatch (VSE strips and single-file selector)
                if resolved[5] in ('VSE', 'FILE'):
                    mismatch = _detect_fps_mismatch(context, video_path)
                    if mismatch:
                        warnings.append(('fps', mismatch[0], mismatch[1]))

                # Cache-collision guard: a strip with no BlendCap identity
                # whose name resolves to a cache folder that already holds
                # another capture's tracking data. Happens when the same
                # file is re-added to the timeline — the first capture
                # renamed the original strip away (to Capture_*), so the
                # newcomer silently reuses the freed name and would
                # overwrite that clip's cache. Stamped strips (proxies,
                # reverted originals) are known identities and skip this.
                if resolved[5] == 'VSE':
                    seq_ed = context.scene.sequence_editor
                    strip = _active_video_strip(seq_ed)
                    if (strip and strip.type in ('MOVIE', 'IMAGE')
                            and "blendcap_source" not in strip
                            and not str(strip.get("blendcap_source_strip",
                                                  "") or "")):
                        clip_name = _resolve_clip_name(context)
                        folder = os.path.join(_cache_dir(),
                                              f"Capture_{clip_name}")
                        has_data = clip_name and (
                            os.path.exists(os.path.join(
                                folder, "mhr_frames.npz"))
                            or os.path.exists(os.path.join(
                                folder, "face_frames.npz")))
                        if has_data:
                            self.collision_detected = True
                            self.collision_new_name = _next_free_clip_name(
                                context, strip.name)
                            warnings.append(('collision', clip_name,
                                             self.collision_new_name))

                if warnings:
                    blendcap_state['_invoke_warnings'] = warnings
                    return context.window_manager.invoke_props_dialog(self, width=420)
        return self.execute(context)

    def draw(self, context):
        layout = self.layout
        warnings = blendcap_state.get('_invoke_warnings', [])
        for warn in warnings:
            if warn[0] == 'frames':
                layout.label(text=f"This clip is {warn[1]} frames long.", icon='ERROR')
                layout.label(text="Processing may take a long time and results can be hard to predict.")
                layout.separator()
            elif warn[0] == 'fps':
                clip_fps, proj_fps = warn[1], warn[2]
                layout.label(text=f"Framerate mismatch: clip {clip_fps:.0f}fps, project {proj_fps:.0f}fps.", icon='INFO')
                layout.prop(self, "fps_mode")
                if self.fps_mode == 'MATCH':
                    if clip_fps > proj_fps:
                        layout.label(text="Drops frames to match project timing.")
                    else:
                        layout.label(text="Duplicates frames to match project timing.")
                elif self.fps_mode == 'KEEP':
                    if clip_fps > proj_fps:
                        layout.label(text="Plays in slow motion (all frames preserved).")
                    else:
                        layout.label(text="Plays in fast motion (all frames preserved).")
                else:
                    layout.label(text="Original video passed directly to tracking.")
                layout.separator()
            elif warn[0] == 'collision':
                clip_label, new_name = warn[1], warn[2]
                layout.label(text=f"Capture data for '{clip_label}' already exists.",
                             icon='ERROR')
                layout.label(text="Choose an action:")
                layout.prop(self, "collision_mode", text="")
                if self.collision_mode == 'NEW':
                    layout.label(text=f"The strip will be renamed to '{new_name}'.")
                else:
                    layout.label(text=f"The cached data for '{clip_label}' will be replaced.")
                layout.separator()
        layout.label(text="Continue?")

    def execute(self, context):
        blendcap_state.pop('_invoke_warnings', None)

        if blendcap_state['running']:
            self.report({'WARNING'}, "A capture is already running. Wait for it to finish.")
            return {'CANCELLED'}

        props = context.scene.blendcap_props
        if not props.track_body and not props.track_face:
            self.report({'ERROR'}, "Enable at least one tracking type (Body or Face).")
            return {'CANCELLED'}

        # Resolve video source
        resolved = _resolve_video_path(context)
        if resolved is None:
            self.report({'ERROR'}, source_mode_error(context))
            return {'CANCELLED'}

        video_path = resolved[0]
        start_frame, max_frames = resolved[1], resolved[2]
        insert_frame, insert_channel = resolved[3], resolved[4]
        mode = resolved[5]

        # Collision-guard decision from invoke(): fork this strip into its
        # own clip identity instead of overwriting the same-named capture.
        # Renaming BEFORE _resolve_clip_name is what routes the run into
        # the fresh cache folder.
        if (self.collision_detected and mode == 'VSE'
                and self.collision_mode == 'NEW' and self.collision_new_name):
            seq_ed = context.scene.sequence_editor
            strip = _active_video_strip(seq_ed)
            if strip and strip.type in ('MOVIE', 'IMAGE'):
                strip.name = self.collision_new_name
                print(f"[BlendCap] Strip renamed to '{strip.name}' to "
                      f"protect the existing capture.")

        # Per-clip cache folder name. Derived from the active VSE strip
        # (or BlendCap Empty in file mode), not the file basename, so a
        # Blender-split source produces separate caches per split.
        clip_name = _resolve_clip_name(context) or "sequence"
        vid_name = f"Capture_{clip_name}"

        # Snap playhead to active strip so the user sees which clip is being
        # processed - unless they turned it off (Disable playhead auto-snap).
        if mode == 'VSE':
            _prefs = addon_prefs(context)
            if not (_prefs and _prefs.disable_timeline_autosnap):
                context.scene.frame_current = int(insert_frame)

        # Determine if conversion is needed: image sequences, unsupported formats, or FPS mismatch
        is_image_seq = False
        fps_mismatch = False
        source_fps = 0
        convert_cmd = None
        img_elements = []
        img_offset_start = 0
        img_offset_end = 0

        if mode == 'VSE':
            seq_ed = context.scene.sequence_editor
            strip = _active_video_strip(seq_ed)
            if strip:
                # Check for IMAGE strip or IMAGE-origin proxy
                orig_type = str(strip.get("blendcap_original_type", strip.type))
                if strip.type == 'IMAGE' or orig_type == 'IMAGE':
                    is_image_seq = True
                    if strip.type == 'IMAGE':
                        # Direct IMAGE strip — read elements and trim
                        if hasattr(strip, 'elements'):
                            img_elements = [e.filename for e in strip.elements]
                        import warnings as _w
                        with _w.catch_warnings():
                            _w.simplefilter("ignore", DeprecationWarning)
                            try:
                                img_offset_start = int(strip.frame_offset_start)
                                img_offset_end = int(getattr(strip, 'frame_offset_end', 0))
                            except Exception:
                                pass
                    else:
                        # IMAGE-origin proxy — read from stored custom properties
                        elem_str = str(strip.get("blendcap_original_elements", ""))
                        img_elements = elem_str.split("|") if elem_str else []
                        img_offset_start = int(strip.get("blendcap_original_image_offset_start", 0))
                        img_offset_end = int(strip.get("blendcap_original_image_offset_end", 0))
                elif strip.type == 'MOVIE':
                    mismatch = _detect_fps_mismatch(context, video_path)
                    if mismatch:
                        source_fps = mismatch[0]
                        project_fps_val = mismatch[1]
                        source_frames = mismatch[2]

                        if self.fps_mode != 'SKIP':
                            fps_mismatch = True
                            mode_label = "match timing" if self.fps_mode == 'MATCH' else "keep all frames"
                        else:
                            mode_label = "skip"
                        print(f"[BlendCap] FPS mismatch: source={source_fps:.1f}fps, "
                              f"project={project_fps_val:.1f}fps. Mode: {mode_label}.")

                        # Detect whether strip timing is in project-space or
                        # source-space, then scale start_frame/max_frames to
                        # source-space. This applies to ALL modes — even SKIP
                        # needs source-space values for the tracking scripts.
                        #
                        # For proxy re-runs: use blendcap_original_frame_duration
                        # (the original strip's frame_duration, stored at first
                        # run) instead of the proxy's own frame_duration.
                        is_proxy = "blendcap_source" in strip
                        if is_proxy:
                            strip_dur = int(strip.get("blendcap_original_frame_duration", 0))
                        else:
                            import warnings as _ws
                            with _ws.catch_warnings():
                                _ws.simplefilter("ignore", DeprecationWarning)
                                strip_dur = int(strip.frame_duration)

                        if source_frames > 0 and strip_dur > 0 and abs(strip_dur - source_frames) <= 1:
                            # Source-space: strip duration matches actual source
                            # frame count. No scaling needed.
                            print(f"[BlendCap] Strip timing in source-space "
                                  f"(strip_dur={strip_dur}, source={source_frames}). "
                                  f"No scaling needed.")
                        elif strip_dur > 0:
                            # Project-space (strip_dur differs from source frames)
                            # or probe unreliable (source_frames=0). Scale both
                            # start_frame and max_frames to source-space.
                            ratio = source_fps / project_fps_val
                            start_frame = int(round(start_frame * ratio))
                            max_frames = int(round(max_frames * ratio)) if max_frames > 0 else 0
                            print(f"[BlendCap] Scaled to source-space "
                                  f"(strip_dur={strip_dur}, source={source_frames}): "
                                  f"ratio={ratio:.3f}, start={start_frame}, "
                                  f"max={max_frames}")
                        else:
                            # No frame_duration available at all (shouldn't happen
                            # but guard against it). Use FPS ratio.
                            ratio = source_fps / project_fps_val
                            start_frame = int(round(start_frame * ratio))
                            max_frames = int(round(max_frames * ratio)) if max_frames > 0 else 0
                            print(f"[BlendCap] No strip_dur available, "
                                  f"fallback ratio={ratio:.3f}, start={start_frame}, "
                                  f"max={max_frames}")
        elif mode == 'FILE':
            # Single-file selector: detect FPS mismatch the same way the
            # VSE path does. There's no strip trim to scale into source
            # space — the whole file is tracked (start_frame/max_frames
            # stay 0), so the convert script handles the entire clip.
            mismatch = _detect_fps_mismatch(context, video_path)
            if mismatch and self.fps_mode != 'SKIP':
                source_fps = mismatch[0]
                project_fps_val = mismatch[1]
                fps_mismatch = True
                mode_label = ("match timing" if self.fps_mode == 'MATCH'
                              else "keep all frames")
                print(f"[BlendCap] FPS mismatch: source={source_fps:.1f}fps, "
                      f"project={project_fps_val:.1f}fps. Mode: {mode_label}.")

        if not is_image_seq and (not video_path or not os.path.exists(video_path)):
            # source_mode_error names the mode being read and, when the OTHER
            # source has a valid selection, points at the switch the user
            # likely forgot. VSE-with-active-strip resolves to the broken-link
            # "Original video not found" message inside the helper.
            self.report({'ERROR'}, source_mode_error(context))
            return {'CANCELLED'}

        if mode == 'VSE':
            print(f"[BlendCap] Resolved: start_frame={start_frame}, max_frames={max_frames}, "
                  f"video={os.path.basename(video_path)}")

        # Paths
        addon_dir = _addon_dir()
        python_exe = _python_exe()
        if not os.path.exists(python_exe):
            # Backstop for the invoke() check (e.g. if the venv was deleted
            # between launching and running). Same friendly dialog.
            bpy.ops.blendcap.deps_missing('INVOKE_DEFAULT')
            return {'CANCELLED'}

        # Cache paths for cancel_cleanup to use later
        blendcap_state['addon_dir'] = addon_dir
        blendcap_state['python_exe'] = python_exe

        # Preserve original frame params for cache validation (conversion resets these to 0)
        cache_start_frame = start_frame
        cache_max_frames = max_frames

        # Build async conversion command if needed (runs as first pipeline step)
        actual_video = video_path
        needs_conversion = (is_image_seq or fps_mismatch
                            or (not is_image_seq and not _is_supported_video(video_path)))
        if needs_conversion:
            export_dir = os.path.join(_cache_dir(), vid_name)
            os.makedirs(export_dir, exist_ok=True)
            export_path = os.path.join(export_dir, "source_export.mp4")
            project_fps = context.scene.render.fps / context.scene.render.fps_base
            convert_script = os.path.join(addon_dir, "convert_source.py")

            if is_image_seq:
                # Write elements list to temp file for the conversion script
                elements_file = os.path.join(export_dir, "_elements.txt")
                with open(elements_file, 'w') as f:
                    f.write('\n'.join(img_elements))
                convert_cmd = [python_exe, convert_script,
                               "--image-dir", video_path,
                               "--elements-file", elements_file,
                               "--output", export_path,
                               "--fps", str(project_fps),
                               "--start-frame", str(img_offset_start),
                               "--offset-end", str(img_offset_end)]
            else:
                # Pass source FPS for rate-aware frame sampling that matches
                # Blender's nearest-frame selection for mismatched-FPS strips
                convert_cmd = [python_exe, convert_script,
                               "--video", video_path,
                               "--output", export_path,
                               "--fps", str(project_fps),
                               "--start-frame", str(start_frame),
                               "--max-frames", str(max_frames)]
                if source_fps > 0:
                    convert_cmd.extend(["--source-fps", str(source_fps)])
                if self.fps_mode == 'MATCH' and source_fps > 0:
                    convert_cmd.extend(["--mode", "match"])

            actual_video = export_path
            start_frame = 0
            max_frames = 0

            label = ("Converting image sequence" if is_image_seq
                     else "Converting to match project framerate" if fps_mismatch
                     else "Converting video format")
            self.report({'INFO'}, f"{label}...")

        # Snapshot active strip properties for replacement (VSE mode only)
        vse_replace = None
        if mode == 'VSE':
            seq_ed = context.scene.sequence_editor
            strip = _active_video_strip(seq_ed)
            if strip:
                vse_replace = _snapshot_strip_props(strip)
        blendcap_state['vse_replace'] = vse_replace

        output_dir = os.path.join(_cache_dir(), vid_name)
        os.makedirs(output_dir, exist_ok=True)

        body_npz = os.path.join(output_dir, "mhr_frames.npz")
        face_npz = os.path.join(output_dir, "face_frames.npz")
        body_preview = os.path.join(output_dir, "preview.mp4")
        face_preview = os.path.join(output_dir, "face_preview.mp4")
        combined_preview = os.path.join(output_dir, "combined_preview.mp4")
        overlay_mp4 = os.path.join(output_dir, "overlay.mp4")

        cmds = []
        proxy_path = ""

        # Project FPS at THIS capture/preview run — written into
        # .cache_params so a later run at a different FPS invalidates
        # the NPZ. Computed up here (not inside the capture branch)
        # so preview mode can stash it in blendcap_state too without
        # an UnboundLocalError.
        project_fps_now = context.scene.render.fps / context.scene.render.fps_base

        # Resolved capture settings + their cache signature. Any change to
        # hand refinement / FOV mode-focal / capture skip / backend must
        # re-capture rather than reuse a cached NPZ made with different
        # settings. Computed up here like project_fps_now (NOT inside the
        # capture branch): preview mode stashes it in blendcap_state too,
        # and the monitor writes .cache_params for any run with a body or
        # face step — an unconditional read of a capture-branch-only
        # variable crashed the Preview button with an UnboundLocalError.
        _ap = addon_prefs(context)
        _hr = getattr(_ap, "refine_hands", 'AUTO') if _ap else 'AUTO'
        _cap_ort = _capture_use_ort_override(context)
        _cap_cpu = _capture_force_cpu(context)
        _non_nvidia = bool(_cap_cpu or _cap_ort == "1")
        if _hr == 'ON':
            capture_no_hands = False
        elif _hr == 'OFF':
            capture_no_hands = True
        else:  # AUTO: refine on NVIDIA, skip on non-NVIDIA / CPU
            capture_no_hands = _non_nvidia
        # Legacy scenes saved with the removed 'AUTO' item resolve to the
        # enum default ('FIRST') when Blender reads the property.
        capture_fov = props.fov_mode
        _focal = f"{props.fov_focal_mm:.4f}" if capture_fov == 'MANUAL' else "-"
        capture_sig = (f"nh={int(capture_no_hands)};fov={capture_fov};"
                       f"focal={_focal};skip={props.capture_skip};"
                       f"ort={_cap_ort};cpu={int(bool(_cap_cpu))}")

        if self.is_update_only:
            # ===== PREVIEW MODE =====
            # Always regenerate previews (fast) to match the current trim
            # Preview Skip counts SKIPPED frames (0 = check every frame), the
            # same convention as Capture Skip; the subprocess --preview-step
            # flag keeps its check-every-Nth-frame meaning, so convert here.
            preview_cli_step = str(props.preview_step + 1)
            # Combined body+face previews render ONE video: the body step
            # writes its sampled detections to a JSON sidecar (no render of
            # its own) and the face step draws body box + face markers
            # together - one full-length encode instead of two.
            both_previews = props.track_body and props.track_face
            preview_boxes = os.path.join(output_dir, "preview_boxes.json")
            if props.track_body:
                cmd = [python_exe, os.path.join(addon_dir, "save_mhr_data.py"),
                       "--video", actual_video, "--preview",
                       "--preview-step", preview_cli_step, "--output", body_preview]
                _append_frame_args(cmd, start_frame, max_frames)
                if both_previews:
                    cmd.extend(["--boxes-out", preview_boxes])
                cmds.append(cmd)
                proxy_path = body_preview

            if props.track_face:
                # Preview Skip applies to the face preview too (all three
                # modes: combined, standalone, and body-NPZ-cropped) - the
                # sampled detections interpolate across the in-between
                # frames, same as the body preview, which is what keeps
                # previews quick on very long clips. CAPTURE is deliberately
                # different: face capture always processes every frame -
                # blinks and lip-sync events are only a few frames long and
                # don't survive interpolation.
                output_file = combined_preview if props.track_body else face_preview
                cmd = [python_exe, os.path.join(addon_dir, "save_face_data.py"),
                       "--video", actual_video, "--preview", "--output", output_file,
                       "--preview-step", preview_cli_step]
                _append_frame_args(cmd, start_frame, max_frames)
                if both_previews:
                    cmd.extend(["--body-boxes", preview_boxes])
                elif os.path.exists(body_npz):
                    cov, _, _ = check_npz_coverage(body_npz, start_frame, max_frames)
                    if cov == 'covers':
                        cmd.extend(["--body", body_npz])
                cmds.append(cmd)
                proxy_path = output_file
        else:
            # ===== CAPTURE MODE =====
            # Check cache: skip tracking steps whose params EXACTLY
            # match. Any difference in trim length, project FPS, or
            # capture settings forces re-tracking — an earlier
            # "cache covers request" shortcut caused overlay/NPZ
            # misalignment when the current trim was much smaller
            # than the cached range, and a project-FPS change silently
            # produced bad data because NPZ frames map per project
            # frame. (capture_sig is resolved above, before the branch.)
            params_match = _cache_params_match(
                output_dir,
                cache_start_frame, cache_max_frames,
                img_offset_start, img_offset_end,
                project_fps_now, capture_sig,
                source_path=video_path)

            body_cached = False
            if props.track_body:
                if params_match:
                    cov, _, _ = check_npz_coverage(body_npz, start_frame, max_frames)
                    if cov == 'covers':
                        body_cached = True
                        print("[BlendCap] Body tracking data already cached, skipping.")
                if not body_cached:
                    cmd = [python_exe, os.path.join(addon_dir, "save_mhr_data.py"),
                           "--video", actual_video, "--output", body_npz]
                    _append_frame_args(cmd, start_frame, max_frames)
                    if capture_no_hands:
                        cmd.append("--no-hands")
                    if capture_fov == 'EVERY':
                        cmd.append("--no-fov-cache")
                    elif capture_fov == 'MANUAL':
                        cmd.extend(["--focal-mm", f"{props.fov_focal_mm:.4f}"])
                    if props.capture_skip > 0:
                        cmd.extend(["--capture-skip", str(props.capture_skip)])
                    cmds.append(cmd)
                    # The ORT (non-PyTorch) backbone path loads a ~1.7 GB
                    # ONNX export that cuda-type installs treat as optional.
                    # When the resolved backend wants it and it isn't on
                    # disk, prepend a one-time pre-flight download step
                    # (always exits 0). The CAPTURE process itself never
                    # downloads — offline by design; if the fetch fails it
                    # demotes to the PyTorch backbone and still completes.
                    # Also self-heals "Non-NVIDIA" picked on a cuda install.
                    if _cap_ort == "1":
                        _bb = os.path.join(addon_dir, "checkpoints", "onnx",
                                           "backbone_dinov3_fp16.onnx")
                        _dl = os.path.join(addon_dir, "installer",
                                           "download_models.py")
                        if (os.path.isfile(_dl)
                                and not (os.path.isfile(_bb)
                                         and os.path.isfile(_bb + "_data"))):
                            cmds.insert(0, [python_exe, _dl,
                                            "--project-root", addon_dir,
                                            "--ensure-backbone-onnx"])

            face_cached = False
            if props.track_face:
                if params_match:
                    cov_f, _, _ = check_npz_coverage(face_npz, start_frame, max_frames)
                    if cov_f == 'covers':
                        face_cached = True
                        # The face refine's jaw gate needs mouth_stats
                        # at version >= 2 (v2 added the central-band
                        # histogram that tells a teeth-bare from an
                        # open mouth). A cached capture without them
                        # would silently run refine with a blind or
                        # half-blind gate — treat it as stale and
                        # re-track this clip once.
                        try:
                            import numpy as _np
                            with _np.load(face_npz, allow_pickle=True) as _fnpz:
                                _msv = (int(_fnpz["mouth_stats_version"])
                                        if "mouth_stats_version" in _fnpz
                                        else 1)
                                if "mouth_stats" not in _fnpz or _msv < 3:
                                    face_cached = False
                                    print("[BlendCap] Cached face data predates "
                                          "mouth-stats v3 capture; re-tracking face.")
                        except Exception:
                            pass
                    if face_cached:
                        print("[BlendCap] Face tracking data already cached, skipping.")
                if not face_cached:
                    cmd = [python_exe, os.path.join(addon_dir, "save_face_data.py"),
                           "--video", actual_video, "--output", face_npz]
                    use_body_crops = False
                    if props.track_body and not body_cached:
                        cmd.extend(["--body", body_npz])
                        use_body_crops = True
                    elif os.path.exists(body_npz):
                        cov2, nf, nl = check_npz_coverage(body_npz, start_frame, max_frames)
                        if cov2 == 'covers':
                            cmd.extend(["--body", body_npz])
                            _append_frame_args(cmd, start_frame, max_frames)
                            use_body_crops = True
                        else:
                            self.report({'WARNING'},
                                        f"Body data covers frames {nf}-{nl}, but clip needs more. Using standalone face tracking.")
                    if not use_body_crops:
                        _append_frame_args(cmd, start_frame, max_frames)
                    cmds.append(cmd)

            # Always regenerate overlay (fast) to match the current trim
            overlay_cmd = [python_exe, os.path.join(addon_dir, "overlay_track.py"),
                           "--video", actual_video, "--output", overlay_mp4]
            has_input = False
            if props.track_body or os.path.exists(body_npz):
                overlay_cmd.extend(["--input", body_npz])
                has_input = True
            if props.track_face or os.path.exists(face_npz):
                overlay_cmd.extend(["--face", face_npz])
                has_input = True
            if has_input:
                cmds.append(overlay_cmd)
            proxy_path = overlay_mp4

        if not cmds:
            self.report({'ERROR'}, "Nothing to run. Enable Body or Face tracking.")
            return {'CANCELLED'}

        # Classify steps
        wmap = {"convert_source": (5, "convert"), "download_models": (10, "download"),
                "save_mhr_data": (70, "body"), "save_face_data": (20 if not self.is_update_only else 30, "face"),
                "overlay_track": (10, "overlay")}

        # Prepend conversion step if needed (runs before tracking in the thread)
        if convert_cmd:
            cmds.insert(0, convert_cmd)

        step_weights = []; step_labels = []; step_outputs = []
        for cmd in cmds:
            sn = os.path.basename(cmd[1]) if len(cmd) > 1 else ""
            w, lbl, out = _classify_step(sn, cmd, wmap)
            step_weights.append(w); step_labels.append(lbl); step_outputs.append(out)

        blendcap_state['track_body'] = props.track_body
        blendcap_state['track_face'] = props.track_face
        blendcap_state['output_dir'] = output_dir
        blendcap_state['vid_name'] = vid_name
        blendcap_state['source_strip_name'] = clip_name
        blendcap_state['is_preview'] = self.is_update_only
        blendcap_state['actual_video'] = actual_video
        blendcap_state['cache_start_frame'] = cache_start_frame
        blendcap_state['cache_max_frames'] = cache_max_frames
        blendcap_state['cache_img_offset_start'] = img_offset_start
        blendcap_state['cache_img_offset_end'] = img_offset_end
        blendcap_state['cache_project_fps'] = project_fps_now
        blendcap_state['cache_capture_sig'] = capture_sig
        blendcap_state['cache_video_path'] = video_path
        blendcap_state['step_labels'] = step_labels
        blendcap_state['step_outputs'] = step_outputs

        capture_use_ort = _capture_use_ort_override(context)
        capture_force_cpu = _capture_force_cpu(context)
        # Pre-set state before the thread starts so monitor_progress_timer
        # (registered with first_interval=0) can't read a previous run's
        # stale running=False / return_code=-2 before the worker's own
        # state update runs — worst case the timer's cancel branch deletes
        # this run's outputs and unregisters, leaving the run unmonitored.
        blendcap_state['running'] = True
        blendcap_state['return_code'] = 0
        blendcap_state['cancel_requested'] = False
        # Launch timestamp: the monitor's cancel branch only deletes
        # "partial" outputs newer than this, so cancelling a RE-capture
        # can't destroy the previous capture's good files.
        blendcap_state['run_started'] = time.time()
        # daemon: never block Blender's shutdown on a capture thread; the
        # helpers atexit hook terminates the child process on quit.
        thread = threading.Thread(
            target=run_mocap_thread,
            args=(cmds, proxy_path, insert_frame, insert_channel,
                  video_path, start_frame, max_frames, mode, step_weights),
            kwargs={'use_ort': capture_use_ort, 'force_cpu': capture_force_cpu},
            daemon=True)
        thread.start()
        bpy.app.timers.register(monitor_progress_timer)

        parts = []
        if props.track_body: parts.append("body")
        if props.track_face: parts.append("face")
        label = "preview" if self.is_update_only else "capture"
        self.report({'INFO'}, f"Running {' + '.join(parts)} {label}...")
        return {'FINISHED'}


class BLENDCAP_OT_cancel(bpy.types.Operator):
    bl_idname = "blendcap.cancel"
    bl_label = "Cancel Running Process"
    bl_description = "Stop the current process"

    def execute(self, context):
        # Set the flag FIRST, then kill the child directly. The worker's
        # reader loop only re-checks the flag between reads, and a capture
        # subprocess can sit silent for minutes loading checkpoints (or
        # wedge entirely) — terminating it closes the pipe, the blocked
        # read returns EOF, and the worker sees cancel_requested and winds
        # down. Flag alone = a Cancel button that can be ignored forever.
        if blendcap_state['running']:
            blendcap_state['cancel_requested'] = True
            _terminate_state_process(blendcap_state)
            self.report({'INFO'}, "Cancelling... please wait.")
        elif bvh_state['running']:
            bvh_state['cancel_requested'] = True
            _terminate_state_process(bvh_state)
            self.report({'INFO'}, "Cancelling BVH export... please wait.")
        else:
            self.report({'INFO'}, "Nothing running to cancel.")
        return {'FINISHED'}


class BLENDCAP_OT_cancel_cleanup(bpy.types.Operator):
    bl_idname = "blendcap.cancel_cleanup"
    bl_label = "Capture Cancelled"
    bl_options = {'REGISTER'}

    action: bpy.props.EnumProperty(
        name="Action",
        items=[('KEEP', "Keep Body Data", "Keep completed body tracking data for later use"),
               ('DISCARD', "Discard All", "Delete all data from this run")],
        default='KEEP'
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=380)

    def draw(self, context):
        layout = self.layout
        layout.label(text="Body tracking completed, but face tracking was cancelled.", icon='INFO')
        layout.label(text="Would you like to keep the body data?")
        layout.separator()
        layout.prop(self, "action", expand=True)

    def execute(self, context):
        output_dir = blendcap_state.get('output_dir', '')

        if self.action == 'KEEP':
            if output_dir:
                for f in ("face_frames.npz", "face_preview.mp4", "combined_preview.mp4"):
                    p = os.path.join(output_dir, f)
                    if os.path.exists(p):
                        try: os.remove(p)
                        except Exception: pass

            body_npz = os.path.join(output_dir, "mhr_frames.npz")
            overlay_mp4 = os.path.join(output_dir, "overlay.mp4")
            source_path = blendcap_state.get('actual_video', '') or blendcap_state.get('source_path', '')

            if os.path.exists(body_npz) and source_path:
                python_exe = blendcap_state.get('python_exe') or _python_exe()
                addon_dir = blendcap_state.get('addon_dir') or _addon_dir()
                overlay_script = os.path.join(addon_dir, "overlay_track.py")
                if os.path.exists(python_exe) and os.path.exists(overlay_script):
                    cmd = [python_exe, overlay_script, "--video", source_path,
                           "--input", body_npz, "--output", overlay_mp4]
                    blendcap_state['step_labels'] = ['overlay']
                    blendcap_state['step_outputs'] = [overlay_mp4]
                    blendcap_state['proxy_path'] = overlay_mp4

                    # Pre-set state before thread starts to prevent race with
                    # monitor_progress_timer (fires with first_interval=0,
                    # could see stale running=False / return_code=-2)
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
                    self.report({'INFO'}, "Body data kept. Generating preview...")
                else:
                    self.report({'WARNING'}, "Body data kept, but could not generate preview.")
            else:
                self.report({'WARNING'}, "Body data kept, but NPZ not found for preview.")
        else:
            if output_dir:
                for f in os.listdir(output_dir):
                    if f.endswith((".npz", ".mp4")):
                        try: os.remove(os.path.join(output_dir, f))
                        except Exception: pass
            self.report({'INFO'}, "All capture data discarded.")
        return {'FINISHED'}


class BLENDCAP_OT_generate_bvh(bpy.types.Operator):
    bl_idname = "blendcap.generate_bvh"
    bl_label = "Generate & Import BVH"
    bl_description = ("Generate a BVH (and/or hands/root/face variants) "
                      "from the cached tracking data of the currently "
                      "selected clip and import it into the scene")

    def invoke(self, context, event):
        """Validate selection + tracking data, then check clip length."""
        if blendcap_state['running'] or bvh_state['running']:
            return self.execute(context)

        props = context.scene.blendcap_props

        # ---- 1. Selection check -------------------------------------
        if not (props.export_body or props.export_hands or props.export_root
                or props.export_face or props.use_separate_face):
            _show_warning_popup([
                "Nothing selected to export.",
                "Enable at least one of: Body, Hands, Root, Face, "
                "or Use Separate Face Data.",
            ])
            return {'CANCELLED'}

        # ---- 2. Resolve source video --------------------------------
        video_path = ""
        if props.video_source == 'FILE':
            video_path, _empty = _file_mode_source(context)
        else:
            seq_ed = context.scene.sequence_editor
            strip = _active_video_strip(seq_ed)
            if strip and strip.type in ('MOVIE', 'IMAGE'):
                if "blendcap_source" in strip:
                    video_path = bpy.path.abspath(str(strip["blendcap_source"]))
                elif strip.type == 'IMAGE':
                    video_path = bpy.path.abspath(strip.directory)
                else:
                    video_path = bpy.path.abspath(strip.filepath)

        if not video_path:
            # Let execute() report the missing-source error in the
            # status bar. Note: the file itself is NOT required to
            # exist — BVH generation reads only the cached NPZ, and
            # the path is only needed to derive the cache folder name.
            return self.execute(context)

        # ---- 3. Locate NPZ caches -----------------------------------
        cache_dir = _cache_dir()
        clip_name = _resolve_clip_name(context) or "sequence"
        vid_name = f"Capture_{clip_name}"
        output_dir = os.path.join(cache_dir, vid_name)
        body_npz = os.path.join(output_dir, "mhr_frames.npz")
        face_npz = os.path.join(output_dir, "face_frames.npz")
        has_body = os.path.exists(body_npz)
        has_face = os.path.exists(face_npz)

        has_separate = False
        if props.use_separate_face and props.separate_face_data != 'NONE':
            sep_npz = os.path.join(cache_dir, props.separate_face_data,
                                   "face_frames.npz")
            has_separate = os.path.exists(sep_npz)

        # ---- 4. Tracking-data check (popup + cancel on miss) --------
        needs_body = (props.export_body or props.export_hands
                      or props.export_root)
        needs_own_face = props.export_face
        needs_sep_face = props.use_separate_face

        if needs_body and not has_body:
            if not has_face and not has_separate:
                msg = ["This clip hasn't been tracked yet.",
                       "Run Capture first to generate motion data."]
            else:
                msg = ["Body tracking data is missing for this clip.",
                       "Run Capture (with body tracking) first."]
            _show_warning_popup(msg)
            return {'CANCELLED'}

        if needs_own_face and not has_face:
            _show_warning_popup([
                "Face tracking data is missing for this clip.",
                "Run Capture (with face tracking) first.",
            ])
            return {'CANCELLED'}

        if needs_sep_face and not has_separate:
            if props.separate_face_data == 'NONE':
                msg = ["No separate face data selected.",
                       "Pick a clip from the Separate Face Data dropdown."]
            else:
                msg = ["Selected separate face data wasn't found.",
                       "Pick a different clip from the dropdown."]
            _show_warning_popup(msg)
            return {'CANCELLED'}

        return self.execute(context)

    def execute(self, context):
        if blendcap_state['running']:
            self.report({'WARNING'}, "A capture is running. Wait for it to finish before generating BVH.")
            return {'CANCELLED'}
        if bvh_state['running']:
            self.report({'WARNING'}, "A BVH export is already running.")
            return {'CANCELLED'}

        props = context.scene.blendcap_props

        # Resolve video path (simplified — BVH only needs the cache folder name)
        if props.video_source == 'FILE':
            video_path, _empty = _file_mode_source(context)
        else:
            seq_ed = context.scene.sequence_editor
            strip = _active_video_strip(seq_ed)
            if strip and strip.type in ('MOVIE', 'IMAGE'):
                if "blendcap_source" in strip:
                    video_path = bpy.path.abspath(str(strip["blendcap_source"]))
                elif strip.type == 'IMAGE':
                    video_path = bpy.path.abspath(strip.directory)
                else:
                    video_path = bpy.path.abspath(strip.filepath)
            else:
                video_path = ""

        # The video file itself is not required — generation reads only
        # the cached NPZ, and the path's only job is naming the cache
        # folder. Requiring existence would block regenerating from a
        # perfectly good cache after the source clip was moved or
        # deleted (Bake already works that way via the rig's tag).
        if not video_path:
            self.report({'ERROR'}, source_mode_error(context))
            return {'CANCELLED'}

        cache_dir = _cache_dir()
        clip_name = _resolve_clip_name(context) or "sequence"
        vid_name = f"Capture_{clip_name}"
        output_dir = os.path.join(cache_dir, vid_name)
        body_npz = os.path.join(output_dir, "mhr_frames.npz")
        face_npz = os.path.join(output_dir, "face_frames.npz")
        bvh_path = os.path.join(output_dir, "output.bvh")

        has_body = os.path.exists(body_npz)
        has_face = os.path.exists(face_npz)

        separate_face_npz = ""
        if props.use_separate_face and props.separate_face_data != 'NONE':
            separate_face_npz = os.path.join(cache_dir, props.separate_face_data, "face_frames.npz")
            if os.path.exists(separate_face_npz):
                has_face = True

        if not has_body and not has_face:
            self.report({'ERROR'}, "No tracking data found. Run Capture first.")
            return {'CANCELLED'}

        addon_dir = _addon_dir()
        python_exe = _python_exe()
        script_path = os.path.join(addon_dir, "npz_to_bvh.py")
        if not os.path.exists(script_path):
            self.report({'ERROR'}, f"npz_to_bvh.py not found at {script_path}")
            return {'CANCELLED'}

        mhr_model = os.path.join(addon_dir, "checkpoints",
                                 "sam-3d-body-dinov3", "assets", "mhr_model.pt")

        # Resolve face source (own face or separate-clip face). Mutually
        # exclusive in the UI, but check explicit existence for safety.
        face_path = ""
        if props.export_face and os.path.exists(face_npz):
            face_path = face_npz
        elif (props.use_separate_face and separate_face_npz
              and os.path.exists(separate_face_npz)):
            face_path = separate_face_npz

        # Common args every sub-command shares
        def _common():
            return [python_exe, script_path, "--mhr-model", mhr_model]

        # Face source identity for filename annotation. Bare clip
        # identifier (no Capture_ prefix) — the prefix only exists on
        # cache folder names for VSE-strip clarity.
        #   - own face  → face_source_name == clip_name (folds into codes as "F")
        #   - separate  → face_source_name == bare clip name of separate cache
        #                 (pulled out to "-{source}_F")
        face_source_name = ""
        if face_path == face_npz:
            face_source_name = clip_name
        elif face_path and face_path == separate_face_npz:
            face_source_name = _strip_capture_prefix(props.separate_face_data)

        # Build job list. Each job: (cmd, intermediate_bvh_path, output_stem)
        # where output_stem is the final filename (without .bvh extension)
        # following the convention: {clip}_{BHRF codes}[-{face_clip}_F].
        jobs = []

        if has_body and props.export_body:
            codes = "B"
            if props.export_hands:
                codes += "H"
            if props.export_root:
                codes += "R"
            if face_path and face_source_name == clip_name:
                codes += "F"
                stem = f"{clip_name}_{codes}"
            elif face_path:
                stem = f"{clip_name}_{codes}-{face_source_name}_F"
            else:
                stem = f"{clip_name}_{codes}"

            cmd = _common()
            cmd.extend(["--input", body_npz, "--rest-pose", "tpose"])
            if props.export_root:
                cmd.append("--root-motion")
            if face_path:
                cmd.extend(["--face", face_path])
            if props.smoothing_enabled and props.smooth_body > 0:
                cmd.extend(["--smooth", str(props.smooth_body)])
            if props.smoothing_enabled and props.smooth_root > 0:
                cmd.extend(["--smooth-root", str(props.smooth_root)])
            if props.smoothing_enabled and props.smooth_vertical > 0:
                cmd.extend(["--smooth-vertical", str(props.smooth_vertical)])
            cmd.extend(["--smooth-face", str(props.smooth_face if props.smoothing_enabled else 0)])
            cmd.extend(_bvh_face_denoise_args(props))
            cmd.extend(_bvh_face_amp_args(props))
            cmd.extend(_bvh_depth_denoise_args(props))
            if props.finger_curl_multiplier > 1.0:
                cmd.extend(["--finger-boost",
                            f"{props.finger_curl_multiplier:.1f}"])
            if props.bvh_hand_solver:
                cmd.extend(["--hand-solver", "on"])
            cmd.extend(["--face-boost",
                        f"{_bvh_face_boost_value(props):.1f}"])
            if not props.export_hands:
                cmd.append("--no-hands")
            # Foot locking flags (Pass 1 + Pass 2)
            if not props.foot_lock_enabled:
                cmd.append("--no-foot-lock")
            cmd.extend(["--foot-sensitivity",
                        f"{props.foot_sensitivity:.3f}"])
            cmd.extend(["--max-gap",
                        f"{props.foot_max_gap_sec:.3f}"])
            # Properties are stored in cm for UI clarity; CLI takes
            # meters, so divide by 100 here.
            cmd.extend(["--max-lift",
                        f"{props.foot_max_lift_cm / 100.0:.3f}"])
            if props.foot_recover_glitches:
                cmd.append("--recover-glitches")
            cmd.extend(["--recovery-lift",
                        f"{props.foot_recovery_lift_cm / 100.0:.3f}"])
            if not props.foot_pin_enabled:
                cmd.append("--no-foot-pin")
            cmd.extend(["--foot-pin-strength",
                        f"{props.foot_pin_strength:.2f}"])
            cmd.extend(["--foot-pin-release",
                        f"{props.foot_pin_release_cm / 100.0:.3f}"])
            if props.foot_define_floor:
                cmd.append("--feet-define-floor")
            if props.body_grounding_enabled:
                cmd.append("--body-grounding")
                cmd.extend(["--body-catch-distance",
                            f"{props.body_catch_distance_cm / 100.0:.3f}"])
                cmd.extend(["--body-skin-radius",
                            f"{props.body_skin_radius_cm / 100.0:.4f}"])
            # Camera angle offset is per-rig (Object.blendcap_camera_offset)
            # and applied via Bake Offset to BVH after the rig is imported,
            # not here.
            cmd.extend(["--output", bvh_path])
            jobs.append((cmd, bvh_path, stem))
        else:
            # Standalone hands armature
            if props.export_hands:
                if not has_body:
                    self.report({'ERROR'},
                                "Hands export requires body tracking data. "
                                "Run Capture first.")
                    return {'CANCELLED'}
                hands_bvh = os.path.join(output_dir, "output_hands.bvh")
                cmd = _common()
                cmd.extend(["--input", body_npz, "--export", "hands"])
                if props.smoothing_enabled and props.smooth_body > 0:
                    cmd.extend(["--smooth", str(props.smooth_body)])
                if props.finger_curl_multiplier > 1.0:
                    cmd.extend(["--finger-boost",
                                f"{props.finger_curl_multiplier:.1f}"])
                if props.bvh_hand_solver:
                    cmd.extend(["--hand-solver", "on"])
                cmd.extend(["--output", hands_bvh])
                jobs.append((cmd, hands_bvh, f"{clip_name}_H"))

            # Standalone root armature
            if props.export_root:
                if not has_body:
                    self.report({'ERROR'},
                                "Root export requires body tracking data. "
                                "Run Capture first.")
                    return {'CANCELLED'}
                root_bvh = os.path.join(output_dir, "output_root.bvh")
                cmd = _common()
                cmd.extend(["--input", body_npz, "--export", "root"])
                if props.smoothing_enabled and props.smooth_root > 0:
                    cmd.extend(["--smooth-root", str(props.smooth_root)])
                if props.smoothing_enabled and props.smooth_vertical > 0:
                    cmd.extend(["--smooth-vertical", str(props.smooth_vertical)])
                cmd.extend(["--output", root_bvh])
                jobs.append((cmd, root_bvh, f"{clip_name}_R"))

            # Standalone face armature (own face or separate).
            # Base name follows the face source — separate face from clipX
            # becomes `clipX_F` regardless of which clip is active.
            if face_path:
                face_bvh = os.path.join(output_dir, "output_face.bvh")
                cmd = _common()
                cmd.extend(["--face", face_path, "--export", "face"])
                cmd.extend(["--smooth-face", str(props.smooth_face if props.smoothing_enabled else 0)])
                cmd.extend(_bvh_face_denoise_args(props))
                cmd.extend(_bvh_face_amp_args(props))
                cmd.extend(["--face-boost",
                            f"{_bvh_face_boost_value(props):.1f}"])
                cmd.extend(["--output", face_bvh])
                jobs.append((cmd, face_bvh, f"{face_source_name}_F"))
        # face_source_name here is already the bare clip identifier.

        if not jobs:
            self.report({'ERROR'},
                        "Nothing selected to export. Enable at least one of: "
                        "Body, Hands, Root, Face, or Use Separate Face Data.")
            return {'CANCELLED'}

        # Generate path: no baked-in camera offset on the new rig.
        # Reset before the worker starts so a stale value from a
        # previous Bake doesn't leak onto the freshly-imported rig.
        bvh_state['baked_pitch_deg'] = 0.0
        bvh_state['baked_roll_deg'] = 0.0
        bvh_state['baked_head_pitch_deg'] = 0.0
        bvh_state['baked_head_roll_deg'] = 0.0
        bvh_state['baked_height_m'] = 0.0
        # Resolve the BVH library dir on THIS (main) thread - run_bvh_thread
        # can't safely read add-on prefs from the worker.
        _bvh_lib = _bvh_library_dir()
        # Pre-set state before the thread starts so monitor_bvh_timer can't
        # read a previous run's stale running/return_code (same race as
        # monitor_progress_timer). daemon: see generate_mocap.
        bvh_state['running'] = True
        bvh_state['return_code'] = 0
        bvh_state['cancel_requested'] = False
        thread = threading.Thread(target=run_bvh_thread,
                                  args=(jobs, clip_name, cache_dir, _bvh_lib),
                                  daemon=True)
        thread.start()
        bpy.app.timers.register(monitor_bvh_timer)
        msg = ("Generating BVH... It will auto-import when finished."
               if len(jobs) == 1
               else f"Generating {len(jobs)} BVHs... They will auto-import "
                    f"as separate armatures when finished.")
        self.report({'INFO'}, msg)
        return {'FINISHED'}


def _parse_bake_flags(stem, vid_name):
    """Recover the export flags the rig was originally generated with by
    parsing its BVH stem. Mirrors the naming scheme in
    BLENDCAP_OT_generate_bvh.execute(): {vid}_{codes}[-{face_source}_F]
    where codes is a substring of "BHRF" (B=body, H=hands, R=root,
    F=face from own clip).

    Returns dict {hands, root, face, face_source} on success, None when
    the stem doesn't match the convention (e.g. an externally-renamed
    file). Caller is responsible for falling back to scene props in
    that case.
    """
    # Body with separate face: vid_<codes>-<face_source>_F
    if stem.startswith(vid_name + "_") and "-" in stem[len(vid_name) + 1:]:
        suffix = stem[len(vid_name) + 1:]
        codes, sep = suffix.split("-", 1)
        if sep.endswith("_F"):
            return {
                "hands": "H" in codes,
                "root": "R" in codes,
                "face": True,
                "face_source": sep[:-2],
            }
        return None
    # Body / standalone with own clip: vid_<codes>
    if stem.startswith(vid_name + "_"):
        codes = stem[len(vid_name) + 1:]
        return {
            "hands": "H" in codes,
            "root": "R" in codes,
            "face": "F" in codes,
            "face_source": "",
        }
    # Standalone face from a separate clip: <face_source>_F
    if stem.endswith("_F"):
        return {
            "hands": False,
            "root": False,
            "face": True,
            "face_source": stem[:-2],
        }
    return None


class BLENDCAP_OT_bake_camera_offset(bpy.types.Operator):
    """Bake the active rig's pitch/roll offset into its BVH file.

    Re-runs npz_to_bvh.py for the source clip tagged on the rig
    (`armature["blendcap_source_video"]` set during the original
    generation), passing the current scene props plus pitch/roll and
    the height offset. The
    output overwrites the existing BVH on disk and monitor_bvh_timer
    re-imports it; mocap.py's _import_bvh path deletes the old armature
    first so the <stem> name transfers cleanly. After re-import the
    new armature's blendcap_camera_offset PropertyGroup defaults to 0/0
    — the offset now lives in the file rather than in Blender state,
    so the live-preview overlay starts from a clean baseline.

    Body vs face-only is detected from the rig's bones (presence of
    Hips). Body bakes use the body command path; face-only bakes use
    the standalone-face path (no body NPZ required).

    Why a separate operator instead of folding it into Generate &
    Import BVH: the user has often imported MULTIPLE clips and is
    iterating on just one of them. Bake is scoped to whichever rig is
    active so the other clips in the scene aren't disturbed.
    """
    bl_idname = "blendcap.bake_camera_offset"
    bl_label = "Bake Settings to BVH"
    bl_description = ("Re-generate the BVH for the currently selected "
                      "armature with the current post-processing "
                      "settings. Overwrites the BVH file on disk and "
                      "re-imports.")
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        if blendcap_state.get('running') or bvh_state.get('running'):
            return False
        rig = context.active_object
        if rig is None or rig.type != 'ARMATURE':
            return False
        return "blendcap_source_video" in rig

    def execute(self, context):
        rig = context.active_object
        vid_name = str(rig.get("blendcap_source_video", "") or "")
        if not vid_name:
            self.report({'ERROR'},
                        "Active rig has no source-clip tag — generate "
                        "the BVH first via 'Generate & Import BVH'.")
            return {'CANCELLED'}

        # Resolve the output stem from the BVH path the rig was
        # imported from. Survives renames (rig.name might no longer
        # match the file). Falls back to rig.name if the custom prop
        # wasn't recorded (e.g. legacy rigs from before the prop
        # existed).
        bvh_path_prop = str(rig.get("blendcap_bvh_path", "") or "")
        if bvh_path_prop:
            stem = os.path.splitext(os.path.basename(bvh_path_prop))[0]
        else:
            stem = rig.name

        offset = rig.blendcap_camera_offset
        slider_pitch = float(offset.pitch_deg)
        slider_roll = float(offset.roll_deg)
        slider_head_pitch = float(offset.head_pitch_deg)
        slider_head_roll = float(offset.head_roll_deg)
        slider_height = float(offset.height_offset_m)
        # Additive bakes: each Bake adds the current slider values on
        # TOP of whatever was previously baked into this rig's BVH.
        # Track the cumulative baked value via custom props on the rig
        # so a sequence of bakes (1° then later 3° more) ends up at
        # the sum (4°) in the BVH file. The new rig that comes back
        # from re-import inherits these totals via bvh_state ->
        # mocap.monitor_bvh_timer.
        prev_baked_pitch = float(rig.get("_blendcap_baked_pitch_deg", 0.0))
        prev_baked_roll = float(rig.get("_blendcap_baked_roll_deg", 0.0))
        prev_baked_head_pitch = float(
            rig.get("_blendcap_baked_head_pitch_deg", 0.0))
        prev_baked_head_roll = float(
            rig.get("_blendcap_baked_head_roll_deg", 0.0))
        prev_baked_height = float(
            rig.get("_blendcap_baked_height_m", 0.0))
        pitch = prev_baked_pitch + slider_pitch
        roll = prev_baked_roll + slider_roll
        head_pitch = prev_baked_head_pitch + slider_head_pitch
        head_roll = prev_baked_head_roll + slider_head_roll
        height = prev_baked_height + slider_height

        # vid_name (loaded from rig prop) is the BARE clip identifier
        # so it can be embedded in BVH stems. The on-disk cache folder
        # carries the Capture_ prefix; reconstruct it here.
        cache_dir = _cache_dir()
        output_dir = os.path.join(cache_dir, f"Capture_{vid_name}")
        body_npz = os.path.join(output_dir, "mhr_frames.npz")
        face_npz = os.path.join(output_dir, "face_frames.npz")

        # Detect export type from the rig structure. Bake should
        # preserve whatever the rig was originally generated with —
        # the user's body/hands/face/root checkboxes apply to
        # Generate, not to re-baking an existing rig.
        pose_bones = rig.pose.bones if rig.pose else None
        has_hips = bool(pose_bones and "Hips" in pose_bones)
        has_head = bool(pose_bones and "Head" in pose_bones)
        has_root = bool(pose_bones and "Root" in pose_bones)
        has_left_hand = bool(pose_bones and "LeftHand" in pose_bones)

        # Recover hands/root/face/face_source from the stem so the
        # bake command exactly matches the original anatomy. Returns
        # None if the stem doesn't follow the naming convention.
        flags = _parse_bake_flags(stem, vid_name)

        addon_dir = _addon_dir()
        python_exe = _python_exe()
        script_path = os.path.join(addon_dir, "npz_to_bvh.py")
        mhr_model = os.path.join(addon_dir, "checkpoints",
                                 "sam-3d-body-dinov3", "assets",
                                 "mhr_model.pt")

        props = context.scene.blendcap_props

        def _resolve_face_path():
            """Locate the face NPZ to use based on parsed flags. Same
            cache folder as the original generate used: own clip when
            face_source is empty, otherwise the separate face clip's
            cache folder."""
            if not flags or not flags["face"]:
                return ""
            if flags["face_source"]:
                sf = os.path.join(cache_dir, f"Capture_{flags['face_source']}",
                                  "face_frames.npz")
                if os.path.exists(sf):
                    return sf
            if os.path.exists(face_npz):
                return face_npz
            return ""

        if has_hips:
            # ---- Body bake (B + optional H / R / F) ----
            if not os.path.exists(body_npz):
                self.report({'ERROR'},
                            f"Body NPZ not found for clip '{vid_name}'. "
                            f"Has the cache been cleared?")
                return {'CANCELLED'}

            # Use parsed flags; fall back to bone-presence heuristics
            # if the stem couldn't be parsed (renamed rig).
            if flags is not None:
                use_hands = flags["hands"]
                use_root = flags["root"]
                use_face = flags["face"]
            else:
                # Stem unparseable — preserve what the rig actually
                # contains rather than touching the scene checkboxes.
                use_hands = has_left_hand
                use_root = has_root
                use_face = bool(pose_bones and any(
                    pb.name not in ("Root", "Hips", "Spine", "Chest",
                                    "Neck", "Head")
                    and pb.parent and pb.parent.name == "Head"
                    for pb in pose_bones))
            face_path = _resolve_face_path() if use_face else ""

            bvh_path = os.path.join(output_dir, "output.bvh")
            cmd = [python_exe, script_path, "--mhr-model", mhr_model,
                   "--input", body_npz, "--rest-pose", "tpose"]
            if use_root:
                cmd.append("--root-motion")
            if face_path:
                cmd.extend(["--face", face_path])
            if props.smoothing_enabled and props.smooth_body > 0:
                cmd.extend(["--smooth", str(props.smooth_body)])
            if props.smoothing_enabled and props.smooth_root > 0:
                cmd.extend(["--smooth-root", str(props.smooth_root)])
            if props.smoothing_enabled and props.smooth_vertical > 0:
                cmd.extend(["--smooth-vertical", str(props.smooth_vertical)])
            cmd.extend(["--smooth-face", str(props.smooth_face if props.smoothing_enabled else 0)])
            cmd.extend(_bvh_face_denoise_args(props))
            cmd.extend(_bvh_face_amp_args(props))
            cmd.extend(_bvh_depth_denoise_args(props))
            if props.finger_curl_multiplier > 1.0:
                cmd.extend(["--finger-boost",
                            f"{props.finger_curl_multiplier:.1f}"])
            if props.bvh_hand_solver:
                cmd.extend(["--hand-solver", "on"])
            cmd.extend(["--face-boost",
                        f"{_bvh_face_boost_value(props):.1f}"])
            if not use_hands:
                cmd.append("--no-hands")
            if not props.foot_lock_enabled:
                cmd.append("--no-foot-lock")
            cmd.extend(["--foot-sensitivity",
                        f"{props.foot_sensitivity:.3f}"])
            cmd.extend(["--max-gap",
                        f"{props.foot_max_gap_sec:.3f}"])
            cmd.extend(["--max-lift",
                        f"{props.foot_max_lift_cm / 100.0:.3f}"])
            if props.foot_recover_glitches:
                cmd.append("--recover-glitches")
            cmd.extend(["--recovery-lift",
                        f"{props.foot_recovery_lift_cm / 100.0:.3f}"])
            if not props.foot_pin_enabled:
                cmd.append("--no-foot-pin")
            cmd.extend(["--foot-pin-strength",
                        f"{props.foot_pin_strength:.2f}"])
            cmd.extend(["--foot-pin-release",
                        f"{props.foot_pin_release_cm / 100.0:.3f}"])
            if props.foot_define_floor:
                cmd.append("--feet-define-floor")
            if props.body_grounding_enabled:
                cmd.append("--body-grounding")
                cmd.extend(["--body-catch-distance",
                            f"{props.body_catch_distance_cm / 100.0:.3f}"])
                cmd.extend(["--body-skin-radius",
                            f"{props.body_skin_radius_cm / 100.0:.4f}"])
            if abs(pitch) > 1e-6:
                cmd.extend(["--camera-pitch", f"{pitch:.3f}"])
            if abs(roll) > 1e-6:
                cmd.extend(["--camera-roll", f"{roll:.3f}"])
            if abs(head_pitch) > 1e-6:
                cmd.extend(["--head-pitch", f"{head_pitch:.3f}"])
            if abs(head_roll) > 1e-6:
                cmd.extend(["--head-roll", f"{head_roll:.3f}"])
            if abs(height) > 1e-6:
                cmd.extend(["--height-offset", f"{height:.4f}"])
            cmd.extend(["--output", bvh_path])

        elif has_head and not has_root:
            # ---- Face-only bake ----
            face_path = _resolve_face_path()
            if not face_path and os.path.exists(face_npz):
                face_path = face_npz
            if not face_path:
                self.report({'ERROR'},
                            f"Face NPZ not found for clip '{vid_name}'. "
                            f"Run face capture first.")
                return {'CANCELLED'}

            bvh_path = os.path.join(output_dir, "output_face.bvh")
            cmd = [python_exe, script_path, "--mhr-model", mhr_model,
                   "--face", face_path, "--export", "face"]
            cmd.extend(["--smooth-face", str(props.smooth_face if props.smoothing_enabled else 0)])
            cmd.extend(_bvh_face_denoise_args(props))
            cmd.extend(_bvh_face_amp_args(props))
            cmd.extend(["--face-boost",
                        f"{_bvh_face_boost_value(props):.1f}"])
            # Camera pitch/roll and height offset silently dropped —
            # no Root to rotate, no Hips to shift.
            cmd.extend(["--output", bvh_path])

        elif has_left_hand and not has_hips:
            # ---- Hands-only bake ----
            if not os.path.exists(body_npz):
                self.report({'ERROR'},
                            f"Body NPZ not found for clip '{vid_name}'. "
                            f"Has the cache been cleared?")
                return {'CANCELLED'}
            bvh_path = os.path.join(output_dir, "output_hands.bvh")
            cmd = [python_exe, script_path, "--mhr-model", mhr_model,
                   "--input", body_npz, "--export", "hands"]
            if props.smoothing_enabled and props.smooth_body > 0:
                cmd.extend(["--smooth", str(props.smooth_body)])
            if props.finger_curl_multiplier > 1.0:
                cmd.extend(["--finger-boost",
                            f"{props.finger_curl_multiplier:.1f}"])
            if props.bvh_hand_solver:
                cmd.extend(["--hand-solver", "on"])
            cmd.extend(["--output", bvh_path])

        elif has_root and not has_hips and not has_head:
            # ---- Root-only bake ----
            if not os.path.exists(body_npz):
                self.report({'ERROR'},
                            f"Body NPZ not found for clip '{vid_name}'. "
                            f"Has the cache been cleared?")
                return {'CANCELLED'}
            bvh_path = os.path.join(output_dir, "output_root.bvh")
            cmd = [python_exe, script_path, "--mhr-model", mhr_model,
                   "--input", body_npz, "--export", "root"]
            if props.smoothing_enabled and props.smooth_root > 0:
                cmd.extend(["--smooth-root", str(props.smooth_root)])
            if props.smoothing_enabled and props.smooth_vertical > 0:
                cmd.extend(["--smooth-vertical", str(props.smooth_vertical)])
            cmd.extend(["--output", bvh_path])

        else:
            self.report({'ERROR'},
                        "Active rig isn't a recognized BlendCap export "
                        "type (no Hips, Head, hand, or Root bones).")
            return {'CANCELLED'}

        jobs = [(cmd, bvh_path, stem)]
        # Stash the cumulative baked totals BEFORE starting the
        # worker thread. The worker's bvh_state.update inside
        # run_bvh_thread does NOT touch these keys, so setting them
        # here ensures monitor_bvh_timer can read the right values
        # after the import completes. Without this pre-launch write,
        # a stale value from a previous bake or an uninitialized
        # state could land on the new rig.
        bvh_state['baked_pitch_deg'] = pitch
        bvh_state['baked_roll_deg'] = roll
        bvh_state['baked_head_pitch_deg'] = head_pitch
        bvh_state['baked_head_roll_deg'] = head_roll
        bvh_state['baked_height_m'] = height
        # Resolve the BVH library dir on THIS (main) thread - run_bvh_thread
        # can't safely read add-on prefs from the worker.
        _bvh_lib = _bvh_library_dir()
        # Pre-set state before the thread starts so monitor_bvh_timer can't
        # read a previous run's stale running/return_code (same race as
        # monitor_progress_timer). daemon: see generate_mocap.
        bvh_state['running'] = True
        bvh_state['return_code'] = 0
        bvh_state['cancel_requested'] = False
        thread = threading.Thread(target=run_bvh_thread,
                                  args=(jobs, vid_name, cache_dir, _bvh_lib),
                                  daemon=True)
        thread.start()
        bpy.app.timers.register(monitor_bvh_timer)
        self.report({'INFO'},
                    f"Baking offsets into '{stem}.bvh' "
                    f"(body pitch {pitch:+.2f}deg roll {roll:+.2f}deg; "
                    f"head pitch {head_pitch:+.2f}deg roll "
                    f"{head_roll:+.2f}deg; height {height:+.3f}m)...")
        return {'FINISHED'}
