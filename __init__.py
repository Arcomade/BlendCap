"""BlendCap - Blender extension entry point.

Blender 4.2+ discovers this package via the sibling blender_manifest.toml
and imports it as ``bl_ext.<repo>.blendcap``. This module owns the add-on
preferences (the dependency-installer UI) and delegates ALL feature
registration to the blendcap/ subpackage's registration module.

Why a thin root package wrapping a blendcap/ subpackage: the helper scripts
the addon shells out to (save_mhr_data.py, npz_to_bvh.py), the installer/,
the venv/, and models/ all live at this root, NEXT TO the blendcap/ package.
helpers._addon_dir() resolves to this folder (parent of blendcap/), so the
subprocess pipeline keeps finding those scripts whether running from a dev
checkout or an installed extension.
"""
import bpy
import os
import textwrap

from .blendcap import registration as _registration
from .blendcap.helpers import (
    _addon_dir, read_install_marker, is_dependencies_installed,
    installer_files_present,
)
from .blendcap.operators_pipeline import draw_install_wizard
# Single shared channel flag (evaluated once at registration in panels):
# True only for the source distribution, where donating is the ask.
from .blendcap.panels import SHOW_DONATE_BUTTON


def _label_wrapped(col, context, text, icon='NONE'):
    """Draw `text` into `col` as labels wrapped to the panel width.

    Blender UI labels don't wrap on their own. Estimate a character budget
    from the region width (minus padding, scaled by the UI scale) and split
    on word boundaries, so help text fills the width like a paragraph instead
    of breaking at hand-placed points or clipping off the right edge."""
    try:
        region = context.region
        width_px = (region.width if region else 600) - 50
        ui = context.preferences.system.ui_scale or 1.0
        chars = max(24, int(width_px / (8.0 * ui)))
    except Exception:
        chars = 70
    lines = textwrap.wrap(text, chars) or [text]
    for i, line in enumerate(lines):
        col.label(text=line, icon=(icon if i == 0 else 'NONE'))


def _redraw_viewport_panels(self, context):
    """Refresh the sidebar / Output panels when a visibility pref toggles, so
    hiding the Setup section or Clear All button takes effect immediately
    instead of waiting for the next viewport interaction."""
    from .blendcap.helpers import _tag_redraw
    _tag_redraw('VIEW_3D', 'PROPERTIES')


def _update_default_video_source(self, context):
    """When Default Source is switched to Video Editor Strip while
    Sequencer Render Warning is 'Ask each time', pop a dialog converting
    the warning pref to one of the two options that work for new files
    (ask-each-time never fires on File > New - there's no interactive
    switch to hook). OK -> Turn off automatically, Cancel -> Leave it
    alone. While the default stays VSE, 'Ask each time' is greyed out in
    the prefs UI (see the draw code).

    Deferred through a one-shot timer, which does double duty: it keeps
    the dialog from firing mid-click on the dropdown, and it re-checks
    both prefs at fire time - so if this update fired during userpref
    loading (properties are assigned in definition order, and the warning
    pref loads AFTER this one, briefly reading as its NOTIFY default) the
    re-check sees the real loaded value and stays quiet.
    """
    if self.default_video_source != 'VSE':
        return
    if self.sequencer_export_behavior != 'NOTIFY':
        return
    from .blendcap.helpers import blendcap_state
    if blendcap_state.get('vse_default_prompt_pending'):
        return
    blendcap_state['vse_default_prompt_pending'] = True
    # The prefs live in their own OS window, but a timer's default context
    # targets the main window - the dialog would open BEHIND Preferences.
    # Capture the window the click happened in and re-target the operator.
    src_window = getattr(context, "window", None)

    def _deferred_prompt():
        blendcap_state['vse_default_prompt_pending'] = False
        from .blendcap.helpers import addon_prefs
        prefs = addon_prefs()
        if (prefs is None
                or prefs.default_video_source != 'VSE'
                or prefs.sequencer_export_behavior != 'NOTIFY'):
            return None
        win = src_window
        if win is not None and win not in list(bpy.context.window_manager.windows):
            win = None  # window was closed between click and timer
        try:
            if win is not None:
                with bpy.context.temp_override(window=win):
                    bpy.ops.blendcap.vse_default_prompt('INVOKE_DEFAULT')
            else:
                bpy.ops.blendcap.vse_default_prompt('INVOKE_DEFAULT')
        except RuntimeError:
            # Context can be invalid (e.g. a script-driven change); the
            # runtime fallback treats a lingering NOTIFY as Leave it
            # alone, so failing quiet is safe.
            pass
        return None  # one-shot

    bpy.app.timers.register(_deferred_prompt, first_interval=0.05)


def _detected_backend_label():
    """What the dependency installer detected/installed, or None.

    The live auto-detection (save_mhr_data._get_device) runs in the venv and
    needs torch, which Blender's bundled Python can't import - so we can't run
    it from draw(). Instead we surface what the installer recorded in
    install_marker.json (gpu_description, e.g. "NVIDIA (CUDA)"). _get_device()
    picks the device matching the installed backend at capture time, so this
    reflects the automatic choice (barring a hardware change since install -
    a Reinstall refreshes the marker)."""
    if not is_dependencies_installed():
        return None
    marker = read_install_marker() or {}
    desc = marker.get("gpu_description")
    if desc:
        return desc
    # Older markers may lack a description; infer from the installed extra.
    extra = (marker.get("extra") or "").lower()
    gtype = (marker.get("gpu_type") or "").lower()
    if extra == "cuda" or gtype == "nvidia":
        return "NVIDIA (CUDA)"
    if extra == "directml" or gtype.startswith("amd"):
        return "AMD/Intel (DirectML)"
    return marker.get("gpu_type") or None


class BLENDCAP_AddonPreferences(bpy.types.AddonPreferences):
    # Must equal the add-on's package name. As an installed extension that is
    # bl_ext.<repo>.blendcap, and __package__ resolves to exactly that.
    bl_idname = __package__

    # ------------------------------------------------------------------
    # Persistent user preferences (saved in userpref.blend, global to the
    # install — NOT per-.blend). Where each one is consumed:
    #   cache_*               -> helpers._cache_dir (cache folder resolution)
    #   bvh_library_*         -> helpers._bvh_library_dir (library folder)
    #   capture_backend       -> operators_pipeline._capture_use_ort_override /
    #                            _capture_force_cpu (USE_ORT + CPU env vars
    #                            on the capture subprocess)
    #   refine_hands          -> operators_pipeline capture cmd (--no-hands)
    #   sequencer_export_behavior -> properties.update_video_source
    #   disable_timeline_autosnap -> mocap.load_proxy_to_vse playhead snap +
    #                                operators_pipeline capture start
    #   hide_setup_section    -> panels.draw_blendcap_ui (Setup block)
    #   hide_clear_all        -> panels.draw_blendcap_ui (Clear All button)
    #   hide_donate_buttons   -> panels.draw_blendcap_ui + this draw()
    #                            (honor-system supporter toggle; only
    #                            relevant on source builds where
    #                            SHOW_DONATE_BUTTON is True)
    # ------------------------------------------------------------------

    # Honor-system supporter toggle. Only ever drawn on source builds
    # (paid builds hide the donate buttons entirely via is_paid_build).
    # Deliberately unverified - the button is an ask, not a gate, and
    # the people who tick this without donating were never going to.
    hide_donate_buttons: bpy.props.BoolProperty(  # type: ignore
        name="I've donated - hide the support buttons",
        default=False,
        update=_redraw_viewport_panels,
        description=(
            "Hide BlendCap's Donate / Support buttons here and in the "
            "3D-viewport panel. Works on the honor system - if you've "
            "chipped in (thank you!) or just want the buttons gone, "
            "tick this. Untick any time to bring them back"))

    # --- File & cache locations ---
    cache_use_custom: bpy.props.BoolProperty(  # type: ignore
        name="Set default cache folder",
        default=False,
        description=(
            "Save tracking data (captures, proxies, NPZ files) to a folder "
            "you choose. By default BlendCap saves next to your .blend file, "
            "or to a temporary folder (cleared when Blender closes) if the "
            "project hasn't been saved yet. Turn this on to use your chosen "
            "folder instead of that temporary one, so the data isn't cleared "
            "when Blender closes. Saved projects still keep their data next "
            "to the .blend file unless you also turn on 'Always use this "
            "folder' below"))
    cache_custom_path: bpy.props.StringProperty(  # type: ignore
        name="Cache Folder",
        subtype='DIR_PATH', default="",
        description="Folder used for tracking data when the option above is on")
    cache_always_use_custom: bpy.props.BoolProperty(  # type: ignore
        name="Always use this folder",
        default=False,
        description=(
            "Use the custom cache folder even after the project has been "
            "saved. When off, a saved project keeps its tracking data next "
            "to the .blend file and the custom folder is only used before "
            "the project has been saved"))

    bvh_library_use_custom: bpy.props.BoolProperty(  # type: ignore
        name="Use a custom BVH library folder",
        default=False,
        description=(
            "Read and save the BVH library from a folder you choose instead "
            "of inside the cache folder. Point several projects at one shared "
            "folder to reuse takes across them"))
    bvh_library_path: bpy.props.StringProperty(  # type: ignore
        name="BVH Library Folder",
        subtype='DIR_PATH', default="",
        description="Folder used for the BVH library when the option above is on")

    # --- Capture & tracking ---
    capture_backend: bpy.props.EnumProperty(  # type: ignore
        name="Capture Backend",
        items=[
            ('AUTO', "Automatic (use what was detected at installation)",
             "Use the backend chosen when dependencies were installed - see "
             "the line below. Best choice unless detection got it wrong"),
            ('GPU', "NVIDIA - CUDA",
             "Force the NVIDIA/CUDA engine. Fastest and most accurate; needs "
             "a supported NVIDIA GPU"),
            ('COMPAT', "Non-NVIDIA - DirectML / ONNX (experimental)",
             "Force the DirectML path (via ONNX Runtime) for AMD/Intel GPUs. "
             "Slower, slightly less accurate, and currently experimental"),
            ('CPU', "CPU Only (no GPU - very slow)",
             "Run capture entirely on the CPU with no GPU acceleration. Much "
             "slower than a GPU, but works on any machine and avoids GPU "
             "out-of-memory crashes. The first CPU capture downloads a "
             "one-time ~1.7 GB fast-CPU backend when online; without it, "
             "capture still runs on an even slower fallback. Use for "
             "testing, or as a fallback if a GPU backend keeps failing"),
        ],
        default='AUTO',
        description=(
            "Which capture engine to use. Leave on Automatic unless hardware "
            "detection picked the wrong one"))
    refine_hands: bpy.props.EnumProperty(  # type: ignore
        name="Refine Hand Tracking",
        description="Extra capture pass that sharpens hand pose (slower)",
        items=[
            ('AUTO', "Automatic",
             "Refine on NVIDIA, skip on non-NVIDIA and CPU (where it's slow)"),
            ('ON', "Always on", "Always refine hands"),
            ('OFF', "Always off", "Never refine hands (faster, rougher hands)"),
        ],
        default='AUTO')
    default_video_source: bpy.props.EnumProperty(  # type: ignore
        name="Default Source",
        items=[
            ('FILE', "File Path",
             "New scenes start with the capture source set to a video file "
             "picked from disk"),
            ('VSE', "Video Editor Strip",
             "New scenes start with the capture source set to the active clip "
             "in Blender's Video Sequencer"),
        ],
        default='FILE',
        update=_update_default_video_source,
        description=(
            "Which BlendCap video capture source is selected by default when "
            "you open Blender. Set this to \"Video Editor Strip\" if you "
            "usually prefer to capture videos from the timeline"))
    sequencer_export_behavior: bpy.props.EnumProperty(  # type: ignore
        name="Sequencer Render Warning",
        items=[
            ('NOTIFY', "Ask each time",
             "Warn and offer to turn off Sequencer export when you switch the "
             "source to a Video Editor strip. Not available while Default "
             "Source is Video Editor Strip - new files start on a strip "
             "without a switch to ask on"),
            ('AUTO_DISABLE', "Turn off automatically",
             "Silently turn off Sequencer export when you switch the source to "
             "a Video Editor strip, or when a new file starts on one via "
             "Default Source. Saved projects keep their own setting"),
            ('LEAVE', "Leave it alone",
             "Don't warn and don't change the Sequencer export setting"),
        ],
        default='NOTIFY',
        description=(
            "What to do with Blender's \"Sequencer\" render option when you "
            "switch the capture source to a Video Editor strip. Leaving it on "
            "can make renders export the video strip instead of your scene"))
    disable_timeline_autosnap: bpy.props.BoolProperty(  # type: ignore
        name="Disable playhead auto-snap",
        default=False,
        description=(
            "Keep the timeline where it is when you run tracking from the "
            "Video Editor. By default the playhead jumps to the start of the "
            "clip to be tracked"))

    # --- Interface ---
    hide_setup_section: bpy.props.BoolProperty(  # type: ignore
        name="Hide the Setup section",
        default=False,
        update=_redraw_viewport_panels,
        description=(
            "Hide the \"Setup\" section at the top of the BlendCap sidebar "
            "panel. Turn this on once your workspace is set up and you no "
            "longer need those buttons"))
    hide_clear_all: bpy.props.BoolProperty(  # type: ignore
        name="Hide the \"Clear All\" cache button",
        default=False,
        update=_redraw_viewport_panels,
        description=(
            "Hide the \"Clear All\" button so the whole capture cache can't "
            "be deleted by accident. \"Clear Cache\" (current clip only) "
            "stays available. Useful when several projects share one cache "
            "folder"))

    # ------------------------------------------------------------------
    # Draw - split into per-section helpers so blocks are easy to reorder.
    # ------------------------------------------------------------------
    def draw(self, context):
        layout = self.layout

        # When the install wizard is active it takes over this panel (it's
        # drawn inline rather than as a popup so its buttons are greyable,
        # correctly ordered, and never clipped). Cancel/Install collapse it
        # back to the idle view below.
        if getattr(context.window_manager, "blendcap_install_active", False):
            draw_install_wizard(context, layout)
            return

        # Mirrors the N-panel's top placement; hidden on paid builds by
        # the same helpers.is_paid_build-derived flag. On source builds
        # the honor-system toggle stays visible either way (it's how a
        # supporter re-shows the buttons), and ticking it swaps the
        # button for a thank-you.
        if SHOW_DONATE_BUTTON:
            if self.hide_donate_buttons:
                layout.label(text="Thank you for supporting BlendCap!",
                             icon='FUND')
            else:
                row = layout.row()
                row.operator("blendcap.donate", text="Donate / Support",
                             icon='FUND')
            layout.prop(self, "hide_donate_buttons")
            layout.separator()

        self._draw_dependencies(context, layout)
        self._draw_locations(context, layout)
        self._draw_capture(context, layout)
        self._draw_interface(context, layout)

    def _draw_dependencies(self, context, layout):
        box = layout.box()
        box.label(text="AI Tracking Dependencies", icon='CONSOLE')

        installed = is_dependencies_installed()
        # The public GitHub repo is a source-availability artifact that
        # ships WITHOUT the installer/ folder; the paid zips include it.
        # Derive the UI from what's on disk so one codebase serves both
        # channels (no forked builds).
        has_installer = installer_files_present()
        if installed:
            box.label(text="Dependencies installed.", icon='CHECKMARK')
            marker = read_install_marker() or {}
            details = []
            gpu = marker.get("gpu_description") or marker.get("gpu_type")
            if gpu:
                details.append(f"GPU: {gpu}")
            if marker.get("torch_version"):
                details.append(f"torch {marker['torch_version']}")
            if marker.get("ort_version"):
                details.append(f"ORT {marker['ort_version']}")
            if details:
                box.label(text="    " + "    ".join(details))
            if has_installer:
                row = box.row()
                row.operator("blendcap.install_dependencies",
                             text="Reinstall Dependencies",
                             icon='FILE_REFRESH').force = True
        elif not has_installer:
            # Source distribution, nothing installed: there is no
            # installer to launch. A hand-built venv at <root>/venv is
            # picked up automatically (the `installed` branch above).
            col = box.column(align=True)
            _label_wrapped(
                col, context,
                "This is the source distribution - the one-click "
                "dependency installer ships with the paid BlendCap. "
                "Running from source means building the Python "
                "environment at venv/ and fetching the model weights "
                "into checkpoints/ yourself (see requirements.txt; "
                "unsupported).", icon='INFO')
        else:
            col = box.column(align=True)
            col.label(text="Not installed yet. This sets up the bundled Python")
            col.label(text="environment and downloads the AI models (about 11 GB).")
            col.label(text="You'll be shown the Meta SAM License to accept first.")
            row = box.row()
            row.scale_y = 1.3
            row.operator("blendcap.install_dependencies",
                         text="Install Dependencies",
                         icon='IMPORT').force = False

        # Resolved install folder - handy when verifying which copy is live.
        info = box.column(align=True)
        info.label(text="Install folder:", icon='FILE_FOLDER')
        info.label(text="    " + _addon_dir())
        # Open it in the OS file browser. wm.path_open is Blender's built-in
        # cross-platform opener (Explorer / Finder / xdg-open) and handles
        # directories, so no dedicated operator is needed here. Uninstall sits
        # alongside it but is only meaningful when an environment exists.
        row = box.row(align=True)
        row.operator("wm.path_open", text="Open Install Folder",
                     icon='FILE_FOLDER').filepath = _addon_dir()
        # Uninstall uses an INLINE arm -> confirm flow (no popup: dialogs
        # launched from the Preferences window get clipped - same reason the
        # install wizard is inline). First click arms; a red confirm/cancel
        # block then appears below.
        armed = getattr(context.window_manager,
                        "blendcap_uninstall_armed", False)
        sub = row.row(align=True)
        sub.enabled = installed and not armed
        sub.alert = True
        sub.operator("blendcap.uninstall_dependencies",
                     text="Uninstall", icon='TRASH').mode = 'ARM'
        if installed and armed:
            warn = box.box()
            warn.alert = True
            col = warn.column(align=True)
            _label_wrapped(col, context,
                           "Delete the Python environment and downloaded AI "
                           "models (several GB)?", icon='ERROR')
            _label_wrapped(col, context,
                           "Blender may freeze briefly while they're removed, "
                           "and you'll need to reinstall before capturing again.")
            r = warn.row(align=True)
            r.operator("blendcap.uninstall_dependencies",
                       text="Confirm Uninstall", icon='TRASH').mode = 'RUN'
            r.operator("blendcap.uninstall_dependencies",
                       text="Cancel", icon='X').mode = 'CANCEL'

    def _draw_locations(self, context, layout):
        box = layout.box()
        box.label(text="File & Cache Locations", icon='FILE_FOLDER')

        # Default cache / assets folder.
        box.prop(self, "cache_use_custom")
        sub = box.column(align=True)
        sub.enabled = self.cache_use_custom
        sub.prop(self, "cache_custom_path")
        sub.prop(self, "cache_always_use_custom")

        box.separator()

        # BVH library folder.
        box.prop(self, "bvh_library_use_custom")
        sub = box.column(align=True)
        sub.enabled = self.bvh_library_use_custom
        sub.prop(self, "bvh_library_path")

    def _draw_capture(self, context, layout):
        box = layout.box()
        box.label(text="Capture & Tracking", icon='CAMERA_DATA')

        # Capture backend / hardware override.
        col = box.column(align=True)
        col.prop(self, "capture_backend")
        if self.capture_backend == 'AUTO':
            detected = _detected_backend_label()
            if detected:
                col.label(text="Detected at installation: " + detected, icon='CHECKMARK')
            elif not is_dependencies_installed():
                col.label(text="Install dependencies to detect your hardware.",
                          icon='INFO')
            else:
                col.label(text="Hardware not recorded yet - try Reinstall.",
                          icon='INFO')
        elif self.capture_backend == 'CPU':
            col.label(text="Runs on CPU only - expect minutes per frame.",
                      icon='INFO')

        box.prop(self, "refine_hands")

        box.separator()

        # Which source a fresh scene starts on (File Path vs Video Editor
        # strip). Applied to new/startup files by registration's load_post
        # handler + startup timer; saved projects keep their stored choice.
        box.prop(self, "default_video_source")

        box.separator()

        # Video Editor behaviors.
        box.label(text="Video Editor:")
        # Drawn as three side-by-side buttons instead of a dropdown so the
        # 'Ask each time' item alone can be greyed out: while Default Source
        # is Video Editor Strip, new files start on a strip with no
        # interactive switch to hook, so ask-each-time can never fire for
        # them (the _update_default_video_source dialog converts users off
        # it; this keeps them from selecting it back).
        seq_col = box.column(align=True)
        seq_col.label(text="Sequencer Render Warning:")
        seq_row = seq_col.row(align=True)
        ask = seq_row.row(align=True)
        ask.enabled = self.default_video_source != 'VSE'
        ask.prop_enum(self, "sequencer_export_behavior", 'NOTIFY')
        seq_row.prop_enum(self, "sequencer_export_behavior", 'AUTO_DISABLE')
        seq_row.prop_enum(self, "sequencer_export_behavior", 'LEAVE')
        if self.default_video_source == 'VSE':
            note = box.column(align=True)
            _label_wrapped(note, context,
                           "\"Ask each time\" is unavailable while Default "
                           "Source is Video Editor Strip - new files start "
                           "on a strip, with no source switch to ask on.",
                           icon='INFO')
        box.prop(self, "disable_timeline_autosnap")

    def _draw_interface(self, context, layout):
        box = layout.box()
        box.label(text="Interface", icon='HIDE_OFF')

        box.prop(self, "hide_setup_section")
        box.prop(self, "hide_clear_all")

        box.separator()
        col = box.column(align=True)
        _label_wrapped(col, context,
                       "In order to keep the BlendCap workspace in Blender by "
                       "default:", icon='INFO')
        _label_wrapped(col, context,
                       "1. Open it once from the sidebar Setup section.")
        _label_wrapped(col, context,
                       "2. File > Defaults > Save Startup File.")
        _label_wrapped(col, context,
                       "3. It's best to do this from a fresh project - the startup "
                       "saves your entire scene, even which monitor Blender opens on.")
        _label_wrapped(col, context,
                       "4. The startup file can be reverted to Blender's default by "
                       "deleting it (path below).")
        # Show the startup-file path plus a button that opens its FOLDER, so the
        # user can find and delete it. Pointing wm.path_open at the .blend itself
        # would just relaunch the file; opening the folder reveals it instead.
        cfg = ""
        try:
            cfg = bpy.utils.user_resource('CONFIG') or ""
        except Exception:
            cfg = ""
        if cfg:
            prow = col.row(align=True)
            prow.label(text="    " + os.path.join(cfg, "startup.blend"))
            prow.operator("wm.path_open", text="Open Folder",
                          icon='FILE_FOLDER').filepath = cfg
        else:
            col.label(text="    (in Blender's user config folder)")


classes = (
    BLENDCAP_AddonPreferences,
)


def register():
    # Feature classes/props first so the operators the preferences UI
    # references already exist, then the preferences panel itself.
    _registration.register()
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
    _registration.unregister()
