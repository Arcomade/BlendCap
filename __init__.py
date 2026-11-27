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
    install_incomplete, INSTALL_INCOMPLETE_NOTE,
    installer_files_present, deps_root, deps_in_addon_folder,
    deps_target, presets_root, _dir_has_venv, deps_unreachable_path,
    _inside_addon_dir, storage_speed_warning, _invalidate_deps_cache,
    pending_user_data_moves,
)
from .blendcap.operators_pipeline import draw_install_wizard


def _deps_pref_changed():
    """The storage prefs changed - the resolved deps location may have
    too. Cache refresh only; nothing on disk moves on a pref change
    (an 11 GB move needs the explicit button)."""
    _invalidate_deps_cache()


def _capture_backend_changed():
    """Capture Backend changed - the hardware half of the Refine Hand
    Tracking 'Automatic' default may have changed with it."""
    try:
        from .blendcap.operators_pipeline import (
            invalidate_refine_hw_default)
        invalidate_refine_hw_default()
    except Exception:
        pass


def _sequencer_guard_pref_changed():
    """Attach or detach the sequencer guard handler to match the two
    'keep project settings' checkboxes (both off = no handler at all)."""
    try:
        from .blendcap import sequencer_guard
        sequencer_guard.sync_handler()
    except Exception as e:
        print(f"[BlendCap] sequencer guard update failed: {e}")


def _presets_pref_changed():
    """Choosing a preset location IS the move - presets are tiny and
    instant, so unlike the dependencies there is no separate button.
    The handler consolidates from the previous location; invalid
    paths resolve to no change and move nothing. A failure must not
    break the prefs UI, but it must be VISIBLE - a swallowed error
    here looks exactly like 'picked a folder, nothing happened'."""
    try:
        from .blendcap.helpers import on_presets_location_changed
        on_presets_location_changed()
    except Exception:
        import traceback
        print("[BlendCap] preset relocation failed:")
        traceback.print_exc()
# Single shared channel flag (evaluated once at registration in panels):
# True only for the source distribution, where donating is the ask.
# Read as a module attribute at draw time (not imported by value): the
# flag is recomputed at register, after the installer relocation.
from .blendcap import panels as _panels


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
    #   keep_view_transform_on_video_add / keep_frame_rate_on_video_add /
    #   disable_sequencer_render_on_video_add /
    #   sequencer_guard_workspace_only -> sequencer_guard.py (+ the
    #                            source-switch trigger in
    #                            properties.update_video_source)
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

    # --- Dependency storage ---
    # Where the installed dependencies (Python environment + tracking
    # models, ~13 GB) live. Default: Blender's per-extension user
    # folder, which survives addon updates. The custom path is the
    # "install to another drive" feature. Changing these only sets the
    # TARGET - installs land there, and the Move button relocates an
    # existing install; nothing moves by itself on a pref change.
    deps_use_custom: bpy.props.BoolProperty(  # type: ignore
        name="Store dependencies in a custom folder",
        default=False,
        update=lambda self, ctx: _deps_pref_changed(),
        description=(
            "Keep the installed dependency files (Python environment and "
            "tracking models, about 11 GB) in a folder you choose - for "
            "example on a bigger drive. With this off they're kept in "
            "Blender's own user folder for BlendCap. Either way they "
            "survive addon updates"))
    deps_custom_path: bpy.props.StringProperty(  # type: ignore
        name="Dependency Folder",
        subtype='DIR_PATH', default="",
        update=lambda self, ctx: _deps_pref_changed(),
        description=(
            "Folder used for the dependency files when the option above "
            "is on. New installs go here; use 'Move Dependency Files' "
            "below to relocate an existing install"))

    # --- Preset storage ---
    # Separate from the dependency location on purpose: presets are
    # kilobytes and some users will want them on a different (always-
    # present) drive than a 13 GB model set.
    presets_use_custom: bpy.props.BoolProperty(  # type: ignore
        name="Store custom presets in a custom folder",
        default=False,
        update=lambda self, ctx: _presets_pref_changed(),
        description=(
            "Keep your custom presets (saved retarget maps and rest "
            "poses) in a 'BlendCap Presets' folder inside a location "
            "you choose. They move there the moment you pick it. With "
            "this off they're kept in Blender's own user folder for "
            "BlendCap. Either way they survive addon updates. The "
            "bundled presets ship with the addon and never move"))
    presets_custom_path: bpy.props.StringProperty(  # type: ignore
        name="Preset Folder",
        subtype='DIR_PATH', default="",
        update=lambda self, ctx: _presets_pref_changed(),
        description=(
            "Your custom presets move into a 'BlendCap Presets' folder "
            "here the moment you pick it (they're small - this is "
            "instant)"))
    hide_storage_warning: bpy.props.BoolProperty(  # type: ignore
        name="Don't show the startup warning about files in the "
             "addon folder",
        default=False,
        description=(
            "Stops the reminder window that appears at startup while "
            "your custom presets or dependency files are still stored "
            "inside the addon folder. The files stay where they are, "
            "and an addon update can still delete them - the warnings "
            "in this panel remain either way"))

    # --- Video Editor: what happens when video is added ---
    # Blender's Add Movie / Add Image operators switch the scene's view
    # transform to Standard and (movies only) the frame rate to the
    # clip's, with no preference to stop it, and every new file has
    # Output > Post Processing > Sequencer on. See sequencer_guard.py.
    keep_view_transform_on_video_add: bpy.props.BoolProperty(  # type: ignore
        name="Keep the project's view transform",
        default=True,
        update=lambda self, ctx: _sequencer_guard_pref_changed(),
        description=(
            "Blender switches the scene's view transform to Standard "
            "when a video or image sequence is added to the sequencer. "
            "Keep the project's own setting (AgX, Filmic...) instead. "
            "Under AgX or Filmic the new clip's colour space is set to "
            "'AgX Base sRGB' or 'Filmic sRGB' so it displays exactly "
            "as-is; linear strips such as EXR are left alone"))
    keep_frame_rate_on_video_add: bpy.props.BoolProperty(  # type: ignore
        name="Keep the project's frame rate",
        default=True,
        update=lambda self, ctx: _sequencer_guard_pref_changed(),
        description=(
            "Blender sets the project frame rate to the clip's when a "
            "video is added to the sequencer. Keep the project's frame "
            "rate instead; the clip still plays at normal speed, and a "
            "capture from a clip with a different frame rate offers "
            "the usual timing options"))
    disable_sequencer_render_on_video_add: bpy.props.BoolProperty(  # type: ignore
        name="Turn off Sequencer rendering",
        default=True,
        update=lambda self, ctx: _sequencer_guard_pref_changed(),
        description=(
            "Blender's Output > Post Processing > Sequencer option is on "
            "in every new file and makes renders output the video strips "
            "instead of your scene. Turn it off when a clip is added to "
            "the sequencer or the capture source switches to Video "
            "Editor Strip. Saved projects keep their own setting until "
            "a clip is added"))
    sequencer_guard_workspace_only: bpy.props.BoolProperty(  # type: ignore
        name="Only while the BlendCap workspace tab is active",
        default=True,
        update=lambda self, ctx: _sequencer_guard_pref_changed(),
        description=(
            "The three options above will only trigger while the "
            "Performance Capture workspace tab is the currently active "
            "workspace. Untick to have them work in any workspace with "
            "the sequencer"))

    # --- Updates (blendcap/updater.py) ---
    check_for_updates: bpy.props.BoolProperty(  # type: ignore
        name="Automatically check for updates",
        default=True,
        description=(
            "Once a day, shortly after Blender starts, look for a newer "
            "BlendCap release and offer to install it. Needs Blender's "
            "online access (Preferences > System)"))
    show_update_notice_in_sidebar: bpy.props.BoolProperty(  # type: ignore
        name="Show update notice in the sidebar",
        default=True,
        update=_redraw_viewport_panels,
        description=(
            "When a newer BlendCap release is known, show a small notice at "
            "the top of the BlendCap sidebar panel. Remind me tomorrow and "
            "Skip this version hide it too"))
    # Bookkeeping for the check, not drawn as controls.
    # Timestamps are kept as text: float properties are 32-bit and lose
    # about two minutes at Unix-time magnitudes.
    update_last_check: bpy.props.StringProperty(  # type: ignore
        name="Last update check", default="", options={'HIDDEN'})
    update_remind_after: bpy.props.StringProperty(  # type: ignore
        name="Remind after", default="", options={'HIDDEN'})
    update_skip_version: bpy.props.StringProperty(  # type: ignore
        name="Skipped version", default="", options={'HIDDEN'})
    update_latest_version: bpy.props.StringProperty(  # type: ignore
        name="Latest known version", default="", options={'HIDDEN'})
    update_latest_url: bpy.props.StringProperty(  # type: ignore
        name="Latest known download", default="", options={'HIDDEN'})
    update_latest_sha256: bpy.props.StringProperty(  # type: ignore
        name="Latest known checksum", default="", options={'HIDDEN'})
    update_latest_notes: bpy.props.StringProperty(  # type: ignore
        name="Latest known notes", default="", options={'HIDDEN'})
    update_latest_installer: bpy.props.StringProperty(  # type: ignore
        name="Latest required installer", default="", options={'HIDDEN'})
    update_latest_deps_revision: bpy.props.IntProperty(  # type: ignore
        name="Latest dependency revision", default=0, options={'HIDDEN'})
    update_latest_source: bpy.props.StringProperty(  # type: ignore
        name="Index the latest result came from", default="",
        options={'HIDDEN'})

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
        update=lambda self, ctx: _capture_backend_changed(),
        description=(
            "Which capture engine to use. Leave on Automatic unless hardware "
            "detection picked the wrong one"))
    show_footage_in_cameras: bpy.props.BoolProperty(  # type: ignore
        name="Show Footage in BlendCap Cameras",
        description="Show the source video as the background of cameras "
                    "BlendCap adds",
        default=True)
    refine_hands: bpy.props.EnumProperty(  # type: ignore
        name="Refine Hand Tracking Default",
        description="Default state of the Refine Hand Tracking "
                    "checkbox in the capture panel. The checkbox can "
                    "be flipped freely during a Blender session; this "
                    "sets where it starts every launch",
        items=[
            ('AUTO', "Automatic",
             "Start on for NVIDIA, off for non-NVIDIA and CPU (where "
             "the extra pass is very slow)"),
            ('ON', "On by default",
             "Start every session with hand refinement on"),
            ('OFF', "Off by default",
             "Start every session with hand refinement off"),
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
        description=(
            "Which BlendCap video capture source is selected by default when "
            "you open Blender. Set this to \"Video Editor Strip\" if you "
            "usually prefer to capture videos from the timeline"))
    # LEGACY (pre-1.0.6), not drawn: the three-way Sequencer Render
    # Warning. Kept so existing preference files still load; its stored
    # value is carried over ONCE into disable_sequencer_render_on_video_add
    # by sequencer_guard._migrate_legacy ('Leave it alone' -> off).
    sequencer_export_behavior: bpy.props.EnumProperty(  # type: ignore
        name="Sequencer Render Warning (legacy)",
        items=[
            ('NOTIFY', "Ask each time", ""),
            ('AUTO_DISABLE', "Turn off automatically", ""),
            ('LEAVE', "Leave it alone", ""),
        ],
        default='NOTIFY',
        options={'HIDDEN'},
        description="Replaced by 'Turn off Sequencer rendering'")
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
    sac_block_seen: bpy.props.BoolProperty(  # type: ignore
        name="Smart App Control blocked a file", default=False,
        options={'HIDDEN'})
    hide_clear_all: bpy.props.BoolProperty(  # type: ignore
        name="Hide the \"Clear All\" cache button",
        default=False,
        update=_redraw_viewport_panels,
        description=(
            "Hide the \"Clear All\" button so the whole capture cache can't "
            "be deleted by accident. \"Clear Cache\" (current clip only) "
            "stays available. Useful when several projects share one cache "
            "folder"))

    # --- Section dropdowns (same arrow-in-the-header pattern as the
    # sidebar). Saved with the preferences, so a section stays the way
    # the user left it. ---
    updates_expanded: bpy.props.BoolProperty(  # type: ignore
        name="Show Updates",
        default=True,
        description="Expand the Updates section.")
    deps_expanded: bpy.props.BoolProperty(  # type: ignore
        name="Show AI Tracking Dependencies",
        default=True,
        description="Expand the AI Tracking Dependencies section.")
    locations_expanded: bpy.props.BoolProperty(  # type: ignore
        name="Show File & Cache Locations",
        default=False,
        description="Expand the File & Cache Locations section.")
    capture_expanded: bpy.props.BoolProperty(  # type: ignore
        name="Show Capture & Tracking",
        default=False,
        description="Expand the Capture & Tracking section.")
    interface_expanded: bpy.props.BoolProperty(  # type: ignore
        name="Show Interface",
        default=False,
        description="Expand the Interface section.")

    # ------------------------------------------------------------------
    # Draw - split into per-section helpers so blocks are easy to reorder.
    # ------------------------------------------------------------------
    def _section(self, layout, prop, title, icon, needs_action=False):
        """A section's box and header, drawn like the sidebar's: an arrow
        that toggles `prop`, then the title. Returns the box to draw the
        body into, or None while the section is collapsed.

        needs_action: the section holds something the user has to deal
        with (not installed, a blocked update, files at risk). It then
        draws open whatever the arrow says, with the arrow greyed, so the
        sidebar's "Preferences" buttons and red strips never land on a
        closed header. The stored choice applies again once it clears."""
        box = layout.box()
        header = box.row(align=True)
        is_open = bool(needs_action or getattr(self, prop))
        arrow = header.row(align=True)
        arrow.enabled = not needs_action
        arrow.prop(self, prop,
                   icon='TRIA_DOWN' if is_open else 'TRIA_RIGHT',
                   emboss=False, text="")
        header.label(text=title, icon=icon)
        return box if is_open else None

    def _updates_need_action(self):
        try:
            from .blendcap import updater
            if updater.managed_repo() or updater.index_url() is None:
                return False
            if updater.repo_lock_stale():
                return True
            # Same test as the sidebar notice: an update the user has
            # not snoozed or skipped.
            return bool(updater.update_available()
                        and not updater.notice_state())
        except Exception:
            return False

    def _deps_need_action(self, context):
        try:
            from .blendcap.helpers import (deps_outdated, installer_outdated,
                                           smart_app_control_blocking)
            from .blendcap.operators_storage import MOVE_STATE
            if not is_dependencies_installed():
                return True
            return bool(
                install_incomplete() or smart_app_control_blocking()
                or deps_outdated() or installer_outdated()
                or MOVE_STATE["running"] or deps_in_addon_folder()
                or getattr(context.window_manager,
                           "blendcap_uninstall_armed", False))
        except Exception:
            return True

    def _locations_need_action(self):
        try:
            return bool(pending_user_data_moves()["presets"])
        except Exception:
            return False

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
        if _panels.SHOW_DONATE_BUTTON:
            if self.hide_donate_buttons:
                layout.label(text="Thank you for supporting BlendCap!",
                             icon='FUND')
            else:
                row = layout.row()
                row.operator("blendcap.donate", text="Donate / Support",
                             icon='FUND')
            layout.prop(self, "hide_donate_buttons")
            layout.separator()

        self._draw_updates(context, layout)
        self._draw_dependencies(context, layout)
        self._draw_locations(context, layout)
        self._draw_capture(context, layout)
        self._draw_interface(context, layout)

    def _draw_updates(self, context, layout):
        from .blendcap import updater
        import time as _time
        box = self._section(layout, "updates_expanded", "Updates",
                            'FILE_REFRESH', self._updates_need_action())
        if box is None:
            return
        # Installed in a remote repository (a store or studio repository
        # under Get Extensions): Blender updates it, and that channel
        # delivers the store's own zip, installer included. This
        # updater stands down - one update channel per install.
        repo_name = updater.managed_repo()
        if repo_name:
            col = box.column()
            col.enabled = False
            col.prop(self, "check_for_updates")
            col.prop(self, "show_update_notice_in_sidebar")
            _label_wrapped(box.column(align=True), context,
                           f"This copy is updated through the '{repo_name}' "
                           f"repository. Use Get Extensions to update it.",
                           icon='INFO')
            box.operator("blendcap.open_docs", text="Documentation",
                         icon='HELP')
            return
        dev = updater.is_dev_build()
        off = updater.index_url() is None
        col = box.column()
        col.enabled = not off
        col.prop(self, "check_for_updates")
        col.prop(self, "show_update_notice_in_sidebar")
        if off:
            box.label(text="Disabled in development builds.", icon='INFO')
            box.operator("blendcap.open_docs", text="Documentation",
                         icon='HELP')
            return
        if dev:
            box.label(text="Development build: checking the test index "
                           "from BLENDCAP_UPDATE_INDEX_URL.", icon='INFO')
        try:
            online = bool(bpy.app.online_access)
        except Exception:
            online = True
        if not online:
            _label_wrapped(box.column(align=True), context,
                           "Blender's online access is off (Preferences > "
                           "System > Network), so BlendCap can't look for "
                           "updates.", icon='INFO')
        row = box.row(align=True)
        row.operator("blendcap.update_check_now", icon='FILE_REFRESH')
        last = updater.pref_time(self, "update_last_check")
        if updater._STATE['checking']:
            row.label(text="Checking...")
        elif last > 0:
            row.label(text="Last checked " + _time.strftime(
                "%Y-%m-%d %H:%M", _time.localtime(last)))
        else:
            row.label(text="Not checked yet.")
        if updater.repo_lock_stale():
            # Blocks Blender's own Install from Disk / drag-and-drop too,
            # so say it here regardless of whether an update is known.
            warn = box.column(align=True)
            warn.alert = True
            _label_wrapped(warn, context,
                           updater.update_block_message('lock'),
                           icon='ERROR')
            warn.operator("blendcap.repo_unlock", text="Unlock Extensions",
                          icon='UNLOCKED')
        latest = updater.known_latest()
        if updater.update_available():
            state = updater.notice_state()
            if not state:
                # The box shows the same choices as the update window,
                # so drawing it counts as having shown the notice.
                updater.notice_shown_in_prefs()
            col = box.column(align=True)
            col.label(text=f"BlendCap {latest['version']} is available "
                           f"(you have {updater.current_version()}).",
                      icon='INFO')
            reason = updater.update_block_reason(latest)
            row = col.row(align=True)
            row.scale_y = 1.2
            sub = row.row(align=True)
            sub.enabled = not reason   # legacy deps / presets / installer too old
            sub.operator("blendcap.update_now", text="Update Now",
                         icon='IMPORT')
            if latest.get('notes'):
                row.operator("blendcap.update_whats_new", icon='URL')
            # Both choices always drawn. Remind greys out while a
            # reminder is running or the version is skipped; Skip greys
            # out once used. Both come back with the next release.
            row = col.row(align=True)
            sub = row.row(align=True)
            sub.enabled = state == ''
            sub.operator("blendcap.update_dismiss",
                         text="Remind Me Tomorrow").action = 'LATER'
            sub = row.row(align=True)
            sub.enabled = state != 'SKIPPED'
            sub.operator("blendcap.update_dismiss",
                         text="Skip This Version").action = 'SKIP'
            if not reason and updater.rebuild_after_update(latest):
                _label_wrapped(col, context,
                               "This update needs a dependency rebuild "
                               "afterwards.", icon='ERROR')
            if state == 'SKIPPED':
                _label_wrapped(col, context,
                               "'Skip this version' has been selected, "
                               "you will be notified the next time an "
                               "update is released.")
            elif state == 'SNOOZED':
                _label_wrapped(col, context,
                               "'Remind me tomorrow' has been selected, "
                               "you will be reminded on the next launch "
                               "after 24 hours.")
            if reason and reason != 'lock':   # 'lock' is drawn above
                warn = box.column(align=True)
                warn.alert = True
                _label_wrapped(warn, context,
                               updater.update_block_message(reason),
                               icon='ERROR')
                if reason == 'installer':
                    warn.operator("wm.url_open", text="Open Store Page",
                                  icon='URL').url = updater.STORE_URL
                elif reason == 'legacy':
                    warn.operator("blendcap.move_deps",
                                  text="Move Dependency Files",
                                  icon='EXPORT')
                elif reason == 'presets':
                    warn.operator("blendcap.move_presets",
                                  text="Move Presets to Safety",
                                  icon='EXPORT')
                elif reason == 'lock':
                    warn.operator("blendcap.repo_unlock",
                                  text="Unlock Extensions",
                                  icon='UNLOCKED')
        elif latest and last > 0:
            box.label(text=f"BlendCap {updater.current_version()} is "
                           f"up to date.", icon='CHECKMARK')
        box.operator("blendcap.open_docs", text="Documentation", icon='HELP')

    def _draw_dependencies(self, context, layout):
        box = self._section(layout, "deps_expanded",
                            "AI Tracking Dependencies", 'CONSOLE',
                            self._deps_need_action(context))
        if box is None:
            return

        installed = is_dependencies_installed()
        # The public GitHub repo is a source-availability artifact that
        # ships WITHOUT the installer/ folder; the paid zips include it.
        # Derive the UI from what's on disk so one codebase serves both
        # channels (no forked builds).
        has_installer = installer_files_present()
        # Windows Smart App Control (enforcing) blocks the engine's
        # binaries: say so first, and grey every Install/Reinstall
        # button below (cached flag, no checking at draw time).
        sac = self._draw_sac_notice(context, box)
        if installed and install_incomplete():
            # A venv exists but the installer never wrote its success
            # marker (closed, crashed, connection died). Reinstall
            # resumes: uv sync and the model downloads pick up where
            # they stopped.
            warn = box.column(align=True)
            warn.alert = True
            _label_wrapped(warn, context, INSTALL_INCOMPLETE_NOTE,
                           icon='ERROR')
            if has_installer:
                row = box.row()
                row.scale_y = 1.3
                row.enabled = not sac
                row.operator("blendcap.install_dependencies",
                             text="Reinstall Dependencies",
                             icon='FILE_REFRESH').force = True
            self._draw_installer_outdated(context, box)
        elif installed:
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
                row.enabled = not sac
                row.operator("blendcap.install_dependencies",
                             text="Reinstall Dependencies",
                             icon='FILE_REFRESH').force = True
            self._draw_deps_outdated(context, box, has_installer)
            self._draw_installer_outdated(context, box)
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
            # An unreachable custom folder (unplugged external drive)
            # gets its own story - the files exist, so "Install
            # Dependencies" (an 11 GB re-download) is the wrong advice.
            missing_at = deps_unreachable_path()
            if missing_at:
                warn = box.column(align=True)
                warn.alert = True
                _label_wrapped(
                    warn, context,
                    "Dependencies not found. They are set to live at "
                    f"{missing_at} but that folder can't be reached "
                    "right now. If it's on an external drive, connect "
                    "the drive - installing below would download "
                    "everything again instead.", icon='ERROR')
            col = box.column(align=True)
            col.label(text="Not installed yet. This sets up the bundled Python")
            col.label(text="environment and downloads the AI models (about 11 GB).")
            col.label(text="You'll be shown the Meta SAM License to accept first.")
            row = box.row()
            row.scale_y = 1.3
            row.enabled = not sac
            row.operator("blendcap.install_dependencies",
                         text="Install Dependencies",
                         icon='IMPORT').force = False

        # The folder that belongs NEXT TO the install/uninstall buttons
        # is where the dependencies actually live - showing the ADDON
        # folder here confused testing 2026-08-29 (that button now
        # lives in File & Cache Locations as "Open Addon Folder").
        info = box.column(align=True)
        info.label(text="Dependency files folder:", icon='FILE_FOLDER')
        info.label(text="    " + deps_root())
        # Open it in the OS file browser. wm.path_open is Blender's built-in
        # cross-platform opener (Explorer / Finder / xdg-open) and handles
        # directories, so no dedicated operator is needed here. Uninstall sits
        # alongside it but is only meaningful when an environment exists.
        row = box.row(align=True)
        row.operator("wm.path_open", text="Open Dependency Folder",
                     icon='FILE_FOLDER').filepath = deps_root()
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

        # --- Dependency storage location -------------------------------
        # Lives HERE, next to Install/Uninstall and the folder display,
        # because the checkbox chooses where the install goes (user
        # request 2026-08-30; it used to sit in File & Cache Locations,
        # two boxes away from the install button it configures).
        # Three states: steady (the folder rows above say it all),
        # custom-chosen-but-not-moved-yet (instruction + Move - a
        # multi-GB move needs an explicit click), and mid-move
        # (progress bar). The path field is HIDDEN while the checkbox
        # is off - nothing suggests a second location exists.
        from .blendcap.operators_storage import MOVE_STATE
        box.separator()
        box.prop(self, "deps_use_custom")
        if self.deps_use_custom:
            sub = box.column(align=True)
            sub.prop(self, "deps_custom_path")
            self._draw_path_feedback(box, context, True,
                                     self.deps_custom_path,
                                     check_speed=True)
        if MOVE_STATE["running"]:
            move_col = box.column(align=True)
            if MOVE_STATE["total"]:
                move_col.progress(
                    factor=MOVE_STATE["done"] / MOVE_STATE["total"],
                    type='BAR',
                    text=MOVE_STATE["msg"] or "Moving...")
            else:
                move_col.label(text=MOVE_STATE["msg"] or "Moving...",
                               icon='TIME')
            move_col.label(text="Keep Blender open until this finishes.")
            # Greys out by itself once the copy phase ends (poll).
            box.operator("blendcap.cancel_move_deps",
                         text="Cancel Move", icon='X')
        else:
            actual = deps_root()
            target = deps_target()
            mismatched = (_dir_has_venv(actual)
                          and os.path.normcase(actual)
                          != os.path.normcase(target))
            if deps_in_addon_folder():
                warn = box.column(align=True)
                warn.alert = True
                _label_wrapped(
                    warn, context,
                    "Dependency files are stored inside the addon "
                    "folder, where an addon update can delete them. "
                    "Click below to move them to:", icon='ERROR')
                dst = box.column(align=True)
                dst.label(text="    " + target)
                box.operator("blendcap.move_deps",
                             text="Move Dependency Files", icon='EXPORT')
            elif mismatched:
                # The folder rows above already show the CURRENT
                # location (deps_root follows the files) with its Open
                # button, so this block only names the destination.
                where = "custom" if self.deps_use_custom else "default"
                col = box.column(align=True)
                col.label(text=f"Click below to move the dependencies "
                               f"to the {where} folder:")
                col.label(text="    " + target)
                box.operator("blendcap.move_deps",
                             text="Move Dependency Files",
                             icon='EXPORT')

        # Startup-notice opt-out: shown only while there is something
        # the notice would fire about, so it never lingers as clutter.
        pend = pending_user_data_moves()
        if pend["presets"] or pend["deps"]:
            box.separator()
            box.prop(self, "hide_storage_warning")

    def _draw_deps_outdated(self, context, box, has_installer):
        """Red note when the installed environment was built for an
        earlier BlendCap (helpers.deps_outdated: the revision stamped in
        the venv is behind DEPS_REVISION). Paid copies rebuild with the
        installer; source copies rebuild by hand from requirements.txt
        and then press Dependencies Rebuilt."""
        from .blendcap.helpers import deps_outdated
        if not deps_outdated():
            return
        warn = box.column(align=True)
        warn.alert = True
        _label_wrapped(
            warn, context,
            "Your dependencies were built for an earlier version of "
            "BlendCap. It is recommended to rebuild them according to "
            "the updated requirements.txt.", icon='ERROR')
        row = box.row(align=True)
        from .blendcap.helpers import smart_app_control_blocking
        row.enabled = not smart_app_control_blocking()
        if has_installer:
            # Paid: the installer rewrites the stamp when it finishes;
            # nothing to confirm by hand.
            row.operator("blendcap.install_dependencies",
                         text="Rebuild Dependencies",
                         icon='FILE_REFRESH').force = True
        else:
            # Source: rebuilt by hand, so the user says when it's done.
            row.operator("blendcap.open_docs", text="Documentation",
                         icon='HELP')
            row.operator("blendcap.deps_mark_rebuilt",
                         text="Dependencies Rebuilt", icon='CHECKMARK')

    def _draw_sac_notice(self, context, box):
        """Red note when Windows Smart App Control is enforcing
        (helpers.smart_app_control_blocking: one cached registry read).
        Returns True when it is, so the caller greys the install
        buttons. Check Again re-reads the setting."""
        from .blendcap.helpers import (smart_app_control_blocking,
                                       smart_app_control_state,
                                       SAC_INFO_NOTE, SAC_ALERT_NOTE,
                                       SAC_EVAL_NOTE)
        if smart_app_control_blocking():
            # Enforcing AND something was actually blocked here.
            warn = box.column(align=True)
            warn.alert = True
            _label_wrapped(warn, context, SAC_ALERT_NOTE, icon='ERROR')
            warn.operator("blendcap.sac_recheck", text="Check Again",
                          icon='FILE_REFRESH')
            return True
        state = smart_app_control_state()
        if state == 'ON':
            # Enforcing, nothing blocked yet: a recommendation in its own
            # red box (Blender's only emphasis colour; a plain box read as
            # noise, user decision 2026-10-30). The text, not the colour,
            # tells "recommended" from "blocked". Buttons stay live.
            note = box.box()
            note.alert = True
            _label_wrapped(note.column(align=True), context, SAC_INFO_NOTE,
                           icon='ERROR')
            note.operator("blendcap.sac_recheck", text="Check Again",
                          icon='FILE_REFRESH')
        elif state == 'EVAL':
            # Evaluation mode blocks nothing yet, but it is how
            # "worked for a week, then stopped" happens: Windows
            # flips it to enforcing later, often after an update.
            _label_wrapped(box.column(align=True), context,
                           SAC_EVAL_NOTE, icon='INFO')
        return False

    def _draw_installer_outdated(self, context, box):
        """Red note when the paid installer on disk is older than this
        addon needs. The paid zip is the delivery route (installing it
        over this copy replaces the stored installer)."""
        from .blendcap.helpers import installer_outdated
        from .blendcap.updater import STORE_URL, managed_repo
        info = installer_outdated()
        if not info:
            return
        cur, req = info
        warn = box.column(align=True)
        warn.alert = True
        repo_name = managed_repo()
        if repo_name:
            # Repository installs get the installer with Blender's own
            # update of the package; no store download involved.
            _label_wrapped(
                warn, context,
                f"The dependency installer on this computer is version "
                f"{cur}; this BlendCap needs {req}. Update BlendCap "
                f"through Get Extensions (the '{repo_name}' repository) "
                f"to replace the installer, then click Reinstall "
                f"Dependencies.", icon='ERROR')
            return
        _label_wrapped(
            warn, context,
            f"The dependency installer on this computer is version {cur}; "
            f"this BlendCap needs {req}. Download the current BlendCap "
            f"package from your store account and install it over this "
            f"one (Install from Disk) to replace the installer.",
            icon='ERROR')
        warn.operator("wm.url_open", text="Open Store Page",
                      icon='URL').url = STORE_URL

    def _draw_locations(self, context, layout):
        box = self._section(layout, "locations_expanded",
                            "File & Cache Locations", 'FILE_FOLDER',
                            self._locations_need_action())
        if box is None:
            return

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

        box.separator()

        # Preset storage. No move button here: choosing a location IS
        # the move (presets are tiny - the update callbacks relocate
        # them the moment the folder changes). The one exception is
        # the at-risk rescue button below.
        box.prop(self, "presets_use_custom")
        if self.presets_use_custom:
            sub = box.column(align=True)
            sub.prop(self, "presets_custom_path")
            self._draw_path_feedback(box, context, True,
                                     self.presets_custom_path,
                                     check_speed=False)
        info = box.column(align=True)
        info.label(text="Current presets folder:", icon='FILE_FOLDER')
        info.label(text="    " + presets_root())
        row = box.row(align=True)
        row.operator("wm.path_open", text="Open Preset Folder",
                     icon='FILE_FOLDER').filepath = presets_root()
        row.operator("blendcap.export_presets", text="Export Presets",
                     icon='EXPORT')
        row.operator("blendcap.import_presets", text="Import Presets",
                     icon='IMPORT')

        # At-risk rescue: presets still inside the addon folder with
        # no pending deps move to sweep them along (the deps move
        # rescues presets automatically). Red, and gone once safe.
        pend = pending_user_data_moves()
        if pend["presets"]:
            warn = box.column(align=True)
            warn.alert = True
            _label_wrapped(
                warn, context,
                "Some of your custom presets are still inside the "
                "addon folder, where an addon update can delete them.",
                icon='ERROR')
            box.operator("blendcap.move_presets",
                         text="Move Presets to Safety", icon='EXPORT')

        box.separator()

        # The addon's own installation folder - a location like the
        # rest, so its Open button lives here (it used to sit in the
        # Dependencies section, where it read as the deps folder).
        info = box.column(align=True)
        info.label(text="Addon installation folder:", icon='FILE_FOLDER')
        info.label(text="    " + _addon_dir())
        box.operator("wm.path_open", text="Open Addon Folder",
                     icon='FILE_FOLDER').filepath = _addon_dir()

    def _draw_path_feedback(self, box, context, enabled, path,
                            check_speed):
        """Red refusal when a custom location points inside the addon
        folder (those values are ignored), and the pre-operation
        slow-drive / removable-drive notes for targets that accept
        heavyweight files."""
        if not enabled or not (path or "").strip():
            return
        if _inside_addon_dir(path):
            warn = box.column(align=True)
            warn.alert = True
            _label_wrapped(
                warn, context,
                "That folder is inside the BlendCap addon, where "
                "updates can delete files. Pick a folder outside the "
                "addon. (This setting is ignored until you do.)",
                icon='ERROR')
            return
        if check_speed:
            hint = storage_speed_warning(path)
            if hint:
                col = box.column(align=True)
                col.alert = hint[0] == 'slow_fs'
                _label_wrapped(col, context, " ".join(hint[1]),
                               icon='ERROR')

    def _draw_capture(self, context, layout):
        box = self._section(layout, "capture_expanded",
                            "Capture & Tracking", 'CAMERA_DATA')
        if box is None:
            return

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
                col.label(text="Hardware not recorded - pick a backend "
                               "above if capture uses the wrong one.",
                          icon='INFO')
        elif self.capture_backend == 'CPU':
            col.label(text="Runs on CPU only - expect minutes per frame.",
                      icon='INFO')

        box.prop(self, "refine_hands")
        box.prop(self, "show_footage_in_cameras")

        box.separator()

        # Which source a fresh scene starts on (File Path vs Video Editor
        # strip). Applied to new/startup files by registration's load_post
        # handler + startup timer; saved projects keep their stored choice.
        box.prop(self, "default_video_source")

        box.prop(self, "disable_timeline_autosnap")

        box.separator()

        # Video Editor behaviors (sequencer_guard.py).
        seq_col = box.column(align=True)
        seq_col.label(text="When video is added to the sequencer:")
        seq_col.prop(self, "keep_view_transform_on_video_add")
        seq_col.prop(self, "keep_frame_rate_on_video_add")
        seq_col.prop(self, "disable_sequencer_render_on_video_add")
        seq_col.prop(self, "sequencer_guard_workspace_only")
        note = box.column(align=True)
        _label_wrapped(note, context,
                       "Blender normally switches the view transform to "
                       "Standard, sets the project frame rate to the "
                       "clip's, and leaves Sequencer rendering on, which "
                       "makes renders show the video instead of your "
                       "scene. With these on, BlendCap turns off "
                       "Blender's \"Set View Transform\" and \"Use Movie "
                       "Framerate\" options for the Add Movie and Add "
                       "Image operators, keeps the project's settings "
                       "either way, sets the clip to display as-is under "
                       "AgX or Filmic, and turns Sequencer rendering "
                       "off. A clip with a different frame rate still "
                       "plays at normal speed.",
                       icon='INFO')

    def _draw_interface(self, context, layout):
        box = self._section(layout, "interface_expanded", "Interface",
                            'HIDE_OFF')
        if box is None:
            return

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
