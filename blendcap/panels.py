"""Shared draw function and the two panels that mount it.

The draw function references operator IDs by string
(`blendcap.generate_mocap` etc.), so the only imports needed here are
the two big state dicts (for progress display) and the strip-position
helpers (for the VSE source label).
"""
import bpy

from .helpers import (
    blendcap_state, bvh_state, retarget_progress_state, arkit_state,
    _strip_visible_start, _strip_visible_end, addon_prefs,
    is_paid_build, _active_video_strip,
)

# Channel detection lives in helpers.is_paid_build (shared with the
# prefs panel's installer checks so the two can never drift). Evaluated
# once at addon registration; editable by hand if a build ever needs to
# force it either way.
SHOW_DONATE_BUTTON = not is_paid_build()


def _source_rig_has_recognized_face_bones(src_rig):
    """True iff the source armature has at least one pose bone whose name
    classifies into a face region via _face_region_from_source.

    Used to grey out Face Expression in Retarget Advanced when the source
    rig isn't a BlendCap-generated face BVH (or any rig that happens to
    use the Mixamo-style face naming the classifier expects). The
    amplification path silently no-ops on unrecognized names, so without
    this gate the sliders appear functional but do nothing.

    Called from draw(); keep it cheap. Worst case is one substring scan
    per pose bone, short-circuited on first match."""
    if src_rig is None or getattr(src_rig, "type", None) != 'ARMATURE':
        return False
    from .retarget import _face_region_from_source
    pose = getattr(src_rig, "pose", None)
    if pose is None:
        return False
    for pb in pose.bones:
        if _face_region_from_source(pb.name) is not None:
            return True
    return False


def _draw_progress_bar(layout, pct, text, scale_y=1.4):
    """Render a real Blender progress bar with text label inside.

    Uses `layout.progress()` (Blender 4.0+). Falls back to a SORTTIME
    label on older Blender so the panel never breaks. The bar widget is
    far more visible than a plain text label — important for long bakes
    so users can see at a glance that work is happening.
    """
    factor = max(0.0, min(1.0, pct / 100.0))
    row = layout.row()
    row.scale_y = scale_y
    try:
        row.progress(factor=factor, type='BAR', text=text)
    except (AttributeError, TypeError):
        row.label(text=text, icon='SORTTIME')


# ============================================================
# IK CHAIN LABEL HELPERS
# ============================================================
# The FK->IK Mapping panel rows show user-friendly labels derived from
# each chain's auto-classified limb_kind ('ARM' / 'LEG' / 'OTHER'). The
# limb_kind itself is set during chain discovery (operators_ik.
# _classify_limb_kind), the side suffix ('.L' / '.R') is read off the
# constraint owner name. Generic (OTHER) chains fall back to neutral
# wording so the panel still reads correctly on non-Rigify rigs.
def _side_suffix(name):
    for s in ('.L', '.R', '_L', '_R', '.l', '.r'):
        if name.endswith(s):
            return s
    return ''

def _chain_label(chain):
    # The locked 4-row layout sets `side` explicitly on each row, so
    # we use it directly. Fall back to the owner suffix for legacy
    # rows that haven't been re-synced yet.
    side = chain.side or _side_suffix(chain.owner).lstrip('._').upper()
    suffix = f".{side}" if side else ''
    if chain.limb_kind == 'ARM':
        return f"Arm{suffix}"
    if chain.limb_kind == 'LEG':
        return f"Leg{suffix}"
    return chain.owner or "(unknown chain)"

def _ik_field_label(chain):
    if chain.limb_kind == 'ARM':
        return "IK Hand Control:"
    if chain.limb_kind == 'LEG':
        return "IK Foot Control:"
    return "IK End Control:"

def _pole_field_label(chain):
    if chain.limb_kind == 'ARM':
        return "Elbow Pole:"
    if chain.limb_kind == 'LEG':
        return "Knee Pole:"
    return "Pole Target:"


# ============================================================
# SHARED UI DRAW FUNCTION
# ============================================================
def _draw_setup_section(layout, hide_setup):
    # --- SETUP --- (suppressed by the "Hide the Setup section" add-on pref)
    if not hide_setup:
        box = layout.box()
        box.label(text="Setup:", icon='PREFERENCES')
        row = box.row(align=True)
        # The dependency installer wizard lives in the Add-on Preferences (drawn
        # inline there); this button jumps the user to that section.
        row.operator("blendcap.install_open_prefs", icon='PREFERENCES',
                     text="Open Preferences")
        row.operator("blendcap.append_workspace", text="Open Workspace", icon='WORKSPACE')

        layout.separator()


def _draw_donate_section(layout, prefs):
    # Shown ONLY to source-distribution / self-install users (paid builds hide
    # it via is_paid_build -> SHOW_DONATE_BUTTON). Supporters can dismiss it
    # with the honor-system prefs toggle (hide_donate_buttons), which stays
    # reachable in the prefs panel after the button is hidden. Boxed + labeled
    # like every other section so it reads as real BlendCap UI (a bare button
    # at the very top read as disconnected); drawn just under Setup to stay
    # prominent without being the disconnected first thing.
    if not SHOW_DONATE_BUTTON:
        return
    if prefs and getattr(prefs, "hide_donate_buttons", False):
        return
    box = layout.box()
    box.label(text="Support BlendCap:", icon='FUND')
    box.operator("blendcap.donate", text="Donate / Support", icon='FUND')
    layout.separator()


def _draw_source_section(layout, context, props):
    # --- SOURCE ---
    box = layout.box()
    src_header = box.row(align=True)
    src_header.prop(props, "video_source_expanded",
                    icon=('TRIA_DOWN' if props.video_source_expanded
                          else 'TRIA_RIGHT'),
                    emboss=False, text="")
    src_header.label(text="Video Source", icon='FILE_MOVIE')
    if props.video_source_expanded:
        # Dropdown (not expand=True toggle row): a toggle row selects on
        # press-drag, so a click that pops the sequencer warning could leave the
        # row "live" and silently revert to File Path when the mouse moved back
        # over it. A dropdown commits cleanly and can't change on hover/drag.
        box.prop(props, "video_source", text="")
        if props.video_source == 'FILE':
            box.prop(props, "video_path", text="")
        else:
            box.label(text="Uses active Video Sequencer strip.", icon='SEQ_SEQUENCER')
            seq_ed = context.scene.sequence_editor
            active = _active_video_strip(seq_ed)
            if active and active.type in ('MOVIE', 'IMAGE'):
                vis_start = _strip_visible_start(active)
                vis_end = _strip_visible_end(active)
                box.label(text=f"{active.name}  (ch.{active.channel}, {vis_start}-{vis_end})",
                           icon='SEQUENCE')
            else:
                box.label(text="No video strip selected.", icon='ERROR')

    layout.separator()


def _draw_capture_section(layout, context, props, is_running, hide_clear_all):
    # --- CAPTURE ---
    box = layout.box()
    cap_header = box.row(align=True)
    cap_header.prop(props, "capture_expanded",
                    icon='TRIA_DOWN' if props.capture_expanded else 'TRIA_RIGHT',
                    emboss=False, text="")
    cap_header.label(text="Capture", icon='VIEW_CAMERA')
    if props.capture_expanded:
        row = box.row()
        row.prop(props, "track_body", text="Body")
        row.prop(props, "track_face", text="Face")

        # FOV / camera field-of-view (body tracking only). Plain dropdown.
        fov_col = box.column(align=True)
        fov_col.enabled = props.track_body
        fov_col.prop(props, "fov_mode")
        if props.fov_mode == 'MANUAL':
            fov_col.prop(props, "fov_focal_mm")

        if props.track_body:
            box.prop(props, "preview_step", slider=True)
            box.prop(props, "capture_skip", slider=True)

        if is_running:
            box.operator("blendcap.cancel", icon='CANCEL', text="Cancel")
            pct = blendcap_state['progress']
            status = blendcap_state.get('status_text', '')
            label = (f"{int(pct)}%  {status}" if status
                     else f"Running... {int(pct)}%")
            _draw_progress_bar(box, pct, label, scale_y=1.4)
            for warn in blendcap_state.get('warnings', []):
                box.label(text=warn, icon='ERROR')
        else:
            row = box.row(align=True)
            row.operator("blendcap.generate_mocap", icon='FILE_REFRESH', text="Run Preview").is_update_only = True
            row.operator("blendcap.generate_mocap", icon='PLAY', text="Run Capture").is_update_only = False

        # Revert button: show when active VSE strip is a BlendCap proxy
        if props.video_source == 'VSE':
            seq_ed = context.scene.sequence_editor
            active = getattr(seq_ed, "active_strip", None) if seq_ed else None
            if active and active.type in ('MOVIE', 'IMAGE') and "blendcap_source" in active:
                box.separator()
                row = box.row()
                row.enabled = not is_running
                row.operator("blendcap.revert_strip", icon='LOOP_BACK',
                             text="Revert to Original Video")
                box.label(text="Proxy clip — edits won't carry over, Revert first", icon='INFO')

        # Extra gap so the destructive cache buttons sit clear of Run Capture.
        box.separator(factor=2.0)
        row = box.row(align=True)
        row.enabled = not is_running
        row.operator("blendcap.clear_current_cache", icon='TRASH', text="Clear Cache")
        # "Clear All" wipes EVERY clip's cache; the add-on pref hides it to
        # prevent an accidental project-wide delete (Clear Cache stays).
        if not hide_clear_all:
            row.operator("blendcap.clear_all_caches", icon='ERROR', text="Clear All")

    layout.separator()


def _draw_bvh_settings_section(layout, context, props, is_running, is_bvh_running):
    # --- BVH SETTINGS ---
    box = layout.box()
    bvh_header = box.row(align=True)
    bvh_header.prop(props, "bvh_settings_expanded",
                    icon=('TRIA_DOWN' if props.bvh_settings_expanded
                          else 'TRIA_RIGHT'),
                    emboss=False, text="")
    bvh_header.label(text="BVH Settings", icon='EXPORT')
    if props.bvh_settings_expanded:
        col = box.column(align=True)
        row = col.row()
        row.prop(props, "export_body")
        row.prop(props, "export_hands")
        row = col.row()
        row.prop(props, "export_face")
        row.prop(props, "export_root")
        col.separator()
        col.prop(props, "use_separate_face")
        if props.use_separate_face:
            col.prop(props, "separate_face_data", text="")

        # Rebuild Finger Motion needs the finger bones exported, so it's
        # gated on Hands - it does nothing (and wastes time) otherwise.
        hs_row = col.row()
        hs_row.enabled = bool(props.export_hands)
        hs_row.prop(props, "bvh_hand_solver")

        box.separator()
        if is_bvh_running:
            box.operator("blendcap.cancel", icon='CANCEL',
                         text="Cancel BVH Export")
            pct = bvh_state.get('progress', 0.0)
            status = bvh_state.get('status_text', '')
            label = (f"{int(pct)}%  {status}" if status
                     else f"Generating BVH... {int(pct)}%")
            _draw_progress_bar(box, pct, label, scale_y=1.4)
        else:
            bvh_row = box.row()
            bvh_row.enabled = not is_running
            bvh_row.operator("blendcap.generate_bvh", icon='IMPORT',
                             text="Generate/Reload BVH")

        _draw_bvh_post_processing(box, context, props)

        # --- BAKE SETTINGS TO BVH ---
        # While a BVH job is running, swap the button for the same
        # Cancel + progress bar pattern used by the Generate button.
        # Blender greys the bake button via the operator's poll() when
        # no BlendCap rig is the active object (so the button is
        # visible-but-disabled in that idle case).
        box.separator()
        if is_bvh_running:
            box.operator("blendcap.cancel", icon='CANCEL',
                         text="Cancel BVH Export")
            pct = bvh_state.get('progress', 0.0)
            status = bvh_state.get('status_text', '')
            label = (f"{int(pct)}%  {status}" if status
                     else f"Baking BVH... {int(pct)}%")
            _draw_progress_bar(box, pct, label, scale_y=1.4)
        else:
            bake_row = box.row(align=True)
            bake_row.scale_y = 1.2
            bake_row.operator("blendcap.bake_camera_offset",
                              icon='IMPORT')

    layout.separator()


def _draw_bvh_post_processing(box, context, props):
    # --- BVH POST PROCESSING ---
    # Non-collapsing category heading for tools that act on the
    # baked BVH via re-running npz_to_bvh.py with updated args. Order
    # is roughly broad-to-specific within body, then face, with the
    # per-rig Camera Angle Offset anchoring the top and the Bake
    # button always-visible at the bottom. All knobs in this
    # section are baked-in (no live preview except camera offset),
    # but the Bake button re-runs the pipeline with the current
    # values so iteration only costs one bake per tweak.
    box.separator()
    box.label(text="BVH Post Processing", icon='MODIFIER')

    active = context.active_object
    is_blendcap_rig = (active is not None
                       and active.type == 'ARMATURE'
                       and "blendcap_source_video" in active)

    # Wrap collapsibles in an aligned column so successive sections
    # pack with the same tight spacing the original BVH Settings
    # column used. `box.separator()` between top-level items gives
    # noticeably wider gaps.
    pp_col = box.column(align=True)

    _draw_bvh_camera_angle_offset(pp_col, context, props, active, is_blendcap_rig)
    _draw_bvh_smoothing(pp_col, props)
    _draw_bvh_depth_noise_filtering(pp_col, props)
    _draw_bvh_foot_locking(pp_col, props)
    _draw_bvh_face_noise_filtering(pp_col, props)
    _draw_bvh_motion_scaling(pp_col, props)


def _draw_bvh_camera_angle_offset(pp_col, context, props, active, is_blendcap_rig):
    # --- CAMERA ANGLE OFFSET ---
    # Per-armature pitch/roll sliders bound to the ACTIVE BlendCap
    # armature's blendcap_camera_offset PropertyGroup. Live-updates
    # Root's rotation keyframes (without touching rest pose). The
    # bake operator below re-runs npz_to_bvh.py for the active
    # rig's source clip with --camera-pitch / --camera-roll set to
    # the slider values, replacing the BVH file on disk. Only body
    # rigs are valid targets: face-only rigs have no Root bone and
    # there's no useful "camera angle" for a face cluster anyway.
    # When the active object isn't a valid body rig, sliders bind
    # to a scene-level placeholder so they stay visible (greyed).
    is_body_rig = (is_blendcap_rig
                   and active.pose
                   and "Root" in active.pose.bones)

    # Library re-imports don't carry forward the cumulative baked
    # totals (no metadata in the BVH file to read them from), so the
    # FIRST bake on a re-imported rig will write `slider` instead of
    # `slider + previous_baked`. Surfaced at the very top of BVH
    # Post Processing (above Camera Angle Offset) so it's visible
    # regardless of which subsection is expanded. Flag is set by
    # BLENDCAP_OT_import_lib_bvh and clears naturally when the bake
    # deletes the old rig and re-imports the freshly-baked one.
    if is_body_rig and active.get("_blendcap_library_reimport"):
        reimport_box = pp_col.box()
        reimport_box.label(text="Re-imported BVH:", icon='INFO')
        reimport_box.label(text="First bake may reset previous offsets.")

    co_header = pp_col.row(align=True)
    co_header.prop(props, "camera_offset_expanded",
                   icon=('TRIA_DOWN' if props.camera_offset_expanded
                         else 'TRIA_RIGHT'),
                   emboss=False, text="")
    co_header.label(text="Camera Angle Offset")
    if props.camera_offset_expanded:
        co_col = pp_col.column(align=True)
        co_col.enabled = is_body_rig
        if is_body_rig:
            offset = active.blendcap_camera_offset
            co_col.label(text=f"Active: {active.name}",
                         icon='OUTLINER_OB_ARMATURE')
        else:
            offset = context.scene.blendcap_camera_offset_placeholder
            co_col.label(text="Select a BlendCap body armature "
                              "to edit its offset.", icon='INFO')
        co_col.prop(offset, "pitch_deg")
        co_col.prop(offset, "roll_deg")
        co_col.prop(offset, "height_offset_m")

        # --- HEAD ANGLE OFFSET (nested label + sliders) ---
        # Same corrective-tilt concept, scoped to the Head bone.
        # Same gating as the body offset (body rigs only) —
        # face-only rigs don't need it, and the bake side only
        # writes head pitch/roll through the body bake command.
        # Inherits co_col's enabled state so the whole section
        # greys together when no body rig is active.
        co_col.separator()
        co_col.label(text="Head Angle Offset")
        co_col.prop(offset, "head_pitch_deg")
        co_col.prop(offset, "head_roll_deg")


def _draw_bvh_smoothing(pp_col, props):
    # --- SMOOTHING ---
    # Face/Body/Root rotation smoothing in seconds. Baked into the
    # BVH via --smooth / --smooth-face / --smooth-root.
    pp_col.separator()
    sm_header = pp_col.row(align=True)
    sm_header.prop(props, "bvh_smoothing_expanded",
                   icon=('TRIA_DOWN' if props.bvh_smoothing_expanded
                         else 'TRIA_RIGHT'),
                   emboss=False, text="")
    sm_header.label(text="Smoothing")
    if props.bvh_smoothing_expanded:
        sm_col = pp_col.column(align=True)
        sm_col.prop(props, "smoothing_enabled")
        sm_sub = sm_col.column(align=True)
        sm_sub.enabled = props.smoothing_enabled
        sm_sub.prop(props, "smooth_face", text="Face (s)")
        sm_sub.prop(props, "smooth_body", text="Body (s)")
        sm_sub.prop(props, "smooth_root", text="Root (s)")
        sm_sub.prop(props, "smooth_vertical", text="Vertical (s)")


def _draw_bvh_depth_noise_filtering(pp_col, props):
    # --- DEPTH NOISE FILTERING ---
    # Smoothstep dead-zone on the residual against a locally-smoothed
    # baseline of each tip's depth-axis world position (feet, hands,
    # head, root). Bakes into the exported BVH via 2-bone IK back-
    # solve (feet/hands), 1-bone swing delta (head), and direct
    # translation edit (root).
    pp_col.separator()
    dd_header = pp_col.row(align=True)
    dd_header.prop(props, "bvh_depth_denoise_expanded",
                   icon=('TRIA_DOWN' if props.bvh_depth_denoise_expanded
                         else 'TRIA_RIGHT'),
                   emboss=False, text="")
    dd_header.label(text="Depth Noise Filtering")
    if props.bvh_depth_denoise_expanded:
        dd_col = pp_col.column(align=True)
        dd_col.prop(props, "bvh_depth_denoise_enabled",
                    text="Use Depth Noise Filtering")
        sub = dd_col.column(align=True)
        sub.enabled = bool(props.bvh_depth_denoise_enabled)
        for _region in ("feet", "hands", "head", "root"):
            sub.prop(props, f"bvh_depth_denoise_{_region}")


def _draw_bvh_foot_locking(pp_col, props):
    # --- FOOT LOCKING ---
    # Master toggle lives inside the collapsible (matches the
    # Face Noise Filtering pattern). Settings greyed out when the
    # toggle is off.
    pp_col.separator()
    fl_header = pp_col.row(align=True)
    fl_header.prop(props, "foot_lock_settings_expanded",
                   icon=('TRIA_DOWN' if props.foot_lock_settings_expanded
                         else 'TRIA_RIGHT'),
                   emboss=False, text="")
    fl_header.label(text="Foot Locking Settings")
    if props.foot_lock_settings_expanded:
        fl_col = pp_col.column(align=True)
        fl_col.prop(props, "foot_lock_enabled")
        sub = fl_col.column(align=True)
        sub.enabled = props.foot_lock_enabled
        # Feet Define the Floor sits directly under the master
        # toggle: it replaces the whole vertical channel and ships
        # default-on — the most consequential row in the section,
        # ahead of the tuning knobs below.
        sub.prop(props, "foot_define_floor")
        sub.prop(props, "foot_sensitivity")
        sub.prop(props, "foot_max_gap_sec")
        sub.prop(props, "foot_max_lift_cm")
        sub.prop(props, "foot_recover_glitches")
        # Recovery Lift only meaningful when Recover Glitches is on.
        # Put it on sub so it inherits the master-toggle greying;
        # the per-row enabled stacks with the parent's state.
        rl_row = sub.row()
        rl_row.enabled = props.foot_recover_glitches
        rl_row.prop(props, "foot_recovery_lift_cm")
        sub.prop(props, "foot_pin_enabled")
        # Strength / Release only meaningful when the pin is on —
        # same stacked-greying pattern as Recovery Lift above.
        pin_col = sub.column(align=True)
        pin_col.enabled = props.foot_pin_enabled
        pin_col.prop(props, "foot_pin_strength")
        pin_col.prop(props, "foot_pin_release_cm")

    # --- BODY GROUNDING ---
    # Grounds non-upright near-floor stretches (rolls, lying,
    # handstands); posture-gated so upright motion and jumps are
    # untouched. Baked at BVH generation time like foot locking.
    pp_col.separator()
    bg_header = pp_col.row(align=True)
    bg_header.prop(props, "body_grounding_expanded",
                   icon=('TRIA_DOWN' if props.body_grounding_expanded
                         else 'TRIA_RIGHT'),
                   emboss=False, text="")
    bg_header.label(text="Body Grounding")
    if props.body_grounding_expanded:
        bg_col = pp_col.column(align=True)
        bg_col.prop(props, "body_grounding_enabled")
        bg_sub = bg_col.column(align=True)
        bg_sub.enabled = props.body_grounding_enabled
        bg_sub.prop(props, "body_catch_distance_cm")
        bg_sub.prop(props, "body_skin_radius_cm")


def _draw_bvh_face_noise_filtering(pp_col, props):
    # --- FACE NOISE FILTERING ---
    # Per-region dead-zone, baked into BVH face data so the filter
    # travels with the export.
    pp_col.separator()
    nf_header = pp_col.row(align=True)
    nf_header.prop(props, "bvh_face_denoise_expanded",
                   icon=('TRIA_DOWN' if props.bvh_face_denoise_expanded
                         else 'TRIA_RIGHT'),
                   emboss=False, text="")
    nf_header.label(text="Face Noise Filtering")
    if props.bvh_face_denoise_expanded:
        nf_col = pp_col.column(align=True)
        nf_col.prop(props, "bvh_face_denoise_enabled",
                    text="Use Face Noise Filtering")
        sub = nf_col.column(align=True)
        sub.enabled = bool(props.bvh_face_denoise_enabled)
        for _region in ("jaw", "gaze", "lids", "brows", "lips",
                        "cheeks", "nose", "tongue"):
            sub.prop(props, f"bvh_face_denoise_{_region}")


def _draw_bvh_motion_scaling(pp_col, props):
    # --- MOTION SCALING ---
    # Global multipliers (finger curl, face expression) plus
    # per-region "Face Expression" sub-collapsible (mirrors the
    # retarget Face Motion Scaling). Settings are baked into the
    # BVH via npz_to_bvh.py args by both Generate BVH and Bake
    # (below).
    pp_col.separator()
    ms_header = pp_col.row(align=True)
    ms_header.prop(props, "bvh_motion_scaling_expanded",
                   icon=('TRIA_DOWN' if props.bvh_motion_scaling_expanded
                         else 'TRIA_RIGHT'),
                   emboss=False, text="")
    ms_header.label(text="Motion Scaling")
    if props.bvh_motion_scaling_expanded:
        ms_col = pp_col.column(align=True)
        # Finger Curl Strength greys out while Rebuild Finger Motion
        # (in the BVH Settings toggles) is on - they're alternatives,
        # the solver replaces the boost.
        fc_row = ms_col.row()
        fc_row.enabled = not bool(props.bvh_hand_solver)
        fc_row.prop(props, "finger_curl_multiplier")
        # Face Expression -- always-visible label + framed box
        # holding global Strength plus per-region overrides.
        # Matches the framed layout of the Retarget version. The
        # face_expression_multiplier (Strength) and the per-region
        # values are mutually exclusive to prevent compounding --
        # when per-region is on, the BVH subprocess forces
        # face_boost=1.0 (see _bvh_face_boost_value).
        ms_col.separator()
        ms_col.label(text="Face Expression")
        fa_box = ms_col.box()
        fa_col = fa_box.column(align=True)
        # Global Strength -- greyed when per-region is on.
        str_row = fa_col.row()
        str_row.enabled = not bool(props.bvh_face_amp_enabled)
        str_row.prop(props, "face_expression_multiplier",
                     text="Strength")
        fa_col.separator()
        fa_col.prop(props, "bvh_face_amp_enabled",
                    text="Use Per-Region")
        fa_sub = fa_col.column(align=True)
        fa_sub.enabled = bool(props.bvh_face_amp_enabled)
        for _region in ("jaw", "gaze", "lids", "brows", "lips",
                        "cheeks", "nose", "tongue"):
            fa_sub.prop(props, f"bvh_face_amp_{_region}")


def _draw_retargeting_section(layout, context, props):
    # --- RETARGETING ---
    scene = context.scene
    box = layout.box()
    header = box.row(align=True)
    header.prop(scene, "blendcap_retarget_expanded",
                icon='TRIA_DOWN' if scene.blendcap_retarget_expanded else 'TRIA_RIGHT',
                emboss=False, text="")
    header.label(text="Retargeting", icon='ARMATURE_DATA')

    if scene.blendcap_retarget_expanded:
        # Rig pickers in their own aligned block (use armature icons).
        rig_col = box.column(align=True)
        rig_col.prop(scene, "blendcap_retarget_source", text="Source Rig",
                     icon='OUTLINER_OB_ARMATURE')
        rig_col.prop(scene, "blendcap_retarget_target", text="Target Rig",
                     icon='OUTLINER_OB_ARMATURE')

        box.separator()

        # Preset / namespace in a SEPARATE column block — own alignment
        # context so the icon space reserved by the rig pickers above
        # doesn't leak into rows that don't have icons.
        col = box.column(align=True)
        # Single combined preset dropdown — selecting a preset auto-loads it.
        # Standard / Custom sub-headers are inside the dropdown itself, with
        # a sentinel "(Unsaved changes…)" row at the top that becomes the
        # selection when the user edits pairs after loading. The trash-can
        # button on the right deletes user-saved presets; it auto-greys out
        # for shipped presets via the operator's poll().
        preset_row = col.row(align=True)
        preset_row.prop(scene, "blendcap_retarget_preset", text="Preset")
        preset_row.operator("blendcap.retarget_delete_preset",
                            text="", icon='TRASH')
        # Source/Target prefix fields. Pair source/target cells store the
        # SHORT bone name; these prefixes prepend at bake time to resolve
        # the actual rig bone. Editing a prefix re-targets every pair to
        # a re-namespaced rig in one keystroke — single Mixamo preset
        # works against rigs named "mixamorig:" or "mixamorig1:" etc.
        # without touching individual rows.
        col.prop(scene, "blendcap_retarget_source_prefix", text="Source Prefix")
        col.prop(scene, "blendcap_retarget_namespace_strip", text="Target Prefix")

        # Reload + Save go here so the user finds them while their eye is
        # already on the preset/prefix row — easier discovery than burying
        # them below the pair table.
        action_row = col.row(align=True)
        action_row.operator("blendcap.retarget_reload_preset", text="Reload Preset", icon='FILE_REFRESH')
        action_row.operator("blendcap.retarget_save_preset", text="Save Map As...", icon='FILE_TICK')

        _draw_retarget_mapper_table(col, scene)

        col.separator()
        col.prop(scene, "blendcap_retarget_auto_scale")
        col.prop(scene, "blendcap_retarget_use_current_pose_as_rest")
        col.prop(scene, "blendcap_retarget_auto_bake_ik")

        _draw_retarget_fk_ik_mapping(col, scene)

        _draw_retarget_custom_rest_pose(col, props)

        _draw_retarget_advanced(col, scene, props)

        # Apply Retargeting + Convert FK->IK — primary actions. While
        # either bake is running, both action buttons collapse into a
        # prominent progress bar (matches the BVH section's layout). The
        # modal retarget bake also exposes a Cancel button below the
        # bar; FK->IK is synchronous and cannot be cancelled mid-run.
        # FK->IK pumps wm.redraw_timer + tag_redraw between batches so
        # the bar actually refreshes mid-execute despite being sync.
        from . import operators_retarget, operators_ik
        if retarget_progress_state.get('running'):
            # Cancel button: route to whichever modal is active. Both
            # bakes share the same progress dict but each has its own
            # cancel operator (cancel_retarget / cancel_fk_to_ik) so the
            # right poll() lights up.
            if operators_retarget._BAKE_RUNNING:
                cancel_row = col.row()
                cancel_row.scale_y = 1.4
                cancel_row.operator("blendcap.cancel_retarget",
                                    icon='CANCEL', text="Cancel")
            elif operators_ik._FK_IK_RUNNING:
                cancel_row = col.row()
                cancel_row.scale_y = 1.4
                cancel_row.operator("blendcap.cancel_fk_to_ik",
                                    icon='CANCEL', text="Cancel")
            phase = retarget_progress_state.get('phase', 'Working')
            pct = retarget_progress_state.get('progress', 0.0)
            status = retarget_progress_state.get('status_text', '')
            label = (f"{phase}: {int(pct)}%  {status}"
                     if status else f"{phase}: {int(pct)}%")
            _draw_progress_bar(col, pct, label, scale_y=1.4)
        else:
            apply_row = col.row()
            apply_row.scale_y = 1.4
            apply_row.operator("blendcap.apply_retarget",
                               icon='BONE_DATA',
                               text="Apply Retargeting")
            convert_row = col.row()
            convert_row.scale_y = 1.2
            convert_row.operator("blendcap.convert_fk_to_ik",
                                 icon='CON_KINEMATIC',
                                 text="Convert FK → IK")

    return box


def _draw_retarget_mapper_table(col, scene):
    # Mapper table
    col.separator()
    pairs = scene.blendcap_retarget_pairs
    valid_count = 0
    src_rig_obj = scene.blendcap_retarget_source
    tgt_rig_obj = scene.blendcap_retarget_target
    # Validity now uses the prefix-aware resolver: a pair stored as
    # "RightLeg" against a Mixamo rig (where the actual bone is
    # "mixamorig:RightLeg") counts as valid via the Source/Target
    # Prefix prepend at lookup time.
    src_prefix = scene.blendcap_retarget_source_prefix or ""
    tgt_prefix = scene.blendcap_retarget_namespace_strip or ""

    def _resolves(rig, name, prefix):
        if not (name and rig and rig.type == 'ARMATURE'):
            return False
        bones = rig.data.bones
        return (name in bones) or (bool(prefix) and (prefix + name) in bones)
    for p in pairs:
        sv = _resolves(src_rig_obj, p.source, src_prefix)
        tv = _resolves(tgt_rig_obj, p.target, tgt_prefix)
        if sv and tv:
            valid_count += 1

    status_row = col.row()
    if len(pairs) == 0:
        status_row.label(text="Map is empty — load a preset or auto-match.", icon='INFO')
    else:
        status_row.label(text=f"{valid_count} / {len(pairs)} pairs valid",
                         icon='CHECKMARK' if valid_count == len(pairs) else 'ERROR')

    # Header labels above the UIList. Uses the SAME row.column()
    # structure with the SAME scale_x ratios as BLENDCAP_UL_retarget_
    # pairs.draw_item so labels and row cells line up vertically as
    # true columns. If you edit the scale_x values here, edit them
    # in operators_retarget.py too.
    # Scale ratios: Source 12 | Target 12 | Channels 9 | Anchor 1.1 | X 1
    header_row = col.row(align=True)
    c = header_row.column(align=True); c.scale_x = 12.0; c.label(text="Source")
    c = header_row.column(align=True); c.scale_x = 12.0; c.label(text="    Target")
    c = header_row.column(align=True); c.scale_x = 9.0;  c.label(text="    Channels")
    c = header_row.column(align=True); c.scale_x = 1.1;  c.label(text="Anchor")
    c = header_row.column(align=True); c.scale_x = 1.0;  c.label(text="")

    col.template_list(
        "BLENDCAP_UL_retarget_pairs", "",
        scene, "blendcap_retarget_pairs",
        scene, "blendcap_retarget_pair_index",
        rows=6,
    )

    edit_row = col.row(align=True)
    edit_row.operator("blendcap.retarget_add_pair", text="Add Pair", icon='ADD')
    edit_row.operator("blendcap.retarget_add_all_source_bones", text="Add All", icon='DUPLICATE')
    edit_row.operator("blendcap.retarget_auto_match", text="Auto-Match", icon='AUTO')
    edit_row.operator("blendcap.retarget_sort_pairs", text="Sort", icon='SORTALPHA')
    edit_row.operator("blendcap.retarget_clear_pairs", text="Clear", icon='TRASH')


def _draw_retarget_fk_ik_mapping(col, scene):
    # --- FK->IK Mapping sub-box (collapsed by default) ---
    # Per-chain IK control + pole control assignments. Discovery
    # populates one row per IK constraint on the target rig (keyed
    # by constraint owner). The user only ever fills in the
    # visible IK end-effector and the pole control bone. Field
    # labels adapt to the auto-classified limb kind (Arm/Leg/
    # Other) so the user sees "IK Hand Control" / "Elbow Pole"
    # instead of generic terms.
    ik_header = col.row(align=True)
    ik_header.prop(scene, "blendcap_retarget_ik_expanded",
                   icon=('TRIA_DOWN' if scene.blendcap_retarget_ik_expanded
                         else 'TRIA_RIGHT'),
                   emboss=False, text="")
    ik_header.label(text="FK → IK Mapping")
    target_rig = scene.blendcap_retarget_target
    n_chains = len(scene.blendcap_retarget_ik_chains)
    if n_chains:
        ik_header.label(text=f"({n_chains})")
    if scene.blendcap_retarget_ik_expanded:
        ik_col = col.column(align=False)
        if target_rig is None:
            ik_col.label(text="Pick a target rig to discover IK chains.",
                         icon='INFO')
        elif n_chains == 0:
            ik_col.label(text="No IK constraints found on the target rig.",
                         icon='INFO')
        else:
            for chain in scene.blendcap_retarget_ik_chains:
                chain_box = ik_col.box()
                chain_box.label(text=_chain_label(chain),
                                icon='CONSTRAINT_BONE')

                ik_row = chain_box.row(align=True)
                ik_row.label(text=_ik_field_label(chain))
                ik_row.prop_search(chain, "ik_control",
                                   target_rig.pose, "bones", text="")

                pole_row = chain_box.row(align=True)
                pole_row.label(text=_pole_field_label(chain))
                pole_row.prop_search(chain, "pole_control",
                                     target_rig.pose, "bones", text="")


def _draw_retarget_custom_rest_pose(col, props):
    # --- Custom Rest Pose sub-section (collapsible) ---
    # Saved-pose alternative to "Use Current Source Pose as Rest". Populates
    # the same `source_rest_override` dict at retarget time but reads the
    # pose from a JSON file instead of the source rig's live pose. Mutually
    # exclusive with the live toggle above; toggling either one clears the
    # other (handled by update callbacks on the BoolProperties).
    crp_header = col.row(align=True)
    crp_header.prop(props, "custom_rest_pose_expanded",
                    icon=('TRIA_DOWN' if props.custom_rest_pose_expanded
                          else 'TRIA_RIGHT'),
                    emboss=False, text="")
    crp_header.label(text="Custom Rest Pose")
    if props.custom_rest_pose_expanded:
        crp_col = col.column(align=True)
        crp_col.prop(props, "use_custom_rest_pose")
        crp_sub = crp_col.column(align=True)
        crp_sub.enabled = props.use_custom_rest_pose
        crp_preset_row = crp_sub.row(align=True)
        crp_preset_row.prop(props, "custom_rest_pose_preset", text="Preset")
        crp_preset_row.operator("blendcap.delete_rest_pose_preset",
                                text="", icon='TRASH')
        # Preview lives inside the enabled-by-checkbox sub-column so it
        # greys out when no preset is selected, matching the dropdown.
        crp_sub.operator("blendcap.preview_rest_pose_preset",
                         text="Preview Selected Pose",
                         icon='HIDE_OFF')
        crp_col.separator()
        crp_col.operator("blendcap.save_rest_pose",
                         text="Save Current Pose as Preset",
                         icon='FILE_TICK')


def _draw_retarget_advanced(col, scene, props):
    # --- Advanced sub-box (collapsed by default) ---
    # Options here are escape hatches for odd rigs — 99% of users never
    # need to touch these. Kept visible (not hidden entirely) so power
    # users can debug / override without hunting through code. Sits
    # ABOVE Apply Retargeting so its toggles are read+set before the
    # user clicks Apply.
    adv_header = col.row(align=True)
    adv_header.prop(scene, "blendcap_retarget_advanced_expanded",
                    icon=('TRIA_DOWN' if scene.blendcap_retarget_advanced_expanded
                          else 'TRIA_RIGHT'),
                    emboss=False, text="")
    adv_header.label(text="Advanced")
    if scene.blendcap_retarget_advanced_expanded:
        adv_col = col.column(align=True)

        # Head bone references used by Anchor (HEAD_LOCAL) pairs. Auto-
        # filled when the user picks a source/target rig (see the
        # rig-picker update callbacks in registration.py) so most users
        # never see or touch these. Exposed as an override for rigs
        # whose head bone doesn't match the common-name list. Bound via
        # prop_search to the corresponding rig's bones — only valid
        # bone names accept the value. Blank = HEAD_LOCAL pairs fall
        # back to BASIS at bake time (with a one-shot auto-detect
        # attempt as a safety net).
        src_rig_for_head = scene.blendcap_retarget_source
        tgt_rig_for_head = scene.blendcap_retarget_target
        head_row_src = adv_col.row(align=True)
        head_row_src.label(text="Head (Source)")
        if src_rig_for_head and src_rig_for_head.type == 'ARMATURE':
            head_row_src.prop_search(
                scene, "blendcap_retarget_face_head_source",
                src_rig_for_head.data, "bones", text="")
        else:
            head_row_src.prop(scene,
                              "blendcap_retarget_face_head_source",
                              text="")
        head_row_tgt = adv_col.row(align=True)
        head_row_tgt.label(text="Head (Target)")
        if tgt_rig_for_head and tgt_rig_for_head.type == 'ARMATURE':
            head_row_tgt.prop_search(
                scene, "blendcap_retarget_face_head_target",
                tgt_rig_for_head.data, "bones", text="")
        else:
            head_row_tgt.prop(scene,
                              "blendcap_retarget_face_head_target",
                              text="")
        adv_col.separator()

        # Include Location & Scale in Rest Pose -- relevant only
        # when a rest-override path is active. Greyed otherwise so
        # the affordance is visible but the consequences clear.
        row = adv_col.row()
        row.enabled = bool(
            scene.blendcap_retarget_use_current_pose_as_rest
            or props.use_custom_rest_pose)
        row.prop(scene, "blendcap_retarget_current_pose_full_matrix")

        # World-space LOC sampling -- escape hatch for source rigs
        # with a non-identity object transform (Mixamo). See the
        # property tooltip in registration.py for full rationale.
        adv_col.prop(scene, "blendcap_retarget_use_world_location")

        # Face Expression -- global Strength + per-region overrides.
        # Layout matches BVH and ARKit Face Expression sections.
        # Strength applies uniformly to every face bone pair (LOC +
        # ROT). Per-Region toggle, when on, replaces Strength with
        # the per-region values (mutually exclusive to prevent
        # compounding). Per-pair explicit loc_scale overrides on
        # the pair table win over both regardless.
        _REGION_LABELS = {
            "jaw":    "Jaw",
            "gaze":   "Gaze",
            "lids":   "Eyelids",
            "brows":  "Brows",
            "lips":   "Lips",
            "cheeks": "Cheeks",
            "nose":   "Nose",
            "tongue": "Tongue",
        }
        adv_col.separator()
        amp_header = adv_col.row(align=True)
        amp_header.prop(scene, "blendcap_face_amp_expanded",
                        icon=('TRIA_DOWN' if scene.blendcap_face_amp_expanded
                              else 'TRIA_RIGHT'),
                        emboss=False, text="")
        amp_header.label(text="Face Expression")
        if scene.blendcap_face_amp_expanded:
            amp_box = adv_col.box()
            amp_col = amp_box.column(align=True)
            # Disable the whole block when the source rig has no
            # bones we'd recognize as face bones. The amplification
            # path classifies source-bone names against a fixed list
            # (Jaw / LeftEye / *Lip* / *Brow* / ...) and silently
            # no-ops on anything else, so we'd rather show greyed
            # sliders + a one-line reason than let the user move
            # sliders that do nothing.
            src_rig_for_face = scene.blendcap_retarget_source
            face_supported = _source_rig_has_recognized_face_bones(
                src_rig_for_face)
            if not face_supported:
                hint_row = amp_col.row()
                hint_row.label(
                    text="Only supported for BlendCap source armatures.",
                    icon='INFO')
                amp_col.separator()
            amp_body = amp_col.column(align=True)
            amp_body.enabled = face_supported
            # Global Strength -- greyed when per-region is on,
            # mirroring the BVH/ARKit pattern.
            str_row = amp_body.row()
            str_row.enabled = not bool(scene.blendcap_face_capture_compensation)
            str_row.prop(scene, "blendcap_face_amp_global",
                         text="Strength")
            amp_body.separator()
            amp_body.prop(scene, "blendcap_face_capture_compensation",
                          text="Use Per-Region")
            amp_sub = amp_body.column(align=True)
            amp_sub.enabled = bool(scene.blendcap_face_capture_compensation)
            for _region, _label in _REGION_LABELS.items():
                amp_sub.prop(scene, f"blendcap_face_amp_{_region}",
                             text=_label)
        # (Face denoising moved out of retarget. It now lives in
        # the BVH and ARKit sections so the filter bakes into the
        # exported data and travels to other programs.)

        # Custom IK source bones -- collapsible sub-table. Sits at
        # the bottom of Advanced (this is power-user territory and
        # rarely touched). End-effector defaults are existing BVH
        # bones (LeftHand etc.); pole defaults are the ephemeral
        # helper tip bones created at FK->IK bake time. Override
        # only for non-BlendCap source rigs. prop_search against
        # the source rig's pose bones gives the same eyedropper +
        # dropdown affordances as the main bone-pair table.
        adv_col.separator()
        ik_src_header = adv_col.row(align=True)
        ik_src_header.prop(scene, "blendcap_retarget_ik_src_expanded",
                           icon=('TRIA_DOWN' if scene.blendcap_retarget_ik_src_expanded
                                 else 'TRIA_RIGHT'),
                           emboss=False, text="")
        ik_src_header.label(text="Custom IK Source Bones")
        if scene.blendcap_retarget_ik_src_expanded:
            ik_src_box = adv_col.box()
            # Master toggle. When off, the FK->IK bake uses the
            # silent BlendCap BVH defaults (LeftHand etc.) and the
            # field rows below are visually disabled. When on,
            # blank fields fall back to the same defaults per-slot
            # -- only filled fields redirect.
            ik_src_box.prop(scene, "blendcap_use_custom_ik_sources")
            src_rig = scene.blendcap_retarget_source
            if src_rig is None or src_rig.type != 'ARMATURE':
                ik_src_box.label(
                    text="Pick a source rig to enable bone search.",
                    icon='INFO')
            else:
                # Labels describe the BONE TO TYPE on the source rig,
                # not the downstream IK control role. "Pole" labels
                # used to mislead users into hunting for explicit
                # elbow/knee bones; the FORMULA path (default) just
                # wants the chain mid — forearm or shin.
                # Order matches the FK->IK Mapping table convention
                # (operators_ik._FIXED_LIMB_ROWS): Arm.L, Arm.R,
                # Leg.L, Leg.R, with each limb's end + pole grouped
                # together. So the user reads each limb as a small
                # 2-field block instead of having to hunt the pole
                # for a given end four rows down.
                _ik_src_rows = (
                    ("blendcap_ik_src_left_arm_end",   "Left Hand"),
                    ("blendcap_ik_src_left_arm_pole",  "Left Forearm"),
                    ("blendcap_ik_src_right_arm_end",  "Right Hand"),
                    ("blendcap_ik_src_right_arm_pole", "Right Forearm"),
                    ("blendcap_ik_src_left_leg_end",   "Left Foot"),
                    ("blendcap_ik_src_left_leg_pole",  "Left Shin"),
                    ("blendcap_ik_src_right_leg_end",  "Right Foot"),
                    ("blendcap_ik_src_right_leg_pole", "Right Shin"),
                )
                sub = ik_src_box.column(align=True)
                sub.enabled = bool(scene.blendcap_use_custom_ik_sources)
                for prop_name, label in _ik_src_rows:
                    row = sub.row(align=True)
                    row.label(text=label)
                    row.prop_search(scene, prop_name,
                                    src_rig.pose, "bones", text="")


def _draw_arkit_section(layout, box, props):
    # ARKit shape keys: collapsible sub-section inside the retargeting
    # box, separate workflow from bone retargeting. Face smoothing is
    # applied automatically (One-Euro, optimal for face capture) and
    # not exposed in the UI.
    box.separator()
    ark_header = box.row(align=True)
    ark_header.prop(props, "arkit_section_expanded",
                    icon=('TRIA_DOWN' if props.arkit_section_expanded
                          else 'TRIA_RIGHT'),
                    emboss=False, text="")
    ark_header.label(text="ARKit Shape Keys", icon='SHAPEKEY_DATA')
    if props.arkit_section_expanded:
        ark_col = box.column(align=True)
        ark_col.label(text="ARKit Face Data:")
        ark_col.prop(props, "arkit_face_data", text="")
        ark_col.separator()
        ark_col.label(text="Face Smoothing")
        ark_col.prop(props, "arkit_face_smooth", text="Face (s)")

        # ARKit Face Noise Filtering -- per-region dead-zone, baked
        # into the keyed Shape Keys.
        ark_col.separator()
        ark_nf_header = ark_col.row(align=True)
        ark_nf_header.prop(props, "arkit_face_denoise_expanded",
                           icon=('TRIA_DOWN' if props.arkit_face_denoise_expanded
                                 else 'TRIA_RIGHT'),
                           emboss=False, text="")
        ark_nf_header.label(text="Face Noise Filtering")
        if props.arkit_face_denoise_expanded:
            ark_nf_col = ark_col.column(align=True)
            ark_nf_col.prop(props, "arkit_face_denoise_enabled",
                            text="Use Face Noise Filtering")
            ark_sub = ark_nf_col.column(align=True)
            ark_sub.enabled = bool(props.arkit_face_denoise_enabled)
            for _region in ("jaw", "gaze", "lids", "brows", "lips",
                            "cheeks", "nose", "tongue"):
                ark_sub.prop(props, f"arkit_face_denoise_{_region}")

        # Face Expression -- global Strength + per-region overrides.
        # Layout matches the Retarget and BVH Face Expression sections.
        # When per-region is on, the apply_arkit pipeline bypasses the
        # global Strength (face_expression_multiplier) to prevent
        # compounding.
        ark_col.separator()
        ark_fa_header = ark_col.row(align=True)
        ark_fa_header.prop(props, "arkit_face_amp_expanded",
                           icon=('TRIA_DOWN' if props.arkit_face_amp_expanded
                                 else 'TRIA_RIGHT'),
                           emboss=False, text="")
        ark_fa_header.label(text="Face Expression")
        if props.arkit_face_amp_expanded:
            ark_fa_col = ark_col.column(align=True)
            # Global Strength -- greyed when per-region is on.
            ark_str_row = ark_fa_col.row()
            ark_str_row.enabled = not bool(props.arkit_face_amp_enabled)
            ark_str_row.prop(props, "face_expression_multiplier",
                             text="Strength")
            ark_fa_col.separator()
            ark_fa_col.prop(props, "arkit_face_amp_enabled",
                            text="Use Per-Region")
            ark_fa_sub = ark_fa_col.column(align=True)
            ark_fa_sub.enabled = bool(props.arkit_face_amp_enabled)
            for _region in ("jaw", "gaze", "lids", "brows", "lips",
                            "cheeks", "nose", "tongue"):
                ark_fa_sub.prop(props, f"arkit_face_amp_{_region}")

        ark_col.separator()
        # While the ARKit apply's face-processing subprocess runs, both
        # action buttons collapse into the shared progress bar + Cancel
        # (matches the Retarget section's layout).
        if arkit_state.get('running'):
            cancel_row = ark_col.row()
            cancel_row.scale_y = 1.2
            cancel_row.operator("blendcap.cancel_arkit_apply",
                                icon='CANCEL', text="Cancel")
            pct = min(95.0, arkit_state.get('progress', 0.0))
            status = arkit_state.get('status_text', '')
            label = (f"Applying ARKit: {int(pct)}%  {status}"
                     if status else f"Applying ARKit: {int(pct)}%")
            _draw_progress_bar(ark_col, pct, label, scale_y=1.2)
        else:
            apply_ark_row = ark_col.row()
            apply_ark_row.scale_y = 1.2
            apply_ark_row.operator("blendcap.apply_arkit",
                                   icon='SHAPEKEY_DATA',
                                   text="Apply ARKit Shape Keys")
            csv_ark_row = ark_col.row()
            csv_ark_row.operator("blendcap.save_arkit_csv",
                                 icon='EXPORT',
                                 text="Save ARKit CSV")

    layout.separator()


def _draw_bvh_library_section(layout, context, props):
    # --- BVH LIBRARY ---
    lib_box = layout.box()
    lib_header = lib_box.row(align=True)
    lib_header.prop(props, "bvh_library_expanded",
                    icon=('TRIA_DOWN' if props.bvh_library_expanded
                          else 'TRIA_RIGHT'),
                    emboss=False, text="")
    lib_header.label(text="BVH Library", icon='ASSET_MANAGER')
    if props.bvh_library_expanded:
        scene = context.scene
        entries = scene.blendcap_bvh_entries
        if len(entries) == 0:
            lib_box.label(text="Library is empty.")
        else:
            lib_box.template_list(
                "BLENDCAP_UL_bvh", "",
                scene, "blendcap_bvh_entries",
                scene, "blendcap_bvh_index",
                rows=3,
            )
            idx = scene.blendcap_bvh_index
            has_sel = 0 <= idx < len(entries)
            sel_name = entries[idx].filename if has_sel else ""
            action_row = lib_box.row(align=True)
            action_row.enabled = has_sel
            action_row.operator("blendcap.import_lib_bvh", text="Import", icon='IMPORT').filename = sel_name
            action_row.operator("blendcap.export_lib_bvh", text="Save BVH", icon='FILE_TICK').source_bvh = sel_name
            action_row.operator("blendcap.export_lib_fbx", text="Save FBX", icon='EXPORT').source_bvh = sel_name
            action_row.operator("blendcap.delete_lib_bvh", text="Delete", icon='TRASH').filename = sel_name

    layout.separator()


def draw_blendcap_ui(self, context):
    layout = self.layout
    props = context.scene.blendcap_props
    prefs = addon_prefs(context)
    hide_setup = bool(prefs and prefs.hide_setup_section)
    hide_clear_all = bool(prefs and prefs.hide_clear_all)

    _draw_setup_section(layout, hide_setup)
    # Donate goes right BELOW Setup, in its own labeled box (see
    # _draw_donate_section): clearly part of BlendCap's UI yet still prominent.
    _draw_donate_section(layout, prefs)
    _draw_source_section(layout, context, props)

    is_running = blendcap_state['running']
    is_bvh_running = bvh_state['running']

    _draw_capture_section(layout, context, props, is_running, hide_clear_all)
    _draw_bvh_settings_section(layout, context, props, is_running, is_bvh_running)
    box = _draw_retargeting_section(layout, context, props)
    _draw_arkit_section(layout, box, props)
    _draw_bvh_library_section(layout, context, props)

# ============================================================
# UI PANELS
# ============================================================
class VIEW3D_PT_blendcap_ui(bpy.types.Panel):
    bl_label = "BlendCap"
    bl_idname = "VIEW3D_PT_blendcap_ui"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BlendCap'
    def draw(self, context): draw_blendcap_ui(self, context)

class OUTPUT_PT_blendcap_ui(bpy.types.Panel):
    bl_label = "BlendCap"
    bl_idname = "OUTPUT_PT_blendcap_ui"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'output'
    def draw(self, context): draw_blendcap_ui(self, context)
