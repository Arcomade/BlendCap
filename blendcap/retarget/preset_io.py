"""Retargeting preset I/O — JSON load/save, directory layout, sanitization,
and face-region classification with amplitude / denoise resolvers.

Shipped presets live in {addon_dir}/retarget_maps/ and ship with the
addon. User-saved presets save alongside the shipped ones now; the
{user_resource('CONFIG')}/blendcap/retarget_maps/ folder is still READ
so legacy custom maps saved there before the change continue to show
up in the dropdown.

The face-region helpers (`_face_region_from_source`, `_resolve_loc_scale`,
`_resolve_denoise`) live here too — they read scene properties whose
values the preset JSON drives, so they form one logical group with the
load/save code.
"""
import os
import json

import bpy

from ..helpers import _addon_dir


# Filenames of presets that ship with the addon. Saving over any of
# these is blocked, and the preset dropdown labels them as "Standard"
# regardless of which folder they were read from. Hardcoded (not derived
# from a directory listing) so a user-saved preset that happens to live
# in the shipped folder doesn't promote itself to read-only.
SHIPPED_PRESET_FILENAMES = frozenset({
    # BlendCap-source presets — full body+hands+face. "new"/"old" on
    # Rigify refers to which face rig system the preset targets.
    # Standalone body / body+hands variants were dropped: a face preset
    # loaded on a body-only rig naturally no-ops its face pairs at bake
    # time, so the separate body presets were redundant. For ARP, only
    # the modern Facial 2 mapping is shipped; the legacy-face variant
    # was dropped since no legacy ARP rigs are in use.
    "blendcap_to_rigify_new.json",
    "blendcap_to_rigify_old.json",
    "blendcap_to_arp.json",
    "blendcap_to_mixamo.json",
    # CloudRig (Sintel-style face) — built/validated against the
    # Blender Studio Sintel rig, 2026-06. Body + hands + full face;
    # mid-lip falloff comes from the rig's own ACT-MouthCorner
    # corrective actions (no per-pair amplitude fields — the map is
    # fully reproducible from the visible UI by design).
    "blendcap_to_cloudrig.json",
    # Mixamo control-rig overlay target with BlendCap BVH source.
    "blendcap_to_mixamo_ctrl.json",
    # Mixamo-source variants — niche case of transferring an already-
    # animated Mixamo skeleton onto a different rig. Body+hands only
    # (Mixamo capture carries no face data, so no Rigify-old variant
    # is needed).
    "mixamo_to_mixamo_ctrl.json",
    "mixamo_to_arp.json",
    "mixamo_to_rigify.json",
    "mixamo_to_cloudrig.json",
})


# Face Amplification region list. Per-region multiplier values live as
# scene FloatProperties (blendcap_face_amp_<region>); they're user-tunable
# via sliders in the panel and override-able per preset via the preset
# JSON "face_amplification" block. Engine defaults are 1.0 for every
# region (pure 1:1 transfer). Multipliers scale BOTH LOC and ROT
# uniformly per pair — e.g. setting Jaw to 0.7 dampens jaw_master's
# rotation by 30% (which is what mutes jaw "noise" on Rigify rigs
# where the jaw motion is rotation-driven). Pre-v57 the multipliers
# scaled LOC only — that didn't help Rigify's rotation-dominant jaw.
# Face presets that need non-1.0 values (ARP's tuned-for-c_jawbone.x
# cascade, rigs with far look-at targets that need gaze boost, etc.)
# ship a face_amplification block in their JSON that overwrites these
# on preset load.
FACE_AMPLIFICATION_REGIONS = ("jaw", "gaze", "lids", "brows", "lips", "cheeks", "nose", "tongue")


def _face_region_from_source(source_name):
    """Classify a source bone name into a face region (or None if it's
    not a face bone). The BVH source uses Mixamo-style face names from
    extract_face_calibration.py's rename map (Jaw, LeftEye, LeftUpperLid,
    LeftBrow, LeftUpperLip, LeftCheekUpper, LeftNostril, ...) — this is
    independent of the target rig's naming convention.

    'gaze' vs 'lids' split: only LeftEye / RightEye drive the look-at
    target (the lever-arm case that needs the heavy boost). Lid bones
    around the eye get their own (smaller) boost so blinking reads as a
    full closure without yanking the lid geometry off the eyeball.
    'lips' vs 'cheeks' are split because target rigs respond to them
    differently. Nostril/nose bones are grouped with cheeks (mid-face)
    since they're rarely driven and the small bump doesn't matter when
    they are."""
    if not source_name:
        return None
    # Case-insensitive substring checks against the Mixamo-style names
    # in the rename map: 'Tongue' routes to its own region (checked
    # BEFORE jaw — tongue bones parent under the jaw run but get their
    # own slider now that tongueOut is a live capture channel);
    # anything containing 'Jaw' / 'Chin' routes to jaw; an exact
    # LeftEye/RightEye match routes to gaze (not to lids); anything
    # containing 'Lid' routes to lids; 'Lip' OR 'MouthCorner' routes to
    # lips (mouth-corner bones are the smile-critical ones, so they
    # have to share the lips amplification); etc.
    n = source_name.lower()
    if "tongue" in n:
        return "tongue"
    if "jaw" in n or "chin" in n:
        return "jaw"
    if n in ("lefteye", "righteye"):
        return "gaze"
    if "lid" in n:
        return "lids"
    if "brow" in n or "forehead" in n or "temple" in n:
        return "brows"
    if "lip" in n or "mouthcorner" in n:
        return "lips"
    if "cheek" in n:
        return "cheeks"
    if "nose" in n or "nostril" in n:
        return "nose"
    return None


def _resolve_loc_scale(pair, compensation_on, scene):
    """Return the per-pair amplification multiplier, applying the
    precedence: explicit preset override > scene per-region slider
    (only if checkbox on) > 1.0. Applied to BOTH the LOC and ROT
    channels — the multiplier scales translation vectors AND rotation
    angles uniformly, so a region slider at 0.7 tames a jaw bone's
    rotation by 30% (same amount it would tame a translated bone).
    (The function name is historical — it was LOC-only pre-v57.)

    Convention: pair.loc_scale == 1.0 (the field default) means "no
    explicit preset override, fall through to the scene's region slider."
    Any other value is treated as an explicit override that wins
    regardless of the checkbox. To opt one face pair OUT of amplification
    while keeping the checkbox on globally, set loc_scale to a value
    like 1.0001 in the preset (or just turn the checkbox off). The
    bypass is intentional — being "explicit about 1.0" isn't worth a
    second boolean field. Negative values flip translation direction
    AND reverse rotation; useful when a target's rest orientation is
    180° from the source's."""
    explicit = float(getattr(pair, 'loc_scale', 1.0))
    if abs(explicit - 1.0) > 1e-6:
        return explicit
    region = _face_region_from_source(getattr(pair, 'source', ''))
    if region is None:
        return 1.0
    # Per-region toggle on -> use the region-specific multiplier.
    # Off -> use the global "Strength" multiplier (default 1.0). The
    # two are mutually exclusive in the UI to prevent compounding.
    if compensation_on:
        return float(getattr(scene, f"blendcap_face_amp_{region}", 1.0))
    return float(getattr(scene, "blendcap_face_amp_global", 1.0))


def _resolve_denoise(pair, denoise_on, scene):
    """Return the per-pair denoise threshold in raw slider units
    (interpreted as mm for LOC channels, degrees for ROT channels at
    the apply site). 0.0 means no denoising. Returns 0.0 immediately
    if the master checkbox is off OR the pair's source bone doesn't
    classify into a face region. No per-pair override field exists —
    denoise is region-only by design (per-bone tuning is what the
    amplitude slider's per-pair loc_scale is for). Apply BEFORE
    _resolve_loc_scale scaling so the threshold compares against raw
    captured magnitudes, not amplitude-modified ones."""
    if not denoise_on:
        return 0.0
    region = _face_region_from_source(getattr(pair, 'source', ''))
    if region is None:
        return 0.0
    return float(getattr(scene, f"blendcap_face_denoise_{region}", 0.0))


def _sanitize_preset_filename(name):
    """Free-form preset name -> safe '.json' filename. Empty input
    returns '' so callers can detect and refuse."""
    s = (name or "").strip().lower().replace(" ", "_")
    for ch in '<>:"/\\|?*':
        s = s.replace(ch, "_")
    if not s:
        return ""
    return s if s.endswith(".json") else s + ".json"


def _retarget_maps_dir_shipped():
    return os.path.join(_addon_dir(), "retarget_maps")

def _retarget_maps_dir_user():
    try:
        base = bpy.utils.user_resource('CONFIG')
    except Exception:
        base = os.path.join(os.path.expanduser("~"), ".config")
    d = os.path.join(base, "blendcap", "retarget_maps")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d

def _list_presets_in(directory, kind_label):
    """Return [(file_path, display_name, tooltip)] for every .json under
    directory, walking subfolders recursively so the user can organize
    presets into nested folders. Save paths haven't changed — presets
    still save into the top-level folder; subfolder organization is a
    manual filesystem move."""
    out = []
    if not os.path.isdir(directory):
        return out
    for root, dirs, files in os.walk(directory):
        # Stable ordering. listdir/walk order is filesystem-defined.
        dirs.sort()
        files.sort()
        for fn in files:
            if not fn.endswith('.json'):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                name = data.get('name', os.path.splitext(fn)[0])
            except Exception:
                name = os.path.splitext(fn)[0]
            rel = os.path.relpath(path, directory).replace("\\", "/")
            out.append((path, name, f"{kind_label} preset: {rel}"))
    return out


def _load_retarget_map(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save_retarget_map(path, name, map_hint, namespace_strip,
                        pairs_collection, auto_bake_ik=False, ik_chains_collection=None,
                        face_head_source="", face_head_target="",
                        face_capture_compensation=None,
                        face_amplification=None,
                        face_amp_global=None,
                        use_world_location=None,
                        custom_ik_sources=None,
                        source_prefix=""):
    """`map_hint` is the preset's target_kind string (e.g. "rigify_control",
    "generic", or empty). `auto_bake_ik` mirrors the "Auto-bake IK on Apply"
    checkbox; `ik_chains_collection` is the FK->IK Mapping collection
    (one row per IK constraint owner). `face_capture_compensation`, when
    not None, records the scene checkbox state so reloading the preset
    restores it. All persist to JSON so re-loading the preset restores
    the full configuration. ik_chains_collection persists every row that
    has data (owner OR a user-filled control), each stamped with its
    fixed limb_kind/side so load can place it even when the auto-detected
    owner is blank.

    `namespace_strip` is the TARGET prefix (legacy JSON key kept for
    back-compat — old presets had this as a both-sides aid, now it's
    target-specific). `source_prefix` is written under a new JSON key
    of the same name; only emitted when non-empty so older readers
    silently ignore it.

    `face_amp_global` is the global Face Expression Strength slider
    (lives next to the per-region values in the UI). When provided, it
    folds into the existing `face_amplification` dict under the reserved
    key "global" so all face-amp values stay in one block.

    `use_world_location` mirrors the Advanced > "Use World Location"
    checkbox. `custom_ik_sources` is the Custom IK Source Bones block
    as a dict — keys: `enabled` (bool), `left_hand`, `right_hand`,
    `left_foot`, `right_foot`, `left_forearm`, `right_forearm`,
    `left_shin`, `right_shin`. Emitted only when the master toggle is
    on OR any bone field is non-empty, so untouched presets stay clean.

    Face denoise state (`face_denoise` / `face_denoising`) was previously
    written here but those controls now live in the BVH and ARKit
    sections, not Retargeting — so they're no longer part of the bone-
    map preset schema. Old presets that carry them are tolerated on
    load (ignored), but new saves never emit them.

    Pair source/target names persist as already stored on the pair
    collection — which, after the smart-strip update callbacks (and
    legacy-strip on load), is the SHORT form. The prefixes above carry
    the namespace info; together they fully round-trip."""
    data = {
        "name": name,
        "version": 1,
        "target_kind": map_hint or "generic",
        "namespace_strip": namespace_strip or "",
        "auto_bake_ik": bool(auto_bake_ik),
        "pairs": [],
    }
    if source_prefix:
        data["source_prefix"] = source_prefix
    if face_head_source:
        data["face_head_source"] = face_head_source
    if face_head_target:
        data["face_head_target"] = face_head_target
    if face_capture_compensation is not None:
        data["face_capture_compensation"] = bool(face_capture_compensation)
    # face_amplification + face_amp_global share one JSON block. Global
    # rides under the reserved key "global"; per-region keys come from
    # FACE_AMPLIFICATION_REGIONS. Emit the block whenever EITHER source
    # provided a value so a preset with only a global tweak still saves
    # cleanly.
    if face_amplification is not None or face_amp_global is not None:
        amp_out = {}
        if face_amplification is not None:
            amp_out.update({k: float(v) for k, v in face_amplification.items()})
        if face_amp_global is not None:
            amp_out["global"] = float(face_amp_global)
        data["face_amplification"] = amp_out
    if use_world_location is not None:
        data["use_world_location"] = bool(use_world_location)
    if custom_ik_sources is not None:
        # Drop the whole block when nothing meaningful is set: master
        # toggle off AND every bone field blank. Keeps untouched presets
        # from gaining a noise-only key.
        enabled = bool(custom_ik_sources.get("enabled", False))
        bones = {k: str(v or "") for k, v in custom_ik_sources.items() if k != "enabled"}
        if enabled or any(bones.values()):
            block = {"enabled": enabled}
            block.update(bones)
            data["custom_ik_sources"] = block
    if ik_chains_collection is not None:
        chains_out = []
        for c in ik_chains_collection:
            # Keep a row if it carries ANY data — the auto-detected owner
            # OR a user-filled ik_control / pole_control. Requiring owner
            # silently dropped rows on rigs whose IK setup discovery can't
            # classify (owner stays blank) even though the user filled the
            # controls. limb_kind + side are stamped so the loader can
            # place the row without needing to parse the owner name.
            if not (c.owner or c.ik_control or c.pole_control):
                continue
            chains_out.append({
                "limb_kind": c.limb_kind,
                "side": c.side,
                "owner": c.owner,
                "ik_control": c.ik_control,
                "pole_control": c.pole_control,
            })
        if chains_out:
            data["ik_chains"] = chains_out
    for p in pairs_collection:
        if not p.source or not p.target:
            continue
        entry = {
            "source": p.source,
            "target": p.target,
            "channels": p.channels,
            "influence": float(p.influence),
        }
        if p.channels in ('LOC', 'LOC_ROT'):
            entry["axes"] = p.axes or "XYZ"
            loc_space = getattr(p, 'loc_space', 'BASIS')
            if loc_space and loc_space != 'BASIS':
                entry["loc_space"] = loc_space.lower()
            loc_scale = float(getattr(p, 'loc_scale', 1.0))
            if abs(loc_scale - 1.0) > 1e-6:
                entry["loc_scale"] = loc_scale
            if bool(getattr(p, 'loc_scale_residual', False)):
                entry["loc_scale_residual"] = True
        data["pairs"].append(entry)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
