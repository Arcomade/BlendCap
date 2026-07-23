"""FK -> IK retargeting by sampling the target's retargeted FK chain.

The source BVH drives FK retargeting onto the target rig's FK chain; this
operator then samples that target FK chain per frame and writes IK control
poses that match. No source-rig sampling — once FK retarget has run, the
target's FK chain carries the BVH motion at the target's exact proportions
and placement, eliminating scale, world-position, and rest-pose mismatch
issues.

  - End-effector positions: target IK control gets the same armature-
    space matrix as the corresponding target FK end bone (e.g.
    `hand_ik.L` ← `hand_fk.L`).
  - Pole positions: computed via the canonical formula
    (_compute_pole_position in retarget.py) on the target FK chain's
    posed root + mid bones. Twist-immune, geometrically correct,
    rig-agnostic.

Target FK chain bones are derived from the FK retarget pair list
(`scene.blendcap_retarget_pairs`): for each chain we look up the FK
pair whose source is the BlendCap source bone (e.g. 'LeftHand',
'LeftForeArm') and use that pair's target as the target FK chain
bone name. This works for any target rig family — Rigify, ARP,
custom — because the user has already configured the FK pair map
for their rig.

Pairs come from `scene.blendcap_retarget_ik_chains` rows (UIList). For
each ARM/LEG row the operator emits:
    end-effector: BONE pair  — target <hand|foot> FK end → row.ik_control
    pole:         POLE pair  — target FK root + mid → row.pole_control

The 8 Custom IK Source Bones overrides specify SOURCE bone names that
identify the chain via the FK pair list. Defaults match BlendCap BVH
(LeftHand/RightHand/LeftFoot/RightFoot for end-effector source;
LeftForeArm/RightForeArm/LeftLeg/RightLeg as pole chain mid sources).

The chain discovery + UIList sync (_discover_ik_chains,
sync_ik_chain_collection, _classify_limb_kind, _chain_label_str) is
preserved — the panel still populates rows from the target rig's IK
constraints.

Pole-vector flip: kept. The IK pairs key positional pole control bones,
but the rig won't actually use them unless `pole_vector` is set to "Pole"
mode (Rigify convention). The flip is gated to ARM/LEG chains being
baked and is never keyed.
"""
import bpy
import re
import types

from .helpers import (
    retarget_progress_state,
    format_workspace_status,
)
from .retarget import _bake_world_pairs_iter


# Pattern for Rigify chain-root FK bones: `<base>_fk.<side>` (also `.fk`,
# `-fk`, and uppercase variants). Used to derive the conventional Rigify
# limb properties bone name (`<base>_parent.<side>`), where pole_vector /
# IK_FK / IK_Stretch sliders live.
_RIGIFY_FK_ROOT_RE = re.compile(r'^(.+?)[._-][Ff][Kk](\.[LRlr])?$')

# Modal-bake state. Mirrors the `_BAKE_RUNNING` / `_BAKE_CANCEL_REQUESTED`
# pattern in operators_retarget.py — module-level so the panel and the
# cancel operator can read them without holding an instance reference.
_FK_IK_RUNNING = False
_FK_IK_CANCEL_REQUESTED = False


# Verbose diagnostic logging for the FK->IK setup phase. Flip to True
# to re-enable when debugging a rig where pair-building / chain detection
# misbehaves. Output lines tag with "[BlendCap FK->IK]" for filtering in
# Window > Toggle System Console.
_DEBUG = False


def _dbg(msg):
    if _DEBUG:
        print(f"[BlendCap FK->IK] {msg}")


def _dbg_row(prefix, i, row):
    _dbg(f"{prefix}[{i}] owner={row.owner!r} kind={row.limb_kind} "
         f"ik_control={row.ik_control!r} pole_control={row.pole_control!r}")


def _dbg_chain(prefix, c):
    sub = c.get('subtarget')
    pole = c.get('pole_subtarget')
    sub_name = sub.name if sub else None
    pole_name = pole.name if pole else None
    ik_names = [b.name for b in c.get('ik_chain') or []]
    fk_names = [b.name for b in c.get('fk_chain') or []]
    _dbg(f"{prefix}owner={c['owner_ik'].name!r} subtarget={sub_name!r} "
         f"pole_subtarget={pole_name!r} "
         f"ik_chain={ik_names} fk_chain={fk_names} "
         f"skip={c.get('skip_reason')!r}")


def _derive_rigify_properties_bone_name(fk_root_name):
    """For a Rigify chain-root FK bone name (`upper_arm_fk.L`, `thigh_fk.L`,
    etc.) return the conventional limb properties bone name
    (`upper_arm_parent.L`, `thigh_parent.L`). Returns None if the input
    doesn't match Rigify's `<base>_fk.<side>` pattern."""
    m = _RIGIFY_FK_ROOT_RE.match(fk_root_name)
    if not m:
        return None
    base, side = m.group(1), m.group(2) or ''
    return f"{base}_parent{side}"


# Substring substitutions tried (in order) to derive a parallel FK bone
# name from an IK bone name. First substitution that produces a name
# present in the rig wins. Used by chain discovery for the UIList — the
# bake itself no longer needs FK partners (it samples the source rig
# directly).
_IK_TO_FK_SUBS = (
    ('_ik', '_fk'),
    ('.ik', '.fk'),
    ('_IK', '_FK'),
    ('.IK', '.FK'),
    ('-ik', '-fk'),
    ('-IK', '-FK'),
)

# Rigify-style metacontrol prefixes stripped before the _ik->_fk swap.
_STRIPPABLE_PREFIXES = (
    'MCH-', 'mch-',
    'DEF-', 'def-',
    'ORG-', 'org-',
)

# CloudRig-style IK control prefixes: 'IK-Hand.L' pairs with plain
# 'Hand.L'. Used by _ik_to_fk_name's prefix-strip rule.
_IK_PREFIX_STRIPS = (
    'IK-', 'ik-',
    'IK_', 'ik_',
    'IK.', 'ik.',
)

# Side-suffix conventions across rig families. Returned side is uppercase
# 'L' or 'R' regardless of source casing — used as a key into the source-
# bone override scene properties.
_SIDE_SUFFIXES = (
    ('.L', 'L'), ('.R', 'R'),
    ('_L', 'L'), ('_R', 'R'),
    ('.l', 'L'), ('.r', 'R'),
)

# BlendCap-default source bone names used when "Use Custom IK Source
# Bones" is OFF. Keyed by (side, kind, role). End-effector bones are
# the BVH wrist / ankle bones; pole bones are the BVH chain MID bones
# whose head is the bend joint (head of forearm = elbow joint, head
# of leg / shin = knee joint). Used by _bake_world_pairs_iter via the
# pair list — no UI surface for these defaults; they're the silent
# fallback.
_BVH_DEFAULT_SOURCES = {
    ('L', 'ARM', 'end'):  'LeftHand',
    ('R', 'ARM', 'end'):  'RightHand',
    ('L', 'LEG', 'end'):  'LeftFoot',
    ('R', 'LEG', 'end'):  'RightFoot',
    ('L', 'ARM', 'pole'): 'LeftForeArm',
    ('R', 'ARM', 'pole'): 'RightForeArm',
    ('L', 'LEG', 'pole'): 'LeftLeg',
    ('R', 'LEG', 'pole'): 'RightLeg',
}

# Locked 4-row layout for the FK->IK Mapping table. The operator never
# adds or removes rows — only the four limb chains are baked, period.
# Order in this tuple defines the order rows appear in the panel.
_FIXED_LIMB_ROWS = (
    ('ARM', 'L'),
    ('ARM', 'R'),
    ('LEG', 'L'),
    ('LEG', 'R'),
)


def _derive_kind_side_from_owner(owner_name):
    """Best-effort (limb_kind, side) derivation from an IK constraint
    owner bone name. Used by the preset loader and the migration path
    to map legacy `ik_chains` entries (keyed by `owner`) onto the fixed
    4-row layout. Returns (None, None) if the owner doesn't look like a
    limb chain.

    Recognizes:
      Rigify  : MCH-shin_ik.L, MCH-forearm_ik.R
      ARP     : leg_ik_nostr.l, forearm_ik_nostr.r
      Custom  : anything with shin/calf/leg keywords for legs and
                forearm/lower_arm/arm keywords for arms.
    """
    if not owner_name:
        return (None, None)
    side = _side_from_owner(owner_name)
    name_lower = owner_name.lower()
    kind = None
    if 'forearm' in name_lower or 'lower_arm' in name_lower or 'lowerarm' in name_lower:
        kind = 'ARM'
    elif ('shin' in name_lower or 'calf' in name_lower or 'knee' in name_lower
            or 'lower_leg' in name_lower or 'lowerleg' in name_lower):
        kind = 'LEG'
    elif 'leg' in name_lower and 'thigh' not in name_lower and 'upper' not in name_lower:
        kind = 'LEG'
    elif ('arm' in name_lower and 'forearm' not in name_lower
            and 'upper' not in name_lower and 'shoulder' not in name_lower):
        kind = 'ARM'
    return (kind, side)


def ensure_fixed_ik_chain_rows(collection):
    """Lock the FK->IK Mapping collection to exactly 4 rows: Arm.L,
    Arm.R, Leg.L, Leg.R, in that order. User edits to ik_control /
    pole_control are preserved across the lock.

    Idempotent — if the collection already has the correct 4-row
    layout, this is a no-op (no clear, no rebuild, no fcurve touch).

    Migration: rows from before the `side` field was added may have
    `side == ''`; we recover side from the owner suffix and replay
    values into the new layout. Rows that don't fit one of the four
    fixed identities (e.g. finger / eye chains the older sync added)
    are dropped — those constraints don't belong in this table.
    """
    fixed_keys = list(_FIXED_LIMB_ROWS)

    # Snapshot existing values, keyed by (limb_kind, side). Migration:
    # if `side` is empty (legacy row), derive it from the owner suffix.
    snapshot = {}
    for row in collection:
        side = row.side
        if not side:
            side = _side_from_owner(row.owner) or ''
            # Try to recover kind too if the row's limb_kind is OTHER —
            # legacy rows might not have been classified.
            if row.limb_kind == 'OTHER':
                derived_kind, _ = _derive_kind_side_from_owner(row.owner)
                if derived_kind:
                    row_kind = derived_kind
                else:
                    continue  # not a limb — drop on rebuild
            else:
                row_kind = row.limb_kind
        else:
            row_kind = row.limb_kind
        key = (row_kind, side)
        if key in fixed_keys and key not in snapshot:
            snapshot[key] = {
                'owner': row.owner,
                'ik_control': row.ik_control,
                'pole_control': row.pole_control,
            }

    # Skip rebuild when the layout already matches — avoids touching
    # the collection (and the dirty/update callbacks chained off the
    # ik_control / pole_control properties) on every sync call.
    if (len(collection) == len(fixed_keys)
            and all((collection[i].limb_kind, collection[i].side) == fixed_keys[i]
                    for i in range(len(fixed_keys)))):
        return

    collection.clear()
    for kind, side in fixed_keys:
        row = collection.add()
        row.limb_kind = kind
        row.side = side
        snap = snapshot.get((kind, side))
        if snap:
            row.owner = snap['owner']
            row.ik_control = snap['ik_control']
            row.pole_control = snap['pole_control']


def _resolve_subtarget_control(subtarget_pb, pose_bones):
    """Resolve mechanism-bone subtargets to the user-visible IK control.

    Heuristic: the user-visible IK control is the bone with a parallel
    `_fk` partner in the rig. Mechanism/intermediate bones don't have FK
    partners — only the bone an animator would key does.

    Walks the parent chain and returns the first bone whose
    `_ik_to_fk_name` succeeds. Rigs without parallel FK chains return
    the original subtarget unchanged.
    """
    pb = subtarget_pb
    seen = set()
    while pb is not None and pb.name not in seen:
        seen.add(pb.name)
        if _ik_to_fk_name(pb.name, pose_bones) is not None:
            return pb
        pb = pb.parent
    return subtarget_pb


def _ik_to_fk_name(ik_name, pose_bones):
    """Return the parallel FK bone name for `ik_name`, or None if no
    candidate matches the substitution patterns + prefix-strip permutations."""
    candidates = [ik_name]
    for prefix in _STRIPPABLE_PREFIXES:
        if ik_name.startswith(prefix):
            candidates.append(ik_name[len(prefix):])
    for c in candidates:
        for old, new in _IK_TO_FK_SUBS:
            if old in c:
                trial = c.replace(old, new)
                if trial in pose_bones:
                    return trial
        # CloudRig-style IK PREFIX: the user control is 'IK-Hand.L' and
        # its FK partner is the bare chain bone 'Hand.L' (CloudRig's FK
        # controls are unprefixed; FK- only appears on spine/carpals).
        # The prefix-stripped name existing IS the FK-partner signal.
        # Mechanism bones stay excluded naturally: 'IK-M-Hand.L' strips
        # to 'M-Hand.L', which doesn't exist.
        for prefix in _IK_PREFIX_STRIPS:
            if c.startswith(prefix):
                trial = c[len(prefix):]
                if trial in pose_bones:
                    return trial
    return None


def _walk_chain(chain_end_pb, chain_count):
    """Return a list of pose bones from chain end (index 0) up the parent
    chain for `chain_count` bones. Stops early if a parent is None."""
    chain = [chain_end_pb]
    current = chain_end_pb
    for _ in range(chain_count - 1):
        if current.parent is None:
            break
        current = current.parent
        chain.append(current)
    return chain


def _discover_ik_chains(rig):
    """Scan the rig for IK constraints and return a list of chain specs.

    Each spec is a dict:
        owner_ik:        pose bone hosting the IK constraint (chain end)
        chain_count:     number of bones the constraint controls
        ik_chain:        [end, end.parent, ...] (length chain_count)
        fk_chain:        parallel FK bones (BEST-EFFORT, may be empty/short
                         on rigs whose mechanism bones don't follow the
                         _ik->_fk substitution — ARP `_nostr` suffix etc.)
        subtarget:       IK control pose bone (constraint.subtarget,
                         resolved through parent walk to find the user-
                         visible control). This is the bone the operator
                         auto-populates `ik_control` from, removing the
                         need for per-rig preset data on the IK side.
        pole_subtarget:  Pole control pose bone (constraint.pole_subtarget)
                         or None. Auto-populates the row's `pole_control`.
        skip_reason:     str if the chain isn't usable, else None.
    """
    if not rig or rig.type != 'ARMATURE':
        return []
    pbs = rig.pose.bones
    chains = []
    seen_keys = set()
    for pb in pbs:
        for con in pb.constraints:
            if con.type != 'IK':
                continue
            if con.influence <= 0.0:
                continue
            cc = int(getattr(con, 'chain_count', 0))
            if cc < 1:
                continue
            sub_name_for_key = getattr(con, 'subtarget', '') or ''
            key = (pb.name, sub_name_for_key, cc)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            ik_chain = _walk_chain(pb, cc)
            spec = {
                'owner_ik': pb,
                'chain_count': cc,
                'ik_chain': ik_chain,
                'fk_chain': [],
                'subtarget': None,
                'pole_subtarget': None,
                'skip_reason': None,
            }
            sub_name = getattr(con, 'subtarget', '') or ''
            if not sub_name or sub_name not in pbs:
                spec['skip_reason'] = f"IK on '{pb.name}' has no usable subtarget"
                chains.append(spec)
                continue
            spec['subtarget'] = _resolve_subtarget_control(pbs[sub_name], pbs)
            # Pole subtarget: take the constraint's pole_subtarget directly
            # (no parent walk — pole bones are typically pure-IK constructs
            # that don't have parallel FK partners, so the walk would over-
            # generalize). Empty / unset is fine; user can fill manually.
            pole_name = getattr(con, 'pole_subtarget', '') or ''
            if pole_name and pole_name in pbs:
                spec['pole_subtarget'] = pbs[pole_name]
            # Best-effort FK chain assembly.
            fk_chain = []
            for ik_pb in ik_chain:
                fk_name = _ik_to_fk_name(ik_pb.name, pbs)
                if fk_name is not None:
                    fk_chain.append(pbs[fk_name])
            spec['fk_chain'] = fk_chain
            chains.append(spec)
    return chains


def _classify_limb_kind(chain_spec):
    """Classify a discovered chain as ARM / LEG / OTHER. Falls back from
    fk_chain to ik_chain when the parallel-FK substitution didn't yield
    anything (ARP / custom rigs)."""
    fk_chain = chain_spec.get('fk_chain') or []
    ik_chain = chain_spec.get('ik_chain') or []
    name = ''
    if fk_chain:
        name = fk_chain[0].name.lower()
    elif ik_chain:
        name = ik_chain[0].name.lower()
    # Most-specific patterns first so we don't over-match. ARP uses
    # 'c_leg_fk.l' for the shin and 'c_arm_fk.l' for the upper arm; the
    # bare 'leg' / 'arm' fallbacks are gated to avoid catching thigh /
    # shoulder / forearm bones.
    if 'forearm' in name or 'lower_arm' in name or 'lowerarm' in name:
        return 'ARM'
    if ('shin' in name or 'calf' in name or 'knee' in name
            or 'lower_leg' in name or 'lowerleg' in name):
        return 'LEG'
    if 'leg' in name and 'thigh' not in name and 'upper' not in name:
        return 'LEG'
    if ('arm' in name and 'forearm' not in name
            and 'upper' not in name and 'shoulder' not in name):
        return 'ARM'
    return 'OTHER'


def apply_ik_chains_from_map(scene, map_dict):
    """Populate scene.blendcap_retarget_ik_chains from a loaded preset's
    `ik_chains` block. Called by retarget._populate_pairs_from_map at
    preset-load time. The loaded preset is authoritative: what it
    specifies IS the final panel state — the loader does not auto-fill
    blank rows from the rig afterward.

    JSON format (one entry per fixed limb row):
        "ik_chains": [
            {"limb_kind": "LEG", "side": "L", "owner": "...",
             "ik_control": "...", "pole_control": "..."},
            ...
        ]

    Each entry is placed by its explicit limb_kind + side (Arm.L / Arm.R
    / Leg.L / Leg.R). Older presets that lack those fall back to deriving
    the identity from the owner name via _derive_kind_side_from_owner.
    Entries that resolve to neither are dropped — the fixed 4-row layout
    has no place for finger / eye / other non-limb chains.

    Reset behavior: the four fixed rows are ensured, then BLANKED
    (owner / ik_control / pole_control cleared) before the preset's
    values are written. This makes a preset that omits `ik_chains` — or
    names only some of the four rows — load deterministically instead of
    inheriting the previously-loaded preset's controls, matching the
    reset-then-apply the rest of the loader uses
    (retarget._populate_pairs_from_map). The loader does NOT auto-fill
    blank rows from the rig afterward, so the panel reflects exactly what
    the loaded preset specifies (blank stays blank); rig discovery only
    populates the panel visibly on target-rig assignment.

    Must run inside the _LOADING_PRESET guard (it does, via the loader):
    ik_control / pole_control carry a dirty-marking update callback, so
    blanking/writing them unguarded would flip the preset to (Unsaved).
    """
    collection = scene.blendcap_retarget_ik_chains
    ensure_fixed_ik_chain_rows(collection)
    # Clear every row first so unspecified rows don't carry over.
    for row in collection:
        row.owner = ""
        row.ik_control = ""
        row.pole_control = ""
    entries = map_dict.get("ik_chains") or []
    # Index fixed rows by (kind, side) for direct write.
    rows_by_key = {(r.limb_kind, r.side): r for r in collection}
    for entry in entries:
        # Prefer the explicit fixed-row identity (limb_kind + side); fall
        # back to deriving it from the owner name for older presets that
        # didn't store them. This lets a row with a user-filled control
        # but a blank/unrecognized owner still land in the right row.
        kind = entry.get("limb_kind") or None
        side = entry.get("side") or None
        if kind not in ('ARM', 'LEG') or side not in ('L', 'R'):
            kind, side = _derive_kind_side_from_owner(entry.get("owner", "") or "")
        if kind is None or side is None:
            continue
        row = rows_by_key.get((kind, side))
        if row is None:
            continue
        row.owner = entry.get("owner", "") or ""
        row.ik_control = entry.get("ik_control", "") or ""
        row.pole_control = entry.get("pole_control", "") or ""


def sync_ik_chain_collection(rig, collection, fill_controls=True):
    """Lock the FK->IK Mapping to the fixed 4 limb rows and (if a rig is
    set) update each row's `owner` from the rig's IK constraints. When
    `fill_controls` is True, also auto-fill any blank ik_control /
    pole_control fields from the discovered IK constraint.

    The collection always ends up with exactly 4 rows: Arm.L, Arm.R,
    Leg.L, Leg.R. Add / remove operations don't happen — only updates
    to existing rows.

    `fill_controls` gates ONLY the blank-field auto-fill (the row lock
    and owner refresh always run). Pass True (default) when discovery
    should populate the panel as a visible, editable suggestion — used
    on target-rig assignment. The bake passes False so it honors exactly
    what the panel shows: a blank ik_control means "skip this chain"
    (_build_ik_pairs reports it), with no silent discovery. Non-blank
    values (preset-supplied or user-typed) are preserved either way.
    """
    # Always lock to the 4 fixed rows first, regardless of rig state.
    # This is what makes the panel show 4 rows even before a rig has
    # been picked / a preset has been loaded.
    ensure_fixed_ik_chain_rows(collection)

    if rig is None or rig.type != 'ARMATURE':
        _dbg(f"sync_ik_chain_collection: no rig — fixed rows ensured, "
             f"discovery skipped")
        return 0

    pbs = rig.pose.bones
    _dbg(f"sync_ik_chain_collection: rig={rig.name!r} ({len(pbs)} pose bones)")

    chains = _discover_ik_chains(rig)
    _dbg(f"  discovered {len(chains)} IK chains on rig:")
    for c in chains:
        _dbg_chain("    ", c)

    # Index discovered chains by (kind, side) for direct lookup against
    # the fixed rows. Non-limb chains and chains with no recognizable
    # side are dropped — they can't bind to one of the 4 rows.
    chain_by_keys = {}
    for c in chains:
        if c.get('skip_reason'):
            continue
        kind = _classify_limb_kind(c)
        if kind not in ('ARM', 'LEG'):
            continue
        side = _side_from_owner(c['owner_ik'].name)
        if side is None:
            continue
        if (kind, side) not in chain_by_keys:
            chain_by_keys[(kind, side)] = c

    for row in collection:
        key = (row.limb_kind, row.side)
        c = chain_by_keys.get(key)
        if c is None:
            _dbg(f"  row {key}: no matching IK constraint on rig — "
                 f"existing values preserved")
            continue
        owner_name = c['owner_ik'].name
        if row.owner != owner_name:
            _dbg(f"  row {key}: updating owner {row.owner!r} -> {owner_name!r}")
            row.owner = owner_name
        sub = c.get('subtarget')
        pole = c.get('pole_subtarget')
        sub_name = sub.name if sub else ''
        pole_name = pole.name if pole else ''
        if fill_controls and not row.ik_control and sub_name:
            _dbg(f"  row {key}: auto-fill ik_control = {sub_name!r}")
            row.ik_control = sub_name
        if fill_controls and not row.pole_control and pole_name:
            _dbg(f"  row {key}: auto-fill pole_control = {pole_name!r}")
            row.pole_control = pole_name

    _dbg(f"  collection AFTER sync ({len(collection)} rows):")
    for i, row in enumerate(collection):
        _dbg_row("    ", i, row)
    return 0


def _chain_label_str(row):
    """Plain-text label for a chain row, used in user-facing messages.
    With the locked 4-row layout, side is set explicitly on each row,
    so we read it directly instead of parsing the owner suffix."""
    side = row.side
    if row.limb_kind == 'ARM':
        return f"Arm.{side}" if side else "Arm"
    if row.limb_kind == 'LEG':
        return f"Leg.{side}" if side else "Leg"
    return row.owner or '(unknown)'


def _side_from_owner(owner_name):
    """Return 'L' or 'R' for a chain owner bone name, or None if neither
    side suffix matches. Handles `.L`, `.R`, `_L`, `_R`, `.l`, `.r`."""
    for suffix, side in _SIDE_SUFFIXES:
        if owner_name.endswith(suffix):
            return side
    return None


def _find_fk_pair_target(fk_pairs, source_bone_name):
    """Look up `source_bone_name` in the FK retarget pair list and
    return the corresponding target bone name (or None if not found).
    First match wins; rotation/location/loc_rot pair kind is ignored
    (the lookup just maps source → target naming)."""
    if not source_bone_name:
        return None
    for p in fk_pairs:
        if p.source == source_bone_name and p.target:
            return p.target
    return None


def _build_ik_pairs(scene, src_rig, tgt_rig):
    """Walk `scene.blendcap_retarget_ik_chains` and emit IK retargeting
    pairs that _bake_world_pairs_iter can consume.

    Target FK chain bones are derived per row from
    `scene.blendcap_retarget_pairs`: for each side/kind we look up the
    FK pair whose source is the BlendCap source bone (override or
    default) and use that pair's target as the target FK chain bone.
    This makes the IK retarget rig-agnostic — Rigify / ARP / custom
    target rigs all work as long as their FK pair map is configured
    (which it must be for FK retarget to function in the first place).

    Two pair kinds emitted (per ARM/LEG row, with pole_control set):
        BONE pair: end-effector. .target = row.ik_control,
            .target_chain_end = target FK end (e.g. 'hand_fk.L'
            or 'c_hand_fk.l').
        POLE pair: pole. .target = row.pole_control,
            .target_chain_root = target FK chain root (e.g.
            'upper_arm_fk.L'), .target_chain_mid = target FK chain
            mid (e.g. 'forearm_fk.L').

    Returns (pairs, skipped_reasons).
    """
    use_custom = bool(getattr(scene, 'blendcap_use_custom_ik_sources', False))
    if use_custom:
        # User opted into custom source bones. Per-field value wins;
        # blank fields fall back to the BlendCap BVH default for that
        # (side, kind, role) so a partial override doesn't break the
        # bake — only the fields the user actually filled in get
        # redirected to non-default bones.
        custom = {
            ('L', 'ARM', 'end'):  scene.blendcap_ik_src_left_arm_end,
            ('R', 'ARM', 'end'):  scene.blendcap_ik_src_right_arm_end,
            ('L', 'LEG', 'end'):  scene.blendcap_ik_src_left_leg_end,
            ('R', 'LEG', 'end'):  scene.blendcap_ik_src_right_leg_end,
            ('L', 'ARM', 'pole'): scene.blendcap_ik_src_left_arm_pole,
            ('R', 'ARM', 'pole'): scene.blendcap_ik_src_right_arm_pole,
            ('L', 'LEG', 'pole'): scene.blendcap_ik_src_left_leg_pole,
            ('R', 'LEG', 'pole'): scene.blendcap_ik_src_right_leg_pole,
        }
        overrides = {k: (custom[k] or _BVH_DEFAULT_SOURCES[k]) for k in custom}
    else:
        overrides = dict(_BVH_DEFAULT_SOURCES)
    # Resolve custom IK source bone names against the source rig: the
    # fields store the SHORT form (after smart-strip), but pair.source
    # has been temporarily resolved to its LONG form for the bake by
    # _resolve_pair_names_for_bake. We resolve here too so the
    # _find_fk_pair_target comparison lands. _resolve_against_rig is
    # exact-match-first, so already-long-form values (e.g. BVH defaults
    # like 'RightHand' on an unprefixed rig) pass through unchanged.
    from .retarget import _resolve_against_rig
    src_prefix = scene.blendcap_retarget_source_prefix or ""
    overrides = {
        k: _resolve_against_rig(v, src_rig, src_prefix)
        for k, v in overrides.items()
    }
    _dbg(f"_build_ik_pairs: use_custom={use_custom}, source bones:")
    for k, v in overrides.items():
        _dbg(f"  {k} = {v!r}")

    pairs = []
    skipped = []
    tgt_pbs = tgt_rig.pose.bones if (tgt_rig and tgt_rig.type == 'ARMATURE') else None
    src_pbs = src_rig.pose.bones if (src_rig and src_rig.type == 'ARMATURE') else None
    fk_pairs = scene.blendcap_retarget_pairs
    _dbg(f"  FK retarget pair count: {len(fk_pairs)}")
    _dbg(f"  source rig: {src_rig.name if src_rig else None!r} "
         f"({len(src_pbs) if src_pbs else 0} pose bones)")
    _dbg(f"  target rig: {tgt_rig.name if tgt_rig else None!r} "
         f"({len(tgt_pbs) if tgt_pbs else 0} pose bones)")
    _dbg(f"  IK chain rows: {len(scene.blendcap_retarget_ik_chains)}")

    for row_idx, row in enumerate(scene.blendcap_retarget_ik_chains):
        _dbg(f"  ---- row[{row_idx}] ----")
        _dbg_row("    ", row_idx, row)
        kind = row.limb_kind
        side = row.side
        # The locked 4-row layout guarantees kind in (ARM, LEG) and
        # side in (L, R); these guards are belt-and-suspenders for
        # legacy collections that haven't been re-synced yet.
        if kind not in ('ARM', 'LEG'):
            _dbg(f"    skipped: limb_kind={kind!r} (not ARM/LEG)")
            continue
        if side not in ('L', 'R'):
            _dbg(f"    skipped: side={side!r} (not L/R)")
            continue
        _dbg(f"    row identity: kind={kind!r} side={side!r}")
        if not row.ik_control:
            _dbg(f"    skipped: ik_control field is empty — user must "
                 f"fill in the FK->IK Mapping panel for chain "
                 f"{row.owner!r} (preset didn't pre-populate it)")
            skipped.append(f"{_chain_label_str(row)}: ik_control empty — skipped")
            continue
        if tgt_pbs is None or row.ik_control not in tgt_pbs:
            _dbg(f"    skipped: ik_control={row.ik_control!r} not on target rig")
            skipped.append(
                f"{_chain_label_str(row)}: ik_control '{row.ik_control}' "
                f"not on target rig — skipped")
            continue

        # ---- End-effector BONE pair ----
        end_src = overrides[(side, kind, 'end')] or ''
        _dbg(f"    end source override = {end_src!r}")
        if not end_src:
            _dbg(f"    skipped: end-effector source bone field is empty")
            skipped.append(
                f"{_chain_label_str(row)}: end-effector source bone is empty "
                f"(check Advanced > Custom IK Source Bones) — skipped")
            continue
        target_chain_end = _find_fk_pair_target(fk_pairs, end_src)
        _dbg(f"    FK pair lookup: {end_src!r} -> {target_chain_end!r}")
        if not target_chain_end:
            _dbg(f"    skipped: no FK pair with source={end_src!r}")
            skipped.append(
                f"{_chain_label_str(row)}: no FK retarget pair maps "
                f"source '{end_src}' to a target bone — skipped")
            continue
        if target_chain_end not in tgt_pbs:
            _dbg(f"    skipped: target FK end {target_chain_end!r} not on target rig")
            skipped.append(
                f"{_chain_label_str(row)}: target FK end '{target_chain_end}' "
                f"(from FK pair list) is not on target rig — skipped")
            continue
        pairs.append(types.SimpleNamespace(
            kind='BONE',
            target=row.ik_control,
            target_chain_end=target_chain_end,
            influence=1.0,
        ))
        _dbg(f"    + BONE pair: target={row.ik_control!r} "
             f"target_chain_end={target_chain_end!r}")

        # ---- Pole POLE pair (only if pole_control set) ----
        if not row.pole_control:
            _dbg(f"    pole_control empty — skipping pole pair (end-only)")
            continue
        if tgt_pbs is None or row.pole_control not in tgt_pbs:
            _dbg(f"    pole_control={row.pole_control!r} not on target rig "
                 f"— skipping pole pair")
            continue
        pole_src = overrides[(side, kind, 'pole')] or ''
        _dbg(f"    pole source = {pole_src!r}  use_custom={use_custom}")
        if not pole_src:
            _dbg(f"    skipped pole: pole source field empty")
            skipped.append(
                f"{_chain_label_str(row)}: pole source bone is empty "
                f"— skipping pole pair")
            continue

        # Two pole modes:
        #   SAMPLE   — sample the source bone's position directly as
        #              the pole. Intended for users with explicit pole
        #              indicator bones on their custom source rig
        #              (rare — Mixamo, Rokoko, Maya HumanIK etc. don't
        #              ship them). Currently UNREACHABLE: the panel
        #              UX (see panels.py around the "Pole" rows) tells
        #              users to type the chain MID bone — forearm or
        #              shin — which is the FORMULA contract, not
        #              SAMPLE's. SAMPLE on a chain-mid bone snaps the
        #              IK pole TO the joint, which is degenerate for
        #              the solver (the elbow plane can't align with
        #              the elbow itself).
        #   FORMULA  — treat the source bone name as the chain MID and
        #              run the canonical perpendicular-to-chain pole
        #              formula on the corresponding TARGET FK chain
        #              bones (root + mid, looked up via FK pair list).
        #              Source-rig-agnostic — works for any biped
        #              source regardless of object scale / rotation /
        #              units, because all sampling is target-side
        #              after FK retarget has populated it.
        # If a future use case needs SAMPLE back (user has a real pole
        # bone like 'pole.L'), gate it behind an explicit per-row toggle
        # (e.g. row.pole_prefer_sample) and emit a pole_mode='SAMPLE'
        # pair here — the downstream sampling math is still in place.

        # FORMULA mode: derive target FK chain (root, mid) bones via
        # FK pair list lookup, run the perpendicular-to-chain formula
        # on those at bake time.
        if src_pbs is None or pole_src not in src_pbs:
            _dbg(f"    skipped pole: source bone {pole_src!r} not on source rig")
            skipped.append(
                f"{_chain_label_str(row)}: pole source '{pole_src}' "
                f"not on source rig — skipping pole pair")
            continue
        mid_pb_src = src_pbs[pole_src]
        if mid_pb_src.parent is None:
            _dbg(f"    skipped pole: {pole_src!r} has no parent")
            skipped.append(
                f"{_chain_label_str(row)}: pole source '{pole_src}' "
                f"has no parent — can't form a chain — skipping pole pair")
            continue
        pole_root_src = mid_pb_src.parent.name
        _dbg(f"    pole mode: FORMULA. chain on source: root={pole_root_src!r} "
             f"mid={pole_src!r}")
        target_chain_mid = _find_fk_pair_target(fk_pairs, pole_src)
        target_chain_root = _find_fk_pair_target(fk_pairs, pole_root_src)
        _dbg(f"    FK pair lookups: mid {pole_src!r}->{target_chain_mid!r}, "
             f"root {pole_root_src!r}->{target_chain_root!r}")
        if not target_chain_mid or not target_chain_root:
            _dbg(f"    skipped pole: missing FK pair entries")
            skipped.append(
                f"{_chain_label_str(row)}: FK retarget pair map missing "
                f"entries for '{pole_src}' or '{pole_root_src}' "
                f"— skipping pole pair")
            continue
        if target_chain_mid not in tgt_pbs or target_chain_root not in tgt_pbs:
            _dbg(f"    skipped pole: derived chain bones not on target rig")
            skipped.append(
                f"{_chain_label_str(row)}: derived target chain bones "
                f"'{target_chain_root}' / '{target_chain_mid}' "
                f"not on target rig — skipping pole pair")
            continue
        pairs.append(types.SimpleNamespace(
            kind='POLE',
            pole_mode='FORMULA',
            target=row.pole_control,
            target_chain_root=target_chain_root,
            target_chain_mid=target_chain_mid,
            influence=1.0,
        ))
        _dbg(f"    + POLE pair (FORMULA): target={row.pole_control!r} "
             f"chain root={target_chain_root!r} mid={target_chain_mid!r}")

    _dbg(f"_build_ik_pairs: produced {len(pairs)} pair(s), {len(skipped)} skipped")
    return pairs, skipped


def _flip_pole_vector(rig, ik_chains_collection):
    """Set `pole_vector` to "Pole" mode on every ARM/LEG chain whose row
    is being baked. Rigify hosts this property on a sibling `*_parent`
    bone (derived from the chain-root FK name). Set-only, never key —
    animators add their own pole_vector keys if they want mid-action
    switching. No-op for chains without a pole_control configured.

    Mirrors the behavior of the pre-rewrite path: gated to chains we
    actually bake, type-matched IDProperty assignment so newer bool /
    older float Rigify variants both work.
    """
    if rig is None or rig.type != 'ARMATURE':
        return
    pbs = rig.pose.bones
    chains = _discover_ik_chains(rig)
    by_owner = {row.owner: row for row in ik_chains_collection}
    for c in chains:
        if c.get('skip_reason'):
            continue
        kind = _classify_limb_kind(c)
        if kind not in ('ARM', 'LEG'):
            continue
        owner_name = c['owner_ik'].name
        row = by_owner.get(owner_name)
        if row is None or not row.pole_control:
            continue
        # Find the pole_vector host bone. Rigify pattern first, fall
        # back to a small parent-walk over candidate bones (subtarget,
        # owner, pole_control).
        candidates = []
        seen = set()
        def _add(pb, depth=0):
            if pb is None or pb.name in seen:
                return
            seen.add(pb.name)
            candidates.append(pb)
            p = pb.parent
            d = 0
            while p is not None and d < depth:
                if p.name not in seen:
                    seen.add(p.name)
                    candidates.append(p)
                p = p.parent
                d += 1
        fk_chain = c.get('fk_chain') or []
        if fk_chain:
            root_fk = fk_chain[-1]
            prop_name = _derive_rigify_properties_bone_name(root_fk.name)
            if prop_name and prop_name in pbs:
                _add(pbs[prop_name], depth=0)
            _add(root_fk, depth=4)
        _add(c.get('subtarget'), depth=6)
        _add(c.get('owner_ik'), depth=6)
        if row.pole_control in pbs:
            _add(pbs[row.pole_control], depth=0)
        for pb in candidates:
            if 'pole_vector' not in pb.keys():
                continue
            try:
                current = pb['pole_vector']
            except Exception:
                continue
            if isinstance(current, bool):
                new_val = True
            elif isinstance(current, int):
                new_val = 1
            else:
                new_val = 1.0
            try:
                pb['pole_vector'] = new_val
            except Exception:
                continue
            print(f"[BlendCap] pole_vector flip: {pb.name} = {new_val!r} "
                  f"(chain owner '{owner_name}')")
            break


class BLENDCAP_OT_convert_fk_to_ik(bpy.types.Operator):
    """Bake the source rig's FK animation onto the target rig's IK
    controls via the same retarget engine used for FK pairs.

    For each ARM/LEG row in the FK->IK Mapping panel, two pairs are sent
    to the bake:
        end-effector  (LOC_ROT) — source hand/foot -> target ik_control
        pole          (LOC)     — FORMULA mode: the pole position is
            derived at bake time from the target FK chain (root + mid)
            via the perpendicular-to-chain formula. No helper bones are
            added to either rig; both are left in their original state.

    Source bone names default to BlendCap BVH conventions; advanced
    overrides on the panel let users redirect to other source rigs.
    """
    bl_idname = "blendcap.convert_fk_to_ik"
    bl_label = "Convert FK → IK (Bake IK Controls)"
    bl_description = (
        "Convert the source rig's FK animation onto the target rig's IK "
        "controls using the FK → IK mapping rows above."
    )
    bl_options = {'REGISTER', 'UNDO'}

    # ----- modal state -----
    _timer = None
    _coroutine = None
    _wm = None
    _scene = None
    _src = None
    _tgt = None
    _pairs = None
    # Snapshot of pre-resolution FK pair source/target names; see
    # retarget._resolve_pair_names_for_bake. Restored in _finalize.
    _pair_name_snapshot = None
    _frame_start = 0
    _frame_end = 0
    _prev_frame = 0
    _prev_mode = None
    _prev_active = None
    _frames_done = 0
    _n_frames = 0

    # Soft per-tick budget. Same convention as operators_retarget.
    TICK_BUDGET = 0.02

    @classmethod
    def poll(cls, context):
        # Mirrors apply_retarget's poll via _validate_rig_slot so the
        # button greys for the deleted-rig / excluded-from-view-layer
        # cases too, not just "field empty / wrong type". The target
        # falls back to context.active_object for the rare scripted
        # path that omits the field — _validate_rig_slot then checks
        # whatever object we resolved.
        from .operators_retarget import _validate_rig_slot
        scene = context.scene
        src = scene.blendcap_retarget_source
        tgt = scene.blendcap_retarget_target or context.active_object
        if _validate_rig_slot(scene, src, "Source") is not None:
            return False
        if _validate_rig_slot(scene, tgt, "Target") is not None:
            return False
        return True

    def invoke(self, context, event):
        try:
            ok, err = self._prepare(context)
        except Exception as e:
            self.report({'ERROR'}, f"FK → IK setup failed: {e}")
            self._finalize(context)
            return {'CANCELLED'}
        if not ok:
            if err:
                self.report({'ERROR'}, err)
            self._finalize(context)
            return {'CANCELLED'}
        wm = context.window_manager
        self._wm = wm
        self._timer = wm.event_timer_add(0.03, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        try:
            ok, err = self._prepare(context)
        except Exception as e:
            self.report({'ERROR'}, f"FK → IK setup failed: {e}")
            self._finalize(context)
            return {'CANCELLED'}
        if not ok:
            if err:
                self.report({'ERROR'}, err)
            self._finalize(context)
            return {'CANCELLED'}
        try:
            for _ in self._coroutine:
                pass
        except Exception as e:
            self.report({'ERROR'}, f"FK → IK failed: {e}")
            self._finalize(context)
            return {'CANCELLED'}
        return self._report_and_finalize(context)

    def modal(self, context, event):
        global _FK_IK_CANCEL_REQUESTED
        if event.type == 'ESC' or _FK_IK_CANCEL_REQUESTED:
            self.report({'WARNING'}, "FK → IK cancelled")
            self._finalize(context)
            return {'CANCELLED'}
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        import time
        start = time.monotonic()
        try:
            while True:
                yielded = next(self._coroutine)
                if isinstance(yielded, int):
                    self._frames_done += 1
                if time.monotonic() - start >= self.TICK_BUDGET:
                    break
        except StopIteration:
            return self._report_and_finalize(context)
        except Exception as e:
            self.report({'ERROR'}, f"FK → IK failed: {e}")
            self._finalize(context)
            return {'CANCELLED'}

        if self._frames_done < self._n_frames:
            pct = 70.0 * self._frames_done / max(self._n_frames, 1)
            short = (f"Frame {self._frames_done}/{self._n_frames}"
                     "  (ESC to cancel)")
        else:
            pct = 85.0
            short = "Writing keyframes…  (ESC to cancel)"
        self._tick_ui(pct, short)
        return {'PASS_THROUGH'}

    # ----- internals -----

    def _tick_ui(self, pct, status):
        retarget_progress_state['progress'] = min(99.9, max(0.0, pct))
        retarget_progress_state['status_text'] = status
        retarget_progress_state['frames_done'] = self._frames_done
        try:
            bpy.context.workspace.status_text_set(
                format_workspace_status("BlendCap FK → IK", pct, status))
        except Exception:
            pass
        try:
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        for region in area.regions:
                            if region.type == 'UI':
                                region.tag_redraw()
        except Exception:
            pass

    def _prepare(self, context):
        scene = context.scene
        self._scene = scene

        _dbg("=" * 60)
        _dbg("Convert FK->IK: _prepare starting")

        src = scene.blendcap_retarget_source
        tgt = scene.blendcap_retarget_target or context.active_object
        _dbg(f"  scene.blendcap_retarget_source = "
             f"{src.name if src else None!r} "
             f"type={src.type if src else None!r}")
        _dbg(f"  scene.blendcap_retarget_target = "
             f"{(scene.blendcap_retarget_target.name if scene.blendcap_retarget_target else None)!r}")
        _dbg(f"  context.active_object = "
             f"{context.active_object.name if context.active_object else None!r}")
        _dbg(f"  resolved tgt = {tgt.name if tgt else None!r} "
             f"type={tgt.type if tgt else None!r}")

        from .operators_retarget import _validate_rig_slot
        err = _validate_rig_slot(scene, src, "Source")
        if err is not None:
            return False, err
        err = _validate_rig_slot(scene, tgt, "Target")
        if err is not None:
            return False, err

        # Lock the 4 rows + refresh each owner from the rig, but do NOT
        # auto-fill blank ik_control / pole_control (fill_controls=False):
        # the bake honors exactly what's in the FK->IK panel. A chain with
        # a blank ik_control is skipped (_build_ik_pairs reports it) rather
        # than silently discovering a control the user didn't choose.
        sync_ik_chain_collection(tgt, scene.blendcap_retarget_ik_chains,
                                 fill_controls=False)

        # Resolve FK pair source/target names to actual rig bones
        # ("RightLeg" -> "mixamorig:RightLeg") so _build_ik_pairs can
        # match pair.source against the custom IK source bone names
        # (which the eyedropper fills with the literal prefixed form).
        # Snapshot is restored in _finalize.
        from .retarget import _resolve_pair_names_for_bake
        self._pair_name_snapshot = _resolve_pair_names_for_bake(scene)

        pairs, skipped = _build_ik_pairs(scene, src, tgt)
        for s in skipped:
            self.report({'WARNING'}, s)
        if not pairs:
            _dbg("  RESULT: no pairs built — bake will not run")
            return False, ("No IK pairs to bake — fill in the FK → IK "
                           "Mapping panel (ik_control / pole_control "
                           "for each chain). The console (Window > "
                           "Toggle System Console) shows what each "
                           "row was missing.")
        _dbg(f"  RESULT: built {len(pairs)} pair(s), proceeding to bake")

        # Pole-vector flip (Rigify carve-out, allowed by Protocols).
        _flip_pole_vector(tgt, scene.blendcap_retarget_ik_chains)

        # CloudRig carve-out (same Protocols carve-out as the FK
        # retarget's switch flip): the convert reads the TARGET's FK
        # chain, and CloudRig's FK controls are constraint-slaved to
        # IK mechanism bones (real IK solver — unsupported in the
        # constraint sim) whenever Properties["ik_<side>_<limb>"] is
        # 1.0. Converting in that state would read a rest-pose chain.
        # Set the switches to FK (0.0) before the bake's constraint
        # scan; the fast engine settles drivers (update_tag +
        # frame_set + view_layer.update) before reading influences.
        from .operators_retarget import _force_cloudrig_fk
        _force_cloudrig_fk(tgt)

        # Frame range from the source rig's action, falling back to the
        # scene range. Same convention as the FK retarget bake.
        action = src.animation_data.action if src.animation_data else None
        if action is not None:
            try:
                astart, aend = action.frame_range
                frame_start = int(astart)
                frame_end = int(aend)
            except Exception:
                frame_start = scene.frame_start
                frame_end = scene.frame_end
        else:
            frame_start = scene.frame_start
            frame_end = scene.frame_end
        if frame_end < frame_start:
            return False, "Scene frame_end is before frame_start"

        prev_frame = scene.frame_current
        prev_mode = context.mode
        prev_active = context.view_layer.objects.active

        # Activate target rig in POSE mode so basis writes land cleanly
        # and the depsgraph evaluates the right armature each tick.
        if context.mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except Exception:
                pass
        try:
            bpy.ops.object.select_all(action='DESELECT')
        except Exception:
            pass
        context.view_layer.objects.active = tgt
        try:
            tgt.select_set(True)
        except Exception:
            pass
        try:
            bpy.ops.object.mode_set(mode='POSE')
        except Exception:
            pass

        n_frames = max(1, frame_end - frame_start + 1)
        self._src = src
        self._tgt = tgt
        self._pairs = pairs
        self._frame_start = frame_start
        self._frame_end = frame_end
        self._prev_frame = prev_frame
        self._prev_mode = prev_mode
        self._prev_active = prev_active
        self._n_frames = n_frames
        self._frames_done = 0

        retarget_progress_state.update({
            'running': True, 'phase': "Converting FK → IK",
            'progress': 0.0, 'status_text': "Starting…",
            'frames_done': 0, 'frames_total': n_frames,
        })
        global _FK_IK_RUNNING, _FK_IK_CANCEL_REQUESTED
        _FK_IK_RUNNING = True
        _FK_IK_CANCEL_REQUESTED = False

        self._coroutine = _bake_world_pairs_iter(
            pairs, src, tgt, frame_start, frame_end)
        return True, None

    def _report_and_finalize(self, context):
        n_pairs = len(self._pairs or [])
        self._finalize(context)
        msg = (f"Baked IK across {n_pairs} pair{'s' if n_pairs != 1 else ''}")
        self.report({'INFO'}, msg)
        return {'FINISHED'}

    def _finalize(self, context):
        """Always-safe cleanup — runs on FINISHED, CANCELLED, and
        exception paths. Restores frame, mode, active object;
        clears progress state."""
        wm = self._wm or context.window_manager
        if self._timer is not None:
            try:
                wm.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None

        if self._scene is not None:
            try:
                self._scene.frame_set(self._prev_frame)
            except Exception:
                pass
        if self._prev_active is not None:
            try:
                context.view_layer.objects.active = self._prev_active
            except Exception:
                pass
        if self._prev_mode and context.mode != self._prev_mode:
            try:
                if self._prev_mode.startswith('EDIT'):
                    bpy.ops.object.mode_set(mode='EDIT')
                elif self._prev_mode == 'OBJECT':
                    bpy.ops.object.mode_set(mode='OBJECT')
                elif self._prev_mode == 'POSE':
                    bpy.ops.object.mode_set(mode='POSE')
            except Exception:
                pass
        try:
            context.workspace.status_text_set(None)
        except Exception:
            pass
        retarget_progress_state.update({
            'running': False, 'phase': "", 'progress': 0.0,
            'status_text': "", 'frames_done': 0, 'frames_total': 0,
        })
        global _FK_IK_RUNNING, _FK_IK_CANCEL_REQUESTED
        _FK_IK_RUNNING = False
        _FK_IK_CANCEL_REQUESTED = False
        # Restore FK pair source/target fields to their stored SHORT
        # form (see retarget._resolve_pair_names_for_bake).
        try:
            from .retarget import _restore_pair_names_after_bake
            _restore_pair_names_after_bake(self._pair_name_snapshot)
        except Exception:
            pass
        self._pair_name_snapshot = None
        try:
            for window in context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        for region in area.regions:
                            if region.type == 'UI':
                                region.tag_redraw()
        except Exception:
            pass
        self._coroutine = None
        self._wm = None


class BLENDCAP_OT_cancel_fk_to_ik(bpy.types.Operator):
    """Set the cancel flag the convert_fk_to_ik modal checks each tick.
    Same effect as ESC. Has no effect on the synchronous execute path
    (e.g. F3 search with EXEC_DEFAULT) because that doesn't run the
    modal handler."""
    bl_idname = "blendcap.cancel_fk_to_ik"
    bl_label = "Cancel FK → IK"
    bl_description = "Stop the current process"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return _FK_IK_RUNNING

    def execute(self, context):
        global _FK_IK_CANCEL_REQUESTED
        _FK_IK_CANCEL_REQUESTED = True
        return {'FINISHED'}
