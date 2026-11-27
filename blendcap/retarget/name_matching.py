"""Bone name matching utilities for auto-pair suggestion, pair-field
smart-stripping, and bake-time bone resolution.

The Source/Target Prefix scene fields are load-bearing: pair source/target
fields store the SHORT name (e.g. "RightLeg") and the prefix prepends at
bake time to find the actual rig bone ("mixamorig:RightLeg"). Editing the
prefix re-targets every pair to a re-namespaced rig in one keystroke.

Three primitives in this module:

  _smart_strip(name, rig, prefix)
      Called from the BlendCapBonePair source/target update callbacks.
      Strips the prefix UNLESS the stripped form ALSO exists on the rig
      as a distinct literal bone — in that case the literal stored name
      wins (the rare collision case where both 'RightLeg' and
      'mixamorig:RightLeg' coexist on one rig).

  _resolve_against_rig(name, rig, prefix)
      Called at bake time. Tries: exact match -> prefix + name ->
      unresolved (returns name as-is so the caller's existence check
      catches it). Exact wins over prefixed when both exist — matches
      the smart-strip rule (an explicit pin stays explicit).

  _auto_match_pairs(...)
      Returns (source, target) tuples in their STRIPPED form, ready
      to drop into the pairs table without further processing.

  _auto_match_pairs_spatial(...)
      Location-based pass for the bones name matching can't pair
      (across naming conventions that is most of them): matches by
      normalized rest-space head position, with bone direction,
      length, and name similarity breaking ties between colocated
      candidates, side markers / X-sign keeping left and right apart,
      hidden bones excluded, one target per source. Same STRIPPED
      output form as _auto_match_pairs.
"""
import difflib
import math


def _strip_namespace(name, prefix):
    """Unconditional prefix strip — name without prefix if name starts
    with prefix, else name unchanged. Used as a low-level helper by
    _smart_strip and _auto_match_pairs."""
    if prefix and name.startswith(prefix):
        return name[len(prefix):]
    return name


def _smart_strip(name, rig, prefix):
    """Conditional prefix strip used by the pair-field update callbacks.

    Returns the stripped name only when:
      - `prefix` is non-empty AND
      - `name` starts with `prefix` AND
      - the stripped form is NOT already a literal bone on the rig
        (avoids destroying intent in the collision case where both
        forms exist as distinct bones).

    If `rig` is None or the stripped form doesn't conflict, strip.
    If the collision case is detected, return `name` unchanged so the
    user's explicit pick survives.
    """
    if not name or not prefix:
        return name
    if not name.startswith(prefix):
        return name
    stripped = name[len(prefix):]
    if not stripped:
        return name
    # Collision check: only suppress the strip when BOTH the prefixed
    # form AND the stripped form exist as distinct rig bones. If the
    # stripped form exists alone, the prefixed form is just an alias —
    # safe to strip.
    if rig is not None and rig.type == 'ARMATURE':
        bones = rig.data.bones
        if stripped in bones and name in bones:
            return name
    return stripped


def _resolve_against_rig(name, rig, prefix):
    """Bake-time name resolution: exact match -> prefix + name -> name.

    Used both when applying a preset (rehydrating stripped names against
    a real rig) and just before bake (resolving each pair's stored
    short name to the actual rig bone). Returns the input `name` unchanged
    when nothing resolves — the caller's `name in pose.bones` check then
    treats the pair as unresolved, same as today.
    """
    if not name or not rig or rig.type != 'ARMATURE':
        return name
    bones = rig.data.bones
    if name in bones:
        return name
    if prefix:
        prefixed = prefix + name
        if prefixed in bones:
            return prefixed
    return name


def _detect_rig_prefix(rig, short_names, fallback_prefix=""):
    """Detect the namespace prefix a rig actually uses for a set of
    expected SHORT bone names, returning the prefix to prepend at resolve
    time.

    Motivation: Mixamo re-namespaces colliding imports as 'mixamorig1:',
    'mixamorig2:', ... instead of the canonical 'mixamorig:'. A preset
    stores short names ('Hips') plus a fixed prefix ('mixamorig:'), so on
    a 'mixamorig1:' rig nothing resolves. This reads the real prefix off
    the rig's own bone names so the preset adapts with no user edit.

    `fallback_prefix` (the preset's stored prefix) is returned unchanged
    when it already resolves every expected name, when no rig is set, or
    when nothing better is found — so a correct preset is never disturbed.
    A detected prefix is adopted only if it resolves STRICTLY MORE names
    than the fallback, and a candidate must end in a separator
    (':', '_', '.', '-') so a bare suffix match ('Arm' inside 'LeftArm')
    can't masquerade as a namespace.
    """
    if rig is None or getattr(rig, 'type', None) != 'ARMATURE':
        return fallback_prefix
    names = [n for n in (short_names or []) if n]
    if not names:
        return fallback_prefix
    bone_names = set(rig.data.bones.keys())

    def _resolved_count(prefix):
        c = 0
        for s in names:
            if s in bone_names or (prefix and (prefix + s) in bone_names):
                c += 1
        return c

    base = _resolved_count(fallback_prefix)
    if base == len(names):
        return fallback_prefix

    # Tally separator-terminated prefixes implied by suffix matches.
    # Longest expected name first so 'LeftArm' wins over 'Arm' on a bone
    # that ends with both; one prefix vote per bone.
    names_by_len = sorted(set(names), key=len, reverse=True)
    tally = {}
    for b in bone_names:
        for s in names_by_len:
            if len(b) > len(s) and b.endswith(s):
                pre = b[:-len(s)]
                if pre and pre[-1] in ':_.-':
                    tally[pre] = tally.get(pre, 0) + 1
                    break
    if not tally:
        return fallback_prefix
    best_pre = max(tally, key=lambda k: tally[k])
    if tally[best_pre] > base:
        return best_pre
    return fallback_prefix


def _present_under_any_namespace(name, bone_names):
    """True if `name` matches a bone in `bone_names` exactly, OR a bone
    that ends with <separator><name> (i.e. the bone exists on the rig
    under some namespace prefix).

    Lets the preset loader tell a fixable prefix / namespace mismatch
    (the bone IS on the rig, just prefixed — keep the row so auto-detect
    or a prefix edit resolves it) apart from a bone the rig genuinely
    LACKS (face / finger bones on a body-only rig — drop the row so it
    doesn't clutter the table with a mapping no prefix could satisfy).

    The separator requirement (':', '_', '.', '-' before `name`) keeps a
    bare substring ('Arm' inside 'UpperArm') from reading as a namespace
    match. `bone_names` is a precomputed set for the O(1) exact check."""
    if not name:
        return True
    if name in bone_names:
        return True
    nl = len(name)
    for b in bone_names:
        if len(b) > nl and b.endswith(name) and b[-nl - 1] in ':_.-':
            return True
    return False


def _normalize_bone_name(name):
    """Lowercase for case-insensitive comparison. Namespace stripping is
    the prefix fields' job — done explicitly via `_strip_namespace`
    before this call."""
    return name.lower()


def _strip_keep_if_collide(name, prefix, bone_set):
    """Strip `prefix` from `name` unless both the prefixed and stripped
    forms exist as distinct bones (same collision rule as _smart_strip,
    without needing the rig object)."""
    if not prefix or not name.startswith(prefix):
        return name
    stripped = name[len(prefix):]
    if stripped and stripped in bone_set and name in bone_set:
        return name
    return stripped if stripped else name


def _auto_match_pairs(source_rig, target_rig, source_strip="", target_strip=""):
    """Return list of (source_bone, target_bone) pairs that share a normalized
    name, in their STRIPPED form (ready to store directly in pairs).

    Strip behavior matches _smart_strip: the prefix is removed unless the
    bone's prefixed AND stripped forms both exist on the same rig, in which
    case the literal name is kept. That keeps the table consistent with
    what the per-field update callbacks would produce on manual entry.
    """
    src_names = [b.name for b in source_rig.data.bones]
    tgt_names = [b.name for b in target_rig.data.bones]
    src_bone_set = set(src_names)
    tgt_bone_set = set(tgt_names)

    tgt_lookup = {}
    for t in tgt_names:
        key = _normalize_bone_name(_strip_namespace(t, target_strip))
        tgt_lookup.setdefault(key, t)
    suggested = []
    for s in src_names:
        key = _normalize_bone_name(_strip_namespace(s, source_strip))
        t = tgt_lookup.get(key)
        if t:
            suggested.append((
                _strip_keep_if_collide(s, source_strip, src_bone_set),
                _strip_keep_if_collide(t, target_strip, tgt_bone_set),
            ))
    return suggested


def _bone_side(name):
    """'L', 'R', or '' from a bone name's side markers: separator-
    delimited tokens ('l', 'r', 'lft', 'rgt', or anything starting/
    ending with 'left'/'right', any case — covers '.l', '_R',
    'LeftArm', 'left_eye'), or a trailing capital L/R glued after a
    lowercase run ('HandL'). Returns '' for center / unmarked bones."""
    n = name.lower()
    tokens = ''.join(
        (c if c not in ':_.- ' else ' ') for c in n).split()
    for t in tokens:
        if (t == 'l' or t == 'lft'
                or t.startswith('left') or t.endswith('left')):
            return 'L'
        if (t == 'r' or t == 'rgt'
                or t.startswith('right') or t.endswith('right')):
            return 'R'
    # Front-glued side letter (DAZ style): 'lThigh', 'rShldrBend',
    # 'RFoot', 'LArm' — a case-preserved part starting with l/L/r/R
    # followed by an uppercase letter, third character lowercase so
    # all-caps parts ('ROOT', 'LOD1') don't read as sides.
    parts = ''.join(
        (c if c not in ':_.- ' else ' ') for c in name).split()
    for p in parts:
        if (p[0] in 'lLrR' and len(p) >= 2 and p[1].isupper()
                and (len(p) < 3 or not p[2].isupper())):
            return p[0].upper()
    if len(name) >= 2 and name[-1] in 'LR' and name[-2].islower():
        return name[-1]
    return ''


# Words marking non-FK helper controls. BlendCap's retarget bakes FK,
# so these are the wrong surface for an unmarked source bone.
_ROLE_HELPER_WORDS = ('pole', 'twist', 'bend', 'tweak', 'stretch',
                      'roll', 'pin', 'parent')


def _bone_role(name):
    """Control-role marker: 'fk', 'ik', 'helper', or ''. Separates the
    controls a rig stacks on one joint (c_hand_fk vs c_hand_ik are
    colocated and identically sized on ARP — only the name says which
    one drives FK)."""
    n = name.lower()
    if 'fk' in n:
        return 'fk'
    tokens = ''.join(
        (c if c not in ':_.- ' else ' ') for c in n).split()
    if 'ik' in tokens or n.endswith('ik'):
        return 'ik'
    # WHOLE tokens only: 'pin' as a substring classified every
    # 'spine' control as a helper and torpedoed all the spine pairs.
    for w in _ROLE_HELPER_WORDS:
        if w in tokens:
            return 'helper'
    return ''


# Body-part keyword groups for semantic classification of BOTH rigs'
# bones (the Auto-Rig Pro remap idea of keyword-driven matching,
# generalized: ARP only classifies the source because its target names
# are its own; here both sides are unknown, so both classify into the
# same category space and the categories are compared).
#
# Substring match against the lowered stripped name, FIRST HIT WINS —
# so containments are ordered out: 'torso' (a pelvis control on
# Rigify/CloudRig) before 'spine' catches 'TORSO-Spine'; 'forehead'
# before 'head'; 'brow'/'lid' before 'eye' ('eyebrow', 'eyelid');
# 'eye' before 'ring'/'ball' ('EyeRing', 'eyeball'); fingers before
# 'hand' ('LeftHandIndex1'); 'forearm' before 'arm' and before 'ear';
# 'thigh'/'upleg' before 'leg'.
_SPATIAL_CATEGORIES = (
    ('hips',     ('hip', 'pelvis', 'torso', 'cog', 'root_master',
                  'waist', 'lowerbody', 'lower_body')),
    ('breast',   ('breast', 'pec')),
    ('spine',    ('spine', 'chest', 'abdomen', 'stomach', 'sternum',
                  'upperbody', 'upper_body')),
    ('neck',     ('neck',)),
    ('forehead', ('forehead', 'temple')),
    ('jaw',      ('jaw', 'chin', 'mandible')),
    ('tongue',   ('tongue', 'tong')),
    ('teeth',    ('teeth', 'tooth')),
    ('brow',     ('brow',)),
    ('lid',      ('lid',)),
    ('nose',     ('nose', 'nostril')),
    ('lip',      ('lip', 'mouth')),
    ('cheek',    ('cheek',)),
    ('eye',      ('eye',)),
    ('head',     ('head', 'skull')),
    ('shoulder', ('shoulder', 'clavicle', 'collar', 'scapula')),
    ('thumb',    ('thumb',)),
    ('index',    ('index',)),
    ('middle',   ('middle',)),
    ('ring',     ('ring',)),
    ('pinky',    ('pinky', 'pinkie', 'little')),
    ('finger',   ('finger',)),
    ('forearm',  ('forearm', 'lowerarm', 'lower_arm', 'elbow',
                  'ulna')),
    ('arm',      ('upperarm', 'upper_arm', 'uparm', 'humerus', 'arm')),
    ('hand',     ('wrist', 'palm', 'hand')),
    ('thigh',    ('thigh', 'upleg', 'upperleg', 'upper_leg', 'femur')),
    ('shin',     ('shin', 'calf', 'knee', 'lowerleg', 'lower_leg',
                  'tibia', 'leg')),
    ('foot',     ('foot', 'ankle', 'heel')),
    ('toe',      ('toe', 'ball')),
    ('ear',      ('ear',)),
    ('tail',     ('tail',)),
)


def _bone_category(name):
    """Body-part category for a (stripped) bone name, or '' when no
    keyword hits. See _SPATIAL_CATEGORIES for the ordering rules."""
    n = name.lower()
    for cat, kws in _SPATIAL_CATEGORIES:
        for kw in kws:
            if kw in n:
                if kw == 'hand' and 'handle' in n:
                    continue  # STR handles etc. are not hands
                return cat
    return ''


def _spatial_features(rig, strip):
    """Normalized rest-space features per candidate bone, keyed by the
    bone's REAL name.

    Candidates are the rig's VISIBLE bones only (not bone.hide, and at
    least one of its bone collections effectively visible) — control
    rigs bury hundreds of MCH/DEF/ORG helpers in hidden collections,
    and those colocate with the controls the user actually poses, so
    matching against them would suggest bones the user can't even see.
    Bones in no collection count as visible; if the filter leaves
    fewer than two bones the whole rig is used.

    Normalization makes features comparable across differently sized,
    differently placed rigs: heads are offset by the candidate-set XY
    centroid and floor (min Z), then everything divides by the
    vertical extent — so a position is "where on the body", not where
    in the world. Each feature dict carries:
        head  — normalized head position (tuple)
        dir   — unit head->tail direction, world frame
        len   — bone length / rig height
        side  — 'L' / 'R' / '' from the stripped name
        cat   — body-part category from _bone_category ('' if none)
        role  — control role from _bone_role ('fk'/'ik'/'helper'/'')
        ndesc — descendant bone count (hierarchy role: the pelvis
                carries the whole skeleton, a helper chain is a leaf)
        norm  — stripped lowercase name (for similarity scoring)
    """
    bones = rig.data.bones
    picked = []
    for b in bones:
        if b.hide:
            continue
        cols = getattr(b, 'collections', None)
        if cols is not None and len(cols) and not any(
                getattr(c, 'is_visible_effectively', c.is_visible)
                for c in cols):
            continue
        picked.append(b)
    if len(picked) < 2:
        picked = list(bones)
    mw = rig.matrix_world
    raw = []
    for b in picked:
        h = mw @ b.head_local
        t = mw @ b.tail_local
        raw.append((b.name, (h.x, h.y, h.z), (t.x, t.y, t.z),
                    len(b.children_recursive)))
    zs = [p[1][2] for p in raw] + [p[2][2] for p in raw]
    z0 = min(zs)
    height = max(zs) - z0
    if height < 1e-6:
        height = 1.0
    x0 = sum(p[1][0] for p in raw) / len(raw)
    y0 = sum(p[1][1] for p in raw) / len(raw)
    feats = {}
    for name, h, t, ndesc in raw:
        d = (t[0] - h[0], t[1] - h[1], t[2] - h[2])
        ln = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        ndir = ((d[0] / ln, d[1] / ln, d[2] / ln) if ln > 1e-9
                else (0.0, 0.0, 1.0))
        short = _strip_namespace(name, strip)
        feats[name] = {
            'head': ((h[0] - x0) / height, (h[1] - y0) / height,
                     (h[2] - z0) / height),
            'dir': ndir,
            'len': ln / height,
            'side': _bone_side(short),
            'cat': _bone_category(short),
            'role': _bone_role(short),
            'ndesc': ndesc,
            'norm': short.lower(),
        }
    return feats


# Location-matching tuning. Distances are in body heights (0.10 = 10%
# of the rig's vertical extent, ~17cm on a human).
_SPATIAL_POS_GATE = 0.10    # candidates farther than this never pair
_SPATIAL_SCORE_GATE = 0.12  # combined-score acceptance cutoff
_SPATIAL_W_DIR = 0.15       # weight: bone direction disagreement
# Length mismatch is scored on the LOG RATIO of the two lengths, not
# the absolute difference: a thigh and a tiny helper bone parked near
# the crotch differ by "only" ~0.2 in absolute normalized units, so a
# small position edge could out-vote the size mismatch and pair the
# thigh with the helper. Ratio scoring makes a 7x+ size difference
# cost 0.12 (the whole score gate) no matter how close the heads sit,
# while small-vs-small (finger to finger) stays cheap.
_SPATIAL_W_LEN = 0.06       # weight: |log(len ratio)|, capped below
_SPATIAL_LEN_CAP = 2.0      # cap: e^2 ≈ 7.4x size difference and up
_SPATIAL_LEN_MIN = 1e-3     # ratio floor for zero/leaf-length bones
# Hierarchy role, scored like length: |log of the descendant-count
# ratio| (+1 so leaves don't divide by zero), capped. A whole-body
# carrier (Hips, a pelvis master) can't pair with a leaf helper chain
# parked nearby — the cap alone nearly spends the score gate. Counts
# CLAMP first: the signal is leaf-vs-carrier, not proportional — a
# 60-bone source Hips against a control rig's 900-descendant pelvis
# master is the RIGHT pair, and the raw ratio was rejecting it.
_SPATIAL_W_HIER = 0.05
_SPATIAL_HIER_CAP = 2.0
_SPATIAL_HIER_CLAMP = 20
# Body-part categories (see _SPATIAL_CATEGORIES): agreeing categories
# earn a bonus, CONFLICTING ones a penalty big enough that only a
# perfect position + name overlap could survive it.
_SPATIAL_W_CAT = 0.05
_SPATIAL_CAT_PENALTY = 0.08
# Control-role prior (see _bone_role). BlendCap retargets FK-only, so
# an unmarked source bone prefers FK-flavored targets and shuns IK /
# helper controls (poles, twists, tweaks, parent gizmos). Matching
# explicit roles (ik->ik) earn the same bonus; explicitly conflicting
# roles pay the penalty.
_SPATIAL_W_ROLE = 0.03
_SPATIAL_ROLE_PENALTY = 0.05
_SPATIAL_W_NAME = 0.08      # bonus: name similarity (subtracted)
_SPATIAL_SIDE_X = 0.02      # |x| beyond which position implies a side
# Anchored round (see _match_anchored): absolute position is replaced
# by the difference in DISTANCE FROM THE ANCHOR pair — pose-invariant,
# so a T-pose source hand still matches an A-pose rig's hand control.
_ANCHOR_REL_GATE = 0.06     # anchor-distance difference limit
_ANCHOR_SCORE_GATE = 0.10   # combined-score acceptance cutoff


def _match_features(src_feats, tgt_feats, used_sources=(), used_targets=(),
                    diag=None):
    """Greedy one-to-one matching over _spatial_features dicts. Pure
    (no bpy) so it unit-tests offline. Returns (src_name, tgt_name)
    REAL-name tuples, best matches first.

    A candidate pair must agree on side (explicit name markers must
    not conflict; nor may clearly off-center X positions point to
    opposite sides) and sit within _SPATIAL_POS_GATE of each other.
    Score = position distance + weighted direction / length-ratio /
    hierarchy-role differences + the body-part category term - a
    name-similarity bonus. Direction and length separate colocated
    controls (FK vs IK vs tweak stacked on one joint); the hierarchy
    term keeps whole-body carriers off leaf helper chains parked
    nearby; the category term makes keyword-classified body parts
    prefer their own kind and shun conflicting ones. Exclusion sets
    are checked against the features' 'norm'/stripped identity via
    the separate `used_*` name sets (already-stripped table names)."""
    def _note(sn, d, tn, reason):
        # Keep the NEAREST target's fate per source bone, so a source
        # that ends up unmatched can report why (diag is optional).
        if diag is not None:
            cur = diag.get(sn)
            if cur is None or d < cur[0]:
                diag[sn] = (d, tn, reason)

    cands = []
    for sn, sf in src_feats.items():
        if sn in used_sources:
            continue
        sh, sd = sf['head'], sf['dir']
        for tn, tf in tgt_feats.items():
            if tn in used_targets:
                continue
            th = tf['head']
            d = math.dist(sh, th)
            if sf['side'] and tf['side'] and sf['side'] != tf['side']:
                _note(sn, d, tn, "left/right conflict")
                continue
            if (abs(sh[0]) > _SPATIAL_SIDE_X
                    and abs(th[0]) > _SPATIAL_SIDE_X
                    and (sh[0] > 0.0) != (th[0] > 0.0)):
                _note(sn, d, tn, "left/right conflict (position)")
                continue
            if d > _SPATIAL_POS_GATE:
                _note(sn, d, tn, f"too far apart ({d:.3f} of body "
                                 f"height, limit {_SPATIAL_POS_GATE})")
                continue
            td = tf['dir']
            ddir = (1.0 - (sd[0] * td[0] + sd[1] * td[1]
                           + sd[2] * td[2])) * 0.5
            ls = max(sf['len'], _SPATIAL_LEN_MIN)
            lt = max(tf['len'], _SPATIAL_LEN_MIN)
            dlen = min(_SPATIAL_LEN_CAP, abs(math.log(ls / lt)))
            # Hierarchy applies ONLY when a side has no category: its
            # job is rejecting anonymous helper chains. Control rigs
            # parent controls to mech bones, so correct, semantically
            # identified targets often carry few descendants — scoring
            # them by hierarchy punished every correct pair on ARP.
            if sf['cat'] and tf['cat']:
                dhier = 0.0
            else:
                dhier = min(_SPATIAL_HIER_CAP,
                            abs(math.log(
                                (1 + min(sf['ndesc'],
                                         _SPATIAL_HIER_CLAMP))
                                / (1 + min(tf['ndesc'],
                                           _SPATIAL_HIER_CLAMP)))))
            if sf['cat'] and tf['cat']:
                cat_term = (-_SPATIAL_W_CAT if sf['cat'] == tf['cat']
                            else _SPATIAL_CAT_PENALTY)
            else:
                cat_term = 0.0
            sr, tr = sf['role'], tf['role']
            if sr == tr:
                role_term = -_SPATIAL_W_ROLE if sr else 0.0
            elif not sr and tr == 'fk':
                role_term = -_SPATIAL_W_ROLE
            elif not sr:  # target is ik / helper, source unmarked
                role_term = _SPATIAL_ROLE_PENALTY
            elif not tr:  # source marked, target plain: neutral
                role_term = 0.0
            else:         # both marked, conflicting (fk vs ik, ...)
                role_term = _SPATIAL_ROLE_PENALTY
            sim = difflib.SequenceMatcher(
                None, sf['norm'], tf['norm']).ratio()
            score = (d + _SPATIAL_W_DIR * ddir + _SPATIAL_W_LEN * dlen
                     + _SPATIAL_W_HIER * dhier + cat_term + role_term
                     - _SPATIAL_W_NAME * sim)
            if score <= _SPATIAL_SCORE_GATE:
                cands.append((score, sn, tn))
                _note(sn, d, tn, "candidate")
            else:
                terms = (f"pos {d:.3f}, dir {_SPATIAL_W_DIR * ddir:.3f}"
                         f", size {_SPATIAL_W_LEN * dlen:.3f}, "
                         f"hierarchy {_SPATIAL_W_HIER * dhier:.3f}, "
                         f"category {cat_term:+.3f}, "
                         f"role {role_term:+.3f}")
                _note(sn, d, tn,
                      f"score {score:.3f} over cutoff "
                      f"{_SPATIAL_SCORE_GATE} ({terms})")
    cands.sort()
    out = []
    taken_s, taken_t = set(), set()
    for _score, sn, tn in cands:
        if sn in taken_s or tn in taken_t:
            continue
        taken_s.add(sn)
        taken_t.add(tn)
        out.append((sn, tn))
    return out


def _parent_map(rig):
    """{bone name: parent bone name or ''} across the WHOLE armature
    (hidden bones included, so ancestor walks can pass through
    filtered-out mech bones)."""
    return {b.name: (b.parent.name if b.parent else '')
            for b in rig.data.bones}


def _match_anchored(src_feats, tgt_feats, src_par, tgt_par,
                    matched, used_s, taken_t):
    """One round of anchor-relative matching for bones round 1 left
    unmatched. Returns new (src, tgt) real-name pairs (may be empty).

    Why: absolute position fails across rest-pose conventions — a
    T-pose source hand sits 10%+ of body height away from an A-pose
    rig's hand control, past any sane position gate. But once the ARM
    pair is known, "how far from the arm" is the same on both rigs
    regardless of pose. So: each unmatched source bone finds its
    nearest MATCHED ancestor (the anchor), candidates are unmatched
    targets descending from the anchor's target, and the position term
    becomes the difference in distance-from-anchor. Direction and
    hierarchy are dropped (pose-dependent / meaningless down here);
    size, category, side, role, and name similarity still apply — with
    per-digit finger categories doing the heavy lifting in the hand.

    Callers loop this until no progress: a hand matched in round N
    anchors its fingers in round N+1."""
    def _has_ancestor(name, anc, par):
        cur = par.get(name, '')
        hops = 0
        while cur and hops < 256:
            if cur == anc:
                return True
            cur = par.get(cur, '')
            hops += 1
        return False

    cands = []
    for sn, sf in src_feats.items():
        if sn in matched or sn in used_s:
            continue
        anc_s = src_par.get(sn, '')
        hops = 0
        while anc_s and anc_s not in matched and hops < 256:
            anc_s = src_par.get(anc_s, '')
            hops += 1
        if not anc_s or anc_s not in matched:
            continue
        anc_t = matched[anc_s]
        if anc_s not in src_feats or anc_t not in tgt_feats:
            continue
        rs = math.dist(sf['head'], src_feats[anc_s]['head'])
        for tn, tf in tgt_feats.items():
            if tn in taken_t:
                continue
            if not _has_ancestor(tn, anc_t, tgt_par):
                continue
            if sf['side'] and tf['side'] and sf['side'] != tf['side']:
                continue
            rt = math.dist(tf['head'], tgt_feats[anc_t]['head'])
            drel = abs(rs - rt)
            if drel > _ANCHOR_REL_GATE:
                continue
            ls = max(sf['len'], _SPATIAL_LEN_MIN)
            lt = max(tf['len'], _SPATIAL_LEN_MIN)
            dlen = min(_SPATIAL_LEN_CAP, abs(math.log(ls / lt)))
            if sf['cat'] and tf['cat']:
                cat_term = (-_SPATIAL_W_CAT
                            if sf['cat'] == tf['cat']
                            else _SPATIAL_CAT_PENALTY)
            else:
                cat_term = 0.0
            sr, tr = sf['role'], tf['role']
            if sr == tr:
                role_term = -_SPATIAL_W_ROLE if sr else 0.0
            elif not sr and tr == 'fk':
                role_term = -_SPATIAL_W_ROLE
            elif not sr:
                role_term = _SPATIAL_ROLE_PENALTY
            elif not tr:
                role_term = 0.0
            else:
                role_term = _SPATIAL_ROLE_PENALTY
            sim = difflib.SequenceMatcher(
                None, sf['norm'], tf['norm']).ratio()
            score = (drel + _SPATIAL_W_LEN * dlen + cat_term
                     + role_term - _SPATIAL_W_NAME * sim)
            if score <= _ANCHOR_SCORE_GATE:
                cands.append((score, sn, tn))
    cands.sort()
    out = []
    taken_s = set()
    for _score, sn, tn in cands:
        if sn in taken_s or tn in taken_t:
            continue
        taken_s.add(sn)
        taken_t.add(tn)
        out.append((sn, tn))
    return out


def _auto_match_pairs_spatial(source_rig, target_rig,
                              source_strip="", target_strip="",
                              used_sources=(), used_targets=(),
                              verbose=False):
    """Location-based auto-match pass. `used_sources` / `used_targets`
    hold STRIPPED names already claimed (by the name pass or by rows
    already in the table) — those bones sit this pass out, on both the
    claimed side and as candidates. Returns STRIPPED (source, target)
    pairs like _auto_match_pairs.

    `verbose` prints one console line per unmatched source bone that
    had a target anywhere near it, naming the nearest target and the
    reason it was rejected — the difference between "auto-match gave
    junk" and knowing which term to fix."""
    src_feats = _spatial_features(source_rig, source_strip)
    tgt_feats = _spatial_features(target_rig, target_strip)
    src_bone_set = set(source_rig.data.bones.keys())
    tgt_bone_set = set(target_rig.data.bones.keys())
    used_s = {n for n, f in src_feats.items()
              if _strip_keep_if_collide(n, source_strip, src_bone_set)
              in set(used_sources)}
    used_t = {n for n, f in tgt_feats.items()
              if _strip_keep_if_collide(n, target_strip, tgt_bone_set)
              in set(used_targets)}
    diag = {} if verbose else None
    matches = _match_features(src_feats, tgt_feats, used_s, used_t,
                              diag=diag)
    # Anchored rounds: bones the absolute-position round couldn't
    # place (rest-pose mismatch: T-pose source vs A-pose rig) match
    # inside their anchor's subtree by distance-from-anchor. Iterate:
    # a hand matched in one round anchors its fingers in the next.
    matched = dict(matches)
    taken_t = {t for _, t in matches} | used_t
    src_par = _parent_map(source_rig)
    tgt_par = _parent_map(target_rig)
    while True:
        new = _match_anchored(src_feats, tgt_feats, src_par, tgt_par,
                              matched, used_s, taken_t)
        if not new:
            break
        for s, t in new:
            matched[s] = t
    if verbose:
        for sn in src_feats:
            if sn in used_s or sn in matched:
                continue
            info = diag.get(sn)
            if info is None or info[0] > _SPATIAL_POS_GATE * 2:
                continue  # nothing anywhere near it - not worth a line
            d, tn, reason = info
            if reason == "candidate":
                reason = ("passed scoring, but the target went to a "
                          "better pair")
            print(f"[BlendCap] auto-match: no match for {sn} - "
                  f"nearest is {tn} at {d:.3f}: {reason}")
    return [(_strip_keep_if_collide(s, source_strip, src_bone_set),
             _strip_keep_if_collide(t, target_strip, tgt_bone_set))
            for s, t in matched.items()]
