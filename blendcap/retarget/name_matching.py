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
"""


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

    def _strip_keep_if_collide(name, prefix, bone_set):
        if not prefix or not name.startswith(prefix):
            return name
        stripped = name[len(prefix):]
        if stripped and stripped in bone_set and name in bone_set:
            return name
        return stripped if stripped else name

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
