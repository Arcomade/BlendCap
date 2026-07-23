"""Bone ordering for the 'Add All Source Bones' table.

Returns bones sorted so the resulting pair list reads top-down while
keeping FK chains grouped together (face / head+neck / arms / spine /
legs / uncategorized). Classification is name-heuristic and handles
SAM3D/BVH, namespaced Camel-case names, and Rigify suffixes.
"""


def _ordered_bones_for_add_all(rig):
    """Return bones ordered for the 'Add All Source Bones' table so the
    result reads top-down while keeping FK chains grouped together:

        1. Face  — feature-grouped top-down (brows → lids/eyes → ears →
                   cheeks → then the nose + mouth + jaw center cluster
                   as one contiguous block). Sided features keep each
                   side's chain whole (all Left brow bones, then all
                   Right); center-cluster features read as
                   center → L → R rows instead.
        2. Head, Neck
        3. Left arm chain  (shoulder → hand → fingers: thumb→pinky)
        4. Right arm chain
        5. Spine chain — top-down (Spine.NNN highest → Hips)
        6. Left leg chain  (thigh → toes)
        7. Right leg chain
        8. Uncategorized — hierarchy order

    Classification is name-heuristic and handles SAM3D/BVH, namespaced
    Camel-case names, and Rigify (suffix .L/.R, underscored). A bone
    that doesn't match any pattern falls into bucket 8; output is less
    intuitive but not broken.
    """
    bones = list(rig.data.bones)
    if not bones:
        return []
    mw = rig.matrix_world
    def wz(b):
        return (mw @ b.head_local).z

    def norm(name):
        n = name.lower()
        if ':' in n:
            n = n.split(':', 1)[1]   # drop 'mixamorig:' style namespaces
        return n

    def side(normed):
        if normed.startswith('left') or normed.endswith('.l') or normed.endswith('_l') or '.l.' in normed or '_l_' in normed:
            return 'L'
        if normed.startswith('right') or normed.endswith('.r') or normed.endswith('_r') or '.r.' in normed or '_r_' in normed:
            return 'R'
        return None

    FINGER_ORDER = ('thumb', 'index', 'middle', 'ring', 'pinky')
    def finger_idx(normed):
        for i, f in enumerate(FINGER_ORDER):
            if f in normed:
                return i
        return len(FINGER_ORDER)

    ARM_KW = ('arm', 'shoulder', 'clavicle', 'hand',
              'thumb', 'index', 'middle', 'ring', 'pinky')
    LEG_KW = ('leg', 'thigh', 'knee', 'shin', 'ankle', 'foot', 'toe', 'heel')
    SPINE_KW = ('spine', 'hips', 'hip', 'torso', 'pelvis')

    # Face = descendants of first bone normalising to 'head'.
    face_names = set()
    head_bone = next((b for b in bones if norm(b.name) == 'head'), None)
    if head_bone:
        stack = list(head_bone.children)
        while stack:
            b = stack.pop()
            if b.name not in face_names:
                face_names.add(b.name)
                stack.extend(b.children)

    neck_bone = next((b for b in bones if norm(b.name) == 'neck'), None)

    def classify(b):
        if b.name in face_names:
            return 'face'
        if b is head_bone or b is neck_bone:
            return 'head_neck'
        n = norm(b.name)
        s = side(n)
        if any(kw in n for kw in ARM_KW):
            return 'arm_L' if s == 'L' else ('arm_R' if s == 'R' else 'other')
        if any(kw in n for kw in LEG_KW):
            return 'leg_L' if s == 'L' else ('leg_R' if s == 'R' else 'other')
        if any(kw in n for kw in SPINE_KW):
            return 'spine'
        return 'other'

    cat = {b.name: classify(b) for b in bones}
    ordered, emitted = [], set()

    def emit(b):
        if b.name in emitted:
            return
        emitted.add(b.name)
        ordered.append(b)

    # 1. Face — feature-grouped, top-down by feature. Sorting face bones
    # by world Z is useless because an entire FK chain (e.g. brow.B.L
    # through brow.B.L.003) sits inside a ~2cm Z band, so pure Z sort
    # interleaves every brow / lid / cheek chain. Instead: name match
    # against a hand-curated feature order, with Mixamo-style side /
    # position prefixes stripped first — "LeftUpperLip" must rank as
    # 'lip', not fall through because it starts with "left" (that
    # fall-through scattered every sided mouth bone into the
    # alphabetical catch-all, splitting the mouth across the table).
    # The list ends with cheeks followed by the nose → lips/mouth →
    # jaw → chin → tongue run, so the center-face cluster reads as ONE
    # contiguous block. Unknown names (glue-only helpers like
    # "chin_end_glue" still match; truly alien ones don't) land in a
    # catch-all bucket at the end.
    FACE_FEATURE_ORDER = (
        'headtop',
        'forehead',
        'temple',
        'brow',        # incl. brow_glue
        'lid',         # eyelid chain – visually sits with eye, list before eye
                       # so the lid FK fans are emitted before the tiny eye bone
        'eye',
        'ear',
        'cheek',       # incl. cheek_glue
        # ---- center cluster: nose + mouth + jaw, kept contiguous ----
        'nose_master',
        'nose',        # incl. nose_glue, nose_end_glue
        'nostril',
        'lip',
        'mouth',       # LeftMouthCorner / RightMouthCorner
        'jaw_master',
        'jaw',
        'chin',        # incl. chin_end_glue
        'tongue',
    )

    # Mixamo/BVH-style names put side and position FIRST (LeftUpperLip,
    # RightCheekLower, LeftNostril). Strip those tokens so the feature
    # match sees the anatomical stem. Rigify-style names (lip.T.L,
    # brow.B.R.001) carry no such prefixes and pass through unchanged.
    _SIDE_POS_PREFIXES = ('left', 'right', 'upper', 'lower',
                          'top', 'bottom', 'inner', 'outer', 'mid')

    def face_stem(normed):
        s = normed
        changed = True
        while changed:
            changed = False
            for p in _SIDE_POS_PREFIXES:
                if s.startswith(p) and len(s) > len(p):
                    s = s[len(p):]
                    changed = True
        return s

    def face_feature_rank(stem):
        for i, feat in enumerate(FACE_FEATURE_ORDER):
            if stem.startswith(feat):
                return i
        return len(FACE_FEATURE_ORDER)

    def face_sort_stem(normed):
        # Side-ONLY strip (left/right), keeping the positional words —
        # 'leftlowerlid' groups as 'lowerlid', so the two sides of one
        # anatomical row share a stem. The full positional strip above
        # is only for feature RANKING.
        s = normed
        for p in ('left', 'right'):
            if s.startswith(p) and len(s) > len(p):
                return s[len(p):]
        return s

    def face_side_rank(normed):
        # Center bones first, then L, then R.
        s = side(normed)
        return 0 if s is None else (1 if s == 'L' else 2)

    # Two grouping modes within a feature:
    #   sided features (brows, lids, eyes, ears, cheeks, …): keep each
    #     side's whole chain together — all Left bones in chain order,
    #     then all Right. A brow/lid/cheek table that ping-pongs
    #     L/R/L/R per row reads as scrambled.
    #   center-cluster features (nose → mouth → jaw → chin → tongue):
    #     these live on/around the centerline, so they read as
    #     anatomical rows instead: LowerLip / LeftLowerLip /
    #     RightLowerLip, then the UpperLip row, then the corners.
    _CENTER_FEATURES = frozenset((
        'nose_master', 'nose', 'nostril', 'lip', 'mouth',
        'jaw_master', 'jaw', 'chin', 'tongue',
    ))

    def face_key(b):
        n = norm(b.name)
        rank_i = face_feature_rank(face_stem(n))
        feat = (FACE_FEATURE_ORDER[rank_i]
                if rank_i < len(FACE_FEATURE_ORDER) else None)
        if feat in _CENTER_FEATURES:
            # Row-major: stem groups the row, center → L → R within it.
            return (rank_i, 0, face_sort_stem(n), face_side_rank(n), b.name)
        # Side-major: whole left chain, then whole right chain; stems
        # alphabetical within a side keep each sub-chain (.001, .002 /
        # Brow1, Brow2) in sequence. Catch-all bucket uses this too.
        return (rank_i, face_side_rank(n), face_sort_stem(n), 0, b.name)

    face_bones = [b for b in bones if cat[b.name] == 'face']
    face_bones.sort(key=face_key)
    for b in face_bones:
        emit(b)

    # 2. Head → Neck (cosmetic: order them top-down).
    if head_bone:
        emit(head_bone)
    if neck_bone:
        emit(neck_bone)

    # 3–4, 6–7: limb chains via hierarchical walk, rooted at each
    # category's top-most bones, children sorted by finger order (for
    # hands) and Z desc otherwise.
    def emit_chain(category):
        roots = [b for b in bones
                 if cat[b.name] == category
                 and (b.parent is None or cat.get(b.parent.name) != category)]
        roots.sort(key=lambda b: (-wz(b), b.name))
        def walk(b):
            if cat[b.name] != category or b.name in emitted:
                return
            emit(b)
            children = sorted(b.children,
                              key=lambda c: (finger_idx(norm(c.name)),
                                             -wz(c), c.name))
            for c in children:
                walk(c)
        for r in roots:
            walk(r)

    emit_chain('arm_L')
    emit_chain('arm_R')

    # 5. Spine — top-down purely by Z (ignoring hierarchy so Hips lands
    # at the bottom of the spine block as the user expects).
    spine_bones = [b for b in bones if cat[b.name] == 'spine']
    spine_bones.sort(key=lambda b: (-wz(b), b.name))
    for b in spine_bones:
        emit(b)

    emit_chain('leg_L')
    emit_chain('leg_R')

    # 8. Anything not categorised: append in the rig's natural order.
    for b in bones:
        emit(b)

    return ordered
