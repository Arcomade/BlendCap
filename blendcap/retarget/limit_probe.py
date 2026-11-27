"""Measure, during the FK bake, how far a Limit Rotation constraint pulls a
mapped bone away from the rotation the retarget writes (support item 14).

Why this exists: the engine cannot bake around a rotation limit. A clamp
maps many inputs to one output, so there is no basis value that comes out
at the pose the capture wanted. The only fix is muting the constraint,
which is the user's call (see ../rig_conflicts.py). What the engine CAN do
is say, after the bake, whether a limit actually changed anything.

Scope, on purpose (measured on Auto-Rig Pro, CloudRig, Rigify and the
Mixamo control rig, 2026-11-12):
  - Only bones the map writes ROTATION to. Every stock rig carries Limit
    Rotation inside its own foot-roll mechanism, clamping by design;
    looking there would warn on every stock rig.
  - Only Limit ROTATION. CloudRig puts Limit Location on mapped face
    controls as their travel range.
A limit counts on a frame only when it moves the bone by more than
EXCESS_THRESHOLD_DEG, so a limit that is never reached reports nothing.

Owner spaces: LOCAL, POSE, WORLD and LOCAL_WITH_PARENT. WORLD matters
most in practice: it is what Blender gives a Limit Rotation added from the
UI, and constraint_sim does not simulate it at all, so this probe reads
the bone's own constraint stack instead of relying on the sim's scan
(found in the first live test, 2026-11-13). CUSTOM space is not measured.

This is a MEASUREMENT. It never changes a key: the bake's forward math
(constraint_sim._apply_constraint_chain) is untouched.

The clamp below follows Blender's rotlimit_evaluate / clamp_angle and is
checked against Blender's own evaluation by the devtools test.
"""
import math

from mathutils import Euler, Matrix, Quaternion

from .constraint_sim import _apply_constraint_chain


# A limit has to move the bone at least this far on a frame to count.
EXCESS_THRESHOLD_DEG = 1.0

MEASURED_SPACES = ('LOCAL', 'POSE', 'WORLD', 'LOCAL_WITH_PARENT')

_EULER_ORDERS = ('XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX')
_TWO_PI = 2.0 * math.pi

# Result of the most recent bake, read by the Apply operator once the
# coroutine finishes. None until a bake completes its per-frame phase.
_LAST = None
# The target rig's real world matrix, handed over by the Apply operator
# before it sets the rig's placement aside for the bake.
_TARGET_WORLD = None


def reset():
    global _LAST
    _LAST = None


def set_target_world(matrix):
    global _TARGET_WORLD
    _TARGET_WORLD = matrix.copy() if matrix is not None else None


def take_last(rig_name):
    """The finished bake's report for `rig_name`, or None. Cleared on read
    so a later, unrelated bake can't be handed an old result."""
    global _LAST
    rep, _LAST = _LAST, None
    if rep is None or rep.get('rig') != rig_name:
        return None
    return rep


def _wrap(a):
    """Wrap to [-pi, pi] (Blender's angle_wrap_rad)."""
    a = (a + math.pi) % _TWO_PI - math.pi
    return a


def clamp_angle(angle, lo, hi):
    """Blender's clamp_angle (non-legacy Limit Rotation)."""
    if (hi - lo) >= _TWO_PI:
        return angle
    if hi <= lo:
        return lo
    lo_w = _wrap(lo - angle)
    hi_w = _wrap(hi - angle)
    if lo_w < hi_w:
        return angle + min(max(0.0, lo_w), hi_w)
    if hi_w >= 0.0 or lo_w <= 0.0:
        return angle
    if abs(hi_w) < abs(lo_w):
        return angle + hi_w
    return angle + lo_w


def euler_order_for(spec, rotation_mode):
    order = spec.get('euler_order', 'AUTO')
    if order in _EULER_ORDERS:
        return order
    return rotation_mode if rotation_mode in _EULER_ORDERS else 'XYZ'


def clamp_rotation(q, spec, rotation_mode):
    """Rotation after `spec` (see limit_spec) acts on `q`, which must
    already be in the constraint's owner space. Returns
    (clamped_quaternion, angle_moved_in_radians)."""
    order = euler_order_for(spec, rotation_mode)
    e = q.to_matrix().to_euler(order)
    vals = [e.x, e.y, e.z]
    legacy = spec.get('legacy', False)
    changed = False
    for i, ax in enumerate('xyz'):
        if not spec['use_limit_' + ax]:
            continue
        lo, hi = spec['min_' + ax], spec['max_' + ax]
        if legacy:
            # Max is tested first and min only otherwise. The order
            # only matters for an inverted range (min above max), and
            # this is the one Blender uses (measured on 5.3).
            v = vals[i]
            if v > hi:
                v = hi
            elif v < lo:
                v = lo
        else:
            v = clamp_angle(vals[i], lo, hi)
        if v != vals[i]:
            vals[i] = v
            changed = True
    if not changed:
        return q, 0.0
    q2 = Euler(vals, order).to_quaternion()
    infl = spec.get('influence', 1.0)
    if infl < 1.0:
        q2 = q.slerp(q2, infl)
    ang = q.rotation_difference(q2).angle
    if ang > math.pi:
        ang = _TWO_PI - ang
    return q2, ang


def limit_spec(con, influence):
    """What clamp_rotation needs from a Limit Rotation constraint."""
    spec = {
        'name': con.name,
        'owner_space': con.owner_space,
        'influence': float(influence),
        'euler_order': getattr(con, 'euler_order', 'AUTO'),
        'legacy': bool(getattr(con, 'use_legacy_behavior', False)),
    }
    for ax in 'xyz':
        spec['use_limit_' + ax] = bool(getattr(con, 'use_limit_' + ax))
        spec['min_' + ax] = float(getattr(con, 'min_' + ax))
        spec['max_' + ax] = float(getattr(con, 'max_' + ax))
    return spec


def to_owner_space(arm_mat, owner_space, P, rest, world):
    """The bone's armature-space pose matrix in the constraint's owner
    space, as Blender converts it (BKE_constraint_mat_convertspace):
      POSE               as is
      LOCAL              relative to the parent's pose and the bone's rest
                         (P = parent pose @ rest offset)
      LOCAL_WITH_PARENT  relative to the bone's own rest matrix
      WORLD              through the armature object's world matrix"""
    if owner_space == 'LOCAL':
        return P.inverted_safe() @ arm_mat
    if owner_space == 'LOCAL_WITH_PARENT':
        return rest.inverted_safe() @ arm_mat
    if owner_space == 'WORLD':
        return world @ arm_mat
    return arm_mat


def from_owner_space(space_mat, owner_space, P, rest, world):
    if owner_space == 'LOCAL':
        return P @ space_mat
    if owner_space == 'LOCAL_WITH_PARENT':
        return rest @ space_mat
    if owner_space == 'WORLD':
        return world.inverted_safe() @ space_mat
    return space_mat


class LimitProbe:
    """One per bake. `bones` is empty on a rig with no Limit Rotation on a
    rotation-mapped bone, and the per-frame loop then skips it with one
    dict lookup."""

    def __init__(self, tgt_rig, target_meta, constraint_meta,
                 eff_influence=None):
        global _TARGET_WORLD
        self.rig_name = tgt_rig.name
        self.bones = {}
        self.stats = {}
        self.frames_hit = set()
        self._threshold = math.radians(EXCESS_THRESHOLD_DEG)
        # Real placement for World Space limits; the object's current
        # matrix when nobody handed one over (a direct engine call).
        world = _TARGET_WORLD
        _TARGET_WORLD = None
        if world is None:
            world = tgt_rig.matrix_world.copy()
        self._world = world
        for name, meta in target_meta.items():
            if not (meta.get('is_mapped') and meta.get('any_rot')):
                continue
            pb = tgt_rig.pose.bones.get(name)
            if pb is None or not pb.constraints:
                continue
            sim_by_name = {}
            for c in constraint_meta.get(name) or ():
                sim_by_name.setdefault(c.get('con_name'), []).append(c)
            # The bone's live constraints in evaluation order: a Limit
            # Rotation in a space we measure becomes a 'limit' step, any
            # other constraint the sim scanned becomes a 'sim' step (it
            # runs first when it comes first), the rest are skipped.
            steps = []
            for con in pb.constraints:
                if eff_influence is not None:
                    mute, infl = eff_influence(con, name)
                else:
                    mute, infl = con.mute, con.influence
                if mute or infl <= 0.0:
                    continue
                if con.type == 'LIMIT_ROTATION':
                    if con.owner_space in MEASURED_SPACES:
                        steps.append(('limit', limit_spec(con, infl)))
                    continue
                for c in sim_by_name.get(con.name, ()):
                    steps.append(('sim', c))
            if not any(kind == 'limit' for kind, _ in steps):
                continue
            rest = tgt_rig.data.bones[name].matrix_local.copy()
            self.bones[name] = (steps, meta['rotation_mode'], rest)
            for kind, spec in steps:
                if kind == 'limit':
                    self.stats[(name, spec['name'])] = [0, 0.0]

    def measure(self, frame, name, basis_loc, basis_rot_q, parent_arm,
                rel_rest, tgt_mw, tgt_mw_inv, computed_tgt_arm_local):
        """Called once per frame for a bone in self.bones, with the basis
        the bake is about to key. Walks the bone's live constraints in
        order so each limit is tested against what it would really be
        handed (anything before it applied first)."""
        steps, rotation_mode, rest = self.bones[name]
        P = parent_arm @ rel_rest if parent_arm is not None else rel_rest
        rot = basis_rot_q if basis_rot_q is not None else Quaternion()
        # Fast path, and a common one (DAZ-style rigs): a single limit in
        # the bone's own space is handed exactly the rotation we key.
        if len(steps) == 1 and steps[0][1]['owner_space'] == 'LOCAL':
            _q, ang = clamp_rotation(rot, steps[0][1], rotation_mode)
            self._note(frame, name, steps[0][1], ang)
            return
        cur = P @ (Matrix.Translation(basis_loc) @ rot.to_matrix().to_4x4())
        for kind, c in steps:
            if kind == 'sim':
                cur = _apply_constraint_chain(
                    cur, [c], tgt_mw, tgt_mw_inv, computed_tgt_arm_local,
                    parent_arm, rel_rest)
                continue
            space = to_owner_space(cur, c['owner_space'], P, rest,
                                   self._world)
            loc, q, scale = space.decompose()
            q2, ang = clamp_rotation(q, c, rotation_mode)
            self._note(frame, name, c, ang)
            if ang > 0.0:
                cur = from_owner_space(
                    Matrix.LocRotScale(loc, q2, scale), c['owner_space'],
                    P, rest, self._world)

    def _note(self, frame, name, spec, ang):
        if ang <= self._threshold:
            return
        st = self.stats[(name, spec['name'])]
        st[0] += 1
        if ang > st[1]:
            st[1] = ang
        self.frames_hit.add(frame)

    def finish(self, frame_start, frame_end, keyed_channels):
        """Publish the report. `keyed_channels` is the set of
        (data_path, array_index) the bake keys, used afterwards to find
        drivers sitting on those same channels."""
        global _LAST
        limits = []
        for (bone, con), (n, worst) in self.stats.items():
            if n > 0:
                limits.append({'bone': bone, 'constraint': con, 'frames': n,
                               'max_deg': math.degrees(worst)})
        limits.sort(key=lambda d: (-d['max_deg'], d['bone']))
        _LAST = {
            'rig': self.rig_name,
            'frames_total': max(0, frame_end - frame_start + 1),
            'frames_hit': len(self.frames_hit),
            'limits': limits,
            'keyed': set(keyed_channels),
        }
        if limits:
            print(f"[BlendCap] retarget: Limit Rotation constraints on "
                  f"'{self.rig_name}' changed the motion on "
                  f"{len(set(d['bone'] for d in limits))} mapped bone(s), "
                  f"{len(self.frames_hit)} of {_LAST['frames_total']} "
                  f"frames:")
            for d in limits:
                print(f"[BlendCap]   {d['bone']} / {d['constraint']}: "
                      f"{d['frames']} frames, up to {d['max_deg']:.1f} deg")
