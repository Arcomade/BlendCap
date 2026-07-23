"""F-curve iteration and bulk write helpers.

`_iter_action_fcurves` papers over Blender's three action storage layouts
(≤4.3 flat fcurves, 4.4 early strip.fcurves, 4.4+/5.x slotted channelbags).
`_bulk_write_keys` primes fcurves via keyframe_insert (the supported API
on Blender 5; action.fcurves.new() was removed) and then fills each
fcurve with a single foreach_set('co', flat) call — orders of magnitude
faster than per-frame keyframe_insert on dense bakes.
`_accumulate_basis_key` is the per-frame helper the bake engines call
to append one bone's keys to the accumulator (with optional quaternion
sign continuity and euler-compat).
"""

from mathutils import Quaternion


def _accumulate_basis_key(key_accum, primed_channels, bone_path, target_name,
                          frame, mode, basis_loc, basis_q,
                          last_q=None, last_e=None):
    """Append one bone's basis-local keys at `frame` into key_accum.

    Used by the FK-fast / IK / IK-fast bake loops. The classic FK bake's
    capture pass (bake_fk.py) reads raw values from pb and is intentionally
    kept inline there — its AXIS_ANGLE branch writes raw `pb.rotation_axis_angle`
    components without round-tripping through a quaternion, and the helper's
    `basis_q.to_axis_angle()` path would introduce subtle numerical drift.

    Parameters:
      key_accum: {(data_path, array_index) -> flat [frame, value, ...]}
      primed_channels: {(bone_name, channel_suffix)} — mutated in place.
      bone_path: pre-escaped `pose.bones["..."]` prefix for the target bone.
      target_name: bone name (for primed_channels keys).
      frame: int.
      mode: target bone's rotation_mode ('QUATERNION' / 'AXIS_ANGLE' / euler).
      basis_loc: Vector or None — None skips location write.
      basis_q: Quaternion or None — None skips rotation write.
      last_q: optional {target_name -> Quaternion} for sign continuity.
              If provided AND basis_q is not None, flips sign when
              prev_q.dot(basis_q) < 0, then stores the result.
      last_e: optional {target_name -> Euler} for euler-compat across frames.
              If provided AND mode is euler AND basis_q is not None, passes
              the stored Euler to basis_q.to_euler(mode, prev_e) as compat
              seed, then stores the result.

    Returns the (possibly sign-flipped) basis_q so callers that need it
    downstream don't have to re-read from last_q.
    """
    if basis_loc is not None:
        primed_channels.add((target_name, 'location'))
        key_accum.setdefault((bone_path + '.location', 0), []).extend((frame, basis_loc.x))
        key_accum.setdefault((bone_path + '.location', 1), []).extend((frame, basis_loc.y))
        key_accum.setdefault((bone_path + '.location', 2), []).extend((frame, basis_loc.z))

    if basis_q is None:
        return basis_q

    if last_q is not None:
        prev_q = last_q.get(target_name)
        if prev_q is not None and prev_q.dot(basis_q) < 0.0:
            basis_q = Quaternion((-basis_q.w, -basis_q.x,
                                  -basis_q.y, -basis_q.z))
        last_q[target_name] = basis_q.copy()

    if mode == 'QUATERNION':
        primed_channels.add((target_name, 'rotation_quaternion'))
        key_accum.setdefault((bone_path + '.rotation_quaternion', 0), []).extend((frame, basis_q.w))
        key_accum.setdefault((bone_path + '.rotation_quaternion', 1), []).extend((frame, basis_q.x))
        key_accum.setdefault((bone_path + '.rotation_quaternion', 2), []).extend((frame, basis_q.y))
        key_accum.setdefault((bone_path + '.rotation_quaternion', 3), []).extend((frame, basis_q.z))
    elif mode == 'AXIS_ANGLE':
        primed_channels.add((target_name, 'rotation_axis_angle'))
        axis, angle = basis_q.to_axis_angle()
        key_accum.setdefault((bone_path + '.rotation_axis_angle', 0), []).extend((frame, angle))
        key_accum.setdefault((bone_path + '.rotation_axis_angle', 1), []).extend((frame, axis.x))
        key_accum.setdefault((bone_path + '.rotation_axis_angle', 2), []).extend((frame, axis.y))
        key_accum.setdefault((bone_path + '.rotation_axis_angle', 3), []).extend((frame, axis.z))
    else:
        primed_channels.add((target_name, 'rotation_euler'))
        if last_e is not None:
            prev_e = last_e.get(target_name)
            if prev_e is not None:
                e = basis_q.to_euler(mode, prev_e)
            else:
                e = basis_q.to_euler(mode)
            last_e[target_name] = e.copy()
        else:
            e = basis_q.to_euler(mode)
        key_accum.setdefault((bone_path + '.rotation_euler', 0), []).extend((frame, e.x))
        key_accum.setdefault((bone_path + '.rotation_euler', 1), []).extend((frame, e.y))
        key_accum.setdefault((bone_path + '.rotation_euler', 2), []).extend((frame, e.z))

    return basis_q


def _iter_action_fcurves(action):
    """Yield every fcurve on `action`, handling all three storage layouts
    we've seen in the wild:
        - Blender ≤4.3: action.fcurves (raises AttributeError on Blender 5).
        - Blender 4.4 early: strip.fcurves directly on a strip.
        - Blender 4.4+ / 5.x slotted: strip.channelbags collection OR
          strip.channelbag(slot) per-slot accessor.
    The slotted API has had wobbly Python bindings — channelbags-as-collection
    isn't always iterable. We try collection first, fall back to per-slot
    lookup via action.slots, then legacy strip.fcurves last.
    """
    if hasattr(action, 'fcurves') and action.fcurves is not None:
        for fc in action.fcurves:
            yield fc
        return
    if not hasattr(action, 'layers'):
        return
    slots = list(getattr(action, 'slots', []) or [])
    for layer in action.layers:
        for strip in layer.strips:
            yielded_via_collection = False
            cbs = getattr(strip, 'channelbags', None)
            if cbs is not None:
                try:
                    for cb in cbs:
                        for fc in cb.fcurves:
                            yield fc
                        yielded_via_collection = True
                except TypeError:
                    yielded_via_collection = False
            if yielded_via_collection:
                continue
            cb_method = getattr(strip, 'channelbag', None)
            if callable(cb_method) and slots:
                got_any = False
                for slot in slots:
                    try:
                        cb = cb_method(slot)
                    except Exception:
                        cb = None
                    if cb is not None:
                        for fc in cb.fcurves:
                            yield fc
                        got_any = True
                if got_any:
                    continue
            if hasattr(strip, 'fcurves'):
                for fc in strip.fcurves:
                    yield fc


def _bulk_write_keys(armature_obj, key_accum, primed_channels, prime_frame):
    """Bulk-write accumulated keyframes onto armature_obj's action.

    GENERATOR: yields None periodically so the caller (the modal retarget
    operator) can let Blender process events between batches. Without these
    yields, a face-rig retarget would prime ~700 fresh fcurves and bulk-
    fill them in one synchronous burst, which on a cold action triggers
    enough internal allocation to make Windows show "Not Responding" for
    a few seconds at the end of the bake. The yields don't make the work
    faster — they just keep the timer ticking so the OS sees the window
    as alive.

    `key_accum`: {(data_path, array_index) -> flat [frame, value, frame, ...]}
    `primed_channels`: {(bone_name, channel_suffix)} — set of bone+channel
        pairs to materialize fcurves for via a single keyframe_insert. Each
        suffix ('location', 'rotation_quaternion', etc.) creates the fcurves
        at all relevant array indices in one call (3 for location/euler,
        4 for quaternion/axis_angle).
    `prime_frame`: frame number used for the priming keyframe_insert. The
        priming key is then overwritten by the bulk fill at the same frame.

    Why prime via keyframe_insert instead of direct fcurve creation:
    Blender 5 removed `action.fcurves.new()` (actions are slotted/layered
    there, and legacy fcurve creation went with the old layout).
    keyframe_insert is the supported API; it creates the action, the slot,
    the layer/strip/channelbag, and the fcurve in one shot. Once the fcurve
    exists, we look it up via the slotted iterator and bulk-fill its
    keyframe_points. foreach_set on `co` accepts a flat [f, v, f, v, ...]
    array and writes all points in a single C call — orders of magnitude
    faster than N keyframe_insert calls on dense bakes.
    """
    # How often to yield during the priming + bulk-fill loops. Small enough
    # that the modal timer (0.03s) gets serviced between batches, large
    # enough that yield overhead doesn't dominate when there's a lot to do.
    YIELD_EVERY = 50

    if not key_accum:
        print("[BlendCap] _bulk_write_keys: empty accumulator — nothing to write")
        return

    # Step 1: prime fcurves — one keyframe_insert per (bone, channel).
    # keyframe_insert returns True/False; we count False as a silent
    # failure (e.g. bone has lock_location flags). Exceptions are also
    # counted and logged.
    prime_ok = 0
    prime_false = 0
    prime_exc = 0
    for i, (bone_name, channel_suffix) in enumerate(primed_channels):
        if bone_name not in armature_obj.pose.bones:
            prime_exc += 1
            continue
        pb = armature_obj.pose.bones[bone_name]
        try:
            ok = pb.keyframe_insert(channel_suffix, frame=prime_frame)
            if ok:
                prime_ok += 1
            else:
                prime_false += 1
                if prime_false <= 3:
                    print(f"[BlendCap] keyframe_insert returned False on "
                          f"{bone_name}.{channel_suffix} (locked / hidden / "
                          f"NLA tweak mode?)")
        except Exception as e:
            prime_exc += 1
            if prime_exc <= 3:
                print(f"[BlendCap] keyframe_insert raised on "
                      f"{bone_name}.{channel_suffix}: {e}")
        if (i + 1) % YIELD_EVERY == 0:
            yield None

    action = None
    if armature_obj.animation_data:
        action = armature_obj.animation_data.action
    if action is None:
        print("[BlendCap] _bulk_write_keys: rig has no action after priming "
              "— no keys written")
        return

    # Index every fcurve once so the bulk-fill loop is O(N) instead of
    # O(N²). A per-key linear scan over _iter_action_fcurves on a face
    # rig (~700 channels) means ~700 × ~700 = ~500k walks just to flush
    # the accumulator — that's the multi-second "Not Responding" freeze
    # at the end of a face retarget.
    fcurve_index = {}
    for fc in _iter_action_fcurves(action):
        fcurve_index[(fc.data_path, fc.array_index)] = fc

    # Step 2: bulk-fill each fcurve from the accumulator.
    written = 0
    missing = 0
    missing_examples = []
    for i, ((data_path, array_index), flat) in enumerate(key_accum.items()):
        fc = fcurve_index.get((data_path, array_index))
        if fc is None:
            missing += 1
            if len(missing_examples) < 3:
                missing_examples.append(f"{data_path}[{array_index}]")
            continue
        n_target = len(flat) // 2
        if n_target <= 0:
            continue
        existing = len(fc.keyframe_points)
        if existing < n_target:
            fc.keyframe_points.add(n_target - existing)
        elif existing > n_target:
            for _ in range(existing - n_target):
                fc.keyframe_points.remove(fc.keyframe_points[-1], fast=True)
        fc.keyframe_points.foreach_set('co', flat)
        fc.update()
        written += 1
        if (i + 1) % YIELD_EVERY == 0:
            yield None
    if missing_examples:
        print(f"[BlendCap] bulk-fill missing fcurves: {missing_examples} "
              f"({missing} total)")
    return written
