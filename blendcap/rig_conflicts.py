"""Rotation limits and drivers that override retargeted motion
(support item 14).

Two things on a target rig can make a correct retarget look wrong, and
the engine cannot bake around either:

  * A Limit Rotation constraint on a bone the map writes to. The keys
    hold the rotation the capture wanted; the constraint clamps it every
    time Blender shows a frame.
  * A driver on a channel the retarget keys. Blender evaluates drivers
    after keyframes, so those keys are ignored.

The only fix for both is muting, and a mute has to stay on for as long as
the animation should play correctly (restoring it after the bake just
re-clamps on playback). So muting is the user's choice:

  * Retargeting > Advanced > "Mute Limits and Drivers" (per target rig),
    with "Include Unmapped Bones" under it for rigs whose limits sit on
    mechanism bones (the Universal Human case).
  * After a retarget, a dialog when a limit actually changed the motion
    (retarget/limit_probe.py measures it) or a driver sits on a keyed
    channel. Its button ticks the checkbox and retargets again.

What BlendCap muted is written on the RIG OBJECT (blendcap_muted_conflicts,
a small JSON string), so it is saved in the .blend and comes along when
the rig is appended. The checkbox has no stored value of its own: it reads
the rig (ticked = the record lists something that is still muted), and
unticking unmutes only what the record lists.

Scope decisions (measured on Auto-Rig Pro, CloudRig, Rigify and the Mixamo
control rig 2026-11-12; none of them may ever trigger the dialog):
  - limits: bones the map writes ROTATION to only (stock rigs clamp on
    purpose inside their foot-roll mechanism);
  - Limit Rotation only (CloudRig limits the LOCATION of mapped face
    controls as their travel range);
  - drivers: only on a channel that is actually keyed (ARP and Rigify
    drive the scale of FK bones the presets map).
"""
import json
import re
import time

import bpy

from .retarget import limit_probe
from .retarget.name_matching import _resolve_against_rig


RECORD_PROP = "blendcap_muted_conflicts"

_PATH_RE = re.compile(
    r'^pose\.bones\["(.*)"\]\.'
    r'(rotation_quaternion|rotation_euler|rotation_axis_angle|location)$')

# The dialog's content between the bake finishing and the user answering.
_PENDING = None
# The window the retarget ran in, so the dialog opens there.
_PENDING_WINDOW = None
# What opening the dialog last returned ({'RUNNING_MODAL'} = it is up).
# For the devtools window test and for reading a support log.
LAST_DIALOG_RESULT = None
# What the last finished retarget found (also read by the devtools tests).
LAST_FINDINGS = None
# rig name -> one line shown under the checkbox (entries that could not
# be found when unticking). Cleared on the next toggle.
_NOTES = {}
# rig name -> (timestamp, bool) for the draw-time "is there anything this
# checkbox could act on" scan.
_CAN_ACT_CACHE = {}


# ---------------------------------------------------------------- record

def _load(rig):
    rec = {'on': False, 'unmapped': False, 'limits': [], 'drivers': []}
    try:
        raw = getattr(rig, RECORD_PROP, "") or ""
        if raw:
            data = json.loads(raw)
            if isinstance(data, dict):
                rec['on'] = bool(data.get('on', False))
                rec['unmapped'] = bool(data.get('unmapped', False))
                rec['limits'] = [list(x) for x in data.get('limits', [])
                                 if isinstance(x, (list, tuple)) and len(x) >= 2]
                rec['drivers'] = [list(x) for x in data.get('drivers', [])
                                  if isinstance(x, (list, tuple)) and len(x) == 2]
    except Exception:
        pass
    for e in rec['limits']:
        if len(e) < 3:
            e.append('m')
    return rec


def _save(rig, rec):
    if not (rec['on'] or rec['limits'] or rec['drivers']):
        setattr(rig, RECORD_PROP, "")
    else:
        setattr(rig, RECORD_PROP, json.dumps(rec, separators=(',', ':')))
    _CAN_ACT_CACHE.pop(rig.name, None)


def editable(rig):
    """False for a linked rig: its constraints can't be changed here."""
    return (rig is not None and rig.type == 'ARMATURE'
            and rig.library is None)


# ---------------------------------------------------------------- lookups

def _limit_con(rig, bone, name):
    pb = rig.pose.bones.get(bone)
    if pb is None:
        return None
    con = pb.constraints.get(name)
    if con is None or con.type != 'LIMIT_ROTATION':
        return None
    return con


def _driver(rig, path, index):
    ad = rig.animation_data
    if ad is None:
        return None
    try:
        return ad.drivers.find(path, index=int(index))
    except Exception:
        return None


def _channel_key(data_path, index):
    """(bone, property, index) for a pose-bone transform channel, else
    None. Compared in this form so the two ways a bone name gets escaped
    in a data path can't cause a miss."""
    m = _PATH_RE.match(data_path)
    if not m:
        return None
    try:
        bone = bpy.utils.unescape_identifier(m.group(1))
    except Exception:
        bone = m.group(1)
    return (bone, m.group(2), int(index))


def normalize_channels(channels):
    out = set()
    for path, index in channels:
        k = _channel_key(path, index)
        if k is not None:
            out.add(k)
    return out


def _rotation_prop(pb):
    mode = pb.rotation_mode
    if mode == 'QUATERNION':
        return 'rotation_quaternion', 4
    if mode == 'AXIS_ANGLE':
        return 'rotation_axis_angle', 4
    return 'rotation_euler', 3


def mapped_targets(scene, rig):
    """(bones the current map writes rotation to, channels it will key)
    for `rig` as the retarget target. Both empty when it isn't the
    target. The channel set is a prediction from the pair table; the bake
    itself reports the exact set afterwards (limit_probe)."""
    rot_bones, keyed = set(), set()
    if scene is None or getattr(scene, 'blendcap_retarget_target', None) != rig:
        return rot_bones, keyed
    prefix = getattr(scene, 'blendcap_retarget_namespace_strip', "") or ""
    for p in scene.blendcap_retarget_pairs:
        if not p.source or not p.target:
            continue
        name = _resolve_against_rig(p.target, rig, prefix)
        pb = rig.pose.bones.get(name)
        if pb is None:
            continue
        if p.channels in ('ROT', 'LOC_ROT'):
            rot_bones.add(name)
            prop, n = _rotation_prop(pb)
            for i in range(n):
                keyed.add((name, prop, i))
        if p.channels in ('LOC', 'LOC_ROT'):
            for i in range(3):
                keyed.add((name, 'location', i))
    return rot_bones, keyed


def live_limits(rig, bones=None):
    """Unmuted Limit Rotation constraints, on `bones` or the whole rig."""
    out = []
    for pb in rig.pose.bones:
        if bones is not None and pb.name not in bones:
            continue
        for con in pb.constraints:
            if con.type == 'LIMIT_ROTATION' and not con.mute:
                out.append((pb, con))
    return out


def driver_conflicts(rig, keyed):
    """Working, unmuted drivers on a keyed channel. `keyed` holds
    (bone, property, index) tuples."""
    out = []
    ad = rig.animation_data
    if ad is None or not keyed:
        return out
    for d in ad.drivers:
        if d.mute:
            continue
        k = _channel_key(d.data_path, d.array_index)
        if k is None or k not in keyed:
            continue
        try:
            if not d.driver.is_valid:
                continue
        except Exception:
            pass
        out.append(d)
    return out


# ---------------------------------------------------------------- state

def _still_muted(rig, rec):
    n = 0
    for bone, name, _scope in rec['limits']:
        con = _limit_con(rig, bone, name)
        if con is not None and con.mute:
            n += 1
    for path, index in rec['drivers']:
        d = _driver(rig, path, index)
        if d is not None and d.mute:
            n += 1
    return n


def is_on(rig):
    """Ticked = BlendCap's record on this rig says on, and (when it lists
    anything) at least one listed item is still muted."""
    if rig is None or rig.type != 'ARMATURE':
        return False
    rec = _load(rig)
    if not rec['on']:
        return False
    if not rec['limits'] and not rec['drivers']:
        return True
    return _still_muted(rig, rec) > 0


def can_act(rig, scene):
    """Is there anything the checkbox could mute or restore? Scans the
    whole rig, so the answer is cached for a second per rig."""
    now = time.monotonic()
    hit = _CAN_ACT_CACHE.get(rig.name)
    if hit is not None and now - hit[0] < 1.0:
        return hit[1]
    res = False
    try:
        rec = _load(rig)
        if rec['on'] or rec['limits'] or rec['drivers']:
            res = True
        elif live_limits(rig):
            res = True
        else:
            _rot, keyed = mapped_targets(scene, rig)
            res = bool(driver_conflicts(rig, keyed))
    except Exception:
        res = True
    _CAN_ACT_CACHE[rig.name] = (now, res)
    return res


# ---------------------------------------------------------------- actions

def apply(rig, scene, extra_keyed=None, unmapped=None):
    """Mute the Limit Rotation constraints on bones the map writes
    rotation to, and the drivers on channels it keys, and record them on
    the rig. With the record's `unmapped` flag, also every other Limit
    Rotation on the rig. Safe to call again (the Apply operator does,
    before each bake, so bones added to the map later are covered).
    `unmapped` True / False sets that flag first; None leaves it as it is.
    Returns (limits muted now, drivers muted now)."""
    rec = _load(rig)
    if unmapped is not None:
        rec['unmapped'] = bool(unmapped)
    rot_bones, keyed = mapped_targets(scene, rig)
    if extra_keyed:
        keyed = set(keyed) | set(extra_keyed)

    limits, seen = [], set()
    for bone, name, scope in rec['limits']:
        con = _limit_con(rig, bone, name)
        if con is not None and con.mute and (bone, name) not in seen:
            seen.add((bone, name))
            limits.append([bone, name, scope])
    drivers, dseen = [], set()
    for path, index in rec['drivers']:
        d = _driver(rig, path, index)
        if d is not None and d.mute and (path, int(index)) not in dseen:
            dseen.add((path, int(index)))
            drivers.append([path, int(index)])

    n_lim = n_drv = 0
    for pb, con in live_limits(rig, rot_bones):
        con.mute = True
        n_lim += 1
        if (pb.name, con.name) not in seen:
            seen.add((pb.name, con.name))
            limits.append([pb.name, con.name, 'm'])
    if rec['unmapped']:
        for pb, con in live_limits(rig):
            con.mute = True
            n_lim += 1
            if (pb.name, con.name) not in seen:
                seen.add((pb.name, con.name))
                limits.append([pb.name, con.name, 'u'])
    for d in driver_conflicts(rig, keyed):
        d.mute = True
        n_drv += 1
        key = (d.data_path, int(d.array_index))
        if key not in dseen:
            dseen.add(key)
            drivers.append([d.data_path, int(d.array_index)])

    rec['on'] = True
    rec['limits'] = limits
    rec['drivers'] = drivers
    _save(rig, rec)
    if n_lim or n_drv:
        rig.update_tag()
        print(f"[BlendCap] '{rig.name}': muted {n_lim} Limit Rotation "
              f"constraint(s) and {n_drv} driver(s).")
    return n_lim, n_drv


def restore(rig, scope=None):
    """Unmute what the record lists (only scope 'u' entries when
    scope='u'). Returns (restored, not found). Anything not found, such
    as a bone or constraint renamed since, is dropped from the record."""
    rec = _load(rig)
    restored = missing = 0
    keep = []
    for bone, name, sc in rec['limits']:
        if scope is not None and sc != scope:
            keep.append([bone, name, sc])
            continue
        con = _limit_con(rig, bone, name)
        if con is None:
            missing += 1
        elif con.mute:
            con.mute = False
            restored += 1
    drivers_keep = []
    if scope is None:
        for path, index in rec['drivers']:
            d = _driver(rig, path, index)
            if d is None:
                missing += 1
            elif d.mute:
                d.mute = False
                restored += 1
    else:
        drivers_keep = rec['drivers']
    rec['limits'] = keep
    rec['drivers'] = drivers_keep
    if scope is None:
        rec['on'] = False
        rec['unmapped'] = False
    else:
        rec['unmapped'] = False
    _save(rig, rec)
    if restored:
        rig.update_tag()
    print(f"[BlendCap] '{rig.name}': restored {restored} muted item(s)"
          + (f", {missing} could not be found." if missing else "."))
    return restored, missing


# ---------------------------------------------------------------- presets
# A bone map preset carries the two checkboxes (the user, 2026-11-27:
# "They should be saved with presets ... so people can make their own
# presets conveniently"). They are not panel values: they are what is
# muted on the target rig, so a preset holds what they were on the rig
# it was saved from, and a load sets them on the rig it is loaded for.
#
#   saved ticked    -> ticked on load (muted for the loaded map)
#   saved unticked  -> unticked on load (what BlendCap muted is unmuted)
#   no value        -> the rig is left alone. Every bundled preset and
#                      every preset saved before 1.1.0, which therefore
#                      behave exactly as they did.

def preset_state(rig):
    """What a preset saves of this rig's two checkboxes: {'on',
    'unmapped'}, or None when there is no rig to read them from."""
    if rig is None or getattr(rig, "type", "") != 'ARMATURE':
        return None
    on = bool(is_on(rig))
    return {"on": on, "unmapped": bool(on and _load(rig)['unmapped'])}


def _preset_wants(state):
    on = bool(state.get("on", False))
    return {"on": on, "unmapped": on and bool(state.get("unmapped", False))}


def apply_preset_state(rig, scene, state):
    """Set the two checkboxes on `rig` to what a preset holds (`state`,
    its 'mute_conflicts' block). Called at the end of a preset load, with
    the pair table already in place. True when the rig was changed."""
    if not isinstance(state, dict) or not editable(rig):
        return False
    want, now = _preset_wants(state), preset_state(rig)
    if want == now:
        if want["on"]:
            # already ticked: cover the bones of the map just loaded
            apply(rig, scene)
        return False
    if want["on"]:
        if now["unmapped"] and not want["unmapped"]:
            restore(rig, scope='u')
        apply(rig, scene, unmapped=want["unmapped"])
    else:
        restore(rig)
    _NOTES.pop(rig.name, None)
    _tag_sidebar()
    return True


def note_preset_drift(rig):
    """After the checkboxes were changed by hand or by Mute and Retarget
    Again: when the selected bone map preset carries them and no longer
    matches the rig, the preset dropdown goes to "(Unsaved changes...)",
    as it does for any other edit of a loaded preset. That also keeps a
    later rig change from re-firing the preset's load and putting its
    saved value back over the user's.
    A preset without the value (every bundled one) is not concerned:
    nothing changes for it, the dropdown stays as it is."""
    try:
        import os
        from . import retarget
        if retarget._LOADING_PRESET:
            return
        scene = bpy.context.scene
        if getattr(scene, 'blendcap_retarget_target', None) != rig:
            return
        path = str(scene.blendcap_retarget_preset or "")
        if not path or path.startswith('__') or not os.path.isfile(path):
            return
        saved = retarget._load_retarget_map(path).get("mute_conflicts")
        if not isinstance(saved, dict) or _preset_wants(saved) == preset_state(rig):
            return
        retarget._LOADING_PRESET = True
        try:
            scene.blendcap_retarget_preset = '__UNSAVED__'
        finally:
            retarget._LOADING_PRESET = False
    except Exception as exc:
        print(f"[BlendCap] Preset not marked as changed: {exc}")


# ---------------------------------------------------------------- properties

def _get_on(self):
    return is_on(self)


def _set_on(self, value):
    scene = bpy.context.scene
    _NOTES.pop(self.name, None)
    if not editable(self):
        return
    if value:
        apply(self, scene)
    else:
        _restored, missing = restore(self)
        if missing:
            _NOTES[self.name] = (
                f"{missing} muted item{'s' if missing != 1 else ''} could "
                f"not be found (renamed?). Unmute by hand if needed.")
    note_preset_drift(self)
    _tag_sidebar()


def _get_unmapped(self):
    return _load(self)['unmapped'] and is_on(self)


def _set_unmapped(self, value):
    scene = bpy.context.scene
    _NOTES.pop(self.name, None)
    if not editable(self):
        return
    if value:
        apply(self, scene, unmapped=True)
    else:
        _restored, missing = restore(self, scope='u')
        if missing:
            _NOTES[self.name] = (
                f"{missing} muted item{'s' if missing != 1 else ''} could "
                f"not be found (renamed?). Unmute by hand if needed.")
    note_preset_drift(self)
    _tag_sidebar()


def _tag_sidebar():
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type in ('VIEW_3D', 'PROPERTIES'):
                    area.tag_redraw()
    except Exception:
        pass


_ON_NAME = "Mute Limits and Drivers"
_ON_DESC = ("Mute Limit Rotation constraints and drivers on the bones this "
            "map writes to, so they can't override the retargeted motion. "
            "Untick to turn them back on")
_UNMAPPED_NAME = "Include Unmapped Bones"
_UNMAPPED_DESC = ("Also mute the Limit Rotation constraints on bones the map "
                  "doesn't write to. Rig features that rely on them, such "
                  "as foot roll, can stop working")


def register_properties():
    bpy.types.Object.blendcap_muted_conflicts = bpy.props.StringProperty(
        name="BlendCap Muted Limits and Drivers", default="",
        options={'HIDDEN'},
        description="What BlendCap muted on this rig (internal record)")
    bpy.types.Object.blendcap_mute_conflicts = bpy.props.BoolProperty(
        name=_ON_NAME, description=_ON_DESC,
        # Not animatable: it is a switch that reads and writes the rig,
        # with no value of its own to key.
        options=set(),
        get=_get_on, set=_set_on)
    bpy.types.Object.blendcap_mute_conflicts_unmapped = bpy.props.BoolProperty(
        name=_UNMAPPED_NAME, description=_UNMAPPED_DESC,
        options=set(),
        get=_get_unmapped, set=_set_unmapped)
    # Inert stand-in so the row can be drawn greyed with no target rig.
    bpy.types.Scene.blendcap_mute_conflicts_placeholder = bpy.props.BoolProperty(
        name=_ON_NAME, description=_ON_DESC, default=False)


def unregister_properties():
    for owner, name in ((bpy.types.Object, "blendcap_muted_conflicts"),
                        (bpy.types.Object, "blendcap_mute_conflicts"),
                        (bpy.types.Object, "blendcap_mute_conflicts_unmapped"),
                        (bpy.types.Scene, "blendcap_mute_conflicts_placeholder")):
        try:
            delattr(owner, name)
        except Exception:
            pass


# ---------------------------------------------------------------- panel

def draw_rows(col, scene):
    """The checkbox rows in Retargeting > Advanced."""
    rig = getattr(scene, 'blendcap_retarget_target', None)
    if rig is None or rig.type != 'ARMATURE':
        row = col.row()
        row.enabled = False
        row.prop(scene, "blendcap_mute_conflicts_placeholder")
        return
    row = col.row()
    row.enabled = editable(rig) and can_act(rig, scene)
    row.prop(rig, "blendcap_mute_conflicts")
    if rig.blendcap_mute_conflicts:
        sub = col.row()
        sub.separator(factor=2.0)
        sub.enabled = editable(rig)
        sub.prop(rig, "blendcap_mute_conflicts_unmapped")
    note = _NOTES.get(rig.name)
    if note:
        col.label(text=note, icon='INFO')


# ---------------------------------------------------------------- after a bake

def before_bake(rig, scene):
    """Apply operator, before the engine scans the rig: with the box
    ticked, mute again so bones added to the map since are covered."""
    try:
        if editable(rig) and is_on(rig):
            apply(rig, scene)
    except Exception as e:
        print(f"[BlendCap] mute before retarget skipped: {e}")


def after_bake(op, rig, scene, show=True, defer=False):
    """Apply operator, once the bake has finished. Works out what the
    limits and drivers did and either handles it quietly (box already
    ticked) or queues the dialog. `defer`: something else is still running
    (auto-bake IK); the caller shows the dialog later via show_pending."""
    global _PENDING, _PENDING_WINDOW, LAST_FINDINGS
    LAST_FINDINGS = None
    rep = limit_probe.take_last(rig.name)
    if rep is None:
        return None
    keyed = normalize_channels(rep['keyed'])
    limits = rep['limits']
    drv = driver_conflicts(rig, keyed)
    drv_bones = sorted(set(_channel_key(d.data_path, d.array_index)[0]
                           for d in drv))
    findings = {
        'rig': rig.name,
        'limits': limits,
        'limit_bones': sorted(set(d['bone'] for d in limits)),
        'frames_hit': rep['frames_hit'],
        'frames_total': rep['frames_total'],
        'driver_bones': drv_bones,
        'keyed': keyed,
        'handled': False,
    }
    LAST_FINDINGS = findings
    if drv:
        print(f"[BlendCap] retarget: drivers override the keyframes on "
              f"{len(drv_bones)} bone(s) of '{rig.name}': "
              + ", ".join(drv_bones))
    if not limits and not drv:
        return findings

    if is_on(rig):
        # Box already ticked. A driver on a channel the pair table didn't
        # predict (a compensated bone keys both location and rotation) is
        # muted now; keys don't depend on it, so no second bake is needed.
        if drv and editable(rig):
            apply(rig, scene, extra_keyed=keyed)
        if limits:
            op.report({'WARNING'},
                      f"Rotation limits still changed the motion on "
                      f"{len(findings['limit_bones'])} bone(s); they could "
                      f"not be muted (see the console).")
        findings['handled'] = True
        return findings

    if not editable(rig):
        op.report({'WARNING'},
                  "Rotation limits or drivers on this linked rig changed "
                  "the retargeted motion (see the console).")
        return findings

    if not show:
        # Script call: nothing may be left waiting for a later, unrelated
        # operator to pop up.
        return findings
    _PENDING = findings
    try:
        _PENDING_WINDOW = bpy.context.window
    except Exception:
        _PENDING_WINDOW = None
    if not defer:
        show_pending()
    return findings


def _open_dialog():
    global LAST_DIALOG_RESULT
    if _PENDING is None or bpy.app.background:
        return None
    try:
        wm = bpy.context.window_manager
        wins = list(wm.windows) if wm else []
        if not wins:
            return None
        win = _PENDING_WINDOW if _PENDING_WINDOW in wins else wins[0]
        with bpy.context.temp_override(window=win):
            LAST_DIALOG_RESULT = bpy.ops.blendcap.retarget_conflicts(
                'INVOKE_DEFAULT')
        print(f"[BlendCap] rig limits dialog: {sorted(LAST_DIALOG_RESULT)}")
    except Exception as e:
        LAST_DIALOG_RESULT = {'ERROR'}
        print(f"[BlendCap] could not show the rig limits dialog: {e}")
    return None


def show_pending():
    """Open the dialog if a retarget left one waiting. Deferred through a
    timer and opened with an explicit window, the same way the update
    notice and the capture warnings open theirs: the caller is a modal
    operator in the middle of finishing."""
    if _PENDING is None:
        return
    try:
        if not bpy.app.timers.is_registered(_open_dialog):
            bpy.app.timers.register(_open_dialog, first_interval=0.2)
    except Exception as e:
        print(f"[BlendCap] could not schedule the rig limits dialog: {e}")


class BLENDCAP_OT_retarget_conflicts(bpy.types.Operator):
    """Shown after a retarget when rotation limits changed the motion or
    drivers override the keyframes. Confirm ticks Mute Limits and Drivers
    and Include Unmapped Bones for the target rig and retargets again."""
    bl_idname = "blendcap.retarget_conflicts"
    bl_label = "BlendCap: Rig Conflict"
    bl_description = ("Mute the rotation limits and drivers that override "
                      "the retargeted motion, then retarget again")
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def invoke(self, context, event):
        if _PENDING is None:
            return {'CANCELLED'}
        return context.window_manager.invoke_props_dialog(
            self, width=520, confirm_text="Mute and Retarget Again")

    def draw(self, context):
        layout = self.layout
        f = _PENDING
        if f is None:
            layout.label(text="Nothing to report.", icon='CHECKMARK')
            return
        # Wording worked out with the user 2026-11-14: plain, for people
        # who don't know what a constraint or a driver is. One fixed
        # sentence ("or", his word, whichever of the two was found), no
        # numbers and no bone names: the console lists them for anyone
        # who wants to look.
        col = layout.column(align=True)
        col.label(text="This rig limits how certain bones can move, or has "
                       "drivers that override animation.", icon='ERROR')
        col = layout.column(align=True)
        col.label(text="      You can mute these so the retargeted animation "
                       "plays correctly.")
        col.label(text="      To undo this later, untick Mute Limits and "
                       "Drivers in Retargeting > Advanced.")

    def cancel(self, context):
        global _PENDING
        _PENDING = None

    def execute(self, context):
        global _PENDING
        f, _PENDING = _PENDING, None
        if f is None:
            return {'CANCELLED'}
        scene = context.scene
        rig = bpy.data.objects.get(f['rig'])
        if rig is None or scene.blendcap_retarget_target != rig:
            self.report({'WARNING'},
                        "The target rig changed; nothing was muted.")
            return {'CANCELLED'}
        # The pop-up's confirm also turns Include Unmapped Bones on (user
        # decision 2026-11-15, after the live DAZ test: a Diffeomorphic
        # Rigify rig only retargets cleanly with the limits on its
        # unmapped bones muted too, and the people who meet this pop-up
        # are the ones least likely to go looking for a second checkbox).
        # Ticking the checkbox by hand still starts with it off.
        apply(rig, scene, extra_keyed=f['keyed'], unmapped=True)
        note_preset_drift(rig)
        _tag_sidebar()

        # Retarget again once this dialog has closed. Started from a
        # timer so the Apply operator's own modal gets a normal window
        # context instead of the popup's.
        window = context.window

        def _rerun():
            try:
                # Same source rest pose as the run this follows ("Use
                # current source pose as rest": the bake's closing
                # frame_set has replaced the user's posed source).
                from . import operators_retarget
                operators_retarget.REUSE_SOURCE_REST_ONCE = True
                win = window
                if win is None or win not in list(
                        bpy.context.window_manager.windows):
                    win = bpy.context.window_manager.windows[0]
                area = next((a for a in win.screen.areas
                             if a.type == 'VIEW_3D'), None)
                if area is not None:
                    region = next((r for r in area.regions
                                   if r.type == 'WINDOW'), None)
                    with bpy.context.temp_override(
                            window=win, area=area, region=region):
                        bpy.ops.blendcap.apply_retarget('INVOKE_DEFAULT')
                else:
                    with bpy.context.temp_override(window=win):
                        bpy.ops.blendcap.apply_retarget('INVOKE_DEFAULT')
            except Exception as e:
                print(f"[BlendCap] could not start the retarget again: {e}")
            return None

        bpy.app.timers.register(_rerun, first_interval=0.1)
        return {'FINISHED'}
