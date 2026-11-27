"""What syncing mode puts ON the Sequencer (multicam.py holds the mode).

While syncing, the timeline shows only the clips being synced: they are
nested, and the Sequencer is stepped inside the nest. Nothing of ours
may sit in the timeline then, so what used to be text strips is drawn
over the editor instead:

  - the camera letters, A to D, on the tiles in the preview;
  - a band across the top of the timeline saying that this is syncing
    mode and how to leave it (not on the preview: it would cover the
    top of the footage, and the header button sits right above it);
  - an Exit Syncing Mode button in the Sequencer's own header, which
    every Sequencer editor has, preview-only ones included.

The drawing is also where a user stepping out of the nest by themselves
(Tab) gets noticed: a Sequencer that is on screen redraws.
"""
import bpy

from . import multicam
from .helpers import blendcap_state

# The band's wording. When the clips could not be nested the rest of
# the timeline is still on screen, so the band must not say otherwise.
# (The user's wording, 2026-11-17. The capture is named because a
# finished capture also brings the timeline back; "run the capture",
# not the button's name in quotes, because it is the capture finishing
# that does it, not the click.)
BAND_NESTED = ('SYNCING MODE: only the clips selected for capture are shown.',
               'Run the capture or click "Exit Syncing Mode" to return to your timeline.')
BAND_OPEN = 'SYNCING MODE: run the capture or click "Exit Syncing Mode" when you are done.'
BAND_TINY = "SYNCING MODE"

BAND_COLOUR = (0.86, 0.42, 0.08, 0.92)
BAND_TEXT = (1.0, 1.0, 1.0, 1.0)

_HANDLES = {}


def band_lines(scene, fits=None):
    """The band's wording as the lines to draw: both sentences on one
    line when that fits, on two lines when it does not, and the bare
    mode name when even those do not (fits(text) -> bool; None means
    everything fits)."""
    def ok(text):
        return fits is None or fits(text)
    if multicam.nest_open(scene):
        if ok(" ".join(BAND_NESTED)):
            return [" ".join(BAND_NESTED)]
        if all(ok(t) for t in BAND_NESTED):
            return list(BAND_NESTED)
    elif ok(BAND_OPEN):
        return [BAND_OPEN]
    return [BAND_TINY]


def _watch(scene):
    """The user stepped out of the nest by themselves: end syncing mode
    (on the next timer tick; drawing may not write scene data). Not
    while a capture runs: its end deals with the nest itself. And a
    clip moved by hand: the nest's range has to follow it."""
    if blendcap_state['running']:
        return
    if multicam.stepped_out(scene):
        multicam.schedule_sync_autoclear()
    elif multicam.nest_stale(scene):
        multicam.schedule_nest_fit()


def _ui_scale(context):
    try:
        return float(context.preferences.system.ui_scale)
    except Exception:
        return 1.0


def _band(context, top_gap):
    """The syncing mode band across the top of the region being drawn,
    `top_gap` pixels down from its top edge."""
    import blf
    import gpu
    from gpu_extras.batch import batch_for_shader
    region = context.region
    ui = _ui_scale(context)
    size = 13.0 * ui
    pad = 7.0 * ui
    font = 0
    blf.size(font, size)
    lines = band_lines(
        context.scene,
        lambda t: blf.dimensions(font, t)[0] <= region.width - 2 * pad)
    step = size * 1.45                       # from one line to the next
    height = size + 2 * pad + step * (len(lines) - 1)
    y1 = region.height - top_gap
    y0 = y1 - height
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(
        shader, 'TRI_FAN',
        {"pos": [(0, y0), (region.width, y0), (region.width, y1), (0, y1)]})
    gpu.state.blend_set('ALPHA')
    shader.bind()
    shader.uniform_float("color", BAND_COLOUR)
    batch.draw(shader)
    gpu.state.blend_set('NONE')
    blf.color(font, *BAND_TEXT)
    for i, text in enumerate(lines):
        tw = blf.dimensions(font, text)[0]
        blf.position(font, max(pad, (region.width - tw) / 2.0),
                     y0 + pad + size * 0.12 + step * (len(lines) - 1 - i), 0)
        blf.draw(font, text)


def letter_spots(scene, clips, to_region):
    """Where each camera's letter goes in the preview, in region pixels:
    [(letter, x, y, size)], x / y the letter's bottom left. `to_region`
    maps a point of the picture (project pixels from its CENTRE, y up,
    which is the preview's own space) to region pixels."""
    res_x = max(1, int(scene.render.resolution_x))
    res_y = max(1, int(scene.render.resolution_y))
    rects = multicam.tile_layout([multicam._strip_size(s) for s in clips],
                                 res_x, res_y)
    # region pixels per project pixel, measured over a long stretch
    # (Blender answers in whole pixels)
    x0, _ = to_region(0.0, 0.0)
    x1, _ = to_region(10000.0, 0.0)
    zoom = abs(float(x1) - float(x0)) / 10000.0
    base = max(24, res_x // 40)
    size = base * 2.8        # twice the size the text strips had (the user, 2026-11-17)
    pad = base * 0.9
    out = []
    for letter, (x, y, _w, _h) in zip(multicam.LETTERS, rects):
        px, py = to_region(x - res_x / 2.0 + pad, res_y / 2.0 - y - pad - size * 0.72)
        out.append((letter, float(px), float(py), size * zoom))
    return out


def _letters(context, clips):
    import blf
    v2d = context.region.view2d
    ui = _ui_scale(context)
    font = 0
    blf.enable(font, blf.SHADOW)
    try:
        blf.shadow(font, 5, 0.0, 0.0, 0.0, 0.8)      # 5: the softest blur there is
    except Exception:
        pass
    blf.color(font, 1.0, 1.0, 1.0, 1.0)
    try:
        for letter, x, y, size in letter_spots(
                context.scene, clips,
                lambda a, b: v2d.view_to_region(a, b, clip=False)):
            size = max(size, 14.0 * ui)
            drop = max(2, int(round(size * 0.07)))   # a drop shadow, down and to the right
            try:
                blf.shadow_offset(font, drop, -drop)
            except Exception:
                pass
            blf.size(font, size)
            blf.position(font, x, y, 0)
            blf.draw(font, letter)
    finally:
        blf.disable(font, blf.SHADOW)


def _draw_timeline():
    try:
        context = bpy.context
        scene = context.scene
        if not multicam.sync_active(scene):
            return
        _watch(scene)
        # below the frame-number ruler along the top of the timeline
        _band(context, 24.0 * _ui_scale(context))
    except Exception:
        pass


def _draw_preview():
    try:
        context = bpy.context
        scene = context.scene
        if not multicam.sync_active(scene):
            return
        _watch(scene)
        clips = multicam.pinned_strips(scene)
        if clips:
            _letters(context, clips)
    except Exception:
        pass


def draw_header(self, context):
    """Appended to the Sequencer's header: the way out of syncing mode,
    in red, right where the user is working."""
    scene = context.scene
    if not multicam.sync_active(scene):
        return
    _watch(scene)
    row = self.layout.row(align=True)
    row.alert = True
    row.enabled = not blendcap_state['running']
    row.operator("blendcap.multicam_sync_exit", icon='LOOP_BACK',
                 text="Exit Syncing Mode")


def arm():
    try:
        bpy.types.SEQUENCER_HT_header.append(draw_header)
    except Exception as exc:
        print(f"[BlendCap] Syncing mode header button not added: {exc}")
    if bpy.app.background or _HANDLES:
        return
    for region, fn in (('WINDOW', _draw_timeline), ('PREVIEW', _draw_preview)):
        try:
            _HANDLES[region] = bpy.types.SpaceSequenceEditor.draw_handler_add(
                fn, (), region, 'POST_PIXEL')
        except Exception as exc:
            print(f"[BlendCap] Syncing mode drawing ({region}) not started: {exc}")


def disarm():
    try:
        bpy.types.SEQUENCER_HT_header.remove(draw_header)
    except Exception:
        pass
    for region, handle in list(_HANDLES.items()):
        try:
            bpy.types.SpaceSequenceEditor.draw_handler_remove(handle, region)
        except Exception:
            pass
    _HANDLES.clear()
