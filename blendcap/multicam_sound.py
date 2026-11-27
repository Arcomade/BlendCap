"""Sync by Sound: how far apart two clips of one performance are, from
what their microphones recorded.

Cameras filming the same performance hear the same room. Every sound
in it (a step, a clap, a word) arrives at each camera at practically
the same moment, so lining the two sound tracks up lines the pictures
up. This module only MEASURES: `clip_lag` answers "camera B started
recording this many seconds after camera A", or says that it cannot
tell. multicam.py turns the answer into Sync Slide values.

How it measures (measure_lag):

  - each track is reduced to its ONSETS, the moments it gets louder, in
    three pitch bands, a thousand values a second. Two microphones in
    different corners of a room record very different waveforms
    (distance, echoes, the microphone itself, automatic gain) but the
    same onsets, so that is what is compared;
  - the two onset tracks are slid against each other over every offset
    that leaves them a few seconds in common, and the offset where they
    agree best is the candidate;
  - then it has to earn belief, because a wrong answer applied to the
    user's timeline is worse than no answer:
      * the best offset must stand far above the run of the others
        (LAG_Z), and clearly above the runner-up (LAG_RATIO): music
        with a beat agrees with itself every beat, and that ambiguity
        shows as a runner-up nearly as good;
      * WITNESSES: the stretch the clips share is cut into pieces and
        each piece is asked on its own, near the candidate. Enough of
        them must give the same answer (WITNESS_MIN), and pieces that
        confidently give a different one count against it. This is
        what catches two clips with nothing in common: one loud sound
        in each will line up at SOME offset and stand out there, but
        the rest of the take does not follow.
    Anything less is "could not tell", and nothing is changed.

What it cannot do: tell apart two clips with no sound in common (it
refuses), or see a camera whose sound track is offset from its own
picture (rare, and then by a fraction of a frame). Sound takes about
3 ms per metre to travel, so a camera 10 m further away hears things
0.7 frames late at 24 fps: the answer is good to about a frame, and
the merge's own fine sync (pipeline/two_camera_merge.py) takes it from
there.

Reading the sound uses Blender's own audio library (aud); the
measuring is plain numpy. Neither is imported until it is needed.
"""
import os

RATE = 8000                # samples per second the sound is read at
BLOCK = 8                  # samples per onset value
ONSET_RATE = RATE // BLOCK # onset values per second (1000)
BANDS = ((150.0, 600.0), (600.0, 1800.0), (1800.0, 3800.0))
MAX_SEC = 1200.0           # seconds of a clip's sound that are used
ONSET_SMOOTH = 0.008       # s: loudness is averaged over this
ONSET_STEP = 0.020         # s: "louder than this long ago" is an onset
MIN_SHARED = 3.0           # s: an offset must leave at least this in common...
MIN_SHARED_PART = 0.3      # ...or this much of the shorter clip, if that is more
MAX_SHARED_NEEDED = 10.0   # ...but never more than this
LAG_Z = 9.0                # the best offset stands this many deviations above the rest
LAG_RATIO = 1.25           # ...and this far above the runner-up
RUNNER_GUARD = 0.10        # s: the runner-up is looked for further away than this
WITNESS_SEC = (1.5, 8.0)   # s: a witness piece is between these lengths
WITNESS_MAX = 12
WITNESS_WINDOW = 0.25      # s: each witness looks this far either side of the candidate
WITNESS_TOL = 0.012        # s: ...and agrees when it lands this close to it
WITNESS_Z = 4.5            # a witness counts when its own peak stands this clear
WITNESS_MIN = 3            # witnesses that must agree

SILENT_PEAK = 0.002        # a track never louder than this (-54 dB) is dead
SILENT_LOOK = 8.0          # s: read at the start, the middle and the end to find out

_HAS_SOUND = {}
_REAL_SOUND = {}


def read_sound(path, max_sec=MAX_SEC):
    """The clip's sound as one channel at RATE samples a second (a
    float32 array), or None when the file has no sound Blender can
    read."""
    import numpy as np
    import aud
    try:
        snd = aud.Sound(path).rechannel(1)
        try:
            snd = snd.resample(RATE, False)
        except TypeError:
            snd = snd.resample(RATE)
        if max_sec:
            snd = snd.limit(0.0, float(max_sec))
        data = np.asarray(snd.data(), dtype=np.float32).reshape(-1)
    except Exception:
        return None
    return data if data.size >= RATE else None


def has_sound(path):
    """True when the file has a sound track (asked once per file: it
    opens the file's header)."""
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return False
    if key not in _HAS_SOUND:
        ok = False
        try:
            import aud
            specs = aud.Sound(path).specs
            ok = bool(specs and specs[0] > 0 and specs[1] > 0)
        except Exception:
            ok = False
        _HAS_SOUND[key] = ok
    return _HAS_SOUND[key]


def has_real_sound(path, seconds=0.0):
    """True when the file's sound track has something on it: there is
    a track, and it is not dead silence (a camera with its microphone
    off still writes one). A quiet room is real sound. Asked once per
    file. `seconds`: how long the track is, when known: then only
    three short stretches of it are read (start, middle, end), so a
    long clip costs no more than a short one."""
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return False
    if key in _REAL_SOUND:
        return _REAL_SOUND[key]
    ok = False
    try:
        import numpy as np
        import aud
        base = aud.Sound(path).rechannel(1)
        look = SILENT_LOOK
        if seconds > 3 * look:
            spans = [(0.0, look), (seconds / 2 - look / 2, seconds / 2 + look / 2),
                     (seconds - look, seconds)]
        else:
            spans = [(0.0, max(seconds, 3 * look))]
        for a, b in spans:
            loud = np.abs(np.asarray(base.limit(float(a), float(b)).data(),
                                     dtype=np.float32).reshape(-1))
            # the loudest thousandth of it, so a single click in a dead
            # track does not count and steady room tone does
            if loud.size and float(np.quantile(loud, 0.999)) >= SILENT_PEAK:
                ok = True
                break
    except Exception:
        ok = False
    _REAL_SOUND[key] = ok
    return ok


def onsets(samples):
    """(bands, n) array: where the sound gets louder, per pitch band,
    ONSET_RATE values a second. None when there is too little of it."""
    import numpy as np
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if x.size < RATE:
        return None
    x = x - x.mean()
    n_fft = 1 << int(np.ceil(np.log2(x.size)))
    spec = np.fft.rfft(x, n_fft)
    freq = np.fft.rfftfreq(n_fft, 1.0 / RATE)
    n = x.size // BLOCK
    smooth = max(1, int(round(ONSET_SMOOTH * ONSET_RATE)))
    step = max(1, int(round(ONSET_STEP * ONSET_RATE)))
    out = np.zeros((len(BANDS), n))
    for i, (lo, hi) in enumerate(BANDS):
        band = np.fft.irfft(np.where((freq >= lo) & (freq < hi), spec, 0.0), n_fft)[:n * BLOCK]
        loud = np.abs(band).reshape(n, BLOCK).mean(axis=1)
        loud = np.convolve(loud, np.ones(smooth) / smooth, mode="same")
        # compressed, so a camera's automatic gain and a loud clap next
        # to a soft step both matter less
        loud = np.power(loud + 1e-9, 0.3)
        rise = np.zeros(n)
        rise[step:] = loud[step:] - loud[:-step]
        out[i] = np.maximum(rise, 0.0)
    return out


def _slide(a, b):
    """How well a[n + k] matches b[n] for every k from -(len(b) - 1) to
    len(a) - 1, each scored over just the stretch the two share at that
    k (so a long overlap has no head start over a short one). Returns
    (scores, shared lengths), index 0 being k = -(len(b) - 1)."""
    import numpy as np
    na, nb = a.size, b.size
    n = 1 << int(np.ceil(np.log2(na + nb)))
    raw = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    cc = np.concatenate([raw[n - (nb - 1):], raw[:na]])
    k = np.arange(-(nb - 1), na)
    a0, a1 = np.maximum(k, 0), np.minimum(na, nb + k)
    b0, b1 = np.maximum(-k, 0), np.minimum(nb, na - k)
    ca = np.concatenate([[0.0], np.cumsum(a * a)])
    cb = np.concatenate([[0.0], np.cumsum(b * b)])
    ea = ca[a1] - ca[a0]
    eb = cb[b1] - cb[b0]
    return cc / np.sqrt(ea * eb + 1e-12), a1 - a0


def measure_lag(on_a, on_b, fps=24.0):
    """How many seconds camera B started recording AFTER camera A
    (negative: before), from their onset tracks.

    Returns a dict: 'ok' (believed), 'lag' (seconds, the candidate even
    when not believed), 'why' (a short reason when not ok), and the
    evidence: 'z', 'ratio', 'witnesses' (agreeing), 'against'
    (confidently disagreeing), 'pieces' (asked)."""
    import numpy as np
    res = {"ok": False, "lag": 0.0, "why": "", "z": 0.0, "ratio": 0.0,
           "witnesses": 0, "against": 0, "pieces": 0}
    if on_a is None or on_b is None:
        res["why"] = "no sound"
        return res
    na, nb = on_a.shape[1], on_b.shape[1]
    need = int(ONSET_RATE * min(MAX_SHARED_NEEDED,
                                max(MIN_SHARED, MIN_SHARED_PART * min(na, nb) / ONSET_RATE)))
    if min(na, nb) < need:
        res["why"] = "too short"
        return res
    score = None
    used = 0
    for i in range(on_a.shape[0]):
        a = on_a[i] - on_a[i].mean()
        b = on_b[i] - on_b[i].mean()
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        s, shared = _slide(a, b)
        score = s if score is None else score + s
        used += 1
    if not used:
        res["why"] = "silent"
        return res
    # A few seconds in common that happen to match score as high as a
    # whole take that matches: lean towards offsets that leave more in
    # common (cameras are usually started close together).
    score = score / used * np.sqrt(shared / float(shared.max()))
    valid = shared >= need
    if not valid.any():
        res["why"] = "too short"
        return res
    lags = np.arange(-(nb - 1), na)
    masked = np.where(valid, score, -np.inf)
    k0 = int(np.argmax(masked))
    peak = float(score[k0])
    guard = int(RUNNER_GUARD * ONSET_RATE)
    others = valid.copy()
    others[max(0, k0 - guard):k0 + guard + 1] = False
    rest = score[others]
    if rest.size < 100:
        res["why"] = "too short"
        return res
    med = float(np.median(rest))
    mad = float(np.median(np.abs(rest - med))) * 1.4826 + 1e-12
    res["z"] = (peak - med) / mad
    runner = float(rest.max())
    res["ratio"] = peak / runner if runner > 1e-9 else 99.0
    lag = int(lags[k0])                      # in onset values; a[n + lag] ~ b[n]
    res["lag"] = lag / float(ONSET_RATE)

    # --- witnesses: pieces of the shared stretch, each asked on its own
    win = int(WITNESS_WINDOW * ONSET_RATE)
    # inset, so every piece can look the full window both ways
    a0, a1 = max(lag, 0) + win, min(na, nb + lag) - win
    shared_n = max(0, a1 - a0)
    piece = int(ONSET_RATE * min(WITNESS_SEC[1], max(WITNESS_SEC[0], shared_n / ONSET_RATE / 6.0)))
    count = int(min(WITNESS_MAX, shared_n // piece))
    tol = max(WITNESS_TOL, 0.4 / max(fps, 1.0)) * ONSET_RATE
    agree = against = asked = 0
    for j in range(count):
        s0 = a0 + int(round(j * (shared_n - piece) / max(1, count - 1))) if count > 1 else a0
        b_lo = s0 - lag - win
        b_hi = s0 - lag + piece + win
        if b_lo < 0 or b_hi > nb:
            continue
        local = np.zeros(2 * win + 1)
        bands = 0
        for i in range(on_a.shape[0]):
            pa = on_a[i, s0:s0 + piece]
            pb = on_b[i, b_lo:b_hi]
            pa = pa - pa.mean()
            pb = pb - pb.mean()
            if pa.std() < 1e-9 or pb.std() < 1e-9:
                continue
            # pa against every stretch of pb as long as itself
            c = np.correlate(pb, pa, mode="valid")
            cs = np.concatenate([[0.0], np.cumsum(pb * pb)])
            eb = cs[piece:] - cs[:-piece]
            local += c[::-1] / np.sqrt((pa * pa).sum() * eb[::-1] + 1e-12)
            bands += 1
        if not bands:
            continue
        asked += 1
        local /= bands
        kk = int(np.argmax(local))
        away = np.ones(local.size, bool)
        away[max(0, kk - guard // 2):kk + guard // 2 + 1] = False
        m = float(np.median(local[away]))
        sd = float(np.median(np.abs(local[away] - m))) * 1.4826 + 1e-12
        zz = (float(local[kk]) - m) / sd
        if zz < WITNESS_Z:
            continue
        if abs(kk - win) <= tol:
            agree += 1
        else:
            against += 1
    res.update(witnesses=agree, against=against, pieces=asked)
    if res["z"] < LAG_Z:
        res["why"] = "nothing stands out"
    elif res["ratio"] < LAG_RATIO:
        res["why"] = "more than one offset fits"
    elif agree < WITNESS_MIN or against * 2 > agree:
        res["why"] = "the parts of the take do not agree"
    else:
        res["ok"] = True
    return res


def clip_onsets(path):
    """onsets() of a video file's sound, or None."""
    data = read_sound(path)
    return None if data is None else onsets(data)
