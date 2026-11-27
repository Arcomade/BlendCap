"""Merge 2 to 4 synchronised single-camera captures into one better take.

(The file keeps its name from when it took two cameras: the addon, its
cache check and the dev tools all know it by it.)

Run by the addon's Multi-Camera Capture (blendcap/multicam.py), and
usable on its own:

    python -m pipeline.two_camera_merge CAM_A_DIR CAM_B_DIR [CAM_C_DIR
                                        [CAM_D_DIR]] -o OUT_DIR

The CAM_x_DIR are Blendcap_Cache capture folders (each holding
mhr_frames.npz and optionally face_frames.npz), camera A first. OUT_DIR
receives a merged mhr_frames.npz + face_frames.npz that npz_to_bvh
consumes unchanged, plus calibration.npz (read back by Rebuild Finger
Motion and Add Camera).

Notes 1 to 5 were written for two cameras and are told that way; every
one of them holds for more, camera by camera against camera A. Note 6
is what three and four cameras add. A TWO-CAMERA TAKE COMES OUT OF THIS
FILE EXACTLY AS IT DID BEFORE IT TOOK MORE (the same numbers to the
last bit, checked on four real takes): wherever the general form would
round differently, the two-camera form is kept as it was. The one
thing that reaches two-camera takes is note 6c, the floor checked
against the stance, switched on for them afterwards by the user's
call: it left those four takes the same to the last bit as well (their
floors are 1.5 to 2.1 degrees from the stance; it acts above 6).

WHAT IT DOES, and why each piece is the way it is. Every design note
below was paid for with a wrong build first; please read before
"simplifying" any of it.

1. CALIBRATION FROM A T-POSE. The clip must open with the performer in
   a T-pose (arms out, standing still) - standard for a deliberate
   two-camera setup and far easier than a printed target. Gravity comes
   from the FLOOR (a plane fitted to foot contacts), never from
   anatomy: forcing hips->head vertical reads ~17 deg off because the
   head sits forward of the hips, which tilts the whole rig and turns
   walking into vertical drift.
   The floor is only there to fit when the performer moves about on
   it. A capture of someone standing in one spot has two ankles and no
   plane: fitted anyway it gave 24 degrees of roll (measured, a 4 s
   capture). So the fit is tested (floor_trusted), and a take that
   fails is levelled from how the performer STANDS instead
   (level_from_stance): while a body is still its centre of mass is
   over what it stands on, two planted feet are at one height, and
   wherever the feet stood is floor. None of that needs the performer
   to move about. This is balance, not posture: "the body is upright"
   is the anatomy mistake above; "the body is not falling over" is
   physics. On whole takes it lands 1.5 to 2.0 degrees from the
   fitted floor (two methods with nothing in common), and on 2 to 4 s
   pieces about 3 degrees from the whole take's floor, where the
   fitted floor gives 9 to 14 and the camera's own tilt was 5. A take
   with neither a floor nor a still moment is NOT levelled: it keeps
   camera A's frame, as a single-camera take does, and says so. A
   wrong levelling is worse than none.
   How the cameras are turned against each other comes from the BODY
   (camera_turn), all three angles, not from the two floors. It used
   to be one yaw found after levelling each camera by its own floor,
   which made it only as good as two separately fitted planes: on 4 s
   pieces of three takes it moved 19 degrees on average against the
   whole take's answer, and the body's 3.5.

2. PER-LIMB CAMERA SELECTION. A camera that cannot see a limb does not
   go quiet, it invents one smoothly and confidently. Detection is
   therefore geometric (capsule ray-trace: is the sightline blocked by
   the body?) plus a robust temporal test. The temporal test MUST use a
   median filter: a gaussian reference is dragged toward the spike it
   is supposed to expose, and a 118 cm one-frame blow-up then measures
   as a few cm.
   Two more tests are borrowed from single-camera capture, because
   neither can be tripped by fast motion: Clean Tracking's detector
   (the bones change length, or the pose collapses to the rest pose:
   that camera's tracker failed on that frame) and Hold Cropped Limbs'
   (a joint outside the picture was not filmed, so that camera loses
   the limb). A failed tracker frame only LEANS the body toward the
   other camera (TRACKER_FAILED): the test speaks for the whole body,
   and handing the whole frame over took good legs away along with
   the bad arms it was right about (measured: better on two groups of
   flagged frames, worse on the third; the lean is better on all
   three). Both have to be read HERE:
   the merged take is rebuilt on fixed bone lengths and keeps one
   camera's keypoints, so neither survives the merge.
   There is NO bridging step. An earlier build drew a straight line
   through any stretch of up to ten frames where both cameras failed
   the temporal test and disagreed by 6 cm. A fast limb does both
   honestly (it leaves its own median path, and two cameras that are
   not synced to the instant disagree about it), so that step covered
   a fifth of the frames of a fast take and flattened one fast move
   in eight (a kick both cameras saw travel 125 cm came out at 50).
   Removing it cost nothing measurable on the slower take.

3. MOTION BLENDING, NOT POSE BLENDING. Two cameras can disagree about a
   limb by 20 cm+. Cross-fading their absolute poses teleports the limb
   (measured 14.5 cm in one frame) - a pop NEITHER camera contains. So
   blend each camera's frame-to-frame CHANGE and integrate it: the
   per-frame step is then bounded by the two sources and a pop is
   impossible by construction, while a slow pull toward the blended
   pose stops it drifting off the evidence.

4. ANISOTROPIC FUSION, FOR POSITION AND FOR ROTATION. Averaging two
   positions is the wrong operator. Each camera is precise ACROSS its
   image and vague ALONG its view axis; at ~90 deg apart each is vague
   exactly where the other is sharp, so a 50/50 average is WORSE than
   the better camera on both horizontal axes. Weight per DIRECTION
   instead. Rotation has the same blindness in the opposite sense (a
   camera sees rotation ABOUT its view axis well and rotation away from
   itself poorly), which is why a flat average loses forward lean to
   whichever camera happens to be front-on, and lands every limb
   BETWEEN the two cameras' versions of it: wrong in both pictures.
   The hips, the spine, and the upper and lower arms and legs are
   fused per rotation axis (axis_precision, q_blend_axes). Measured by
   putting the merged skeleton back in front of each camera (devtools
   two_camera_reproject.py): the flat blend was no closer to the two
   cameras' keypoints than one camera on its own; per axis closed
   about half of the gap to the best these captures allow on one take
   and a quarter on another, with slightly LESS jitter. Feet, hands,
   collarbones, neck and head keep the flat blend: the two cameras'
   disagreement about them showed no such pattern, so there is nothing
   for the weighting to use.

5. PLANT FREEZE. While a foot is planted, stop switching that leg. A
   handover migrates the whole limb over ~0.5 s, and on a planted foot
   that migration IS sliding. A stationary foot tells neither camera
   anything new, so there is nothing to gain and a seam to lose.

6. THREE AND FOUR CAMERAS. Everything above is done for every camera
   against camera A (its turn from the body, its fine sync, its place
   from the T-pose), and every blend takes all of them at once: a limb
   is shared out by how bad each camera looks against the BEST one
   (camera_shares; for two cameras that is the old formula), and each
   rotation axis by who can see it (q_blend_axes_many). Measured on
   the one four-camera take there is (four different cameras around a
   small room; lying, rolling, crouching; devtools
   multi_camera_reproject.py): the merged skeleton sits 3.7 cm from
   the four cameras' own keypoints, where cameras A and B alone give
   6.3 and the best single camera 5.1, with a third of the pops.
   Three things came with them (a and b need more than two cameras;
   c was found with four and holds for two):
   a. FINE SYNC FROM EVERY PAIR (sync_between_all). One pair has to
      be very sure before it corrects anything. Three pairs that add
      up check each other, so a camera whose own pairing with camera A
      is poor is still placed, by the others. On that take camera A
      was the poor one and no pair with it passed; all together put
      the cameras 1.5, 1.3 and -0.2 frames from it. Correcting that
      took the merged skeleton from 4.1 to 3.8 cm off the keypoints
      and kept more of the fast moves' speed (81 to 87 percent).
   b. WHICH CAMERA IS ALONE (cameras_apart). Two cameras that disagree
      cannot tell which is wrong; three can. A camera far (AGREE_NEAR
      .. AGREE_FAR) from where the others put a joint loses the limb.
      Set loose on purpose: at 5..15 cm it took good ankles away
      (cameras always differ by a few cm), at 10..30 it changes
      nothing but the gross cases (one camera 6 cm off its own picture
      for the last seconds of that take, because another, too close
      and cut off at the knees, was being believed: 1.7 with this).
   c. THE FLOOR IS CHECKED AGAINST THE STANCE (FLOOR_STANCE_APART). On
      that take two cameras' fitted floors agreed with each other and
      were 10 degrees from how the performer stands, and the T-pose
      said the stance was right (legs 3 degrees off vertical under it,
      8 under the floor). Ankles on a take spent on the floor are not
      floor contacts. Two cameras are checked the same way (the two
      floors agreeing with each other, floor_trusted, does not catch
      this: on that take cameras A and B's floors were 6.2 degrees
      apart and both wrong). Merged from cameras A and B alone, that
      take's T-pose has the legs 8.5 degrees off vertical under the
      floor and 1.9 under the stance. A take with no stance to read
      keeps its floor.
   Camera A still matters more than the others: the take is in its
   frame, the T-pose is found in it, and which foot is planted is read
   from it. A poor camera A costs less than it did (it is one of
   four), not nothing.

KNOWN LIMITS: validated on one take, one subject, ~90 deg rig.
Thresholds are in centimetres and were tuned at ~5 m subject distance;
they should be scaled by focal/distance before this meets footage shot
at another scale. Whole-frame sync is a parameter (the clips are lined
up by hand); what is left of it, a frame or so and never a whole
number, is measured and corrected here (measure_sync).
"""
import argparse
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, median_filter

from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.body_skeleton import load_skeleton  # noqa: E402
from pipeline.bvh_math import compute_tpose_globals  # noqa: E402
from pipeline.cropped_limbs import keypoints_inside  # noqa: E402
from pipeline.tracking_repair import _detect_bad_frames  # noqa: E402

# ---------------------------------------------------------------- skeleton
LETTERS = "ABCD"                       # the cameras, camera A first
N_JOINTS = 127
HIPS, HEAD = 1, 123
HIP_L, KNEE_L, ANK_L = 2, 3, 4
HIP_R, KNEE_R, ANK_R = 18, 19, 20
SHO_R, ELB_R, HAND_R = 39, 40, 41
SHO_L, ELB_L, HAND_L = 75, 76, 77
NECK = 110
TARGETS = [HIP_R, HIP_L, KNEE_R, KNEE_L, ANK_R, ANK_L,
           SHO_R, SHO_L, ELB_R, ELB_L, HAND_R, HAND_L]
CHAINS = {"arm R": [SHO_R, ELB_R, HAND_R], "arm L": [SHO_L, ELB_L, HAND_L],
          "leg R": [HIP_R, KNEE_R, ANK_R], "leg L": [HIP_L, KNEE_L, ANK_L]}
# each of those joints' 2D keypoint (MHR70 layout), for "was it filmed"
KP_OF = {SHO_L: 5, SHO_R: 6, ELB_L: 7, ELB_R: 8, HIP_L: 9, HIP_R: 10,
         KNEE_L: 11, KNEE_R: 12, ANK_L: 13, ANK_R: 14,
         HAND_R: 41, HAND_L: 62}
TORSO = [34, 35, 36, 37, 38, 74, NECK] + list(range(111, N_JOINTS))
# body as capsules, for the sightline test. radii in metres.
CAPSULES = [((HIPS, 37), 0.135), ((HIP_R, HIP_L), 0.100),
            ((SHO_R, SHO_L), 0.080), ((NECK, HEAD), 0.095),
            ((SHO_R, ELB_R), 0.052), ((SHO_L, ELB_L), 0.052),
            ((ELB_R, HAND_R), 0.045), ((ELB_L, HAND_L), 0.045),
            ((HIP_R, KNEE_R), 0.085), ((HIP_L, KNEE_L), 0.085),
            ((KNEE_R, ANK_R), 0.060), ((KNEE_L, ANK_L), 0.060)]

F_FLIP = np.diag([1.0, -1.0, -1.0])   # global_rots live in a y-up frame
UP = np.array([0.0, -1.0, 0.0])       # camera frame: +y is DOWN
ALPHA = 0.08                          # slow pull, ambiguous moments
ALPHA_SURE = 0.30                     # fast pull, confident decisions
# Fine sync (measure_sync). Times are in seconds so they mean the same
# at any capture rate.
SYNC_SMOOTH_SEC = 0.0625   # both cameras' signals get this, identically
SYNC_MAX_SEC = 0.25        # furthest apart the clips are looked for
SYNC_WINDOW_SEC = 3.0      # the offset is read in windows this long...
SYNC_STEP_SEC = 1.0        # ...this far apart
SYNC_AGREE = 0.5           # the take counts when the cameras agree this well
SYNC_TRUST = 0.85          # a window counts when the cameras agree this well
SYNC_WITNESS = 0.35        # frames: arms / legs / left / right must agree
SYNC_SCATTER = 0.25        # frames: windows may sit this far off the model
SYNC_SHAPE_WINDOWS = 10    # windows needed before a step or a drift is believed
SYNC_MIN = 0.2             # frames: below this the clips are in step
# How bad a camera counts on a frame where its tracker failed (Clean
# Tracking's detector), on the 0..1 scale the other tests use. 0.075
# leaves it a quarter of the say when the other camera is clean.
TRACKER_FAILED = 0.075
# A camera that did not see the performer on a frame at all (a person of
# a Multi-Camera & Multi-Person take, hidden or lost in that camera: its
# capture carries an `absent` flag per frame, and what it holds there is
# filled in, people_tracks.extract_person_cameras). The other cameras
# have the frame to themselves. No ordinary capture has the flag.
CAMERA_ABSENT = 1.0
# A camera with NO VIEW of a joint (the joint is outside its picture, or
# the performer was not seen by it at all on that frame) counts this
# much worse again than the worst any other test can make a camera
# look. Every other test is a judgement about a camera that is looking
# at the limb, and they all top out at 1.0: on a fast move the cameras
# that SEE the limb are marked down by them (a quick kick is far from
# its own recent path, the same as a glitch is), and without this they
# tie with the cameras that are only guessing at it. A tie shares the
# limb evenly (camera_shares), so a kick two cameras saw clearly came
# out half way to the folded leg the other two invented (the user's
# take of 2026-11-24, a low sweep around frame 275: all four cameras at
# 1.00 for a dozen frames, a quarter of the leg each). More than the
# 0.15 at which camera_shares gives a camera nothing: a camera with no
# view gets no share while any camera has one.
NO_VIEW = 0.5
# The cameras that SEE a limb joint are ranked by how well each one's
# version of it fits what the others see (fits_others): a camera whose
# version is this much further (metres) from the others' sightlines
# than the best camera's is, starts to lose the joint, and has lost it
# at the second number. Where two or more cameras see a joint this
# REPLACES the "far from its own recent path" tests: those cannot tell
# a fast move from a glitch, mark every camera that sees a quick limb
# as bad as can be, and leave them tied; another camera's picture can
# tell. (The user's take of 2026-11-24, performer 1's sweep at frames
# 268 to 284, where only cameras A and C saw the leg and took turns
# being right: knees 8.5 -> 6.0 cm from where the seeing cameras'
# sightlines meet, ankles 7.2 -> 5.3 and at worst 38 -> 24; the leg at
# frame 274 is 0.92 of straight, as camera A had it, where the even
# share of two tied cameras gave 0.64. Added as one more test beside
# the others it changed nothing, for the reason above. Five
# one-performer takes come out the same either way.)
FIT_NEAR, FIT_FAR = 0.03, 0.15
# Levelling: when the floor fitted to the foot contacts is believed
# (floor_trusted). Set on pieces of three takes cut to 2..20 s: of the
# floors these let through, the worst was 5.4 degrees off the whole
# take's and the average 2.4; of the ones they stop, the average was
# over 20. Three takes, two rooms: revisit with more footage.
FLOOR_MIN_SPREAD = 0.06     # metres the contacts must reach, narrow way
FLOOR_MAX_APART = 10.0      # degrees the two cameras' floors may differ
# The fitted floor is also checked against how the performer stands
# (level_from_stance), and is not used when the two are further apart
# than this (degrees). Any number of cameras. One take's worth of
# evidence for the number: three takes where the floor is right sit
# 1.5 to 2.1 from the stance, the one where it is wrong 10.
FLOOR_STANCE_APART = 6.0
# What the console says when a take could not be levelled at all (two
# lines). Console only, by the user's call: not in the result popup.
LEVEL_NOT = ("Levelling: the floor could not be measured in this take, so "
             "it has camera A's tilt.\n"
             "It takes the performer standing still for a moment, or moving "
             "across the floor. Camera Angle Offset can level the take.")

# Levelling without a floor to fit (level_from_stance). Speeds in m/s
# and times in seconds, so they mean the same at any capture rate.
UNDER_L, TOE_L, UNDER_R, TOE_R = 6, 8, 22, 24   # under the ankle, toe
STANCE_SPEED = 0.29       # a foot slower than this is standing
STANCE_MIN_SEC = 0.15
STILL_SPEED = 0.15        # a centre of mass slower than this is balanced
STILL_MIN_SEC = 0.2
# where along the foot the weight sits when standing, from under the
# ankle (0) to the toe (1): a little ahead of the ankle
WEIGHT_ALONG_FOOT = 0.35
# How far each piece of evidence is believed, in metres. The first two
# are measured: on the 23 still two-footed stretches of three whole
# takes, against the take's fitted floor, the centre of mass sat 2.9
# cm (rms) off the line between the feet and one foot read 2.7 cm
# above the other. (The two cameras' readings of one stretch move
# together, r = 0.7 and 0.9, which is why they share one say below.)
# A foot that stood somewhere is located to about 2 cm.
SIG_BALANCE = 0.03
SIG_FEET_LEVEL = 0.03
SIG_STOOD = 0.02
# Balance only speaks for a body that is STANDING on its feet. A leg
# counts as stood on when it is at least this straight (hip to ankle
# against thigh plus shin: 0.85 is a knee bent to about 120 degrees),
# which leaves out sitting and kneeling, where the weight is on
# something else. Deep stances are left out with them; that costs
# evidence, never correctness.
LEG_STRAIGHT = 0.85
# ...and when the line from the centre of mass to the feet is within
# this of the camera's own "down" (degrees): lying or sitting on the
# floor points it sideways, and no camera is tilted this far.
STANDING_CONE = 35.0
# A planted foot this much higher than the other (metres) is not on
# the floor: it is held up, and the body stands on the other one.
FOOT_HELD_UP = 0.10
# What is assumed of camera A before any evidence, in degrees: a camera
# on a stand is rarely rolled much and is often pitched. Weak on
# purpose: it only decides what the evidence leaves open (someone
# standing with their feet together says little about roll), and with
# no evidence at all it does not pass LEVEL_MAX_SD.
CAMERA_ROLL_SD = 5.0
CAMERA_PITCH_SD = 15.0
# With no still stretch to read balance from, where the feet stood has
# to hold "down" at least this tightly (degrees) or the take is not
# levelled at all. A still stretch always levels: what it leaves open
# (one stance holds the tilt along the line between the feet to about
# 6 degrees) stays near the camera's own, which is no worse than not
# levelling.
LEVEL_MAX_SD = 5.0
# Share of the body's mass in each segment and where its centre sits
# between two joints (de Leva 1996, rounded; the trunk is added in
# centre_of_mass). Only where the mass is over the feet matters here,
# and that is not sensitive to these.
_MASS = [
    (0.069, NECK, HEAD, 0.5),
    (0.027, SHO_L, ELB_L, 0.58), (0.027, SHO_R, ELB_R, 0.58),
    (0.016, ELB_L, HAND_L, 0.46), (0.016, ELB_R, HAND_R, 0.46),
    (0.006, HAND_L, HAND_L, 0.0), (0.006, HAND_R, HAND_R, 0.0),
    (0.142, HIP_L, KNEE_L, 0.41), (0.142, HIP_R, KNEE_R, 0.41),
    (0.043, KNEE_L, ANK_L, 0.45), (0.043, KNEE_R, ANK_R, 0.45),
    (0.014, ANK_L, TOE_L, 0.5), (0.014, ANK_R, TOE_R, 0.5),
]
_MASS_TRUNK = 0.435
_MASS_JOINTS = sorted({HIPS, HIP_L, HIP_R, SHO_L, SHO_R, UNDER_L, UNDER_R}
                      | {j for _w, a, b, _t in _MASS for j in (a, b)})

# Three or four cameras (cameras_apart): a camera whose version of a
# joint is further than AGREE_NEAR from where the cameras agree it is,
# across its own picture, starts losing that limb, and has lost it at
# AGREE_FAR. Metres. AGREE_SCALE is how far off a camera has to be to
# lose its say in WHERE they agree.
AGREE_NEAR = 0.10
AGREE_FAR = 0.30
AGREE_SCALE = 0.05
AGREE_ROUNDS = 4

# Per-axis fusion of a joint's rotation (see q_blend_axes). Each joint
# lists the joints its rotation carries around: a rotation is only
# visible to a camera through where it moves those.
K_AXIS = 0.85     # how much less a camera is believed along its sightline
EPS_AXIS = 0.03   # an axis nothing reveals is shared by the limb weight
SPINE = [34, 35, 36, 37]
AXIS_LEVERS = {
    34: [35, 36, 37, NECK, SHO_R, SHO_L],
    35: [36, 37, NECK, SHO_R, SHO_L],
    36: [37, NECK, SHO_R, SHO_L],
    37: [NECK, SHO_R, SHO_L],
    SHO_R: [ELB_R, HAND_R], ELB_R: [HAND_R],
    SHO_L: [ELB_L, HAND_L], ELB_L: [HAND_L],
    HIP_R: [KNEE_R, ANK_R], KNEE_R: [ANK_R],
    HIP_L: [KNEE_L, ANK_L], KNEE_L: [ANK_L],
}


# ------------------------------------------------------------- visibility
def _seg_seg_dist(p1, p2, q1, q2):
    """Min distance between segments p1p2 and q1q2, per frame."""
    d1, d2, r = p2 - p1, q2 - q1, p1 - q1
    a = np.einsum('fi,fi->f', d1, d1)
    e = np.einsum('fi,fi->f', d2, d2)
    f = np.einsum('fi,fi->f', d2, r)
    c = np.einsum('fi,fi->f', d1, r)
    b = np.einsum('fi,fi->f', d1, d2)
    den = np.maximum(a * e - b * b, 1e-12)
    s = np.clip((b * f - c * e) / den, 0.0, 1.0)
    t = np.clip((b * s + f) / np.maximum(e, 1e-12), 0.0, 1.0)
    s = np.clip((b * t - c) / np.maximum(a, 1e-12), 0.0, 1.0)
    return np.linalg.norm(p1 + d1 * s[:, None] - q1 - d2 * t[:, None], axis=1)


def sightline_clearance(world):
    """For each target joint, how far the camera's sightline clears the
    nearest blocking body part. Negative = blocked. The camera sits at
    the origin of its own frame, so this needs no calibration."""
    n = len(world)
    origin = np.zeros((n, 3))
    out = np.full((n, len(TARGETS)), 10.0)
    for ti, j in enumerate(TARGETS):
        # stop short of the joint so its own flesh is not the occluder
        end = origin + (world[:, j] - origin) * 0.93
        best = np.full(n, 10.0)
        for (a, b), rad in CAPSULES:
            if j in (a, b):
                continue
            best = np.minimum(best, _seg_seg_dist(origin, end,
                                                  world[:, a], world[:, b]) - rad)
        out[:, ti] = best
    return out


# ------------------------------------------------------------ quaternions
def _n(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _mul(a, b):
    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                     w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], axis=-1)


def _conj(q):
    return np.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], -1)


def _continuous(q):
    flip = np.concatenate([[False], (q[1:] * q[:-1]).sum(-1) < 0])
    return q * np.where(np.cumsum(flip) % 2 == 1, -1.0, 1.0)[:, None]


def q_blend(Ma, Mb, w, alpha=None):
    """q_blend_many for two streams, w being the second one's share."""
    return q_blend_many([Ma, Mb], [1 - w, w], alpha)


def q_blend_many(Ms, Ws, alpha=None):
    """Blend rotation streams without ever producing a pop. Ms: one
    (n, 3, 3) stream per camera; Ws: each one's share, (n,), summing
    to 1.

    Blends MOTION (frame-to-frame change) and integrates it, so the
    output's per-frame step is bounded by the inputs; a slow pull
    toward the blended pose keeps it anchored to the evidence. Every
    axis gets the same weight; q_blend_axes_many is the per-axis
    version."""
    qa = _continuous(_n(Rot.from_matrix(Ms[0]).as_quat()))
    dA = _mul(_conj(qa[:-1]), qa[1:])
    tgt = qa * Ws[0][:, None]
    d = dA * Ws[0][1:][:, None]
    for M, w in zip(Ms[1:], Ws[1:]):
        qb = _continuous(_n(Rot.from_matrix(M).as_quat()))
        qb = np.where(((qa * qb).sum(-1) < 0)[:, None], -qb, qb)
        tgt = tgt + qb * w[:, None]
        dB = _mul(_conj(qb[:-1]), qb[1:])
        dB = np.where(((dA * dB).sum(-1) < 0)[:, None], -dB, dB)
        d = d + dB * w[1:][:, None]
    tgt, d = _n(tgt), _n(d)

    al = np.full(len(qa), ALPHA) if alpha is None else np.asarray(alpha)
    out = np.empty_like(qa)
    out[0] = tgt[0]
    for f in range(1, len(qa)):
        q = _mul(out[f - 1], d[f - 1])
        t = tgt[f] if (q @ tgt[f]) >= 0 else -tgt[f]
        out[f] = _n(q * (1 - al[f]) + t * al[f])
    return Rot.from_quat(out).as_matrix()


def _skew(v):
    """Cross-product matrices: _skew(v) @ w == v x w, per frame."""
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], -1),
                     np.stack([v[:, 2], z, -v[:, 0]], -1),
                     np.stack([-v[:, 1], v[:, 0], z], -1)], -2)


def axis_precision(sight, levers):
    """How well one camera can tell a joint's rotation, per axis, per
    frame, as a 3x3 matrix in the frame `sight` and `levers` are in.

    A rotation is visible only through what it moves. Turning a joint
    by a small w moves a point at lever d by w x d, and the camera
    reads that movement well ACROSS its image and poorly ALONG its
    sightline. Summed over the joints the rotation carries, that is

        P = sum  [d]x^T (I - K s s^T) [d]x

    which says, without any special cases: a bone lying across the
    image has its in-picture swing seen well and its tilt toward the
    camera seen badly; a bone pointing AT the camera is the other way
    round (both swings move its tip across the image); and a twist
    about the bone is seen only if something further down the limb
    sticks out sideways (a bent elbow). EPS_AXIS keeps the unseen axes
    solvable; they end up shared by the plain limb weight."""
    Q = np.eye(3) - K_AXIS * sight[:, :, None] * sight[:, None, :]
    P = np.zeros((len(sight), 3, 3))
    tot = np.zeros(len(sight))
    for d in levers:
        X = _skew(d)
        P += np.einsum('fji,fjk,fkl->fil', X, Q, X)
        tot += (d * d).sum(-1)
    return P / np.maximum(tot, 1e-9)[:, None, None] + EPS_AXIS * np.eye(3)


def q_blend_axes(Ma, Mb, w, alpha, PA, PB):
    """q_blend_axes_many for two streams, w being the second one's
    share."""
    return q_blend_axes_many([Ma, Mb], [1 - w, w], alpha, [PA, PB])


def q_blend_axes_many(Ms, Ws, alpha, Ps):
    """q_blend_many, with each rotation AXIS weighted by who can see it.

    Ms are the cameras' orientations of one joint in a shared frame;
    Ps say how well each camera sees a rotation about each axis of
    that frame (axis_precision). The limb weights Ws still have the
    last word: a camera with the whole weight is the result, and in
    between each axis leans toward the cameras that see it. Motion is
    blended the same way (a punch thrown straight at camera A is
    nearly invisible to A and plain to a camera beside it, so its
    motion comes from that one), then integrated with the slow pull,
    so the no-pop property of q_blend_many is kept.

    The pose it pulls toward is the weighted-least-squares answer to
    "which rotation best fits every camera, given what each can see".
    With two cameras that is one step from the first camera's (how the
    two-camera merge was built and measured, kept to the last bit).
    With more, one camera can be far from the rest, and far is where a
    single step from it goes wrong: so it starts from the plain blend
    of them all and is solved from there."""
    rs = [Rot.from_matrix(M) for M in Ms]
    ra = rs[0]
    ws = [w[:, None, None] for w in Ws]
    tot = ws[0] * Ps[0]
    for w, P in zip(ws[1:], Ps[1:]):
        tot = tot + w * P
    tot = tot + 1e-9 * np.eye(3)
    # each camera's share of each axis
    Ss = [np.linalg.solve(tot, w * P) for w, P in zip(ws[1:], Ps[1:])]

    dA = (ra[1:] * ra[:-1].inv()).as_rotvec()
    dv = dA
    for r, S in zip(rs[1:], Ss):
        dv = dv + np.einsum('fij,fj->fi', S[1:],
                            (r[1:] * r[:-1].inv()).as_rotvec() - dA)
    d = Rot.from_rotvec(dv).as_quat()

    if len(rs) == 2:
        rel = (rs[1] * ra.inv()).as_rotvec()
        tgt = (Rot.from_rotvec(np.einsum('fij,fj->fi', Ss[0], rel))
               * ra).as_quat()
    else:
        qa = ra.as_quat()
        q0 = qa * Ws[0][:, None]
        for r, w in zip(rs[1:], Ws[1:]):
            q = r.as_quat()
            q0 = q0 + np.where(((qa * q).sum(-1) < 0)[:, None], -q, q) \
                * w[:, None]
        near = Rot.from_quat(_n(q0))
        S0 = np.linalg.solve(tot, ws[0] * Ps[0])
        for _ in range(2):
            step = np.einsum('fij,fj->fi', S0,
                             (ra * near.inv()).as_rotvec())
            for r, S in zip(rs[1:], Ss):
                step = step + np.einsum('fij,fj->fi', S,
                                        (r * near.inv()).as_rotvec())
            near = Rot.from_rotvec(step) * near
        tgt = near.as_quat()

    al = np.full(len(tgt), ALPHA) if alpha is None else np.asarray(alpha)
    out = np.empty_like(tgt)
    out[0] = tgt[0]
    for f in range(1, len(tgt)):
        q = _mul(d[f - 1], out[f - 1])
        t = tgt[f] if (q @ tgt[f]) >= 0 else -tgt[f]
        out[f] = _n(q * (1 - al[f]) + t * al[f])
    return Rot.from_quat(out).as_matrix()


def _deviation(p):
    """Distance from where this stream plainly ought to be. MEDIAN
    filter, never a mean: a mean reference is dragged toward the spike
    it is meant to expose."""
    return np.linalg.norm(p - median_filter(p, size=(9, 1), mode="nearest"),
                          axis=-1)


def _rot_deviation(g):
    q = _continuous(_n(Rot.from_matrix(g).as_quat()))
    med = _n(median_filter(q, size=(9, 1), mode="nearest"))
    return np.degrees(2 * np.arccos(np.abs((q * med).sum(-1)).clip(0, 1)))


# ------------------------------------------------------------- T-pose etc
def find_tpose(jc):
    """Longest still, arms-extended run in the first 15% of the clip."""
    pa = jc - jc[:, HIPS:HIPS + 1, :]
    sR = np.linalg.norm(pa[:, HAND_R] - pa[:, SHO_R], axis=1)
    sL = np.linalg.norm(pa[:, HAND_L] - pa[:, SHO_L], axis=1)
    spd = np.linalg.norm(np.gradient(pa[:, [HAND_R, HAND_L]], axis=0),
                         axis=2).max(axis=1)
    ok = ((sR > 0.92 * np.percentile(sR, 99))
          & (sL > 0.92 * np.percentile(sL, 99))
          & (spd < np.percentile(spd, 40)))
    best, cur = (0, 1), 0
    for f in range(int(0.15 * len(jc))):
        cur = cur + 1 if ok[f] else 0
        if cur > best[1] - best[0]:
            best = (f - cur + 1, f + 1)
    return best


def floor_plane(world_feet):
    """The plane through the foot contacts: (normal, spread). The
    normal points down (camera frame, +y down). spread is how far the
    contacts reach, in metres (one standard deviation): along the
    plane's long way, its narrow way, and off the plane."""
    pts = world_feet.reshape(-1, 3)
    sel = pts[pts[:, 1] > np.percentile(pts[:, 1], 80)]   # y is DOWN
    if len(sel) < 3:
        # a frame or two: nothing to fit, and floor_trusted will say so
        return -UP, np.zeros(3)
    _, s, vt = np.linalg.svd(sel - sel.mean(0))
    nrm = vt[2] / np.linalg.norm(vt[2])
    if nrm[1] < 0:
        nrm = -nrm
    return nrm, s / np.sqrt(len(sel))


def _level_to(nrm):
    """The rotation that stands `nrm` (pointing down) straight down,
    and how far it turns, in degrees."""
    down = -UP
    v = np.cross(nrm, down)
    sn = np.linalg.norm(v)
    if sn < 1e-9:
        return np.eye(3), 0.0
    ang = np.arctan2(sn, float(nrm @ down))
    return Rot.from_rotvec(v / sn * ang).as_matrix(), np.degrees(ang)


def gravity_from_floor(world_feet):
    """Vertical from a plane fitted to foot contacts. A floor is level
    by definition; anatomy is not (hips->head reads ~17 deg off)."""
    nrm, spread = floor_plane(world_feet)
    G, deg = _level_to(nrm)
    return G, deg, spread[2]


def camera_turn(pa, pb, rounds=6):
    """How camera B is turned against camera A, from the body: the
    rotation R with pa ~ R pb, and the mean distance it leaves
    (metres). pa / pb are (f, k, 3): the same joints at the same
    moments as each camera saw them, relative to the hips.

    All three angles are free, and nothing but the body is used: it is
    in every frame of every take, which the floor is not (see the note
    at the top). Fitted to the mean DISTANCE, not the mean square (a
    few rounds of reweighting): a frame where one camera invents a
    limb must not pull it."""
    P, Q = pa.reshape(-1, 3), pb.reshape(-1, 3)
    w = np.ones(len(P))
    R, res = np.eye(3), np.zeros(len(P))
    for _ in range(rounds):
        U, _, Vt = np.linalg.svd((Q * w[:, None]).T @ P)
        d = 1.0 if np.linalg.det(Vt.T @ U.T) >= 0 else -1.0
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        res = np.linalg.norm(P - Q @ R.T, axis=1)
        w = 1.0 / np.maximum(res, 0.01)
    return R, float(res.mean()) if len(res) else 0.0


def floor_trusted(planeA, planeB, R):
    """Whether the floor fitted to camera A's foot contacts can level
    the take: (yes or no, degrees the two cameras' floors are apart).

    planeA / planeB: floor_plane() of each camera's contacts. R:
    camera_turn(). Two things have to hold. The contacts must cover
    some floor, in both cameras: feet that stay in one spot give a
    line or a point, and a plane through those is noise. And the two
    cameras must have found the SAME floor: it is one floor, so two
    fits that differ (compared through the body's rotation, which
    does not use the floor) are wrong at least once."""
    (nA, sA), (nB, sB) = planeA, planeB
    apart = float(np.degrees(np.arccos(np.clip(nA @ (R @ nB), -1.0, 1.0))))
    ok = (min(float(sA[1]), float(sB[1])) >= FLOOR_MIN_SPREAD
          and apart <= FLOOR_MAX_APART)
    return bool(ok), apart


def centre_of_mass(P):
    """The body's centre of mass, (f, 3), from (f, 127, 3) joints."""
    c = _MASS_TRUNK * 0.25 * (P[:, HIP_L] + P[:, HIP_R]
                              + P[:, SHO_L] + P[:, SHO_R])
    total = _MASS_TRUNK
    for w, a, b, t in _MASS:
        c = c + w * (P[:, a] + t * (P[:, b] - P[:, a]))
        total += w
    return c / total


def _runs(mask, least):
    """[(start, end)] of the runs of True at least `least` long."""
    out, f, n = [], 0, len(mask)
    while f < n:
        if mask[f]:
            g = f
            while g < n and mask[g]:
                g += 1
            if g - f >= least:
                out.append((f, g))
            f = g
        else:
            f += 1
    return out


def _all_cameras(worlds, turns, joints):
    """The joints as one set of points in camera A's frame, each camera
    believed across its picture and not along its sightline. Used only
    to tell WHEN something is still: one camera's own depth jitters
    enough to make a planted foot look like it is moving.

    worlds: each camera's (f, 127, 3) joints in its own frame, camera
    A first. turns: each camera's rotation into camera A's frame
    (camera_turn; the identity for camera A)."""
    wA = worlds[0]
    eye = np.eye(3)
    tot = rhs = None
    for w, R in zip(worlds, turns):
        if w is wA:
            p = w[:, joints]
            d = p / np.maximum(np.linalg.norm(p, axis=-1, keepdims=True),
                               1e-9)
        else:
            p = np.einsum('ij,fkj->fki', R, w[:, joints])
            p = p + np.median(wA[:, HIPS] - np.einsum(
                'ij,fj->fi', R, w[:, HIPS]), axis=0)
            s = w[:, joints]
            d = np.einsum('ij,fkj->fki', R, s / np.maximum(
                np.linalg.norm(s, axis=-1, keepdims=True), 1e-9))
        M = eye - 0.95 * d[..., :, None] * d[..., None, :]
        part = np.einsum('fkij,fkj->fki', M, p)
        tot = M if tot is None else tot + M
        rhs = part if rhs is None else rhs + part
    out = np.zeros((len(wA), N_JOINTS, 3))
    out[:, joints] = np.linalg.solve(tot, rhs[..., None])[..., 0]
    return out


def level_from_stance(worlds, poses, turns, fps):
    """Which way is down, without a floor to fit: from how the
    performer stands. Returns (down, sd, what): the direction in camera
    A's frame, how loosely the evidence holds it (degrees: the weaker
    of the two ways it can tilt), and a few words for the log. down is
    None when there is nothing to go by (no still stretch, and too
    little stood on: LEVEL_MAX_SD).

    worlds: each camera's (f, 127, 3) joints in its own frame, camera
    A first. poses: the same relative to the hips. turns: each
    camera's rotation into camera A's frame (camera_turn; the identity
    for camera A).

    Three things are known about a standing body whether or not it
    goes anywhere:
      1. While it is still and standing, its centre of mass is over
         what it stands on: over the one planted foot, or over the
         LINE between two planted feet (where along that line depends
         on which leg has the weight, so that part says nothing).
         Standing only: a body that sits, kneels or lies has its
         weight on something else (LEG_STRAIGHT, STANDING_CONE).
      2. Two planted feet are at the same height.
      3. Wherever a foot stood is floor: all the places feet stood,
         through the take, are in one plane.
    (And, weakly, that camera A stands roughly level: CAMERA_ROLL_SD.)
    Each is a direction (or a set of points) that "down" has to be
    square to, believed to a known number of cm, so they add up as
    weighted evidence and the answer is the direction they leave
    least room around. 1 and 2 are read from each camera's own POSE
    (turned into camera A's frame): no depth, no lens, no travel. 3 uses each camera's own positions. On a whole take 3
    carries it and this agrees with the fitted floor; on a few seconds
    of standing 1 and 2 do.

    A foot held still in the air is not floor: the floor is the
    LOWEST level the feet stood at, and stances above it lose their
    say (same idea as the fitted floor's lowest fifth, without its
    weakness of picking the far end of a tilted floor)."""
    wA = worlds[0]
    n = len(wA)
    none = (None, 90.0, "nothing was still and nowhere was stood")
    if n < 3:
        return none
    P = _all_cameras(worlds, turns, _MASS_JOINTS + [TOE_L, TOE_R])
    # every camera's pose and positions, in camera A's frame
    poses = [poses[0]] + [np.einsum('ij,fkj->fki', R, p)
                          for p, R in zip(poses[1:], turns[1:])]
    placed = [wA] + [np.einsum('ij,fkj->fki', R, w)
                     for w, R in zip(worlds[1:], turns[1:])]
    say_each = 1.0 / len(poses)
    # ---- when each foot stood
    sig = max(1.0, 0.08 * fps)
    stances = []
    for foot, j in ((0, UNDER_L), (1, UNDER_R)):
        p = gaussian_filter1d(P[:, j], sig, axis=0)
        v = np.concatenate([[0.0], np.linalg.norm(np.diff(p, axis=0),
                                                  axis=1)]) * fps
        stances += [(foot, a, b) for a, b in _runs(
            v < STANCE_SPEED, max(2, int(round(STANCE_MIN_SEC * fps))))]
    on = np.full((n, 2), -1)
    for i, (foot, a, b) in enumerate(stances):
        on[a:b, foot] = i
    # ---- when the body was still, by which feet it stood on
    c = gaussian_filter1d(centre_of_mass(P), max(1.0, 0.1 * fps), axis=0)
    cv = np.concatenate([[0.0], np.linalg.norm(np.diff(c, axis=0),
                                               axis=1)]) * fps
    stood = {}
    for f in np.flatnonzero((cv <= STILL_SPEED) & (on.max(axis=1) >= 0)):
        stood.setdefault((int(on[f, 0]), int(on[f, 1])), []).append(int(f))
    # what each camera's pose says about each still stretch
    seen = []      # per stretch and camera: (stretch, centre of mass, feet)
    for k, ((sl, sr), fr) in enumerate(stood.items()):
        if len(fr) < max(2, int(STILL_MIN_SEC * fps)):
            continue
        fr = np.asarray(fr)
        for X in poses:
            feet = []
            for s, ju, jt, hip, knee, ank in (
                    (sl, UNDER_L, TOE_L, HIP_L, KNEE_L, ANK_L),
                    (sr, UNDER_R, TOE_R, HIP_R, KNEE_R, ANK_R)):
                if s < 0:
                    continue
                reach = np.linalg.norm(X[fr, hip] - X[fr, ank], axis=1)
                leg = (np.linalg.norm(X[fr, hip] - X[fr, knee], axis=1)
                       + np.linalg.norm(X[fr, knee] - X[fr, ank], axis=1))
                if np.median(reach / np.maximum(leg, 1e-9)) < LEG_STRAIGHT:
                    continue
                feet.append(((X[fr, ju] + WEIGHT_ALONG_FOOT
                              * (X[fr, jt] - X[fr, ju])).mean(axis=0),
                             X[fr, ju].mean(axis=0)))
            if feet:
                seen.append((k, centre_of_mass(X[fr]).mean(axis=0), feet))

    def balance(down, held_up):
        """Evidence from the still stretches, and which of them gave
        any, given the best idea of "down" so far (to tell a foot on
        the floor from one held up)."""
        Q, used = np.zeros((3, 3)), set()
        for k, cm, feet in seen:
            if len(feet) == 2:
                # the higher foot, if it is clearly higher, is in the air
                hl, hr = -(feet[0][1] @ down), -(feet[1][1] @ down)
                if abs(hl - hr) > held_up:
                    feet = [feet[0] if hl < hr else feet[1]]
            if len(feet) == 2 \
                    and np.linalg.norm(feet[1][0] - feet[0][0]) > 0.05:
                line = feet[1][0] - feet[0][0]
                drop = 0.5 * (feet[0][0] + feet[1][0]) - cm
            else:
                line = None
                drop = np.mean([w for w, _u in feet], axis=0) - cm
            size = np.linalg.norm(drop)
            if size < 1e-6 or np.degrees(np.arccos(np.clip(
                    drop @ -UP / size, -1.0, 1.0))) > STANDING_CONE:
                continue
            used.add(k)
            # the cameras on one stance are ONE piece of evidence about
            # where the weight is: an equal part of one say each
            if line is not None:
                q = np.cross(line / np.linalg.norm(line), drop)
                Q += say_each * np.outer(q, q) / SIG_BALANCE ** 2
                d = feet[1][1] - feet[0][1]
                Q += say_each * np.outer(d, d) / SIG_FEET_LEVEL ** 2
            else:
                e1 = np.cross(drop / size, [1.0, 0.0, 0.0])
                e1 /= np.linalg.norm(e1)
                e2 = np.cross(drop / size, e1)
                for e in (e1, e2):
                    # one foot: the weight is somewhere along it
                    Q += say_each * np.outer(e, e) * size ** 2 \
                        / (1.5 * SIG_BALANCE) ** 2
        return Q, used

    # ---- where the feet stood, as each camera placed them
    sets = []
    if len(stances) >= 3:
        for W in placed:
            sets.append(np.array([np.median(
                W[a:b, UNDER_L if foot == 0 else UNDER_R], axis=0)
                for foot, a, b in stances]))
    prior = np.diag([1.0 / np.radians(CAMERA_ROLL_SD) ** 2, 0.0,
                     1.0 / np.radians(CAMERA_PITCH_SD) ** 2])
    ws = [np.ones(len(p)) for p in sets]
    down, ev, balanced = -UP, np.zeros(3), set()
    for rnd in range(8):
        # a first look with a loose idea of "held up" (the camera's own
        # down can be 15 degrees out, which lifts one foot of a wide
        # stance by 10 cm), then with the answer so far
        Qt, balanced = balance(
            down, 2 * FOOT_HELD_UP if rnd == 0 else FOOT_HELD_UP)
        Qt = Qt + prior
        mids = []
        for p, w in zip(sets, ws):
            mid = (p * w[:, None]).sum(axis=0) / max(w.sum(), 1e-9)
            d = p - mid
            Qt = Qt + (d * w[:, None]).T @ d / SIG_STOOD ** 2 / len(sets)
            mids.append(mid)
        ev, vec = np.linalg.eigh(Qt)
        new = vec[:, 0] if vec[1, 0] > 0 else -vec[:, 0]
        settled = rnd > 0 and float(new @ down) > 1.0 - 1e-9
        down = new
        for i, (p, mid) in enumerate(zip(sets, mids)):
            h = -(p - mid) @ down                 # height above the plane
            low = np.percentile(h, 25) if len(h) >= 4 else h.min()
            w = 1.0 / (1.0 + (np.clip(h - low - 0.03, 0, None) / 0.03) ** 2)
            settled = settled and np.allclose(w, ws[i], atol=1e-3)
            ws[i] = w
        if settled:
            break
    sd = min(float(np.degrees(1.0 / np.sqrt(max(ev[1], 1e-9)))), 90.0)
    still = len(balanced)
    what = (f"{still} still stretch{'' if still == 1 else 'es'}, "
            f"{len(stances)} stance{'' if len(stances) == 1 else 's'}")
    if not balanced and sd > LEVEL_MAX_SD:
        return None, sd, what
    return down, sd, what


def _freeze_during(w, mask):
    """Hold w at its entry value for each run of `mask`."""
    out, f = w.copy(), 0
    while f < len(w):
        if mask[f]:
            g = f
            while g + 1 < len(w) and mask[g + 1]:
                g += 1
            out[f:g + 1] = w[max(f - 1, 0)]
            f = g + 1
        else:
            f += 1
    return out


# -------------------------------------------------------------- fine sync
SYNC_JOINTS = [HAND_R, HAND_L, ELB_R, ELB_L, KNEE_R, KNEE_L, ANK_R, ANK_L]
SYNC_WITNESSES = (("arms", [0, 1, 2, 3]), ("legs", [4, 5, 6, 7]),
                  ("right side", [0, 2, 4, 6]), ("left side", [1, 3, 5, 7]))


def resample_frames(arr, t, kind="linear"):
    """Read a per-frame array at fractional frame positions t (n,).

    kind: 'linear' for positions and other plain numbers, 'rotation'
    for stacks of rotation matrices (..., 3, 3) (spherical, per
    matrix), 'nearest' for anything that cannot be mixed (flags,
    integers)."""
    arr = np.asarray(arr)
    t = np.clip(np.asarray(t, dtype=np.float64), 0, len(arr) - 1)
    if kind == "nearest":
        return arr[np.rint(t).astype(int)]
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, len(arr) - 1)
    w = t - i0
    if kind == "linear":
        ww = w.reshape((-1,) + (1,) * (arr.ndim - 1))
        return arr[i0] * (1.0 - ww) + arr[i1] * ww
    shape = arr[i0].shape
    q0 = Rot.from_matrix(arr[i0].reshape(-1, 3, 3)).as_quat()
    q1 = Rot.from_matrix(arr[i1].reshape(-1, 3, 3)).as_quat()
    q1 = np.where(((q0 * q1).sum(-1) < 0)[:, None], -q1, q1)
    ww = np.repeat(w, int(np.prod(shape[1:-2], dtype=int)))[:, None]
    om = np.arccos(np.clip((q0 * q1).sum(-1), -1.0, 1.0))[:, None]
    so = np.sin(om)
    near = so < 1e-6
    so = np.where(near, 1.0, so)
    q = np.where(near, q0 * (1.0 - ww) + q1 * ww,
                 (np.sin((1.0 - ww) * om) * q0 + np.sin(ww * om) * q1) / so)
    return Rot.from_quat(_n(q)).as_matrix().reshape(shape)


def _sync_speeds(jc, G, sigma):
    """What both cameras see the same way: how fast each wrist, elbow,
    knee and ankle is moving UP or DOWN, relative to the hips. Height
    is the one direction neither camera has to guess."""
    rel = -np.einsum('ij,fkj->fki', G,
                     jc - jc[:, HIPS:HIPS + 1])[:, SYNC_JOINTS, 1]
    return np.diff(gaussian_filter1d(rel, sigma, axis=0), axis=0)


def _sync_peak(vA, vB, lo, hi, L, cols=slice(None)):
    """How many frames camera B shows a move BEFORE camera A, over
    frames lo..hi: (shift, agreement, sharpness), shift None when the
    best lag is at the end of the range looked at.

    The two signals are compared at whole-frame lags only and the peak
    is placed between lags by a parabola. Searching with an
    interpolated signal instead is biased: interpolating halfway
    between frames smooths the signal, a smoother signal always fits
    better, and the answer drifts to the half-frame marks."""
    lo, hi = max(lo, 0), min(hi, len(vA))
    a = vA[lo + L:hi - L, cols]
    if len(a) < 4:
        return None, 0.0, 0.0
    c = np.empty(2 * L + 1)
    for k in range(-L, L + 1):
        b = vB[lo + L - k:hi - L - k, cols]
        c[k + L] = (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()
                                           + 1e-30)
    k = int(np.argmax(c))
    if k in (0, 2 * L):
        return None, float(c[k]), 0.0
    den = c[k - 1] - 2.0 * c[k] + c[k + 1]
    frac = 0.5 * (c[k - 1] - c[k + 1]) / den if den < 0 else 0.0
    sharp = c[k] - max(c[max(k - 2, 0)], c[min(k + 2, 2 * L)])
    return float(k - L + frac), float(c[k]), float(sharp)


def sync_between_all(jcs, Gs, fps):
    """With three or four cameras: how far each runs ahead of the
    others, from every PAIR of cameras together. Returns (ahead, tied):
    ahead[k] is one number of frames for camera k, or None where the
    footage does not say; tied is False when camera A is not among the
    cameras it could place, and the numbers are then against those
    cameras' own average instead of against camera A.

    measure_sync asks one pair, and asks a lot of it (the arms, the
    legs, each side and every window must agree) before it corrects
    anything, because one pair has nothing to check itself against.
    Three cameras do: A to B, B to C and A to C are three separate
    measurements, and they can only add up if they are right. So here
    every pair gives its plain best offset over the whole take, the
    offsets are fitted to one number per camera, and a pair that does
    not fit with the rest (SYNC_WITNESS) is dropped. A camera is only
    placed when at least two pairs that fit hold it.

    This is what places a camera whose own pairing with camera A is
    poor (camera A sees the performer badly, or from the opposite
    side): it is held by the other cameras instead. One number per
    camera: a step or a drift is still measure_sync's to find."""
    K = len(jcs)
    none = [None] * K
    n = min(len(x) for x in jcs)
    L = max(2, int(np.ceil(SYNC_MAX_SEC * fps)))
    if K < 3 or n < 8 * L:
        return none, True
    sigma = SYNC_SMOOTH_SEC * fps
    v = [_sync_speeds(jc[:n], G, sigma) for jc, G in zip(jcs, Gs)]
    edges = []
    for i in range(K):
        for j in range(i + 1, K):
            s, agree, _ = _sync_peak(v[i], v[j], 0, n - 1, L)
            if s is not None and agree >= SYNC_AGREE:
                edges.append((i, j, s, agree))
    while True:
        # only cameras held by two pairs or more
        while True:
            count = [sum(1 for e in edges if k in e[:2]) for k in range(K)]
            keep = [e for e in edges if count[e[0]] >= 2 and count[e[1]] >= 2]
            if len(keep) == len(edges):
                break
            edges = keep
        if not edges:
            return none, True
        # one number per camera, by least squares; the last row holds
        # the average at zero, which only settles what the pairs leave
        # open (they say nothing about all the cameras moving together)
        rows, rhs = [], []
        for i, j, s, w in edges:
            r = np.zeros(K)
            r[j], r[i] = np.sqrt(w), -np.sqrt(w)
            rows.append(r)
            rhs.append(np.sqrt(w) * s)
        used = sorted({k for e in edges for k in e[:2]})
        for k in range(K):
            if k not in used:
                r = np.zeros(K)
                r[k] = 1.0
                rows.append(r)
                rhs.append(0.0)
        rows.append(np.full(K, 1e-3))
        rhs.append(0.0)
        c = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
        miss = [abs(c[j] - c[i] - s) for i, j, s, _w in edges]
        worst = int(np.argmax(miss))
        if miss[worst] <= SYNC_WITNESS:
            break
        edges.pop(worst)
    # the cameras held together with camera A, or failing that the
    # largest group that holds together without it
    groups, left = [], set(used)
    while left:
        g, grow = {min(left)}, True
        while grow:
            grow = False
            for i, j, _s, _w in edges:
                if (i in g) != (j in g):
                    g |= {i, j}
                    grow = True
        groups.append(sorted(g))
        left -= g
    group = next((g for g in groups if 0 in g), max(groups, key=len))
    tied = 0 in group
    base = c[0] if tied else float(np.mean([c[k] for k in group]))
    ahead = [float(c[k] - base) if k in group else None for k in range(K)]
    return ahead, tied


def measure_sync(jcA, jcB, GA, GB, fps, name="B", agreed=None,
                 agreed_with="camera A"):
    """How far two captures are out of step, frame by frame: camera A
    and one other (name: its letter, for the text).

    agreed: with three or four cameras, what all of them together say
    about this one (sync_between_all), in frames ahead of agreed_with.
    It is used only where this pair cannot say by itself.

    Returns (shift, text, act). shift is (n,) frames camera B runs
    AHEAD of camera A (B shows a move that much earlier), or None when
    the clips are in step or the footage does not say clearly; text is
    one line for the log; act is True when the user has something to
    do about it (the clips are too far apart to correct), which is the
    one case the capture result reports.

    Clips lined up by eye are a frame or so out, and never by a whole
    number of frames, since the cameras do not start their frames at
    the same instant. A fast limb travels 20 cm in a frame, so the
    merge then sees every fast move at two different moments. The
    offset is not always one number either: a camera that drops a
    frame steps it, and two cameras on slightly different clocks
    (23.976 next to 24) drift. So it is read in windows through the
    take and the simplest of three shapes that fits is used: constant,
    one step, or a steady drift. (On slow footage a step and a drift
    fit about equally and either may be chosen; they correct it about
    equally too. A take too short for SYNC_SHAPE_WINDOWS windows gets
    one number or nothing.)

    Nothing is corrected unless the footage is clear about it: the
    arms, the legs, the left side and the right side each have to give
    the same answer on their own, and the windows have to sit on the
    shape chosen. A slow take says nothing either way and is left as
    it was synced by hand."""
    n = min(len(jcA), len(jcB))
    L = max(2, int(np.ceil(SYNC_MAX_SEC * fps)))
    if n < 8 * L:
        return None, "Fine sync: the take is too short to measure.", False
    sigma = SYNC_SMOOTH_SEC * fps
    vA, vB = _sync_speeds(jcA[:n], GA, sigma), _sync_speeds(jcB[:n], GB, sigma)

    def how(s, of=True):
        if abs(s) < 0.05:
            return "in step with" if of else "in step"
        return (f"{abs(s):.1f} frames "
                + (("ahead of" if of else "ahead") if s > 0
                   else ("behind" if of else "behind")))

    def left(text):
        """This pair cannot say: what all the cameras together say,
        when they do."""
        if agreed is None or abs(agreed) > L - 1:
            return None, text, False
        if abs(agreed) < SYNC_MIN:
            return None, (f"Fine sync: camera {name} is in step (within "
                          f"{SYNC_MIN:g} of a frame), going by all the "
                          f"cameras together."), False
        return (np.full(n, float(agreed)),
                f"Fine sync: camera {name} was {how(agreed)} "
                f"{agreed_with}, going by all the cameras together. "
                f"Corrected.", False)

    whole, agree, _ = _sync_peak(vA, vB, 0, n - 1, L)
    unclear = ("Fine sync: the footage does not say clearly how far the "
               "clips are out of step; left as synced.")
    if agree < SYNC_AGREE:
        # nothing the two cameras both show moving: standing still, or
        # one camera not seeing the performer
        return left("Fine sync: not enough movement both cameras can "
                    "see to measure; left as synced.")
    if whole is None:
        return None, (f"Fine sync: the clips look more than {L - 1} frames "
                      f"out of step. Sync them closer and capture "
                      f"again."), True
    for _name, cols in SYNC_WITNESSES:
        s, _a, _sh = _sync_peak(vA, vB, 0, n - 1, L, cols)
        if s is None or abs(s - whole) > SYNC_WITNESS:
            return left(unclear)

    # the offset through the take, from the windows the cameras agree in
    W = int(round(SYNC_WINDOW_SEC * fps))
    step = max(1, int(round(SYNC_STEP_SEC * fps)))
    xs, ys, ws = [], [], []
    for lo in range(0, max(n - 1 - W, 0) + 1, step):
        s, a, sh = _sync_peak(vA, vB, lo, lo + W, L)
        if s is not None and a >= SYNC_TRUST:
            xs.append(lo + 0.5 * W)
            ys.append(s)
            ws.append(max(sh, 0.02))
    xs, ys, ws = np.array(xs), np.array(ys), np.array(ws)

    def fit(xs, ys, ws):
        """The simplest shape the windows sit on: (name, per-window
        model, parameters). Constant unless a drift or one step at
        least halves the misfit."""
        def sse(model):
            return float((ws * (ys - model) ** 2).sum())
        mean = (ws * ys).sum() / ws.sum()
        best = ("constant", np.full(len(xs), mean), None)
        e_const = sse(best[1])
        A_ = np.stack([np.ones(len(xs)), xs], 1) * np.sqrt(ws)[:, None]
        coef = np.linalg.lstsq(A_, ys * np.sqrt(ws), rcond=None)[0]
        line = coef[0] + coef[1] * xs
        e_step, cut = np.inf, None
        for k in range(3, len(xs) - 2):
            l1 = (ws[:k] * ys[:k]).sum() / ws[:k].sum()
            l2 = (ws[k:] * ys[k:]).sum() / ws[k:].sum()
            e = sse(np.concatenate([np.full(k, l1),
                                    np.full(len(xs) - k, l2)]))
            if e < e_step:
                e_step, cut = e, (k, l1, l2)
        if len(xs) < SYNC_SHAPE_WINDOWS:
            # A short take has too few windows to tell a step from a
            # drift from noise (with six, a step was read as a drift
            # and the start of the take was put half a frame wrong):
            # it is one number, or it is left alone.
            return best
        if (sse(line) <= e_step and sse(line) < 0.5 * e_const
                and abs(coef[1]) * n >= 2 * SYNC_MIN):
            best = ("drift", line, coef)
        elif (cut is not None and e_step < 0.5 * e_const
                and abs(cut[2] - cut[1]) >= 2 * SYNC_MIN):
            k, l1, l2 = cut
            best = ("step", np.concatenate([np.full(k, l1),
                                            np.full(len(xs) - k, l2)]), cut)
        return best

    shape, shift, at = "constant", np.full(n, whole), 0
    if len(xs) >= 6:
        shape, model, par = fit(xs, ys, ws)
        # One stray window (a moment where a camera lost a limb) must
        # not decide it: windows far off the shape are set aside once,
        # as long as they are few, and the shape is fitted again.
        res = np.abs(ys - model)
        keep = res <= max(3.0 * 1.4826 * float(np.median(res)), 0.3)
        if not keep.all() and keep.mean() >= 0.85 and keep.sum() >= 6:
            xs, ys, ws = xs[keep], ys[keep], ws[keep]
            shape, model, par = fit(xs, ys, ws)
        if np.sqrt(float((ws * (ys - model) ** 2).sum()) / ws.sum()) \
                > SYNC_SCATTER:
            return left(unclear)
        if shape == "constant":
            if abs(float(model[0]) - whole) > SYNC_WITNESS:
                return left(unclear)
        elif shape == "drift":
            # held at its end values outside the windows it was
            # measured in: a line is not to be trusted past its data
            shift = par[0] + par[1] * np.clip(np.arange(n), xs[0], xs[-1])
        else:
            k, l1, l2 = par
            # where exactly: the frame that splits the two levels best,
            # judged on the speeds themselves between the two windows
            t = np.arange(len(vB), dtype=np.float64)
            e1 = ((vA - np.stack([np.interp(t - l1, t, vB[:, c])
                                  for c in range(vB.shape[1])], 1)) ** 2).sum(1)
            e2 = ((vA - np.stack([np.interp(t - l2, t, vB[:, c])
                                  for c in range(vB.shape[1])], 1)) ** 2).sum(1)
            lo_f = int(max(xs[k - 1] - 0.5 * W, 0))
            hi_f = int(min(xs[k] + 0.5 * W, len(vB) - 1))
            cost = (np.cumsum(e1[lo_f:hi_f]) + e2[lo_f:hi_f].sum()
                    - np.cumsum(e2[lo_f:hi_f]))
            at = lo_f + int(np.argmin(cost)) + 1
            shift = np.where(np.arange(n) < at, l1, l2)
    if np.abs(shift).max() < SYNC_MIN:
        return None, (f"Fine sync: the clips are in step (within "
                      f"{SYNC_MIN:g} of a frame)."), False
    if np.abs(shift).max() > L - 1:
        return left(unclear)
    if shape == "constant":
        text = f"camera {name} was {how(whole)} camera A"
    else:
        # no cause is named: a dropped frame and two clocks that differ
        # can both produce either shape, and the footage cannot tell
        text = (f"camera {name} was {how(shift[0])} camera A at the start "
                f"and {how(shift[-1], False)} at the end")
    return shift, f"Fine sync: {text}. Corrected.", False


def _single_camera_faults(Z, cut, n, g, jc, parents, R0):
    """What single-camera capture already knows about one camera being
    wrong, frame by frame. Neither test looks at how fast anything is
    moving.

    failed: (n,) 0 or TRACKER_FAILED, Clean Tracking's detector
        (tracking_repair): the bones changed length or the pose
        collapsed to the rest pose. The model stretches the whole body
        together, so this speaks for the camera's whole frame, not for
        one limb, which is why it is a lean and not a hand-over.
    outside: {joint: (n,) 0/1}, Hold Cropped Limbs' test (cropped_limbs):
        the joint's 2D keypoint is outside the picture, so the model
        invented it. None when the capture does not record its frame
        size (captures from before that was stored).

    Both are spread one frame each way: the merge blends frame-to-frame
    CHANGE, and a bad frame spoils the change into it and out of it."""
    def spread(mask):
        return maximum_filter1d(np.asarray(mask, dtype=np.float64), 3,
                                mode="nearest")
    failed = np.zeros(n)
    if R0 is not None:
        failed = TRACKER_FAILED * spread(
            _detect_bad_frames(g, jc, parents, R0)[0])
    outside = None
    if "keypoints_2d" in Z.files and "frame_size" in Z.files:
        inside = keypoints_inside(Z["keypoints_2d"][cut][:n], Z["frame_size"])
        outside = {j: spread(~inside[:, k]) for j, k in KP_OF.items()}
    return failed, outside


# -------------------------------------------------------------------- main
def camera_shares(bads):
    """Each camera's share of a limb, from how bad each looks: (k, n)
    in, (k, n) out, summing to 1 on every frame.

    A camera as good as the best one shares evenly with it; one 0.15
    worse than the best has no share; in between it falls off in a
    straight line against the best. Written out for two cameras that
    is 0.5 + (bad A - bad B) / 0.30 for camera B, held to 0..1, which
    is how the two-camera merge has always computed it (and still
    does, to the last bit)."""
    b = np.asarray(bads, dtype=np.float64)
    if len(b) == 2:
        wb = np.clip(0.5 + (b[0] - b[1]) / 0.30, 0, 1)
        return np.stack([1.0 - wb, wb])
    gap = b - b.min(axis=0)
    s = np.clip(0.5 - gap / 0.30, 0, None) / (0.5 + gap / 0.30)
    return s / s.sum(axis=0)


def _settle_shares(W, hold=None):
    """Ease each camera's share over a frame or so, hold them through
    `hold` (a planted foot), and give camera A what the others leave,
    so the shares still sum to 1."""
    out = [None]
    for w in W[1:]:
        w = gaussian_filter1d(w, 0.8)
        out.append(w if hold is None else _freeze_during(w, hold))
    rest = out[1]
    for w in out[2:]:
        rest = rest + w
    out[0] = 1.0 - rest if len(out) == 2 else np.clip(1.0 - rest, 0.0, 1.0)
    return out


def cameras_apart(poses, sights, joints, seen=None):
    """With three or four cameras: how far each camera's version of a
    joint is from where the cameras agree it is. (k, n, j) metres,
    measured across that camera's own picture.

    poses: each camera's (n, 127, 3) joints relative to the hips, in
    camera A's frame. sights: the unit direction each camera looks
    along to reach each joint, same frame. joints: which ones.

    Where the cameras agree is found the way two cameras' positions
    are fused (each believed across its picture, not along its
    sightline), with one addition that only three or more allow: a
    camera far from the others loses its say, so the point settles
    where MOST of them put it (it starts from the middle camera on
    each axis, and is refitted a few times). Two cameras that disagree
    cannot tell which of them is wrong; three can, and this is where
    that is read.

    seen: (k, n, j) 0..1, how much of a view each camera has of each
    joint (0: the joint is outside its picture, or the performer is
    not in it). A camera without a view has no say in where the others
    agree; where NO camera has one, they all have their say as before."""
    K = len(poses)
    P = [p[:, joints] for p in poses]
    S = [s[:, joints] for s in sights]
    eye = np.eye(3)
    c = np.median(P, axis=0)
    say_of = None
    if seen is not None:
        say_of = np.asarray(seen, dtype=np.float64)
        nobody = say_of.max(axis=0) < 0.5
        say_of = np.where(nobody[None], 1.0, say_of)
        # start from the middle of the cameras that can see the joint
        Pm = np.where((say_of >= 0.5)[..., None], np.asarray(P), np.nan)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mid = np.nanmedian(Pm, axis=0)
        c = np.where(np.isfinite(mid), mid, c)
    apart = None
    for rnd in range(AGREE_ROUNDS + 1):
        apart = []
        for k in range(K):
            d = P[k] - c
            d = d - S[k] * (d * S[k]).sum(-1, keepdims=True)
            apart.append(np.linalg.norm(d, axis=-1))
        if rnd == AGREE_ROUNDS:
            break
        tot = rhs = None
        for k in range(K):
            w = 1.0 / (1.0 + (apart[k] / AGREE_SCALE) ** 2)
            if say_of is not None:
                w = w * say_of[k]
            Q = w[..., None, None] * (
                eye - 0.95 * S[k][..., :, None] * S[k][..., None, :])
            part = np.einsum('fkij,fkj->fki', Q, P[k])
            tot = Q if tot is None else tot + Q
            rhs = part if rhs is None else rhs + part
        c = np.linalg.solve(tot + 1e-6 * eye, rhs[..., None])[..., 0]
    return np.array(apart)


def agreeing_floors(planes, turns):
    """Which cameras found the same floor: (cameras, apart). cameras is
    the largest set whose contacts cover some floor (FLOOR_MIN_SPREAD)
    and whose fitted floors are within FLOOR_MAX_APART of one of them,
    as indices, or [] when no two agree. apart: (k, k) degrees between
    the cameras' floors, compared through the body's rotation (turns).

    For two cameras this is floor_trusted."""
    K = len(planes)
    nrm = [turns[k] @ planes[k][0] for k in range(K)]
    apart = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i != j:
                apart[i, j] = np.degrees(np.arccos(np.clip(
                    nrm[i] @ nrm[j], -1.0, 1.0)))
    wide = [float(planes[k][1][1]) >= FLOOR_MIN_SPREAD for k in range(K)]
    best = []
    for c in range(K):
        if wide[c]:
            g = [k for k in range(K)
                 if wide[k] and apart[c, k] <= FLOOR_MAX_APART]
            if len(g) > len(best):
                best = g
    return (best if len(best) >= 2 else []), apart


def fits_others(pose, sights, seeing):
    """How far each camera's version of each joint is from what the
    OTHER cameras see: (k, n, joints) metres.

    pose: per camera, the joints relative to the hips, in camera A's
    frame, (n, joints, 3). sights: the same shape, the direction each
    camera looks along to reach each joint (unit vectors, camera A's
    frame). seeing: (k, n, joints), over 0.5 where the camera has the
    joint in view.

    A camera's version is compared with every other camera's that has
    the joint in view, ACROSS that other camera's line of sight, which
    is the part of its own version that camera really sees (along its
    line of sight it is guessing, like every camera). The middle one
    when several see it; 0 where no other camera does. A limb a camera
    got wrong is usually wrong along its own line of sight, nearer or
    further than it is, which its own picture cannot show and another
    camera sees sideways."""
    import warnings
    K = len(pose)
    out = np.zeros((K,) + pose[0].shape[:2])
    for k in range(K):
        ds = []
        for o in range(K):
            if o == k:
                continue
            d = pose[k] - pose[o]
            d = d - sights[o] * (d * sights[o]).sum(-1, keepdims=True)
            ds.append(np.where(np.asarray(seeing[o]) > 0.5,
                               np.linalg.norm(d, axis=-1), np.nan))
        ds = np.array(ds)
        some = np.isfinite(ds).any(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mid = np.nanmedian(ds, axis=0)
        out[k] = np.where(some, mid, 0.0)
    return out


def place_from_take(wld, turns, offs, fail, outside, say):
    """Where the cameras of an ordinary multi-camera take stand, from
    the WHOLE take (pipeline/people_place.py, with one person in it) in
    place of the T-pose alone: the places, camera A's the origin, or
    None when the take does not settle it (the T-pose's placement
    stands then).

    wld: each camera's joints in its own frame, already in step.
    turns: how each camera is turned against camera A (camera_turn,
    read off the body). They are KEPT: only where the cameras stand is
    worked out here. offs: where the T-pose put them, to start from.
    fail, outside: the single-camera tests (_single_camera_faults): a
    frame counts for a camera when its tracker held and the hips and
    shoulders were inside its picture.

    The T-pose says where the cameras stand from one moment, and from
    how far away each camera took the performer to be at that moment,
    which is the weakest thing a camera knows. This goes by where each
    camera sees the body ACROSS its picture, on every frame it shares
    with another camera, so it does not need the T-pose: a take that
    opens without one leaves find_tpose a single early frame to go by.

    Measured 2026-11-24 on five takes, by the merged limb joints seen
    through each camera against that camera's own tracking of them in
    its picture (pixels of a 1920 picture, the middle one). With the
    T-pose in the take this and the T-pose come out the same (14.4 ->
    14.6, 10.7 -> 11.1, 11.9 -> 11.7, 11.7 -> 12.0, 21.5 -> 19.7) and
    the cameras move 1 to 12 cm. With the take cut to start after its
    T-pose, or in the middle of the action (nine cuts of three takes),
    this is the same or better every time (24.0 -> 21.3 and 16.6 ->
    14.5 at best, 14.6 -> 14.9 at worst).
    Letting the turns go too, as a multi-person take does (where two
    people pin them down), moved the cameras 13 to 44 cm and came out
    a little WORSE by the same measure on three of the five (14.4 ->
    16.1, 10.7 -> 12.0): the limbs are fused on the turns read off the
    body, and a body placed with other turns no longer lines up with
    them in the cameras' pictures. So the turns stay."""
    from pipeline.people_place import place
    K = len(wld)
    joints = [HIPS, NECK, SHO_R, SHO_L]         # the hips first (people_place)
    obs = {}
    for k in range(K):
        seen = np.asarray(fail[k]) < 0.5 * TRACKER_FAILED
        if outside[k] is not None:
            for j in (HIP_R, HIP_L, SHO_R, SHO_L):
                seen = seen & (np.asarray(outside[k][j]) < 0.5)
        obs[(0, k)] = (wld[k][:, joints], seen)
    why = []
    got = place(obs, [(np.asarray(turns[k], dtype=np.float64),
                       np.asarray(offs[k], dtype=np.float64))
                      for k in range(K)], log=lambda *_a: None, why=why,
                keep_turns=True)
    if got is None:
        if why and why[0][0] == "apart":
            reason = (f"placed from the whole take they disagree on the body "
                      f"by {why[0][1] * 100:.1f} cm across their pictures")
        else:
            reason = ("too few frames where two cameras had the hips and "
                      "shoulders in their pictures at once")
        say(f"  position: the cameras stand where the T-pose put them "
            f"({reason})")
        return None
    moved = [float(np.linalg.norm(got["places"][k] - np.asarray(offs[k])))
             for k in range(1, K)]
    say(f"  position: the cameras placed from the whole take; they agree on "
        f"the body to {got['across'] * 100:.1f} cm across their pictures "
        f"(from where the T-pose put them: "
        + ", ".join(f"{LETTERS[k]} {moved[k - 1] * 100:.0f} cm"
                    for k in range(1, K)) + ")")
    return [got["places"][k] for k in range(K)]


def merge(dir_a, dir_b, out_dir, mhr_model, sync_offset=0, quiet=False):
    """merge_cameras for two cameras."""
    return merge_cameras([dir_a, dir_b], out_dir, mhr_model, [sync_offset],
                         quiet)


def merge_cameras(dirs, out_dir, mhr_model, sync_offsets=None, quiet=False,
                  level=None, cameras=None):
    """Merge 2 to 4 cameras' captures of one take (dirs, camera A
    first) into out_dir. sync_offsets: for each camera after A, how
    many frames camera A leads it by (default 0: the captures start
    on the same frame, as the addon makes them).

    level: which way is up, already known (the rotation that stands
    camera A's frame level), in place of working it out from this
    performer. Given for a person of a Multi-Camera & Multi-Person take
    (people_cameras.take_level): everybody in it stands on ONE floor,
    and levelling each of them on their own evidence leaves them
    leaning a few degrees against each other.

    cameras: where the cameras stand and how big this person is,
    already known ({"turns", "places", "scale"}, one turn and place per
    camera in `dirs` order, camera A's the identity and the origin), in
    place of placing them from this performer's T-pose. Given for a
    person of a Multi-Camera & Multi-Person take
    (people_tracks.take_cameras, pipeline/people_place.py): placed from
    one person at a time, every camera is tied to camera A's idea of
    how far away THAT person is, which is the weakest thing it knows,
    and two people end up the difference of two such errors apart. Only
    WHERE the body is comes from these; how the limbs are turned is
    still fused on the turns read off this performer's own poses."""
    say = (lambda *a: None) if quiet else print
    K = len(dirs)
    if not 2 <= K <= len(LETTERS):
        raise ValueError(f"2 to {len(LETTERS)} cameras, got {K}")
    names = LETTERS[:K]
    Z = [np.load(os.path.join(d, "mhr_frames.npz")) for d in dirs]
    A = Z[0]
    parents, offsets_cm, prerot = load_skeleton(mhr_model)
    fps = float(A["fps"])

    # sync: a positive offset means camera A leads that camera by N frames
    offs_in = [0] + [int(o) for o in (sync_offsets or [])][:K - 1]
    offs_in += [0] * (K - len(offs_in))
    first = max(0, max(offs_in))
    starts = [first - o for o in offs_in]
    cuts = [slice(s, None) for s in starts]
    jc = [z["joint_coords"].astype(np.float64)[c] for z, c in zip(Z, cuts)]
    gr = [z["global_rots"].astype(np.float64)[c] for z, c in zip(Z, cuts)]
    ct = [z["cam_t"].astype(np.float64)[c] for z, c in zip(Z, cuts)]
    n = min(len(x) for x in jc)
    jc, gr, ct = ([x[:n] for x in v] for v in (jc, gr, ct))
    o = offs_in[1]
    say(f"{n} frames after sync (offset "
        f"{o if K == 2 else ', '.join(str(v) for v in offs_in[1:])})")

    # what single-camera capture can already tell about each camera
    R0 = None if prerot is None else compute_tpose_globals(parents, prerot)
    fail, outside, unseen = [], [], []
    for k in range(K):
        fl, ou = _single_camera_faults(Z[k], cuts[k], n, gr[k], jc[k],
                                       parents, R0)
        not_there = np.zeros(n)
        if "absent" in Z[k].files:
            # (spread a frame each way, like the tests above: the merge
            # blends frame-to-frame change)
            gone = np.asarray(Z[k]["absent"], dtype=np.float64)[cuts[k]][:n]
            not_there = maximum_filter1d(gone, 3, mode="nearest")
            fl = np.maximum(fl, CAMERA_ABSENT * not_there)
        fail.append(fl)
        outside.append(ou)
        unseen.append(not_there)
    zero = np.zeros(n)

    def depth(j):
        d = 0
        while parents[j] >= 0:
            j = int(parents[j])
            d += 1
        return d
    order = sorted(range(N_JOINTS), key=depth)

    pa = [x - x[:, HIPS:HIPS + 1, :] for x in jc]
    wld = [x + t[:, None, :] for x, t in zip(jc, ct)]

    # --- 1. calibration: the cameras against each other from the body,
    # gravity from the floor when there is a floor to fit ---------------
    W = find_tpose(jc[0])
    say(f"T-pose: frames {W[0]}-{W[1] - 1} ({W[1] - W[0]} frames)")
    J = TARGETS + [HEAD]
    feet = [w[:, [ANK_R, ANK_L]] for w in wld]
    planes = [floor_plane(x) for x in feet]
    # (before fine sync: a frame or so out of step does not move this
    # enough to matter for the test, and fine sync needs the vertical)
    Rs = [np.eye(3)] + [camera_turn(pa[0][:, J], pa[k][:, J])[0]
                        for k in range(1, K)]
    floors, apart = agreeing_floors(planes, Rs)
    fits = [gravity_from_floor(x) for x in feet]
    if K == 2:
        say(f"  floor: A {fits[0][1]:.2f} deg (contacts reach "
            f"{planes[0][1][0] * 100:.0f} x {planes[0][1][1] * 100:.0f} cm, "
            f"flat to {fits[0][2] * 100:.1f}), B {fits[1][1]:.2f} deg ("
            f"{planes[1][1][0] * 100:.0f} x {planes[1][1][1] * 100:.0f} cm, "
            f"flat to {fits[1][2] * 100:.1f}); the two "
            f"are {apart[0, 1]:.1f} deg apart")
    else:
        for k in range(K):
            say(f"  floor: {names[k]} {fits[k][1]:.2f} deg (contacts reach "
                f"{planes[k][1][0] * 100:.0f} x {planes[k][1][1] * 100:.0f} "
                f"cm, flat to {fits[k][2] * 100:.1f})"
                + ("" if k == 0 else f", {apart[0, k]:.1f} deg from A's"))
        say("  floor: the cameras that agree: "
            + (", ".join(names[k] for k in floors) or "none"))
    levelled = True
    stance = None
    if floors:
        # The fitted floor has to agree with how the performer stands as
        # well (FLOOR_STANCE_APART). A take with no stance to read keeps
        # its floor.
        mean = np.sum([Rs[k] @ planes[k][0] for k in floors], axis=0)
        floor_down = planes[0][0] if 0 in floors else mean / np.linalg.norm(mean)
        stance = level_from_stance(wld, pa, Rs, fps)
        if stance[0] is not None:
            off = float(np.degrees(np.arccos(np.clip(
                floor_down @ stance[0], -1.0, 1.0))))
            say(f"  floor: {off:.1f} deg from how the performer stands "
                f"({stance[2]})"
                + ("" if off <= FLOOR_STANCE_APART else ": not used"))
            if off > FLOOR_STANCE_APART:
                floors = []
    if floors:
        if 0 in floors:
            GA = gravity_from_floor(feet[0])[0]
            level_text = "Levelling: from the floor (camera A's foot contacts)."
        else:
            # camera A's own contacts do not make a floor (its view of
            # the feet is blocked, or cut off), and other cameras agree
            # on one: theirs, turned into camera A's frame
            mean = np.sum([Rs[k] @ planes[k][0] for k in floors], axis=0)
            GA = _level_to(mean / np.linalg.norm(mean))[0]
            level_text = ("Levelling: from the floor (the foot contacts of "
                          "cameras " + " and ".join(names[k] for k in floors)
                          + ").")
        # fine sync reads heights: a camera's own floor is its vertical
        Gs = [GA] + [gravity_from_floor(feet[k])[0] if k in floors
                     else GA @ Rs[k] for k in range(1, K)]
    else:
        # No floor to fit: from how the performer stands. The other
        # cameras' vertical is camera A's, through the body.
        down, sd, what = (stance if stance is not None
                          else level_from_stance(wld, pa, Rs, fps))
        if down is not None:
            GA, deg = _level_to(down)
            level_text = (f"Levelling: from how the performer stands, "
                          + ("no floor to fit" if stance is None
                             else "the fitted floor does not agree with it")
                          + f" ({what}; camera A {deg:.1f} deg "
                          f"off level, held to {sd:.1f}).")
        else:
            # Nothing to go by: the take stays in camera A's frame.
            levelled = False
            GA = np.eye(3)
            level_text = LEVEL_NOT
            say(f"  levelling: not from how the performer stands either "
                f"({what}; held to {sd:.0f} deg)")
        Gs = [GA] + [GA @ Rs[k] for k in range(1, K)]
    if level is not None:
        GA = np.asarray(level, dtype=np.float64)
        Gs = [GA] + [GA @ Rs[k] for k in range(1, K)]
        levelled = True
        level_text = ("Levelling: the take's own (the floor under everybody "
                      "in it).")
    say(level_text)

    # --- 1b. fine sync: re-time every other camera onto camera A's
    # frames ------------------------------------------------------------
    shifts, times = [None] * K, [None] * K
    pa_own, root_own = [None] * K, [None] * K
    sync_lines, sync_acts = [], []
    # with three or four cameras, what every pair says, taken together
    net, tied = sync_between_all(jc, Gs, fps)
    for k in range(1, K):
        if K == 2:
            shift, text, act = measure_sync(jc[0], jc[k], GA, Gs[k], fps)
        else:
            shift, text, act = measure_sync(
                jc[0], jc[k], GA, Gs[k], fps, names[k], net[k],
                "camera A" if tied else "the other cameras")
        if K > 2:
            text = text.replace("Fine sync:", f"Fine sync, camera "
                                              f"{names[k]}:", 1)
        say(text)
        sync_lines.append(text)
        if act:
            sync_acts.append(text)
        if shift is None:
            continue
        shifts[k] = shift
        times[k] = t = np.arange(n) - shift
        # The camera as it was captured, for the "is it jumping" tests
        # below. Reading between two frames smooths what is read, and a
        # smoother camera would win those tests for that reason alone:
        # so they run on its own frames and their VERDICTS are
        # re-timed, not the data they judge.
        pa_own[k] = jc[k] - jc[k][:, HIPS:HIPS + 1, :]
        root_own[k] = gr[k][:, HIPS].copy()
        jc[k] = resample_frames(jc[k], t)
        ct[k] = resample_frames(ct[k], t)
        gr[k] = resample_frames(gr[k], t, "rotation")
        fail[k] = resample_frames(fail[k], t)
        unseen[k] = resample_frames(unseen[k], t)
        if outside[k] is not None:
            outside[k] = {j: resample_frames(v, t)
                          for j, v in outside[k].items()}
        pa[k] = jc[k] - jc[k][:, HIPS:HIPS + 1, :]
        wld[k] = jc[k] + ct[k][:, None, :]

    # the cameras against each other, now that they are in step
    agree, yaws = [0.0], [0.0]
    for k in range(1, K):
        Rs[k], a_k = camera_turn(pa[0][:, J], pa[k][:, J])
        agree.append(a_k)
    Gs = [GA] + [GA @ Rs[k] for k in range(1, K)]    # each frame -> the take's
    for k in range(1, K):
        # for the record: how far the camera points from camera A,
        # about the vertical
        fa = GA[:, 2] * [1.0, 0.0, 1.0]
        fb = Gs[k][:, 2] * [1.0, 0.0, 1.0]
        yaws.append(float(np.degrees(np.arctan2(UP @ np.cross(fa, fb),
                                                fa @ fb))))
    if K == 2:
        say(f"  calibration: cameras {yaws[1]:.2f} deg apart, pose agreement "
            f"{agree[1] * 100:.2f} cm")
    else:
        for k in range(1, K):
            say(f"  calibration: camera {names[k]} {yaws[k]:.2f} deg from "
                f"camera A, pose agreement {agree[k] * 100:.2f} cm")
        # two cameras against each other directly, and through camera
        # A: the same turn if the calibration holds together
        for i in range(1, K):
            for j in range(i + 1, K):
                direct = camera_turn(pa[i][:, J], pa[j][:, J])[0]
                gap = np.degrees(Rot.from_matrix(
                    direct.T @ Rs[i].T @ Rs[j]).magnitude())
                say(f"  calibration: {names[i]} to {names[j]} directly and "
                    f"through A differ by {gap:.2f} deg")
    Cflip = [F_FLIP @ R @ F_FLIP for R in Rs]
    pose = [pa[0]] + [np.einsum('ij,fkj->fki', Rs[k], pa[k])
                      for k in range(1, K)]

    clr = [sightline_clearance(w) for w in wld]

    def locals_of(g):
        L = np.tile(np.eye(3), (n, N_JOINTS, 1, 1))
        for j in range(N_JOINTS):
            p = int(parents[j])
            if p >= 0:
                L[:, j] = np.einsum('fji,fjk->fik', g[:, p], g[:, j])
        return L
    Ls = [locals_of(g) for g in gr]

    # --- 2. which foot is planted (for the plant freeze) --------------
    feet_a = np.einsum('ij,fkj->fki', GA, wld[0][:, [ANK_R, ANK_L]])
    plant = {}
    for i, name in ((0, "leg R"), (1, "leg L")):
        h = feet_a[:, i, 1] - np.percentile(feet_a[:, i, 1], 95)
        v = np.concatenate([[0.0], np.linalg.norm(
            np.diff(feet_a[:, i], axis=0), axis=1)])
        plant[name] = gaussian_filter1d(
            ((np.abs(h) < 0.05) & (gaussian_filter1d(v, 2) < 0.012)
             ).astype(float), 1.0) > 0.5
        say(f"  {name} planted {100 * plant[name].mean():3.0f}% of frames")

    # with three or four cameras: how far each is from where the
    # cameras agree each limb joint is (and the shoulders, for the root)
    def blind(k, j):
        """0..1: this camera has no view of this joint on the frame
        (NO_VIEW): outside its picture, or the performer not seen by
        it at all (a person of a multi-person take).

        Every take since 2026-11-24 (the user: "Apply both the limb
        fix and the camera placer to multi-cam capture"). Before that
        only a person of a multi-person take, where it was seen and
        measured (limb joints more than 15 cm from where the seeing
        cameras put them on 5.2% of joint-frames without it, 2.8%
        with)."""
        out_j = (zero if outside[k] is None or j not in outside[k]
                 else outside[k][j])
        return np.clip(np.maximum(out_j, unseen[k]), 0.0, 1.0)

    sights_a = [_n(wld[0])] + [np.einsum('ij,fkj->fki', Rs[k], _n(wld[k]))
                               for k in range(1, K)]
    seeing = np.array([np.stack([1.0 - blind(k, j) for j in TARGETS], axis=1)
                       for k in range(K)])
    apart_j = None
    if K > 2:
        # (a camera with no view of a joint has no say in where the
        # cameras agree it is: two cameras guessing at a leg they cannot
        # see guess much the same thing, and outvoted the two that saw it)
        apart_j = cameras_apart(pose, sights_a, TARGETS, seen=seeing)
    # each camera's version of each limb joint against what the others
    # see of it (FIT_NEAR)
    fit_j = fits_others([p[:, TARGETS] for p in pose],
                        [s_[:, TARGETS] for s_ in sights_a], seeing)
    fit_best = np.min(np.where(seeing > 0.5, fit_j, np.inf), axis=0)
    fit_best = np.where(np.isfinite(fit_best), fit_best, 0.0)

    def disagrees(k, cols):
        """0..1: this camera is alone in where it puts these joints."""
        return np.clip((apart_j[k][:, cols].max(axis=1) - AGREE_NEAR)
                       / (AGREE_FAR - AGREE_NEAR), 0, 1)

    def others(vals, k):
        """What the other cameras show: the one other, or the middle
        of the others."""
        rest = [v for i, v in enumerate(vals) if i != k]
        return rest[0] if len(rest) == 1 else np.median(rest, axis=0)

    def best_other(vals, k):
        rest = [v for i, v in enumerate(vals) if i != k]
        return rest[0] if len(rest) == 1 else np.max(rest, axis=0)

    def take_text(Wl):
        if K == 2:
            w = Wl[1]
            return (f"take-B {100 * (w > 0.75).mean():3.0f}%  "
                    f"take-A {100 * (w < 0.25).mean():3.0f}%  "
                    f"blend {100 * ((w >= 0.25) & (w <= 0.75)).mean():3.0f}%")
        return "  ".join(f"{names[k]} {100 * Wl[k].mean():3.0f}%"
                         for k in range(K)) + "  (average share)"

    # --- 3. per-limb camera selection --------------------------------
    plan = {}       # joint -> (shares, pull), for every blended joint
    limb_w = {}     # limb -> each camera's share of it, per frame
    limb_log = []
    for limb, chain in CHAINS.items():
        bads = []
        for j in chain:
            ti = TARGETS.index(j)
            dev = []
            for k in range(K):
                if times[k] is None:
                    dev.append(_deviation(pose[k][:, j]))
                else:
                    dev.append(resample_frames(_deviation(np.einsum(
                        'ij,fj->fi', Rs[k], pa_own[k][:, j])), times[k]))
            # blocked sightline
            occ = [np.clip(-c[:, ti] / 0.04, 0, 1) for c in clr]
            # with room to spare
            room = [np.clip((c[:, ti] - 0.10) / 0.05, 0, 1) for c in clr]
            each = []
            for k in range(K):
                t_k, t_o = dev[k], others(dev, k)
                # flying away from its own robust trend
                absk = np.clip((t_k - 0.03) / 0.05, 0, 1)
                # ...and worse than the other cameras at the same moment
                rel = (np.clip((t_k - 0.02) / 0.04, 0, 1)
                       * np.clip(t_k / (t_o + 1e-6) - 1, 0, 1))
                b = np.clip(np.maximum(np.maximum(occ[k], rel), absk), 0, 1)
                # seen by this camera AND another: the two "far from
                # its own path" tests give way to how well this
                # camera's version fits the others' pictures, against
                # the camera that fits best (FIT_NEAR)
                shared = (seeing[k][:, ti] > 0.5) & (np.max(
                    [seeing[o_][:, ti] for o_ in range(K) if o_ != k],
                    axis=0) > 0.5)
                rank = np.clip((fit_j[k][:, ti] - fit_best[:, ti] - FIT_NEAR)
                               / (FIT_FAR - FIT_NEAR), 0, 1)
                b = np.where(shared, np.maximum(
                    occ[k], gaussian_filter1d(rank, 1.0)), b)
                # this one blocked, another with room to spare: not a
                # close call
                dom = best_other(room, k) * occ[k]
                # the tracker failed on this frame, or the joint was not
                # in the picture (the single-camera tests)
                tests = [b, dom, fail[k],
                         zero if outside[k] is None else outside[k][j]]
                if apart_j is not None:
                    # the other cameras agree with each other and not
                    # with this one
                    tests.append(disagrees(k, [ti]))
                each.append(np.maximum.reduce(tests) + NO_VIEW * blind(k, j))
            bads.append(each)

        # the worst joint speaks for the limb: mixing weights within one
        # chain makes the limb incoherent and costs jitter
        bad = [np.max([x[k] for x in bads], axis=0) for k in range(K)]
        conf = gaussian_filter1d(np.max(bad, axis=0) - np.min(bad, axis=0)
                                 if K > 2 else np.abs(bad[0] - bad[1]), 0.8)
        alpha = ALPHA + (ALPHA_SURE - ALPHA) * np.clip((conf - 0.25) / 0.35, 0, 1)
        Wl = _settle_shares(camera_shares(bad), plant.get(limb))

        limb_w[limb] = Wl
        for j in chain:
            # the joint itself, and the helper joints that hang off it
            for jj in [j] + [int(k) for k in np.flatnonzero(parents == j)
                             if int(k) not in chain]:
                plan[jj] = (Wl, alpha)
        limb_log.append(f"  {limb:<6} " + take_text(Wl))
    # collarbones, neck and head: shared evenly, except on a frame
    # where one camera's tracker failed
    w_torso = [np.full(n, 1.0 / K) for _ in range(K)]
    if K == 2:
        w_torso[0] = 1.0 - w_torso[1]
    if any(x.any() for x in fail):
        w_torso = _settle_shares(camera_shares(fail))
    for j in TORSO:
        plan[j] = (w_torso, None)

    # --- 4. the root: arbitrated, and blended per axis ----------------
    # every camera's rotations, in camera A's frame
    gr_c = [gr[0]] + [np.einsum('ij,fnjk->fnik', Cflip[k], gr[k])
                      for k in range(1, K)]
    tiR, tiL = TARGETS.index(HIP_R), TARGETS.index(HIP_L)
    bR = []
    for k in range(K):
        occR = np.clip(-0.5 * (clr[k][:, tiL] + clr[k][:, tiR]) / 0.04, 0, 1)
        # the single-camera tests again: the tracker failed, or a hip
        # was not in the picture
        hips = zero if outside[k] is None else np.maximum(
            outside[k][HIP_R], outside[k][HIP_L])
        devR = (_rot_deviation(gr_c[k][:, HIPS]) if times[k] is None
                else resample_frames(_rot_deviation(np.einsum(
                    'ij,fjk->fik', Cflip[k], root_own[k])), times[k]))
        tests = [occR, fail[k], hips, np.clip((devR - 3.0) / 6.0, 0, 1)]
        if apart_j is not None:
            tests.append(disagrees(k, [TARGETS.index(SHO_R),
                                       TARGETS.index(SHO_L)]))
        bR.append(np.clip(np.maximum.reduce(tests), 0, 1))
    conf_r = gaussian_filter1d(np.max(bR, axis=0) - np.min(bR, axis=0)
                               if K > 2 else np.abs(bR[0] - bR[1]), 0.8)
    al_root = ALPHA + (ALPHA_SURE - ALPHA) * np.clip((conf_r - 0.25) / 0.35, 0, 1)
    # (the same for the root: a camera with the hips out of its picture,
    # or the performer not in it, has no share of the pelvis while
    # another camera sees it. bR itself stays 0..1: the position fusion
    # below reads it as a weight.)
    w_root = _settle_shares(camera_shares(
        [bR[k] + NO_VIEW * np.maximum(blind(k, HIP_R), blind(k, HIP_L))
         for k in range(K)]))

    # a camera sees rotation ABOUT its view axis well, away from itself
    # poorly - so weight each rotation axis by who can actually see it.
    # The axes are WORLD axes (q_blend_axes works in the world): until
    # this was applied in the hips' own frame instead, which is the
    # same thing only while the performer faces camera A, and backwards
    # once they have turned side-on to it.
    view = [F_FLIP @ np.array([0.0, 0.0, 1.0])] + [
        F_FLIP @ (Rs[k] @ np.array([0.0, 0.0, 1.0])) for k in range(1, K)]
    Pv = [(1 - K_AXIS) * np.eye(3) + K_AXIS * np.outer(d, d) for d in view]
    root_new = q_blend_axes_many(
        [g[:, HIPS] for g in gr_c], w_root, al_root,
        [np.tile(P, (n, 1, 1)) for P in Pv])
    # The spine is fused per axis too, and it goes by the root's
    # weight: fusing the pelvis and the chest on different weights
    # invents a bend at the waist that neither camera contains.
    for j in SPINE:
        plan[j] = (w_root, al_root)

    # --- 5. every other joint, parents first --------------------------
    # In camera A's frame, y-up like global_rots. pos_f: each camera's
    # own skeleton; sight: the direction that camera looks along to
    # reach each joint.
    pos_f = [p @ F_FLIP for p in pose]
    sight = [_n(wld[0]) @ F_FLIP] + [_n(wld[k]) @ Rs[k].T @ F_FLIP
                                     for k in range(1, K)]
    G_new = np.tile(np.eye(3), (n, N_JOINTS, 1, 1))
    L_new = Ls[0].copy()
    for j in order:
        p = int(parents[j])
        if p < 0 or j == HIPS:
            G_new[:, j] = root_new
            continue
        if j in plan:
            Wl, alpha = plan[j]
            if j in AXIS_LEVERS:
                # The joint's orientation in the WORLD is what each
                # camera observed, so that is what gets fused; the
                # local rotation is then whatever takes the merged
                # parent there. (Fusing the local rotations instead
                # carries each camera's error in the PARENT down the
                # limb, and measured clearly worse.)
                lev = AXIS_LEVERS[j]
                g = q_blend_axes_many(
                    [x[:, j] for x in gr_c], Wl, alpha,
                    [axis_precision(sight[k][:, j],
                                    [pos_f[k][:, t] - pos_f[k][:, j]
                                     for t in lev]) for k in range(K)])
                L_new[:, j] = np.einsum('fji,fjk->fik', G_new[:, p], g)
            else:
                L_new[:, j] = q_blend_many([L[:, j] for L in Ls], Wl, alpha)
        G_new[:, j] = np.einsum('fij,fjk->fik', G_new[:, p], L_new[:, j])

    def frames(x):
        return int((np.asarray(x) > 0).sum())
    say("single-camera tests: tracker failed on "
        + " / ".join(f"{names[k]} {frames(fail[k])}" for k in range(K))
        + " frames (counting one frame either side)")
    if any(x is None for x in outside):
        say("  limb outside the picture: not checked, no frame size "
            "recorded by camera "
            + " and ".join(names[k] for k in range(K)
                           if outside[k] is None))
    else:
        say("  a limb outside the picture on "
            + " / ".join(
                f"{names[k]} "
                f"{frames(np.max(list(outside[k].values()), axis=0))}"
                for k in range(K)) + " frames")
    say("per-limb decisions:")
    for line in limb_log:
        say(line)
    say("root: " + take_text(w_root))
    tot_P = Pv[0]
    for P in Pv[1:]:
        tot_P = tot_P + P
    for k in range(1, K):
        say(f"  root rotation, {names[k]} share per axis "
            f"{np.diag(np.linalg.inv(tot_P) @ Pv[k]).round(2)} "
            f"({1.0 / K:.2g} would be a flat average)")

    # the merge lives in camera A's frame, so A's floor is its gravity
    gr_new = np.einsum('ij,fnjk->fnik', F_FLIP @ GA @ F_FLIP, G_new)

    # --- 6. anisotropic position fusion ------------------------------
    k_an = 0.95
    size = 1.0
    if cameras is None:
        Rp = Rs
        hips_in_a = [wld[0][:, HIPS]] + [np.einsum('ij,fj->fi', Rs[k],
                                                   wld[k][:, HIPS])
                                         for k in range(1, K)]
        # where each camera stands against camera A: from the T-pose
        offs = [np.zeros(3)] + [np.median(hips_in_a[0][W[0]:W[1]]
                                          - hips_in_a[k][W[0]:W[1]], axis=0)
                                for k in range(1, K)]
        # ...and then from the whole take, when the take settles it
        # (place_from_take; the turns are kept, so hips_in_a stands)
        own = place_from_take(wld, Rs, offs, fail, outside, say)
        if own is not None:
            offs = own
    else:
        # The take's own cameras, placed once from everybody in it, and
        # this person's real size against the size the capture drew
        # them at. A camera takes a smaller body to be further away by
        # the same proportion, so its whole idea of where they are
        # scales with their size (the direction it sees them in does
        # not change).
        Rp = [np.asarray(r, dtype=np.float64) for r in cameras["turns"]]
        offs = [np.asarray(t, dtype=np.float64) for t in cameras["places"]]
        size = float(cameras.get("scale", 1.0))
        hips_in_a = [size * np.einsum('ij,fj->fi', Rp[k], wld[k][:, HIPS])
                     for k in range(K)]
        say(f"  position: the take's cameras (placed once for everybody), "
            f"this person {size:.3f} of the body model's size")
    placed = [hips_in_a[0]] + [hips_in_a[k] + offs[k] for k in range(1, K)]
    axes = [np.array([0.0, 0.0, 1.0])] + [Rp[k] @ np.array([0.0, 0.0, 1.0])
                                          for k in range(1, K)]
    Ms = [(1.0 - 0.9 * bR[k])[:, None, None]
          * (np.eye(3) - k_an * np.outer(axes[k], axes[k]))
          for k in range(K)]
    tot_M = Ms[0]
    rhs = np.einsum('fij,fj->fi', Ms[0], placed[0])
    for k in range(1, K):
        tot_M = tot_M + Ms[k]
        rhs = rhs + np.einsum('fij,fj->fi', Ms[k], placed[k])
    # One 3x3 system per frame. The right-hand side goes in as a column
    # (..., 3, 1): since numpy 2.0, solve() reads a 2-D b as a MATRIX,
    # not a stack of vectors, and (n, 3, 3) against (n, 3) raises.
    pos = np.linalg.solve(tot_M + 1e-6 * np.eye(3), rhs[..., None])[..., 0]

    def wobble(p):
        return np.linalg.norm(p - gaussian_filter1d(p, 8, axis=0), axis=1).std()
    mean_pos = placed[0]
    for p in placed[1:]:
        mean_pos = mean_pos + p
    say("position wobble: "
        + "  ".join(f"{names[k]} {wobble(placed[k]) * 100:.2f}"
                    for k in range(K))
        + f"  average {wobble(mean_pos / K) * 100:.2f}"
        f"  ANISO {wobble(pos) * 100:.2f} cm")
    for k in range(1, K):
        say(f"  cameras A and {names[k]} disagree on body position by p50 "
            f"{np.median(np.linalg.norm(placed[0] - placed[k], axis=1)) * 100:.1f} cm")
    traj = np.einsum('ij,fj->fi', GA, pos)

    lows = [np.percentile(np.einsum('ij,fkj->fki', GA, size * wld[0][
        :, [ANK_R, ANK_L]])[:, :, 1], 95)]
    for k in range(1, K):
        lows.append(np.percentile(np.einsum('ij,fkj->fki', GA, size * np.einsum(
            'ij,fkj->fki', Rp[k], wld[k][:, [ANK_R, ANK_L]])
            + offs[k])[:, :, 1], 95))
    say("  floor: " + "  ".join(f"{names[k]} {lows[k]:.3f}" for k in range(K))
        + f" m (differ {(max(lows) - min(lows)) * 100:.1f} cm)")

    # --- write ---------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for k in A.files:
        v = A[k]
        out[k] = (v[cuts[0]][:n] if (hasattr(v, "ndim") and v.ndim > 0
                                     and len(v) == len(A["cam_t"])) else v)
    out["global_rots"] = gr_new.astype(np.float32)
    out["cam_t"] = (traj - jc[0][0, HIPS][None, :]).astype(np.float32)
    # Tells the BVH export this cam_t is already gravity-aligned, so the
    # camera-pitch detrend must not run: its depth axis is now a WORLD
    # direction the performer walks along, not a camera view axis, and
    # regressing height against it removes real vertical motion.
    # 0 when the take could not be levelled: it is then in camera A's
    # own frame, and the detrend runs as it does for one camera. (The
    # key is also how the export knows a take is merged: keep it.)
    out["gravity_aligned"] = np.int64(1 if levelled else 0)
    # joint_coords must describe the SAME body the output rotations
    # render. Camera A's jc do not: wherever a limb came from another
    # camera the rendered pose and A's positions disagree by up to ~10
    # cm, and everything downstream that reads jc (floor grounding,
    # Feet Define the Floor plant detection, foot locking) is then
    # referenced to feet that are not the ones on screen. So rebuild jc
    # by FK from the merged rotations (same pos[j] = pos[p] + G[p] @
    # off[j] the export uses; feet land within ~1 cm of SAM's own jc),
    # flip back to the NPZ's y-down frame, and re-pin the hips anchor.
    # (a person of a multi-person take is drawn at their own size:
    # `size` is 1.0 for everybody else)
    offs_m = offsets_cm * 0.01 * size
    fk = np.zeros((n, N_JOINTS, 3))
    for j in order:
        p = int(parents[j])
        if p < 0:
            fk[:, j] = offs_m[j]
        else:
            fk[:, j] = fk[:, p] + gr_new[:, p] @ offs_m[j]
    jc_new = np.einsum('ij,fkj->fki', F_FLIP, fk - fk[:, HIPS:HIPS + 1]) \
        + jc[0][0, HIPS][None, None, :]
    out["joint_coords"] = jc_new.astype(np.float32)
    # The measured floor level, so the export can ground the take
    # absolutely instead of calling the clip's single lowest frame the
    # floor (one downward foot glitch floats the whole take). Same
    # foot/toe joints and world convention (jc + cam_t, y-down) the
    # export's own grounding metric uses; p95 over the whole clip is
    # robust to the downward spikes that break the min-frame rule.
    wout = out["joint_coords"][:, [5, 7, 21, 23], :] + \
        out["cam_t"][:, None, :]
    out["floor_y"] = np.float64(np.percentile(wout[:, :, 1], 95))
    more = {}
    if cameras is not None:
        # The person's size against the body model's, for the export to
        # draw the skeleton at (modes.run_body). Only a person of a
        # multi-person take has it; no other capture's files change.
        out["body_scale"] = np.float64(size)
        # body_scale again, and the turns WHERE the body is was worked
        # out with (the take's; `turns` stays this performer's own)
        more = {"body_scale": np.float64(size), "place_turns": np.stack(Rp)}
    np.savez(os.path.join(out_dir, "mhr_frames.npz"), **out)
    # What everything that goes back to the cameras' own captures
    # reads (Rebuild Finger Motion through modes.merged_take_cameras,
    # Add Camera through capture_camera). Camera A first in every
    # per-camera array:
    #   cameras          how many were merged
    #   turns, offsets   each camera's rotation into camera A's frame
    #                    and where it stands in it
    #   starts           the frame of each camera's capture that the
    #                    merged take's first frame is
    #   sync_shifts      frames each camera was read later than its own
    #                    frame number, per frame (fine sync; zeros when
    #                    it changed nothing). Whatever reads a camera's
    #                    capture has to apply it too (resample_frames).
    #   arm_right, arm_left   each camera's share of that arm, per
    #                    frame: which camera saw each hand
    # GA: camera A's frame -> the levelled take's.
    # sync_text / sync_act: what fine sync found, and whether the user
    # has to do something about it; sync_act_text is the part of it
    # they have to act on (multicam.merge_lines reports that).
    # level_text / level_act: how the take was levelled, and 1 when it
    # could not be (nothing to level by, so it was left in camera A's
    # frame). Printed above; nothing reads them back.
    # R, yaw, GB, offset, sync_offset, sync_shift, arm_w_right,
    # arm_w_left: camera B's, under the names a two-camera merge has
    # always written them by.
    shift_all = np.stack([np.zeros(n) if s is None else s for s in shifts])
    np.savez(os.path.join(out_dir, "calibration.npz"), R=Rs[1], yaw=yaws[1],
             GA=GA, GB=Gs[1], offset=offs[1], sync_offset=o,
             level_text=np.str_(level_text),
             level_act=np.int64(0 if levelled else 1),
             sync_shift=shift_all[1].astype(np.float32),
             sync_text=np.str_("\n".join(sync_lines)),
             sync_act=np.int64(1 if sync_acts else 0),
             sync_act_text=np.str_("\n".join(sync_acts)),
             arm_w_right=limb_w["arm R"][1].astype(np.float32),
             arm_w_left=limb_w["arm L"][1].astype(np.float32),
             cameras=np.int64(K), turns=np.stack(Rs), offsets=np.stack(offs),
             yaws=np.array(yaws), starts=np.array(starts, dtype=np.int64),
             sync_shifts=shift_all.astype(np.float32),
             arm_right=np.stack(limb_w["arm R"]).astype(np.float32),
             arm_left=np.stack(limb_w["arm L"]).astype(np.float32), **more)

    merge_face(dirs, out_dir, cuts, n, say, shifts)
    say(f"wrote {out_dir}")


FACE_STEADY_MIN = 12     # pairs of frames a camera's face needs before its steadiness is judged
FACE_EASE_S = 0.25       # a change of camera slides over this long instead of cutting
FACE_EASED = ("face_blendshapes", "eye_gaze", "face_landmarks")
# the eye outlines the export places a face by (face_export.compute_head_local_landmarks)
FACE_L_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
FACE_R_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


def _face_placed_as(src, dst):
    """Landmarks `src` (478, 3: one camera's picture) moved, turned and
    sized in the picture so their eyes sit on those of `dst` (another
    camera's): the first camera's face SHAPE where the second camera
    sees the face. Unchanged when either has no usable eyes."""
    if src.shape != dst.shape or src.ndim != 2 or len(src) <= max(FACE_L_EYE + FACE_R_EYE):
        return src
    sl, sr = src[FACE_L_EYE].mean(axis=0), src[FACE_R_EYE].mean(axis=0)
    dl, dr = dst[FACE_L_EYE].mean(axis=0), dst[FACE_R_EYE].mean(axis=0)
    sv, dv = (sr - sl)[:2], (dr - dl)[:2]
    ns, nd = float(np.linalg.norm(sv)), float(np.linalg.norm(dv))
    if ns < 1e-6 or nd < 1e-6:
        return src
    k = nd / ns
    a = np.arctan2(dv[1], dv[0]) - np.arctan2(sv[1], sv[0])
    c, s = np.cos(a), np.sin(a)
    rel = (src - (sl + sr) / 2) * k
    out = np.empty_like(src, dtype=float)
    out[:, 0] = c * rel[:, 0] - s * rel[:, 1]
    out[:, 1] = s * rel[:, 0] + c * rel[:, 1]
    out[:, 2] = rel[:, 2]
    return out + (dl + dr) / 2


def merge_face(dirs, out_dir, cuts, n, say, shifts=None):
    """The face comes whole from ONE camera at a time, and that camera
    is kept for as long as it has the face. Only when it loses the face
    is another taken: of the cameras that have the face on that frame,
    the one whose reading is the steadiest over the take (FACE_STEADY:
    how much its blendshapes change from one frame to the next; all
    the cameras watch the same performance, so what differs between
    them is noise). Level on that, the one that goes on having the
    face for longest, then the larger face. A face is 2D data: every
    camera reads the same face a little differently, so each change of
    camera is a jump in the expression, and nothing is gained by
    changing for a bigger view (the earlier rule, the largest detection
    on each frame, changed camera 36 times in a 36 second take). Frames
    on which no camera has the face are frames without a face, as in a
    single-camera capture.

    A change of camera with no such frame in between is eased: over
    FACE_EASE_S the expression (and the gaze, and the shape of the
    landmarks, placed where the new camera sees the face) slides from
    the last reading of the camera left to the readings of the camera
    taken. That is a blend in time at the handover, never of two
    cameras' data on one frame. After frames without a face there is
    nothing to ease from: the face is at rest there, as it is after any
    dropout of a single-camera capture.

    A camera with no face data at all (it never saw the face: a camera
    behind the performer) is not a reason to drop the face: the others'
    is used. With none, a merged face file left by an earlier run is
    removed, so the take cannot pick up a face that does not belong to
    it.

    shifts: fine sync's per-frame offset for each camera, None for one
    it left alone (measure_sync). The face is taken whole from one
    camera, so a re-timed camera's frames are read at the nearest
    frame, not mixed."""
    K = len(dirs)
    names = LETTERS[:K]

    def cut(k, v):
        v = v[cuts[k]][:n]
        if shifts is None or shifts[k] is None:
            return v
        return resample_frames(v, np.arange(len(v)) - shifts[k][:len(v)],
                               "nearest")
    paths = [os.path.join(d, "face_frames.npz") for d in dirs]
    out_face = os.path.join(out_dir, "face_frames.npz")
    have = [k for k in range(K) if os.path.exists(paths[k])]
    if not have:
        say("face: " + ("neither capture has" if K == 2 else "no capture has")
            + " face data, skipping")
        if os.path.exists(out_face):
            os.remove(out_face)
        return
    if len(have) == 1:
        k = have[0]
        F1 = np.load(paths[k], allow_pickle=True)
        ln1 = len(np.asarray(F1["face_detected"]))
        one = {}
        for key in F1.files:
            v = np.asarray(F1[key])
            one[key] = cut(k, v) if (v.ndim > 0 and len(v) == ln1) else v
        np.savez(out_face, **one)
        say(f"face: only camera {names[k]} has face data, using it "
            f"({100 * np.asarray(one['face_detected']).mean():.1f}% of frames)")
        return
    Fs = [np.load(paths[k], allow_pickle=True) for k in have]
    ln = len(np.asarray(Fs[0]["face_detected"]))
    det = [cut(k, np.asarray(F["face_detected"])) for k, F in zip(have, Fs)]
    area = []
    for k, F in zip(have, Fs):
        b = cut(k, np.asarray(F["face_bbox"]).astype(float))
        area.append(np.abs((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
    # which camera has the face on each frame: the one in use while it
    # still has it, else the steadiest one that has it
    D, A = np.stack(det), np.stack(area)
    rough = []                            # lower = steadier
    for k, F, d in zip(have, Fs, det):
        both = d[1:] & d[:-1]
        if "face_blendshapes" in F.files and both.sum() >= FACE_STEADY_MIN:
            b = cut(k, np.asarray(F["face_blendshapes"]).astype(float))
            rough.append(float(np.abs(b[1:] - b[:-1])[both].mean()))
        else:
            rough.append(np.inf)          # too little to judge: last
    T = D.shape[1]
    keeps = np.zeros(D.shape, int)        # frames it still has the face for
    for f in range(T - 1, -1, -1):
        keeps[:, f] = np.where(D[:, f],
                               1 + (keeps[:, f + 1] if f + 1 < T else 0), 0)
    pick = np.zeros(T, int)
    cur, changes = -1, 0
    for f in range(T):
        if D[:, f].any() and (cur < 0 or not D[cur, f]):
            new = min((i for i in range(len(have)) if D[i, f]),
                      key=lambda i: (rough[i], -keeps[i, f], -A[i, f]))
            changes += cur >= 0
            cur = new
        pick[f] = max(cur, 0)             # (no face anywhere yet: the first)
    fout = {}
    for key in Fs[0].files:
        va = np.asarray(Fs[0][key])
        if va.ndim > 0 and len(va) == ln:
            rows = [cut(k, np.asarray(F[key])) for k, F in zip(have, Fs)]
            res = rows[0]
            for i in range(1, len(rows)):
                res = np.where((pick == i).reshape(
                    (-1,) + (1,) * (res.ndim - 1)), rows[i], res)
            fout[key] = res
        else:
            fout[key] = va
    # ease the straight handovers
    fps = float(np.asarray(Fs[0]["fps"])) if "fps" in Fs[0].files else 24.0
    E = max(1, int(round(FACE_EASE_S * fps)))
    eased = 0
    for f in range(1, T):
        new, old = pick[f], pick[f - 1]
        if new == old or not (D[new, f] and D[old, f - 1]):
            continue
        m = min(E, int(keeps[new, f]))
        eased += 1
        for key in FACE_EASED:
            v = fout.get(key)
            if v is None or v.ndim < 2 or len(v) != T or v.dtype.kind != "f":
                continue
            v = fout[key] = np.array(v)   # (our own copy)
            hold = v[f - 1].astype(float)
            for i in range(m):
                x = (i + 1) / (E + 1.0)
                w = x * x * (3.0 - 2.0 * x)
                goal = v[f + i].astype(float)
                frm = (_face_placed_as(hold, goal)
                       if key == "face_landmarks" else hold)
                v[f + i] = (1.0 - w) * frm + w * goal
    any_det = det[0]
    for d in det[1:]:
        any_det = any_det | d
    fout["face_detected"] = any_det
    np.savez(out_face, **fout)
    if K == 2:
        say(f"face: A {100 * det[0].mean():.1f}%  B {100 * det[1].mean():.1f}%  "
            f"merged {100 * any_det.mean():.1f}%  (B used on "
            f"{100 * (pick == 1).mean():.0f}%, {changes} "
            f"change{'' if changes == 1 else 's'} of camera, "
            f"{eased} eased)")
    else:
        say("face: " + "  ".join(f"{names[k]} {100 * d.mean():.1f}%"
                                 for k, d in zip(have, det))
            + f"  merged {100 * any_det.mean():.1f}%  (used: "
            + ", ".join(f"{names[k]} {100 * (pick == i).mean():.0f}%"
                        for i, k in enumerate(have))
            + f"; {changes} change{'' if changes == 1 else 's'} of camera, "
            f"{eased} eased)")


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cams", nargs="+", metavar="CAM_DIR",
                    help="the cameras' capture folders, camera A first "
                         "(2 to 4)")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--sync-offset", type=int, nargs="+", default=[0],
                    help="frames camera A leads each other camera by, in "
                         "order (negative if that camera leads). Sync by "
                         "hand to within a few frames; the rest is "
                         "measured and corrected.")
    ap.add_argument("--mhr-model", default=os.path.join(
        os.environ.get("BLENDCAP_CHECKPOINTS")
        or os.path.join(here, "checkpoints"),
        "sam-3d-body-dinov3", "assets", "mhr_model.pt"))
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()
    if not 2 <= len(a.cams) <= len(LETTERS):
        ap.error(f"give 2 to {len(LETTERS)} capture folders")
    merge_cameras(a.cams, a.out, a.mhr_model, a.sync_offset, a.quiet)


if __name__ == "__main__":
    main()
