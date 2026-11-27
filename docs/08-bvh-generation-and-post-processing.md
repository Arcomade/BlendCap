# BVH Generation and Post-Processing

A capture only produces tracking data. This is where you turn that data into an animated armature (a **BVH**) and clean it up: the **BVH Settings** section generates the armature, and the **BVH Post Processing** controls inside it shape the result.

---

## Generating the BVH

Expand the **BVH Settings** section, choose what the armature should include, and click **Generate/Reload BVH**. BlendCap builds the armature from your selected clip's capture and imports it into the scene.

The included toggles:

- **Body**: the full set of body bones.
- **Hands**: finger articulation.
- **Face**: facial motion as face bones (see [Face Capture](06-face-capture.md)).
- **Root Motion**: the actor's travel along the ground plane. Turn it off to generate the performance in place, for example when you'd rather animate the character's path yourself, or when a moving camera has bled its own travel into the capture.
- **Use Separate Face Data**: combine the selected clip's body with another clip's face capture (see [Face Capture](06-face-capture.md)).
- **Rebuild Finger Motion (Experimental)**: reconstruct the fingers from their tracked keypoints instead of the default finger motion, recovering distinct poses like a point or a counted number. Available when **Hands** is enabled; covered in full in [Hand & Finger Tracking](07-hand-tracking.md).
- **Hold Cropped Limbs**: holds body parts still when they're outside the picture, instead of animating them from a best guess (see "Hold Cropped Limbs" below).

![The BVH Settings section](images/16-bvh-settings.png)

> **How generated armatures are named.** Each armature (and its BVH file in the library) is named after its clip plus letters for what it contains: `B` body, `H` hands, `R` root motion, `F` face, so `walk_BHRF` is a full export of the clip "walk", and a separate face capture shows as `walk_BH-closeup_F`. Re-generating with the same toggles **replaces** that armature in the scene, and overwrites the previous iteration's data on disk, while changing the toggles creates a differently named armature **alongside** the old one. The other capture modes add to the name: a multi-camera take puts its number of cameras in front of the letters (`A0314_2cBHR` for 2 cameras), and a multi-person capture adds each person's number (`walk_P2_BHR`, or `A0314_P2_4cBHR` with 4 cameras).

## Applying changed settings

Every post-processing control below is **baked in when the BVH is generated**, so after changing them, apply them in one of two ways:

- **Generate/Reload BVH** builds the armature for the **active clip**, and it's the only button that applies the toggles above. Use it for the initial generation, after a re-capture, or whenever you change which body parts or motion data the armature should include. It always starts fresh, so any camera-angle correction baked into a previous armature is not carried over.
- **Bake Settings to BVH** re-generates the **currently selected armature**. The armature's composition does not change, but any modified post-processing settings will be applied. The armature's source clip does not need to be active. This is the only button that writes the Camera Angle Offset sliders into the BVH file (each bake stacks on top of the last).

Post-processing only changes how the capture is cleaned up, it never re-runs the capture itself, so each re-generation is much faster than the capture stage. Experiment freely to find the right amount for each clip.

---

## Hold Cropped Limbs

Normally BlendCap animates the whole body even when part of it is outside the picture, using the body model's best guess for what it can't see. On footage framed from the waist up, that means the legs get animated even though they were never filmed.

**Hold Cropped Limbs** holds those parts still instead. Legs that are out of frame stay in a standing pose, and arms, hands and the head keep the pose they had when they left the frame. It's worked out frame by frame from the edges of the picture, once a body part has been out of frame for about half a second.

- If the hips themselves are out of frame, the body stays in place and upright and only turns with the shoulders.
- It only cares about the edges of the frame. A hand hidden behind the body is not detected.
- It needs a capture made with BlendCap 1.1.0 or newer, so re-capture older clips before using it.

See [Filming Tips](13-filming-tips.md) for how to frame upper-body shots.

---

## Add Camera

The **Camera** section at the bottom of **BVH Settings** allows you to add a camera to your scene that attempts to match the capture footage's viewpoint: placed where the real camera stood, with a matching lens, and with the source video as its background. Look through it and you'll see your armature, and later your character, lined up over the footage.

![Add Camera](images/38-add-camera.png)

- It works on the selected BlendCap armature. With nothing selected, it uses the selected clip's armature, as long as that clip only has one person tracked.
- You can click it any time after the armature has been generated, there's no need to generate again. When the button is greyed out, its tooltip tells you why.
- The camera is a regular Blender camera that sits next to the armature, it isn't parented to it. Move, edit or delete it freely. It isn't set as the scene's active camera unless it's the only camera available.
- Generating or baking the armature again, or adjusting **Camera Angle Offset**, keeps the camera matched.
- On a multi-camera take, a **Camera View** dropdown above the button sets which camera angle to use.
- It needs a capture made with BlendCap 1.1.0 or newer.

A few things to keep in mind. It's one fixed camera per clip, so camera movement in the footage isn't tracked. The view only lines up when your scene's render resolution has the same shape as the video. It matches best with **Root Motion** enabled. And a tilted camera still needs **Camera Angle Offset**, BlendCap doesn't detect the tilt for you on single-camera captures. If you'd rather not have the footage behind the camera, untick **Show Footage in BlendCap Cameras** in [Preferences](11-preferences.md), or turn off background images in the camera itself.

---

## Multi-person captures

When the selected capture is a multi-person one, **BVH Settings** gains a **People** list at the top, a **Generate/Reload All** button, and a **Keep Captured Position** checkbox. They're covered on the [Multi-Person Capture](05b-multi-person-capture.md) page.

---

## Post-processing controls

Single-camera capture is never perfectly clean: feet slide and float, depth jitters, and the whole performance can come in tilted. These controls mitigate most of these issues without the need for manual cleanup.

![The BVH Post Processing controls](images/16-1-post-processing-panel.png)

---

## Camera Angle Offset

Sometimes a captured performance comes in tilted, the whole body leaning forward, back, or to one side. It's an occasional quirk of reconstructing 3D motion from a single camera. (Multi-camera takes are levelled automatically, and rarely need it.) **Camera Angle Offset** straightens it, per armature, with a live viewport preview:

- **Pitch** and **Roll** tilt the whole body forward/back and side to side. Adjust until your character stands upright, the viewport updates as you drag.
- **Height** shifts the whole performance up or down, for when the floor level was misjudged and the character rides above or sinks below your floor.
- **Head Angle Offset** adds an extra **Head Pitch** / **Head Roll** for just the head bone, for when the head still reads as looking too far up or down after the body is corrected.

These sliders preview live on the selected armature; click **Bake Settings to BVH** to write them into the BVH file itself (which is what retargeting and exports read).

![Straightening a tilted performance](images/17-camera-offset.png)

> **Baking stacks.** Each bake adds the current slider values on top of what's already baked into that armature, and the sliders reset to zero afterward. One thing to know: if you re-import a clip from your library, BlendCap can't read how much was previously baked into it, so the panel reminds you that the first bake on a re-imported armature starts fresh.

---

## Jitter Control

The **Jitter Control** section groups the four tools that calm unwanted shake: **Smoothing**, **Depth Noise Filtering**, **Head Noise Filtering** and **Face Noise Filtering**. Each one is covered below.

![The Jitter Control section](images/39-jitter-control.png)

---
## Smoothing

Time-based smoothing that reduces frame-to-frame jitter, measured in **seconds** (0 = off). **Enable Smoothing** turns the whole section on and off (on by default), and there are separate values for the **Face**, the **Head**, the **Body**, the **Root**, and **Vertical**. The head and neck are left out of Body smoothing and use the **Head** value instead. Root smoothing works along the ground plane only; the **Vertical** slider handles up-and-down motion (jumps, squats) separately, with a deliberately small default so jump arcs survive the smoothing. A little goes a long way, too much and you start to lose the snap of fast motion, so start near the defaults and adjust until the jitter is gone but the performance still feels alive.

> **More than one performer in fast action?** Root smoothing rounds off each performer's path on its own, so in a fast exchange (a fight scene, for example) it can change the distance between them for a moment. Lower the **Root** value if they drift apart or into each other.

---

## Depth Noise Filtering

The trickiest axis for any single-camera system is **depth** (toward and away from the camera), there's simply less information there, so it's where most of the wobble lives. Depth Noise Filtering calms it while leaving real movement (steps, reaches) untouched.

There are separate strengths for the **Feet**, **Hands**, **Head**, and **Root** (the whole body's depth position), each measured in centimeters: motion smaller than the threshold is treated as noise and smoothed away, while larger, real motion passes through. It's on by default with safe values, increase a slider if a particular body part still shimmers in depth, or set it to 0 to leave that part untouched.

>If your character's entire body is randomly shifting by large amounts towards and away from the camera, try setting your capture's FOV mode to **Sample first frame only** (the default) and re-capturing (See [Capturing](05-capturing.md)), or turning off your BVH's **Root Motion** (See **Generating the BVH** above) instead of messing with depth noise filtering or smoothing settings. 

---

## Head Noise Filtering

A head that trembles or nods slightly on its own is easy to spot in a capture. **Head Noise Filtering** holds the head steady: small rotations below a threshold are treated as noise and removed, while deliberate nods, turns and tilts pass through untouched. It's on by default, with a threshold in degrees for each direction:

- **Nod**: up-and-down wobble.
- **Turn**: left-right wobble.
- **Tilt**: ear-to-shoulder wobble.

Raise a value if the head still shakes in that direction, or set it to 0 to leave that direction alone. Untick **Use Head Noise Filtering** to get the raw head motion back.

> **Which control for which head problem?** If your character's head is looking around weirdly and seems to be popping from one position to the next, use **Head Smoothing**. If your character's head is consistently vibrating over an entire take, but the larger motions all seem fine, use **Head Noise Filtering**. Generally the default values work fine on most footage, and you shouldn't need to touch these values much if at all.

---

## Face Noise Filtering

Snaps very small movements in face capture animations to zero, per facial region, so a trembling jaw or flickering eyes go quiet without deadening real expressions. Raise a region's value only as far as needed to kill its jitter. Very small values usually work best here (around 0.05 - 0.2).

---
## Foot Locking

Usually the most valuable cleanup step: it detects when a foot is in contact with the ground, pins it to the floor, and helps to remove the sliding ("footskate") or jittery vertical motion in the feet. **Enable Foot Locking** is on by default, and its settings live in the **Foot Locking Settings** sub-section:

> Note that these settings will help remove footskate from the feet themselves, but cannot repair any sliding that comes from the armature's root motion, which needs to be cleaned up manually if present.

- **Feet Define the Floor**: keeps the character on the ground plane while trying to conserve jumps. Feet in contact with the ground stay pinned to it. Helps with fixing floating captures or knees not bending enough. **Weaknesses**: if camera motion follows a jump this setting may not detect it properly, and feet dangling in the air during seated or hanging positions can be mistaken for standing or squatting. Furthermore, climbing onto elevated surfaces will not be properly detected while this setting is active. On by default, disable if your character is too stuck to the ground.

![Feet Define Floor Fixes bad squat](images/18-1-feet-floor.png)

- **Sensitivity**: how aggressively BlendCap decides a foot is in contact with the ground. Higher values catch more contacts (good for footage with lots of held poses); lower is more conservative (good when there are lots of genuine quick steps). It's also the fix for floating feet: a contact that drifted off the floor can sit too high for the detector to recognize it as one, which is exactly what keeps it from being corrected, and a higher setting widens what counts so grounding can catch it and pull it back down. 
- **Max Gap**: any contacts closer together in time than this (in seconds) are merged into one continuous contact.
- **Max Lift**: any lifts that go higher than this value (in cm) between contacts will be considered a real footstep or leg lift, and will not be merged.
- **Bridge Phantom Lifts**: bridges the gaps Max Gap can't: with this on, a contact can break for _any_ length of time and still be merged, as long as the foot stayed low and came back down in the same spot, a sign the tracker lost the contact rather than the foot really stepping. The **Recovery Lift** value sets how high the foot can rise while still being treated as one of these misreads.
- **Pin Planted Feet**: once a contact is detected, this is what actually holds the foot still so it doesn't slide. It's on by default; turn it off to let the feet keep more of their original motion through a contact. **Pinning Strength** sets how firmly a planted foot holds its spot, lower values let some of the original movement through. **Release Distance** unpins a contact when the foot drifts farther than the set distance from where it planted; this allows you to free a genuine slide that was mistaken for a plant.

![Foot locking fixes bad feet](images/18-foot-locking.png)
### Suggested starting points for example cases
- **Lots of dynamic movement and held poses:** raise **Sensitivity** to about 0.75 and raise **Recovery Lift** under **Bridge Phantom Lifts** to about 30 cm.
- **Lively footwork with short quick steps:** keep the defaults, adjust **Sensitivity** and **Bridge Phantom Lifts** depending on results.
- **My character's feet are floating when they shouldn't be:** raise **Sensitivity**, and make sure **Feet Define the Floor** is enabled. 
- **Knees appear stiff or jumps aren't detected**: Lower **Sensitivity** to around 0.2, and maybe disable **Feet Define the Floor**.
- **Character stuck to the floor while climbing**: Disable **Feet Define the Floor** and adjust **Sensitivity** until you get your preferred results.

---

## Body Grounding

Foot Locking keeps standing motion planted; **Body Grounding** covers everything that happens on the floor: rolls, lying down, crawls, and kneels. It finds the stretches where the body's weight is on something other than the feet, measures where that support actually touches the ground, and seats the body on it, while trying to keep it level with the floor.

Upright motion and jumps are never touched, and airborne tricks (flips, dives) are protected automatically. It's on by default, with two sliders:

- **Catch Distance**: the farthest the body may be pulled to reach the floor (default 30 cm). A pose hovering further off the ground than this is left where it is.
- **Skin Radius**: the gap kept between the skeleton and the floor (default 1.5 cm), so the final retargeted body rests on its surface instead of its bones.

---
## Motion Scaling

**Finger Curl Strength** amplifies finger curl. By default capture tends to underestimate how far fingers open or close, so raise this if fists/grips look too loose, or hands don't seem to open far enough. Can safely raise to around 3.00 - 4.00 without obvious artifacts depending on the capture. It's disabled while **Rebuild Finger Motion** is active (see [Hand & Finger Tracking](07-hand-tracking.md)). 

**Face Expression** scales the strength of the facial performance: **Strength** scales everything uniformly, or turn on **Use Per-Region** to push or pull each region independently, this lets you exaggerate the eye movement of a character with larger eyes, or soften the mouth movement of characters with a smaller jaw/mouth, or selectively tone down a performance that was exaggerated for tracking. Values above 1 amplify, below 1 soften. Generally looks best within the 0.5 - 1.5 range. **Tongue** is a length dial: capture reports a tongue-out at full strength, so this sets how far your character's tongue actually travels. On the BVH path, **Use Per-Region** ships on with tuned starting values for each region, so you're fine-tuning a sensible baseline rather than dialing in from scratch.

![Face Noise Filtering and Face Expression sliders](images/18-1-face-tuning.png)

> These bake into the exported BVH (and the ARKit versions into the keyed shape keys), so the cleaned-up, scaled result travels with the file into other programs.

> The same **Face Expression** sliders also live under **Retargeting ▸ Advanced**, but they work differently there, scaling your rig's face bones rather than the captured expression. That's blunter, since a bone can carry motion from its neighbors, so use these here to adjust expression strength precisely, and the Retargeting ones for more general adjustments. See [Retargeting](09-retargeting.md).

---

## Recommended order

A good default cleanup pass:

1. **Camera Angle Offset**: get the character standing upright first.
2. **Smoothing**: calm the overall jitter (usually the default values are fine).
3. **Depth Noise Filtering**: settle the depth wobble (again, usually the default values work fine).
4. **Foot Locking/Pinning:** plant the feet last, so it works on already-clean motion (on by default, but you may want to weaken or disable it if legs look too stiff, or strengthen it if feet lose contact with the ground or slide around too much).

You don't have to use all four on every clip, reach for the ones a given capture needs.