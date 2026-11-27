# Filming Tips

BlendCap can only capture what the camera saw, so a few minutes of care when you record pays off more than any setting. This page is a practical checklist for shooting footage that captures cleanly, based on how BlendCap's tracking models actually behave.

![Good vs. poor capture framing](images/23-filming-framing.png)

---

## The essentials

- **One person in frame.** In **Single-Camera Capture**, BlendCap tracks a single subject per capture: it locks onto the main person and follows them between frames, and other people in the shot can steal that lock (it warns you when it keeps seeing more than one). Keep others out of frame, or at least small, far away, and never overlapping your subject. To capture multiple people at once, use [Multi-Person Capture](05b-multi-person-capture.md) (see "Shooting multiple people" below).
- **Keep the whole body in frame** for body capture, head to feet, with a little margin. The body model handles a partly hidden body well, but foot grounding needs to *see* the feet, if they leave the frame, planting and floor contact have nothing to work with. For footage that's deliberately framed tighter, see "Upper-body and cropped shots" below.
- **Bright, even light.** Good light helps every stage, the person detector, the body model, and especially face tracking. Avoid strong backlight.
- **Hold the camera steady.** A tripod is ideal. BlendCap captures motion relative to the camera, so shake reads as jitter and a moving shot mixes the camera's travel into the character's motion. Camera tilt may cause subjects to lean forwards or backwards, but can be corrected with **Camera Angle Offset** later. Footage from a moving camera is still usable, but you may have to generate the armature without **Root Motion** and animate the character's travel manually (see [BVH Generation and Post-Processing](08-bvh-generation-and-post-processing.md)).
- **A clean background helps detection.** The body model is built for messy real-world footage, but the person detector still has to find your subject, and mannequins, posters and other people confuse it more than ordinary clutter does.

---

## Occlusion: brief is fine, long is not

The body model is robust to partial occlusion, a hand crossing the torso, a prop in front of a leg, and BlendCap rides through brief full losses by re-using the subject's last known position. What can't be captured is a subject hidden or out of frame for long stretches, those frames simply have nothing in them. When in doubt, run a **Preview** to check detection coverage before committing to a long capture, see [Capturing](05-capturing.md).

---

## Framing & camera position

- **Give the performance room.** Let the subject have some free space in the frame rather than filling every pixel. Keep a bit of headroom and footroom so the body stays entirely in frame through the whole take for an ideal capture.
- **Avoid fisheye and heavy lens distortion.** BlendCap estimates the camera's field of view assuming a normal, undistorted lens; strongly distorted footage skews the depth estimate. A standard phone lens is fine, the ultra-wide one less so.
- **Keep the shot static, and if you must zoom, tell BlendCap.** Pans and handheld drift bake into the motion (see above), and a zoom changes the field of view mid-clip, set **FOV Mode** to sample every frame if your footage has zooms (see [Capturing](05-capturing.md)).

---

## The subject

- **Normal clothing is fine, extremes aren't.** The body model is trained on a huge variety of real-world clothing, but very loose or flowing garments (robes, long dresses) hide the pose underneath and can confuse the prediction. Also avoid outfits that blend into the background.
- **Expect depth to be the noisy axis.** Toward-and-away-from-camera is the hardest thing for any single camera to judge, that's what **Depth Noise Filtering** exists for. Motion *across* the frame will always read more cleanly than motion straight at the lens.

---

## For face capture

Face tracking is pickier than body tracking, it needs detail the camera has to actually deliver:

- **Get the face large and clear in frame.** A small, distant face degrades tracking quality, a close or medium shot is far better. Fine mouth and teeth detail needs the most size, in wide full-body shots BlendCap automatically falls back to simpler face tracking. For the most detail, shoot the face as its own closer clip and combine it with the body (see [Face Capture](06-face-capture.md)).
- **Stay within a three-quarter view.** Face tracking loses the face beyond roughly 80° away from the camera, a profile shot won't track.
- **Keep the face unobstructed.** Tracking fails outright when less than half the face is visible, and hands, hair and props crossing the face degrade it well before that.
- **Light the face and keep it steady.** Dim, noisy or blurry footage shows up as jitter in the tracked face. Face Smoothing and Noise Filtering can clean it up, but cleaner footage needs less of them.
- **Keep other faces away from your subject's.** The face tracker expects one face in its search area, someone overlapping or right beside your subject can confuse it.
- **Perform expressions clearly.** Motion the camera can barely see may read as flat (cheeks especially). Performing slightly big and pulling it back later with **Face Expression** scaling beats re-shooting.

---

## Upper-body and cropped shots

Waist-up and torso-only footage works better when you turn on **Hold Cropped Limbs** in BVH Settings, which holds the parts of the body that were never filmed still instead of guessing at them (see [BVH Generation and Post-Processing](08-bvh-generation-and-post-processing.md)).

- **Keep the hips in frame if you can.** That gives the best result. Without them the body stays in place and only turns with the shoulders.
- **Expect the feet to slide** when the body turns or travels, since no footwork was filmed.
- **A close-up of just the hands won't capture.** The tracker has to find a person first, so keep the shoulders and some of the torso in the picture.

---

## Shooting with multiple cameras

For [Multi-Camera Capture](05a-multi-camera-capture.md):

- **Film the same take on every camera**, start to finish, and keep the cameras still.
- **Spread the cameras out.** Around 90° apart is the usual advice. Two cameras facing each other from opposite sides of the performer see nearly the same thing, and don't add much.
- **Start your takes with a T-pose when you can**, standing still, in view of every camera. It isn't required, but it makes the results a little more consistent.
- **Clap once at the start**, or make sure there's some distinct sound in the take, so **Sync by Sound** can line the clips up for you.
- **Use the same frame rate on every camera**, and note down each camera's lens and focal length.
- **How many cameras?** In general, 2 cameras are the sweet-spot, but if you have a lot going on and want to make sure you get the most accurate result possible, a third or fourth camera helps. In my tests the second camera is the big step up. A third is a smaller but still decent improvement, mostly in the wrists and ankles. A fourth adds very little on clean footage, but it's good insurance against a camera that ends up blocked or badly placed.

**Recommended camera setups**, seen from above. The orange figure is the performer in the T-pose at the start of the take, and the arrow shows which way they face.

**2 cameras:** about 90° apart, with the performer where the two views cross. For the T-pose, face diagonally between the cameras. Facing straight at one camera points the arms at the other, which then can't see them properly.

![Two-camera setup seen from above](images/43-camera-setup-2cam.png)

**3 cameras:** spaced evenly around the performer, about 120° apart. T-pose in the middle, facing between two of the cameras.

![Three-camera setup seen from above](images/44-camera-setup-3cam.png)

**4 cameras:** spaced evenly around the performer, each about 90° from the next. T-pose in the middle, facing diagonally between two of the cameras.

![Four-camera setup seen from above](images/45-camera-setup-4cam.png)

In every setup, keep each camera far enough back to hold the whole body in frame, arms out included.

---

## Shooting multiple people

For [Multi-Person Capture](05b-multi-person-capture.md) and [Multi-Camera & Multi-Person](05c-multi-camera-and-multi-person.md):

- **Keep everybody fully in frame.** People crossing in front of each other, or stepping out of the shot and back in, are where the tracking loses track of who is who.
- **Faces need to be large and turned toward the camera** to be picked up, which is harder with a group in a wide shot. Separate face captures are highly recommended for these types of captures. 
- **With several cameras, it helps if everybody starts in a T-pose**, standing apart with room for their arms, and visible to every camera.
- **Place the extra camera(s) where it can see everybody for the whole take.** The more a camera can see of the performers without any occlusions, the more it will add to the overall result.
- **Two cameras can work for two people, but it's worth using at least three for any captures with more than one person**, as long as that third camera keeps everyone in sight.
- **Close contact (grappling, hugging) is the hardest case.** More cameras help.

---

## Technical settings

- **Color space: standard sRGB/Rec.709.** BlendCap reads your footage as ordinary video, so log picture profiles (S-Log, V-Log, C-Log and the like) and HDR footage reach the tracker flat and washed out, which noticeably hurts tracking quality. Shoot in a standard profile, or convert/grade the clip to Rec.709 before capturing, it makes a bigger difference than you'd expect.
- **Frame rate:** match (or convert to) your scene's frame rate. BlendCap offers the conversion for you, see [Capturing](05-capturing.md).
- **Resolution:** a clear, well-exposed 1080p clip is plenty. Higher resolution costs processing time and memory during capture without rescuing poor light or framing.
- **Shutter:** nothing special required, a standard 180° shutter captures fine. A faster shutter can sharpen fast motion a touch, but isn't typically necessary.

---

## Quick pre-shoot checklist

- [ ] One person (unless you're using a multi-person mode), whole body in frame, feet included, with margin
- [ ] Even, bright lighting; no strong backlight
- [ ] Camera steady (tripod)
- [ ] Background reasonably clean; clothing distinct from it, not overly loose
- [ ] For face: face large in frame, unobstructed, within a three-quarter view
- [ ] Standard color profile (sRGB/Rec.709), no log or HDR
- [ ] Frame rate matches your project
- [ ] Several cameras: same frame rate on every camera, focal lengths noted, and ideally a T-pose and a clap at the start

Get these right and most of the cleanup work is already done before you capture.