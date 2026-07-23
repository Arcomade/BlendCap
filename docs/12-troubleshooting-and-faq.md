# Troubleshooting & FAQ

If something isn't working, start here. If you're still stuck, see [Getting help](#getting-help) at the bottom.

---

## Frequently asked questions

### Do I need an internet connection to use BlendCap?
Only **once**, during installation, to download the models (~11 GB). After that, **capture runs completely offline**: your footage never leaves your computer. (One exception: if you later switch the Capture Backend to **CPU Only** or **Non-NVIDIA**, the first capture on that backend fetches a one-time extra component when online.)

### What kind of video do I need?
Ordinary video, a phone clip, a webcam recording, or existing footage. For best results: one person, fully in frame, decent even lighting, fairly steady camera. See [Filming Tips](13-filming-tips.md).

### Can I capture more than one person at once?
BlendCap focuses on a **single subject** per capture and tracks the main person in frame (it warns you if it sees several). For multiple characters, capture each person's clip separately. Support for true multi-person-capture is currently planned as a feature for a future release, but is still under development.

### How long does a capture take?
It depends on your GPU and the clip length. On a modern NVIDIA card it's roughly a third of a second per frame for the body; face capture is much lighter. The **CPU Only** backend should work on almost any machine but is much slower. Use **Capture Skip** to speed up captures in exchange for lower accuracy.

### Does BlendCap work on macOS?
No, the paid build does not currently support macOS. See [System Requirements](02-system-requirements.md).

### Does it work without a GPU?
There's a **CPU Only** capture backend that runs almost anywhere, but it's much slower than GPU capture.

### What rigs does it support?
Rigify, Auto-Rig Pro, CloudRig, Mixamo skeletons and Mixamo control rigs via the bundled presets, and most other/custom rigs via the bone-map editor. See [Retargeting](09-retargeting.md).

### Can I use my captures outside Blender?
Yes. The BVH Library's **Save BVH** and **Save FBX** buttons export any capture for other software. See [BVH Library & Cache](10-bvh-library-and-cache.md).

### Can I use the animations I create commercially?
Yes, the animation you capture is yours to use, including in commercial projects. See [Licensing & Credits](14-licensing-and-credits.md).

### Where are my captures stored?
Finished takes go to your BVH Library; working data goes to a per-project cache. Both locations are configurable. See [BVH Library & Cache](10-bvh-library-and-cache.md).

---

## Installation problems

### The install download was interrupted
Just run **Install Dependencies** (or **Reinstall Dependencies**) again, it **resumes** where it left off rather than starting the ~11 GB download over.

### Setup stopped with a "SHA256 mismatch" error
A downloaded model file didn't match its expected fingerprint, so setup stopped rather than continue with a bad file. The file has already been removed, so just run the setup again and it will re-download it. If it happens repeatedly, something on your network (often antivirus or a proxy) is interfering with the download.

### "Nothing happens" when I start the install (Linux)
On Linux the installer opens a **terminal window** so you can watch progress and approve system packages. If you don't see one, check for a terminal behind other windows, and make sure your system has a terminal application available.

### Flatpak Blender: install is blocked
Flatpak runs Blender in a sandbox, so BlendCap needs a one-time permission to run its installer on the host. The install guide shows the exact command with a **Copy Command** button: run it in a terminal once, restart Blender, and try again. See [Installation](03-installation.md).

### Windows: a required tool is missing
The installer fetches the few system tools it needs automatically. If your system can't do that, the installer tells you which component to install and where to get it. Install it and run the setup again.

### Blender says the AI dependencies aren't installed
Clicking a capture button before the one-time setup pops a dialog with an **Open Preferences** button, follow it and run **Install Dependencies** ([Installation](03-installation.md)).

---

## Capture problems

### The Preview shows lots of "no person" frames
The subject isn't being detected reliably. Improve framing (get the whole person in shot), lighting, and reduce occlusion (furniture, props, other people crossing). Re-run Preview until coverage is solid before doing a full capture. It may also be a false alarm. If you believe your footage is clean enough, try running a capture and see how it turns out.

### My GPU isn't recognized / capture runs on the wrong backend
Check the detected-hardware line in **Preferences ▸ BlendCap ▸ AI Tracking Dependencies** (also shown under the Capture Backend dropdown). If you changed or upgraded your GPU after installing, run **Reinstall Dependencies** so BlendCap re-detects it. Confirm your card meets the [System Requirements](02-system-requirements.md) (NVIDIA RTX 20 / GTX 16 series or newer).

### "Out of memory" during capture
This usually means limited VRAM (under 6 GB). Close other GPU-heavy apps, try a shorter or lower-resolution clip, or switch the **Capture Backend** to **CPU Only**, much slower, but it can't run out of video memory.

### A capture failed with an error
BlendCap shows the last lines of the capture's output when it fails. Note that message, it usually points at the cause (out of memory, a problem reading the video, and so on), and include it if you contact support.

### A capture seems frozen
The first-run model load can be silent for a couple of minutes; that's normal, and BlendCap never kills a capture that may still be working. If nothing has happened for over ten minutes, BlendCap adds a "Cancel if stuck" hint to the status, at that point, **Cancel** (it stops cleanly) and try again. A short clip is a good way to confirm the pipeline works.

### I changed a setting but got the same result
Capture settings re-capture automatically, and post-processing settings need a **Generate/Reload BVH** (or **Bake Settings to BVH**) to apply. If you still suspect a stale result, use **Clear Cache** on that clip and capture again (see [BVH Library & Cache](10-bvh-library-and-cache.md)).

---

## Retargeting & cleanup problems

### The retarget looks wrong / limbs are twisted
Most often the **bone map doesn't match the rig**. Load the preset that matches your rig type, check the "pairs valid" count above the table (red rows mean bones weren't found, often a **prefix** issue, see [Retargeting](09-retargeting.md)), and if your character's rest pose isn't a standard T/A-pose, set a rest pose (Custom Rest Pose, or Use Current Source Pose as Rest).

>For Rigify rigs, you may want to make sure "head follow" and "neck follow" are both set to "1.000" before retargeting so they have standard FK behavior and will avoid twisting artifacts.

### The IK looks off
IK conversion reads from the FK result, so it must run **after** the FK retarget. Let BlendCap auto-convert (**Auto-bake IK on Apply**), or make sure you clicked **Apply Retargeting** first. Also check the **FK → IK Mapping** rows point at the right IK and pole controls.

### The character floats, sinks, or the feet slide
Use **Foot Locking** in BVH Post Processing: for sliding, and for feet that float during holds and deep squats, raise **Sensitivity**; for brief pops, turn on **Bridge Phantom Lifts**. For a whole performance floating above the floor, or bobbing and drifting vertically (a moving camera does this), try **Feet Define the Floor** in Foot Locking Settings; if the level is still off, nudge **Height** in Camera Angle Offset, then **Bake Settings to BVH** to make it permanent. If the entire body is sliding around, you may need to turn off root motion and animation movement along the ground plane manually. See [BVH Generation and Post-Processing](08-bvh-generation-and-post-processing.md).

### The whole performance is tilted
An occasional quirk of single-camera capture. Use **Camera Angle Offset** to correct the body (and, if needed, the head separately). See [BVH Generation and Post-Processing](08-bvh-generation-and-post-processing.md).

### The face looks flat or barely moves
Check that the expressions were clearly performed and the face was well lit and large in frame. Then raise the **Face Expression** strength (in Motion Scaling, the ARKit section, or Retargeting ▸ Advanced) to push the performance on your character. See [Face Capture](06-face-capture.md).

---

## Getting help

Reach out through the store you bought BlendCap from: use the contact or message option on the BlendCap product page (for example on Superhive or Gumroad).

When reporting a problem, it helps a lot to include: your OS, GPU, Blender version, what you were doing, and any error message BlendCap showed.
