# Troubleshooting & FAQ

If something isn't working, start here. If you're still stuck, see [Getting help](#getting-help) at the bottom.

---

## Frequently asked questions

### Do I need an internet connection to use BlendCap?
Only **once**, during installation, to download the models (~11 GB). After that, **capture runs completely offline**: your footage never leaves your computer. (One exception: if you later switch the Capture Backend to **CPU Only** or **Non-NVIDIA**, the first capture on that backend fetches a one-time extra component when online.) BlendCap also checks for a newer version once a day while Blender's online access is on, which you can switch off in [Preferences](11-preferences.md).

### What kind of video do I need?
Ordinary video, a phone clip, a webcam recording, or existing footage. For best results: one person (or multiple people, with [Multi-Person Capture](05b-multi-person-capture.md)), fully in frame, decent even lighting, fairly steady camera. See [Filming Tips](13-filming-tips.md).

### Can I capture more than one person at once?
Yes. **Multi-Person Capture** follows everybody in one video and gives you a separate armature for each person, and **Multi-Camera & Multi-Person** does the same from several camera angles. The regular **Single-Camera Capture** mode still tracks one subject, the main person in frame. See [Multi-Person Capture](05b-multi-person-capture.md).

### Can I film with more than one camera?
Yes. **Multi-Camera Capture** combines up to 4 camera angles of the same take into one more accurate result. See [Multi-Camera Capture](05a-multi-camera-capture.md).

### How long does a capture take?
It depends on your GPU and the clip length. On a modern NVIDIA card it's roughly a third of a second per frame for the body; face capture is much lighter. The **CPU Only** backend should work on almost any machine but is much slower. Use **Capture Skip** to speed up captures in exchange for lower accuracy. More people in the shot and more camera angles both add to the time (see [Capturing](05-capturing.md)).

### Does BlendCap work on macOS?
No, the paid build does not currently support macOS. See [System Requirements](02-system-requirements.md).

### Does it work without a GPU?
There's a **CPU Only** capture backend that runs almost anywhere, but it's much slower than GPU capture.

### What rigs does it support?
Rigify, Auto-Rig Pro, CloudRig, Mixamo skeletons and Mixamo control rigs via the bundled presets, and most other/custom rigs via the bone-map editor. See [Retargeting](09-retargeting.md).

### Can I use my captures outside Blender?
Yes. The BVH Library's **Save BVH** and **Save FBX** buttons export any capture for other software, with options for scale, axis orientation and an added rest pose frame. See [BVH Library & Cache](10-bvh-library-and-cache.md).

### Can I use the animations I create commercially?
Yes, the animation you capture is yours to use, including in commercial projects. See [Licensing & Credits](14-licensing-and-credits.md).

### Where are my captures stored?
Finished takes go to your BVH Library; working data goes to a per-project cache. Both locations are configurable. See [BVH Library & Cache](10-bvh-library-and-cache.md).

---

## Installation problems

### The install download was interrupted
Just run **Install Dependencies** (or **Reinstall Dependencies**) again, it **resumes** where it left off rather than starting the ~11 GB download over. BlendCap shows a red "install didn't finish" note in its preferences and sidebar until the install has completed.

### Setup stopped with a "SHA256 mismatch" error
A downloaded model file didn't match its expected fingerprint, so setup stopped rather than continue with a bad file. The file has already been removed, so just run the setup again and it will re-download it. If it happens repeatedly, something on your network (often antivirus or a proxy) is interfering with the download.

### "Nothing happens" when I start the install (Linux)
On Linux the installer opens a **terminal window** so you can watch progress and approve system packages. If you don't see one, check for a terminal behind other windows, and make sure your system has a terminal application available.

### Flatpak Blender: install is blocked
Flatpak runs Blender in a sandbox, so BlendCap needs a one-time permission to run its installer on the host. The install guide shows the exact command with a **Copy Command** button: run it in a terminal once, restart Blender, and try again. See [Installation](03-installation.md).

### "An Application Control policy has blocked this file", or capture suddenly stops working on Windows 11
Smart App Control is on and has blocked one of the capture engine's files. It is not antivirus: it checks each file against Microsoft's cloud service and can block the open-source components (Python, PyTorch and others) that BlendCap's capture engine is built from, since Microsoft hasn't signed them. Which files it blocks can change from day to day, and it can also switch itself on after a Windows update, which is why BlendCap can stop working with no change on your side.

Fix: Go to Windows Security > App & browser control > Smart App Control settings > Off. Your antivirus will keep running as usual. Then run the capture, installer, or update again.

Note: on Windows builds older than 24H2 with the April 2026 update, Smart App Control can't be turned back on without resetting Windows. On newer builds it can.

A capture or install that fails with exit code 3236495362 is the same block: Windows refused to start one of BlendCap's helper programs.

### I removed BlendCap and my presets or dependencies are gone
Removing an extension in Blender also deletes its user folder, which is where BlendCap keeps both by default. Reinstall BlendCap and run **Install Dependencies** again. Custom presets can't be brought back unless you exported them, so click **Export Presets** in [Preferences](11-preferences.md) before removing the add-on next time.

---

## Update problems

### An update fails, or Update Now is greyed out
BlendCap should say why, in the message or in a note under the button. The usual causes: the add-on's folder is open in another program (close it and update again), the new version needs the full package from your store page (install it with **Install from Disk**), or an earlier install was interrupted and left Blender's extensions locked (click **Unlock Extensions**).

### I never see update notices
The daily check needs Blender's online access (**Preferences ▸ System ▸ Network**), and **Automatically check for updates** has to be ticked in BlendCap's preferences. You can always check by hand with **Check now**.

---

## Capture problems

### The Preview shows lots of "no person" frames
The subject isn't being detected reliably. Improve framing (get the whole person in shot), lighting, and reduce occlusion (furniture, props, other people crossing). Re-run Preview until coverage is solid before doing a full capture. It may also be a false alarm. If you believe your footage is clean enough, try running a capture and see how it turns out.

### "Path too long for Blender on Windows"
Blender on Windows can't use file paths longer than about 260 characters, even with Windows' own long path setting switched on. Save the project and video closer to the top of the drive, use shorter clip names, or point the cache at a short folder with **Set default cache folder** and **Always use this folder** in [Preferences](11-preferences.md).

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

### Generate/Reload BVH runs but no armature appears
BlendCap uses Blender's own BVH importer to bring the animation into the scene. It ships with Blender and is on by default, but it can be switched off. Enable **BioVision Motion Capture (BVH) format** under **Edit ▸ Preferences ▸ Add-ons** (search for "BVH"), then generate again; BlendCap checks before it generates and offers to switch it back on for you. The .bvh is written to the BVH library either way, so nothing is lost if you need to import it another way (see [BVH Library & Cache](10-bvh-library-and-cache.md)).

### The face bones look huge in an FBX I exported and imported back into Blender
That's how Blender draws bones that have no end point in the file. The motion is fine. Save with **Add Leaf Bones** enabled and tick **Ignore Leaf Bones** when importing. See [BVH Library & Cache](10-bvh-library-and-cache.md).

---

## Multi-camera problems

### The rest of my timeline disappeared when I clicked Sync Clips
It's only hidden while you're in syncing mode. Run the capture, or click **Exit Syncing Mode**, and it comes back.

### A multi-camera take came out tilted
Multi-camera takes are levelled from the floor the performer moves across, or from how they stand. A short take where they never stand still on their feet has neither, so straighten it with **Camera Angle Offset** (see [BVH Generation and Post-Processing](08-bvh-generation-and-post-processing.md)). A T-pose at the start of the take gives BlendCap a still moment to level from, which helps avoid this.

---

## Multi-person problems

### The People list got someone wrong
One performer picked up twice, two people trading numbers, somebody coming back with a new number, or a bystander showing up as a person are all fixed with **Join**, **Swap** and **Remove** under the People list, without capturing again. See "Fixing mistakes" in [Multi-Person Capture](05b-multi-person-capture.md). If only one camera of a multi-camera take got it wrong, set the dropdown above the buttons to that camera first.

### The Face checkbox or Run Capture is greyed out
In the multi-person modes, face tracking needs **Body** to be enabled.

### All the armatures are standing on top of each other
Tick **Keep Captured Position** in BVH Settings before generating, so each person lands where they stood in the footage.

---

## Retargeting & cleanup problems

### The retarget looks wrong / limbs are twisted
Most often the **bone map doesn't match the rig**. Load the preset that matches your rig type, check the "pairs valid" count above the table (red rows mean bones weren't found, often a **prefix** issue, see [Retargeting](09-retargeting.md)), and if your character's rest pose isn't a standard T/A-pose, set a rest pose (Custom Rest Pose, or Use Current Source Pose as Rest).

If the limbs bend oddly or stop short on a rig with built-in limits, the rig's own rotation limits or drivers are likely overriding the animation. Use **Mute and Retarget Again** in the **Rig Conflict** window, or tick **Mute Limits and Drivers** under **Retargeting ▸ Advanced**.

>For Rigify rigs, you may want to make sure "head follow" and "neck follow" are both set to "1.000" before retargeting so they have standard FK behavior and will avoid twisting artifacts.

### The IK looks off
IK conversion reads from the FK result, so it must run **after** the FK retarget. Let BlendCap auto-convert (**Auto-bake IK on Apply**), or make sure you clicked **Apply Retargeting** first. Also check the **FK → IK Mapping** rows point at the right IK and pole controls.

### A rest pose preset does nothing, or is refused
A rest pose preset only works on the kind of skeleton it was saved from. The bundled **T-POSE** and **A-POSE** are for BlendCap's own armature, so for any other source rig, save your own preset. See [Retargeting](09-retargeting.md).

### The character floats, sinks, or the feet slide
For poses on the floor (lying, rolling, crawling, kneeling), **Body Grounding** in BVH Post Processing does the seating and is on by default, so first check it's still enabled and that **Catch Distance** is large enough to reach your pose. For standing motion, use **Foot Locking**: for sliding, and for feet that float during holds and deep squats, raise **Sensitivity**; for brief pops, check that **Bridge Phantom Lifts** is still enabled. For a whole performance floating above the floor, or bobbing and drifting vertically (a moving camera does this), make sure **Feet Define the Floor** is enabled in Foot Locking Settings; if the level is still off, you can nudge **Height** in Camera Angle Offset, then **Bake Settings to BVH** to make it permanent. If the entire body is sliding around, you may need to turn off root motion and animation movement along the ground plane manually. See [BVH Generation and Post-Processing](08-bvh-generation-and-post-processing.md). You may also want to make sure your capture had the FOV set to a constant value, or manually enter the camera's focal length. See [Capturing](05-capturing.md).

### The whole performance is tilted
An occasional quirk of single-camera capture (a short multi-camera take can come in tilted too, see "Multi-camera problems" above). Use **Camera Angle Offset** to correct the body (and, if needed, the head separately). See [BVH Generation and Post-Processing](08-bvh-generation-and-post-processing.md).

### The face looks flat or barely moves
Check that the expressions were clearly performed and the face was well lit and large in frame. Then raise the **Face Expression** strength (in Motion Scaling, the ARKit section, or Retargeting ▸ Advanced) to push the performance on your character. See [Face Capture](06-face-capture.md).

---

## Getting help

Reach out through the store you bought BlendCap from: use the contact or message option on the BlendCap product page (for example on Superhive or Gumroad).

When reporting a problem, it helps a lot to include: your OS, GPU, Blender version, what you were doing, and any error message BlendCap showed.
