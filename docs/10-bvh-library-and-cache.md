# BVH Library & Cache

BlendCap keeps your captures organized so you can re-use them, export them, and free up space when you're done. Two pieces do that work: the **BVH Library** (your finished, reusable takes) and the **cache** (the raw working data behind each capture).

---

## The BVH Library

Every armature you generate is saved as a **BVH file** in the library, and the **BVH Library** section at the bottom of the panel lists them. Select one and:

- **Import** brings it back into the scene, without re-running the capture. Handy for retargeting the same performance onto a different character, or coming back to a clip later.
- **Save BVH** saves a copy of the BVH file wherever you choose, for use in any program that reads BVH.
- **Save FBX** converts it to an FBX file, for game engines and other software that prefer FBX.
- **Delete** removes it from the library (it asks you to confirm first).

![The BVH Library section](images/21-bvh-library.png)

### Save options

**Save BVH** and **Save FBX** both have a few options in the side panel of the save window:

- **Add Rest Pose Frame**: adds one extra frame in front of the motion with the armature in a rest pose, picked from the **Pose** list (BlendCap's rest pose, T-POSE, A-POSE, or one of your own rest pose presets made on a BlendCap armature). Some programs set up their retargeting from the first frame of a file, and this gives them a clean pose to start from. In an FBX the motion keeps its frame numbers and the pose sits on the frame before it. A BVH is simply one frame longer.
- **Scale**: Blender and BlendCap work in meters, so set this to 100 for programs that expect centimeters.
- **Forward** and **Up**: which way the armature faces and which axis points up, to suit the program you're exporting to.
- **Add Leaf Bones** (FBX only): adds an end bone to every chain. FBX doesn't store how long a bone is, so without them the last bone of each finger, the toes and the face bones have no length of their own.

![The Save FBX options](images/41-save-fbx-options.png)

The file always holds the whole performance, whatever your scene's frame range is set to. In an FBX, the armature and its animation are named after the take with `_e1` on the end.

By default the library lives in a `bvh_exports` folder inside the project's cache, and you can point it somewhere else, including one shared folder that several projects read from, with **Use a custom BVH library folder** in [Preferences](11-preferences.md).

---

## The cache

When you capture a clip, BlendCap stores the raw intermediate results, the tracking data and the preview overlay, in a **cache folder** for that clip. This is what makes re-captures faster, and allows you to generate the BVH multiple times with different post-processing settings while the slow capture step only has to run once.

BlendCap also remembers **which settings** produced each cached result. If you change anything capture-affecting (the frame range, capture backend, hand refinement, FOV mode, capture skip), it notices and will clear the cache the next time you run the capture, so you never get a stale result that doesn't match your settings. If you suspect a stale result anyway, you can always clear the cache manually (see below).

### Where it's stored

- For a **saved** project, the cache is a `Blendcap_Cache` folder next to your `.blend` file, so it travels with the project.
- For an **unsaved** project, it goes to a temporary folder that is **cleared when Blender closes**. Save the project and BlendCap automatically moves the cache in next to the new `.blend` file, so capturing first and saving after is safe, just save before you quit. 
- A multi-person capture gets one extra cache folder for each person you generate, next to the capture's own, and a multi-camera take keeps each camera's capture in its own subfolder. **Clear Cache** removes them along with the capture.

>You can change where your cache is stored by default in [Preferences](11-preferences.md).

> **Windows and long paths.** Blender on Windows can't use file paths longer than about 260 characters. If you get a "Path too long for Blender on Windows" message, save the project and video closer to the top of the drive, use shorter clip names, or point the cache at a short folder with **Set default cache folder** and **Always use this folder** in [Preferences](11-preferences.md).

---

## Clearing space

The **Capture** section gives you two cleanup buttons to help work around unexpected bugs:

- **Clear Cache** removes the current clip's cached data, so its next capture runs fresh. Use it if you ever suspect a stale result. On a multi-camera strip it also brings your original clips back.
- **Clear All** removes every clip's cache in the project, reverts any capture strips in the Sequencer back to the original videos, and removes the preview overlays from the viewport, allowing you to start completely fresh.

![The Capture controls](images/08-quickstart-capture.png)

> Both are safe by design: they only ever delete inside the `Blendcap_Cache` folder BlendCap owns, and **your BVH Library is not affected**, finished takes stay. If you're worried about hitting **Clear All** by accident, you can hide the button in [Preferences](11-preferences.md).