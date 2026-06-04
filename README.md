<!--
  ┌─────────────────────────────────────────────────────────────────────┐
  │  MEDIA TO ADD BEFORE PUBLISHING (drop into an `assets/` folder):     │
  │    assets/banner.png     – wide logo/title banner (optional)         │
  │    assets/hero.gif       – the money shot: video in → animation out  │
  │    assets/demo-body.gif  – full-body + hands capture                 │
  │    assets/demo-face.gif  – face capture / blendshapes               │
  │    assets/demo-retarget.gif – retargeted onto a rig                  │
  │    assets/demo-cleanup.gif  – foot lock / denoise before-after       │
  │  Until these exist the images show as broken icons — add them or     │
  │  comment the <img> tags out before the first push.                   │
  └─────────────────────────────────────────────────────────────────────┘
-->

<div align="center">

<!-- Optional banner. Delete this line if you don't have one. -->
<img src="assets/banner.png" alt="BlendCap" width="640">

# BlendCap

**Markerless full-body performance capture for Blender — from a single video.**

Drop in ordinary footage and get clean body, hand, and face animation retargeted
onto your own rig. Runs locally on your GPU. No suit. No markers. No cloud.

[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](#-license)
[![Blender](https://img.shields.io/badge/Blender-4.2%2B-orange.svg?logo=blender&logoColor=white)](https://www.blender.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](#-requirements)
[![Status](https://img.shields.io/badge/status-v1.0-brightgreen.svg)](#-project-status)

<br>

<!-- The hero demo. Make this your best single clip: raw video on the left,
     finished Blender animation on the right. -->
<img src="assets/hero.gif" alt="BlendCap: video in, animation out" width="85%">

</div>

---

## What is BlendCap?

**BlendCap turns a normal video of a person moving into a fully animated Blender
character.** Point it at a clip, and it tracks the full body (127 joints,
including hands), captures the face, and retargets everything onto your own
Rigify or Auto-Rig Pro rig — all inside Blender, all on your own machine.

It's built on Meta's [SAM 3D Body](https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/)
for body and hand tracking and [MediaPipe](https://github.com/google-ai-edge/mediapipe)
for facial performance, wrapped in a Blender add-on that handles the messy parts:
foot grounding, denoising, rig retargeting, and export. After the one-time model
download, **BlendCap runs completely offline — your footage never leaves your
computer.**

---

## ✨ Showcase

<div align="center">
<table>
  <tr>
    <td width="50%" align="center">
      <img src="assets/demo-body.gif" width="100%"><br>
      <sub><b>Full body + hands</b> — 127 joints from one camera</sub>
    </td>
    <td width="50%" align="center">
      <img src="assets/demo-face.gif" width="100%"><br>
      <sub><b>Facial capture</b> — ARKit-compatible blendshapes</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" align="center">
      <img src="assets/demo-retarget.gif" width="100%"><br>
      <sub><b>One-click retarget</b> — straight onto your rig</sub>
    </td>
    <td width="50%" align="center">
      <img src="assets/demo-cleanup.gif" width="100%"><br>
      <sub><b>Auto cleanup</b> — foot locking + depth denoise</sub>
    </td>
  </tr>
</table>
</div>

---

## Features

**🎥 Capture**
- Full-body **and hand** tracking (127 joints) from a single ordinary video — no suit, no markers, no capture studio.
- **Facial performance capture** to ARKit-compatible blendshapes via MediaPipe.
- **100% local & offline.** GPU-accelerated; nothing is uploaded and there are no per-frame fees.
- Fast **detector-only preview** to confirm your subject is tracked before committing to a full capture.

**🦴 Retarget**
- **One-click retargeting** onto your own **Rigify** or **Auto-Rig Pro** rig (FK *and* IK), with tuned presets that ship in the box.
- **Mixamo-standard** preset for control-rig overlays.
- Custom **rest-pose presets** and per-rig **camera-angle offsets** for footage shot from below/above.

**🧹 Clean up**
- 2-pass **foot grounding** + footskate cleanup.
- **Depth-axis denoise** — kills per-frame wobble while letting real strides pass through.
- Tracking-failure repair, rotation smoothing, and per-region motion scaling.

**🧰 Workflow**
- Everything is driven from a single panel in Blender's 3D Viewport — capture, convert, retarget, clean up, export.
- **BVH export** plus a reusable BVH library so captures are easy to re-apply.

---

## 📦 Get BlendCap

BlendCap comes in two flavors. **Same capture engine — the only difference is how
much setup you do yourself.**

| | 🆓 **Free** (this repo) | ⭐ **Paid** (one-click installer) |
|---|---|---|
| **What it is** | The complete, open-source add-on. | The same add-on, packaged to *just work*. |
| **Setup** | You bring Python, a HuggingFace account, and run a setup script. | One button inside Blender. No terminal, no Python, model access bundled. |
| **GPU setup** | Manual (edit a line for non-NVIDIA). | Auto-detected and configured. |
| **Best for** | Developers, tinkerers, and Linux users. | Artists who just want it to run. |
| **Get it** | [Quick start ↓](#-quick-start-free--diy) | **[Gumroad / Blender Market »](LINK_TO_YOUR_STORE)** |

> The paid version funds development. The full source lives here for free under the
> GPL — buying the installer is paying for the convenience, not the code.

---

## 🖥️ Requirements

- **Blender 4.2+**
- A **CUDA-capable NVIDIA GPU** (≥ 6 GB VRAM recommended) — this is the validated, supported path.
  - **AMD / Intel on Windows** run through DirectML / ONNX Runtime — *experimental*.
  - **Apple Silicon** runs through MPS from the source path — *experimental*.
  - CPU technically works but is impractically slow (hours per clip) and isn't recommended.
- **Python 3.10+** *(free / DIY path only — the paid installer brings its own)*.
- A free **HuggingFace account** *(free path)* to accept Meta's SAM License and download the model weights.
- **~8 GB** of free disk for the models and Python environment.

| Platform | Status |
|---|---|
| Windows + NVIDIA | ✅ Fully supported (validated) |
| Linux + NVIDIA | ✅ Supported |
| Windows + AMD / Intel (DirectML) | 🧪 Experimental |
| macOS Apple Silicon (MPS) | 🧪 Experimental, source path only |
| AMD on Linux (ROCm) | ❌ Not currently supported |

---

## 🚀 Quick start (free / DIY)

```bash
git clone https://github.com/Arcomade/BlendCap.git
cd BlendCap

# Windows
setup_env.bat

# Linux / macOS
./setup_env.sh
```

The setup script builds a local Python environment and downloads the models. When
prompted, log in with `huggingface-cli login` — you'll first need to accept Meta's
SAM License at
[huggingface.co/facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3).

Then load the add-on in Blender:

1. Open Blender → **Scripting** workspace
2. **Text Editor → Open →** pick `blendcap_ui.py` from the cloned folder
3. Click **Run Script**
4. The **BlendCap** panel appears in the 3D Viewport's **N-panel**

> 📖 **Full install notes, GPU compatibility, and troubleshooting:**
> [`setup/github_version/README.md`](setup/github_version/README.md)

---

## 🕹️ Using BlendCap

1. **Point** BlendCap at a video (a file or a Sequencer strip).
2. **Capture** the body and/or face. Run the quick preview first to confirm the subject is tracked across the clip.
3. BlendCap **converts** the capture to BVH and imports it for you.
4. **Pick your rig and a preset**, then bake the retarget (FK, then IK).
5. **Clean up** — foot locking, depth denoise, smoothing — and **export**.

---

## 🧠 Built on

BlendCap stands on excellent open research and tools:

- **[SAM 3D Body](https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/)** (Meta) — full-body + hand mesh recovery
- **[Fast-SAM-3D-Body](https://github.com/yangtiming/Fast-SAM-3D-Body)** — the faster, lighter inference backend
- **[DINOv3](https://github.com/facebookresearch/dinov3)** (Meta) — vision backbone
- **[MoGe](https://github.com/microsoft/MoGe)** (Microsoft) — geometry / focal-length estimation
- **[MediaPipe](https://github.com/google-ai-edge/mediapipe)** (Google) — face landmarks + blendshapes
- **[YOLO11](https://github.com/ultralytics/ultralytics)** (Ultralytics) — person detection
- **[Blender](https://www.blender.org/)** + **Rigify**

---

## 📜 License

BlendCap is **GPL-3.0-or-later**, © 2026 Arcomade.

The bundled model weights and third-party components ship under their own terms —
including Meta's **SAM License** and **DINOv3 License**, and the **AGPL-3.0** license
of YOLO11. Full license texts and notices are in
[`installer/licenses/`](installer/licenses/).

The SAM 3D Body weights are permitted for both research and commercial use under the
SAM License, which prohibits military, weapons, and espionage applications.

---

## ⭐ Project status

BlendCap is at **v1.0** and in active use. Windows + NVIDIA is the directly
validated, fully supported path; other platforms ship as experimental. Issues and
feedback are welcome.

<div align="center">
<sub>Made for animators who'd rather skip the suit.</sub>
</div>
