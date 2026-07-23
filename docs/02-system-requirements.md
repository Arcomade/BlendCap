# System Requirements

BlendCap runs a fairly heavy pose estimation model locally on your machine, so the hardware matters more than it does for most add-ons. This page covers what BlendCap needs to run well, and what to expect on different hardware.

---

## At a glance

|                      | Requirement                                                                                     |
| -------------------- | ----------------------------------------------------------------------------------------------- |
| **Blender**          | 4.2 or newer                                                                                    |
| **Operating system** | Windows 10 / 11 (64-bit), fully supported. Linux, supported. macOS, not supported (see below).  |
| **Best experience**  | NVIDIA GPU, RTX 20-series / GTX 16-series or newer, 6 GB VRAM or more                           |
| **Disk space**       | ~32 GB free (for the installed models + dependencies)                                           |
| **Internet**         | Required **once**, during installation (~11 GB download). Capture runs fully offline afterward. |

---

## Blender

BlendCap is a Blender **extension** and requires **Blender 4.2 or newer**.

## Operating system

### Windows 10 / 11: fully supported
This is the primary, directly validated platform and the one with the strongest reliability guarantee. If you're on Windows with an NVIDIA GPU, you're on the happy path.

### Linux: supported
Linux is supported, including both standard installs and **Flatpak** Blender. The one-click installer opens a terminal so you can watch the setup and approve the system packages it needs.

### macOS: not supported (paid build)
The paid BlendCap build does **not** support macOS. Apple's notarization requirements and the lack of a supported GPU acceleration path make a reliable one-click Mac product impractical for now. (BlendCap's source is available under its open-source license for advanced Mac users who want to set it up manually, but this is unsupported and not recommended for most users.)

---

## Graphics card (GPU)

BlendCap's body capture runs a large pose estimation model. A capable GPU is what makes capture practical rather than painfully slow.

### NVIDIA: recommended, fully supported
- **Minimum architecture:** RTX 20-series / GTX 16-series, or newer (this is the "Turing" generation). Newer cards including the RTX 30, 40 and 50 series are supported.
- **VRAM:** 6 GB or more recommended. Cards with less will still install but may run out of memory and lead to a failed capture, the installer warns you if it sees less than 6 GB.
- **Older cards:** GTX 10-series and older are **not supported**. The installer detects this and tells you clearly rather than failing later.
- **Typical performance:** roughly a third of a second per frame on a modern NVIDIA card.

### AMD / Intel GPUs: experimental
- **Windows (AMD/Intel):** supported on an **experimental** basis through DirectML. The code path is in place but has not been hardware-validated across the range of cards, so treat it as best-effort.
- **Linux (AMD):** experimental GPU acceleration via ROCm on supported discrete AMD cards (RX 6000 / 7000 class). Without a supported AMD GPU stack, Linux falls back to CPU-only capture.

### CPU-only: works, but slow
A **CPU Only** capture mode is available as a universal fallback. It needs no GPU and should complete a capture on basically any machine, but it is **much slower** than GPU capture, expect to leave longer clips running. Use it for short tests, or when no supported GPU is available.

---

## Disk space

The BlendCap download itself is small (around 2 MB). The first time you install its dependencies, it downloads roughly **11 GB** of models and supporting libraries. Allow **at least 32 GB of free space** on the drive where BlendCap is installed.

## Internet connection

An internet connection is required **once**, during installation, to download the models and dependencies. After that, **capture runs completely offline**: your video footage never leaves your computer, which is good for both privacy and working on the move.