# BlendCap Documentation

This folder holds the source for the BlendCap user documentation, written so it can be published on **Superhive (Blender Market)** and on the **product website** (arcomade.com/blendcap).

The pages are plain **Markdown** so you can edit them in any text editor, drop in screenshots and video links as you go, and convert them to HTML for the website or paste them into a marketplace description.

---

## How to read / publish these docs

- **Website (multi-page):** publish each numbered file as its own page, in order. The filenames double as the page order.
- **Marketplace (single page):** concatenate the numbered files into one long page, or copy just the sections you want (Introduction, System Requirements, Installation, Quick Start are the most important for a listing).
- **PDF (to package with the add-on):** Superhive recommends including a written PDF in the product download. These Markdown files convert cleanly to a single PDF once the screenshots are in (e.g. via Pandoc).

### These pages are public, but write them for owners
Superhive shows your documentation on the product page, so a prospective buyer might read it before purchasing. Write the pages for someone who already owns BlendCap, and let the Gumroad and Superhive product descriptions do the selling. The one thing that serves both audiences is accuracy: keep the pages correct and clear (especially [System Requirements](02-system-requirements.md) and [Installation](03-installation.md)), since that is what a prospect skims and an owner relies on.

## Table of contents

| # | Page | What it covers |
|---|------|----------------|
| - | [README](README.md) | This file, conventions and index |
| 01 | [Introduction](01-introduction.md) | What BlendCap is, guide index, getting help |
| 02 | [System Requirements](02-system-requirements.md) | OS, GPU, VRAM, disk, internet |
| 03 | [Installation](03-installation.md) | Installing the add-on + the one-click dependency installer |
| 04 | [Quick Start](04-quick-start.md) | Your first capture, start to finish |
| 05 | [Capturing](05-capturing.md) | Video sources, capture settings, Preview, the cache buttons |
| 06 | [Face Capture](06-face-capture.md) | Facial performance, ARKit shape keys, face bones, expression tuning |
| 07 | [Hand & Finger Tracking](07-hand-tracking.md) | How fingers are tracked, and the controls that refine them |
| 08 | [BVH Generation and Post-Processing](08-bvh-generation-and-post-processing.md) | Generate the armature, then clean it up: camera angle, smoothing, depth, foot locking |
| 09 | [Retargeting](09-retargeting.md) | Transferring motion onto your rig, presets, bone maps, FK → IK |
| 10 | [BVH Library & Cache](10-bvh-library-and-cache.md) | Managing captures, exporting BVH/FBX, the cache |
| 11 | [Preferences](11-preferences.md) | Every add-on preference, explained |
| 12 | [Troubleshooting & FAQ](12-troubleshooting-and-faq.md) | Common questions and fixes |
| 13 | [Filming Tips](13-filming-tips.md) | How to shoot video that captures well |
| 14 | [Licensing & Credits](14-licensing-and-credits.md) | Licenses and attributions |

---

## Conventions

- **Bold** marks exact UI labels (buttons, options, section names), these have been verified against the add-on's code, so keep them matching the build if the UI changes.
- Image tags point at [`images/`](images/). Images `01`–`14` exist already (docs 01–05); the rest are the to-shoot list, see [`images/README.md`](images/README.md) for what each one should show. Numbering continues the existing scheme: a new number per topic, `-1`/`-2` suffixes for follow-on shots of the same topic.
- A few `<!-- NOTE ... -->` HTML comments remain where something must happen before publishing (they're invisible when rendered).

---

## Before you publish: a short checklist

- [ ] Shoot the remaining screenshots in [`images/README.md`](images/README.md) and confirm each renders in place.
- [ ] Add the tutorial video link in [01-introduction](01-introduction.md) (currently points at the channel).
- [ ] Publish the GitHub repo **before** the docs/product go live, [14-licensing-and-credits](14-licensing-and-credits.md) and NOTICE.txt cite it as the source location.
- [ ] Re-read [System Requirements](02-system-requirements.md) and [Installation](03-installation.md) against the final build, these are the two pages people get tripped up by most when they're wrong.
- [ ] Export a PDF (docs + images) to package with the add-on download, as Superhive recommends.
