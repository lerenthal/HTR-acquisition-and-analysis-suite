# HTR Studio

Acquisition and analysis of the mouse **head-twitch response**, for the Arducam
ezBOX GS B0634 (OV9281 monochrome global shutter) recording 320×240 at 240 fps on
Windows.

A self-contained folder: copy it anywhere, run `install.bat`, and it works. The
Python environment, FFmpeg, settings, logs and (by default) recordings all live
inside it. Nothing touches the registry or PATH.

```
install.bat            install (needs internet once, ~5 minutes)
run_htr_studio.bat     start the app
run_selftest.bat       verify the installation
tools\camera_diagnostics.bat   collect camera diagnostics when the camera is not found
```

Start with **`docs/USER_MANUAL.md`**. At the rig, print
**`docs/ACQUISITION_QUICKCARD.md`**.

---

## What it does

| Tab | |
|---|---|
| **Acquisition** | Find the camera, probe what it actually delivers, set exposure and gain against live contrast / clipping / focus / uniformity readouts, test the lighting for PWM flicker, and record a trial with its metadata and a one-click dose-time marker. |
| **Library** | Every recording with its animal, group, dose, route, T0, camera settings and analysis state — stored in a JSON sidecar beside the video, so copying the folder copies the study. **Inspect / archive** measures foreign footage and converts long recordings to something storable without destroying the oscillation. |
| **Review** | Watch each candidate twitch at 1/8 speed with the tracking overlay and the signal trace, and accept, reject or add it. The curated set is what gets exported. |
| **Analysis** | The detector and every parameter behind it, with the reason for each default, plus validation against a ground-truth CSV. When a run finds nothing it says which of the possible causes it looks like, and **Signal diagnostics** and **Tune tracking** let you see the answer rather than guess it. |
| **Batch** | A queue of trials to a study summary table and a group time-course figure. Overnight-safe: a failure does not stop the queue. |
| **Settings** | Tool paths, folders, recording defaults, report options, diagnostics. |

---

## Why the defaults are what they are

An HTR is **5–11 oscillations at 78–98 Hz** (Halberstadt et al., *J Neurosci
Methods*, 2014), so a burst lasts 50–150 ms. Three consequences drive everything
else:

* **240 fps is required, not preferred.** At 100 fps the Nyquist limit is 50 Hz
  and the oscillation aliases, unrecoverably. That is why the production mode is
  320×240 / 240 fps rather than the better-looking 640×480 / 100 fps.
* **The pass band is 60–110 Hz**, not something wider. A band reaching down to
  20 Hz passes grooming and locomotion harmonics.
* **The detection threshold comes from the pre-dose baseline**, not from the whole
  recording — otherwise the drug effect raises the bar it has to clear, and the
  bias is worst in the animals with the biggest response.

And one hardware fact that no software setting can compensate for: a PWM-dimmed
LED panel modulates the whole frame inside the HTR band, which produces
whole-field false positives. Use DC constant-current panels, and use the
Acquisition tab's flicker test to confirm you have.

---

## Measured performance

Same files, same ground truth, default settings:

| | First-generation `scripts/htr_analysis.py` | HTR Studio |
|---|---|---|
| `benchmark_videos/synthetic_mouse_htr.mp4` | precision 0.75, recall 1.00, **F1 0.86** | precision 1.00, recall 1.00, **F1 1.00** |
| New benchmark with grooming, fast turns and simulated PWM flicker | — | precision 1.00, recall 1.00, **F1 1.00**, no false positive on any confounder |
| Realistic benchmark: bedding texture, cylinder rim, lighting gradient, moving cast shadow, low-contrast animal | — | tracked 100 %, **F1 1.00** (70.9 % tracked and F1 0.667 with plain thresholding — this is what background segmentation fixed) |
| 8× slow-motion render of a 240 fps capture | — | 0 detections at the file's declared rate; **F1 1.00** once the capture rate is recovered |
| Longest recording analysable | ~50 s (12 000-frame RAM cap) | unlimited — streaming, memory flat in recording length |

Throughput: 300–500 frames/s per thread, so a 70-minute trial is 8–15 minutes on
8 cores.

Disk, measured at 2456 bytes per frame: **1.06 GB** per 30-minute trial, **2.5 GB**
per 70-minute trial, at 320×240 / 240 fps in the camera's own MJPEG. The same
frames captured *uncompressed* would be **66 GB** per 30 minutes — 27× larger for
no gain, since MJPEG is already all-intra. If recordings come out unmanageably
large, that is why; the fix is the pixel format, never the frame rate.

`docs/PIPELINE_REVIEW.md` documents every defect found in the first pipeline and
what was done about it, with measurements.

---

## Documentation

| | |
|---|---|
| `docs/USER_MANUAL.md` | the full manual — every tab, every parameter, the study workflow, the output formats |
| `docs/INSTALLATION.md` | install, move to another machine, offline and proxy routes |
| `docs/ACQUISITION_QUICKCARD.md` | one page for the rig |
| `docs/TROUBLESHOOTING.md` | symptom → cause → fix |
| `docs/PIPELINE_REVIEW.md` | what changed from the first pipeline, and the evidence |
| `CHANGELOG.md` | what changed in each release, with the measurements |

---

## Relationship to the existing pipeline

This folder is additive. The original `HTR_setup/scripts/`,
`MOUSE_HTR_PROTOCOL_SOP.md`, `.venv/`, `benchmark_videos/` and test recordings are
untouched and still work. HTR Studio has its own environment and its own settings
and shares nothing with them, so both can be run side by side — which is also how
the comparison above was measured.

The SOP remains the reference for rig geometry, lighting specification and the
oral-dosing rationale. Two of its figures are corrected here: the per-trial disk
requirement (~2.5 GB, not ~17–25 GB), and the burst-duration window (45–600 ms,
since 5–11 oscillations at 78–98 Hz is 51–141 ms, so its 100 ms floor would reject
the shortest genuine twitches).

---

## Optional: pose estimation

`install_dlc.bat` adds DeepLabCut and PyTorch in a **separate** environment, so a
dependency conflict there cannot stop this app from opening. You probably do not
need it — the shipped detector uses no trained model. It is worth it for keypoints
on other behaviours, or if real footage defeats blob tracking. Only the angle
source changes; the whole detector chain after it is unchanged.

---

## References

1. Halberstadt AL, Geyer MA. *Characterization of the head-twitch response
   induced by hallucinogens in mice.* J Neurosci Methods, 2014. PMC3866102 —
   5–11 oscillations at 78–98 Hz.
2. Maitland AD et al. *Rapid, Open-Source, and Automated Quantification of the
   Head Twitch Response in C57BL/6J Mice Using DeepLabCut and Simple Behavioral
   Analysis.* ACS Pharmacol Transl Sci, 2025. doi:10.1021/acsptsci.5c00305.
3. Arducam ezBOX GS B0634 datasheet.
4. `MOUSE_HTR_PROTOCOL_SOP.md` v1.2 in the parent folder — rig geometry,
   lighting, and the oral-dose time-course rationale.

Version 1.0.0
