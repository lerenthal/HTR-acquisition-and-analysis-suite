# Changelog

## 1.3.0-dev

### The preview now shows what will be recorded

The old preview was OpenCV, and on Windows the DirectShow backend negotiates on
OpenCV's behalf: ask the B0634 for MJPG and the driver returns YUY2, and
`CAP_PROP_FPS` echoes the request rather than reporting what was agreed.
FFmpeg, asked for `-vcodec mjpeg -framerate 200`, gets exactly that. Two
different media types at two different frame rates have two different exposure
windows — and since brightness follows integration time, the preview could be
several times brighter than the file. An operator setting lighting against it
was setting it against the wrong picture.

* **`camera.CapturePreview`** opens the camera through FFmpeg with
  `recorder.dshow_input_args(spec)` — the same argument list the recording
  uses, not a copy of it, so the two cannot drift. It is the default; the
  OpenCV preview remains as "Fast" for framing and focus.
* A reader thread owns the pipe and holds only the newest frame. A pipe read
  blocks until a whole frame arrives, and doing that on the GUI thread cost a
  measured 89 ms per tick. Steady-state ticks are now **under 1 ms median**,
  with one ~30 ms tick when the first frame arrives (Qt and OpenCV lazy
  initialisation at the frame's size).
* FFmpeg cannot set UVC controls, so a control change closes the device, sets
  it, and reopens — about a second. A slider drag emits one change per pixel of
  travel, so a debounce timer collapses a drag into **one** reopen.
* **Camera properties…** opens the driver's own DirectShow property page while
  FFmpeg holds the device in the capture mode. On this camera that is the only
  way to set exposure and gain in the mode the trial will use.
* The **flicker test** now runs through the capture pipeline at the capture
  frame rate. Whether a PWM ripple folds into the 55–125 Hz band depends on the
  sampling rate, so a verdict taken at the preview's rate said nothing about a
  200 fps recording.
* The lighting readouts are recomputed 4×/s rather than every frame: measured at
  3.8 ms (QVGA) and 14.5 ms (VGA) per call, nearly all of it the Laplacian
  variance behind the focus score.

### The analysis settings advisor

`htr_studio/advisor.py`. The shipped defaults can only be right for one kind of
file; point the tool at a 120 fps published clip and several are not merely
suboptimal but wrong.

* **`suggest_settings`** reads the container and calibrates on a frame sample,
  then lists what to change *with the arithmetic for each*. Nothing is applied
  until Apply is pressed, and the reasons go into the result sidecar under
  `settings_provenance`.
* **The band-pass band is the setting everything hangs off.** Above Nyquist the
  78–98 Hz twitch folds rather than vanishing: at 120 fps it lands at 22–42 Hz,
  so the default 60–110 Hz band filters empty spectrum. Measured on the paper
  footage: `example_vid3_LSD` went from **0 detections to 10**, and
  `example_vid1_DOI` from **0 to 6**, same frames, same threshold rule.
* Folding is computed over the whole band, not its endpoints, because it
  reflects at every multiple of `fps/2`: 100–140 Hz at 120 fps folds to 0–20 Hz
  while both endpoints land on 20. Endpoint arithmetic would have proposed a
  band containing almost none of the signal.
* A folded band reaching below 15 Hz is flagged as **bad**, not merely
  suggested: there it overlaps locomotion and grooming, and no threshold will
  separate them.
* The 240 fps benchmark is unaffected — the advisor proposes only cosmetic
  changes there and **F1 stays 1.000** with and without them.

### Calibrating a session from an untreated control

An untreated animal should produce no head twitches, so every burst found in a
vehicle recording is a false positive by construction. Sweeping the threshold
over it measures the rig's own noise floor.

* **`calibrate_from_control`** sweeps `threshold_k`, plots the false-positive
  curve, and picks **one step below** the lowest setting at which the control
  falls silent. Sitting exactly at the silent point means tuning until the
  detector can just barely not see the loudest non-twitch present, which is a
  hair from not seeing a real twitch either.
* It reports the margin between the chosen threshold and the tallest control
  peak, and **refuses to invent a number when there isn't one**. Run on
  `example_vid2_noI`, it reported 4 calls remaining at k = 12 and said so:
  *"That is not a threshold problem: something in this footage is producing
  movement in the detection band."*
* Results save as named **session profiles** carrying the control file, the
  sweep and the margin, so a count can be traced to how its threshold was
  fitted.

### Four defects found by the operator

* **Every integer slider was dead on drag.** `raw / self._scale` is a float even
  when the scale is 1 (`240 / 1 == 240.0`), and `QSpinBox.setValue` refuses a
  float; the `TypeError` aborted the handler before the value was pushed to the
  camera. Exposure, gain, brightness, contrast and gamma could only be set by
  typing.
* **Review kept the previous file's burst regions.** `AppContext.entryChanged`
  existed and nothing was connected to it, so each open path told the tabs by
  hand and "Open in Analysis" forgot Review. `set_entry` is now the only writer
  of `ctx.entry` and both tabs subscribe.
* **"Finished: 13,260 frames, 0 B"** — the segment muxer reports no size on
  FFmpeg's stats line, and the folder-size fallback sat inside the
  still-running branch, so it ran during the recording but not for the summary.
* **`ffmpeg exited with code 4294967291`** was `-video_size 320x245` from the
  mouse wheel passing over an unfocused spin box. Wheel events are now ignored
  on unfocused spin and combo boxes (61 widgets), the exit code is translated
  into plain language, and pre-flight rejects a capture size that is not a
  multiple of 8.

### A settings default that had drifted, and did not reach the rig

`config.DEFAULTS["analysis"]` was a hand-written copy of
`analysis.AnalysisConfig`'s field defaults, and it had drifted: it still carried
`max_area_frac: 0.45` after the dataclass was lowered to **0.25** — 0.45 being
the ceiling that let the lit arena floor (33 % of frame, perfectly stable) be
accepted as the animal. Eight other fields were simply absent. Since a stored
value wins over a default, a rig that already had a `settings.json` kept 0.45
and the arena-floor fix was never actually active there; "Reset to defaults"
restored the broken value.

* `DEFAULTS["analysis"]` is now **derived from the dataclass**, so there is one
  place a default lives.
* A **schema migration** (version 3) corrects a stored 0.45 ceiling and a stored
  fixed 9 px tail kernel, matching on the exact old default so a deliberately
  chosen value is kept.
* **Reset to defaults** now resets from the dataclass, lists what will change,
  and asks first.

### Tests

`tools/regress_1_2_2.py`, `tools/test_true_preview.py`, `tools/test_advisor.py`
and `tools/test_advisor_ui.py` — 100 checks. The preview and dialog tests drive
the real widgets with stand-in cameras and footage, which is the generalisation
of the `%LOCALAPPDATA%` lesson: a path that only runs on Windows with hardware
attached is untested by construction.

## 1.1.0

Three problems reported from real use, with what was measured in each case.

### Analysis failed on real footage

A global brightness threshold cannot separate a dark mouse from dark bedding or
from the bench outside the arena, so on real footage the mask grabbed all three
and the body axis pointed somewhere between them. Reproduced first on a new
harder benchmark (`tools/make_realistic_benchmark.py`: bedding speckle, cylinder
rim and highlight, the dark bench outside the arena, a lighting gradient, a
moving cast shadow, a low-contrast animal), where plain thresholding scored
**70.9 % tracked, F1 = 0.667**.

* **Background segmentation.** The per-pixel temporal median of sampled frames is
  the empty arena; subtracting it cancels everything static. **100 % tracked,
  F1 = 1.000**, with the clean and legacy fixtures unchanged at F1 = 1.000.
* `auto` keeps thresholding while its hit rate is >= 90 % and its blob-area
  coefficient of variation <= 0.35, and falls back only when it is not —
  background subtraction scored marginally *worse* on clean fixtures, so
  switching eagerly would have traded good footage for bad.
* **Auto-ROI and edge-blob rejection are off by default**, both on measurement:
  auto-ROI took F1 from 1.000 to 0.500 (the ROI edge clips the animal against the
  wall), and edge rejection took tracking from 100 % to 67.8 % and cost the
  legacy fixture all six of its events. The clip test now examines the
  morphologically opened core, so a tail at the frame edge no longer condemns a
  frame.
* **Tune tracking** dialog: frame-by-frame segmentation running the same
  `segment_frame` the extraction pass uses, with a 200-frame scan reporting hit
  rate and area variation — the two numbers to optimise.

### "Clear twitches on the video, nothing detected"

A 240 fps capture written at 30 fps for playback puts a 90 Hz twitch at 11.25 Hz
in the file's own timebase: plainly visible to the eye, completely invisible to a
78–98 Hz detector working from file timestamps. Measured on a reconstructed 8x
render: 0 detections as the file declares itself, **F1 = 1.000** at the true rate
— same frames, same settings.

* **Capture-rate override** on the Analysis tab, cleared when a different
  recording is selected, recorded in the exported report.
* **Signal diagnostics** dialog: power spectrum against the HTR band and the
  file's Nyquist limit, per-source (including a background-only control for
  lighting flicker), with the implied capture rate offered for one click.
* **What to do next** after a run that finds nothing: names the most likely cause
  — tracking, then timebase, then band, then detector gates.
* The rate suggestion is gated on the band being *empty*, not on the tallest peak
  in the spectrum: any trial with locomotion peaks at 5–10 Hz, and gating on that
  told a correctly timed 240 fps clip to change its frame rate. The band's own
  peak is reported separately.

### Acquisition and long captures

* **Frame loss was under-reported.** FFmpeg's `drop=` counts only muxer drops;
  frames lost in the capture buffer appear on a line containing no word the error
  scan matched, so they were discarded silently. Both are now counted, summed in
  the live tile, recorded separately in the sidecar, and restated on completion.
* **`-thread_queue_size` was unset**, so FFmpeg's default of 8 packets applied —
  33 ms at 240 fps, shorter than one disk flush, and the standard cause of
  "Thread message queue blocking" plus the frame loss that follows. Now 4096, and
  exposed in Settings.
* **Write-path pre-flight check.** The failure is a stall, not a throughput
  shortfall, so it writes 6 MB in fsynced 256 kB pieces and reports the worst
  single pause. Over 250 ms will overflow the capture buffer.
* **Segmenting pre-flight check** for recordings over ten minutes.
* **Inspect / archive** (Library): what a file actually is — codec, measured
  versus declared frame rate, all-intra or not, bytes per frame — plus arena
  detection, conversion options measured on the operator's own footage, and a
  capture planner.
  * Every option is all-intra and none changes the frame rate.
  * Sizes are measured, not tabulated, because the conventional advice is wrong
    here: lossless FFV1 came out **5.3x larger** than the MJPEG source, since it
    must reproduce the camera's JPEG artefacts and sensor noise exactly. Measured
    on the same file: MJPEG re-quantised 98 %, all-intra H.264 CRF 15 89 %, and
    **cropping to the arena 19 %** — the largest saving and the only free one.
  * Conversions are verified by frame count in and out; a mismatch is a failure.
* **Disk arithmetic, documented**: 320x240 MJPEG at 240 fps is 1.06 GB per
  30 minutes. The same frames uncompressed are 66 GB — 27x for no gain. Large
  files come from not using the camera's MJPEG encoder, never from the frame
  rate, which must not be traded for size.

### Also

* Self-test gained a *failure diagnostics and archiving* section: spectrum,
  slow-motion recovery, all three segmentation modes, inspection, and an archive
  round trip that re-analyses the converted copy and checks it scores the same.
* `ffprobe` now reports pixel format.

## 1.0.0

First release. See `docs/PIPELINE_REVIEW.md`.
