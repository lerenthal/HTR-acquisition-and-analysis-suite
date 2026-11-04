# Changelog

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
