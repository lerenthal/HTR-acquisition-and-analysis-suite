# Review of the first-generation HTR pipeline

What was already in `HTR_setup/`, what it got right, what it got wrong, and what
HTR Studio does differently. Nothing in the original was changed — `scripts/`,
`MOUSE_HTR_PROTOCOL_SOP.md`, `.venv/` and the test videos are all untouched.

Files reviewed: `scripts/htr_analysis.py` (504 lines), `scripts/test_arducam_acquisition.py`,
`scripts/record_trial.sh`, `scripts/probe_fps.py`, `scripts/generate_synthetic_video.py`,
`scripts/headless_smoke.py`, `scripts/htr_capture.bonsai`, `scripts/htr_analysis_env.txt`,
`MOUSE_HTR_PROTOCOL_SOP.md` (v1.2, 649 lines), `prompt.md`.

---

## 1. What the first pass got right

These were real findings and they are carried forward unchanged:

**FFmpeg with `-c:v copy` is the only viable recorder.** The measurement behind
that — 7209 frames in 30.01 s, 240.3 fps, zero drops — is the single most
valuable result in the original work. Handing the camera's own MJPEG packets
straight to disk avoids a decode and re-encode per frame, which is what capped
the OpenCV `VideoWriter` path near 100 fps. HTR Studio records the same way.

**320×240 is the right production mode, not 640×480.** The B0634's MJPEG encoder
runs on a fixed clock, so VGA tops out near 100 fps whatever you request, and
100 fps gives a Nyquist limit of 50 Hz — below the phenomenon being measured.
320×240 delivers 240 fps, Nyquist 120 Hz, which actually covers the 78–98 Hz
band. Choosing temporal resolution over spatial resolution was correct.

**ffprobe `nb_read_frames / duration` is ground truth for frame rate.** OpenCV
reports the wrong FPS for MJPEG-in-AVI. Spotting that saved every timing figure
downstream.

**A synthetic benchmark with planted events.** Building a video with known
twitch times, rather than waiting for mice, was the right instinct and is what
made it possible to verify the rewrite quantitatively.

**The 650 nm IR-cut filter finding.** The B0634 cannot see 850 nm IR, so the
original brief's IR panels would have produced a black frame. Catching that
before anything was ordered was worth the whole exercise.

---

## 2. Defects found, and what was done about each

### 2.1 The analysis could not process a real trial (blocking)

`htr_analysis.py` reads every frame into a Python list of grayscale arrays:

```python
self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
if len(self.frames) >= max_frames:   # max_frames = 12000
    break
```

At 320×240 a grayscale frame is 76.8 kB, so the 12 000-frame cap is **50 seconds
at 240 fps**. The protocol in the same repository specifies a 70-minute trial:
1 008 000 frames, about **77 GB** of RAM if the cap were lifted. The analysis as
written could not have run on the recordings the protocol tells you to make, and
would have silently truncated them at 50 s if the cap were raised naively.

**Now:** extraction is streaming. Each frame is decoded, reduced to a handful of
scalars, and discarded. Only the derived 1-D signals are kept — about 40 MB for a
full 70-minute trial, independent of recording length. The frame cap still exists
but defaults to "no limit" and is there for convenience, not for memory.

### 2.2 One twitch counted two or three times

The original thresholded a moving-average envelope and treated each excursion
above it as a separate event:

```python
above = envelope > vel_thresh_deg_s
# contiguous runs -> bursts
```

An HTR burst is amplitude-modulated: 5–11 oscillations that grow and fade. Every
time the envelope dips back through the threshold mid-burst, the burst is split.
The SOP records the consequence honestly — "6 planted bursts, 6 detected, 2
over-splits", precision 0.75, **F1 = 0.86**.

**Now:** two mechanisms, both measured.

* *Hysteresis.* A burst starts above the threshold and continues down to 50% of
  it, so a dip does not end the event — the same reason an oscilloscope trigger
  has a release level.
* *Merge gap.* Sub-threshold gaps shorter than 60 ms are closed before anything
  is counted. Two HTRs that close together are not physiologically plausible;
  each burst is itself 50–150 ms long.

Measured effect on the original benchmark video, with everything else held
constant: precision 0.75 → **1.00**, F1 0.86 → **1.00**.

### 2.3 The drug effect inflated its own detection threshold

```python
baseline = np.percentile(envelope, 50)
mad = np.percentile(envelope, 75) - baseline
vel_thresh_deg_s = baseline + 3 * sigma
```

Robust statistics, but computed over the **whole recording**. In a drug trial the
drug effect is most of the recording, so the twitches raise the median and the
spread they are measured against. The stronger the effect, the higher the bar it
has to clear — the effect partly hides itself, and the bias is worse in the
animals with the biggest response. That is the wrong direction for a dose–response
study.

**Now:** the threshold is derived from the **pre-dose baseline window**, which the
protocol already provides (10 minutes before gavage). T0 is recorded at the
moment of dosing, so the window needs no guessing. With no T0 marked, the first
10% of the recording is used and the report says so. "Whole recording" and a
manual value remain available and are recorded in the output either way.

### 2.4 The pass band admitted things that are not twitches

The original band-passed 20–100 Hz. The reference it cites (Halberstadt et al.,
*J Neurosci Methods* 2014) reports **78–98 Hz**. A band starting at 20 Hz passes
grooming harmonics, locomotion, and the body-rotation component of a fast turn.

**Now:** 60–110 Hz by default, which leaves margin around the published range
without reaching down into locomotion. Two further gates are applied per
candidate, because being in-band is not the same as oscillating:

* *dominant frequency*, measured from zero crossings inside the candidate, must
  fall in 55–125 Hz;
* *at least 6 half-cycles*, against the 10–22 that 5–11 oscillations imply.

A broadband lurch crosses the threshold but fails both, and the rejection reason
is recorded so the call can be audited.

### 2.5 Lost tracking was filled with the previous value

```python
if not contours:
    return prev_angle   # hold last value if mouse briefly out of frame
```

Holding a value produces a flat segment, and leaving it produces a step. A step
is broadband, so it rings the band-pass — lost tracking manufactured exactly the
signal the detector was looking for.

**Now:** a frame with no usable blob is marked invalid and its angle is NaN.
Short gaps (under 60 ms) are interpolated; longer ones are interpolated too,
because the filter needs a finite array, but stay flagged, and the joins are
smoothed so they do not leave a kink. Any candidate burst with more than 25% of
its frames invalid is rejected. The report states what fraction of the recording
tracked successfully, which is the first number to look at when a trial's counts
look odd.

### 2.6 The head-angle estimator was the largest single source of false positives

Two problems, compounding.

**180° ambiguity.** `cv2.fitEllipse` returns an angle in [0, 180), so the signal
flips by 180° whenever the apparent long axis swaps ends. The original tried to
patch this with a `while` loop that nudged by ±180 when the step exceeded 90°,
which cannot distinguish a real fast rotation from a flip.

**Pixel quantisation.** Fitting the ellipse to the whole blob — body plus tail
plus head — means the measured axis is dominated by the body, and the head
rotation that defines an HTR is a small perturbation on it.

A head-direction estimator was built to replace it, and the first version made
things *worse*: taking the farthest contour **pixel** from the centroid gives a
quantised angle, and the chosen pixel hops between neighbours from frame to
frame. On a 320×240 frame that hop is several degrees at the frame rate — a
square wave sitting right inside the 60–110 Hz band. Measured on the original
benchmark: 4 spurious bursts, precision 0.60.

**Now**, in three steps:

1. A morphological opening removes the tail (a few pixels wide, against tens for
   the body), leaving body and head.
2. The projection axis comes from the blob's **principal axis** — smooth and
   sub-pixel — with the extremal pixel used only to decide *which end* is the
   head (by thickness on the first frame, by continuity after that; the head end
   is the narrower one).
3. The reported position is the **centroid of the leading 60%** of the blob along
   that axis, which moves continuously rather than in pixel steps.

The result is a 0–360° head direction with no flips. Measured on the original
benchmark: band-passed noise floor in a quiet stretch fell from 169 to 49 deg/s
RMS against a burst amplitude of 342 — about a sevenfold improvement in
signal-to-noise — and the 4 spurious bursts disappeared. The legacy ellipse and
moment estimators are still selectable for comparison.

### 2.7 Segmentation was brittle

A hard-coded grayscale threshold of 100, inverted, with a 200-pixel minimum
contour area and no upper bound and no polarity option. That works on the
synthetic video it was tuned against. On real footage it depends on the arena
floor, the lamp distance, and whether anyone adjusted the exposure.

**Now:** the threshold is Otsu's, computed from frames spread across the file and
then **frozen** for the whole recording (a per-frame Otsu would make the tracked
area jitter as the animal moves). Polarity is auto-detected by trying both and
keeping whichever locates the animal more often. Blob area is gated as a fraction
of the frame at both ends. All three are overridable, and the chosen values appear
in the report.

**And then this turned out not to be enough**, which is worth recording because
it was found the way such things usually are — by a user pointing a working tool
at real footage. Published experiment footage failed: the head direction and body
axis were wrong and the visible twitches were not detected.

Reproducing it first, before changing anything, mattered. A harder benchmark was
built (`tools/make_realistic_benchmark.py`): bedding speckle with dark chips, a
cylinder highlight and dark rim, the dark bench *outside* the arena, a lighting
gradient with vignetting, a moving cast shadow, and a soft low-contrast animal
(tone 104 against a 188 floor). Plain thresholding scored **70.9 % tracked,
F1 = 0.667** on it. The failure was the one the arithmetic predicts: a global
brightness threshold cannot distinguish a dark mouse from dark bedding or from
the bench, so the mask grabbed all three and the blob's principal axis pointed
somewhere between them.

The fix is **background subtraction**. The per-pixel temporal median of sampled
frames is the empty arena — the animal is somewhere different in each sample, so
at any given pixel it is the minority and the median returns the floor.
Subtracting that cancels everything static: bedding, rim, gradient, bench.
Result: **100 % tracked, F1 = 1.000**, with the clean and legacy fixtures
unchanged at F1 = 1.000.

Three decisions around it are worth stating, because each was the opposite of
the obvious one and each came from a measurement:

* **`auto` keeps thresholding while it is working.** Background subtraction is
  not uniformly better — on the clean fixtures it scored marginally worse. So
  `auto` keeps threshold while its hit rate is ≥ 90 % *and* its blob-area
  coefficient of variation ≤ 0.35, and falls back only when it is not. Switching
  eagerly would have traded accuracy on good footage for accuracy on bad.
* **Auto-ROI is off.** Deriving the arena from the background model and masking
  outside it sounds strictly beneficial. Measured, it took F1 from **1.000 to
  0.500** and raised area variation from 0.09 to 0.22: the ROI edge clips the
  animal when it presses against the wall, and a clipped outline puts the head
  direction somewhere it never was. Nine false positives. The option remains, off.
* **Edge-blob rejection is off.** Same reasoning, opposite direction: rejecting
  frames where the animal touches the edge is correct in principle and ruinous in
  practice on a tight arena — tracking fell from 100 % to 67.8 % on one fixture
  and the legacy fixture lost all six of its events. The clip test was also
  narrowed to the morphologically *opened* core, so a trailing tail at the frame
  edge no longer condemns a frame.

Both defaults are off on evidence, and both are documented as such in the manual,
so the next person does not "fix" them back.

### 2.8 Frame-rate handling was heuristic

```python
if self.fps > 1000 or self.fps <= 0 or abs(self.fps - round(self.fps / 100) * 100) < 1:
    # ...re-probe with ffprobe
```

The third clause treats any frame rate near a multiple of 100 as suspect — which
includes a genuine 100 fps *and* a genuine 240 fps capture reported correctly.
Harmless in effect, but it meant the real frame rate was decided by a guess about
whether the container could be trusted.

**Now:** the exact frame rate is measured once, by `ffprobe -count_frames`, and
**cached in the recording's sidecar**, so it is paid for once and then read. The
Library shows whether a given file's rate is verified or still a header value,
and the Analysis tab refuses to be quiet about it: an unverified rate gets a
visible warning, because every timing figure and the entire band definition
depend on it.

### 2.8a A file's declared frame rate is not always its capture rate

Measuring the frame rate exactly (2.8) answers "how many frames per second does
this file *say* it has". It does not answer "how many frames per second did the
camera take", and for published footage those differ routinely: a 240 fps capture
written at 30 fps for playback is a slow-motion video in which every frame is
present and only the timebase is wrong.

The consequence is specific and brutal. A 90 Hz twitch appears at
90 × 30/240 = **11.25 Hz** in the file's own timebase. It is plainly visible to
the eye — that is the point of a slow-motion render — and completely invisible to
a detector band-passing 78–98 Hz on file timestamps. The detector reports zero
and is, in its own terms, correct.

Measured on a reconstructed 8× render of a 240 fps capture:

| Analysed at | Tracking | Detected | F1 |
|---|---|---|---|
| the container's declared 30 fps | 100 % | 0 | 0.000 |
| the true 240 fps | 100 % | 4 of 4 | 1.000 |

Same frames, same settings; only the timebase changed. This is the most likely
single explanation for "clear twitches on the video, nothing detected", and it
costs nothing to fix once recognised.

**Now:** three things. A **capture-rate override** on the Analysis tab, cleared
whenever a different recording is selected and recorded in the exported report so
a run's timebase is auditable. A **spectrum view** that plots the power spectrum
against the HTR band and the file's Nyquist limit, so "is the signal reachable"
becomes something the operator can see. And a detector that recognises the
pattern — an oscillation at an integer fraction of 88 Hz in a file whose declared
rate is low — and offers the implied capture rate for one-click application.

One subtlety found while testing it: gating that suggestion on the *tallest peak*
in the spectrum is wrong. Any trial with locomotion has its tallest single peak at
5–10 Hz, because locomotion runs continuously while a twitch is 100 ms in seventy
minutes; the self-test caught this immediately, telling a correctly-timed 240 fps
clip to change its frame rate. The suggestion is gated on whether the band is
*empty* instead, and the band's own peak is reported separately from the global
one.

### 2.9 Absolute paths pinned the tool to one machine

```python
str(Path("E:/obsidian_vault_14_9_26/HTR_setup/benchmark_videos"))
default = Path("E:/obsidian_vault_14_9_26/HTR_setup/benchmark_videos/htr_report.csv")
```

These appear in the file dialogs. On the analysis computer the folder does not
exist, so the dialogs open somewhere arbitrary. The Bonsai workflow and
`record_trial.sh` have the same problem with hard-coded output paths.

**Now:** every path is resolved relative to the installation folder, and the
recordings root is a setting. `config/settings.json` lives inside the install, so
copying the folder carries the configuration with it and nothing is written to
the registry or the user profile.

### 2.10 Per-minute counts were not aligned to the dose

```python
def per_minute_counts(times, bursts, duration_s):
    for b in bursts:
        minute = int(b["start_s"] // 60)
```

Minutes are counted from the start of the *recording*, which includes the
pre-dose baseline. With 10 minutes of baseline, the first post-dose minute is
labelled 10. Across animals whose baselines ran slightly long or short, the time
courses do not line up — and a time course is the study endpoint. (`times` and
`duration_s` are accepted and never used, which is a hint that the binning was
not fully thought through.)

**Now:** bins carry both an absolute `start_s` and a `bin_rel_t0`, where 0 is the
first bin after dosing and the baseline bins are negative. The exported study
summary adds latency to first twitch, peak rate, the peak's minute relative to
T0, and separate baseline and post-dose counts.

### 2.11 The burst-duration window contradicted its own citation

The SOP specifies "burst duration (100–500 ms)" and, two sections earlier, cites
5–11 oscillations at 78–98 Hz. Those are inconsistent: 5 cycles at 98 Hz is
**51 ms** and 11 cycles at 78 Hz is **141 ms**. A 100 ms floor rejects the
shortest genuine twitches; the code's own default, 80 ms, still cuts into the
range.

**Now:** 45–600 ms. The floor follows the arithmetic, with margin for the
envelope smoothing widening a measured burst slightly. The ceiling is kept
generous because the cost of a loose ceiling is low — grooming bouts last 1–2 s
and are rejected on frequency anyway — while a tight one throws away data.
Measured burst durations now come out at 108–154 ms for planted bursts of
65–132 ms, which is the expected small widening.

### 2.12 Nothing recorded what a trial *was*

`record_trial.sh` writes `trial.avi`. There is no animal ID, no group, no dose,
no route, no dose time, no record of the exposure or gain the camera was set to.
Six weeks later, "what exposure was that trial shot at" has no answer, and
neither does "which of these twelve files is animal M007".

It is also a **bash** script on a Windows host, so it needs Git Bash or WSL; it
hard-codes 100 fps at 640×480, which is the aliasing mode the SOP warns against;
and it writes one unsegmented file.

**Now:** every recording gets a JSON sidecar holding animal, group, treatment,
dose, route, operator, notes, T0, the full camera configuration, the lighting
metrics measured at record time, the exact FFmpeg command line, and what FFmpeg
observed (frames, effective rate, drops, any errors). The Library reads and edits
it. Recording is native Windows with no shell dependency, and segmented by
default.

### 2.12a Frame loss during capture was under-reported, and the buffers were at FFmpeg's defaults

The recorder displayed FFmpeg's `drop=` counter and treated any non-zero value as
the signal to act on. `drop=` counts frames the **muxer** discarded. Frames thrown
away earlier — in the DirectShow ring buffer, because the writer could not keep up
— never reach the muxer and never appear there. FFmpeg does report them, on a line
reading *"real-time buffer … too full! frame dropped!"*, but that line contains no
word the error scan matched (`error`, `could not`, `invalid`, …), so it was
discarded silently. A capture could lose a third of its frames with the on-screen
drop counter reading zero.

Worse, `-thread_queue_size` was not set at all, so FFmpeg's default applied:
**8 packets**, which at 240 fps is 33 ms of tolerance — shorter than a single NTFS
flush. That default is the standard cause of *"Thread message queue blocking"* on
long high-rate captures, and the frame loss that follows it.

**Now:** both overflow and queue-blocking lines are parsed and counted, summed in
the live tile, recorded separately in the sidecar (`rt_buffer_drops`,
`input_queue_blocked`, `capture_warnings`), and restated in the completion
message. `-thread_queue_size` ships at 4096 (~17 s of 240 fps video, ~10 MB of
RAM at this frame size) and is exposed in Settings. The status line gives a
verdict — *"The capture buffer overflowed N times: frames are being lost before
they reach the file"* — rather than a number to interpret.

A pre-flight **write-path measurement** was added alongside, because the failure
is a *stall*, not a throughput shortfall: 240 fps of MJPEG is 0.6 MB/s, which any
disk sustains, but an SMB round trip or a USB drive parking its heads for 400 ms
overflows the buffer while the camera carries on. The check writes 6 MB in 256 kB
pieces, each fsynced, and reports the **worst single pause** rather than the
average rate.

### 2.12b Long captures had no archiving story, and the obvious one is wrong

A 30-minute trial at full temporal resolution needed an answer, and the
arithmetic turns out to be reassuring in one direction and alarming in the other:

| Mode | Per frame | 30 min | Write rate |
|---|---|---|---|
| 320×240 MJPEG @ 240 fps | 2456 B | **1.06 GB** | 0.59 MB/s |
| 320×240 uncompressed YUYV @ 240 fps | 153 600 B | **66.4 GB** | 36.9 MB/s |

So a long high-frame-rate trial is not intrinsically unmanageable — **27× of the
size comes from not using the camera's own MJPEG encoder**, and MJPEG is
all-intra, so nothing is gained by avoiding it. The instinctive response to large
files, lowering the frame rate, is the one change that destroys the measurement.

For footage that is genuinely large, the constraint is that **an HTR recording
must stay all-intra**. An inter-frame codec exists to remove small frame-to-frame
differences: motion estimation turns the head's displacement into a block vector,
the residual is small, and the quantiser discards it. H.264 at default settings
destroys HTR footage that looks perfectly fine to a human, because a human cannot
see 90 Hz. Every option offered therefore keeps `-g 1` and no B-frames, and none
changes the frame rate.

And here the conventional advice — archive losslessly — proved **wrong** for this
footage, which is the reason the tool measures rather than tabulating. Lossless
coding must reproduce the camera's own JPEG ringing and sensor noise exactly, and
that costs more than the JPEG did: FFV1 came out **5.3× larger than the MJPEG
source** on a 240 fps benchmark. FFV1 is right for *uncompressed* captures, where
it roughly halves them. Measured on the same file: re-quantised MJPEG 98 %,
all-intra H.264 at CRF 15 89 %, and **cropping to the arena 19 %** — by far the
largest saving and the only one that is free scientifically, since pixels outside
the cylinder carry no information about the animal. The arena is detected from the
same background model the analysis uses, so it is one click.

Every conversion is verified by comparing frame counts in and out, and a mismatch
is a **failure**, not a warning: a file that quietly lost its last thirty seconds
would not be noticed for a year. The self-test additionally re-analyses an
archived copy and checks it scores the same.

### 2.13 The disk-space figure was out by an order of magnitude

The SOP states "produces ~17 GB of raw .avi at 240 fps × 320×240" for 70 minutes
and the checklist asks for "≥ 25 GB free per trial". Its own measurement
contradicts this: 17.7 MB for 30 s is 2456 bytes per frame, so 70 minutes at
240 fps is **about 2.5 GB**. The error would have driven unnecessary storage
purchasing and, worse, a disk check that passes trials it should not.

**Now:** the estimate is computed from the measured bytes-per-frame and scaled by
pixel count and delivered rate. The Acquisition tab shows it before you start and
the pre-flight check compares it against actual free space.

### 2.14 No way to tell lighting flicker from a twitch

The SOP recommends DC constant-current panels and explains why. There was no way
to check that the panels in the room actually are DC.

This matters more than any software setting. A PWM-dimmed LED modulates the whole
frame at a few hundred hertz; under a sub-millisecond global-shutter exposure
that becomes frame-to-frame brightness stepping, which rings the HTR band-pass
across the entire recording. It produces whole-field false positives that no
amount of analysis can undo, and it is invisible to the eye.

**Now:** two independent checks.

* *Before recording* — the Acquisition tab's flicker test records 2 s, takes the
  per-frame mean intensity, removes the trend, and reports the dominant ripple
  frequency and its depth. Ripple inside 55–125 Hz, or near a mains harmonic, is
  reported as a blocking problem with the hardware fix named.
* *During analysis* — frame-difference energy is measured **outside** the animal
  as well as inside. If the empty arena is oscillating in the HTR band, the
  result carries a loud warning, and individual bursts whose background
  oscillates as strongly as the mouse are rejected as flicker.

This check is not hypothetical: it fires on the original
`benchmark_videos/synthetic_mouse_htr.mp4`. That generator reseeds its sensor
noise from `int(t * 100)`, so the noise pattern is held for about 2.4 frames at
240 fps and then changes — a whole-field artefact at roughly 100 Hz, inside the
HTR band. The legacy benchmark is therefore a *pessimistic* yardstick, and is
kept as a regression fixture rather than as the measure of quality.

### 2.15 No human in the loop, and no audit trail

The original exported the bursts it accepted. Rejected candidates were discarded,
so there was no way to see what the detector nearly called, and no way for a
scorer to correct it. HTR counts are conventionally human-verified; a tool that
cannot record a human decision cannot be used that way.

**Now:** the Review tab steps through candidates with the footage and the signal
side by side, and each one can be accepted, rejected, or added by hand. The
curated set is saved with the recording and is what the export uses. Rejected
candidates are kept with the reason they were rejected. A batch run never
overwrites a curated result.

### 2.16 Smaller things

* `htr_capture.bonsai` needs the `Bonsai.Vision` NuGet package, which the SOP
  documents installing by hand, and `Take()` segfaults on teardown. HTR Studio
  drops Bonsai from the capture path entirely — the FFmpeg route is the one that
  was actually measured at 240 fps, and it has no package-management problem.
* `htr_analysis.py` imports `collections.deque` and never uses it;
  `per_minute_counts` takes two arguments it ignores.
* `test_arducam_acquisition.py` scores cameras by preferring 640×480, which is
  the mode the SOP tells you not to use.
* `htr_analysis_env.txt` pins `deeplabcut[nn]` into the same environment as the
  GUI. DLC's own pins have conflicted with that in practice, which is why the new
  installer puts it in a separate environment that cannot break the app.
* The SOP's numbering jumps (section 9 before section 8).

---

## 3. Measured result

Same input file, same ground truth, default settings on each side:

| | Legacy `htr_analysis.py` | HTR Studio |
|---|---|---|
| `benchmark_videos/synthetic_mouse_htr.mp4` | precision 0.75, recall 1.00, **F1 0.86** | precision 1.00, recall 1.00, **F1 1.00** |
| New benchmark, with grooming / fast turns / simulated PWM flicker | not applicable | precision 1.00, recall 1.00, **F1 1.00**, zero false positives on any confounder |
| Longest recording that can be analysed | ~50 s (12 000-frame cap) | unlimited; RAM use independent of length |
| Stable across threshold multiplier k = 4…8 | not tested | yes, F1 1.00 throughout |
| Parallel extraction | none | bit-identical to single-threaded, verified in the self-test |
| Realistic benchmark: bedding, cylinder rim, lighting gradient, moving shadow, low-contrast animal | not applicable | 70.9 % tracked / F1 0.667 by thresholding → **100 % tracked / F1 1.000** with background subtraction |
| 8× slow-motion render of a 240 fps capture | not applicable | 0 detections as the file declares itself; **F1 1.000** with the capture rate recovered |
| Frame loss during capture | `drop=` only, which misses buffer overflows entirely | both counted, summed in the live tile, recorded separately in the sidecar |
| Archiving a long trial | not addressed | all-intra options measured on the operator's own footage, frame count verified in and out |

The second row is the one worth weight. Scoring well on a file containing only
true positives says little; what limits a real HTR count is firing on grooming
and on fast turns. The new benchmark plants both, plus a stretch of simulated PWM
lighting, and the detector ignores all of them.

Measured throughput on a 2-core test machine: about 300–500 frames/s per thread,
so a 70-minute trial is roughly 35–55 minutes single-threaded. On the reference
host (i7-10700, 8 cores) with 4–6 worker threads, expect 8–15 minutes per trial.
A twelve-animal study is an overnight batch, which is what the Batch tab is for.

---

## 4. Things left for you to decide

**The threshold multiplier is the one knob that matters.** It is stable from 4 to
8 on synthetic data, but synthetic data has a stationary noise floor and real
footage does not. After the first three to five animals, look at the baseline
periods: they should score near zero. If they do not, raise `threshold_k` and
re-run the batch — it is quick, because the recordings do not change.

**The oral-dose time course is still an open question.** The SOP's estimate —
peak at 20–45 min post-gavage, tail to 60–90 min — is an extrapolation from
intraperitoneal data, and it says so. The 70-minute window is a reasonable bet,
not a finding. After a handful of animals the exported time courses will show
where the peak actually falls, and the window can be shortened or extended. The
`bin_rel_t0` column exists so those curves line up across animals.

**Validate against a human scorer at least once.** Pick two or three recordings,
have someone score them by eye from the footage, and compare against the Review
tab's curated output. That comparison is what lets you state a detection
performance figure in a methods section. Until it exists, the F1 = 1.00 above is
a statement about synthetic video, and should be described that way.

**If blob tracking struggles on real footage**, the pose-estimation route is
still open and `install_dlc.bat` sets it up. Only the angle source changes; the
entire detector chain after it — band-pass, envelope, hysteresis, gates, binning
— is unchanged, which is also the structure the original work anticipated.
Try background segmentation and the tuning view first, though: the one case of
"real footage defeats blob tracking" that has been examined so far was fixed by
background subtraction, and the measurement to make the decision on — scan 200
frames, read the hit rate and the area variation — is in the tuning view.

**Decide the archive policy once, for the study, not per trial.** The defensible
version: keep the original capture until that trial's analysis is finished and
curated; then archive with crop plus all-intra H.264, confirm the archived copy
re-analyses to the same count, and only then delete. Writing that down in the SOP
is worth more than any individual codec choice, because the failure mode is not
picking the wrong codec — it is deleting an original before anyone checked.
