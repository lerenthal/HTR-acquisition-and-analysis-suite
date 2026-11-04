# HTR Studio — user manual

Acquisition and analysis of the mouse head-twitch response, built around the
Arducam ezBOX GS B0634 (OV9281 monochrome global shutter) recording 320×240 at
240 fps.

- [Before you start: what the measurement actually is](#before-you-start-what-the-measurement-actually-is)
- [Five minutes to your first result](#five-minutes-to-your-first-result)
- [The Acquisition tab](#the-acquisition-tab)
- [The Library tab](#the-library-tab)
- [The Review tab](#the-review-tab)
- [The Analysis tab](#the-analysis-tab)
- [The Batch tab](#the-batch-tab)
- [The Settings tab](#the-settings-tab)
- [Running a study end to end](#running-a-study-end-to-end)
- [Output files](#output-files)
- [Validating the detector](#validating-the-detector)
- [Adding a pose model later](#adding-a-pose-model-later)
- [Keyboard shortcuts](#keyboard-shortcuts)

---

## Before you start: what the measurement actually is

Three facts shape every default in this program. They are worth holding onto,
because almost every way of getting an HTR count wrong follows from ignoring one
of them.

**1. An HTR is a 90 Hz oscillation, not a movement.** Halberstadt et al.
(*J Neurosci Methods*, 2014) characterised it as **5–11 oscillations at 78–98 Hz**
— so a burst lasts 50 to 150 ms. That frequency is the signature. A fast turn of
the head is a much larger movement with no 90 Hz content, and distinguishing the
two is the whole job.

**2. 240 fps is not a luxury.** To resolve a 98 Hz oscillation you need to sample
above 196 Hz; practically, 240 fps gives a Nyquist limit of 120 Hz with margin.
At 100 fps the Nyquist limit is 50 Hz and the oscillation **aliases** — it folds
down into a lower frequency and the measurement is not recoverable. This is why
the production mode is 320×240 at 240 fps rather than 640×480 at 100 fps, even
though VGA looks nicer. Spatial resolution you can afford to lose; temporal
resolution you cannot.

**3. Lighting flicker is indistinguishable from a twitch unless you check.** A
PWM-dimmed LED panel modulates brightness at a few hundred hertz. Under a
sub-millisecond global-shutter exposure that becomes frame-to-frame brightness
stepping across the entire frame, which rings the 90 Hz band everywhere at once.
You cannot see it, and no amount of filtering removes it. Use DC
constant-current panels, and run the flicker test to confirm you have.

---

## Five minutes to your first result

No camera and no mice needed — this is how to confirm the installation works and
to learn the interface.

1. Start the app: `run_htr_studio.bat`.
2. **Tools → Make a synthetic benchmark video**, confirm, and wait a couple of
   minutes. It writes `data\benchmark\synthetic_htr_benchmark.avi` plus a
   ground-truth CSV. The clip contains six head twitches at known times and three
   kinds of confounder: grooming bouts, fast turns, and a stretch of simulated
   PWM lighting flicker.
3. **Library** → set the library root to `data\benchmark` → **Rescan**. The clip
   appears.
4. Select it → **Analyse**. On the Analysis tab, press **Run analysis**. It takes
   under a minute.
5. You should see **6 twitches**. Press **Validate against ground truth**, pick
   `synthetic_htr_benchmark_truth.csv`, and you should get
   `precision=1.000 recall=1.000 F1=1.000`.
6. Switch to **Review**. Step through the six bursts with the `burst ▶▶` button,
   watch each at 1/8 speed, and accept or reject it. This is exactly the loop you
   will use on real trials.
7. Back on **Analysis**, **Save CSV report** and **Save figure**. Open them and
   you have seen the whole output format.

---

## The Acquisition tab

Five numbered groups, in the order you use them.

### 1 — Camera

**Refresh** enumerates DirectShow devices (what FFmpeg records from) and OpenCV
indices (what the live preview uses). The B0634 normally appears as
*Arducam ezBOX VGA GS Mono*; if several cameras are attached, pick the right one
by hand.

**List modes** asks the driver what it advertises. Treat that as a claim, not a
fact — see Probe, below.

**Camera properties…** opens the driver's own DirectShow property page. This is
the authoritative place to set exposure and gain: values set there are stored in
the driver and survive the camera handle being closed, so FFmpeg inherits them
when recording starts. The sliders in this tab try to do the same thing through
OpenCV, which works on most UVC drivers but is not guaranteed — if the app warns
that a value did not read back as set, use this dialog instead.

### 2 — Capture mode

Pick a preset. The notes on each one are measurements, not opinions:

| Preset | Delivered | Use it when |
|---|---|---|
| **320×240 @ 200 request** | **240 fps** | Always, for HTR. Nyquist 120 Hz covers the band. |
| 640×480 @ 120 request | ~100 fps | Collecting frames to label for a pose model; any scoring that is not HTR. **Aliases the twitch.** |
| 640×480 @ 100 | 100 fps | As above. |
| 160×120 @ 400 | 400 fps | Experimental. Very coarse spatially; blob tracking may fail on a 15–20 cm arena. |
| 320×240 @ 100 YUV | 100 fps | No MJPEG artefacts, ~6× the disk rate, and it aliases. Rarely worth it. |

**Probe this mode** records two seconds and reports what the camera actually
delivered. **Probe all presets** does the lot. Do this once when the rig is set
up and again if anything about the USB path changes. The camera *negotiates*: it
delivers 240 fps for a 200 fps request at 320×240, and about 100 fps for a 120 fps
request at VGA. Probing is how you find out rather than assume.

> The camera is claimed exclusively while probing, so the preview stops first.

### 3 — Exposure and lighting

Start the preview (right-hand panel) and work with the numbers, not your eye.

**Auto-exposure: leave it off.** It hunts as the mouse crosses the frame, and
that brightness modulation lands in the band you are measuring.

**Exposure** is the UVC log2 value, so −7 is about 7.8 ms of *permitted*
integration, clamped by the frame interval (4.2 ms at 240 fps). More negative is
shorter: −9 ≈ 2 ms, −10 ≈ 1 ms. Shorter exposure freezes the oscillation; too
short and you lose contrast. Start at −7 and shorten until the clipped-pixel
figure is near zero and contrast is still comfortable.

**Gain: 0.** Gain amplifies sensor noise, and sensor noise is broadband, which
means it lands in the HTR band along with everything else.

The five tiles under the preview are what to judge by:

| Tile | Target | If it is wrong |
|---|---|---|
| preview fps | near the mode's rate | Preview is not the recording path; a low figure here is not a problem by itself. |
| contrast | **> 70** of 255 | Below 40 blob tracking will be unreliable. Matte black floor under the cylinder; more light; longer exposure. |
| clipped | **< 1 %** | Blown highlights. Less light, or shorter exposure. |
| focus | **> 40** | Refocus, then lock the ring (epoxy or hot glue — a focus that drifts mid-study is unrecoverable). |
| uniformity | **> 0.70** | Corner-to-corner brightness ratio. Move the panels further out, or add diffusion. |

Verdict lines underneath name the problem and the fix.

**Run flicker test (2 s)** — do this once per session, and after anyone touches
the lights. It records two seconds, takes the per-frame mean intensity, removes
the trend, and reports the dominant ripple frequency and its depth:

* *under 0.5 % ripple* — steady light, DC drive confirmed.
* *ripple at 55–125 Hz, or near a mains harmonic* — **stop**. This will generate
  false positives across the whole recording. Switch the panels to constant-current
  DC drive (not PWM dimming) and set brightness with distance or diffusion.
* *ripple elsewhere* — usable, but a DC panel would be cleaner.

### 4 — Trial

Fill this in. It is written into a JSON sidecar next to the video, and it is what
makes the recording interpretable six weeks later: animal ID, group, treatment,
dose, route, operator, notes.

**Total duration** defaults to 70 min: 10 min pre-dose baseline plus 60 min after.
The baseline is not padding — it is where the detection threshold comes from.

**Segment length** defaults to 5 min. Segmenting writes the trial as a folder of
5-minute files with a manifest tying them together. An interruption then costs at
most the last segment instead of the whole 70 minutes, and the pieces are small
enough to copy. The Library shows the set as one row. Set it to 0 for a single
file.

The line underneath shows the estimated size (about **2.5 GB** for a 70-minute
trial at 240 fps) and how many segments that is.

### 5 — Record

**Pre-flight check** first, every time. FFmpeg and FFprobe present, camera
selected, output folder writable, enough free disk for the estimate, **the write
path measured**, **segmenting sane for this duration**, output name not already
taken, T0 not yet marked, exposure mode manual, and whether the chosen mode
actually resolves the HTR band. Blocking problems are marked and the Start button
refuses until they are fixed; warnings ask for confirmation.

**Write path** deserves a word, because it is where long captures actually fail.
240 fps of MJPEG is about 0.6 MB/s, which any disk can sustain — on paper. What
breaks a capture is not throughput but a *stall*: an SMB round trip, a USB drive
parking its heads, a sync that takes 400 ms. The camera does not wait. So the
check writes 6 MB in 256 kB pieces, each one flushed and fsynced, and reports the
**worst single pause** rather than the average rate. A pause over 250 ms is long
enough to overflow the capture buffer and lose frames. Record to a local SSD and
copy the files afterwards; never record straight to a network share.

**Start recording** stops the preview (a DirectShow device admits one client at a
time), pushes the exposure and gain into the driver, and launches FFmpeg.

**Mark T0 (dose)** — press it **at the moment of gavage**. It writes the offset
from the start of the recording into the sidecar. Everything before it is treated
as baseline, which is where the threshold comes from, and the exported time
course is binned relative to it so animals line up even when a baseline ran long
or short. If you forget, you can enter it by hand later in the Library — but
pressing the button is one click and guessing afterwards is not.

While recording, the tiles show elapsed media time, instantaneous fps, frame
count, **frames lost** and file size, and the status line gives a verdict rather
than a number.

**Frames lost counts two different things, deliberately.** FFmpeg's own `drop=`
counter reports only frames the *muxer* discarded. Frames thrown away earlier —
in the DirectShow ring buffer, because the writer could not keep up — never
appear there at all. FFmpeg does say so, on a line reading *"real-time buffer …
too full! frame dropped!"*, but that line contains no word a generic error scan
would match, so it is easy to lose a third of a recording with the drop counter
reading zero on screen. Both are now counted, reported separately in the sidecar
(`dropped` and `rt_buffer_drops`), and summed in the tile. If the capture
overflows even once, the status line says so in red and the recording is suspect:
every lost frame is a step in the angle trace that the detector can read as a
twitch.

Buffer sizes live in **Settings**. Two matter:

* `rtbufsize` (512 MB) — DirectShow's ring buffer before FFmpeg sees a packet. At
  2.5 kB a frame that is minutes of slack.
* `thread_queue_size` (4096) — packets queued between the capture thread and the
  writer. **FFmpeg's own default is 8**, which at 240 fps is 33 ms of tolerance —
  shorter than a single NTFS flush. That default is why long high-rate captures
  report *"Thread message queue blocking"* and lose frames; 4096 packets is about
  17 s of 240 fps video and ~10 MB of RAM at 320×240. If a capture still
  overflows on a local SSD, raise these before changing anything else.

If the effective rate drifts below 90 % of what the mode should deliver, stop and
re-probe — the camera is not in the mode that was requested.

When the recording finishes, the frame rate is verified automatically for
single-file recordings — an exact frame count against the expected rate — and any
frame loss is restated in the completion message. That is the moment to catch a
bad capture, while the animal is still in the room.

### How big will a 30-minute trial be?

Smaller than people expect, and the arithmetic is worth internalising because the
instinctive response to "huge files" is to lower the frame rate, which is the one
change that destroys the measurement.

A 320×240 MJPEG frame from this camera is about 2456 bytes, measured.

| Mode | Per frame | 30 min | 70 min | Write rate |
|---|---|---|---|---|
| 320×240 MJPEG @ 240 fps | 2.5 kB | **1.06 GB** | 2.48 GB | 0.59 MB/s |
| 640×480 MJPEG @ 100 fps | 9.8 kB | 1.77 GB | 4.13 GB | 0.98 MB/s |
| 320×240 **uncompressed** YUYV @ 240 fps | 150 kB | **66.4 GB** | 155 GB | 36.9 MB/s |

A gigabyte for half an hour at full temporal resolution is not a problem. **66 GB
is**, and that is what an uncompressed capture costs — 27× the size for the same
frames, and it gains nothing: the camera encodes MJPEG in hardware, and MJPEG is
all-intra, so every frame is already independent. If your recordings are
unmanageably large, this is almost certainly why. The fix is to let the camera
encode (pixel format `mjpeg`), never to slow the camera down.

The **Plan a capture** page under Library → Inspect / archive does this
arithmetic for any combination, including the sustained write rate to check
against the pre-flight measurement.

Segmenting (5 min by default) is on for the same reason it always was: a single
30-minute AVI whose index is written at the end is unreadable if the capture is
interrupted, whereas an interruption costs at most one 180 MB segment, and each
segment is small enough to copy and to open.

---

## The Library tab

Every recording under the library root, with its metadata and what has been done
to it.

Scanning never decodes video, because `ffprobe -count_frames` on a 70-minute
trial takes minutes. The table shows container header values; the **fps** column
marks an unverified figure with `~`.

**Verify frame rate** does the exact count on demand and caches it in the
sidecar, so it is paid for once. Do this for every trial you intend to analyse.
The container frame rate for MJPEG-in-AVI is routinely wrong, and the band
definition, the burst durations and the time course all depend on getting it
right.

The detail panel on the left edits trial metadata — including **T0 offset** if you
forgot to press the button. The panel on the right shows everything known about
the file: what mode it was recorded in, what the camera was set to, the lighting
metrics measured at record time, what FFmpeg observed, the verified frame rate,
and the current analysis state.

Actions: **Open in Review**, **Analyse**, **Add selection to Batch**, **Verify
frame rate**, **Inspect / archive**, **Show in Explorer**, and **Move to recycle
bin** (never an unrecoverable delete — trials are irreplaceable and a mis-click in
a file list should not be able to destroy one).

### Inspect / archive

Use this on any footage that did not come from this tool's own Acquisition tab —
a file from another capture program, or a paper's supplementary video. It has
three pages.

**What is this file?** Codec, the measured frame rate against the declared one,
whether every frame is coded independently, and bytes per frame. It then says
what to do about what it found. The one judgement it makes that is worth
understanding: an **inter-frame** codec (H.264, HEVC, VP9, MPEG-4) describes each
frame in terms of its neighbours, and that is precisely a mechanism for removing
small fast changes. Motion estimation turns the head's displacement into a block
vector, the residual is small, and the quantiser discards it. A twitch can be
smeared or gone from an H.264 re-encode that looks perfectly fine to the eye —
because a person cannot see 90 Hz. Counts from such a file are provisional;
analyse the original capture if one exists.

**Archive / convert** — see the next section.

**Plan a capture** — what a given duration, frame rate, frame size and pixel
format will cost on disk, and whether that is sane.

### Archiving long recordings

The constraint that governs every option here: **an HTR recording must stay
all-intra**, every frame coded independently. So every conversion offered keeps
`-g 1` and no B-frames, and *none of them changes the frame rate*. There is no
"archive at a lower frame rate" option because there is no correct way to do it —
halving 240 fps to 120 takes the Nyquist limit to 60 Hz, below the band, and
destroys the measurement in exchange for a file half the size.

Within that constraint the savings are still large. In order of how much they
save per unit of risk:

1. **Crop to the arena.** Free, scientifically — pixels outside the cylinder
   carry no information about the animal. Press **Find the arena** and it is
   detected from the same background model the analysis uses, so it is one click
   rather than four numbers guessed off a screenshot. Measured: a 640×480 clip
   whose arena occupied 438×438 lost 38 % of its pixels; a 320×240 clip with a
   150×150 arena lost 81 %.
2. **All-intra H.264** (`-g 1 -bf 0 -crf 15`). Typically the smallest option.
   Lossy, so it is for storage after the analysis is finished and checked.
3. **Re-quantised MJPEG.** The same codec the camera produces, so nothing about
   the decode path changes. The least disruptive lossy option.
4. **Repackage only** (`-c:v copy`). Rewrites the container and rebuilds the
   index without touching a pixel. Use it to rescue a half-written AVI. Identical
   size, zero risk.
5. **FFV1**, mathematically lossless — *and usually the wrong choice here*. See
   below.

**Measure before you choose.** Press **Measure** and each option encodes a few
seconds of your actual footage; the result is scaled to the whole recording.
Nothing is predicted from a bitrate table, because MJPEG's cost is set almost
entirely by how much texture the bedding has and FFV1's by how noisy the sensor
is. The conventional advice — archive losslessly — turns out to be *wrong* for
MJPEG camera footage, and only a measurement reveals it: lossless coding has to
reproduce the camera's own JPEG ringing and sensor noise exactly, which costs
more than the JPEG did. On a 240 fps benchmark FFV1 came out **5.3× larger than
the source**. FFV1 is the right answer for footage captured *uncompressed*, where
it typically halves the size.

Every conversion is verified by counting frames in and out, and a mismatch is
reported as a **failure**, not a warning: a file that quietly lost its last
thirty seconds will not be noticed for a year. The original is never modified or
deleted — keep it until the analysis for that trial is finished and checked, then
confirm the archived copy analyses to the same answer before removing anything.

A segmented session shows as one row with a folder icon. Analysis and Review work
on its first segment; to analyse all the segments, add the session to the Batch
tab.

---

## The Review tab

This is where a count becomes a result you can defend.

Automated scoring is not a substitute for watching the video; it is what makes
watching affordable. A 70-minute trial is a million frames. The detector reduces
it to a few dozen candidates, and this tab puts each one on screen with its
signal so you can confirm it in seconds.

The layout: footage on the left, the head-angle trace and band-pass envelope on
the right (sharing one x-axis, with a draggable cursor), and the candidate list
underneath.

**Watch at 1/8 speed.** A 90 Hz oscillation is invisible at real speed. The
speed selector goes down to 1/16×.

**`burst ▶▶` / `◀◀ burst`** jump between candidates, which is the fast way
through a trial. Clicking a row in the table also jumps to it.

**Show tracking overlay** draws the fitted body centroid and the head direction
from the last analysis. This is how you tell "the detector saw a twitch" from
"the detector lost the mouse and the tracking jumped". If the arrow is pointing
at the rump, or flicking between ends, the segmentation needs attention before
the counts mean anything.

For each candidate: **Accept**, **Reject**, or **Toggle**. **Add burst at cursor**
records one the detector missed, with the duration from the spin box beside it.
**Show rejected candidates** lists what the detector nearly called, with the
reason it was rejected — and a rejected candidate can be promoted if you disagree.

**Save curated result** writes your decisions next to the recording. From then on
the Library shows the trial as *curated*, the export uses your counts, and a
batch run will not overwrite it. **Export curated CSV** writes the report from
the curated set rather than the raw detector output.

Rejecting a detected burst keeps it in the record with your rejection, rather than
deleting it. That disagreement is the audit trail, and it is also labelled
training data if a pose model is ever trained.

---

## The Analysis tab

Every parameter has a reason, and the tooltips carry it. The defaults are the
validated ones — F1 = 1.00 on both synthetic benchmarks — so change them
deliberately.

### Mouse tracking

| Setting | Default | What it does |
|---|---|---|
| Angle source | head direction, sub-pixel | Full 0–360° heading, so no 180° flips. The ellipse and moment axes are ambiguous by 180°, and each flip is a step that the band-pass reads as a twitch. They remain selectable for comparison. |
| Segmentation | auto | How the animal is separated from the arena. See below — on real footage this is the setting that matters most. |
| Mouse vs background | auto | Tries both polarities on sampled frames, keeps whichever locates the animal more often. |
| Segmentation threshold | otsu | Computed once from frames across the file and then frozen, so segmentation does not drift as the animal moves. |
| Background frames | 40 | How many frames are pooled into the empty-arena median. |
| Region of interest | none | Off by default — see below. |
| Blob area min / max | 0.004 / 0.25 of the frame | Rejects dust at one end and scenery at the other. An animal viewed top-down is ~1% of the frame on published footage; the old 0.45 ceiling let a 33% blob (the lit arena floor) be accepted as the animal. |
| Clean-up kernel | 5 px | Morphological open and close. |
| Tail-removal kernel | 9 px | Must be wider than the tail and narrower than the head; the opening removes the tail so the head end can be found. |
| Interpolate gaps up to | 60 ms | Short tracking losses are bridged. Longer ones stay flagged, and a burst sitting on them is rejected. |
| Mark edge-touching frames as untracked | off | Off by default — see below. |

#### Threshold or background?

**Threshold** separates the animal by brightness alone. It is correct on a clean,
evenly lit arena and it is what the first-generation pipeline did.

**Background** builds an empty-arena image — the per-pixel median of 40 sampled
frames, in which the animal is somewhere different each time and so is the
minority at every pixel — and subtracts it. Everything static cancels: bedding
texture, the cylinder rim, the lighting gradient, the dark bench outside the
arena. This is the change that makes real footage work. On a 640×480 benchmark
with bedding, a rim highlight and a moving cast shadow, plain thresholding
tracked 70.9 % of frames and scored F1 = 0.667; background subtraction tracked
100 % and scored F1 = 1.000.

**auto** calibrates both and keeps threshold while it is working — hit rate ≥ 90 %
*and* blob-area coefficient of variation ≤ 0.35 — falling back to background when
it is not. It is deliberately conservative in that direction, because background
subtraction is not always better: on the clean fixtures it was marginally worse,
and silently switching would have cost accuracy there.

If a trial tracks badly, change this before touching any detector parameter. A
threshold that grabs the bedding as well as the mouse produces an angle trace
that no detector setting can rescue.

#### Two defaults that look wrong and are not

**Region of interest is off.** An ROI drawn tight to the arena clips the animal
whenever it presses against the wall, and a clipped outline puts the computed
head direction somewhere it never was. Measured on the realistic benchmark,
turning auto-ROI on took F1 from 1.000 to 0.500 and raised the blob-area
variation from 0.09 to 0.22 — it created nine false positives while removing
contrast that background subtraction already handles.

**Edge-touching frames are not rejected.** The reasoning for rejecting them is
sound: a clipped outline gives a wrong angle. The measurement disagrees about the
cost — on a tight arena it rejected so many frames that genuine twitches went
with them (tracking fell from 100 % to 67.8 %, and the legacy fixture lost all
six of its events). Turn it on only if you can see clipping causing false
positives, and check the tracked-frame percentage when you do.

### Twitch detection

| Setting | Default | Why |
|---|---|---|
| Detector | head-angle band-pass | The validated one. |
| Pass band | 60–110 Hz | Published range is 78–98 Hz; this leaves margin without reaching into locomotion harmonics. |
| Threshold from | baseline | The pre-dose period, so the drug effect cannot raise its own threshold. |
| Robust sigma multiplier | 6.0 | median + k × MAD-sigma. Stable from 4 to 8 on synthetic data. **This is the one knob worth tuning on real footage.** |
| Burst release fraction | 0.50 | Hysteresis. A burst starts above the threshold and continues down to half of it, so one twitch stays one event. |
| Baseline window | 0–0 (auto) | Uses the recorded pre-dose period. Set explicitly to override. |
| Burst duration | 45–600 ms | 5–11 oscillations at 78–98 Hz is 51–141 ms, so 45 ms is the floor; the ceiling is deliberately loose. |
| Merge gaps shorter than | 60 ms | Two twitches that close together are not plausible. |
| Require in-burst frequency | 55–125 Hz | Measured from zero crossings inside the candidate. Rejects broadband lurches. |
| Minimum half-cycles | 6 | A twitch oscillates; 5–11 cycles is 10–22 half-cycles, so 6 is forgiving. |
| Max invalid frames | 0.25 | A burst on lost tracking is an artefact. |

**The four detector choices:**

* **Head-angle band-pass** — the default and the recommendation. F1 = 1.00 on
  both benchmarks.
* **Motion-energy spectral** — independent of the angle fit. It measures whether
  the pixels inside the mouse oscillate in band, using a band-power *ratio*
  (70–110 Hz against 8–50 Hz) so that overall activity cancels out. High
  precision, **low recall**: its 256 ms analysis window dilutes a 100 ms burst,
  so it finds roughly a third of them. A cross-check, not a primary detector.
* **Both must agree** — most conservative. Inherits the spectral detector's
  recall, so it will miss roughly two thirds of real twitches. Use it only when a
  false positive is much worse than a miss, and say so in the methods.
* **Either** — most sensitive. F1 = 1.00 on the benchmarks too, but it has more
  ways to fire on a trial that is not synthetic.

### Performance

**Worker threads** (auto by default) decodes separate ranges of the file in
parallel. The self-test verifies the parallel result is bit-identical to the
single-threaded one, so there is no accuracy reason to prefer one thread. Only
used on files over ~20 000 frames and only when frame-exact seeking works, which
it does for MJPEG in AVI.

**Frame limit** caps the analysis for a quick look at the start of a long trial.
Analysis is streaming, so there is no memory reason to use it.

### Reading the result

Five tiles: twitches, peak rate per minute, threshold, tracking valid percentage,
rejected candidates. Below them, the full diagnostic text — warnings, the
background flicker figures, every rejected candidate with its reason, and every
accepted event with its duration, dominant frequency and half-cycle count.

**Read the warnings.** They are not decoration. The ones that matter most:

* *"The arena background is itself oscillating in the HTR band"* — lighting
  flicker. The counts from this recording cannot be trusted. Fix the lighting.
* *"tracking succeeded on only N% of frames"* — segmentation is failing. Fix
  contrast or tune the threshold before believing anything.
* *"at N fps the Nyquist limit is only M Hz, which cuts into the HTR band"* —
  the recording was made in an aliasing mode. Not recoverable in software *if the
  camera really ran at that rate* — but read the next section first, because a
  slow-motion render produces exactly this warning and is completely recoverable.

### When the analysis finds nothing

"0 twitches" is the same output for two opposite situations: an animal that did
not twitch, and a pipeline that could not have seen a twitch if it had. The
difference matters enormously and is invisible in the burst list, so the run ends
with a **What to do next** line that names the most likely cause. It checks, in
order:

1. **Tracking below 80 %.** The angle trace is mostly interpolation; no count
   from it is defensible. Nothing else is worth examining until this is fixed —
   open **Tune tracking** and try background segmentation.
2. **The oscillation is far below the band.** The file is probably a slow-motion
   render. See below.
3. **The frame rate cannot reach the band.** Below ~196 fps a 78–98 Hz
   oscillation aliases at the sensor and no setting recovers it.
4. **The band is empty but the rate is fine.** The trace genuinely has no twitch
   in it. Check a burst by eye before loosening anything.
5. **The band has energy but nothing passed the gates.** Now — and only now — the
   detector thresholds are the place to look.

#### Signal diagnostics

**Signal diagnostics** (next to the capture-rate override) plots the power
spectrum of the head-angle velocity with the 78–98 Hz band and the file's Nyquist
limit drawn on it. Whether the signal is reachable stops being a question of
judgement and becomes something you can see.

Pick a different source to interrogate further. **Frame-to-frame motion,
background only** is the control: in-band power *there* is lighting flicker or
sensor noise rather than the animal, and it will produce false positives
everywhere.

Two numbers are reported separately and should not be confused. The **strongest
peak** is the tallest single component anywhere in the spectrum — in any trial
with locomotion that is 5–10 Hz, because locomotion runs continuously while a
twitch is 100 ms in seventy minutes. The **power in band** is what says whether
the twitch is there at all. A recording can have 60 % of its movement energy
inside the HTR band while its tallest peak sits at 8 Hz, and that recording is
perfectly good.

#### Slow-motion renders, and the capture-rate override

A published supplementary video captured at 240 fps is routinely written at 30
fps so it plays as slow motion. Every frame is still there; only the declared
rate is wrong. A 90 Hz twitch then sits at 11 Hz in the file's own timebase, so
it is plainly visible to the eye and completely invisible to a detector
band-passing 78–98 Hz on file timestamps.

This is the single most likely reason that footage with obvious twitches yields
nothing. It is also trivially fixable. Signal diagnostics recognises the pattern
— an oscillation at an integer fraction of the HTR centre frequency in a file
whose declared rate is low — and offers the implied capture rate; one click puts
it in the **Capture-rate override** box and the next run uses it.

Measured on a reconstructed 8× render of a 240 fps capture:

| Analysed at | Tracking | Detected | F1 |
|---|---|---|---|
| the file's declared 30 fps | 100 % | 0 | 0.000 |
| the true 240 fps | 100 % | 4 of 4 | 1.000 |

Two cautions. **Confirm it against the footage**: if the animal moves at normal
speed, the file is not slow motion and the low peak is something else (grooming
is 6–9 Hz). And the override cannot rescue a recording genuinely captured below
~196 fps — there the oscillation aliased at the sensor and the information is
gone. The override is cleared whenever you select a different recording, because
carrying it onto the next trial would put every timing figure in that trial out
by the same factor. It is recorded in the attached result and the exported
report, so a run's timebase is always auditable.

#### Tune tracking

**Tune tracking** opens a frame-by-frame segmentation view that runs the *same*
`segment_frame` the extraction pass uses — what you see is what the analysis will
do, not an approximation. Scrub to any frame; switch **Show** between the overlay,
the binary mask, the background model and the difference image when the overlay
looks plausible but the angle does not.

Two readouts are the ones to optimise, and both come from **Scan 200 frames**:
the percentage of frames in which the animal is found, and the coefficient of
variation of the blob area. Below about 90 % found, or above 0.35 area variation,
the angle trace is not trustworthy and no twitch count from it means anything.
A steady blob area matters as much as finding the animal: an area that jumps
around means the mask is grabbing and releasing bedding, which moves the computed
centroid and therefore the head direction.

**Use these settings** copies the tuned values back into the Analysis tab, where
they are saved and exported with the run like any other parameter.

### Export

* **Save CSV report** — three CSVs (see [Output files](#output-files)).
* **Save figure** — a three-panel PNG: rate time course, envelope with the
  threshold, head angle with lost-tracking shaded.
* **Save traces** — the full per-frame signals as `.npz`, for re-plotting or
  analysing elsewhere without re-running.
* **Attach result to the recording** — stores it beside the video so the Library
  shows it and the Review tab can curate it. Do this; it is what makes the batch
  summary work.

---

## The Batch tab

For a study rather than a trial. Twelve animals at 70 minutes is about twelve
million frames — an overnight job.

Add videos, a whole folder, or a selection from the Library. Set the output
folder. Press **Run batch**.

Each trial gets its own CSV and figure, and the result is attached to the
recording. The run ends with **`study_summary.csv`** — one row per trial with
total, baseline and post-dose counts, peak rate, the peak's minute relative to
T0, latency to first twitch, tracking quality and any warnings — plus
**`study_rate_by_group.png`**, the mean ± SEM rate time course by group.

Three options worth knowing:

* **Skip trials already analysed** — on by default, so a re-run only does the new
  ones. A **curated** result is never overwritten regardless, because a human
  decision outranks a re-run.
* **Write a figure per trial** — worth the disk. A trial that went wrong is
  usually obvious in the figure and invisible in the summary row.
* **Rebuild summary from the queue** — regenerates the study table from results
  already attached to recordings, without re-analysing. Use this after curating.

A trial that fails does not stop the queue. It is marked FAILED with its reason
and the rest continue.

> Only the first three groups appear on the group figure. Past three, the
> categorical colours stop being distinguishable to a reader with colour-vision
> deficiency, so a fourth group gets its own figure rather than a fourth hue. The
> figure says so when it has left groups out.

---

## The Settings tab

Settings live in `config\settings.json` inside the installation folder — not in
the registry, not in your user profile — so copying the folder carries the
configuration with it.

**External tools** — leave the FFmpeg boxes blank to use the bundled build.
**Check tools** reports versions and paths.

**Folders** — **Recordings root** is the one to change: point it at the drive
with the space. Buttons open the exports, logs and install folders.

**Recording defaults** — file-name template (`{animal} {group} {treatment}
{dose} {route} {operator} {date} {time}`), container, FFmpeg `rtbufsize` (raise
it to `1024M` if frames drop), free-space warning threshold, and the two
behaviour switches: verify the frame rate after each recording, and push exposure
and gain into the driver before recording.

**Reports** — time-course bin (60 s gives the conventional twitches-per-minute
endpoint) and theme. The theme applies to exported figures as well as the app.

**Diagnostics** — **Show environment** prints every version, which is what to
paste into a bug report. **Open the log file** goes to
`data\logs\htr_studio.log`, the first place to look when something is wrong.

---

## Running a study end to end

**Once, when the rig is set up**

1. Mount the camera top-down, 25–35 cm above the arena floor, perpendicular.
2. 15–20 cm clear acrylic cylinder on a matte black surface.
3. Two DC constant-current white LED panels at 45°, 300–400 lux at the floor,
   diffuse, no hot spots. **Not PWM-dimmed.** (The B0634 has a 650 nm IR-cut
   filter, so 850 nm IR panels are invisible to it — visible white light is the
   way.)
4. Acquisition → **Probe all presets**. Confirm 320×240 delivers ~240 fps.
5. Preview, set exposure and gain, check all five tiles, lock the focus ring.
6. **Run flicker test.** Do not proceed until it says the light is steady.
7. Settings → Recordings root → the data drive. Save settings.

**Every session**

8. Flicker test again (someone may have changed a lamp).
9. Preview and confirm contrast, clipping and uniformity.

**Every animal**

10. Fill in the trial fields. Acclimatise the animal ≥ 5 min.
11. **Pre-flight check**, then **Start recording**.
12. Confirm the first few seconds look right.
13. Dose, and press **Mark T0** as you do.
14. Let it run. Watch for dropped frames.

**After the session**

15. Library → **Rescan** → **Verify frame rate** on each trial.
16. Analyse one trial by hand. Check the warnings, check tracking percentage,
    look at the figure. Satisfy yourself the settings suit this footage before
    committing to a batch.
17. Batch → add the session's trials → **Run batch**.
18. Review → curate at least the trials you intend to report. Save each.
19. Batch → **Rebuild summary from the queue** so the summary reflects the
    curated counts.
20. `study_summary.csv` and `study_rate_by_group.png` are the study output. For
    statistics, a linear mixed-effects model on the per-bin rates with animal as
    a random effect and treatment × time as fixed effects is the conventional
    analysis; `*_htr_rate.csv` is already in the right shape for it.

**After the first three to five animals, re-check the threshold.** The baseline
periods should score near zero. If they do not, raise the robust sigma multiplier
and re-run the batch — it is quick, because the recordings do not change. And
look at where the peak actually falls: the 70-minute window is an extrapolation
from intraperitoneal data, not a finding about oral dosing, so it may want
shortening or extending.

---

## Output files

### Beside each recording

| File | Contents |
|---|---|
| `<name>.avi` | the video |
| `<name>.avi.htr.json` | trial metadata, camera configuration, lighting metrics, the FFmpeg command line, what FFmpeg observed, the verified frame rate |
| `<name>.avi.htrresult.json` | the analysis: every burst with its measurements, every rejected candidate with its reason, the binned rates, the full parameter set, and whether a human curated it |
| `session.htrsession.json` | in a segmented trial's folder: the segment list and the shared metadata |

### In the export folder

| File | Contents |
|---|---|
| `<name>_htr_report.csv` | the human-facing report: a commented header (frames, rate, tracking quality, threshold and its basis, every warning), the total, the event table, the rejected candidates, and the per-bin rates |
| `<name>_htr_events.csv` | one table, one row per accepted event — for scripting |
| `<name>_htr_rate.csv` | one table, one row per time bin, with `bin_rel_t0` — for plotting and modelling |
| `<name>_htr.png` | the three-panel figure |
| `study_summary.csv` | one row per trial |
| `study_rate_by_group.png` | mean ± SEM rate time course by group |
| `<name>_traces.npz` | per-frame signals, if you asked for them |

Event table columns: `burst_id, start_s, end_s, center_s, center_min_rel_t0,
duration_ms, dominant_hz, half_cycles, peak_velocity_deg_s, peak_envelope,
invalid_frac, detector, accepted, manual, note`.

Rate table columns: `bin_index, start_s, end_s, bin_rel_t0, count, rate_per_min`.
`bin_rel_t0 = 0` is the first bin after dosing; baseline bins are negative.

---

## Validating the detector

Three levels, in increasing order of what they prove.

**1. The self-test** (`run_selftest.bat`). Renders a 20-second clip with three
twitches, one grooming bout and one fast turn. Expect perfect detection and zero
false positives. Run it after installing and after any change to the defaults.

**2. The synthetic benchmark** (Tools → Make a synthetic benchmark video). A
45-second clip with six twitches at jittered frequencies (78–98 Hz, 5–11
oscillations), three grooming bouts, two fast turns, and a 2-second stretch of
simulated PWM flicker. Analyse it, then **Validate against ground truth** with
the CSV it wrote. Current result with the defaults: precision 1.00, recall 1.00,
F1 1.00, nothing fired on any confounder.

**3. Against a human scorer.** This is the one that supports a claim in a methods
section. Pick two or three real recordings, have someone score them by eye from
the footage, and compare against the Review tab's curated output. Until you have
done this, the F1 = 1.00 figures above are statements about synthetic video and
should be described as such.

`Validate against ground truth` takes any CSV with `time_sec` and
`twitch_active` columns, so a hand-scored trial exported in that shape can be
used directly.

---

## Adding a pose model later

You probably do not need one. The shipped detector uses no trained model.

A pose model is worth it when you want keypoints for other behaviours — grooming,
rearing, locomotion — or if real footage defeats blob tracking: very low contrast,
loose bedding, two animals in frame. `install_dlc.bat` sets up DeepLabCut and
PyTorch in a **separate** environment so a dependency conflict there cannot stop
this app from opening.

The integration point is small, which was also the first pipeline's plan. Only
the **angle source** changes; everything after it — band-pass, Hilbert envelope,
hysteresis, merge, duration and frequency gates, binning, export — is unchanged.

In outline:

1. Record 8–12 representative trials. For labelling frames, 640×480 is fine and
   better: spatial detail matters for a pose model and the HTR band does not.
2. `deeplabcut.extract_frames` with k-means clustering, 200–400 frames per video.
3. Label at minimum **nose tip, left ear, right ear, neck**. Those four give the
   head axis; more keypoints only help other behaviours.
4. Train ResNet-50, ~50 000 iterations. On CPU that is 1.5–3 hours for 200–400
   frames; on an RTX 3060 or better, 10–15 minutes. Model files are portable, so
   train on a GPU machine and run inference here.
5. Run `deeplabcut.analyze_videos` to get per-frame keypoints. Inference at
   320×240 is around 10 fps on CPU, so a 60-minute trial is roughly a day —
   batch it overnight.
6. In `htr_studio/analysis.py`, add an angle source that reads the keypoint
   table and returns, per frame,
   `degrees(atan2(nose_y − neck_y, nose_x − neck_x))` plus a validity flag from
   the DLC likelihood. Everything downstream already handles NaN for invalid
   frames.

Augmentation note: train with motion blur and brightness jitter. A head moving at
90 Hz is blurred in every frame that matters, and a model trained only on sharp
frames will lose the animal exactly during a twitch.

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| `Ctrl+O` | open a video |
| `F5` | rescan the library |
| `Ctrl+S` | save settings |
| `Ctrl+1` … `Ctrl+6` | Acquisition, Library, Review, Analysis, Batch, Settings |
| `Ctrl+Q` | quit |

Dragging a video file onto `run_htr_studio.bat` opens it in Review.
