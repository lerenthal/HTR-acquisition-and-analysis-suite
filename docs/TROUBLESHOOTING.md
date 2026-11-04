# Troubleshooting

Symptom, cause, fix. The log is at `data\logs\htr_studio.log` and is the first
place to look; **Settings → Show environment** prints every version, which is
what to include if you need to ask someone.

- [Installation](#installation)
- [Starting the app](#starting-the-app)
- [The camera](#the-camera)
- [Recording](#recording)
- [Lighting and image quality](#lighting-and-image-quality)
- [Analysis](#analysis)
- [Results that look wrong](#results-that-look-wrong)
- [Files, library and performance](#files-library-and-performance)

---

## Installation

### `install.bat` says the folder is not writable

The installation folder has to be writable: the Python environment, settings and
logs all live inside it. Move `windows_HTR` out of `C:\Program Files` to your
user folder or a data drive, and run it again.

### "could not download uv" or "could not download FFmpeg"

No internet, or a proxy in the way.

* With a proxy, set it first in the same console:
  `set HTTPS_PROXY=http://user:pass@proxy:8080` then `install.bat`.
* Without internet, use the staged install in
  `INSTALLATION.md` → *Installing without internet access*. The short version:
  put `uv.exe` in `tools\bin\`, `ffmpeg.exe` and `ffprobe.exe` in
  `tools\ffmpeg\bin\`, and a pip wheelhouse in the folder.
* If `uv` is missing but the machine has Python 3.10–3.12, the installer falls
  back to the built-in `venv` module automatically — the uv failure is then
  harmless.

### "No usable Python and uv is not available either"

Nothing to build the environment with. Either get the machine online briefly and
re-run, or install Python 3.11 from python.org (tick *Add python.exe to PATH*)
and re-run.

### Package installation fails partway

Usually a dropped connection. Re-run `install.bat` — it reuses what succeeded. If
it keeps failing on one package, `install.bat /force` rebuilds the environment
cleanly.

### The self-test reports failures right after installing

Read which check failed; each line names the fix. The common ones:

| Check | Meaning |
|---|---|
| `import <something>` | the package did not install — re-run `install.bat` |
| `ffmpeg available: NOT FOUND` | recording and frame verification will not work; see the FFmpeg entry above |
| `GUI constructs` | PyQt5 did not install correctly, or the machine has no display session |
| `detect planted twitches` below F1 1.00 | a detector default has been changed — Analysis → **Reset to defaults** |

---

## Starting the app

### "HTR Studio is not installed in this folder yet"

`.venv` is missing. Run `install.bat`.

### Nothing happens when I double-click `run_htr_studio.bat`

It launches with `pythonw.exe`, which has no console, so a startup error is
invisible. Run it with the console instead, from a command prompt in the folder:

```
.venv\Scripts\python.exe -m htr_studio --debug
```

That prints the error, and `--debug` also turns up the logging.

### It opens and then closes immediately

Check `data\logs\htr_studio.log`. The usual causes are a corrupt
`config\settings.json` (delete it; defaults are restored) or a graphics driver
problem with Qt (try **Settings → theme → light**, or update the display driver).

---

## The camera

### Collect everything at once

Before working through the camera entries below, double-click

```
tools\camera_diagnostics.bat
```

It writes `data\logs\camera_diagnostics.txt` with the FFmpeg version, the **raw
unparsed** DirectShow device list, the Windows device database's view, the
desktop-app camera privacy setting, and what HTR Studio itself enumerates — then
opens it in Notepad. Nothing is changed and the camera is opened only
momentarily, but **close the Windows Camera app first**: it holds the device while
it is open.

Section [2] is the one that answers most questions. If the camera appears there
but not in the app, the parser is the problem and that output is what identifies
it; if it appears nowhere, the problem is below the application.

### The device list is empty but the camera is plugged in

If the status line says FFmpeg listed no devices **but OpenCV found cameras**,
the camera is attached and only the listing failed. FFmpeg changed the format of
`-list_devices` output at version 7.0 — it dropped the "DirectShow video
devices" headers and started appending `(video)` or `(audio)` to each line.
HTR Studio 1.0.0 handles both formats; if you are seeing this on an older copy,
update `htr_studio/ffmpeg_tools.py`.

Either way there is a way through: the device box is **editable**. Run

```
ffmpeg -list_devices true -f dshow -i dummy
```

in a terminal (it always exits with "Immediate exit requested" — that is normal,
the list is the output above it), and type the exact quoted name into the box.
Recording does not depend on the dropdown having found it.

### "No DirectShow video device found"

1. Confirm Windows itself sees it: open the Camera app, or Device Manager →
   *Imaging devices*.
2. Nothing else may hold the camera: close Teams, Zoom, OBS, the Camera app, and
   any other copy of HTR Studio.
3. Try a different USB port. **Avoid hubs** — the B0634 at 240 fps needs the
   bandwidth, and a shared hub is the most common cause of dropped frames.
4. Press **Refresh** again; enumeration takes a second or two per index.

### Several cameras appear and the wrong one is chosen

Pick it from the dropdown. The app matches on the configured name, falling back
to anything containing "arducam", "ezbox", "ov9281" or "global shutter". Once
selected, **File → Save settings** remembers it.

### "Could not open camera index N" for the preview

A recording is running (a DirectShow device admits one client at a time), or
another program has it. Stop the recording, or close the other program, then
press **Refresh**.

### The exposure slider has no effect, or the value reads back differently

Not all UVC drivers accept controls through OpenCV. Use **Camera properties…**,
which opens the driver's own DirectShow page: values set there are stored in the
driver and survive the handle being closed, so FFmpeg inherits them. The app
warns when a value did not read back as set.

### Probe reports "I/O error" or zero frames

The mode is not supported at that rate. Known from the reference host: 400 fps at
320×240 is rejected; 320×240 at a 200 fps request delivers 240 fps. Use **Probe
all presets** and work from what the camera actually delivered.

### The camera delivers a different rate than requested

Expected. The B0634's MJPEG encoder runs on a fixed clock, so it negotiates:
240 fps for a 200 fps request at 320×240, and about 100 fps for a 120 fps request
at VGA. Always work from the probed figure, and verify the frame rate of each
recording afterwards.

---

## Recording

### Dropped frames during a recording

Each lost frame is a hole in the timeline — and a hole in a 240 fps angle trace
is a step, which the band-pass can read as a twitch. So this is worth fixing
rather than tolerating.

Note what the **frames lost** tile counts. FFmpeg's own `drop=` reports only
frames the *muxer* discarded; frames thrown away earlier, in the capture ring
buffer, appear only on a log line reading *"real-time buffer … too full! frame
dropped!"* — which contains no word a generic error scan would match. Both are
now counted and summed in that tile, and recorded separately in the sidecar as
`dropped` and `rt_buffer_drops`. A capture can lose a third of its frames with
`drop=` at zero, so if the tile is non-zero, treat the trial as suspect.

In order:

1. **Record to a local SSD** and copy the files afterwards. Not a network share,
   not a USB drive. Run the pre-flight check and read the **Write path** line: it
   reports the worst single write pause, and anything over 250 ms is long enough
   to overflow the capture buffer. This is the most common cause by a wide
   margin.
2. Raise **Settings → FFmpeg thread_queue_size** to `8192`. FFmpeg's own default
   for this is **8 packets** — 33 ms at 240 fps, shorter than a single NTFS flush
   — which is why high-rate captures block; this build ships 4096 instead. If the
   log says *"Thread message queue blocking"*, this is the setting.
3. Raise **Settings → FFmpeg rtbufsize** to `1024M`.
4. Put the camera on its own USB controller — not a hub shared with other devices.
5. Close other programs, particularly anything doing video or backup.
6. Check the destination drive is not nearly full.

### Recording stops early

Check the recording status line, which shows what FFmpeg last reported, and then
the log. Common causes: the disk filled up, the camera was unplugged or lost
power, or the USB connection reset. Segmented recordings keep everything up to
the last complete segment, which is the reason segmenting is on by default.

### Verification says the frame rate is wrong

```
7209 frames in 30.01 s = 240.2 fps - matches the expected 240 fps     good
2403 frames in 30.01 s =  80.1 fps, but 240 fps was expected          bad
```

Frames were dropped, or the camera negotiated a different mode. Re-probe the mode
and work through the dropped-frames list above before recording more animals. A
recording at the wrong rate is not fixable afterwards: the band definition and
every duration depend on the true rate, and if that rate is below ~200 fps the
oscillation has aliased.

### I forgot to press Mark T0

Enter it by hand: **Library** → select the trial → **T0 offset** → the number of
seconds from the start of the recording to dosing → **Save metadata**. If you do
not know it even approximately, leave T0 unset: the detection threshold falls
back to the first 10% of the recording and the report says so, but the time course
will not be aligned with the other animals.

---

## Lighting and image quality

### The flicker test reports ripple in the HTR band

This is the single most damaging problem in the whole pipeline and it cannot be
fixed in software. A PWM-dimmed LED modulates the whole frame at a few hundred
hertz; under a sub-millisecond global-shutter exposure that becomes frame-to-frame
brightness stepping, which rings the 90 Hz band across the entire recording.

Fix the hardware: panels on **constant-current DC drive**, with brightness set by
distance or diffusion rather than by dimming. Then re-run the test and confirm
the ripple is under 0.5%.

### Contrast is too low

Blob tracking needs the mouse clearly separated from the floor:

* matte black surface under the cylinder — the single biggest improvement
* more light, or a longer exposure (less negative exposure value)
* clean the acrylic; scuffs scatter light and wash out the contrast
* check the panels are not reflecting off the cylinder wall into the lens

### Too many clipped pixels

Reduce the light or shorten the exposure (more negative). Over 1% is worth
fixing; blown highlights destroy the edges the tracker uses.

### Uniformity below 0.70

Corner-to-corner brightness ratio. Move the panels further out, add diffusion, or
raise them — a close panel makes a hot spot.

### Focus score is low

Refocus on the arena floor, then **lock the ring** with epoxy or hot glue. A
focus that drifts mid-study is not recoverable, and the drift is gradual enough
that nobody notices until the data is analysed.

---

## Analysis

### "tracking succeeded on only N% of frames"

Segmentation is losing the mouse. Fix it before believing any count — below about
80 % the angle trace is mostly interpolation.

1. **Analysis → Tune tracking**. It runs the same segmentation code the analysis
   does, on real frames, so you can see the failure instead of inferring it. Press
   **Scan 200 frames** for the two numbers that matter: the percentage of frames
   in which the animal is found, and the coefficient of variation of the blob
   area. Aim for ≥ 90 % and ≤ 0.35.
2. **Set *Segmentation* to `background`.** This is the fix for most real footage.
   Plain thresholding separates the animal by brightness alone, so bedding
   texture, a cylinder rim, a lighting gradient or a dark bench outside the arena
   all get grabbed along with the mouse. Background mode builds an empty-arena
   image from the median of sampled frames and subtracts it, so everything static
   cancels. Measured on a benchmark with bedding and a cast shadow: thresholding
   tracked 70.9 % and scored F1 = 0.667, background tracked 100 % and scored
   F1 = 1.000.
3. Switch **Show** in the tuning view to **binary mask**. If the mask includes the
   bedding, that confirms it; if the mask is right but the head arrow flips end to
   end between frames, the **tail-removal kernel** is wrong — it must be wider
   than the tail and narrower than the head.
4. If the polarity is being guessed wrongly, set *Mouse vs background* explicitly
   instead of `auto`.
5. If a reflection or a shadow is being tracked instead, raise *Blob area min*.
6. If the mouse leaves the frame, that is a framing problem for next time; the
   affected stretches are excluded from detection automatically.

Do **not** reach for *Region of interest* or *Mark edge-touching frames as
untracked* as a first move. Both are off by default on measured evidence: a
tight ROI clips the animal against the wall and took F1 from 1.000 to 0.500 on
the realistic benchmark, and edge rejection removed so many frames that genuine
twitches went with them.

### "at N fps the Nyquist limit is only M Hz, which cuts into the HTR band"

**Check whether the file is a slow-motion render before concluding anything.**
Play it: if the animal moves in slow motion, the frames are all there and only
the declared rate is wrong — see *Clear twitches on the video, nothing detected*
below, which is completely fixable.

If the animal moves at normal speed, the recording really was made in a mode that
cannot represent the twitch. Not recoverable in software — the oscillation
aliased at the sensor. Re-record at 320×240 / 240 fps.

A note saying the high cutoff was "trimmed to stay clear of the Nyquist limit" at
240 fps is routine housekeeping — the 78–98 Hz band is still fully covered.

### "The arena background is itself oscillating in the HTR band"

Lighting flicker; see above. Counts from this recording cannot be trusted.

### Analysis is very slow

Expect 300–500 frames per second per thread, so a 70-minute trial is 35–55
minutes single-threaded and 8–15 minutes with 4–6 threads.

* **Worker threads** set to `auto` on an 8-core machine should use 4.
* The parallel path needs frame-exact seeking, which MJPEG-in-AVI has. If the
  warning says seeking is not exact, the file has been re-encoded into a format
  with keyframes — analyse the original.
* Analyse from a local disk. A network share throttles the decode.
* Use **Frame limit** for a quick look at the start of a long trial.

### Analysis fails with "cannot open video"

The file is in use, moved, or truncated. Confirm it plays in a media player, and
check **Library → Verify frame rate** can read it.

### Nothing is detected at all

The run now ends with a **What to do next** line naming the most likely cause, so
read that first — it checks tracking, then the timebase, then the band, then the
gates, in that order. The reasoning behind the order, if you want to work through
it by hand:

1. **Tracking below 80 %** — nothing else matters until this is fixed. See above.
2. **The frame rate is wrong, so the band is wrong.** **Library → Verify frame
   rate** for an exact count. Then press **Signal diagnostics** on the Analysis
   tab: it plots the spectrum with the 78–98 Hz band and the Nyquist limit drawn
   on, which turns "is the signal reachable" from a judgement into something you
   can see.
3. **The threshold is too high** — lower the **robust sigma multiplier** from 6 to
   4 and re-run, watching the rejected list for the reason each candidate was
   refused.
4. **There really were no twitches**: wrong dose, too early, vehicle animal.
   Check a positive control.

### Clear twitches on the video, nothing detected

Almost always a **slow-motion render**, and it is completely fixable.

A supplementary video captured at 240 fps is routinely written at 30 fps so it
plays as slow motion. Every frame is still there; only the declared rate is
wrong. A 90 Hz twitch then sits at 11 Hz in the file's own timebase — plainly
visible to the eye, and completely invisible to a detector band-passing 78–98 Hz
on file timestamps.

1. Run the analysis once. Tracking will be fine and the count will be zero.
2. Press **Signal diagnostics**. If the strongest oscillation is at an integer
   fraction of 88 Hz and the band is empty, it says so and offers the implied
   capture rate.
3. Click **Use *N* fps and re-run**. It fills in the **Capture-rate override** and
   the next run uses that timebase.

Measured on a reconstructed 8× render of a 240 fps capture: 0 detections at the
file's declared 30 fps, 4 of 4 with F1 = 1.000 at the true 240 fps — same frames,
same settings, only the timebase changed.

**Confirm it against the footage.** If the animal moves at normal speed, the file
is not slow motion and the low peak is something else — grooming is 6–9 Hz. The
override is cleared when you select a different recording, and it is recorded in
the exported report so the timebase of every run stays auditable.

### A count that disagrees between the original file and a converted copy

Check what the copy is in: **Library → Inspect / archive**. If the codec is
H.264, HEVC, VP9 or MPEG-4, the copy is the problem. Inter-frame codecs describe
each frame in terms of its neighbours, which is precisely a mechanism for
removing small fast changes — motion estimation turns the head displacement into
a block vector and the quantiser discards the residual. A twitch can be smeared
or gone from a re-encode that looks perfect to the eye, because a person cannot
see 90 Hz.

Analyse the original capture. If you must convert, use the options under
**Archive / convert**, which are all-intra by construction.

---

## Results that look wrong

### Twitches detected during the baseline

The baseline should score near zero. If it does not:

1. Run the flicker test — in-band lighting ripple produces exactly this.
2. Check the tracking percentage; lost tracking produces steps, and a step is
   broadband.
3. Raise the **robust sigma multiplier**. It is stable from 4 to 8 on synthetic
   data, and real footage has a noisier floor, so 6–8 is often right.
4. Open the trial in Review and look at what was called. If the "twitches" are
   grooming or fast turns, tighten *Require in-burst frequency* and raise
   *Minimum half-cycles*.

### Counts are far higher than the literature

The literature range after a psychedelic in C57BL/6J mice is tens per 5–10
minutes, not hundreds per minute. A count that is an order of magnitude high is
almost always one of: lighting flicker, failing tracking, or a threshold set too
low. Work through the baseline list above.

### The same twitch is counted twice

Raise **Merge gaps shorter than** from 60 ms, or raise the **burst release
fraction** from 0.50 towards 0.7 so the hysteresis holds the burst together
longer. Then re-check against the Review tab's footage.

### Detections look right but the time course is not aligned across animals

T0 is missing or wrong on some trials. Check the **T0** column in the Batch queue
and the T0 offset in the Library for each one. The `bin_rel_t0` column is what
aligns the curves, and it is only meaningful when T0 is right.

### The curated counts are not in the summary

Re-run **Batch → Rebuild summary from the queue** after curating. The summary is
built from the stored results, and curation changes those.

---

## Files, library and performance

### The library is empty

Check **Library root** points at the folder with the recordings, and press
**Rescan**. Subfolders are included. Recognised extensions: `.avi`, `.mp4`,
`.mkv`, `.mov`, `.m4v`.

### Scanning is slow

Scanning reads container headers only, which is fast; **Verify frame rate**
decodes the whole file and is slow by nature. Verify the trials you intend to
analyse, not the whole library, and the result is cached in the sidecar so it is
paid for once.

### The fps column shows a `~`

That rate came from the container header and has not been verified. For
MJPEG-in-AVI the header is routinely wrong. Press **Verify frame rate**.

### A segmented session only analyses its first segment

By design, for the single-trial view. Add the session to the **Batch** tab to
process every segment. Keeping the segments separate also keeps the per-segment
timing honest.

### "Send2Trash is not installed, so nothing was deleted"

Optional package that did not install. Re-run `install.bat`, or delete the file
in Explorer. Nothing is ever hard-deleted by the app.

### Can I run two copies at once?

Analysis, yes. Acquisition, no — the camera admits one client, so the second copy
cannot open it. Two copies writing to the same recording's sidecar can also
overwrite each other's metadata edits.

### How do I move everything to another computer?

Copy the whole `windows_HTR` folder and run `run_selftest.bat` on the target. If
it reports import failures, `install.bat /force` rebuilds the environment.
Details in `INSTALLATION.md` → *Moving it to the analysis computer*.
