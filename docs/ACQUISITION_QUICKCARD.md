# HTR acquisition — quick card

One page. Print it and keep it at the rig.

---

## Rig geometry

```
        [LED panel]                      [LED panel]
           45°  \                        /  45°
                 \                      /
                  \                    /
                   +--------+
                   | B0634  |   top-down, perpendicular
                   +--------+
                       |  25-35 cm
                       v
                  +-----------+
                  |   MOUSE   |   15-20 cm clear acrylic cylinder
                  +-----------+
                   ###########     matte black floor
```

- **Camera:** 90° top-down, 25–35 cm above the arena floor, focus ring **locked**
  (epoxy or hot glue — a focus that drifts mid-study cannot be recovered).
- **Lighting:** two **DC constant-current** white LED panels at 45°, 5000–6500 K,
  300–400 lux at the floor, diffuse, no hot spots.
  **Not PWM-dimmed.** Set brightness with distance or diffusion.
- **No infrared.** The B0634 has a 650 nm IR-cut filter, so 850 nm panels give a
  black frame.
- **Background:** matte black under the cylinder, for contrast.

---

## Capture mode

| | |
|---|---|
| Resolution | **320 × 240** |
| Pixel format | **MJPEG** — never uncompressed. See the disk table below. |
| Requested rate | **200 fps** |
| Delivered rate | **240 fps** (the camera negotiates — probe to confirm) |
| Auto-exposure | **OFF** |
| Exposure | −7 to −10 (≈ 8 ms down to 1 ms permitted) |
| Gain | **0** |

**Not 640×480.** VGA delivers ~100 fps, Nyquist 50 Hz, and the 78–98 Hz twitch
**aliases**. Use VGA only for collecting frames to label for a pose model.

---

## Before the first animal of the session

- [ ] **Flicker test** (Acquisition → Run flicker test). Must report steady light.
      Ripple at 55–125 Hz means false positives across the whole recording — stop
      and fix the lighting.
- [ ] Preview tiles:
      - contrast **> 70** / 255
      - clipped **< 1 %**
      - focus **> 40**
      - uniformity **> 0.70**
- [ ] Arena wiped clean.
- [ ] Free disk ≥ **5 GB per trial** (a 70-min trial is about 2.5 GB).
- [ ] **Recording to a local SSD**, never a network share or a USB drive. The
      pre-flight **Write path** line reports the worst single write pause; over
      250 ms is long enough to overflow the capture buffer and lose frames.

---

## Every animal

1. Fill in: animal ID, group, treatment, dose, route, operator.
2. Acclimatise the animal in the arena **≥ 5 min**.
3. **Pre-flight check** — must show no blocking problems.
4. **Start recording.**
5. Glance at the first 5 s: mouse in frame, exposure right.
6. Dose, and press **Mark T0 (dose)** as you do.
7. Watch the **frames lost** tile and the status line. Any non-zero value is a
   hole in the timeline, and a hole in a 240 fps trace is a step the detector can
   read as a twitch. The tile counts buffer overflows as well as muxer drops —
   FFmpeg's own `drop=` misses the first kind entirely. If it moves: record to a
   local SSD, close other programs, raise `thread_queue_size` to 8192 and
   `rtbufsize` to `1024M` in Settings.

Default timing: **10 min baseline + 60 min post-dose = 70 min**, written as
fourteen 5-minute segments.

---

## Straight after the session

- [ ] Library → **Rescan**.
- [ ] **Verify frame rate** on each trial. Expect ≈ 240 fps. Anything else means
      frames were dropped or the camera negotiated a different mode.
- [ ] Analyse one trial by hand and read the warnings before batching the rest.

---

## Three things that ruin a trial, and what they look like

| Symptom | Cause | Fix |
|---|---|---|
| Twitches "detected" throughout, including baseline | PWM lighting flicker in the HTR band | DC constant-current panels. Re-run the flicker test. Not recoverable in software. |
| Tracking valid below ~75 % | too little contrast; mouse leaving frame | matte black floor, more light, longer exposure, check framing |
| Verified frame rate well below 240 | lost frames, or a renegotiated mode | record to a local SSD, raise `thread_queue_size` and `rtbufsize`, close other programs, re-probe the mode, use a different USB port — avoid hubs |

---

## Numbers worth remembering

| | |
|---|---|
| HTR frequency | **78–98 Hz**, 5–11 oscillations, so 50–150 ms per burst |
| Nyquist needed | > 196 Hz sampling; 240 fps gives margin |
| At 100 fps | Nyquist 50 Hz — the twitch aliases, unrecoverably |
| Disk, MJPEG | **1.06 GB** per 30 min, **2.5 GB** per 70 min (2456 bytes/frame measured) |
| Disk, uncompressed YUYV | **66 GB** per 30 min — 27× the size, for identical frames. This is why a capture "turns out huge". |
| Write rate needed | 0.59 MB/s at 320×240 / 240 fps MJPEG; 37 MB/s uncompressed |
| `thread_queue_size` | ships at **4096**. FFmpeg's own default is **8** — 33 ms at 240 fps, and the usual cause of "Thread message queue blocking" plus silent frame loss. |
| Analysis time | 8–15 min per trial on 8 cores; 35–55 min single-threaded |
| Expected baseline rate | near zero. If not, the threshold or the lighting is wrong. |
