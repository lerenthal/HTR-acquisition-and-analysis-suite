"""Generate a *realistic* HTR clip: the things that break blob tracking.

``make_synthetic_benchmark.py`` renders a clean dark mouse on a clean light
floor. Real footage -- including the supplementary videos published with papers
-- is nothing like that, and a detector validated only against the clean case
will fail on the real one. This generator adds, deliberately, every nuisance
that matters:

* **Bedding texture** -- a static speckled floor with the same brightness range
  as the animal, which a global threshold cannot separate from it.
* **A cylinder wall** -- a bright arc and a dark rim near the frame edge, which
  is often the largest dark object in the frame.
* **Vignetting and a lighting gradient** -- one side brighter than the other, so
  a single global threshold is wrong somewhere in the frame by construction.
* **A cast shadow** that moves with the animal, roughly its own size.
* **A soft, low-contrast mouse** rather than a hard-edged blob.
* **Sensor noise and MJPEG-like blur.**

And two things about *timing* that catch people out:

* ``--slowmo N`` writes the clip at 1/N of the capture rate, which is how most
  published supplementary videos are rendered. A 240 fps capture written for
  30 fps playback puts a 90 Hz twitch at 11.25 Hz **in the file's own
  timebase**, so a detector band-passing 60-110 Hz on file timestamps finds
  nothing at all while the twitch is plainly visible to the eye.
* ``--fps`` lets you render a genuinely low-rate capture, where the oscillation
  is aliased and unrecoverable.

Usage::

    python tools/make_realistic_benchmark.py                    # 240 fps, hard
    python tools/make_realistic_benchmark.py --slowmo 8         # 8x slow motion
    python tools/make_realistic_benchmark.py --easy             # no nuisances
    python tools/make_realistic_benchmark.py --fps 30           # aliased

Writes ``<out>/realistic_htr.avi`` and ``<out>/realistic_htr_truth.csv``. The
truth file's ``time_sec`` column is in **real** time; when ``--slowmo`` is used
a ``file_time_sec`` column is added so you can line it up with the file.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from htr_studio import paths  # noqa: E402


def build_background(w: int, h: int, rng: np.random.Generator, hard: bool
                     ) -> np.ndarray:
    """A static arena: bedding speckle, cylinder rim, vignette, gradient."""
    bg = np.full((h, w), 188.0, dtype=np.float32)
    if not hard:
        return np.clip(bg, 0, 255).astype(np.uint8)

    # Bedding: coarse speckle, smoothed so it reads as texture rather than noise.
    speck = rng.normal(0, 26, size=(h, w)).astype(np.float32)
    speck = cv2.GaussianBlur(speck, (0, 0), 1.6)
    bg += speck
    # A few darker chips of bedding, the same tone as the animal.
    for _ in range(int(0.0012 * w * h)):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(bg, (x, y), int(rng.integers(1, 4)), float(rng.integers(70, 120)),
                   -1, cv2.LINE_AA)

    # Lighting gradient: one panel closer than the other.
    gx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    gy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    bg += 26.0 * gx + 8.0 * gy

    # Vignette.
    r = np.sqrt((gx * 1.0) ** 2 + (gy * 1.0) ** 2)
    bg *= (1.0 - 0.22 * np.clip(r, 0, 1.4) ** 2)

    # Cylinder: a bright wall highlight and a dark rim just inside the frame.
    cx, cy = w / 2, h / 2
    rad = int(min(w, h) * 0.46)
    cv2.circle(bg, (int(cx), int(cy)), rad, 232.0, 7, cv2.LINE_AA)   # highlight
    cv2.circle(bg, (int(cx), int(cy)), rad + 6, 52.0, 5, cv2.LINE_AA)  # dark rim
    # Outside the cylinder: dark bench. This is the single largest dark region
    # in the frame, and a naive threshold picks it instead of the mouse.
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), rad + 4, 255, -1)
    bg[mask == 0] = 42.0

    return np.clip(bg, 0, 255).astype(np.uint8)


def draw_mouse(canvas: np.ndarray, cx: float, cy: float, heading: float,
               head_rot: float, scale: float, hard: bool,
               rear: float = 0.0) -> None:
    """A soft, low-contrast animal with a cast shadow.

    ``rear`` is 0 for a mouse flat on the floor and 1 for one stretched up
    against the arena wall. Seen from above, rearing foreshortens the animal
    along its own axis and widens it slightly, so the body goes from a clear
    ellipse to something close to a disc. That matters more than it sounds:
    the long axis of a disc is noise, so the body-axis estimate -- and with it
    the head direction -- becomes meaningless for as long as the animal is up.
    A detector validated only on a flat, elongated mouse never meets this case.
    """
    ux, uy = math.cos(math.radians(heading)), math.sin(math.radians(heading))
    squash = 1.0 - 0.55 * rear          # foreshortening along the body axis
    body_len, body_wid = 34 * scale * squash, 15 * scale * (1.0 + 0.30 * rear)

    layer = np.zeros(canvas.shape, np.float32)
    alpha = np.zeros(canvas.shape, np.float32)

    if hard:
        # Cast shadow, offset from the animal, almost as large as it is.
        sx, sy = cx + 7 * scale, cy + 9 * scale
        cv2.ellipse(alpha, (int(sx), int(sy)),
                    (int(body_len * 1.05), int(body_wid * 1.15)), heading,
                    0, 360, 0.45, -1, cv2.LINE_AA)
        cv2.ellipse(layer, (int(sx), int(sy)),
                    (int(body_len * 1.05), int(body_wid * 1.15)), heading,
                    0, 360, 96.0, -1, cv2.LINE_AA)

    def blob(center, axes, ang, tone, a=1.0):
        cv2.ellipse(alpha, center, axes, ang, 0, 360, a, -1, cv2.LINE_AA)
        cv2.ellipse(layer, center, axes, ang, 0, 360, tone, -1, cv2.LINE_AA)

    body_tone = 104.0 if hard else 55.0   # low contrast against 188 bedding
    blob((int(cx), int(cy)), (int(body_len), int(body_wid)), heading, body_tone)

    # Tail
    tail_len = 58 * scale * (1.0 - 0.6 * rear)
    cv2.line(alpha, (int(cx - ux * body_len * 0.9), int(cy - uy * body_len * 0.9)),
             (int(cx - ux * (body_len + tail_len)),
              int(cy - uy * (body_len + tail_len))), 1.0,
             max(1, int(round(2 * scale))), cv2.LINE_AA)
    cv2.line(layer, (int(cx - ux * body_len * 0.9), int(cy - uy * body_len * 0.9)),
             (int(cx - ux * (body_len + tail_len)),
              int(cy - uy * (body_len + tail_len))), body_tone + 8,
             max(1, int(round(2 * scale))), cv2.LINE_AA)

    # Head, carrying the twitch rotation
    neck_x, neck_y = cx + ux * body_len * 0.88, cy + uy * body_len * 0.88
    ha = heading + head_rot
    hax, hay = math.cos(math.radians(ha)), math.sin(math.radians(ha))
    hl, hw = 15 * scale, 10 * scale
    hcx, hcy = neck_x + hax * hl * 0.8, neck_y + hay * hl * 0.8
    blob((int(hcx), int(hcy)), (int(hl), int(hw)), ha, body_tone - 6)
    # Snout, so the leading end is narrower
    cv2.circle(alpha, (int(hcx + hax * hl * 0.75), int(hcy + hay * hl * 0.75)),
               max(1, int(round(4 * scale))), 1.0, -1, cv2.LINE_AA)
    cv2.circle(layer, (int(hcx + hax * hl * 0.75), int(hcy + hay * hl * 0.75)),
               max(1, int(round(4 * scale))), body_tone - 10, -1, cv2.LINE_AA)
    # Ears
    epx, epy = -hay, hax
    for side in (-1, 1):
        p = (int(hcx + side * epx * hw * 0.95), int(hcy + side * epy * hw * 0.95))
        cv2.circle(alpha, p, max(1, int(round(3 * scale))), 1.0, -1, cv2.LINE_AA)
        cv2.circle(layer, p, max(1, int(round(3 * scale))), body_tone - 14, -1,
                   cv2.LINE_AA)

    if hard:
        # Soft edges: a real animal is fur, not a vector shape.
        alpha = cv2.GaussianBlur(alpha, (0, 0), 1.3)
        layer = cv2.GaussianBlur(layer, (0, 0), 1.3)
    np.clip(alpha, 0, 1, out=alpha)
    canvas[:] = canvas * (1.0 - alpha) + layer * alpha


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="")
    ap.add_argument("--fps", type=float, default=240.0,
                    help="capture frame rate (the real one)")
    ap.add_argument("--slowmo", type=float, default=1.0,
                    help="write the file at capture_fps/N, as a slow-motion render")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds of real time")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--easy", action="store_true",
                    help="no bedding, shadow, rim or gradient")
    ap.add_argument("--simple-motion", action="store_true",
                    help="the v1.1 motion model: a nearly fixed heading and no "
                         "rearing. Kept so earlier results reproduce exactly.")
    ap.add_argument("--amp-min", type=float, default=6.0,
                    help="smallest twitch amplitude in degrees of head rotation")
    ap.add_argument("--amp-max", type=float, default=40.0,
                    help="largest twitch amplitude in degrees")
    ap.add_argument("--amp-sweep", action="store_true", default=True,
                    help="sweep amplitude linearly from --amp-min to --amp-max "
                         "across the recording, so recall can be plotted "
                         "against amplitude and a detection floor measured")
    ap.add_argument("--no-amp-sweep", dest="amp_sweep", action="store_false")
    args = ap.parse_args(argv)

    hard = not args.easy
    out_dir = Path(args.out) if args.out else (paths.DATA_DIR / "benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "realistic_htr.avi"
    truth_path = out_dir / "realistic_htr_truth.csv"

    rng = np.random.default_rng(args.seed)
    n_frames = int(round(args.fps * args.duration))
    scale = min(args.width / 640.0, args.height / 480.0) * 1.4
    file_fps = args.fps / max(args.slowmo, 1e-9)

    bg = build_background(args.width, args.height, rng, hard)

    # HTR bursts: 5-11 oscillations at 78-98 Hz, in REAL time.
    #
    # Amplitude is swept from faint to obvious rather than drawn from a narrow
    # band of large values. The previous 26-44 deg range made every planted
    # twitch an easy one, so a perfect score on it said nothing about the
    # events that actually get missed. With a sweep, the ground-truth file
    # carries each event's amplitude and recall can be plotted against it --
    # which turns "did it find them" into "how small an event can it find",
    # the number that matters when the question is whether a real, subtler
    # twitch would be caught.
    bursts = []
    t = 2.5
    schedule = []
    while t < args.duration - 2.0:
        schedule.append(t)
        t += float(rng.uniform(2.2, 4.0))
    for k, start in enumerate(schedule):
        freq = float(rng.uniform(78, 98))
        cycles = int(rng.integers(5, 12))
        if args.amp_sweep and len(schedule) > 1:
            amp = args.amp_min + (args.amp_max - args.amp_min) * k / (len(schedule) - 1)
        else:
            amp = float(rng.uniform(args.amp_min, args.amp_max))
        bursts.append((start, cycles / freq, freq, float(amp)))

    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"),
                             file_fps, (args.width, args.height), isColor=False)
    if not writer.isOpened():
        print(f"ERROR: could not open a writer at {video_path}", file=sys.stderr)
        return 1

    cx, cy = args.width / 2, args.height / 2
    heading, vx, vy, omega = 35.0, 0.0, 0.0, 0.0
    margin = 70 * scale
    rows = []

    # ---- behaviour model -------------------------------------------------
    # A mouse in an arena is not a blob drifting at a fixed heading. It walks,
    # stops, turns through large angles, grooms in place, and rears against the
    # wall. Each of those breaks a different assumption in the tracker, and the
    # reared state breaks the most important one: a foreshortened animal has no
    # reliable long axis, so the body-axis estimate collapses exactly when the
    # animal is most likely to be twitching. Planting twitches in every state
    # is what makes the benchmark able to show that.
    STATES = ("rest", "walk", "turn", "rear", "groom")
    DWELL = {"rest": (0.6, 2.0), "walk": (0.8, 2.5), "turn": (0.3, 1.1),
             "rear": (0.7, 2.2), "groom": (0.8, 2.4)}
    NEXT = {
        "rest":  ("walk", "turn", "groom", "rear"),
        "walk":  ("turn", "rest", "rear", "walk"),
        "turn":  ("walk", "rest", "walk", "groom"),
        "rear":  ("rest", "walk", "turn", "rest"),
        "groom": ("rest", "walk", "turn", "rest"),
    }
    state = "walk"
    state_until = float(rng.uniform(*DWELL[state]))
    rear_now = 0.0           # smoothed 0..1 so the posture change is gradual
    turn_rate = 0.0
    state_log: list[str] = []

    for i in range(n_frames):
        tr = i / args.fps                      # real time

        if args.simple_motion:
            vx += rng.normal(0, 0.30 * scale)
            vy += rng.normal(0, 0.30 * scale)
            omega += rng.normal(0, 0.22)
            vx, vy, omega = vx * 0.945, vy * 0.945, omega * 0.90
            heading += omega * 0.06
            rear_target = 0.0
            state = "walk"
        else:
            if tr >= state_until:
                state = str(rng.choice(NEXT[state]))
                state_until = tr + float(rng.uniform(*DWELL[state]))
                if state == "turn":
                    # A real turn is tens to hundreds of degrees, not a drift.
                    turn_rate = float(rng.choice([-1, 1])) * float(
                        rng.uniform(90, 320))          # deg/s
            if state == "walk":
                speed = 2.6 * scale
                vx += (math.cos(math.radians(heading)) * speed - vx) * 0.06
                vy += (math.sin(math.radians(heading)) * speed - vy) * 0.06
                heading += rng.normal(0, 18.0) / args.fps
                rear_target = 0.0
            elif state == "turn":
                vx *= 0.90; vy *= 0.90
                heading += turn_rate / args.fps
                rear_target = 0.0
            elif state == "rear":
                # Rearing happens at the wall: drift outward, then hold.
                ang = math.atan2(cy - args.height / 2, cx - args.width / 2)
                vx += (math.cos(ang) * 1.1 * scale - vx) * 0.05
                vy += (math.sin(ang) * 1.1 * scale - vy) * 0.05
                heading += rng.normal(0, 35.0) / args.fps
                rear_target = 1.0
            elif state == "groom":
                vx *= 0.86; vy *= 0.86
                heading += rng.normal(0, 55.0) / args.fps
                rear_target = 0.25
            else:                                   # rest
                vx *= 0.88; vy *= 0.88
                heading += rng.normal(0, 8.0) / args.fps
                rear_target = 0.0

        rear_now += (rear_target - rear_now) * (3.0 / args.fps)
        cx = float(np.clip(cx + vx / args.fps * 30.0, margin, args.width - margin))
        cy = float(np.clip(cy + vy / args.fps * 30.0, margin, args.height - margin))
        heading %= 360.0

        rot = 0.0
        active = False
        amp_now = 0.0
        for bs, d, f, amp in bursts:
            if bs <= tr < bs + d:
                taper = math.sin(math.pi * (tr - bs) / d) ** 0.5
                rot += amp * taper * math.sin(2 * math.pi * f * (tr - bs))
                active = True
                amp_now = amp

        frame = bg.astype(np.float32).copy()
        draw_mouse(frame, cx, cy, heading, rot, scale, hard, rear=rear_now)
        if hard:
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
        frame += rng.normal(0, 3.2 if hard else 2.0, size=frame.shape)
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))

        state_log.append(state)
        rows.append((i, f"{tr:.6f}", f"{i / file_fps:.6f}",
                     1 if active else 0, f"{rot:.3f}",
                     f"{amp_now:.1f}", state, f"{rear_now:.3f}",
                     f"{heading:.2f}"))

    writer.release()

    with open(truth_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["frame_index", "time_sec", "file_time_sec", "twitch_active",
                     "head_rotation_deg", "twitch_amplitude_deg", "behaviour",
                     "rear", "body_heading_deg"])
        wr.writerows(rows)

    print(f"wrote {video_path}")
    print(f"  {n_frames} frames, {args.width}x{args.height}, "
          f"{video_path.stat().st_size / 1e6:.1f} MB")
    print(f"  capture rate {args.fps:g} fps, written as {file_fps:g} fps"
          + (f"  ({args.slowmo:g}x SLOW MOTION)" if args.slowmo != 1 else ""))
    print(f"  nuisances: {'ON (bedding, rim, shadow, gradient, soft edges)' if hard else 'off'}")
    print(f"wrote {truth_path}")
    if not args.simple_motion:
        from collections import Counter
        c = Counter(state_log)
        tot = max(len(state_log), 1)
        share = ", ".join(f"{k} {100*v/tot:.0f}%" for k, v in c.most_common())
        print(f"  behaviour: {share}")
        reared = sum(1 for st, r in zip(state_log, (float(r[7]) for r in rows))
                     if r > 0.5)
        print(f"  reared (foreshortened) in {100*reared/tot:.0f}% of frames - "
              "the body axis is unreliable in those")
    print(f"\n{len(bursts)} HTR burst(s), in real time:")
    at_state = {bs: state_log[min(int(bs * args.fps), len(state_log) - 1)]
                for bs, _d, _f, _a in bursts}
    for bs, d, f, amp in bursts:
        print(f"   t={bs:6.2f}s  {d * 1000:5.0f} ms  {f:.1f} Hz  amp {amp:5.1f} deg"
              f"  during '{at_state[bs]}'"
              + (f"   -> file t={bs / max(args.slowmo, 1e-9):6.2f}s,"
                 f" apparent {f / max(args.slowmo, 1e-9):.1f} Hz"
                 if args.slowmo != 1 else ""))
    if args.amp_sweep and len(bursts) > 1:
        print(f"\n  Amplitude is swept {bursts[0][3]:.0f} -> {bursts[-1][3]:.0f} deg "
              "across the recording. The ground-truth CSV carries each event's")
        print("  amplitude, so recall can be plotted against it and the "
              "detection floor measured rather than assumed.")
    if args.slowmo != 1:
        print(f"\n  NOTE: in this file's own timebase the twitch sits at "
              f"{bursts[0][2] / args.slowmo:.1f} Hz, not {bursts[0][2]:.0f} Hz.")
        print(f"  Analyse it with the capture rate set to {args.fps:g} fps, not "
              f"the container's {file_fps:g}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
