"""Generate a synthetic HTR benchmark video with ground truth and confounders.

Why not just reuse ``benchmark_videos/synthetic_mouse_htr.mp4``? That file is
still useful as a regression fixture, but it has two properties that make it a
poor yardstick:

* Its sensor noise is reseeded from ``int(t * 100)``, so the noise pattern is
  held for ~2.4 frames at 240 fps and then changes. That is a whole-field
  flicker artefact at about 100 Hz -- inside the HTR band. HTR Studio's
  background-flicker check fires on it, correctly.
* It contains only true positives. A detector can score well on it and still
  fire on every grooming bout and every fast turn, which is what actually
  limits a real HTR count.

This generator fixes both: independent per-frame noise, realistic proportions
(long thin tail, small ears, so the tail is what the morphological opening
removes), and three classes of confounder that a naive band-pass detector gets
wrong:

====================  ===========================================
Event                 What it tests
====================  ===========================================
``htr``               True positives: 78-98 Hz, 5-11 oscillations
``groom``             A 6-9 Hz head movement - below the band
``dart``              A fast whole-body turn - broadband, no 90 Hz
``flicker``           Whole-field PWM lighting at ~90 Hz
====================  ===========================================

Usage::

    python tools/make_synthetic_benchmark.py                 # defaults
    python tools/make_synthetic_benchmark.py --no-confounders
    python tools/make_synthetic_benchmark.py --duration 60 --fps 240 --out mydir

Outputs ``<out>/synthetic_htr_benchmark.avi`` (MJPEG, like the real camera) and
``<out>/synthetic_htr_benchmark_truth.csv`` with one row per frame:
``frame_index, time_sec, twitch_active, event_type, head_rotation_deg``.
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


def build_events(duration: float, rng: np.random.Generator, confounders: bool
                 ) -> list[dict]:
    """Lay out non-overlapping events across the recording."""
    events: list[dict] = []
    t = 3.0
    kinds = ["htr", "groom", "dart", "htr", "flicker", "htr", "groom", "htr", "dart", "htr"]
    if not confounders:
        kinds = ["htr"] * 10
    i = 0
    while t < duration - 3.0:
        kind = kinds[i % len(kinds)]
        if kind == "htr":
            # Halberstadt 2014: 5-11 oscillations at 78-98 Hz.
            freq = float(rng.uniform(78, 98))
            cycles = int(rng.integers(5, 12))
            dur = cycles / freq
            events.append({"kind": "htr", "start": t, "dur": dur, "freq": freq,
                           "amp": float(rng.uniform(25, 45))})
        elif kind == "groom":
            freq = float(rng.uniform(6, 9))
            dur = float(rng.uniform(1.2, 2.5))
            events.append({"kind": "groom", "start": t, "dur": dur, "freq": freq,
                           "amp": float(rng.uniform(15, 30))})
        elif kind == "dart":
            dur = float(rng.uniform(0.25, 0.5))
            events.append({"kind": "dart", "start": t, "dur": dur, "freq": 0.0,
                           "amp": float(rng.uniform(70, 130))})
        else:  # flicker
            dur = 2.0
            events.append({"kind": "flicker", "start": t, "dur": dur,
                           "freq": float(rng.uniform(85, 95)), "amp": 18.0})
        t += dur + float(rng.uniform(1.8, 3.5))
        i += 1
    return events


def render(w: int, h: int, cx: float, cy: float, heading_deg: float,
           head_rot_deg: float, scale: float, rng: np.random.Generator,
           brightness_offset: float) -> np.ndarray:
    """Top-down mouse: elongated body, small head, long thin tail, two ears."""
    img = np.full((h, w), 215, dtype=np.uint8)

    hd = math.radians(heading_deg)
    ux, uy = math.cos(hd), math.sin(hd)
    px, py = -uy, ux

    body_len = 34 * scale
    body_wid = 15 * scale
    # Body: ellipse along the heading.
    cv2.ellipse(img, (int(cx), int(cy)),
                (int(body_len), int(body_wid)), heading_deg, 0, 360, 55, -1, cv2.LINE_AA)

    # Tail: long, thin, trailing. This is what the opening in the tracker
    # removes, so it has to be genuinely thin relative to the body.
    tail_len = 60 * scale
    cv2.line(img, (int(cx - ux * body_len * 0.9), int(cy - uy * body_len * 0.9)),
             (int(cx - ux * (body_len + tail_len)), int(cy - uy * (body_len + tail_len))),
             60, max(1, int(round(2 * scale))), cv2.LINE_AA)

    # Head: smaller ellipse at the front, carrying the twitch rotation.
    neck_x = cx + ux * body_len * 0.88
    neck_y = cy + uy * body_len * 0.88
    head_ang = heading_deg + head_rot_deg
    hax, hay = math.cos(math.radians(head_ang)), math.sin(math.radians(head_ang))
    head_len = 15 * scale
    head_wid = 10 * scale
    hcx = neck_x + hax * head_len * 0.8
    hcy = neck_y + hay * head_len * 0.8
    cv2.ellipse(img, (int(hcx), int(hcy)), (int(head_len), int(head_wid)),
                head_ang, 0, 360, 48, -1, cv2.LINE_AA)
    # Ears: small, so they do not dominate what the opening strips away.
    ear_off = head_wid * 0.95
    epx, epy = -hay, hax
    for side in (-1, 1):
        cv2.circle(img, (int(hcx + side * epx * ear_off), int(hcy + side * epy * ear_off)),
                   max(1, int(round(3 * scale))), 40, -1, cv2.LINE_AA)
    # Snout tip, so the leading end is narrower than the rump.
    cv2.circle(img, (int(hcx + hax * head_len * 0.75), int(hcy + hay * head_len * 0.75)),
               max(1, int(round(4 * scale))), 45, -1, cv2.LINE_AA)

    out = img.astype(np.int16)
    # Independent noise every frame: no spurious periodicity.
    out += rng.integers(-6, 7, size=out.shape, dtype=np.int16)
    if brightness_offset:
        out += int(round(brightness_offset))
    return np.clip(out, 0, 255).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="", help="output directory (default: data/benchmark)")
    ap.add_argument("--fps", type=float, default=240.0)
    ap.add_argument("--duration", type=float, default=45.0, help="seconds")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-confounders", action="store_true",
                    help="true positives only, like the first-generation benchmark")
    args = ap.parse_args(argv)

    out_dir = Path(args.out) if args.out else (paths.DATA_DIR / "benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "synthetic_htr_benchmark.avi"
    truth_path = out_dir / "synthetic_htr_benchmark_truth.csv"

    rng = np.random.default_rng(args.seed)
    n_frames = int(round(args.fps * args.duration))
    scale = min(args.width / 320.0, args.height / 240.0)
    events = build_events(args.duration, rng, not args.no_confounders)

    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"),
                             args.fps, (args.width, args.height), isColor=False)
    if not writer.isOpened():
        print(f"ERROR: could not open a writer at {video_path}", file=sys.stderr)
        return 1

    cx, cy = args.width / 2, args.height / 2
    heading = 40.0
    vx = vy = 0.0
    omega = 0.0
    margin = 45 * scale
    rows: list[tuple] = []

    for i in range(n_frames):
        t = i / args.fps
        active = [e for e in events if e["start"] <= t < e["start"] + e["dur"]]
        kind = active[0]["kind"] if active else "none"

        head_rot = 0.0
        bright = 0.0
        dart = False
        for e in active:
            dt = t - e["start"]
            if e["kind"] == "htr":
                # Taper the ends so the burst grows and fades, as a real one does.
                taper = math.sin(math.pi * min(max(dt / e["dur"], 0.0), 1.0)) ** 0.5
                head_rot += e["amp"] * taper * math.sin(2 * math.pi * e["freq"] * dt)
            elif e["kind"] == "groom":
                head_rot += e["amp"] * math.sin(2 * math.pi * e["freq"] * dt)
            elif e["kind"] == "dart":
                dart = True
            elif e["kind"] == "flicker":
                bright += e["amp"] * math.sin(2 * math.pi * e["freq"] * dt)

        # Locomotion: damped random walk, with a hard turn during a dart.
        vx += rng.normal(0, 0.35 * scale)
        vy += rng.normal(0, 0.35 * scale)
        omega += rng.normal(0, 0.25)
        if dart:
            omega += active[0]["amp"] * 0.08 * (1 if int(t * 3) % 2 else -1)
            vx += math.cos(math.radians(heading)) * 1.2 * scale
            vy += math.sin(math.radians(heading)) * 1.2 * scale
        vx *= 0.94
        vy *= 0.94
        omega *= 0.90
        cx = float(np.clip(cx + vx, margin, args.width - margin))
        cy = float(np.clip(cy + vy, margin, args.height - margin))
        heading += omega * 0.06

        frame = render(args.width, args.height, cx, cy, heading, head_rot,
                       scale, rng, bright)
        writer.write(frame)
        rows.append((i, f"{t:.5f}", 1 if kind == "htr" else 0, kind,
                     f"{head_rot:.3f}"))

    writer.release()

    with open(truth_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["frame_index", "time_sec", "twitch_active", "event_type",
                     "head_rotation_deg"])
        wr.writerows(rows)

    by_kind: dict[str, int] = {}
    for e in events:
        by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
    print(f"wrote {video_path}  ({n_frames} frames, {args.fps:g} fps, "
          f"{args.width}x{args.height}, {video_path.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {truth_path}")
    print("events:")
    for e in events:
        extra = f" {e['freq']:.1f} Hz" if e["freq"] else ""
        print(f"  {e['kind']:>8s}  t={e['start']:6.2f}s  dur={e['dur']*1000:6.0f} ms{extra}")
    print(f"summary: {by_kind}")
    print(f"\nTrue HTR events (the only rows with twitch_active=1): "
          f"{by_kind.get('htr', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
