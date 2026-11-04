"""HTR signal extraction and burst detection.

Design notes, and how this differs from the first-generation
``scripts/htr_analysis.py``
---------------------------------------------------------------
1. **Streaming, not RAM-bound.** The original read every frame into a Python
   list of grayscale arrays and capped itself at 12 000 frames -- 50 s at
   240 fps. A 70-minute trial is ~1 008 000 frames, about 77 GB as raw
   grayscale. Here each frame is decoded, reduced to a handful of scalars, and
   discarded; only the derived 1-D signals are kept (~40 MB for a full trial).

2. **Bursts are merged before the duration gate.** The old detector thresholded
   a moving-average envelope and treated every excursion as a separate event,
   so one real twitch whose envelope dipped for a few milliseconds was counted
   twice. That is the "2 over-splits" that held the published benchmark to
   F1 = 0.86. Sub-threshold gaps shorter than ``merge_gap_ms`` are now closed
   before anything is counted.

3. **The threshold comes from the baseline window.** The old adaptive threshold
   was ``median + 3*MAD`` over the *whole* recording. In a drug trial the drug
   effect is most of the recording, so the signal inflated its own threshold and
   the effect partly hid itself. The protocol already records 10 minutes of
   pre-gavage baseline; that is what the threshold is derived from now.

4. **The pass band matches the phenomenon.** Halberstadt et al. (2014,
   PMC3866102) measured 5-11 oscillations per burst at 78-98 Hz. The old
   20-100 Hz band also passed locomotion and grooming harmonics. Default is now
   60-110 Hz, with a hard Nyquist guard and an in-burst frequency gate.

5. **Lost tracking is NaN, not the previous value.** The old code held the last
   angle when the blob was missed, which injects a flat segment and then a step
   -- a step is broadband, so it rings the band-pass and looks like a twitch.
   Invalid frames are now marked; short gaps are interpolated, long gaps stay
   invalid, and a candidate burst sitting on lost tracking is rejected.

6. **A second, independent detector.** ``motion_spectral`` works on frame-to-
   frame motion energy inside the mouse, not on a fitted angle, so it does not
   depend on segmentation succeeding every frame. Running both and requiring
   agreement is the most conservative setting available here.

Everything in this module is pure NumPy/OpenCV/SciPy and importable without Qt,
so ``tools/selftest.py`` can exercise it headless.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import butter, hilbert, sosfiltfilt, stft

log = logging.getLogger(__name__)

ProgressCB = Callable[[int, int, str], bool]
"""``(done, total, message) -> keep_going``. Return False to cancel."""

#: Below this many frames the thread pool costs more than it saves.
PARALLEL_MIN_FRAMES = 20_000


# ==========================================================================
# Configuration
# ==========================================================================
@dataclass
class AnalysisConfig:
    # segmentation
    #: "auto" tries both and keeps whichever locates the animal more reliably.
    #: "background" subtracts a temporal-median background model, which is what
    #: makes real footage work: bedding, a cylinder rim, a lighting gradient and
    #: the dark bench outside the arena are all static, so they cancel.
    #: "threshold" is the original global-threshold path.
    segmentation: str = "auto"
    polarity: str = "auto"
    threshold_mode: str = "otsu"
    threshold_value: int = 100
    bg_samples: int = 40          # frames pooled into the background model
    bg_threshold_value: int = 0   # 0 = Otsu on the difference image
    #: Restrict segmentation to part of the frame. "auto" derives the arena from
    #: the background model. Off by default: background subtraction already
    #: removes static clutter, and an ROI boundary *clips* the animal when it
    #: reaches the wall, which distorts the fitted angle. Use it only when
    #: something moves outside the arena (a hand, a cage-mate, a reflection).
    roi_mode: str = "none"        # none | auto | manual
    roi_frac: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    min_area_frac: float = 0.004
    #: An animal viewed top-down in an arena is a small part of the frame --
    #: measured at 0.9-1.1% on published footage and ~4% on the close-cropped
    #: synthetic fixtures. The old ceiling of 0.45 was meant only to catch "the
    #: whole frame went dark", and it let a 33% blob (the lit arena floor) be
    #: accepted as the animal. 0.25 is still four times the largest real animal
    #: seen and rejects scenery outright.
    max_area_frac: float = 0.25
    #: Treat a frame as untracked when the *body* touches the frame or ROI edge.
    #: A clipped animal has a truncated outline, so its fitted angle is wrong.
    #: Off by default: measured on the reference clips it discarded 17-32% of
    #: perfectly good frames, because in a small arena the animal spends a lot
    #: of its time against the wall. Turn it on for a generously framed rig
    #: where the animal never reaches the edge, or when bursts are being called
    #: at moments the Review tab shows the animal half out of frame.
    reject_edge_blobs: bool = False
    edge_margin_px: int = 2
    morph_kernel: int = 5
    #: Tail-suppression kernel for the head-point estimator, in pixels.
    #: 0 means "measure it", which is the default because the right value
    #: depends on how many pixels the animal occupies, and that changes with
    #: resolution and camera height. A fixed 9 px -- correct for a 320x240
    #: frame where the mouse is ~40 px long -- removed only 0.4-7% of a
    #: 60-98 px wide animal on 906x680 footage, so the tail stayed in the
    #: "body", dragging the long axis toward it and putting the head arrow on
    #: the wrong end.
    core_kernel: int = 0
    # angle
    angle_source: str = "headpoint"
    max_gap_ms: float = 60.0
    # detector
    detector: str = "angle_bandpass"
    band_low_hz: float = 60.0
    band_high_hz: float = 110.0
    filter_order: int = 4
    envelope_smooth_ms: float = 15.0
    threshold_source: str = "baseline"
    threshold_k: float = 6.0
    threshold_manual: float = 0.0
    threshold_low_frac: float = 0.5  # hysteresis: extend a burst down to this
    baseline_window_s: tuple[float, float] = (0.0, 0.0)
    merge_gap_ms: float = 60.0
    min_burst_ms: float = 45.0
    max_burst_ms: float = 600.0
    require_freq_hz: tuple[float, float] = (55.0, 125.0)
    min_half_cycles: int = 6
    max_invalid_frac: float = 0.25
    # motion-spectral detector
    spec_window_ms: float = 256.0
    spec_overlap: float = 0.75
    spec_signal_band: tuple[float, float] = (70.0, 110.0)
    spec_noise_band: tuple[float, float] = (8.0, 50.0)
    spec_ratio_k: float = 5.0
    spec_power_k: float = 6.0  # absolute in-band power must also rise
    spec_smooth_steps: int = 3  # median filter over STFT frames
    # performance
    workers: int = 0
    max_frames: int = 0
    progress_every: int = 2000

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {}
        for k, v in (d or {}).items():
            if k not in known:
                continue
            if k == "roi_frac" and isinstance(v, (list, tuple)) and len(v) == 4:
                v = tuple(float(x) for x in v)
            elif k in ("baseline_window_s", "require_freq_hz", "spec_signal_band",
                       "spec_noise_band") and isinstance(v, (list, tuple)):
                v = (float(v[0]), float(v[1]))
            kwargs[k] = v
        return cls(**kwargs)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("baseline_window_s", "require_freq_hz", "spec_signal_band",
                  "spec_noise_band", "roi_frac"):
            d[k] = list(d[k])
        return d


# ==========================================================================
# Signals
# ==========================================================================
@dataclass
class Signals:
    """Per-frame derived quantities. All arrays share length ``n``."""
    fps: float
    n: int
    t: np.ndarray                 # seconds
    angle_raw: np.ndarray         # degrees, NaN where tracking failed
    angle: np.ndarray             # unwrapped + gap-interpolated, finite
    valid: np.ndarray             # bool, True where the blob was found
    area: np.ndarray              # px^2
    cx: np.ndarray
    cy: np.ndarray
    hx: np.ndarray                # head point (NaN unless angle_source=headpoint)
    hy: np.ndarray
    motion_all: np.ndarray        # mean |frame difference| inside the mouse
    motion_head: np.ndarray       # same, restricted to the head region
    motion_bg: np.ndarray         # same, outside the mouse (lighting-flicker probe)
    angle_period: float = 360.0   # 180 for ellipse/moments, 360 for headpoint
    threshold_used: float = 0.0
    polarity: str = ""
    warnings: list[str] = field(default_factory=list)
    source_path: str = ""
    angle_source: str = ""

    @property
    def duration_s(self) -> float:
        return float(self.t[-1]) if self.n else 0.0

    @property
    def valid_frac(self) -> float:
        return float(self.valid.mean()) if self.n else 0.0

    def summary(self) -> dict:
        return {
            "path": self.source_path,
            "fps": round(self.fps, 3),
            "frames": self.n,
            "duration_s": round(self.duration_s, 3),
            "tracking_valid_pct": round(self.valid_frac * 100, 2),
            "angle_source": self.angle_source,
            "polarity": self.polarity,
            "segmentation_threshold": round(self.threshold_used, 1),
            "warnings": list(self.warnings),
        }


# ==========================================================================
# Segmentation helpers
# ==========================================================================
def _odd(n: int) -> int:
    n = max(1, int(n))
    return n if n % 2 else n + 1


@dataclass
class Blob:
    mask: np.ndarray            # uint8, 255 inside
    area: float
    cx: float
    cy: float
    x0: int                     # bounding box, from the connected-components stats
    y0: int
    x1: int
    y1: int


def _largest_component(mask: np.ndarray, min_area: float, max_area: float
                       ) -> Blob | None:
    """The biggest connected blob whose area falls in the accepted range."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1]
    for oi in order:
        area = float(areas[oi])
        if area < min_area:
            break  # sorted descending: nothing smaller will qualify either
        if area > max_area:
            continue
        lab = int(oi) + 1
        comp = (labels == lab).astype(np.uint8) * 255
        cx, cy = centroids[lab]
        x0 = int(stats[lab, cv2.CC_STAT_LEFT])
        y0 = int(stats[lab, cv2.CC_STAT_TOP])
        return Blob(comp, area, float(cx), float(cy), x0, y0,
                    x0 + int(stats[lab, cv2.CC_STAT_WIDTH]),
                    y0 + int(stats[lab, cv2.CC_STAT_HEIGHT]))
    return None


def _threshold_frame(gray: np.ndarray, thr: float, dark_on_light: bool) -> np.ndarray:
    mode = cv2.THRESH_BINARY_INV if dark_on_light else cv2.THRESH_BINARY
    _, m = cv2.threshold(gray, float(thr), 255, mode)
    return m


def _moments_angle(comp: np.ndarray) -> float:
    """Orientation of the second central moments, in degrees, range (-90, 90]."""
    m = cv2.moments(comp, binaryImage=True)
    if m["m00"] <= 0:
        return float("nan")
    mu20 = m["mu20"] / m["m00"]
    mu02 = m["mu02"] / m["m00"]
    mu11 = m["mu11"] / m["m00"]
    denom = mu20 - mu02
    if abs(denom) < 1e-12 and abs(mu11) < 1e-12:
        return float("nan")
    return float(math.degrees(0.5 * math.atan2(2.0 * mu11, denom)))


def _ellipse_angle(comp: np.ndarray) -> float:
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return float("nan")
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return float("nan")
    return float(cv2.fitEllipse(c)[2])


@dataclass
class HeadAxis:
    """Both ends of the mouse's long axis, resolved to sub-pixel precision."""
    hx: float   # chosen head end (sub-pixel centroid of the leading segment)
    hy: float
    rx: float   # the opposite end (rump)
    ry: float
    cx: float   # body centroid, tail removed
    cy: float
    ex: float   # extremal contour pixel of the chosen end, for continuity
    ey: float


def _end_centroid(xs: np.ndarray, ys: np.ndarray, proj: np.ndarray,
                  cx: float, cy: float, limit: float, leading: bool,
                  fallback: tuple[float, float]) -> tuple[float, float]:
    sel = (proj >= limit) if leading else (proj <= limit)
    if sel.sum() < 4:
        sel = (proj >= limit * 0.5) if leading else (proj <= limit * 0.5)
    if sel.sum() < 1:
        return fallback
    return float(xs[sel].mean()), float(ys[sel].mean())


def _tail_centroid(comp: np.ndarray, core: np.ndarray, cx: float, cy: float
                   ) -> tuple[float, float] | None:
    """Where the tail is, or None if no tail is confidently visible.

    The tail is what the morphological opening removed -- but so are the snout,
    the paws and any ragged edge, and the snout is at the *wrong* end, so taking
    the whole removed region would point the arrow backwards about as often as
    it pointed it forwards. The tail is distinguished by reach: it extends well
    beyond the body outline, which nothing else removed by the opening does.

    So: of the removed pieces, take the one that reaches farthest from the body
    centroid, and accept it only if it reaches clearly past the body itself and
    carries enough pixels to be a tail rather than a speckle.
    """
    removed = cv2.subtract(comp, core)
    if cv2.countNonZero(removed) < 20:
        return None
    n, labels, stats, cents = cv2.connectedComponentsWithStats(removed, 8)
    if n <= 1:
        return None

    core_pts = cv2.findNonZero(core)
    if core_pts is None:
        return None
    cp = core_pts.reshape(-1, 2).astype(np.float64)
    body_reach = float(np.max(np.hypot(cp[:, 0] - cx, cp[:, 1] - cy)))
    if body_reach <= 0:
        return None

    best = None
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 15:
            continue
        pts = cv2.findNonZero((labels == i).astype(np.uint8))
        if pts is None:
            continue
        q = pts.reshape(-1, 2).astype(np.float64)
        reach = float(np.max(np.hypot(q[:, 0] - cx, q[:, 1] - cy)))
        if best is None or reach > best[0]:
            best = (reach, float(cents[i][0]), float(cents[i][1]),
                    int(stats[i, cv2.CC_STAT_AREA]))
    if best is None:
        return None
    reach, tx, ty, area = best
    # A tail sticks out past the body; a snout or a paw does not.
    if reach < body_reach * 1.12:
        return None
    return tx, ty


def _head_axis(comp: np.ndarray, core_kernel: int,
               prev_extreme: tuple[float, float] | None,
               head_frac: float = 0.6) -> HeadAxis | None:
    """Head direction of the mouse, to sub-pixel precision.

    Why a head *direction* and not a body axis: the ellipse/moments axis is
    ambiguous by 180 degrees, so every time the apparent long axis flips the
    trace takes a 180-degree step, and a step is broadband energy that the
    band-pass reads as a twitch. A 0-360 degree direction has no such flip.

    Why sub-pixel: picking the single farthest contour *pixel* makes the angle
    quantised, and the chosen pixel hops between neighbours from frame to frame.
    On a 320x240 frame that hop is several degrees at the frame rate -- a square
    wave sitting right inside the 60-110 Hz band, i.e. a false-positive
    generator. (Measured on the synthetic benchmark: the pixel-extremum version
    produced four spurious bursts, the sub-pixel version none.) So the extremal
    pixel only decides *which end* is the head; the reported position is the
    centroid of the leading ``head_frac`` of the blob along that direction.

    Steps: open the mask to drop the tail (a few pixels wide, against tens for
    the body) -> body centroid -> the two extremities -> pick the head end by
    continuity with the previous frame, or, with no previous frame, as the end
    *farther from the tail* (the tail attaches at the rump) -> sub-pixel
    centroid of each end.

    Both ends are returned because the choice is only locally consistent:
    :func:`extract_signals` re-checks it at every thread-chunk boundary and
    swaps a chunk wholesale if it picked the other end.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(core_kernel),) * 2)
    core = cv2.morphologyEx(comp, cv2.MORPH_OPEN, k)
    if cv2.countNonZero(core) < 20:
        core = comp

    m = cv2.moments(core, binaryImage=True)
    if m["m00"] <= 0:
        return None
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]

    cnts, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    pts = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(pts) < 8:
        return None

    d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    p1 = pts[int(np.argmax(d))]
    a1 = math.atan2(p1[1] - cy, p1[0] - cx)

    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    delta = np.abs(np.arctan2(np.sin(ang - a1), np.cos(ang - a1)))
    far = delta > math.radians(120)
    p2 = pts[int(np.argmax(np.where(far, d, -1.0)))] if far.any() else None

    if p2 is None:
        ex, ey = float(p1[0]), float(p1[1])
    elif prev_extreme is not None:
        d1 = math.hypot(p1[0] - prev_extreme[0], p1[1] - prev_extreme[1])
        d2 = math.hypot(p2[0] - prev_extreme[0], p2[1] - prev_extreme[1])
        ex, ey = (p1 if d1 <= d2 else p2)
    else:
        tail = _tail_centroid(comp, core, cx, cy)
        if tail is not None:
            # The tail attaches at the rump, so the head is the end farther from
            # it. This is a far stronger cue than relative thickness, which was
            # what this used before: measured over sampled frames of four
            # published clips, thickness put the arrow on the correct end in
            # 100%, 92%, 62% and 12% of frames respectively -- i.e. on one clip
            # it was backwards almost always. A mouse's head end is simply not
            # reliably thinner than its rump once the opening has eroded the
            # snout away, whereas the tail is unambiguous whenever it is visible.
            d1 = math.hypot(p1[0] - tail[0], p1[1] - tail[1])
            d2 = math.hypot(p2[0] - tail[0], p2[1] - tail[1])
            ex, ey = (p1 if d1 >= d2 else p2)
        else:
            # No usable tail (curled under the body, or out of frame). Fall back
            # to thickness, which is weak but better than an arbitrary choice.
            dt = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
            h, w = dt.shape

            def thickness(p: np.ndarray, r: int = 5) -> float:
                x0, x1 = max(0, int(p[0]) - r), min(w, int(p[0]) + r + 1)
                y0, y1 = max(0, int(p[1]) - r), min(h, int(p[1]) + r + 1)
                patch = dt[y0:y1, x0:x1]
                return float(patch.mean()) if patch.size else 0.0

            ex, ey = (p1 if thickness(p1) <= thickness(p2) else p2)
    ex, ey = float(ex), float(ey)

    # The projection axis comes from the blob's principal axis, not from the
    # vector to the extremal pixel: that pixel hops between neighbours from
    # frame to frame, and a hopping axis makes the leading-pixel selection -- and
    # therefore the angle -- jitter at the frame rate. The principal axis is
    # sub-pixel and smooth; the extremal pixel is used only for its sign.
    theta = _moments_angle(core)
    if math.isnan(theta):
        norm = math.hypot(ex - cx, ey - cy)
        if norm < 1e-6:
            return HeadAxis(ex, ey, cx, cy, cx, cy, ex, ey)
        ux, uy = (ex - cx) / norm, (ey - cy) / norm
    else:
        ux, uy = math.cos(math.radians(theta)), math.sin(math.radians(theta))
        if (ex - cx) * ux + (ey - cy) * uy < 0:
            ux, uy = -ux, -uy
    ys, xs = np.nonzero(core)
    proj = (xs - cx) * ux + (ys - cy) * uy
    if proj.size == 0:
        return HeadAxis(ex, ey, cx, cy, cx, cy, ex, ey)
    pmax, pmin = float(proj.max()), float(proj.min())
    hx, hy = _end_centroid(xs, ys, proj, cx, cy, head_frac * pmax, True, (ex, ey))
    rx, ry = _end_centroid(xs, ys, proj, cx, cy, head_frac * pmin, False, (cx, cy))
    return HeadAxis(hx, hy, rx, ry, float(cx), float(cy), ex, ey)


# ==========================================================================
# Calibration pass
# ==========================================================================
@dataclass
class Calibration:
    """Everything the per-frame segmentation needs, decided once up front."""
    mode: str                       # "threshold" | "background"
    threshold: float                # grey level, or difference level in bg mode
    dark_on_light: bool             # threshold mode only
    polarity_name: str
    sampled: int
    hit_rate: float
    area_cv: float = 0.0            # coefficient of variation of the blob area
    #: Median centroid displacement between sampled frames, as a fraction of the
    #: frame diagonal. Near zero means the "animal" is not moving between frames
    #: taken minutes apart, i.e. it is scenery.
    travel: float = 0.0
    #: Tail-suppression kernel actually used, in pixels: cfg.core_kernel when
    #: the operator set one, otherwise measured from the animal's own width.
    core_kernel: int = 9
    background: np.ndarray | None = None   # uint8, the temporal-median arena
    roi_mask: np.ndarray | None = None     # uint8 255 inside, 0 outside
    roi_note: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def describe(self) -> str:
        bits = [f"mode={self.mode}", f"threshold={self.threshold:.0f}"]
        if self.mode == "threshold":
            bits.append(self.polarity_name)
        if self.roi_mask is not None:
            bits.append("roi")
        bits.append(f"found in {self.hit_rate:.0%} of sampled frames")
        bits.append(f"travel {self.travel:.3f}")
        bits.append(f"tail kernel {self.core_kernel}px")
        return ", ".join(bits)


def _sample_frames(video: Path | str, count: int) -> tuple[list[np.ndarray], int]:
    """Grab ``count`` frames spread evenly across the file."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idxs = (np.linspace(0, max(n - 1, 0), min(count, max(n, 1)), dtype=int)
                if n > 1 else np.array([0]))
        grays: list[np.ndarray] = []
        for i in idxs:
            if n > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok:
                continue
            grays.append(fr if fr.ndim == 2 else cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        if not grays:
            raise RuntimeError(f"no decodable frames in {video}")
        return grays, n
    finally:
        cap.release()


def build_background(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Per-pixel temporal median of sampled frames: the empty arena.

    The animal is somewhere different in every sampled frame, so at any given
    pixel it is the minority and the median returns the floor. Everything static
    -- bedding, the cylinder rim, a lighting gradient, the bench outside the
    arena -- survives into the model and therefore cancels when subtracted.
    This is the single change that makes real footage work.
    """
    stack = np.stack([f.astype(np.uint8) for f in frames], axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def auto_roi_from_background(bg: np.ndarray, shrink_frac: float = 0.02
                             ) -> tuple[np.ndarray | None, str]:
    """Derive the arena from the background model.

    The arena floor is the large bright region; the bench outside a cylinder is
    dark and is frequently the single largest dark object in frame, which a
    global threshold grabs in preference to the mouse. Keeping only the floor
    removes that whole class of failure.
    """
    h, w = bg.shape[:2]
    blur = cv2.GaussianBlur(bg, (0, 0), max(1.0, min(h, w) / 160.0))
    thr, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return None, "no bright region found; ROI left at the full frame"
    areas = stats[1:, cv2.CC_STAT_AREA]
    lab = int(np.argmax(areas)) + 1
    frac = float(areas[lab - 1]) / float(h * w)
    if frac < 0.12 or frac > 0.97:
        return None, (f"the bright region covers {frac:.0%} of the frame, which "
                      "does not look like an arena; ROI left at the full frame")
    mask = (labels == lab).astype(np.uint8) * 255
    # Pull the edge in so the rim highlight itself is excluded.
    r = max(3, int(round(min(h, w) * shrink_frac)))
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r)))
    kept = float(cv2.countNonZero(mask)) / float(h * w)
    return mask, (f"arena ROI from the background model: {kept:.0%} of the frame "
                  f"(Otsu level {thr:.0f})")


def roi_from_fractions(shape: tuple[int, int],
                       frac: Sequence[float]) -> np.ndarray | None:
    h, w = shape[:2]
    x0, y0, x1, y1 = (float(v) for v in frac)
    if (x0, y0, x1, y1) == (0.0, 0.0, 1.0, 1.0):
        return None
    mask = np.zeros((h, w), np.uint8)
    xa, xb = int(round(x0 * w)), int(round(x1 * w))
    ya, yb = int(round(y0 * h)), int(round(y1 * h))
    xa, xb = max(0, min(xa, w - 1)), max(1, min(xb, w))
    ya, yb = max(0, min(ya, h - 1)), max(1, min(yb, h))
    if xb <= xa or yb <= ya:
        return None
    mask[ya:yb, xa:xb] = 255
    return mask


def _blob_is_clipped(blob: "Blob", shape: tuple[int, int], cal: Calibration,
                     margin: int, core_kernel: int = 9) -> bool:
    """True when the *body* runs into the frame edge or the ROI boundary.

    A clipped animal has a truncated outline, so the fitted axis is wrong; and
    an angle that is wrong only while the animal is at the wall produces a step
    on the way in and another on the way out, which is broadband energy the
    band-pass reports as a twitch.

    The test deliberately ignores the tail. A tail leaving the frame is the
    normal state of affairs in a small arena and says nothing about the head
    angle, so the check is applied to the morphologically opened core -- body
    plus head -- rather than to the whole blob. Testing the whole blob instead
    cost 17 percentage points of tracking on the reference clips, for no gain.
    """
    h, w = shape[:2]
    if (blob.x0 > margin and blob.y0 > margin
            and blob.x1 < w - margin and blob.y1 < h - margin
            and cal.roi_mask is None):
        return False  # comfortably inside, nothing more to check

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(core_kernel),) * 2)
    core = cv2.morphologyEx(blob.mask, cv2.MORPH_OPEN, k)
    if cv2.countNonZero(core) < 20:
        core = blob.mask
    cx0, cy0, cw, ch = cv2.boundingRect(core)
    if (cx0 <= margin or cy0 <= margin
            or cx0 + cw >= w - margin or cy0 + ch >= h - margin):
        return True
    if cal.roi_mask is not None and cal.roi_mask.shape == (h, w):
        # Dilating by the margin and testing against the ROI's complement
        # catches a curved boundary, which a bounding box cannot.
        kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * max(1, margin) + 1,) * 2)
        grown = cv2.dilate(core, kk)
        if cv2.countNonZero(cv2.bitwise_and(grown, cv2.bitwise_not(cal.roi_mask))):
            return True
    return False


def _foreground_mask(gray: np.ndarray, cal: Calibration,
                     kern: np.ndarray) -> np.ndarray:
    """Binary mask of the animal, by whichever route calibration chose."""
    if cal.mode == "background" and cal.background is not None:
        diff = cv2.absdiff(gray, cal.background)
        _, m = cv2.threshold(diff, float(cal.threshold), 255, cv2.THRESH_BINARY)
    else:
        m = _threshold_frame(gray, cal.threshold, cal.dark_on_light)
    if cal.roi_mask is not None and cal.roi_mask.shape == m.shape:
        m = cv2.bitwise_and(m, cal.roi_mask)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kern)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern)
    return m


def _score_mode(frames: Sequence[np.ndarray], cal: Calibration,
                cfg: AnalysisConfig) -> tuple[float, float, float]:
    """(hit rate, area coefficient of variation, travel) for a calibration.

    Hit rate alone is not enough: a threshold that grabs the bench outside the
    cylinder, or the lit arena floor itself, "finds" something in every frame.

    Area stability is not enough either, and getting that wrong cost real
    accuracy -- on published footage of a black mouse in a lit cylinder, the
    polarity search chose *light-on-dark* and locked onto the arena floor, which
    filled a third of the frame with an area coefficient of variation of 0.01.
    It scored better than the animal on both hit rate and stability, because a
    static object is perfectly stable by definition. Every head direction out of
    that run was the orientation of the floor.

    So the decisive test is **travel**: the sampled frames are spread across the
    whole recording, and between two such frames an animal has moved a long way
    while the arena has not moved at all. ``travel`` is the median centroid
    displacement between consecutive samples, as a fraction of the frame
    diagonal. A real animal scores some tenths; scenery scores ~0.
    """
    fa = float(frames[0].size)
    min_area, max_area = cfg.min_area_frac * fa, cfg.max_area_frac * fa
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(cfg.morph_kernel),) * 2)
    h, w = frames[0].shape[:2]
    diag = float(math.hypot(w, h)) or 1.0
    areas: list[float] = []
    centres: list[tuple[float, float]] = []
    for g in frames:
        blob = _largest_component(_foreground_mask(g, cal, kern), min_area, max_area)
        if blob is not None:
            areas.append(blob.area)
            centres.append((blob.cx, blob.cy))
    hit = len(areas) / len(frames)
    if len(areas) < 2:
        return hit, 1.0, 0.0
    mean = float(np.mean(areas))
    cv_ = float(np.std(areas) / mean) if mean > 0 else 1.0
    steps = [math.hypot(b[0] - a[0], b[1] - a[1]) / diag
             for a, b in zip(centres, centres[1:])]
    travel = float(np.median(steps)) if steps else 0.0
    return hit, cv_, travel


#: Below this median between-sample centroid travel (fraction of the frame
#: diagonal), a candidate is scenery rather than an animal. Sampled frames are
#: spread across the whole recording, so between two of them a mouse has moved
#: and the arena has not.
#:
#: The separation is large but the absolute numbers are small, which is why this
#: floor is low. Measured on four published clips (906x680, 10-30 s): the lit
#: arena floor travelled 0.000-0.001, the animal 0.004-0.012 -- a factor of ten,
#: but only ~5-14 px, because a twitching mouse stays in one part of the arena.
#: On the longer synthetic fixtures the animal reaches 0.08. So the test is
#: "did it move at all", not "did it move far", and candidates are additionally
#: compared against each other.
MIN_TRAVEL_FRAC = 0.004


def _measure_core_kernel(frames: Sequence[np.ndarray], cal: Calibration,
                         cfg: AnalysisConfig) -> tuple[int, str]:
    """Pick a tail-removal kernel from the animal's own width.

    The opening has to be wider than the tail and narrower than the body, and
    the correct size is set by how many pixels the animal occupies -- which
    changes with resolution and camera height, so a fixed value cannot be right
    for both a 320x240 fixture and 906x680 footage.

    The factor is deliberately conservative. A larger opening strips the tail
    more reliably but also erodes the neck, and the head point is the centroid
    of the leading part of what survives, so over-opening makes the angle noisy
    and manufactures false positives. Measured: on a synthetic fixture whose
    animal is 44 px wide, going from a 9 px to an 11 px opening cost a false
    positive; on the published clips (60-98 px wide) a 9 px opening left the
    tail in the body and put the head arrow on the wrong end in up to 88% of
    frames. 0.21 of the body width spans both, with a 9 px floor: below that the
    opening stops removing the tail at all, which cost three false positives on
    the clean fixture.
    """
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (_odd(cfg.morph_kernel),) * 2)
    fa = float(frames[0].size)
    widths: list[float] = []
    for g in frames:
        blob = _largest_component(_foreground_mask(g, cal, kern),
                                  cfg.min_area_frac * fa, cfg.max_area_frac * fa)
        if blob is None:
            continue
        dt = cv2.distanceTransform(blob.mask, cv2.DIST_L2, 5)
        widths.append(2.0 * float(dt.max()))
    if not widths:
        return 9, "no blob to measure; tail-removal kernel left at 9 px"
    w = float(np.median(widths))
    k = int(np.clip(_odd(int(round(0.21 * w))), 9, 41))
    return k, (f"tail-removal kernel {k} px, from a measured body width of "
               f"{w:.0f} px")


def calibrate(video: Path | str, cfg: AnalysisConfig, samples: int = 24
              ) -> Calibration:
    """Choose the segmentation route, its threshold, and the ROI.

    Done once and frozen for the whole recording: a per-frame decision would
    make the tracked area jitter as the animal moves, and jitter in the tracked
    shape is jitter in the angle, which is exactly what the detector must not
    see.
    """
    n_samples = max(samples, cfg.bg_samples if cfg.segmentation != "threshold" else samples)
    grays, _n = _sample_frames(video, n_samples)
    notes: list[str] = []
    shape = grays[0].shape[:2]

    # ---- background model (needed for bg mode and for the auto ROI) ----
    want_bg = cfg.segmentation in ("auto", "background") or cfg.roi_mode == "auto"
    background = build_background(grays) if want_bg else None

    # ---- ROI ----
    roi_mask, roi_note = None, ""
    if cfg.roi_mode == "auto" and background is not None:
        roi_mask, roi_note = auto_roi_from_background(background)
    elif cfg.roi_mode == "manual":
        roi_mask = roi_from_fractions(shape, cfg.roi_frac)
        roi_note = ("manual ROI " + str([round(v, 3) for v in cfg.roi_frac])
                    if roi_mask is not None else "manual ROI covers the whole frame")
    if roi_note:
        notes.append(roi_note)

    # ---- candidate: global threshold ----
    def make_threshold_cal() -> Calibration:
        if cfg.threshold_mode == "fixed":
            thr = float(cfg.threshold_value)
            note = f"fixed threshold {thr:.0f}"
        else:
            otsus = [float(cv2.threshold(g, 0, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
                     for g in grays]
            thr = float(np.median(otsus))
            note = f"Otsu threshold (median of {len(otsus)} frames) = {thr:.0f}"
        c = Calibration(mode="threshold", threshold=thr, dark_on_light=True,
                        polarity_name="dark_on_light", sampled=len(grays),
                        hit_rate=0.0, background=background, roi_mask=roi_mask)
        c.notes.append(note)
        if cfg.polarity == "dark_on_light":
            c.dark_on_light, c.polarity_name = True, "dark_on_light"
        elif cfg.polarity == "light_on_dark":
            c.dark_on_light, c.polarity_name = False, "light_on_dark"
        else:
            c.dark_on_light = True
            hit_d, cv_d, mv_d = _score_mode(grays, c, cfg)
            c.dark_on_light = False
            hit_l, cv_l, mv_l = _score_mode(grays, c, cfg)
            # A candidate that does not travel is scenery, whatever else it
            # scores. Only when both (or neither) move does stability decide.
            moves_d, moves_l = mv_d >= MIN_TRAVEL_FRAC, mv_l >= MIN_TRAVEL_FRAC
            if moves_d != moves_l:
                dark_better = moves_d
            else:
                dark_better = (hit_d, -cv_d) >= (hit_l, -cv_l)
            c.dark_on_light = dark_better
            c.polarity_name = "dark_on_light" if dark_better else "light_on_dark"
            c.notes.append(
                f"polarity auto: dark-on-light {hit_d:.0%} (area CV {cv_d:.2f}, "
                f"travel {mv_d:.3f}) vs light-on-dark {hit_l:.0%} "
                f"(area CV {cv_l:.2f}, travel {mv_l:.3f})")
        c.hit_rate, c.area_cv, c.travel = _score_mode(grays, c, cfg)
        return c

    # ---- candidate: background subtraction ----
    def make_background_cal() -> Calibration | None:
        if background is None:
            return None
        if cfg.bg_threshold_value > 0:
            thr = float(cfg.bg_threshold_value)
            note = f"fixed difference threshold {thr:.0f}"
        else:
            diffs = [cv2.absdiff(g, background) for g in grays[:12]]
            otsus = [float(cv2.threshold(d, 0, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
                     for d in diffs]
            # Otsu on a mostly-empty difference image can land very low; keep a
            # floor so sensor noise alone does not become "the animal".
            thr = max(float(np.median(otsus)), 12.0)
            note = (f"background-difference threshold (median Otsu of "
                    f"{len(otsus)} frames) = {thr:.0f}")
        c = Calibration(mode="background", threshold=thr, dark_on_light=True,
                        polarity_name="difference", sampled=len(grays),
                        hit_rate=0.0, background=background, roi_mask=roi_mask)
        c.notes.append(note)
        c.hit_rate, c.area_cv, c.travel = _score_mode(grays, c, cfg)
        return c

    if cfg.segmentation == "threshold":
        cal = make_threshold_cal()
    elif cfg.segmentation == "background":
        cal = make_background_cal() or make_threshold_cal()
    else:
        t_cal = make_threshold_cal()
        b_cal = make_background_cal()
        if b_cal is None:
            cal = t_cal
        else:
            # Conservative ordering: stay on the plain threshold while it is
            # doing its job, and only move to background subtraction when it
            # is not. Switching on a tie would change the behaviour that was
            # validated on clean footage for no measured gain -- and on one
            # reference clip (whose synthetic noise has a whole-field periodic
            # component) background subtraction is actually the worse of the
            # two. "Is the simple method failing?" is the question that
            # generalises; "which scores marginally better here" is not.
            # Travel first: a candidate that does not move between frames
            # taken minutes apart is scenery, and no other score redeems it.
            t_moves = t_cal.travel >= MIN_TRAVEL_FRAC
            b_moves = b_cal.travel >= MIN_TRAVEL_FRAC
            t_ok = (t_moves and t_cal.hit_rate >= 0.90
                    and t_cal.area_cv <= 0.35)
            if t_moves and not b_moves:
                cal, why = t_cal, "only the threshold blob actually moves"
            elif b_moves and not t_moves:
                cal, why = b_cal, ("the threshold blob does not move between "
                                   "sampled frames, so it is scenery, not the "
                                   "animal")
            elif t_ok:
                cal, why = t_cal, "threshold is working, so it was kept"
            elif b_cal.hit_rate > t_cal.hit_rate + 0.05:
                cal, why = b_cal, "background finds the animal more often"
            elif b_cal.hit_rate >= 0.90 and b_cal.area_cv < t_cal.area_cv:
                cal, why = b_cal, "background gives a steadier blob"
            else:
                cal = b_cal if b_cal.hit_rate > t_cal.hit_rate else t_cal
                why = "neither is convincing; took the higher hit rate"
            notes.append(
                f"segmentation auto: threshold {t_cal.hit_rate:.0%} "
                f"(area CV {t_cal.area_cv:.2f}, travel {t_cal.travel:.3f}) vs "
                f"background {b_cal.hit_rate:.0%} "
                f"(area CV {b_cal.area_cv:.2f}, travel {b_cal.travel:.3f}) "
                f"-> chose {cal.mode}; {why}")

    if cfg.core_kernel and cfg.core_kernel > 0:
        cal.core_kernel = int(cfg.core_kernel)
    else:
        cal.core_kernel, kn = _measure_core_kernel(grays, cal, cfg)
        notes.append(kn)

    cal.notes = notes + cal.notes
    if cal.hit_rate < 0.5:
        cal.notes.append(
            f"WARNING: the animal was located in only {cal.hit_rate:.0%} of "
            "sampled frames. Use 'Tune tracking' to see the mask and adjust the "
            "threshold, the ROI or the area limits before trusting any count.")
    elif cal.travel < MIN_TRAVEL_FRAC:
        cal.notes.append(
            f"WARNING: the tracked object moved by only {cal.travel:.3f} of the "
            "frame between sampled frames taken across the whole recording. "
            "That is scenery, not an animal - most often the lit arena floor or "
            "the bench outside it. Open 'Tune tracking', set the segmentation "
            "to 'background', and lower the maximum blob area.")
    elif cal.area_cv > 0.5:
        cal.notes.append(
            f"WARNING: the tracked area varies by {cal.area_cv:.0%} between "
            "frames, which usually means something other than the animal is "
            "being picked up part of the time. Check it in 'Tune tracking'.")
    log.info("calibration: %s", cal.describe)
    return cal


@dataclass
class FrameSegmentation:
    """Everything the tuning view needs for one frame."""
    gray: np.ndarray
    mask: np.ndarray
    blob: "Blob | None"
    head: HeadAxis | None
    angle_deg: float = float("nan")
    area_frac: float = 0.0
    clipped: bool = False
    note: str = ""


def segment_frame(gray: np.ndarray, cfg: AnalysisConfig, cal: Calibration,
                  prev_extreme: tuple[float, float] | None = None
                  ) -> FrameSegmentation:
    """Segment a single frame exactly as the extraction pass would.

    Shared with the interactive tuning view, so what the operator sees is what
    the analysis will do -- not an approximation of it.
    """
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(cfg.morph_kernel),) * 2)
    mask = _foreground_mask(gray, cal, kern)
    fa = float(gray.size)
    blob = _largest_component(mask, cfg.min_area_frac * fa, cfg.max_area_frac * fa)
    out = FrameSegmentation(gray=gray, mask=mask, blob=blob, head=None)
    if blob is None:
        n, _l, stats, _c = cv2.connectedComponentsWithStats(mask, 8)
        biggest = (float(stats[1:, cv2.CC_STAT_AREA].max()) / fa) if n > 1 else 0.0
        out.note = (
            f"no blob in the accepted size range. Largest object is "
            f"{biggest:.3%} of the frame; the range is "
            f"{cfg.min_area_frac:.3%}-{cfg.max_area_frac:.0%}."
            if biggest else "nothing above threshold in this frame.")
        return out
    out.area_frac = blob.area / fa
    if cfg.reject_edge_blobs and _blob_is_clipped(
            blob, gray.shape, cal, cfg.edge_margin_px, cal.core_kernel):
        out.note = ("the animal is touching the frame or ROI edge, so its "
                    "outline is clipped and the angle would be wrong - this "
                    "frame is marked untracked")
        out.blob = blob          # still drawn, so the operator can see why
        out.clipped = True
        return out
    ha = _head_axis(blob.mask, cal.core_kernel, prev_extreme)
    out.head = ha
    if ha is not None:
        if cfg.angle_source == "headpoint":
            out.angle_deg = math.degrees(math.atan2(ha.hy - ha.cy, ha.hx - ha.cx))
        elif cfg.angle_source == "ellipse":
            out.angle_deg = _ellipse_angle(blob.mask)
        else:
            out.angle_deg = _moments_angle(blob.mask)
    return out


# ==========================================================================
# Extraction
# ==========================================================================
@dataclass
class _RawChunk:
    start: int
    angle_raw: np.ndarray
    valid: np.ndarray
    area: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    hx: np.ndarray
    hy: np.ndarray
    rx: np.ndarray
    ry: np.ndarray
    motion_all: np.ndarray
    motion_head: np.ndarray
    motion_rear: np.ndarray
    motion_bg: np.ndarray

    def swap_ends(self) -> None:
        """Flip the head/rump assignment for this whole chunk."""
        self.hx, self.rx = self.rx, self.hx
        self.hy, self.ry = self.ry, self.hy
        self.motion_head, self.motion_rear = self.motion_rear, self.motion_head


def _process_range(video: str, start: int, count: int, pad: int,
                   cfg: AnalysisConfig, cal: Calibration,
                   tick: Callable[[int], bool] | None = None) -> _RawChunk:
    """Decode ``[start-pad, start+count)`` and emit signals for ``[start, …)``.

    The pad frames are decoded but not reported: they seed the frame-difference
    and the head-point continuity so a chunk boundary does not create a step.
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")

    a = np.full(count, np.nan, dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    area = np.zeros(count, dtype=np.float32)
    cxs = np.full(count, np.nan, dtype=np.float32)
    cys = np.full(count, np.nan, dtype=np.float32)
    hxs = np.full(count, np.nan, dtype=np.float32)
    hys = np.full(count, np.nan, dtype=np.float32)
    rxs = np.full(count, np.nan, dtype=np.float32)
    rys = np.full(count, np.nan, dtype=np.float32)
    mot = np.zeros(count, dtype=np.float32)
    moth = np.zeros(count, dtype=np.float32)
    motr = np.zeros(count, dtype=np.float32)
    motb = np.zeros(count, dtype=np.float32)

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(cfg.morph_kernel),) * 2)
    clipped = 0
    want_head = cfg.angle_source == "headpoint"
    prev_extreme: tuple[float, float] | None = None
    prev_blur: np.ndarray | None = None
    first = max(0, start - pad)

    try:
        if first > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        min_area = max_area = None
        i = first
        produced = 0
        while i < start + count:
            ok, frame = cap.read()
            if not ok:
                break
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if min_area is None:
                fa = float(gray.size)
                min_area, max_area = cfg.min_area_frac * fa, cfg.max_area_frac * fa

            m = _foreground_mask(gray, cal, kern)
            blob = _largest_component(m, min_area, max_area)
            if (blob is not None and cfg.reject_edge_blobs
                    and _blob_is_clipped(blob, gray.shape, cal, cfg.edge_margin_px,
                                      cal.core_kernel)):
                blob = None
                clipped += 1
            comp = blob.mask if blob is not None else None
            ar = blob.area if blob is not None else 0.0
            cx = blob.cx if blob is not None else float("nan")
            cy = blob.cy if blob is not None else float("nan")

            ang = float("nan")
            hx = hy = rx = ry = float("nan")
            if comp is not None:
                # The head axis is computed whatever the angle source: the
                # motion-spectral detector needs a head region of interest, and
                # the Review tab draws it as an overlay. Only the *angle* comes
                # from the configured source, and for the head-point source it
                # is computed at merge time, once the end assignment is settled.
                ha = _head_axis(comp, cal.core_kernel, prev_extreme)
                if ha is not None:
                    hx, hy, rx, ry = ha.hx, ha.hy, ha.rx, ha.ry
                    prev_extreme = (ha.ex, ha.ey)
                    cx, cy = ha.cx, ha.cy
                if want_head:
                    ang = 0.0 if ha is not None else float("nan")
                elif cfg.angle_source == "ellipse":
                    ang = _ellipse_angle(comp)
                else:
                    ang = _moments_angle(comp)

            # Frame-difference motion energy. The 3x3 blur matters: sensor noise
            # is spatially uncorrelated, so blurring cuts it by roughly a factor
            # of three while leaving the edge displacement that actual movement
            # produces, which is what the spectral detector needs to see.
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            m_all = m_head = m_rear = m_bg = 0.0
            if prev_blur is not None and prev_blur.shape == blur.shape:
                diff = cv2.absdiff(blur, prev_blur)
                if comp is not None:
                    m_all = float(cv2.mean(diff, mask=comp)[0])
                    # Background = everything outside the mouse's padded bounding
                    # box, by subtraction so no extra mask has to be built.
                    pad_px = 8
                    y0 = max(0, blob.y0 - pad_px)
                    y1 = min(diff.shape[0], blob.y1 + pad_px)
                    x0 = max(0, blob.x0 - pad_px)
                    x1 = min(diff.shape[1], blob.x1 + pad_px)
                    box_n = (y1 - y0) * (x1 - x0)
                    outside_n = diff.size - box_n
                    if outside_n > 64:
                        total = float(diff.sum(dtype=np.int64))
                        inbox = float(diff[y0:y1, x0:x1].sum(dtype=np.int64))
                        m_bg = (total - inbox) / outside_n
                    if np.isfinite(hx):
                        r = max(6, int(round(math.sqrt(max(ar, 1.0)) * 0.35)))

                        def _patch_mean(px: float, py: float) -> float:
                            ay0, ay1 = max(0, int(py) - r), min(diff.shape[0], int(py) + r + 1)
                            ax0, ax1 = max(0, int(px) - r), min(diff.shape[1], int(px) + r + 1)
                            pa = diff[ay0:ay1, ax0:ax1]
                            return float(pa.mean()) if pa.size else m_all

                        m_head = _patch_mean(hx, hy)
                        m_rear = _patch_mean(rx, ry) if np.isfinite(rx) else m_all
                    else:
                        m_head = m_rear = m_all
                else:
                    m_all = m_head = m_rear = m_bg = float(diff.mean())
            prev_blur = blur

            if i >= start:
                j = i - start
                a[j] = ang
                valid[j] = comp is not None and np.isfinite(ang)
                area[j] = ar
                cxs[j], cys[j] = cx, cy
                hxs[j], hys[j] = hx, hy
                rxs[j], rys[j] = rx, ry
                mot[j], moth[j], motr[j], motb[j] = m_all, m_head, m_rear, m_bg
                produced += 1
                if tick is not None and produced % max(1, cfg.progress_every) == 0:
                    if not tick(produced):
                        break
            i += 1
    finally:
        cap.release()

    return _RawChunk(start, a, valid, area, cxs, cys, hxs, hys, rxs, rys,
                     mot, moth, motr, motb)


def _seek_is_reliable(video: str, n_frames: int) -> bool:
    """MJPEG/AVI is all-intra so frame seeking is exact; check before trusting it."""
    if n_frames < 100:
        return False
    cap = cv2.VideoCapture(video)
    try:
        if not cap.isOpened():
            return False
        target = n_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ok, _ = cap.read()
        return bool(ok) and abs(pos - target) <= 2
    except cv2.error:
        return False
    finally:
        cap.release()


def frame_count_and_fps(video: Path | str, fps_override: float = 0.0
                        ) -> tuple[int, float, list[str]]:
    """Frame count and frame rate, preferring an externally supplied true fps.

    OpenCV reports a bogus FPS for MJPEG-in-AVI, so the caller should pass the
    ffprobe ``nb_read_frames / duration`` value as ``fps_override``. Without one
    this falls back to the container value and says so.
    """
    warns: list[str] = []
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps_container = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()

    fps = fps_override if fps_override and fps_override > 1 else fps_container
    if not (fps_override and fps_override > 1):
        warns.append(
            f"using the container frame rate ({fps_container:.2f} fps). OpenCV "
            "misreports this for MJPEG AVI files; verify with ffprobe "
            "(-count_frames) so timing and the HTR band are right.")
    if fps <= 1:
        fps = 240.0
        warns.append("frame rate unreadable; assuming 240 fps")
    return n, fps, warns


def extract_signals(video: Path | str, cfg: AnalysisConfig, fps: float = 0.0,
                    n_frames: int = 0, progress: ProgressCB | None = None,
                    calibration: Calibration | None = None) -> Signals:
    """Full extraction pass. Memory use is independent of recording length."""
    video_s = str(video)
    n_meta, fps_used, warns = frame_count_and_fps(video_s, fps)
    total = n_frames or n_meta
    if cfg.max_frames and total:
        total = min(total, cfg.max_frames)
    if total <= 0:
        total = 0  # unknown: sequential pass will discover it

    cal = calibration or calibrate(video_s, cfg)
    warns.extend(cal.notes)

    workers = cfg.workers or max(1, min(6, (os.cpu_count() or 2) // 2))
    pad = max(2, int(round(0.25 * fps_used)))

    use_parallel = (workers > 1 and total > PARALLEL_MIN_FRAMES
                    and _seek_is_reliable(video_s, total))
    if workers > 1 and not use_parallel and total > PARALLEL_MIN_FRAMES:
        warns.append("frame seeking is not exact in this file; extracting on a "
                     "single thread (slower but correct)")

    cancelled = threading.Event()
    done_lock = threading.Lock()
    done = [0]

    def make_tick(label: str) -> Callable[[int], bool]:
        last = [0]

        def tick(produced: int) -> bool:
            with done_lock:
                done[0] += produced - last[0]
                last[0] = produced
                d = done[0]
            if progress is not None and not progress(d, total, label):
                cancelled.set()
                return False
            return not cancelled.is_set()

        return tick

    chunks: list[_RawChunk] = []
    if use_parallel:
        bounds = np.linspace(0, total, workers + 1, dtype=int)
        jobs = [(int(bounds[i]), int(bounds[i + 1] - bounds[i]))
                for i in range(workers) if bounds[i + 1] > bounds[i]]
        log.info("parallel extraction: %d threads over %d frames", len(jobs), total)
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futs = [pool.submit(_process_range, video_s, s, c, pad, cfg, cal,
                                make_tick(f"extracting ({len(jobs)} threads)"))
                    for s, c in jobs]
            for f in futs:
                chunks.append(f.result())
    else:
        if total <= 0:
            total = 10 ** 9  # streaming until EOF; progress shows frames done
        chunks.append(_process_range(video_s, 0, total, 0, cfg, cal,
                                     make_tick("extracting")))

    chunks.sort(key=lambda c: c.start)

    # Each chunk decides for itself which end of the mouse is the head (by
    # continuity within the chunk). Two chunks can disagree, which would put a
    # 180-degree step at the boundary -- broadband energy that reads as a
    # twitch. So compare the last tracked head of one chunk with the first of
    # the next: if the next chunk's *other* end is the closer match, that chunk
    # chose the opposite end and is flipped wholesale. One frame separates the
    # two samples, so the mouse has barely moved and the comparison is safe.
    swaps = 0
    for prev_c, cur in zip(chunks, chunks[1:]):
        pv = np.nonzero(np.isfinite(prev_c.hx))[0]
        cv_ = np.nonzero(np.isfinite(cur.hx))[0]
        if pv.size == 0 or cv_.size == 0:
            continue
        ph = (float(prev_c.hx[pv[-1]]), float(prev_c.hy[pv[-1]]))
        ch = (float(cur.hx[cv_[0]]), float(cur.hy[cv_[0]]))
        cr = (float(cur.rx[cv_[0]]), float(cur.ry[cv_[0]]))
        if not all(map(np.isfinite, cr)):
            continue
        if math.dist(ph, cr) < math.dist(ph, ch):
            cur.swap_ends()
            swaps += 1
    if swaps:
        log.info("head/rump assignment flipped for %d of %d chunks", swaps, len(chunks))

    angle_raw = np.concatenate([c.angle_raw for c in chunks])
    valid = np.concatenate([c.valid for c in chunks])
    area = np.concatenate([c.area for c in chunks])
    cx = np.concatenate([c.cx for c in chunks])
    cy = np.concatenate([c.cy for c in chunks])
    hx = np.concatenate([c.hx for c in chunks])
    hy = np.concatenate([c.hy for c in chunks])
    mot = np.concatenate([c.motion_all for c in chunks])
    moth = np.concatenate([c.motion_head for c in chunks])
    motb = np.concatenate([c.motion_bg for c in chunks])

    # Trim the tail of a short read (file ended early, or the user cancelled).
    last = int(np.max(np.nonzero(valid)[0])) + 1 if valid.any() else 0
    if last and last < len(valid) - 1:
        keep = last
        angle_raw, valid, area = angle_raw[:keep], valid[:keep], area[:keep]
        cx, cy, hx, hy = cx[:keep], cy[:keep], hx[:keep], hy[:keep]
        mot, moth, motb = mot[:keep], moth[:keep], motb[:keep]

    n = len(angle_raw)
    t = np.arange(n, dtype=np.float64) / fps_used
    period = 360.0 if cfg.angle_source == "headpoint" else 180.0
    if cfg.angle_source == "headpoint":
        with np.errstate(invalid="ignore"):
            angle_raw = np.degrees(np.arctan2(hy - cy, hx - cx)).astype(np.float32)
        angle_raw[~valid] = np.nan
    angle = _unwrap_and_fill(angle_raw, valid, fps_used, cfg.max_gap_ms, period)

    if n and valid.mean() < 0.75:
        warns.append(
            f"tracking succeeded on only {valid.mean():.0%} of frames. Detection "
            "will be conservative; improve contrast or tune the threshold.")

    sig = Signals(fps=fps_used, n=n, t=t, angle_raw=angle_raw, angle=angle,
                  valid=valid, area=area, cx=cx, cy=cy, hx=hx, hy=hy,
                  motion_all=mot, motion_head=moth, motion_bg=motb,
                  angle_period=period,
                  threshold_used=cal.threshold, polarity=cal.polarity_name,
                  warnings=warns, source_path=video_s, angle_source=cfg.angle_source)
    log.info("extracted %d frames at %.2f fps (%.1f%% tracked) from %s",
             n, fps_used, sig.valid_frac * 100, Path(video_s).name)
    return sig


def _unwrap_and_fill(angle_raw: np.ndarray, valid: np.ndarray, fps: float,
                     max_gap_ms: float, period: float) -> np.ndarray:
    """Interpolate short tracking gaps, unwrap, and keep the result finite.

    Long gaps are also interpolated -- the filter needs a finite array -- but
    ``valid`` still marks them, and :func:`detect` rejects bursts that overlap
    them, so an interpolated stretch can never become a detection.
    """
    n = len(angle_raw)
    out = np.array(angle_raw, dtype=np.float64)
    if n == 0:
        return out
    good = np.isfinite(out)
    if not good.any():
        return np.zeros(n, dtype=np.float64)

    idx = np.arange(n)
    out = np.interp(idx, idx[good], out[good])
    out = np.unwrap(out, period=period)

    # Smooth across interpolated runs longer than max_gap_ms so the join does
    # not leave a kink that the band-pass would ring on.
    gap_frames = max(1, int(round(max_gap_ms / 1000.0 * fps)))
    bad = ~good
    if bad.any():
        lab, nlab = ndimage.label(bad)
        if nlab:
            sigma = max(1.0, gap_frames / 2.0)
            smooth = ndimage.gaussian_filter1d(out, sigma=sigma, mode="nearest")
            for sl in ndimage.find_objects(lab):
                if (sl[0].stop - sl[0].start) > gap_frames:
                    lo = max(0, sl[0].start - gap_frames)
                    hi = min(n, sl[0].stop + gap_frames)
                    out[lo:hi] = smooth[lo:hi]
    return out


# ==========================================================================
# Burst detection
# ==========================================================================
@dataclass
class Burst:
    start_s: float
    end_s: float
    start_frame: int
    end_frame: int
    duration_ms: float
    peak_envelope: float = 0.0
    peak_velocity_deg_s: float = 0.0
    dominant_hz: float = 0.0
    half_cycles: int = 0
    invalid_frac: float = 0.0
    detector: str = ""
    accepted: bool = True        # curation flag, editable in the Review tab
    manual: bool = False         # True when a human added it
    note: str = ""

    @property
    def center_s(self) -> float:
        return 0.5 * (self.start_s + self.end_s)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Burst":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class DetectionResult:
    bursts: list[Burst]
    detector: str
    threshold: float
    fps: float
    duration_s: float
    diagnostics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # Traces kept for plotting (decimated by the UI, not here).
    filtered: np.ndarray | None = None
    envelope: np.ndarray | None = None
    spec_t: np.ndarray | None = None
    spec_ratio: np.ndarray | None = None
    spec_threshold: float = 0.0
    rejected: list[Burst] = field(default_factory=list)

    @property
    def accepted_bursts(self) -> list[Burst]:
        return [b for b in self.bursts if b.accepted]

    @property
    def count(self) -> int:
        return len(self.accepted_bursts)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as half-open [start, stop) index pairs."""
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    stops = list(np.flatnonzero(d == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        stops.append(len(mask))
    return list(zip(starts, stops))


def _hysteresis_runs(x: np.ndarray, thr_high: float, thr_low: float
                     ) -> list[tuple[int, int]]:
    """Runs above ``thr_low`` that contain at least one sample above ``thr_high``.

    A Schmitt trigger, for the same reason oscilloscopes use one. An HTR burst is
    amplitude-modulated: 5-11 oscillations that grow and fade. A single threshold
    clips the burst wherever the envelope dips through it, so one twitch gets
    counted two or three times -- the over-counting that held the first-generation
    pipeline to F1 = 0.86. Triggering high and releasing low keeps a burst whole
    without lowering the bar for starting one.
    """
    if thr_low >= thr_high:
        return _runs(x > thr_high)
    out: list[tuple[int, int]] = []
    for s, e in _runs(x > thr_low):
        if np.any(x[s:e] > thr_high):
            out.append((s, e))
    return out


def _merge_runs(runs: Sequence[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    out = [list(runs[0])]
    for s, e in runs[1:]:
        if s - out[-1][1] <= max_gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [(int(a), int(b)) for a, b in out]


def _robust_threshold(x: np.ndarray, k: float) -> tuple[float, dict]:
    """median + k * (MAD-derived sigma). Insensitive to the bursts themselves."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return 0.0, {"median": 0.0, "sigma": 0.0}
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    sigma = mad * 1.4826
    if sigma <= 0:  # degenerate (e.g. a synthetic constant baseline)
        sigma = float(np.std(finite)) or 1e-6
    return med + k * sigma, {"median": med, "sigma": sigma}


def _baseline_slice(sig: Signals, cfg: AnalysisConfig,
                    baseline_default_s: float = 0.0) -> tuple[int, int, str]:
    lo_s, hi_s = cfg.baseline_window_s
    if hi_s > lo_s >= 0:
        note = f"baseline window {lo_s:.0f}-{hi_s:.0f} s (explicit)"
    elif baseline_default_s > 0 and baseline_default_s < sig.duration_s * 0.9:
        lo_s, hi_s = 0.0, baseline_default_s
        note = f"baseline window 0-{hi_s:.0f} s (trial pre-gavage period)"
    else:
        lo_s, hi_s = 0.0, max(1.0, sig.duration_s * 0.1)
        note = (f"baseline window 0-{hi_s:.1f} s (first 10% of the recording - "
                "no pre-dose period recorded)")
    lo = max(0, int(round(lo_s * sig.fps)))
    hi = min(sig.n, int(round(hi_s * sig.fps)))
    if hi - lo < int(sig.fps):  # under one second is not a baseline
        lo, hi = 0, sig.n
        note = "baseline window too short; using the whole recording"
    return lo, hi, note


def _bandpass(x: np.ndarray, fps: float, lo: float, hi: float, order: int
              ) -> tuple[np.ndarray, list[str]]:
    warns: list[str] = []
    nyq = fps / 2.0
    hi_eff = min(hi, 0.45 * fps)
    if hi_eff < hi:
        if hi_eff >= 98.0:
            # Still above the top of the published HTR band, so this is a
            # housekeeping note rather than a problem.
            warns.append(
                f"high cutoff trimmed from {hi:.0f} Hz to {hi_eff:.1f} Hz to stay "
                f"clear of the {nyq:.0f} Hz Nyquist limit at {fps:.0f} fps. The "
                "78-98 Hz HTR band is still fully covered.")
        else:
            warns.append(
                f"high cutoff reduced from {hi:.0f} Hz to {hi_eff:.1f} Hz: at "
                f"{fps:.0f} fps the Nyquist limit is only {nyq:.0f} Hz, which cuts "
                "into the 78-98 Hz HTR band. The oscillation will alias and counts "
                "from this recording will be unreliable. Record at 320x240 to get "
                "240 fps.")
    lo_eff = min(lo, hi_eff * 0.5)
    if lo_eff >= hi_eff:
        warns.append("frame rate too low for any usable HTR band; "
                     "falling back to an unfiltered velocity trace")
        return x.astype(np.float64), warns
    sos = butter(order, [lo_eff, hi_eff], btype="bandpass", fs=fps, output="sos")
    return sosfiltfilt(sos, x), warns


def detect_angle_bandpass(sig: Signals, cfg: AnalysisConfig,
                          baseline_default_s: float = 0.0) -> DetectionResult:
    res = DetectionResult(bursts=[], detector="angle_bandpass", threshold=0.0,
                          fps=sig.fps, duration_s=sig.duration_s)
    if sig.n < 32:
        res.warnings.append("recording too short to analyse")
        return res

    velocity = np.gradient(sig.angle, sig.t)
    filt, warns = _bandpass(velocity, sig.fps, cfg.band_low_hz, cfg.band_high_hz,
                            cfg.filter_order)
    res.warnings.extend(warns)

    env = np.abs(hilbert(filt))
    sigma = max(1.0, cfg.envelope_smooth_ms / 1000.0 * sig.fps / 2.0)
    env = ndimage.gaussian_filter1d(env, sigma=sigma, mode="nearest")

    if cfg.threshold_source == "manual" and cfg.threshold_manual > 0:
        thr = float(cfg.threshold_manual)
        res.diagnostics["threshold_note"] = "manual"
    elif cfg.threshold_source == "whole":
        thr, stats = _robust_threshold(env, cfg.threshold_k)
        res.diagnostics.update(stats)
        res.diagnostics["threshold_note"] = "robust statistics over whole recording"
    else:
        lo, hi, note = _baseline_slice(sig, cfg, baseline_default_s)
        thr, stats = _robust_threshold(env[lo:hi], cfg.threshold_k)
        res.diagnostics.update(stats)
        res.diagnostics["threshold_note"] = note
        res.diagnostics["baseline_frames"] = [lo, hi]
    res.threshold = thr
    res.filtered = filt
    res.envelope = env

    merge = max(1, int(round(cfg.merge_gap_ms / 1000.0 * sig.fps)))
    thr_low = thr * max(0.0, min(cfg.threshold_low_frac, 0.99))
    res.diagnostics["threshold_low"] = round(thr_low, 4)
    runs = _merge_runs(_hysteresis_runs(env, thr, thr_low), merge)

    min_f = int(round(cfg.min_burst_ms / 1000.0 * sig.fps))
    max_f = int(round(cfg.max_burst_ms / 1000.0 * sig.fps))
    f_lo, f_hi = cfg.require_freq_hz

    for s, e in runs:
        dur_f = e - s
        dur_ms = dur_f / sig.fps * 1000.0
        seg = filt[s:e]
        crossings = int(np.count_nonzero(np.diff(np.signbit(seg)))) if seg.size > 1 else 0
        dom_hz = crossings / (2.0 * (dur_f / sig.fps)) if dur_f else 0.0
        inval = 1.0 - float(sig.valid[s:e].mean()) if dur_f else 1.0
        b = Burst(
            start_s=float(sig.t[s]), end_s=float(sig.t[min(e, sig.n) - 1]),
            start_frame=s, end_frame=e, duration_ms=dur_ms,
            peak_envelope=float(env[s:e].max()) if dur_f else 0.0,
            peak_velocity_deg_s=float(np.abs(seg).max()) if seg.size else 0.0,
            dominant_hz=dom_hz, half_cycles=crossings, invalid_frac=inval,
            detector="angle_bandpass",
        )
        if dur_f < min_f:
            b.note = f"rejected: {dur_ms:.0f} ms shorter than {cfg.min_burst_ms:.0f} ms"
        elif dur_f > max_f:
            b.note = f"rejected: {dur_ms:.0f} ms longer than {cfg.max_burst_ms:.0f} ms"
        elif crossings < cfg.min_half_cycles:
            b.note = (f"rejected: {crossings} half-cycles, fewer than the "
                      f"{cfg.min_half_cycles} expected of an HTR burst")
        elif not (f_lo <= dom_hz <= f_hi):
            b.note = (f"rejected: dominant {dom_hz:.0f} Hz outside the "
                      f"{f_lo:.0f}-{f_hi:.0f} Hz gate")
        elif inval > cfg.max_invalid_frac:
            b.note = f"rejected: {inval:.0%} of frames had no tracking"
        if b.note:
            res.rejected.append(b)
        else:
            res.bursts.append(b)

    res.diagnostics["candidate_runs"] = len(runs)
    res.diagnostics["rejected"] = len(res.rejected)
    return res


@dataclass
class BandRatio:
    t: np.ndarray            # STFT frame centres, seconds
    ratio: np.ndarray        # in-band power / activity-band power
    power: np.ndarray        # absolute in-band power
    nperseg: int = 0
    noverlap: int = 0


def _band_ratio(x: np.ndarray, fps: float, cfg: AnalysisConfig) -> BandRatio:
    """STFT band-power ratio (HTR band vs general-activity band) of a signal.

    The *ratio* is the discriminant rather than raw power because it cancels the
    overall activity level: a running mouse raises both bands, a twitching one
    raises only the HTR band. The absolute in-band power is returned alongside
    it so a caller can require both -- a ratio alone will occasionally look good
    on pure sensor noise, where both bands are small.
    """
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    x = x - ndimage.gaussian_filter1d(x, sigma=max(1.0, 0.05 * fps), mode="nearest")
    nper = max(16, int(round(cfg.spec_window_ms / 1000.0 * fps)))
    nover = int(round(nper * min(max(cfg.spec_overlap, 0.0), 0.95)))
    f, tt, Z = stft(x, fs=fps, nperseg=nper, noverlap=nover, window="hann",
                    padded=True, boundary="zeros")
    P = np.abs(Z) ** 2
    s_lo, s_hi = cfg.spec_signal_band
    s_hi = min(s_hi, fps * 0.45)
    n_lo, n_hi = cfg.spec_noise_band
    sb = (f >= s_lo) & (f <= s_hi)
    nb = (f >= n_lo) & (f <= n_hi)
    if sb.sum() == 0 or nb.sum() == 0:
        z = np.zeros_like(tt)
        return BandRatio(tt, z, z, nper, nover)
    sig_pw = P[sb].mean(axis=0)
    ratio = sig_pw / (P[nb].mean(axis=0) + 1e-12)
    k = max(1, int(cfg.spec_smooth_steps))
    if k > 1 and ratio.size > k:
        ratio = ndimage.median_filter(ratio, size=k, mode="nearest")
        sig_pw = ndimage.median_filter(sig_pw, size=k, mode="nearest")
    return BandRatio(tt, ratio, sig_pw, nper, nover)


def background_flicker_check(sig: Signals, cfg: AnalysisConfig) -> dict:
    """Is the *arena*, not the mouse, oscillating in the HTR band?

    A PWM-dimmed LED panel modulates the whole field at hundreds of hertz. Under
    a sub-millisecond global-shutter exposure that becomes frame-to-frame
    brightness stepping, which rings the HTR band-pass everywhere at once. Since
    the mouse occupies a small part of the frame, the background tells us
    whether the oscillation is the animal or the light.
    """
    out = {"checked": False}
    if sig.n < 128 or not np.isfinite(sig.motion_bg).any() or sig.motion_bg.max() <= 0:
        return out
    br = _band_ratio(sig.motion_bg, sig.fps, cfg)
    bg_ratio = br.ratio
    med = float(np.median(bg_ratio)) if bg_ratio.size else 0.0
    p99 = float(np.percentile(bg_ratio, 99)) if bg_ratio.size else 0.0
    out.update({"checked": True, "bg_band_ratio_median": round(med, 3),
                "bg_band_ratio_p99": round(p99, 3)})
    if med > 1.5 or p99 > 6.0:
        out["message"] = (
            f"The arena background is itself oscillating in the HTR band "
            f"(median band ratio {med:.2f}, p99 {p99:.2f}). That is almost always "
            "PWM-dimmed lighting, not the animal. Re-run the flicker test on the "
            "Acquisition tab and switch the panels to constant-current DC drive "
            "before trusting any counts from this recording.")
        out["suspect"] = True
    else:
        out["suspect"] = False
    return out


def detect_motion_spectral(sig: Signals, cfg: AnalysisConfig,
                           baseline_default_s: float = 0.0) -> DetectionResult:
    """Spectral detector on frame-difference energy.

    Independent of the angle fit: it asks whether the pixels inside the mouse
    are oscillating in the HTR band, so it keeps working when segmentation
    briefly loses the animal. The discriminant is a band-power *ratio*
    (70-110 Hz against 8-50 Hz), which cancels out overall activity level --
    a running mouse raises both bands, a twitching one raises only the first.
    """
    res = DetectionResult(bursts=[], detector="motion_spectral", threshold=0.0,
                          fps=sig.fps, duration_s=sig.duration_s)
    if sig.n < 64:
        res.warnings.append("recording too short to analyse")
        return res

    base = sig.motion_head if np.isfinite(sig.motion_head).any() else sig.motion_all
    if cfg.spec_signal_band[1] > sig.fps * 0.45:
        res.warnings.append(
            f"signal band upper edge {cfg.spec_signal_band[1]:.0f} Hz exceeds the "
            f"usable {sig.fps * 0.45:.0f} Hz at {sig.fps:.0f} fps; clipped")
    br = _band_ratio(base, sig.fps, cfg)
    tt, ratio, power, nper, nover = br.t, br.ratio, br.power, br.nperseg, br.noverlap
    if ratio.size == 0 or not np.any(ratio):
        res.warnings.append("frame rate too low to form the spectral bands")
        return res

    # Background probe, used below to veto whole-field (lighting) oscillation.
    bg_ratio = None
    if np.isfinite(sig.motion_bg).any() and sig.motion_bg.max() > 0:
        bg = _band_ratio(sig.motion_bg, sig.fps, cfg)
        if bg.ratio.shape == ratio.shape:
            bg_ratio = bg.ratio

    if cfg.threshold_source == "manual" and cfg.threshold_manual > 0:
        thr = float(cfg.threshold_manual)
        pw_thr = 0.0
        note = "manual"
    else:
        if cfg.threshold_source == "baseline":
            lo, hi, note = _baseline_slice(sig, cfg, baseline_default_s)
            m = (tt >= lo / sig.fps) & (tt <= hi / sig.fps)
            use = m if m.sum() > 4 else np.ones_like(tt, dtype=bool)
        else:
            use, note = np.ones_like(tt, dtype=bool), "robust statistics over whole recording"
        thr, stats = _robust_threshold(ratio[use], cfg.spec_ratio_k)
        pw_thr, _ = _robust_threshold(power[use], cfg.spec_power_k)
        res.diagnostics.update(stats)
    res.threshold = thr
    res.spec_t, res.spec_ratio, res.spec_threshold = tt, ratio, thr
    res.diagnostics["threshold_note"] = note
    res.diagnostics["stft_window_frames"] = nper
    res.diagnostics["in_band_power_threshold"] = round(float(pw_thr), 6)

    hop_s = (nper - nover) / sig.fps
    merge_steps = max(1, int(round(cfg.merge_gap_ms / 1000.0 / max(hop_s, 1e-9))))
    thr_low = thr * max(0.0, min(cfg.threshold_low_frac, 0.99))
    # Both tests must pass: the band ratio has to rise *and* the absolute
    # in-band power has to rise. The ratio alone can look favourable on pure
    # sensor noise, where both bands are near zero.
    ratio = np.where(power >= pw_thr, ratio, 0.0) if pw_thr > 0 else ratio
    runs = _merge_runs(_hysteresis_runs(ratio, thr, thr_low), merge_steps)

    for s, e in runs:
        t0 = float(tt[s])
        t1 = float(tt[min(e, len(tt) - 1)])
        dur_ms = (t1 - t0) * 1000.0
        f0 = int(round(t0 * sig.fps))
        f1 = min(sig.n, int(round(t1 * sig.fps)) + 1)
        inval = 1.0 - float(sig.valid[f0:f1].mean()) if f1 > f0 else 1.0
        b = Burst(start_s=t0, end_s=t1, start_frame=f0, end_frame=f1,
                  duration_ms=dur_ms,
                  peak_envelope=float(ratio[s:e].max()) if e > s else 0.0,
                  invalid_frac=inval, detector="motion_spectral")
        bg_here = float(bg_ratio[s:e].max()) if (bg_ratio is not None and e > s) else 0.0
        if dur_ms < cfg.min_burst_ms:
            b.note = f"rejected: {dur_ms:.0f} ms shorter than {cfg.min_burst_ms:.0f} ms"
        elif dur_ms > cfg.max_burst_ms:
            b.note = f"rejected: {dur_ms:.0f} ms longer than {cfg.max_burst_ms:.0f} ms"
        elif inval > cfg.max_invalid_frac:
            b.note = f"rejected: {inval:.0%} of frames had no tracking"
        elif bg_here >= b.peak_envelope * 0.8 and bg_here > thr:
            b.note = ("rejected: the empty arena oscillates just as strongly here - "
                      "this is lighting flicker, not the animal")
        if b.note:
            res.rejected.append(b)
        else:
            res.bursts.append(b)

    res.diagnostics["candidate_runs"] = len(runs)
    res.diagnostics["rejected"] = len(res.rejected)
    return res


def _overlaps(a: Burst, b: Burst, pad_s: float = 0.05) -> bool:
    return (a.start_s - pad_s) <= b.end_s and (b.start_s - pad_s) <= a.end_s


def detect(sig: Signals, cfg: AnalysisConfig,
           baseline_default_s: float = 0.0) -> DetectionResult:
    """Run the configured detector (or both, combined)."""
    mode = cfg.detector
    flick = background_flicker_check(sig, cfg)

    def finish(r: DetectionResult) -> DetectionResult:
        r.diagnostics["background_flicker"] = flick
        if flick.get("suspect"):
            r.warnings.insert(0, flick["message"])
        return r

    if mode == "angle_bandpass":
        return finish(detect_angle_bandpass(sig, cfg, baseline_default_s))
    if mode == "motion_spectral":
        return finish(detect_motion_spectral(sig, cfg, baseline_default_s))

    a = detect_angle_bandpass(sig, cfg, baseline_default_s)
    m = detect_motion_spectral(sig, cfg, baseline_default_s)
    out = DetectionResult(bursts=[], detector=mode, threshold=a.threshold,
                          fps=sig.fps, duration_s=sig.duration_s)
    out.warnings = a.warnings + m.warnings
    out.filtered, out.envelope = a.filtered, a.envelope
    out.spec_t, out.spec_ratio, out.spec_threshold = m.spec_t, m.spec_ratio, m.spec_threshold
    out.diagnostics = {"angle": a.diagnostics, "motion": m.diagnostics}

    if mode == "both_agree":
        for b in a.bursts:
            if any(_overlaps(b, o) for o in m.bursts):
                b.detector = "both"
                out.bursts.append(b)
            else:
                b.note = "rejected: the motion-spectral detector did not agree"
                out.rejected.append(b)
        out.rejected.extend(a.rejected)
        out.diagnostics["agreement"] = (
            f"{len(out.bursts)} of {len(a.bursts)} angle bursts confirmed by motion")
    else:  # "either"
        out.bursts = list(a.bursts)
        for b in m.bursts:
            if not any(_overlaps(b, o) for o in out.bursts):
                out.bursts.append(b)
        out.bursts.sort(key=lambda x: x.start_s)
        out.rejected = a.rejected + m.rejected
    return finish(out)


# ==========================================================================
# Spectrum diagnostics
# ==========================================================================
#: Published HTR frequency range (Halberstadt et al. 2014).
HTR_BAND_HZ = (78.0, 98.0)


@dataclass
class SpectrumResult:
    freqs: np.ndarray
    power: np.ndarray
    source: str
    fps: float
    peaks: list[tuple[float, float]] = field(default_factory=list)  # (Hz, power)
    band_fraction: float = 0.0     # share of 1-Nyquist power inside the HTR band
    band_peak_hz: float = 0.0      # strongest component *inside* the HTR band
    notes: list[str] = field(default_factory=list)

    @property
    def peak_hz(self) -> float:
        return self.peaks[0][0] if self.peaks else 0.0


def spectrum(sig: Signals, source: str = "angle_velocity",
             fmin: float = 2.0, nperseg_s: float = 1.0) -> SpectrumResult:
    """Power spectrum of a derived signal, for "where is the energy?".

    This exists because the most common way to get nothing out of this pipeline
    is for the band to be in the wrong place -- either the recording aliased, or
    the file is a slow-motion render whose declared frame rate is the playback
    rate rather than the capture rate. Both are invisible in the burst list and
    obvious in the spectrum, so the spectrum is what the operator should look at
    when a video with visible twitches yields no detections.
    """
    from scipy.signal import welch

    if source == "motion_head":
        x = np.nan_to_num(sig.motion_head.astype(np.float64))
    elif source == "motion_all":
        x = np.nan_to_num(sig.motion_all.astype(np.float64))
    elif source == "motion_bg":
        x = np.nan_to_num(sig.motion_bg.astype(np.float64))
    elif source == "angle":
        x = np.nan_to_num(sig.angle.astype(np.float64))
    else:
        source = "angle_velocity"
        x = np.gradient(sig.angle, sig.t) if sig.n > 2 else np.zeros(sig.n)
    x = np.asarray(x, dtype=np.float64)
    res = SpectrumResult(freqs=np.zeros(0), power=np.zeros(0), source=source,
                         fps=sig.fps)
    if x.size < 64:
        res.notes.append("recording too short for a spectrum")
        return res

    # Remove the mean and any slow trend so locomotion does not swamp the plot.
    x = x - ndimage.gaussian_filter1d(x, sigma=max(1.0, 0.25 * sig.fps),
                                      mode="nearest")
    nper = int(min(max(64, round(nperseg_s * sig.fps)), x.size))
    f, P = welch(x, fs=sig.fps, nperseg=nper, noverlap=nper // 2,
                 window="hann", detrend="constant")
    res.freqs, res.power = f, P

    usable = (f >= fmin) & (f <= sig.fps * 0.45)
    if usable.sum() < 4:
        res.notes.append("frame rate too low to resolve anything useful")
        return res
    fu, Pu = f[usable], P[usable]

    # Peaks: local maxima, strongest first, well separated.
    order = np.argsort(Pu)[::-1]
    picked: list[tuple[float, float]] = []
    for i in order:
        fi = float(fu[i])
        if any(abs(fi - pf) < max(2.0, 0.08 * fi) for pf, _ in picked):
            continue
        picked.append((fi, float(Pu[i])))
        if len(picked) >= 6:
            break
    res.peaks = picked

    in_band = (fu >= HTR_BAND_HZ[0]) & (fu <= HTR_BAND_HZ[1])
    total = float(Pu.sum())
    res.band_fraction = float(Pu[in_band].sum() / total) if total > 0 else 0.0
    # Reported separately from the global peak, because the two answer different
    # questions. A trial with any locomotion in it has its tallest single peak
    # down at 5-10 Hz -- locomotion is continuous, while a twitch is 100 ms in a
    # 70-minute recording -- so the global peak is a poor guide to whether the
    # band is populated. The band's own peak and the band fraction are what say
    # that, and they are what the rate suggestion is gated on.
    if in_band.any():
        res.band_peak_hz = float(fu[in_band][int(np.argmax(Pu[in_band]))])

    if sig.fps < 200:
        res.notes.append(
            f"At {sig.fps:.0f} fps the usable range stops at "
            f"{sig.fps * 0.45:.0f} Hz, so the 78-98 Hz HTR band is "
            + ("partly" if sig.fps * 0.45 > HTR_BAND_HZ[0] else "entirely")
            + " out of reach of this file as timed.")
    if res.peaks and res.band_fraction < 0.05:
        res.notes.append(
            f"Only {res.band_fraction:.1%} of the signal power falls in the "
            f"78-98 Hz band; the strongest component is at "
            f"{res.peak_hz:.1f} Hz.")
    return res


#: Above this, an "implied capture rate" is not a camera, it is arithmetic on
#: the wrong peak. Used to stop the slow-motion diagnostic firing on ordinary
#: locomotion. High-speed rigs of this kind run at 120-1000 fps.
MAX_PLAUSIBLE_FPS = 1000.0


@dataclass
class RateSuggestion:
    container_fps: float
    peak_hz: float
    implied_fps: float
    slowmo_factor: float
    confidence: str          # "likely" | "possible" | "unlikely"
    message: str


def suggest_capture_rate(spec: SpectrumResult,
                         expect_hz: float = 88.0) -> RateSuggestion | None:
    """If this file is a slow-motion render, what was the capture rate?

    Published supplementary videos are routinely written at 30 fps for playback
    even when captured at several hundred. The frames are all there; only the
    declared rate is wrong, so the twitch is plainly visible by eye and sits at
    ``capture_rate / playback_rate`` times its true frequency in the file's own
    timebase. A detector band-passing 78-98 Hz on file timestamps then finds
    nothing at all.

    Given the strongest oscillation in the file and the published HTR centre
    frequency, this reports the capture rate that would put the two together.
    It is a suggestion to check, not a measurement: a 10 Hz peak could equally
    be grooming in a genuinely 30 fps recording, which is why the confidence
    field is there and why the operator is asked to confirm against the footage.
    """
    if not spec.peaks or spec.fps <= 0:
        return None
    # The decisive question is whether the band is EMPTY, not where the tallest
    # peak sits. Any trial with locomotion in it has its tallest single peak at
    # 5-10 Hz, because locomotion runs continuously while a twitch is 100 ms in
    # a 70-minute recording; keying off the global peak alone would tell a
    # perfectly good 240 fps recording to change its frame rate. So a populated
    # band ends the enquiry, whatever else is louder.
    if spec.band_fraction >= 0.05 or spec.band_peak_hz > 0:
        reachable = spec.fps * 0.45 >= HTR_BAND_HZ[1]
        if reachable and spec.band_fraction >= 0.05:
            return None
    peak = spec.peak_hz
    if peak <= 0:
        return None
    factor = expect_hz / peak
    implied = spec.fps * factor
    if 0.8 <= factor <= 1.25:
        return None  # already about right; nothing to suggest

    if factor > 1.0:
        # Integer-ish factors are the giveaway: renders are made at 1/2, 1/4,
        # 1/8 speed, not at 1/6.37.
        nearest = min((2, 3, 4, 5, 6, 8, 10, 12, 16, 20),
                      key=lambda k: abs(k - factor))
        off = abs(nearest - factor) / nearest
        if off < 0.18 and spec.fps <= 120:
            conf = "likely"
        elif spec.fps <= 120:
            conf = "possible"
        else:
            conf = "unlikely"
        # Plausibility bound. The peak this is divided into is whatever is
        # loudest, and in a trial with any locomotion that is 2-10 Hz, which
        # implies a capture rate of thousands of fps. No camera in this kind of
        # rig runs at that, so an implied rate that high means the peak was
        # never a twitch -- it was the animal walking. Without this the
        # diagnostic confidently reported a "44x slow-motion render of a
        # ~5280 fps capture" for an ordinary 120 fps recording.
        if implied > MAX_PLAUSIBLE_FPS or factor > 20.0:
            conf = "unlikely"
            return RateSuggestion(
                spec.fps, peak, implied, factor, conf,
                f"The strongest oscillation is at {peak:.1f} Hz. Treating that "
                f"as an HTR would imply a capture rate of {implied:.0f} fps, "
                "which no camera in this kind of rig runs at - so that peak is "
                "the animal moving, not a twitch, and this is not a "
                "slow-motion render. If nothing is being detected, the reason "
                "is elsewhere.")
        msg = (
            f"The strongest oscillation in this file is at {peak:.1f} Hz. If it "
            f"is an HTR (published centre {expect_hz:.0f} Hz), the file is a "
            f"{factor:.1f}x slow-motion render and the real capture rate was "
            f"about {implied:.0f} fps"
            + (f" -- consistent with a {nearest}x render of a "
               f"{spec.fps * nearest:.0f} fps capture."
               if off < 0.18 else ".")
            + " Set the capture-rate override to that value and re-run. If the "
            "animal in the video is moving at normal speed, this is not a "
            "slow-motion file and the peak is something else (grooming is "
            "6-9 Hz).")
        return RateSuggestion(spec.fps, peak, implied, factor, conf, msg)

    msg = (f"The strongest oscillation is at {peak:.1f} Hz, above the published "
           f"{expect_hz:.0f} Hz. If the file was sped up, the capture rate would "
           f"be about {implied:.0f} fps. More often a peak this high is sensor "
           "noise or lighting flicker rather than the animal.")
    return RateSuggestion(spec.fps, peak, implied, factor, "unlikely", msg)


# ==========================================================================
# Binning and validation
# ==========================================================================
def bin_counts(bursts: Iterable[Burst], duration_s: float, bin_s: float = 60.0,
               t0_s: float = 0.0) -> list[dict]:
    """Counts per time bin, with bin labels relative to T0 (dose time).

    With a 10-minute pre-gavage baseline, ``t0_s=600`` makes the baseline bins
    come out as -10..-1 and the first post-dose minute as 0.
    """
    bin_s = max(1.0, float(bin_s))
    n_bins = max(1, int(math.ceil(duration_s / bin_s)))
    counts = [0] * n_bins
    for b in bursts:
        if not b.accepted:
            continue
        i = int(b.center_s // bin_s)
        if 0 <= i < n_bins:
            counts[i] += 1
    out = []
    for i, c in enumerate(counts):
        start = i * bin_s
        out.append({
            "bin_index": i,
            "start_s": round(start, 3),
            "end_s": round(min(start + bin_s, duration_s), 3),
            "bin_rel_t0": int(math.floor((start - t0_s) / bin_s)),
            "count": c,
            "rate_per_min": round(c * 60.0 / bin_s, 3),
        })
    return out


@dataclass
class ValidationResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    tolerance_s: float = 0.15
    matched: list[tuple[float, float]] = field(default_factory=list)
    missed: list[tuple[float, float]] = field(default_factory=list)
    spurious: list[float] = field(default_factory=list)

    def line(self) -> str:
        return (f"TP={self.tp} FP={self.fp} FN={self.fn}  "
                f"precision={self.precision:.3f} recall={self.recall:.3f} "
                f"F1={self.f1:.3f}")


def validate(bursts: Sequence[Burst], truth: Sequence[tuple[float, float]],
             tolerance_s: float = 0.15) -> ValidationResult:
    """One-to-one greedy match of detections to ground-truth intervals.

    A detection counts as a hit when it overlaps a truth interval, or when its
    centre lands within ``tolerance_s`` of it. Each truth interval can be
    claimed once, so a double-count scores as one TP and one FP -- which is
    exactly the failure mode this pipeline was built to remove.
    """
    res = ValidationResult(tolerance_s=tolerance_s)
    det = sorted((b for b in bursts if b.accepted), key=lambda b: b.start_s)
    used = [False] * len(truth)
    for b in det:
        hit = -1
        best = float("inf")
        for i, (ts, te) in enumerate(truth):
            if used[i]:
                continue
            overlap = min(b.end_s, te) - max(b.start_s, ts)
            dist = 0.0 if overlap > 0 else min(abs(b.start_s - te), abs(ts - b.end_s))
            if (overlap > 0 or dist <= tolerance_s) and dist < best:
                best, hit = dist, i
        if hit >= 0:
            used[hit] = True
            res.tp += 1
            res.matched.append(truth[hit])
        else:
            res.fp += 1
            res.spurious.append(b.center_s)
    res.fn = used.count(False)
    res.missed = [truth[i] for i, u in enumerate(used) if not u]
    res.precision = res.tp / (res.tp + res.fp) if (res.tp + res.fp) else 0.0
    res.recall = res.tp / (res.tp + res.fn) if (res.tp + res.fn) else 0.0
    res.f1 = (2 * res.precision * res.recall / (res.precision + res.recall)
              if (res.precision + res.recall) else 0.0)
    return res


def truth_from_ground_truth_csv(path: Path | str) -> list[tuple[float, float]]:
    """Read ``benchmark_videos/ground_truth.csv`` into (start_s, end_s) intervals.

    Expected columns: ``frame_index, time_sec, twitch_active, ...``.
    """
    import csv

    intervals: list[tuple[float, float]] = []
    start: float | None = None
    prev_t = 0.0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                t = float(row["time_sec"])
                active = int(float(row["twitch_active"]))
            except (KeyError, TypeError, ValueError):
                continue
            if active and start is None:
                start = t
            elif not active and start is not None:
                intervals.append((start, prev_t))
                start = None
            prev_t = t
    if start is not None:
        intervals.append((start, prev_t))
    return intervals


# ==========================================================================
# Persistence of a completed analysis
# ==========================================================================
def save_result(path: Path, video: Path, sig: Signals, det: DetectionResult,
                cfg: AnalysisConfig, bins: list[dict], extra: dict | None = None
                ) -> Path:
    doc = {
        "htr_studio_result_version": 1,
        "video": str(video),
        "signals": sig.summary(),
        "config": cfg.to_dict(),
        "detector": det.detector,
        "threshold": det.threshold,
        "diagnostics": det.diagnostics,
        "warnings": det.warnings,
        "bursts": [b.to_dict() for b in det.bursts],
        "rejected": [b.to_dict() for b in det.rejected],
        "bins": bins,
        "n_accepted": det.count,
    }
    if extra:
        doc.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(path)
    log.info("saved analysis result to %s", path)
    return path


def load_result(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
