"""Work out sensible analysis settings for a given recording, and explain why.

Two jobs, both of which exist because the defaults can only be right for one
kind of file.

**Per-file advice** (:func:`suggest_settings`). The defaults in
:class:`analysis.AnalysisConfig` assume a 240 fps QVGA recording of a mouse
in an arena, because that is this rig. Point the tool at a 30 fps published
clip, or a 906x680 crop, or a rig at a different height, and several settings
are no longer merely suboptimal but wrong -- most importantly the band-pass
band, which at 120 fps must sit where the 78-98 Hz twitch *aliases to* rather
than where it really is. This inspects the file and says what to change, with
the arithmetic behind each change, and changes nothing by itself.

**Per-session calibration** (:func:`calibrate_from_control`). An untreated
recording is a null experiment: whatever the detector finds in it is a false
positive, by construction. Sweeping the threshold over that recording
therefore measures this rig's noise floor directly -- this arena, this
lighting, this camera height, this animal's grooming -- which is a far better
basis for a threshold than any constant chosen by someone else. The result is
saved as a profile and applied to the treated recordings of the same session.

The honest limits of both are stated in the returned text rather than hidden:
a suggestion is a suggestion, a sweep over one control animal is one control
animal, and neither replaces checking the calls in the Review tab.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from . import analysis, ffmpeg_tools, paths

log = logging.getLogger(__name__)

#: Published head-twitch oscillation band (Halberstadt 2014, magnetometry).
TRUE_BAND_HZ = analysis.HTR_BAND_HZ

#: Below this, the oscillation cannot be recovered from the frames at all and
#: a band-pass on head angle is measuring nothing. Glatfelter 2025 measured
#: F1 falling 95.5 -> 88.6 -> 58.9 going 120 -> 60 -> 30 fps with a trained
#: pose model, which is a far more capable detector than a band-pass.
MIN_USABLE_FPS = 60.0

#: At or above this the true band is under Nyquist and needs no folding.
DIRECT_BAND_FPS = 2.2 * TRUE_BAND_HZ[1]      # ~216 fps


# ==========================================================================
# Per-file advice
# ==========================================================================
@dataclass
class Change:
    """One proposed setting change, with the reasoning kept attached to it."""
    key: str
    label: str
    current: object
    suggested: object
    why: str
    level: str = "info"          # info | warn | bad

    @property
    def differs(self) -> bool:
        a, b = self.current, self.suggested
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return [round(float(x), 6) for x in a] != [round(float(x), 6) for x in b]
        if isinstance(a, float) or isinstance(b, float):
            try:
                return abs(float(a) - float(b)) > 1e-6
            except (TypeError, ValueError):
                return a != b
        return a != b


@dataclass
class Advice:
    source: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    fps_exact: bool = False
    duration_s: float = 0.0
    changes: list[Change] = field(default_factory=list)
    notes: list[tuple[str, str]] = field(default_factory=list)   # (level, text)
    calibration: str = ""
    usable: bool = True

    @property
    def pending(self) -> list[Change]:
        return [c for c in self.changes if c.differs]

    def apply_to(self, cfg: analysis.AnalysisConfig) -> analysis.AnalysisConfig:
        """A copy of ``cfg`` with every proposed change applied."""
        d = cfg.to_dict()
        for c in self.pending:
            d[c.key] = c.suggested
        return analysis.AnalysisConfig.from_dict(d)

    def provenance(self) -> dict:
        """What went into the sidecar, so a figure can be traced to it."""
        return {
            "source": self.source,
            "measured": {"width": self.width, "height": self.height,
                         "fps": round(self.fps, 4), "fps_exact": self.fps_exact,
                         "duration_s": round(self.duration_s, 3)},
            "changes": [{k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in asdict(c).items()} for c in self.pending],
            "notes": [{"level": lv, "text": tx} for lv, tx in self.notes],
            "calibration": self.calibration,
        }


def fold_hz(f: float, fps: float) -> float:
    """Where a real frequency ``f`` appears when sampled at ``fps``."""
    r = f % fps
    return min(r, fps - r)


def fold_band(fps: float, lo: float = TRUE_BAND_HZ[0],
              hi: float = TRUE_BAND_HZ[1]) -> tuple[float, float]:
    """The image of the band ``[lo, hi]`` under sampling at ``fps``.

    Sampled densely rather than at the endpoints, because folding is not
    monotonic: it reflects at every multiple of ``fps/2``. A band that
    straddles one of those multiples has an image wider than its endpoints
    suggest -- 100-140 Hz at 120 fps folds to 0-20 Hz, while its endpoints
    both land on 20. Endpoint arithmetic would miss that entirely and propose
    a band that excludes most of the signal.
    """
    xs = np.linspace(lo, hi, 257)
    ys = np.array([fold_hz(float(x), fps) for x in xs])
    return float(ys.min()), float(ys.max())


def band_for_rate(fps: float) -> tuple[tuple[float, float], str]:
    """The band to filter in, for footage sampled at ``fps``.

    Above Nyquist for the twitch, the band is simply where the twitch is.
    Below it, the oscillation does not vanish -- it folds. A real 88 Hz
    movement sampled at 120 fps appears at ``|88 - 120| = 32`` Hz, so
    filtering 60-110 Hz finds nothing at all while the signal sits plainly
    at 32. The band to use is then the image of 78-98 Hz under folding,
    widened for the frame-rate uncertainty that comes with trusting a rate
    declared by a container.

    Returned with its reasoning, because a folded band needs explaining
    wherever it appears: nobody reading these settings later should have to
    work out for themselves why a 78-98 Hz phenomenon is being filtered at
    17-47 Hz.
    """
    lo, hi = TRUE_BAND_HZ
    nyquist = fps / 2.0
    if hi < nyquist:
        # Under Nyquist: filter where the twitch is, with margin either side,
        # but never past Nyquist itself.
        band = (60.0, min(110.0, nyquist * 0.98))
        why = (f"{fps:.0f} fps puts the whole 78-98 Hz band under Nyquist "
               f"({nyquist:.0f} Hz), so the twitch is filtered where it "
               "actually is")
        if band[1] < 110.0:
            why += (f"; the upper edge is held at {band[1]:.0f} Hz because "
                    f"Nyquist is {nyquist:.0f} Hz")
        return band, why

    a, b = fold_band(fps)
    # Widen by 25% of the span, floored at 5 Hz, for rate uncertainty. The
    # cap is Nyquist, not a fraction of it: the folded band is already inside
    # [0, fps/2] by construction and truncating it further would cut signal.
    pad = max(5.0, 0.25 * (b - a))
    band = (max(0.5, a - pad), min(nyquist * 0.98, b + pad))
    why = (f"at {fps:.0f} fps the 78-98 Hz twitch is above Nyquist "
           f"({nyquist:.0f} Hz) and folds to {a:.0f}-{b:.0f} Hz, so that is "
           f"where it has to be filtered; padded to {band[0]:.0f}-"
           f"{band[1]:.0f} Hz for frame-rate uncertainty")
    return band, why


#: Below this, head movement is dominated by locomotion, grooming and
#: rearing. A folded twitch band reaching down here is confounded with them.
LOCOMOTION_HZ = 15.0


def _gates_for_rate(fps: float) -> list[tuple[str, object, str]]:
    """Duration and half-cycle gates that a given frame rate can support.

    A burst of 5-11 half-cycles at 78-98 Hz lasts 51-141 ms. The gates have
    to be reachable: requiring 6 half-cycles of a folded oscillation is a
    different count from 6 half-cycles of the real one, and a minimum burst
    length shorter than a couple of frames is not a gate at all.
    """
    out: list[tuple[str, object, str]] = []
    frame_ms = 1000.0 / max(fps, 1e-6)
    # Keep the published 51-141 ms burst window, but never ask for a duration
    # the sampling cannot resolve.
    min_ms = max(45.0, 4.0 * frame_ms)
    out.append(("min_burst_ms", round(min_ms, 1),
                f"a frame is {frame_ms:.1f} ms, so the shortest burst worth "
                f"calling spans at least four of them ({min_ms:.0f} ms); "
                "published bursts run 51-141 ms"))
    band, _ = band_for_rate(fps)
    # The in-burst frequency gate must describe the band being filtered.
    freq = (round(band[0] * 0.9, 1), round(band[1] * 1.15, 1))
    out.append(("require_freq_hz", list(freq),
                f"the in-burst frequency gate has to describe the band "
                f"actually filtered ({band[0]:.0f}-{band[1]:.0f} Hz), with "
                "a little tolerance either side"))
    if fps < DIRECT_BAND_FPS:
        # A folded oscillation completes fewer visible half-cycles per burst.
        half = 4
        out.append(("min_half_cycles", half,
                    f"a folded oscillation shows fewer resolvable half-cycles "
                    f"than the real one, so requiring 6 at {fps:.0f} fps "
                    "rejects genuine bursts; 4 is the reachable equivalent"))
    return out


def suggest_settings(video: Path | str, cfg: analysis.AnalysisConfig,
                     ffprobe: str = "", samples: int = 24,
                     progress=None) -> Advice:
    """Inspect a recording and propose settings for it.

    Reads the container for size and true frame rate, then calibrates on a
    sample of frames -- the same :func:`analysis.calibrate` the analysis
    itself uses, so the numbers come from the real segmentation rather than
    an approximation of it.
    """
    path = Path(video)
    adv = Advice(source=str(path))
    if progress:
        progress(1, 3, "reading the container…")
    info = ffmpeg_tools.probe_video(path, ffprobe, count_frames=False)
    adv.width, adv.height = info.width, info.height
    adv.fps, adv.fps_exact = info.true_fps, info.exact
    adv.duration_s = info.duration_s
    fps = adv.fps or 0.0

    if fps <= 0:
        adv.usable = False
        adv.notes.append(("bad", "the container declares no usable frame rate, "
                                 "so nothing can be advised; set the rate "
                                 "override on the Analysis tab by hand"))
        return adv
    if not adv.fps_exact:
        adv.notes.append(
            ("warn", f"the {fps:.2f} fps figure is the container's declared "
                     "rate, not a counted one. Every burst time and duration "
                     "scales with it, so confirm it with 'Verify rate' "
                     "before quoting anything from this file"))

    # ---- frame rate: the setting everything else depends on --------------
    if fps < MIN_USABLE_FPS:
        adv.usable = False
        adv.notes.append(
            ("bad", f"{fps:.0f} fps cannot resolve a 78-98 Hz oscillation by "
                    "any means: at this rate the twitch occupies about "
                    f"{fps * 0.1:.0f} frames in total. Published work puts a "
                    "trained pose model at F1 0.59 on 30 fps footage against "
                    "0.96 at 120. Treat any result from this file as "
                    "indicative only"))
    elif fps < DIRECT_BAND_FPS:
        adv.notes.append(
            ("warn", f"{fps:.0f} fps is below the ~216 fps needed to sample "
                     "the twitch directly, so the detector works on the "
                     "folded (aliased) oscillation. That is sound, but it "
                     "throws away the half-cycle structure that separates a "
                     "twitch from a fast head turn, so expect more false "
                     "positives than on the 240 fps rig"))

    band, band_why = band_for_rate(fps)
    folded = TRUE_BAND_HZ[1] >= fps / 2.0
    lvl = "warn" if folded else "info"
    adv.changes.append(Change(
        "band_low_hz", "Band-pass low", cfg.band_low_hz, round(band[0], 1),
        band_why, lvl))
    adv.changes.append(Change(
        "band_high_hz", "Band-pass high", cfg.band_high_hz, round(band[1], 1),
        band_why, lvl))
    if folded and band[0] < LOCOMOTION_HZ:
        adv.notes.append(
            ("bad", f"at {fps:.0f} fps the twitch folds down to "
                    f"{band[0]:.0f}-{band[1]:.0f} Hz, which overlaps the "
                    f"range below {LOCOMOTION_HZ:.0f} Hz where locomotion, "
                    "grooming and rearing live. The detector cannot tell a "
                    "twitch from a brisk head turn in that band, so expect "
                    "false positives that no threshold will remove. This is "
                    "a property of the frame rate, not of the settings: the "
                    f"rate is near a multiple of 78-98 Hz, and recording a "
                    "few fps away from it would move the folded band clear"))
    for key, val, why in _gates_for_rate(fps):
        adv.changes.append(Change(
            key, key.replace("_", " "), getattr(cfg, key), val, why))

    # ---- what the frames themselves say ----------------------------------
    if progress:
        progress(2, 3, f"calibrating on {samples} frames…")
    probe_cfg = adv.apply_to(cfg)
    try:
        cal = analysis.calibrate(path, probe_cfg, samples=samples)
    except Exception as e:                       # pragma: no cover
        adv.notes.append(("warn", f"could not calibrate on this file: {e}"))
        return adv
    adv.calibration = cal.describe

    if cal.hit_rate < 0.5:
        adv.notes.append(
            ("bad", f"the animal was located in only {cal.hit_rate:.0%} of "
                    "sampled frames. Settings advice cannot fix that - it is "
                    "a segmentation problem. Open 'Tune tracking' and check "
                    "the polarity and area limits against the footage before "
                    "trusting any analysis of this file"))
    elif cal.hit_rate < 0.85:
        adv.notes.append(
            ("warn", f"the animal was located in {cal.hit_rate:.0%} of "
                     "sampled frames; below about 85% the angle trace has "
                     "enough gaps to cost real detections"))
    if cal.travel < analysis.MIN_TRAVEL_FRAC:
        adv.notes.append(
            ("bad", f"the tracked object barely moves between sampled frames "
                    f"(travel {cal.travel:.4f}, animals measure 0.004-0.012). "
                    "This is the signature of locking onto scenery - the "
                    "arena floor or a cage wall - not an animal"))

    # Area limits from the animal actually in this frame, not from a constant.
    if progress:
        progress(3, 3, "sizing the animal…")
    area = _measure_area_frac(path, probe_cfg, cal, samples=min(samples, 16))
    if area > 0:
        # Four times the measured median, which is the margin the shipped
        # default was chosen to give on this rig.
        ceiling = min(0.9, max(0.05, round(area * 4.0, 3)))
        adv.changes.append(Change(
            "max_area_frac", "Max blob area", cfg.max_area_frac, ceiling,
            f"the animal measures {area:.2%} of the frame here, so a ceiling "
            f"of {ceiling:.1%} is four times its size - loose enough for a "
            "rear or a stretch, tight enough to reject the arena floor, "
            "which is what a too-high ceiling lets through"))
        floor = max(0.0005, round(area * 0.15, 4))
        adv.changes.append(Change(
            "min_area_frac", "Min blob area", cfg.min_area_frac, floor,
            f"15% of the measured animal area ({area:.2%}), so a partly "
            "occluded animal still counts while specks do not"))

    # core_kernel is already measured per-file when left at 0; say so rather
    # than proposing a number that would freeze it.
    if cfg.core_kernel != 0:
        adv.changes.append(Change(
            "core_kernel", "Tail kernel", cfg.core_kernel, 0,
            f"0 means 'measure it from the animal's width', which on this "
            f"file gives {cal.core_kernel} px. A fixed value correct for one "
            "resolution puts the head arrow on the tail at another, which is "
            "exactly how this setting has gone wrong before"))

    if adv.width * adv.height > 640 * 480:
        adv.notes.append(
            ("info", f"{adv.width}x{adv.height} is larger than this rig "
                     "records. Nothing needs changing for it, but extraction "
                     "will be slower in proportion to the pixel count"))
    return adv


def _measure_area_frac(video: Path | str, cfg: analysis.AnalysisConfig,
                       cal: analysis.Calibration, samples: int = 16) -> float:
    """Median blob area as a fraction of the frame, over sampled frames."""
    frames, _ = analysis._sample_frames(video, samples)
    fracs = []
    for g in frames:
        seg = analysis.segment_frame(g, cfg, cal)
        if seg.blob is not None and seg.area_frac > 0:
            fracs.append(seg.area_frac)
    return float(np.median(fracs)) if fracs else 0.0


# ==========================================================================
# Per-session calibration from an untreated control
# ==========================================================================
#: The sweep. threshold_k is a multiple of the baseline MAD, so these span a
#: very permissive detector to a very conservative one.
K_SWEEP = (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 12.0)


@dataclass
class ControlCalibration:
    source: str = ""
    duration_s: float = 0.0
    fps: float = 0.0
    sweep: list[tuple[float, int]] = field(default_factory=list)  # (k, bursts)
    chosen_k: float = 0.0
    zero_k: float = 0.0           # the lowest k with no calls at all
    calls_at_chosen: int = 0
    peak_envelope: float = 0.0    # tallest envelope peak in the control
    threshold_at_chosen: float = 0.0
    margin: float = 0.0           # threshold / peak; >1 means peak is below it
    ok: bool = False
    message: str = ""
    cfg: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sweep"] = [[float(k), int(n)] for k, n in self.sweep]
        return d


def calibrate_from_control(video: Path | str, cfg: analysis.AnalysisConfig,
                           ffprobe: str = "", fps_override: float = 0.0,
                           progress=None, should_cancel=None
                           ) -> ControlCalibration:
    """Pick a detection threshold from a recording with no drug in it.

    The logic is the whole point: an untreated animal should produce no head
    twitches, so every burst the detector calls on this file is a false
    positive. Sweeping ``threshold_k`` and counting calls therefore traces
    this rig's false-positive curve directly -- including the parts no
    constant can anticipate, like how much this arena's bedding rustles and
    how vigorously this cohort grooms.

    The threshold chosen is one step *below* the lowest k that gives zero
    calls. Sitting exactly at the zero point would mean the detector was
    tuned until it could just barely not see the loudest non-twitch in the
    control, which is a hair's breadth from not seeing a real twitch either;
    one step down keeps sensitivity and costs, by construction, only the
    handful of calls that the sweep showed at that step.

    What this cannot do: one control animal is one control animal. The margin
    reported is the honest measure of how much room there is, and a thin
    margin means the rig is noisy rather than that the number is wrong.
    """
    path = Path(video)
    out = ControlCalibration(source=str(path))
    if progress:
        progress(0, len(K_SWEEP) + 1, "extracting the angle trace once…")

    # Extract once. The sweep only changes the threshold, which is applied to
    # the envelope, so re-extracting per step would be the same work 13 times.
    sig = analysis.extract_signals(path, cfg, fps=fps_override)
    out.fps, out.duration_s = sig.fps, sig.n / max(sig.fps, 1e-9)
    if sig.n < int(max(sig.fps, 1.0) * 5):
        out.message = ("the control recording is under five seconds long; "
                       "a threshold fitted to it would mean nothing")
        return out

    best_zero = None
    for i, k in enumerate(K_SWEEP, start=1):
        if should_cancel is not None and should_cancel():
            out.message = "cancelled"
            return out
        if progress:
            progress(i, len(K_SWEEP) + 1, f"threshold k = {k:g}…")
        trial = analysis.AnalysisConfig.from_dict({**cfg.to_dict(),
                                                   "threshold_k": k})
        det = analysis.detect(sig, trial)
        n = len([b for b in det.bursts if b.accepted])
        out.sweep.append((k, n))
        if det.envelope is not None and out.peak_envelope == 0.0:
            finite = np.asarray(det.envelope, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                out.peak_envelope = float(np.max(finite))
        if n == 0 and best_zero is None:
            best_zero = (k, det.threshold)

    if best_zero is None:
        hardest_k, hardest_n = out.sweep[-1]
        out.message = (
            f"even at k = {hardest_k:g} this control recording still yields "
            f"{hardest_n} call(s). That is not a threshold problem: something "
            "in this footage is producing movement in the detection band. "
            "Run the flicker test on the rig, and open these calls in the "
            "Review tab - if they are real twitches the animal was not "
            "untreated, and if they are not, the tracking is the place to "
            "look, not the threshold")
        out.cfg = cfg.to_dict()
        return out

    out.zero_k = best_zero[0]
    ks = [k for k, _n in out.sweep]
    idx = ks.index(out.zero_k)
    # One step more sensitive than the zero point, unless zero came at the
    # most permissive k the sweep covers -- in which case there is nothing
    # below it and the rig is simply quiet.
    out.chosen_k = ks[max(0, idx - 1)]
    out.calls_at_chosen = dict(out.sweep)[out.chosen_k]

    chosen = analysis.AnalysisConfig.from_dict({**cfg.to_dict(),
                                                "threshold_k": out.chosen_k})
    det = analysis.detect(sig, chosen)
    out.threshold_at_chosen = float(det.threshold)
    if out.peak_envelope > 0:
        out.margin = out.threshold_at_chosen / out.peak_envelope
    out.cfg = chosen.to_dict()
    out.ok = True

    mins = out.duration_s / 60.0
    rate = (out.calls_at_chosen / mins * 30.0) if mins > 0 else 0.0
    if out.chosen_k == out.zero_k:
        head = (f"k = {out.chosen_k:g} is the most permissive setting swept "
                "and this control is already silent at it, so the rig is "
                "quiet; nothing was backed off")
    else:
        head = (f"k = {out.chosen_k:g}, one step below the {out.zero_k:g} at "
                "which this control falls silent")
    out.message = (
        f"{head}. At the chosen setting the control yields "
        f"{out.calls_at_chosen} call(s) in {mins:.1f} min, about "
        f"{rate:.1f} per 30 min of false positives. The threshold sits at "
        f"{out.threshold_at_chosen:.1f} deg/s against a tallest control peak "
        f"of {out.peak_envelope:.1f} deg/s"
        + (f" - a margin of {out.margin:.2f}x, which is comfortable"
           if out.margin >= 1.3 else
           f" - a margin of only {out.margin:.2f}x, so this rig is noisy and "
           "the calls on the treated recordings are worth checking by hand"
           if out.margin > 0 else "")
        + ". Apply this to the treated recordings of the same session only: "
        "it is fitted to this lighting, exposure, zoom and arena.")
    return out


# ==========================================================================
# Session profiles
# ==========================================================================
def profiles_dir() -> Path:
    d = paths.DATA_DIR / "profiles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_profile(name: str, cfg: analysis.AnalysisConfig,
                 cal: "ControlCalibration | None" = None,
                 advice: "Advice | None" = None) -> Path:
    """Write a named settings profile, with how it was arrived at.

    The provenance is not decoration. A profile is going to be applied to
    every treated recording in a session and its threshold will decide the
    counts that end up in a figure, so what it was fitted to -- which control
    file, which sweep, what margin -- has to travel with it.
    """
    safe = "".join(ch if (ch.isalnum() or ch in "-_ ") else "_"
                   for ch in name).strip() or "profile"
    doc = {
        "name": name,
        "saved_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": cfg.to_dict(),
        "control_calibration": cal.to_dict() if cal is not None else None,
        "file_advice": advice.provenance() if advice is not None else None,
    }
    p = profiles_dir() / f"{safe}.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(p)
    log.info("saved session profile to %s", p)
    return p


def list_profiles() -> list[dict]:
    out = []
    for p in sorted(profiles_dir().glob("*.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        doc["path"] = str(p)
        out.append(doc)
    return out


def load_profile(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
