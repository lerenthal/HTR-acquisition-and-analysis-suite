"""Exports: per-trial CSV, study-level summary CSV, and report figures.

Three things a lab actually needs out of an HTR run:

1. **An event table** -- one row per detected twitch, with the numbers that let
   somebody else check the call (duration, dominant frequency, half-cycles,
   whether a human accepted it).
2. **A rate time course** -- twitches per minute against minutes from dose. This
   is the study endpoint; it goes straight into a mixed-effects model.
3. **A figure to look at** -- because a number nobody eyeballed is a number
   nobody trusts. The figures carry the detection threshold and the tracking
   quality so a bad trial is visible at a glance.

The per-trial CSV is deliberately two tables in one file (events, then bins)
separated by a blank line, which is what the first-generation pipeline wrote and
what the existing spreadsheets expect. A machine-readable one-table-per-file pair
is written alongside it for anyone scripting against the output.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from . import theme
from .analysis import Burst, DetectionResult, Signals, ValidationResult, bin_counts

log = logging.getLogger(__name__)

EVENT_COLUMNS = [
    "burst_id", "start_s", "end_s", "center_s", "center_min_rel_t0",
    "duration_ms", "dominant_hz", "half_cycles", "peak_velocity_deg_s",
    "peak_envelope", "invalid_frac", "detector", "accepted", "manual", "note",
]
BIN_COLUMNS = ["bin_index", "start_s", "end_s", "bin_rel_t0", "count", "rate_per_min"]


@dataclass
class TrialReport:
    video: Path
    out_dir: Path
    bursts_csv: Path | None = None
    bins_csv: Path | None = None
    combined_csv: Path | None = None
    figure_png: Path | None = None
    traces_npz: Path | None = None


def _event_rows(bursts: Sequence[Burst], t0_s: float, bin_s: float
                ) -> list[list]:
    rows = []
    for i, b in enumerate(sorted(bursts, key=lambda x: x.start_s), 1):
        rel_min = (b.center_s - max(t0_s, 0.0)) / 60.0
        rows.append([
            i, f"{b.start_s:.4f}", f"{b.end_s:.4f}", f"{b.center_s:.4f}",
            f"{rel_min:.3f}", f"{b.duration_ms:.1f}", f"{b.dominant_hz:.1f}",
            b.half_cycles, f"{b.peak_velocity_deg_s:.1f}",
            f"{b.peak_envelope:.3f}", f"{b.invalid_frac:.3f}", b.detector,
            int(bool(b.accepted)), int(bool(b.manual)), b.note,
        ])
    return rows


def write_trial_csv(video: Path, out_dir: Path, sig: Signals,
                    det: DetectionResult, bin_s: float = 60.0,
                    t0_s: float = 0.0, meta: dict | None = None,
                    include_rejected: bool = True) -> TrialReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video.stem
    rep = TrialReport(video=video, out_dir=out_dir)

    accepted = [b for b in det.bursts if b.accepted]
    bins = bin_counts(accepted, sig.duration_s, bin_s, t0_s)

    # --- the combined, human-facing file ---
    rep.combined_csv = out_dir / f"{stem}_htr_report.csv"
    with open(rep.combined_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# HTR Studio trial report"])
        w.writerow(["# video", str(video)])
        w.writerow(["# generated", datetime.now().astimezone().isoformat(timespec="seconds")])
        w.writerow(["# frames", sig.n, "fps", f"{sig.fps:.4f}",
                    "duration_s", f"{sig.duration_s:.3f}"])
        w.writerow(["# tracking_valid_pct", f"{sig.valid_frac * 100:.2f}"])
        w.writerow(["# detector", det.detector, "threshold", f"{det.threshold:.4f}"])
        w.writerow(["# threshold_basis", det.diagnostics.get("threshold_note", "")])
        w.writerow(["# t0_offset_s", f"{t0_s:.3f}",
                    "(bin_rel_t0 = 0 is the first bin after dosing)"])
        for k, v in (meta or {}).items():
            if v not in ("", None):
                w.writerow([f"# {k}", v])
        for warn in det.warnings + sig.warnings:
            w.writerow(["# WARNING", warn])
        w.writerow([])
        w.writerow(["TOTAL_ACCEPTED_TWITCHES", len(accepted)])
        w.writerow([])
        w.writerow(EVENT_COLUMNS)
        for row in _event_rows(accepted, t0_s, bin_s):
            w.writerow(row)
        if include_rejected and det.rejected:
            w.writerow([])
            w.writerow([f"# {len(det.rejected)} candidate(s) rejected by the gates, "
                        "listed for audit"])
            w.writerow(EVENT_COLUMNS)
            for row in _event_rows(det.rejected, t0_s, bin_s):
                w.writerow(row)
        w.writerow([])
        w.writerow(BIN_COLUMNS)
        for b in bins:
            w.writerow([b[c] for c in BIN_COLUMNS])

    # --- machine-readable single-table files ---
    rep.bursts_csv = out_dir / f"{stem}_htr_events.csv"
    with open(rep.bursts_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(EVENT_COLUMNS)
        for row in _event_rows(accepted, t0_s, bin_s):
            w.writerow(row)

    rep.bins_csv = out_dir / f"{stem}_htr_rate.csv"
    with open(rep.bins_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(BIN_COLUMNS)
        for b in bins:
            w.writerow([b[c] for c in BIN_COLUMNS])

    log.info("wrote trial report for %s (%d accepted events)", video.name, len(accepted))
    return rep


def save_traces(path: Path, sig: Signals, det: DetectionResult,
                decimate: int = 1) -> Path:
    """Compressed per-frame traces, for re-plotting without re-analysing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    s = max(1, int(decimate))
    arrays = {
        "t": sig.t[::s].astype(np.float32),
        "angle": sig.angle[::s].astype(np.float32),
        "valid": sig.valid[::s],
        "motion_head": sig.motion_head[::s],
        "motion_bg": sig.motion_bg[::s],
        "fps": np.array([sig.fps]),
        "decimate": np.array([s]),
    }
    if det.filtered is not None:
        arrays["filtered"] = det.filtered[::s].astype(np.float32)
    if det.envelope is not None:
        arrays["envelope"] = det.envelope[::s].astype(np.float32)
    if det.spec_t is not None and det.spec_ratio is not None:
        arrays["spec_t"] = det.spec_t.astype(np.float32)
        arrays["spec_ratio"] = det.spec_ratio.astype(np.float32)
    np.savez_compressed(path, **arrays)
    return path


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _decimate(t: np.ndarray, y: np.ndarray, max_points: int = 6000
              ) -> tuple[np.ndarray, np.ndarray]:
    """Min/max envelope decimation, so a 1-million-sample trace stays honest.

    Plain subsampling of a 90 Hz oscillation at 1-in-200 would make the bursts
    disappear from the figure. Keeping the min and max of each block preserves
    the visible envelope.
    """
    n = len(y)
    if n <= max_points:
        return t, y
    block = int(np.ceil(n / (max_points / 2)))
    usable = (n // block) * block
    yb = y[:usable].reshape(-1, block)
    tb = t[:usable].reshape(-1, block)
    lo, hi = yb.min(axis=1), yb.max(axis=1)
    tt = tb[:, block // 2]
    out_t = np.repeat(tt, 2)
    out_y = np.empty(out_t.shape, dtype=y.dtype)
    out_y[0::2], out_y[1::2] = lo, hi
    return out_t, out_y


def _bin_label(bin_s: float) -> str:
    """'min' reads wrong for a 10-second bin; say seconds when it is seconds."""
    return f"{bin_s / 60:g} min" if bin_s >= 60 else f"{bin_s:g} s"


def plot_trial(png: Path, sig: Signals, det: DetectionResult, bin_s: float = 60.0,
               t0_s: float = 0.0, title: str = "", theme_name: str = "light"
               ) -> Path:
    """Three stacked panels sharing one x-axis -- never two y-scales on one panel.

    Top: the HTR rate time course, which is the study endpoint.
    Middle: the band-pass envelope with its threshold, so the reader can see how
    far above the noise each call sits.
    Bottom: the head-angle trace, with the stretches where tracking was lost
    shaded, because a burst sitting on lost tracking is not a burst.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    pal = theme.palette(theme_name)
    theme.apply_matplotlib(pal)

    accepted = [b for b in det.bursts if b.accepted]
    bins = bin_counts(accepted, sig.duration_s, bin_s, t0_s)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1.0, 1.0],
                                          "hspace": 0.32})
    ax_rate, ax_env, ax_ang = axes

    # ---- panel 1: rate time course (one series, so no legend box) ----
    centers = [(b["start_s"] + b["end_s"]) / 2 for b in bins]
    counts = [b["count"] for b in bins]
    width = bin_s * 0.82
    bars = ax_rate.bar(centers, counts, width=width, color=pal.s1,
                       edgecolor=pal.surface, linewidth=2.0)
    ax_rate.set_title(f"Head twitches per {_bin_label(bin_s)}"
                      f"   ·   {len(accepted)} total", loc="left")
    ax_rate.set_ylabel("twitches")
    ax_rate.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    # Label only the peaks: a number on every bar is noise.
    if counts:
        peak = max(counts)
        for rect, c in zip(bars, counts):
            if c and (c == peak or c >= 0.8 * peak):
                ax_rate.annotate(str(c), (rect.get_x() + rect.get_width() / 2, c),
                                 textcoords="offset points", xytext=(0, 3),
                                 ha="center", fontsize=8, color=pal.ink_secondary)

    # ---- panel 2: envelope + threshold ----
    if det.envelope is not None:
        te, ye = _decimate(sig.t, det.envelope)
        ax_env.plot(te, ye, color=pal.s2, linewidth=1.2)
        ax_env.axhline(det.threshold, color=pal.critical, linewidth=1.6,
                       linestyle="--")
        ax_env.annotate(f"detection threshold  {det.threshold:.0f}",
                        (sig.duration_s, det.threshold), xytext=(-6, 6),
                        textcoords="offset points", ha="right", fontsize=8,
                        color=pal.critical)
        lo = det.diagnostics.get("threshold_low")
        if lo:
            ax_env.axhline(lo, color=pal.serious, linewidth=1.0, linestyle=":")
            # Anchor this one on the left: at 50% of the threshold the two
            # labels would otherwise sit on top of each other.
            ax_env.annotate(f"burst release  {lo:.0f}", (0.0, lo), xytext=(6, -11),
                            textcoords="offset points", ha="left", fontsize=8,
                            color=pal.ink_muted)
            # Room under the axis so that label is not clipped.
            top = ax_env.get_ylim()[1]
            ax_env.set_ylim(bottom=-0.07 * top, top=top)
        ax_env.set_ylabel("deg/s")
        ax_env.set_title(f"Band-pass envelope, {det.detector}", loc="left")
    elif det.spec_ratio is not None:
        ax_env.plot(det.spec_t, det.spec_ratio, color=pal.s2, linewidth=1.2)
        ax_env.axhline(det.threshold, color=pal.critical, linewidth=1.6, linestyle="--")
        ax_env.set_ylabel("band-power ratio")
        ax_env.set_title("Motion-energy band-power ratio", loc="left")

    # ---- panel 3: head angle, with lost tracking shaded ----
    ta, ya = _decimate(sig.t, sig.angle)
    ax_ang.plot(ta, ya, color=pal.s1, linewidth=1.0)
    ax_ang.set_ylabel("head angle (deg)")
    ax_ang.set_xlabel("time from start of recording (s)")
    ax_ang.set_title(f"Head angle · {sig.angle_source} · "
                     f"tracking valid {sig.valid_frac * 100:.1f}%", loc="left")
    if sig.n and not sig.valid.all():
        from scipy import ndimage as ndi
        lab, nlab = ndi.label(~sig.valid)
        for sl in ndi.find_objects(lab)[:400]:
            a, b = sl[0].start / sig.fps, sl[0].stop / sig.fps
            if b - a > 0.05:
                ax_ang.axvspan(a, b, color=pal.ink_muted, alpha=0.18, linewidth=0)

    # ---- shared annotations ----
    for ax in (ax_env, ax_ang):
        for b in accepted:
            ax.axvspan(b.start_s, b.end_s, color=pal.s3, alpha=0.30, linewidth=0)
    if t0_s > 0:
        for ax in axes:
            ax.axvline(t0_s, color=pal.ink, linewidth=1.4, linestyle="-", alpha=0.8)
        ax_rate.annotate("T0  dose", (t0_s, ax_rate.get_ylim()[1]),
                         xytext=(4, -10), textcoords="offset points",
                         fontsize=8, color=pal.ink, ha="left")

    head = title or sig.source_path
    fig.suptitle(head, x=0.012, ha="left", fontsize=13, fontweight="bold",
                 color=pal.ink)
    note = ("shaded spans = accepted twitch bursts (aqua); grey spans = tracking lost"
            if sig.n and not sig.valid.all()
            else "shaded spans = accepted twitch bursts (aqua)")
    fig.text(0.012, 0.005, note, fontsize=8, color=pal.ink_muted, ha="left")

    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png)
    plt.close(fig)
    log.info("wrote figure %s", png)
    return png


# --------------------------------------------------------------------------
# Study-level summary
# --------------------------------------------------------------------------
STUDY_COLUMNS = [
    "video", "animal_id", "group", "treatment", "dose_mg_kg", "route",
    "duration_min", "fps", "tracking_valid_pct", "detector", "threshold",
    "t0_offset_s", "total_twitches", "baseline_twitches", "post_dose_twitches",
    "peak_rate_per_min", "peak_minute_rel_t0", "latency_to_first_min_rel_t0",
    "curated", "warnings",
]


def study_row(video: Path, sig_summary: dict, det: DetectionResult,
              bursts: Sequence[Burst], t0_s: float, bin_s: float,
              trial_meta: dict, curated: bool = False) -> dict:
    accepted = [b for b in bursts if b.accepted]
    duration = float(sig_summary.get("duration_s", 0.0) or 0.0)
    bins = bin_counts(accepted, duration, bin_s, t0_s)
    post = [b for b in accepted if b.center_s >= t0_s] if t0_s > 0 else accepted
    pre = [b for b in accepted if b.center_s < t0_s] if t0_s > 0 else []
    peak = max(bins, key=lambda b: b["count"]) if bins else None
    first_post = min((b.center_s for b in post), default=None)
    return {
        "video": str(video),
        "animal_id": trial_meta.get("animal_id", ""),
        "group": trial_meta.get("group", ""),
        "treatment": trial_meta.get("treatment", ""),
        "dose_mg_kg": trial_meta.get("dose_mg_kg", ""),
        "route": trial_meta.get("route", ""),
        "duration_min": round(duration / 60.0, 2),
        "fps": sig_summary.get("fps", ""),
        "tracking_valid_pct": sig_summary.get("tracking_valid_pct", ""),
        "detector": det.detector,
        "threshold": round(det.threshold, 4),
        "t0_offset_s": round(t0_s, 2),
        "total_twitches": len(accepted),
        "baseline_twitches": len(pre),
        "post_dose_twitches": len(post),
        "peak_rate_per_min": peak["rate_per_min"] if peak else 0,
        "peak_minute_rel_t0": peak["bin_rel_t0"] if peak else "",
        "latency_to_first_min_rel_t0": (round((first_post - t0_s) / 60.0, 2)
                                        if first_post is not None else ""),
        "curated": int(bool(curated)),
        "warnings": " | ".join(det.warnings[:3]),
    }


def write_study_summary(path: Path, rows: Iterable[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=STUDY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log.info("wrote study summary %s (%d trials)", path, len(rows))
    return path


def plot_study(png: Path, rows: Sequence[dict], rate_csvs: dict[str, Path],
               bin_s: float = 60.0, theme_name: str = "light") -> Path | None:
    """Group mean rate time course, one line per group, at most three groups.

    Three is the cap on purpose: past three series the categorical slots stop
    clearing the colour-vision separation floor, and a reader with deuteranopia
    could not tell two of the lines apart. A fourth group gets its own figure
    rather than a fourth hue.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pal = theme.palette(theme_name)
    theme.apply_matplotlib(pal)

    series: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for r in rows:
        grp = str(r.get("group") or "ungrouped")
        p = rate_csvs.get(str(r.get("video")))
        if not p or not Path(p).exists():
            continue
        xs, ys = [], []
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    xs.append(float(row["bin_rel_t0"]))
                    ys.append(float(row["rate_per_min"]))
                except (KeyError, TypeError, ValueError):
                    continue
        if xs:
            series.setdefault(grp, []).append((np.array(xs), np.array(ys)))
    if not series:
        return None

    groups = sorted(series)[:3]
    fig, ax = plt.subplots(figsize=(10, 5))
    label_points: list[tuple[float, float, str, str]] = []
    for i, grp in enumerate(groups):
        trials = series[grp]
        grid = np.arange(min(t.min() for t, _ in trials),
                         max(t.max() for t, _ in trials) + 1)
        stack = np.vstack([np.interp(grid, t, y, left=np.nan, right=np.nan)
                           for t, y in trials])
        mean = np.nanmean(stack, axis=0)
        n = np.sum(~np.isnan(stack), axis=0)
        sem = np.nanstd(stack, axis=0) / np.sqrt(np.maximum(n, 1))
        color = pal.series[i]
        ax.plot(grid, mean, color=color, linewidth=2.0,
                label=f"{grp} (n={len(trials)})")
        ax.fill_between(grid, mean - sem, mean + sem, color=color, alpha=0.18,
                        linewidth=0)
        last = np.flatnonzero(~np.isnan(mean))
        if last.size:
            j = last[-1]
            label_points.append((float(grid[j]), float(mean[j]), grp, color))

    # Direct labels as well as the legend, so identity is never carried by
    # colour alone. Two groups that end at the same value would print on top of
    # each other, so nudge them apart vertically first.
    if label_points:
        lo, hi = ax.get_ylim()
        gap = 0.07 * (hi - lo)
        label_points.sort(key=lambda p: p[1])
        placed_y: list[float] = []
        for x, y, grp, color in label_points:
            while placed_y and y - placed_y[-1] < gap:
                y = placed_y[-1] + gap
            placed_y.append(y)
            ax.annotate(grp, (x, y), xytext=(8, 0), textcoords="offset points",
                        color=color, fontsize=9, va="center", fontweight="bold")

    ax.axvline(0, color=pal.ink, linewidth=1.4, alpha=0.8)
    ax.annotate("T0  dose", (0, ax.get_ylim()[1]), xytext=(4, -10),
                textcoords="offset points", fontsize=8, color=pal.ink)
    ax.set_xlabel(f"time from dose ({_bin_label(bin_s)} bins)")
    ax.set_ylabel("twitches per min")
    ax.set_title("HTR rate time course by group  ·  mean ± SEM", loc="left")
    ax.legend(loc="upper right")
    if len(series) > 3:
        fig.text(0.012, 0.005,
                 f"{len(series) - 3} further group(s) omitted - at most three "
                 "series share one panel so the colours stay distinguishable",
                 fontsize=8, color=pal.ink_muted, ha="left")

    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png)
    plt.close(fig)
    return png


def write_validation_report(path: Path, val: ValidationResult, label: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# HTR Studio detector validation", label])
        w.writerow(["tolerance_s", val.tolerance_s])
        w.writerow(["true_positives", val.tp])
        w.writerow(["false_positives", val.fp])
        w.writerow(["false_negatives", val.fn])
        w.writerow(["precision", round(val.precision, 4)])
        w.writerow(["recall", round(val.recall, 4)])
        w.writerow(["f1", round(val.f1, 4)])
        w.writerow([])
        w.writerow(["missed_start_s", "missed_end_s"])
        for a, b in val.missed:
            w.writerow([f"{a:.4f}", f"{b:.4f}"])
        w.writerow([])
        w.writerow(["spurious_center_s"])
        for c in val.spurious:
            w.writerow([f"{c:.4f}"])
    return path
