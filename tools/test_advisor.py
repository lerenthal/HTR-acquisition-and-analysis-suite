"""Regressions for the settings advisor, the reset, and control calibration.

The arithmetic here is the part worth pinning. A folded band-pass band looks
wrong to anyone who has not worked through the aliasing, so if it ever
silently changes, nobody will notice by reading the settings -- only by
getting the wrong counts months later.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                             # noqa: E402

from htr_studio import advisor, analysis, config                # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("  ok   " if ok else "  FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def test_folding() -> None:
    """Where a frequency lands when it is sampled too slowly."""
    for f, fps, want in ((88, 240, 88), (88, 120, 32), (88, 200, 88),
                         (98, 120, 22), (78, 120, 42), (88, 100, 12),
                         (88, 60, 28), (30, 240, 30)):
        got = advisor.fold_hz(float(f), float(fps))
        check(f"{f} Hz at {fps} fps folds to {want}", abs(got - want) < 0.01,
              f"{got:.2f}")

    # Folding reflects at every multiple of fps/2, so an interval that
    # straddles one has an image wider than its endpoints. Endpoint-only
    # arithmetic gives 20-20 here and would propose a band containing almost
    # none of the signal.
    lo, hi = advisor.fold_band(120.0, 100.0, 140.0)
    check("a band crossing Nyquist folds to its true image",
          abs(lo - 0.0) < 0.01 and abs(hi - 20.0) < 0.01, f"{lo:.1f}-{hi:.1f}")
    check("endpoint arithmetic would have missed it",
          advisor.fold_hz(100, 120) == advisor.fold_hz(140, 120) == 20)


def test_band_for_rate() -> None:
    """The proposed band must contain the signal and stay under Nyquist."""
    for fps in (480.0, 240.0, 200.0, 180.0, 120.0, 100.0, 60.0, 30.0):
        band, why = advisor.band_for_rate(fps)
        nyq = fps / 2.0
        check(f"{fps:.0f} fps: band is ordered and under Nyquist",
              band[0] < band[1] <= nyq, f"{band[0]:.1f}-{band[1]:.1f} (nyq {nyq:.0f})")
        check(f"{fps:.0f} fps: the reasoning is stated", len(why) > 40)
        # The folded centre of the real band must be inside the proposal.
        centre = advisor.fold_hz(88.0, fps)
        check(f"{fps:.0f} fps: the folded 88 Hz centre is in the band",
              band[0] <= centre <= band[1],
              f"centre {centre:.1f} in {band[0]:.1f}-{band[1]:.1f}")

    # The 240 fps rig must keep the band the benchmarks were validated with.
    band, _ = advisor.band_for_rate(240.0)
    check("240 fps keeps the validated 60-110 Hz band", band == (60.0, 110.0),
          str(band))
    # 120 fps must NOT be told to filter where the twitch really is.
    band, _ = advisor.band_for_rate(120.0)
    check("120 fps does not propose the unfolded band", band[1] < 60.0,
          f"{band[0]:.1f}-{band[1]:.1f}")


def test_config_defaults_derive_from_dataclass() -> None:
    """The settings defaults must not be a second copy that can drift."""
    d = config.DEFAULTS["analysis"]
    shipped = analysis.AnalysisConfig()
    drift = []
    for f in shipped.__dataclass_fields__:
        sv = getattr(shipped, f)
        if isinstance(sv, tuple):
            sv = list(sv)
        if f not in d:
            drift.append(f"{f} missing")
        elif d[f] != sv:
            drift.append(f"{f}: {d[f]!r} != {sv!r}")
    check("every analysis default matches the dataclass", not drift,
          "; ".join(drift[:4]))
    check("max_area_frac is the corrected 0.25, not 0.45",
          d["max_area_frac"] == 0.25, str(d["max_area_frac"]))


def test_migration() -> None:
    """A rig that already has a settings file must get corrected defaults."""
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "settings.json"
        f.write_text(json.dumps({
            "schema_version": 2,
            "analysis": {"max_area_frac": 0.45, "core_kernel": 9,
                         "threshold_k": 4.5}}))
        c = config.Config.load(f)
        check("a stale 0.45 ceiling is corrected",
              c.get("analysis.max_area_frac") == 0.25,
              str(c.get("analysis.max_area_frac")))
        check("a stale fixed tail kernel is corrected",
              c.get("analysis.core_kernel") == 0,
              str(c.get("analysis.core_kernel")))
        check("the operator's own threshold is preserved",
              c.get("analysis.threshold_k") == 4.5,
              str(c.get("analysis.threshold_k")))
        check("the schema version is bumped",
              c.get("schema_version") == config.SCHEMA_VERSION,
              str(c.get("schema_version")))

    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "settings.json"
        f.write_text(json.dumps({
            "schema_version": 2,
            "analysis": {"max_area_frac": 0.33, "core_kernel": 15}}))
        c = config.Config.load(f)
        check("a deliberately chosen ceiling is left alone",
              c.get("analysis.max_area_frac") == 0.33,
              str(c.get("analysis.max_area_frac")))
        check("a deliberately chosen tail kernel is left alone",
              c.get("analysis.core_kernel") == 15,
              str(c.get("analysis.core_kernel")))

    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "settings.json"
        f.write_text(json.dumps({"schema_version": config.SCHEMA_VERSION,
                                 "analysis": {"max_area_frac": 0.45}}))
        c = config.Config.load(f)
        check("an already-migrated file is not touched again",
              c.get("analysis.max_area_frac") == 0.45,
              str(c.get("analysis.max_area_frac")))


def test_change_and_advice_objects() -> None:
    c = advisor.Change("band_low_hz", "low", 60.0, 60.0, "why")
    check("an unchanged value is not pending", not c.differs)
    c2 = advisor.Change("band_low_hz", "low", 60.0, 17.0, "why")
    check("a changed value is pending", c2.differs)
    c3 = advisor.Change("require_freq_hz", "freq", (55.0, 125.0),
                        [55.0, 125.0], "why")
    check("a tuple equal to a list is not pending", not c3.differs)

    base = analysis.AnalysisConfig()
    adv = advisor.Advice(fps=120.0)
    adv.changes = [c2, advisor.Change("min_half_cycles", "half", 6, 4, "why")]
    out = adv.apply_to(base)
    check("apply_to sets the suggested values",
          out.band_low_hz == 17.0 and out.min_half_cycles == 4,
          f"{out.band_low_hz} {out.min_half_cycles}")
    check("apply_to leaves everything else alone",
          out.band_high_hz == base.band_high_hz
          and out.threshold_k == base.threshold_k)
    check("the base config is not mutated", base.band_low_hz == 60.0)
    prov = adv.provenance()
    check("provenance carries the reasons",
          len(prov["changes"]) == 2 and all(c["why"] for c in prov["changes"]))


def test_control_sweep_logic() -> None:
    """The sweep picks one step below silence, on a synthetic envelope.

    Driven through analysis.detect with a hand-built Signals object, so the
    choice rule is tested without a 70-minute recording.
    """
    fps = 240.0
    n = int(fps * 30)
    t = np.arange(n) / fps
    rng = np.random.default_rng(7)
    # Noise only, no twitches: this is what a control recording looks like.
    angle = np.cumsum(rng.normal(0, 0.35, n))
    angle += 2.0 * np.sin(2 * np.pi * 2.0 * t)      # locomotion
    z = np.zeros(n)
    sig = analysis.Signals(
        source_path="synthetic-control", fps=fps, n=n, t=t,
        angle=angle, angle_raw=angle.copy(),
        valid=np.ones(n, bool), area=np.full(n, 0.01),
        cx=z.copy(), cy=z.copy(), hx=z.copy(), hy=z.copy(),
        motion_all=z.copy(), motion_head=z.copy(), motion_bg=z.copy(),
        angle_source="headpoint")
    cfg = analysis.AnalysisConfig()
    counts = []
    for k in advisor.K_SWEEP:
        trial = analysis.AnalysisConfig.from_dict({**cfg.to_dict(),
                                                   "threshold_k": k})
        det = analysis.detect(sig, trial)
        counts.append(len([b for b in det.bursts if b.accepted]))
    print(f"    sweep on synthetic control: {list(zip(advisor.K_SWEEP, counts))}")
    check("the sweep is monotonically non-increasing in k",
          all(a >= b for a, b in zip(counts, counts[1:])), str(counts))

    # And the choice rule, in isolation.
    sweep = [(3.0, 9), (4.0, 4), (5.0, 1), (6.0, 0), (7.0, 0)]
    ks = [k for k, _ in sweep]
    zero = next(k for k, c in sweep if c == 0)
    chosen = ks[max(0, ks.index(zero) - 1)]
    check("the chosen k is one step below silence", chosen == 5.0, str(chosen))
    sweep2 = [(3.0, 0), (4.0, 0)]
    ks2 = [k for k, _ in sweep2]
    zero2 = next(k for k, c in sweep2 if c == 0)
    check("a quiet rig keeps the most permissive k",
          ks2[max(0, ks2.index(zero2) - 1)] == 3.0)


def test_profiles() -> None:
    cfg = analysis.AnalysisConfig.from_dict(
        {**analysis.AnalysisConfig().to_dict(), "threshold_k": 4.5})
    cal = advisor.ControlCalibration(
        source="control.avi", chosen_k=4.5, zero_k=5.0, ok=True,
        sweep=[(4.0, 2), (4.5, 1), (5.0, 0)], margin=1.4,
        message="fitted on control.avi")
    p = advisor.save_profile("test session ☺ /unsafe", cfg, cal=cal)
    check("the profile file is written", p.is_file(), p.name)
    check("the filename is sanitised", "/" not in p.name, p.name)
    doc = advisor.load_profile(p)
    check("the config round-trips", doc["config"]["threshold_k"] == 4.5)
    check("the calibration travels with it",
          doc["control_calibration"]["chosen_k"] == 4.5
          and doc["control_calibration"]["sweep"] == [[4.0, 2], [4.5, 1], [5.0, 0]])
    check("it is listed", any(d["path"] == str(p) for d in advisor.list_profiles()))
    p.unlink()


def main() -> int:
    print("1. frequency folding")
    test_folding()
    print("2. band for a frame rate")
    test_band_for_rate()
    print("3. config defaults derive from the dataclass")
    test_config_defaults_derive_from_dataclass()
    print("4. settings migration")
    test_migration()
    print("5. change / advice objects")
    test_change_and_advice_objects()
    print("6. control sweep")
    test_control_sweep_logic()
    print("7. session profiles")
    test_profiles()
    print(f"\n{'ADVISOR OK' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
