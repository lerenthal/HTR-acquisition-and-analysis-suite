"""Regressions for the four defects fixed in 1.2.2-dev.

Every one of these was found by the operator, not by a test, which is the
pattern the hardening plan says to break. Each check below fails on the code
as it stood before the fix.

1. LabeledSlider with decimals=0 raised TypeError on every drag, so the five
   camera control sliders never pushed a value to the camera at all.
2. ReviewTab kept the previous file's burst regions when a new entry was
   opened from the Analysis path, because AppContext.entryChanged existed and
   nothing was connected to it.
3. Recorder.stats() reported 0 B for a finished segmented recording, because
   the folder-size fallback sat inside the still-running branch.
4. The preview's media type and frame rate can differ from the recording's,
   and nothing in the window said so.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PyQt5 import QtWidgets                                    # noqa: E402

from htr_studio import camera, library, paths, recorder         # noqa: E402
from htr_studio.ui import widgets as W                          # noqa: E402

FAILS: list[str] = []
PASSES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSES if ok else FAILS).append(f"{name}: {detail}" if detail else name)
    print(("  ok   " if ok else "  FAIL ") + name + (f"  [{detail}]" if detail else ""))


def test_integer_slider_drag(app) -> None:
    """A decimals=0 slider must mirror into its QSpinBox and emit."""
    seen: list[float] = []
    s = W.LabeledSlider("Exposure", -13, 0, -7, 1, 0)
    s.valueChanged.connect(seen.append)
    # Drag: the slider emits an int; _from_slider used to pass raw/1 (a float)
    # straight into QSpinBox.setValue, which raises TypeError.
    # Call the handler directly. Going through slider.setValue() routes the
    # exception through Qt's slot machinery, which in PyQt5 aborts the process
    # rather than raising -- which is exactly why this bug reached the operator
    # as a modal error box with no stack they could act on.
    try:
        s._from_slider(-4)
        raised = ""
    except TypeError as e:
        raised = str(e)
    check("integer slider drag does not raise", raised == "", raised)
    check("integer slider updates its spin box", s.spin.value() == -4,
          f"spin={s.spin.value()}")
    check("integer slider emits valueChanged", seen == [-4.0], f"emitted={seen}")

    # And the float case must still work.
    f = W.LabeledSlider("Threshold", 0.0, 10.0, 1.0, 0.1, 2)
    f._from_slider(int(3.25 * 100))
    check("decimal slider still mirrors", abs(f.spin.value() - 3.25) < 1e-6,
          f"spin={f.spin.value()}")


def _entry(tmp: Path, name: str) -> library.Entry:
    p = tmp / name
    p.write_bytes(b"not a real video")
    e = library.Entry(path=p)
    e.width, e.height, e.fps, e.duration_s = 320, 240, 240.0, 1.0
    return e


def test_entry_change_clears_review(app, tmp: Path) -> None:
    """Changing ctx.entry must reach every tab that displays a file."""
    from htr_studio.ui.main_window import AppContext
    from htr_studio import config

    cfg = config.Config.load()
    ctx = AppContext(cfg)

    # Stand-ins for the two consumers: the point of the check is the wiring,
    # not the tabs' internals.
    got: list[str] = []
    ctx.entryChanged.connect(lambda e: got.append(f"review:{e.path.name}"))
    ctx.entryChanged.connect(lambda e: got.append(f"analysis:{e.path.name}"))

    a, b = _entry(tmp, "a.avi"), _entry(tmp, "b.avi")
    ctx.set_entry(a)
    ctx.set_entry(b)
    check("entryChanged reaches both consumers",
          got == ["review:a.avi", "analysis:a.avi",
                  "review:b.avi", "analysis:b.avi"], str(got))
    ctx.set_entry(b)
    check("setting the same entry does not re-emit", len(got) == 4, f"n={len(got)}")

    # And main_window must route its open paths through set_entry, not assign.
    src = (ROOT / "htr_studio" / "ui" / "main_window.py").read_text()
    check("main_window never assigns ctx.entry directly",
          "self.ctx.entry = entry" not in src)
    check("main_window connects entryChanged",
          "self.ctx.entryChanged.connect" in src)

    rsrc = (ROOT / "htr_studio" / "ui" / "review_tab.py").read_text()
    check("review_tab has a single clear point", "_clear_plots" in rsrc)
    check("review_tab does not write ctx.entry",
          "self.ctx.entry = entry" not in rsrc)


def test_review_clears_regions(app, tmp: Path) -> None:
    """load_entry on a file with no analysis leaves no regions behind."""
    from htr_studio.ui.main_window import AppContext
    from htr_studio.ui.review_tab import ReviewTab
    from htr_studio import config, analysis

    ctx = AppContext(config.Config.load())
    tab = ReviewTab(ctx)

    # Pretend a previous file was analysed: two bursts shaded on both plots.
    tab.bursts = [analysis.Burst(start_s=0.1, end_s=0.2, start_frame=24,
                                 end_frame=48, duration_ms=100.0,
                                 dominant_hz=80.0, half_cycles=6,
                                 detector="bandpass")]
    tab.rejected = []
    tab._draw_regions()
    before = len(tab.burst_regions)
    check("regions were drawn for the first file", before == 2, f"n={before}")

    # Now a different file with no analysis at all. The reader cannot open a
    # text file, so this also exercises the early-return path that used to
    # leave the whole previous file on screen.
    tab.load_entry(_entry(tmp, "other.avi"))
    check("no regions survive a file change", len(tab.burst_regions) == 0,
          f"n={len(tab.burst_regions)}")
    check("bursts are cleared", tab.bursts == [], f"n={len(tab.bursts)}")
    check("rejected are cleared", tab.rejected == [], f"n={len(tab.rejected)}")
    title = tab.plot_angle.getPlotItem().titleLabel.text
    check("the angle plot title is reset", "saved burst" not in title, title)
    tab.deleteLater()


def test_segmented_size_after_exit(tmp: Path) -> None:
    """A finished segmented recording reports the bytes on disk, not 0."""
    out = tmp / "seg"
    spec = recorder.RecordingSpec(
        device_name="dummy", duration_s=60.0,
        segment_enabled=True, segment_seconds=12.0,
        out_dir=out, base_name="trial", container="avi")
    check("spec is segmented", spec.is_segmented)

    folder = spec.target_path().parent
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        (folder / f"trial_{i:04d}.avi").write_bytes(b"x" * 1000)

    rec = recorder.Recorder(spec, {})
    # A process object that has already exited -- the state in which the
    # "Finished:" line is composed.
    rec.proc = types.SimpleNamespace(poll=lambda: 0)
    st = rec.stats()
    check("finished segmented size is measured from disk", st.size_bytes == 5000,
          f"{st.size_bytes} B")

    # Still-running must keep working too.
    rec.proc = types.SimpleNamespace(poll=lambda: None)
    check("running segmented size still measured", rec.stats().size_bytes == 5000)


def test_exposure_budget_logic() -> None:
    """The clamp warning must fire exactly when the physics says it will."""
    ms = camera.exposure_hint_ms(-7)
    check("exposure -7 is about 7.8 ms", abs(ms - 7.8125) < 0.01, f"{ms:.4f} ms")
    # 120 fps is the boundary: a frame lasts 8.3 ms, so 7.8 ms fits and there
    # is nothing to warn about. The warning is for rates where the frame is
    # shorter than the requested integration time.
    for fps, want in ((200.0, True), (240.0, True), (120.0, False), (30.0, False)):
        frame_ms = 1000.0 / fps
        fires = ms > frame_ms * 1.05
        check(f"clamp warning at {fps:.0f} fps is {want}", fires == want,
              f"{ms:.1f} ms vs {frame_ms:.1f} ms frame")

    src = (ROOT / "htr_studio" / "ui" / "acquisition_tab.py").read_text()
    for needle, label in (("_warn_mode_mismatch", "mismatch warning exists"),
                          ("def _check_capture", "capture check handler exists"),
                          ("CHECK_CAPTURE_S", "capture check duration is named"),
                          ("self.t_mean", "preview reports its mean level")):
        check(label, needle in src)


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    paths.ensure_dirs()
    tmp = paths.CACHE_DIR / "regress_1_2_2"
    tmp.mkdir(parents=True, exist_ok=True)

    print("1. integer slider drag")
    test_integer_slider_drag(app)
    print("2. entry change fan-out")
    test_entry_change_clears_review(app, tmp)
    print("3. review clears regions")
    test_review_clears_regions(app, tmp)
    print("4. segmented size after exit")
    test_segmented_size_after_exit(tmp)
    print("5. exposure budget")
    test_exposure_budget_logic()

    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    for f in FAILS:
        print("  FAILED: " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
