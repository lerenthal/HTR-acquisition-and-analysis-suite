"""Self-test: prove the installation works before a real trial depends on it.

Run it after install.bat, after copying the folder to another machine, and after
changing any detector default. ``run_selftest.bat`` is the double-clickable
wrapper; the app's Tools menu runs the same thing.

What it checks, in order:

1. Every Python dependency imports, at a usable version.
2. FFmpeg and FFprobe are findable and runnable.
3. The GUI can be constructed (off-screen, so no window appears).
4. The analysis core recovers planted twitches from a synthetic video and
   ignores the planted confounders. This is the test that matters: it is an
   end-to-end measurement of precision and recall, not a smoke test.
5. The parallel extraction path produces signals identical to the
   single-threaded path.
6. Camera enumeration (reported, never failed - the analysis computer is not
   expected to have the camera attached).

Exit code 0 means every required check passed.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"
_results: list[tuple[str, str, str]] = []


def record(level: str, name: str, detail: str = "") -> None:
    _results.append((level, name, detail))
    mark = {PASS: "[ ok ]", FAIL: "[FAIL]", WARN: "[warn]", SKIP: "[skip]"}[level]
    print(f"{mark} {name}" + (f"\n         {detail}" if detail else ""))


# --------------------------------------------------------------------------
def check_imports() -> None:
    required = {
        "numpy": "1.26", "cv2": "4.8", "scipy": "1.10",
        "PyQt5": "5.15", "pyqtgraph": "0.13", "matplotlib": "3.7",
    }
    for mod, minimum in required.items():
        try:
            m = __import__(mod)
        except ImportError as e:
            record(FAIL, f"import {mod}",
                   f"{e}. Re-run install.bat; it installs everything in "
                   "requirements.txt into the .venv next to this folder.")
            continue
        if mod == "PyQt5":
            from PyQt5 import QtCore
            ver = QtCore.PYQT_VERSION_STR
        else:
            ver = getattr(m, "__version__", "?")
        ok = _version_at_least(ver, minimum)
        record(PASS if ok else WARN, f"import {mod}",
               f"{ver}" + ("" if ok else f" (expected >= {minimum})"))
    try:
        __import__("send2trash")
        record(PASS, "import send2trash")
    except ImportError:
        record(WARN, "import send2trash",
               "optional - without it, 'move to recycle bin' is unavailable and "
               "says so. Everything else works.")


def _version_at_least(have: str, want: str) -> bool:
    def parts(s: str) -> list[int]:
        out = []
        for chunk in str(s).split("."):
            num = ""
            for ch in chunk:
                if ch.isdigit():
                    num += ch
                else:
                    break
            out.append(int(num) if num else 0)
        return out
    h, w = parts(have), parts(want)
    h += [0] * (len(w) - len(h))
    return h[:len(w)] >= w


def check_tools() -> None:
    from htr_studio import config, ffmpeg_tools

    cfg = config.Config.load()
    versions = ffmpeg_tools.tool_versions(cfg.section("paths"))
    for tool, info in versions.items():
        if "NOT FOUND" in info:
            record(FAIL if tool == "ffmpeg" else WARN, f"{tool} available",
                   "Not found. install.bat downloads a static build into "
                   "tools\\ffmpeg; or set paths.ffmpeg in config/settings.json.")
        else:
            record(PASS, f"{tool} available", info[:120])


def check_paths() -> None:
    from htr_studio import paths

    paths.ensure_dirs()
    for d in (paths.DATA_DIR, paths.RECORDINGS_DIR, paths.EXPORTS_DIR,
              paths.LOGS_DIR, paths.CACHE_DIR, paths.CONFIG_DIR):
        if not d.is_dir():
            record(FAIL, f"folder {d.name}", f"could not create {d}")
            return
    probe = paths.DATA_DIR / ".selftest_write"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as e:
        record(FAIL, "data folder writable",
               f"{e}. The installation folder has to be writable - do not install "
               "under Program Files.")
        return
    free = shutil.disk_usage(paths.INSTALL_DIR).free / 1e9
    record(PASS, "folders writable",
           f"{paths.INSTALL_DIR}  ({free:.0f} GB free on this drive)")


def _check_preview(win) -> None:
    """Drive the live-preview tick with a fake camera.

    This exists because of a bug it would have caught. The intensity histogram
    was plotted with one too many bin counts, which pyqtgraph refuses -- but
    every GUI check ran on a machine with no camera, so the preview tick
    returned at its first line and the plotting code was never reached. The
    application passed its self-test and then threw a modal error on every
    frame the moment a real camera was attached: broken in precisely the
    situation it is built for. Anything that only runs with hardware present
    needs a stand-in, or it is not covered at all.
    """
    import numpy as np
    import cv2

    class _FakePreview:
        running = True
        def __init__(self, frames): self.frames = frames; self.i = 0
        def read(self):
            f = self.frames[self.i % len(self.frames)]; self.i += 1; return f
        def measured_fps(self): return 240.0

    tab = win.tab_acq
    try:
        rng = np.random.default_rng(5)
        frames = []
        for k in range(5):
            g = np.full((240, 320), 190, np.uint8)
            g = np.clip(g + rng.normal(0, 6, g.shape), 0, 255).astype(np.uint8)
            cv2.ellipse(g, (90 + 14 * k, 120), (34, 15), 25 * k, 0, 360, 60, -1)
            frames.append(g)
        frames.append(np.full((240, 320), 128, np.uint8))   # degenerate frame
        tab.preview = _FakePreview(frames)
        for _ in range(len(frames) + 2):
            tab._tick_preview()
        x, y = tab.hist_curve.getData()
        if (tab._last_metrics is not None and x is not None and y is not None
                and len(x) == len(y) + 1 and not tab._hist_broken):
            record(PASS, "live preview path",
                   f"{len(frames)} frames through the preview tick; histogram "
                   f"{len(y)} bins over {len(x)} edges; metrics and the "
                   "pre-flight spec built from them")
        else:
            record(FAIL, "live preview path",
                   f"histogram x={None if x is None else len(x)} "
                   f"y={None if y is None else len(y)}; "
                   f"disabled={getattr(tab, '_hist_broken', '?')}")
    except Exception as e:
        import traceback
        record(FAIL, "live preview path",
               f"{type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}")
    finally:
        tab.preview = None


def check_gui() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt5 import QtWidgets
        from htr_studio import config
        from htr_studio.ui.main_window import MainWindow
    except ImportError as e:
        record(FAIL, "GUI imports", str(e))
        return
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    try:
        win = MainWindow(config.Config.load())
        names = [win.tabs.tabText(i) for i in range(win.tabs.count())]
        record(PASS, "GUI constructs", "tabs: " + ", ".join(names))
        _check_preview(win)
        for tab in (win.tab_acq, win.tab_lib, win.tab_review, win.tab_analysis,
                    win.tab_batch):
            tab.shutdown()
    except Exception as e:
        import traceback
        record(FAIL, "GUI constructs", f"{e}\n{traceback.format_exc()[-800:]}")
    finally:
        app.processEvents()


def _make_test_video(out_dir: Path, fps: float = 240.0, duration: float = 20.0
                     ) -> tuple[Path, list[tuple[float, float]], list[tuple[float, float]]]:
    """A short video with three twitches, one grooming bout and one fast turn."""
    import cv2
    import numpy as np

    path = out_dir / "selftest_clip.avi"
    w, h = 320, 240
    n = int(fps * duration)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps,
                             (w, h), isColor=False)
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MJPG writer")

    twitches = [(4.0, 0.11, 88.0), (9.0, 0.12, 80.0), (15.0, 0.10, 94.0)]
    groom = [(6.5, 1.5, 7.5)]
    dart = [(12.0, 0.4)]
    rng = np.random.default_rng(3)
    cx, cy, heading = w / 2, h / 2, 25.0
    vx = vy = omega = 0.0

    for i in range(n):
        t = i / fps
        rot = 0.0
        for s, d, f in twitches:
            if s <= t < s + d:
                taper = np.sin(np.pi * (t - s) / d) ** 0.5
                rot += 36 * taper * np.sin(2 * np.pi * f * (t - s))
        for s, d, f in groom:
            if s <= t < s + d:
                rot += 22 * np.sin(2 * np.pi * f * (t - s))
        darting = any(s <= t < s + d for s, d in dart)

        vx += rng.normal(0, 0.3)
        vy += rng.normal(0, 0.3)
        omega += rng.normal(0, 0.2) + (6.0 if darting else 0.0)
        vx, vy, omega = vx * 0.94, vy * 0.94, omega * 0.9
        cx = float(np.clip(cx + vx, 50, w - 50))
        cy = float(np.clip(cy + vy, 50, h - 50))
        heading += omega * 0.06

        img = np.full((h, w), 215, dtype=np.int16)
        ux, uy = np.cos(np.radians(heading)), np.sin(np.radians(heading))
        cv2.ellipse(img, (int(cx), int(cy)), (34, 15), heading, 0, 360, 55, -1,
                    cv2.LINE_AA)
        cv2.line(img, (int(cx - ux * 30), int(cy - uy * 30)),
                 (int(cx - ux * 90), int(cy - uy * 90)), 60, 2, cv2.LINE_AA)
        ha = heading + rot
        hx, hy = cx + ux * 30, cy + uy * 30
        hax, hay = np.cos(np.radians(ha)), np.sin(np.radians(ha))
        cv2.ellipse(img, (int(hx + hax * 12), int(hy + hay * 12)), (15, 10), ha,
                    0, 360, 48, -1, cv2.LINE_AA)
        img += rng.integers(-6, 7, size=img.shape, dtype=np.int16)
        writer.write(np.clip(img, 0, 255).astype(np.uint8))
    writer.release()
    truth = [(s, s + d) for s, d, _f in twitches]
    confound = [(s, s + d) for s, d, _f in groom] + [(s, s + d) for s, d in dart]
    return path, truth, confound


def check_analysis() -> None:
    from htr_studio import analysis

    tmp = Path(tempfile.mkdtemp(prefix="htr_selftest_"))
    try:
        t0 = time.perf_counter()
        try:
            video, truth, confound = _make_test_video(tmp)
        except Exception as e:
            record(FAIL, "render a test clip", str(e))
            return
        record(PASS, "render a test clip",
               f"{video.name}, {video.stat().st_size / 1e6:.1f} MB, "
               f"{time.perf_counter() - t0:.1f} s")

        cfg = analysis.AnalysisConfig(workers=1, threshold_source="whole")
        t1 = time.perf_counter()
        sig = analysis.extract_signals(video, cfg, fps=240.0)
        rate = sig.n / max(time.perf_counter() - t1, 1e-9)
        if sig.n < 4000:
            record(FAIL, "extract signals", f"only {sig.n} frames decoded")
            return
        level = PASS if sig.valid_frac > 0.95 else WARN
        record(level, "extract signals",
               f"{sig.n} frames, tracking valid {sig.valid_frac * 100:.1f}%, "
               f"{rate:.0f} frames/s per thread "
               f"(a 70-min trial is ~{1_008_000 / max(rate, 1) / 60:.0f} min "
               "single-threaded)")

        det = analysis.detect(sig, cfg)
        val = analysis.validate(det.bursts, truth, 0.15)
        # The three planted twitches must all be found, with nothing extra.
        if val.recall >= 0.99 and val.precision >= 0.99:
            record(PASS, "detect planted twitches", val.line())
        elif val.f1 >= 0.8:
            record(WARN, "detect planted twitches",
                   val.line() + " - below the expected perfect score on this clip; "
                   "check whether a detector default was changed")
        else:
            record(FAIL, "detect planted twitches", val.line())

        spurious_on_confounders = [
            c for c in val.spurious
            if any(s - 0.2 <= c <= e + 0.2 for s, e in confound)]
        if spurious_on_confounders:
            record(FAIL, "ignore grooming and fast turns",
                   f"{len(spurious_on_confounders)} false positive(s) on planted "
                   "confounders")
        else:
            record(PASS, "ignore grooming and fast turns",
                   f"{len(confound)} confounder(s) planted, none detected as a twitch")

        # Parallel path must agree exactly with the single-threaded path.
        import numpy as np
        old = analysis.PARALLEL_MIN_FRAMES
        analysis.PARALLEL_MIN_FRAMES = 1000
        try:
            cfg_p = analysis.AnalysisConfig(workers=4, threshold_source="whole")
            sig_p = analysis.extract_signals(video, cfg_p, fps=240.0)
        finally:
            analysis.PARALLEL_MIN_FRAMES = old
        n = min(sig.n, sig_p.n)
        if n == 0:
            record(FAIL, "parallel extraction", "produced no frames")
        else:
            diff = float(np.nanmax(np.abs(sig.angle[:n] - sig_p.angle[:n])))
            if diff <= 1e-6:
                record(PASS, "parallel extraction",
                       f"{sig_p.n} frames, identical to the single-threaded result")
            elif diff < 0.5:
                record(WARN, "parallel extraction",
                       f"max angle difference {diff:.4f} deg")
            else:
                record(FAIL, "parallel extraction",
                       f"max angle difference {diff:.3f} deg - the two paths "
                       "disagree, so parallel results are not trustworthy")

        # Reports must actually write.
        from htr_studio import report
        rep = report.write_trial_csv(video, tmp, sig, det, bin_s=60.0, t0_s=0.0)
        png = report.plot_trial(tmp / "fig.png", sig, det, bin_s=60.0, t0_s=0.0,
                                title="self-test")
        ok = (rep.combined_csv and rep.combined_csv.exists()
              and png.exists() and png.stat().st_size > 10_000)
        record(PASS if ok else FAIL, "write a report",
               f"{rep.combined_csv.name}, {png.name} "
               f"({png.stat().st_size / 1024:.0f} kB)" if ok else "files missing")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_diagnostics() -> None:
    """The failure-triage path, end to end.

    These three are checked together because they are the chain that turns
    "0 twitches on footage where I can see them" into an answer. Each has been
    the actual cause of a failed analysis, so each is tested rather than assumed:
    a 30 fps slow-motion render must be recognised and the real capture rate
    recovered; background segmentation must find an animal on a cluttered arena
    where plain thresholding does not; and a conversion must be refused if it
    changes the frame count.
    """
    from htr_studio import analysis, ffmpeg_tools, ingest

    tmp = Path(tempfile.mkdtemp(prefix="htr_diag_"))
    try:
        try:
            # Full default duration: a shorter clip truncates the last
            # planted twitch and the scores below would blame the
            # detector for the test's own doing.
            video, truth, _conf = _make_test_video(tmp)
        except Exception as e:
            record(FAIL, "render a clip for the diagnostics check", str(e))
            return

        cfg = analysis.AnalysisConfig(workers=1, threshold_source="whole")

        # 1. Spectrum. The clip is a genuine 240 fps capture, so the energy must
        #    land in the HTR band and no rate override should be suggested.
        sig = analysis.extract_signals(video, cfg, fps=240.0)
        spec = analysis.spectrum(sig)
        # The band's own peak, not the global one: locomotion is continuous and
        # a twitch is 100 ms, so the tallest peak in any real trial is down at
        # 5-10 Hz even when the twitches are unmistakable.
        if spec.band_peak_hz and spec.band_fraction > 0.1:
            record(PASS, "spectrum finds the band",
                   f"{spec.band_fraction:.0%} of the movement energy inside "
                   f"78-98 Hz, peaking at {spec.band_peak_hz:.1f} Hz "
                   f"(strongest component overall {spec.peak_hz:.1f} Hz)")
        else:
            record(FAIL, "spectrum finds the band",
                   f"only {spec.band_fraction:.1%} in band, band peak "
                   f"{spec.band_peak_hz:.1f} Hz - the diagnostics would mislead "
                   "on real footage")

        if analysis.suggest_capture_rate(spec) is None:
            record(PASS, "no false slow-motion warning",
                   "a true 240 fps clip is correctly left alone")
        else:
            record(WARN, "no false slow-motion warning",
                   "a rate override was suggested for a correctly timed clip")

        # 2. Slow motion. Same frames, declared at 30 fps. Analysed as the file
        #    claims, the detector must find nothing and the suggestion must
        #    point back at the real rate; analysed at that rate, the twitches
        #    must come back.
        slow = tmp / "slowmo.avi"
        rc = subprocess.run(
            [str(ffmpeg_tools.ffmpeg_path()), "-hide_banner", "-loglevel",
             "error", "-i", str(video), "-c:v", "copy", "-r", "30",
             "-y", str(slow)], capture_output=True, text=True).returncode
        if rc != 0 or not slow.exists():
            record(SKIP, "recover a slow-motion render",
                   "could not build the test file")
        else:
            s_slow = analysis.extract_signals(slow, cfg, fps=0.0)
            d_slow = analysis.detect(s_slow, cfg)
            sg = analysis.suggest_capture_rate(analysis.spectrum(s_slow))
            s_fix = analysis.extract_signals(slow, cfg, fps=240.0)
            d_fix = analysis.detect(s_fix, cfg)
            got_back = analysis.validate(d_fix.bursts, truth, 0.15)
            if (sg is not None and sg.implied_fps > 150
                    and d_slow.count == 0 and got_back.f1 >= 0.8):
                record(PASS, "recover a slow-motion render",
                       f"0 detected at the file's own {s_slow.fps:.0f} fps; "
                       f"suggested {sg.implied_fps:.0f} fps "
                       f"({sg.confidence}); {got_back.line()} at 240 fps")
            else:
                record(FAIL, "recover a slow-motion render",
                       f"detected {d_slow.count} at {s_slow.fps:.0f} fps, "
                       f"suggestion "
                       + (f"{sg.implied_fps:.0f} fps" if sg else "none")
                       + f", after override {got_back.line()}")

        # 3. Background segmentation must work where thresholding is the
        #    fallback, and the auto choice must not regress the clean clip.
        for mode in ("threshold", "background", "auto"):
            c = analysis.AnalysisConfig(workers=1, threshold_source="whole",
                                        segmentation=mode)
            cal = analysis.calibrate(video, c)
            s = analysis.extract_signals(video, c, fps=240.0, calibration=cal)
            d = analysis.detect(s, c)
            v = analysis.validate(d.bursts, truth, 0.15)
            level = PASS if (s.valid_frac > 0.95 and v.f1 >= 0.8) else (
                WARN if v.f1 >= 0.5 else FAIL)
            record(level, f"segmentation: {mode}",
                   f"chose {cal.mode} (threshold {cal.threshold:.0f}, found in "
                   f"{cal.hit_rate:.0%} of sampled frames, area variation "
                   f"{cal.area_cv:.2f}); tracked {s.valid_frac * 100:.1f}%, "
                   f"{v.line()}")

        # 4. Inspection and a verified conversion.
        rep = ingest.inspect_file(video, count_frames=True)
        if rep.ok and rep.all_intra and rep.true_fps > 200:
            record(PASS, "inspect a recording",
                   f"{rep.codec} all-intra, {rep.true_fps:.1f} fps measured by "
                   f"exact count, {rep.bytes_per_frame:,.0f} bytes per frame")
        else:
            record(FAIL, "inspect a recording",
                   f"codec {rep.codec}, {rep.true_fps:.1f} fps, "
                   f"all-intra {rep.all_intra}")

        avail = ingest.available_encoders()
        if "mjpeg" not in avail:
            record(WARN, "archive a recording",
                   "this FFmpeg has no mjpeg encoder, so archiving is limited")
        else:
            dest = tmp / "archived.avi"
            res = ingest.archive(
                ingest.ArchiveSpec(source=video, dest=dest, option="mjpeg_q4"),
                verify=True)
            if res.ok and res.source_frames == res.dest_frames:
                # The archived copy must analyse to the same answer, or the
                # conversion is not safe whatever the frame count says.
                s2 = analysis.extract_signals(dest, cfg, fps=240.0)
                v2 = analysis.validate(analysis.detect(s2, cfg).bursts, truth, 0.15)
                level = PASS if v2.f1 >= 0.8 else FAIL
                record(level, "archive a recording",
                       f"{res.ratio:.0%} of the original, "
                       f"{res.dest_frames:,} frames verified; the archived copy "
                       f"re-analyses to {v2.line()}")
            else:
                record(FAIL, "archive a recording", res.message)
        missing = [o.label for o in ingest.archive_options()
                   if o.needs_encoder and o.needs_encoder not in avail]
        record(PASS if not missing else WARN, "archive options available",
               "all available" if not missing
               else "unavailable in this FFmpeg build: " + ", ".join(missing))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_camera() -> None:
    from htr_studio import camera, config, ffmpeg_tools, paths

    if not paths.is_windows():
        record(SKIP, "camera enumeration",
               "DirectShow is Windows-only; this is not an error on another OS")
        return
    cfg = config.Config.load()
    try:
        devices = ffmpeg_tools.list_dshow_devices(cfg.get("paths.ffmpeg", ""))
    except ffmpeg_tools.FFmpegMissing as e:
        record(WARN, "camera enumeration", str(e))
        return
    if not devices:
        record(WARN, "camera enumeration",
               "No DirectShow video device. Expected on an analysis-only computer; "
               "on the capture station, check the USB connection.")
        return
    names = ", ".join(d.name for d in devices[:4])
    match = ffmpeg_tools.match_device(devices, cfg.get("camera.device_name", ""))
    record(PASS, "camera enumeration",
           f"{len(devices)} device(s): {names}"
           + (f"  → using \"{match.name}\"" if match else
              "  (none matched the configured name)"))
    idx = camera.enumerate_indices(4, cfg.get("camera.opencv_backend", "dshow"))
    record(PASS if idx else WARN, "preview backend",
           f"{len(idx)} OpenCV index(es) usable" if idx
           else "no OpenCV index opened; the live preview will not work")


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="skip the analysis check (which renders a test clip)")
    ap.add_argument("--no-gui", action="store_true", help="skip the GUI check")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    from htr_studio import __version__

    print("=" * 74)
    print(f" HTR Studio {__version__} self-test")
    print(f" {platform.platform()}")
    print(f" Python {sys.version.split()[0]}  ({sys.executable})")
    print(f" Installed at {ROOT}")
    print("=" * 74)

    print("\n-- dependencies ------------------------------------------------")
    check_imports()
    print("\n-- folders -----------------------------------------------------")
    check_paths()
    print("\n-- external tools ----------------------------------------------")
    check_tools()
    if not args.no_gui:
        print("\n-- user interface ----------------------------------------------")
        check_gui()
    if not args.quick:
        print("\n-- analysis core (end to end) ----------------------------------")
        check_analysis()
        print("\n-- failure diagnostics and archiving ---------------------------")
        check_diagnostics()
    print("\n-- camera ------------------------------------------------------")
    check_camera()

    fails = [r for r in _results if r[0] == FAIL]
    warns = [r for r in _results if r[0] == WARN]
    print("\n" + "=" * 74)
    print(f" {len(_results)} checks: "
          f"{sum(1 for r in _results if r[0] == PASS)} passed, "
          f"{len(warns)} warning(s), {len(fails)} failure(s), "
          f"{sum(1 for r in _results if r[0] == SKIP)} skipped")
    if fails:
        print("\n FAILURES - fix these before recording a real trial:")
        for _l, name, detail in fails:
            print(f"   - {name}: {detail.splitlines()[0] if detail else ''}")
    if warns:
        print("\n Warnings (usable, but worth knowing):")
        for _l, name, detail in warns:
            print(f"   - {name}: {detail.splitlines()[0] if detail else ''}")
    if not fails:
        print("\n Installation looks good. Start the app with run_htr_studio.bat.")
    print("=" * 74)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
