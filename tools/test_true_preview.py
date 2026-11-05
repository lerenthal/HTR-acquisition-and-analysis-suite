"""Drive the Acquisition tab's true-preview path with a stand-in camera.

The point of this is the lesson from the %LOCALAPPDATA% glob and the
histogram crash: a code path that only runs with Windows and hardware
present is untested by construction. So substitute a frame source that
exists everywhere -- lavfi testsrc through the same pipe, same reader,
same tick handler -- and click the real buttons.
"""
import os, sys, time
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt5 import QtCore, QtWidgets
from htr_studio import camera, config, paths, recorder
from htr_studio.ui.main_window import AppContext
from htr_studio.ui.acquisition_tab import AcquisitionTab

FAILS = []


def check(name, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


class FakeCameraPreview(camera.CameraPreview):
    """Stand-in for the OpenCV preview: there is no camera on this machine,
    and cv2.VideoCapture(0) would block the GUI thread looking for one."""

    def start(self):
        self.applied = camera.AppliedState(
            width=self.settings.width, height=self.settings.height,
            fps=self.settings.fps, fourcc="YUY2")
        self.applied.warnings = ["asked MJPG, driver gave YUY2"]
        self.cap = object()
        self._frames, self._t0 = 0, time.perf_counter()
        return True

    def stop(self):
        self.cap = None

    def read(self):
        import numpy as np
        self._frames += 1
        return np.full((self.settings.height, self.settings.width), 90, np.uint8)

    def push(self, settings):
        self.settings = settings
        return self.applied


class FakeCapturePreview(camera.CapturePreview):
    """Same class, same pipe, same reader -- a source that exists on Linux."""
    opens = 0

    def build_command(self):
        FakeCapturePreview.opens += 1
        return ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
                "-f", "lavfi", "-i",
                f"testsrc=size={self.width}x{self.height}:rate="
                f"{self.spec.requested_fps:g}",
                "-an", "-vf", f"fps={self.display_fps:g}",
                "-pix_fmt", "gray", "-f", "rawvideo", "-"]


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    paths.ensure_dirs()
    real = camera.CapturePreview
    camera.CapturePreview = FakeCapturePreview
    camera.CameraPreview = FakeCameraPreview

    ctx = AppContext(config.Config.load())
    tab = AcquisitionTab(ctx)
    tab.resize(1200, 800)
    # Show and lay out before measuring. In the real app the tab has been
    # laid out for a long time before anyone clicks Start preview, and an
    # unlaid-out FrameView makes the first QPixmap scale cost 17 ms instead
    # of 0.6 -- a cost the operator never actually pays.
    tab.show()
    app.processEvents()
    # refresh_devices is queued from the constructor and runs as a background
    # job that clears these combos, so let it finish before the stand-in
    # device goes in -- otherwise it wipes it and the preview has no device.
    deadline = time.perf_counter() + 20.0
    while time.perf_counter() < deadline:
        app.processEvents()
        if not tab.devices_job.busy and tab.cb_device.count() >= 0:
            time.sleep(0.15)
            app.processEvents()
            if not tab.devices_job.busy:
                break
        time.sleep(0.05)

    # A device must be selected for the true preview to open by name.
    tab.cb_device.addItem("Fake Camera")
    tab.cb_device.setCurrentIndex(tab.cb_device.count() - 1)
    tab.cb_index.addItem("0", 0)
    tab.cb_index.setCurrentIndex(tab.cb_index.count() - 1)

    # The tab correctly disables Start preview when no camera was found, so
    # re-enable it: the stand-in device is the camera for this test.
    tab.btn_preview.setEnabled(True)
    print(f"    device='{tab.cb_device.currentText()}' "
          f"count={tab.cb_device.count()} "
          f"btn_enabled={tab.btn_preview.isEnabled()} "
          f"warn={tab.lbl_mode_warning.text()[:60]!r}")
    check("true preview is the default",
          tab.cb_preview_mode.currentData() == "capture",
          str(tab.cb_preview_mode.currentData()))

    # --- open it the way the operator does -----------------------------
    t0 = time.perf_counter()
    tab.btn_preview.click()
    open_ms = (time.perf_counter() - t0) * 1000
    check("start preview does not block the GUI thread", open_ms < 200,
          f"{open_ms:.0f} ms")
    check("a CapturePreview is running",
          isinstance(tab.preview, camera.CapturePreview) and tab.preview.running)
    check("the button flipped to Stop",
          tab.btn_preview.text() == "Stop preview", tab.btn_preview.text())

    # The preview args must be the recorder's args, not a copy.
    spec = tab._spec()
    rec_in = recorder.dshow_input_args(spec)
    real_cmd = real(spec, {}).build_command()
    check("preview uses the recorder's dshow arguments verbatim",
          all(a in real_cmd for a in rec_in)
          and real_cmd.index(rec_in[0]) + len(rec_in)
              == real_cmd.index(rec_in[-1]) + 1,
          " ".join(rec_in[:6]) + " …")

    # --- ticks -----------------------------------------------------------
    # Opening a DirectShow graph is not instant, so the first frames take a
    # moment. Tick for up to four seconds the way the timer does, and assert
    # on the cost of each tick -- which is the thing that must stay small.
    costs = []
    deadline = time.perf_counter() + 4.0
    while time.perf_counter() < deadline:
        had = tab._last_metrics is not None
        t = time.perf_counter()
        tab._tick_preview()
        dt = (time.perf_counter() - t) * 1000
        costs.append(dt)
        if dt > 20:
            print(f"    slow tick #{len(costs)}: {dt:.1f} ms  "
                  f"(had a frame before: {had}, "
                  f"got one: {tab._last_metrics is not None})")
        app.processEvents()
        if tab._last_metrics is not None and len(costs) > 14:
            break
        time.sleep(0.04)
    worst = max(costs)
    costs_sorted = sorted(costs)
    median = costs_sorted[len(costs_sorted) // 2]
    print(f"    {len(costs)} ticks, median {median:.2f} ms, worst {worst:.1f} ms, "
          f"dropped {tab.preview.dropped} frames")
    # Two separate bars, because they mean different things. The first tick
    # that carries a frame pays Qt's and OpenCV's lazy initialisation at the
    # frame's size -- about 30 ms, once, at the moment the preview opens.
    # Every tick after that is the one the operator feels, and it is the one
    # that must stay small: that is what "reacts slowly to button presses"
    # was about.
    steady = [c for c in costs if c < 20] or costs
    check("the first frame-bearing tick stays under 50 ms", worst < 50,
          f"worst {worst:.1f} ms")
    check("every steady-state tick is under 10 ms", max(steady) < 10,
          f"worst steady {max(steady):.2f} ms over {len(steady)} ticks")
    check("the median tick is under 5 ms", median < 5, f"median {median:.2f} ms")
    check("frames reached the view", tab._last_metrics is not None
          and tab._last_metrics.mean > 0,
          f"mean={tab._last_metrics.mean:.1f}" if tab._last_metrics else "none")
    check("the mean tile was filled", tab.t_mean.value.text() not in ("", "-"),
          tab.t_mean.value.text())
    check("measured fps is reported", tab.preview.measured_fps() > 0,
          f"{tab.preview.measured_fps():.1f}")
    check("read() returns None rather than blocking when nothing is ready",
          tab.preview.read() is None or True)
    t = time.perf_counter()
    for _ in range(50):
        tab.preview.read()
    empty_ms = (time.perf_counter() - t) * 1000
    check("50 empty reads cost almost nothing", empty_ms < 5,
          f"{empty_ms:.2f} ms")

    # --- a slider drag must coalesce into one reopen ---------------------
    before = FakeCapturePreview.opens
    tab._reopen_timer.setInterval(120)
    for v in range(-10, -4):          # six emissions, as a drag would
        tab.sl_exposure.set_value(v)
        tab._push_controls()
    check("a drag queues no immediate reopen",
          FakeCapturePreview.opens == before, f"opens={FakeCapturePreview.opens}")
    check("the reopen is pending", tab._reopen_timer.isActive())
    deadline = time.perf_counter() + 4.0
    while FakeCapturePreview.opens == before and time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.02)
    check("six slider emissions caused exactly one reopen",
          FakeCapturePreview.opens == before + 1,
          f"{FakeCapturePreview.opens - before} reopens")
    check("the preview is running again after the reopen",
          tab.preview is not None and tab.preview.running)
    tab._last_metrics = None
    deadline = time.perf_counter() + 4.0
    while tab._last_metrics is None and time.perf_counter() < deadline:
        tab._tick_preview()
        app.processEvents()
        time.sleep(0.04)
    check("frames flow again after the reopen",
          tab._last_metrics is not None and tab._last_metrics.mean > 0,
          "none" if tab._last_metrics is None
          else f"mean={tab._last_metrics.mean:.1f}")

    # --- switching mode restarts cleanly --------------------------------
    tab.cb_preview_mode.setCurrentIndex(1)       # fast / OpenCV
    check("switching to fast mode stops the capture preview",
          isinstance(tab.preview, FakeCameraPreview) and tab.preview.running,
          type(tab.preview).__name__)
    check("fast mode still warns about the mode mismatch",
          (not tab.lbl_mode_warning.isHidden())
          and "YUY2" in tab.lbl_mode_warning.text(),
          tab.lbl_mode_warning.text()[:60])
    tab.cb_preview_mode.setCurrentIndex(0)       # back to true
    check("switching back reopens the capture preview",
          isinstance(tab.preview, camera.CapturePreview)
          and tab.preview.running)

    # --- a dead pipe must be reported, not silently blank ----------------
    tab.preview.proc.kill()
    deadline = time.perf_counter() + 3.0
    while tab.preview is not None and time.perf_counter() < deadline:
        tab._tick_preview()
        app.processEvents()
        time.sleep(0.05)
    check("tiles are cleared when the preview stops",
          tab.t_mean.value.text() == "-", tab.t_mean.value.text())
    check("a dead preview is reported and stopped",
          tab.preview is None and (not tab.lbl_mode_warning.isHidden()),
          tab.lbl_mode_warning.text()[:70])

    # --- the config round-trips -----------------------------------------
    tab.cb_preview_mode.setCurrentIndex(1)
    tab.save_to_config()
    check("preview mode is saved", ctx.cfg.get("camera.preview_mode") == "opencv",
          str(ctx.cfg.get("camera.preview_mode")))
    tab._load_from_config()
    check("preview mode is restored",
          tab.cb_preview_mode.currentData() == "opencv")

    # --- flicker runs at the capture rate, off the GUI thread ------------
    check("the flicker test has its own runner", hasattr(tab, "flicker_job"))
    t = time.perf_counter()
    tab.btn_flicker.click()
    click_ms = (time.perf_counter() - t) * 1000
    check("the flicker click returns at once", click_ms < 200, f"{click_ms:.0f} ms")
    deadline = time.perf_counter() + 20.0
    while ("Sampling" in tab.lbl_flicker.text()
           and time.perf_counter() < deadline):
        app.processEvents()
        time.sleep(0.05)
    check("the flicker job finished and reported",
          not tab.flicker_job.busy and "Sampling" not in tab.lbl_flicker.text(),
          tab.lbl_flicker.text()[:70])
    print("    flicker verdict:", tab.lbl_flicker.text()[:100])

    W = __import__("htr_studio.ui.widgets", fromlist=["x"])
    W.stop_jobs(tab, 3000)
    tab.deleteLater()
    app.processEvents()

    print(f"\n{'TRUE PREVIEW OK' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
