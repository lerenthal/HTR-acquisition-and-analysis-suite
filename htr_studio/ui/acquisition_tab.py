"""Acquisition tab: camera setup, lighting checks, and trial recording.

The three jobs this tab exists to do, in the order an operator does them:

1. **Find and configure the camera.** Pick the DirectShow device, pick a mode,
   and -- the part that is easy to get wrong -- actually verify that the camera
   delivered the mode it was asked for. The B0634 negotiates: ask for 200 fps at
   320x240 and it gives 240; ask for 120 at VGA and it gives about 100. "Probe"
   measures it instead of trusting the request.

2. **Get the picture right before the animal is in the arena.** Live preview with
   exposure and gain, plus the numbers that actually decide whether tracking will
   work: contrast, clipped pixels, illumination uniformity, focus. And a flicker
   test, because a PWM-dimmed LED panel modulates the whole frame at a few
   hundred hertz, which under a sub-millisecond global-shutter exposure lands
   inside the HTR band and produces whole-field false positives that no amount of
   analysis can undo.

3. **Record the trial with its metadata attached.** Animal, group, dose, route,
   operator -- and T0, marked at the moment of gavage, which is what makes the
   time course interpretable and what the detector uses to find its baseline.

Preview and recording are mutually exclusive: a DirectShow device admits one
client at a time. Starting a recording stops the preview first, and says so.
"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

from .. import camera, config, ffmpeg_tools, paths, recorder, theme
from . import widgets as W

log = logging.getLogger(__name__)

#: Preview refresh bounds. 40 ms is 25 Hz, which is as fast as a human eye
#: needs for framing and focus; the upper bound keeps the preview visibly
#: alive even on a machine where a frame is expensive.
PREVIEW_MIN_INTERVAL_MS = 40
PREVIEW_MAX_INTERVAL_MS = 250

#: How long "Check capture" records. Long enough for the sensor's exposure and
#: gain to settle after the mode switch, short enough that nobody minds.
CHECK_CAPTURE_S = 2.0


def _check_capture(spec, cfg_paths: dict, progress=None) -> dict:
    """Record a two-second clip with the real command, then measure the file.

    This exists because the preview is not a preview of the recording, and
    every honest reading of the code says so. The preview is OpenCV holding
    the camera in whatever media type the DSHOW backend negotiated -- on the
    B0634, YUY2, because the driver refuses MJPG to OpenCV -- at whatever
    frame rate that mode runs, because the DSHOW backend's CAP_PROP_FPS is an
    echo of what was set, not a negotiation result. FFmpeg then opens the same
    camera in MJPEG at an explicit ``-framerate``, which dshow does honour.

    Two different media types at two different frame rates means two different
    exposure windows, and brightness is proportional to exposure time. A
    preview whose mode allows 7.8 ms of integration will look several times
    brighter than a 200 fps capture that physically cannot integrate for more
    than 5 ms. No amount of pushing UVC controls fixes that, because it is not
    a settings problem: it is the frame interval.

    So rather than argue about it, measure it. Record with the exact command
    the trial will use, decode the file, and report its own numbers. The file
    is the only thing that tells the truth about what the analysis will see.
    """
    import cv2

    out: dict = {"ok": False, "note": "", "metrics": None, "info": None}
    if progress:
        progress(1, 3, f"recording {CHECK_CAPTURE_S:.0f} s with the real command…")

    rec = recorder.Recorder(spec, cfg_paths)
    cmd = rec.build_command()
    target = spec.target_path()
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                          text=True, errors="replace",
                          **({"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
                             if paths.is_windows() else {}))
    if not target.is_file() or target.stat().st_size == 0:
        tail = [ln for ln in (proc.stderr or "").splitlines() if ln.strip()][-6:]
        out["note"] = recorder.explain_exit(proc.returncode, tail)
        return out

    if progress:
        progress(2, 3, "measuring the file…")
    out["info"] = ffmpeg_tools.probe_video(target, cfg_paths.get("ffprobe", ""),
                                           count_frames=True)

    cap = cv2.VideoCapture(str(target))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # Skip the first 10%: the first frames after a mode switch are taken while
    # the sensor is still settling and are not representative of the trial.
    picks = [int(n * f) for f in (0.2, 0.4, 0.6, 0.8)] if n > 10 else [0]
    mets = []
    for idx in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mets.append(camera.frame_metrics(gray))
    cap.release()
    if not mets:
        out["note"] = "the clip was written but no frame could be decoded from it"
        return out

    # The median frame by mean level, so one odd frame cannot set the verdict.
    out["metrics"] = sorted(mets, key=lambda m: m.mean)[len(mets) // 2]
    out["frames_sampled"] = len(mets)
    out["path"] = target
    out["ok"] = True
    if progress:
        progress(3, 3, "done")
    return out


class AcquisitionTab(QtWidgets.QWidget):
    recordingFinished = QtCore.pyqtSignal(str)   # path of the finished recording
    statusMessage = QtCore.pyqtSignal(str)

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.pal = theme.palette(ctx.cfg.get("ui.theme", "light"))
        self.preview: camera.CameraPreview | None = None
        self.rec: recorder.Recorder | None = None
        self.worker: W.Worker | None = None
        self.devices: list[ffmpeg_tools.DShowDevice] = []
        self.indices: list[camera.ProbedIndex] = []
        self.presets = config.load_capture_presets()
        self._last_metrics: camera.FrameMetrics | None = None
        self._hist_broken = False
        self._start_after_preflight = False

        # Runners come before _build: the widget construction binds buttons to
        # them so that a running job disables its own button.
        self._hist_broken = False
        self._start_after_preflight = False
        self.preflight_job = W.JobRunner("preflight", self)
        self.preflight_job.finished.connect(self._preflight_done)
        self.preflight_job.failed.connect(
            lambda m, tb: self.lbl_rec.set(f"Pre-flight failed: {m}", "bad"))
        self.devices_job = W.JobRunner("enumerate-devices", self)
        self.devices_job.finished.connect(self._devices_done)
        self.devices_job.failed.connect(
            lambda m, tb: self.lbl_device.set(f"Enumeration failed: {m}", "bad"))
        self.probe_job = W.JobRunner("mode-probe", self)
        self.probe_job.finished.connect(self._probe_done)
        self.probe_job.failed.connect(self._job_failed)
        self.verify_job = W.JobRunner("verify", self)
        self.verify_job.finished.connect(
            lambda v: self.lbl_rec.set(v.message, "ok" if v.ok else "warn"))
        self.verify_job.failed.connect(self._job_failed)
        self._check_was_previewing = False
        self._preview_metrics_at_check = None
        self.check_job = W.JobRunner("capture-check", self)
        self.check_job.finished.connect(self._check_capture_done)
        self.check_job.failed.connect(self._job_failed)
        self._flicker_was_previewing = False
        self._metrics_at = 0.0
        self.flicker_job = W.JobRunner("flicker", self)
        self.flicker_job.finished.connect(self._flicker_done)
        self.flicker_job.failed.connect(
            lambda m, tb: self.lbl_flicker.set(f"Flicker test failed: {m}", "bad"))

        self._build()
        W.no_wheel(self)
        self._load_from_config()

        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setInterval(PREVIEW_MIN_INTERVAL_MS)
        self.preview_timer.timeout.connect(self._tick_preview)

        self.rec_timer = QtCore.QTimer(self)
        self.rec_timer.setInterval(500)
        self.rec_timer.timeout.connect(self._tick_recording)

        # Coalesces a slider drag into a single reopen of the true preview.
        # A drag emits a change per pixel of travel and each reopen costs
        # about a second, so without this a drag across the track would queue
        # dozens of them.
        self._reopen_timer = QtCore.QTimer(self)
        self._reopen_timer.setSingleShot(True)
        self._reopen_timer.setInterval(
            int(self.ctx.cfg.get("camera.preview_reopen_debounce_ms", 500)))
        self._reopen_timer.timeout.connect(self._reopen_true_preview)

        QtCore.QTimer.singleShot(0, self.refresh_devices)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build(self) -> None:
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        left_inner = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(left_inner)
        left.setContentsMargins(0, 0, 6, 0)
        left.addWidget(self._group_device())
        left.addWidget(self._group_mode())
        left.addWidget(self._group_image())
        left.addStretch(1)

        right_inner = QtWidgets.QWidget()
        right = QtWidgets.QVBoxLayout(right_inner)
        right.setContentsMargins(6, 0, 0, 0)
        right.addWidget(self._group_preview(), 1)
        right.addWidget(self._group_trial())
        right.addWidget(self._group_record())

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(W.scrollable(left_inner))
        split.addWidget(W.scrollable(right_inner))
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 5)
        outer.addWidget(split)

    # ---- camera device ----
    def _group_device(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("1 - Camera")
        form = QtWidgets.QFormLayout(g)

        self.cb_device = QtWidgets.QComboBox()
        self.cb_device.setMinimumWidth(240)
        # Editable on purpose: FFmpeg has changed its -list_devices output
        # format between releases, and if a future build changes it again this
        # box still lets the exact device name be typed in by hand.
        self.cb_device.setEditable(True)
        W.form_row(form, "DirectShow device", self.cb_device,
                   "The name FFmpeg records from. The B0634 usually appears as "
                   "'Arducam ezBOX VGA GS Mono'. You can type a name here if "
                   "the list comes up empty - run\n"
                   "  ffmpeg -list_devices true -f dshow -i dummy\n"
                   "in a terminal to see what Windows calls it.")

        self.cb_index = QtWidgets.QComboBox()
        W.form_row(form, "OpenCV index (preview)", self.cb_index,
                   "Used only for the live preview and for pushing exposure/gain. "
                   "Recording always goes through FFmpeg.")

        row = QtWidgets.QHBoxLayout()
        self.btn_refresh = W.button("Refresh", tooltip="Re-enumerate cameras")
        self.btn_modes = W.button("List modes",
                                  tooltip="Ask the driver which resolutions and "
                                          "frame rates it advertises")
        self.btn_props = W.button(
            "Camera properties…",
            tooltip="Open the vendor DirectShow property page. Values set there "
                    "are stored in the driver and survive the handle closing, "
                    "which makes it the reliable place to set exposure.")
        for b in (self.btn_refresh, self.btn_modes, self.btn_props):
            row.addWidget(b)
        row.addStretch(1)
        form.addRow(row)

        self.lbl_device = W.StatusLabel("Looking for cameras…", "info")
        form.addRow(self.lbl_device)

        self.btn_refresh.clicked.connect(self.refresh_devices)
        self.btn_modes.clicked.connect(self._list_modes)
        self.btn_props.clicked.connect(self._open_props)
        return g

    # ---- capture mode ----
    def _group_mode(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("2 - Capture mode")
        form = QtWidgets.QFormLayout(g)

        self.cb_preset = QtWidgets.QComboBox()
        for p in self.presets:
            self.cb_preset.addItem(p["name"], p)
        self.cb_preset.currentIndexChanged.connect(self._apply_preset)
        form.addRow("Preset", self.cb_preset)

        self.lbl_preset_note = QtWidgets.QLabel("")
        self.lbl_preset_note.setObjectName("hint")
        self.lbl_preset_note.setWordWrap(True)
        form.addRow(self.lbl_preset_note)

        hw = QtWidgets.QHBoxLayout()
        self.sp_w = QtWidgets.QSpinBox()
        self.sp_w.setRange(64, 4096)
        self.sp_h = QtWidgets.QSpinBox()
        self.sp_h.setRange(64, 4096)
        self.sp_fps = QtWidgets.QDoubleSpinBox()
        self.sp_fps.setRange(1, 1000)
        self.sp_fps.setDecimals(0)
        self.cb_pixfmt = QtWidgets.QComboBox()
        self.cb_pixfmt.addItems(["mjpeg", "yuyv422"])
        for wdg, lab in ((self.sp_w, "W"), (self.sp_h, "H"),
                         (self.sp_fps, "fps"), (self.cb_pixfmt, "fmt")):
            hw.addWidget(QtWidgets.QLabel(lab))
            hw.addWidget(wdg)
        hw.addStretch(1)
        form.addRow("Requested", hw)

        row = QtWidgets.QHBoxLayout()
        self.btn_probe_one = W.button("Probe this mode")
        self.btn_probe_all = W.button("Probe all presets")
        row.addWidget(self.btn_probe_one)
        row.addWidget(self.btn_probe_all)
        row.addStretch(1)
        form.addRow(row)

        self.tbl_probe = QtWidgets.QTableWidget(0, 5)
        self.tbl_probe.setHorizontalHeaderLabels(
            ["Mode", "Requested", "Delivered", "Frames", "Result"])
        self.tbl_probe.verticalHeader().setVisible(False)
        self.tbl_probe.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        self.tbl_probe.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)
        self.tbl_probe.setMaximumHeight(150)
        form.addRow(self.tbl_probe)

        self.lbl_probe = W.StatusLabel("", "info")
        form.addRow(self.lbl_probe)

        self.btn_probe_one.clicked.connect(lambda: self._probe([self._current_mode()]))
        self.btn_probe_all.clicked.connect(
            lambda: self._probe([(p["width"], p["height"], p["requested_fps"],
                                  p.get("pixel_format", "mjpeg")) for p in self.presets]))
        return g

    # ---- exposure and lighting ----
    def _group_image(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("3 - Exposure and lighting")
        v = QtWidgets.QVBoxLayout(g)

        self.chk_auto = QtWidgets.QCheckBox("Auto-exposure (not recommended)")
        self.chk_auto.setToolTip(
            "Auto-exposure hunts when the mouse moves across the frame, and that "
            "brightness modulation lands in the HTR band. Keep it off.")
        v.addWidget(self.chk_auto)

        self.sl_exposure = W.LabeledSlider(
            "Exposure (UVC log2 value)", -13, 0, -7, 1, 0,
            hint=lambda x: (f"≈ {camera.exposure_hint_ms(x):.2f} ms integration "
                            f"(clamped by the frame interval). More negative = shorter."))
        self.sl_gain = W.LabeledSlider("Gain", 0, 255, 0, 1, 0,
                                       hint=lambda x: "0 keeps sensor noise out of the trace")
        self.sl_brightness = W.LabeledSlider("Brightness", -64, 64, 0, 1, 0)
        self.sl_contrast = W.LabeledSlider("Contrast", 0, 100, 32, 1, 0)
        self.sl_gamma = W.LabeledSlider("Gamma", 1, 500, 100, 1, 0)
        for s in (self.sl_exposure, self.sl_gain, self.sl_brightness,
                  self.sl_contrast, self.sl_gamma):
            v.addWidget(s)
            s.valueChanged.connect(self._push_controls)
        self.chk_auto.toggled.connect(self._push_controls)

        row = QtWidgets.QHBoxLayout()
        self.btn_flicker = W.button(
            "Run flicker test (2 s)",
            tooltip="Looks for brightness ripple from PWM-dimmed lighting. "
                    "Ripple in the 55-125 Hz band produces whole-field false "
                    "positives that cannot be fixed in analysis.\n\n"
                    "Measured through the capture pipeline at the capture "
                    "frame rate, because whether a ripple folds into the HTR "
                    "band depends on the sampling rate. The preview stops for "
                    "the two seconds it takes and comes back afterwards.")
        self.flicker_job._button = self.btn_flicker
        row.addWidget(self.btn_flicker)
        row.addStretch(1)
        v.addLayout(row)
        self.lbl_flicker = W.StatusLabel("", "info")
        v.addWidget(self.lbl_flicker)

        self.plot_hist = pg.PlotWidget(title="Intensity histogram")
        self.plot_hist.setMaximumHeight(130)
        self.plot_hist.setLabel("bottom", "pixel value")
        self.plot_hist.showGrid(x=False, y=True, alpha=0.2)
        self.hist_curve = pg.PlotCurveItem(
            pen=pg.mkPen(self.pal.s1, width=2),
            brush=pg.mkBrush(self.pal.s1 + "40"), fillLevel=0)
        self.plot_hist.addItem(self.hist_curve)
        v.addWidget(self.plot_hist)

        self.btn_flicker.clicked.connect(self._flicker)
        return g

    # ---- preview ----
    def _group_preview(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Live preview")
        v = QtWidgets.QVBoxLayout(g)

        row = QtWidgets.QHBoxLayout()
        self.cb_preview_mode = QtWidgets.QComboBox()
        self.cb_preview_mode.addItem("True (capture pipeline)", "capture")
        self.cb_preview_mode.addItem("Fast (OpenCV)", "opencv")
        self.cb_preview_mode.setToolTip(
            "<b>True</b> opens the camera with the recorder's own FFmpeg "
            "arguments - the same media type, size and frame rate - so the "
            "picture is the one that will be written to disk, with the same "
            "exposure window. FFmpeg holds the device while it runs, so "
            "changing a control reopens the preview (about a second), and "
            "exposure and gain are set through 'Camera properties'.<br><br>"
            "<b>Fast</b> is the OpenCV preview: controls respond instantly "
            "and there is no blink, but on this camera the driver refuses "
            "MJPG to OpenCV and falls back to YUY2 at a different frame "
            "rate, so the brightness is not the recording's. Use it for "
            "framing and focus, which it shows perfectly well.")
        row.addWidget(QtWidgets.QLabel("Preview:"))
        row.addWidget(self.cb_preview_mode)
        self.btn_preview = W.button("Start preview", "primary")
        self.btn_snapshot = W.button("Save snapshot")
        self.btn_snapshot.setEnabled(False)
        self.btn_props = W.button("Camera properties…")
        self.btn_props.setToolTip(
            "Opens the camera driver's own DirectShow property page while "
            "FFmpeg holds the device in the capture mode, so exposure and "
            "gain are set against the mode the trial will actually use. On "
            "this camera that is the only way to do it: OpenCV cannot get "
            "the MJPEG mode from the driver at all.")
        self.btn_check_capture = W.button("Check capture")
        self.check_job._button = self.btn_check_capture
        self.btn_check_capture.setToolTip(
            "Records a two-second clip with the exact command the trial will "
            "use, then measures that file.\n\n"
            "Worth doing whenever the lighting or the mode changes, because the "
            "preview and the recording are not the same thing. The preview is "
            "OpenCV holding the camera in whatever mode the DirectShow backend "
            "negotiated - often YUY2, because the driver refuses MJPG to "
            "OpenCV - at that mode's own frame rate. FFmpeg opens the same "
            "camera in MJPEG at an explicit frame rate, which the driver does "
            "honour. Different mode, different frame interval, different "
            "exposure window, so a different brightness. The file is what the "
            "analysis sees, so the file is what to judge.")
        row.addWidget(self.btn_preview)
        row.addWidget(self.btn_snapshot)
        row.addWidget(self.btn_props)
        row.addWidget(self.btn_check_capture)
        row.addStretch(1)
        self.lbl_negotiated = QtWidgets.QLabel("")
        self.lbl_negotiated.setObjectName("hint")
        row.addWidget(self.lbl_negotiated)
        v.addLayout(row)

        self.lbl_mode_warning = W.StatusLabel("", "info")
        self.lbl_mode_warning.setVisible(False)
        v.addWidget(self.lbl_mode_warning)

        self.view = W.FrameView()
        self.view.set_palette_tokens(self.pal)
        v.addWidget(self.view, 1)

        tiles = QtWidgets.QHBoxLayout()
        self.t_fps = W.MetricTile("preview fps")
        self.t_contrast = W.MetricTile("contrast", "-", "/255")
        self.t_clip = W.MetricTile("clipped", "-", "%")
        self.t_focus = W.MetricTile("focus")
        self.t_uniform = W.MetricTile("uniformity")
        self.t_mean = W.MetricTile("mean level", "-", "/255")
        self.t_mean.setToolTip(
            "Average pixel value of the preview. Compare it with the mean that "
            "'Check capture' reports for the recorded file - if the file is "
            "much darker, the preview is running a longer exposure than the "
            "capture frame rate allows.")
        for t in (self.t_fps, self.t_mean, self.t_contrast, self.t_clip,
                  self.t_focus, self.t_uniform):
            tiles.addWidget(t)
        v.addLayout(tiles)

        self.verdicts = QtWidgets.QVBoxLayout()
        self.verdict_labels = [W.StatusLabel("", "info") for _ in range(4)]
        for lbl in self.verdict_labels:
            self.verdicts.addWidget(lbl)
        v.addLayout(self.verdicts)

        self.btn_preview.clicked.connect(self._toggle_preview)
        self.btn_snapshot.clicked.connect(self._snapshot)
        self.btn_props.clicked.connect(self._open_properties)
        self.btn_check_capture.clicked.connect(self._check_capture)
        self.cb_preview_mode.currentIndexChanged.connect(self._preview_mode_changed)
        return g

    # ---- trial metadata ----
    def _group_trial(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("4 - Trial")
        grid = QtWidgets.QGridLayout(g)
        self.in_animal = QtWidgets.QLineEdit()
        self.in_animal.setPlaceholderText("M001")
        self.in_group = QtWidgets.QLineEdit()
        self.in_group.setPlaceholderText("vehicle / psilocybin")
        self.in_treatment = QtWidgets.QLineEdit()
        self.in_dose = QtWidgets.QLineEdit()
        self.in_dose.setPlaceholderText("mg/kg")
        self.cb_route = QtWidgets.QComboBox()
        self.cb_route.setEditable(True)
        self.cb_route.addItems(["oral gavage", "intraperitoneal", "subcutaneous",
                                "drinking water", "none (baseline)"])
        self.in_operator = QtWidgets.QLineEdit()
        self.in_notes = QtWidgets.QLineEdit()

        fields = [("Animal ID", self.in_animal), ("Group", self.in_group),
                  ("Treatment", self.in_treatment), ("Dose", self.in_dose),
                  ("Route", self.cb_route), ("Operator", self.in_operator)]
        for i, (lab, wdg) in enumerate(fields):
            grid.addWidget(QtWidgets.QLabel(lab), i // 2, (i % 2) * 2)
            grid.addWidget(wdg, i // 2, (i % 2) * 2 + 1)
        r = len(fields) // 2
        grid.addWidget(QtWidgets.QLabel("Notes"), r, 0)
        grid.addWidget(self.in_notes, r, 1, 1, 3)

        self.sp_total = QtWidgets.QDoubleSpinBox()
        self.sp_total.setRange(0.1, 600)
        self.sp_total.setDecimals(1)
        self.sp_total.setSuffix(" min")
        self.sp_baseline = QtWidgets.QDoubleSpinBox()
        self.sp_baseline.setRange(0, 120)
        self.sp_baseline.setDecimals(1)
        self.sp_baseline.setSuffix(" min")
        self.sp_segment = QtWidgets.QDoubleSpinBox()
        self.sp_segment.setRange(0, 120)
        self.sp_segment.setDecimals(1)
        self.sp_segment.setSuffix(" min")
        timing = [("Total duration", self.sp_total), ("Pre-dose baseline", self.sp_baseline),
                  ("Segment length", self.sp_segment)]
        for i, (lab, wdg) in enumerate(timing):
            grid.addWidget(QtWidgets.QLabel(lab), r + 1 + i // 2, (i % 2) * 2)
            grid.addWidget(wdg, r + 1 + i // 2, (i % 2) * 2 + 1)
        for wdg in (self.sp_total, self.sp_segment):
            wdg.valueChanged.connect(self._update_estimate)

        rr = r + 1 + (len(timing) + 1) // 2
        self.in_outdir = QtWidgets.QLineEdit()
        self.btn_outdir = W.button("…")
        self.btn_outdir.setMaximumWidth(32)
        self.btn_outdir.clicked.connect(self._pick_outdir)
        box = QtWidgets.QHBoxLayout()
        box.addWidget(self.in_outdir)
        box.addWidget(self.btn_outdir)
        grid.addWidget(QtWidgets.QLabel("Save to"), rr, 0)
        grid.addLayout(box, rr, 1, 1, 3)

        self.lbl_estimate = QtWidgets.QLabel("")
        self.lbl_estimate.setObjectName("hint")
        self.lbl_estimate.setWordWrap(True)
        grid.addWidget(self.lbl_estimate, rr + 1, 0, 1, 4)
        return g

    # ---- recording ----
    def _group_record(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("5 - Record")
        v = QtWidgets.QVBoxLayout(g)

        row = QtWidgets.QHBoxLayout()
        self.btn_preflight = W.button("Pre-flight check")
        self.btn_start = W.button("Start recording", "primary")
        self.btn_t0 = W.button("Mark T0 (dose)")
        self.btn_t0.setEnabled(False)
        self.btn_stop = W.button("Stop", "danger")
        self.btn_stop.setEnabled(False)
        for b in (self.btn_preflight, self.btn_start, self.btn_t0, self.btn_stop):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)

        self.checks = W.ChecklistTable()
        self.checks.setMaximumHeight(170)
        v.addWidget(self.checks)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        v.addWidget(self.progress)

        tiles = QtWidgets.QHBoxLayout()
        self.r_elapsed = W.MetricTile("elapsed")
        self.r_fps = W.MetricTile("fps")
        self.r_frames = W.MetricTile("frames")
        self.r_drop = W.MetricTile("dropped")
        self.r_size = W.MetricTile("file size")
        for t in (self.r_elapsed, self.r_fps, self.r_frames, self.r_drop, self.r_size):
            tiles.addWidget(t)
        v.addLayout(tiles)

        self.lbl_rec = W.StatusLabel("Idle.", "info")
        v.addWidget(self.lbl_rec)

        self.btn_preflight.clicked.connect(self._preflight)
        self.preflight_job._button = self.btn_preflight
        self.btn_start.clicked.connect(self._start_recording)
        self.btn_t0.clicked.connect(self._mark_t0)
        self.btn_stop.clicked.connect(self._stop_recording)
        return g

    # ------------------------------------------------------------------
    # Config binding
    # ------------------------------------------------------------------
    def _load_from_config(self) -> None:
        c = self.ctx.cfg
        name = c.get("camera.preset", "")
        idx = self.cb_preset.findText(name)
        if idx >= 0:
            self.cb_preset.setCurrentIndex(idx)
        self.sp_w.setValue(int(c.get("camera.width", 320)))
        self.sp_h.setValue(int(c.get("camera.height", 240)))
        self.sp_fps.setValue(float(c.get("camera.requested_fps", 200)))
        pf = self.cb_pixfmt.findText(str(c.get("camera.pixel_format", "mjpeg")))
        self.cb_pixfmt.setCurrentIndex(max(0, pf))
        self._apply_preset(self.cb_preset.currentIndex(), push_mode=False)

        self.chk_auto.setChecked(bool(c.get("camera.auto_exposure", False)))
        self.sl_exposure.set_value(float(c.get("camera.exposure", -7)))
        self.sl_gain.set_value(float(c.get("camera.gain", 0)))
        self.sl_brightness.set_value(float(c.get("camera.brightness", 0) or 0))
        self.sl_contrast.set_value(float(c.get("camera.contrast", 32) or 32))
        self.sl_gamma.set_value(float(c.get("camera.gamma", 100) or 100))

        pm = self.cb_preview_mode.findData(str(c.get("camera.preview_mode", "capture")))
        self.cb_preview_mode.blockSignals(True)
        self.cb_preview_mode.setCurrentIndex(max(0, pm))
        self.cb_preview_mode.blockSignals(False)

        self.sp_total.setValue(float(c.get("recording.total_seconds", 4200)) / 60.0)
        self.sp_baseline.setValue(float(c.get("recording.baseline_seconds", 600)) / 60.0)
        seg = float(c.get("recording.segment_seconds", 300)) / 60.0
        self.sp_segment.setValue(seg if c.get("recording.segment_enabled", True) else 0)
        self.in_outdir.setText(str(self.ctx.cfg.recordings_root()))
        self.in_operator.setText(str(c.get("recording.operator", "")))
        self._update_estimate()

    def save_to_config(self) -> None:
        c = self.ctx.cfg
        c.set("camera.preset", self.cb_preset.currentText())
        c.set("camera.width", int(self.sp_w.value()))
        c.set("camera.height", int(self.sp_h.value()))
        c.set("camera.requested_fps", float(self.sp_fps.value()))
        c.set("camera.pixel_format", self.cb_pixfmt.currentText())
        c.set("camera.auto_exposure", self.chk_auto.isChecked())
        c.set("camera.exposure", self.sl_exposure.value())
        c.set("camera.gain", self.sl_gain.value())
        c.set("camera.brightness", self.sl_brightness.value())
        c.set("camera.contrast", self.sl_contrast.value())
        c.set("camera.gamma", self.sl_gamma.value())
        c.set("camera.preview_mode", self.cb_preview_mode.currentData() or "capture")
        if self.cb_device.currentText():
            c.set("camera.device_name", self.cb_device.currentText())
        data = self.cb_index.currentData()
        if data is not None:
            c.set("camera.opencv_index", int(data))
        c.set("recording.total_seconds", self.sp_total.value() * 60.0)
        c.set("recording.baseline_seconds", self.sp_baseline.value() * 60.0)
        seg_min = self.sp_segment.value()
        c.set("recording.segment_enabled", seg_min > 0)
        if seg_min > 0:
            c.set("recording.segment_seconds", seg_min * 60.0)
        c.set("recording.root", self.in_outdir.text().strip())
        c.set("recording.operator", self.in_operator.text().strip())

    # ------------------------------------------------------------------
    # Device discovery
    # ------------------------------------------------------------------
    def refresh_devices(self) -> None:
        """Enumerate cameras, off the GUI thread.

        Both halves of this are slow, and the second is punishing: probing
        OpenCV indices opens up to six DirectShow devices, and on Windows an
        index with nothing behind it can block for a second or more before it
        gives up. enumerate_indices' own docstring says to call it from a
        worker thread; it was being called from the GUI thread, at startup and
        on every Refresh, which is what made the Acquisition tab feel stuck.
        """
        if not paths.is_windows():
            self.lbl_device.set(
                "Camera enumeration uses Windows DirectShow, which is not available "
                "here. Analysis, Library and Review all work; acquisition does not.",
                "warn")
            self.btn_preview.setEnabled(False)
            self.btn_start.setEnabled(False)
            return
        if self.preview is not None:
            self.lbl_device.set(
                "Stop the preview first - enumerating claims the camera.", "warn")
            return
        self.lbl_device.set("Enumerating cameras…", "info")
        ffpath = self.ctx.cfg.get("paths.ffmpeg", "")
        backend = self.ctx.cfg.get("camera.opencv_backend", "dshow")

        def job(progress=None):
            devices = ffmpeg_tools.list_dshow_devices(ffpath)
            if progress:
                progress(1, 2, "probing preview indices")
            indices = camera.enumerate_indices(6, backend)
            return devices, indices

        if not self.devices_job.start(job):
            self.lbl_device.set("Already enumerating…", "warn")

    def _devices_done(self, result) -> None:
        self.devices, self.indices = result
        self.cb_device.clear()
        for d in self.devices:
            self.cb_device.addItem(d.name)
        wanted = self.ctx.cfg.get("camera.device_name", "")
        match = ffmpeg_tools.match_device(self.devices, wanted)
        if match:
            self.cb_device.setCurrentText(match.name)

        self.cb_index.clear()
        for p in self.indices:
            self.cb_index.addItem(p.label, p.index)
        cfg_idx = int(self.ctx.cfg.get("camera.opencv_index", -1))
        pick = cfg_idx if cfg_idx >= 0 else camera.guess_arducam_index(self.indices)
        for i in range(self.cb_index.count()):
            if self.cb_index.itemData(i) == pick:
                self.cb_index.setCurrentIndex(i)
                break

        from_pnp = [d for d in self.devices if d.source == "pnp"]
        if not self.devices:
            typed = str(self.ctx.cfg.get("camera.device_name", "") or "")
            if typed:
                self.cb_device.setEditText(typed)
            hint = (f" The configured name \"{typed}\" is filled in and can still "
                    "be used - recording does not depend on this list." if typed
                    else "")
            if self.indices:
                self.lbl_device.set(
                    f"Neither FFmpeg nor Windows listed a camera, but OpenCV "
                    f"opened {len(self.indices)} device(s), so one is attached. "
                    "Run tools\\camera_diagnostics.bat and send me the file it "
                    "writes, or type the device name into the box above." + hint,
                    "warn")
            else:
                self.lbl_device.set(
                    "No DirectShow video device found. Check the USB connection, "
                    "then confirm the camera appears in Windows Camera. Note that "
                    "nothing else may hold the camera open." + hint, "bad")
        elif from_pnp:
            name = match.name if match else self.devices[0].name
            self.lbl_device.set(
                f"FFmpeg listed no devices, so these {len(self.devices)} name(s) "
                f"came from the Windows device database instead; using "
                f"\"{name}\". That name is usually exactly what DirectShow "
                "calls it, but not always - if recording fails, run "
                "tools\\camera_diagnostics.bat and correct the name here.", "warn")
        elif match:
            self.lbl_device.set(
                f"Found {len(self.devices)} camera(s); using \"{match.name}\". "
                f"{len(self.indices)} OpenCV index(es) available for preview.", "ok")
        else:
            self.lbl_device.set(
                f"Found {len(self.devices)} camera(s), but none looks like the "
                f"Arducam B0634. Pick the right one by hand.", "warn")

    def _list_modes(self) -> None:
        name = self.cb_device.currentText()
        if not name:
            return
        try:
            text = ffmpeg_tools.device_options(
                self.ctx.cfg.get("paths.ffmpeg", ""), name)
        except ffmpeg_tools.FFmpegMissing as e:
            W.warn_box(self, "FFmpeg missing", str(e))
            return
        modes = ffmpeg_tools.parse_device_options(text)
        if not modes:
            W.info_box(self, "Camera modes",
                       "The driver did not report any modes. Raw output:\n\n"
                       + text[-2500:])
            return
        lines = "\n".join(f"  {m.label}" for m in modes)
        W.info_box(self, "Camera modes advertised by the driver",
                   f"{name}\n\n{lines}\n\n"
                   "Advertised is not the same as delivered. Use 'Probe this mode' "
                   "to measure what the camera actually sustains.")

    def _open_props(self) -> None:
        name = self.cb_device.currentText()
        if not name:
            return
        was_running = self.preview is not None and self.preview.running
        if was_running:
            self._stop_preview()
        try:
            ffmpeg_tools.open_device_property_dialog(
                self.ctx.cfg.get("paths.ffmpeg", ""), name)
        except ffmpeg_tools.FFmpegMissing as e:
            W.warn_box(self, "FFmpeg missing", str(e))
            return
        W.info_box(self, "Camera properties",
                   "The driver's own property page is opening in a separate window.\n\n"
                   "Set exposure and gain there if the sliders here do not hold: "
                   "values set in the driver dialog persist after the camera handle "
                   "closes, so FFmpeg inherits them when recording starts.\n\n"
                   "Close that window when you are done, then start the preview again.")

    # ------------------------------------------------------------------
    # Mode and probing
    # ------------------------------------------------------------------
    def _apply_preset(self, index: int, push_mode: bool = True) -> None:
        data = self.cb_preset.itemData(index)
        if not isinstance(data, dict):
            return
        self.lbl_preset_note.setText(data.get("note", ""))
        if push_mode:
            self.sp_w.setValue(int(data["width"]))
            self.sp_h.setValue(int(data["height"]))
            self.sp_fps.setValue(float(data["requested_fps"]))
            i = self.cb_pixfmt.findText(str(data.get("pixel_format", "mjpeg")))
            self.cb_pixfmt.setCurrentIndex(max(0, i))
        self._update_estimate()

    def _current_mode(self) -> tuple[int, int, float, str]:
        return (int(self.sp_w.value()), int(self.sp_h.value()),
                float(self.sp_fps.value()), self.cb_pixfmt.currentText())

    def _probe(self, modes: list[tuple[int, int, float, str]]) -> None:
        name = self.cb_device.currentText()
        if not name:
            W.warn_box(self, "No camera", "Select a DirectShow device first.")
            return
        if self.preview is not None and self.preview.running:
            self._stop_preview()
        self.lbl_probe.set(f"Probing {len(modes)} mode(s); the camera is claimed "
                           "while this runs…", "info")
        self._set_busy(True)

        ffpath = self.ctx.cfg.get("paths.ffmpeg", "")
        rtbuf = str(self.ctx.cfg.get("camera.rtbufsize", "512M"))

        def job(progress=None):
            out = []
            for i, (w, h, fps, pf) in enumerate(modes):
                if progress and not progress(i, len(modes), f"{w}x{h} @ {fps:g}"):
                    break
                out.append(ffmpeg_tools.probe_mode(ffpath, name, w, h, fps,
                                                   seconds=2.0, pixel_format=pf,
                                                   rtbufsize=rtbuf))
            return out

        if not self.probe_job.start(job):
            self.lbl_device.set("A probe is already running.", "warn")
            self._set_busy(False)

    def _probe_done(self, results: list[ffmpeg_tools.ProbeResult]) -> None:
        self._set_busy(False)
        self.tbl_probe.setRowCount(len(results))
        best = None
        for r, res in enumerate(results):
            vals = [f"{res.width}x{res.height}", f"{res.requested_fps:g} fps",
                    f"{res.achieved_fps:.1f} fps" if res.achieved_fps else "-",
                    str(res.frames), res.message[:60]]
            for c, text in enumerate(vals):
                self.tbl_probe.setItem(r, c, QtWidgets.QTableWidgetItem(text))
            if res.ok and (best is None or res.achieved_fps > best.achieved_fps):
                best = res
        if best is None:
            self.lbl_probe.set(
                "No mode delivered any frames. The camera may be held by another "
                "program, or the device name may be wrong.", "bad")
            return
        req = best.requested_fps
        note = ""
        if abs(best.achieved_fps - req) / max(req, 1) > 0.08:
            note = (f" Note the camera delivered {best.achieved_fps:.0f} fps for a "
                    f"{req:g} fps request - it negotiates, so always record against "
                    "the measured figure.")
        level = "ok" if best.achieved_fps >= 200 else "warn"
        extra = ("" if best.achieved_fps >= 200 else
                 " Below 200 fps the 78-98 Hz HTR band is not safely resolved.")
        self.lbl_probe.set(
            f"Best: {best.width}x{best.height} delivered {best.achieved_fps:.1f} fps "
            f"({best.frames} frames in {best.duration_s:.2f} s).{note}{extra}", level)

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    def _uvc(self) -> camera.UVCSettings:
        w, h, fps, pf = self._current_mode()
        return camera.UVCSettings(
            width=w, height=h, fps=fps,
            fourcc="MJPG" if pf.startswith("mjp") else "YUY2",
            auto_exposure=self.chk_auto.isChecked(),
            exposure=self.sl_exposure.value(), gain=self.sl_gain.value(),
            brightness=self.sl_brightness.value(),
            contrast=self.sl_contrast.value(), gamma=self.sl_gamma.value())

    def _toggle_preview(self) -> None:
        if self.preview is not None and self.preview.running:
            self._stop_preview()
        else:
            self._start_preview()

    @property
    def _true_preview(self) -> bool:
        return self.cb_preview_mode.currentData() == "capture"

    def _start_preview(self) -> None:
        if self.rec is not None and self.rec.running:
            W.warn_box(self, "Recording in progress",
                       "The camera is busy recording. Stop the recording first - a "
                       "DirectShow device only admits one client at a time.")
            return
        self._warm_up()
        if self._true_preview:
            self._start_true_preview()
        else:
            self._start_opencv_preview()

    #: Frame shapes already warmed, so the cost is paid once per mode.
    _warmed: set = set()

    def _warm_up(self) -> None:
        """Pay OpenCV's and Qt's first-call costs before the camera is live.

        Measured: the first ``frame_metrics`` call costs 17 ms against 3 ms
        thereafter, and the first ``show_frame`` 21 ms against 1.2 ms -- 37 ms
        of lazy initialisation inside OpenCV and Qt's pixmap path. Left
        alone, all of it lands on the first preview frame, which is precisely
        the moment the operator is watching, and it showed up as a 46 ms tick
        in the latency test.

        At the *frame's own size*, not a token 16x16 one: warming a tiny
        array was tried and did nothing, because OpenCV allocates its
        scratch buffers per shape. So the warm-up has to use the shape the
        preview will deliver, which is the configured capture size.
        """
        try:
            w, h, _fps, _pf = self._current_mode()
            key = (int(h), int(w))
            if key in AcquisitionTab._warmed or min(key) <= 0:
                return
            AcquisitionTab._warmed.add(key)
            # Not zeros: an all-zero frame short-circuits some of the work.
            blank = np.full(key, 128, dtype=np.uint8)
            blank[:: max(1, key[0] // 8)] = 60
            camera.frame_metrics(blank)
            self.view.show_frame(blank, {"crosshair": True}, "")
            self.view.setText("")
        except Exception:  # pragma: no cover - warm-up must never block a trial
            log.debug("preview warm-up failed", exc_info=True)

    def _start_true_preview(self) -> None:
        """Preview through the capture pipeline, so it shows the real thing."""
        if not self.cb_device.currentText():
            W.warn_box(self, "No device",
                       "Pick a DirectShow device first - the true preview opens "
                       "the camera the same way the recording does, by name.")
            return
        try:
            spec = self._spec()
        except Exception as e:  # pragma: no cover - during construction
            W.warn_box(self, "Could not build the capture settings", str(e))
            return
        self.preview = camera.CapturePreview(
            spec, dict(self.ctx.cfg.get("paths", {}) or {}),
            display_fps=float(self.ctx.cfg.get("camera.preview_display_fps", 25)))
        if not self.preview.start():
            W.warn_box(self, "Could not start the true preview",
                       self.preview.last_error)
            self.preview = None
            return
        self.lbl_negotiated.setText(
            f"capture pipeline: {spec.width}x{spec.height} @ "
            f"{spec.requested_fps:g} {spec.pixel_format}")
        self.lbl_mode_warning.set(
            "True preview: this is the capture pipeline, so what you see is "
            "what will be recorded - same mode, frame rate and exposure "
            "window. FFmpeg holds the camera, so a control change reopens it "
            "(about a second). Use 'Camera properties' to set exposure and "
            "gain in the driver's own page, which is the only way to set them "
            "in the capture mode on this camera.", "info")
        self.lbl_mode_warning.setVisible(True)
        self.btn_preview.setText("Stop preview")
        self.btn_snapshot.setEnabled(True)
        self.preview_timer.start()
        self.statusMessage.emit("True preview running (capture pipeline)")

    def _start_opencv_preview(self) -> None:
        data = self.cb_index.currentData()
        if data is None:
            W.warn_box(self, "No preview device",
                       "No OpenCV camera index was found. Press Refresh; if the list "
                       "stays empty, close anything else using the camera.")
            return
        self.preview = camera.CameraPreview(
            int(data), self._uvc(), self.ctx.cfg.get("camera.opencv_backend", "dshow"))
        if not self.preview.start():
            W.warn_box(self, "Could not open camera", self.preview.last_error)
            self.preview = None
            return
        a = self.preview.applied
        self.lbl_negotiated.setText(
            f"driver gave {a.width}x{a.height} @ {a.fps:.0f} {a.fourcc}")
        for wmsg in a.warnings:
            log.warning("preview: %s", wmsg)
        self._warn_mode_mismatch(a)
        self.btn_preview.setText("Stop preview")
        self.btn_snapshot.setEnabled(True)
        self.preview_timer.start()
        self.statusMessage.emit("Fast preview running (OpenCV)")

    def _stop_preview(self) -> None:
        self.preview_timer.stop()
        self._reopen_timer.stop()
        if self.preview is not None:
            self.preview.stop()
            self.preview = None
        # Reset, so the next start shows "waiting for the first frame" rather
        # than the previous session's numbers beside a blank pane. The spec
        # written into a sidecar reads these, so stale ones would be recorded
        # as this trial's lighting.
        self._last_metrics = None
        self._metrics_at = 0.0
        for t in (self.t_fps, self.t_mean, self.t_contrast, self.t_clip,
                  self.t_focus, self.t_uniform):
            t.set_value("-")
        self.btn_preview.setText("Start preview")
        self.btn_snapshot.setEnabled(False)
        self.statusMessage.emit("Preview stopped")

    def _preview_mode_changed(self, *_args) -> None:
        """Switching mode restarts a running preview, and nothing else."""
        if self.preview is not None and self.preview.running:
            self._stop_preview()
            self._start_preview()
        else:
            self.lbl_mode_warning.setVisible(False)

    def _open_properties(self) -> None:
        if self.rec is not None and self.rec.running:
            W.warn_box(self, "Recording in progress",
                       "Stop the recording before opening the driver's page.")
            return
        if not self.cb_device.currentText():
            W.warn_box(self, "No device", "Pick a DirectShow device first.")
            return
        # The page belongs to the capture filter, so the device has to be free.
        was_running = self.preview is not None and self.preview.running
        if was_running:
            self._stop_preview()
        msg = camera.open_device_properties(
            self._spec(), dict(self.ctx.cfg.get("paths", {}) or {}))
        self.lbl_mode_warning.set(msg, "info")
        self.lbl_mode_warning.setVisible(True)
        self.statusMessage.emit(msg)
        if was_running:
            W.info_box(self, "Preview stopped",
                       "The preview was stopped so the driver's property page "
                       "could take the camera. Start it again when you have "
                       "closed the page.")

    def _push_controls(self, *_args) -> None:
        """A control changed. How that is applied depends on the preview.

        The fast OpenCV preview owns the handle, so a control can be pushed
        into it and the effect is immediate. The true preview is FFmpeg
        holding the device, and FFmpeg cannot set UVC controls, so the only
        way to apply one is to close the device, set it through a brief
        OpenCV handle, and reopen - about a second. A slider drag emits one
        change per pixel of travel, so that cannot happen per emission: the
        timer coalesces a drag into one reopen after the operator lets go.
        """
        if self.preview is None or not self.preview.running:
            return
        if self._true_preview:
            self._reopen_timer.start()
            return
        applied = self.preview.push(self._uvc())
        if applied.warnings:
            self.lbl_negotiated.setText(" | ".join(applied.warnings)[:120])
        self._warn_mode_mismatch(applied)

    def _reopen_true_preview(self) -> None:
        """Apply the pending control change and bring the preview back."""
        if self.preview is None or not self._true_preview:
            return
        self.preview_timer.stop()
        self.preview.stop()
        idx = self.cb_index.currentData()
        if idx is not None and self.ctx.cfg.get("camera.apply_uvc_before_record", True):
            st = camera.apply_uvc_oneshot(
                int(idx), self._uvc(),
                self.ctx.cfg.get("camera.opencv_backend", "dshow"))
            for wmsg in st.warnings:
                log.info("true preview control push: %s", wmsg)
        if not self.preview.start():
            self.lbl_mode_warning.set(
                "Could not reopen the preview: " + self.preview.last_error, "bad")
            self.lbl_mode_warning.setVisible(True)
            self.btn_preview.setText("Start preview")
            self.btn_snapshot.setEnabled(False)
            self.preview = None
            return
        self.preview_timer.start()

    def _warn_mode_mismatch(self, applied) -> None:
        """Say, in the window, when the preview cannot represent the capture.

        Two independent reasons the recorded file will not look like the
        preview, both of which were previously visible only as a line in the
        log file:

        **Different media type.** The DirectShow backend negotiates on
        OpenCV's behalf and the B0634's driver refuses MJPG to it, falling
        back to YUY2. FFmpeg asks for MJPEG explicitly and gets it. Those are
        two different sensor modes with their own default exposure and gain
        behaviour, and the UVC controls pushed while the camera is in one do
        not necessarily carry into the other.

        **Exposure longer than the capture frame interval.** This is the one
        that actually makes the file darker, and it is physics, not a
        setting: at 200 fps a frame lasts 5 ms, so the sensor cannot integrate
        for the 7.8 ms that an exposure value of -7 asks for. The driver
        clamps it. If the preview is running at a lower rate -- which it is
        whenever the media type differs, because the DSHOW backend's
        CAP_PROP_FPS is an echo of the request rather than a negotiated value
        -- the preview gets the full 7.8 ms and the capture gets 5 ms or less.
        Brightness is proportional to integration time, so the ratio of the
        two is roughly the ratio of the brightness.

        The right answer in that case is more light, not a longer exposure:
        7.8 ms of integration smears a 78-98 Hz twitch across most of a
        cycle, which is the signal this whole tool exists to measure.
        """
        msgs: list[str] = []
        _, _, want_fps, want_pf = self._current_mode()
        want_fourcc = "MJPG" if want_pf.startswith("mjp") else "YUY2"
        got = (applied.fourcc or "").upper()
        if got and got not in (want_fourcc.upper(), ""):
            msgs.append(
                f"the preview is running in {applied.fourcc}, but the recording "
                f"will use {want_pf.upper()} - two different sensor modes, so "
                "the brightness you see here is not the brightness you will get")

        if not self.chk_auto.isChecked() and want_fps > 0:
            want_ms = camera.exposure_hint_ms(self.sl_exposure.value())
            frame_ms = 1000.0 / want_fps
            if want_ms > frame_ms * 1.05:
                msgs.append(
                    f"exposure asks for {want_ms:.1f} ms but a frame at "
                    f"{want_fps:.0f} fps lasts {frame_ms:.1f} ms, so the "
                    f"driver will clamp it - expect the capture to be about "
                    f"{want_ms / frame_ms:.1f}x darker than this preview. "
                    "Add light rather than lengthening the exposure: at this "
                    "rate a longer exposure would blur the twitch away")

        if msgs:
            self.lbl_mode_warning.set("  •  ".join(msgs), "warn")
            self.lbl_mode_warning.setVisible(True)
        else:
            self.lbl_mode_warning.setVisible(False)


    def _draw_histogram(self, m) -> None:
        """Update the intensity histogram, and never let it stop a recording.

        stepMode wants one more x than y: the x are bin EDGES and the y are the
        bin counts. Padding the counts with a duplicate last bin made both
        arrays the same length, so pyqtgraph rejected the call -- on every
        preview tick, through a modal error box, which made the Acquisition tab
        impossible to use with a camera attached.

        The guard matters as much as the fix. This is a cosmetic readout; if a
        future pyqtgraph changes its mind about the arguments again, the right
        outcome is a dead histogram and a line in the log, not a blocked trial
        with the animal already dosed.
        """
        if self._hist_broken or m.histogram is None or not len(m.histogram):
            return
        try:
            counts = np.asarray(m.histogram, dtype=float).ravel()
            edges = np.linspace(0, 256, counts.size + 1)
            self.hist_curve.setData(edges, counts, stepMode="center")
        except Exception:
            self._hist_broken = True
            log.exception("intensity histogram disabled after a plotting error")
            self.plot_hist.setTitle("Intensity histogram (unavailable - see log)")

    def _tick_preview(self) -> None:
        """One preview frame: draw it and update the lighting readouts.

        The interval adapts to what the tick actually costs. A repeating timer
        whose handler takes longer than its interval does not merely run late
        -- it leaves no gap for anything else, so clicks sit in the queue
        behind the next frame and the whole window feels stuck. The metrics
        are the expensive part and they scale with frame area, so a VGA
        preview on a slower machine can exceed the 40 ms period while a QVGA
        one never comes close. Rather than guess, measure and back off, and
        recover when it gets cheap again.
        """
        if self.preview is None:
            return
        if not self.preview.running:
            # Do not just return. The camera being unplugged, or FFmpeg being
            # killed, leaves a stopped preview and a frozen last frame, and
            # bailing out here would show that stale frame indefinitely with
            # no indication anything was wrong. Say what happened and tidy up.
            err = self.preview.last_error or "the camera stopped delivering frames"
            self.lbl_mode_warning.set(f"Preview stopped: {err}", "bad")
            self.lbl_mode_warning.setVisible(True)
            log.warning("preview stopped: %s", err)
            self._stop_preview()
            return
        t_start = time.perf_counter()
        gray = self.preview.read()
        if gray is None:
            if self._last_metrics is None:
                # Opening a DirectShow graph takes about a second, so the
                # first frame is not instant. Say so, rather than leaving a
                # blank pane that looks like a failure.
                self.view.setText("waiting for the first frame from the camera…")
            return

        # Drawing is cheap (1 ms at QVGA, 0.3 at VGA); the metrics are not --
        # measured at 3.8 ms for QVGA and 14.5 ms for VGA, nearly all of it
        # the Laplacian variance behind the focus score. Recomputing those at
        # the display rate would spend 36% of the event loop at VGA on
        # numbers nobody can read changing 25 times a second. So the picture
        # updates every tick and the readouts a few times a second.
        self.view.show_frame(gray, {"crosshair": True},
                             caption=f"{gray.shape[1]}x{gray.shape[0]}")
        self.t_fps.set_value(f"{self.preview.measured_fps():.0f}")

        now = time.perf_counter()
        due = (self._last_metrics is None
               or (now - self._metrics_at) * 1000.0 >= self.METRICS_INTERVAL_MS)
        if due:
            self._metrics_at = now
            m = camera.frame_metrics(gray)
            self._last_metrics = m
            self.t_mean.set_value(f"{m.mean:.0f}")
            self.t_contrast.set_value(f"{m.contrast:.0f}")
            self.t_clip.set_value(f"{m.clipped_high_pct:.1f}")
            self.t_focus.set_value(f"{m.focus_score:.0f}")
            self.t_uniform.set_value(f"{m.uniformity:.2f}")
            self._draw_histogram(m)
            for i, lbl in enumerate(self.verdict_labels):
                if i < len(m.verdicts):
                    level, text = m.verdicts[i]
                    lbl.set(text, level)
                else:
                    lbl.set("", "info")
        self._pace_preview((time.perf_counter() - t_start) * 1000.0)

    #: Never let the preview consume more than this share of the event loop.
    PREVIEW_DUTY = 0.5

    #: How often the lighting readouts are recomputed. They are judgement
    #: aids for a human setting up a rig, not a signal, and four times a
    #: second is already faster than anyone can read them.
    METRICS_INTERVAL_MS = 250.0

    def _pace_preview(self, cost_ms: float) -> None:
        """Keep the preview below PREVIEW_DUTY of the event loop's time."""
        self._tick_cost_ms = (0.8 * getattr(self, "_tick_cost_ms", cost_ms)
                              + 0.2 * cost_ms)
        want = int(max(PREVIEW_MIN_INTERVAL_MS,
                       min(PREVIEW_MAX_INTERVAL_MS,
                           self._tick_cost_ms / self.PREVIEW_DUTY)))
        cur = self.preview_timer.interval()
        # Only move in steps, so the rate does not oscillate frame to frame.
        if want > cur * 1.3 or want < cur * 0.7:
            self.preview_timer.setInterval(want)
            log.debug("preview interval %d -> %d ms (tick costs %.1f ms)",
                      cur, want, self._tick_cost_ms)

    def _snapshot(self) -> None:
        paths.ensure_dirs()
        default = paths.EXPORTS_DIR / "camera_snapshot.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save snapshot", str(default), "PNG (*.png)")
        if path and self.view.save_png(Path(path)):
            self.statusMessage.emit(f"Snapshot saved to {path}")

    def _check_capture(self) -> None:
        """Record two seconds with the real command and report the file."""
        if self.rec is not None and self.rec.running:
            W.warn_box(self, "Recording in progress",
                       "Stop the trial recording first.")
            return
        if not self.cb_device.currentText():
            W.warn_box(self, "No device", "Pick a DirectShow device first.")
            return
        # The camera admits one client at a time, so the preview has to go.
        was_previewing = self.preview is not None and self.preview.running
        if was_previewing:
            self._stop_preview()
        self._preview_metrics_at_check = self._last_metrics

        import copy
        spec = copy.deepcopy(self._spec())
        spec.duration_s = CHECK_CAPTURE_S
        spec.segment_enabled = False
        spec.out_dir = paths.CACHE_DIR
        spec.base_name = "capture_check"
        self._check_was_previewing = was_previewing
        self.check_job.start(_check_capture, spec,
                             dict(self.ctx.cfg.get("paths", {}) or {}))

    def _check_capture_done(self, res: dict) -> None:
        if self._check_was_previewing:
            self._start_preview()
        if not res.get("ok"):
            self.lbl_mode_warning.set(
                "Check capture could not record: " + (res.get("note") or "unknown"),
                "bad")
            self.lbl_mode_warning.setVisible(True)
            return

        m = res["metrics"]
        info = res.get("info")
        pre = getattr(self, "_preview_metrics_at_check", None)
        lines = [
            f"<b>The recorded file</b>: mean {m.mean:.0f}/255, contrast "
            f"{m.contrast:.0f}/255, {m.clipped_high_pct:.1f}% blown out, "
            f"{m.clipped_low_pct:.0f}% crushed to black, uniformity "
            f"{m.uniformity:.2f}, focus {m.focus_score:.0f}"]
        if info is not None:
            lines.append(
                f"{info.width}x{info.height} {info.codec} at "
                f"{info.true_fps:.1f} fps measured over {CHECK_CAPTURE_S:.0f} s")
        if pre is not None and pre.mean > 1:
            ratio = m.mean / pre.mean
            verb = ("darker" if ratio < 0.95 else
                    "brighter" if ratio > 1.05 else "the same brightness")
            lines.append(
                f"<b>Preview was mean {pre.mean:.0f}/255</b>, so the file is "
                + (f"{1 / ratio:.1f}x {verb} than the preview."
                   if ratio < 0.95 else
                   f"{ratio:.1f}x {verb} than the preview."
                   if ratio > 1.05 else "the same brightness as the preview."))
        for level, text in m.verdicts:
            lines.append(f"[{level}] {text}")

        worst = "bad" if any(v[0] == "bad" for v in m.verdicts) else \
                "warn" if any(v[0] == "warn" for v in m.verdicts) else "ok"
        W.info_box(self, "What the recording actually looks like",
                   "<br>".join(lines))
        self.lbl_mode_warning.set(
            f"Checked capture: file mean {m.mean:.0f}/255, contrast "
            f"{m.contrast:.0f}/255"
            + (f" (preview was {pre.mean:.0f})" if pre is not None else ""),
            worst)
        self.lbl_mode_warning.setVisible(True)
        log.info("capture check: %s", " | ".join(
            ln.replace("<b>", "").replace("</b>", "") for ln in lines))

    def _flicker(self) -> None:
        """Measure light ripple at the capture frame rate.

        Not at the preview's rate. Whether a PWM ripple lands in the
        55-125 Hz band depends on the sampling rate -- a 400 Hz ripple folds
        to 160 Hz at 240 fps and somewhere else entirely at 25 -- so the only
        verdict worth having is the one taken at the rate the trial will run.
        That means FFmpeg and the capture pipeline, not the preview handle.
        """
        if self.rec is not None and self.rec.running:
            W.warn_box(self, "Recording in progress",
                       "Stop the recording before running the flicker test.")
            return
        if not self.cb_device.currentText():
            W.warn_box(self, "No device", "Pick a DirectShow device first.")
            return
        self._flicker_was_previewing = (self.preview is not None
                                       and self.preview.running)
        if self._flicker_was_previewing:
            self._stop_preview()
        self.lbl_flicker.set("Sampling at the capture frame rate…", "info")
        self.flicker_job.start(
            camera.flicker_from_capture, self._spec(),
            dict(self.ctx.cfg.get("paths", {}) or {}),
            seconds=float(self.ctx.cfg.get("camera.flicker_seconds", 2.0)),
            mains_hz=float(self.ctx.cfg.get("camera.mains_hz", 50.0)))

    def _flicker_done(self, res) -> None:
        if getattr(self, "_flicker_was_previewing", False):
            self._start_preview()
        self.lbl_flicker.set(
            f"{res.verdict}  [{res.frames} frames at {res.fps:.0f} fps]", res.level)
        log.info("flicker: %s [%d frames at %.1f fps]",
                 res.verdict, res.frames, res.fps)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def _pick_outdir(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Recordings folder", self.in_outdir.text() or str(paths.RECORDINGS_DIR))
        if d:
            self.in_outdir.setText(d)
            self._update_estimate()

    def _meta(self) -> recorder.TrialMeta:
        return recorder.TrialMeta(
            animal_id=self.in_animal.text().strip(),
            group=self.in_group.text().strip(),
            treatment=self.in_treatment.text().strip(),
            dose_mg_kg=self.in_dose.text().strip(),
            route=self.cb_route.currentText().strip(),
            operator=self.in_operator.text().strip(),
            notes=self.in_notes.text().strip(),
            planned_baseline_s=self.sp_baseline.value() * 60.0)

    def _spec(self) -> recorder.RecordingSpec:
        w, h, fps, pf = self._current_mode()
        meta = self._meta()
        out_dir = Path(self.in_outdir.text().strip() or paths.RECORDINGS_DIR)
        seg_min = self.sp_segment.value()
        return recorder.RecordingSpec(
            device_name=self.cb_device.currentText(),
            width=w, height=h, requested_fps=fps, pixel_format=pf,
            duration_s=self.sp_total.value() * 60.0,
            segment_enabled=seg_min > 0,
            segment_seconds=max(seg_min, 0.0) * 60.0,
            out_dir=out_dir,
            base_name=recorder.default_base_name(
                str(self.ctx.cfg.get("recording.name_template",
                                     "{animal}_{group}_{date}_{time}")), meta),
            container=str(self.ctx.cfg.get("recording.container", "avi")),
            rtbufsize=str(self.ctx.cfg.get("camera.rtbufsize", "512M")),
            thread_queue_size=int(self.ctx.cfg.get("camera.thread_queue_size", 4096)),
            camera_settings={
                "width": w, "height": h, "requested_fps": fps, "pixel_format": pf,
                "auto_exposure": self.chk_auto.isChecked(),
                "exposure": self.sl_exposure.value(),
                "exposure_estimate_ms": round(
                    camera.exposure_hint_ms(self.sl_exposure.value()), 3),
                "gain": self.sl_gain.value(),
                "brightness": self.sl_brightness.value(),
                "contrast": self.sl_contrast.value(),
                "gamma": self.sl_gamma.value(),
                "opencv_index": self.cb_index.currentData(),
                "lighting_metrics": ({
                    "mean": round(self._last_metrics.mean, 2),
                    "contrast": round(self._last_metrics.contrast, 2),
                    "clipped_high_pct": round(self._last_metrics.clipped_high_pct, 3),
                    "uniformity": round(self._last_metrics.uniformity, 3),
                    "focus_score": round(self._last_metrics.focus_score, 1),
                } if self._last_metrics else None),
            },
            meta=meta)

    def _update_estimate(self) -> None:
        try:
            spec = self._spec()
        except Exception:  # pragma: no cover - during construction
            return
        n_seg = (int(spec.duration_s // spec.segment_seconds) + 1
                 if spec.is_segmented else 1)
        self.lbl_estimate.setText(
            f"{spec.duration_s / 60:.0f} min at {spec.expected_fps():.0f} fps "
            f"≈ {spec.estimated_bytes() / 1e9:.1f} GB"
            + (f", written as {n_seg} segment(s) of {spec.segment_seconds / 60:.0f} min "
               f"in a folder named after the trial." if spec.is_segmented
               else ", as one file.")
            + "  Segmenting means an interruption costs only the last segment.")

    def _preflight(self) -> None:
        """Run the pre-flight checks off the GUI thread.

        They are not cheap and two of them are genuinely slow: the write-path
        probe writes 6 MB in fsync'd pieces, and the free-space and name checks
        touch the recordings disk, which may be a network share. Pressing the
        button should not stop the window, least of all on a rig where that
        disk is the thing being investigated.
        """
        spec = self._spec()
        cfg_paths = self.ctx.cfg.section("paths")
        min_free = float(self.ctx.cfg.get("recording.min_free_gb", 30))
        self.lbl_rec.set("Running pre-flight checks…", "info")
        if not self.preflight_job.start(
                lambda progress=None: recorder.preflight(spec, cfg_paths,
                                                         min_free)):
            self.lbl_rec.set("Pre-flight is already running.", "warn")

    def _preflight_done(self, checks: list) -> None:
        rows = [("ok" if c.ok else ("bad" if c.fatal else "warn"), c.name, c.detail)
                for c in checks]
        self.checks.set_rows(rows, self.pal)
        fatal = [c for c in checks if not c.ok and c.fatal]
        if fatal:
            self.lbl_rec.set(f"{len(fatal)} blocking problem(s) - fix before recording.",
                             "bad")
        elif any(not c.ok for c in checks):
            self.lbl_rec.set("Warnings only; recording is possible.", "warn")
        else:
            self.lbl_rec.set("All checks passed.", "ok")
        if self._start_after_preflight:
            self._start_after_preflight = False
            self._start_checked(checks)

    def _start_recording(self) -> None:
        """Start a trial, once the checks have run.

        The checks are asynchronous now, so this arms a flag and returns; the
        recording itself begins in _start_checked when the results arrive.
        """
        self._start_after_preflight = True
        self._preflight()

    def _start_checked(self, checks: list) -> None:
        if [c for c in checks if not c.ok and c.fatal]:
            W.warn_box(self, "Cannot record",
                       "Fix the blocking items in the checklist first.")
            return
        warn = [c for c in checks if not c.ok and not c.fatal]
        if warn and not W.ask_yes_no(
                self, "Proceed with warnings?",
                "These are not blocking, but worth a look first:\n\n"
                + "\n".join(f"• {c.name}: {c.detail}" for c in warn)
                + "\n\nStart the recording anyway?"):
            return
        if self.preview is not None and self.preview.running:
            self.statusMessage.emit("Stopping preview so FFmpeg can claim the camera")
            self._apply_settings_then_release()
            self._stop_preview()
        elif self.ctx.cfg.get("camera.apply_uvc_before_record", True):
            self._apply_settings_then_release()

        spec = self._spec()
        self.rec = recorder.Recorder(spec, self.ctx.cfg.section("paths"))
        try:
            self.rec.start()
        except Exception as e:
            W.warn_box(self, "Could not start recording", str(e))
            self.rec = None
            return
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_t0.setEnabled(True)
        self.btn_preview.setEnabled(False)
        self.progress.setRange(0, int(spec.duration_s))
        self.rec_timer.start()
        self.lbl_rec.set(f"Recording to {spec.target_path()}", "ok")
        self.statusMessage.emit("Recording started")

    def _apply_settings_then_release(self) -> None:
        """Push exposure/gain into the driver, then let go of the camera.

        Most Windows UVC drivers keep manual exposure and gain after the handle
        closes, which is how the preview's settings reach the FFmpeg recording.
        It is best-effort: if the read-back disagrees, say so rather than
        pretending the setting took.
        """
        data = self.cb_index.currentData()
        if data is None:
            return
        st = camera.apply_uvc_oneshot(
            int(data), self._uvc(),
            self.ctx.cfg.get("camera.opencv_backend", "dshow"))
        if st.warnings:
            self.statusMessage.emit(
                "Camera settings may not have stuck: " + "; ".join(st.warnings)
                + " - use the driver's property page instead.")

    def _mark_t0(self) -> None:
        if self.rec is None or not self.rec.running:
            return
        off = self.rec.mark_t0()
        self.btn_t0.setText(f"T0 marked at {W.hms(off)}")
        self.btn_t0.setEnabled(False)
        self.lbl_rec.set(
            f"T0 recorded at {W.hms(off)} into the recording. Analysis will treat "
            f"everything before it as baseline.", "ok")

    def _tick_recording(self) -> None:
        if self.rec is None:
            return
        st = self.rec.stats()
        self.r_elapsed.set_value(W.hms(st.time_s or st.wall_elapsed_s))
        self.r_fps.set_value(f"{st.fps_instant:.0f}")
        self.r_frames.set_value(f"{st.frames:,}")
        # Shows total loss, not just the muxer's drop= counter: a capture can
        # throw away frames in the DirectShow buffer with drop= still at zero,
        # and a tile reading 0 through that is worse than no tile at all.
        self.r_drop.set_value(f"{st.total_lost:,}")
        self.r_size.set_value(W.human_bytes(st.size_bytes))
        self.progress.setValue(int(min(st.time_s, self.progress.maximum())))

        expected = self.rec.spec.expected_fps()
        if st.errors:
            self.lbl_rec.set("FFmpeg reported: " + st.errors[-1], "bad")
        else:
            level, verdict = st.health(expected)
            if level != "ok":
                self.lbl_rec.set(verdict, level)

        if not self.rec.running:
            self.rec_timer.stop()
            self._recording_finished()

    def _stop_recording(self) -> None:
        if self.rec is None:
            return
        if not W.ask_yes_no(self, "Stop recording?",
                            "The file is kept and closed cleanly, but the trial will "
                            "be shorter than planned. Stop now?"):
            return
        self.rec.stop()

    def _recording_finished(self) -> None:
        if self.rec is None:
            return
        spec = self.rec.spec
        st = self.rec.stats()
        rc = self.rec.exit_code
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_t0.setEnabled(False)
        self.btn_t0.setText("Mark T0 (dose)")
        self.btn_preview.setEnabled(True)

        target = spec.target_path()
        where = target.parent if spec.is_segmented else target
        if rc not in (0, 255) and st.frames == 0:
            self.lbl_rec.set(
                recorder.explain_exit(rc, st.errors)
                + "  No frames were written."
                + (f"\n\nFFmpeg said: {st.errors[-1]}" if st.errors else ""),
                "bad")
        else:
            msg = (f"Finished: {st.frames:,} frames, {W.hms(st.time_s)}, "
                   f"{W.human_bytes(st.size_bytes)} → {where}")
            lost = st.total_lost
            if lost:
                got = st.effective_fps
                msg += (f"\nWARNING: {lost:,} frame(s) were lost during this "
                        f"capture ({st.rt_buffer_drops} buffer overflow(s), "
                        f"{st.dropped} muxer drop(s)), and the file averages "
                        f"{got:.0f} fps against an expected {spec.expected_fps():.0f}. "
                        "Each loss is a step in the angle trace that the "
                        "detector can read as a twitch. Record to a local SSD, "
                        "close other programs, and raise the buffer sizes in "
                        "Settings before repeating the trial.")
            self.lbl_rec.set(msg, "warn" if lost else "ok")
            self.statusMessage.emit(msg.splitlines()[0])
            self.recordingFinished.emit(str(where))

            if self.ctx.cfg.get("recording.auto_verify_after_record", True) \
                    and not spec.is_segmented and target.exists():
                self._verify(target, spec.expected_fps())
        self.rec = None

    def _verify(self, target: Path, expected_fps: float) -> None:
        self.statusMessage.emit("Verifying the recording's true frame rate…")
        ffprobe = self.ctx.cfg.get("paths.ffprobe", "")

        def job(progress=None):
            return recorder.verify_recording(target, expected_fps, ffprobe)

        self.verify_job.start(job)

    # ------------------------------------------------------------------
    def _set_busy(self, busy: bool) -> None:
        for b in (self.btn_probe_one, self.btn_probe_all, self.btn_preview,
                  self.btn_start, self.btn_refresh):
            b.setEnabled(not busy)

    def _job_failed(self, msg: str, tb: str) -> None:
        self._set_busy(False)
        log.error("acquisition job failed: %s\n%s", msg, tb)
        W.warn_box(self, "Operation failed", msg)

    def shutdown(self) -> None:
        W.stop_jobs(self)
        self.preview_timer.stop()
        self.rec_timer.stop()
        if self.preview is not None:
            self.preview.stop()
        if self.rec is not None and self.rec.running:
            self.rec.stop()
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
