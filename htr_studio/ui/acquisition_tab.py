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
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

from .. import camera, config, ffmpeg_tools, paths, recorder, theme
from . import widgets as W

log = logging.getLogger(__name__)


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

        self._build()
        self._load_from_config()

        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setInterval(40)  # 25 Hz is plenty for a human eye
        self._hist_broken = False
        self.probe_job = W.JobRunner("mode-probe", self)
        self.probe_job.finished.connect(self._probe_done)
        self.probe_job.failed.connect(self._job_failed)
        self.verify_job = W.JobRunner("verify", self)
        self.verify_job.finished.connect(
            lambda v: self.lbl_rec.set(v.message, "ok" if v.ok else "warn"))
        self.verify_job.failed.connect(self._job_failed)
        self.preview_timer.timeout.connect(self._tick_preview)

        self.rec_timer = QtCore.QTimer(self)
        self.rec_timer.setInterval(500)
        self.rec_timer.timeout.connect(self._tick_recording)

        QtCore.QTimer.singleShot(300, self.refresh_devices)

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
                    "positives that cannot be fixed in analysis.")
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
        self.btn_preview = W.button("Start preview", "primary")
        self.btn_snapshot = W.button("Save snapshot")
        self.btn_snapshot.setEnabled(False)
        row.addWidget(self.btn_preview)
        row.addWidget(self.btn_snapshot)
        row.addStretch(1)
        self.lbl_negotiated = QtWidgets.QLabel("")
        self.lbl_negotiated.setObjectName("hint")
        row.addWidget(self.lbl_negotiated)
        v.addLayout(row)

        self.view = W.FrameView()
        self.view.set_palette_tokens(self.pal)
        v.addWidget(self.view, 1)

        tiles = QtWidgets.QHBoxLayout()
        self.t_fps = W.MetricTile("preview fps")
        self.t_contrast = W.MetricTile("contrast", "-", "/255")
        self.t_clip = W.MetricTile("clipped", "-", "%")
        self.t_focus = W.MetricTile("focus")
        self.t_uniform = W.MetricTile("uniformity")
        for t in (self.t_fps, self.t_contrast, self.t_clip, self.t_focus, self.t_uniform):
            tiles.addWidget(t)
        v.addLayout(tiles)

        self.verdicts = QtWidgets.QVBoxLayout()
        self.verdict_labels = [W.StatusLabel("", "info") for _ in range(4)]
        for lbl in self.verdict_labels:
            self.verdicts.addWidget(lbl)
        v.addLayout(self.verdicts)

        self.btn_preview.clicked.connect(self._toggle_preview)
        self.btn_snapshot.clicked.connect(self._snapshot)
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
        self.lbl_device.set("Enumerating cameras…", "info")

        if not paths.is_windows():
            self.lbl_device.set(
                "Camera enumeration uses Windows DirectShow, which is not available "
                "here. Analysis, Library and Review all work; acquisition does not.",
                "warn")
            self.btn_preview.setEnabled(False)
            self.btn_start.setEnabled(False)
            return

        try:
            self.devices = ffmpeg_tools.list_dshow_devices(
                self.ctx.cfg.get("paths.ffmpeg", ""))
        except ffmpeg_tools.FFmpegMissing as e:
            self.lbl_device.set(str(e), "bad")
            return

        self.cb_device.clear()
        for d in self.devices:
            self.cb_device.addItem(d.name)
        wanted = self.ctx.cfg.get("camera.device_name", "")
        match = ffmpeg_tools.match_device(self.devices, wanted)
        if match:
            self.cb_device.setCurrentText(match.name)

        self.indices = camera.enumerate_indices(
            6, self.ctx.cfg.get("camera.opencv_backend", "dshow"))
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

    def _start_preview(self) -> None:
        if self.rec is not None and self.rec.running:
            W.warn_box(self, "Recording in progress",
                       "The camera is busy recording. Stop the recording first - a "
                       "DirectShow device only admits one client at a time.")
            return
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
        self.btn_preview.setText("Stop preview")
        self.btn_snapshot.setEnabled(True)
        self.preview_timer.start()
        self.statusMessage.emit("Preview running")

    def _stop_preview(self) -> None:
        self.preview_timer.stop()
        if self.preview is not None:
            self.preview.stop()
            self.preview = None
        self.btn_preview.setText("Start preview")
        self.btn_snapshot.setEnabled(False)
        self.statusMessage.emit("Preview stopped")

    def _push_controls(self, *_args) -> None:
        if self.preview is not None and self.preview.running:
            applied = self.preview.push(self._uvc())
            if applied.warnings:
                self.lbl_negotiated.setText(" | ".join(applied.warnings)[:120])


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
        if self.preview is None or not self.preview.running:
            return
        gray = self.preview.read()
        if gray is None:
            return
        m = camera.frame_metrics(gray)
        self._last_metrics = m
        self.view.show_frame(gray, {"crosshair": True},
                             caption=f"{gray.shape[1]}x{gray.shape[0]}")
        self.t_fps.set_value(f"{self.preview.measured_fps():.0f}")
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

    def _snapshot(self) -> None:
        paths.ensure_dirs()
        default = paths.EXPORTS_DIR / "camera_snapshot.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save snapshot", str(default), "PNG (*.png)")
        if path and self.view.save_png(Path(path)):
            self.statusMessage.emit(f"Snapshot saved to {path}")

    def _flicker(self) -> None:
        if self.preview is None or not self.preview.running:
            self.lbl_flicker.set("Start the preview first.", "warn")
            return
        self.preview_timer.stop()
        try:
            res = self.preview.run_flicker_test(2.0)
        finally:
            self.preview_timer.start()
        self.lbl_flicker.set(
            f"{res.verdict}  [{res.frames} frames at {res.fps:.0f} fps]", res.level)

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

    def _preflight(self) -> list:
        spec = self._spec()
        checks = recorder.preflight(spec, self.ctx.cfg.section("paths"),
                                    float(self.ctx.cfg.get("recording.min_free_gb", 30)))
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
        return checks

    def _start_recording(self) -> None:
        checks = self._preflight()
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
                f"FFmpeg exited with code {rc} and wrote no frames. "
                + (st.errors[-1] if st.errors else "See the log for detail."), "bad")
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
        self.preview_timer.stop()
        self.rec_timer.stop()
        if self.preview is not None:
            self.preview.stop()
        if self.rec is not None and self.rec.running:
            self.rec.stop()
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
