"""Two dialogs that answer the two questions a failed analysis actually raises.

A detector that reports "0 twitches" on footage where the twitches are plainly
visible tells the operator nothing about which of the two possible failures
happened, and the two have opposite fixes:

**"Is the band in the wrong place?"** -> :class:`SpectrumDialog`. The oscillation
may be outside the pass band, either because the recording aliased (captured
below ~200 fps, so a 90 Hz twitch folds down to something else and cannot be
recovered) or because the file is a slow-motion render whose declared frame rate
is the *playback* rate. The second case is the common one with published
supplementary videos and is entirely recoverable: the frames are all there, only
the timebase is wrong. The spectrum makes both obvious at a glance, and the
dialog offers the implied capture rate as a one-click override.

**"Is the animal being found at all?"** -> :class:`TuneDialog`. Segmentation can
fail quietly: a mask that includes the bedding, a tail that was not opened away
so the head end flips, an animal against the cylinder wall whose outline is
clipped. The tuning view runs the *same* ``segment_frame`` the extraction pass
uses, so what the operator sees is what the analysis will do, and the parameter
that fixes it can be found in seconds instead of by re-running a 30-minute file.

Neither dialog changes anything on its own. Both hand their result back to the
Analysis tab, which is where the decision is recorded.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from .. import analysis, theme
from . import widgets as W

log = logging.getLogger(__name__)

SPECTRUM_SOURCES = [
    ("angle_velocity", "Head-angle velocity (what the detector uses)"),
    ("angle", "Head angle"),
    ("motion_head", "Frame-to-frame motion, head region"),
    ("motion_all", "Frame-to-frame motion, whole animal"),
    ("motion_bg", "Frame-to-frame motion, background only (flicker check)"),
]


# ==========================================================================
# Spectrum
# ==========================================================================
class SpectrumDialog(QtWidgets.QDialog):
    """Where is the energy in this recording, and is the band reachable?

    ``rateChosen`` carries a frame rate the operator accepted as the true
    capture rate. The Analysis tab writes it into the capture-rate override and
    re-runs; nothing is changed here.
    """

    rateChosen = QtCore.pyqtSignal(float)

    def __init__(self, sig: analysis.Signals, pal: theme.Palette, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Signal diagnostics - where is the energy?")
        self.resize(900, 620)
        self.sig = sig
        self.pal = pal
        self._spec: analysis.SpectrumResult | None = None
        self._sugg: analysis.RateSuggestion | None = None
        self._build()
        self._recompute()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        v = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Signal"))
        self.cb_source = QtWidgets.QComboBox()
        for key, label in SPECTRUM_SOURCES:
            self.cb_source.addItem(label, key)
        self.cb_source.setToolTip(
            "Head-angle velocity is the signal the detector thresholds, so it "
            "is the one that matters. The background-only trace is the control: "
            "in-band power there is lighting flicker or sensor noise, not the "
            "animal, and it will create false positives everywhere.")
        self.cb_source.currentIndexChanged.connect(self._recompute)
        top.addWidget(self.cb_source, 1)
        self.chk_log = QtWidgets.QCheckBox("Log power")
        self.chk_log.setChecked(True)
        self.chk_log.toggled.connect(self._redraw)
        top.addWidget(self.chk_log)
        v.addLayout(top)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.setLabel("left", "Power spectral density")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.curve = self.plot.plot(pen=pg.mkPen(self.pal.s1, width=2))
        # The published HTR band, drawn so "in or out" needs no arithmetic.
        self.band = pg.LinearRegionItem(
            values=analysis.HTR_BAND_HZ, movable=False,
            brush=pg.mkBrush(self.pal.s3 + "33"),
            pen=pg.mkPen(self.pal.s3, width=1))
        self.band.setZValue(-10)
        self.plot.addItem(self.band)
        self.lbl_band = pg.TextItem("78-98 Hz HTR band", color=self.pal.s3,
                                    anchor=(0, 1))
        self.plot.addItem(self.lbl_band)
        self.line_nyq = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(self.pal.critical, width=2, style=QtCore.Qt.DashLine),
            label="Nyquist {value:.0f} Hz",
            labelOpts={"color": self.pal.critical, "position": 0.9})
        self.plot.addItem(self.line_nyq)
        self.marker = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(self.pal.s2),
                                         pen=pg.mkPen(self.pal.surface, width=1))
        self.plot.addItem(self.marker)
        v.addWidget(self.plot, 1)

        tiles = QtWidgets.QHBoxLayout()
        self.t_fps = W.MetricTile("file frame rate", "-", "fps")
        self.t_nyq = W.MetricTile("usable to", "-", "Hz")
        self.t_peak = W.MetricTile("strongest peak", "-", "Hz")
        self.t_band = W.MetricTile("power in band", "-", "%")
        for t in (self.t_fps, self.t_nyq, self.t_peak, self.t_band):
            tiles.addWidget(t)
        v.addLayout(tiles)

        self.txt = QtWidgets.QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setMaximumHeight(170)
        v.addWidget(self.txt)

        row = QtWidgets.QHBoxLayout()
        self.btn_apply = W.button("Use the implied capture rate", "primary")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._apply)
        row.addWidget(self.btn_apply)
        self.btn_copy = W.button("Copy this report")
        self.btn_copy.clicked.connect(self._copy)
        row.addWidget(self.btn_copy)
        row.addStretch(1)
        close = W.button("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        v.addLayout(row)

    # ------------------------------------------------------------------
    def _recompute(self) -> None:
        src = self.cb_source.currentData()
        try:
            self._spec = analysis.spectrum(self.sig, source=src)
        except Exception as e:                      # scipy/short-signal guard
            log.exception("spectrum failed")
            self.txt.setPlainText(f"Could not compute a spectrum: {e}")
            return
        # The rate suggestion is only meaningful for the animal's own movement.
        self._sugg = (analysis.suggest_capture_rate(self._spec)
                      if src in ("angle_velocity", "angle", "motion_head")
                      else None)
        self._redraw()

    def _redraw(self) -> None:
        sp = self._spec
        if sp is None:
            return
        keep = sp.freqs > 0
        f, P = sp.freqs[keep], sp.power[keep]
        if self.chk_log.isChecked():
            y = 10.0 * np.log10(np.maximum(P, np.finfo(float).tiny))
            self.plot.setLabel("left", "Power spectral density (dB)")
        else:
            y = P
            self.plot.setLabel("left", "Power spectral density")
        self.curve.setData(f, y)
        nyq = sp.fps / 2.0
        self.line_nyq.setValue(nyq)
        self.plot.setXRange(0, min(max(nyq * 1.05, 20.0), 500.0), padding=0)
        if y.size:
            self.lbl_band.setPos(analysis.HTR_BAND_HZ[0], float(np.max(y)))

        pts = []
        for fi, Pi in sp.peaks[:4]:
            yi = (10.0 * np.log10(max(Pi, np.finfo(float).tiny))
                  if self.chk_log.isChecked() else Pi)
            pts.append({"pos": (fi, yi)})
        self.marker.setData(pts)

        self.t_fps.set_value(f"{sp.fps:.2f}")
        self.t_nyq.set_value(f"{sp.fps * 0.45:.0f}")
        self.t_peak.set_value(f"{sp.peak_hz:.1f}" if sp.peaks else "-")
        self.t_band.set_value(f"{sp.band_fraction * 100:.1f}")
        self.txt.setPlainText(self._report())

        ok = bool(self._sugg) and self._sugg.confidence in ("likely", "possible")
        self.btn_apply.setEnabled(ok)
        if ok:
            self.btn_apply.setText(
                f"Use {self._sugg.implied_fps:.0f} fps and re-run")
        else:
            self.btn_apply.setText("Use the implied capture rate")

    def _report(self) -> str:
        sp = self._spec
        if sp is None:
            return ""
        lines = [f"signal: {sp.source}    file frame rate: {sp.fps:.3f} fps",
                 f"power inside the 78-98 Hz HTR band: {sp.band_fraction:.2%}", ""]
        if sp.peaks:
            lines.append("strongest components")
            top = sp.peaks[0][1] or 1.0
            for fi, Pi in sp.peaks[:6]:
                mark = "  <- in the HTR band" if (
                    analysis.HTR_BAND_HZ[0] <= fi <= analysis.HTR_BAND_HZ[1]) else ""
                lines.append(f"  {fi:7.1f} Hz   relative power "
                             f"{Pi / top:6.3f}{mark}")
            lines.append("")
        for n in sp.notes:
            lines.append(f"NOTE  {n}")
        if sp.notes:
            lines.append("")
        if self._sugg is not None:
            lines.append(f"CAPTURE-RATE CHECK  ({self._sugg.confidence})")
            lines.append(self._sugg.message)
        elif sp.band_fraction >= 0.05:
            lines.append("The energy is where an HTR should be, so the frame "
                         "rate and pass band are consistent with each other. If "
                         "the counts are still wrong, the problem is in the "
                         "detector thresholds or in the tracking, not the "
                         "timebase.")
        return "\n".join(lines)

    def _apply(self) -> None:
        if self._sugg is None:
            return
        self.rateChosen.emit(float(self._sugg.implied_fps))
        self.accept()

    def _copy(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self.txt.toPlainText())


# ==========================================================================
# Tracking
# ==========================================================================
class TuneDialog(QtWidgets.QDialog):
    """Interactive segmentation on real frames, using the analysis code path.

    ``configChanged`` carries the segmentation fields the operator settled on.
    The Analysis tab copies them into its own widgets, so the tuned values are
    saved and exported with the run like any other parameter.
    """

    configChanged = QtCore.pyqtSignal(object)      # dict of field -> value

    def __init__(self, video: Path | str, cfg: analysis.AnalysisConfig,
                 pal: theme.Palette, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tune tracking - is the animal being found?")
        self.resize(1080, 700)
        self.video = Path(video)
        self.cfg = cfg
        self.pal = pal
        self.cal: analysis.Calibration | None = None
        self.reader = W.FrameReader(self.video, cache=32)
        self.n_frames = 0
        self._build()
        QtCore.QTimer.singleShot(0, self._open)

    # ------------------------------------------------------------------
    def _build(self) -> None:
        outer = QtWidgets.QHBoxLayout(self)

        # ---- controls
        panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(panel)
        panel.setMaximumWidth(430)

        self.cb_seg = QtWidgets.QComboBox()
        self.cb_seg.addItems(["auto", "threshold", "background"])
        W.form_row(form, "Segmentation", self.cb_seg,
                   "'threshold' separates the animal by brightness alone and is "
                   "right on a clean, evenly lit arena. 'background' builds an "
                   "empty-arena image from the median of sampled frames and "
                   "subtracts it, which cancels bedding, the cylinder rim and a "
                   "lighting gradient. 'auto' keeps threshold while it is "
                   "working and falls back to background when it is not.")

        self.cb_polarity = QtWidgets.QComboBox()
        self.cb_polarity.addItems(["auto", "dark_on_light", "light_on_dark"])
        W.form_row(form, "Mouse vs background", self.cb_polarity,
                   "Threshold mode only. A dark mouse on a pale floor is "
                   "'dark_on_light'.")

        self.cb_thresh_mode = QtWidgets.QComboBox()
        self.cb_thresh_mode.addItems(["otsu", "fixed"])
        W.form_row(form, "Threshold rule", self.cb_thresh_mode,
                   "Otsu is computed once from sampled frames and frozen, so "
                   "segmentation cannot drift with the animal's position.")
        self.sp_thresh = QtWidgets.QSpinBox()
        self.sp_thresh.setRange(1, 254)
        form.addRow("  fixed grey level", self.sp_thresh)

        self.sp_bg_samples = QtWidgets.QSpinBox()
        self.sp_bg_samples.setRange(8, 400)
        W.form_row(form, "Background frames", self.sp_bg_samples,
                   "How many frames are pooled into the empty-arena median. "
                   "More is steadier and slower; 40 is enough unless the animal "
                   "sits still for most of the recording.")
        self.sp_bg_thr = QtWidgets.QSpinBox()
        self.sp_bg_thr.setRange(0, 200)
        self.sp_bg_thr.setSpecialValueText("auto (Otsu)")
        W.form_row(form, "Difference level", self.sp_bg_thr,
                   "Background mode. How far a pixel must depart from the empty "
                   "arena to count as animal. Leave on auto unless the mask is "
                   "visibly wrong.")

        self.cb_roi = QtWidgets.QComboBox()
        self.cb_roi.addItems(["none", "auto", "manual"])
        W.form_row(form, "Region of interest", self.cb_roi,
                   "'auto' infers the arena from the background model. It is off "
                   "by default because a tight ROI clips the animal when it "
                   "presses against the wall, which distorts the angle far more "
                   "than the stray contrast it removes.")
        roi = QtWidgets.QHBoxLayout()
        self.sp_roi = []
        for name in ("x", "y", "w", "h"):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(0.0, 1.0)
            s.setDecimals(3)
            s.setSingleStep(0.02)
            s.setPrefix(f"{name} ")
            self.sp_roi.append(s)
            roi.addWidget(s)
        form.addRow("  manual (frame fractions)", W.form_wrap(roi))

        self.sp_min_area = QtWidgets.QDoubleSpinBox()
        self.sp_min_area.setRange(0.0001, 0.5)
        self.sp_min_area.setDecimals(4)
        self.sp_min_area.setSingleStep(0.001)
        self.sp_max_area = QtWidgets.QDoubleSpinBox()
        self.sp_max_area.setRange(0.01, 1.0)
        self.sp_max_area.setDecimals(3)
        self.sp_max_area.setSingleStep(0.01)
        W.form_row(form, "Blob area min (frame fraction)", self.sp_min_area)
        W.form_row(form, "Blob area max", self.sp_max_area)

        self.sp_morph = QtWidgets.QSpinBox()
        self.sp_morph.setRange(1, 31)
        W.form_row(form, "Clean-up kernel (px)", self.sp_morph,
                   "Removes speckle. Too large and the animal's outline is eaten.")
        self.sp_core = QtWidgets.QSpinBox()
        self.sp_core.setRange(0, 41)
        self.sp_core.setSpecialValueText("auto (measured)")
        W.form_row(form, "Tail-removal kernel (px)", self.sp_core,
                   "Must be wider than the tail and narrower than the body. The "
                   "head arrow points away from whatever the opening removed, "
                   "so if the arrow sits on the tail end, this is the parameter "
                   "to change - too small and the tail stays inside the body. "
                   "'auto' measures it from the animal's own width; the "
                   "calibration line reports what it chose.")

        self.chk_edge = QtWidgets.QCheckBox(
            "Mark frames where the animal touches the edge as untracked")
        self.chk_edge.setToolTip(
            "Off by default. A clipped outline gives a wrong angle, but on a "
            "tight arena this rejects so many frames that real twitches are "
            "lost with them.")
        form.addRow(self.chk_edge)

        self.cb_view = QtWidgets.QComboBox()
        self.cb_view.addItems(["overlay on the frame", "binary mask",
                               "background model", "difference from background"])
        W.form_row(form, "Show", self.cb_view,
                   "Check the mask directly when the overlay looks plausible but "
                   "the angle does not.")

        row = QtWidgets.QHBoxLayout()
        self.btn_cal = W.button("Recalibrate", "primary",
                                "Re-samples the file with these settings. Needed "
                                "after changing segmentation, polarity, the "
                                "threshold rule or the ROI.")
        self.btn_cal.clicked.connect(self._calibrate)
        row.addWidget(self.btn_cal)
        self.btn_scan = W.button("Scan 200 frames",
                                 "Segments 200 frames spread across the file and "
                                 "reports how often the animal is found and how "
                                 "steady its area is. This is the number to "
                                 "optimise: below about 90% found, or an area "
                                 "coefficient of variation above 0.35, the angle "
                                 "trace is not trustworthy.")
        self.btn_scan.clicked.connect(self._scan)
        row.addWidget(self.btn_scan)
        form.addRow(W.form_wrap(row))

        self.lbl_cal = W.StatusLabel("Opening the file…", "info")
        self.lbl_cal.setWordWrap(True)
        form.addRow(self.lbl_cal)
        self.lbl_scan = W.StatusLabel("", "info")
        self.lbl_scan.setWordWrap(True)
        form.addRow(self.lbl_scan)

        for w in (self.sp_thresh, self.sp_bg_thr, self.sp_min_area,
                  self.sp_max_area, self.sp_morph, self.sp_core):
            w.valueChanged.connect(self._refresh)
        for w in (self.cb_view,):
            w.currentIndexChanged.connect(self._refresh)
        self.chk_edge.toggled.connect(self._refresh)
        self.cb_thresh_mode.currentIndexChanged.connect(self._refresh)

        # ---- viewer
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        self.view = W.FrameView()
        self.view.set_palette_tokens(self.pal)
        rv.addWidget(self.view, 1)

        nav = QtWidgets.QHBoxLayout()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._refresh)
        nav.addWidget(self.slider, 1)
        self.lbl_pos = QtWidgets.QLabel("-")
        self.lbl_pos.setMinimumWidth(130)
        nav.addWidget(self.lbl_pos)
        rv.addLayout(nav)
        self.lbl_frame = W.StatusLabel("", "info")
        self.lbl_frame.setWordWrap(True)
        rv.addWidget(self.lbl_frame)

        buttons = QtWidgets.QHBoxLayout()
        self.btn_use = W.button("Use these settings", "primary")
        self.btn_use.clicked.connect(self._accept)
        buttons.addWidget(self.btn_use)
        cancel = W.button("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        buttons.addStretch(1)
        rv.addLayout(buttons)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(W.scrollable(panel))
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        outer.addWidget(split)
        self._load(self.cfg)

    # ------------------------------------------------------------------
    def _load(self, a: analysis.AnalysisConfig) -> None:
        self.cb_seg.setCurrentText(a.segmentation)
        self.cb_polarity.setCurrentText(a.polarity)
        self.cb_thresh_mode.setCurrentText(a.threshold_mode)
        self.sp_thresh.setValue(int(a.threshold_value))
        self.sp_bg_samples.setValue(int(a.bg_samples))
        self.sp_bg_thr.setValue(int(a.bg_threshold_value))
        self.cb_roi.setCurrentText(a.roi_mode)
        for s, val in zip(self.sp_roi, a.roi_frac):
            s.setValue(float(val))
        self.sp_min_area.setValue(a.min_area_frac)
        self.sp_max_area.setValue(a.max_area_frac)
        self.sp_morph.setValue(a.morph_kernel)
        self.sp_core.setValue(a.core_kernel)
        self.chk_edge.setChecked(bool(a.reject_edge_blobs))

    def current_fields(self) -> dict:
        return {
            "segmentation": self.cb_seg.currentText(),
            "polarity": self.cb_polarity.currentText(),
            "threshold_mode": self.cb_thresh_mode.currentText(),
            "threshold_value": int(self.sp_thresh.value()),
            "bg_samples": int(self.sp_bg_samples.value()),
            "bg_threshold_value": int(self.sp_bg_thr.value()),
            "roi_mode": self.cb_roi.currentText(),
            "roi_frac": tuple(float(s.value()) for s in self.sp_roi),
            "min_area_frac": float(self.sp_min_area.value()),
            "max_area_frac": float(self.sp_max_area.value()),
            "morph_kernel": int(self.sp_morph.value()),
            "core_kernel": int(self.sp_core.value()),
            "reject_edge_blobs": bool(self.chk_edge.isChecked()),
        }

    def _work_cfg(self) -> analysis.AnalysisConfig:
        import dataclasses
        return dataclasses.replace(self.cfg, **self.current_fields())

    # ------------------------------------------------------------------
    def _open(self) -> None:
        if not self.reader.open():
            self.lbl_cal.set(f"Could not open {self.video.name}.", "bad")
            return
        self.n_frames = int(self.reader.n_frames or 0)
        self.slider.setRange(0, max(0, self.n_frames - 1))
        self.slider.setValue(min(self.n_frames // 3, max(0, self.n_frames - 1)))
        self._calibrate()

    def _calibrate(self) -> None:
        cfg = self._work_cfg()
        self.lbl_cal.set("Sampling the file…", "info")
        QtWidgets.QApplication.processEvents()
        try:
            self.cal = analysis.calibrate(self.video, cfg)
        except Exception as e:
            log.exception("calibrate failed")
            self.lbl_cal.set(f"Calibration failed: {e}", "bad")
            return
        cal = self.cal
        level = "ok" if (cal.hit_rate >= 0.90 and cal.area_cv <= 0.35) else (
            "warn" if cal.hit_rate >= 0.70 else "bad")
        self.lbl_cal.set(
            f"{cal.describe}; blob area varies by {cal.area_cv:.2f} "
            f"(coefficient of variation)."
            + ("" if level == "ok" else
               "  Below 90% found or above 0.35 area variation, try the other "
               "segmentation mode before touching the detector."),
            level)
        if cal.notes:
            log.info("tuning calibration: %s", "; ".join(cal.notes))
        self._refresh()

    def _scan(self) -> None:
        if self.cal is None:
            return
        cfg = self._work_cfg()
        self.lbl_scan.set("Scanning…", "info")
        QtWidgets.QApplication.processEvents()
        cap = cv2.VideoCapture(str(self.video))
        if not cap.isOpened():
            self.lbl_scan.set("Could not reopen the file for a scan.", "bad")
            return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or self.n_frames
        want = 200
        idx = (np.linspace(0, max(total - 1, 0), want).astype(int)
               if total > want else np.arange(max(total, 1)))
        found, areas, clipped = 0, [], 0
        try:
            for i in idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, frame = cap.read()
                if not ok:
                    continue
                gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        if frame.ndim == 3 else frame)
                seg = analysis.segment_frame(gray, cfg, self.cal)
                if seg.clipped:
                    clipped += 1
                if seg.head is not None:
                    found += 1
                    areas.append(seg.area_frac)
        finally:
            cap.release()
        n = max(1, len(idx))
        hit = found / n
        cv_ = (float(np.std(areas) / np.mean(areas))
               if areas and np.mean(areas) > 0 else float("nan"))
        level = "ok" if (hit >= 0.90 and cv_ <= 0.35) else (
            "warn" if hit >= 0.70 else "bad")
        msg = (f"{found} of {n} sampled frames tracked ({hit:.0%}); "
               f"area variation {cv_:.2f}")
        if clipped:
            msg += f"; {clipped} rejected as edge-clipped"
        if level != "ok":
            msg += (".  The angle trace is not trustworthy at this hit rate - "
                    "fix tracking before interpreting any twitch count.")
        self.lbl_scan.set(msg, level)

    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        if self.cal is None:
            return
        i = int(self.slider.value())
        gray = self.reader.get(i)
        if gray is None:
            self.lbl_frame.set(f"Frame {i} could not be decoded.", "warn")
            return
        cfg = self._work_cfg()
        try:
            seg = analysis.segment_frame(gray, cfg, self.cal)
        except Exception as e:
            log.exception("segment_frame failed")
            self.lbl_frame.set(f"Segmentation failed on this frame: {e}", "bad")
            return

        mode = self.cb_view.currentIndex()
        if mode == 1:
            base = seg.mask
        elif mode == 2 and self.cal.background is not None:
            base = self.cal.background
        elif mode == 3 and self.cal.background is not None:
            base = cv2.absdiff(gray, self.cal.background)
            base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
            base = base.astype(np.uint8)
        else:
            base = gray

        ov: dict = {}
        if mode == 0:
            cnts, _ = cv2.findContours(seg.mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            ov["mask_contour"] = cnts
            if seg.head is not None:
                ov["centroid"] = (seg.head.cx, seg.head.cy)
                ov["head"] = (seg.head.hx, seg.head.hy)
            ov["burst"] = bool(seg.note)

        cap = f"frame {i}"
        if seg.head is not None:
            cap += f"   heading {seg.angle_deg:7.2f} deg   area {seg.area_frac:.3%}"
        self.view.show_frame(base, ov, cap)
        self.lbl_pos.setText(f"{i:,} / {max(self.n_frames - 1, 0):,}")

        if seg.head is not None:
            self.lbl_frame.set(
                f"Tracked. Heading {seg.angle_deg:.2f} deg, blob "
                f"{seg.area_frac:.3%} of the frame.", "ok")
        else:
            self.lbl_frame.set(seg.note or "Not tracked on this frame.",
                               "warn" if seg.clipped else "bad")

    def _accept(self) -> None:
        self.configChanged.emit(self.current_fields())
        self.accept()

    def closeEvent(self, ev) -> None:        # noqa: N802 - Qt naming
        self.reader.close()
        super().closeEvent(ev)
