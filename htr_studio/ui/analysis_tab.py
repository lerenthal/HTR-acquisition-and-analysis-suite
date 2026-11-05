"""Analysis tab: detector parameters, the run, diagnostics and export.

Every parameter here has a reason, and the tooltips carry it, because a detector
whose settings are folklore produces counts nobody can defend. The ones that
matter most:

* **Pass band 60-110 Hz.** Halberstadt et al. (2014) measured 5-11 oscillations
  per HTR at 78-98 Hz. A wider band (the first-generation pipeline used 20-100 Hz)
  passes locomotion and grooming harmonics.
* **Threshold from the baseline window.** Deriving it from the whole recording
  lets the drug effect inflate its own threshold, so the effect partly hides
  itself. The protocol records 10 minutes of pre-dose baseline; that is the
  noise floor to measure against.
* **Hysteresis and merge gap.** An HTR burst grows and fades. A single threshold
  clips it wherever the envelope dips through, counting one twitch two or three
  times.
* **Frequency and half-cycle gates.** A twitch is an oscillation, not a lurch. A
  candidate that does not actually oscillate in band gets rejected, and the
  reason is recorded so the call can be audited.

"Validate against ground truth" is here too: point it at a CSV with
``time_sec`` and ``twitch_active`` columns and it reports precision, recall and
F1 for the current settings.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets

from .. import advisor, analysis, library, paths, report, theme
from . import widgets as W

log = logging.getLogger(__name__)

DETECTORS = [
    ("angle_bandpass", "Head-angle band-pass (recommended)"),
    ("motion_spectral", "Motion-energy spectral (cross-check)"),
    ("both_agree", "Both must agree (most conservative)"),
    ("either", "Either detector (most sensitive)"),
]
ANGLE_SOURCES = [
    ("headpoint", "Head direction, sub-pixel (recommended)"),
    ("ellipse", "Fitted-ellipse axis (legacy, 180° ambiguous)"),
    ("moments", "Second-moment axis (180° ambiguous)"),
]


class AnalysisTab(QtWidgets.QWidget):
    statusMessage = QtCore.pyqtSignal(str)

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.pal = theme.palette(ctx.cfg.get("ui.theme", "light"))
        self.worker: W.Worker | None = None
        #: How the current settings were arrived at, written into the result
        #: sidecar. A threshold fitted to a control animal and a band folded
        #: for a frame rate are both decisions a reader of the numbers needs
        #: to be able to see.
        self._advice_provenance: dict | None = None
        self._control_calibration: dict | None = None
        self.run_job = W.JobRunner("analysis", self)
        self.run_job.progressed.connect(self._on_progress)
        self.run_job.finished.connect(self._on_done)
        self.run_job.failed.connect(self._on_failed)
        self.run_job.cancelled.connect(
            lambda: self.lbl_run.set("Analysis cancelled.", "warn"))
        self._build()
        W.no_wheel(self)
        self.run_job._button = self.btn_run
        self.run_job._cancel_button = self.btn_cancel
        self.run_job._progress = self.progress
        self.btn_cancel.clicked.connect(self.run_job.cancel)
        self._load()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        left_inner = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(left_inner)
        left.setContentsMargins(0, 0, 6, 0)
        left.addWidget(self._group_target())
        left.addWidget(self._group_tracking())
        left.addWidget(self._group_detector())
        left.addWidget(self._group_perf())
        left.addStretch(1)

        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(6, 0, 0, 0)
        rv.addWidget(self._group_run())
        rv.addWidget(self._group_results(), 1)
        rv.addWidget(self._group_export())

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(W.scrollable(left_inner))
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 4)
        outer.addWidget(split)

    def _group_target(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Recording")
        v = QtWidgets.QVBoxLayout(g)
        self.lbl_target = QtWidgets.QLabel("Nothing selected.")
        self.lbl_target.setWordWrap(True)
        v.addWidget(self.lbl_target)
        row = QtWidgets.QHBoxLayout()
        self.btn_open = W.button("Open a video file…")
        self.btn_open.clicked.connect(self._open_file)
        row.addWidget(self.btn_open)
        row.addStretch(1)
        v.addLayout(row)
        self.lbl_fps_note = W.StatusLabel("", "info")
        v.addWidget(self.lbl_fps_note)

        rate = QtWidgets.QHBoxLayout()
        self.sp_rate_override = QtWidgets.QDoubleSpinBox()
        self.sp_rate_override.setRange(0, 5000)
        self.sp_rate_override.setDecimals(2)
        self.sp_rate_override.setSpecialValueText("use the file's own rate")
        self.sp_rate_override.setSuffix(" fps")
        self.sp_rate_override.setMinimumWidth(190)
        self.sp_rate_override.setToolTip(
            "Analyse the frames at a different rate from the one the file "
            "declares.\n\n"
            "This is for slow-motion renders. A published supplementary video "
            "captured at 240 fps and written at 30 fps for playback still has "
            "every frame, but a 90 Hz twitch sits at 11 Hz in the file's own "
            "timebase, so a 78-98 Hz detector finds nothing while the twitch is "
            "plainly visible by eye. Setting the true capture rate here fixes it "
            "outright.\n\n"
            "It cannot rescue a recording that was genuinely captured below "
            "~200 fps: there the oscillation aliased at the sensor and the "
            "information is gone. 'Signal diagnostics' tells the two apart.")
        rate.addWidget(QtWidgets.QLabel("Capture-rate override"))
        rate.addWidget(self.sp_rate_override)
        self.btn_spectrum = W.button(
            "Signal diagnostics…",
            tooltip="Shows where the energy in this recording actually is, "
                    "against the published 78-98 Hz HTR band and the file's "
                    "Nyquist limit. Run an analysis first.")
        self.btn_spectrum.setEnabled(False)
        self.btn_spectrum.clicked.connect(self._show_spectrum)
        rate.addWidget(self.btn_spectrum)
        rate.addStretch(1)
        v.addLayout(rate)
        self.lbl_rate_note = W.StatusLabel("", "info")
        self.lbl_rate_note.setWordWrap(True)
        self.lbl_rate_note.setVisible(False)
        v.addWidget(self.lbl_rate_note)
        self.sp_rate_override.valueChanged.connect(self._on_rate_override)
        return g

    def _group_tracking(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Mouse tracking")
        form = QtWidgets.QFormLayout(g)

        self.cb_angle = QtWidgets.QComboBox()
        for key, label in ANGLE_SOURCES:
            self.cb_angle.addItem(label, key)
        W.form_row(form, "Angle source", self.cb_angle,
                   "The head-direction estimator reports a full 0-360° heading, "
                   "so it has no 180° flips; the ellipse and moment axes do, and "
                   "each flip is a step that the band-pass reads as a twitch.")

        self.cb_seg = QtWidgets.QComboBox()
        self.cb_seg.addItems(["auto", "threshold", "background"])
        W.form_row(form, "Segmentation", self.cb_seg,
                   "'threshold' separates the animal by brightness alone, which "
                   "is right on a clean, evenly lit arena. 'background' builds "
                   "an empty-arena image from the median of sampled frames and "
                   "subtracts it, which cancels bedding, the cylinder rim and a "
                   "lighting gradient - use it on real footage where plain "
                   "thresholding grabs the litter as well as the mouse. 'auto' "
                   "keeps threshold while it is working and falls back to "
                   "background when it is not.")

        self.cb_polarity = QtWidgets.QComboBox()
        self.cb_polarity.addItems(["auto", "dark_on_light", "light_on_dark"])
        W.form_row(form, "Mouse vs background", self.cb_polarity,
                   "A dark mouse on a pale floor is 'dark_on_light'. 'auto' tries "
                   "both on sampled frames and keeps whichever finds the animal "
                   "more often.")

        self.cb_thresh_mode = QtWidgets.QComboBox()
        self.cb_thresh_mode.addItems(["otsu", "fixed"])
        W.form_row(form, "Segmentation threshold", self.cb_thresh_mode,
                   "Otsu is computed once from frames across the whole file and then "
                   "frozen, so segmentation does not drift with the animal's "
                   "position.")
        self.sp_thresh = QtWidgets.QSpinBox()
        self.sp_thresh.setRange(1, 254)
        form.addRow("  fixed value", self.sp_thresh)

        self.sp_bg_samples = QtWidgets.QSpinBox()
        self.sp_bg_samples.setRange(8, 400)
        W.form_row(form, "Background frames", self.sp_bg_samples,
                   "How many frames are pooled into the empty-arena median. "
                   "More is steadier and slower; 40 is enough unless the animal "
                   "spends most of the trial in one spot, which puts it into "
                   "the median.")
        self.sp_bg_thr = QtWidgets.QSpinBox()
        self.sp_bg_thr.setRange(0, 200)
        self.sp_bg_thr.setSpecialValueText("auto (Otsu)")
        form.addRow("  difference level", self.sp_bg_thr)

        self.cb_roi = QtWidgets.QComboBox()
        self.cb_roi.addItems(["none", "auto", "manual"])
        W.form_row(form, "Region of interest", self.cb_roi,
                   "Off by default, deliberately. An ROI drawn tight to the "
                   "arena clips the animal whenever it presses against the "
                   "wall, and a clipped outline distorts the angle far more "
                   "than the stray contrast the ROI removes. Measured on the "
                   "realistic benchmark: auto-ROI took F1 from 1.00 to 0.50.")
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
        W.form_row(form, "Blob area min (frame fraction)", self.sp_min_area,
                   "Rejects dust and reflections.")
        W.form_row(form, "Blob area max", self.sp_max_area,
                   "Rejects the whole frame going dark.")

        self.sp_morph = QtWidgets.QSpinBox()
        self.sp_morph.setRange(1, 31)
        self.sp_core = QtWidgets.QSpinBox()
        self.sp_core.setRange(0, 41)
        W.form_row(form, "Clean-up kernel (px)", self.sp_morph)
        self.sp_core.setSpecialValueText("auto (measured)")
        W.form_row(form, "Tail-removal kernel (px)", self.sp_core,
                   "Must be wider than the tail and narrower than the body. The "
                   "opening removes the tail so the head end can be identified, "
                   "and the head arrow points away from whatever it removed.\n\n"
                   "Leave it on 'auto': the right value is set by how many "
                   "pixels the animal occupies, so it changes with resolution "
                   "and camera height. A fixed 9 px - correct at 320x240 - left "
                   "the tail inside the body on 906x680 footage and put the "
                   "arrow on the wrong end in up to 88% of frames.")

        self.sp_gap = QtWidgets.QDoubleSpinBox()
        self.sp_gap.setRange(0, 2000)
        self.sp_gap.setSuffix(" ms")
        W.form_row(form, "Interpolate gaps up to", self.sp_gap,
                   "Short tracking losses are bridged. Longer ones stay marked "
                   "invalid, and a candidate burst sitting on them is rejected.")

        self.chk_edge = QtWidgets.QCheckBox(
            "Mark frames where the animal touches the edge as untracked")
        self.chk_edge.setToolTip(
            "Off by default. A clipped outline does give a wrong angle, but on "
            "a tight arena this rejects so many frames that genuine twitches "
            "go with them - it cost the legacy fixture every one of its six "
            "events in testing.")
        form.addRow(self.chk_edge)

        self.btn_tune = W.button(
            "Tune tracking…", "primary",
            "Opens the frame-by-frame segmentation view. It runs the same code "
            "the analysis does, so what you see there is what the analysis "
            "will do.")
        self.btn_tune.clicked.connect(self._tune)
        form.addRow(self.btn_tune)
        return g

    def _group_detector(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Twitch detection")
        form = QtWidgets.QFormLayout(g)

        self.cb_detector = QtWidgets.QComboBox()
        for key, label in DETECTORS:
            self.cb_detector.addItem(label, key)
        W.form_row(form, "Detector", self.cb_detector,
                   "'Both must agree' buys precision at a real cost in recall: the "
                   "spectral detector needs a sustained oscillation and misses short "
                   "bursts. Prefer it only when a false positive is worse than a "
                   "miss.")

        band = QtWidgets.QHBoxLayout()
        self.sp_band_lo = QtWidgets.QDoubleSpinBox()
        self.sp_band_lo.setRange(1, 400)
        self.sp_band_lo.setDecimals(0)
        self.sp_band_hi = QtWidgets.QDoubleSpinBox()
        self.sp_band_hi.setRange(2, 500)
        self.sp_band_hi.setDecimals(0)
        band.addWidget(self.sp_band_lo)
        band.addWidget(QtWidgets.QLabel("to"))
        band.addWidget(self.sp_band_hi)
        band.addWidget(QtWidgets.QLabel("Hz"))
        W.form_row(form, "Pass band", band,
                   "Published HTR frequency is 78-98 Hz (Halberstadt 2014). 60-110 Hz "
                   "leaves margin without admitting locomotion harmonics.")

        self.cb_thr_src = QtWidgets.QComboBox()
        self.cb_thr_src.addItems(["baseline", "whole", "manual"])
        W.form_row(form, "Threshold from", self.cb_thr_src,
                   "'baseline' uses the pre-dose period, so the drug effect cannot "
                   "raise its own threshold.")
        self.sp_thr_k = QtWidgets.QDoubleSpinBox()
        self.sp_thr_k.setRange(1, 30)
        self.sp_thr_k.setDecimals(1)
        W.form_row(form, "  robust sigma multiplier", self.sp_thr_k,
                   "median + k x MAD-sigma. Validated stable from k=4 to k=8 on the "
                   "synthetic benchmark; raise it if a noisy trial over-reports.")
        self.sp_thr_manual = QtWidgets.QDoubleSpinBox()
        self.sp_thr_manual.setRange(0, 1e6)
        self.sp_thr_manual.setDecimals(1)
        form.addRow("  manual value", self.sp_thr_manual)
        self.sp_thr_low = QtWidgets.QDoubleSpinBox()
        self.sp_thr_low.setRange(0.05, 0.99)
        self.sp_thr_low.setDecimals(2)
        self.sp_thr_low.setSingleStep(0.05)
        W.form_row(form, "  burst release fraction", self.sp_thr_low,
                   "Hysteresis. A burst starts above the threshold and continues "
                   "down to this fraction of it, which keeps one twitch as one "
                   "event instead of two or three.")

        base = QtWidgets.QHBoxLayout()
        self.sp_base_lo = QtWidgets.QDoubleSpinBox()
        self.sp_base_lo.setRange(0, 1e5)
        self.sp_base_lo.setDecimals(0)
        self.sp_base_hi = QtWidgets.QDoubleSpinBox()
        self.sp_base_hi.setRange(0, 1e5)
        self.sp_base_hi.setDecimals(0)
        base.addWidget(self.sp_base_lo)
        base.addWidget(QtWidgets.QLabel("to"))
        base.addWidget(self.sp_base_hi)
        base.addWidget(QtWidgets.QLabel("s"))
        W.form_row(form, "Baseline window", base,
                   "Leave both at 0 to use the trial's recorded pre-dose period.")

        dur = QtWidgets.QHBoxLayout()
        self.sp_min_ms = QtWidgets.QDoubleSpinBox()
        self.sp_min_ms.setRange(5, 2000)
        self.sp_min_ms.setDecimals(0)
        self.sp_max_ms = QtWidgets.QDoubleSpinBox()
        self.sp_max_ms.setRange(20, 5000)
        self.sp_max_ms.setDecimals(0)
        dur.addWidget(self.sp_min_ms)
        dur.addWidget(QtWidgets.QLabel("to"))
        dur.addWidget(self.sp_max_ms)
        dur.addWidget(QtWidgets.QLabel("ms"))
        W.form_row(form, "Burst duration", dur,
                   "5-11 oscillations at 78-98 Hz is 51-141 ms, so 45 ms is the floor. "
                   "The first-generation pipeline's 100 ms minimum would reject the "
                   "shortest genuine twitches.")

        self.sp_merge = QtWidgets.QDoubleSpinBox()
        self.sp_merge.setRange(0, 1000)
        self.sp_merge.setDecimals(0)
        self.sp_merge.setSuffix(" ms")
        W.form_row(form, "Merge gaps shorter than", self.sp_merge,
                   "Two twitches this close together are not physiologically "
                   "plausible, so a gap this short is one burst, not two.")

        freq = QtWidgets.QHBoxLayout()
        self.sp_freq_lo = QtWidgets.QDoubleSpinBox()
        self.sp_freq_lo.setRange(1, 400)
        self.sp_freq_lo.setDecimals(0)
        self.sp_freq_hi = QtWidgets.QDoubleSpinBox()
        self.sp_freq_hi.setRange(2, 500)
        self.sp_freq_hi.setDecimals(0)
        freq.addWidget(self.sp_freq_lo)
        freq.addWidget(QtWidgets.QLabel("to"))
        freq.addWidget(self.sp_freq_hi)
        freq.addWidget(QtWidgets.QLabel("Hz"))
        W.form_row(form, "Require in-burst frequency", freq,
                   "Measured from zero crossings inside the candidate. Rejects "
                   "broadband lurches that cross the threshold without oscillating.")

        self.sp_half = QtWidgets.QSpinBox()
        self.sp_half.setRange(0, 60)
        W.form_row(form, "Minimum half-cycles", self.sp_half,
                   "A twitch is 5-11 oscillations, i.e. 10-22 half-cycles. Six is a "
                   "forgiving floor.")
        self.sp_invalid = QtWidgets.QDoubleSpinBox()
        self.sp_invalid.setRange(0, 1)
        self.sp_invalid.setDecimals(2)
        self.sp_invalid.setSingleStep(0.05)
        W.form_row(form, "Max invalid frames in a burst", self.sp_invalid,
                   "Fraction. A burst on lost tracking is an artefact.")
        return g

    def _group_perf(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Performance")
        form = QtWidgets.QFormLayout(g)
        self.sp_workers = QtWidgets.QSpinBox()
        self.sp_workers.setRange(0, 32)
        self.sp_workers.setSpecialValueText("auto")
        W.form_row(form, "Worker threads", self.sp_workers,
                   "Frames are decoded in parallel over separate ranges of the file. "
                   "Verified to give bit-identical signals to a single-threaded run. "
                   "Only used on files longer than ~20 000 frames and only when "
                   "frame-exact seeking works (it does for MJPEG in AVI).")
        self.sp_maxframes = QtWidgets.QSpinBox()
        self.sp_maxframes.setRange(0, 100_000_000)
        self.sp_maxframes.setSpecialValueText("no limit")
        self.sp_maxframes.setSingleStep(10_000)
        W.form_row(form, "Frame limit", self.sp_maxframes,
                   "For a quick look at the first part of a long trial. Analysis is "
                   "streaming, so there is no memory reason to cap it.")
        return g

    def _group_run(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Run")
        v = QtWidgets.QVBoxLayout(g)
        row = QtWidgets.QHBoxLayout()
        self.btn_run = W.button("Run analysis", "primary")
        self.btn_cancel = W.button("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_defaults = W.button(
            "Reset to defaults",
            tooltip="Restores every analysis setting to the values the code "
                    "ships with. Lists what will change and asks first.")
        self.btn_validate = W.button(
            "Validate against ground truth…",
            tooltip="Scores the current settings against a CSV of known twitch "
                    "times. Columns: time_sec, twitch_active.")
        for b in (self.btn_run, self.btn_cancel, self.btn_defaults, self.btn_validate):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)

        row2 = QtWidgets.QHBoxLayout()
        self.btn_suggest = W.button(
            "Suggest settings for this file…",
            tooltip="Reads the recording's resolution and true frame rate, "
                    "calibrates on a sample of its frames, and lists the "
                    "settings that do not suit it - with the reasoning for "
                    "each. Nothing changes until you press Apply.\n\n"
                    "The frame rate is the setting everything else hangs "
                    "off: below about 200 fps the 78-98 Hz twitch is above "
                    "Nyquist and folds down to a lower frequency, so the "
                    "band-pass has to move to where the signal actually "
                    "lands rather than where the twitch really is.")
        self.btn_control = W.button(
            "Calibrate from a control video…",
            tooltip="Fits the detection threshold to an untreated recording, "
                    "where every call is a false positive by construction. "
                    "Sweeps the threshold, shows the false-positive curve, "
                    "and picks one step below the point at which the control "
                    "falls silent.\n\n"
                    "Run this once per session on a vehicle animal, then "
                    "apply it to that session's treated recordings - it is "
                    "fitted to one lighting, exposure, zoom and arena.")
        self.cb_profile = QtWidgets.QComboBox()
        self.cb_profile.setToolTip(
            "Session profiles saved from a control-video calibration. Each "
            "one carries the control file, the sweep and the margin it was "
            "fitted with, so a count can be traced back to how its "
            "threshold was arrived at.")
        self.btn_profile_apply = W.button("Apply profile")
        self.btn_profile_apply.setEnabled(False)
        row2.addWidget(self.btn_suggest)
        row2.addWidget(self.btn_control)
        row2.addStretch(1)
        row2.addWidget(QtWidgets.QLabel("Session profile:"))
        row2.addWidget(self.cb_profile)
        row2.addWidget(self.btn_profile_apply)
        v.addLayout(row2)
        self.btn_suggest.clicked.connect(self._suggest_settings)
        self.btn_control.clicked.connect(self._calibrate_control)
        self.btn_profile_apply.clicked.connect(self._apply_profile)
        self.cb_profile.currentIndexChanged.connect(
            lambda i: self.btn_profile_apply.setEnabled(
                self.cb_profile.itemData(i) is not None))
        self._refresh_profiles()
        self.progress = QtWidgets.QProgressBar()
        v.addWidget(self.progress)
        self.lbl_run = W.StatusLabel("Pick a recording in the Library, then press "
                                     "Run analysis.", "info")
        v.addWidget(self.lbl_run)

        self.btn_run.clicked.connect(self._run)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_defaults.clicked.connect(self._reset_defaults)
        self.btn_validate.clicked.connect(self._validate)
        return g

    def _group_results(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Result")
        v = QtWidgets.QVBoxLayout(g)
        tiles = QtWidgets.QHBoxLayout()
        self.t_count = W.MetricTile("twitches")
        self.t_rate = W.MetricTile("peak rate", "-", "/min")
        self.t_thr = W.MetricTile("threshold")
        self.t_track = W.MetricTile("tracking", "-", "%")
        self.t_rej = W.MetricTile("rejected")
        for t in (self.t_count, self.t_rate, self.t_thr, self.t_track, self.t_rej):
            tiles.addWidget(t)
        v.addLayout(tiles)
        self.txt_diag = QtWidgets.QPlainTextEdit()
        self.txt_diag.setReadOnly(True)
        v.addWidget(self.txt_diag, 1)
        return g

    def _group_export(self) -> QtWidgets.QGroupBox:
        g = QtWidgets.QGroupBox("Export")
        row = QtWidgets.QHBoxLayout(g)
        self.btn_csv = W.button("Save CSV report", "primary")
        self.btn_png = W.button("Save figure (PNG)")
        self.btn_traces = W.button("Save traces (.npz)")
        self.btn_save_result = W.button("Attach result to the recording")
        for b in (self.btn_csv, self.btn_png, self.btn_traces, self.btn_save_result):
            b.setEnabled(False)
            row.addWidget(b)
        row.addStretch(1)
        self.btn_csv.clicked.connect(self._export_csv)
        self.btn_png.clicked.connect(self._export_png)
        self.btn_traces.clicked.connect(self._export_traces)
        self.btn_save_result.clicked.connect(self._attach_result)
        return g

    # ------------------------------------------------------------------
    # Config binding
    # ------------------------------------------------------------------
    def _load(self) -> None:
        self.apply_config(
            analysis.AnalysisConfig.from_dict(self.ctx.cfg.section("analysis")))

    def apply_config(self, a: analysis.AnalysisConfig) -> None:
        """Push a config into the widgets.

        The inverse of :meth:`current_config`, and the single way anything
        sets these controls -- stored settings, the shipped defaults, the
        per-file advisor and a session profile all come through here, so a
        control added later is handled by all four at once.
        """
        self._set_combo(self.cb_angle, a.angle_source)
        self.cb_seg.setCurrentText(a.segmentation)
        self.cb_polarity.setCurrentText(a.polarity)
        self.cb_thresh_mode.setCurrentText(a.threshold_mode)
        self.sp_thresh.setValue(int(a.threshold_value))
        self.sp_bg_samples.setValue(int(a.bg_samples))
        self.sp_bg_thr.setValue(int(a.bg_threshold_value))
        self.cb_roi.setCurrentText(a.roi_mode)
        for s, val in zip(self.sp_roi, a.roi_frac):
            s.setValue(float(val))
        self.chk_edge.setChecked(bool(a.reject_edge_blobs))
        self.sp_min_area.setValue(a.min_area_frac)
        self.sp_max_area.setValue(a.max_area_frac)
        self.sp_morph.setValue(a.morph_kernel)
        self.sp_core.setValue(a.core_kernel)
        self.sp_gap.setValue(a.max_gap_ms)
        self._set_combo(self.cb_detector, a.detector)
        self.sp_band_lo.setValue(a.band_low_hz)
        self.sp_band_hi.setValue(a.band_high_hz)
        self.cb_thr_src.setCurrentText(a.threshold_source)
        self.sp_thr_k.setValue(a.threshold_k)
        self.sp_thr_manual.setValue(a.threshold_manual)
        self.sp_thr_low.setValue(a.threshold_low_frac)
        self.sp_base_lo.setValue(a.baseline_window_s[0])
        self.sp_base_hi.setValue(a.baseline_window_s[1])
        self.sp_min_ms.setValue(a.min_burst_ms)
        self.sp_max_ms.setValue(a.max_burst_ms)
        self.sp_merge.setValue(a.merge_gap_ms)
        self.sp_freq_lo.setValue(a.require_freq_hz[0])
        self.sp_freq_hi.setValue(a.require_freq_hz[1])
        self.sp_half.setValue(a.min_half_cycles)
        self.sp_invalid.setValue(a.max_invalid_frac)
        self.sp_workers.setValue(a.workers)
        self.sp_maxframes.setValue(a.max_frames)

    @staticmethod
    def _set_combo(cb: QtWidgets.QComboBox, key: str) -> None:
        for i in range(cb.count()):
            if cb.itemData(i) == key:
                cb.setCurrentIndex(i)
                return

    def current_config(self) -> analysis.AnalysisConfig:
        return analysis.AnalysisConfig(
            segmentation=self.cb_seg.currentText(),
            polarity=self.cb_polarity.currentText(),
            threshold_mode=self.cb_thresh_mode.currentText(),
            threshold_value=int(self.sp_thresh.value()),
            bg_samples=int(self.sp_bg_samples.value()),
            bg_threshold_value=int(self.sp_bg_thr.value()),
            roi_mode=self.cb_roi.currentText(),
            roi_frac=tuple(float(s.value()) for s in self.sp_roi),
            reject_edge_blobs=bool(self.chk_edge.isChecked()),
            min_area_frac=float(self.sp_min_area.value()),
            max_area_frac=float(self.sp_max_area.value()),
            morph_kernel=int(self.sp_morph.value()),
            core_kernel=int(self.sp_core.value()),
            angle_source=self.cb_angle.currentData(),
            max_gap_ms=float(self.sp_gap.value()),
            detector=self.cb_detector.currentData(),
            band_low_hz=float(self.sp_band_lo.value()),
            band_high_hz=float(self.sp_band_hi.value()),
            threshold_source=self.cb_thr_src.currentText(),
            threshold_k=float(self.sp_thr_k.value()),
            threshold_manual=float(self.sp_thr_manual.value()),
            threshold_low_frac=float(self.sp_thr_low.value()),
            baseline_window_s=(float(self.sp_base_lo.value()),
                               float(self.sp_base_hi.value())),
            merge_gap_ms=float(self.sp_merge.value()),
            min_burst_ms=float(self.sp_min_ms.value()),
            max_burst_ms=float(self.sp_max_ms.value()),
            require_freq_hz=(float(self.sp_freq_lo.value()),
                             float(self.sp_freq_hi.value())),
            min_half_cycles=int(self.sp_half.value()),
            max_invalid_frac=float(self.sp_invalid.value()),
            workers=int(self.sp_workers.value()),
            max_frames=int(self.sp_maxframes.value()),
        )

    def save_to_config(self) -> None:
        self.ctx.cfg.data["analysis"] = self.current_config().to_dict()

    def _reset_defaults(self) -> None:
        """Back to the dataclass defaults, with a confirm step.

        Reset from :class:`analysis.AnalysisConfig` itself rather than from
        the stored settings file, so "defaults" means the values the code
        ships with even if the stored section has been edited by hand or
        written by an older version. The confirm is there because a reset
        discards a threshold that may have taken a control-video sweep to
        arrive at.
        """
        cur = self.current_config()
        shipped = analysis.AnalysisConfig()
        changed = [f for f in shipped.__dataclass_fields__
                   if getattr(cur, f) != getattr(shipped, f)]
        if not changed:
            self.lbl_run.set("Already at the shipped defaults - "
                             "nothing to reset.", "info")
            return
        if not W.ask_yes_no(
                self, "Reset analysis settings",
                f"{len(changed)} setting(s) differ from the shipped defaults "
                "and will be restored:<br><br><tt>"
                + ", ".join(sorted(changed))
                + "</tt><br><br>If any of these came from a control-video "
                  "calibration, save that as a session profile first or it "
                  "will have to be measured again."):
            return
        self.apply_config(shipped)
        self.ctx.cfg.data["analysis"] = shipped.to_dict()
        self.lbl_run.set(
            f"Reset {len(changed)} setting(s) to the shipped defaults: "
            + ", ".join(sorted(changed)), "ok")
        log.info("analysis settings reset to defaults: %s", ", ".join(changed))

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------
    def set_entry(self, entry: library.Entry) -> None:
        # ctx.entry is set by AppContext.set_entry, which is what called this.
        # A rate override belongs to the file it was worked out for. Carrying it
        # silently onto the next recording would put every timing figure in that
        # trial out by the same factor, so it is cleared here.
        self.sp_rate_override.blockSignals(True)
        self.sp_rate_override.setValue(0.0)
        self.sp_rate_override.blockSignals(False)
        self.lbl_rate_note.setVisible(False)
        self.btn_spectrum.setEnabled(False)
        # Per-file advice belongs to the file it was measured on, so it is
        # cleared here. A session calibration deliberately is not: it is
        # fitted once per session and applied across that session's
        # recordings, which is the whole point of it.
        self._advice_provenance = None
        fps = entry.fps_for_analysis()
        self.lbl_target.setText(
            f"<b>{entry.name}</b><br>{entry.path}<br>"
            f"{entry.width}x{entry.height} · {entry.duration_hms} · "
            f"{W.human_bytes(entry.size_bytes)}"
            + (f"<br>{len(entry.segments)} segment(s); the first is analysed - "
               "analyse the rest from the Batch tab" if entry.is_session else ""))
        if entry.fps_exact or (entry.sidecar.get("measured") or {}).get("exact"):
            self.lbl_fps_note.set(
                f"Frame rate {fps:.3f} fps, verified by an exact frame count.", "ok")
        elif entry.expected_fps > 1:
            self.lbl_fps_note.set(
                f"Using {fps:.3f} fps from the recording's own sidecar. Run "
                "'Verify frame rate' in the Library to confirm it by counting "
                "frames - every timing figure depends on it.", "warn")
        else:
            self.lbl_fps_note.set(
                f"Using {fps:.3f} fps from the container header, which is "
                "unreliable for MJPEG in AVI. Verify it in the Library before "
                "trusting the counts.", "bad")
        base = entry.baseline_s
        if base > 0 and self.sp_base_hi.value() == 0:
            self.lbl_run.set(
                f"Ready. The threshold will come from the first {base / 60:.0f} min "
                "(the recorded pre-dose baseline).", "info")
        else:
            self.lbl_run.set("Ready.", "info")

    def effective_fps(self) -> float:
        """The rate the analysis will use: the override if set, else the file's.

        Kept in one place because every timing figure in the study -- burst
        duration, dominant frequency, counts per minute -- is derived from it,
        and a run whose rate came from somewhere unexpected is a run whose
        numbers cannot be defended.
        """
        over = float(self.sp_rate_override.value())
        if over > 0:
            return over
        return self.ctx.entry.fps_for_analysis() if self.ctx.entry else 0.0

    def _on_rate_override(self, value: float) -> None:
        if value <= 0:
            self.lbl_rate_note.setVisible(False)
            return
        entry = self.ctx.entry
        native = entry.fps_for_analysis() if entry else 0.0
        self.lbl_rate_note.setVisible(True)
        if native > 0:
            factor = value / native
            self.lbl_rate_note.set(
                f"Analysing at {value:.2f} fps instead of the file's "
                f"{native:.2f} fps, i.e. treating this file as a {factor:.1f}x "
                "slow-motion render. Every time and frequency in the result is "
                "now in real time, not file time. The exported report records "
                "the override.", "warn")
        else:
            self.lbl_rate_note.set(
                f"Analysing at {value:.2f} fps.", "warn")

    def _show_spectrum(self) -> None:
        if self.ctx.signals is None:
            W.warn_box(self, "Nothing to look at",
                       "Run an analysis first - the spectrum is computed from "
                       "the extracted head-angle trace.")
            return
        from .diagnostics import SpectrumDialog
        dlg = SpectrumDialog(self.ctx.signals, self.pal, self)
        dlg.rateChosen.connect(self._adopt_rate)
        dlg.exec_()

    def _adopt_rate(self, fps: float) -> None:
        self.sp_rate_override.setValue(float(fps))
        self.lbl_run.set(
            f"Capture-rate override set to {fps:.0f} fps. Press Run analysis "
            "again - the extraction is repeated at the new timebase.", "warn")

    def _tune(self) -> None:
        entry = self.ctx.entry
        if entry is None:
            W.warn_box(self, "Nothing to tune",
                       "Select a recording in the Library, or use 'Open a video "
                       "file' here.")
            return
        from .diagnostics import TuneDialog
        dlg = TuneDialog(entry.analysis_target, self.current_config(),
                         self.pal, self)
        dlg.configChanged.connect(self._adopt_tracking)
        dlg.exec_()

    # ------------------------------------------------------------------
    # Settings advice and session calibration
    # ------------------------------------------------------------------
    def _suggest_settings(self) -> None:
        entry = self.ctx.entry
        if entry is None:
            W.warn_box(self, "No recording selected",
                       "Select a recording in the Library, or use 'Open a "
                       "video file' here, so there is something to inspect.")
            return
        from .advisor_dialog import SuggestDialog
        dlg = SuggestDialog(self.ctx, entry.analysis_target,
                            self.current_config(), self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted and dlg.result_cfg:
            self.apply_config(dlg.result_cfg)
            # Keep the reasoning: it goes into the result sidecar, so the
            # settings a figure was produced under carry their justification.
            self._advice_provenance = (dlg.advice.provenance()
                                       if dlg.advice else None)
            n = len(dlg.advice.pending) if dlg.advice else 0
            self.lbl_run.set(
                f"Applied {n} suggested setting(s) for this recording. Run "
                "the analysis to use them; the reasoning is saved with the "
                "result.", "warn")

    def _calibrate_control(self) -> None:
        entry = self.ctx.entry
        if entry is None:
            W.warn_box(self, "No recording selected",
                       "Select the untreated control recording in the "
                       "Library first - the calibration is fitted to it.")
            return
        if not W.ask_yes_no(
                self, "Is this the control recording?",
                f"<b>{entry.path.name}</b><br><br>"
                "This fits the detection threshold on the assumption that "
                "the animal in this recording received no drug, so that "
                "every burst found in it is a false positive.<br><br>"
                "Running it on a treated recording would tune the threshold "
                "until the real twitches disappeared. Is this the control?"):
            return
        from .advisor_dialog import ControlCalibrationDialog
        dlg = ControlCalibrationDialog(
            self.ctx, entry.analysis_target, self.current_config(),
            fps_override=float(self.sp_rate_override.value()), parent=self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted and dlg.result_cfg:
            self.apply_config(dlg.result_cfg)
            self._control_calibration = (dlg.cal.to_dict() if dlg.cal else None)
            self.lbl_run.set(
                f"Threshold set from the control recording: k = "
                f"{dlg.result_cfg.threshold_k:g}. Apply this to the treated "
                "recordings of the same session only.", "warn")
        self._refresh_profiles()

    def _refresh_profiles(self) -> None:
        self.cb_profile.blockSignals(True)
        self.cb_profile.clear()
        self.cb_profile.addItem("(none)", None)
        for doc in advisor.list_profiles():
            cal = doc.get("control_calibration") or {}
            k = (doc.get("config") or {}).get("threshold_k")
            label = str(doc.get("name") or Path(doc["path"]).stem)
            if k is not None:
                label += f"  (k={float(k):g})"
            self.cb_profile.addItem(label, doc["path"])
            self.cb_profile.setItemData(
                self.cb_profile.count() - 1,
                (cal.get("message") or "no calibration recorded"),
                QtCore.Qt.ToolTipRole)
        self.cb_profile.blockSignals(False)
        self.btn_profile_apply.setEnabled(False)

    def _apply_profile(self) -> None:
        path = self.cb_profile.currentData()
        if not path:
            return
        try:
            doc = advisor.load_profile(path)
        except (OSError, ValueError) as e:
            W.warn_box(self, "Could not read the profile", str(e))
            return
        cfg = analysis.AnalysisConfig.from_dict(doc.get("config") or {})
        self.apply_config(cfg)
        self._control_calibration = doc.get("control_calibration")
        cal = doc.get("control_calibration") or {}
        self.lbl_run.set(
            f"Applied the session profile '{doc.get('name', '')}'. "
            + (cal.get("message") or ""), "warn")

    def _adopt_tracking(self, fields: dict) -> None:
        self.cb_seg.setCurrentText(fields["segmentation"])
        self.cb_polarity.setCurrentText(fields["polarity"])
        self.cb_thresh_mode.setCurrentText(fields["threshold_mode"])
        self.sp_thresh.setValue(int(fields["threshold_value"]))
        self.sp_bg_samples.setValue(int(fields["bg_samples"]))
        self.sp_bg_thr.setValue(int(fields["bg_threshold_value"]))
        self.cb_roi.setCurrentText(fields["roi_mode"])
        for s, val in zip(self.sp_roi, fields["roi_frac"]):
            s.setValue(float(val))
        self.sp_min_area.setValue(float(fields["min_area_frac"]))
        self.sp_max_area.setValue(float(fields["max_area_frac"]))
        self.sp_morph.setValue(int(fields["morph_kernel"]))
        self.sp_core.setValue(int(fields["core_kernel"]))
        self.chk_edge.setChecked(bool(fields["reject_edge_blobs"]))
        self.lbl_run.set("Tracking settings adopted. Run the analysis again to "
                         "apply them to the whole recording.", "warn")

    def _open_file(self) -> None:
        start = str(self.ctx.cfg.recordings_root())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open a video", start,
            "Videos (*.avi *.mp4 *.mkv *.mov);;All files (*)")
        if not path:
            return
        e = library.Entry(path=Path(path))
        from .. import ffmpeg_tools
        info = ffmpeg_tools.probe_video(path, self.ctx.cfg.get("paths.ffprobe", ""),
                                        count_frames=False)
        e.width, e.height, e.codec = info.width, info.height, info.codec
        e.duration_s, e.fps, e.fps_exact = info.duration_s, info.true_fps, info.exact
        try:
            e.size_bytes = Path(path).stat().st_size
        except OSError:
            pass
        e.sidecar = library.read_json(library.sidecar_for(Path(path)))
        e.result = library.read_json(library.result_for(Path(path)))
        self.set_entry(e)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _run(self) -> None:
        entry = self.ctx.entry
        if entry is None:
            W.warn_box(self, "Nothing to analyse",
                       "Select a recording in the Library, or use 'Open a video "
                       "file' here.")
            return
        cfg = self.current_config()
        target = entry.analysis_target
        fps = self.effective_fps()
        baseline = entry.baseline_s

        def job(progress=None):
            sig = analysis.extract_signals(target, cfg, fps=fps, progress=progress)
            det = analysis.detect(sig, cfg, baseline_default_s=baseline)
            return sig, det

        if not self.run_job.start(job):
            self.lbl_run.set("An analysis is already running. Cancel it first.",
                             "warn")
            return
        self.lbl_run.set(f"Analysing {target.name}…", "info")

    def _on_progress(self, done: int, total: int, msg: str) -> None:
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(min(done, total))
        self.lbl_run.set(f"{msg}: {done:,} of {total:,} frames", "info")

    def _cancel(self) -> None:
        self.run_job.cancel()
        self.lbl_run.set("Cancelling…", "warn")

    def _on_failed(self, msg: str, tb: str) -> None:
        self.lbl_run.set(msg, "bad")

    def _on_done(self, result) -> None:
        if result is None:
            return
        sig, det = result
        # Button state is the runner's job now, not this handler's.
        self.ctx.set_analysis(sig, det)

        entry = self.ctx.entry
        t0 = max(entry.t0_offset_s, 0.0) if entry else 0.0
        bin_s = float(self.ctx.cfg.get("report.bin_seconds", 60))
        bins = analysis.bin_counts([b for b in det.bursts if b.accepted],
                                   sig.duration_s, bin_s, t0)
        peak = max((b["rate_per_min"] for b in bins), default=0.0)

        self.t_count.set_value(str(det.count))
        self.t_rate.set_value(f"{peak:.1f}")
        self.t_thr.set_value(f"{det.threshold:.0f}")
        self.t_track.set_value(f"{sig.valid_frac * 100:.1f}")
        self.t_rej.set_value(str(len(det.rejected)))

        lines = [f"detector: {det.detector}",
                 f"frames: {sig.n:,} at {sig.fps:.3f} fps "
                 f"({sig.duration_s / 60:.2f} min)",
                 f"threshold: {det.threshold:.3f}  "
                 f"({det.diagnostics.get('threshold_note', '')})",
                 f"accepted: {det.count}   rejected: {len(det.rejected)}"]
        if det.warnings or sig.warnings:
            lines.append("")
            lines.append("WARNINGS")
            for w in det.warnings + sig.warnings:
                lines.append(f"  - {w}")
        flick = det.diagnostics.get("background_flicker") or {}
        if flick.get("checked"):
            lines.append("")
            lines.append(
                f"background band ratio: median {flick.get('bg_band_ratio_median')}, "
                f"p99 {flick.get('bg_band_ratio_p99')} "
                f"({'SUSPECT' if flick.get('suspect') else 'clean'})")
        if det.rejected:
            lines.append("")
            lines.append(f"REJECTED CANDIDATES ({len(det.rejected)}, first 25)")
            for b in det.rejected[:25]:
                lines.append(f"  {b.start_s:9.3f}s  {b.duration_ms:6.0f} ms  {b.note}")
        lines.append("")
        lines.append("ACCEPTED EVENTS")
        for i, b in enumerate(det.bursts, 1):
            lines.append(
                f"  {i:3d}  {b.start_s:9.3f}-{b.end_s:9.3f}s  "
                f"{b.duration_ms:6.0f} ms  {b.dominant_hz:5.1f} Hz  "
                f"{b.half_cycles:3d} half-cycles")
        self.txt_diag.setPlainText("\n".join(lines))

        level = "ok"
        if det.warnings:
            level = "warn"
        if sig.valid_frac < 0.6 or flick.get("suspect"):
            level = "bad"
        advice = self._triage(sig, det)
        if advice:
            level = "bad" if level == "ok" else level
            lines.insert(0, "WHAT TO DO NEXT")
            lines.insert(1, "  " + advice.replace("\n", "\n  "))
            lines.insert(2, "")
            self.txt_diag.setPlainText("\n".join(lines))
        self.lbl_run.set(
            f"Done: {det.count} twitch(es), peak {peak:.1f}/min, "
            f"tracking valid on {sig.valid_frac * 100:.1f}% of frames."
            + ("  Check the warnings before using these counts."
               if level != "ok" and not advice else "")
            + (f"  {advice.splitlines()[0]}" if advice else ""),
            level)
        for b in (self.btn_csv, self.btn_png, self.btn_traces, self.btn_save_result):
            b.setEnabled(True)
        self.btn_spectrum.setEnabled(True)
        self.statusMessage.emit(
            f"Analysis complete: {det.count} twitches in {Path(sig.source_path).name}")

    # ------------------------------------------------------------------
    def _triage(self, sig, det) -> str:
        """One sentence naming the most likely reason a run found nothing.

        "0 twitches" is the same output for two opposite situations -- an animal
        that did not twitch, and a pipeline that could not have seen a twitch if
        it had. Separating those is the whole point of the diagnostics, so the
        run says which one it looks like rather than leaving the operator to
        guess. Ordered most-conclusive first; tracking is checked before the
        timebase because a spectrum of a trace that never found the animal says
        nothing about the recording.
        """
        if sig.valid_frac < 0.80:
            return (f"Tracking only held on {sig.valid_frac * 100:.0f}% of "
                    "frames, so the angle trace is mostly interpolation and no "
                    "count from it is defensible.\n"
                    "Open 'Tune tracking' and try the background segmentation "
                    "mode; on cluttered bedding it is usually the whole fix.")
        if det.count > 0:
            return ""
        try:
            spec = analysis.spectrum(sig)
        except Exception:                       # diagnostics must never break a run
            log.exception("triage spectrum failed")
            return ""
        sugg = analysis.suggest_capture_rate(spec)
        if sugg is not None and sugg.confidence in ("likely", "possible"):
            return (f"Nothing detected, but the animal is being tracked and its "
                    f"strongest oscillation is at {spec.peak_hz:.1f} Hz - far "
                    f"below the 78-98 Hz HTR band.\n"
                    f"This file looks like a {sugg.slowmo_factor:.0f}x "
                    f"slow-motion render of a ~{sugg.implied_fps:.0f} fps "
                    "capture. Open 'Signal diagnostics' and apply that rate.")
        if sig.fps < 200:
            msg = (f"Nothing detected, and at {sig.fps:.0f} fps this file "
                   "cannot resolve the HTR band at all: a 78-98 Hz "
                   "oscillation needs more than 196 fps to sample, so it has "
                   "aliased and cannot be recovered by any setting.\n")
            # If candidates were found and then rejected on frequency, say
            # where they landed. An operator watching a clear bump sit just
            # under the threshold will reach for the sigma multiplier, and on
            # an aliased recording that cannot work: the events are refused on
            # frequency, not amplitude, so lowering the threshold only produces
            # more rejections. Naming the alias band is what ends that search.
            lo_a, hi_a = max(0.0, sig.fps - 98.0), max(0.0, sig.fps - 78.0)
            freqs = [b.dominant_hz for b in det.rejected
                     if "outside the" in (b.note or "") and b.dominant_hz > 0]
            if freqs and hi_a > lo_a:
                import statistics
                med = statistics.median(freqs)
                msg += (
                    f"{len(det.rejected)} candidate(s) were found and then "
                    f"rejected, most on frequency: they measure around "
                    f"{med:.0f} Hz. That is where a real 78-98 Hz twitch folds "
                    f"to at {sig.fps:.0f} fps ({lo_a:.0f}-{hi_a:.0f} Hz), so "
                    "the movement is almost certainly genuine - its frequency "
                    "simply no longer means anything.\n"
                    "Lowering the robust sigma multiplier will NOT help: these "
                    "are refused on frequency, not amplitude. Widening the "
                    "frequency gate to admit them was tested and does not "
                    "work either - on a no-injection control it counted as "
                    "many events as on DOI, because at this rate an aliased "
                    "twitch is indistinguishable from vigorous grooming.\n")
            msg += ("If the camera really ran at this rate, the recording has "
                    "to be repeated at 240 fps. If the file is a slow-motion "
                    "render, set the capture-rate override instead.")
            return msg
        if spec.band_fraction < 0.02:
            return ("Nothing detected. Tracking and frame rate are both fine, "
                    f"and only {spec.band_fraction:.1%} of the movement energy "
                    "is in the HTR band, so the trace genuinely has no twitch "
                    "in it.\n"
                    "Either the animal did not twitch in this window, or the "
                    "head is not being resolved - check a burst by eye in "
                    "'Tune tracking' before loosening the detector.")
        return ("Nothing detected, but there is energy in the HTR band, so the "
                "detector gates are the place to look.\n"
                "Lower the robust sigma multiplier a step at a time and watch "
                "the rejected list for the reason each candidate was refused.")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        if self.ctx.detection is None:
            W.warn_box(self, "Nothing to validate", "Run an analysis first.")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Ground-truth CSV", str(paths.DATA_DIR),
            "CSV (*.csv);;All files (*)")
        if not path:
            return
        try:
            truth = analysis.truth_from_ground_truth_csv(path)
        except (OSError, KeyError, ValueError) as e:
            W.warn_box(self, "Could not read the CSV",
                       f"{e}\n\nExpected columns: time_sec, twitch_active.")
            return
        if not truth:
            W.warn_box(self, "No events",
                       "No twitch intervals were found. The CSV needs a "
                       "'twitch_active' column that is 1 during an event.")
            return
        val = analysis.validate(self.ctx.detection.bursts, truth, 0.15)
        level = "ok" if val.f1 >= 0.9 else ("warn" if val.f1 >= 0.75 else "bad")
        self.lbl_run.set(f"Validation vs {Path(path).name}: {val.line()}", level)
        out = paths.EXPORTS_DIR / f"validation_{Path(path).stem}.csv"
        report.write_validation_report(out, val, label=str(path))
        self.txt_diag.setPlainText(
            f"{val.line()}\n\ntolerance: {val.tolerance_s} s\n"
            f"missed intervals: {val.missed}\n"
            f"spurious detections at: {[round(c, 3) for c in val.spurious]}\n\n"
            f"written to {out}\n\n" + self.txt_diag.toPlainText())

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _ready(self) -> bool:
        return self.ctx.signals is not None and self.ctx.detection is not None

    def _t0_and_bin(self) -> tuple[float, float]:
        entry = self.ctx.entry
        t0 = max(entry.t0_offset_s, 0.0) if entry else 0.0
        return t0, float(self.ctx.cfg.get("report.bin_seconds", 60))

    def _export_csv(self) -> None:
        if not self._ready():
            return
        t0, bin_s = self._t0_and_bin()
        entry = self.ctx.entry
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Where to write the report", str(paths.EXPORTS_DIR))
        if not d:
            return
        rep = report.write_trial_csv(
            Path(self.ctx.signals.source_path), Path(d), self.ctx.signals,
            self.ctx.detection, bin_s=bin_s, t0_s=t0,
            meta=entry.trial if entry else None)
        self.lbl_run.set(f"Wrote {rep.combined_csv.name} and two single-table CSVs "
                         f"to {d}", "ok")

    def _export_png(self) -> None:
        if not self._ready():
            return
        t0, bin_s = self._t0_and_bin()
        entry = self.ctx.entry
        stem = Path(self.ctx.signals.source_path).stem
        default = paths.EXPORTS_DIR / f"{stem}_htr.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save figure", str(default), "PNG (*.png)")
        if not path:
            return
        title = stem
        if entry and entry.animal:
            title = (f"{entry.animal}  {entry.group}  {entry.treatment} "
                     f"{entry.dose} {entry.route}").strip()
        report.plot_trial(Path(path), self.ctx.signals, self.ctx.detection,
                          bin_s=bin_s, t0_s=t0, title=title,
                          theme_name=self.ctx.cfg.get("ui.theme", "light"))
        self.lbl_run.set(f"Figure saved to {path}", "ok")

    def _export_traces(self) -> None:
        if not self._ready():
            return
        stem = Path(self.ctx.signals.source_path).stem
        default = paths.EXPORTS_DIR / f"{stem}_traces.npz"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save traces", str(default), "NumPy archive (*.npz)")
        if not path:
            return
        report.save_traces(Path(path), self.ctx.signals, self.ctx.detection,
                           decimate=1)
        self.lbl_run.set(f"Traces saved to {path}", "ok")

    def _attach_result(self) -> None:
        if not self._ready() or self.ctx.entry is None:
            return
        entry = self.ctx.entry
        t0, bin_s = self._t0_and_bin()
        sig, det = self.ctx.signals, self.ctx.detection
        bins = analysis.bin_counts([b for b in det.bursts if b.accepted],
                                   sig.duration_s, bin_s, t0)
        doc = {
            "htr_studio_result_version": 1,
            "video": str(sig.source_path),
            "signals": sig.summary(),
            "config": self.current_config().to_dict(),
            "fps_used": sig.fps,
            "fps_declared_by_file": (entry.fps_for_analysis() if entry else 0.0),
            "capture_rate_override": float(self.sp_rate_override.value()) or None,
            "detector": det.detector,
            "threshold": det.threshold,
            "diagnostics": det.diagnostics,
            "warnings": det.warnings,
            "bursts": [b.to_dict() for b in det.bursts],
            "rejected": [b.to_dict() for b in det.rejected],
            "bins": bins,
            "n_accepted": det.count,
            "curated": False,
            # How these settings were arrived at. Without this a sidecar
            # records a threshold of 4.5 and a band of 17-47 Hz with no way
            # to tell whether either was deliberate.
            "settings_provenance": {
                "file_advice": self._advice_provenance,
                "control_calibration": self._control_calibration,
            } if (self._advice_provenance or self._control_calibration) else None,
        }
        try:
            p = library.save_analysis(entry, doc)
        except OSError as e:
            self.lbl_run.set(f"Could not write the result file: {e}", "bad")
            return
        self.lbl_run.set(f"Result attached to the recording as {p.name}. It now "
                         "shows in the Library and can be curated in Review.", "ok")

    def shutdown(self) -> None:
        W.stop_jobs(self)
        self.run_job.wait(5000)
