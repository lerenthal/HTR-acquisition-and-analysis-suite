"""Two dialogs over :mod:`htr_studio.advisor`.

:class:`SuggestDialog` shows what the advisor would change for one file --
setting, current value, suggested value, and the reasoning -- and changes
nothing until Apply is pressed. Showing the reasoning is not politeness: a
folded band-pass band looks like a mistake to anyone who has not worked
through the aliasing, and a setting applied without a reason attached is a
setting nobody will dare touch later.

:class:`ControlCalibrationDialog` runs the threshold sweep over an untreated
recording and shows the false-positive curve it traced, so the chosen
threshold can be seen in context rather than taken on trust.

Both run their work in a background job, because both are measured in tens of
seconds on a real recording.
"""
from __future__ import annotations

import logging

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

from .. import advisor, analysis, theme
from . import widgets as W

log = logging.getLogger(__name__)

_LEVEL_MARK = {"bad": "✕", "warn": "⚠", "info": "•", "ok": "✓"}


def _suggest(video, cfg, ffprobe, samples, progress=None):
    """Module-level so it can be pickled into a worker thread cleanly."""
    return advisor.suggest_settings(video, cfg, ffprobe=ffprobe,
                                    samples=samples, progress=progress)


def _calibrate(video, cfg, ffprobe, fps_override, progress=None,
               should_cancel=None):
    return advisor.calibrate_from_control(
        video, cfg, ffprobe=ffprobe, fps_override=fps_override,
        progress=progress, should_cancel=should_cancel)


class SuggestDialog(QtWidgets.QDialog):
    """What the advisor would change for this file, and why."""

    COLUMNS = ("Setting", "Current", "Suggested", "Why")

    def __init__(self, ctx, video, cfg: analysis.AnalysisConfig, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.video = video
        self.base_cfg = cfg
        self.advice: advisor.Advice | None = None
        self.result_cfg: analysis.AnalysisConfig | None = None
        self.pal = theme.palette(ctx.cfg.get("ui.theme", "light"))
        self.setWindowTitle("Settings for this recording")
        self.resize(980, 620)

        self.job = W.JobRunner("suggest-settings", self)
        self.job.finished.connect(self._done)
        self.job.failed.connect(self._failed)
        self.job.progressed.connect(self._progress)

        self._build()
        W.no_wheel(self)
        QtCore.QTimer.singleShot(0, self._start)

    def _build(self) -> None:
        v = QtWidgets.QVBoxLayout(self)
        self.lbl_file = QtWidgets.QLabel("")
        self.lbl_file.setWordWrap(True)
        v.addWidget(self.lbl_file)

        self.notes = QtWidgets.QVBoxLayout()
        v.addLayout(self.notes)

        self.table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(True)
        hh = self.table.horizontalHeader()
        for i in range(3):
            hh.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        v.addWidget(self.table, 1)

        self.lbl_cal = QtWidgets.QLabel("")
        self.lbl_cal.setObjectName("hint")
        self.lbl_cal.setWordWrap(True)
        v.addWidget(self.lbl_cal)

        self.progress = QtWidgets.QProgressBar()
        v.addWidget(self.progress)
        self.lbl_status = W.StatusLabel("Inspecting the recording…", "info")
        v.addWidget(self.lbl_status)

        row = QtWidgets.QHBoxLayout()
        self.btn_apply = W.button("Apply these settings", "primary")
        self.btn_apply.setEnabled(False)
        self.btn_close = W.button("Close without changing anything")
        row.addStretch(1)
        row.addWidget(self.btn_close)
        row.addWidget(self.btn_apply)
        v.addLayout(row)
        self.btn_apply.clicked.connect(self._apply)
        self.btn_close.clicked.connect(self.reject)

    # ------------------------------------------------------------------
    def _start(self) -> None:
        self.progress.setRange(0, 3)
        self.job.start(_suggest, self.video, self.base_cfg,
                       str(self.ctx.cfg.get("paths.ffprobe", "")),
                       int(self.ctx.cfg.get("analysis_ui.advisor_samples", 24)))

    def _progress(self, done: int, total: int, msg: str) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        if msg:
            self.lbl_status.set(msg, "info")

    def _failed(self, msg: str, _tb: str) -> None:
        self.progress.setValue(0)
        self.lbl_status.set(f"Could not inspect the recording: {msg}", "bad")

    def _done(self, adv: advisor.Advice) -> None:
        self.advice = adv
        self.progress.setValue(self.progress.maximum())
        self.lbl_file.setText(
            f"<b>{adv.source}</b><br>{adv.width}x{adv.height}, "
            f"{adv.fps:.2f} fps "
            + ("(counted)" if adv.fps_exact else "(declared by the container)")
            + f", {adv.duration_s:.1f} s")
        if adv.calibration:
            self.lbl_cal.setText("Calibration on this file: " + adv.calibration)

        while self.notes.count():
            w = self.notes.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        for level, text in adv.notes:
            lbl = W.StatusLabel(text, level)
            lbl.setWordWrap(True)
            self.notes.addWidget(lbl)

        pending = adv.pending
        self.table.setRowCount(len(pending))
        for r, c in enumerate(pending):
            mark = _LEVEL_MARK.get(c.level, "•")
            for col, text in enumerate((f"{mark} {c.label}",
                                        _fmt(c.current), _fmt(c.suggested),
                                        c.why)):
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(c.why)
                self.table.setItem(r, col, item)
        self.table.resizeRowsToContents()

        if not pending:
            self.lbl_status.set(
                "Nothing to change: the current settings already suit this "
                "recording." + ("" if adv.usable else
                                " The notes above still apply."),
                "ok" if adv.usable else "warn")
            self.btn_apply.setEnabled(False)
            self.btn_close.setText("Close")
            return
        self.btn_apply.setEnabled(True)
        self.lbl_status.set(
            f"{len(pending)} setting(s) would change. Nothing has been "
            "applied yet; the reasons are recorded in the result sidecar if "
            "you apply them.",
            "warn" if any(c.level in ("warn", "bad") for c in pending) else "info")

    def _apply(self) -> None:
        if self.advice is None:
            return
        self.result_cfg = self.advice.apply_to(self.base_cfg)
        self.accept()

    def closeEvent(self, ev) -> None:  # noqa: N802 - Qt naming
        W.stop_jobs(self, 2000)
        super().closeEvent(ev)

    def shutdown(self) -> None:
        W.stop_jobs(self, 2000)


def _fmt(v) -> str:
    if isinstance(v, (list, tuple)):
        return ", ".join(f"{float(x):g}" for x in v)
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


class ControlCalibrationDialog(QtWidgets.QDialog):
    """Fit the detection threshold to an untreated recording."""

    def __init__(self, ctx, video, cfg: analysis.AnalysisConfig,
                 fps_override: float = 0.0, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.video = video
        self.base_cfg = cfg
        self.fps_override = fps_override
        self.cal: advisor.ControlCalibration | None = None
        self.result_cfg: analysis.AnalysisConfig | None = None
        self.pal = theme.palette(ctx.cfg.get("ui.theme", "light"))
        self.setWindowTitle("Calibrate from a control recording")
        self.resize(900, 680)

        self.job = W.JobRunner("control-calibration", self)
        self.job.finished.connect(self._done)
        self.job.failed.connect(self._failed)
        self.job.progressed.connect(self._progress)

        self._build()
        W.no_wheel(self)
        QtCore.QTimer.singleShot(0, self._start)

    def _build(self) -> None:
        v = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "<b>What this does.</b> An untreated animal should produce no "
            "head twitches, so every burst the detector calls in this "
            "recording is a false positive. Sweeping the threshold over it "
            "measures this rig's noise floor directly - this arena, this "
            "lighting, this camera height, this cohort's grooming - which is "
            "a better basis for a threshold than any constant chosen "
            "elsewhere.<br><br>"
            "The threshold picked is one step below the lowest setting at "
            "which the control falls silent. Sitting exactly at the silent "
            "point would mean tuning until the detector can just barely not "
            "see the loudest non-twitch here, which is a hair from not "
            "seeing a real twitch either.<br><br>"
            "<b>The limit.</b> One control animal is one control animal, and "
            "the result applies only to recordings made in the same session "
            "with the same exposure, frame rate, zoom and lighting.")
        intro.setWordWrap(True)
        v.addWidget(intro)

        self.lbl_file = QtWidgets.QLabel("")
        self.lbl_file.setWordWrap(True)
        v.addWidget(self.lbl_file)

        theme.apply_pyqtgraph(self.pal, bool(self.ctx.cfg.get("ui.plot_antialias", True)))
        self.plot = pg.PlotWidget(title="False positives in the control, by threshold")
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.setLabel("bottom", "threshold k (multiples of baseline MAD)")
        self.plot.setLabel("left", "bursts called")
        self.plot.setMinimumHeight(220)
        v.addWidget(self.plot, 1)
        self.curve = self.plot.plot(
            pen=pg.mkPen(self.pal.s1, width=2),
            symbol="o", symbolSize=6, symbolBrush=self.pal.s1)
        self.line_chosen = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(self.pal.s3, width=2, style=QtCore.Qt.DashLine),
            label="chosen", labelOpts={"position": 0.9})
        self.plot.addItem(self.line_chosen)
        self.line_chosen.setVisible(False)

        self.txt = QtWidgets.QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setMaximumHeight(130)
        v.addWidget(self.txt)

        self.progress = QtWidgets.QProgressBar()
        v.addWidget(self.progress)
        self.lbl_status = W.StatusLabel("Extracting the angle trace…", "info")
        self.lbl_status.setWordWrap(True)
        v.addWidget(self.lbl_status)

        row = QtWidgets.QHBoxLayout()
        self.in_profile = QtWidgets.QLineEdit()
        self.in_profile.setPlaceholderText("session profile name, e.g. 2026-11-05 cohort B")
        self.btn_save = W.button("Save as a session profile")
        self.btn_save.setEnabled(False)
        self.btn_apply = W.button("Apply this threshold", "primary")
        self.btn_apply.setEnabled(False)
        self.btn_cancel = W.button("Cancel")
        self.btn_close = W.button("Close without changing anything")
        row.addWidget(self.in_profile, 1)
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.btn_close)
        row.addWidget(self.btn_apply)
        v.addLayout(row)
        self.job._cancel_button = self.btn_cancel
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.job.cancel)
        self.btn_save.clicked.connect(self._save_profile)
        self.btn_apply.clicked.connect(self._apply)
        self.btn_close.clicked.connect(self.reject)

    # ------------------------------------------------------------------
    def _start(self) -> None:
        self.lbl_file.setText(f"<b>Control recording:</b> {self.video}")
        self.btn_cancel.setEnabled(True)
        self.job.start(_calibrate, self.video, self.base_cfg,
                       str(self.ctx.cfg.get("paths.ffprobe", "")),
                       self.fps_override)

    def _progress(self, done: int, total: int, msg: str) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        if msg:
            self.lbl_status.set(msg, "info")

    def _failed(self, msg: str, _tb: str) -> None:
        self.btn_cancel.setEnabled(False)
        self.lbl_status.set(f"Calibration failed: {msg}", "bad")

    def _done(self, cal: advisor.ControlCalibration) -> None:
        self.cal = cal
        self.btn_cancel.setEnabled(False)
        self.progress.setValue(self.progress.maximum())
        self.lbl_file.setText(
            f"<b>Control recording:</b> {cal.source}<br>"
            f"{cal.duration_s:.1f} s at {cal.fps:.2f} fps")
        if cal.sweep:
            ks = [k for k, _n in cal.sweep]
            ns = [n for _k, n in cal.sweep]
            self.curve.setData(ks, ns)
            self.txt.setPlainText(
                "threshold k :  " + "  ".join(f"{k:>5g}" for k in ks) + "\n"
                "bursts      :  " + "  ".join(f"{n:>5d}" for n in ns))
        if cal.ok:
            self.line_chosen.setValue(cal.chosen_k)
            self.line_chosen.setVisible(True)
            self.btn_apply.setEnabled(True)
            self.btn_save.setEnabled(True)
            self.btn_close.setText("Close without applying")
        self.lbl_status.set(
            cal.message,
            "ok" if (cal.ok and cal.margin >= 1.3) else
            "warn" if cal.ok else "bad")
        log.info("control calibration: %s", cal.message)

    def chosen_config(self) -> analysis.AnalysisConfig:
        if self.cal is None or not self.cal.ok:
            return self.base_cfg
        return analysis.AnalysisConfig.from_dict(self.cal.cfg)

    def _save_profile(self) -> None:
        if self.cal is None or not self.cal.ok:
            return
        name = self.in_profile.text().strip()
        if not name:
            W.warn_box(self, "Name the profile",
                       "Give the profile a name that identifies the session, "
                       "so it is obvious later which recordings it belongs "
                       "to - the date and the cohort, for instance.")
            return
        p = advisor.save_profile(name, self.chosen_config(), cal=self.cal)
        self.lbl_status.set(
            f"Saved as a session profile: {p.name}. It carries the control "
            "file, the sweep and the margin with it, so the counts it "
            "produces can be traced back to how it was fitted.", "ok")

    def _apply(self) -> None:
        self.result_cfg = self.chosen_config()
        self.accept()

    def closeEvent(self, ev) -> None:  # noqa: N802 - Qt naming
        W.stop_jobs(self, 3000)
        super().closeEvent(ev)

    def shutdown(self) -> None:
        W.stop_jobs(self, 3000)
