"""Review tab: watch the footage, check every call, and correct it.

Automated HTR scoring is not a replacement for a human looking at the video; it
is a way of making that look affordable. A 70-minute trial holds a million
frames, and the detector reduces it to a few dozen candidates. This tab puts
each candidate on screen with its signal, and lets the scorer accept it, reject
it, or add one the detector missed.

That curation is saved with the recording, so the exported counts can be the
human-confirmed ones, and so the disagreements are available later as labelled
training data if a pose model gets trained.

Frames are read on demand, never loaded wholesale: the file may be 25 GB.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

from .. import analysis, library, paths, report, theme
from . import widgets as W

log = logging.getLogger(__name__)

BURST_COLUMNS = ["#", "Start (s)", "Dur (ms)", "Hz", "Half-cyc", "Detector",
                 "Source", "Accepted"]


class ReviewTab(QtWidgets.QWidget):
    statusMessage = QtCore.pyqtSignal(str)

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.pal = theme.palette(ctx.cfg.get("ui.theme", "light"))
        self.reader: W.FrameReader | None = None
        self.bursts: list[analysis.Burst] = []
        self.frame_index = 0
        self._syncing = False
        self._build()

        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._advance)
        self.ctx.analysisChanged.connect(self._on_analysis_changed)

    # ------------------------------------------------------------------
    def _build(self) -> None:
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)

        bar = QtWidgets.QHBoxLayout()
        self.lbl_file = QtWidgets.QLabel("No recording loaded.")
        self.lbl_file.setWordWrap(True)
        bar.addWidget(self.lbl_file, 1)
        self.btn_reload = W.button("Reload video")
        self.btn_reload.clicked.connect(lambda: self.load_entry(self.ctx.entry))
        bar.addWidget(self.btn_reload)
        v.addLayout(bar)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self._left())
        split.addWidget(self._right())
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 3)
        v.addWidget(split, 1)

        self.lbl_status = W.StatusLabel("", "info")
        v.addWidget(self.lbl_status)

    def _left(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(0, 0, 6, 0)

        self.view = W.FrameView()
        self.view.set_palette_tokens(self.pal)
        v.addWidget(self.view, 1)

        nav = QtWidgets.QHBoxLayout()
        self.btn_prev_burst = W.button("◀◀ burst")
        self.btn_step_back = W.button("◀ frame")
        self.btn_play = W.button("Play", "primary")
        self.btn_step_fwd = W.button("frame ▶")
        self.btn_next_burst = W.button("burst ▶▶")
        for b in (self.btn_prev_burst, self.btn_step_back, self.btn_play,
                  self.btn_step_fwd, self.btn_next_burst):
            nav.addWidget(b)
        nav.addWidget(QtWidgets.QLabel("speed"))
        self.cb_speed = QtWidgets.QComboBox()
        for label, mult in (("1/16x", 1 / 16), ("1/8x", 1 / 8), ("1/4x", 0.25),
                            ("1/2x", 0.5), ("1x", 1.0)):
            self.cb_speed.addItem(label, mult)
        self.cb_speed.setCurrentIndex(1)
        self.cb_speed.setToolTip(
            "A 90 Hz oscillation is invisible at real speed. 1/8x is about right "
            "for eyeballing a twitch.")
        nav.addWidget(self.cb_speed)
        nav.addStretch(1)
        v.addLayout(nav)

        srow = QtWidgets.QHBoxLayout()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        self.lbl_time = QtWidgets.QLabel("t = 0.000 s   frame 0")
        self.lbl_time.setMinimumWidth(190)
        srow.addWidget(self.slider, 1)
        srow.addWidget(self.lbl_time)
        v.addLayout(srow)

        self.chk_overlay = QtWidgets.QCheckBox("Show tracking overlay")
        self.chk_overlay.setChecked(bool(self.ctx.cfg.get("review.overlay_tracking", True)))
        self.chk_overlay.setToolTip(
            "Draws the fitted body centroid and the head direction from the last "
            "analysis, so you can see whether tracking actually followed the mouse.")
        self.chk_overlay.toggled.connect(lambda *_: self._show_frame(self.frame_index))
        v.addWidget(self.chk_overlay)

        self.btn_export_frame = W.button("Save this frame…")
        self.btn_export_frame.clicked.connect(self._save_frame)
        v.addWidget(self.btn_export_frame)

        self.btn_prev_burst.clicked.connect(lambda: self._jump_burst(-1))
        self.btn_next_burst.clicked.connect(lambda: self._jump_burst(+1))
        self.btn_step_back.clicked.connect(lambda: self._seek(self.frame_index - 1))
        self.btn_step_fwd.clicked.connect(lambda: self._seek(self.frame_index + 1))
        self.btn_play.clicked.connect(self._toggle_play)
        return box

    def _right(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(6, 0, 0, 0)

        theme.apply_pyqtgraph(self.pal, bool(self.ctx.cfg.get("ui.plot_antialias", True)))
        self.plot_angle = pg.PlotWidget(title="Head angle (deg)")
        self.plot_env = pg.PlotWidget(title="Band-pass envelope (deg/s)")
        for p in (self.plot_angle, self.plot_env):
            p.showGrid(x=True, y=True, alpha=0.18)
            p.setLabel("bottom", "time", units="s")
            p.setMinimumHeight(130)
            v.addWidget(p)
        self.plot_env.setXLink(self.plot_angle)

        self.curve_angle = self.plot_angle.plot(pen=pg.mkPen(self.pal.s1, width=1))
        self.curve_env = self.plot_env.plot(pen=pg.mkPen(self.pal.s2, width=1))
        self.line_thr = pg.InfiniteLine(angle=0, movable=False,
                                        pen=pg.mkPen(self.pal.critical, width=2,
                                                     style=QtCore.Qt.DashLine))
        self.plot_env.addItem(self.line_thr)
        self.cursors = []
        for p in (self.plot_angle, self.plot_env):
            c = pg.InfiniteLine(angle=90, movable=True,
                                pen=pg.mkPen(self.pal.ink, width=1))
            c.sigPositionChanged.connect(self._on_cursor)
            p.addItem(c)
            self.cursors.append(c)
        self.burst_regions: list[pg.LinearRegionItem] = []

        self.table = QtWidgets.QTableWidget(0, len(BURST_COLUMNS))
        self.table.setHorizontalHeaderLabels(BURST_COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        self.table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_row)
        v.addWidget(self.table, 1)

        act = QtWidgets.QGridLayout()
        self.btn_accept = W.button("Accept")
        self.btn_reject = W.button("Reject")
        self.btn_toggle = W.button("Toggle")
        self.btn_add = W.button("Add burst at cursor")
        self.sp_add_ms = QtWidgets.QDoubleSpinBox()
        self.sp_add_ms.setRange(20, 2000)
        self.sp_add_ms.setValue(120)
        self.sp_add_ms.setSuffix(" ms")
        self.btn_delete = W.button("Delete manual burst")
        self.btn_show_rejected = QtWidgets.QCheckBox("Show rejected candidates")
        self.btn_show_rejected.setToolTip(
            "Lists, with its reason, every candidate that crossed the "
            "threshold and was then refused by one of the gates - duration, "
            "in-burst frequency, half-cycle count or tracking quality.\n\n"
            "They are always marked on the plots with a dashed outline, so a "
            "peak that was considered and refused never looks the same as one "
            "that was never noticed.")
        self.btn_save = W.button("Save curated result", "primary")
        self.btn_export = W.button("Export curated CSV")
        act.addWidget(self.btn_accept, 0, 0)
        act.addWidget(self.btn_reject, 0, 1)
        act.addWidget(self.btn_toggle, 0, 2)
        act.addWidget(self.btn_add, 1, 0)
        act.addWidget(self.sp_add_ms, 1, 1)
        act.addWidget(self.btn_delete, 1, 2)
        act.addWidget(self.btn_show_rejected, 2, 0, 1, 3)
        act.addWidget(self.btn_save, 3, 0, 1, 2)
        act.addWidget(self.btn_export, 3, 2)
        v.addLayout(act)

        self.btn_accept.clicked.connect(lambda: self._set_accepted(True))
        self.btn_reject.clicked.connect(lambda: self._set_accepted(False))
        self.btn_toggle.clicked.connect(lambda: self._set_accepted(None))
        self.btn_add.clicked.connect(self._add_manual)
        self.btn_delete.clicked.connect(self._delete_manual)
        self.btn_show_rejected.toggled.connect(lambda *_: self._fill_table())
        self.btn_save.clicked.connect(self._save_curated)
        self.btn_export.clicked.connect(self._export_curated)
        return box

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_entry(self, entry: library.Entry | None) -> None:
        if entry is None:
            return
        self.ctx.entry = entry
        target = entry.analysis_target
        if self.reader is not None:
            self.reader.close()
        self.reader = W.FrameReader(target, int(self.ctx.cfg.get("review.frame_cache", 240)))
        if self.reader.error:
            self.lbl_status.set(self.reader.error, "bad")
            return
        fps = entry.fps_for_analysis() or self.reader.fps or 240.0
        self.fps = fps
        n = self.reader.n_frames or int(entry.duration_s * fps)
        self.slider.setRange(0, max(0, n - 1))
        self.slider.setEnabled(True)
        self.lbl_file.setText(
            f"<b>{target.name}</b> · {self.reader.width}x{self.reader.height} "
            f"· {n:,} frames at {fps:.2f} fps · {entry.duration_hms}"
            + (f" · segment 1 of {len(entry.segments)}" if entry.is_session else ""))

        # Prefer the in-memory analysis if it is for this file; else the saved one.
        sig = self.ctx.signals
        if sig is not None and Path(sig.source_path) == target:
            self._load_from_memory()
        elif entry.result:
            self._load_from_saved(entry.result)
        else:
            self.bursts = []
            self.curve_angle.clear()
            self.curve_env.clear()
            self._fill_table()
            self.lbl_status.set(
                "No analysis for this recording yet. Run one on the Analysis tab, "
                "then come back to check the calls.", "info")
        self._seek(0)

    def _on_analysis_changed(self) -> None:
        sig = self.ctx.signals
        if sig is None:
            return
        if self.reader is None or Path(sig.source_path) != Path(self.reader.path):
            entry = self.ctx.entry
            if entry is not None:
                self.load_entry(entry)
                return
        self._load_from_memory()

    def _load_from_memory(self) -> None:
        sig, det = self.ctx.signals, self.ctx.detection
        if sig is None or det is None:
            return
        self.bursts = [analysis.Burst.from_dict(b.to_dict()) for b in det.bursts]
        self.rejected = [analysis.Burst.from_dict(b.to_dict()) for b in det.rejected]
        self._plot_traces(sig.t, sig.angle, det.envelope, det.threshold,
                          det.spec_t, det.spec_ratio)
        self._fill_table()
        n_rej = len(getattr(self, "rejected", []))
        self.lbl_status.set(
            f"{len(self.bursts)} accepted candidate(s) from the last analysis"
            + (f", and {n_rej} the detector examined and refused (dashed "
               "outline on the plots; tick 'Show rejected candidates' to list "
               "them with the reason)." if n_rej else ".")
            + "  Step through them and accept or reject each one.", "info")

    def _load_from_saved(self, doc: dict) -> None:
        self.bursts = [analysis.Burst.from_dict(b) for b in (doc.get("bursts") or [])]
        self.rejected = [analysis.Burst.from_dict(b) for b in (doc.get("rejected") or [])]
        self.curve_angle.clear()
        self.curve_env.clear()
        self.line_thr.setValue(float(doc.get("threshold", 0) or 0))
        self._fill_table()
        self.lbl_status.set(
            f"Loaded a saved analysis: {len(self.bursts)} candidate(s), "
            f"{'curated' if doc.get('curated') else 'not yet curated'}. "
            "Traces are not stored with the result, so re-run the analysis if you "
            "want the signal plots.", "info")

    def _plot_traces(self, t, angle, envelope, threshold, spec_t, spec_ratio) -> None:
        # Decimate for drawing: a million points would make scrubbing unusable,
        # and min/max blocks keep the oscillation visible rather than aliasing it
        # away.
        td, ad = report._decimate(t, angle, 8000)
        self.curve_angle.setData(td, ad)
        if envelope is not None:
            te, ee = report._decimate(t, envelope, 8000)
            self.curve_env.setData(te, ee)
            self.plot_env.setTitle("Band-pass envelope (deg/s)")
        elif spec_ratio is not None and spec_t is not None:
            self.curve_env.setData(spec_t, spec_ratio)
            self.plot_env.setTitle("Motion-energy band-power ratio")
        self.line_thr.setValue(float(threshold))
        self._draw_regions()

    def _draw_regions(self) -> None:
        for r in self.burst_regions:
            for p in (self.plot_angle, self.plot_env):
                try:
                    p.removeItem(r)
                except Exception:  # pragma: no cover
                    pass
        self.burst_regions = []
        for b in self.bursts:
            colour = self.pal.s3 if b.accepted else self.pal.ink_muted
            for p in (self.plot_angle, self.plot_env):
                region = pg.LinearRegionItem(
                    values=(b.start_s, b.end_s), movable=False,
                    brush=pg.mkBrush(colour + "4D"),
                    pen=pg.mkPen(colour, width=1))
                region.setZValue(-10)
                region.setToolTip("accepted" if b.accepted else "rejected by hand")
                p.addItem(region)
                self.burst_regions.append(region)

        # Candidates the detector looked at and refused. Drawn in outline only,
        # because the alternative -- leaving them off -- makes a peak that was
        # examined and rejected look identical to one that was never noticed.
        # That is the single most confusing thing the graph can do: on one clip
        # the tallest envelope peak in the whole recording, at twice the
        # threshold, carried no mark at all, and the only way to discover it had
        # been considered was to read the rejected list in the text panel.
        for b in getattr(self, "rejected", []):
            colour = self.pal.serious
            for p in (self.plot_angle, self.plot_env):
                region = pg.LinearRegionItem(
                    values=(b.start_s, b.end_s), movable=False,
                    brush=pg.mkBrush(colour + "1A"),
                    pen=pg.mkPen(colour, width=1, style=QtCore.Qt.DashLine))
                region.setZValue(-11)
                region.setToolTip(b.note or "rejected by the detector")
                p.addItem(region)
                self.burst_regions.append(region)

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------
    def _rows(self) -> list[analysis.Burst]:
        rows = list(self.bursts)
        if self.btn_show_rejected.isChecked():
            rows += getattr(self, "rejected", [])
        return sorted(rows, key=lambda b: b.start_s)

    def _fill_table(self) -> None:
        rows = self._rows()
        self.table.setRowCount(len(rows))
        for r, b in enumerate(rows):
            in_main = b in self.bursts
            vals = [str(r + 1), f"{b.start_s:.3f}", f"{b.duration_ms:.0f}",
                    f"{b.dominant_hz:.1f}", str(b.half_cycles), b.detector,
                    "manual" if b.manual else ("candidate" if in_main else "rejected"),
                    "yes" if (in_main and b.accepted) else "no"]
            for c, text in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(text)
                if c == 0:
                    item.setData(QtCore.Qt.UserRole, r)
                if b.note:
                    item.setToolTip(b.note)
                self.table.setItem(r, c, item)
        n_acc = sum(1 for b in self.bursts if b.accepted)
        self.table.setToolTip(f"{n_acc} accepted of {len(self.bursts)} candidates")
        self._draw_regions()

    def _selected_burst(self) -> analysis.Burst | None:
        rows = self._rows()
        sel = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not sel:
            return None
        i = sel[0].row()
        return rows[i] if 0 <= i < len(rows) else None

    def _on_row(self) -> None:
        b = self._selected_burst()
        if b is None or not hasattr(self, "fps"):
            return
        self._seek(int(round(b.center_s * self.fps)))

    def _set_accepted(self, value: bool | None) -> None:
        b = self._selected_burst()
        if b is None:
            return
        if b not in self.bursts:
            # Promoting a rejected candidate is an explicit scoring decision.
            self.rejected = [x for x in getattr(self, "rejected", []) if x is not b]
            b.note = (b.note + " | promoted by reviewer").strip(" |")
            b.accepted = True
            self.bursts.append(b)
        else:
            b.accepted = (not b.accepted) if value is None else value
        self._fill_table()
        n = sum(1 for x in self.bursts if x.accepted)
        self.lbl_status.set(f"{n} accepted of {len(self.bursts)} candidate(s). "
                            "Remember to save the curated result.", "warn")

    def _add_manual(self) -> None:
        if not hasattr(self, "fps"):
            return
        t = self.frame_index / self.fps
        dur = self.sp_add_ms.value() / 1000.0
        b = analysis.Burst(
            start_s=max(0.0, t - dur / 2), end_s=t + dur / 2,
            start_frame=int(max(0, (t - dur / 2) * self.fps)),
            end_frame=int((t + dur / 2) * self.fps),
            duration_ms=dur * 1000.0, detector="manual", accepted=True,
            manual=True, note="added by reviewer")
        self.bursts.append(b)
        self._fill_table()
        self.lbl_status.set(f"Added a manual burst at {t:.3f} s. Save to keep it.",
                            "warn")

    def _delete_manual(self) -> None:
        b = self._selected_burst()
        if b is None:
            return
        if not b.manual:
            self.lbl_status.set(
                "Only reviewer-added bursts can be deleted. Reject a detected one "
                "instead - rejecting keeps the record of the disagreement.", "warn")
            return
        self.bursts = [x for x in self.bursts if x is not b]
        self._fill_table()
        self.lbl_status.set("Manual burst removed.", "warn")

    # ------------------------------------------------------------------
    # Playback and seeking
    # ------------------------------------------------------------------
    def _toggle_play(self) -> None:
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.btn_play.setText("Play")
            return
        if self.reader is None:
            return
        mult = float(self.cb_speed.currentData() or 0.125)
        interval = max(10, int(1000.0 / max(self.fps * mult, 1.0)))
        self.play_timer.setInterval(interval)
        self.play_timer.start()
        self.btn_play.setText("Pause")

    def _advance(self) -> None:
        if self.reader is None or self.frame_index >= self.slider.maximum():
            self.play_timer.stop()
            self.btn_play.setText("Play")
            return
        self._seek(self.frame_index + 1)

    def _seek(self, index: int) -> None:
        if self.reader is None:
            return
        index = int(np.clip(index, 0, max(self.slider.maximum(), 0)))
        self.frame_index = index
        self._syncing = True
        self.slider.setValue(index)
        for c in self.cursors:
            c.setValue(index / max(self.fps, 1e-9))
        self._syncing = False
        self._show_frame(index)

    def _on_slider(self, value: int) -> None:
        if not self._syncing:
            self._seek(value)

    def _on_cursor(self, line) -> None:
        if self._syncing or not hasattr(self, "fps"):
            return
        self._seek(int(round(float(line.value()) * self.fps)))

    def _show_frame(self, index: int) -> None:
        if self.reader is None:
            return
        gray = self.reader.get(index)
        if gray is None:
            self.lbl_status.set(f"Could not decode frame {index}.", "warn")
            return
        t = index / max(self.fps, 1e-9)
        overlays: dict = {}
        sig = self.ctx.signals
        if (self.chk_overlay.isChecked() and sig is not None
                and Path(sig.source_path) == Path(self.reader.path)
                and index < sig.n):
            overlays["centroid"] = (float(sig.cx[index]), float(sig.cy[index]))
            overlays["head"] = (float(sig.hx[index]), float(sig.hy[index]))
        in_burst = any(b.start_s <= t <= b.end_s for b in self.bursts if b.accepted)
        overlays["burst"] = in_burst
        caption = f"t={t:.3f}s  frame {index}" + ("   TWITCH" if in_burst else "")
        self.view.show_frame(gray, overlays, caption)
        self.lbl_time.setText(f"t = {t:.3f} s   frame {index:,}"
                              + ("   [in burst]" if in_burst else ""))
        if not self.play_timer.isActive():
            self.reader.prefetch(index + 1, 12)

    def _jump_burst(self, direction: int) -> None:
        if not self.bursts or not hasattr(self, "fps"):
            return
        t = self.frame_index / self.fps
        ordered = sorted(self.bursts, key=lambda b: b.center_s)
        if direction > 0:
            nxt = next((b for b in ordered if b.center_s > t + 1e-6), None)
        else:
            nxt = next((b for b in reversed(ordered) if b.center_s < t - 1e-6), None)
        if nxt is None:
            self.lbl_status.set("No further burst in that direction.", "info")
            return
        self._seek(int(round(nxt.center_s * self.fps)))
        for r, b in enumerate(self._rows()):
            if b is nxt:
                self.table.selectRow(r)
                break

    def _save_frame(self) -> None:
        if self.reader is None:
            return
        default = paths.EXPORTS_DIR / f"{Path(self.reader.path).stem}_f{self.frame_index}.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save frame", str(default), "PNG (*.png)")
        if path and self.view.save_png(Path(path)):
            self.lbl_status.set(f"Frame saved to {path}", "ok")

    # ------------------------------------------------------------------
    # Saving curation
    # ------------------------------------------------------------------
    def _save_curated(self) -> None:
        entry = self.ctx.entry
        if entry is None:
            return
        base = dict(entry.result) if entry.result else {}
        if self.ctx.detection is not None and self.ctx.signals is not None \
                and Path(self.ctx.signals.source_path) == Path(self.reader.path
                                                               if self.reader else ""):
            det, sig = self.ctx.detection, self.ctx.signals
            base.update({
                "htr_studio_result_version": 1,
                "video": str(sig.source_path),
                "signals": sig.summary(),
                "detector": det.detector,
                "threshold": det.threshold,
                "diagnostics": det.diagnostics,
                "warnings": det.warnings,
            })
        t0 = max(entry.t0_offset_s, 0.0)
        bin_s = float(self.ctx.cfg.get("report.bin_seconds", 60))
        duration = (self.ctx.signals.duration_s if self.ctx.signals
                    else entry.duration_s)
        accepted = [b for b in self.bursts if b.accepted]
        base["bursts"] = [b.to_dict() for b in sorted(self.bursts,
                                                      key=lambda x: x.start_s)]
        base["rejected"] = [b.to_dict() for b in getattr(self, "rejected", [])]
        base["bins"] = analysis.bin_counts(accepted, duration, bin_s, t0)
        base["n_accepted"] = len(accepted)
        base["curated"] = True
        base["curated_by"] = str(self.ctx.cfg.get("recording.operator", "")) or ""
        try:
            p = library.save_analysis(entry, base)
        except OSError as e:
            self.lbl_status.set(f"Could not save: {e}", "bad")
            return
        self.lbl_status.set(
            f"Curated result saved to {p.name}: {len(accepted)} confirmed twitch(es) "
            f"of {len(self.bursts)} candidate(s).", "ok")
        self.statusMessage.emit(f"Curation saved for {entry.name}")

    def _export_curated(self) -> None:
        entry = self.ctx.entry
        if entry is None or self.ctx.signals is None:
            W.warn_box(self, "Need the traces",
                       "Re-run the analysis for this recording first; the CSV "
                       "export includes the signal summary.")
            return
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Where to write the curated report", str(paths.EXPORTS_DIR))
        if not d:
            return
        det = self.ctx.detection
        # Export what the reviewer decided, not what the detector first said.
        curated = analysis.DetectionResult(
            bursts=self.bursts, detector=(det.detector if det else "curated") + "+curated",
            threshold=det.threshold if det else 0.0,
            fps=self.ctx.signals.fps, duration_s=self.ctx.signals.duration_s,
            diagnostics=dict(det.diagnostics) if det else {},
            warnings=list(det.warnings) if det else [],
            envelope=det.envelope if det else None,
            filtered=det.filtered if det else None,
            rejected=getattr(self, "rejected", []))
        rep = report.write_trial_csv(
            Path(self.ctx.signals.source_path), Path(d), self.ctx.signals, curated,
            bin_s=float(self.ctx.cfg.get("report.bin_seconds", 60)),
            t0_s=max(entry.t0_offset_s, 0.0), meta=entry.trial)
        self.lbl_status.set(f"Curated report written to {rep.combined_csv}", "ok")

    def shutdown(self) -> None:
        self.play_timer.stop()
        if self.reader is not None:
            self.reader.close()
