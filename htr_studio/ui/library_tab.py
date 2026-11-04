"""Library tab: every recording, its metadata, and what has been done to it.

The point of this tab is that a trial is not just a video file. It is a video
plus an animal ID, a group, a dose, a route, a T0 and a set of camera settings,
and the analysis is only interpretable with those attached. They live in a JSON
sidecar next to the video, so copying the folder copies the study.

Scanning never decodes video: ``ffprobe -count_frames`` on a 70-minute trial
takes minutes. The table shows the container's header values, and "Verify frame
rate" does the exact count on demand and caches it in the sidecar, because the
container frame rate for MJPEG-in-AVI is routinely wrong and every timing figure
downstream depends on getting it right.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from .. import library, paths, recorder, theme
from . import widgets as W

log = logging.getLogger(__name__)

COLUMNS = ["Name", "Animal", "Group", "Dose", "Route", "Recorded", "Duration",
           "Mode", "fps", "Size", "Status"]


class LibraryTab(QtWidgets.QWidget):
    openInReview = QtCore.pyqtSignal(object)    # library.Entry
    analyseRequested = QtCore.pyqtSignal(object)
    batchRequested = QtCore.pyqtSignal(list)
    statusMessage = QtCore.pyqtSignal(str)

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.pal = theme.palette(ctx.cfg.get("ui.theme", "light"))
        self.entries: list[library.Entry] = []
        self.worker: W.Worker | None = None
        self.scan_job = W.JobRunner("library-scan", self)
        self.scan_job.finished.connect(self._scan_done)
        self.scan_job.failed.connect(self._job_failed)
        self.verify_job = W.JobRunner("library-verify", self)
        self.verify_job.failed.connect(self._job_failed)
        self._build()
        self.scan_job._button = self.btn_rescan
        self.verify_job._button = self.btn_verify
        QtCore.QTimer.singleShot(0, self.rescan)

    # ------------------------------------------------------------------
    def _build(self) -> None:
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Library root"))
        self.in_root = QtWidgets.QLineEdit(str(self.ctx.cfg.recordings_root()))
        bar.addWidget(self.in_root, 1)
        self.btn_browse = W.button("…")
        self.btn_browse.setMaximumWidth(32)
        self.btn_rescan = W.button("Rescan", "primary")
        self.in_filter = QtWidgets.QLineEdit()
        self.in_filter.setPlaceholderText("filter by name, animal, group, dose…")
        self.in_filter.setClearButtonEnabled(True)
        for wdg in (self.btn_browse, self.btn_rescan):
            bar.addWidget(wdg)
        bar.addWidget(self.in_filter, 1)
        v.addLayout(bar)

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        self.table.setSelectionMode(QtWidgets.QTableWidget.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.table.doubleClicked.connect(lambda *_: self._open_review())
        split.addWidget(self.table)

        split.addWidget(self._detail_panel())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        v.addWidget(split, 1)

        self.lbl_summary = W.StatusLabel("", "info")
        v.addWidget(self.lbl_summary)

        self.btn_browse.clicked.connect(self._browse)
        self.btn_rescan.clicked.connect(self.rescan)
        self.in_filter.textChanged.connect(self._refill)

    def _detail_panel(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)

        meta_g = QtWidgets.QGroupBox("Trial metadata")
        form = QtWidgets.QFormLayout(meta_g)
        self.m_animal = QtWidgets.QLineEdit()
        self.m_group = QtWidgets.QLineEdit()
        self.m_treatment = QtWidgets.QLineEdit()
        self.m_dose = QtWidgets.QLineEdit()
        self.m_route = QtWidgets.QLineEdit()
        self.m_operator = QtWidgets.QLineEdit()
        self.m_notes = QtWidgets.QLineEdit()
        self.m_t0 = QtWidgets.QDoubleSpinBox()
        self.m_t0.setRange(-1, 100000)
        self.m_t0.setDecimals(1)
        self.m_t0.setSuffix(" s")
        self.m_t0.setSpecialValueText("not marked")
        self.m_baseline = QtWidgets.QDoubleSpinBox()
        self.m_baseline.setRange(0, 100000)
        self.m_baseline.setDecimals(1)
        self.m_baseline.setSuffix(" s")
        for lab, wdg in (("Animal ID", self.m_animal), ("Group", self.m_group),
                         ("Treatment", self.m_treatment), ("Dose", self.m_dose),
                         ("Route", self.m_route), ("Operator", self.m_operator),
                         ("Notes", self.m_notes)):
            form.addRow(lab, wdg)
        W.form_row(form, "T0 offset", self.m_t0,
                   "Seconds from the start of the recording to dosing. Everything "
                   "before it is the baseline the detection threshold comes from.")
        W.form_row(form, "Planned baseline", self.m_baseline,
                   "Used when no T0 was marked.")
        self.btn_save_meta = W.button("Save metadata", "primary")
        self.btn_save_meta.clicked.connect(self._save_meta)
        form.addRow(self.btn_save_meta)
        h.addWidget(meta_g, 1)

        info_g = QtWidgets.QGroupBox("File and analysis")
        iv = QtWidgets.QVBoxLayout(info_g)
        self.txt_info = QtWidgets.QPlainTextEdit()
        self.txt_info.setReadOnly(True)
        self.txt_info.setMaximumHeight(210)
        iv.addWidget(self.txt_info)

        grid = QtWidgets.QGridLayout()
        self.btn_review = W.button("Open in Review", "primary")
        self.btn_analyse = W.button("Analyse")
        self.btn_batch = W.button("Add selection to Batch")
        self.btn_verify = W.button(
            "Verify frame rate",
            tooltip="Decodes the whole file and counts frames, which is the only "
                    "trustworthy frame rate for MJPEG in AVI. Slow on a long trial; "
                    "the result is cached in the sidecar.")
        self.btn_ingest = W.button(
            "Inspect / archive…",
            tooltip="What this file actually is - codec, measured frame rate, "
                    "whether every frame is coded independently - and the "
                    "conversions that shrink it without destroying the "
                    "oscillation. Use it on any footage that did not come from "
                    "this tool's own Acquisition tab.")
        self.btn_reveal = W.button("Show in Explorer")
        self.btn_trash = W.button("Move to recycle bin", "danger")
        for i, b in enumerate((self.btn_review, self.btn_analyse, self.btn_batch,
                               self.btn_verify, self.btn_ingest, self.btn_reveal,
                               self.btn_trash)):
            grid.addWidget(b, i // 2, i % 2)
        iv.addLayout(grid)
        self.lbl_detail = W.StatusLabel("", "info")
        iv.addWidget(self.lbl_detail)
        h.addWidget(info_g, 1)

        self.btn_review.clicked.connect(self._open_review)
        self.btn_analyse.clicked.connect(self._analyse)
        self.btn_batch.clicked.connect(self._batch)
        self.btn_verify.clicked.connect(self._verify)
        self.btn_ingest.clicked.connect(self._ingest)
        self.btn_reveal.clicked.connect(self._reveal)
        self.btn_trash.clicked.connect(self._trash)
        self._set_actions_enabled(False)
        return box

    # ------------------------------------------------------------------
    def _browse(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Library root", self.in_root.text() or str(paths.RECORDINGS_DIR))
        if d:
            self.in_root.setText(d)
            self.ctx.cfg.set("recording.root", d)
            self.rescan()

    def rescan(self) -> None:
        root = Path(self.in_root.text().strip() or paths.RECORDINGS_DIR)
        if not root.exists():
            self.lbl_summary.set(
                f"{root} does not exist yet. It is created when the first trial is "
                "recorded, or point the root at an existing folder of videos.", "warn")
            self.entries = []
            self._refill()
            return
        self.lbl_summary.set(f"Scanning {root}…", "info")
        self.btn_rescan.setEnabled(False)
        cfg_paths = self.ctx.cfg.section("paths")

        def job(progress=None):
            return library.scan(root, deep=False, cfg_paths=cfg_paths)

        if not self.scan_job.start(job):
            self.lbl_summary.set("A scan is already running.", "warn")
            self.btn_rescan.setEnabled(True)

    def _scan_done(self, entries: list) -> None:
        self.btn_rescan.setEnabled(True)
        self.entries = entries
        self._refill()
        s = library.library_summary(entries)
        if not entries:
            self.lbl_summary.set(
                "No videos found under the library root. Record a trial on the "
                "Acquisition tab, or point the root at a folder that has some.", "info")
        else:
            self.lbl_summary.set(
                f"{s['recordings']} recording(s), {s['sessions']} segmented session(s), "
                f"{s['analysed']} analysed, {s['curated']} human-curated · "
                f"{s['total_hours']} h, {s['total_gb']} GB · "
                f"{len(s['animals'])} animal(s) in {len(s['groups'])} group(s)", "ok")
        self.statusMessage.emit(f"Library: {len(entries)} entries")

    def _visible(self) -> list[library.Entry]:
        q = self.in_filter.text().strip().lower()
        if not q:
            return self.entries
        out = []
        for e in self.entries:
            hay = " ".join([e.name, e.animal, e.group, e.treatment, e.dose,
                            e.route, e.status]).lower()
            if all(tok in hay for tok in q.split()):
                out.append(e)
        return out

    def _refill(self) -> None:
        rows = self._visible()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        for r, e in enumerate(rows):
            vals = [
                ("\U0001F4C1 " if e.is_session else "") + e.name,
                e.animal, e.group, e.dose, e.route, e.mtime_iso, e.duration_hms,
                f"{e.width}x{e.height} {e.codec}" if e.width else "-",
                f"{e.fps:.1f}" + ("" if e.fps_exact else " ~"),
                W.human_bytes(e.size_bytes), e.status,
            ]
            for c, text in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(text)
                if c == 0:
                    item.setData(QtCore.Qt.UserRole, r)
                    item.setToolTip(str(e.path))
                if e.error:
                    item.setToolTip(e.error)
                self.table.setItem(r, c, item)
        self.table.setSortingEnabled(True)
        self._set_actions_enabled(False)

    def _selected(self) -> list[library.Entry]:
        rows = self._visible()
        out = []
        for idx in self.table.selectionModel().selectedRows() if \
                self.table.selectionModel() else []:
            item = self.table.item(idx.row(), 0)
            if item is None:
                continue
            i = item.data(QtCore.Qt.UserRole)
            if isinstance(i, int) and 0 <= i < len(rows):
                out.append(rows[i])
        return out

    def _current(self) -> library.Entry | None:
        sel = self._selected()
        return sel[0] if sel else None

    def _set_actions_enabled(self, on: bool) -> None:
        for b in (self.btn_review, self.btn_analyse, self.btn_verify,
                  self.btn_ingest, self.btn_reveal, self.btn_trash,
                  self.btn_save_meta):
            b.setEnabled(on)
        self.btn_batch.setEnabled(on)

    def _on_select(self) -> None:
        e = self._current()
        self._set_actions_enabled(e is not None)
        if e is None:
            self.txt_info.setPlainText("")
            return
        t = e.trial
        self.m_animal.setText(str(t.get("animal_id", "")))
        self.m_group.setText(str(t.get("group", "")))
        self.m_treatment.setText(str(t.get("treatment", "")))
        self.m_dose.setText(str(t.get("dose_mg_kg", "")))
        self.m_route.setText(str(t.get("route", "")))
        self.m_operator.setText(str(t.get("operator", "")))
        self.m_notes.setText(str(t.get("notes", "")))
        self.m_t0.setValue(float(t.get("t0_offset_s", -1) or -1))
        self.m_baseline.setValue(float(t.get("planned_baseline_s", 0) or 0))

        cap = e.capture
        lines = [f"Path: {e.path}"]
        if e.is_session:
            lines.append(f"Segments: {len(e.segments)}")
        lines += [
            f"Container says: {e.width}x{e.height} {e.codec}, "
            f"{e.fps:.3f} fps{'' if e.fps_exact else '  (header value, not verified)'}",
            f"Frame rate used for analysis: {e.fps_for_analysis():.3f} fps",
            f"Duration: {e.duration_hms}   Size: {W.human_bytes(e.size_bytes)}",
        ]
        if cap:
            lines.append(
                f"Recorded as: requested {cap.get('requested_fps')} fps, "
                f"expected {cap.get('expected_fps')} fps, {cap.get('pixel_format')}, "
                f"device {cap.get('device')}")
        cs = e.sidecar.get("camera_settings") or {}
        if cs:
            lines.append(
                f"Camera: exposure {cs.get('exposure')} "
                f"(≈{cs.get('exposure_estimate_ms')} ms), gain {cs.get('gain')}, "
                f"auto-exposure {'on' if cs.get('auto_exposure') else 'off'}")
            lm = cs.get("lighting_metrics") or {}
            if lm:
                lines.append(
                    f"Lighting at record time: contrast {lm.get('contrast')}/255, "
                    f"clipped {lm.get('clipped_high_pct')}%, "
                    f"uniformity {lm.get('uniformity')}, focus {lm.get('focus_score')}")
        obs = e.sidecar.get("observed") or {}
        if obs:
            lines.append(
                f"Observed at record time: {obs.get('frames')} frames, "
                f"{obs.get('effective_fps')} fps, dropped {obs.get('dropped')}")
            for err in (obs.get("errors") or [])[:2]:
                lines.append(f"  FFmpeg said: {err}")
        meas = e.sidecar.get("measured") or {}
        if meas:
            lines.append(
                f"Verified: {meas.get('frames')} frames / {meas.get('duration_s')} s "
                f"= {meas.get('true_fps')} fps  ({meas.get('measured_iso', '')[:16]})")
        if e.result:
            lines.append(
                f"Analysis: {e.result.get('detector')} · "
                f"{e.n_bursts} accepted twitch(es) · "
                f"{'human-curated' if e.curated else 'automatic'} · "
                f"{e.analysis_date[:16]}")
            for warn in (e.result.get("warnings") or [])[:3]:
                lines.append(f"  warning: {warn}")
        else:
            lines.append("Analysis: none yet")
        if e.error:
            lines.append(f"ERROR: {e.error}")
        self.txt_info.setPlainText("\n".join(lines))
        self.lbl_detail.set("", "info")

    # ------------------------------------------------------------------
    def _save_meta(self) -> None:
        e = self._current()
        if e is None:
            return
        try:
            sc = library.update_trial_meta(e, {
                "animal_id": self.m_animal.text().strip(),
                "group": self.m_group.text().strip(),
                "treatment": self.m_treatment.text().strip(),
                "dose_mg_kg": self.m_dose.text().strip(),
                "route": self.m_route.text().strip(),
                "operator": self.m_operator.text().strip(),
                "notes": self.m_notes.text().strip(),
                "t0_offset_s": float(self.m_t0.value()),
                "planned_baseline_s": float(self.m_baseline.value()),
            })
        except OSError as err:
            self.lbl_detail.set(f"Could not write the sidecar: {err}", "bad")
            return
        self.lbl_detail.set(f"Saved to {sc.name}", "ok")
        self._refill()

    def _open_review(self) -> None:
        e = self._current()
        if e is not None:
            self.openInReview.emit(e)

    def _analyse(self) -> None:
        e = self._current()
        if e is not None:
            self.analyseRequested.emit(e)

    def _batch(self) -> None:
        sel = self._selected()
        if sel:
            self.batchRequested.emit(sel)
            self.statusMessage.emit(f"Added {len(sel)} item(s) to the batch queue")

    def _ingest(self) -> None:
        e = self._current()
        if e is None:
            return
        from .ingest_dialog import IngestDialog
        dlg = IngestDialog(e.analysis_target, self.ctx.cfg, self.pal, self)
        dlg.converted.connect(lambda p: self.statusMessage.emit(
            f"Converted copy written to {p}"))
        dlg.exec_()
        self.rescan()

    def _verify(self) -> None:
        e = self._current()
        if e is None:
            return
        target = e.analysis_target
        expected = e.expected_fps or e.fps
        self.lbl_detail.set(
            "Decoding the whole file to count frames; this takes a while on a long "
            "trial…", "info")
        self.btn_verify.setEnabled(False)
        ffprobe = self.ctx.cfg.get("paths.ffprobe", "")

        def job(progress=None):
            return recorder.verify_recording(target, expected, ffprobe)

        def done(v):
            self.lbl_detail.set(v.message, "ok" if v.ok else "warn")
            self.rescan()

        try:
            self.verify_job.finished.disconnect()
        except TypeError:
            pass
        self.verify_job.finished.connect(done)
        if not self.verify_job.start(job):
            self.lbl_detail.set("A verification is already running.", "warn")

    def _reveal(self) -> None:
        e = self._current()
        if e is None:
            return
        p = e.path if e.is_session else e.path.parent
        try:
            if paths.is_windows():  # pragma: no cover
                if e.is_session:
                    os.startfile(str(p))  # noqa: S606
                else:
                    subprocess.Popen(["explorer", "/select,", str(e.path)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except OSError as err:
            self.lbl_detail.set(f"Could not open the folder: {err}", "warn")

    def _trash(self) -> None:
        sel = self._selected()
        if not sel:
            return
        names = "\n".join(f"• {e.name}" for e in sel[:12])
        more = f"\n…and {len(sel) - 12} more" if len(sel) > 12 else ""
        if not W.ask_yes_no(
                self, "Move to recycle bin?",
                f"This moves {len(sel)} item(s), with their metadata and analysis "
                f"files, to the Windows recycle bin:\n\n{names}{more}\n\n"
                "Nothing is permanently deleted - you can restore it from the bin."):
            return
        ok_n = 0
        msgs = []
        for e in sel:
            ok, msg = library.move_to_trash(e)
            ok_n += ok
            msgs.append(msg)
        self.lbl_detail.set(f"{ok_n} of {len(sel)} moved. {msgs[-1]}",
                            "ok" if ok_n == len(sel) else "warn")
        self.rescan()

    def _job_failed(self, msg: str, tb: str) -> None:
        self.btn_rescan.setEnabled(True)
        self.btn_verify.setEnabled(True)
        log.error("library job failed: %s\n%s", msg, tb)
        self.lbl_summary.set(msg, "bad")

    def entry_for_path(self, path: str) -> library.Entry | None:
        p = Path(path)
        for e in self.entries:
            if e.path == p or p in e.segments:
                return e
        return None

    def shutdown(self) -> None:
        W.stop_jobs(self)
        self.scan_job.wait(3000)
        self.verify_job.wait(3000)
