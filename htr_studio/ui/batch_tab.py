"""Batch tab: analyse a queue of trials and build the study-level summary.

A study is not one video. Twelve animals at 70 minutes each is roughly 12
million frames, so this runs unattended and has to be honest about what it did:
every trial gets its own CSV and figure, the result is attached to the recording,
and the run ends with one summary table plus a group time-course figure.

Failures do not stop the queue. A trial that cannot be analysed is marked failed
with its reason and the rest continue, because coming back to a batch that
aborted on item three of twelve wastes a night.
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from .. import analysis, library, paths, report
from . import widgets as W

log = logging.getLogger(__name__)

COLUMNS = ["Recording", "Animal", "Group", "Duration", "T0", "Status", "Twitches",
           "Detail"]


class BatchTab(QtWidgets.QWidget):
    statusMessage = QtCore.pyqtSignal(str)

    def __init__(self, ctx, analysis_tab, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.analysis_tab = analysis_tab
        self.queue: list[library.Entry] = []
        self.results: dict[str, dict] = {}
        self.worker: W.Worker | None = None
        self.run_job = W.JobRunner("batch", self)
        self.run_job.progressed.connect(self._on_progress)
        self.run_job.finished.connect(self._on_done)
        self.run_job.failed.connect(self._on_failed)
        self.scan_job = W.JobRunner("batch-scan", self)
        self.scan_job.finished.connect(self.add_entries)
        self.scan_job.failed.connect(
            lambda m, tb: self.lbl_status.set(f"Scan failed: {m}", "bad"))
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)

        bar = QtWidgets.QHBoxLayout()
        self.btn_add = W.button("Add videos…")
        self.btn_add_folder = W.button("Add a whole folder…")
        self.btn_remove = W.button("Remove selected")
        self.btn_clear = W.button("Clear queue")
        for b in (self.btn_add, self.btn_add_folder, self.btn_remove, self.btn_clear):
            bar.addWidget(b)
        bar.addStretch(1)
        self.chk_figures = QtWidgets.QCheckBox("Write a figure per trial")
        self.chk_figures.setChecked(True)
        self.chk_attach = QtWidgets.QCheckBox("Attach the result to each recording")
        self.chk_attach.setChecked(True)
        self.chk_skip = QtWidgets.QCheckBox("Skip trials already analysed")
        self.chk_skip.setChecked(True)
        self.chk_skip.setToolTip(
            "A curated result is never overwritten by a batch run, whatever this "
            "is set to - a human decision outranks a re-run.")
        for c in (self.chk_figures, self.chk_attach, self.chk_skip):
            bar.addWidget(c)
        v.addLayout(bar)

        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        self.table.setSelectionMode(QtWidgets.QTableWidget.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(len(COLUMNS) - 1, QtWidgets.QHeaderView.Stretch)
        v.addWidget(self.table, 1)

        out = QtWidgets.QHBoxLayout()
        out.addWidget(QtWidgets.QLabel("Output folder"))
        self.in_out = QtWidgets.QLineEdit(str(paths.EXPORTS_DIR))
        out.addWidget(self.in_out, 1)
        self.btn_out = W.button("…")
        self.btn_out.setMaximumWidth(32)
        out.addWidget(self.btn_out)
        v.addLayout(out)

        run = QtWidgets.QHBoxLayout()
        self.btn_run = W.button("Run batch", "primary")
        self.btn_cancel = W.button("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_summary = W.button("Rebuild summary from the queue")
        for b in (self.btn_run, self.btn_cancel, self.btn_summary):
            run.addWidget(b)
        run.addStretch(1)
        v.addLayout(run)

        self.progress_overall = QtWidgets.QProgressBar()
        self.progress_item = QtWidgets.QProgressBar()
        v.addWidget(self.progress_overall)
        v.addWidget(self.progress_item)
        self.lbl_status = W.StatusLabel("Queue is empty.", "info")
        v.addWidget(self.lbl_status)

        self.btn_add.clicked.connect(self._add_files)
        self.btn_add_folder.clicked.connect(self._add_folder)
        self.btn_remove.clicked.connect(self._remove)
        self.btn_clear.clicked.connect(self._clear)
        self.btn_out.clicked.connect(self._pick_out)
        self.btn_run.clicked.connect(self._run)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_summary.clicked.connect(self._write_summary)

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------
    def add_entries(self, entries: list[library.Entry]) -> None:
        known = {str(e.path) for e in self.queue}
        for e in entries:
            if str(e.path) not in known:
                self.queue.append(e)
        self._refill()

    def _add_files(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add videos", str(self.ctx.cfg.recordings_root()),
            "Videos (*.avi *.mp4 *.mkv *.mov);;All files (*)")
        if not files:
            return
        cfg_paths = self.ctx.cfg.section("paths")
        added = []
        for f in files:
            e = library._entry_from_video(Path(f), cfg_paths, deep=False)
            added.append(e)
        self.add_entries(added)

    def _add_folder(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Add every recording in a folder", str(self.ctx.cfg.recordings_root()))
        if not d:
            return
        self.lbl_status.set(f"Scanning {d}…", "info")
        paths_cfg = self.ctx.cfg.section("paths")
        if not self.scan_job.start(
                lambda progress=None: library.scan(d, deep=False,
                                                   cfg_paths=paths_cfg)):
            self.lbl_status.set("A folder scan is already running.", "warn")

    def _remove(self) -> None:
        rows = sorted((i.row() for i in
                       (self.table.selectionModel().selectedRows()
                        if self.table.selectionModel() else [])), reverse=True)
        for r in rows:
            if 0 <= r < len(self.queue):
                self.queue.pop(r)
        self._refill()

    def _clear(self) -> None:
        self.queue = []
        self.results = {}
        self._refill()

    def _pick_out(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Batch output folder", self.in_out.text() or str(paths.EXPORTS_DIR))
        if d:
            self.in_out.setText(d)

    def _refill(self) -> None:
        self.table.setRowCount(len(self.queue))
        for r, e in enumerate(self.queue):
            res = self.results.get(str(e.path), {})
            t0 = e.t0_offset_s
            vals = [e.name, e.animal, e.group, e.duration_hms,
                    (W.hms(t0) if t0 > 0 else "-"),
                    res.get("status", "curated" if e.curated
                            else ("analysed" if e.analysed else "queued")),
                    str(res.get("count", e.n_bursts if e.analysed else "")),
                    res.get("detail", "")]
            for c, text in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(str(text))
                if c == 0:
                    item.setToolTip(str(e.path))
                self.table.setItem(r, c, item)
        n = len(self.queue)
        self.lbl_status.set(
            "Queue is empty. Add videos, or select some in the Library and press "
            "'Add selection to Batch'." if not n
            else f"{n} trial(s) queued.", "info" if n else "info")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _run(self) -> None:
        if not self.queue:
            return
        acfg = self.analysis_tab.current_config()
        out_dir = Path(self.in_out.text().strip() or paths.EXPORTS_DIR)
        bin_s = float(self.ctx.cfg.get("report.bin_seconds", 60))
        want_fig = self.chk_figures.isChecked()
        want_attach = self.chk_attach.isChecked()
        skip_done = self.chk_skip.isChecked()
        theme_name = self.ctx.cfg.get("ui.theme", "light")
        items = list(self.queue)

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_overall.setRange(0, len(items))
        self.progress_overall.setValue(0)
        self.results = {}

        def job(progress=None):
            rows: list[dict] = []
            rate_csvs: dict[str, Path] = {}
            for i, entry in enumerate(items):
                key = str(entry.path)
                if progress and not progress(i, len(items), f"START|{key}"):
                    break
                # A curated result represents a human decision; never overwrite it.
                if entry.curated or (skip_done and entry.analysed):
                    if progress:
                        progress(i, len(items),
                                 f"SKIP|{key}|{'curated' if entry.curated else 'already analysed'}")
                    continue
                try:
                    target = entry.analysis_target
                    fps = entry.fps_for_analysis()

                    def sub(done, total, msg, _i=i, _key=key):
                        if progress:
                            return progress(_i, len(items), f"ITEM|{_key}|{done}|{total}")
                        return True

                    sig = analysis.extract_signals(target, acfg, fps=fps, progress=sub)
                    det = analysis.detect(sig, acfg,
                                          baseline_default_s=entry.baseline_s)
                    t0 = max(entry.t0_offset_s, 0.0)
                    rep = report.write_trial_csv(
                        target, out_dir, sig, det, bin_s=bin_s, t0_s=t0,
                        meta=entry.trial)
                    if rep.bins_csv:
                        rate_csvs[str(target)] = rep.bins_csv
                    if want_fig:
                        title = (f"{entry.animal} {entry.group} {entry.treatment} "
                                 f"{entry.dose}").strip() or target.stem
                        report.plot_trial(out_dir / f"{target.stem}_htr.png", sig, det,
                                          bin_s=bin_s, t0_s=t0, title=title,
                                          theme_name=theme_name)
                    row = report.study_row(target, sig.summary(), det, det.bursts,
                                           t0, bin_s, entry.trial)
                    rows.append(row)
                    if want_attach:
                        library.save_analysis(entry, {
                            "htr_studio_result_version": 1,
                            "video": str(target),
                            "signals": sig.summary(),
                            "config": acfg.to_dict(),
                            "detector": det.detector,
                            "threshold": det.threshold,
                            "diagnostics": det.diagnostics,
                            "warnings": det.warnings,
                            "bursts": [b.to_dict() for b in det.bursts],
                            "rejected": [b.to_dict() for b in det.rejected],
                            "bins": analysis.bin_counts(
                                [b for b in det.bursts if b.accepted],
                                sig.duration_s, bin_s, t0),
                            "n_accepted": det.count,
                            "curated": False,
                            "batch": True,
                        })
                    if progress:
                        progress(i + 1, len(items),
                                 f"DONE|{key}|{det.count}|"
                                 f"tracking {sig.valid_frac * 100:.0f}%"
                                 + (f"; {len(det.warnings)} warning(s)"
                                    if det.warnings else ""))
                except Exception as e:  # keep the queue moving
                    log.exception("batch item failed: %s", entry.path)
                    if progress:
                        progress(i + 1, len(items), f"FAIL|{key}|{e}")
            return rows, rate_csvs, out_dir, bin_s, theme_name

        if not self.run_job.start(job):
            self.lbl_status.set("A batch is already running.", "warn")
            return
        self.statusMessage.emit(f"Batch started: {len(items)} trial(s)")

    def _on_progress(self, done: int, total: int, msg: str) -> None:
        self.progress_overall.setRange(0, total)
        self.progress_overall.setValue(done)
        parts = msg.split("|")
        kind = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        if kind == "ITEM" and len(parts) >= 4:
            try:
                d, t = int(parts[2]), int(parts[3])
            except ValueError:
                return
            self.progress_item.setRange(0, max(t, 1))
            self.progress_item.setValue(min(d, t))
            self.results.setdefault(key, {})["status"] = "running"
            self.results[key]["detail"] = f"{d:,}/{t:,} frames"
        elif kind == "START":
            self.results[key] = {"status": "running", "detail": ""}
        elif kind == "SKIP":
            self.results[key] = {"status": "skipped",
                                 "detail": parts[2] if len(parts) > 2 else ""}
        elif kind == "DONE":
            self.results[key] = {"status": "done", "count": parts[2],
                                 "detail": parts[3] if len(parts) > 3 else ""}
        elif kind == "FAIL":
            self.results[key] = {"status": "FAILED",
                                 "detail": parts[2] if len(parts) > 2 else ""}
        self._refill_status_only()

    def _refill_status_only(self) -> None:
        for r, e in enumerate(self.queue):
            res = self.results.get(str(e.path))
            if not res:
                continue
            for c, val in ((5, res.get("status", "")), (6, str(res.get("count", ""))),
                           (7, res.get("detail", ""))):
                item = self.table.item(r, c)
                if item is None:
                    item = QtWidgets.QTableWidgetItem()
                    self.table.setItem(r, c, item)
                item.setText(str(val))

    def _cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.lbl_status.set("Cancelling after the current trial…", "warn")

    def _on_failed(self, msg: str, tb: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        log.error("batch failed: %s\n%s", msg, tb)
        self.lbl_status.set(msg, "bad")

    def _on_done(self, result) -> None:
        rows, rate_csvs, out_dir, bin_s, theme_name = result
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_item.setValue(0)
        self._last_rows, self._last_rate_csvs = rows, rate_csvs

        failed = [k for k, v in self.results.items() if v.get("status") == "FAILED"]
        skipped = [k for k, v in self.results.items() if v.get("status") == "skipped"]
        if not rows:
            self.lbl_status.set(
                "Nothing was analysed. "
                + (f"{len(skipped)} trial(s) were skipped as already analysed; "
                   "untick 'Skip trials already analysed' to redo them."
                   if skipped else "See the log for why."),
                "warn" if skipped else "bad")
            return

        summary = report.write_study_summary(out_dir / "study_summary.csv", rows)
        fig = report.plot_study(out_dir / "study_rate_by_group.png", rows, rate_csvs,
                                bin_s=bin_s, theme_name=theme_name)
        total = sum(int(r["total_twitches"]) for r in rows)
        msg = (f"Analysed {len(rows)} trial(s), {total} twitch(es) in total. "
               f"Summary: {summary.name}"
               + (f"; figure: {fig.name}" if fig else ""))
        if failed:
            msg += f"  — {len(failed)} trial(s) FAILED, see the Detail column."
        if skipped:
            msg += f"  {len(skipped)} skipped."
        self.lbl_status.set(msg, "bad" if failed else "ok")
        self.statusMessage.emit(msg)

    def _write_summary(self) -> None:
        """Rebuild the study table from results already attached to recordings."""
        out_dir = Path(self.in_out.text().strip() or paths.EXPORTS_DIR)
        bin_s = float(self.ctx.cfg.get("report.bin_seconds", 60))
        rows, rate_csvs = [], {}
        for e in self.queue:
            if not e.result:
                continue
            det = analysis.DetectionResult(
                bursts=[analysis.Burst.from_dict(b) for b in (e.result.get("bursts") or [])],
                detector=str(e.result.get("detector", "")),
                threshold=float(e.result.get("threshold", 0) or 0),
                fps=float((e.result.get("signals") or {}).get("fps", 0) or 0),
                duration_s=float((e.result.get("signals") or {}).get("duration_s", 0) or 0),
                warnings=list(e.result.get("warnings") or []))
            rows.append(report.study_row(
                e.analysis_target, e.result.get("signals") or {}, det, det.bursts,
                max(e.t0_offset_s, 0.0), bin_s, e.trial,
                curated=bool(e.result.get("curated"))))
            csv_path = out_dir / f"{e.analysis_target.stem}_htr_rate.csv"
            if csv_path.exists():
                rate_csvs[str(e.analysis_target)] = csv_path
        if not rows:
            self.lbl_status.set(
                "None of the queued trials has a stored result. Run the batch, or "
                "analyse them individually first.", "warn")
            return
        p = report.write_study_summary(out_dir / "study_summary.csv", rows)
        fig = report.plot_study(out_dir / "study_rate_by_group.png", rows, rate_csvs,
                                bin_s=bin_s,
                                theme_name=self.ctx.cfg.get("ui.theme", "light"))
        self.lbl_status.set(
            f"Rebuilt {p.name} from {len(rows)} stored result(s)"
            + (f" and {fig.name}" if fig else
               " (no rate CSVs in the output folder, so no group figure)"), "ok")

    def shutdown(self) -> None:
        self.scan_job.wait(3000)
        self.run_job.wait(8000)
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(8000)
