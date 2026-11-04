"""Inspect a video, and convert it to something storable without breaking it.

Opened from the Library. It does three things in the order they are needed:

1. **Says what the file is.** Codec, measured frame rate against the declared
   one, whether every frame is coded independently, bytes per frame. For footage
   from another capture program or from a paper, this is the only way to know
   whether it can carry the measurement at all.

2. **Finds the arena.** Cropping to the cylinder is the one size reduction that
   costs nothing scientifically, and it is usually the largest -- measured at 81%
   off a 320x240 benchmark whose arena occupied a 150x150 square. The arena is
   detected from the same background model the analysis uses, so the button is a
   single click rather than four numbers guessed off a screenshot.

3. **Measures each conversion on this footage before committing to one.** The
   usual advice (archive losslessly) is wrong for MJPEG camera footage: lossless
   coding has to reproduce the camera's own JPEG artefacts and sensor noise
   exactly, which costs more than the JPEG did -- FFV1 came out five times larger
   than the source in testing. There is no table of bitrates that would have
   told the operator that, so the dialog encodes a few seconds of the real file
   with each option and scales the result.

Every option it offers is all-intra, and none of them changes the frame rate.
Both of those are deliberate and neither is configurable: an inter-frame codec
removes exactly the fast head displacement the measurement consists of, and
halving the frame rate takes the Nyquist limit below the HTR band. The
conversion is verified by comparing frame counts, and a mismatch is reported as
a failure rather than a warning, because a file that quietly lost its last
thirty seconds will not be noticed for a year.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets

from .. import analysis, ingest, paths, theme
from . import widgets as W

log = logging.getLogger(__name__)


def _build_arena(video: str, progress=None):
    """Background model and arena bounding box. Runs on a worker thread.

    Returns (background, (x, y, w, h) or None, note). Free of Qt so it can be
    called off the GUI thread and tested without a window.
    """
    frames, _n = analysis._sample_frames(video, 40)
    if not frames:
        return None, None, "No frames could be decoded from this file."
    if progress is not None:
        progress(1, 2, "building the background model")
    bg = analysis.build_background(frames)
    mask, note = analysis.auto_roi_from_background(bg)
    if mask is None:
        return bg, None, note
    x, y, w, h = cv2.boundingRect(mask)
    # Several encoders refuse an odd width or height.
    w -= w % 2
    h -= h % 2
    return bg, (x, y, w, h), note


class IngestDialog(QtWidgets.QDialog):
    """Inspection, arena crop, measured size options, and the conversion."""

    converted = QtCore.pyqtSignal(str)       # path of the new file

    def __init__(self, video: Path | str, cfg, pal: theme.Palette, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inspect and archive footage")
        self.resize(1040, 720)
        self.video = Path(video)
        self.cfg = cfg
        self.pal = pal
        self.report: ingest.FileReport | None = None
        self.estimates: list[ingest.SizeEstimate] = []
        self.worker: W.Worker | None = None
        self._bg: np.ndarray | None = None
        self.inspect_job = W.JobRunner("inspect", self)
        self.inspect_job.finished.connect(self._inspect_done)
        self.inspect_job.failed.connect(self._failed)
        self.measure_job = W.JobRunner("measure", self)
        self.measure_job.finished.connect(self._measure_done)
        self.measure_job.failed.connect(self._failed)
        self.convert_job = W.JobRunner("convert", self)
        self.convert_job.progressed.connect(self._on_progress)
        self.convert_job.finished.connect(self._convert_done)
        self.convert_job.failed.connect(self._failed)
        self.arena_job = W.JobRunner("find-arena", self)
        self.arena_job.finished.connect(self._arena_done)
        self.arena_job.failed.connect(
            lambda m, tb: self.lbl_arena.set(f"Could not sample the file: {m}",
                                             "bad"))
        self._build()
        self.arena_job._button = self.btn_find_arena
        self.inspect_job._button = self.btn_count
        self.measure_job._button = self.btn_measure
        self.convert_job._button = self.btn_convert
        self.convert_job._cancel_button = self.btn_cancel
        self.convert_job._progress = self.progress
        QtCore.QTimer.singleShot(0, lambda: self._inspect(count_frames=False))

    # ------------------------------------------------------------------
    def _build(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._page_inspect(), "What is this file?")
        tabs.addTab(self._page_archive(), "Archive / convert")
        tabs.addTab(self._page_plan(), "Plan a capture")
        outer.addWidget(tabs, 1)

        row = QtWidgets.QHBoxLayout()
        self.lbl_status = W.StatusLabel("Reading the file…", "info")
        self.lbl_status.setWordWrap(True)
        row.addWidget(self.lbl_status, 1)
        close = W.button("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        outer.addLayout(row)

    # ---- page 1: inspection ----
    def _page_inspect(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)

        row = QtWidgets.QHBoxLayout()
        self.btn_count = W.button(
            "Count frames exactly", "primary",
            "Decodes the whole file. Slow - tens of seconds for a 30-minute "
            "trial - and the only way to know the true frame rate of MJPEG in "
            "AVI, where the header's figure is routinely wrong. Every timing "
            "number in the study rests on this.")
        self.btn_count.clicked.connect(lambda: self._inspect(count_frames=True))
        row.addWidget(self.btn_count)
        self.btn_save_rep = W.button("Save this report beside the file")
        self.btn_save_rep.clicked.connect(self._save_report)
        self.btn_save_rep.setEnabled(False)
        row.addWidget(self.btn_save_rep)
        row.addStretch(1)
        v.addLayout(row)

        tiles = QtWidgets.QHBoxLayout()
        self.t_codec = W.MetricTile("codec")
        self.t_fps = W.MetricTile("measured rate", "-", "fps")
        self.t_nyq = W.MetricTile("Nyquist", "-", "Hz")
        self.t_size = W.MetricTile("size", "-", "GB")
        self.t_bpf = W.MetricTile("per frame", "-", "B")
        for t in (self.t_codec, self.t_fps, self.t_nyq, self.t_size, self.t_bpf):
            tiles.addWidget(t)
        v.addLayout(tiles)

        self.txt_rep = QtWidgets.QPlainTextEdit()
        self.txt_rep.setReadOnly(True)
        self.txt_rep.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        v.addWidget(self.txt_rep, 1)
        return page

    # ---- page 2: archive ----
    def _page_archive(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(page)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        left.setMaximumWidth(430)

        gc = QtWidgets.QGroupBox("Crop to the arena")
        cf = QtWidgets.QFormLayout(gc)
        self.chk_crop = QtWidgets.QCheckBox("Crop before encoding")
        self.chk_crop.setToolTip(
            "The one reduction that is free scientifically: the pixels outside "
            "the cylinder carry no information about the animal. On the "
            "benchmark it saved 81%, more than any codec choice.")
        cf.addRow(self.chk_crop)
        crop = QtWidgets.QHBoxLayout()
        self.sp_crop = []
        for name, hi in (("x", 10000), ("y", 10000), ("w", 10000), ("h", 10000)):
            s = QtWidgets.QSpinBox()
            s.setRange(0, hi)
            s.setPrefix(f"{name} ")
            self.sp_crop.append(s)
            crop.addWidget(s)
        cf.addRow(W.form_wrap(crop))
        self.btn_find_arena = W.button(
            "Find the arena", "primary",
            "Builds the empty-arena background model - the same one the "
            "analysis uses - and takes the bounding box of the floor.")
        self.btn_find_arena.clicked.connect(self._find_arena)
        cf.addRow(self.btn_find_arena)
        self.lbl_arena = W.StatusLabel("", "info")
        self.lbl_arena.setWordWrap(True)
        cf.addRow(self.lbl_arena)
        lv.addWidget(gc)

        gm = QtWidgets.QGroupBox("Measure the options on this footage")
        mv = QtWidgets.QVBoxLayout(gm)
        mv.addWidget(QtWidgets.QLabel(
            "Each option encodes a few seconds of this file and the result is "
            "scaled to the whole recording. Nothing is predicted from a table: "
            "MJPEG's cost depends on how much texture the bedding has, and a "
            "lossless codec on already-compressed footage can come out larger "
            "than the original."))
        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(QtWidgets.QLabel("Sample"))
        self.sp_sample = QtWidgets.QDoubleSpinBox()
        self.sp_sample.setRange(1.0, 60.0)
        self.sp_sample.setValue(5.0)
        self.sp_sample.setSuffix(" s")
        srow.addWidget(self.sp_sample)
        self.btn_measure = W.button("Measure", "primary")
        self.btn_measure.clicked.connect(self._measure)
        srow.addWidget(self.btn_measure)
        srow.addStretch(1)
        mv.addLayout(srow)
        lv.addWidget(gm)

        gd = QtWidgets.QGroupBox("Convert")
        df = QtWidgets.QFormLayout(gd)
        self.cb_option = QtWidgets.QComboBox()
        for o in ingest.archive_options():
            self.cb_option.addItem(o.label, o.key)
        self.cb_option.currentIndexChanged.connect(self._option_changed)
        W.form_row(df, "Option", self.cb_option)
        self.lbl_why = QtWidgets.QLabel()
        self.lbl_why.setWordWrap(True)
        self.lbl_why.setObjectName("hint")
        df.addRow(self.lbl_why)
        self.in_dest = QtWidgets.QLineEdit()
        pick = QtWidgets.QHBoxLayout()
        pick.addWidget(self.in_dest, 1)
        b = W.button("…")
        b.setMaximumWidth(36)
        b.clicked.connect(self._pick_dest)
        pick.addWidget(b)
        W.form_row(df, "Write to", W.form_wrap(pick),
                   "The original is never modified or deleted. Check the new "
                   "file analyses the same before you remove anything.")
        self.chk_verify = QtWidgets.QCheckBox(
            "Verify by counting frames afterwards")
        self.chk_verify.setChecked(True)
        self.chk_verify.setToolTip(
            "Compares the frame count in and out. A conversion that silently "
            "truncated the recording would otherwise go unnoticed.")
        df.addRow(self.chk_verify)
        brow = QtWidgets.QHBoxLayout()
        self.btn_convert = W.button("Convert", "primary")
        self.btn_convert.clicked.connect(self._convert)
        brow.addWidget(self.btn_convert)
        self.btn_cancel = W.button("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        brow.addWidget(self.btn_cancel)
        brow.addStretch(1)
        df.addRow(W.form_wrap(brow))
        self.progress = QtWidgets.QProgressBar()
        df.addRow(self.progress)
        lv.addWidget(gd)
        lv.addStretch(1)

        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.addWidget(QtWidgets.QLabel("Measured size for this recording"))
        self.tbl = QtWidgets.QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(
            ["option", "predicted", "of original", "lossless"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        self.tbl.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        rv.addWidget(self.tbl, 1)
        self.view = W.FrameView()
        self.view.set_palette_tokens(self.pal)
        self.view.setMinimumHeight(220)
        rv.addWidget(self.view)
        self.lbl_notes = QtWidgets.QPlainTextEdit()
        self.lbl_notes.setReadOnly(True)
        self.lbl_notes.setMaximumHeight(110)
        rv.addWidget(self.lbl_notes)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(W.scrollable(left))
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        outer.addWidget(split)
        self._option_changed()
        return page

    # ---- page 3: planning ----
    def _page_plan(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.addWidget(QtWidgets.QLabel(
            "What a capture will cost on disk. The point of this page is that a "
            "long high-frame-rate trial is not the problem it looks like: half "
            "an hour at 240 fps in the camera's own MJPEG is about a gigabyte. "
            "It is uncompressed capture that produces unmanageable files, and "
            "the fix is to let the camera encode, never to lower the frame rate."))
        form = QtWidgets.QFormLayout()
        self.sp_mins = QtWidgets.QDoubleSpinBox()
        self.sp_mins.setRange(1, 600)
        self.sp_mins.setValue(30)
        self.sp_mins.setSuffix(" min")
        form.addRow("Duration", self.sp_mins)
        self.sp_pfps = QtWidgets.QDoubleSpinBox()
        self.sp_pfps.setRange(1, 2000)
        self.sp_pfps.setValue(240)
        self.sp_pfps.setSuffix(" fps")
        form.addRow("Frame rate", self.sp_pfps)
        wh = QtWidgets.QHBoxLayout()
        self.sp_pw = QtWidgets.QSpinBox()
        self.sp_pw.setRange(32, 8192)
        self.sp_pw.setValue(320)
        self.sp_ph = QtWidgets.QSpinBox()
        self.sp_ph.setRange(32, 8192)
        self.sp_ph.setValue(240)
        wh.addWidget(self.sp_pw)
        wh.addWidget(QtWidgets.QLabel("x"))
        wh.addWidget(self.sp_ph)
        form.addRow("Frame size", W.form_wrap(wh))
        self.cb_pf = QtWidgets.QComboBox()
        self.cb_pf.addItems(["mjpeg", "yuyv422", "gray"])
        form.addRow("Pixel format", self.cb_pf)
        self.sp_seg = QtWidgets.QDoubleSpinBox()
        self.sp_seg.setRange(0, 120)
        self.sp_seg.setValue(5)
        self.sp_seg.setSuffix(" min")
        self.sp_seg.setSpecialValueText("one file")
        form.addRow("Segment length", self.sp_seg)
        v.addLayout(form)
        for w in (self.sp_mins, self.sp_pfps, self.sp_pw, self.sp_ph, self.sp_seg):
            w.valueChanged.connect(self._replan)
        self.cb_pf.currentIndexChanged.connect(self._replan)

        tiles = QtWidgets.QHBoxLayout()
        self.p_total = W.MetricTile("whole trial", "-", "GB")
        self.p_seg = W.MetricTile("per segment", "-", "GB")
        self.p_bpf = W.MetricTile("per frame", "-", "B")
        self.p_rate = W.MetricTile("write rate", "-", "MB/s")
        for t in (self.p_total, self.p_seg, self.p_bpf, self.p_rate):
            tiles.addWidget(t)
        v.addLayout(tiles)
        self.txt_plan = QtWidgets.QPlainTextEdit()
        self.txt_plan.setReadOnly(True)
        v.addWidget(self.txt_plan, 1)
        QtCore.QTimer.singleShot(0, self._replan)
        return page

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def _inspect(self, count_frames: bool) -> None:
        self.lbl_status.set(
            "Counting frames - this decodes the whole file…" if count_frames
            else "Reading the header…", "info")
        self.btn_count.setEnabled(False)

        def job(progress=None):
            return ingest.inspect_file(
                self.video, self.cfg.get("paths.ffprobe", ""),
                count_frames=count_frames)

        if not self.inspect_job.start(job):
            self.lbl_status.set("Already reading this file…", "warn")

    def _inspect_done(self, rep) -> None:
        self.report = rep
        self.btn_count.setEnabled(True)
        self.btn_save_rep.setEnabled(True)
        self.t_codec.set_value(rep.codec or "?")
        self.t_fps.set_value(f"{rep.true_fps:.2f}")
        self.t_nyq.set_value(f"{rep.nyquist_hz:.0f}")
        self.t_size.set_value(f"{rep.size_bytes / 1e9:.3f}")
        self.t_bpf.set_value(f"{rep.bytes_per_frame:,.0f}")
        self.txt_rep.setPlainText(rep.as_text())
        self.lbl_status.set(
            rep.findings[0] if rep.findings
            else "Nothing wrong with this file. Add it to the Library and "
                 "analyse it.",
            rep.verdict or "ok")
        for s, val in zip(self.sp_crop, (0, 0, rep.width, rep.height)):
            if s.value() == 0 or s is self.sp_crop[2] or s is self.sp_crop[3]:
                s.setValue(int(val))
        if not self.in_dest.text().strip():
            self.in_dest.setText(str(self._default_dest()))

    def _save_report(self) -> None:
        if self.report is None:
            return
        out = self.video.with_name(self.video.stem + ".ingest.json")
        try:
            ingest.write_report(out, self.report, self.estimates)
        except OSError as e:
            self.lbl_status.set(f"Could not write the report: {e}", "bad")
            return
        self.lbl_status.set(f"Report written to {out.name}.", "ok")

    # ------------------------------------------------------------------
    # Arena
    # ------------------------------------------------------------------
    def _find_arena(self) -> None:
        """Build the background model and take the arena's bounding box.

        Off the GUI thread: it samples 40 frames scattered through the file and
        was measured at 4.5 s, growing with recording length.
        """
        self.lbl_arena.set("Building the background model…", "info")
        if not self.arena_job.start(_build_arena, str(self.video)):
            self.lbl_arena.set("Already looking for the arena…", "warn")

    def _arena_done(self, res) -> None:
        bg, box, note = res
        self._bg = bg
        if bg is None:
            self.lbl_arena.set(note, "bad")
            return
        if box is None:
            self.lbl_arena.set(note + "  Set the crop by hand, or leave it off.",
                               "warn")
            self.view.show_frame(bg, {}, "background model")
            return
        x, y, w, h = box
        for sp, val in zip(self.sp_crop, (x, y, w, h)):
            sp.setValue(int(val))
        self.chk_crop.setChecked(True)
        kept = (w * h) / float(bg.shape[0] * bg.shape[1])
        self.lbl_arena.set(
            f"Arena found at {w}x{h} starting ({x},{y}): {kept:.0%} of the "
            f"frame, so cropping removes {1 - kept:.0%} of the pixels at no "
            "cost to the measurement.", "ok")
        vis = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis, (x, y), (x + w, y + h),
                      W.FrameView._hex_to_bgr(self.pal.s2), 1)
        self.view.show_frame(vis, {}, f"arena {w}x{h}")

    def _crop(self) -> tuple[int, int, int, int] | None:
        if not self.chk_crop.isChecked():
            return None
        x, y, w, h = (int(s.value()) for s in self.sp_crop)
        if w <= 0 or h <= 0:
            return None
        return (w, h, x, y)                  # ffmpeg crop order

    # ------------------------------------------------------------------
    # Measuring and converting
    # ------------------------------------------------------------------
    def _measure(self) -> None:
        self.btn_measure.setEnabled(False)
        self.lbl_status.set("Encoding samples with each option…", "info")
        keys = [o.key for o in ingest.archive_options() if o.key != "copy"]
        crop = self._crop()
        secs = float(self.sp_sample.value())

        def job(progress=None):
            return ingest.estimate_sizes(
                self.video, keys, sample_seconds=secs, crop=crop,
                ffmpeg=self.cfg.get("paths.ffmpeg", ""),
                ffprobe=self.cfg.get("paths.ffprobe", ""),
                progress=(lambda m: progress and progress(0, 0, m)))

        if not self.measure_job.start(job):
            self.lbl_status.set("A measurement is already running.", "warn")

    def _measure_done(self, ests) -> None:
        self.btn_measure.setEnabled(True)
        self.estimates = ests
        orig = self.report.size_bytes if self.report else 0
        self.tbl.setRowCount(0)
        notes = []
        best, best_key = None, ""
        for e in ests:
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)
            self.tbl.setItem(r, 0, QtWidgets.QTableWidgetItem(e.label))
            if e.predicted_bytes:
                self.tbl.setItem(r, 1, QtWidgets.QTableWidgetItem(
                    f"{e.predicted_bytes / 1e9:.3f} GB"))
                self.tbl.setItem(r, 2, QtWidgets.QTableWidgetItem(
                    f"{e.ratio * 100:.0f}%"))
            else:
                self.tbl.setItem(r, 1, QtWidgets.QTableWidgetItem("-"))
                self.tbl.setItem(r, 2, QtWidgets.QTableWidgetItem("-"))
            self.tbl.setItem(r, 3, QtWidgets.QTableWidgetItem(
                "yes" if e.lossless else "no"))
            if e.note:
                notes.append(f"{e.label}: {e.note}")
            if e.predicted_bytes and (best is None or e.predicted_bytes < best):
                best, best_key = e.predicted_bytes, e.option
        self.tbl.resizeColumnsToContents()
        if orig:
            notes.insert(0, f"The original is {orig / 1e9:.3f} GB.")
        if best_key:
            notes.append(
                f"Smallest here is '{best_key}'. Size is not the only "
                "criterion: keep a lossless or original copy until the "
                "analysis for this trial is finished and checked.")
            self._set_option(best_key)
        self.lbl_notes.setPlainText("\n".join(notes))
        self.lbl_status.set("Measured. The predictions are from this file's own "
                            "footage, not a bitrate table.", "ok")

    def _set_option(self, key: str) -> None:
        for i in range(self.cb_option.count()):
            if self.cb_option.itemData(i) == key:
                self.cb_option.setCurrentIndex(i)
                return

    def _option_changed(self) -> None:
        key = self.cb_option.currentData()
        opt = next((o for o in ingest.archive_options() if o.key == key), None)
        self.lbl_why.setText(opt.why if opt else "")
        self.in_dest.setText(str(self._default_dest()))
        self.chk_crop.setEnabled(key != "copy")
        if key == "copy" and self.chk_crop.isChecked():
            self.chk_crop.setChecked(False)
            self.lbl_arena.set("Repackaging cannot crop - it does not touch "
                               "pixels. Pick a re-encode to crop.", "warn")

    def _default_dest(self) -> Path:
        key = self.cb_option.currentData() or "ffv1"
        ext = "avi" if key in ("mjpeg_q4",) else "mkv"
        d = paths.EXPORTS_DIR / "archive"
        return d / f"{self.video.stem}_{key}.{ext}"

    def _pick_dest(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Write the converted file to", self.in_dest.text(),
            "Matroska (*.mkv);;AVI (*.avi);;All files (*)")
        if path:
            self.in_dest.setText(path)

    def _convert(self) -> None:
        dest = Path(self.in_dest.text().strip())
        if not dest.name:
            self.lbl_status.set("Choose where to write the converted file.", "bad")
            return
        if dest.resolve() == self.video.resolve():
            self.lbl_status.set("That is the source file. Choose another name - "
                                "the original is never overwritten.", "bad")
            return
        if dest.exists() and not W.ask_yes_no(
                self, "Replace it?", f"{dest.name} already exists. Replace it?"):
            return
        dest.unlink(missing_ok=True)
        spec = ingest.ArchiveSpec(source=self.video, dest=dest,
                                  option=self.cb_option.currentData(),
                                  crop=self._crop())
        verify = self.chk_verify.isChecked()
        self.btn_convert.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.lbl_status.set(f"Converting to {dest.name}…", "info")

        ffm = self.cfg.get("paths.ffmpeg", "")
        ffp = self.cfg.get("paths.ffprobe", "")

        def job(progress=None):
            return ingest.archive(
                spec, ffmpeg=ffm, ffprobe=ffp, verify=verify,
                progress=(lambda d, t, m: bool(progress(d, t, m))
                          if progress else True))

        if not self.convert_job.start(job):
            self.lbl_status.set("A conversion is already running.", "warn")

    def _on_progress(self, done: int, total: int, msg: str) -> None:
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(min(done, total))
            self.lbl_status.set(f"{msg}: {done:,} of {total:,} frames", "info")
        else:
            self.lbl_status.set(msg, "info")

    def _cancel(self) -> None:
        self.convert_job.cancel()
        self.lbl_status.set("Cancelling…", "warn")

    def _convert_done(self, res) -> None:
        self.btn_convert.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if not res.ok:
            self.lbl_status.set(res.message, "bad")
            return
        self.lbl_status.set(
            f"Written to {res.dest}.  {res.message}  The original is untouched; "
            "keep it until this trial's analysis is finished and checked.", "ok")
        self.converted.emit(str(res.dest))

    def _failed(self, msg: str, tb: str) -> None:
        # Button state belongs to whichever runner owns each button: this
        # handler used to re-enable all four, which would un-disable a button
        # whose own job was still running if two failed close together.
        log.error("ingest failed: %s\n%s", msg, tb)
        self.lbl_status.set(msg, "bad")

    # ------------------------------------------------------------------
    def _replan(self) -> None:
        plan = ingest.plan_capture(
            float(self.sp_mins.value()), float(self.sp_pfps.value()),
            int(self.sp_pw.value()), int(self.sp_ph.value()),
            self.cb_pf.currentText(), float(self.sp_seg.value()))
        self.p_total.set_value(f"{plan.total_bytes / 1e9:.2f}")
        self.p_seg.set_value(f"{plan.segment_bytes / 1e9:.2f}")
        self.p_bpf.set_value(f"{plan.bytes_per_frame:,.0f}")
        self.p_rate.set_value(
            f"{plan.bytes_per_frame * plan.fps / 1e6:.2f}")
        lines = [
            f"{plan.minutes:.0f} min at {plan.fps:.0f} fps, "
            f"{plan.width}x{plan.height}, {plan.pixel_format}",
            f"  {plan.bytes_per_frame:,.0f} bytes per frame",
            f"  {plan.bytes_per_frame * plan.fps / 1e6:.2f} MB/s sustained write",
            f"  {plan.total_bytes / 1e9:.2f} GB for the trial, in "
            f"{plan.segments} file(s)",
            "",
        ]
        if plan.notes:
            lines.append("NOTES")
            lines += [f"  - {n}" for n in plan.notes]
        self.txt_plan.setPlainText("\n".join(lines))

    def closeEvent(self, ev) -> None:        # noqa: N802 - Qt naming
        for job in (self.arena_job, self.inspect_job, self.measure_job,
                    self.convert_job):
            job.wait(3000)
        super().closeEvent(ev)
