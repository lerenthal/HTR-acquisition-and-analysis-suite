"""Settings tab: tool paths, folders, report options, theme, and diagnostics.

Settings live in ``config/settings.json`` inside the installation folder, not in
the registry and not in the user profile, so copying the ``windows_HTR`` folder
to another machine carries the configuration with it.
"""
from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from .. import ffmpeg_tools, paths, theme
from . import widgets as W

log = logging.getLogger(__name__)


class SettingsTab(QtWidgets.QWidget):
    themeChanged = QtCore.pyqtSignal(str)
    statusMessage = QtCore.pyqtSignal(str)

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.tools_job = W.JobRunner("tool-check", self)
        self.tools_job.finished.connect(self._tools_done)
        self.env_job = W.JobRunner("environment", self)
        self._build()
        W.no_wheel(self)
        # txt_env exists only after _build.
        self.env_job.finished.connect(self.txt_env.setPlainText)
        self._load()

    def _build(self) -> None:
        inner = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(inner)

        # ---- tools ----
        g_tools = QtWidgets.QGroupBox("External tools")
        f = QtWidgets.QFormLayout(g_tools)
        self.in_ffmpeg = QtWidgets.QLineEdit()
        self.in_ffprobe = QtWidgets.QLineEdit()
        for lab, wdg, hint in (
                ("ffmpeg.exe", self.in_ffmpeg,
                 "Leave blank to use the copy in tools\\ffmpeg that install.bat "
                 "downloads, or any ffmpeg on PATH."),
                ("ffprobe.exe", self.in_ffprobe,
                 "Needed for exact frame counts; normally sits beside ffmpeg.")):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(wdg, 1)
            b = W.button("…")
            b.setMaximumWidth(32)
            b.clicked.connect(lambda _=None, w=wdg: self._pick_file(w))
            row.addWidget(b)
            box = QtWidgets.QWidget()
            box.setLayout(row)
            W.form_row(f, lab, box, hint)
        self.btn_check_tools = W.button("Check tools", "primary")
        self.btn_check_tools.clicked.connect(self._check_tools)
        f.addRow(self.btn_check_tools)
        self.lbl_tools = QtWidgets.QPlainTextEdit()
        self.lbl_tools.setReadOnly(True)
        self.lbl_tools.setMaximumHeight(90)
        f.addRow(self.lbl_tools)
        v.addWidget(g_tools)

        # ---- folders ----
        g_dirs = QtWidgets.QGroupBox("Folders")
        f2 = QtWidgets.QFormLayout(g_dirs)
        self.in_root = QtWidgets.QLineEdit()
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.in_root, 1)
        b = W.button("…")
        b.setMaximumWidth(32)
        b.clicked.connect(self._pick_root)
        row.addWidget(b)
        box = QtWidgets.QWidget()
        box.setLayout(row)
        W.form_row(f2, "Recordings root", box,
                   f"Blank means {paths.RECORDINGS_DIR}. Point this at the drive "
                   "with the space; a trial is about 2.5 GB.")
        self.lbl_dirs = QtWidgets.QLabel()
        self.lbl_dirs.setObjectName("hint")
        self.lbl_dirs.setWordWrap(True)
        f2.addRow(self.lbl_dirs)
        row2 = QtWidgets.QHBoxLayout()
        for name, target in (("Open exports folder", paths.EXPORTS_DIR),
                             ("Open logs folder", paths.LOGS_DIR),
                             ("Open install folder", paths.INSTALL_DIR)):
            bb = W.button(name)
            bb.clicked.connect(lambda _=None, t=target: self._open_dir(t))
            row2.addWidget(bb)
        row2.addStretch(1)
        f2.addRow(row2)
        v.addWidget(g_dirs)

        # ---- recording defaults ----
        g_rec = QtWidgets.QGroupBox("Recording defaults")
        f3 = QtWidgets.QFormLayout(g_rec)
        self.in_template = QtWidgets.QLineEdit()
        W.form_row(f3, "File-name template", self.in_template,
                   "Fields: {animal} {group} {treatment} {dose} {route} {operator} "
                   "{date} {time}")
        self.cb_container = QtWidgets.QComboBox()
        self.cb_container.addItems(["avi", "mkv"])
        W.form_row(f3, "Container", self.cb_container,
                   "AVI is what the validated pipeline used. MKV survives an "
                   "interrupted write better but some older tools dislike it.")
        self.in_rtbuf = QtWidgets.QLineEdit()
        W.form_row(f3, "FFmpeg rtbufsize", self.in_rtbuf,
                   "DirectShow's ring buffer, before FFmpeg sees a packet. A "
                   "320x240 MJPEG frame is about 2.5 kB, so 512M holds minutes "
                   "of video; raise it to 1024M only if the recording still "
                   "reports buffer overflows on a local disk.")
        self.sp_tqs = QtWidgets.QSpinBox()
        self.sp_tqs.setRange(8, 65536)
        self.sp_tqs.setSingleStep(512)
        W.form_row(f3, "FFmpeg thread_queue_size", self.sp_tqs,
                   "Packets queued between the capture thread and the writer. "
                   "FFmpeg's default of 8 is 33 ms of slack at 240 fps, which "
                   "is shorter than one disk flush - that is why the default "
                   "loses frames on long captures. 4096 is about 17 s of "
                   "240 fps video and ~10 MB of RAM at 320x240.")
        self.sp_minfree = QtWidgets.QDoubleSpinBox()
        self.sp_minfree.setRange(0, 10000)
        self.sp_minfree.setSuffix(" GB")
        W.form_row(f3, "Warn below free space", self.sp_minfree)
        self.chk_verify = QtWidgets.QCheckBox(
            "Verify the true frame rate after each recording")
        self.chk_verify.setToolTip(
            "Counts frames with ffprobe once the recording finishes. Takes a while "
            "on a long trial, but catches a bad capture while the animal is still "
            "in the room.")
        f3.addRow(self.chk_verify)
        self.chk_apply_uvc = QtWidgets.QCheckBox(
            "Push exposure and gain into the driver before recording")
        self.chk_apply_uvc.setToolTip(
            "Opens the camera briefly through OpenCV, applies the preview's "
            "settings, and closes it so FFmpeg inherits them. Most UVC drivers keep "
            "them; if yours does not, use the driver's own property page instead.")
        f3.addRow(self.chk_apply_uvc)
        v.addWidget(g_rec)

        # ---- reports ----
        g_rep = QtWidgets.QGroupBox("Reports")
        f4 = QtWidgets.QFormLayout(g_rep)
        self.sp_bin = QtWidgets.QSpinBox()
        self.sp_bin.setRange(5, 3600)
        self.sp_bin.setSuffix(" s")
        W.form_row(f4, "Time-course bin", self.sp_bin,
                   "60 s gives the conventional twitches-per-minute endpoint.")
        self.chk_plots = QtWidgets.QCheckBox("Write a figure alongside each report")
        f4.addRow(self.chk_plots)
        self.cb_theme = QtWidgets.QComboBox()
        self.cb_theme.addItems(["light", "dark"])
        W.form_row(f4, "Theme", self.cb_theme,
                   "Applies to the app and to exported figures. Takes effect on "
                   "restart for some panels.")
        v.addWidget(g_rep)

        # ---- diagnostics ----
        g_diag = QtWidgets.QGroupBox("Diagnostics")
        f5 = QtWidgets.QVBoxLayout(g_diag)
        row3 = QtWidgets.QHBoxLayout()
        self.btn_env = W.button("Show environment")
        self.btn_env.clicked.connect(self._show_env)
        self.btn_log = W.button("Open the log file")
        self.btn_log.clicked.connect(
            lambda: self._open_dir(paths.LOGS_DIR / "htr_studio.log"))
        self.btn_reset = W.button("Reset all settings", "danger")
        self.btn_reset.clicked.connect(self._reset)
        for b in (self.btn_env, self.btn_log, self.btn_reset):
            row3.addWidget(b)
        row3.addStretch(1)
        f5.addLayout(row3)
        self.txt_env = QtWidgets.QPlainTextEdit()
        self.txt_env.setReadOnly(True)
        self.txt_env.setMinimumHeight(140)
        f5.addWidget(self.txt_env)
        v.addWidget(g_diag)

        row4 = QtWidgets.QHBoxLayout()
        self.btn_save = W.button("Save settings", "primary")
        self.btn_save.clicked.connect(self._save)
        row4.addWidget(self.btn_save)
        row4.addStretch(1)
        self.lbl_saved = W.StatusLabel("", "info")
        row4.addWidget(self.lbl_saved, 1)
        v.addLayout(row4)
        v.addStretch(1)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(W.scrollable(inner))

    # ------------------------------------------------------------------
    def _load(self) -> None:
        c = self.ctx.cfg
        self.in_ffmpeg.setText(str(c.get("paths.ffmpeg", "")))
        self.in_ffprobe.setText(str(c.get("paths.ffprobe", "")))
        self.in_root.setText(str(c.get("recording.root", "")))
        self.in_template.setText(str(c.get("recording.name_template", "")))
        self.cb_container.setCurrentText(str(c.get("recording.container", "avi")))
        self.in_rtbuf.setText(str(c.get("camera.rtbufsize", "512M")))
        self.sp_tqs.setValue(int(c.get("camera.thread_queue_size", 4096)))
        self.sp_minfree.setValue(float(c.get("recording.min_free_gb", 30)))
        self.chk_verify.setChecked(bool(c.get("recording.auto_verify_after_record", True)))
        self.chk_apply_uvc.setChecked(bool(c.get("camera.apply_uvc_before_record", True)))
        self.sp_bin.setValue(int(c.get("report.bin_seconds", 60)))
        self.chk_plots.setChecked(bool(c.get("report.export_plots", True)))
        self.cb_theme.setCurrentText(str(c.get("ui.theme", "light")))
        self.lbl_dirs.setText(
            f"Install: {paths.INSTALL_DIR}\nData: {paths.DATA_DIR}\n"
            f"Settings file: {paths.SETTINGS_FILE}")

    def _save(self) -> None:
        c = self.ctx.cfg
        old_theme = str(c.get("ui.theme", "light"))
        c.set("paths.ffmpeg", self.in_ffmpeg.text().strip())
        c.set("paths.ffprobe", self.in_ffprobe.text().strip())
        c.set("recording.root", self.in_root.text().strip())
        c.set("recording.name_template", self.in_template.text().strip()
              or "{animal}_{group}_{date}_{time}")
        c.set("recording.container", self.cb_container.currentText())
        c.set("camera.rtbufsize", self.in_rtbuf.text().strip() or "512M")
        c.set("camera.thread_queue_size", int(self.sp_tqs.value()))
        c.set("recording.min_free_gb", float(self.sp_minfree.value()))
        c.set("recording.auto_verify_after_record", self.chk_verify.isChecked())
        c.set("camera.apply_uvc_before_record", self.chk_apply_uvc.isChecked())
        c.set("report.bin_seconds", int(self.sp_bin.value()))
        c.set("report.export_plots", self.chk_plots.isChecked())
        new_theme = self.cb_theme.currentText()
        c.set("ui.theme", new_theme)
        try:
            c.save()
        except OSError as e:
            self.lbl_saved.set(f"Could not write {c.file}: {e}", "bad")
            return
        self.lbl_saved.set(f"Saved to {c.file}", "ok")
        if new_theme != old_theme:
            self.themeChanged.emit(new_theme)

    def _pick_file(self, widget: QtWidgets.QLineEdit) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select the executable", widget.text() or str(paths.TOOLS_DIR),
            "Executables (*.exe);;All files (*)")
        if path:
            widget.setText(path)

    def _pick_root(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Recordings root", self.in_root.text() or str(paths.RECORDINGS_DIR))
        if d:
            self.in_root.setText(d)

    def _open_dir(self, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            if paths.is_windows():  # pragma: no cover
                os.startfile(str(target))  # noqa: S606
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except OSError as e:
            W.warn_box(self, "Could not open", f"{target}\n\n{e}")

    def _check_tools(self) -> None:
        """Ask the tools their versions. Off-thread: each is a process launch,
        and on Windows a ~100 MB static binary that anti-virus inspects."""
        cfg = {"ffmpeg": self.in_ffmpeg.text().strip(),
               "ffprobe": self.in_ffprobe.text().strip()}
        self.lbl_tools.setPlainText("Checking…")
        if not self.tools_job.start(
                lambda progress=None: ffmpeg_tools.tool_versions(cfg)):
            self.lbl_tools.setPlainText("Already checking…")

    def _tools_done(self, versions: dict) -> None:
        lines = [f"{k}: {v}" for k, v in versions.items()]
        missing = [k for k, v in versions.items() if "NOT FOUND" in v]
        if missing:
            lines.append("")
            lines.append("Run install.bat to download a static FFmpeg build into "
                         "tools\\ffmpeg, or point the boxes above at an existing one.")
        self.lbl_tools.setPlainText("\n".join(lines))

    def _show_env(self) -> None:
        """Collect the environment report off-thread (it probes the tools)."""
        self.txt_env.setPlainText("Collecting…")
        if not self.env_job.start(lambda progress=None: self._env_text()):
            self.txt_env.setPlainText("Already collecting…")

    def _env_text(self) -> str:
        import numpy
        import cv2
        import scipy
        import PyQt5.QtCore as QC

        try:
            import pyqtgraph
            pg_ver = pyqtgraph.__version__
        except ImportError:  # pragma: no cover
            pg_ver = "not installed"
        try:
            import matplotlib
            mpl_ver = matplotlib.__version__
        except ImportError:  # pragma: no cover
            mpl_ver = "not installed"
        from .. import __version__

        lines = [
            f"HTR Studio {__version__}",
            f"Python {sys.version.split()[0]} ({sys.executable})",
            f"Platform {platform.platform()}",
            f"CPU count {os.cpu_count()}",
            f"numpy {numpy.__version__}",
            f"opencv {cv2.__version__}",
            f"scipy {scipy.__version__}",
            f"PyQt5 {QC.PYQT_VERSION_STR} / Qt {QC.QT_VERSION_STR}",
            f"pyqtgraph {pg_ver}",
            f"matplotlib {mpl_ver}",
            "",
            f"install dir {paths.INSTALL_DIR}",
            f"settings    {paths.SETTINGS_FILE}",
            f"recordings  {self.ctx.cfg.recordings_root()}",
            "",
        ]
        for k, v in ffmpeg_tools.tool_versions(self.ctx.cfg.section("paths")).items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)

    def _reset(self) -> None:
        if not W.ask_yes_no(
                self, "Reset all settings?",
                "Every setting goes back to the shipped default. Recordings, "
                "metadata and analysis results are untouched."):
            return
        from .. import config as cfgmod
        self.ctx.cfg.data = cfgmod._deep_merge(cfgmod.DEFAULTS, {})
        try:
            self.ctx.cfg.save()
        except OSError as e:
            self.lbl_saved.set(f"Could not write the settings file: {e}", "bad")
            return
        self._load()
        self.lbl_saved.set("Settings reset. Restart the app to reload every panel.",
                           "ok")

    def shutdown(self) -> None:
        """Stop background jobs. The window calls this before closing."""
        W.stop_jobs(self)
