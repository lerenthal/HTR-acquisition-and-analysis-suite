"""The main window: shared state, the six tabs, and the wiring between them."""
from __future__ import annotations

import logging
import webbrowser
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

from .. import __version__, analysis, config, library, paths, theme
from . import widgets as W
from .acquisition_tab import AcquisitionTab
from .analysis_tab import AnalysisTab
from .batch_tab import BatchTab
from .library_tab import LibraryTab
from .review_tab import ReviewTab
from .settings_tab import SettingsTab

log = logging.getLogger(__name__)


class AppContext(QtCore.QObject):
    """State shared between tabs, plus the signals that announce changes."""

    analysisChanged = QtCore.pyqtSignal()
    entryChanged = QtCore.pyqtSignal(object)

    def __init__(self, cfg: config.Config):
        super().__init__()
        self.cfg = cfg
        self.entry: library.Entry | None = None
        self.signals: analysis.Signals | None = None
        self.detection: analysis.DetectionResult | None = None

    def set_analysis(self, sig: analysis.Signals,
                     det: analysis.DetectionResult) -> None:
        self.signals, self.detection = sig, det
        self.analysisChanged.emit()

    def set_entry(self, entry: library.Entry) -> None:
        self.entry = entry
        self.entryChanged.emit(entry)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg: config.Config, initial_video: str | None = None):
        super().__init__()
        self.ctx = AppContext(cfg)
        self.setWindowTitle(f"HTR Studio {__version__}  —  head-twitch response "
                            "acquisition and analysis")
        self.resize(1520, 960)

        pal = theme.palette(cfg.get("ui.theme", "light"))
        theme.apply_pyqtgraph(pal, bool(cfg.get("ui.plot_antialias", True)))
        self.setStyleSheet(theme.stylesheet(cfg.get("ui.theme", "light")))

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab_acq = AcquisitionTab(self.ctx)
        self.tab_lib = LibraryTab(self.ctx)
        self.tab_review = ReviewTab(self.ctx)
        self.tab_analysis = AnalysisTab(self.ctx)
        self.tab_batch = BatchTab(self.ctx, self.tab_analysis)
        self.tab_settings = SettingsTab(self.ctx)

        self.tabs.addTab(self.tab_acq, "Acquisition")
        self.tabs.addTab(self.tab_lib, "Library")
        self.tabs.addTab(self.tab_review, "Review")
        self.tabs.addTab(self.tab_analysis, "Analysis")
        self.tabs.addTab(self.tab_batch, "Batch")
        self.tabs.addTab(self.tab_settings, "Settings")

        self._wire()
        self._build_menu()

        self.status = self.statusBar()
        self.status.showMessage("Ready")
        self._restore_geometry()

        if initial_video:
            self._open_video_path(initial_video)

    # ------------------------------------------------------------------
    def _wire(self) -> None:
        for tab in (self.tab_acq, self.tab_lib, self.tab_review,
                    self.tab_analysis, self.tab_batch, self.tab_settings):
            if hasattr(tab, "statusMessage"):
                tab.statusMessage.connect(self._status)

        # Library drives the other tabs.
        self.tab_lib.openInReview.connect(self._open_in_review)
        self.tab_lib.analyseRequested.connect(self._open_in_analysis)
        self.tab_lib.batchRequested.connect(self._add_to_batch)

        # A finished recording appears in the Library without a manual rescan.
        self.tab_acq.recordingFinished.connect(lambda _p: self.tab_lib.rescan())

        self.tab_settings.themeChanged.connect(self._apply_theme)

    def _build_menu(self) -> None:
        m_file = self.menuBar().addMenu("&File")
        act_open = m_file.addAction("&Open a video…")
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._open_video_dialog)
        act_rescan = m_file.addAction("&Rescan the library")
        act_rescan.setShortcut("F5")
        act_rescan.triggered.connect(self.tab_lib.rescan)
        m_file.addSeparator()
        act_save = m_file.addAction("&Save settings")
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._save_settings)
        m_file.addSeparator()
        act_quit = m_file.addAction("&Quit")
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)

        m_go = self.menuBar().addMenu("&Go")
        for i, (name, key) in enumerate((("Acquisition", "Ctrl+1"),
                                         ("Library", "Ctrl+2"),
                                         ("Review", "Ctrl+3"),
                                         ("Analysis", "Ctrl+4"),
                                         ("Batch", "Ctrl+5"),
                                         ("Settings", "Ctrl+6"))):
            a = m_go.addAction(name)
            a.setShortcut(key)
            a.triggered.connect(lambda _c=False, idx=i: self.tabs.setCurrentIndex(idx))

        m_tools = self.menuBar().addMenu("&Tools")
        a = m_tools.addAction("Make a synthetic benchmark video…")
        a.triggered.connect(self._make_benchmark)
        a2 = m_tools.addAction("Run the self-test")
        a2.triggered.connect(self._run_selftest)

        m_help = self.menuBar().addMenu("&Help")
        for label, fname in (("User manual", "USER_MANUAL.md"),
                             ("Installation guide", "INSTALLATION.md"),
                             ("Acquisition quick card", "ACQUISITION_QUICKCARD.md"),
                             ("Troubleshooting", "TROUBLESHOOTING.md"),
                             ("What changed from the first pipeline",
                              "PIPELINE_REVIEW.md")):
            act = m_help.addAction(label)
            act.triggered.connect(lambda _c=False, f=fname: self._open_doc(f))
        m_help.addSeparator()
        m_help.addAction("About").triggered.connect(self._about)

    # ------------------------------------------------------------------
    def _status(self, msg: str) -> None:
        self.status.showMessage(msg, 15000)
        log.info("status: %s", msg)

    def _open_in_review(self, entry: library.Entry) -> None:
        self.ctx.entry = entry
        self.tab_review.load_entry(entry)
        self.tab_analysis.set_entry(entry)
        self.tabs.setCurrentWidget(self.tab_review)

    def _open_in_analysis(self, entry: library.Entry) -> None:
        self.ctx.entry = entry
        self.tab_analysis.set_entry(entry)
        self.tabs.setCurrentWidget(self.tab_analysis)

    def _add_to_batch(self, entries: list) -> None:
        self.tab_batch.add_entries(entries)
        self.tabs.setCurrentWidget(self.tab_batch)

    def _open_video_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open a video", str(self.ctx.cfg.recordings_root()),
            "Videos (*.avi *.mp4 *.mkv *.mov);;All files (*)")
        if path:
            self._open_video_path(path)

    def _open_video_path(self, path: str) -> None:
        from .. import ffmpeg_tools
        p = Path(path)
        if not p.exists():
            W.warn_box(self, "Not found", f"{p} does not exist.")
            return
        e = library.Entry(path=p)
        info = ffmpeg_tools.probe_video(p, self.ctx.cfg.get("paths.ffprobe", ""),
                                        count_frames=False)
        e.width, e.height, e.codec = info.width, info.height, info.codec
        e.duration_s, e.fps, e.fps_exact = info.duration_s, info.true_fps, info.exact
        try:
            e.size_bytes = p.stat().st_size
        except OSError:
            pass
        e.sidecar = library.read_json(library.sidecar_for(p))
        e.result = library.read_json(library.result_for(p))
        self._open_in_review(e)

    def _save_settings(self) -> None:
        self.tab_acq.save_to_config()
        self.tab_analysis.save_to_config()
        self.ctx.cfg.set("ui.last_tab", self.tabs.currentIndex())
        self.ctx.cfg.set("ui.window_geometry",
                         bytes(self.saveGeometry().toHex()).decode("ascii"))
        try:
            self.ctx.cfg.save()
            self._status(f"Settings saved to {self.ctx.cfg.file}")
        except OSError as e:
            W.warn_box(self, "Could not save settings", str(e))

    def _apply_theme(self, name: str) -> None:
        self.setStyleSheet(theme.stylesheet(name))
        pal = theme.palette(name)
        theme.apply_pyqtgraph(pal, bool(self.ctx.cfg.get("ui.plot_antialias", True)))
        self._status(f"Theme set to {name}. Reopen a recording to redraw the plots "
                     "in the new palette.")

    def _restore_geometry(self) -> None:
        geo = str(self.ctx.cfg.get("ui.window_geometry", "") or "")
        if geo:
            try:
                self.restoreGeometry(QtCore.QByteArray.fromHex(geo.encode("ascii")))
            except (ValueError, TypeError):
                pass
        idx = int(self.ctx.cfg.get("ui.last_tab", 0) or 0)
        if 0 <= idx < self.tabs.count():
            self.tabs.setCurrentIndex(idx)

    def _open_doc(self, name: str) -> None:
        p = paths.DOCS_DIR / name
        if not p.exists():
            W.warn_box(self, "Document missing",
                       f"{p} is not in this installation.")
            return
        webbrowser.open(p.as_uri())

    def _make_benchmark(self) -> None:
        """Generate the synthetic benchmark so the install can be verified."""
        out = paths.DATA_DIR / "benchmark"
        if not W.ask_yes_no(
                self, "Make a synthetic benchmark?",
                "This renders a 45-second 320x240 video at 240 fps with six known "
                "head twitches and three kinds of confounder (grooming, fast turns, "
                "and a stretch of simulated PWM lighting flicker), plus a "
                "ground-truth CSV.\n\n"
                f"It goes to {out} and takes a couple of minutes.\n\n"
                "Analyse it and use 'Validate against ground truth' on the Analysis "
                "tab to confirm the installation detects all six and none of the "
                "confounders."):
            return
        import subprocess
        import sys as _sys
        script = paths.INSTALL_DIR / "tools" / "make_synthetic_benchmark.py"
        try:
            subprocess.Popen([_sys.executable, str(script), "--out", str(out)])
        except OSError as e:
            W.warn_box(self, "Could not start", str(e))
            return
        self._status(f"Benchmark generation started; the files will appear in {out}")

    def _run_selftest(self) -> None:
        import subprocess
        import sys as _sys
        script = paths.INSTALL_DIR / "tools" / "selftest.py"
        try:
            r = subprocess.run([_sys.executable, str(script)], capture_output=True,
                               text=True, timeout=600)
        except (OSError, subprocess.SubprocessError) as e:
            W.warn_box(self, "Could not run the self-test", str(e))
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Self-test")
        dlg.resize(900, 560)
        v = QtWidgets.QVBoxLayout(dlg)
        txt = QtWidgets.QPlainTextEdit()
        txt.setReadOnly(True)
        txt.setPlainText((r.stdout or "") + ("\n" + r.stderr if r.stderr else ""))
        v.addWidget(txt)
        b = W.button("Close", "primary")
        b.clicked.connect(dlg.accept)
        v.addWidget(b)
        dlg.exec_()

    def _about(self) -> None:
        W.info_box(
            self, "About HTR Studio",
            f"HTR Studio {__version__}\n\n"
            "Acquisition and analysis of the mouse head-twitch response, built "
            "around the Arducam ezBOX GS B0634 (OV9281 global shutter) at "
            "320x240 / 240 fps.\n\n"
            "Detection follows the published description of the response: 5-11 "
            "oscillations at 78-98 Hz (Halberstadt et al., J Neurosci Methods, "
            "2014), band-passed at 60-110 Hz with the threshold taken from the "
            "pre-dose baseline.\n\n"
            f"Installed at {paths.INSTALL_DIR}\n"
            f"Settings: {paths.SETTINGS_FILE}\n"
            f"Log: {paths.LOGS_DIR / 'htr_studio.log'}\n\n"
            "Help menu has the user manual and the review of what changed from "
            "the first-generation pipeline.")

    # ------------------------------------------------------------------
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.tab_acq.rec is not None and self.tab_acq.rec.running:
            if not W.ask_yes_no(
                    self, "A recording is running",
                    "Closing now stops the recording. The file is closed cleanly, "
                    "but the trial will be short. Close anyway?"):
                event.ignore()
                return
        self._save_settings()
        for tab in (self.tab_acq, self.tab_lib, self.tab_review,
                    self.tab_analysis, self.tab_batch, self.tab_settings):
            try:
                tab.shutdown()
            except Exception:  # pragma: no cover
                log.exception("error shutting down %s", type(tab).__name__)
        # Anything a tab could not stop in time was detached rather than
        # destroyed; join it here, because the process is about to exit and a
        # thread still running at interpreter teardown is the same abort by a
        # slower route.
        try:
            W.JobRunner.join_all()
        except Exception:  # pragma: no cover
            log.exception("error joining detached workers")
        event.accept()
