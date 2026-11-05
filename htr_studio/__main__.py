"""Entry point: ``python -m htr_studio [video]``.

``run_htr_studio.bat`` calls this through the virtual environment that
``install.bat`` created.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="htr_studio",
        description="HTR Studio - mouse head-twitch response acquisition and analysis")
    ap.add_argument("video", nargs="?", default=None,
                    help="optional video to open in the Review tab at startup")
    ap.add_argument("--debug", action="store_true", help="verbose console logging")
    ap.add_argument("--selftest", action="store_true",
                    help="run the self-test and exit, without opening a window")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    from . import config, logging_setup, paths

    paths.ensure_dirs()
    logging_setup.setup(logging.DEBUG if args.debug else logging.INFO)
    log = logging.getLogger("htr_studio")

    cfg = config.Config.load()
    # Persist a schema migration the first time it runs. Config.load applies
    # it in memory on every start, so the app always uses the corrected
    # values either way -- but leaving the file saying 0.45 while the app
    # uses 0.25 makes settings.json a liar, and repeats the migration warning
    # in the log every launch. Written once, on the launch that migrated.
    if cfg.migrated_from is not None:
        log.warning("settings file migrated from schema %s to %s; saving",
                    cfg.migrated_from, config.SCHEMA_VERSION)
        try:
            cfg.save()
        except OSError as e:          # read-only install: carry on regardless
            log.warning("could not save the migrated settings: %s", e)
    cfg.write_defaults_reference()
    config.write_capture_presets_reference()

    if args.selftest:
        from tools import selftest  # type: ignore[import-not-found]
        return selftest.main([])

    try:
        from PyQt5 import QtCore, QtWidgets
    except ImportError as e:
        print("PyQt5 is not available in this environment.\n"
              f"  {e}\n"
              "Run install.bat, then start the app with run_htr_studio.bat.",
              file=sys.stderr)
        return 2

    # Qt's high-DPI handling has to be set before the application exists.
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("HTR Studio")
    app.setOrganizationName("HTR Studio")

    from .ui.main_window import MainWindow

    # Rate limiting, per error signature.
    #
    # The previous handler showed one modal box per exception. Anything raised
    # from a timer callback therefore produced one box per tick, and that is
    # how an off-by-one in the preview histogram became an application that
    # could not be used at all: boxes appeared faster than they could be
    # dismissed, so the Acquisition tab could not be reached to stop a
    # recording. A bug in a cosmetic readout must not be able to do that.
    #
    # So: the first occurrence of each distinct error is shown, repeats are
    # counted and logged, and the same error is not shown again for a minute.
    _seen: dict[str, float] = {}
    _repeats: dict[str, int] = {}
    QUIET_FOR_S = 60.0

    def excepthook(exc_type, exc, tb):
        """Never let a stray exception silently close the window mid-trial."""
        import time
        import traceback
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        frames = traceback.extract_tb(tb)
        where = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "?"
        sig = f"{exc_type.__name__}@{where}"
        now = time.monotonic()
        last = _seen.get(sig)
        _repeats[sig] = _repeats.get(sig, 0) + 1
        if last is not None and now - last < QUIET_FOR_S:
            log.error("unhandled exception (%s, repeat %d, dialog suppressed): %s",
                      sig, _repeats[sig], exc)
            return
        _seen[sig] = now
        log.error("unhandled exception (%s):\n%s", sig, text)

        # Only claim a recording is running if one actually is: the old text
        # said so unconditionally, which sent people to the Acquisition tab
        # after an error that had nothing to do with recording.
        recording = False
        try:
            w = QtWidgets.QApplication.activeWindow()
            for top in QtWidgets.QApplication.topLevelWidgets():
                rec = getattr(getattr(top, "tab_acq", None), "rec", None)
                if rec is not None and getattr(rec, "running", False):
                    recording = True
                    break
            del w
        except Exception:  # pragma: no cover - never fail inside the handler
            log.exception("could not determine the recording state")

        repeats = _repeats[sig]
        extra = (f"\n\nThis has now happened {repeats} times; further copies "
                 "are being logged instead of shown." if repeats > 1 else "")
        try:
            box = QtWidgets.QMessageBox(
                QtWidgets.QMessageBox.Critical, "Unexpected error",
                f"{exc}\n\nThe full traceback is in\n"
                f"{paths.LOGS_DIR / 'htr_studio.log'}\n\n"
                + ("A recording IS still running - it has not been stopped. "
                   "Go to the Acquisition tab." if recording else
                   "No recording is running.")
                + extra)
            copy = box.addButton("Copy details",
                                 QtWidgets.QMessageBox.ActionRole)
            box.addButton(QtWidgets.QMessageBox.Ok)
            box.exec_()
            if box.clickedButton() is copy:
                QtWidgets.QApplication.clipboard().setText(text)
        except Exception:  # pragma: no cover
            pass

    sys.excepthook = excepthook

    win = MainWindow(cfg, initial_video=args.video)
    win.show()
    log.info("HTR Studio started")
    return app.exec_()


if __name__ == "__main__":
    # Required so a frozen or spawned child process does not re-enter the GUI.
    multiprocessing.freeze_support()
    sys.exit(main())
