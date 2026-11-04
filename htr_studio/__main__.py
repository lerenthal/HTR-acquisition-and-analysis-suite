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

    def excepthook(exc_type, exc, tb):
        """Never let a stray exception silently close the window mid-trial."""
        import traceback
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        log.error("unhandled exception:\n%s", text)
        try:
            QtWidgets.QMessageBox.critical(
                None, "Unexpected error",
                f"{exc}\n\nThe full traceback is in\n"
                f"{paths.LOGS_DIR / 'htr_studio.log'}\n\n"
                "If a recording was running it has not been stopped; check the "
                "Acquisition tab.")
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
