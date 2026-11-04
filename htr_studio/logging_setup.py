"""Rotating file log plus console output.

The log file is the first thing to look at when something goes wrong on the
analysis computer; TROUBLESHOOTING.md points the user at it.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys

from . import paths

_CONFIGURED = False
FMT = "%(asctime)s %(levelname)-7s %(name)-24s %(message)s"


def setup(level: int = logging.INFO, console: bool = True) -> logging.Logger:
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        return root

    paths.ensure_dirs()
    root.setLevel(logging.DEBUG)

    try:
        fh = logging.handlers.RotatingFileHandler(
            paths.LOGS_DIR / "htr_studio.log",
            maxBytes=2_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(FMT))
        root.addHandler(fh)
    except OSError:
        pass  # read-only install folder: carry on with console only

    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(ch)

    # Qt and matplotlib are chatty at DEBUG.
    for noisy in ("matplotlib", "PIL", "PyQt5"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    root.info("--- HTR Studio log start (install=%s) ---", paths.INSTALL_DIR)
    return root
