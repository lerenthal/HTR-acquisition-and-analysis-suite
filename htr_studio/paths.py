"""Portable path resolution.

Everything lives relative to the installation folder so the whole tree can be
copied to another Windows machine (or a USB drive) and still work. Nothing is
written to the registry and nothing outside the install folder is required,
except the optional user-chosen recordings root.

Layout::

    windows_HTR/                  <- INSTALL_DIR
      htr_studio/                 <- this package
      config/
        settings.json             <- live settings (created on first run)
        default_settings.json     <- shipped defaults (read-only reference)
        capture_presets.json
      tools/
        bin/uv.exe                <- bootstrapped by install.bat
        ffmpeg/bin/ffmpeg.exe     <- bootstrapped by install.bat
      data/
        recordings/               <- default recordings root
        exports/                  <- CSV / PNG reports
        logs/
        cache/                    <- thumbnails, analysis caches
      .venv/                      <- created by install.bat
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

#: Root of the installation (the folder that contains ``htr_studio``).
INSTALL_DIR: Path = Path(__file__).resolve().parent.parent

CONFIG_DIR: Path = INSTALL_DIR / "config"
SETTINGS_FILE: Path = CONFIG_DIR / "settings.json"
DEFAULT_SETTINGS_FILE: Path = CONFIG_DIR / "default_settings.json"
CAPTURE_PRESETS_FILE: Path = CONFIG_DIR / "capture_presets.json"

TOOLS_DIR: Path = INSTALL_DIR / "tools"
TOOLS_BIN: Path = TOOLS_DIR / "bin"
FFMPEG_DIR: Path = TOOLS_DIR / "ffmpeg"

DATA_DIR: Path = INSTALL_DIR / "data"
RECORDINGS_DIR: Path = DATA_DIR / "recordings"
EXPORTS_DIR: Path = DATA_DIR / "exports"
LOGS_DIR: Path = DATA_DIR / "logs"
CACHE_DIR: Path = DATA_DIR / "cache"

DOCS_DIR: Path = INSTALL_DIR / "docs"

#: Suffix for the per-recording metadata sidecar.
SIDECAR_SUFFIX = ".htr.json"
#: Suffix for the per-recording analysis result file.
RESULT_SUFFIX = ".htrresult.json"
#: Name of the manifest written into a segmented-session folder.
SESSION_MANIFEST = "session.htrsession.json"


def ensure_dirs() -> None:
    """Create the writable data directories if they do not exist."""
    for d in (CONFIG_DIR, DATA_DIR, RECORDINGS_DIR, EXPORTS_DIR, LOGS_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def is_windows() -> bool:
    return os.name == "nt" or sys.platform.startswith("win")


def exe(name: str) -> str:
    """Append ``.exe`` on Windows so the same code works when testing on Linux."""
    return f"{name}.exe" if is_windows() else name


def short(path: os.PathLike | str, max_len: int = 64) -> str:
    """Middle-truncate a path for display in a status bar."""
    s = str(path)
    if len(s) <= max_len:
        return s
    keep = (max_len - 3) // 2
    return f"{s[:keep]}...{s[-keep:]}"
