"""Settings: shipped defaults, user overrides, deep-merge, atomic save.

The whole configuration is one JSON document. ``DEFAULTS`` below is the
authoritative schema; ``config/settings.json`` only has to carry the keys the
user actually changed, so adding a new key in a later version does not break an
existing installation.
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

from . import paths

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
# Rationale for the scientific defaults is in docs/PIPELINE_REVIEW.md.
DEFAULTS: dict[str, Any] = {
    "schema_version": 3,
    "camera": {
        # Substring matched case-insensitively against the dshow device names.
        "device_name": "Arducam ezBOX VGA GS Mono",
        # OpenCV index used for the live preview (-1 = auto-detect).
        "opencv_index": -1,
        # Preferred OpenCV backend for preview: "dshow" | "msmf" | "any".
        "opencv_backend": "dshow",
        "preset": "320x240@200 (240 fps actual) - production",
        "width": 320,
        "height": 240,
        "requested_fps": 200,
        "pixel_format": "mjpeg",
        # UVC controls. OpenCV's DSHOW exposure scale on OV9281 UVC is
        # logarithmic: -7 is roughly 1/128 s -> ~1 ms at 240 fps.
        "auto_exposure": False,
        "exposure": -7,
        "gain": 0,
        "brightness": 0,
        "contrast": 32,
        "gamma": 100,
        "apply_uvc_before_record": True,
        # Live preview. "capture" runs the preview through FFmpeg with the
        # recorder's own DirectShow arguments, so the picture is the one that
        # will be written -- same media type, frame rate and exposure window.
        # "opencv" is the responsive preview, which on this camera negotiates
        # YUY2 at a different rate and so shows a different brightness.
        "preview_mode": "capture",
        # Frames per second shipped down the preview pipe. The camera still
        # runs at the capture rate; this only decimates what is displayed.
        "preview_display_fps": 25,
        # FFmpeg holds the device, so applying a UVC control means closing and
        # reopening it. A slider drag emits a change per pixel of travel, so
        # the reopen waits this long after the last one.
        "preview_reopen_debounce_ms": 500,
        "flicker_seconds": 2.0,
        "mains_hz": 50.0,
        "rtbufsize": "512M",
        # FFmpeg's own default is 8 packets, which at 240 fps is 33 ms of
        # tolerance for a disk stall - less than a single NTFS flush. This is
        # the setting that stops "Thread message queue blocking" and the frame
        # loss that comes with it on long high-rate captures.
        "thread_queue_size": 4096,
    },
    "recording": {
        "root": "",  # empty -> paths.RECORDINGS_DIR
        "container": "avi",
        "name_template": "{animal}_{group}_{date}_{time}",
        "segment_enabled": True,
        "segment_seconds": 300,
        "total_seconds": 4200,  # 70 min = 10 min baseline + 60 min post-gavage
        "baseline_seconds": 600,
        "min_free_gb": 30.0,
        "write_sidecar": True,
        "auto_verify_after_record": True,
    },
    # "analysis" is filled in below from analysis.AnalysisConfig itself --
    # see _analysis_defaults(). It is NOT written out by hand here.
    "review": {
        "frame_cache": 240,
        "overlay_tracking": True,
        "overlay_trace_seconds": 2.0,
    },
    "report": {
        "bin_seconds": 60,
        "t0_relative": True,
        "export_plots": True,
        "export_signals": False,  # full per-frame traces (large)
    },
    "ui": {
        "theme": "light",
        "last_tab": 0,
        "window_geometry": "",
        "plot_antialias": True,
    },
    "paths": {
        "ffmpeg": "",  # empty -> auto-discover
        "ffprobe": "",
        "dlc_python": "",  # .venv-dlc interpreter, filled by install_dlc.bat
    },
}


def _analysis_defaults() -> dict:
    """The analysis defaults, taken from the dataclass that documents them.

    This used to be a hand-written copy of :class:`analysis.AnalysisConfig`'s
    field defaults, and it drifted, which is what a duplicated constant
    always eventually does. The copy still carried ``max_area_frac: 0.45``
    after the dataclass had been lowered to 0.25 -- 0.45 being the ceiling
    that let a 33% blob, the lit arena floor, be accepted as the animal. So a
    fresh install wrote 0.45 into its settings file, the app read the settings
    file in preference to the dataclass, and the fix was only half applied;
    "Reset to defaults" then restored the broken value. Eight other fields
    were simply missing, so they only ever got their dataclass value by
    accident of absence.

    Deriving it removes the possibility: there is one place a default lives,
    next to the comment explaining why it has that value.
    """
    from .analysis import AnalysisConfig          # local: avoids a cycle
    d = AnalysisConfig().to_dict()
    # progress_every is a config-only knob; to_dict already carries it.
    return d


DEFAULTS["analysis"] = _analysis_defaults()


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


SCHEMA_VERSION = 3

#: Values a stored settings file may hold that are now known to be wrong,
#: with the version that corrects them. Each entry is
#: ``(dotted key, exact stored value to replace, new value, why)``. Matching
#: on the *exact* old default is deliberate: an operator who deliberately
#: chose some other number keeps it, while the one who simply inherited a
#: bad default gets the fix.
_CORRECTIONS = [
    ("analysis.max_area_frac", 0.45, 0.25,
     "0.45 let a blob covering nearly half the frame be accepted as the "
     "animal, which is how the lit arena floor (33% of frame, perfectly "
     "stable) came to be tracked instead of the mouse. The real animal "
     "measures 0.9-4% of the frame, so 0.25 is still several times its size."),
    ("analysis.core_kernel", 9, 0,
     "a fixed 9 px tail-suppression kernel is right for a 320x240 frame and "
     "far too small for a larger one, where it left the tail attached and "
     "put the head arrow on the wrong end. 0 means measure it from the "
     "animal's own width."),
]


def _migrate(raw: dict[str, Any]) -> tuple[dict[str, Any], int | None]:
    """Bring a stored settings document up to the current schema.

    Needed because a stored value wins over a default, which is normally
    right -- it is how an operator's choices survive an upgrade -- but means
    a default corrected in the code does not reach a rig that already has a
    settings file. The arena-floor fix was exactly that: the dataclass went
    to 0.25, the installed settings.json kept 0.45, and the rig carried on
    with the broken ceiling while the code claimed to have fixed it.

    Returns the document and, when a migration happened, the version it came
    from -- so the caller can persist it once rather than re-running it and
    re-logging it on every launch.
    """
    if not raw:
        return raw, None
    out = copy.deepcopy(raw)
    was = int(out.get("schema_version", 0) or 0)
    if was >= SCHEMA_VERSION:
        return out, None
    for dotted, old, new, why in _CORRECTIONS:
        parts = dotted.split(".")
        node = out
        for p in parts[:-1]:
            if not isinstance(node, dict) or p not in node:
                node = None
                break
            node = node[p]
        if not isinstance(node, dict):
            continue
        key = parts[-1]
        if key in node and node[key] == old:
            node[key] = new
            log.warning("settings migration: %s %r -> %r. %s",
                        dotted, old, new, why)
    out["schema_version"] = SCHEMA_VERSION
    return out, was


class Config:
    """Dotted-path access to the settings document."""

    def __init__(self, data: dict[str, Any] | None = None, file: Path | None = None):
        self.file = file or paths.SETTINGS_FILE
        migrated, from_version = _migrate(data or {})
        #: The schema version a migration ran from, or None if none did. The
        #: launcher saves once when this is set, so the stored file stops
        #: disagreeing with what the app is actually using.
        self.migrated_from: int | None = from_version
        self.data = _deep_merge(DEFAULTS, migrated)

    # ---- construction -------------------------------------------------
    @classmethod
    def load(cls, file: Path | None = None) -> "Config":
        f = file or paths.SETTINGS_FILE
        raw: dict[str, Any] = {}
        if f.exists():
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                log.warning("could not read %s (%s); falling back to defaults", f, e)
                raw = {}
        return cls(raw, file=f)

    # ---- access -------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def section(self, name: str) -> dict[str, Any]:
        v = self.get(name, {})
        return v if isinstance(v, dict) else {}

    # ---- persistence ---------------------------------------------------
    def save(self) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=False), encoding="utf-8")
        tmp.replace(self.file)
        log.info("settings saved to %s", self.file)

    def reset_section(self, name: str) -> None:
        if name in DEFAULTS:
            self.data[name] = copy.deepcopy(DEFAULTS[name])

    # ---- derived -------------------------------------------------------
    def recordings_root(self) -> Path:
        r = (self.get("recording.root") or "").strip()
        return Path(r) if r else paths.RECORDINGS_DIR

    def write_defaults_reference(self) -> None:
        """Dump the shipped defaults next to settings.json for reference.

        Rewritten whenever the stored version is older than the current
        schema, not only when the file is absent. The shipped copy in the
        repository had gone stale at schema 2 with ``max_area_frac: 0.45`` in
        it -- a read-only reference that documented the wrong value is worse
        than none, because it is the file someone checks when a setting
        looks odd.
        """
        try:
            paths.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            f = paths.DEFAULT_SETTINGS_FILE
            if f.exists():
                try:
                    old = json.loads(f.read_text(encoding="utf-8"))
                    if int(old.get("schema_version", 0) or 0) >= SCHEMA_VERSION:
                        return
                except (OSError, json.JSONDecodeError):
                    pass
            f.write_text(json.dumps(DEFAULTS, indent=2), encoding="utf-8")
        except OSError:
            pass


# --------------------------------------------------------------------------
# Capture presets
# --------------------------------------------------------------------------
# Measured on the host that validated this pipeline (see MOUSE_HTR_PROTOCOL_SOP
# section 1.4). The B0634's MJPEG encoder runs on a fixed clock, so the
# delivered rate is not always the requested rate.
CAPTURE_PRESETS: list[dict[str, Any]] = [
    {
        "name": "320x240@200 (240 fps actual) - production",
        "width": 320,
        "height": 240,
        "requested_fps": 200,
        "expected_fps": 240,
        "pixel_format": "mjpeg",
        "note": "Validated: 7209 frames / 30.01 s = 240.3 fps, zero drops. "
                "Nyquist 120 Hz covers the 78-98 Hz HTR band. Recommended.",
    },
    {
        "name": "640x480@120 (~100 fps actual) - high spatial detail",
        "width": 640,
        "height": 480,
        "requested_fps": 120,
        "expected_fps": 100,
        "pixel_format": "mjpeg",
        "note": "MJPEG encoder saturates near 100 fps at VGA. Nyquist 50 Hz is "
                "BELOW the HTR band - the 90 Hz oscillation will alias. Use only "
                "for pose-model training frames or non-HTR scoring.",
    },
    {
        "name": "640x480@100 - VGA baseline",
        "width": 640,
        "height": 480,
        "requested_fps": 100,
        "expected_fps": 100,
        "pixel_format": "mjpeg",
        "note": "Same aliasing caveat as above. 3002 frames / 30 s verified.",
    },
    {
        "name": "160x120@400 - maximum temporal fidelity",
        "width": 160,
        "height": 120,
        "requested_fps": 400,
        "expected_fps": 400,
        "pixel_format": "mjpeg",
        "note": "Experimental. Very coarse spatially; blob tracking may fail on "
                "a 15-20 cm arena. Probe before trusting.",
    },
    {
        "name": "320x240@100 - YUV uncompressed",
        "width": 320,
        "height": 240,
        "requested_fps": 100,
        "expected_fps": 100,
        "pixel_format": "yuyv422",
        "note": "No MJPEG compression artifacts, but limited to 100 fps and "
                "~6x the disk rate. Aliases the HTR band.",
    },
]


def load_capture_presets() -> list[dict[str, Any]]:
    """Presets from config/capture_presets.json, falling back to the builtins."""
    f = paths.CAPTURE_PRESETS_FILE
    if f.exists():
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return data
        except (OSError, json.JSONDecodeError) as e:
            log.warning("bad capture_presets.json (%s); using builtins", e)
    return copy.deepcopy(CAPTURE_PRESETS)


def write_capture_presets_reference() -> None:
    try:
        paths.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if not paths.CAPTURE_PRESETS_FILE.exists():
            paths.CAPTURE_PRESETS_FILE.write_text(
                json.dumps(CAPTURE_PRESETS, indent=2), encoding="utf-8"
            )
    except OSError:
        pass
