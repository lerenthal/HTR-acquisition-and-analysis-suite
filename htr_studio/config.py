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
    "schema_version": 2,
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
    "analysis": {
        # ---- segmentation of the mouse ----
        "polarity": "auto",  # auto | dark_on_light | light_on_dark
        "threshold_mode": "otsu",  # otsu | fixed
        "threshold_value": 100,
        "min_area_frac": 0.004,  # fraction of frame area
        "max_area_frac": 0.45,
        "morph_kernel": 5,
        # ---- head angle estimator ----
        "angle_source": "headpoint",  # headpoint | ellipse | moments
        "max_gap_ms": 60.0,  # interpolate invalid runs shorter than this
        # ---- detector ----
        "detector": "angle_bandpass",  # angle_bandpass | motion_spectral | both_agree | either
        "band_low_hz": 60.0,
        "band_high_hz": 110.0,
        "filter_order": 4,
        "envelope_smooth_ms": 15.0,
        "threshold_source": "baseline",  # baseline | whole | manual
        "threshold_k": 6.0,  # robust sigma multiplier
        "threshold_manual": 0.0,
        "threshold_low_frac": 0.5,
        "baseline_window_s": [0.0, 0.0],  # [0,0] -> use recording.baseline_seconds / whole file
        "merge_gap_ms": 60.0,
        "min_burst_ms": 45.0,
        "max_burst_ms": 600.0,
        "require_freq_hz": [55.0, 125.0],  # dominant in-burst frequency gate
        "min_half_cycles": 6,  # Halberstadt 2014: 5-11 oscillations per burst
        "max_invalid_frac": 0.25,  # reject bursts sitting on lost tracking
        # motion-spectral detector
        "spec_window_ms": 256.0,
        "spec_overlap": 0.75,
        "spec_signal_band": [70.0, 110.0],
        "spec_noise_band": [8.0, 50.0],
        "spec_ratio_k": 5.0,
        "spec_power_k": 6.0,
        "spec_smooth_steps": 3,
        # ---- performance ----
        "workers": 0,  # 0 = auto
        "max_frames": 0,  # 0 = no cap (streaming, so RAM is not a constraint)
        "progress_every": 2000,
    },
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


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    """Dotted-path access to the settings document."""

    def __init__(self, data: dict[str, Any] | None = None, file: Path | None = None):
        self.file = file or paths.SETTINGS_FILE
        self.data = _deep_merge(DEFAULTS, data or {})

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
        """Dump the shipped defaults next to settings.json for reference."""
        try:
            paths.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            paths.DEFAULT_SETTINGS_FILE.write_text(
                json.dumps(DEFAULTS, indent=2), encoding="utf-8"
            )
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
