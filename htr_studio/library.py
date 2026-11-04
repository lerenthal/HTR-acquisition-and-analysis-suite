"""The video library: discovery, metadata sidecars, session grouping.

A "recording" is one video file. A "session" is a folder of segments produced by
one segmented trial, held together by ``session.htrsession.json``; the Library
shows it as a single row because it is one animal and one dose, and the analysis
treats its segments as one continuous timeline.

Scanning never decodes video. ``ffprobe -count_frames`` on a 70-minute trial
takes minutes, so the fast scan reads the container header only and the exact
frame count is fetched on request (and then cached in the sidecar, so it is paid
for once).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import ffmpeg_tools, paths

log = logging.getLogger(__name__)

VIDEO_EXTS = {".avi", ".mp4", ".mkv", ".mov", ".m4v"}


# --------------------------------------------------------------------------
# Sidecars
# --------------------------------------------------------------------------
def sidecar_for(video: Path) -> Path:
    return video.with_name(video.name + paths.SIDECAR_SUFFIX)


def result_for(video: Path) -> Path:
    return video.with_name(video.name + paths.RESULT_SUFFIX)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.debug("could not read %s: %s", path, e)
        return {}


def write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------
# Entries
# --------------------------------------------------------------------------
@dataclass
class Entry:
    """One row in the Library: a single video, or a segmented session."""
    path: Path                     # the video file, or the session folder
    is_session: bool = False
    segments: list[Path] = field(default_factory=list)
    size_bytes: int = 0
    mtime: float = 0.0
    width: int = 0
    height: int = 0
    codec: str = ""
    duration_s: float = 0.0
    fps: float = 0.0
    fps_exact: bool = False
    sidecar: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    error: str = ""

    # ---- derived display fields ----
    @property
    def name(self) -> str:
        return self.path.name

    @property
    def animal(self) -> str:
        return str(self.trial.get("animal_id", "") or "")

    @property
    def group(self) -> str:
        return str(self.trial.get("group", "") or "")

    @property
    def treatment(self) -> str:
        return str(self.trial.get("treatment", "") or "")

    @property
    def dose(self) -> str:
        return str(self.trial.get("dose_mg_kg", "") or "")

    @property
    def route(self) -> str:
        return str(self.trial.get("route", "") or "")

    @property
    def trial(self) -> dict:
        t = self.sidecar.get("trial")
        return t if isinstance(t, dict) else {}

    @property
    def capture(self) -> dict:
        c = self.sidecar.get("capture")
        return c if isinstance(c, dict) else {}

    @property
    def t0_offset_s(self) -> float:
        try:
            return float(self.trial.get("t0_offset_s", -1.0))
        except (TypeError, ValueError):
            return -1.0

    @property
    def baseline_s(self) -> float:
        """Seconds of pre-dose recording, for deriving the detection threshold."""
        t0 = self.t0_offset_s
        if t0 > 0:
            return t0
        try:
            return float(self.trial.get("planned_baseline_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @property
    def expected_fps(self) -> float:
        try:
            return float(self.capture.get("expected_fps", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @property
    def analysed(self) -> bool:
        return bool(self.result)

    @property
    def n_bursts(self) -> int:
        if not self.result:
            return 0
        bursts = self.result.get("bursts") or []
        return sum(1 for b in bursts if b.get("accepted", True))

    @property
    def analysis_date(self) -> str:
        return str(self.result.get("analysed_iso", "") or "")

    @property
    def curated(self) -> bool:
        return bool(self.result.get("curated"))

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        if self.curated:
            return f"curated ({self.n_bursts})"
        if self.analysed:
            return f"analysed ({self.n_bursts})"
        return "new"

    @property
    def mtime_iso(self) -> str:
        try:
            return datetime.fromtimestamp(self.mtime).strftime("%Y-%m-%d %H:%M")
        except (OSError, OverflowError, ValueError):
            return ""

    @property
    def duration_hms(self) -> str:
        s = int(round(self.duration_s))
        return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

    @property
    def analysis_target(self) -> Path:
        """The file to feed to the analysis (the first segment for a session)."""
        if self.is_session and self.segments:
            return self.segments[0]
        return self.path

    def fps_for_analysis(self) -> float:
        """Best available true frame rate.

        Priority: an exact measured value cached in the sidecar, then the
        recorder's expected value, then the container header -- which OpenCV and
        AVI together get wrong often enough that it is the last resort.
        """
        meas = self.sidecar.get("measured")
        if isinstance(meas, dict):
            try:
                v = float(meas.get("true_fps", 0.0))
                if v > 1:
                    return v
            except (TypeError, ValueError):
                pass
        if self.expected_fps > 1:
            return self.expected_fps
        return self.fps


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------
def _entry_from_video(video: Path, cfg_paths: dict, deep: bool) -> Entry:
    e = Entry(path=video)
    try:
        stat = video.stat()
        e.size_bytes, e.mtime = stat.st_size, stat.st_mtime
    except OSError as err:
        e.error = str(err)
        return e
    e.sidecar = read_json(sidecar_for(video))
    e.result = read_json(result_for(video))
    info = ffmpeg_tools.probe_video(video, cfg_paths.get("ffprobe", ""),
                                    count_frames=deep,
                                    timeout=900 if deep else 60)
    if info.error:
        e.error = info.error
    e.width, e.height, e.codec = info.width, info.height, info.codec
    e.duration_s, e.fps, e.fps_exact = info.duration_s, info.true_fps, info.exact
    if deep and info.ok:
        cache_measured(video, info)
        e.sidecar = read_json(sidecar_for(video))
    return e


def cache_measured(video: Path, info: ffmpeg_tools.VideoInfo) -> None:
    """Store an exact frame count in the sidecar so it is measured only once."""
    sc = sidecar_for(video)
    doc = read_json(sc)
    doc.setdefault("htr_studio_sidecar_version", 1)
    doc.setdefault("kind", "recording")
    doc.setdefault("name", video.stem)
    doc["measured"] = {
        "frames": info.frames,
        "duration_s": round(info.duration_s, 4),
        "true_fps": round(info.true_fps, 4),
        "width": info.width,
        "height": info.height,
        "codec": info.codec,
        "exact": info.exact,
        "measured_iso": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    try:
        write_json(sc, doc)
    except OSError as e:  # pragma: no cover
        log.warning("could not write %s: %s", sc, e)


def _entry_from_session(folder: Path, manifest: dict, cfg_paths: dict,
                        deep: bool) -> Entry:
    e = Entry(path=folder, is_session=True, sidecar=manifest)
    segs = sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    e.segments = segs
    try:
        e.mtime = folder.stat().st_mtime
    except OSError:
        pass
    e.size_bytes = sum(p.stat().st_size for p in segs if p.exists())
    e.result = read_json(folder / (folder.name + paths.RESULT_SUFFIX))

    total = 0.0
    for i, p in enumerate(segs):
        # Probing every segment header is cheap; a deep probe of a whole session
        # would be minutes, so only the first segment is ever decoded.
        info = ffmpeg_tools.probe_video(p, cfg_paths.get("ffprobe", ""),
                                        count_frames=deep and i == 0,
                                        timeout=900 if deep else 60)
        if i == 0:
            e.width, e.height, e.codec = info.width, info.height, info.codec
            e.fps, e.fps_exact = info.true_fps, info.exact
            if info.error:
                e.error = info.error
            if deep and info.ok:
                cache_measured(p, info)
        total += info.duration_s
    e.duration_s = total
    return e


def scan(root: Path | str, deep: bool = False, cfg_paths: dict | None = None,
         max_entries: int = 5000) -> list[Entry]:
    """Walk ``root`` and return one Entry per recording or session.

    ``deep=True`` decodes each file to count frames exactly. Use it on a small
    selection, not on a whole library.
    """
    cfg_paths = cfg_paths or {}
    root = Path(root)
    entries: list[Entry] = []
    if not root.exists():
        return entries

    seen_sessions: set[Path] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        dirnames[:] = [x for x in dirnames if not x.startswith(".")]
        manifest_file = d / paths.SESSION_MANIFEST
        if manifest_file.exists():
            manifest = read_json(manifest_file)
            entries.append(_entry_from_session(d, manifest, cfg_paths, deep))
            seen_sessions.add(d)
            dirnames[:] = []  # a session folder is a leaf
            continue
        if any(parent in seen_sessions for parent in d.parents):
            continue
        for fn in sorted(filenames):
            p = d / fn
            if p.suffix.lower() not in VIDEO_EXTS:
                continue
            entries.append(_entry_from_video(p, cfg_paths, deep))
            if len(entries) >= max_entries:
                log.warning("library scan stopped at %d entries", max_entries)
                return entries
    entries.sort(key=lambda x: x.mtime, reverse=True)
    return entries


# --------------------------------------------------------------------------
# Metadata editing
# --------------------------------------------------------------------------
TRIAL_FIELDS = ("animal_id", "group", "treatment", "dose_mg_kg", "route",
                "operator", "notes", "t0_offset_s", "planned_baseline_s")


def update_trial_meta(entry: Entry, updates: dict) -> Path:
    """Write edited trial metadata back to the sidecar (or session manifest)."""
    if entry.is_session:
        sc = entry.path / paths.SESSION_MANIFEST
    else:
        sc = sidecar_for(entry.path)
    doc = read_json(sc)
    doc.setdefault("htr_studio_sidecar_version", 1)
    doc.setdefault("kind", "session" if entry.is_session else "recording")
    doc.setdefault("name", entry.path.stem)
    trial = doc.get("trial") if isinstance(doc.get("trial"), dict) else {}
    for k, v in updates.items():
        if k in TRIAL_FIELDS:
            trial[k] = v
    doc["trial"] = trial
    doc["edited_iso"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_json(sc, doc)
    entry.sidecar = doc
    return sc


def save_analysis(entry: Entry, doc: dict) -> Path:
    """Persist an analysis result next to its recording."""
    if entry.is_session:
        target = entry.path / (entry.path.name + paths.RESULT_SUFFIX)
    else:
        target = result_for(entry.path)
    doc = dict(doc)
    doc["analysed_iso"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_json(target, doc)
    entry.result = doc
    return target


def move_to_trash(entry: Entry) -> tuple[bool, str]:
    """Send a recording to the recycle bin, never an unrecoverable delete.

    Trials are expensive and irreplaceable; a mis-click in a file list should
    not be able to destroy one.
    """
    targets = entry.segments + [entry.path] if entry.is_session else [entry.path]
    extras: list[Path] = []
    for t in targets:
        if t.is_file():
            extras += [sidecar_for(t), result_for(t)]
    if entry.is_session:
        extras += [entry.path / paths.SESSION_MANIFEST,
                   entry.path / (entry.path.name + paths.RESULT_SUFFIX)]
    try:
        from send2trash import send2trash
    except ImportError:
        return False, ("Send2Trash is not installed, so nothing was deleted. "
                       "Remove the file in Explorer, or re-run install.bat.")
    moved = 0
    try:
        for p in [*(x for x in extras if x.exists()),
                  *(x for x in targets if x.exists())]:
            send2trash(str(p))
            moved += 1
    except OSError as e:
        return False, f"moved {moved} item(s) to the recycle bin, then failed: {e}"
    return True, f"moved {moved} item(s) to the recycle bin"


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------
def library_summary(entries: list[Entry]) -> dict:
    total_bytes = sum(e.size_bytes for e in entries)
    total_s = sum(e.duration_s for e in entries)
    return {
        "recordings": len(entries),
        "sessions": sum(1 for e in entries if e.is_session),
        "analysed": sum(1 for e in entries if e.analysed),
        "curated": sum(1 for e in entries if e.curated),
        "total_hours": round(total_s / 3600, 2),
        "total_gb": round(total_bytes / 1e9, 2),
        "animals": sorted({e.animal for e in entries if e.animal}),
        "groups": sorted({e.group for e in entries if e.group}),
    }
