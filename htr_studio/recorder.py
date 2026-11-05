"""Trial recording through FFmpeg, with live statistics and a metadata sidecar.

Why FFmpeg and not OpenCV: ``-c:v copy`` writes the camera's own MJPEG packets
to disk with no decode and no re-encode. That is what sustains 240 fps on this
camera; OpenCV's VideoWriter re-encodes and tops out near 100 fps.

Why segmenting is on by default: a 70-minute trial at 240 fps is a single ~25 GB
AVI. If anything interrupts it -- a USB hiccup, a full disk, someone closing the
app -- a single-file AVI can end up with a broken index and be awkward to
recover. Cutting it into 5-minute segments means an interruption costs at most
the last segment, and the segments are small enough to copy and to open. A
``session.htrsession.json`` manifest ties them back together so the Library and
the analysis treat the set as one trial.

What gets written next to the video, every time:

* ``<name>.htr.json``        -- trial metadata sidecar (animal, group, dose,
  route, T0, operator, camera settings, the exact FFmpeg command line)
* ``session.htrsession.json`` -- for a segmented trial, the segment list

Recording the provenance matters as much as recording the video: six weeks later
the question is always "what exposure was that trial shot at", and the answer
has to be in the folder, not in someone's memory.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import ffmpeg_tools, paths

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Trial metadata
# --------------------------------------------------------------------------
@dataclass
class TrialMeta:
    animal_id: str = ""
    group: str = ""
    treatment: str = ""
    dose_mg_kg: str = ""
    route: str = "oral gavage"
    operator: str = ""
    notes: str = ""
    # Seconds from the start of the recording to dose administration. -1 until
    # the operator presses "Mark T0"; the protocol's intent is 600 (10 min of
    # pre-dose baseline).
    t0_offset_s: float = -1.0
    planned_baseline_s: float = 600.0
    started_iso: str = ""
    ended_iso: str = ""

    def filled(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in ("", None)}


@dataclass
class RecordingSpec:
    device_name: str
    width: int = 320
    height: int = 240
    requested_fps: float = 200.0
    pixel_format: str = "mjpeg"
    duration_s: float = 4200.0
    segment_enabled: bool = True
    segment_seconds: float = 300.0
    out_dir: Path = field(default_factory=lambda: paths.RECORDINGS_DIR)
    base_name: str = "trial"
    container: str = "avi"
    #: DirectShow's own ring buffer, before FFmpeg sees a packet. At 240 fps a
    #: 320x240 MJPEG frame is ~2.5 kB, so 512M is minutes of slack -- generous
    #: on purpose, because the cost of over-allocating is RAM the machine has
    #: and the cost of under-allocating is a lost trial.
    rtbufsize: str = "512M"
    #: Packets queued between the demuxer thread and the muxer thread. FFmpeg's
    #: default is 8, which at 240 fps is 33 ms of tolerance -- less than a single
    #: NTFS write stall, which is why the default produces "Thread message queue
    #: blocking" and dropped frames on long high-rate captures. 4096 packets is
    #: ~17 s of 240 fps video and about 10 MB of RAM at this frame size.
    thread_queue_size: int = 4096
    camera_settings: dict = field(default_factory=dict)
    meta: TrialMeta = field(default_factory=TrialMeta)

    @property
    def is_segmented(self) -> bool:
        return self.segment_enabled and self.segment_seconds > 0 and \
            self.duration_s > self.segment_seconds

    def target_path(self) -> Path:
        if self.is_segmented:
            return self.out_dir / self.base_name / f"{self.base_name}_%04d.{self.container}"
        return self.out_dir / f"{self.base_name}.{self.container}"

    def estimated_bytes(self) -> int:
        """Rough size estimate for the pre-flight disk check.

        Measured reference: 320x240 MJPEG at 240 fps produced 17.7 MB for 30 s,
        i.e. about 590 kB/s, which is ~2.5 kB per frame. Scale that by the pixel
        count and the delivered rate.
        """
        per_frame = 2500 * (self.width * self.height) / (320 * 240)
        fps = self.expected_fps()
        return int(per_frame * fps * self.duration_s)

    def expected_fps(self) -> float:
        """What the camera actually delivers, which is not what we request."""
        if (self.width, self.height) == (320, 240) and self.requested_fps >= 200:
            return 240.0
        if (self.width, self.height) == (640, 480):
            return min(self.requested_fps, 100.0)
        return self.requested_fps


# --------------------------------------------------------------------------
# Pre-flight
# --------------------------------------------------------------------------
@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = False


def _disk_speed_check(spec: RecordingSpec) -> Check:
    """Measure the write path, because this is where long captures actually fail.

    240 fps of MJPEG is only ~600 kB/s, which any disk can sustain -- on paper.
    What breaks a capture is not throughput but a *stall*: an SMB round trip, a
    USB drive parking, a sync that takes 400 ms. The camera does not wait, so the
    frames arriving during the stall go into the real-time buffer and, if the
    stall is long enough, out of it again. So this writes a few megabytes in
    small pieces and reports the worst single pause, not the average rate.
    """
    name = "Write path"
    try:
        spec.out_dir.mkdir(parents=True, exist_ok=True)
        probe = spec.out_dir / ".htr_speed_test"
        chunk = b"\0" * (256 * 1024)
        worst, total = 0.0, 0.0
        with open(probe, "wb") as fh:
            for _ in range(24):                   # 6 MB in 256 kB pieces
                t = time.perf_counter()
                fh.write(chunk)
                fh.flush()
                os.fsync(fh.fileno())
                dt = time.perf_counter() - t
                worst = max(worst, dt)
                total += dt
        rate = (24 * 256 * 1024) / total / 1e6 if total > 0 else 0.0
        probe.unlink(missing_ok=True)
    except OSError as e:
        return Check(name, False, f"could not measure: {e}")

    need_mb_s = spec.estimated_bytes() / max(spec.duration_s, 1.0) / 1e6
    # The real-time buffer absorbs a stall; how long a one depends on its size.
    budget_s = 0.25
    ok = worst <= budget_s and rate >= max(need_mb_s * 3, 5.0)
    detail = (f"{rate:.0f} MB/s sustained, worst single write pause "
              f"{worst * 1000:.0f} ms. This capture needs {need_mb_s:.1f} MB/s.")
    if worst > budget_s:
        detail += (f"  A pause over {budget_s * 1000:.0f} ms is long enough to "
                   "overflow the capture buffer and lose frames. Record to a "
                   "local SSD, not a network share or a USB drive, and copy the "
                   "files afterwards.")
    elif not ok:
        detail += "  That is less headroom than a 30-minute capture wants."
    return Check(name, ok, detail)


def preflight(spec: RecordingSpec, cfg_paths: dict | None = None,
              min_free_gb: float = 30.0) -> list[Check]:
    """Everything worth knowing before committing to a 70-minute recording."""
    cfg_paths = cfg_paths or {}
    out: list[Check] = []

    ff = ffmpeg_tools.find_tool("ffmpeg", cfg_paths.get("ffmpeg", ""))
    out.append(Check("FFmpeg", ff is not None,
                     str(ff) if ff else "not found - run install.bat", fatal=True))
    fp = ffmpeg_tools.find_tool("ffprobe", cfg_paths.get("ffprobe", ""))
    out.append(Check("FFprobe", fp is not None,
                     str(fp) if fp else "not found - verification will be skipped"))

    out.append(Check("Camera selected", bool(spec.device_name),
                     spec.device_name or "no device chosen", fatal=True))

    try:
        spec.out_dir.mkdir(parents=True, exist_ok=True)
        probe = spec.out_dir / ".htr_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        out.append(Check("Output folder writable", True, str(spec.out_dir)))
    except OSError as e:
        out.append(Check("Output folder writable", False, f"{spec.out_dir}: {e}",
                         fatal=True))

    need = spec.estimated_bytes()
    try:
        free = shutil.disk_usage(spec.out_dir).free
        enough = free > max(need * 1.25, min_free_gb * 1e9 * 0.0)
        out.append(Check(
            "Disk space", enough,
            f"{free / 1e9:.1f} GB free, this trial needs about "
            f"{need / 1e9:.1f} GB ({spec.duration_s / 60:.0f} min at "
            f"{spec.expected_fps():.0f} fps)", fatal=not enough))
    except OSError as e:
        out.append(Check("Disk space", False, str(e)))

    out.append(_disk_speed_check(spec))

    if spec.is_segmented:
        out.append(Check(
            "Segmenting", True,
            f"{spec.duration_s / 60:.0f} min will be written as "
            f"{math.ceil(spec.duration_s / spec.segment_seconds)} files of "
            f"{spec.segment_seconds / 60:.0f} min "
            f"(~{spec.estimated_bytes() / max(1, math.ceil(spec.duration_s / spec.segment_seconds)) / 1e9:.2f} GB each). "
            "An interruption costs at most the last segment, and each file is "
            "small enough to copy and to open."))
    elif spec.duration_s > 600:
        out.append(Check(
            "Segmenting", False,
            f"off, so this will be one {spec.estimated_bytes() / 1e9:.1f} GB "
            "file. A single long AVI whose index is written at the end is "
            "unreadable if the capture is interrupted. Turn segmenting on."))

    target = spec.target_path()
    clash = (target.parent / target.name.replace("%04d", "0000")) if spec.is_segmented else target
    out.append(Check("Output name free", not clash.exists(),
                     "already exists - it would be overwritten" if clash.exists()
                     else str(clash)))

    if spec.meta.t0_offset_s < 0 and spec.meta.planned_baseline_s > 0:
        out.append(Check(
            "T0 (dose time)", True,
            f"not marked yet - press 'Mark T0' at gavage. Planned baseline is "
            f"{spec.meta.planned_baseline_s / 60:.0f} min; without a T0 the "
            "analysis falls back to that figure."))

    exp = spec.camera_settings.get("exposure")
    if spec.camera_settings.get("auto_exposure"):
        out.append(Check("Exposure mode", False,
                         "auto-exposure is ON. It hunts on dark frames and that "
                         "modulates brightness inside the HTR band. Switch to manual."))
    elif exp is not None:
        out.append(Check("Exposure mode", True, f"manual, value {exp}"))

    # A mode the camera does not offer is refused by DirectShow with a generic
    # I/O error, after the animal is already in the arena. Common sizes are
    # multiples of 8 or 16; an odd one is almost always a mis-scrolled spin box
    # (320x245 for 320x240 cost a trial on the reference rig).
    if spec.width % 8 or spec.height % 8:
        out.append(Check(
            "Capture size", False,
            f"{spec.width}x{spec.height} is not a size cameras normally offer - "
            "they are multiples of 8. Check the W and H boxes; a stray mouse "
            "wheel over them is the usual cause. Press 'List modes' to see what "
            "this camera actually supports.", fatal=True))

    if spec.expected_fps() < 200:
        out.append(Check(
            "Temporal resolution", False,
            f"{spec.expected_fps():.0f} fps gives a Nyquist limit of "
            f"{spec.expected_fps() / 2:.0f} Hz, below the 78-98 Hz HTR band, so the "
            "oscillation will alias. Use the 320x240 preset for 240 fps."))
    else:
        out.append(Check("Temporal resolution", True,
                         f"{spec.expected_fps():.0f} fps - Nyquist "
                         f"{spec.expected_fps() / 2:.0f} Hz covers the HTR band"))
    return out


# --------------------------------------------------------------------------
# Live statistics
# --------------------------------------------------------------------------
def dshow_input_args(spec: "RecordingSpec") -> list[str]:
    """The DirectShow input half of the FFmpeg command line.

    Shared by the recorder and the live preview on purpose. The preview's
    whole value is that it shows what the recording will look like, and it
    can only do that if it opens the camera with the same arguments -- same
    media type, same size, same ``-framerate``, which is what fixes the
    sensor's integration window and therefore the brightness. Two copies of
    this list would drift, and the first symptom of the drift would be a
    preview that lies, which is the problem it exists to solve.
    """
    args = ["-f", "dshow", "-rtbufsize", spec.rtbufsize,
            "-thread_queue_size", str(max(8, int(spec.thread_queue_size))),
            "-framerate", str(spec.requested_fps),
            "-video_size", f"{spec.width}x{spec.height}"]
    if spec.pixel_format in ("mjpeg", "mjpg"):
        args += ["-vcodec", "mjpeg"]
    else:
        args += ["-pixel_format", spec.pixel_format]
    return args + ["-i", f"video={spec.device_name}"]


def explain_exit(rc: int | None, stderr_lines: list[str] | None = None) -> str:
    """Turn an FFmpeg exit status into something an operator can act on.

    Windows reports a negative exit as a large unsigned number, so a DirectShow
    I/O failure arrives as 4294967291 -- a number with no meaning to anyone.
    The common cause is asking the camera for a mode it does not have, which
    DirectShow refuses with "Could not set video options" and then a generic
    I/O error, so the mode is worth naming in the message.
    """
    if rc is None:
        return "FFmpeg is still running."
    signed = rc - 2 ** 32 if rc > 2 ** 31 else rc
    text = " ".join(stderr_lines or []).lower()
    if "could not set video options" in text:
        return (f"FFmpeg could not open the camera (exit {signed}): the driver "
                "refused the requested capture mode. The size, frame rate or "
                "pixel format is not one this camera offers - press 'List "
                "modes' and pick a row, or use a preset.")
    if "i/o error" in text or signed == -5:
        return (f"FFmpeg could not open the camera (exit {signed}, I/O error). "
                "Either the requested mode is not supported, or something else "
                "holds the camera open - close Windows Camera and any preview.")
    if ("could not find video device" in text or "cannot find a device" in text
            or "no such file or directory" in text):
        return (f"FFmpeg could not find that camera (exit {signed}). The "
                "DirectShow name must match exactly, including trailing "
                "spaces - press Refresh and pick the device from the list "
                "rather than typing it.")
    if "device or resource busy" in text or "in use" in text:
        return (f"The camera is already in use (exit {signed}). Close the other "
                "program and try again.")
    if "no space left" in text:
        return f"The disk filled up (exit {signed}). The recording stopped."
    if "error opening input" in text or "unknown input format" in text:
        return (f"FFmpeg could not open the camera (exit {signed}). "
                + (f"It said: {stderr_lines[-1]}" if stderr_lines
                   else "See the log for its output."))
    if signed == 0:
        return "FFmpeg finished normally."
    # The last stderr line is almost always the useful one, and making the
    # operator go and find the log to see it is a step for no reason.
    return (f"FFmpeg exited with {signed}"
            + (f" (reported by Windows as {rc})" if signed != rc else "")
            + (f". It said: {stderr_lines[-1]}" if stderr_lines
               else ". See the log for its output."))


@dataclass
class RecordStats:
    frames: int = 0
    fps_instant: float = 0.0
    time_s: float = 0.0
    size_bytes: int = 0
    dropped: int = 0
    duplicated: int = 0
    speed: float = 0.0
    wall_elapsed_s: float = 0.0
    last_line: str = ""
    errors: list[str] = field(default_factory=list)
    #: Frames the capture layer threw away before FFmpeg could write them. These
    #: never appear in ``drop=``, which counts only frames the *muxer* dropped,
    #: so a recording can lose a third of its frames with drop=0 on screen. Each
    #: one is a hole in the angle trace, and a hole in a 240 fps trace is a step
    #: that the band-pass reads as a twitch.
    rt_buffer_drops: int = 0
    queue_blocked: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def effective_fps(self) -> float:
        return self.frames / self.time_s if self.time_s > 0 else 0.0

    @property
    def total_lost(self) -> int:
        return self.dropped + self.rt_buffer_drops

    def health(self, expected_fps: float) -> tuple[str, str]:
        """(level, one-line verdict) for the live indicator.

        Written as a verdict rather than a number because "drop=0" next to a
        real-time buffer that is discarding frames is exactly the reading that
        lets a bad trial through.
        """
        if self.rt_buffer_drops or self.queue_blocked:
            return ("bad",
                    f"The capture buffer overflowed {self.rt_buffer_drops} "
                    f"time(s): frames are being lost before they reach the file. "
                    "Stop, and either raise the buffer sizes in Settings or "
                    "record to a faster local disk.")
        if expected_fps > 0 and self.time_s > 2.0:
            got = self.effective_fps
            if got < expected_fps * 0.90:
                return ("bad", f"Writing {got:.0f} fps against an expected "
                               f"{expected_fps:.0f}. The camera is not "
                               "delivering the mode that was requested.")
            if got < expected_fps * 0.97:
                return ("warn", f"Writing {got:.0f} fps, a little under the "
                                f"expected {expected_fps:.0f}.")
        if self.duplicated:
            return ("warn", f"{self.duplicated} duplicated frame(s): the input "
                            "stalled and the timestamps were padded.")
        return ("ok", "Capture is keeping up.")


_PROG = re.compile(
    r"frame=\s*(?P<frame>\d+).*?fps=\s*(?P<fps>[\d.]+).*?"
    r"(?:size=\s*(?P<size>[\d.]+)(?P<unit>[kKmMgG]?)i?B.*?)?"
    r"time=(?P<h>\d+):(?P<m>\d+):(?P<s>[\d.]+)"
)
_DROP = re.compile(r"drop=\s*(\d+)")
_DUP = re.compile(r"dup=\s*(\d+)")
_SPEED = re.compile(r"speed=\s*([\d.]+)x")

#: FFmpeg's capture-side complaints. None of these contain the word "error", so
#: a plain keyword scan misses every one of them -- which is how a recording can
#: look clean on screen while throwing frames away.
_RT_FULL = re.compile(r"real-?time buffer.*?(?:too )?full", re.I)
_QUEUE = re.compile(r"thread message queue blocking|thread_queue_size", re.I)


def _parse_progress(line: str, st: RecordStats) -> bool:
    m = _PROG.search(line)
    if not m:
        return False
    st.frames = int(m.group("frame"))
    try:
        st.fps_instant = float(m.group("fps"))
    except (TypeError, ValueError):
        pass
    st.time_s = (int(m.group("h")) * 3600 + int(m.group("m")) * 60
                 + float(m.group("s")))
    if m.group("size"):
        mult = {"": 1, "k": 1024, "K": 1024, "m": 1024 ** 2, "M": 1024 ** 2,
                "g": 1024 ** 3, "G": 1024 ** 3}[m.group("unit") or ""]
        st.size_bytes = int(float(m.group("size")) * mult)
    for rx, attr in ((_DROP, "dropped"), (_DUP, "duplicated")):
        mm = rx.search(line)
        if mm:
            setattr(st, attr, int(mm.group(1)))
    ms = _SPEED.search(line)
    if ms:
        st.speed = float(ms.group(1))
    return True


# --------------------------------------------------------------------------
# Recorder
# --------------------------------------------------------------------------
class Recorder:
    """Runs one FFmpeg capture and reports progress.

    Thread model: FFmpeg runs as a child process; one reader thread drains its
    stderr into :class:`RecordStats`. The UI polls :meth:`stats` from a timer,
    so no Qt object is touched off the main thread.
    """

    def __init__(self, spec: RecordingSpec, cfg_paths: dict | None = None):
        self.spec = spec
        self.cfg_paths = cfg_paths or {}
        self.proc: subprocess.Popen | None = None
        self._stats = RecordStats()
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._t_start = 0.0
        self._stopping = False
        self.command: list[str] = []
        self.sidecar_path: Path | None = None
        self.on_finished: Callable[[int], None] | None = None

    # ---- command construction ----
    def build_command(self) -> list[str]:
        s = self.spec
        ff = ffmpeg_tools.ffmpeg_path(self.cfg_paths.get("ffmpeg", ""))
        target = s.target_path()
        target.parent.mkdir(parents=True, exist_ok=True)

        args = ([str(ff), "-hide_banner", "-loglevel", "info", "-stats"]
                + dshow_input_args(s)
                + ["-t", f"{s.duration_s:.3f}",
                   # Keep the camera's own packets: no decode, no re-encode.
                   "-c:v", "copy", "-an"])
        if s.is_segmented:
            args += ["-f", "segment",
                     "-segment_time", f"{s.segment_seconds:.3f}",
                     "-segment_format", s.container,
                     "-reset_timestamps", "1",
                     "-segment_list", str(target.parent / "segments.csv"),
                     "-segment_list_type", "csv"]
        args += ["-y", str(target)]
        return args

    # ---- lifecycle ----
    def start(self) -> None:
        if self.proc is not None:
            raise RuntimeError("recorder already running")
        self.command = self.build_command()
        self.spec.meta.started_iso = datetime.now().astimezone().isoformat(timespec="seconds")
        log.info("recording: %s", " ".join(self.command))

        creation = {}
        if paths.is_windows():  # pragma: no cover
            creation["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        self.proc = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True, errors="replace", bufsize=1, **creation)
        self._t_start = time.perf_counter()
        self._stopping = False
        self._stats = RecordStats()
        self._reader = threading.Thread(target=self._drain, daemon=True,
                                        name="ffmpeg-stderr")
        self._reader.start()
        self.write_sidecar()

    def _drain(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        # FFmpeg rewrites its progress line with \r, so split on both.
        buf = ""
        while True:
            ch = self.proc.stderr.read(1)
            if not ch:
                break
            if ch in ("\r", "\n"):
                line, buf = buf.strip(), ""
                if line:
                    self._handle_line(line)
            else:
                buf += ch
        if buf.strip():
            self._handle_line(buf.strip())
        rc = self.proc.wait()
        log.info("ffmpeg exited with %s: %s", rc,
                 explain_exit(rc, self._stats.errors))
        with self._lock:
            self._stats.wall_elapsed_s = time.perf_counter() - self._t_start
        self.spec.meta.ended_iso = datetime.now().astimezone().isoformat(timespec="seconds")
        try:
            self.write_sidecar(final=True, exit_code=rc)
        except OSError as e:  # pragma: no cover
            log.warning("could not finalise sidecar: %s", e)
        if self.on_finished:
            try:
                self.on_finished(rc)
            except Exception:  # pragma: no cover - callback must not kill the thread
                log.exception("on_finished callback failed")

    def _handle_line(self, line: str) -> None:
        with self._lock:
            self._stats.last_line = line
            self._stats.wall_elapsed_s = time.perf_counter() - self._t_start
            if _parse_progress(line, self._stats):
                return
            if _RT_FULL.search(line):
                self._stats.rt_buffer_drops += 1
                if len(self._stats.warnings) < 20:
                    self._stats.warnings.append(line)
                log.warning("ffmpeg capture buffer overflow: %s", line)
                return
            if _QUEUE.search(line):
                self._stats.queue_blocked += 1
                if len(self._stats.warnings) < 20:
                    self._stats.warnings.append(line)
                log.warning("ffmpeg input queue blocked: %s", line)
                return
            low = line.lower()
            if any(k in low for k in ("error", "i/o error", "could not",
                                      "invalid", "no such", "permission denied")):
                if line not in self._stats.errors:
                    self._stats.errors.append(line)
                log.warning("ffmpeg: %s", line)

    def stats(self) -> RecordStats:
        with self._lock:
            st = RecordStats(**{k: (list(v) if isinstance(v, list) else v)
                                for k, v in asdict(self._stats).items()})
        if self.proc is not None and self.proc.poll() is None:
            st.wall_elapsed_s = time.perf_counter() - self._t_start
        # FFmpeg's -stats line reports no size for the segment muxer, so the
        # folder has to be measured instead. This used to sit inside the
        # still-running branch above, which meant the final summary -- the one
        # the operator actually reads -- said "0 B" for every segmented trial
        # while the live readout during the same recording was correct.
        if self.spec.is_segmented and st.size_bytes == 0:
            try:
                folder = self.spec.target_path().parent
                st.size_bytes = sum(f.stat().st_size for f in folder.glob("*")
                                    if f.is_file())
            except OSError:
                pass
        return st

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def exit_code(self) -> int | None:
        return self.proc.poll() if self.proc is not None else None

    def mark_t0(self) -> float:
        """Record the dose time as an offset from the start of the recording."""
        st = self.stats()
        # The media clock is the honest one: it is what the analysis indexes by.
        offset = st.time_s if st.time_s > 0 else (time.perf_counter() - self._t_start)
        self.spec.meta.t0_offset_s = float(offset)
        log.info("T0 marked at %.2f s into the recording", offset)
        self.write_sidecar()
        return offset

    def stop(self, timeout: float = 10.0) -> None:
        """Ask FFmpeg to finish cleanly so the file keeps a valid index."""
        if self.proc is None or self.proc.poll() is not None:
            return
        self._stopping = True
        log.info("stopping recording")
        try:
            if self.proc.stdin:
                self.proc.stdin.write("q")
                self.proc.stdin.flush()
        except (OSError, ValueError):
            pass
        try:
            self.proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            log.warning("ffmpeg did not quit on request; terminating")
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover
            self.proc.kill()

    # ---- sidecars ----
    def write_sidecar(self, final: bool = False, exit_code: int | None = None) -> Path:
        s = self.spec
        target = s.target_path()
        if s.is_segmented:
            folder = target.parent
            folder.mkdir(parents=True, exist_ok=True)
            sidecar = folder / paths.SESSION_MANIFEST
            segments = sorted(p.name for p in folder.glob(f"*.{s.container}"))
        else:
            sidecar = target.with_name(target.name + paths.SIDECAR_SUFFIX)
            segments = []

        st = self.stats()
        doc = {
            "htr_studio_sidecar_version": 1,
            "kind": "session" if s.is_segmented else "recording",
            "name": s.base_name,
            "created": s.meta.started_iso or datetime.now().astimezone().isoformat(
                timespec="seconds"),
            "finalised": bool(final),
            "exit_code": exit_code,
            "trial": asdict(s.meta),
            "capture": {
                "device": s.device_name,
                "width": s.width,
                "height": s.height,
                "requested_fps": s.requested_fps,
                "expected_fps": s.expected_fps(),
                "pixel_format": s.pixel_format,
                "planned_duration_s": s.duration_s,
                "segmented": s.is_segmented,
                "segment_seconds": s.segment_seconds if s.is_segmented else None,
                "container": s.container,
                "rtbufsize": s.rtbufsize,
                "thread_queue_size": s.thread_queue_size,
            },
            "camera_settings": s.camera_settings,
            "ffmpeg_command": self.command,
            "observed": {
                "frames": st.frames,
                "media_time_s": round(st.time_s, 3),
                "effective_fps": round(st.effective_fps, 3),
                "dropped": st.dropped,
                "duplicated": st.duplicated,
                # Recorded separately from "dropped" because they mean different
                # things: this is loss upstream of the muxer, and it is the one
                # that invalidates a trial without showing in drop=.
                "rt_buffer_drops": st.rt_buffer_drops,
                "input_queue_blocked": st.queue_blocked,
                "bytes": st.size_bytes,
                "errors": st.errors,
                "capture_warnings": st.warnings,
            },
        }
        if s.is_segmented:
            doc["segments"] = segments
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        tmp.replace(sidecar)
        self.sidecar_path = sidecar
        return sidecar


# --------------------------------------------------------------------------
# Post-recording verification
# --------------------------------------------------------------------------
@dataclass
class Verification:
    path: Path
    ok: bool
    true_fps: float
    frames: int
    duration_s: float
    expected_fps: float
    message: str


def verify_recording(path: Path, expected_fps: float,
                     configured_ffprobe: str = "") -> Verification:
    """Check a finished file against the frame rate it was supposed to deliver.

    This is the ``nb_read_frames / duration`` check from the protocol, done for
    the operator instead of by hand. It decodes the whole file, so it is slow on
    a long trial -- which is why the Library runs it on request rather than on
    every scan.
    """
    info = ffmpeg_tools.probe_video(path, configured_ffprobe, count_frames=True)
    if not info.ok:
        return Verification(path, False, 0.0, 0, 0.0, expected_fps,
                            info.error or "could not read the file")
    true_fps = info.true_fps
    tol = 0.05
    ok = expected_fps <= 0 or abs(true_fps - expected_fps) / expected_fps <= tol
    if ok:
        msg = (f"{info.frames} frames in {info.duration_s:.2f} s = "
               f"{true_fps:.2f} fps - matches the expected {expected_fps:.0f} fps")
    else:
        msg = (f"{info.frames} frames in {info.duration_s:.2f} s = {true_fps:.2f} fps, "
               f"but {expected_fps:.0f} fps was expected. Frames were dropped, or the "
               "camera negotiated a different mode. Check the log and re-probe the "
               "camera before recording more animals.")
    return Verification(path, ok, true_fps, info.frames, info.duration_s,
                        expected_fps, msg)


def default_base_name(template: str, meta: TrialMeta) -> str:
    """Expand a name template such as ``{animal}_{group}_{date}_{time}``."""
    now = datetime.now()
    fields = {
        "animal": meta.animal_id or "animal",
        "group": meta.group or "grp",
        "treatment": meta.treatment or "tx",
        "dose": str(meta.dose_mg_kg or "").replace(".", "p"),
        "route": (meta.route or "").replace(" ", "-"),
        "operator": meta.operator or "op",
        "date": now.strftime("%Y%m%d"),
        "time": now.strftime("%H%M%S"),
    }
    try:
        name = template.format(**fields)
    except (KeyError, IndexError, ValueError):
        name = f"trial_{fields['date']}_{fields['time']}"
    # Windows-safe.
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return name or "trial"
