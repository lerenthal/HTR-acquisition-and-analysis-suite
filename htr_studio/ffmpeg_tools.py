"""FFmpeg / FFprobe discovery and the read-only probing helpers.

Why FFmpeg at all, when OpenCV can also open the camera? Because on this
hardware it is the only path that sustains the target frame rate. The validated
measurement on the reference host was 7209 frames in 30.01 s (240.3 fps, zero
drops) through ``-f dshow ... -c:v copy``, versus ~100 fps through OpenCV's
VideoWriter, which re-encodes every frame. ``-c:v copy`` hands the camera's own
MJPEG packets straight to disk.

A second reason: OpenCV reports the wrong FPS for MJPEG-in-AVI files. Ground
truth is ``nb_read_frames / duration`` from ffprobe, which :func:`probe_video`
computes.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import paths

log = logging.getLogger(__name__)

#: Passed to every subprocess so no console window flashes up on Windows.
_NO_WINDOW = {}
if paths.is_windows():  # pragma: no cover - platform specific
    _NO_WINDOW = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}


class FFmpegMissing(RuntimeError):
    """Raised when neither the bundled nor a system FFmpeg can be found."""


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def _candidates(tool: str, configured: str = "") -> list[Path]:
    out: list[Path] = []
    if configured:
        out.append(Path(configured))
    out.append(paths.FFMPEG_DIR / "bin" / paths.exe(tool))
    out.append(paths.TOOLS_BIN / paths.exe(tool))
    which = shutil.which(tool)
    if which:
        out.append(Path(which))
    if paths.is_windows():  # pragma: no cover
        local = os.environ.get("LOCALAPPDATA", "")
        if local:
            # The reference host had a Gyan full build unpacked under LOCALAPPDATA.
            out.extend(Path(local).glob(f"**/ffmpeg-*/bin/{paths.exe(tool)}"))
    return out


def find_tool(tool: str, configured: str = "") -> Path | None:
    for c in _candidates(tool, configured):
        try:
            if c.is_file():
                return c
        except OSError:
            continue
    return None


def ffmpeg_path(configured: str = "") -> Path:
    p = find_tool("ffmpeg", configured)
    if p is None:
        raise FFmpegMissing(
            "ffmpeg not found. Run install.bat (it downloads a static build into "
            "tools\\ffmpeg), or set paths.ffmpeg in config/settings.json."
        )
    return p


def ffprobe_path(configured: str = "") -> Path:
    p = find_tool("ffprobe", configured)
    if p is None:
        raise FFmpegMissing(
            "ffprobe not found. Run install.bat, or set paths.ffprobe in "
            "config/settings.json."
        )
    return p


def tool_versions(cfg_paths: dict | None = None) -> dict[str, str]:
    cfg_paths = cfg_paths or {}
    out: dict[str, str] = {}
    for tool, key in (("ffmpeg", "ffmpeg"), ("ffprobe", "ffprobe")):
        p = find_tool(tool, cfg_paths.get(key, ""))
        if p is None:
            out[tool] = "NOT FOUND"
            continue
        try:
            r = subprocess.run([str(p), "-version"], capture_output=True, text=True,
                               timeout=20, **_NO_WINDOW)
            first = (r.stdout or r.stderr).splitlines()[0] if (r.stdout or r.stderr) else "?"
            out[tool] = f"{first}  [{p}]"
        except (OSError, subprocess.SubprocessError) as e:
            out[tool] = f"error: {e}"
    return out


def run(args: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess:
    """Run a tool, capturing text output. FFmpeg writes diagnostics to stderr."""
    log.debug("run: %s", " ".join(args))
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                          errors="replace", **_NO_WINDOW)


# --------------------------------------------------------------------------
# DirectShow device enumeration
# --------------------------------------------------------------------------
@dataclass
class DShowDevice:
    name: str
    alternative_name: str = ""
    is_audio: bool = False
    #: "dshow" when FFmpeg listed it, "pnp" when it came from the Windows
    #: device database as a fallback. A pnp name is the device's friendly
    #: name, which is *usually* byte-identical to its DirectShow name but is
    #: not guaranteed to be, so the UI flags it as a best guess.
    source: str = "dshow"

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.name


def list_dshow_devices(configured_ffmpeg: str = "") -> list[DShowDevice]:
    """Enumerate DirectShow video devices.

    Two independent routes, because this is the step that strands people.

    1. ``ffmpeg -list_devices true -f dshow -i dummy``. It always exits
       non-zero ("Immediate exit requested") -- that is expected, the list is
       on stderr. :func:`parse_dshow_devices` handles both the pre-7.0 and the
       7.0+ output formats.
    2. If that yields nothing, ask Windows directly through PowerShell. FFmpeg
       has changed this output format before and may again, and a camera that
       the Camera app can see should never be invisible here. The raw FFmpeg
       output is written to the log either way, so a format change is
       diagnosable rather than mysterious.
    """
    if not paths.is_windows():
        log.info("dshow enumeration is Windows-only; returning empty list")
        return []
    ff = ffmpeg_path(configured_ffmpeg)
    r = run([str(ff), "-hide_banner", "-list_devices", "true", "-f", "dshow",
             "-i", "dummy"], timeout=40)
    text = (r.stderr or "") + (r.stdout or "")

    devices = parse_dshow_devices(text)
    video = [d for d in devices if not d.is_audio]
    if video:
        log.info("dshow video devices: %s", [d.name for d in video])
        return video

    log.warning("FFmpeg listed no DirectShow video devices. Raw output:\n%s",
                text[-3000:] if text.strip() else "(nothing on stdout or stderr)")
    fallback = list_pnp_cameras()
    if fallback:
        log.info("falling back to the Windows device database: %s", fallback)
        return [DShowDevice(name=n, source="pnp") for n in fallback]
    log.info("no cameras from either route")
    return []


_PS_CAMERA_SCRIPT = (
    "$ErrorActionPreference='SilentlyContinue';"
    "$n = @();"
    "$n += Get-PnpDevice -Class Camera,Image -Status OK -ErrorAction SilentlyContinue |"
    " Select-Object -ExpandProperty FriendlyName;"
    "if ($n.Count -eq 0) {"
    " $n += Get-CimInstance Win32_PnPEntity -ErrorAction SilentlyContinue |"
    " Where-Object { $_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image' } |"
    " Select-Object -ExpandProperty Name };"
    "$n | Where-Object { $_ } | ForEach-Object { $_ }"
)


def list_pnp_cameras() -> list[str]:
    """Camera friendly names straight from the Windows device database.

    Used only when FFmpeg's own listing comes back empty. These names normally
    match the DirectShow device name exactly -- both derive from the device's
    friendly name -- but a driver is free to differ, which is why the device
    box stays editable.
    """
    if not paths.is_windows():
        return []
    ps = shutil.which("powershell") or (
        r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
    try:
        r = run([ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                 "-Command", _PS_CAMERA_SCRIPT], timeout=45)
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("PowerShell camera enumeration failed: %s", e)
        return []
    names, seen = [], set()
    for line in (r.stdout or "").splitlines():
        n = line.strip()
        # Skip obvious non-cameras that share the Image PnP class.
        if not n or n.lower() in seen:
            continue
        if any(w in n.lower() for w in ("scanner", "printer", "wia ")):
            continue
        seen.add(n.lower())
        names.append(n)
    return names


#: A device line, in either of the two formats FFmpeg has used.
#:
#: Up to FFmpeg 6.x the listing was grouped under "DirectShow video devices"
#: and "DirectShow audio devices" headers, and each line ended with the quoted
#: name. From FFmpeg 7.0 the headers were dropped and the kind is appended to
#: the line instead, as ``"Some Camera" (video)``. A parser that only handles
#: the old shape finds nothing on a current build -- which looks exactly like
#: "no camera attached", and is the most confusing possible failure.
_DSHOW_LINE = re.compile(r'"(?P<name>[^"]+)"\s*(?:\((?P<kind>video|audio)\))?\s*$')
_DSHOW_ALT = re.compile(r'Alternative name\s+"(?P<alt>[^"]+)"')


def parse_dshow_devices(text: str) -> list[DShowDevice]:
    """Parse ``ffmpeg -list_devices true -f dshow -i dummy`` output.

    Handles both the pre-7.0 section-header format and the 7.0+ inline
    ``(video)`` / ``(audio)`` marker. When a line carries an explicit marker it
    wins; otherwise the most recent section header decides.
    """
    devices: list[DShowDevice] = []
    section_audio = False
    saw_header = False
    last: DShowDevice | None = None
    for line in text.splitlines():
        if "DirectShow video devices" in line:
            section_audio, saw_header = False, True
            continue
        if "DirectShow audio devices" in line:
            section_audio, saw_header = True, True
            continue

        m_alt = _DSHOW_ALT.search(line)
        if m_alt and last is not None:
            last.alternative_name = m_alt.group("alt")
            continue
        if "Alternative name" in line:
            continue

        m = _DSHOW_LINE.search(line)
        if not m:
            continue
        kind = m.group("kind")
        if kind is not None:
            is_audio = kind == "audio"
        elif saw_header:
            is_audio = section_audio
        else:
            # No header and no marker: assume video rather than discard it.
            is_audio = False
        last = DShowDevice(name=m.group("name"), is_audio=is_audio)
        devices.append(last)
    return devices


def match_device(devices: list[DShowDevice], wanted: str) -> DShowDevice | None:
    """Case-insensitive exact-then-substring match, then a loose token match."""
    if not devices:
        return None
    if not wanted:
        return devices[0]
    w = wanted.strip().lower()
    for d in devices:
        if d.name.lower() == w:
            return d
    for d in devices:
        if w in d.name.lower() or d.name.lower() in w:
            return d
    # Loose: any device sharing a distinctive token (e.g. "arducam").
    for token in ("arducam", "ezbox", "ov9281", "global shutter"):
        for d in devices:
            if token in d.name.lower():
                return d
    return None


def device_options(configured_ffmpeg: str, device_name: str) -> str:
    """Raw ``-list_options`` text for a device: every mode it advertises."""
    ff = ffmpeg_path(configured_ffmpeg)
    r = run([str(ff), "-hide_banner", "-list_options", "true", "-f", "dshow",
             "-i", f"video={device_name}"], timeout=40)
    return (r.stderr or "") + (r.stdout or "")


_OPT_RE = re.compile(
    r"(?:pixel_format=(?P<pix>\S+)|vcodec=(?P<codec>\S+)).*?"
    r"min s=(?P<w>\d+)x(?P<h>\d+) fps=(?P<minfps>[\d.]+) "
    r"max s=(?P<w2>\d+)x(?P<h2>\d+) fps=(?P<maxfps>[\d.]+)"
)


@dataclass
class CameraMode:
    width: int
    height: int
    min_fps: float
    max_fps: float
    pixel_format: str = ""
    codec: str = ""

    @property
    def label(self) -> str:
        fmt = self.codec or self.pixel_format or "?"
        return f"{self.width}x{self.height} {fmt} {self.min_fps:g}-{self.max_fps:g} fps"


def parse_device_options(text: str) -> list[CameraMode]:
    modes: list[CameraMode] = []
    for m in _OPT_RE.finditer(text):
        modes.append(CameraMode(
            width=int(m.group("w2")), height=int(m.group("h2")),
            min_fps=float(m.group("minfps")), max_fps=float(m.group("maxfps")),
            pixel_format=m.group("pix") or "", codec=m.group("codec") or "",
        ))
    # De-duplicate, keeping the widest fps range per (w, h, format).
    best: dict[tuple, CameraMode] = {}
    for mo in modes:
        key = (mo.width, mo.height, mo.codec or mo.pixel_format)
        cur = best.get(key)
        if cur is None or mo.max_fps > cur.max_fps:
            best[key] = mo
    return sorted(best.values(), key=lambda m: (-m.max_fps, -m.width))


def open_device_property_dialog(configured_ffmpeg: str, device_name: str) -> None:
    """Open the vendor DirectShow property page for the camera.

    This is the authoritative place to set exposure and gain: values set here
    are stored in the driver and survive the handle being closed, which is not
    guaranteed for values pushed through OpenCV. FFmpeg exits as soon as the
    dialog is dismissed.
    """
    ff = ffmpeg_path(configured_ffmpeg)
    args = [str(ff), "-hide_banner", "-f", "dshow",
            "-show_video_device_dialog", "true",
            "-i", f"video={device_name}", "-t", "0.1", "-f", "null", "-"]
    log.info("opening dshow property dialog: %s", " ".join(args))
    subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# --------------------------------------------------------------------------
# FPS probing
# --------------------------------------------------------------------------
@dataclass
class ProbeResult:
    requested_fps: float
    width: int
    height: int
    negotiated_fps: float | None = None
    frames: int = 0
    achieved_fps: float = 0.0
    duration_s: float = 0.0
    ok: bool = False
    message: str = ""


def probe_mode(configured_ffmpeg: str, device_name: str, width: int, height: int,
               requested_fps: float, seconds: float = 2.0,
               pixel_format: str = "mjpeg", rtbufsize: str = "256M") -> ProbeResult:
    """Capture a short clip to a temp file and report what the camera delivered."""
    ff = ffmpeg_path(configured_ffmpeg)
    paths.ensure_dirs()
    tmp = paths.CACHE_DIR / f"probe_{width}x{height}_{int(requested_fps)}.avi"
    args = [str(ff), "-hide_banner", "-f", "dshow", "-rtbufsize", rtbufsize,
            "-framerate", str(requested_fps), "-video_size", f"{width}x{height}"]
    if pixel_format in ("mjpeg", "mjpg"):
        args += ["-vcodec", "mjpeg"]
    else:
        args += ["-pixel_format", pixel_format]
    args += ["-i", f"video={device_name}", "-t", f"{seconds}",
             "-c:v", "copy", "-an", "-y", str(tmp)]

    res = ProbeResult(requested_fps=requested_fps, width=width, height=height)
    try:
        r = run(args, timeout=seconds + 40)
    except subprocess.TimeoutExpired:
        res.message = "timed out"
        return res
    text = (r.stderr or "") + (r.stdout or "")

    m = re.search(r"Stream #0:0:.*?([\d.]+)\s*fps", text)
    if m:
        try:
            res.negotiated_fps = float(m.group(1))
        except ValueError:
            pass
    for fm in re.finditer(r"frame=\s*(\d+).*?time=(\d+):(\d+):([\d.]+)", text):
        res.frames = int(fm.group(1))
        res.duration_s = (int(fm.group(2)) * 3600 + int(fm.group(3)) * 60
                          + float(fm.group(4)))
    if res.frames and res.duration_s > 0:
        res.achieved_fps = res.frames / res.duration_s

    if r.returncode != 0 and res.frames == 0:
        snippet = [ln for ln in text.splitlines() if "rror" in ln or "I/O" in ln]
        res.message = snippet[-1].strip() if snippet else f"ffmpeg exit {r.returncode}"
    else:
        res.ok = res.frames > 0
        res.message = "ok"

    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    return res


# --------------------------------------------------------------------------
# Video inspection
# --------------------------------------------------------------------------
@dataclass
class VideoInfo:
    path: Path
    width: int = 0
    height: int = 0
    codec: str = ""
    container_fps: float = 0.0
    frames: int = 0
    duration_s: float = 0.0
    size_bytes: int = 0
    pix_fmt: str = ""
    exact: bool = False  # True when frames were counted, not estimated
    error: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def true_fps(self) -> float:
        """Ground-truth frame rate: counted frames divided by duration."""
        if self.frames and self.duration_s > 0:
            return self.frames / self.duration_s
        return self.container_fps

    @property
    def ok(self) -> bool:
        return not self.error and self.frames > 0


def probe_video(path: Path | str, configured_ffprobe: str = "",
                count_frames: bool = True, timeout: float = 900.0) -> VideoInfo:
    """Inspect a video file.

    ``count_frames=True`` decodes the whole file to count frames exactly. That
    is slow for a 70-minute trial (minutes), so the library view calls this with
    ``count_frames=False`` first and upgrades on demand.
    """
    p = Path(path)
    info = VideoInfo(path=p)
    try:
        info.size_bytes = p.stat().st_size
    except OSError as e:
        info.error = str(e)
        return info

    try:
        fp = ffprobe_path(configured_ffprobe)
    except FFmpegMissing as e:
        info.error = str(e)
        return info

    entries = ("stream=width,height,codec_name,pix_fmt,r_frame_rate,"
               "avg_frame_rate,nb_frames,duration,nb_read_frames")
    args = [str(fp), "-v", "error", "-select_streams", "v:0"]
    if count_frames:
        args.append("-count_frames")
    args += ["-show_entries", entries, "-of", "json", str(p)]

    try:
        r = run(args, timeout=timeout)
    except subprocess.TimeoutExpired:
        info.error = "ffprobe timed out"
        return info
    if r.returncode != 0:
        info.error = (r.stderr or "ffprobe failed").strip().splitlines()[-1:][0] \
            if (r.stderr or "").strip() else "ffprobe failed"
        return info

    try:
        doc = json.loads(r.stdout or "{}")
        streams = doc.get("streams") or []
        if not streams:
            info.error = "no video stream"
            return info
        s = streams[0]
    except json.JSONDecodeError as e:
        info.error = f"bad ffprobe output: {e}"
        return info

    info.width = int(s.get("width") or 0)
    info.height = int(s.get("height") or 0)
    info.codec = s.get("codec_name") or ""
    info.pix_fmt = s.get("pix_fmt") or ""

    def _rate(key: str) -> float:
        v = s.get(key) or ""
        if "/" in str(v):
            num, den = str(v).split("/")
            try:
                return float(num) / float(den) if float(den) else 0.0
            except ValueError:
                return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    info.container_fps = _rate("r_frame_rate") or _rate("avg_frame_rate")
    try:
        info.duration_s = float(s.get("duration") or 0.0)
    except (TypeError, ValueError):
        info.duration_s = 0.0

    nb_read = s.get("nb_read_frames")
    nb = s.get("nb_frames")
    if nb_read not in (None, "N/A"):
        info.frames = int(nb_read)
        info.exact = True
    elif nb not in (None, "N/A"):
        info.frames = int(nb)
    elif info.duration_s and info.container_fps:
        info.frames = int(round(info.duration_s * info.container_fps))

    # AVI files with no duration field: derive it from the container rate.
    if info.duration_s <= 0 and info.frames and info.container_fps:
        info.duration_s = info.frames / info.container_fps

    info.extra = {"r_frame_rate": s.get("r_frame_rate"),
                  "avg_frame_rate": s.get("avg_frame_rate")}
    return info


def extract_frame(path: Path | str, t_seconds: float, out_png: Path,
                  configured_ffmpeg: str = "") -> bool:
    """Pull a single frame at a timestamp (used for library thumbnails)."""
    try:
        ff = ffmpeg_path(configured_ffmpeg)
    except FFmpegMissing:
        return False
    out_png.parent.mkdir(parents=True, exist_ok=True)
    args = [str(ff), "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, t_seconds)}",
            "-i", str(path), "-frames:v", "1", "-q:v", "3", "-y", str(out_png)]
    try:
        r = run(args, timeout=120)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0 and out_png.exists()
