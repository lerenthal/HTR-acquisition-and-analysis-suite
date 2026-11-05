"""Live camera preview, UVC control, and image-quality / lighting metrics.

Division of labour between OpenCV and FFmpeg
--------------------------------------------
* OpenCV (DirectShow backend) is used for the *live preview* and for *pushing
  UVC control values* (exposure, gain, brightness, contrast, gamma). It is the
  only one of the two that can show a frame and change a control interactively.
* FFmpeg does the *recording*, because it is the only path that sustains
  240 fps on this camera (see ffmpeg_tools).

A DirectShow device can only be opened once at a time, so preview must be
stopped before recording starts. :class:`CameraPreview` owns that lifecycle.

About exposure values
---------------------
OpenCV's DSHOW backend passes ``CAP_PROP_EXPOSURE`` through as the UVC
*log2 exposure-time* value, so -7 means 2**-7 s = 7.8 ms of *allowed*
integration, which the sensor clamps to the frame interval (4.2 ms at 240 fps).
Going more negative shortens it: -9 is ~2 ms, -10 ~1 ms. Those are the useful
values for freezing a 90 Hz head oscillation. The mapping is driver-dependent,
which is why :func:`exposure_hint_ms` is labelled an estimate and why the
property dialog remains available as the authoritative control.
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import paths

log = logging.getLogger(__name__)

BACKENDS = {
    "dshow": getattr(cv2, "CAP_DSHOW", 700),
    "msmf": getattr(cv2, "CAP_MSMF", 1400),
    "any": getattr(cv2, "CAP_ANY", 0),
    "v4l2": getattr(cv2, "CAP_V4L2", 200),
}


def exposure_hint_ms(value: float) -> float:
    """Approximate integration time in milliseconds for an OpenCV exposure value."""
    try:
        return (2.0 ** float(value)) * 1000.0
    except (OverflowError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------
# Enumeration
# --------------------------------------------------------------------------
@dataclass
class ProbedIndex:
    index: int
    width: int
    height: int
    fps: float
    fourcc: str
    backend: str

    @property
    def label(self) -> str:
        return (f"index {self.index} - {self.width}x{self.height} "
                f"{self.fps:.0f} fps {self.fourcc} [{self.backend}]")


def _fourcc_str(value: float) -> str:
    v = int(value)
    try:
        return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4)).strip()
    except ValueError:
        return "?"


def enumerate_indices(max_index: int = 6, backend: str = "dshow") -> list[ProbedIndex]:
    """Probe OpenCV camera indices.

    Opening an index is slow (up to ~1 s each) and briefly claims the device, so
    only call this from a worker thread and not while recording.
    """
    api = BACKENDS.get(backend, BACKENDS["any"])
    found: list[ProbedIndex] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, api)
        try:
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append(ProbedIndex(
                        index=i,
                        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
                        fourcc=_fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)),
                        backend=cap.getBackendName(),
                    ))
        except cv2.error as e:  # pragma: no cover
            log.debug("index %d probe error: %s", i, e)
        finally:
            cap.release()
    log.info("opencv indices found: %s", [p.index for p in found])
    return found


def guess_arducam_index(probed: list[ProbedIndex]) -> int:
    """Pick the index most likely to be the B0634.

    The B0634 exposes only 640x480 natively and prefers MJPG, so VGA + MJPG is
    the strongest signal. Falls back to the first working index.
    """
    if not probed:
        return -1

    def score(p: ProbedIndex) -> int:
        s = 0
        if (p.width, p.height) == (640, 480):
            s += 100
        if p.fourcc.upper() in ("MJPG", "YUY2"):
            s += 10
        if p.fps >= 100:
            s += 20
        return s

    return max(probed, key=score).index


# --------------------------------------------------------------------------
# Control set
# --------------------------------------------------------------------------
@dataclass
class UVCSettings:
    width: int = 320
    height: int = 240
    fps: float = 200.0
    fourcc: str = "MJPG"
    auto_exposure: bool = False
    exposure: float = -7.0
    gain: float = 0.0
    brightness: float | None = 0.0
    contrast: float | None = 32.0
    gamma: float | None = 100.0

    @classmethod
    def from_config(cls, cam: dict) -> "UVCSettings":
        return cls(
            width=int(cam.get("width", 320)),
            height=int(cam.get("height", 240)),
            fps=float(cam.get("requested_fps", 200)),
            fourcc="MJPG" if str(cam.get("pixel_format", "mjpeg")).startswith("mjp") else "YUY2",
            auto_exposure=bool(cam.get("auto_exposure", False)),
            exposure=float(cam.get("exposure", -7)),
            gain=float(cam.get("gain", 0)),
            brightness=cam.get("brightness"),
            contrast=cam.get("contrast"),
            gamma=cam.get("gamma"),
        )


@dataclass
class AppliedState:
    """What the driver actually accepted, for display next to what was asked."""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    fourcc: str = ""
    exposure: float = float("nan")
    gain: float = float("nan")
    auto_exposure: float = float("nan")
    warnings: list[str] = field(default_factory=list)


def apply_uvc(cap: "cv2.VideoCapture", s: UVCSettings) -> AppliedState:
    """Push the requested mode and controls, then read back what stuck."""
    st = AppliedState()
    try:
        # FourCC first: changing it after the size resets the size on some drivers.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*s.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.height)
        cap.set(cv2.CAP_PROP_FPS, s.fps)

        # On the DSHOW backend 0.25 = manual, 0.75 = auto. Some builds instead
        # take 1 = manual, 3 = auto. Write both; the driver ignores the invalid one.
        if s.auto_exposure:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, s.exposure)
        cap.set(cv2.CAP_PROP_GAIN, s.gain)
        for prop, val in ((cv2.CAP_PROP_BRIGHTNESS, s.brightness),
                          (cv2.CAP_PROP_CONTRAST, s.contrast),
                          (cv2.CAP_PROP_GAMMA, s.gamma)):
            if val is not None:
                cap.set(prop, float(val))
    except cv2.error as e:  # pragma: no cover
        st.warnings.append(f"OpenCV rejected a control: {e}")

    st.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    st.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    st.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    st.fourcc = _fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
    st.exposure = float(cap.get(cv2.CAP_PROP_EXPOSURE))
    st.gain = float(cap.get(cv2.CAP_PROP_GAIN))
    st.auto_exposure = float(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))

    if (st.width, st.height) != (s.width, s.height):
        st.warnings.append(
            f"asked {s.width}x{s.height}, driver gave {st.width}x{st.height}")
    if st.fourcc.upper() not in (s.fourcc.upper(), ""):
        st.warnings.append(f"asked {s.fourcc}, driver gave {st.fourcc}")
    if not s.auto_exposure and abs(st.exposure - s.exposure) > 0.51:
        st.warnings.append(
            f"exposure read back as {st.exposure:g}, not {s.exposure:g} - "
            "use the camera property dialog instead")
    return st


def apply_uvc_oneshot(index: int, s: UVCSettings, backend: str = "dshow",
                      settle_frames: int = 10) -> AppliedState:
    """Open the camera briefly, push controls, grab a few frames, close.

    Most Windows UVC drivers keep manual exposure/gain in the device after the
    handle closes, so this is how the preview's settings get carried into the
    FFmpeg recording. It is best-effort: the returned warnings say whether the
    values stuck, and the property dialog is the fallback.
    """
    api = BACKENDS.get(backend, BACKENDS["any"])
    cap = cv2.VideoCapture(index, api)
    if not cap.isOpened():
        st = AppliedState()
        st.warnings.append(f"could not open camera index {index}")
        return st
    try:
        st = apply_uvc(cap, s)
        for _ in range(settle_frames):
            cap.read()
        return st
    finally:
        cap.release()


# --------------------------------------------------------------------------
# Frame quality / lighting metrics
# --------------------------------------------------------------------------
@dataclass
class FrameMetrics:
    mean: float = 0.0
    median: float = 0.0
    p01: float = 0.0
    p99: float = 0.0
    clipped_low_pct: float = 0.0
    clipped_high_pct: float = 0.0
    contrast: float = 0.0          # p99 - p01, 0-255
    focus_score: float = 0.0       # variance of Laplacian
    uniformity: float = 0.0        # 1.0 = perfectly even illumination
    quadrant_means: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    histogram: np.ndarray | None = None
    verdicts: list[tuple[str, str]] = field(default_factory=list)  # (level, text)


def frame_metrics(gray: np.ndarray) -> FrameMetrics:
    """Everything the Acquisition tab needs to judge exposure and lighting."""
    m = FrameMetrics()
    if gray is None or gray.size == 0:
        return m
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    f = g.astype(np.float32)

    m.mean = float(f.mean())
    m.median = float(np.median(f))
    m.p01, m.p99 = (float(x) for x in np.percentile(f, [1, 99]))
    total = f.size
    m.clipped_low_pct = float((g <= 2).sum()) / total * 100.0
    m.clipped_high_pct = float((g >= 253).sum()) / total * 100.0
    m.contrast = m.p99 - m.p01
    m.focus_score = float(cv2.Laplacian(g, cv2.CV_64F).var())

    h, w = g.shape[:2]
    hh, hw = h // 2, w // 2
    quads = (f[:hh, :hw], f[:hh, hw:], f[hh:, :hw], f[hh:, hw:])
    qm = tuple(float(q.mean()) for q in quads)
    m.quadrant_means = qm  # type: ignore[assignment]
    if max(qm) > 0:
        m.uniformity = float(min(qm) / max(qm))

    m.histogram = cv2.calcHist([g], [0], None, [64], [0, 256]).flatten()

    # Verdicts, in the order the operator should act on them.
    if m.clipped_high_pct > 1.0:
        m.verdicts.append(("bad", f"{m.clipped_high_pct:.1f}% of pixels blown out - "
                                  "reduce light or shorten exposure"))
    if m.clipped_low_pct > 20.0:
        m.verdicts.append(("bad", f"{m.clipped_low_pct:.0f}% of pixels crushed to black - "
                                  "add light or lengthen exposure"))
    if m.contrast < 40:
        m.verdicts.append(("bad", f"mouse/arena contrast only {m.contrast:.0f}/255 - "
                                  "blob tracking will be unreliable"))
    elif m.contrast < 70:
        m.verdicts.append(("warn", f"contrast {m.contrast:.0f}/255 is marginal; "
                                   "a matte black floor under the cylinder helps"))
    if m.uniformity < 0.70:
        m.verdicts.append(("warn", f"illumination uneven (corner ratio {m.uniformity:.2f}) - "
                                   "move the panels further out or add diffusion"))
    if m.focus_score < 40:
        m.verdicts.append(("warn", f"focus score {m.focus_score:.0f} is low - "
                                   "refocus, then lock the ring"))
    if not m.verdicts:
        m.verdicts.append(("ok", "exposure, contrast, uniformity and focus all within range"))
    return m


@dataclass
class FlickerResult:
    frames: int = 0
    fps: float = 0.0
    dominant_hz: float = 0.0
    modulation_pct: float = 0.0
    mains_hz: float = 0.0
    verdict: str = ""
    level: str = "ok"
    freqs: np.ndarray | None = None
    power: np.ndarray | None = None


def flicker_test(cap: "cv2.VideoCapture", seconds: float = 2.0,
                 nominal_fps: float = 0.0, mains_hz: float = 50.0) -> FlickerResult:
    """Detect PWM-dimmed or mains-flickering light.

    A PWM-driven LED panel modulates brightness at hundreds of Hz. Under a
    sub-millisecond global-shutter exposure that turns into frame-to-frame
    brightness steps, which the HTR band-pass happily reports as oscillation -
    a whole-field false positive. The fix is hardware (DC constant-current
    drive), so this test exists to catch it before a 70-minute trial.

    Method: record the mean intensity of each frame, remove the trend, and look
    at the amplitude spectrum. Flat spectrum = clean DC light.
    """
    res = FlickerResult(mains_hz=mains_hz)
    means: list[float] = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        means.append(float(g.mean()))
    elapsed = time.perf_counter() - t0

    return flicker_from_means(
        means, nominal_fps or (len(means) / max(elapsed, 1e-6)), mains_hz)


def flicker_from_means(means: "list[float] | np.ndarray", fps: float,
                       mains_hz: float = 50.0) -> FlickerResult:
    """The spectrum half of :func:`flicker_test`, over per-frame mean levels.

    Split out so the measurement can be made from any source of frames. It
    matters most at the *capture* frame rate: whether a PWM ripple aliases
    into the 55-125 Hz band depends on the sampling rate, so a verdict taken
    from a 25 fps preview says nothing useful about a 200 fps recording.
    """
    res = FlickerResult(mains_hz=mains_hz)
    res.frames = len(means)
    if res.frames < 32:
        res.verdict = "not enough frames captured to judge flicker"
        res.level = "warn"
        return res

    res.fps = float(fps)
    x = np.asarray(means, dtype=np.float64)
    baseline = float(np.mean(x))
    # Remove slow drift (auto-gain hunting, warm-up) so only fast ripple remains.
    k = max(3, int(res.fps * 0.25) | 1)
    trend = np.convolve(x, np.ones(k) / k, mode="same")
    ripple = x - trend

    window = np.hanning(len(ripple))
    spec = np.abs(np.fft.rfft(ripple * window)) * (2.0 / np.sum(window))
    freqs = np.fft.rfftfreq(len(ripple), d=1.0 / res.fps)
    res.freqs, res.power = freqs, spec

    if len(spec) > 2:
        i = int(np.argmax(spec[1:]) + 1)
        res.dominant_hz = float(freqs[i])
        res.modulation_pct = float(spec[i] / max(baseline, 1e-6) * 100.0)

    near_mains = any(abs(res.dominant_hz - h) < 3.0
                     for h in (mains_hz, 2 * mains_hz, 4 * mains_hz))
    in_htr_band = 55.0 <= res.dominant_hz <= 125.0

    if res.modulation_pct < 0.5:
        res.level, res.verdict = "ok", (
            f"light is steady ({res.modulation_pct:.2f}% ripple) - DC drive confirmed")
    elif in_htr_band or near_mains:
        res.level, res.verdict = "bad", (
            f"{res.modulation_pct:.1f}% brightness ripple at {res.dominant_hz:.0f} Hz. "
            "This sits in or near the HTR band and will generate false positives. "
            "Switch the panels to constant-current DC drive (no PWM dimming) and "
            "set brightness with distance or diffusion instead.")
    else:
        res.level, res.verdict = "warn", (
            f"{res.modulation_pct:.1f}% ripple at {res.dominant_hz:.0f} Hz - outside the "
            "HTR band, but a DC-driven panel would be cleaner")
    return res


# --------------------------------------------------------------------------
# Preview handle
# --------------------------------------------------------------------------
class CameraPreview:
    """Owns a short-lived OpenCV capture for the live preview.

    Deliberately not a QThread: the Acquisition tab polls :meth:`read` from a
    QTimer, which keeps all OpenCV calls on one thread (the DSHOW backend is not
    thread-safe) and makes stopping before a recording trivial.
    """

    def __init__(self, index: int, settings: UVCSettings, backend: str = "dshow"):
        self.index = index
        self.settings = settings
        self.backend = backend
        self.cap: cv2.VideoCapture | None = None
        self.applied = AppliedState()
        self._frames = 0
        self._t0 = 0.0
        self._last_err = ""

    # ---- lifecycle ----
    def start(self) -> bool:
        self.stop()
        api = BACKENDS.get(self.backend, BACKENDS["any"])
        cap = cv2.VideoCapture(self.index, api)
        if not cap.isOpened():
            self._last_err = (
                f"Could not open camera index {self.index} with the {self.backend} "
                "backend. Another program (or a running recording) may hold it.")
            cap.release()
            return False
        self.cap = cap
        self.applied = apply_uvc(cap, self.settings)
        self._frames = 0
        self._t0 = time.perf_counter()
        self._last_err = ""
        log.info("preview started: index=%s applied=%sx%s @%.1f %s",
                 self.index, self.applied.width, self.applied.height,
                 self.applied.fps, self.applied.fourcc)
        return True

    def stop(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except cv2.error:  # pragma: no cover
                pass
            self.cap = None
            log.info("preview stopped")

    @property
    def running(self) -> bool:
        return self.cap is not None

    @property
    def last_error(self) -> str:
        return self._last_err

    # ---- frames ----
    def read(self) -> np.ndarray | None:
        """Return the newest frame as grayscale, or None."""
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok:
            return None
        self._frames += 1
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def measured_fps(self) -> float:
        dt = time.perf_counter() - self._t0
        return self._frames / dt if dt > 0.2 else 0.0

    def reset_fps_counter(self) -> None:
        self._frames = 0
        self._t0 = time.perf_counter()

    def push(self, settings: UVCSettings) -> AppliedState:
        """Apply changed controls to the open handle (used by the sliders)."""
        self.settings = settings
        if self.cap is None:
            return AppliedState()
        self.applied = apply_uvc(self.cap, settings)
        self.reset_fps_counter()
        return self.applied

    def run_flicker_test(self, seconds: float = 2.0, mains_hz: float = 50.0) -> FlickerResult:
        if self.cap is None:
            r = FlickerResult()
            r.verdict = "start the preview first"
            r.level = "warn"
            return r
        return flicker_test(self.cap, seconds=seconds,
                            nominal_fps=self.applied.fps, mains_hz=mains_hz)

    def __enter__(self) -> "CameraPreview":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


# --------------------------------------------------------------------------
# True preview: the capture pipeline itself
# --------------------------------------------------------------------------
class CapturePreview:
    """Preview through FFmpeg, with the recorder's own DirectShow arguments.

    Why this exists. :class:`CameraPreview` is OpenCV, and on Windows the
    DSHOW backend negotiates on OpenCV's behalf: ask the B0634 for MJPG and
    the driver hands back YUY2, and ``CAP_PROP_FPS`` echoes the requested
    rate rather than reporting what was agreed. FFmpeg, asked for
    ``-vcodec mjpeg -framerate 200``, gets exactly that. Two different media
    types at two different frame rates have two different exposure windows,
    and since brightness follows integration time, the OpenCV preview can be
    several times brighter than the file. An operator setting lighting
    against that preview is setting it against the wrong picture.

    So this class opens the camera with ``recorder.dshow_input_args(spec)``
    -- the same list the recording uses, not a copy of it -- and decodes to
    8-bit gray on stdout. What appears here is what will be written, mode,
    rate, exposure clamp and all.

    The cost. FFmpeg holds the device, so UVC controls cannot be pushed into
    a running preview the way :meth:`CameraPreview.push` does: the Acquisition
    tab stops this, applies the control, and reopens. Opening a DirectShow
    graph takes about a second, which is the blink the operator sees.

    Frame rate on the pipe. The camera runs at the capture rate but a preview
    needs about 25 fps, so an ``fps`` filter decimates on FFmpeg's side.
    Decoding every frame first is unavoidable -- the decimation has to happen
    after decode -- but at QVGA that is cheap, and it keeps the pipe at
    320x240x25 bytes/s rather than eight times that.

    Threading. A pipe read blocks until a whole frame has arrived, so reading
    it from the GUI thread would stall the window for up to a frame interval
    -- measured at 89 ms on the first tick, which is exactly the kind of
    "reacts slowly to button presses" this codebase has already been through
    once. A reader thread keeps the pipe drained and holds only the newest
    frame; :meth:`read` takes that frame without blocking and returns None
    when nothing new has arrived. One slot, not a queue: a preview that falls
    behind should skip to the live edge, never work through a backlog.
    """

    def __init__(self, spec, cfg_paths: dict | None = None,
                 display_fps: float = 25.0):
        self.spec = spec
        self.cfg_paths = cfg_paths or {}
        self.display_fps = max(1.0, float(display_fps))
        self.proc: "subprocess.Popen | None" = None
        self.width = int(spec.width)
        self.height = int(spec.height)
        self.command: list[str] = []
        self._frames = 0
        self._t0 = 0.0
        self._last_err = ""
        self._stderr: list[str] = []
        self._err_thread: threading.Thread | None = None
        self._frame_thread: threading.Thread | None = None
        self._latest: "np.ndarray | None" = None
        self._dropped = 0
        self._stopping = threading.Event()
        self._lock = threading.Lock()

    # ---- command ----
    def build_command(self) -> list[str]:
        from . import ffmpeg_tools, recorder
        ff = ffmpeg_tools.ffmpeg_path(self.cfg_paths.get("ffmpeg", ""))
        return ([str(ff), "-hide_banner", "-loglevel", "warning", "-nostdin"]
                + recorder.dshow_input_args(self.spec)
                + ["-an",
                   "-vf", f"fps={self.display_fps:g}",
                   "-pix_fmt", "gray", "-f", "rawvideo", "-"])

    # ---- lifecycle ----
    def start(self) -> bool:
        self.stop()
        self.command = self.build_command()
        log.info("true preview: %s", " ".join(self.command))
        creation = {}
        if paths.is_windows():  # pragma: no cover - Windows only
            creation["creationflags"] = getattr(
                subprocess, "CREATE_NO_WINDOW", 0x08000000)
        try:
            self.proc = subprocess.Popen(
                self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL, bufsize=0, **creation)
        except OSError as e:
            self._last_err = f"could not start FFmpeg: {e}"
            self.proc = None
            return False
        self._frames = 0
        self._dropped = 0
        self._t0 = time.perf_counter()
        self._stderr = []
        self._latest = None
        self._stopping.clear()
        self._err_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._err_thread.start()
        self._frame_thread = threading.Thread(target=self._drain_frames, daemon=True)
        self._frame_thread.start()
        return True

    def _drain_frames(self) -> None:
        """Keep the pipe empty, holding only the newest frame.

        Runs off the GUI thread, so a blocking read costs nothing visible.
        Frames that arrive while the last one is still unread are counted and
        discarded -- the preview wants the live edge, not a backlog.
        """
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        need = self.width * self.height
        while not self._stopping.is_set():
            buf = bytearray()
            while len(buf) < need:
                try:
                    chunk = proc.stdout.read(need - len(buf))
                except (OSError, ValueError):  # pipe closed under us
                    chunk = b""
                if not chunk:
                    if self._stopping.is_set():
                        return
                    from . import recorder
                    with self._lock:
                        self._last_err = recorder.explain_exit(
                            proc.poll(), list(self._stderr))
                    return
                buf += chunk
            frame = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(
                self.height, self.width)
            with self._lock:
                if self._latest is not None:
                    self._dropped += 1
                self._latest = frame
                self._frames += 1

    def _drain_stderr(self) -> None:
        """Keep FFmpeg's stderr pipe empty, and keep the last few lines.

        A full stderr pipe blocks the child, which on a 70-minute trial is a
        lost experiment; here it would merely freeze the preview, but the
        habit is worth keeping. The retained lines are what
        :func:`recorder.explain_exit` turns into something actionable.
        """
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if not line:
                continue
            with self._lock:
                self._stderr.append(line)
                del self._stderr[:-20]
            log.debug("true preview ffmpeg: %s", line)

    def stop(self) -> None:
        self._stopping.set()
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:  # pragma: no cover
                proc.kill()
                proc.wait(timeout=2.0)
        except OSError:  # pragma: no cover
            pass
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:  # pragma: no cover
                pass
        log.info("true preview stopped")

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_err

    def stderr_tail(self) -> list[str]:
        with self._lock:
            return list(self._stderr)

    # ---- frames ----
    def read(self) -> "np.ndarray | None":
        """The newest frame, or None if none has arrived since the last call.

        Never blocks: the reader thread owns the pipe. None is the normal
        answer when the display timer runs faster than the pipe, and the
        caller should simply leave the last frame on screen.
        """
        with self._lock:
            frame, self._latest = self._latest, None
        if frame is None and not self.running:
            self._stop_threads_only()
        return frame

    #: Kept so the Acquisition tab can treat both previews alike.
    def drain_to_latest(self, max_frames: int = 8) -> "np.ndarray | None":
        return self.read()

    def _stop_threads_only(self) -> None:
        self._stopping.set()

    @property
    def dropped(self) -> int:
        """Frames the display never saw because a newer one replaced them."""
        with self._lock:
            return self._dropped

    def measured_fps(self) -> float:
        """Frames per second arriving on the pipe.

        Counted by the reader thread, so it reflects what FFmpeg is actually
        delivering rather than how often the display happened to ask.
        """
        dt = time.perf_counter() - self._t0
        with self._lock:
            n = self._frames
        return n / dt if dt > 0.2 else 0.0

    def reset_fps_counter(self) -> None:
        with self._lock:
            self._frames = 0
            self._dropped = 0
        self._t0 = time.perf_counter()

    def __enter__(self) -> "CapturePreview":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def flicker_from_capture(spec, cfg_paths: dict | None = None,
                         seconds: float = 2.0, mains_hz: float = 50.0,
                         progress=None) -> FlickerResult:
    """Flicker test through the capture pipeline, at the capture frame rate.

    Strictly better than testing through the preview, because aliasing
    depends on the sampling rate: a 400 Hz PWM ripple sampled at 240 fps
    folds to 160 Hz and sampled at 25 fps folds to somewhere else entirely.
    The only rate whose verdict means anything is the one the trial will use,
    so this runs FFmpeg undecimated for a couple of seconds and measures
    every frame.
    """
    from . import ffmpeg_tools, recorder
    ff = ffmpeg_tools.ffmpeg_path((cfg_paths or {}).get("ffmpeg", ""))
    cmd = ([str(ff), "-hide_banner", "-loglevel", "warning", "-nostdin"]
           + recorder.dshow_input_args(spec)
           + ["-an", "-t", f"{seconds:.3f}",
              "-pix_fmt", "gray", "-f", "rawvideo", "-"])
    log.info("flicker (capture rate): %s", " ".join(cmd))
    creation = {}
    if paths.is_windows():  # pragma: no cover - Windows only
        creation["creationflags"] = getattr(
            subprocess, "CREATE_NO_WINDOW", 0x08000000)
    need = int(spec.width) * int(spec.height)
    means: list[float] = []
    if progress:
        progress(1, 2, f"sampling {seconds:.0f} s at {spec.requested_fps:g} fps…")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                stdin=subprocess.DEVNULL, bufsize=0, **creation)
    except OSError as e:
        res = FlickerResult(mains_hz=mains_hz)
        res.level, res.verdict = "warn", f"could not start FFmpeg: {e}"
        return res
    try:
        while True:
            buf = proc.stdout.read(need) if proc.stdout else b""
            if len(buf) < need:
                break
            g = np.frombuffer(buf, dtype=np.uint8)
            means.append(float(g.mean()))
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):  # pragma: no cover
            proc.kill()
        err = (proc.stderr.read().decode("utf-8", "replace")
               if proc.stderr else "")
    if len(means) < 32:
        res = FlickerResult(mains_hz=mains_hz)
        res.frames = len(means)
        res.level = "warn"
        tail = [ln for ln in err.splitlines() if ln.strip()][-3:]
        res.verdict = (f"only {len(means)} frames arrived - "
                       + (recorder.explain_exit(proc.returncode, tail)
                          if tail else "the camera delivered too little to judge"))
        return res
    if progress:
        progress(2, 2, "computing the spectrum…")
    # The measured rate, not the requested one: the camera negotiates.
    return flicker_from_means(means, len(means) / seconds, mains_hz)


def open_device_properties(spec, cfg_paths: dict | None = None) -> str:
    """Open the camera driver's own DirectShow property page.

    The authoritative way to set exposure and gain on Windows, and the only
    way to set them *in the capture media type*: FFmpeg holds the device in
    MJPEG at the capture frame rate while the page is up, so whatever the
    operator sets is set against the mode the trial will use. OpenCV cannot
    do this on this camera -- the driver refuses MJPG to it.

    Returns a message for the status bar; the dialog is modal to FFmpeg, not
    to the application, so this call returns as soon as FFmpeg is launched.
    """
    from . import ffmpeg_tools, recorder
    ff = ffmpeg_tools.ffmpeg_path((cfg_paths or {}).get("ffmpeg", ""))
    cmd = ([str(ff), "-hide_banner", "-loglevel", "error",
            "-f", "dshow", "-show_video_device_dialog", "true"]
           + recorder.dshow_input_args(spec)[2:]
           + ["-t", "0.1", "-f", "null", "-"])
    log.info("device properties: %s", " ".join(cmd))
    try:
        subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        return f"Could not open the driver's property page: {e}"
    return ("The camera driver's own property page is opening. Set exposure "
            "and gain there, close it, then press Check capture to confirm "
            "what the file looks like.")
