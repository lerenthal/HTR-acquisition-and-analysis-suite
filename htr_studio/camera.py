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
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

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

    res.frames = len(means)
    if res.frames < 32:
        res.verdict = "not enough frames captured to judge flicker"
        res.level = "warn"
        return res

    res.fps = nominal_fps if nominal_fps > 0 else res.frames / max(elapsed, 1e-6)
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
