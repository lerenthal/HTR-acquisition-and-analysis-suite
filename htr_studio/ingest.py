"""Bringing in foreign footage, and getting long recordings down to a size
that can be stored -- without destroying the signal in the process.

Two jobs that share one constraint.

**Ingest.** Footage from another capture program, or from a paper's
supplementary material, arrives with no sidecar and often with a frame rate that
is not the rate it was captured at. :func:`inspect_file` measures what is
actually in the file -- true frame rate from an exact frame count, whether it is
all-intra, how many bytes a frame costs -- and says what to do about it.

**Archive.** This is where almost everyone goes wrong, so the reasoning is worth
stating plainly: **an HTR recording must stay all-intra.** The twitch is a few
pixels of head displacement oscillating at 78-98 Hz. An inter-frame codec exists
to *remove* small frame-to-frame differences: motion estimation describes the
head's movement as a block vector, the residual is small, and the quantiser
throws it away. What survives is a head that moves in visible jumps between
keyframes, which is the opposite of the signal. H.264 at default settings has
destroyed HTR footage that looked perfectly fine to the eye -- it is a codec
tuned for how video looks to a person, and a person cannot see 90 Hz.

So every option here keeps ``-g 1`` (every frame a keyframe) and no B-frames.
Within that constraint the savings are still large, and they come, in order of
how much they save per unit of risk:

1. **Crop to the arena.** Free, scientifically. A 320x240 frame of which the
   cylinder occupies 150x150 is 60% wasted pixels, and cropping them costs
   nothing at all.
2. **All-intra H.264 or FFV1.** FFV1 is mathematically lossless and typically
   55-70% of MJPEG. All-intra H.264 at a high quality level is smaller again and
   visually indistinguishable, but it is lossy, so it is the choice for storage
   after analysis rather than before it.
3. **Re-quantised MJPEG.** Same codec the camera produced, lower quality. The
   least disruptive change of all, since nothing about the decode path changes.

What is *not* offered: reducing the frame rate. Dropping from 240 to 120 fps
halves the file and destroys the measurement -- the Nyquist limit falls to 60 Hz,
below the band. There is no "archive at a lower frame rate" option because there
is no correct way to do it.

Sizes are predicted by measuring a sample of the real file rather than by a
table of bitrates, because MJPEG's cost depends almost entirely on how much
texture the bedding has.
"""
from __future__ import annotations

import json
import logging
import math
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import ffmpeg_tools, paths

log = logging.getLogger(__name__)

#: Codecs that code every frame independently. Only these are safe for HTR.
ALL_INTRA = {"mjpeg", "mjpg", "ffv1", "rawvideo", "huffyuv", "ffvhuff",
             "jpeg2000", "jpegls", "dvvideo", "cinepak", "msvideo1", "qtrle",
             "png", "tiff", "utvideo", "v210", "yuv4"}
#: Codecs that describe a frame in terms of its neighbours. These smear the
#: oscillation; a file in one of them can still be analysed, but the result has
#: to be treated as provisional until it is checked against the original.
INTER_FRAME = {"h264", "hevc", "mpeg4", "mpeg2video", "vp8", "vp9", "av1",
               "wmv2", "wmv3", "vc1", "theora", "msmpeg4v3", "flv1"}


# ==========================================================================
# Inspection
# ==========================================================================
@dataclass
class FileReport:
    path: Path
    ok: bool = False
    codec: str = ""
    width: int = 0
    height: int = 0
    container_fps: float = 0.0
    true_fps: float = 0.0
    exact: bool = False
    frames: int = 0
    duration_s: float = 0.0
    size_bytes: int = 0
    bytes_per_frame: float = 0.0
    all_intra: bool | None = None       # None = unknown codec
    pix_fmt: str = ""
    verdict: str = ""                   # "ok" | "warn" | "bad"
    findings: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)

    @property
    def nyquist_hz(self) -> float:
        return self.true_fps / 2.0

    def as_text(self) -> str:
        lines = [
            f"file           {self.path.name}",
            f"codec          {self.codec or '?'}"
            + ("  (all-intra - safe)" if self.all_intra
               else "  (inter-frame - see below)" if self.all_intra is False
               else "  (unrecognised)"),
            f"frame size     {self.width}x{self.height}  {self.pix_fmt}",
            f"declared rate  {self.container_fps:.3f} fps",
            f"measured rate  {self.true_fps:.3f} fps"
            + ("  (exact frame count)" if self.exact else "  (from the header)"),
            f"frames         {self.frames:,}",
            f"duration       {self.duration_s / 60:.2f} min",
            f"size           {self.size_bytes / 1e9:.3f} GB "
            f"({self.bytes_per_frame:,.0f} bytes per frame)",
            f"Nyquist limit  {self.nyquist_hz:.1f} Hz "
            + ("- covers the 78-98 Hz HTR band"
               if self.nyquist_hz >= 98 else "- BELOW the HTR band"),
            "",
        ]
        if self.findings:
            lines.append("FINDINGS")
            lines += [f"  - {f}" for f in self.findings]
            lines.append("")
        if self.actions:
            lines.append("WHAT TO DO")
            lines += [f"  {i}. {a}" for i, a in enumerate(self.actions, 1)]
        return "\n".join(lines)


def inspect_file(path: Path | str, ffprobe: str = "",
                 count_frames: bool = True) -> FileReport:
    """Measure a video and say whether it can carry an HTR measurement.

    ``count_frames`` decodes the whole file to get an exact count. It is slow
    (tens of seconds for a 30-minute trial) and it is the only way to know the
    real frame rate of MJPEG in AVI, where the header rate is routinely wrong.
    """
    p = Path(path)
    rep = FileReport(path=p)
    if not p.exists():
        rep.findings.append("the file does not exist")
        rep.verdict = "bad"
        return rep
    try:
        rep.size_bytes = p.stat().st_size
    except OSError:
        pass

    info = ffmpeg_tools.probe_video(str(p), ffprobe, count_frames=count_frames)
    rep.ok = bool(info.width and info.height)
    rep.codec = (info.codec or "").lower()
    rep.width, rep.height = info.width, info.height
    rep.container_fps = info.container_fps if hasattr(info, "container_fps") else 0.0
    rep.true_fps = info.true_fps
    rep.exact = info.exact
    rep.frames = getattr(info, "frames", 0) or 0
    rep.duration_s = info.duration_s
    rep.pix_fmt = getattr(info, "pix_fmt", "") or ""
    if not rep.container_fps:
        rep.container_fps = rep.true_fps
    if rep.frames and rep.size_bytes:
        rep.bytes_per_frame = rep.size_bytes / rep.frames

    if rep.codec in ALL_INTRA:
        rep.all_intra = True
    elif rep.codec in INTER_FRAME:
        rep.all_intra = False

    if not rep.ok:
        rep.findings.append("FFprobe could not read a video stream from this file")
        rep.verdict = "bad"
        return rep

    worst = "ok"

    def flag(level: str, msg: str) -> None:
        nonlocal worst
        rep.findings.append(msg)
        if level == "bad" or (level == "warn" and worst == "ok"):
            worst = level

    if rep.all_intra is False:
        flag("warn",
             f"{rep.codec} codes each frame from its neighbours. Motion "
             "estimation and quantisation remove exactly the small, fast head "
             "displacement an HTR consists of, so a twitch may be smeared or "
             "gone even though the video looks fine. Counts from this file are "
             "provisional; confirm against the original capture if one exists.")
    elif rep.all_intra is None:
        flag("warn", f"codec '{rep.codec}' is not one this tool recognises, so "
                     "whether it preserves fast motion is unknown.")

    if rep.true_fps < 196:
        if rep.true_fps < 60:
            flag("bad",
                 f"At {rep.true_fps:.1f} fps the Nyquist limit is "
                 f"{rep.nyquist_hz:.1f} Hz, far below the 78-98 Hz HTR band. "
                 "Either this file is a slow-motion render whose declared rate "
                 "is the playback rate -- common in published supplementary "
                 "videos, and entirely fixable with the capture-rate override "
                 "-- or it was genuinely captured too slowly, in which case the "
                 "oscillation aliased at the sensor and no analysis can recover "
                 "it.")
        else:
            flag("bad",
                 f"{rep.true_fps:.1f} fps gives a Nyquist limit of "
                 f"{rep.nyquist_hz:.1f} Hz, below the 78-98 Hz HTR band. The "
                 "oscillation aliases and the counts will be wrong.")

    if (abs(rep.container_fps - rep.true_fps) > max(1.0, 0.02 * rep.true_fps)
            and rep.exact):
        flag("warn",
             f"the header says {rep.container_fps:.3f} fps but counting frames "
             f"gives {rep.true_fps:.3f} fps. Timing is taken from the measured "
             "rate.")
    if not rep.exact and count_frames:
        flag("warn", "the frame count could not be verified, so the frame rate "
                     "is the header's claim. Every timing figure depends on it.")

    short = min(rep.width, rep.height)
    if short and short > 500:
        flag("warn",
             f"{rep.width}x{rep.height} is larger than this measurement needs. "
             "Frame rate buys the signal; resolution only has to resolve the "
             "head against the body, which 320x240 does. A larger frame means a "
             "lower achievable frame rate and a much bigger file.")

    rep.verdict = worst
    rep.actions = _actions_for(rep)
    return rep


def _actions_for(rep: FileReport) -> list[str]:
    acts: list[str] = []
    if rep.true_fps < 60:
        acts.append(
            "Play the file. If the animal moves at normal speed it was captured "
            "at this rate and cannot be used; if it moves in slow motion, open "
            "it on the Analysis tab, press 'Signal diagnostics' and apply the "
            "capture rate it suggests.")
    if rep.all_intra is False:
        acts.append(
            "If the original capture still exists, analyse that instead of this "
            "re-encode.")
    if not rep.exact:
        acts.append("Run 'Verify frame rate' in the Library to count frames "
                    "exactly before trusting any timing.")
    if rep.size_bytes > 5e9:
        acts.append(
            f"At {rep.size_bytes / 1e9:.1f} GB this file is worth archiving - "
            "see 'Archive / convert', which keeps the footage all-intra.")
    if not acts:
        acts.append("Nothing to fix. Add it to the Library and analyse it.")
    return acts


# ==========================================================================
# Archiving
# ==========================================================================
@dataclass
class ArchiveOption:
    key: str
    label: str
    why: str
    args: list[str]
    lossless: bool
    needs_encoder: str = ""


def archive_options() -> list[ArchiveOption]:
    """The conversions that are safe for an HTR recording.

    Every one is all-intra (``-g 1``, no B-frames). Nothing here changes the
    frame rate or the frame count, so a file converted with any of them analyses
    to the same timebase as the original.
    """
    return [
        ArchiveOption(
            "copy", "Repackage only (no re-encode)",
            "Rewrites the container and rebuilds the index without touching a "
            "single pixel. Use it to turn a half-written AVI into a readable "
            "file, or to convert AVI to MKV. Identical size, zero risk.",
            ["-c:v", "copy"], lossless=True),
        ArchiveOption(
            "ffv1", "FFV1, mathematically lossless",
            "Every pixel recoverable bit for bit. Right for footage captured "
            "uncompressed, where it typically saves half. WRONG for footage "
            "that is already MJPEG: lossless coding has to preserve the JPEG "
            "ringing and the sensor noise exactly, and that costs more than the "
            "JPEG did - measured at 5x larger on 240 fps camera footage. "
            "Measure before choosing it.",
            ["-c:v", "ffv1", "-level", "3", "-coder", "1", "-context", "1",
             "-g", "1", "-slices", "4", "-slicecrc", "1"],
            lossless=True, needs_encoder="ffv1"),
        ArchiveOption(
            "mjpeg_q4", "MJPEG, high quality",
            "Same codec the camera produces, re-quantised. Nothing about the "
            "decode path changes, which makes it the least disruptive lossy "
            "option. Expect 60-80% of the original.",
            ["-c:v", "mjpeg", "-q:v", "4", "-g", "1"],
            lossless=False, needs_encoder="mjpeg"),
        ArchiveOption(
            "h264_intra", "H.264, all-intra, visually lossless",
            "Every frame a keyframe, no B-frames, no motion estimation between "
            "frames - so none of the inter-frame smearing that makes ordinary "
            "H.264 unsafe here. Typically 25-40% of the MJPEG original. The "
            "choice for long-term storage after the analysis is done.",
            ["-c:v", "libx264", "-preset", "slow", "-crf", "15",
             "-g", "1", "-bf", "0", "-x264-params", "keyint=1:scenecut=0",
             "-pix_fmt", "yuv420p"],
            lossless=False, needs_encoder="libx264"),
        ArchiveOption(
            "h264_intra_lossless", "H.264, all-intra, lossless",
            "Bit-exact within the chosen pixel format, all-intra. Smaller than "
            "FFV1 on some footage and larger on others; worth measuring both.",
            ["-c:v", "libx264", "-preset", "veryslow", "-qp", "0",
             "-g", "1", "-bf", "0", "-x264-params", "keyint=1:scenecut=0"],
            lossless=True, needs_encoder="libx264"),
    ]


@dataclass
class ArchiveSpec:
    source: Path
    dest: Path
    option: str = "ffv1"
    crop: tuple[int, int, int, int] | None = None   # w, h, x, y
    container: str = "mkv"

    def filters(self) -> list[str]:
        if not self.crop:
            return []
        w, h, x, y = self.crop
        return ["-vf", f"crop={w}:{h}:{x}:{y}"]


def build_archive_command(spec: ArchiveSpec, ffmpeg: str = "") -> list[str]:
    opt = next((o for o in archive_options() if o.key == spec.option), None)
    if opt is None:
        raise ValueError(f"unknown archive option {spec.option!r}")
    if spec.crop and opt.key == "copy":
        raise ValueError("cropping requires a re-encode; 'copy' cannot crop")
    ff = ffmpeg_tools.ffmpeg_path(ffmpeg)
    return ([str(ff), "-hide_banner", "-loglevel", "info", "-stats",
             "-i", str(spec.source)]
            + spec.filters() + opt.args
            + ["-an", "-y", str(spec.dest)])


@dataclass
class SizeEstimate:
    option: str
    label: str
    sample_frames: int
    sample_bytes: int
    predicted_bytes: int
    ratio: float
    lossless: bool
    note: str = ""


def estimate_sizes(source: Path | str, options: list[str] | None = None,
                   sample_seconds: float = 6.0,
                   crop: tuple[int, int, int, int] | None = None,
                   ffmpeg: str = "", ffprobe: str = "",
                   progress: Callable[[str], None] | None = None
                   ) -> list[SizeEstimate]:
    """Encode a few seconds with each option and scale the result up.

    A bitrate table would be quicker and would be wrong: MJPEG's cost is set by
    how much texture the bedding has, and FFV1's by how noisy the sensor is, so
    the only honest prediction comes from the operator's own footage. The sample
    is taken from a third of the way in, past any settling at the start.
    """
    src = Path(source)
    rep = inspect_file(src, ffprobe, count_frames=False)
    keys = options or ["ffv1", "mjpeg_q4", "h264_intra"]
    avail = available_encoders(ffmpeg)
    out: list[SizeEstimate] = []
    start = max(0.0, rep.duration_s / 3.0)
    total = max(rep.size_bytes, 1)

    tmp = paths.DATA_DIR / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for key in keys:
        opt = next((o for o in archive_options() if o.key == key), None)
        if opt is None:
            continue
        if opt.needs_encoder and opt.needs_encoder not in avail:
            out.append(SizeEstimate(
                key, opt.label, 0, 0, 0, 0.0, opt.lossless,
                note=f"the {opt.needs_encoder} encoder is not in this FFmpeg "
                     "build, so this option is unavailable"))
            continue
        if progress:
            progress(f"measuring {opt.label}…")
        dest = tmp / f"sizeprobe_{key}.{'mkv' if key != 'mjpeg_q4' else 'avi'}"
        dest.unlink(missing_ok=True)
        cmd = ([str(ffmpeg_tools.ffmpeg_path(ffmpeg)), "-hide_banner",
                "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src),
                "-t", f"{sample_seconds:.3f}"]
               + (["-vf", "crop={}:{}:{}:{}".format(*crop)] if crop else [])
               + opt.args + ["-an", "-y", str(dest)])
        try:
            r = _run(cmd)
        except OSError as e:
            out.append(SizeEstimate(key, opt.label, 0, 0, 0, 0.0, opt.lossless,
                                    note=f"could not run FFmpeg: {e}"))
            continue
        if r.returncode != 0 or not dest.exists():
            out.append(SizeEstimate(
                key, opt.label, 0, 0, 0, 0.0, opt.lossless,
                note=(r.stderr or "").strip().splitlines()[-1][:180]
                     if r.stderr else "encode failed"))
            continue
        sample_bytes = dest.stat().st_size
        got = ffmpeg_tools.probe_video(str(dest), ffprobe, count_frames=True)
        n = getattr(got, "frames", 0) or 0
        if n <= 0 and got.true_fps > 0:
            n = int(round(got.duration_s * got.true_fps))
        per_frame = sample_bytes / n if n else 0.0
        predicted = int(per_frame * rep.frames) if rep.frames else int(
            sample_bytes * (rep.duration_s / max(sample_seconds, 0.001)))
        est = SizeEstimate(key, opt.label, n, sample_bytes, predicted,
                           predicted / total if total else 0.0, opt.lossless)
        if key != "copy" and est.ratio > 1.0:
            # Worth saying out loud: a lossless codec applied to an already
            # lossy source has to encode the compression artefacts too, and on
            # noisy high-frame-rate footage that is bigger than the source.
            est.note = (
                f"larger than the original ({est.ratio:.1f}x). "
                + ("Lossless coding has to reproduce the source's own JPEG "
                   "artefacts and sensor noise exactly, which costs more than "
                   "the JPEG did. Use this only on uncompressed footage."
                   if opt.lossless else
                   "This footage is too noisy for the codec to help. Crop to "
                   "the arena instead, or keep the original."))
        out.append(est)
        dest.unlink(missing_ok=True)
    return out


def available_encoders(ffmpeg: str = "") -> set[str]:
    """Which of the encoders the options need this FFmpeg build actually has."""
    try:
        r = _run([str(ffmpeg_tools.ffmpeg_path(ffmpeg)), "-hide_banner",
                  "-encoders"])
    except OSError:
        return set()
    text = (r.stdout or "") + (r.stderr or "")
    found = set()
    for name in ("ffv1", "mjpeg", "libx264", "libx265", "utvideo", "ffvhuff"):
        if re.search(rf"^\s*\S+\s+{re.escape(name)}\b", text, re.M):
            found.add(name)
    return found


def _run(cmd: list[str], timeout: float = 600.0):
    creation = {}
    if paths.is_windows():  # pragma: no cover
        creation["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW",
                                            0x08000000)
    return subprocess.run(cmd, capture_output=True, text=True,
                          errors="replace", timeout=timeout, **creation)


# ==========================================================================
# Conversion with progress, and verification afterwards
# ==========================================================================
@dataclass
class ArchiveResult:
    ok: bool
    dest: Path
    source_bytes: int = 0
    dest_bytes: int = 0
    source_frames: int = 0
    dest_frames: int = 0
    message: str = ""

    @property
    def ratio(self) -> float:
        return self.dest_bytes / self.source_bytes if self.source_bytes else 0.0


_FRAME = re.compile(r"frame=\s*(\d+)")


def archive(spec: ArchiveSpec, ffmpeg: str = "", ffprobe: str = "",
            progress: Callable[[int, int, str], bool] | None = None,
            verify: bool = True) -> ArchiveResult:
    """Convert, then prove the conversion kept every frame.

    The verification is not optional padding. A conversion that silently drops
    the last few seconds, or that re-times the stream, produces a file whose
    analysis disagrees with the original for reasons nobody will think to look
    for a year later. So the frame count is compared, and a mismatch is a
    failure, not a warning.
    """
    src, dest = spec.source, spec.dest
    before = inspect_file(src, ffprobe, count_frames=verify)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return ArchiveResult(False, dest, message=f"{dest.name} already exists")

    cmd = build_archive_command(spec, ffmpeg)
    log.info("archiving: %s", " ".join(cmd))
    creation = {}
    if paths.is_windows():  # pragma: no cover
        creation["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW",
                                            0x08000000)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True,
                            errors="replace", bufsize=1, **creation)
    tail: list[str] = []
    cancelled = False
    buf = ""
    assert proc.stderr is not None
    while True:
        ch = proc.stderr.read(1)
        if not ch:
            break
        if ch in ("\r", "\n"):
            line, buf = buf.strip(), ""
            if not line:
                continue
            tail.append(line)
            del tail[:-40]
            m = _FRAME.search(line)
            if m and progress is not None:
                if not progress(int(m.group(1)), before.frames, "converting"):
                    cancelled = True
                    proc.terminate()
                    break
        else:
            buf += ch
    rc = proc.wait()

    if cancelled:
        dest.unlink(missing_ok=True)
        return ArchiveResult(False, dest, message="cancelled; the partial file "
                                                  "was removed")
    if rc != 0 or not dest.exists():
        detail = next((t for t in reversed(tail) if t), "no detail")
        dest.unlink(missing_ok=True)
        return ArchiveResult(False, dest,
                             message=f"FFmpeg exited with {rc}: {detail[:200]}")

    res = ArchiveResult(True, dest, source_bytes=before.size_bytes,
                        dest_bytes=dest.stat().st_size,
                        source_frames=before.frames)
    if verify:
        after = inspect_file(dest, ffprobe, count_frames=True)
        res.dest_frames = after.frames
        if before.frames and after.frames and before.frames != after.frames:
            res.ok = False
            res.message = (
                f"the conversion changed the frame count: {before.frames:,} in, "
                f"{after.frames:,} out. The output is NOT equivalent to the "
                "original and must not replace it.")
            return res
        res.message = (
            f"{res.dest_bytes / 1e9:.3f} GB, {res.ratio:.0%} of the original, "
            f"{after.frames:,} frames verified identical in count.")
    else:
        res.message = (f"{res.dest_bytes / 1e9:.3f} GB, {res.ratio:.0%} of the "
                       "original (frame count not verified).")
    return res


# ==========================================================================
# Planning a long capture
# ==========================================================================
@dataclass
class CapturePlan:
    minutes: float
    fps: float
    width: int
    height: int
    pixel_format: str
    bytes_per_frame: float
    total_bytes: int
    segments: int
    segment_bytes: int
    notes: list[str] = field(default_factory=list)


#: Measured on the B0634: 320x240 MJPEG at 240 fps produced 17.7 MB in 30 s,
#: i.e. ~2456 bytes per frame. Scaled by pixel count for other modes.
MJPEG_BYTES_PER_FRAME_320x240 = 2456.0


def plan_capture(minutes: float, fps: float, width: int, height: int,
                 pixel_format: str = "mjpeg",
                 segment_minutes: float = 5.0) -> CapturePlan:
    """What a given capture will cost on disk, and whether that is sane.

    This exists because the usual reaction to "30 minutes at high frame rate" is
    to assume the files will be unmanageable, and then to reach for a lower frame
    rate -- which is the one change that destroys the measurement. The arithmetic
    says otherwise: MJPEG at 320x240 is about 2.5 kB a frame, so half an hour at
    240 fps is near 1 GB. It is *uncompressed* capture that is unmanageable, and
    the fix for that is to let the camera's own MJPEG encoder do its job, not to
    slow the camera down.
    """
    px = max(1, width * height)
    notes: list[str] = []
    if pixel_format in ("mjpeg", "mjpg"):
        bpf = MJPEG_BYTES_PER_FRAME_320x240 * px / (320 * 240)
    elif pixel_format in ("yuyv422", "yuyv", "uyvy422"):
        bpf = px * 2.0
        notes.append(
            "Uncompressed YUYV is about 27 times the size of the camera's own "
            "MJPEG for the same frames, and it gains nothing: the camera "
            "encodes MJPEG in hardware, and MJPEG is all-intra, so every frame "
            "is still independent. This is the usual reason a high-frame-rate "
            "recording turns out unmanageably large.")
    elif pixel_format in ("gray", "y8"):
        bpf = px * 1.0
        notes.append("Uncompressed 8-bit grey is about 13 times the camera's "
                     "MJPEG for the same frames.")
    else:
        bpf = px * 1.5
        notes.append(f"'{pixel_format}' is not a format this planner has "
                     "measured; the estimate assumes 1.5 bytes per pixel.")

    total = int(bpf * fps * minutes * 60.0)
    seg_n = max(1, int(math.ceil(minutes / segment_minutes))) if segment_minutes > 0 else 1
    plan = CapturePlan(minutes, fps, width, height, pixel_format, bpf, total,
                       seg_n, total // seg_n, notes)

    if fps < 196:
        plan.notes.append(
            f"{fps:.0f} fps cannot carry this measurement: the Nyquist limit is "
            f"{fps / 2:.0f} Hz and the HTR band is 78-98 Hz. Frame rate is the "
            "one parameter that must not be traded for file size.")
    if total > 20e9:
        plan.notes.append(
            f"{total / 1e9:.0f} GB per trial. Record a smaller frame, or use the "
            "camera's MJPEG mode, before accepting a figure like this.")
    if seg_n > 1:
        plan.notes.append(
            f"Written as {seg_n} segments of {plan.segment_bytes / 1e9:.2f} GB, "
            "so an interruption costs at most one segment and each file can be "
            "copied and opened on its own.")
    free = _free_bytes(paths.RECORDINGS_DIR)
    if free and total * 1.25 > free:
        plan.notes.append(
            f"The recordings disk has {free / 1e9:.1f} GB free, which is not "
            f"enough for this trial plus headroom.")
    return plan


def _free_bytes(where: Path) -> int:
    try:
        where.mkdir(parents=True, exist_ok=True)
        return shutil.disk_usage(where).free
    except OSError:
        return 0


def write_report(path: Path, rep: FileReport,
                 estimates: list[SizeEstimate] | None = None) -> Path:
    """Save an inspection (and any size measurements) next to the footage."""
    doc = {
        "htr_studio_ingest_version": 1,
        "file": str(rep.path),
        "codec": rep.codec,
        "all_intra": rep.all_intra,
        "width": rep.width, "height": rep.height,
        "container_fps": rep.container_fps,
        "true_fps": rep.true_fps,
        "frame_rate_verified": rep.exact,
        "frames": rep.frames,
        "duration_s": rep.duration_s,
        "size_bytes": rep.size_bytes,
        "bytes_per_frame": round(rep.bytes_per_frame, 1),
        "nyquist_hz": rep.nyquist_hz,
        "verdict": rep.verdict,
        "findings": rep.findings,
        "actions": rep.actions,
    }
    if estimates:
        doc["archive_estimates"] = [
            {"option": e.option, "label": e.label, "lossless": e.lossless,
             "predicted_bytes": e.predicted_bytes,
             "fraction_of_original": round(e.ratio, 4), "note": e.note}
            for e in estimates]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path
