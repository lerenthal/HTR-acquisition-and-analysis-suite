"""Reusable Qt pieces: status labels, metric tiles, frame view, workers.

Two conventions worth stating because the rest of the UI depends on them:

* **A status colour never travels alone.** :class:`StatusLabel` always shows a
  glyph and a sentence, so the three verdict levels are readable in greyscale,
  in print, and to a colour-blind reader.
* **Nothing touches a Qt widget off the main thread.** Long jobs run in a
  :class:`Worker` thread and report back through signals, or write into a
  plain object that a QTimer polls.
"""
from __future__ import annotations

import logging
import threading
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .. import theme

log = logging.getLogger(__name__)

GLYPH = {"ok": "✓", "warn": "⚠", "bad": "✕", "info": "•"}


# --------------------------------------------------------------------------
# Small display widgets
# --------------------------------------------------------------------------
class StatusLabel(QtWidgets.QLabel):
    """A one-line verdict: glyph + text, coloured by level."""

    def __init__(self, text: str = "", level: str = "info", parent=None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.set(text, level)

    def set(self, text: str, level: str = "info") -> None:
        self._level = level
        name = theme.STATUS_OBJECT_NAMES.get(level, "hint")
        self.setObjectName(name)
        self.setText(f"{GLYPH.get(level, GLYPH['info'])}  {text}" if text else "")
        # Re-apply the stylesheet so the new object name takes effect.
        self.style().unpolish(self)
        self.style().polish(self)


class MetricTile(QtWidgets.QFrame):
    """A label and a big number. Used where a chart would be overkill."""

    def __init__(self, caption: str, value: str = "-", unit: str = "", parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.setSpacing(0)
        self.caption = QtWidgets.QLabel(caption)
        self.caption.setObjectName("hint")
        self.value = QtWidgets.QLabel(value)
        self.value.setObjectName("metric")
        self.unit = QtWidgets.QLabel(unit)
        self.unit.setObjectName("hint")
        lay.addWidget(self.caption)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(self.value)
        row.addWidget(self.unit, 0, QtCore.Qt.AlignBottom)
        row.addStretch(1)
        lay.addLayout(row)

    def set_value(self, value: str, unit: str | None = None) -> None:
        self.value.setText(value)
        if unit is not None:
            self.unit.setText(unit)


class LabeledSlider(QtWidgets.QWidget):
    """Slider plus a spin box, kept in sync, with an optional live hint."""

    valueChanged = QtCore.pyqtSignal(float)

    def __init__(self, label: str, minimum: float, maximum: float, value: float,
                 step: float = 1.0, decimals: int = 0, suffix: str = "",
                 hint: Callable[[float], str] | None = None, parent=None):
        super().__init__(parent)
        self._scale = 10 ** decimals
        self._hint_fn = hint
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setVerticalSpacing(1)

        self.label = QtWidgets.QLabel(label)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(int(minimum * self._scale), int(maximum * self._scale))
        self.slider.setSingleStep(max(1, int(step * self._scale)))
        self.spin = (QtWidgets.QDoubleSpinBox() if decimals else QtWidgets.QSpinBox())
        if decimals:
            self.spin.setDecimals(decimals)
            self.spin.setSingleStep(step)
        self.spin.setRange(minimum, maximum) if decimals else \
            self.spin.setRange(int(minimum), int(maximum))
        if suffix:
            self.spin.setSuffix(f" {suffix}")
        self.hint = QtWidgets.QLabel("")
        self.hint.setObjectName("hint")

        lay.addWidget(self.label, 0, 0)
        lay.addWidget(self.spin, 0, 1)
        lay.addWidget(self.slider, 1, 0, 1, 2)
        lay.addWidget(self.hint, 2, 0, 1, 2)
        lay.setColumnStretch(0, 1)

        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self.set_value(value)

    def _from_slider(self, raw: int) -> None:
        """Mirror the slider into the spin box.

        ``raw / self._scale`` is a float even when ``_scale`` is 1, because
        Python 3 ``/`` always returns a float: 240 / 1 == 240.0. A QSpinBox
        (the widget used whenever decimals == 0) refuses a float outright,
        so dragging ANY integer slider raised TypeError here before
        ``_emit`` could run -- the spin box never updated and, worse, the
        valueChanged signal never fired, so the value was never pushed to
        the camera. Every exposure/gain/brightness/contrast/gamma slider on
        the Acquisition tab was dead on drag; only typing into the spin box
        worked. Cast to the type the widget actually accepts.
        """
        v = raw / self._scale
        self.spin.blockSignals(True)
        self.spin.setValue(v if self._scale != 1 else int(raw))
        self.spin.blockSignals(False)
        self._emit(v)

    def _from_spin(self, v) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(float(v) * self._scale))
        self.slider.blockSignals(False)
        self._emit(float(v))

    def _emit(self, v: float) -> None:
        if self._hint_fn:
            self.hint.setText(self._hint_fn(v))
        self.valueChanged.emit(v)

    def value(self) -> float:
        return float(self.spin.value())

    def set_value(self, v: float) -> None:
        self.spin.blockSignals(True)
        self.slider.blockSignals(True)
        self.spin.setValue(v if isinstance(self.spin, QtWidgets.QDoubleSpinBox) else int(v))
        self.slider.setValue(int(v * self._scale))
        self.spin.blockSignals(False)
        self.slider.blockSignals(False)
        if self._hint_fn:
            self.hint.setText(self._hint_fn(v))


class FrameView(QtWidgets.QLabel):
    """Displays a grayscale frame, scaled to fit, with optional overlays."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self._pal = theme.palette("light")
        self.setStyleSheet(f"background: {self._pal.plane}; "
                           f"border: 1px solid {self._pal.grid}; border-radius: 4px;")
        self.setText("no frame")
        self._last_bgr: np.ndarray | None = None

    def set_palette_tokens(self, pal: theme.Palette) -> None:
        self._pal = pal
        self.setStyleSheet(f"background: {pal.plane}; "
                           f"border: 1px solid {pal.grid}; border-radius: 4px;")

    @staticmethod
    def _hex_to_bgr(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)

    def show_frame(self, gray: np.ndarray, overlays: dict | None = None,
                   caption: str = "") -> None:
        if gray is None or getattr(gray, "size", 0) == 0:
            self.setText("no frame")
            return
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()
        h, w = vis.shape[:2]
        ov = overlays or {}

        if ov.get("mask_contour") is not None:
            cv2.drawContours(vis, ov["mask_contour"], -1,
                             self._hex_to_bgr(self._pal.s3), 1, cv2.LINE_AA)
        c, hp = ov.get("centroid"), ov.get("head")
        if c and all(np.isfinite(c)):
            cv2.circle(vis, (int(c[0]), int(c[1])), 3,
                       self._hex_to_bgr(self._pal.s2), -1, cv2.LINE_AA)
        if hp and all(np.isfinite(hp)):
            cv2.circle(vis, (int(hp[0]), int(hp[1])), 4,
                       self._hex_to_bgr(self._pal.s1), -1, cv2.LINE_AA)
            if c and all(np.isfinite(c)):
                cv2.arrowedLine(vis, (int(c[0]), int(c[1])), (int(hp[0]), int(hp[1])),
                                self._hex_to_bgr(self._pal.s1), 1, cv2.LINE_AA,
                                tipLength=0.18)
        if ov.get("crosshair"):
            col = self._hex_to_bgr(self._pal.axis)
            cv2.line(vis, (w // 2, 0), (w // 2, h), col, 1)
            cv2.line(vis, (0, h // 2), (w, h // 2), col, 1)
        if ov.get("burst"):
            cv2.rectangle(vis, (1, 1), (w - 2, h - 2),
                          self._hex_to_bgr(self._pal.critical), 2)
        if caption:
            cv2.putText(vis, caption, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        self._hex_to_bgr(self._pal.s1), 1, cv2.LINE_AA)

        self._last_bgr = vis
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img).scaled(
            self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(pix)

    def save_png(self, path: Path) -> bool:
        if self._last_bgr is None:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(path), self._last_bgr))


class ChecklistTable(QtWidgets.QTableWidget):
    """Pre-flight style list: one row per check, verdict and detail."""

    def __init__(self, parent=None):
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["", "Check", "Detail"])
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QTableWidget.NoSelection)
        self.setAlternatingRowColors(True)
        self.setWordWrap(True)
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Fixed)
        self.setColumnWidth(0, 28)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)

    def set_rows(self, rows: list[tuple[str, str, str]], pal: theme.Palette) -> None:
        """``rows`` is a list of ``(level, name, detail)``."""
        colours = {"ok": pal.good, "warn": pal.serious, "bad": pal.critical,
                   "info": pal.ink_muted}
        self.setRowCount(len(rows))
        for r, (level, name, detail) in enumerate(rows):
            g = QtWidgets.QTableWidgetItem(GLYPH.get(level, GLYPH["info"]))
            g.setForeground(QtGui.QColor(colours.get(level, pal.ink)))
            g.setTextAlignment(QtCore.Qt.AlignCenter)
            self.setItem(r, 0, g)
            self.setItem(r, 1, QtWidgets.QTableWidgetItem(name))
            self.setItem(r, 2, QtWidgets.QTableWidgetItem(detail))
        self.resizeRowsToContents()


# --------------------------------------------------------------------------
# Background work
# --------------------------------------------------------------------------
class Worker(QtCore.QThread):
    """Runs a callable off the main thread.

    The callable is handed a ``progress(done, total, message) -> bool`` function;
    returning False from it (which happens once :meth:`cancel` is called) asks
    the job to stop. Results and errors come back as signals.
    """
    progressed = QtCore.pyqtSignal(int, int, str)
    finishedOk = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str, str)

    def __init__(self, fn: Callable[..., Any], *args, parent=None, **kwargs):
        super().__init__(parent)
        self._fn, self._args, self._kwargs = fn, args, kwargs
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def _progress(self, done: int, total: int, message: str = "") -> bool:
        self.progressed.emit(int(done), int(total), message)
        return not self._cancelled

    def run(self) -> None:  # pragma: no cover - thread body
        try:
            result = self._fn(*self._args, progress=self._progress, **self._kwargs)
        except Exception as e:
            log.exception("worker failed")
            self.failed.emit(str(e), traceback.format_exc())
            return
        self.finishedOk.emit(result)


class JobRunner(QtCore.QObject):
    """Owns one background job at a time, for one button.

    Two problems, one object.

    **The crash.** The pattern this replaces was ``self.worker = Worker(...)``
    at nine call sites, with nothing stopping a second start. Pressing the
    button twice rebinds the attribute, and if the first ``QThread`` is still
    running when Python collects it, Qt aborts the whole process with "QThread:
    Destroyed while thread is still running" -- the window vanishes, mid-trial,
    with no warning. A runner refuses to start while it is busy, so the race
    cannot be reached from the UI at all.

    **The freeze.** The modules that had no worker used
    ``QApplication.processEvents()`` instead, which repaints without yielding:
    the window looks alive while input queues up behind an operation measured at
    up to 23.8 seconds. Making a job easy to run off-thread -- button handling,
    progress, cancellation and re-enabling all included -- is what removes the
    temptation.

    The runner keeps a reference to the thread for its whole life, so nothing is
    collected while running, and it always re-enables its button, including on
    failure and cancellation.
    """

    started = QtCore.pyqtSignal()
    progressed = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str, str)
    cancelled = QtCore.pyqtSignal()
    busyChanged = QtCore.pyqtSignal(bool)

    def __init__(self, name: str = "job", parent=None,
                 button: QtWidgets.QAbstractButton | None = None,
                 cancel_button: QtWidgets.QAbstractButton | None = None,
                 progress: QtWidgets.QProgressBar | None = None):
        super().__init__(parent)
        self.name = name
        self._worker: Worker | None = None
        self._button = button
        self._cancel_button = cancel_button
        self._progress = progress
        if cancel_button is not None:
            cancel_button.setEnabled(False)
            cancel_button.clicked.connect(self.cancel)

    # ------------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def start(self, fn: Callable[..., Any], *args, **kwargs) -> bool:
        """Run ``fn`` off the main thread. False if a job is already running."""
        if self.busy:
            log.debug("%s: ignored a second start while busy", self.name)
            return False
        w = Worker(fn, *args, parent=self, **kwargs)
        self._worker = w
        w.progressed.connect(self._on_progress)
        w.finishedOk.connect(self._on_done)
        w.failed.connect(self._on_failed)
        self._set_busy(True)
        self.started.emit()
        w.start()
        return True

    def cancel(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()

    #: Workers that outlived their owner. See :meth:`wait`.
    _orphans: set = set()

    def wait(self, msec: int = 5000) -> None:
        """Stop and join. Called from shutdown paths, so it must not raise.

        A job that does not stop within the budget is **detached** rather than
        left to be destroyed. Closing a dialog deletes its children, and
        deleting a running ``QThread`` aborts the process -- which is the crash
        this class exists to prevent, so it must not be reintroduced by the
        cleanup path. A long scan cancelled at the last moment can easily still
        be inside a decode when the budget expires; reparenting it to nothing
        and holding a reference until it finishes lets it end quietly instead.
        """
        w = self._worker
        self._worker = None
        if w is None:
            return
        try:
            if not w.isRunning():
                return
            w.cancel()
            if w.wait(msec):
                return
            log.warning("%s did not stop within %d ms; detaching it so closing "
                        "the window cannot destroy a running thread",
                        self.name, msec)
            w.setParent(None)
            JobRunner._orphans.add(w)
            w.finished.connect(lambda w=w: JobRunner._orphans.discard(w))
        except RuntimeError:  # pragma: no cover - already destroyed by Qt
            pass

    @staticmethod
    def join_all(msec: int = 8000) -> None:
        """Wait for detached workers. Called once on application exit."""
        for w in list(JobRunner._orphans):
            try:
                if w.isRunning():
                    w.cancel()
                    w.wait(msec)
            except RuntimeError:  # pragma: no cover
                pass
            JobRunner._orphans.discard(w)

    # ------------------------------------------------------------------
    def _set_busy(self, busy: bool) -> None:
        if self._button is not None:
            self._button.setEnabled(not busy)
        if self._cancel_button is not None:
            self._cancel_button.setEnabled(busy)
        if self._progress is not None:
            self._progress.setVisible(True)
            if busy:
                # Indeterminate until the job reports a total it knows.
                self._progress.setRange(0, 0)
            else:
                self._progress.setRange(0, 100)
                self._progress.setValue(0)
        self.busyChanged.emit(busy)

    def _on_progress(self, done: int, total: int, msg: str) -> None:
        if self._progress is not None and total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(min(done, total))
        self.progressed.emit(done, total, msg)

    def _finish(self) -> None:
        self._set_busy(False)
        self._worker = None

    def _on_done(self, result) -> None:
        was_cancelled = self._worker is not None and self._worker.cancelled
        self._finish()
        if was_cancelled:
            self.cancelled.emit()
        else:
            self.finished.emit(result)

    def _on_failed(self, msg: str, tb: str) -> None:
        log.error("%s failed: %s\n%s", self.name, msg, tb)
        self._finish()
        self.failed.emit(msg, tb)


class FrameReader:
    """On-demand frame access with a small LRU cache.

    Scrubbing a 70-minute trial cannot load the file into memory -- that is the
    limitation this whole rewrite exists to remove -- so frames are seeked on
    request. MJPEG in AVI is all-intra, which makes frame-exact seeking cheap;
    for anything else, seeking lands on a keyframe, so the reader checks where it
    actually landed and steps forward frame by frame to the one asked for.
    """

    def __init__(self, path: Path | str, cache: int = 240):
        self.path = str(path)
        self.cap: cv2.VideoCapture | None = None
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._limit = max(8, cache)
        self.n_frames = 0
        self.fps = 0.0
        self.width = self.height = 0
        self.error = ""
        self._pos = -1
        self.open()

    def open(self) -> bool:
        self.close()
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            self.error = f"could not open {Path(self.path).name}"
            return False
        self.cap = cap
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.error = ""
        self._pos = 0
        return True

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._cache.clear()
        self._pos = -1

    def get(self, index: int) -> np.ndarray | None:
        if self.cap is None:
            return None
        index = max(0, int(index))
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]

        try:
            if self._pos != index:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                landed = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                # Walk forward if the backend put us on an earlier keyframe.
                steps = 0
                while landed < index and steps < 600:
                    if not self.cap.grab():
                        break
                    landed += 1
                    steps += 1
                self._pos = landed
            ok, frame = self.cap.read()
        except cv2.error as e:  # pragma: no cover
            self.error = str(e)
            return None
        if not ok:
            return None
        self._pos += 1
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._cache[index] = gray
        while len(self._cache) > self._limit:
            self._cache.popitem(last=False)
        return gray

    def prefetch(self, index: int, count: int = 24) -> None:
        """Read a short run forward, which is what scrubbing does next."""
        for i in range(index, index + count):
            if i not in self._cache:
                if self.get(i) is None:
                    break

    # -- thread-safety ------------------------------------------------
    # AsyncFrameReader drives one of these from a single decoder thread while
    # the GUI thread reads the cache. Only the cache is shared, and dict
    # operations are atomic under the GIL, so no lock is needed here -- but
    # nothing else may touch the VideoCapture concurrently.

    def peek(self, index: int) -> np.ndarray | None:
        """Return a cached frame, or None. Never decodes, never blocks."""
        return self._cache.get(int(index))

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------
# Form helpers
# --------------------------------------------------------------------------
def form_row(layout: QtWidgets.QFormLayout, label: str,
             widget: QtWidgets.QWidget | QtWidgets.QLayout,
             hint: str = "") -> QtWidgets.QWidget:
    """Add a labelled row, with the explanation under the control.

    Accepts a layout as well as a widget, so a composite row (two spin boxes and
    a unit, say) gets the same treatment.
    """
    if isinstance(widget, QtWidgets.QLayout):
        holder = QtWidgets.QWidget()
        holder.setLayout(widget)
        widget = holder
    if hint:
        widget.setToolTip(hint)
        box = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(1)
        v.addWidget(widget)
        h = QtWidgets.QLabel(hint)
        h.setObjectName("hint")
        h.setWordWrap(True)
        v.addWidget(h)
        layout.addRow(label, box)
        return box
    layout.addRow(label, widget)
    return widget


class _NoWheel(QtCore.QObject):
    """Swallow wheel events on controls that do not have focus.

    Qt's spin boxes and combo boxes accept the wheel whether or not they are
    focused, so scrolling a form changes whatever happens to be under the
    pointer. On this application that is not a cosmetic annoyance: it silently
    set the capture height to 320x245, a mode the camera does not have, and
    the recording failed with a raw DirectShow I/O error. The same slip set the
    segment length to 0.2 min. A control should change when it is used, not
    when it is scrolled past.
    """

    def eventFilter(self, obj, ev):  # noqa: N802 - Qt naming
        if ev.type() == QtCore.QEvent.Wheel and not obj.hasFocus():
            ev.ignore()
            return True
        return False


_NO_WHEEL = _NoWheel()


def no_wheel(root: QtWidgets.QWidget) -> int:
    """Protect every spin box and combo box under ``root``. Returns the count."""
    kinds = (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)
    n = 0
    for w in root.findChildren(QtWidgets.QWidget):
        if isinstance(w, kinds):
            w.installEventFilter(_NO_WHEEL)
            w.setFocusPolicy(QtCore.Qt.StrongFocus)
            n += 1
    return n


def stop_jobs(owner: object, msec: int = 5000) -> int:
    """Stop every JobRunner attribute of ``owner``. Returns how many it found.

    Shutdown used to name each runner by hand, which is a list that goes stale
    the moment someone adds one: the Settings tab was not even in the window's
    shutdown loop, so its jobs were still running when Qt tore the widget down
    -- the QThread abort, reintroduced by omission rather than by design.
    Sweeping the attributes cannot go stale.
    """
    found = 0
    for name in dir(owner):
        if not name.endswith("_job"):
            continue
        try:
            runner = getattr(owner, name)
        except Exception:  # pragma: no cover - properties that raise
            continue
        if isinstance(runner, JobRunner):
            runner.wait(msec)
            found += 1
    return found


class AsyncFrameReader(QtCore.QObject):
    """Decodes frames on a worker thread and coalesces seek requests.

    Scrubbing was the worst everyday cost in the application: each decode was
    measured at 127-197 ms on the GUI thread, and a slider drag emits one
    request per pixel of travel. Those requests queued, so a drag across a
    70-minute trial committed the interface to minutes of work with no way to
    interrupt it -- and because they were served in order, the frames arrived
    long after the mouse had moved on.

    Two changes fix it. Decoding moves to its own thread, and the queue has
    room for exactly one pending request: a newer one replaces the one waiting,
    because a frame the user has already scrolled past is worth nothing. The
    cost of a drag is then bounded by how fast frames decode, not by how many
    pixels the mouse travelled, and the view always converges on where the
    slider actually stopped.

    ``frameReady(index, frame)`` arrives on the GUI thread. A request whose
    frame is already cached is answered synchronously and returns True, so the
    common case -- stepping one frame -- stays immediate.
    """

    frameReady = QtCore.pyqtSignal(int, object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._reader: FrameReader | None = None
        self._lock = threading.Lock()
        self._wanted: int | None = None
        self._new_request = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._prefetch = 0

    # ------------------------------------------------------------------
    def open(self, path: Path | str, cache: int = 240, prefetch: int = 8) -> bool:
        self.close()
        reader = FrameReader(path, cache=cache)
        if reader.cap is None:
            self.failed.emit(reader.error or f"could not open {Path(path).name}")
            return False
        self._reader = reader
        self._prefetch = max(0, int(prefetch))
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="frame-decoder")
        self._thread.start()
        return True

    def close(self) -> None:
        self._stop.set()
        self._new_request.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=3.0)
        self._thread = None
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    @property
    def n_frames(self) -> int:
        return self._reader.n_frames if self._reader else 0

    @property
    def fps(self) -> float:
        return self._reader.fps if self._reader else 0.0

    @property
    def width(self) -> int:
        return self._reader.width if self._reader else 0

    @property
    def height(self) -> int:
        return self._reader.height if self._reader else 0

    # ------------------------------------------------------------------
    def request(self, index: int) -> bool:
        """Ask for a frame. True if it was already cached and emitted now."""
        if self._reader is None:
            return False
        index = max(0, int(index))
        cached = self._reader.peek(index)
        if cached is not None:
            self.frameReady.emit(index, cached)
            return True
        with self._lock:
            # One slot: a newer request supersedes whatever was waiting.
            self._wanted = index
        self._new_request.set()
        return False

    def _run(self) -> None:  # pragma: no cover - thread body
        while not self._stop.is_set():
            self._new_request.wait(0.2)
            self._new_request.clear()
            while not self._stop.is_set():
                with self._lock:
                    idx = self._wanted
                    self._wanted = None
                if idx is None:
                    break
                reader = self._reader
                if reader is None:
                    break
                try:
                    frame = reader.get(idx)
                except Exception as e:
                    log.exception("frame decode failed")
                    self.failed.emit(str(e))
                    break
                if frame is None:
                    continue
                with self._lock:
                    superseded = self._wanted is not None
                if not superseded:
                    # Only paint the frame the user is still waiting for.
                    self.frameReady.emit(idx, frame)
                    for j in range(idx + 1, idx + 1 + self._prefetch):
                        with self._lock:
                            if self._wanted is not None or self._stop.is_set():
                                break
                        if reader.peek(j) is None and reader.get(j) is None:
                            break


def form_wrap(layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
    """A layout as a widget, so it can be dropped into a QFormLayout row."""
    holder = QtWidgets.QWidget()
    holder.setLayout(layout)
    return holder


def h_line() -> QtWidgets.QFrame:
    f = QtWidgets.QFrame()
    f.setFrameShape(QtWidgets.QFrame.HLine)
    f.setFrameShadow(QtWidgets.QFrame.Plain)
    return f


def button(text: str, kind: str = "", tooltip: str = "") -> QtWidgets.QPushButton:
    b = QtWidgets.QPushButton(text)
    if kind:
        b.setObjectName(kind)
    if tooltip:
        b.setToolTip(tooltip)
    return b


def scrollable(inner: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QtWidgets.QFrame.NoFrame)
    area.setWidget(inner)
    return area


def info_box(parent, title: str, text: str) -> None:
    QtWidgets.QMessageBox.information(parent, title, text)


def warn_box(parent, title: str, text: str) -> None:
    QtWidgets.QMessageBox.warning(parent, title, text)


def ask_yes_no(parent, title: str, text: str) -> bool:
    return QtWidgets.QMessageBox.question(
        parent, title, text,
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        QtWidgets.QMessageBox.No) == QtWidgets.QMessageBox.Yes


def human_bytes(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def hms(seconds: float) -> str:
    s = int(max(0, round(seconds)))
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
