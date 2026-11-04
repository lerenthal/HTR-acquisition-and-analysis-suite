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
        v = raw / self._scale
        self.spin.blockSignals(True)
        self.spin.setValue(v)
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
