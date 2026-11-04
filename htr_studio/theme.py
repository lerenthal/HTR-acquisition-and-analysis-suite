"""Colour and typography tokens, shared by the Qt UI and the exported plots.

One place for the palette so a printed report and the screen agree. The
categorical slots and the status colours are a validated set (lightness band,
chroma floor, colour-vision-deficiency separation and normal-vision separation
all checked against both the light and the dark chart surface), so they are used
as given and not re-mixed per chart.

Usage rules that go with them, and that the plots here follow:

* Only the first three categorical slots are used on any one chart. Those three
  clear the separation floors for every pair, which is what a scatter or an
  overlay needs; beyond three, pairs start failing and a reader with deuteranopia
  cannot tell two series apart.
* Status colours mean state (good / warning / serious / critical) and never
  stand in for a fourth series, and they always travel with a word, never
  colour alone.
* Aqua is below 3:1 against the light surface, so anything drawn in it also
  carries a visible label.
* Never two y-axes on one panel. Two quantities with different units get two
  stacked panels sharing the x-axis.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    name: str
    surface: str
    plane: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    axis: str
    series: tuple[str, ...]
    good: str
    warning: str
    serious: str
    critical: str

    @property
    def s1(self) -> str:
        return self.series[0]

    @property
    def s2(self) -> str:
        return self.series[1]

    @property
    def s3(self) -> str:
        return self.series[2]


LIGHT = Palette(
    name="light",
    surface="#fcfcfb", plane="#f9f9f7",
    ink="#0b0b0b", ink_secondary="#52514e", ink_muted="#898781",
    grid="#e1e0d9", axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
            "#008300", "#6250d6", "#e34948"),
    good="#0ca30c", warning="#fab219", serious="#ec835a", critical="#d03b3b",
)

DARK = Palette(
    name="dark",
    surface="#1a1a19", plane="#0d0d0d",
    ink="#f0efec", ink_secondary="#c3c2b7", ink_muted="#898781",
    grid="#2c2c2a", axis="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
            "#008300", "#9085e9", "#e66767"),
    good="#0ca30c", warning="#fab219", serious="#ec835a", critical="#d03b3b",
)

FONT_STACK = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def palette(name: str = "light") -> Palette:
    return DARK if str(name).lower() == "dark" else LIGHT


# --------------------------------------------------------------------------
# Matplotlib
# --------------------------------------------------------------------------
def apply_matplotlib(pal: Palette) -> None:
    """Set rcParams so every exported figure is consistent without per-plot code."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": pal.surface,
        "savefig.facecolor": pal.surface,
        "axes.facecolor": pal.surface,
        "axes.edgecolor": pal.axis,
        "axes.labelcolor": pal.ink_secondary,
        "axes.titlecolor": pal.ink,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": pal.grid,
        "grid.linewidth": 0.6,
        "xtick.color": pal.ink_muted,
        "ytick.color": pal.ink_muted,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "text.color": pal.ink,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


# --------------------------------------------------------------------------
# pyqtgraph
# --------------------------------------------------------------------------
def apply_pyqtgraph(pal: Palette, antialias: bool = True) -> None:
    import pyqtgraph as pg

    pg.setConfigOption("background", pal.surface)
    pg.setConfigOption("foreground", pal.ink_secondary)
    pg.setConfigOptions(antialias=antialias)


QSS_LIGHT = """
QWidget { background: #f9f9f7; color: #0b0b0b; font-size: 12px; }
QGroupBox { border: 1px solid #e1e0d9; border-radius: 6px; margin-top: 14px;
            padding: 10px 8px 8px 8px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px;
                   color: #52514e; }
QPushButton { background: #ffffff; border: 1px solid #c3c2b7; border-radius: 5px;
              padding: 5px 12px; }
QPushButton:hover:!disabled { border-color: #2a78d6; }
QPushButton:disabled { color: #898781; background: #f1f1ee; }
QPushButton#primary { background: #2a78d6; color: #ffffff; border-color: #2a78d6;
                      font-weight: 600; }
QPushButton#primary:hover:!disabled { background: #256abf; }
QPushButton#danger { background: #d03b3b; color: #ffffff; border-color: #d03b3b;
                     font-weight: 600; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {
    background: #ffffff; border: 1px solid #c3c2b7; border-radius: 5px; padding: 3px 6px; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #2a78d6; }
QTableWidget, QTableView, QTreeWidget { background: #ffffff; border: 1px solid #e1e0d9;
    gridline-color: #e1e0d9; alternate-background-color: #f9f9f7; }
QHeaderView::section { background: #f1f1ee; border: none;
    border-bottom: 1px solid #c3c2b7; padding: 5px; font-weight: 600; color: #52514e; }
QTabWidget::pane { border: 1px solid #e1e0d9; border-radius: 6px; top: -1px; }
QTabBar::tab { background: transparent; padding: 7px 16px; color: #52514e; }
QTabBar::tab:selected { background: #fcfcfb; border: 1px solid #e1e0d9;
    border-bottom-color: #fcfcfb; border-top-left-radius: 6px;
    border-top-right-radius: 6px; color: #0b0b0b; font-weight: 600; }
QProgressBar { border: 1px solid #c3c2b7; border-radius: 5px; text-align: center;
    background: #ffffff; height: 16px; }
QProgressBar::chunk { background: #2a78d6; border-radius: 4px; }
QSlider::groove:horizontal { height: 4px; background: #e1e0d9; border-radius: 2px; }
QSlider::handle:horizontal { background: #2a78d6; width: 14px; margin: -6px 0;
    border-radius: 7px; }
QLabel#hint { color: #898781; }
QLabel#statusOk { color: #006300; font-weight: 600; }
QLabel#statusWarn { color: #52514e; font-weight: 600; }
QLabel#statusBad { color: #d03b3b; font-weight: 600; }
QLabel#metric { font-size: 20px; font-weight: 600; }
QSplitter::handle { background: #e1e0d9; }
"""

QSS_DARK = """
QWidget { background: #0d0d0d; color: #f0efec; font-size: 12px; }
QGroupBox { border: 1px solid #2c2c2a; border-radius: 6px; margin-top: 14px;
            padding: 10px 8px 8px 8px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px;
                   color: #c3c2b7; }
QPushButton { background: #1a1a19; border: 1px solid #383835; border-radius: 5px;
              padding: 5px 12px; color: #f0efec; }
QPushButton:hover:!disabled { border-color: #3987e5; }
QPushButton:disabled { color: #898781; background: #141413; }
QPushButton#primary { background: #3987e5; color: #ffffff; border-color: #3987e5;
                      font-weight: 600; }
QPushButton#danger { background: #d03b3b; color: #ffffff; border-color: #d03b3b;
                     font-weight: 600; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {
    background: #1a1a19; border: 1px solid #383835; border-radius: 5px;
    padding: 3px 6px; color: #f0efec; }
QTableWidget, QTableView, QTreeWidget { background: #1a1a19; border: 1px solid #2c2c2a;
    gridline-color: #2c2c2a; alternate-background-color: #141413; }
QHeaderView::section { background: #141413; border: none;
    border-bottom: 1px solid #383835; padding: 5px; font-weight: 600; color: #c3c2b7; }
QTabWidget::pane { border: 1px solid #2c2c2a; border-radius: 6px; top: -1px; }
QTabBar::tab { background: transparent; padding: 7px 16px; color: #c3c2b7; }
QTabBar::tab:selected { background: #1a1a19; border: 1px solid #2c2c2a;
    border-bottom-color: #1a1a19; border-top-left-radius: 6px;
    border-top-right-radius: 6px; color: #f0efec; font-weight: 600; }
QProgressBar { border: 1px solid #383835; border-radius: 5px; text-align: center;
    background: #1a1a19; height: 16px; }
QProgressBar::chunk { background: #3987e5; border-radius: 4px; }
QSlider::groove:horizontal { height: 4px; background: #2c2c2a; border-radius: 2px; }
QSlider::handle:horizontal { background: #3987e5; width: 14px; margin: -6px 0;
    border-radius: 7px; }
QLabel#hint { color: #898781; }
QLabel#statusOk { color: #0ca30c; font-weight: 600; }
QLabel#statusWarn { color: #c3c2b7; font-weight: 600; }
QLabel#statusBad { color: #e66767; font-weight: 600; }
QLabel#metric { font-size: 20px; font-weight: 600; }
QSplitter::handle { background: #2c2c2a; }
"""


def stylesheet(theme: str = "light") -> str:
    return QSS_DARK if str(theme).lower() == "dark" else QSS_LIGHT


#: Qt object names used for the three verdict levels, so a status colour always
#: arrives with text rather than on its own.
STATUS_OBJECT_NAMES = {"ok": "statusOk", "warn": "statusWarn", "bad": "statusBad"}
