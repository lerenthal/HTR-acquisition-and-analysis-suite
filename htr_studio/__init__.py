"""HTR Studio — Head-Twitch Response acquisition and analysis suite.

A GUI front end for the Arducam ezBOX GS B0634 (OV9281 global shutter)
HTR pipeline: camera setup, trial recording, video library management,
footage review with human curation, and automated burst detection.

Package layout
--------------
paths          portable path resolution (install dir, data dir, tools)
config         JSON settings with defaults + migration
logging_setup  rotating file log + console
ffmpeg_tools   locate/driving ffmpeg + ffprobe, dshow enumeration, mode probing
camera         OpenCV preview, UVC property control, image-quality metrics
recorder       segmented ffmpeg recording with live stats and trial sidecars
library        recording discovery, sidecar metadata, session grouping
analysis       streaming signal extraction + burst detectors + validation
report         CSV/JSON/plot export
ui.*           PyQt5 widgets
"""

__version__ = "1.3.0-dev"
__all__ = ["__version__"]
