"""
BAS Experiment Copilot - Local Video Recorder
==============================================
SIH 26174 requirement: "Stream the video of the experiment to specific IP
and also store the video locally."

The annotated stream goes out over /video_feed (HTTP MJPEG, reachable from
any LAN IP the server binds); this module simultaneously writes the same
annotated frames to a local timestamped video file for post-flight review.

Writes MJPEG-in-AVI (robust, no codec packs needed) into the session
directory, downsampled to ~12 fps to keep files small.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("bas.recorder")

RECORD_FPS = 12.0


class VideoRecorder:
    def __init__(self, session_dir: Path):
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path: Optional[Path] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._lock = threading.Lock()
        self._last_write = 0.0
        self._frame_count = 0

    def write(self, frame_bgr: np.ndarray):
        """Write one annotated frame (throttled to RECORD_FPS)."""
        now = time.time()
        if now - self._last_write < 1.0 / RECORD_FPS:
            return
        self._last_write = now
        with self._lock:
            if self._writer is None:
                h, w = frame_bgr.shape[:2]
                self.path = self.dir / "experiment_video.avi"
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self._writer = cv2.VideoWriter(str(self.path), fourcc,
                                               RECORD_FPS, (w, h))
                if not self._writer.isOpened():
                    log.warning("VideoWriter failed to open %s - recording "
                                "disabled", self.path)
                    self._writer = None
                    self.path = None
                    return
                log.info("Local recording started: %s", self.path)
            try:
                self._writer.write(frame_bgr)
                self._frame_count += 1
            except Exception as exc:
                log.warning("Recorder write failed: %s", exc)

    @property
    def filename(self) -> Optional[str]:
        return self.path.name if self.path else None

    @property
    def frames(self) -> int:
        return self._frame_count

    def release(self):
        with self._lock:
            if self._writer is not None:
                try:
                    self._writer.release()
                    log.info("Local recording saved: %s (%d frames)",
                             self.path, self._frame_count)
                except Exception:
                    pass
                self._writer = None
