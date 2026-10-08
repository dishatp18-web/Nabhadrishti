"""
BAS Experiment Copilot - Camera Abstraction
============================================
Robust video capture that supports, in priority order:

  1. An explicit source given on the command line / env var:
       - integer index          -> local webcam            e.g. "0"
       - http(s) URL            -> DroidCam / IP webcam    e.g. "http://192.168.1.7:4747/video"
       - rtsp URL               -> network camera
       - existing file path     -> recorded video loop
  2. Auto-probing the first few local webcam indices (works for USB webcams
     and DroidCam's Windows DirectShow driver).
  3. A synthetic test-pattern generator so the HUD always has a signal,
     clearly flagged with CameraStatus.SIMULATED.

Design rules enforced here (perception.py and server.py never touch cv2
capture themselves):
  * Opening a source is retried with backoff - DroidCam often refuses the
    first connection while its stream warms up.
  * Any read that fails, times out, or returns an EMPTY / BLACK / FROZEN
    frame marks the source degraded; after N consecutive bad frames the
    source is torn down and re-opened (or the next fallback is tried).
"""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .models import CameraStatus

log = logging.getLogger("bas.camera")

# A frame is considered "black" when its mean and stddev are both tiny.
BLACK_MEAN_THRESH = 4.0
BLACK_STD_THRESH = 3.0
# Consecutive bad reads before we declare the source dead and re-open it.
MAX_BAD_FRAMES = 30
RECONNECT_DELAY_S = 1.5


def _parse_source(raw: str):
    """Return ('index', int) | ('url', str) | ('file', str) for a source spec."""
    raw = raw.strip()
    if raw.isdigit():
        return "index", int(raw)
    if raw.lower().startswith(("http://", "https://", "rtsp://")):
        return "url", raw
    return "file", raw


class _AutoProbe:
    """Find a local camera index by actually trying to read a frame from it."""

    MAX_INDEX = 4

    @staticmethod
    def find() -> Optional[Tuple[str, object]]:
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        for idx in range(_AutoProbe.MAX_INDEX):
            cap = cv2.VideoCapture(idx, backend)
            ok, frame = False, None
            try:
                # Give DroidCam-style virtual drivers a moment to surface.
                time.sleep(0.2)
                ok, frame = cap.read()
            finally:
                cap.release()
            if ok and frame is not None and frame.mean() > BLACK_MEAN_THRESH:
                log.info("Auto-probe found local camera index %d", idx)
                return "index", idx
        return None


_NAME_CACHE: dict = {"ts": 0.0, "names": []}


def _is_video_device_name(name: str) -> bool:
    n = name.lower()
    return (("cam" in n or "video" in n or "uvc" in n or "droidcam" in n)
            and "audio" not in n and "microphone" not in n)


def list_camera_names(max_age_s: float = 30.0) -> List[str]:
    """Friendly names of active camera devices, in enumeration order.

    On Windows this maps 1:1-ish to DirectShow/MSMF indices used by cv2,
    which lets us pick a specific device (e.g. the DroidCam virtual webcam)
    by name instead of guessing indices. Cached briefly - it shells out to
    PowerShell.
    """
    now = time.time()
    if _NAME_CACHE["names"] and now - _NAME_CACHE["ts"] < max_age_s:
        return _NAME_CACHE["names"]
    import subprocess
    try:
        # DroidCam's virtual webcam registers under MEDIA (not Camera/Image),
        # so query both classes and merge, keeping PnP order.
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-PnpDevice -Class Camera,Image,MEDIA -ErrorAction SilentlyContinue | "
             "Where-Object { $_.Status -eq 'OK' -and "
             "($_.Class -ne 'MEDIA' -or $_.FriendlyName -match 'cam|video|droidcam') } | "
             "Select-Object -ExpandProperty FriendlyName"],
            capture_output=True, text=True, timeout=20)
        seen, names = set(), []
        for l in out.stdout.splitlines():
            l = l.strip()
            if l and l not in seen and _is_video_device_name(l):
                seen.add(l)
                names.append(l)
        _NAME_CACHE["names"] = names
        _NAME_CACHE["ts"] = now
        return names
    except Exception as exc:
        log.warning("Cannot enumerate camera names: %s", exc)
        return _NAME_CACHE["names"]


def _camera_name_for_index(index: int) -> str:
    names = list_camera_names()
    if 0 <= index < len(names):
        return names[index]
    return f"Camera {index}"


def find_virtual_camera_index(keyword: str = "droidcam") -> Optional[int]:
    """Find a local camera index whose device name matches `keyword`.

    Probes both Windows backends - DroidCam's virtual webcam is
    Media-Foundation-only on recent drivers.
    """
    names = list_camera_names()
    backends = ([cv2.CAP_DSHOW, cv2.CAP_MSMF]
                if platform.system() == "Windows" else [cv2.CAP_ANY])
    for i, name in enumerate(names):
        if keyword.lower() not in name.lower():
            continue
        for backend in backends:
            cap = cv2.VideoCapture(i, backend)
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None and frame.mean() > BLACK_MEAN_THRESH:
                log.info("Virtual camera '%s' found at index %d (%s)",
                         name, i, "MSMF" if backend == cv2.CAP_MSMF else "DSHOW")
                return i
        log.warning("'%s' maps to index %d but delivers no frames", name, i)
    return None


_ENUM_CACHE: dict = {"ts": 0.0, "cams": None}


def _probe_index_works(index: int) -> bool:
    backends = ([cv2.CAP_DSHOW, cv2.CAP_MSMF]
                if platform.system() == "Windows" else [cv2.CAP_ANY])
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None and frame.mean() > BLACK_MEAN_THRESH:
            return True
    return False


def enumerate_camera_sources(active_id: Optional[str] = None,
                             max_age_s: float = 30.0,
                             probe: bool = True) -> List[dict]:
    """List selectable local cameras for the HUD dropdown.

    Each entry: {id, name, kind, active, verified}. The currently active
    camera is trusted without probing (its device is already open); others
    are verified to actually deliver frames so the dropdown never offers a
    dead choice.
    """
    now = time.time()
    if _ENUM_CACHE["cams"] and now - _ENUM_CACHE["ts"] < max_age_s:
        cams = [dict(c) for c in _ENUM_CACHE["cams"]]
    else:
        cams = [{"id": str(i), "name": n, "kind": "index"}
                for i, n in enumerate(list_camera_names())]
        _ENUM_CACHE["cams"] = [dict(c) for c in cams]

    for c in cams:
        c["active"] = (active_id is not None and c["id"] == str(active_id))
        if c["active"] or not probe:
            c["verified"] = None
            continue
        c["verified"] = _probe_index_works(int(c["id"]))
    return [c for c in cams if c.get("verified") is not False]


def find_droidcam(timeout_per_host: float = 0.4,
                  port: int = 4747) -> Optional[str]:
    """Scan the local subnet(s) for a DroidCam app and return its MJPEG URL.

    DroidCam serves `http://<phone-ip>:4747/video`. Scanning 254 hosts with
    short socket timeouts across a thread pool takes ~2-4 s.
    Returns None if no phone answers (check: phone app open + same Wi-Fi).
    """
    import socket
    from concurrent.futures import ThreadPoolExecutor

    candidates: List[str] = []
    try:
        hostname, _, _ = socket.gethostbyname_ex(socket.gethostname())
        addrs = [a for a in ([hostname] + socket.gethostbyname_ex(hostname)[2])
                 if a]
        # Also common virtual adapter ranges are skipped by construction:
        # we only scan the /24 of each real local address.
        for a in addrs:
            if a.startswith("127.") or a.endswith(".0") or a.endswith(".255"):
                continue
            base = a.rsplit(".", 1)[0]
            candidates.extend(f"{base}.{i}" for i in range(1, 255))
    except OSError as exc:
        log.warning("DroidCam discovery: cannot enumerate local addresses (%s)", exc)
        return None

    def probe(ip: str) -> Optional[str]:
        try:
            with socket.create_connection((ip, port), timeout=timeout_per_host):
                return f"http://{ip}:{port}/video"
        except OSError:
            return None

    log.info("Scanning %d hosts for DroidCam (port %d)...",
             len(candidates), port)
    with ThreadPoolExecutor(max_workers=128) as pool:
        for url in pool.map(probe, candidates):
            if url:
                log.info("DroidCam found at %s", url)
                return url
    return None


class _SyntheticSource:
    """Last-resort signal: an animated test pattern so the HUD is never dead."""

    def __init__(self, size: Tuple[int, int] = (1280, 720)):
        self.size = size

    def read(self):
        import math
        w, h = self.size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        t = time.time()
        # Deep-space gradient background.
        img[:, :, 0] = 40
        img[:, :, 1] = 18
        img[:, :, 2] = 10
        # Drifting grid lines (parallax feel).
        off = int((t * 30) % 64)
        for x in range(-64, w + 64, 64):
            cv2.line(img, (x + off, 0), (x + off, h), (70, 50, 30), 1)
        for y in range(0, h, 64):
            cv2.line(img, (0, y), (w, y), (70, 50, 30), 1)
        # A slowly moving "payload" puck the HUD can lock onto.
        cx = int(w / 2 + math.sin(t / 3.1) * w * 0.28)
        cy = int(h / 2 + math.cos(t / 4.3) * h * 0.22)
        cv2.circle(img, (cx, cy), 34, (90, 180, 250), -1)
        cv2.circle(img, (cx, cy), 34, (255, 255, 255), 2)
        cv2.putText(img, "SIM PAYLOAD", (cx - 55, cy - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "SYNTHETIC FEED - NO CAMERA", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        return True, img

    def release(self):
        pass

    def isOpened(self):
        return True


class CameraManager:
    """Owns the active cv2.VideoCapture and heals it automatically.

    Usage:
        cam = CameraManager(explicit_source="http://192.168.1.7:4747/video")
        ok, frame, status = cam.read()
    """

    def __init__(self, explicit_source: Optional[str] = None):
        env_src = os.environ.get("BAS_CAMERA_SOURCE")
        self._sources: List[str] = []
        if explicit_source:
            self._sources.append(explicit_source)
        if env_src and env_src not in self._sources:
            self._sources.append(env_src)
        # Auto-probe / synthetic are appended lazily as fallbacks.
        self._probe_done = False
        self._synthetic_enabled = True

        self._cap = None
        self._current_spec: Optional[Tuple[str, object]] = None
        self._lock = threading.Lock()
        self.status = CameraStatus.OFFLINE
        self.source_label = ""
        self.source_name = ""
        self._bad_frames = 0
        self._last_good_frame_ts = 0.0
        self._width = 0
        self._height = 0
        # Failed full acquisition sweeps (probe + fallbacks). Only this
        # counter gates giving up - slow probing must not exhaust it.
        self._sweep_failures = 0
        # Consecutive pitch-black (but healthy) frames - a covered lens or
        # a camera facing total darkness. NOT a dead source: report it,
        # don't reopen.
        self._dark_streak = 0

    # ---------------------------------------------------------------- open

    def _open_source(self, spec: Tuple[str, object]) -> bool:
        kind, value = spec
        try:
            if kind == "index":
                # DroidCam's virtual webcam is Media-Foundation-only on
                # recent drivers, while USB webcams open via DirectShow -
                # so try both backends.
                backends = ([cv2.CAP_DSHOW, cv2.CAP_MSMF]
                            if platform.system() == "Windows" else [cv2.CAP_ANY])
                cap = None
                for backend in backends:
                    candidate = cv2.VideoCapture(value, backend)
                    ok, frame = candidate.read()
                    if ok and frame is not None and frame.size:
                        candidate.set(cv2.CAP_PROP_POS_MSEC, 0)
                        cap = candidate
                        break
                    candidate.release()
                if cap is None:
                    return False
                # NOTE: do NOT force a resolution on virtual cameras -
                # renegotiating MSMF stride mid-open corrupts reads
                # (cv2 Mat _step assertion). Use the device's native mode.
                self._cap = cap
                self._current_spec = spec
                self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                self._bad_frames = 0
                self._dark_streak = 0
                self._last_good_frame_ts = time.time()
                self.source_label = str(value)
                self.source_name = _camera_name_for_index(int(value))
                self.status = CameraStatus.LIVE
                self._sweep_failures = 0
                log.info("Camera opened: index %s '%s' (backend %s)", value,
                         self.source_name,
                         "dshow" if cap.getBackendName() == "DSHOW" else "msmf")
                return True
            if kind == "url":
                cap = cv2.VideoCapture(value)
            else:  # file
                if not os.path.isfile(value):
                    log.warning("Video file not found: %s", value)
                    return False
                cap = cv2.VideoCapture(value)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            # Warm-up read: DroidCam's first frames often come back empty.
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                cap.release()
                return False
            self._cap = cap
            self._current_spec = spec
            self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or frame.shape[1]
            self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or frame.shape[0]
            self._bad_frames = 0
            self._dark_streak = 0
            self._last_good_frame_ts = time.time()
            self.source_label = str(value)
            self.source_name = ("DroidCam (Wi-Fi)" if kind == "url"
                                and "4747" in str(value)
                                else os.path.basename(str(value)) if kind == "file"
                                else "Network camera")
            self.status = CameraStatus.LIVE
            self._sweep_failures = 0
            log.info("Camera opened: %s (%s)", value, kind)
            return True
        except Exception as exc:  # cv2 can raise on bad URLs on some builds
            log.warning("Failed to open %s: %s", value, exc)
            return False

    def _acquire_any(self) -> bool:
        """Try every fallback strategy until something opens."""
        with self._lock:
            # 1. CLI / env sources (retry each a few times - DroidCam warms up).
            while self._sources:
                spec = _parse_source(self._sources[0])
                for attempt in range(3):
                    if self._open_source(spec):
                        return True
                    time.sleep(RECONNECT_DELAY_S)
                log.warning("Source unusable, dropping from list: %s", self._sources[0])
                self._sources.pop(0)

            # 2. Auto-probe local webcams / virtual DroidCam drivers.
            if not self._probe_done:
                self._probe_done = True
                found = _AutoProbe.find()
                if found and self._open_source(found):
                    return True

            # 3. Synthetic fallback.
            if self._synthetic_enabled:
                self._cap = _SyntheticSource()
                self._current_spec = ("synthetic", None)
                self._width, self._height = self._cap.size
                self._bad_frames = 0
                self._dark_streak = 0
                self._last_good_frame_ts = time.time()
                self.source_label = "synthetic"
                self.source_name = "Synthetic Test Pattern"
                self.status = CameraStatus.SIMULATED
                log.warning("No camera found - using synthetic test pattern")
                return True
        return False

    def open(self) -> bool:
        if self._acquire_any():
            self._sweep_failures = 0
            return True
        self._sweep_failures += 1
        return False

    # ---------------------------------------------------- runtime switching

    @property
    def current_source_id(self) -> Optional[str]:
        """Stable id of the active source (index / url / file), if any."""
        if self._current_spec is None:
            return None
        kind, value = self._current_spec
        return None if kind == "synthetic" else str(value)

    def switch_source(self, spec: str) -> Tuple[bool, str]:
        """Hot-swap to a new source at runtime. Never leaves the pipeline
        without a camera: on failure the previous source is restored.

        Returns (ok, friendly_source_name).
        """
        spec = (spec or "").strip()
        if not spec:
            return False, self.source_name

        with self._lock:
            old_spec = self._current_spec
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

            if spec in ("sim", "synthetic"):
                self._cap = _SyntheticSource()
                self._current_spec = ("synthetic", None)
                self._width, self._height = self._cap.size
                self._bad_frames = 0
                self._dark_streak = 0
                self.source_label = "synthetic"
                self.source_name = "Synthetic Test Pattern"
                self.status = CameraStatus.SIMULATED
                self._sources = []
                self._sweep_failures = 0
                log.info("Camera switched to synthetic test pattern")
                return True, self.source_name

            if self._open_source(_parse_source(spec)):
                self._sources = [spec]
                self._probe_done = True  # stay on the user's choice
                log.info("Camera switched to %s ('%s')", spec, self.source_name)
                return True, self.source_name

            # Failure: restore the previous source so the pipeline keeps
            # running with the old camera.
            log.warning("Switch to '%s' failed - restoring previous source", spec)
            if old_spec is not None:
                if self._open_source(old_spec):
                    return False, self.source_name
            self.status = CameraStatus.RECONNECTING
            return False, self.source_name

    # ---------------------------------------------------------------- read

    @staticmethod
    def _is_black(frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(gray.mean()) < BLACK_MEAN_THRESH and float(gray.std()) < BLACK_STD_THRESH

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read one frame, healing the pipeline when the source misbehaves."""
        if self._cap is None:
            if self._sweep_failures >= 8:
                return False, None
            if not self._acquire_any():
                self._sweep_failures += 1
                self.status = CameraStatus.RECONNECTING
                return False, None
            self._sweep_failures = 0

        with self._lock:
            cap = self._cap
            ok, frame = (False, None)
            try:
                ok, frame = cap.read()
            except Exception as exc:
                log.warning("Camera read raised: %s", exc)
                ok = False

            degraded = (
                not ok
                or frame is None
                or getattr(frame, "size", 0) == 0
            )
            # A pitch-black frame is a HEALTHY camera pointed at darkness
            # (covered lens, phone face-down). Stream it and flag it - do
            # not treat it as a dead source, or DroidCam in a dark room
            # triggers an endless reopen loop.
            dark = bool(
                ok and frame is not None
                and self._current_spec and self._current_spec[0] != "synthetic"
                and self._is_black(frame)
            )
            self._dark_streak = self._dark_streak + 1 if dark else 0

            if degraded:
                self._bad_frames += 1
                if self._bad_frames == 1:
                    self.status = CameraStatus.RECONNECTING
                    log.warning("Degraded camera frame (bad=%d)", self._bad_frames)
                if self._bad_frames >= MAX_BAD_FRAMES:
                    log.warning("Camera source dead after %d bad frames - reopening",
                                self._bad_frames)
                    try:
                        cap.release()
                    except Exception:
                        pass
                    self._cap = None
                    self.status = CameraStatus.RECONNECTING
                    # Re-open the SAME source first (retry loop inside
                    # _acquire_any handles DroidCam warm-up failures).
                    if self._current_spec and self._current_spec[0] != "synthetic":
                        self._sources.insert(0, str(self._current_spec[1]))
                    self._sweep_failures = 0
                return False, None

            self._bad_frames = 0
            self._last_good_frame_ts = time.time()
            self.status = (CameraStatus.SIMULATED
                           if (self._current_spec and self._current_spec[0] == "synthetic")
                           else CameraStatus.LIVE)
            return True, frame

    # ---------------------------------------------------------------- misc

    @property
    def is_dark(self) -> bool:
        """Camera delivers frames but sees (almost) no light."""
        return self._dark_streak > 12

    @property
    def frame_size(self) -> Tuple[int, int]:
        return self._width, self._height

    def release(self):
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
            self.status = CameraStatus.OFFLINE
