"""
BAS Experiment Copilot - Backend Server (The "Nerve Center")
=============================================================
FastAPI application wiring together:

  * camera.CameraManager        - robust frame acquisition
  * perception.PerceptionEngine - hands + objects + interactions
  * fsm.ProtocolFSM             - deterministic protocol state machine
  * action_model.ActionWatcher  - GRU activity recognition (advisory only)
  * voice.VoiceEngine           - offline voice alerts
  * audit.AuditLogger           - JSONL evidence + text report
  * recorder.VideoRecorder      - local video storage

Endpoints
---------
  GET  /                    - the HUD
  GET  /video_feed          - multipart MJPEG stream (Content-Length per part)
  WS   /ws/telemetry        - JSON telemetry (Pydantic `Telemetry`) @10 Hz
  GET  /api/health          - liveness probe
  GET  /api/audit/report    - human-readable experiment report (.txt)
  GET  /api/audit/log       - structured JSONL evidence trail
  GET  /api/recordings      - local video recordings index

Concurrency model: one pipeline thread does capture -> perception -> FSM ->
recorder -> JPEG encode, publishing into shared slots; async endpoints only
read those slots, so network clients can never stall the vision loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, JSONResponse, PlainTextResponse,
                               Response, StreamingResponse)
from pydantic import BaseModel

from .action_model import ActionWatcher, extract_features
from .audit import AuditLogger
from .camera import CameraManager, enumerate_camera_sources
from .experiments import ensure_seed_experiments, get_all_experiments, get_experiment_detail
from .fsm import ProtocolFSM
from .models import Alert, AlertLevel, SystemHealth, Telemetry
from .perception import PerceptionEngine
from .protocols import get_protocol
from .recorder import VideoRecorder
from .voice import VoiceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
log = logging.getLogger("bas.server")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "recordings"

TARGET_STREAM_FPS = 24
TELEMETRY_HZ = 10

session_id = datetime.now(timezone.utc).strftime("session_%Y%m%d_%H%M%S")
session_dir = RECORDINGS_DIR / session_id


# --------------------------------------------------------------------------
# The pipeline: one thread, published slots
# --------------------------------------------------------------------------

class Pipeline:
    def __init__(self):
        protocol = get_protocol(os.environ.get("BAS_PROTOCOL", "flask"))
        self.camera = CameraManager()
        self.perception = PerceptionEngine()
        self.fsm = ProtocolFSM(protocol)
        self.perception.set_vocabulary(self.fsm.vocabulary())
        self.voice = VoiceEngine()
        self.audit = AuditLogger(session_dir)
        self.recorder = VideoRecorder(session_dir)
        self.watcher = ActionWatcher(step_ids=[p.id for p in protocol])

        self._vocabulary = self.fsm.vocabulary()
        self._jpeg: Optional[bytes] = None
        self._jpeg_cond = threading.Condition()
        self._telemetry = Telemetry()
        self._tlock = threading.Lock()
        self._seq = 0
        self._start_ts = time.time()
        self._running = True
        self._thread = threading.Thread(target=self._run, name="bas-pipeline",
                                        daemon=True)
        self._last_ai_warn_ts = 0.0
        self._last_dark_alert_ts = 0.0

    def start(self):
        self._thread.start()

    def stop(self):
        self._running = False
        try:
            self.recorder.release()
        except Exception:
            pass
        try:
            self.camera.release()
        except Exception:
            pass

    # ------------------------------------------------------------ producers

    def _run(self):
        interval = 1.0 / TARGET_STREAM_FPS
        last_cam_open_warn = 0.0
        while self._running:
            loop_start = time.time()
            ok, frame = self.camera.read()
            if not ok or frame is None:
                if time.time() - last_cam_open_warn > 5.0:
                    log.info("No frame from camera (status=%s) - retrying",
                             self.camera.status.value)
                    last_cam_open_warn = time.time()
                self._publish_telemetry(perception_ok=False)
                time.sleep(0.1)
                continue

            try:
                pframe = self.perception.process(frame)
            except Exception:
                log.exception("Perception failed on one frame - skipping")
                time.sleep(0.05)
                continue

            # Advisory AI activity recognition (never changes FSM state).
            try:
                self.watcher.observe(
                    extract_features(pframe.hands, pframe.objects,
                                     self._vocabulary))
            except Exception:
                log.exception("Action watcher failed once - continuing")

            ai_alert = None
            dark_alert = None
            now_s = time.time()
            if (self.camera.is_dark
                    and now_s - self._last_dark_alert_ts > 30.0):
                self._last_dark_alert_ts = now_s
                dark_alert = Alert(
                    level=AlertLevel.WARNING, code="CAMERA_DARK",
                    message="Camera sees no light - uncover the lens, "
                            "point it at the payload, or switch on "
                            "lighting.")
            try:
                alerts = self.fsm.update(
                    hands_count=len(pframe.hands),
                    objects=pframe.objects,
                    interactions=pframe.interactions,
                )
                # Early-warning: AI recognises a different activity than
                # the FSM expects -> advisory HOLD before the mistake lands.
                ai = self.watcher
                if (ai.disagreeing and not self.fsm.finished
                        and ai.action != self.fsm.current_def.id
                        and time.time() - self._last_ai_warn_ts > 15.0):
                    self._last_ai_warn_ts = time.time()
                    ai_alert = Alert(
                        level=AlertLevel.WARNING, code="AI_DEVIATION",
                        message=f"AI activity model recognises '{ai.action}' "
                                f"({ai.confidence:.0%}) while protocol "
                                f"expects '{self.fsm.current_def.id}' - "
                                f"advisory HOLD.",
                        step_index=self.fsm.current_index)
            except Exception:
                log.exception("FSM update failed - continuing")
                alerts = []

            # ---- side channels: voice, audit, local video storage ----
            try:
                for a in alerts:
                    self.voice.alert(a.level.value, a.code,
                                     self._voice_detail(a))
                    self._audit_alert(a)
                if any(a.code == "STEP_COMPLETE" for a in alerts):
                    self._audit_step_complete()
                    self._voice_next_step()
                if ai_alert is not None:
                    self.voice.alert("warning", "HOLD",
                                     "Activity mismatch detected.")
                    self._audit_alert(ai_alert)
                if dark_alert is not None:
                    self.voice.alert("warning", "HOLD",
                                     "Camera sees no light. Check the lens.")
                    self._audit_alert(dark_alert)
            except Exception:
                log.exception("Voice/audit dispatch failed once")

            self._publish_jpeg(pframe.annotated_bgr)
            try:
                self.recorder.write(pframe.annotated_bgr)
            except Exception:
                log.exception("Recorder failed once")

            self._publish_telemetry(
                perception_ok=True, pframe=pframe,
                new_alerts=(alerts
                            + ([ai_alert] if ai_alert else [])
                            + ([dark_alert] if dark_alert else [])))

            spent = time.time() - loop_start
            if spent < interval:
                time.sleep(interval - spent)

    @staticmethod
    def _voice_detail(alert: Alert) -> str:
        if alert.code == "STALLED":
            return str((alert.step_index or 0) + 1)
        return alert.message

    def _audit_alert(self, alert: Alert):
        self.audit.log(alert.code, level=alert.level.value, code=alert.code,
                       step=alert.step_index, detail=alert.message,
                       state="AI_DEVIATION" == alert.code and "HOLD" or "alert")

    def _audit_step_complete(self):
        protocol = get_protocol(os.environ.get("BAS_PROTOCOL", "flask"))
        done_idx = (self.fsm.current_index - 1) if not self.fsm.finished \
            else len(protocol) - 1
        step_def = protocol[done_idx]
        duration = getattr(self.fsm, "_last_step_duration_s", 0.0)
        self.audit.step_complete(done_idx, step_def.id, step_def.name,
                                 1.0, duration)

    def _voice_next_step(self):
        if self.fsm.finished:
            self.voice.alert("info", "PROTOCOL_COMPLETE")
            return
        nxt = self.fsm.current_def
        if nxt is not None:
            self.voice.alert("info", "NEXT_STEP",
                             f"{nxt.name}. {nxt.description}")

    def _publish_jpeg(self, frame):
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        payload = buf.tobytes()
        with self._jpeg_cond:
            self._jpeg = payload
            self._jpeg_cond.notify_all()

    def _publish_telemetry(self, perception_ok: bool, pframe=None,
                           new_alerts=None):
        now = time.time()
        with self._tlock:
            self._seq += 1
            prev_alerts = self._telemetry.alerts
            for alert in (new_alerts or []):
                prev_alerts.insert(0, alert)
            del prev_alerts[ProtocolFSM.ALERT_RING_SIZE:]

            if perception_ok and pframe is not None:
                step_idx, steps, _ = self.fsm.snapshot()
                health = SystemHealth(
                    camera=self.camera.status,
                    camera_source=(self.camera.source_name
                                   or self.camera.source_label),
                    detector=self.perception.detector.mode,
                    hands_model=self.perception.hands.mode,
                    voice=self.voice.mode,
                    action_model=("gru-loaded" if self.watcher.available
                                  else "untrained"),
                    recording_file=self.recorder.filename,
                    fps=pframe.fps,
                    frame_width=pframe.frame_w,
                    frame_height=pframe.frame_h,
                    uptime_s=round(now - self._start_ts, 1),
                )
                state = "nominal"
                for a in prev_alerts[:5]:
                    if a.level == AlertLevel.WARNING and now - a.ts < 10:
                        state = "caution"
                    if a.level == AlertLevel.CRITICAL and now - a.ts < 15:
                        state = "critical"
                        break
                # AI advisory layer.
                ai = self.watcher
                expected = self.fsm.current_def.id if self.fsm.current_def else None
                agrees = (ai.action == expected) if ai.available else None
                hold = bool(ai.available and ai.disagreeing
                            and now - self._last_ai_warn_ts < 10)
                self._telemetry = Telemetry(
                    seq=self._seq,
                    ts=now,
                    health=health,
                    current_step_index=step_idx,
                    protocol=steps,
                    hands=pframe.hands,
                    objects=pframe.objects,
                    interactions=pframe.interactions,
                    alerts=prev_alerts,
                    state=state,
                    ai_action=ai.action,
                    ai_confidence=ai.confidence,
                    ai_agrees=agrees,
                    hold=hold,
                )
            else:
                self._telemetry.seq = self._seq
                self._telemetry.ts = now
                self._telemetry.health.camera = self.camera.status
                self._telemetry.health.fps = 0.0
                self._telemetry.state = "caution"

    # ------------------------------------------------------------ consumers

    def latest_jpeg(self, timeout: float = 2.0) -> Optional[bytes]:
        with self._jpeg_cond:
            if self._jpeg is None:
                self._jpeg_cond.wait(timeout)
            return self._jpeg

    def latest_telemetry(self) -> Telemetry:
        with self._tlock:
            return self._telemetry.model_copy(deep=True)


# --------------------------------------------------------------------------
# FastAPI application
# --------------------------------------------------------------------------

pipeline = Pipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_seed_experiments(RECORDINGS_DIR)
    pipeline.start()
    pipeline.audit.log("SESSION_START",
                       detail="Nabhadrishti copilot session started",
                       protocol=os.environ.get("BAS_PROTOCOL", "flask"),
                       camera=os.environ.get("BAS_CAMERA_SOURCE", "auto"))
    pipeline.voice.speak("Nabhadrishti online. Awaiting video feed.")
    log.info("Nabhadrishti pipeline started (session %s)", session_id)
    yield
    pipeline.audit.log("SESSION_END", detail="Nabhadrishti copilot session ended")
    pipeline.stop()
    log.info("Nabhadrishti pipeline stopped")


app = FastAPI(title="Nabhadrishti - BAS Payload Operations Copilot",
              version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/styles.css")
async def styles():
    return FileResponse(FRONTEND_DIR / "styles.css", media_type="text/css")


@app.get("/app.js")
async def app_js():
    return FileResponse(FRONTEND_DIR / "app.js",
                        media_type="application/javascript")


@app.get("/api/health")
async def health():
    t = pipeline.latest_telemetry()
    return JSONResponse({"status": "ok", "camera": t.health.camera,
                         "state": t.state, "session": session_id})


@app.get("/api/audit/report",
         response_class=PlainTextResponse,
         responses={200: {"content": {"text/plain": {}}}})
async def audit_report(download: bool = False):
    headers = {}
    if download:
        headers["Content-Disposition"] = \
            f'attachment; filename="{session_id}_report.txt"'
    return PlainTextResponse(pipeline.audit.report_text(), headers=headers)


@app.get("/api/audit/log",
         response_class=PlainTextResponse,
         responses={200: {"content": {"application/jsonl": {}}}})
async def audit_log(download: bool = False):
    headers = {}
    if download:
        headers["Content-Disposition"] = \
            f'attachment; filename="{session_id}_audit.jsonl"'
    return PlainTextResponse(pipeline.audit.events_jsonl(), headers=headers)


@app.get("/api/recordings")
async def recordings():
    items = []
    if RECORDINGS_DIR.exists():
        for p in sorted(RECORDINGS_DIR.rglob("*")):
            if p.is_file():
                items.append({"file": str(p.relative_to(RECORDINGS_DIR)),
                              "bytes": p.stat().st_size,
                              "modified": p.stat().st_mtime})
    return JSONResponse({"session": session_id, "files": items})


@app.get("/api/experiments")
async def list_experiments():
    """List all recorded experiments (past and active) with metrics and status."""
    exps = get_all_experiments(RECORDINGS_DIR, session_id)
    return JSONResponse({
        "current_session": session_id,
        "total": len(exps),
        "experiments": exps,
    })


@app.get("/api/experiments/{target_id}")
async def experiment_detail(target_id: str):
    """Deep details, step progression, and anomaly logs for a specific experiment."""
    data = get_experiment_detail(RECORDINGS_DIR, target_id, session_id)
    if not data:
        return JSONResponse({"error": "Experiment session not found"}, status_code=404)
    return JSONResponse(data)


@app.get("/api/experiments/{target_id}/report",
         response_class=PlainTextResponse,
         responses={200: {"content": {"text/plain": {}}}})
async def experiment_report_download(target_id: str, download: bool = False):
    """Retrieve human-readable report.txt for an experiment."""
    data = get_experiment_detail(RECORDINGS_DIR, target_id, session_id)
    if not data:
        return PlainTextResponse("Experiment session not found", status_code=404)
    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{target_id}_report.txt"'
    return PlainTextResponse(data.get("report_text", ""), headers=headers)


@app.get("/api/experiments/{target_id}/log",
         response_class=PlainTextResponse,
         responses={200: {"content": {"application/jsonl": {}}}})
async def experiment_log_download(target_id: str, download: bool = False):
    """Retrieve structured audit.jsonl for an experiment."""
    p = RECORDINGS_DIR / target_id / "audit.jsonl"
    if not p.exists():
        return PlainTextResponse("Audit log not found", status_code=404)
    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{target_id}_audit.jsonl"'
    return PlainTextResponse(p.read_text(encoding="utf-8"), headers=headers)


@app.get("/api/experiments/{target_id}/video")
async def experiment_video_download(target_id: str):
    """Retrieve or download recorded .avi video for an experiment."""
    p = RECORDINGS_DIR / target_id / "experiment_video.avi"
    if not p.exists():
        return JSONResponse({"error": "Video recording not found for this experiment"}, status_code=404)
    return FileResponse(p, media_type="video/x-msvideo", filename=f"{target_id}_video.avi")


class CameraSelectRequest(BaseModel):
    source: str


@app.get("/api/cameras")
def list_cameras():
    """Selectable cameras for the HUD dropdown (verified + active)."""
    cams = enumerate_camera_sources(
        active_id=pipeline.camera.current_source_id)
    return JSONResponse({
        "current": pipeline.camera.current_source_id,
        "current_name": pipeline.camera.source_name or
                        pipeline.camera.source_label,
        "cameras": cams,
    })


@app.post("/api/camera/select")
def select_camera(req: CameraSelectRequest):
    """Hot-swap the camera at runtime; the FSM, HUD and recording continue."""
    ok, name = pipeline.camera.switch_source(req.source)
    pipeline.audit.log("CAMERA_SWITCH", ok=ok, source=req.source,
                       detail=f"Camera switched to '{name}'"
                              + ("" if ok else " (switch FAILED, previous "
                                            "source restored)"))
    if ok:
        pipeline.voice.speak(f"Camera switched to {name}")
    return JSONResponse({"ok": ok, "source": name})


BOUNDARY = "basframe"


def _mjpeg_part(jpeg: bytes) -> bytes:
    """One multipart part with an explicit Content-Length."""
    headers = (
        f"--{BOUNDARY}\r\n"
        f"Content-Type: image/jpeg\r\n"
        f"Content-Length: {len(jpeg)}\r\n\r\n"
    ).encode("ascii")
    return headers + jpeg + b"\r\n"


@app.get("/video_feed")
async def video_feed():
    async def stream():
        loop = asyncio.get_running_loop()
        while True:
            jpeg = await loop.run_in_executor(None, pipeline.latest_jpeg, 2.0)
            if jpeg is None:
                await asyncio.sleep(0.2)
                continue
            yield _mjpeg_part(jpeg)
            await asyncio.sleep(1.0 / TARGET_STREAM_FPS)

    return StreamingResponse(
        stream(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    await ws.accept()
    log.info("Telemetry client connected")
    interval = 1.0 / TELEMETRY_HZ
    try:
        while True:
            payload = pipeline.latest_telemetry().model_dump_json()
            await ws.send_text(payload)
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        log.info("Telemetry client disconnected")
    except Exception as exc:
        log.info("Telemetry client dropped: %s", exc)
        try:
            await ws.close()
        except Exception:
            pass


_ = cv2
