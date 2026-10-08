"""
BAS Experiment Copilot - Data Contracts
========================================
Explicit Pydantic schemas defining every piece of JSON that crosses the
API boundary. Perception (CV), the FSM, and the WebSocket producer all
serialize through these models, so the frontend contract is defined in
exactly one place.

All bounding boxes are NORMALIZED (0.0 - 1.0) relative to the frame,
with origin at the top-left corner. The HUD draws SVG overlays directly
from these values, so nothing in the UI needs to know frame resolution.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Geometry / perception primitives
# --------------------------------------------------------------------------

class BBox(BaseModel):
    """Normalized bounding box. x, y = top-left corner; w, h = extents."""
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    w: float = Field(..., ge=0.0, le=1.0)
    h: float = Field(..., ge=0.0, le=1.0)


class Landmark(BaseModel):
    """Normalized hand landmark (MediaPipe hands, 21 points)."""
    x: float
    y: float
    z: float = 0.0


class Hand(BaseModel):
    """A detected hand."""
    hand_id: int = 0
    label: str = "Hand"            # "Left" / "Right"
    box: BBox
    landmarks: Optional[List[Landmark]] = None


class TrackedObject(BaseModel):
    """A detected scientific object (YOLOv8 detection, tracked frame to frame)."""
    object_id: int = 0
    label: str
    confidence: float = 0.0
    box: BBox
    held_by_hand_id: Optional[int] = None
    nearest_hand_dist: Optional[float] = None   # normalized center distance


class Interaction(BaseModel):
    """A proximity/contact event between a hand and an object this frame."""
    kind: str                        # "contact" | "near" | "lift"
    hand_id: int
    object_id: int
    object_label: str
    strength: float = 0.0            # IoU for contact, 1 - dist for near


# --------------------------------------------------------------------------
# FSM / telemetry payloads
# --------------------------------------------------------------------------

class StepStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETE = "complete"
    FAILED = "failed"


class ProtocolStep(BaseModel):
    """One step of the experiment protocol as reported to the HUD."""
    index: int
    id: str
    name: str
    description: str = ""
    status: StepStatus = StepStatus.PENDING
    target_object: Optional[str] = None
    progress: float = Field(0.0, ge=0.0, le=1.0)   # dwell-based completion 0..1


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Alert(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    level: AlertLevel
    code: str                        # machine code, e.g. "SKIPPED_STEP"
    message: str
    step_index: Optional[int] = None
    ts: float = Field(default_factory=time.time)


class CameraStatus(str, Enum):
    OFFLINE = "offline"
    RECONNECTING = "reconnecting"
    LIVE = "live"
    SIMULATED = "simulated"          # synthetic test-pattern fallback


class SystemHealth(BaseModel):
    camera: CameraStatus = CameraStatus.OFFLINE
    camera_source: str = ""
    detector: str = "unavailable"    # "yolov8", "heuristic", "unavailable"
    hands_model: str = "unavailable" # "mediapipe", "unavailable"
    voice: str = "unavailable"       # "pyttsx3-offline", "beep-fallback"
    action_model: str = "unavailable"  # "gru-loaded", "untrained"
    recording_file: Optional[str] = None
    fps: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    uptime_s: float = 0.0


class Telemetry(BaseModel):
    """The single JSON document streamed over /ws/telemetry, ~10 Hz."""
    seq: int = 0
    ts: float = Field(default_factory=time.time)
    health: SystemHealth = Field(default_factory=SystemHealth)
    current_step_index: int = 0
    protocol: List[ProtocolStep] = Field(default_factory=list)
    hands: List[Hand] = Field(default_factory=list)
    objects: List[TrackedObject] = Field(default_factory=list)
    interactions: List[Interaction] = Field(default_factory=list)
    alerts: List[Alert] = Field(default_factory=list)   # recent ring buffer, newest first
    state: str = "nominal"           # nominal | caution | critical
    # --- AI action-recognition layer ("AI proposes, FSM disposes") ---
    ai_action: Optional[str] = None          # recognised activity (step id)
    ai_confidence: float = 0.0
    ai_agrees: Optional[bool] = None         # None = model not loaded
    hold: bool = False                       # low-confidence advisory hold
