"""
BAS Experiment Copilot - Protocol Library
==========================================
SIH 26174 sample experiment: a container holds two smaller boxes, red and
yellow. The protocol below encodes that sequence; edit it freely or point
BAS_PROTOCOL at a custom protocol defined here.

Protocols are pure data - adding a new experiment requires no code changes
in perception or the server (the "config-driven protocols" innovation).
Object labels are either COCO classes (stock yolov8n.pt works offline) or
color-qualified labels produced by perception.ColorTagger
("red_box" / "yellow_box").
"""

from __future__ import annotations

from .fsm import DEFAULT_PROTOCOL, ProtocolStepDef

SAMPLE_EXPERIMENT: list[ProtocolStepDef] = [
    ProtocolStepDef(
        id="INSPECT",
        name="Open Stowage Box",
        description="Open the container so that BOTH the red and the "
                    "yellow box are clearly visible to the payload camera.",
        target_object=None,
        requires_visible=["red_box", "yellow_box"],
        requires_dwell_s=2.0,
    ),
    ProtocolStepDef(
        id="EXTRACT_RED",
        name="Extract Red Box",
        description="Lift the RED box out of the container and hold it up.",
        target_object="red_box",
        requires_hold=True,
        requires_dwell_s=1.5,
    ),
    ProtocolStepDef(
        id="STOW_RED",
        name="Place Red Box Aside",
        description="Set the red box down clear of the container and "
                    "retract your hands.",
        target_object="red_box",
        requires_release=True,
        requires_dwell_s=1.5,
    ),
    ProtocolStepDef(
        id="EXTRACT_YELLOW",
        name="Extract Yellow Box",
        description="Lift the YELLOW box and hold it up.",
        target_object="yellow_box",
        requires_hold=True,
        requires_dwell_s=1.5,
    ),
    ProtocolStepDef(
        id="NEST_YELLOW",
        name="Nest Yellow in Red",
        description="Bring the yellow box to the red box and place it "
                    "inside / adjacent, then release.",
        target_object="yellow_box",
        requires_proximity_to="red_box",
        requires_release=True,
        requires_dwell_s=2.0,
    ),
    ProtocolStepDef(
        id="VERIFY",
        name="Final Verification",
        description="Both boxes assembled and hands clear of the payload.",
        target_object=None,
        requires_visible=["red_box", "yellow_box"],
        requires_dwell_s=2.0,
        is_final=True,
    ),
]

# Hand-gesture mission: no props needed - the astronaut IS the payload.
# Gestures are recognised from MediaPipe landmarks (backend/gestures.py)
# and emitted as pseudo-objects ("gesture_peace", "gesture_wave", ...).
GESTURE_MISSION: list[ProtocolStepDef] = [
    ProtocolStepDef(
        id="SHOW_PALM",
        name="Show Open Palm",
        description="Raise your hand and show an OPEN PALM to the camera.",
        target_object="gesture_open_palm",
        requires_dwell_s=2.0,
    ),
    ProtocolStepDef(
        id="PEACE",
        name="Show Two Fingers",
        description="Make a V sign - index and middle finger extended.",
        target_object="gesture_peace",
        requires_dwell_s=2.0,
    ),
    ProtocolStepDef(
        id="WAVE",
        name="Wave Goodbye",
        description="Wave your open hand side to side at the camera.",
        target_object="gesture_wave",
        requires_dwell_s=2.5,
    ),
    ProtocolStepDef(
        id="FINGER_GUN",
        name="Make a Finger Gun",
        description="Index finger pointing, thumb raised - finger gun pose.",
        target_object="gesture_gun",
        requires_dwell_s=2.0,
    ),
    ProtocolStepDef(
        id="FIST",
        name="Make a Fist",
        description="Curl all fingers into a fist and hold.",
        target_object="gesture_fist",
        requires_dwell_s=2.0,
    ),
]

PROTOCOLS = {
    "flask": DEFAULT_PROTOCOL,
    "sample": SAMPLE_EXPERIMENT,
    "gesture": GESTURE_MISSION,
}


def get_protocol(name: str) -> list[ProtocolStepDef]:
    return PROTOCOLS.get((name or "flask").lower(), DEFAULT_PROTOCOL)
