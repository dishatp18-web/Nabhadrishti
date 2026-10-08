"""
BAS Experiment Copilot - Protocol Finite State Machine
=======================================================
Pure logic: consumes a perception snapshot from the CV engine and produces
step transitions plus anomaly alerts. No OpenCV, no network, no UI code
lives here - the FSM is fully unit-testable from synthetic snapshots.

Protocol model
--------------
A protocol is an ordered list of ProtocolStepDef entries. Each step declares:
  * target_object       - which scientific object the astronaut must handle
  * requires_hold       - step completes only while the object is LIFTED/held
  * requires_proximity_to - ...while near this auxiliary object
  * requires_dwell_s    - seconds the condition must hold continuously

Default demo protocol (reagent transfer experiment):
  1. PICKUP_FLASK   - hold the reagent flask
  2. POUR           - bring the flask near the beaker while holding it
  3. PLACE_HEATER   - release the flask near the heater pad, hands retracted

Anomaly detection
-----------------
  * SKIPPED_STEP   - interaction with a FUTURE step's target object while an
                     earlier step is still active.
  * CONTAMINATION  - contact with an off-protocol object that has appeared
                     in earlier or later steps of the protocol.
  * STALLED        - active step has not progressed for > STALL_TIMEOUT_S.
  * OBJECT_MISSING - current step's target object unseen for > MISSING_TIMEOUT_S.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .models import (
    Alert,
    AlertLevel,
    Interaction,
    ProtocolStep,
    StepStatus,
    TrackedObject,
)

log = logging.getLogger("bas.fsm")

# Normalized center distance below which two objects count as "near".
NEAR_DIST = 0.30


# --------------------------------------------------------------------------
# Protocol definitions
# --------------------------------------------------------------------------

@dataclass
class ProtocolStepDef:
    id: str
    name: str
    description: str
    target_object: Optional[str]
    requires_hold: bool = False
    requires_proximity_to: Optional[str] = None
    requires_dwell_s: float = 1.2
    is_final: bool = False
    # All listed objects must be visible simultaneously (e.g. "open the
    # box so red and yellow are both seen") - no manipulation required.
    requires_visible: Optional[List[str]] = None
    # Object must be set down: visible near its aux (if any) and NOT held.
    requires_release: bool = False


DEFAULT_PROTOCOL: List[ProtocolStepDef] = [
    ProtocolStepDef(
        id="PICKUP_FLASK",
        name="Retrieve Reagent Flask",
        description="Lift and hold the reagent flask from the stowage rack.",
        target_object="bottle",
        requires_hold=True,
        requires_dwell_s=1.5,
    ),
    ProtocolStepDef(
        id="POUR",
        name="Dispense Reagent",
        description="Bring the flask over the beaker and hold position to pour.",
        target_object="bottle",
        requires_proximity_to="cup",
        requires_hold=True,
        requires_dwell_s=2.5,
    ),
    ProtocolStepDef(
        id="PLACE_HEATER",
        name="Secure Sample on Heater",
        description="Release the flask onto the heater pad and retract hands.",
        target_object="bottle",
        requires_proximity_to="oven",
        requires_hold=False,
        requires_dwell_s=1.5,
        is_final=True,
    ),
]


@dataclass
class _StepRuntime:
    condition_since: Optional[float] = None
    progress: float = 0.0
    last_progress_ts: float = field(default_factory=time.time)
    first_seen_ts: Optional[float] = None


class ProtocolFSM:
    """Evaluates perception snapshots and drives protocol state."""

    STALL_TIMEOUT_S = 12.0
    MISSING_TIMEOUT_S = 20.0
    ALERT_RING_SIZE = 40
    # Detections below this confidence are ignored entirely: the FSM never
    # changes state on weak evidence ("low confidence -> HOLD").
    MIN_CONFIDENCE = 0.45

    def __init__(self, protocol: Optional[List[ProtocolStepDef]] = None):
        self._protocol = protocol if protocol is not None else list(DEFAULT_PROTOCOL)
        self._index = 0
        self._status: List[StepStatus] = [StepStatus.PENDING] * len(self._protocol)
        self._runtime = [_StepRuntime() for _ in self._protocol]
        self._status[0] = StepStatus.ACTIVE
        self._alerts: List[Alert] = []
        self._finished = False
        self._skipped_flags: set = set()
        self._last_target_seen = time.time()
        self._last_stall_alert_ts = 0.0
        self._last_missing_alert_ts = 0.0
        self._step_started_ts = time.time()

    # ------------------------------------------------------------- accessors

    @property
    def step_started_ts(self) -> float:
        return self._step_started_ts

    def vocabulary(self) -> List[str]:
        """Ordered unique object labels referenced by the protocol."""
        labels: List[str] = []
        for p in self._protocol:
            for lbl in ([p.target_object, p.requires_proximity_to]
                        + list(p.requires_visible or [])):
                if lbl and lbl not in labels:
                    labels.append(lbl)
        return labels

    @property
    def current_index(self) -> int:
        return self._index

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def current_def(self) -> Optional[ProtocolStepDef]:
        if self._index < len(self._protocol):
            return self._protocol[self._index]
        return None

    # ------------------------------------------------------------- main tick

    def update(self,
               hands_count: int,
               objects: List[TrackedObject],
               interactions: List[Interaction]) -> List[Alert]:
        """Feed one perception snapshot; returns alerts raised this tick."""
        now = time.time()
        new_alerts: List[Alert] = []
        if self._finished:
            return new_alerts

        # Confidence gate: never let weak detections drive the protocol.
        objects = [o for o in objects if o.confidence >= self.MIN_CONFIDENCE]

        step = self._protocol[self._index]
        rt = self._runtime[self._index]
        held = {i.object_id for i in interactions if i.kind == "lift"}
        touched = {i.object_id for i in interactions}
        target_obj = self._find_object(objects, step.target_object)

        if target_obj is not None:
            if rt.first_seen_ts is None:
                rt.first_seen_ts = now
            self._last_target_seen = now

        # --- anomaly: interacting with a FUTURE step's target ------------
        for obj in objects:
            if step.target_object is not None and obj.label == step.target_object:
                continue  # current target is legitimately being handled
            future_idx = self._future_step_for(obj.label)
            if future_idx is None or future_idx <= self._index:
                continue
            if obj.object_id in touched and (future_idx, obj.object_id) not in self._skipped_flags:
                self._skipped_flags.add((future_idx, obj.object_id))
                new_alerts.append(self._raise(
                    AlertLevel.WARNING, "SKIPPED_STEP",
                    f"Interaction with '{obj.label}' detected while step "
                    f"{self._index + 1} ('{step.name}') is still active - "
                    f"possible sequence violation.",
                    step_index=self._index))

        # --- anomaly: off-protocol contact -------------------------------
        for obj in objects:
            if obj.object_id not in touched:
                continue
            # Hand poses are not physical objects - an idle hand naturally
            # forms protocol gestures, so never flag contamination on them.
            if obj.label.startswith("gesture_"):
                continue
            is_target = step.target_object is not None and obj.label == step.target_object
            is_aux = step.requires_proximity_to is not None \
                and obj.label == step.requires_proximity_to
            if not is_target and not is_aux and self._step_for_label(obj.label) is not None:
                new_alerts.append(self._raise(
                    AlertLevel.WARNING, "CONTAMINATION",
                    f"Off-protocol contact with '{obj.label}' during step "
                    f"{self._index + 1} - contamination risk flagged.",
                    step_index=self._index))

        # --- evaluate current step ---------------------------------------
        condition_met = self._evaluate(step, objects, target_obj, held, hands_count)
        if condition_met:
            if rt.condition_since is None:
                rt.condition_since = now
            held_for = now - rt.condition_since
            rt.progress = min(1.0, held_for / max(step.requires_dwell_s, 0.1))
            rt.last_progress_ts = now
            if rt.progress >= 1.0:
                done_idx = self._index
                duration = now - self._step_started_ts
                self._complete_step(now)
                self._step_started_ts = now
                new_alerts.append(self._raise(
                    AlertLevel.INFO, "STEP_COMPLETE",
                    f"Step {done_idx + 1} complete: {self._protocol[done_idx].name}",
                    step_index=done_idx))
                # Stash the verified duration for the audit trail.
                self._last_step_duration_s = duration
        else:
            rt.condition_since = None
            rt.progress = 0.0
            stalled_for = now - rt.last_progress_ts
            if stalled_for > self.STALL_TIMEOUT_S and now - self._last_stall_alert_ts > self.STALL_TIMEOUT_S:
                self._last_stall_alert_ts = now
                new_alerts.append(self._raise(
                    AlertLevel.WARNING, "STALLED",
                    f"Step {self._index + 1} '{step.name}' has not progressed "
                    f"in {int(stalled_for)}s.",
                    step_index=self._index))
            if (step.target_object and target_obj is None
                    and now - self._last_target_seen > self.MISSING_TIMEOUT_S
                    and now - self._last_missing_alert_ts > self.MISSING_TIMEOUT_S):
                self._last_missing_alert_ts = now
                new_alerts.append(self._raise(
                    AlertLevel.WARNING, "OBJECT_MISSING",
                    f"Target '{step.target_object}' not visible for "
                    f"{int(now - self._last_target_seen)}s.",
                    step_index=self._index))

        return new_alerts

    # ------------------------------------------------------------- internals

    def _evaluate(self, step: ProtocolStepDef,
                  objects: List[TrackedObject],
                  target_obj: Optional[TrackedObject],
                  held: set, hands_count: int) -> bool:
        """Is the step's completion condition satisfied right now?"""
        # Visibility-only conditions (open box, final verification, ...).
        for lbl in (step.requires_visible or []):
            if self._find_object(objects, lbl) is None:
                return False
        if step.target_object is None:
            return True
        if target_obj is None:
            return False
        met = True

        aux = None
        if step.requires_proximity_to is not None:
            aux = self._find_object(objects, step.requires_proximity_to)
            if aux is None or self._center_dist(target_obj, aux) > NEAR_DIST:
                met = False

        if step.requires_hold:
            if target_obj.object_id not in held:
                met = False

        if step.requires_release or (step.is_final and not step.requires_hold):
            # The object must be set down, not carried.
            if target_obj.held_by_hand_id is not None or target_obj.object_id in held:
                met = False

        return met

    @staticmethod
    def _center_dist(a: TrackedObject, b: TrackedObject) -> float:
        ax, ay = a.box.x + a.box.w / 2, a.box.y + a.box.h / 2
        bx, by = b.box.x + b.box.w / 2, b.box.y + b.box.h / 2
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    def _find_object(self, objects: List[TrackedObject],
                     label: Optional[str]) -> Optional[TrackedObject]:
        if not label:
            return None
        best, best_conf = None, 0.0
        for o in objects:
            if o.label == label and o.confidence >= best_conf:
                best, best_conf = o, o.confidence
        return best

    def _step_for_label(self, label: str) -> Optional[int]:
        for i, pdef in enumerate(self._protocol):
            if pdef.target_object == label:
                return i
        return None

    def _future_step_for(self, label: str) -> Optional[int]:
        for i in range(self._index + 1, len(self._protocol)):
            if self._protocol[i].target_object == label:
                return i
        return None

    def _complete_step(self, now: float):
        self._status[self._index] = StepStatus.COMPLETE
        self._runtime[self._index].progress = 1.0
        if self._index + 1 < len(self._protocol):
            self._index += 1
            self._status[self._index] = StepStatus.ACTIVE
            self._runtime[self._index].last_progress_ts = now
            log.info("FSM advanced to step %d: %s",
                     self._index + 1, self._protocol[self._index].name)
        else:
            self._finished = True
            log.info("Protocol complete")

    def _raise(self, level: AlertLevel, code: str, message: str,
               step_index: Optional[int] = None) -> Alert:
        alert = Alert(level=level, code=code, message=message,
                      step_index=step_index)
        self._alerts.insert(0, alert)
        del self._alerts[self.ALERT_RING_SIZE:]
        log.log(logging.INFO if level == AlertLevel.INFO else logging.WARNING,
                "[%s] %s", code, message)
        return alert

    # ------------------------------------------------------------- snapshot

    def snapshot(self) -> Tuple[int, List[ProtocolStep], List[Alert]]:
        """Current index, protocol steps for the HUD, and recent alerts."""
        steps = []
        for i, (pdef, rt) in enumerate(zip(self._protocol, self._runtime)):
            steps.append(ProtocolStep(
                index=i,
                id=pdef.id,
                name=pdef.name,
                description=pdef.description,
                status=self._status[i],
                target_object=pdef.target_object,
                progress=rt.progress,
            ))
        return self._index, steps, list(self._alerts)
