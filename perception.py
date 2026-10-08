"""
BAS Experiment Copilot - Perception Engine (The "Eyes")
========================================================
Wraps MediaPipe Hands and YOLOv8 and turns raw frames into the typed
perception snapshot the FSM consumes:

    PerceptionFrame(hands, objects, interactions, annotated_bgr, fps, ...)

Decoupling rules:
  * Knows nothing about the FSM or the web server.
  * Owns no capture logic - it is handed frames from camera.CameraManager.
  * Heavy models are imported LAZILY and every failure degrades to a
    lighter mode instead of crashing the pipeline:

        detector:  yolov8  ->  heuristic  ->  unavailable
        hands:     mediapipe  ->  unavailable

    "heuristic" mode detects saturated/colorful blobs (typical lab glass
    and hardware) with classic OpenCV so the demo still runs on machines
    without the ultralytics stack.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .models import BBox, Hand, Interaction, Landmark, TrackedObject
from .gestures import GestureRecognizer

log = logging.getLogger("bas.perception")

# Interaction thresholds (normalized coordinates).
CONTACT_IOU = 0.12        # hand box overlaps object box -> touching
NEAR_DIST = 0.22          # normalized center distance -> "near"
LIFT_DROP_FRACTION = 0.35 # object raised above its lowest seen position -> held

# Colors (BGR) used to annotate the MJPEG stream.
COL_HAND = (90, 230, 80)       # green
COL_OBJ = (60, 185, 255)       # amber
COL_HELD = (80, 80, 255)       # red when held
COL_CONTACT = (60, 220, 255)   # yellow
COL_TEXT = (240, 240, 240)


@dataclass
class PerceptionFrame:
    """One complete perception snapshot + annotated frame for the stream."""
    hands: List[Hand] = field(default_factory=list)
    objects: List[TrackedObject] = field(default_factory=list)
    interactions: List[Interaction] = field(default_factory=list)
    annotated_bgr: Optional[np.ndarray] = None
    fps: float = 0.0
    frame_w: int = 0
    frame_h: int = 0


class HandTracker:
    """MediaPipe Hands wrapper. Degrades to an empty tracker if unavailable."""

    def __init__(self):
        self.mode = "unavailable"
        self._hands = None
        try:
            import mediapipe as mp  # heavy import - keep lazy
            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
            )
            self.mode = "mediapipe"
            log.info("Hand tracking: MediaPipe")
        except Exception as exc:
            log.warning("MediaPipe unavailable (%s) - hand tracking disabled", exc)

    def process(self, frame_bgr: np.ndarray) -> List[Hand]:
        if self._hands is None:
            return []
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # MediaPipe requires read-only contiguous arrays; copy is cheap insurance.
        rgb.flags.writeable = False
        result = self._hands.process(rgb)
        hands: List[Hand] = []
        if not result.multi_hand_landmarks:
            return hands
        handedness = (result.multi_handedness
                      if result.multi_handedness else [None] * len(result.multi_hand_landmarks))
        for i, lm_set in enumerate(result.multi_hand_landmarks):
            xs = [lm.x for lm in lm_set.landmark]
            ys = [lm.y for lm in lm_set.landmark]
            x0, y0 = max(min(xs), 0.0), max(min(ys), 0.0)
            x1, y1 = min(max(xs), 1.0), min(max(ys), 1.0)
            label = "Hand"
            if i < len(handedness) and handedness[i] is not None:
                try:
                    label = handedness[i].classification[0].label
                except Exception:
                    pass
            # OpenCV frames are un-mirrored (raw sensor), but MediaPipe
            # computes handedness assuming a mirrored selfie view - so the
            # labels arrive swapped. Flip them back so the HUD reports the
            # astronaut's actual hand.
            label = {"Left": "Right", "Right": "Left"}.get(label, label)
            hands.append(Hand(
                hand_id=i,
                label=label,
                box=BBox(x=x0, y=y0, w=max(x1 - x0, 0.01), h=max(y1 - y0, 0.01)),
                landmarks=[Landmark(x=lm.x, y=lm.y, z=lm.z) for lm in lm_set.landmark],
            ))
        return hands

    def close(self):
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception:
                pass


class ColorTagger:
    """Color-qualifies object labels using hue statistics inside the bbox.

    Turns a generic 'box' detection into 'red_box' / 'yellow_box' so
    color-driven protocols (SIH 26174 sample experiment) work with a stock
    COCO model, no retraining required. Only applied when the active
    protocol's vocabulary actually references the color labels.
    """

    # HSV hue ranges (OpenCV hue 0-179).
    RANGES = {
        "red": [(0, 10), (170, 179)],
        "yellow": [(20, 35)],
    }
    TAGGABLE = {"box", "bottle", "cup", "vase", "bowl"}

    @classmethod
    def classify(cls, frame_hsv: np.ndarray, box: BBox) -> Optional[str]:
        h, w = frame_hsv.shape[:2]
        x0 = max(int(box.x * w), 0)
        y0 = max(int(box.y * h), 0)
        x1 = min(int((box.x + box.w) * w), w - 1)
        y1 = min(int((box.y + box.h) * h), h - 1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        crop = frame_hsv[y0:y1, x0:x1]
        sat = crop[:, :, 1].astype(np.float32)
        hue = crop[:, :, 0].astype(np.float32)
        # Only strongly saturated pixels define the object's color.
        mask = sat > 90
        if mask.mean() < 0.15:
            return None
        hue_m = hue[mask]
        for color, ranges in cls.RANGES.items():
            frac = 0.0
            for lo, hi in ranges:
                if lo <= hi:
                    frac += float(((hue_m >= lo) & (hue_m <= hi)).mean())
                else:  # wraps around 180
                    frac += float(((hue_m >= lo) | (hue_m <= hi)).mean())
            if frac > 0.5:
                return color
        return None


# Open-vocabulary classes for YOLO-World ("--wide" mode): recognizes any
# listed concept zero-shot, no training needed. Stock yolov8n only covers
# the 80 COCO classes (cell phone, laptop, book, scissors...) which misses
# pens, pencils, glasses, headphones etc. - this vocabulary fills that gap.
OPEN_VOCAB_CLASSES = [
    # stationery / desk
    "pen", "pencil", "eraser", "marker", "ruler", "scissors", "stapler",
    "paper clip", "notebook", "book", "paper", "envelope", "glue stick",
    # personal items
    "spectacles", "sunglasses", "headphones", "earphones", "cell phone",
    "smartwatch", "keyboard", "mouse", "laptop", "monitor", "remote",
    "charger", "cable", "usb drive", "battery", "power bank", "wallet",
    "key", "card", "watch", "torch",
    # lab / household
    "bottle", "cup", "mug", "bowl", "plate", "spoon", "fork", "knife",
    "beaker", "flask", "test tube", "syringe", "funnel", "glove", "mask",
    "tissue", "tape", "screwdriver", "pliers", "hammer", "wrench",
    "flashlight", "candle", "box", "cloth", "towel", "brush", "comb",
]


class ObjectDetector:
    """YOLOv8 / YOLO-World wrapper with a classic-CV heuristic fallback.

    Model selection priority (first that loads wins):
      1. BAS_YOLO_MODEL env var - a custom fine-tuned weights path
         (e.g. trained with tools/train_detector.py).
      2. Open-vocabulary mode (BAS_OPEN_VOCAB=1 or run.py --wide):
         YOLO-World recognizes the OPEN_VOCAB_CLASSES list zero-shot -
         pens, pencils, spectacles, headphones, tools, etc. One-time
         model download (~50 MB), then fully offline.
      3. Stock yolov8n.pt (80 COCO classes).
      4. Heuristic blob detector (no ultralytics installed).

    The fallback looks for bright, saturated, contoured blobs - good enough
    to track a flask / beaker demo rig with colored markers, and it keeps
    the whole system runnable on machines without torch installed.
    """

    def __init__(self, conf: float = 0.35):
        self.mode = "unavailable"
        self._model = None
        self._min_area_frac = 0.002

        custom = os.environ.get("BAS_YOLO_MODEL", "").strip()
        wide = os.environ.get("BAS_OPEN_VOCAB", "").lower() in ("1", "true", "yes", "on")
        vocab = [c.strip() for c in
                 os.environ.get("BAS_OPEN_VOCAB_CLASSES", "").split(",") if c.strip()] \
            or OPEN_VOCAB_CLASSES

        # Inference resolution + frame-skipping (FPS tuning).
        # Open-vocab mode defaults to a smaller image and detects every 2nd
        # frame with a cached result in between - hands/voice stay real-time
        # while object boxes refresh at ~6 Hz.
        self.imgsz = int(os.environ.get("BAS_YOLO_IMGSZ", "0") or 0) \
            or (416 if wide else 640)
        self.every_n = int(os.environ.get("BAS_DETECT_EVERY", "0") or 0) \
            or (2 if wide else 1)
        self._frame_i = 0
        self._cache: List[Tuple[str, float, BBox]] = []

        if custom:
            self._load(custom, conf, "custom")
        elif wide:
            # Open-vocabulary: the smallest official YOLO-World is the
            # small model (~100ms CPU at imgsz 416 with frame-skipping).
            # Override via BAS_OPEN_VOCAB_MODEL. Open-vocab scores run
            # cooler than closed-set YOLO -> low conf threshold.
            self._load(os.environ.get("BAS_OPEN_VOCAB_MODEL", "").strip()
                       or "yolov8s-worldv2.pt", 0.22, "yolo-world", vocab) \
                or (log.warning("YOLO-World unavailable - falling back to yolov8n"),
                    self._load("yolov8n.pt", conf, "yolov8"))
        else:
            if not self._load("yolov8n.pt", conf, "yolov8"):
                log.warning("Ultralytics unavailable - using heuristic blob detector")
                self.mode = "heuristic"

    def _load(self, model_name: str, conf: float, mode: str,
              classes: Optional[List[str]] = None) -> bool:
        try:
            from ultralytics import YOLO  # heavy import - keep lazy
            model = YOLO(model_name)
            if classes:
                model.set_classes(classes)
            self._model = model
            self.conf = conf
            self.mode = mode
            log.info("Object detection: %s (%s)%s", mode, model_name,
                     f" - {len(classes)} open-vocab classes" if classes else "")
            return True
        except Exception as exc:
            log.warning("Could not load %s: %s", model_name, exc)
            return False

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[str, float, BBox]]:
        """Return [(label, confidence, normalized_bbox), ...]

        In every_n>1 mode, results are refreshed only every Nth frame and
        served from cache in between (keeps the pipeline real-time when the
        detector is the bottleneck, e.g. open-vocabulary on CPU).
        """
        if self._model is None:
            return self._detect_heuristic(frame_bgr)
        self._frame_i += 1
        if self.every_n > 1 and self._cache \
                and (self._frame_i % self.every_n) != 0:
            return list(self._cache)
        out = self._detect_yolo(frame_bgr)
        self._cache = list(out)
        return out

    def _detect_yolo(self, frame_bgr: np.ndarray) -> List[Tuple[str, float, BBox]]:
        h, w = frame_bgr.shape[:2]
        out: List[Tuple[str, float, BBox]] = []
        try:
            results = self._model.predict(frame_bgr, conf=self.conf,
                                          imgsz=self.imgsz, verbose=False)
        except Exception as exc:
            log.warning("YOLO inference failed once: %s", exc)
            return out
        for res in results:
            for box in res.boxes:
                try:
                    cls_id = int(box.cls[0])
                    label = res.names.get(cls_id, str(cls_id))
                    conf = float(box.conf[0])
                    x0, y0, x1, y1 = box.xyxyn[0].tolist()
                    out.append((label, conf,
                                BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0)))
                except Exception:
                    continue
        return out

    def _detect_heuristic(self, frame_bgr: np.ndarray) -> List[Tuple[str, float, BBox]]:
        h, w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # Saturated, bright-ish regions (colored lab markers / glass contents).
        mask = cv2.inRange(hsv, (0, 90, 80), (179, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        out: List[Tuple[str, float, BBox]] = []
        for c in contours[:12]:
            area = cv2.contourArea(c)
            if area < self._min_area_frac * w * h:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            # Classify crudely by aspect ratio so the FSM has stable labels.
            aspect = bh / max(bw, 1)
            label = "bottle" if aspect > 1.4 else "cup"
            out.append((label, 0.60, BBox(x=x / w, y=y / h,
                                          w=bw / w, h=bh / h)))
        return out

    def close(self):
        pass


class InteractionEngine:
    """Proximity / contact / lift reasoning between hands and objects.

    Keeps, per tracked object id:
      * the previous box  - for greedy IoU id-stability across frames
      * the lowest center-y ever seen - its "resting" height, so LIFT is
        detected when the object currently sits well above that rest line
    """

    def __init__(self):
        self._prev_boxes: dict = {}
        self._rest_y: dict = {}
        self._next_object_id = 1

    def assign_ids(self, objects: List[TrackedObject]):
        for obj in objects:
            best_id, best_iou = None, 0.25
            for oid, prev_box in self._prev_boxes.items():
                if oid in {o.object_id for o in objects}:
                    continue
                iou = self._iou(obj.box, prev_box)
                if iou > best_iou:
                    best_id, best_iou = oid, iou
            if best_id is None:
                best_id = self._next_object_id
                self._next_object_id += 1
            obj.object_id = best_id
            self._prev_boxes[best_id] = obj.box

    @staticmethod
    def _iou(a: BBox, b: BBox) -> float:
        x0 = max(a.x, b.x)
        y0 = max(a.y, b.y)
        x1 = min(a.x + a.w, b.x + b.w)
        y1 = min(a.y + a.h, b.y + b.h)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        inter = (x1 - x0) * (y1 - y0)
        union = a.w * a.h + b.w * b.h - inter
        return inter / max(union, 1e-9)

    @staticmethod
    def _center(box: BBox) -> Tuple[float, float]:
        return box.x + box.w / 2, box.y + box.h / 2

    def evaluate(self, hands: List[Hand],
                 objects: List[TrackedObject]) -> List[Interaction]:
        """Annotate objects with held/near info and emit interactions."""
        interactions: List[Interaction] = []
        for obj in objects:
            ocx, ocy = self._center(obj.box)

            best_hand, best_dist = None, 1e9
            for hand in hands:
                hcx, hcy = self._center(hand.box)
                dist = ((ocx - hcx) ** 2 + (ocy - hcy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_hand = dist, hand

            contact = False
            for hand in hands:
                if self._iou(hand.box, obj.box) >= CONTACT_IOU:
                    contact = True
                    interactions.append(Interaction(
                        kind="contact", hand_id=hand.hand_id,
                        object_id=obj.object_id, object_label=obj.label,
                        strength=self._iou(hand.box, obj.box)))
            obj.nearest_hand_dist = best_dist if best_hand is not None else None

            # Lift: object center sits well above its lowest seen position.
            rest = self._rest_y.get(obj.object_id, ocy)
            self._rest_y[obj.object_id] = min(rest, ocy)
            lifted = rest - ocy > LIFT_DROP_FRACTION * max(obj.box.h, 0.01)

            if lifted and (contact or (best_hand is not None and best_dist < NEAR_DIST * 2)):
                interactions.append(Interaction(
                    kind="lift", hand_id=best_hand.hand_id if best_hand else -1,
                    object_id=obj.object_id, object_label=obj.label,
                    strength=1.0))

            if best_hand is not None:
                if contact or lifted:
                    obj.held_by_hand_id = best_hand.hand_id
                elif best_dist < NEAR_DIST:
                    interactions.append(Interaction(
                        kind="near", hand_id=best_hand.hand_id,
                        object_id=obj.object_id, object_label=obj.label,
                        strength=1.0 - best_dist / NEAR_DIST))

        return interactions


class PerceptionEngine:
    """Facade combining hand tracking, object detection and interactions."""

    def __init__(self):
        self.hands = HandTracker()
        self.detector = ObjectDetector()
        self.interactions = InteractionEngine()
        self.gestures = GestureRecognizer()
        self._fps = 0.0
        self._last_tick = time.time()
        # Protocol vocabulary (set by the server); color tagging activates
        # only when the protocol references red_box/yellow_box labels, and
        # gesture recognition only when it references gesture_* labels.
        self._vocabulary: set = set()

    def set_vocabulary(self, labels: List[str]):
        self._vocabulary = set(labels)

    def process(self, frame_bgr: np.ndarray) -> PerceptionFrame:
        h, w = frame_bgr.shape[:2]

        # FPS via exponential moving average.
        now = time.time()
        inst = 1.0 / max(now - self._last_tick, 1e-6)
        self._fps = inst if self._fps == 0 else 0.9 * self._fps + 0.1 * inst
        self._last_tick = now

        detected_hands = self.hands.process(frame_bgr)

        color_tagging = bool(self._vocabulary & {"red_box", "yellow_box"})
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV) if color_tagging else None

        raw_objects: List[TrackedObject] = []
        for label, conf, bbox in self.detector.detect(frame_bgr):
            if color_tagging and (label in ColorTagger.TAGGABLE
                                  or label.startswith(("red_", "yellow_"))):
                color = ColorTagger.classify(frame_hsv, bbox)
                if color in ("red", "yellow"):
                    label = f"{color}_box"
            raw_objects.append(TrackedObject(label=label, confidence=conf, box=bbox))

        # Hand gestures (only for gesture-driven protocols).
        if any(lbl.startswith("gesture_") for lbl in self._vocabulary):
            try:
                raw_objects.extend(self.gestures.process(detected_hands))
            except Exception:
                log.exception("Gesture recognition failed once - continuing")

        self.interactions.assign_ids(raw_objects)
        inter = self.interactions.evaluate(detected_hands, raw_objects)

        annotated = self._annotate(frame_bgr.copy(), detected_hands,
                                   raw_objects, inter)

        return PerceptionFrame(
            hands=detected_hands,
            objects=raw_objects,
            interactions=inter,
            annotated_bgr=annotated,
            fps=round(self._fps, 1),
            frame_w=w,
            frame_h=h,
        )

    # ------------------------------------------------------------- annotate

    def _annotate(self, img, hands: List[Hand], objects: List[TrackedObject],
                  inter: List[Interaction]) -> np.ndarray:
        h, w = img.shape[:2]
        held_ids = {i.object_id for i in inter if i.kind == "lift"}
        contact_ids = {i.object_id for i in inter if i.kind == "contact"}

        slots: List[Tuple[int, int, int, int]] = []  # occupied label chips

        def draw_label(x: int, y: int, text: str, color,
                       below: bool = False):
            """Draw text on a dark chip; stack downward if slots collide."""
            fscale = 0.46
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                          fscale, 1)
            pad = 4
            bw = tw + 2 * pad
            bh = th + 2 * pad
            # Objects: chip above the box; hands: chip below the box.
            by = y - bh - 2 if not below else y + 2
            bx = max(0, min(x, w - bw))
            slot = [bx, max(0, by), bx + bw, max(0, by) + bh]
            while any(not (slot[2] < s[0] or slot[0] > s[2]
                           or slot[3] < s[1] or slot[1] > s[3])
                      for s in slots):
                slot = [slot[0], slot[1] + bh + 2, slot[2], slot[3] + bh + 2]
            slots.append(tuple(slot))
            cv2.rectangle(img, (slot[0], slot[1]), (slot[2], slot[3]),
                          (16, 16, 22), -1)
            cv2.rectangle(img, (slot[0], slot[1]), (slot[2], slot[3]),
                          color, 1)
            cv2.putText(img, text, (slot[0] + pad, slot[3] - pad - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, fscale, color, 1,
                        cv2.LINE_AA)

        for obj in objects:
            x, y = int(obj.box.x * w), int(obj.box.y * h)
            bw, bh = int(obj.box.w * w), int(obj.box.h * h)
            color = COL_HELD if obj.object_id in held_ids else COL_OBJ
            cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 1)
            tag = (obj.label.replace("gesture_", "").upper()
                   + f" #{obj.object_id} {obj.confidence:.2f}")
            if obj.object_id in held_ids:
                tag += " HELD"
            elif obj.object_id in contact_ids:
                tag += " CONTACT"
            draw_label(x, y, tag, color)

        for hand in hands:
            x, y = int(hand.box.x * w), int(hand.box.y * h)
            bw, bh = int(hand.box.w * w), int(hand.box.h * h)
            cv2.rectangle(img, (x, y), (x + bw, y + bh), COL_HAND, 1)
            draw_label(x, y + bh, hand.label.upper(), COL_HAND, below=True)
            if hand.landmarks:
                # Light skeleton dots - small and muted so they never drown
                # out the labels.
                for lm in hand.landmarks:
                    cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 2,
                               (150, 90, 200), -1, cv2.LINE_AA)

        return img

    def close(self):
        self.hands.close()
        self.detector.close()
