"""
BAS Experiment Copilot - Gesture Recognition
=============================================
Turns MediaPipe's 21 hand landmarks into discrete protocol gestures:

  open_palm   all four fingers extended
  peace       index + middle extended (V sign / "two fingers")
  gun         index extended + thumb flag raised ("finger gun")
  pointing    index only, thumb tucked
  fist        all fingers curled
  thumbs_up   fist with thumb straight up
  wave        temporal: open palm oscillating side-to-side

Gestures are emitted as TrackedObjects (label "gesture_<name>", box = the
hand's box) so the FSM, HUD tracking list, and SVG overlay all work on
them with zero changes - a hand gesture is just another "object" the
protocol can demand.

Two noise guards keep idle hands from polluting the protocol:
  * persistence - a gesture must hold for PERSIST_S before it is emitted
    (kills single-frame flicker from relaxed hands);
  * wave hysteresis - the wave state lingers ~0.9 s so its dwell timer
    accumulates across brief tracking gaps.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import List, Optional, Tuple

from .models import Hand, TrackedObject

log = logging.getLogger("bas.gestures")


def _d(a, b) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


# MediaPipe hand landmark indices.
WRIST, THUMB_TIP, THUMB_IP = 0, 4, 3
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20
PINKY_MCP = 17


class GestureRecognizer:
    PERSIST_S = 0.35
    WAVE_WINDOW_S = 2.2
    WAVE_MIN_AMPLITUDE = 0.08
    WAVE_MIN_TURNS = 3

    def __init__(self):
        self._pending: dict = {}       # (hand_key, label) -> first seen ts
        self._wave_hist: dict = {}     # hand_key -> deque[(ts, x, open?)]
        self._wave_until: dict = {}    # hand_key -> ts wave stays active

    # ------------------------------------------------------------ public

    def process(self, hands: List[Hand]) -> List[TrackedObject]:
        now = time.time()
        out: List[TrackedObject] = []
        seen = set()

        for hand in hands:
            key = hand.hand_id
            label, conf = self.classify(hand)

            # Wave is temporal and evaluated from the x-trajectory of an
            # open hand; it supersedes the static open_palm label.
            hist = self._wave_hist.setdefault(key, deque(maxlen=64))
            hist.append((now, hand.box.x + hand.box.w / 2.0, label == "open_palm"))
            if self._wave_active(key, now):
                label, conf = "wave", max(conf, 0.80)

            if label is None:
                continue
            pkey = (key, label)
            seen.add(pkey)
            first = self._pending.setdefault(pkey, now)
            if now - first < self.PERSIST_S:
                continue
            out.append(TrackedObject(
                label=f"gesture_{label}",
                confidence=conf,
                box=hand.box.model_copy(),
            ))

        # Purge debounce entries for hands/gestures that disappeared.
        for k in list(self._pending):
            if k not in seen:
                del self._pending[k]
        for k in list(self._wave_hist):
            if not any(now - t < 1.0 for t, _, _ in self._wave_hist[k]):
                del self._wave_hist[k]
        return out

    # ------------------------------------------------------------ classify

    def classify(self, hand: Hand) -> Tuple[Optional[str], float]:
        lm = hand.landmarks
        if not lm or len(lm) < 21:
            return None, 0.0
        wrist = lm[WRIST]
        hand_size = _d(lm[WRIST], lm[MIDDLE_MCP])
        if hand_size < 1e-4:
            return None, 0.0

        def ext(tip: int, pip: int) -> bool:
            return _d(lm[tip], wrist) > _d(lm[pip], wrist) * 1.12

        idx = ext(INDEX_TIP, INDEX_PIP)
        mid = ext(MIDDLE_TIP, MIDDLE_PIP)
        rng = ext(RING_TIP, RING_PIP)
        pky = ext(PINKY_TIP, PINKY_PIP)
        n_ext = sum([idx, mid, rng, pky])

        # Thumb "raised" = tip far from the middle MCP relative to hand size
        # (tucked-in thumbs sit close; gun/thumbs-up thumbs stick out).
        thumb_up = _d(lm[THUMB_TIP], lm[MIDDLE_MCP]) > 0.55 * hand_size

        if n_ext >= 4:
            return "open_palm", 0.90
        if idx and mid and not rng and not pky:
            return "peace", 0.88
        if idx and not mid and not rng and not pky:
            return ("gun", 0.85) if thumb_up else ("pointing", 0.82)
        if n_ext == 0:
            if thumb_up and lm[THUMB_TIP].y < wrist.y - 0.04:
                return "thumbs_up", 0.85
            return "fist", 0.88
        return None, 0.0

    # ------------------------------------------------------------ wave

    def _wave_active(self, key, now: float) -> bool:
        hist = self._wave_hist.get(key)
        if not hist:
            return False
        recent = [x for (t, x, opened) in hist
                  if opened and now - t <= self.WAVE_WINDOW_S]
        if len(recent) >= 6:
            amp = max(recent) - min(recent)
            diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)
                     if abs(recent[i + 1] - recent[i]) > 0.015]
            turns = sum(1 for i in range(len(diffs) - 1)
                        if (diffs[i] > 0) != (diffs[i + 1] > 0))
            if amp >= self.WAVE_MIN_AMPLITUDE and turns >= self.WAVE_MIN_TURNS:
                self._wave_until[key] = now + 0.9   # hysteresis
        return now < self._wave_until.get(key, 0.0)
