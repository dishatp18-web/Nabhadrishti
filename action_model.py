"""
BAS Experiment Copilot - Temporal Action Recognition (the "AI proposes" layer)
===============================================================================
A lightweight GRU sequence model that classifies *which protocol step the
astronaut is currently performing* from a sliding window of perception
features. Its prediction never changes protocol state - the deterministic
FSM owns state. The AI output is used for:

  * surfacing "AI: <activity> (p=..%)" in the HUD, and
  * EARLY warning ("warns before the mistake"): when the AI confidently
    recognises an activity other than the FSM's expected step, the copilot
    raises a HOLD advisory + voice alert BEFORE the operator completes a
    wrong action.

Features (fixed-length 44-dim, per frame, all normalized):
  hands        : 2 x (x, y, w, h)                      ->  8
  per label (up to MAX_OBJECTS): vis, x, y, w, h, held ->  6 x MAX_OBJECTS = 36

Weights: backend/action_weights.pt (labels in action_labels.json).
Absent weights -> the watcher is inactive and the system runs purely on
the FSM (documented degradation, not a crash). Train it with
tools/collect_dataset.py + tools/train_action_model.py.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .models import TrackedObject

log = logging.getLogger("bas.action")

MAX_OBJECTS = 6
HAND_DIM = 8
OBJ_DIM = 6
FEATURE_DIM = HAND_DIM + OBJ_DIM * MAX_OBJECTS  # 44
SEQ_LEN = 16                                    # frames per inference window

WEIGHTS_PATH = Path(__file__).resolve().parent / "action_weights.pt"
LABELS_PATH = Path(__file__).resolve().parent / "action_labels.json"


def extract_features(hands, objects: List[TrackedObject],
                     vocabulary: Sequence[str]) -> np.ndarray:
    """Build the fixed-length feature vector for one frame.

    `hands` is a list of models.Hand; `objects` the tracked objects.
    """
    feats = np.zeros(FEATURE_DIM, dtype=np.float32)
    for i, hand in enumerate(hands[:2]):
        feats[i * 4 + 0] = hand.box.x
        feats[i * 4 + 1] = hand.box.y
        feats[i * 4 + 2] = hand.box.w
        feats[i * 4 + 3] = hand.box.h
    base = HAND_DIM
    for i, label in enumerate(vocabulary[:MAX_OBJECTS]):
        match = next((o for o in objects if o.label == label), None)
        if match is None:
            continue
        o = base + i * OBJ_DIM
        feats[o + 0] = 1.0                      # visible
        feats[o + 1] = match.box.x
        feats[o + 2] = match.box.y
        feats[o + 3] = match.box.w
        feats[o + 4] = match.box.h
        feats[o + 5] = 1.0 if match.held_by_hand_id is not None else 0.0
    return feats


class ActionWatcher:
    """Maintains the feature ring buffer and runs GRU inference when
    trained weights are available. Never raises into the pipeline."""

    def __init__(self, step_ids: List[str], infer_every: int = 4):
        self.labels: List[str] = ["other"] + list(step_ids)
        self.vocabulary_hint: List[str] = []
        self.infer_every = infer_every
        self._buffer: deque = deque(maxlen=SEQ_LEN)
        self._frame_count = 0
        self.action: Optional[str] = None
        self.confidence: float = 0.0
        self.available = False
        self._model = None
        self._load()

    def _load(self):
        if not (WEIGHTS_PATH.exists() and LABELS_PATH.exists()):
            log.info("Action model weights not found (%s) - AI action "
                     "recognition inactive; FSM-only mode.", WEIGHTS_PATH.name)
            return
        try:
            import torch
            try:
                self.labels = json.loads(LABELS_PATH.read_text())
            except Exception:
                pass
            self._model = _GRU(input_size=FEATURE_DIM,
                               hidden=48, num_classes=len(self.labels))
            self._model.load_state_dict(
                torch.load(WEIGHTS_PATH, map_location="cpu"))
            self._model.eval()
            self.available = True
            log.info("Action model loaded: %d classes, window=%d frames",
                     len(self.labels), SEQ_LEN)
        except Exception as exc:
            log.warning("Action model failed to load (%s) - FSM-only mode", exc)

    def observe(self, features: np.ndarray):
        """Push one frame of features; occasionally run inference."""
        self._buffer.append(features)
        self._frame_count += 1
        if not self.available or len(self._buffer) < SEQ_LEN:
            return
        if self._frame_count % self.infer_every != 0:
            return
        try:
            import torch
            window = np.stack(self._buffer)[-SEQ_LEN:]  # (T, F)
            x = torch.from_numpy(window).unsqueeze(0)   # (1, T, F)
            with torch.no_grad():
                probs = torch.softmax(self._model(x)[0], dim=-1)
            conf, idx = float(probs.max()), int(probs.argmax())
            label = self.labels[idx] if idx < len(self.labels) else "other"
            self.action, self.confidence = label, round(conf, 3)
        except Exception as exc:
            log.warning("Action inference failed once: %s", exc)

    @property
    def disagreeing(self) -> bool:
        """AI confidently recognises a different activity than 'other'-
        style noise; the server compares against the FSM's expected step."""
        return (self.available and self.action not in (None, "other")
                and self.confidence >= 0.70)


import torch.nn as nn  # noqa: E402  (kept last so torch stays lazy for FSM tests)


class _GRU(nn.Module):
    def __init__(self, input_size: int, hidden: int, num_classes: int):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))

    def forward(self, x):                       # x: (B, T, F)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])         # logits (B, C)
