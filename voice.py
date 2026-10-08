"""
BAS Experiment Copilot - Offline Voice Alerts
==============================================
Voice-based alerts (explicit SIH 26174 requirement: "It should be a voice
based alert") using offline TTS only - no cloud calls, works on an
air-gapped edge box.

Windows: SAPI5 via pyttsx3. Linux: espeak via pyttsx3. If TTS is entirely
unavailable the engine degrades to console beeps (print + terminal bell)
so the copilot is never silent, and logs the degradation once.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

log = logging.getLogger("bas.voice")


class VoiceEngine:
    """Non-blocking offline TTS. speak() never blocks the vision pipeline."""

    def __init__(self, enabled: bool = True, rate: int = 172):
        self.enabled = enabled
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self.mode = "unavailable"
        self._engine = None

        if enabled:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", rate)
                self._engine.setProperty("volume", 1.0)
                self.mode = "pyttsx3-offline"
            except Exception as exc:
                log.warning("Offline TTS unavailable (%s) - falling back to "
                            "console beeps", exc)
                self.mode = "beep-fallback"

        self._thread = threading.Thread(target=self._loop, name="bas-voice",
                                        daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ public

    def speak(self, text: str, interrupt: bool = False):
        """Queue a phrase. interrupt=True cancels pending speech (urgent)."""
        if not self.enabled:
            return
        if interrupt:
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
        self._q.put(text)

    def alert(self, level: str, code: str, detail: str = ""):
        """Map a copilot event to a spoken phrase."""
        phrases = {
            "STEP_COMPLETE": f"Step complete. {detail}" if detail else "Step complete.",
            "NEXT_STEP": f"Next step: {detail}",
            "SKIPPED_STEP": f"Warning. Sequence violation. {detail}",
            "CONTAMINATION": f"Warning. Contamination risk. {detail}",
            "OBJECT_MISSING": "Caution. Target object not in view.",
            "STALLED": f"Caution. Step {detail} is taking too long." if detail
                       else "Caution. Step is taking too long.",
            "HOLD": f"Hold. {detail}" if detail else "Hold. Low confidence. Verify posture.",
            "PROTOCOL_COMPLETE": "Experiment sequence complete. Well done.",
        }
        phrase = phrases.get(code)
        if phrase:
            self.speak(phrase, interrupt=(level == "critical"))

    def close(self):
        self._q.put(None)

    # ------------------------------------------------------------ worker

    def _loop(self):
        while True:
            text = self._q.get()
            if text is None:
                break
            try:
                if self._engine is not None:
                    self._engine.say(text)
                    self._engine.runAndWait()
                else:
                    print(f"\a[VOICE] {text}", flush=True)
            except Exception as exc:
                log.warning("TTS speak failed: %s", exc)
