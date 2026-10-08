"""
BAS Experiment Copilot - Structured Audit Trail
================================================
SIH 26174 requirement: "generate a timestamped and structured lightweight
text file of the conducted steps with outcomes/status."

Produces, per session (under recordings/<session>/):
  * audit.jsonl - one JSON object per event (machine-readable evidence)
  * report.txt  - human-readable timestamped step log with outcome/status

Every event carries UTC time, step, confidence and reason - the
"reproducible evidence" line from the idea deck.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("bas.audit")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class AuditLogger:
    def __init__(self, session_dir: Path):
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self.dir / "audit.jsonl"
        self._report_path = self.dir / "report.txt"
        self._lock = threading.Lock()
        self._events: list = []
        self._steps: list = []

    # ------------------------------------------------------------ events

    def log(self, event: str, **fields):
        """Append one structured event to audit.jsonl."""
        record = {"ts": utcnow(), "event": event, **fields}
        with self._lock:
            self._events.append(record)
            with self._jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.info("AUDIT %s %s", event, fields.get("detail", ""))

    def step_complete(self, index: int, step_id: str, name: str,
                      confidence: float, duration_s: float):
        self.log("STEP_COMPLETE", step=index + 1, step_id=step_id,
                 name=name, status="OK", confidence=round(confidence, 3),
                 duration_s=round(duration_s, 1),
                 detail=f"Step {index + 1} '{name}' verified")
        with self._lock:
            self._steps.append({
                "step": index + 1, "id": step_id, "name": name,
                "status": "COMPLETE", "confidence": round(confidence, 3),
                "ts": utcnow(), "duration_s": round(duration_s, 1),
            })
        self._write_report()

    def _write_report(self):
        """Rewrite the lightweight text report (cheap: steps only)."""
        lines = [
            "=" * 74,
            "NABHADRISHTI · BAS PAYLOAD OPERATIONS COPILOT - EXECUTION REPORT",
            f"Session : {self.dir.name}",
            f"Generated (UTC): {utcnow()}",
            "=" * 74,
            "",
            f"{'UTC TIME':<24} {'STEP':<6} {'ID':<18} {'STATUS':<10} "
            f"{'CONF':<7} {'DUR(s)':<7}",
            "-" * 74,
        ]
        if not self._steps:
            lines.append("  (Awaiting first verified step transition...)")
        else:
            for s in self._steps:
                lines.append(
                    f"{s['ts']:<24} {s['step']:<6} {s['id']:<18} "
                    f"{s['status']:<10} {s['confidence']:<7} {s['duration_s']:<7}")
        lines += ["", "-" * 74, "ANOMALY LOG", "-" * 74]
        anomalies = [e for e in self._events
                     if e.get("level") in ("warning", "critical")]
        if not anomalies:
            lines.append("(none)")
        else:
            for e in anomalies:
                lines.append(f"{e['ts']:<24} [{e.get('code', e['event'])}] "
                             f"{e.get('detail', '')}")
        lines += ["", "END OF REPORT"]
        with self._lock:
            self._report_path.write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------ access

    def report_text(self) -> str:
        with self._lock:
            if not self._report_path.exists():
                self._write_report()
            return self._report_path.read_text(encoding="utf-8")

    def events_jsonl(self) -> str:
        with self._lock:
            if self._jsonl_path.exists():
                return self._jsonl_path.read_text(encoding="utf-8")
            return ""
