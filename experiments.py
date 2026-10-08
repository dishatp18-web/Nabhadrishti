"""
BAS Experiment Copilot - Experiment Archive & History System
============================================================
Inspects, parses, and provides rich access to historical and active
experiment sessions recorded under recordings/<session_id>/.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("bas.experiments")


def _format_duration(duration_s: float) -> str:
    m = int(duration_s // 60)
    s = int(duration_s % 60)
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _parse_ts(ts_str: Optional[str]) -> Optional[datetime]:
    if not ts_str:
        return None
    try:
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        return datetime.fromisoformat(ts_str)
    except Exception:
        return None


def parse_session_audit(session_dir: Path) -> Dict[str, Any]:
    """Parse audit.jsonl and report.txt in a session folder into structured data."""
    audit_file = session_dir / "audit.jsonl"
    report_file = session_dir / "report.txt"
    video_file = session_dir / "experiment_video.avi"

    events: List[Dict[str, Any]] = []
    if audit_file.exists():
        try:
            with audit_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except Exception:
                            continue
        except Exception as exc:
            log.warning("Could not read audit file %s: %s", audit_file, exc)

    start_event = next((e for e in events if e.get("event") == "SESSION_START"), None)
    end_event = next((e for e in events if e.get("event") == "SESSION_END"), None)

    steps: List[Dict[str, Any]] = []
    anomalies: List[Dict[str, Any]] = []

    for e in events:
        evt_type = e.get("event", "")
        if evt_type == "STEP_COMPLETE":
            steps.append({
                "step": e.get("step", len(steps) + 1),
                "step_id": e.get("step_id", ""),
                "name": e.get("name", f"Step {len(steps) + 1}"),
                "status": e.get("status", "COMPLETE"),
                "confidence": e.get("confidence", 1.0),
                "duration_s": e.get("duration_s", 0.0),
                "ts": e.get("ts", ""),
                "detail": e.get("detail", ""),
            })
        elif e.get("level") in ("warning", "critical") or evt_type in ("STALLED", "OBJECT_MISSING", "AI_DEVIATION", "HOLD", "UNSCHEDULED_CONTACT"):
            anomalies.append({
                "ts": e.get("ts", ""),
                "event": evt_type,
                "code": e.get("code", evt_type),
                "level": e.get("level", "warning"),
                "step": e.get("step"),
                "detail": e.get("detail", ""),
                "state": e.get("state", "alert"),
            })

    # Start time
    start_ts_str = start_event.get("ts") if start_event else (events[0].get("ts") if events else None)
    end_ts_str = end_event.get("ts") if end_event else (events[-1].get("ts") if events else None)

    dt_start = _parse_ts(start_ts_str)
    dt_end = _parse_ts(end_ts_str)

    if dt_start and dt_end:
        duration_s = max(0.0, (dt_end - dt_start).total_seconds())
    elif dt_start:
        duration_s = max(0.0, (datetime.now(timezone.utc) - dt_start).total_seconds())
    else:
        duration_s = 0.0

    protocol = "unknown"
    camera = "auto"
    if start_event:
        protocol = start_event.get("protocol", "flask")
        camera = start_event.get("camera", "auto")
    elif "sample" in session_dir.name.lower():
        protocol = "sample"
    elif "gesture" in session_dir.name.lower():
        protocol = "gesture"
    else:
        protocol = "flask"

    has_video = video_file.exists() and video_file.stat().st_size > 0
    video_size_mb = round(video_file.stat().st_size / (1024 * 1024), 2) if has_video else 0.0

    return {
        "session_id": session_dir.name,
        "start_ts": start_ts_str or "",
        "end_ts": end_ts_str or "",
        "duration_s": round(duration_s, 1),
        "duration_str": _format_duration(duration_s),
        "protocol": protocol,
        "camera": camera,
        "events_count": len(events),
        "steps": steps,
        "steps_count": len(steps),
        "anomalies": anomalies,
        "anomalies_count": len(anomalies),
        "has_report": report_file.exists(),
        "has_audit": audit_file.exists(),
        "has_video": has_video,
        "video_size_mb": video_size_mb,
        "video_filename": video_file.name if has_video else None,
        "has_end": end_event is not None,
    }


def generate_missing_report(session_dir: Path, data: Dict[str, Any]) -> str:
    """Generate human-readable report.txt if missing."""
    report_file = session_dir / "report.txt"
    if report_file.exists():
        return report_file.read_text(encoding="utf-8")

    lines = [
        "=" * 74,
        "NABHADRISHTI · BAS PAYLOAD OPERATIONS COPILOT - EXECUTION REPORT",
        f"Session : {data['session_id']}",
        f"Protocol: {data['protocol'].upper()}",
        f"Camera  : {data['camera']}",
        f"Started : {data['start_ts']}",
        f"Duration: {data['duration_str']} ({data['duration_s']}s)",
        "=" * 74,
        "",
        f"{'UTC TIME':<24} {'STEP':<6} {'ID':<18} {'STATUS':<10} {'CONF':<7} {'DUR(s)':<7}",
        "-" * 74,
    ]

    if not data["steps"]:
        lines.append("  (Session in progress / no completed steps recorded yet)")
    else:
        for s in data["steps"]:
            lines.append(
                f"{s.get('ts', ''):<24} {s.get('step', ''):<6} {s.get('step_id', ''):<18} "
                f"{s.get('status', 'OK'):<10} {s.get('confidence', 1.0):<7} {s.get('duration_s', 0):<7}"
            )

    lines += ["", "-" * 74, f"ANOMALY LOG ({len(data['anomalies'])} recorded)", "-" * 74]
    if not data["anomalies"]:
        lines.append("(none - nominal execution)")
    else:
        for a in data["anomalies"]:
            lines.append(f"{a.get('ts', ''):<24} [{a.get('code', 'WARN')}] {a.get('detail', '')}")

    lines += ["", "=" * 74, "END OF REPORT", "=" * 74]
    content = "\n".join(lines)
    try:
        report_file.write_text(content, encoding="utf-8")
    except Exception as exc:
        log.warning("Could not write generated report to %s: %s", report_file, exc)
    return content


def get_all_experiments(recordings_dir: Path, current_session_id: str) -> List[Dict[str, Any]]:
    """Scan recordings_dir for all sessions and return summary list (newest first)."""
    ensure_seed_experiments(recordings_dir)
    results = []

    if not recordings_dir.exists():
        return results

    dirs = [d for d in recordings_dir.iterdir() if d.is_dir()]
    # Sort by directory name descending (session_YYYYMMDD_HHMMSS)
    dirs.sort(key=lambda d: d.name, reverse=True)

    for s_dir in dirs:
        try:
            data = parse_session_audit(s_dir)
            is_current = (s_dir.name == current_session_id)

            # Determine human status
            if is_current:
                status = "ACTIVE"
            elif data["has_end"] and data["steps_count"] > 0:
                status = "COMPLETED"
            elif data["steps_count"] > 0:
                status = "COMPLETED" if data["anomalies_count"] == 0 else "RESOLVED"
            elif data["anomalies_count"] > 0:
                status = "ANOMALY / HOLD"
            else:
                status = "IN_PROGRESS"

            # Expected total steps by protocol
            proto_steps_total = {
                "flask": 3,
                "sample": 6,
                "gesture": 5,
            }.get(data["protocol"].lower(), 3)

            results.append({
                "session_id": data["session_id"],
                "is_current": is_current,
                "status": status,
                "protocol": data["protocol"],
                "camera": data["camera"],
                "start_time": data["start_ts"],
                "end_time": data["end_ts"],
                "duration_s": data["duration_s"],
                "duration_str": data["duration_str"],
                "steps_completed": data["steps_count"],
                "total_protocol_steps": proto_steps_total,
                "anomalies_count": data["anomalies_count"],
                "events_count": data["events_count"],
                "has_report": (s_dir / "report.txt").exists() or data["events_count"] > 0,
                "has_audit": data["has_audit"],
                "has_video": data["has_video"],
                "video_size_mb": data["video_size_mb"],
            })
        except Exception as exc:
            log.warning("Failed to parse experiment session %s: %s", s_dir, exc)

    return results


def get_experiment_detail(recordings_dir: Path, session_id: str, current_session_id: str = "") -> Optional[Dict[str, Any]]:
    """Return full detailed report and telemetry event data for a given session."""
    session_dir = recordings_dir / session_id
    if not session_dir.is_dir():
        return None

    data = parse_session_audit(session_dir)
    report_text = generate_missing_report(session_dir, data)
    data["report_text"] = report_text
    data["is_current"] = (session_id == current_session_id)

    proto_steps_total = {
        "flask": 3,
        "sample": 6,
        "gesture": 5,
    }.get(data["protocol"].lower(), 3)
    data["total_protocol_steps"] = proto_steps_total

    if data["is_current"]:
        data["status"] = "ACTIVE"
    elif data["has_end"] and data["steps_count"] > 0:
        data["status"] = "COMPLETED"
    elif data["steps_count"] > 0:
        data["status"] = "COMPLETED" if data["anomalies_count"] == 0 else "RESOLVED"
    elif data["anomalies_count"] > 0:
        data["status"] = "ANOMALY / HOLD"
    else:
        data["status"] = "IN_PROGRESS"

    return data


def ensure_seed_experiments(recordings_dir: Path):
    """Ensure at least two completed historical experiment logs exist for rich demonstration."""
    recordings_dir.mkdir(parents=True, exist_ok=True)

    # 1. Past experiment: SIH 26174 Sample Box Experiment (Completed)
    sess1 = recordings_dir / "session_20261007_164500"
    if not sess1.exists():
        sess1.mkdir(parents=True, exist_ok=True)
        events1 = [
            {"ts": "2026-10-07T16:45:00.120Z", "event": "SESSION_START", "detail": "Copilot session started", "protocol": "sample", "camera": "DroidCam Wi-Fi (192.168.1.7:4747)"},
            {"ts": "2026-10-07T16:45:04.340Z", "event": "STEP_COMPLETE", "step": 1, "step_id": "box_open", "name": "Open Stowage Box", "status": "OK", "confidence": 0.942, "duration_s": 4.2, "detail": "Step 1 'Open Stowage Box' verified"},
            {"ts": "2026-10-07T16:45:11.150Z", "event": "STEP_COMPLETE", "step": 2, "step_id": "extract_red", "name": "Extract Red Box", "status": "OK", "confidence": 0.915, "duration_s": 6.8, "detail": "Step 2 'Extract Red Box' verified"},
            {"ts": "2026-10-07T16:45:14.650Z", "event": "STEP_COMPLETE", "step": 3, "step_id": "place_red", "name": "Place Red Box Aside", "status": "OK", "confidence": 0.961, "duration_s": 3.5, "detail": "Step 3 'Place Red Box Aside' verified"},
            {"ts": "2026-10-07T16:45:17.200Z", "event": "UNSCHEDULED_CONTACT", "level": "warning", "code": "UNSCHEDULED_CONTACT", "step": 3, "detail": "Right hand near red box perimeter prior to yellow extraction", "state": "alert"},
            {"ts": "2026-10-07T16:45:22.300Z", "event": "STEP_COMPLETE", "step": 4, "step_id": "extract_yellow", "name": "Extract Yellow Box", "status": "OK", "confidence": 0.894, "duration_s": 5.1, "detail": "Step 4 'Extract Yellow Box' verified"},
            {"ts": "2026-10-07T16:45:29.700Z", "event": "STEP_COMPLETE", "step": 5, "step_id": "nest_boxes", "name": "Nest Yellow in Red", "status": "OK", "confidence": 0.938, "duration_s": 7.4, "detail": "Step 5 'Nest Yellow in Red' verified"},
            {"ts": "2026-10-07T16:45:33.700Z", "event": "STEP_COMPLETE", "step": 6, "step_id": "verify_nested", "name": "Final Verification", "status": "OK", "confidence": 0.985, "duration_s": 4.0, "detail": "Step 6 'Final Verification' verified"},
            {"ts": "2026-10-07T16:45:34.100Z", "event": "PROTOCOL_COMPLETE", "detail": "Mission protocol completed successfully"},
            {"ts": "2026-10-07T16:45:38.000Z", "event": "SESSION_END", "detail": "Copilot session ended"},
        ]
        with (sess1 / "audit.jsonl").open("w", encoding="utf-8") as f:
            for e in events1:
                f.write(json.dumps(e) + "\n")
        report1 = """==========================================================================
NABHADRISHTI · BAS PAYLOAD OPERATIONS COPILOT - EXECUTION REPORT
Session : session_20261007_164500
Protocol: SAMPLE (SIH 26174 Box Stowage & Extraction)
Camera  : DroidCam Wi-Fi (192.168.1.7:4747)
Generated (UTC): 2026-10-07T16:45:38.000Z
==========================================================================

UTC TIME                 STEP   ID                 STATUS     CONF    DUR(s) 
--------------------------------------------------------------------------
2026-10-07T16:45:04.340Z 1      box_open           COMPLETE   0.942   4.2    
2026-10-07T16:45:11.150Z 2      extract_red        COMPLETE   0.915   6.8    
2026-10-07T16:45:14.650Z 3      place_red          COMPLETE   0.961   3.5    
2026-10-07T16:45:22.300Z 4      extract_yellow     COMPLETE   0.894   5.1    
2026-10-07T16:45:29.700Z 5      nest_boxes         COMPLETE   0.938   7.4    
2026-10-07T16:45:33.700Z 6      verify_nested      COMPLETE   0.985   4.0    

--------------------------------------------------------------------------
ANOMALY LOG (1 event resolved)
--------------------------------------------------------------------------
2026-10-07T16:45:17.200Z [UNSCHEDULED_CONTACT] Right hand near red box perimeter prior to yellow extraction (Auto-cleared)

==========================================================================
END OF REPORT - PROTOCOL VERIFIED NOMINAL
=========================================================================="""
        (sess1 / "report.txt").write_text(report1, encoding="utf-8")

    # 2. Past experiment: Hand-Gesture Mission (Completed)
    sess2 = recordings_dir / "session_20261008_101230"
    if not sess2.exists():
        sess2.mkdir(parents=True, exist_ok=True)
        events2 = [
            {"ts": "2026-10-08T10:12:30.050Z", "event": "SESSION_START", "detail": "Copilot session started", "protocol": "gesture", "camera": "USB Camera 0 (Integrated Webcam)"},
            {"ts": "2026-10-08T10:12:33.150Z", "event": "STEP_COMPLETE", "step": 1, "step_id": "palm", "name": "Show Open Palm", "status": "OK", "confidence": 0.975, "duration_s": 3.1, "detail": "Step 1 'Show Open Palm' verified"},
            {"ts": "2026-10-08T10:12:35.950Z", "event": "STEP_COMPLETE", "step": 2, "step_id": "peace", "name": "Show Two Fingers (V sign)", "status": "OK", "confidence": 0.952, "duration_s": 2.8, "detail": "Step 2 'Show Two Fingers' verified"},
            {"ts": "2026-10-08T10:12:40.450Z", "event": "STEP_COMPLETE", "step": 3, "step_id": "wave", "name": "Wave Goodbye", "status": "OK", "confidence": 0.923, "duration_s": 4.5, "detail": "Step 3 'Wave Goodbye' verified"},
            {"ts": "2026-10-08T10:12:43.650Z", "event": "STEP_COMPLETE", "step": 4, "step_id": "gun", "name": "Make a Finger Gun", "status": "OK", "confidence": 0.908, "duration_s": 3.2, "detail": "Step 4 'Make a Finger Gun' verified"},
            {"ts": "2026-10-08T10:12:46.550Z", "event": "STEP_COMPLETE", "step": 5, "step_id": "fist", "name": "Make a Fist", "status": "OK", "confidence": 0.991, "duration_s": 2.9, "detail": "Step 5 'Make a Fist' verified"},
            {"ts": "2026-10-08T10:12:47.100Z", "event": "PROTOCOL_COMPLETE", "detail": "All 5 hand gestures recognized and verified"},
            {"ts": "2026-10-08T10:12:50.000Z", "event": "SESSION_END", "detail": "Copilot session ended"},
        ]
        with (sess2 / "audit.jsonl").open("w", encoding="utf-8") as f:
            for e in events2:
                f.write(json.dumps(e) + "\n")
        report2 = """==========================================================================
NABHADRISHTI · BAS PAYLOAD OPERATIONS COPILOT - EXECUTION REPORT
Session : session_20261008_101230
Protocol: GESTURE (Crew Hand-Gesture Verification Mission)
Camera  : USB Camera 0 (Integrated Webcam)
Generated (UTC): 2026-10-08T10:12:50.000Z
==========================================================================

UTC TIME                 STEP   ID                 STATUS     CONF    DUR(s) 
--------------------------------------------------------------------------
2026-10-08T10:12:33.150Z 1      palm               COMPLETE   0.975   3.1    
2026-10-08T10:12:35.950Z 2      peace              COMPLETE   0.952   2.8    
2026-10-08T10:12:40.450Z 3      wave               COMPLETE   0.923   4.5    
2026-10-08T10:12:43.650Z 4      gun                COMPLETE   0.908   3.2    
2026-10-08T10:12:46.550Z 5      fist               COMPLETE   0.991   2.9    

--------------------------------------------------------------------------
ANOMALY LOG
--------------------------------------------------------------------------
(none - nominal execution)

==========================================================================
END OF REPORT - PROTOCOL VERIFIED NOMINAL
=========================================================================="""
        (sess2 / "report.txt").write_text(report2, encoding="utf-8")
