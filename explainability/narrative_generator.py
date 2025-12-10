"""
narrative_generator.py (Gemini 2.5 Edition — FINAL VERSION)

LLM-based forensic narrative generator for Hawkeye.
Outputs:
    - incident_narrative.json
    - incident_narrative.txt

Run:
    python -m narrative_generator incidents/incident_xxxxx --save
"""

import os
import json
import time
import argparse
from typing import Dict, Any, Optional, List

# -------------------------------------------------------------
# AUTO LOAD .env (so GEMINI_API_KEY works without terminal export)
# -------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------------------
# CONFIGURE GEMINI
# -------------------------------------------------------------
import google.generativeai as genai

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("ERROR: GEMINI_API_KEY not found. Add it to your .env file.")

genai.configure(api_key=API_KEY)

# ⭐ Use the correct model based on your SDK — you DO have this one
MODEL_NAME = "models/gemini-2.5-flash"


# -------------------------------------------------------------
# PROMPT TEMPLATES
# -------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a careful forensic analyst. "
    "You must output ONLY valid JSON. No text before or after. "
    "Avoid hallucinations. Be conservative. Clearly express uncertainties."
)

INSTRUCTION_TEMPLATE = """
Incident Summary
----------------
Incident ID: {incident_id}
Start Time: {start_time}
End Time: {end_time}
Frame Count: {frame_count}

Keyframe: {keyframe}
Overlay Keyframe: {overlay_keyframe}

Persons:
{persons_summary}

Audio:
{audio_summary}

=====================
REQUIRED JSON FORMAT:
=====================

{{
  "headline": "One sentence.",
  "short_summary": "2–3 sentence overview.",
  "detailed_narrative": "Multi-paragraph, chronological, forensic explanation.",
  "evidence_stack": [
    {{
      "type": "pose | heatmap | audio | motion",
      "timestamp": 0.0,
      "person_id": "P1",
      "short_desc": "description",
      "confidence_score": 0.0,
      "file_refs": ["optional_filename.png"]
    }}
  ],
  "uncertainty": ["unknowns or confidence limits"],
  "suggested_actions": ["step1", "step2", "step3"]
}}

RULES:
- Output ONLY JSON.
- No markdown.
- No commentary.
- Do not invent details.
- If unknown, state it under 'uncertainty'.
"""


# -------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------
def _short_person_summary(persons: Dict[str, Dict[str, Any]]) -> str:
    if not persons:
        return "None"

    lines = []
    for pid, p in persons.items():
        rt = p.get("risk_timeline", [])
        peak = max(rt, key=lambda x: x[1]) if rt else (None, None)
        lines.append(
            f"- {pid}: risk_peak={peak[1]} at {peak[0]}s, "
            f"last_bbox={p.get('last_bbox')}, "
            f"keypoints={p.get('last_keypoints_count')}"
        )
    return "\n".join(lines)


def _audio_summary(events: Optional[List[Dict[str, Any]]]):
    if not events:
        return "None"
    return "\n".join([f"- {e['type']} at {e['time']}s (score={e['score']})" for e in events])


def _load_incident_report(incident_dir: str):
    """
    Attempts to load:
        - incident_report.json
        - evidence_out/incident_report.json
        - evidence.json (fallback)
    """
    candidates = [
        os.path.join(incident_dir, "incident_report.json"),
        os.path.join(incident_dir, "evidence_out", "incident_report.json")
    ]

    for p in candidates:
        if os.path.exists(p):
            return json.load(open(p))

    ev_path = os.path.join(incident_dir, "evidence.json")
    if os.path.exists(ev_path):
        frames = json.load(open(ev_path))
        return {
            "incident_dir": incident_dir,
            "frame_count": len(frames),
            "start_time": frames[0]["timestamp"],
            "end_time": frames[-1]["timestamp"],
            "per_person": {}
        }

    raise FileNotFoundError("incident_report.json or evidence.json not found.")


def _extract_persons(report):
    persons = report.get("per_person") or {}
    out = {}

    for pid, p in persons.items():
        out[pid] = {
            "risk_timeline": p.get("risk_timeline", []),
            "last_bbox": p.get("last_bbox"),
            "last_keypoints_count": p.get("last_keypoints_count", 0)
        }
    return out


# -------------------------------------------------------------
# GEMINI CALL
# -------------------------------------------------------------
def _call_gemini(prompt_text: str) -> str:
    model = genai.GenerativeModel(MODEL_NAME)

    response = model.generate_content(
        SYSTEM_PROMPT + "\n\n" + prompt_text,
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": 2000
        }
    )

    return response.text


# -------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------
def generate_narrative_from_report(incident_dir: str, save_output=True):
    report = _load_incident_report(incident_dir)

    persons = _extract_persons(report)
    prompt = INSTRUCTION_TEMPLATE.format(
        incident_id=os.path.basename(incident_dir),
        start_time=report.get("start_time"),
        end_time=report.get("end_time"),
        frame_count=report.get("frame_count"),
        keyframe=report.get("keyframe"),
        overlay_keyframe=report.get("overlay_keyframe"),
        persons_summary=_short_person_summary(persons),
        audio_summary=_audio_summary(report.get("audio_events", []))
    )

    raw = _call_gemini(prompt)

    # Try strict JSON parsing
    try:
        parsed = json.loads(raw)
    except:
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except:
                parsed = None
        else:
            parsed = None

    if parsed is None:
        parsed = {
            "headline": "Parsing error",
            "short_summary": "",
            "detailed_narrative": raw,
            "evidence_stack": [],
            "uncertainty": ["Could not parse JSON output"],
            "suggested_actions": []
        }

    result = {
        "incident_id": os.path.basename(incident_dir),
        "generated_at": time.time(),
        "raw_text": raw,
        "parsed": parsed
    }

    if save_output:
        out_json = os.path.join(incident_dir, "incident_narrative.json")
        out_txt = os.path.join(incident_dir, "incident_narrative.txt")

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(parsed.get("detailed_narrative", raw))

    return result


# -------------------------------------------------------------
# CLI ENTRY
# -------------------------------------------------------------
def _cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("incident_dir")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    print("Generating narrative using Gemini 2.5 Flash…")
    res = generate_narrative_from_report(args.incident_dir, args.save)

    print("Done. Headline:", res["parsed"].get("headline"))


if __name__ == "__main__":
    _cli_main()
