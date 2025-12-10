# server/routers/incidents.py
from fastapi import APIRouter, HTTPException
import os
import json

router = APIRouter()
INCIDENT_ROOT = "incidents"


@router.get("/list")
def list_incidents():
    if not os.path.exists(INCIDENT_ROOT):
        return []
    entries = [d for d in os.listdir(INCIDENT_ROOT) if os.path.isdir(os.path.join(INCIDENT_ROOT, d))]
    entries.sort()
    return {"incidents": entries}


@router.get("/{incident_id}/evidence")
def get_evidence(incident_id: str):
    incident_dir = os.path.join(INCIDENT_ROOT, incident_id)
    if not os.path.exists(incident_dir):
        raise HTTPException(status_code=404, detail="Incident folder not found")

    # try canonical evidence_out/evidence.json, then evidence.json
    candidates = [
        os.path.join(incident_dir, "evidence_out", "evidence.json"),
        os.path.join(incident_dir, "evidence.json"),
        os.path.join(incident_dir, "evidence_out", "incident_report.json"),
        os.path.join(incident_dir, "incident_report.json")
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return json.load(open(p, "r", encoding="utf-8"))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load {p}: {e}")

    raise HTTPException(status_code=404, detail="No evidence file found for this incident")
