# server/routers/narrative.py
from fastapi import APIRouter, HTTPException
import os

router = APIRouter()
INCIDENT_ROOT = "incidents"


@router.get("/{incident_id}")
def get_narrative(incident_id: str):
    incident_dir = os.path.join(INCIDENT_ROOT, incident_id)
    if not os.path.exists(incident_dir):
        raise HTTPException(status_code=404, detail="Incident folder not found")

    # Try to load an existing narrative file first
    narrative_json = os.path.join(incident_dir, "incident_narrative.json")
    if os.path.exists(narrative_json):
        import json
        try:
            return json.load(open(narrative_json, "r", encoding="utf-8"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read {narrative_json}: {e}")

    # Otherwise generate a new narrative using your narrative_generator
    try:
        from narrative_generator import generate_narrative_from_report
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"narrative_generator not available: {e}")

    try:
        result = generate_narrative_from_report(incident_dir, save_output=True)
        return result.get("parsed", result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate narrative: {e}")
