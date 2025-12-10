# server/routers/trigger.py
from fastapi import APIRouter
from explain.replay.engine import ReplayEngine
import time

router = APIRouter(prefix="/trigger", tags=["trigger"])

# GLOBAL ENGINE
engine = ReplayEngine(pre_seconds=5, post_seconds=5)

@router.post("/replay")
def trigger_replay():
    try:
        print("[Trigger] Manual trigger activated.")

        # Internally record the trigger moment
        incident = engine.trigger()

        # Build the incident and return metadata
        result = engine.export_incident()

        return {
            "success": True,
            "message": "Replay captured",
            "incident": result
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
