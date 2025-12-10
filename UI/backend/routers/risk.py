
from fastapi import APIRouter
from backend.mock_state import get_state

router = APIRouter(prefix="/risk", tags=["Risk Engine"])

@router.get("/status")
async def risk_status():
    state = get_state()
    return {
        "timestamp": state.last_update,
        "overall_risk_score": state.risk_score,
        "contributors": {
            "vision": state.vision_data["vision_risk_factors"],
            "audio": state.audio_data["audio_risk_score"],
            "motion": state.motion_data["panic_score"]
        }
    }
