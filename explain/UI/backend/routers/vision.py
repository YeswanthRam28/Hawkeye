
from fastapi import APIRouter
from backend.mock_state import get_state

router = APIRouter(prefix="/vision", tags=["Vision"])

@router.get("/frame-analysis")
async def frame_analysis():
    state = get_state()
    return {
        "timestamp": state.vision_data["timestamp"],
        "objects": state.vision_data["objects"],
        "poses": state.vision_data["poses"],
        "vision_risk_factors": state.vision_data["vision_risk_factors"]
    }

@router.get("/raw-features")
async def raw_features():
    state = get_state()
    return {
        "timestamp": state.vision_data["timestamp"],
        "raw_heatmap": state.vision_data["raw_heatmap"],
        "pose_map": state.vision_data["pose_map"],
        "object_scores_matrix": state.vision_data["object_scores_matrix"]
    }

@router.get("/events")
async def events():
    state = get_state()
    # Convert vision risk factors to event list for consistency
    events_list = []
    if state.vision_data["vision_risk_factors"]["weapon_detected"]:
        events_list.append({"type": "weapon", "confidence": 0.95})
    if state.vision_data["vision_risk_factors"]["fall_detected"]:
        events_list.append({"type": "fall", "confidence": 0.88})
        
    return {
        "timestamp": state.vision_data["timestamp"],
        "events": events_list
    }
