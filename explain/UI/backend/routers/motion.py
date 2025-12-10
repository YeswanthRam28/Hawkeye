
from fastapi import APIRouter
from backend.mock_state import get_state

router = APIRouter(prefix="/motion", tags=["Motion"])

@router.get("/crowd-analysis")
async def crowd_analysis():
    state = get_state()
    return {
        "timestamp": state.motion_data["timestamp"],
        "crowd_density": state.motion_data["crowd_density"],
        "surge_detected": state.motion_data["surge_detected"],
        "surge_direction": state.motion_data["surge_direction"],
        "panic_score": state.motion_data["panic_score"]
    }

@router.get("/motion-vectors")
async def motion_vectors():
    state = get_state()
    return {
        "timestamp": state.motion_data["timestamp"],
        "optical_flow_vectors": state.motion_data["optical_flow_vectors"],
        "velocity_map": state.motion_data["velocity_map"],
        "avg_speed": state.motion_data["avg_speed"],
        "anomaly_regions": state.motion_data["anomaly_regions"]
    }

@router.get("/events")
async def events():
    state = get_state()
    events_list = []
    if state.motion_data["surge_detected"]:
        events_list.append({"type": "crowd_surge", "confidence": 0.85})
    if state.motion_data["panic_score"] > 0.7:
        events_list.append({"type": "panic_behavior", "confidence": 0.8})
        
    return {
        "timestamp": state.motion_data["timestamp"],
        "events": events_list
    }
