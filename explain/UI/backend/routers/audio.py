
from fastapi import APIRouter
from backend.mock_state import get_state

router = APIRouter(prefix="/audio", tags=["Audio"])

@router.get("/audio-analysis")
async def audio_analysis():
    state = get_state()
    return {
        "timestamp": state.audio_data["timestamp"],
        "events": state.audio_data["events"],
        "audio_risk_score": state.audio_data["audio_risk_score"]
    }

@router.get("/raw-features")
async def raw_features():
    state = get_state()
    return {
        "timestamp": state.audio_data["timestamp"],
        "spectrogram": state.audio_data["spectrogram"],
        "mfcc": state.audio_data["mfcc"],
        "peaks": state.audio_data["peaks"]
    }

@router.get("/events")
async def events():
    state = get_state()
    # Map internal events to the events endpoint format if different, 
    # but here they are similar. The user example showed "explosion" etc.
    # We'll just use the generated events.
    formatted_events = []
    for evt in state.audio_data["events"]:
        formatted_events.append({"type": evt["label"], "confidence": evt["confidence"]})

    return {
        "timestamp": state.audio_data["timestamp"],
        "events": formatted_events
    }
