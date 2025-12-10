
# 🔌 Hawkeye Integration Guide

**For: Members 1, 2, 3 (Vision, Audio, Motion Engineers)** 
**From: Member 6 (Platform Engineer)**

This guide explains how to connect your AI Models to the main Hawkeye Dashboard.

---

## 🟢 Option A: Run Your Model on a Separate Port (Recommended)

1.  Run your code as a simple API (Flask/FastAPI) on your own port.
    *   **Vision**: Port `8001`
    *   **Audio**: Port `8002`
    *   **Motion**: Port `8003`

2.  Your API must return the **Exact JSON Format** shared in our group chat.

3.  **Open `backend/config.py`** in this repo and change the URLs:
    ```python
    # Example Change for Vision
    VISION_ENDPOINT = "http://localhost:8001/vision/frame-analysis"
    USE_MOCK_VISION = False # <--- Switch this to False!
    ```

4.  **Restart the Backend**. The dashboard will now show YOUR data!

---

## 🟡 Option B: The "All-in-One" Method

If you just have a python function (e.g., `detect(image)`), copy your file into `backend/routers/` and import it.

1.  Paste your `vision_model.py` into `backend/`.
2.  Edit `backend/routers/vision.py`:
    ```python
    from backend.vision_model import detect_objects
    
    @router.get("/frame-analysis")
    def frame_analysis():
        # Call your function directly
        return detect_objects()
    ```

---

## 🚨 JSON Requirements (Reminder)

**Vision (`/vision/frame-analysis`):**
```json
{
  "timestamp": 1234567.89,
  "objects": [{"label": "person", "bbox": [0,0,100,100], "confidence": 0.9}],
  "vision_risk_factors": {"weapon_detected": true}
}
```

**Audio (`/audio/audio-analysis`):**
```json
{
  "timestamp": 1234567.89,
  "events": [{"label": "scream", "confidence": 0.8}],
  "audio_risk_score": 0.9
}
```

**Motion (`/motion/crowd-analysis`):**
```json
{
  "timestamp": 1234567.89,
  "panic_score": 0.7,
  "surge_detected": false
}
```
