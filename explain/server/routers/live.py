# server/routers/live.py

import cv2
import base64
import asyncio
from fastapi import APIRouter, WebSocket
from explain.person_pipeline import PersonPipeline
from explain.explainability.composite_overlay import combine_overlays

router = APIRouter()

# Load YOLO model once
pipeline = PersonPipeline(model_path="yolov8s-pose.pt")

# Open camera
cap = cv2.VideoCapture(0)

def frame_to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


@router.websocket("/ws/live")
async def live_stream(ws: WebSocket):
    await ws.accept()

    await ws.send_json({"status": "connected", "message": "Live explainability feed started"})

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                await ws.send_json({"error": "camera frame read failed"})
                await asyncio.sleep(0.1)
                continue

            persons = pipeline.process_frame(frame)

            # Build explainability overlays
            heatmaps = [p["heatmap"] for p in persons]
            poses = [p["keypoints"] for p in persons]
            flows = [p["flow"] for p in persons]
            risk_scores = {f"P{p['id']}": p["risk"] for p in persons}

            composite = combine_overlays(
                frame.copy(),
                heatmaps=heatmaps,
                poses=poses,
                motions=flows,
                risk_scores=risk_scores
            )

            # Encode frame
            b64 = frame_to_base64(composite)

            # Send data packet
            await ws.send_json({
                "success": True,
                "frame_b64": b64,
                "persons": persons
            })

            await asyncio.sleep(0.03)  # ~30 FPS

        except Exception as e:
            await ws.send_json({"success": False, "error": str(e)})
            break

    await ws.close()
