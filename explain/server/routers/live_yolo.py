# server/routers/live_yolo.py
import cv2
import base64
import numpy as np
from fastapi import APIRouter, WebSocket
from explain.person_pipeline import PersonPipeline
from explain.explainability.composite_overlay import combine_overlays

router = APIRouter()
pipeline = PersonPipeline(model_path="yolov8s-pose.pt")

def encode_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")

@router.websocket("/ws/live-yolo")
async def live_yolo_stream(ws: WebSocket):
    await ws.accept()
    print("🔵 Client connected to Live YOLO WebSocket")

    while True:
        try:
            data = await ws.receive_bytes()

            # Convert bytes -> numpy image
            np_data = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            # Run YOLO + tracking + explainability
            persons = pipeline.process_frame(frame)

            heatmaps = [p["heatmap"] for p in persons]
            poses = [p["keypoints"] for p in persons]
            flows = [p["flow"] for p in persons]
            risk_scores = {f"P{p['id']}": p["risk"] for p in persons}

            output = combine_overlays(
                frame.copy(),
                heatmaps=heatmaps,
                poses=poses,
                motions=flows,
                risk_scores=risk_scores
            )

            encoded = encode_frame(output)

            # Send processed frame back to frontend
            await ws.send_text(encoded)

        except Exception as e:
            print("❌ WebSocket Closed:", e)
            break