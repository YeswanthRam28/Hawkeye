# server/routers/explain.py
from fastapi import APIRouter
import cv2
import base64
import numpy as np

from explain.person_pipeline import PersonPipeline
from explain.explainability.composite_overlay import combine_overlays

router = APIRouter(prefix="/frame", tags=["frame"])

pipeline = PersonPipeline()

def encode_img(img):
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("utf-8")

@router.get("/explain")
def explain_frame():
    try:
        frame, result = pipeline.process_frame()

        persons = result.get("persons", {})

        # ---- Convert dict → lists for overlay ----
        pose_list = []
        heatmap_list = []
        motion_list = []

        for pid, pdata in persons.items():
            pose_list.append(pdata.get("keypoints"))
            heatmap_list.append(pdata.get("heatmap"))
            motion_list.append(pdata.get("flow"))

        # ---- Risk scores as dict stays dict ----
        risk_scores = result.get("risk_scores", {})

        # ---- Build overlay ----
        overlay = combine_overlays(
            frame.copy(),
            heatmaps=heatmap_list,
            poses=pose_list,
            motions=motion_list,
            risk_scores=risk_scores
        )

        encoded = encode_img(overlay)

        return {
            "success": True,
            "risk_overall": result.get("risk_overall", 0),
            "persons": list(persons.keys()),
            "frame_b64": encoded
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
