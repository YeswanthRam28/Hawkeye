# prepare_inputs.py
import numpy as np
import cv2

def prepare_explainability_inputs(frame, persons):
    poses = []
    heatmaps = []
    motions = []
    risk_scores = {}

    h, w = frame.shape[:2]

    for p in persons:
        # 1️⃣ Pose
        poses.append(p["keypoints"])

        # 2️⃣ Heatmap (simple bounding-box region heatmap for now)
        heatmap = np.zeros((h, w), dtype=np.float32)
        x1, y1, x2, y2 = p["bbox"]
        heatmap[y1:y2, x1:x2] = 1.0     # highlight person area
        heatmaps.append(heatmap)

        # 3️⃣ Motion (placeholder until optical flow integration)
        flow = np.zeros((h, w, 2), dtype="float32")
        motions.append(flow)

        # 4️⃣ Fake risk score (temporary for demo)
        risk_scores[f"P{p['id']+1}"] = p["confidence"]

    return poses, heatmaps, motions, risk_scores
