# explainability/composite_overlay.py
import cv2
import numpy as np
from .heatmap import generate_heatmap
from .pose_overlay import draw_pose
from .motion_overlay import draw_motion_vectors


def _normalize_to_list(obj):
    """
    Converts single object → list.
    Returns [] if None.
    """
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        return obj
    return [obj]


# ----------------------------------------------------------
#   RISK HUD (MULTI-PERSON)
# ----------------------------------------------------------
def _draw_risk_hud(frame, risk_scores):
    """
    risk_scores: dict → {"P1":0.62, "P2":0.30}

    Draws stacked HUD labels in top-left corner.
    Style remains identical to previous version.
    """
    if not risk_scores:
        return frame

    out = frame.copy()
    x0, y0 = 10, 10
    box_height = 38
    box_width = 260
    spacing = 6

    for idx, (pid, score) in enumerate(risk_scores.items()):
        top = y0 + idx * (box_height + spacing)
        bottom = top + box_height

        # Background box
        cv2.rectangle(out, (x0, top), (x0 + box_width, bottom), (0, 0, 0), -1)

        # Text
        text = f"{pid} RISK: {int(score * 100)}%"
        cv2.putText(out, text, (x0 + 10, top + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)

    return out


# ----------------------------------------------------------
#   MASTER COMPOSITE OVERLAY SYSTEM (MULTI-PERSON)
# ----------------------------------------------------------
def combine_overlays(frame, heatmaps=None, poses=None, motions=None, risk_scores=None):
    """
    Hybrid API for multi-person explainability.

    heatmaps:  single heatmap OR list of heatmaps
    poses:     single keypoint set OR list of persons
    motions:   single optical flow OR list of flows
    risk_scores: dict like {"P1":0.62, "P2":0.41}

    Output order:
      1. Heatmap (background layer)
      2. Motion vectors (middle layer)
      3. Pose skeleton (top layer for visual clarity)
      4. Risk HUD (UI layer)
    """

    out = frame.copy()

    # 1) HEATMAP LAYER
    out = generate_heatmap(out, heatmaps)

    # 2) MOTION LAYER
    out = draw_motion_vectors(out, motions)

    # 3) POSE LAYER
    out = draw_pose(out, poses)

    # 4) RISK HUD LAYER
    out = _draw_risk_hud(out, risk_scores)

    return out
