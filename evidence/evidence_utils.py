# evidence/evidence_utils.py
import os
import json
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

def load_evidence_json(evidence_json_path: str) -> List[Dict[str, Any]]:
    """
    Load the evidence JSON produced by replay engine.
    Returns list of frame entries: {"index", "timestamp", "metadata", "image"}
    """
    if not os.path.exists(evidence_json_path):
        raise FileNotFoundError(f"Evidence JSON not found: {evidence_json_path}")
    with open(evidence_json_path, "r") as f:
        data = json.load(f)
    return data

def load_frame_image(frame_entry: Dict[str, Any]) -> Tuple[str, Any]:
    """
    Given a single entry from evidence.json, return image path and BGR numpy array.
    """
    img_path = frame_entry.get("image") or frame_entry.get("frame_path")
    if img_path is None:
        # try to infer a path in same folder
        raise FileNotFoundError("Frame image path missing in evidence entry")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Frame image not found: {img_path}")
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    return img_path, img

def find_highest_risk_frame(evidence_frames: List[Dict[str, Any]]) -> int:
    """
    Return index in evidence_frames of the frame with the highest max-person risk.
    Each frame metadata expected to have 'metadata'->'persons' list with 'risk' or frame-wide 'risk_overall'.
    """
    best_idx = 0
    best_val = -1.0
    for i, entry in enumerate(evidence_frames):
        meta = entry.get("metadata", {})
        # prefer aggregated risk_overall if present
        ro = meta.get("risk_overall")
        if ro is not None:
            val = float(ro)
        else:
            # check persons
            persons = meta.get("persons", [])
            if persons:
                val = max([p.get("risk", 0.0) for p in persons])
            else:
                val = 0.0
        if val > best_val:
            best_val = val
            best_idx = i
    return best_idx

def aggregate_person_timeline(evidence_frames: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build per-person timeline dict keyed by "P{track_id}" with structure:
      {"risk_timeline": [(time_rel_s, risk)], "bbox_timeline": [(time_rel_s, bbox)], "keypoints_last": [...], ...}
    time_rel_s is seconds relative to start_time (first frame timestamp).
    """
    out = {}
    if not evidence_frames:
        return out
    start_ts = evidence_frames[0].get("timestamp", 0.0)
    for idx, entry in enumerate(evidence_frames):
        ts = entry.get("timestamp", start_ts)
        rel = float(ts - start_ts)
        meta = entry.get("metadata", {}) or {}
        persons = meta.get("persons", [])
        for p in persons:
            pid = p.get("id")
            if pid is None:
                continue
            key = f"P{pid}"
            if key not in out:
                out[key] = {
                    "risk_timeline": [],
                    "bbox_timeline": [],
                    "last_keypoints": p.get("keypoints", []),
                    "vision_conf": p.get("vision_conf", p.get("confidence", 0.0))
                }
            out[key]["risk_timeline"].append((rel, float(p.get("risk", 0.0))))
            out[key]["bbox_timeline"].append((rel, p.get("bbox")))
            # update last seen keypoints
            if p.get("keypoints"):
                out[key]["last_keypoints"] = p.get("keypoints")
    return out

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
