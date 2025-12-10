# motion_per_person.py
import cv2
import numpy as np

def compute_dense_flow(prev_gray, cur_gray):
    """
    Farneback dense optical flow. Returns flow (H,W,2) float32.
    """
    flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray,
                                        None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    return flow

def compute_person_flow_global(flow, bbox):
    """
    Given full-frame flow and a bbox, returns a full-frame flow where outside bbox is zero.
    This matches your explainability API which expects full-frame flows per person.
    bbox: [x1,y1,x2,y2]
    """
    h, w = flow.shape[:2]
    pf = np.zeros_like(flow)
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w-1, int(x2)); y2 = min(h-1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return pf
    pf[y1:y2, x1:x2, :] = flow[y1:y2, x1:x2, :]
    return pf

def aggregate_flow_in_bbox(flow, bbox, downsample=4):
    """
    Returns average dx, dy and magnitude inside bbox.
    """
    x1,y1,x2,y2 = bbox
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(flow.shape[1]-1, int(x2)); y2 = min(flow.shape[0]-1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 0.0
    sub = flow[y1:y2:downsample, x1:x2:downsample]
    if sub.size == 0:
        return 0.0, 0.0, 0.0
    avg = sub.reshape(-1, 2).mean(axis=0)
    mag = float((avg**2).sum()**0.5)
    return float(avg[0]), float(avg[1]), mag
