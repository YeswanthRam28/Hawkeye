# explainability/heatmap.py
import cv2
import numpy as np

def _to_uint8_heatmap(hmap):
    """
    Normalizes input heatmap to uint8 0..255.
    """
    if hmap is None:
        return None

    arr = hmap.astype(np.float32)

    # If data is 0..1, scale to 0..255
    if arr.max() <= 1.0:
        arr = arr * 255.0

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def generate_heatmap(frame, heatmaps, alpha=0.6):
    """
    Multi-person hybrid heatmap generator.

    Supports:
    - Single heatmap (2D array)
    - List of heatmaps for multiple persons

    Fixes brightness issue by:
    → Applying GLOBAL normalization so all heatmaps have equal intensity.
    """

    out = frame.copy()

    # Convert single → list
    if isinstance(heatmaps, np.ndarray):
        heatmaps = [heatmaps]
    elif heatmaps is None:
        return out

    # ------------------------------------
    # 1) GLOBAL NORMALIZATION
    # ------------------------------------
    # Compute GLOBAL max value across all heatmaps
    max_vals = [np.max(h) for h in heatmaps if h is not None]
    global_max = max(max_vals) if max_vals else 1.0

    # Prepare combined grayscale heatmap
    h, w = frame.shape[:2]
    combined = np.zeros((h, w), dtype=np.float32)

    for hmap in heatmaps:
        if hmap is None:
            continue

        # Resize if needed
        if hmap.shape != (h, w):
            hmap = cv2.resize(hmap, (w, h))

        # Normalize using GLOBAL max (ensures equal brightness)
        normalized = (hmap.astype(np.float32) / global_max) * 255.0
        combined = np.maximum(combined, normalized)

    combined = combined.astype(np.uint8)

    # ------------------------------------
    # 2) APPLY COLOR MAP
    # ------------------------------------
    heatmap_color = cv2.applyColorMap(combined, cv2.COLORMAP_JET)

    # ------------------------------------
    # 3) BLEND WITH FRAME
    # ------------------------------------
    out = cv2.addWeighted(heatmap_color, alpha, out, 1 - alpha, 0)

    return out
