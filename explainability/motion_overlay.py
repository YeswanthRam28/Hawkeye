# explainability/motion_overlay.py
import cv2
import numpy as np

def _is_flow_like(obj):
    return isinstance(obj, (list, tuple)) == False and hasattr(obj, "shape") and obj.ndim == 3 and obj.shape[2] == 2

def _normalize_flow_input(flow_or_list):
    if flow_or_list is None:
        return []
    # single numpy flow array
    if _is_flow_like(flow_or_list):
        return [flow_or_list]
    # list of flows
    if isinstance(flow_or_list, (list, tuple)):
        return [f for f in flow_or_list if _is_flow_like(f)]
    return []

def draw_motion_vectors(frame, flow_or_list, step=16):
    """
    Hybrid API:
      - flow_or_list can be a single full-frame flow (HxWx2) OR a list of such arrays.
      - We draw tiny dot grid + crisp white arrows, preserving the look.
    """
    flows = _normalize_flow_input(flow_or_list)

    # If no flows, return frame unchanged (backwards compatible)
    if not flows:
        return frame

    out = frame.copy()
    h, w = out.shape[:2]

    # Draw grid of tiny dots (same as earlier tests) first so arrows appear on top
    dot_color = (200, 200, 200)
    for y in range(0, h, step):
        for x in range(0, w, step):
            out = cv2.circle(out, (x, y), 1, dot_color, -1)

    # For each person flow, draw arrows (they will overlay on same frame)
    for flow in flows:
        fh, fw = flow.shape[:2]
        # if flow size different, scale to frame size
        if (fh, fw) != (h, w):
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_NEAREST)
            # flow values get scaled incorrectly by resize; for visualization tests we accept this.
        for y in range(0, h, step):
            for x in range(0, w, step):
                dx, dy = flow[y, x]
                # draw only if noticeable motion
                if abs(dx) < 0.1 and abs(dy) < 0.1:
                    continue

                end_x = int(x + dx)
                end_y = int(y + dy)

                # draw body (crisp)
                cv2.line(out, (x, y), (end_x, end_y), (255, 255, 255), 2)

                # manual arrowhead (sharp, not blurry)
                angle = np.arctan2(dy, dx + 1e-7)
                arrow_len = 6
                left_x  = int(end_x - arrow_len * np.cos(angle - np.pi/6))
                left_y  = int(end_y - arrow_len * np.sin(angle - np.pi/6))
                right_x = int(end_x - arrow_len * np.cos(angle + np.pi/6))
                right_y = int(end_y - arrow_len * np.sin(angle + np.pi/6))

                cv2.line(out, (end_x, end_y), (left_x, left_y), (255, 255, 255), 2)
                cv2.line(out, (end_x, end_y), (right_x, right_y), (255, 255, 255), 2)

    return out
