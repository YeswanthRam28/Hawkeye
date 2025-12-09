import cv2
import json
import time
import base64
import numpy as np

from motion import compute_optical_flow
from metrics import compute_motion_metrics

# ---- PARAMETERS ----
alpha = 0.2  # smoothing factor
FRAME_W, FRAME_H = 400, 300  # panel sizes

# ---- SMOOTHED METRICS FOR BOTH FEEDS ----
smooth_cam = {"panic": 0.0, "density": 0.0, "variation": 0.0}
smooth_vid = {"panic": 0.0, "density": 0.0, "variation": 0.0}

# ---- OPEN CAMERA + VIDEO ----
cap_cam = cv2.VideoCapture(0)                # Camera feed
cap_vid = cv2.VideoCapture("sample.mp4")     # Video feed

if not cap_cam.isOpened():
    print("Error: Cannot open camera.")
    exit()

if not cap_vid.isOpened():
    print("Error: Cannot open video file 'sample.mp4'.")
    exit()

# ---- GET FIRST FRAMES ----
ret_cam, prev_cam = cap_cam.read()
ret_vid, prev_vid = cap_vid.read()

if not ret_cam:
    print("Error reading camera first frame.")
    exit()

if not ret_vid:
    print("Error reading video first frame.")
    exit()

# ---- MAIN LOOP ----
while True:

    # Read next frames
    ret_cam, frame_cam = cap_cam.read()
    ret_vid, frame_vid = cap_vid.read()

    # Camera failure -> exit
    if not ret_cam:
        print("Camera feed dropped.")
        break

    # Loop video if ended
    if not ret_vid:
        cap_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_vid, frame_vid = cap_vid.read()
        if not ret_vid:
            print("Video cannot restart.")
            break

    # ---- OPTICAL FLOW ----
    mag_cam, ang_cam = compute_optical_flow(prev_cam, frame_cam)
    mag_vid, ang_vid = compute_optical_flow(prev_vid, frame_vid)

    # ---- COLOR HEATMAPS ----
    motion_map_cam = cv2.normalize(mag_cam, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    color_map_cam = cv2.applyColorMap(motion_map_cam, cv2.COLORMAP_JET)

    motion_map_vid = cv2.normalize(mag_vid, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    color_map_vid = cv2.applyColorMap(motion_map_vid, cv2.COLORMAP_JET)

    # ---- RAW METRICS (CAMERA) ----
    metrics_cam = compute_motion_metrics(mag_cam)
    panic_cam = metrics_cam["panic_score"]
    density_cam = metrics_cam["crowd_density"]
    variation_cam = metrics_cam["motion_variation"]

    # ---- RAW METRICS (VIDEO) ----
    metrics_vid = compute_motion_metrics(mag_vid)
    panic_vid = metrics_vid["panic_score"]
    density_vid = metrics_vid["crowd_density"]
    variation_vid = metrics_vid["motion_variation"]

    # ---- SMOOTHING ----
    smooth_cam["panic"] = alpha * panic_cam + (1 - alpha) * smooth_cam["panic"]
    smooth_cam["density"] = alpha * density_cam + (1 - alpha) * smooth_cam["density"]
    smooth_cam["variation"] = alpha * variation_cam + (1 - alpha) * smooth_cam["variation"]

    smooth_vid["panic"] = alpha * panic_vid + (1 - alpha) * smooth_vid["panic"]
    smooth_vid["density"] = alpha * density_vid + (1 - alpha) * smooth_vid["density"]
    smooth_vid["variation"] = alpha * variation_vid + (1 - alpha) * smooth_vid["variation"]

    # ---- OVERLAYS: ALERTS + METRICS (CAMERA FEED) ----
    if smooth_cam["panic"] > 3:
        cv2.putText(color_map_cam, "PANIC DETECTED!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    if smooth_cam["density"] > 0.30:
        cv2.putText(color_map_cam, "CROWD SURGE!", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)

    if smooth_cam["variation"] > 2:
        cv2.putText(color_map_cam, "MOTION ANOMALY!", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)

    cv2.putText(color_map_cam, f"Panic: {smooth_cam['panic']:.2f}", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(color_map_cam, f"Density: {smooth_cam['density']:.2f}", (20, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(color_map_cam, f"Variation: {smooth_cam['variation']:.2f}", (20, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # ---- OVERLAYS: ALERTS + METRICS (VIDEO) ----
    if smooth_vid["panic"] > 3:
        cv2.putText(color_map_vid, "PANIC DETECTED!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    if smooth_vid["density"] > 0.30:
        cv2.putText(color_map_vid, "CROWD SURGE!", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)

    if smooth_vid["variation"] > 2:
        cv2.putText(color_map_vid, "MOTION ANOMALY!", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)

    cv2.putText(color_map_vid, f"Panic: {smooth_vid['panic']:.2f}", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(color_map_vid, f"Density: {smooth_vid['density']:.2f}", (20, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(color_map_vid, f"Variation: {smooth_vid['variation']:.2f}", (20, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # ---- PANEL LABELS ----
    cv2.putText(frame_cam, "CAMERA FEED", (20, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame_vid, "VIDEO FEED", (20, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # ---- RESIZE + CONCAT PANELS ----
    raw_cam_resized = cv2.resize(frame_cam, (FRAME_W, FRAME_H))
    map_cam_resized = cv2.resize(color_map_cam, (FRAME_W, FRAME_H))
    panel_cam = cv2.hconcat([raw_cam_resized, map_cam_resized])

    raw_vid_resized = cv2.resize(frame_vid, (FRAME_W, FRAME_H))
    map_vid_resized = cv2.resize(color_map_vid, (FRAME_W, FRAME_H))
    panel_vid = cv2.hconcat([raw_vid_resized, map_vid_resized])

    # ---- FINAL DASHBOARD ----
    dashboard = cv2.hconcat([panel_cam, panel_vid])
    cv2.imshow("Hawkeye Motion AI - Camera | Video", dashboard)

    # --------------------------------------------------------------------
    # 🎯 NEW SECTION: JSON OUTPUT FOR API ENDPOINTS
    # --------------------------------------------------------------------

    # ========= 1) CROWD ANALYSIS JSON =========
    crowd_analysis = {
        "timestamp": time.time(),
        "crowd_density": float(smooth_cam["density"]),
        "surge_detected": bool(smooth_cam["density"] > 0.30),
        "surge_direction": "north-east",  # placeholder
        "panic_score": float(smooth_cam["panic"])
    }

    with open("crowd_analysis.json", "w") as f:
        json.dump(crowd_analysis, f, indent=2)

    # ========= 2) MOTION VECTORS JSON =========

    # Convert velocity map to base64
    velocity_map = cv2.normalize(mag_cam, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    _, buffer = cv2.imencode(".jpg", velocity_map)
    velocity_map_b64 = base64.b64encode(buffer).decode("utf-8")

    motion_vectors = {
        "timestamp": time.time(),
        "optical_flow_vectors": velocity_map_b64,
        "velocity_map": velocity_map_b64,
        "avg_speed": float(mag_cam.mean()),
        "anomaly_regions": [
            {"region_id": 1, "magnitude": float(np.max(mag_cam))}
        ]
    }

    with open("motion_vectors.json", "w") as f:
        json.dump(motion_vectors, f, indent=2)

    # ========= 3) MOTION EVENTS JSON =========
    events = []

    if smooth_cam["density"] > 0.30:
        events.append({"type": "crowd_surge", "confidence": 0.86})

    if smooth_cam["panic"] > 3:
        events.append({"type": "sudden_run", "confidence": 0.74})

    motion_events = {
        "timestamp": time.time(),
        "events": events
    }

    with open("motion_events.json", "w") as f:
        json.dump(motion_events, f, indent=2)

    # --------------------------------------------------------------------

    # Prepare next iteration
    prev_cam = frame_cam.copy()
    prev_vid = frame_vid.copy()

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_cam.release()
cap_vid.release()
cv2.destroyAllWindows()
