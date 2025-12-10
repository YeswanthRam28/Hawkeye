# main_pipeline.py
"""
Unified pipeline (Step 3 + Step 4):
 - Runs PersonPipeline (YOLO pose, tracking, flow, risk)
 - Feeds all frames + metadata to ReplayEngine
 - Auto-triggers when ANY person risk > RISK_TRIGGER
 - Saves incident clip + raw frames + JSON
 - Immediately calls Evidence Builder to generate:
       ✔ keyframe
       ✔ overlay keyframe
       ✔ pose image
       ✔ motion image
       ✔ risk timeline PNG
       ✔ evidence.json summary
 - Optional: Step 5 LLM narrative generator (if available)
"""

import cv2
import time
import os
import json

# ------------------------------
# CONFIG
# ------------------------------
RISK_TRIGGER = 0.70        # Trigger replay if any person exceeds 70%
COOLDOWN = 4.0             # Minimum time between triggers
PRE_SECONDS = 5
POST_SECONDS = 5
FPS = 20
SHOW_PREVIEW = True
OUT_DIR = "incidents_auto"

# ------------------------------
# IMPORT MODULES
# ------------------------------
from person_pipeline import PersonPipeline
from replay.engine import ReplayEngine
from evidence.evidence_builder import build_evidence_packet

# Optional Step 5 (LLM narrative)
try:
    from narrative_generator import generate_narrative
    HAVE_NARRATIVE = True
except:
    HAVE_NARRATIVE = False


# ------------------------------
# HELPERS
# ------------------------------
def extract_metadata(person_list):
    """Extracts clean metadata for ReplayEngine."""
    if not person_list:
        return {"risk_overall": 0.0, "persons": []}

    risk_overall = max([p.get("risk", 0) for p in person_list])

    cleaned = []
    for p in person_list:
        cleaned.append({
            "id": int(p["id"]),
            "bbox": [int(x) for x in p["bbox"]],
            "risk": float(p["risk"]),
            "vision_conf": float(p.get("vision_conf", 0)),
            "has_heatmap": p.get("heatmap") is not None,
            "has_flow": p.get("flow") is not None
        })

    return {
        "risk_overall": risk_overall,
        "persons": cleaned
    }


# ------------------------------
# MAIN PIPELINE LOOP
# ------------------------------
def main(camera_index=0):
    print("\n=== HAWKEYE — Unified Detection + Replay + Evidence Pipeline ===")

    # Init YOLO pipeline + replay engine
    pipeline = PersonPipeline(model_path="yolov8s-pose.pt")
    replay = ReplayEngine(pre_seconds=PRE_SECONDS,
                          post_seconds=POST_SECONDS,
                          fps=FPS,
                          incident_root=OUT_DIR)

    cap = cv2.VideoCapture(camera_index)
    last_trigger = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera feed lost.")
            break

        # Run full multi-person processing
        persons = pipeline.process_frame(frame)

        # Build replay metadata
        metadata = extract_metadata(persons)
        risk_now = metadata["risk_overall"]

        # Add to replay buffer
        incident_info = replay.add_frame(frame, metadata)

        # If replay finished saving — build evidence
        if incident_info:
            print("\n[ReplayEngine] Incident saved:", incident_info)

            incident_dir = incident_info["incident_dir"]

            # STEP 4 — Build Evidence Packet
            evidence = build_evidence_packet(incident_dir)
            print("[Evidence] Packet created:", evidence)

            # Optional Narrative (Step 5)
            if HAVE_NARRATIVE:
                try:
                    summary = {
                        "video_id": os.path.basename(incident_dir),
                        "start_time": evidence["start_time"],
                        "end_time": evidence["end_time"],
                        "persons": {}
                    }
                    out_json = os.path.join(incident_dir, "evidence.json")
                    with open(out_json, "r") as f:
                        full_e = json.load(f)

                    # Summaries per person
                    for pid, pdata in full_e.get("per_person", {}).items():
                        summary["persons"][pid] = {
                            "risk": pdata.get("risk_timeline", []),
                            "bbox": pdata.get("last_bbox", None)
                        }

                    narrative = generate_narrative(**summary)
                    with open(os.path.join(incident_dir, "narrative.txt"), "w") as f:
                        f.write(narrative)

                    print("[Narrative] LLM output saved.")
                except Exception as e:
                    print("[Narrative] Failed:", e)

        # AUTO TRIGGER
        now = time.time()
        if risk_now >= RISK_TRIGGER and (now - last_trigger > COOLDOWN):
            print("\n[AUTO] Triggering incident — risk:", risk_now)
            replay.trigger()
            last_trigger = now

        # PREVIEW FRAME
        if SHOW_PREVIEW:
            preview = frame.copy()
            cv2.putText(preview, f"RISK: {risk_now:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255) if risk_now > RISK_TRIGGER else (0, 255, 0), 2)

            cv2.imshow("Hawkeye Live Pipeline", preview)
            key = cv2.waitKey(1)

            if key == ord('t'):
                print("\n[MANUAL] Triggered by user.")
                replay.trigger()
            elif key == 27:
                print("Exiting...")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
