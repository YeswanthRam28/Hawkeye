# evidence/evidence_builder.py
import os
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any
from explain.evidence.evidence_utils import (
    load_evidence_json,
    load_frame_image,
    find_highest_risk_frame,
    aggregate_person_timeline,
    ensure_dir
)

# Optional: use your composite overlay to create pose/heatmap/motion snapshots
# We attempt to import combine_overlays from your explainability module.
try:
    # adapt path if your explainability package exports combine_overlays at top-level
    from explainability.composite_overlay import combine_overlays
except Exception:
    try:
        from explainability import combine_overlays
    except Exception:
        combine_overlays = None  # we'll not use overlays if import fails

def _save_image(path: str, img):
    cv2.imwrite(path, img)

def _generate_risk_plot(risk_timeline_dict, out_path: str):
    """
    risk_timeline_dict: {person_id: [(t, risk), ...], ...}
    Saves a matplotlib plot to out_path
    """
    plt.figure(figsize=(8,3), dpi=150)
    for pid, data in risk_timeline_dict.items():
        times = [t for t, r in data]
        risks = [r*100 for t, r in data]
        plt.plot(times, risks, label=pid, linewidth=2)
        # mark last point
        if times:
            plt.text(times[-1], risks[-1], f"{int(risks[-1])}%", fontsize=9)
    plt.xlabel("Time (s)")
    plt.ylabel("Risk (%)")
    plt.title("Per-person Risk Timeline")
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def build_evidence_packet(incident_dir: str, out_dir: str = None) -> Dict[str, Any]:
    """
    Main entry: build evidence packet for incident directory saved by ReplayEngine.
    - incident_dir: path to the saved incident folder (contains clip.mp4, frame_XXXX.jpg, evidence.json)
    - out_dir: optional output folder (if None uses incident_dir/evidence_out)
    Returns summary dict with paths to saved assets.
    """
    if not os.path.isdir(incident_dir):
        raise FileNotFoundError(f"Incident directory not found: {incident_dir}")

    evidence_json_candidates = [
        os.path.join(incident_dir, "evidence.json"),
        os.path.join(incident_dir, "incident_*_evidence.json")
    ]
    evidence_json = os.path.join(incident_dir, "evidence.json")
    if not os.path.exists(evidence_json):
        # try search for any json in folder
        found = [p for p in os.listdir(incident_dir) if p.endswith(".json")]
        if found:
            evidence_json = os.path.join(incident_dir, found[0])
        else:
            raise FileNotFoundError("evidence.json not found in incident dir")

    frames = load_evidence_json(evidence_json)

    if out_dir is None:
        out_dir = os.path.join(incident_dir, "evidence_out")
    ensure_dir(out_dir)

    # 1) Find highest risk frame and load it
    best_idx = find_highest_risk_frame(frames)
    best_entry = frames[best_idx]
    best_img_path, best_img = load_frame_image(best_entry)

    # Save keyframe
    keyframe_path = os.path.join(out_dir, "key_frame.png")
    _save_image(keyframe_path, best_img)

    # 2) Create an overlay snapshot (pose + heatmap + motion) for the key frame if possible
    overlay_path = os.path.join(out_dir, "overlay_keyframe.png")
    try:
        meta = best_entry.get("metadata", {})
        if combine_overlays is not None:
            # prepare inputs
            persons_meta = meta.get("persons", [])
            heatmaps = [p.get("heatmap") for p in persons_meta]
            poses = [p.get("keypoints") for p in persons_meta]
            flows = [p.get("flow") for p in persons_meta]
            risk_map = {f"P{p['id']}": p.get("risk", 0.0) for p in persons_meta}
            overlay = combine_overlays(best_img.copy(), heatmaps=heatmaps, poses=poses, motions=flows, risk_scores=risk_map)
            _save_image(overlay_path, overlay)
        else:
            # fallback: just save the key frame as overlay
            _save_image(overlay_path, best_img)
    except Exception as e:
        # fallback to saving the raw frame
        _save_image(overlay_path, best_img)

    # 3) Build per-person summary timelines
    person_timelines = aggregate_person_timeline(frames)

    # 4) Generate risk timeline plot
    # convert to {pid: [(t, r),...]}
    risks_for_plot = {pid: data["risk_timeline"] for pid, data in person_timelines.items()}
    risk_plot_path = os.path.join(out_dir, "risk_timeline.png")
    _generate_risk_plot(risks_for_plot, risk_plot_path)

    # 5) Optionally produce pose-only and motion-only snapshots for the highest risk frame
    pose_snap = os.path.join(out_dir, "pose_keyframe.png")
    motion_snap = os.path.join(out_dir, "motion_keyframe.png")
    try:
        # try to draw only skeletons (very simple overlay)
        img_pose = best_img.copy()
        persons_meta = best_entry.get("metadata", {}).get("persons", [])
        for p in persons_meta:
            kps = p.get("keypoints", [])
            # draw joints
            for kp in kps:
                x = int(kp.get("x", 0))
                y = int(kp.get("y", 0))
                score = kp.get("score", 1.0)
                if score > 0.2:
                    cv2.circle(img_pose, (x, y), 4, (0,255,0), -1)
            # draw bbox
            bbox = p.get("bbox")
            if bbox:
                x1,y1,x2,y2 = bbox
                cv2.rectangle(img_pose, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        _save_image(pose_snap, img_pose)
    except Exception:
        _save_image(pose_snap, best_img)

    try:
        # motion snapshot: draw small arrows using flow if present in metadata
        img_motion = best_img.copy()
        persons_meta = best_entry.get("metadata", {}).get("persons", [])
        for p in persons_meta:
            flow = p.get("flow")  # expected full-frame flow HxWx2 or None
            if flow is None:
                continue
            # upscale small arrows inside bbox for readability (draw a few)
            h, w = img_motion.shape[:2]
            # sample a coarse grid
            step = max(8, min(h, w) // 40)
            for y in range(0, h, step*3):
                for x in range(0, w, step*3):
                    try:
                        dx = int(flow[y, x, 0]*2.5)
                        dy = int(flow[y, x, 1]*2.5)
                        if abs(dx) + abs(dy) < 1:
                            continue
                        cv2.arrowedLine(img_motion, (x,y), (x+dx, y+dy), (0,0,255), 1, tipLength=0.3)
                    except Exception:
                        continue
        _save_image(motion_snap, img_motion)
    except Exception:
        _save_image(motion_snap, best_img)

    # 6) Save per-person JSON summary
    per_person_out = {}
    for pid, info in person_timelines.items():
        # find peak risk and its time
        if info["risk_timeline"]:
            peak_t, peak_r = max(info["risk_timeline"], key=lambda x: x[1])
        else:
            peak_t, peak_r = 0.0, 0.0
        per_person_out[pid] = {
            "vision_conf": info.get("vision_conf", 0.0),
            "risk_peak": peak_r,
            "risk_peak_time": peak_t,
            "last_bbox": info["bbox_timeline"][-1] if info["bbox_timeline"] else None,
            "last_keypoints_count": len(info.get("last_keypoints", []))
        }

    # 7) Compose final incident report JSON
    report = {
        "incident_dir": incident_dir,
        "keyframe": keyframe_path,
        "overlay_keyframe": overlay_path,
        "pose_keyframe": pose_snap,
        "motion_keyframe": motion_snap,
        "risk_plot": risk_plot_path,
        "per_person": per_person_out,
        "frame_count": len(frames),
        "start_time": frames[0].get("timestamp") if frames else None,
        "end_time": frames[-1].get("timestamp") if frames else None,
        "evidence_json": evidence_json
    }

    # Save report
    report_path = os.path.join(out_dir, "incident_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python evidence_builder.py <incident_dir>")
        print("Example: python evidence_builder.py incidents/incident_1765298346")
        sys.exit(1)
    inc = sys.argv[1]
    r = build_evidence_packet(inc)
    print("Evidence built:", r)
