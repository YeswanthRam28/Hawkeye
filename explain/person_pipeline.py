# person_pipeline.py
import cv2
import numpy as np
from explain.yolo_pose import YOLOPoseDetector
from explain.simple_tracker import CentroidTracker
from explain.motion_per_person import compute_dense_flow, compute_person_flow_global, aggregate_flow_in_bbox
from explain.simple_risk_fuser import fuse_risk  # tiny fuser you already have / reuse

class PersonPipeline:
    def __init__(self, model_path="yolov8s-pose.pt"):
        self.detector = YOLOPoseDetector(model_path)
        self.tracker = CentroidTracker(max_disappeared=30)
        self.prev_gray = None
        self.prev_frame = None
        # buffer of recent flows if needed
        self.last_flow = None

    def process_frame(self, frame):
        """
        frame: BGR numpy array
        returns: list of person dicts:
          {
            "id": int,
            "bbox": [x1,y1,x2,y2],
            "keypoints": [...],
            "vision_conf": float,
            "flow": full_frame_flow_masked (HxWx2),
            "motion": {"dx","dy","mag"},
            "heatmap": 2D array (HxW),
            "risk": 0..1
          }
        """
        H, W = frame.shape[:2]

        # 1) Detect persons and keypoints
        persons = self.detector.detect(frame)  # returns list with bbox,keypoints,confidence

        # build list of boxes for tracker
        rects = [p["bbox"] for p in persons]

        # 2) Tracker: get mapping id->bbox (ensures persistent id)
        id_to_box = self.tracker.update(rects)  # dict id->bbox

        # 3) Compute full-frame dense flow (if previous frame exists)
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        full_flow = None
        if self.prev_frame is not None:
            prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            full_flow = compute_dense_flow(prev_gray, cur_gray)
            self.last_flow = full_flow
        else:
            # zeros
            full_flow = np.zeros((H, W, 2), dtype="float32")
            self.last_flow = full_flow

        # 4) Associate detections -> tracker IDs (by IoU or centroid nearest)
        # prepare simple association: for each detection find nearest tracked id
        outputs = []
        for det in persons:
            db = det["bbox"]
            # find best matching id based on IoU or centroid distance
            best_id = None
            best_iou = 0.0
            for oid, tbox in id_to_box.items():
                if tbox is None:
                    continue
                # compute IoU
                ix1 = max(db[0], tbox[0]); iy1 = max(db[1], tbox[1])
                ix2 = min(db[2], tbox[2]); iy2 = min(db[3], tbox[3])
                iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
                inter = iw*ih
                union = max(1, (db[2]-db[0])*(db[3]-db[1]) + (tbox[2]-tbox[0])*(tbox[3]-tbox[1]) - inter)
                iou = inter/union
                if iou > best_iou:
                    best_iou = iou; best_id = oid
            if best_id is None:
                # fallback: nearest centroid
                cx = (db[0] + db[2])//2; cy = (db[1] + db[3])//2
                min_d = 1e9
                for oid, tbox in id_to_box.items():
                    if tbox is None:
                        continue
                    tcx = (tbox[0] + tbox[2])//2; tcy = (tbox[1] + tbox[3])//2
                    d = (tcx - cx)**2 + (tcy - cy)**2
                    if d < min_d:
                        min_d = d; best_id = oid

            pid = best_id if best_id is not None else -1

            # 5) compute per-person flow (masked full-frame)
            person_flow = compute_person_flow_global(self.last_flow, db)

            # 6) aggregate motion statistics inside bbox
            dx, dy, mag = aggregate_flow_in_bbox(self.last_flow, db)

            # 7) create per-person heatmap (soft gaussian inside bbox)
            hx, hy, xw, yh = db[0], db[1], db[2]-db[0], db[3]-db[1]
            heatmap = np.zeros((H, W), dtype=np.float32)
            if xw > 0 and yh > 0:
                cx = db[0] + xw//2
                cy = db[1] + yh//2
                # gaussian radius proportional to box size
                rr = max(int(max(xw, yh)*0.6), 30)
                ys, xs = np.ogrid[:H, :W]
                mask = np.exp(-((xs - cx)**2 + (ys - cy)**2) / (2*(rr**2)))
                heatmap = mask * 255.0

            # 8) compute simple fused risk (vision_conf + motion magnitude)
            vision_conf = det.get("confidence", 0.0)
            # normalize motion mag using heuristic (tune for real data)
            motion_norm = min(mag / 10.0, 1.0)
            risk_score = fuse_risk(vision_conf, motion_norm, audio_event_strength=0.0)

            outputs.append({
                "id": pid,
                "bbox": db,
                "keypoints": det.get("keypoints", []),
                "vision_conf": vision_conf,
                "flow": person_flow,
                "motion": {"dx": dx, "dy": dy, "mag": mag},
                "heatmap": heatmap,
                "risk": risk_score
            })

        # 9) update prev frame and return
        self.prev_frame = frame.copy()
        return outputs
