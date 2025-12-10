# yolo_pose.py
from ultralytics import YOLO
import numpy as np

class YOLOPoseDetector:
    def __init__(self, model_path="yolov8s-pose.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Runs YOLO pose detection and returns list of persons:
        [
            {
                "id": 0,
                "bbox": [x1,y1,x2,y2],
                "keypoints": [{"x":..,"y":..,"score":..}, ...],
                "confidence": 0.87
            },
            ...
        ]
        """
        results = self.model(frame, verbose=False)[0]
        persons = []

        boxes = results.boxes
        keypoints = results.keypoints

        for idx, (kp, box) in enumerate(zip(keypoints, boxes)):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            kp_list = []
            for k in kp.data[0]:
                kx, ky, kconf = float(k[0]), float(k[1]), float(k[2])
                kp_list.append({
                    "x": int(kx),
                    "y": int(ky),
                    "score": float(kconf)
                })

            persons.append({
                "id": idx,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "keypoints": kp_list,
                "confidence": conf
            })

        return persons
