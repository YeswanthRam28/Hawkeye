# simple_tracker.py
# Lightweight centroid tracker for stable IDs across frames
import numpy as np
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=80):
        self.next_object_id = 0
        self.objects = OrderedDict()   # id -> centroid (x,y)
        self.boxes = OrderedDict()     # id -> bbox
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, bbox):
        # bbox: [x1,y1,x2,y2]
        cX = int((bbox[0] + bbox[2]) / 2.0)
        cY = int((bbox[1] + bbox[3]) / 2.0)
        self.objects[self.next_object_id] = (cX, cY)
        self.boxes[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.boxes[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        """
        rects: list of bboxes [[x1,y1,x2,y2], ...]
        returns mapping id -> bbox (some may be None)
        """
        if len(rects) == 0:
            # mark all as disappeared
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return dict(self.boxes)

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for bbox in rects:
                self.register(bbox)
            return dict(self.boxes)

        # build distance matrix
        object_ids = list(self.objects.keys())
        object_centroids = np.array([self.objects[i] for i in object_ids])

        D = np.linalg.norm(object_centroids[:, None, :] - input_centroids[None, :, :], axis=2)

        # greedy assignment
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assigned_rows, assigned_cols = set(), set()
        assigned = {}

        for r, c in zip(rows, cols):
            if r in assigned_rows or c in assigned_cols:
                continue
            if D[r, c] > self.max_distance:
                continue
            oid = object_ids[r]
            self.objects[oid] = tuple(input_centroids[c])
            self.boxes[oid] = rects[c]
            self.disappeared[oid] = 0
            assigned_rows.add(r)
            assigned_cols.add(c)
            assigned[oid] = rects[c]

        # register new detections not matched
        for i in range(len(rects)):
            if i not in assigned_cols:
                self.register(rects[i])

        # mark disappeared
        for r_idx, oid in enumerate(object_ids):
            if r_idx not in assigned_rows:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

        return dict(self.boxes)
