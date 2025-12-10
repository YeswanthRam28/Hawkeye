# replay/engine.py
import cv2
import time
import os
import json
from collections import deque
from typing import Callable, List, Dict, Any

class ReplayEngine:
    """
    TimeWarp Replay Engine (callback-enabled)
    - register_callback(fn) to get called with incident_info when an incident completes.
    """

    def __init__(self, pre_seconds=5, post_seconds=5, fps=20):
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.fps = fps

        # PRE circular buffer
        self.max_pre_frames = int(pre_seconds * fps)
        self.buffer = deque(maxlen=self.max_pre_frames)

        # State during incident
        self.recording = False
        self.post_frames_left = 0
        self.incident_frames = []

        # Output folder
        self.root_dir = "incidents"
        os.makedirs(self.root_dir, exist_ok=True)

        # Last saved incident metadata
        self.last_incident_info = None

        # Callbacks (call with one arg: incident_info: dict)
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []

    # ------------------------------------------------------------
    def add_frame(self, frame, metadata):
        """
        Called every frame from the main pipeline.
        Stores (frame, metadata, timestamp) in rolling buffer.
        If incident is active → also stores into incident_frames.
        """
        timestamp = time.time()
        self.buffer.append((frame.copy(), metadata, timestamp))

        if self.recording:
            self.incident_frames.append((frame.copy(), metadata, timestamp))
            self.post_frames_left -= 1

            if self.post_frames_left <= 0:
                return self._finalize_incident()

        return None

    # ------------------------------------------------------------
    def trigger(self):
        """
        Starts incident recording. Copies PRE frames immediately.
        Returns True if new incident started, False if already recording.
        """
        if self.recording:
            return False

        print("[ReplayEngine] Trigger received → starting incident record.")
        self.recording = True
        self.post_frames_left = int(self.post_seconds * self.fps)

        # Copy PRE frames into incident buffer
        self.incident_frames = [(f.copy(), m, t) for (f, m, t) in list(self.buffer)]
        return True

    # ------------------------------------------------------------
    def _finalize_incident(self):
        """
        Internal: saves clip.mp4, frames, evidence.json.
        Calls registered callbacks with incident metadata.
        """
        self.recording = False
        timestamp = int(time.time())

        incident_dir = os.path.join(self.root_dir, f"incident_{timestamp}")
        os.makedirs(incident_dir, exist_ok=True)

        # Save video
        first_frame = self.incident_frames[0][0]
        h, w = first_frame.shape[:2]
        video_path = os.path.join(incident_dir, "clip.mp4")

        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (w, h)
        )

        metadata_json = []
        for idx, (frame, meta, ts) in enumerate(self.incident_frames):
            writer.write(frame)
            frame_path = os.path.join(incident_dir, f"frame_{idx:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            metadata_json.append({
                "index": idx,
                "timestamp": ts,
                "metadata": meta,
                "image": frame_path
            })

        writer.release()

        json_path = os.path.join(incident_dir, "evidence.json")
        with open(json_path, "w") as f:
            json.dump(metadata_json, f, indent=4)

        print(f"[ReplayEngine] Incident saved → {incident_dir}")

        # Save last incident info
        self.last_incident_info = {
            "incident_dir": incident_dir,
            "video": video_path,
            "json": json_path,
            "frame_count": len(self.incident_frames),
            "start_time": metadata_json[0]["timestamp"],
            "end_time": metadata_json[-1]["timestamp"]
        }

        # Notify callbacks (safely; callbacks are synchronous here)
        for cb in list(self._callbacks):
            try:
                cb(self.last_incident_info)
            except Exception as e:
                # callbacks should handle exceptions; log and continue
                print(f"[ReplayEngine] callback error: {e}")

        return self.last_incident_info

    # ------------------------------------------------------------
    def export_incident(self):
        """Return metadata of the most recent incident (or error message)."""
        if not self.last_incident_info:
            return {"error": "No incident recorded yet."}
        return self.last_incident_info

    # ------------------------------------------------------------
    def register_callback(self, fn: Callable[[Dict[str, Any]], None]):
        """
        Register a callback that will be invoked with incident_info dict
        when an incident is finalized. The callback should be quick.
        """
        if fn not in self._callbacks:
            self._callbacks.append(fn)

    def unregister_callback(self, fn: Callable[[Dict[str, Any]], None]):
        if fn in self._callbacks:
            self._callbacks.remove(fn)
