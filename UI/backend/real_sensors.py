
import sys
import os
import time
import cv2
import numpy as np
import traceback

# Add project root to path so we can import from sibling directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- Import Fusion Engine ---
try:
    from fusion_engine.fusion_core import FusionEngine
    from fusion_engine.feature_schema import SensorPacket, VisionData, AudioData, MotionData
except ImportError:
    print("WARNING: Could not import FusionEngine. Is 'fusion_engine' in path?")
    FusionEngine = None

# --- Import Vision (YOLO) ---
try:
    from ultralytics import YOLO
    # Locate model file in vision directory
    VISION_DIR = os.path.join(PROJECT_ROOT, "vision")
    YOLO_MODEL_PATH = os.path.join(VISION_DIR, "yolov8s.pt")
except ImportError:
    print("WARNING: Could not import ultralytics.")
    YOLO = None

# --- Import Motion ---
try:
    # motion_ai/motion.py
    from motion_ai.motion import compute_optical_flow
    from motion_ai.metrics import compute_motion_metrics
except ImportError:
    print("WARNING: Could not import motion_ai.")
    compute_optical_flow = None


class RealSensors:
    def __init__(self):
        self.yolo_model = None
        self.fusion_engine = None
        
        self.prev_frame = None
        self.prev_gray = None
        
        # State
        self.last_result = {}

    def init(self):
        print("[RealSensors] Initializing...")
        
        # 1. Load YOLO
        if YOLO and os.path.exists(YOLO_MODEL_PATH):
            try:
                print(f"[RealSensors] Loading YOLO from {YOLO_MODEL_PATH}...")
                self.yolo_model = YOLO(YOLO_MODEL_PATH)
                # warmup
                self.yolo_model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
            except Exception as e:
                print(f"[RealSensors] YOLO init failed: {e}")
        else:
            print("[RealSensors] YOLO model not found or library missing.")

        # 2. Init Fusion
        if FusionEngine:
            try:
                self.fusion_engine = FusionEngine()
                print("[RealSensors] FusionEngine initialized.")
            except Exception as e:
                print(f"[RealSensors] FusionEngine init failed: {e}")

        print("[RealSensors] Ready.")

    def process(self, frame_bgr, audio_chunk=None):
        """
        Process a single frame through Vision, Motion, (Audio stub), and Fusion.
        Returns a dict with:
          - risk_score
          - overlays (boxes, skeletons, etc.)
          - fusion_data
        """
        timestamp = time.time()
        
        # --- 1. Vision Analysis ---
        vision_data = VisionData(object_count=0, threat_score=0.0, bounding_boxes=[])
        overlays = {"boxes": [], "skeletons": [], "heatmap": []}
        
        if self.yolo_model:
            try:
                results = self.yolo_model.predict(frame_bgr, conf=0.4, verbose=False)
                r = results[0]
                
                # Parse boxes
                boxes = []
                obj_count = 0
                max_conf = 0.0
                has_threat = False
                
                # Collect boxes for frontend
                frontend_boxes = []
                
                if r.boxes:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        label = self.yolo_model.names[cls_id]
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].tolist()
                        
                        obj_count += 1
                        if conf > max_conf:
                            max_conf = conf
                            
                        # Simple threat check
                        is_threat = label.lower() in ["knife", "gun", "weapon", "fire"]
                        if is_threat:
                            has_threat = True
                        
                        # Add to vision data
                        boxes.append(xyxy)
                        
                        # Add to frontend overlays
                        # Frontend expects: x, y, w, h
                        x1, y1, x2, y2 = xyxy
                        frontend_boxes.append({
                            "id": f"{label}_{obj_count}",
                            "x": int(x1),
                            "y": int(y1),
                            "w": int(x2 - x1),
                            "h": int(y2 - y1),
                            "label": label,
                            "conf": round(conf, 2),
                            "color": "red" if is_threat else "lime"
                        })
                
                overlays["boxes"] = frontend_boxes
                
                # Calc threat score
                v_score = 0.0
                if has_threat:
                    v_score = 0.9
                else:
                    v_score = 0.1 * min(obj_count, 5) # Crowd factor
                    
                vision_data = VisionData(
                    object_count=obj_count,
                    threat_score=round(v_score, 2),
                    bounding_boxes=boxes
                )
            except Exception as e:
                print(f"[RealSensors] YOLO error: {e}")

        # --- 2. Motion Analysis ---
        motion_data = MotionData(speed=0.0, acceleration=0.0, jerk=0.0)
        
        if compute_optical_flow:
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                if self.prev_gray is not None:
                    mag, ang = compute_optical_flow(self.prev_gray, gray)
                    metrics = compute_motion_metrics(mag)
                    
                    # Calculate Dynamics
                    current_speed = float(metrics.get("panic_score", 0.0))
                    
                    # Delta time (approximate based on loop)
                    dt = timestamp - self.last_timestamp if hasattr(self, 'last_timestamp') else 0.06
                    self.last_timestamp = timestamp

                    # Acceleration (change in speed)
                    accel = (current_speed - self.last_speed) / dt if hasattr(self, 'last_speed') else 0.0
                    
                    # Jerk (change in acceleration)
                    jerk = (accel - self.last_accel) / dt if hasattr(self, 'last_accel') else 0.0
                    
                    self.last_speed = current_speed
                    self.last_accel = accel

                    motion_data = MotionData(
                        speed=round(current_speed, 2),
                        acceleration=round(accel, 2),
                        jerk=round(jerk, 2)
                    )
                
                self.prev_gray = gray
            except Exception as e:
                print(f"[RealSensors] Motion error: {e}")


        # --- 3. Audio Analysis ---
        # (Stub for now, or use volume if audio_chunk provided)
        audio_score = 0.0
        if audio_chunk is not None:
            # Simple volume calculation
            try:
                rms = np.sqrt(np.mean(audio_chunk**2))
                # Normalize arbitrary RMS (assuming 16-bit audio)
                vol = min(rms / 10000.0, 1.0)
                if vol > 0.5:
                    audio_score = 0.8 # Loud noise
                else:
                    audio_score = 0.1
            except Exception:
                pass
        
        audio_data = AudioData(
            volume_db=0.0, # Not calculating true DB
            anomaly_score=round(audio_score, 2),
            keywords_detected=[]
        )

        # --- 4. Fusion ---
        final_risk = 0.0
        fused_result = {}
        
        if self.fusion_engine:
            try:
                packet = SensorPacket(
                    vision=vision_data,
                    audio=audio_data,
                    motion=motion_data,
                    timestamp=timestamp
                )
                
                # Process
                fused_result = self.fusion_engine.process(packet)
                
                # Extract score logic matches api_server.py logic
                if isinstance(fused_result, dict):
                    final_risk = fused_result.get("final_score") or fused_result.get("fused_threat") or 0.0
                else:
                    final_risk = 0.5 # Default error
                    
            except Exception as e:
                print(f"[RealSensors] Fusion error: {e}")
                # Fallback to simple max
                final_risk = max(vision_data.threat_score, audio_data.anomaly_score)

        return {
            "risk_score": final_risk,
            "overlays": overlays,
            "fusion_debug": fused_result
        }

