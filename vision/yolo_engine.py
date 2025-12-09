# yolo_engine.py
"""
Full-featured YOLO engine for Hawkeye (Option B).

See original docstring in your repo for usage examples.
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import websockets
import asyncio
from pydantic import BaseModel

# -----------------------------
# Module paths & defaults
# -----------------------------
MODULE_DIR = os.path.dirname(__file__) or "."
STATE_PATH_DEFAULT = os.path.join(MODULE_DIR, "hawkeye_state.json")

API_ONLY = os.environ.get("HAWKEYE_API_ONLY", "") == "1"
STATE_PATH = os.environ.get("HAWKEYE_STATE_FILE", STATE_PATH_DEFAULT)
REPLAY_DIR = os.path.join(MODULE_DIR, "vision_replay")
os.makedirs(REPLAY_DIR, exist_ok=True)


def dump_replay_to_disk(replay_buf, max_images=120):
    """
    Writes last up to max_images frames from replay_buf to REPLAY_DIR as vision_replay_<frame_id>.jpg
    Returns metadata list.
    Safe to call frequently but it rewrites existing files.
    """
    meta = []
    try:
        os.makedirs(REPLAY_DIR, exist_ok=True)
        for item in list(replay_buf)[-max_images:]:
            fid = item["frame_id"]
            fname = os.path.join(REPLAY_DIR, f"vision_replay_{fid}.jpg")
            try:
                cv2.imwrite(fname, item["frame_bgr"])
            except Exception as e:
                print("[dump_replay_to_disk] write failed for", fname, e)
                continue
            meta.append({"frame_id": fid, "file": fname, "timestamp": item["timestamp"]})
    except Exception as e:
        print("[dump_replay_to_disk] error:", e)
    return meta


# -----------------------------
# API-only block (uvicorn import)
# -----------------------------
if API_ONLY:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    import glob
    import shutil

    app = FastAPI(title="Hawkeye Unified Vision API (read-only)")

    # -----------------------------
    # Frame analysis request model
    # -----------------------------
    class FrameAnalysisRequest(BaseModel):
        image_b64: str
        include_pose: bool = True

    # -----------------------------
    # Utilities
    # -----------------------------
    def decode_b64_image(b64str):
        try:
            jpg = base64.b64decode(b64str)
            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception:
            return None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _read_state_file_for_api():
        p = os.environ.get("HAWKEYE_STATE_FILE", STATE_PATH_DEFAULT)
        if not os.path.exists(p):
            return {"status": "no_state", "message": f"state file {p} not found"}
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # initialize YOLO model inside API mode (used by /vision/frame-analysis)
    print("[API] Loading YOLO model for frame-analysis...")
    model_path = os.environ.get("API_YOLO_MODEL", "yolov8s.pt")
    device_id = int(os.environ.get("API_DEVICE_ID", "0"))
    try:
        yolo_model = YOLO(model_path)
        if torch.cuda.is_available():
            yolo_model.to(f"cuda:{device_id}")
    except Exception as e:
        print("[API] failed to load YOLO model:", e)
        yolo_model = None

    model_names = yolo_model.names if yolo_model is not None else {}

    # simple risk function fallback
    def compute_risk(detections):
        score = 0.0
        for d in detections:
            lbl = d.get("class", "").lower()
            if lbl in {"knife", "gun", "pistol"}:
                score += 0.8
            else:
                score += d.get("confidence", 0.0) * 0.2
        return min(score, 1.0)

    # pose server URL from env
    pose_ws_url = os.environ.get("POSE_WS_URL", None)

    @app.get("/status")
    def api_status():
        st = _read_state_file_for_api()
        return {"status": st.get("status", "unknown"), "updated_at": st.get("updated_at")}

    @app.get("/latest")
    def api_latest():
        st = _read_state_file_for_api()
        return st.get("latest_packet", None)

    @app.get("/risk")
    def api_risk():
        st = _read_state_file_for_api()
        return st.get("risk_summary", {"low": 0, "medium": 0, "high": 0})

    @app.get("/roi")
    def api_roi():
        st = _read_state_file_for_api()
        return {"roi": st.get("roi", None)}

    @app.get("/fps")
    def api_fps():
        st = _read_state_file_for_api()
        return {"fps": st.get("fps", 0.0)}

    @app.get("/poses")
    def api_poses():
        st = _read_state_file_for_api()
        return st.get("poses", [])

    @app.get("/pose_events")
    def api_pose_events():
        st = _read_state_file_for_api()
        return st.get("pose_events", [])

    @app.get("/pose_fps")
    def api_pose_fps():
        st = _read_state_file_for_api()
        return {"pose_fps": st.get("pose_fps", 0.0)}

    @app.post("/vision/frame-analysis")
    async def frame_analysis(req: FrameAnalysisRequest):
        if yolo_model is None:
            return {"error": "yolo_model_not_loaded"}
        frame = decode_b64_image(req.image_b64)
        if frame is None:
            return {"error": "invalid_image"}

        # --- YOLO DETECTION ---
        try:
            results = yolo_model.predict(frame, conf=0.3, device=device_id)
            r = results[0]
        except Exception as e:
            return {"error": "yolo_failed", "detail": str(e)}

        detections = []
        if hasattr(r, "boxes") and r.boxes is not None:
            try:
                for b in r.boxes:
                    cls = int(b.cls)
                    conf = float(b.conf)
                    # b.xyxy may be tensor; convert safely
                    try:
                        xyxy = [float(x) for x in b.xyxy[0].tolist()]
                    except Exception:
                        # fallback: attempt to read .xyxy property differently
                        xyxy = [float(x) for x in getattr(b, "xyxy", [0, 0, 0, 0])]
                    detections.append({
                        "class": model_names.get(cls, str(cls)),
                        "confidence": conf,
                        "bbox": xyxy
                    })
            except Exception as e:
                print("[API frame-analysis] parsing boxes error:", e)

        risk_score = compute_risk(detections)

        # ----------------------------
        # Optional Pose Analysis
        # ----------------------------
        poses = []
        pose_events = []
        pose_fps = 0.0

        if req.include_pose and pose_ws_url:
            try:
                async with websockets.connect(pose_ws_url, max_size=2**24) as ws:
                    packet = {
                        "type": "frame",
                        "frame_id": 0,
                        "timestamp": time.time(),
                        "image_jpeg_b64": req.image_b64,
                        "crop": False
                    }
                    await ws.send(json.dumps(packet))
                    resp = await ws.recv()
                    data = json.loads(resp)
                    poses = data.get("poses", [])
                    pose_events = data.get("pose_events", [])
                    pose_fps = data.get("fps", 0.0)
            except Exception as e:
                print("[FrameAnalysis] Pose WS failed:", e)

        return {
            "detections": detections,
            "risk_score": risk_score,
            "poses": poses,
            "pose_events": pose_events,
            "yolo_fps": round(getattr(r, "speed", {}).get("inference", 0.0), 2) if hasattr(r, "speed") else None,
            "pose_fps": pose_fps
        }

    @app.get("/replay")
    def replay_list():
        files = []
        try:
            for fname in sorted(os.listdir(REPLAY_DIR)):
                if fname.endswith(".jpg") and fname.startswith("vision_replay_"):
                    try:
                        fid = int(fname.replace("vision_replay_", "").replace(".jpg", ""))
                    except Exception:
                        fid = None
                    files.append({"frame_id": fid, "file": fname})
        except Exception:
            pass
        return files

    @app.get("/replay/frame/{frame_id}")
    def replay_frame(frame_id: int):
        fname = os.path.join(REPLAY_DIR, f"vision_replay_{frame_id}.jpg")
        if not os.path.exists(fname):
            raise HTTPException(status_code=404, detail="frame not found")
        return FileResponse(fname, media_type="image/jpeg")

    @app.get("/replay/video")
    def replay_video():
        """
        Builds an MP4 video from replay images and returns it.
        Uses a temporary snapshot folder to avoid read/write conflicts.
        """

        import glob
        import shutil

        # 1. Collect frames from REPLAY_DIR
        frame_paths = sorted(
            glob.glob(os.path.join(REPLAY_DIR, "vision_replay_*.jpg")),
            key=lambda x: int(os.path.basename(x).replace("vision_replay_", "").replace(".jpg", ""))
        )

        if not frame_paths:
            raise HTTPException(status_code=404, detail="No replay frames found")

        # 2. Create a temp snapshot directory
        TMP_DIR = os.path.join(REPLAY_DIR, "tmp_video")
        if os.path.exists(TMP_DIR):
            try:
                shutil.rmtree(TMP_DIR)
            except Exception:
                pass
        os.makedirs(TMP_DIR, exist_ok=True)

        # 3. Copy frames safely
        for fp in frame_paths:
            try:
                shutil.copy(fp, TMP_DIR)
            except Exception:
                print("[ReplayVideo] Skipping unreadable frame:", fp)

        # 4. Reload from TMP_DIR only
        snapshot_paths = sorted(
            glob.glob(os.path.join(TMP_DIR, "vision_replay_*.jpg")),
            key=lambda x: int(os.path.basename(x).replace("vision_replay_", "").replace(".jpg", ""))
        )

        if not snapshot_paths:
            raise HTTPException(status_code=500, detail="Replay snapshot empty")

        # 5. Load first frame size
        first_frame = cv2.imread(snapshot_paths[0])
        if first_frame is None:
            raise HTTPException(status_code=500, detail="Replay frames corrupted")

        height, width = first_frame.shape[:2]

        # 6. Output video path
        output_path = os.path.join(REPLAY_DIR, "replay.mp4")

        # 7. Build video
        fps = 12
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for fp in snapshot_paths:
            img = cv2.imread(fp)
            if img is None:
                print("[ReplayVideo] Skipping unreadable frame:", fp)
                continue
            writer.write(img)

        writer.release()

        return FileResponse(output_path, media_type="video/mp4", filename="replay.mp4")


# -----------------------------
# Main YOLO process (full-featured)
# -----------------------------
else:
    # Config & defaults
    DEFAULT_MODEL = "yolov8s.pt"
    DEFAULT_DEVICE = 0
    DEFAULT_IMG_SIZE = 640
    DEFAULT_SKIP = 1
    DEFAULT_REPLAY = 120
    DEFAULT_CONF = 0.3

    DEFAULT_HIGH_RISK_CLASSES = {"knife", "pistol", "gun", "rifle", "firearm", "weapon", "explosion"}

    RISK_COLORS = {
        "low": (0, 255, 0),
        "medium": (0, 255, 255),
        "high": (0, 0, 255)
    }

    # -----------------------------
    # Thread-safe latest pose storage (populated by WS consumer)
    # -----------------------------
    _latest_pose_lock = threading.Lock()
    _latest_pose_state = None

    def _set_latest_pose_state(obj):
        global _latest_pose_state
        with _latest_pose_lock:
            _latest_pose_state = obj

    def _get_latest_pose_state():
        with _latest_pose_lock:
            return None if _latest_pose_state is None else dict(_latest_pose_state)

    # -----------------------------
    # Outgoing WebSocket sender (fusion)
    # -----------------------------
    class WebsocketSender:
        def __init__(self, url: str, max_queue=512):
            self.url = url
            self.queue = Queue(maxsize=max_queue)
            self._running = True
            self._thread = Thread(target=self._start_loop, daemon=True)
            self._thread.start()

        def _start_loop(self):
            asyncio.run(self._ws_loop())

        async def _ws_loop(self):
            reconnect_delay = 1.0
            while self._running:
                try:
                    async with websockets.connect(self.url, max_size=2**24) as ws:
                        print(f"[WebsocketSender] Connected to {self.url}")
                        reconnect_delay = 1.0
                        while self._running:
                            try:
                                packet = await asyncio.get_event_loop().run_in_executor(None, self.queue.get)
                                if packet is None:
                                    return
                                await ws.send(json.dumps(packet))
                            except Exception as e:
                                print("[WebsocketSender] send error:", e)
                                break
                except Exception as e:
                    print(f"[WebsocketSender] connection failed: {e}. Retrying in {reconnect_delay}s")
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 10.0)

        def send(self, packet: dict):
            if not self._running:
                return
            try:
                self.queue.put_nowait(packet)
            except Exception:
                pass

        def stop(self):
            self._running = False
            try:
                self.queue.put_nowait(None)
            except Exception:
                pass

    # -----------------------------
    # Pose WS client (send frames -> receive pose_result)
    # -----------------------------
    class PoseWSClient:
        """
        Sends 'frame' JSONs (base64 jpeg) and listens for 'pose_result' JSON.
        Keeps last pose_result in-memory (thread-safe) via _set_latest_pose_state.
        """
        def __init__(self, url: str, max_queue=4, jpeg_quality=70):
            self.url = url
            self.jpeg_quality = int(jpeg_quality)
            self._running = True
            self._send_queue = deque(maxlen=max_queue)
            self._lock = threading.Lock()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

        def send_frame(self, frame_id, frame_bgr):
            try:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                ok, jpg = cv2.imencode('.jpg', frame_bgr, encode_param)
                if not ok:
                    return
                b64 = base64.b64encode(jpg.tobytes()).decode('ascii')
                msg = {
                    "type": "frame",
                    "frame_id": int(frame_id),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "crop": False,
                    "image_jpeg_b64": b64,
                }
                with self._lock:
                    self._send_queue.append(msg)
            except Exception as e:
                print("[PoseWSClient] encode error:", e)

        def stop(self):
            self._running = False
            try:
                self._thread.join(timeout=0.5)
            except Exception:
                pass

        def _run_loop(self):
            asyncio.run(self._ws_loop())

        async def _ws_loop(self):
            backoff = 1.0
            while self._running:
                try:
                    async with websockets.connect(self.url, max_size=2**24) as ws:
                        print(f"[PoseWS Client] connected to {self.url}")
                        backoff = 1.0
                        consumer = asyncio.create_task(self._consumer(ws))
                        try:
                            while self._running:
                                msg = None
                                with self._lock:
                                    if self._send_queue:
                                        msg = self._send_queue.popleft()
                                if msg is None:
                                    await asyncio.sleep(0.01)
                                    continue
                                try:
                                    await ws.send(json.dumps(msg))
                                except Exception as e:
                                    print("[PoseWS Client] send error:", e)
                                    with self._lock:
                                        if len(self._send_queue) < self._send_queue.maxlen:
                                            self._send_queue.appendleft(msg)
                                    break
                        finally:
                            consumer.cancel()
                except Exception as e:
                    print(f"[PoseWS Client] connection failed: {e}. reconnect in {backoff}s")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 10.0)

        async def _consumer(self, ws):
            try:
                async for msg in ws:
                    try:
                        resp = json.loads(msg)
                    except Exception as e:
                        print("[PoseWS Client] invalid resp:", e)
                        continue
                    if resp.get("type") == "pose_result":
                        _set_latest_pose_state(resp)
            except Exception:
                pass

    # -----------------------------
    # Geometry helpers, ROI, risk, state file helpers
    # -----------------------------
    def point_in_poly(x, y, poly):
        inside = False
        n = len(poly)
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[(i + 1) % n]
            intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi)
            if intersect:
                inside = not inside
        return inside

    def bbox_center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def auto_generate_roi(frame, min_area_ratio=0.05):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [(0, 0), (w, 0), (w, h), (0, h)]

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        c = contours[0]
        area = cv2.contourArea(c)
        if area < min_area_ratio * (w * h):
            return [(0, 0), (w, 0), (w, h), (0, h)]

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        poly = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
        if len(poly) < 3:
            return [(0, 0), (w, 0), (w, h), (0, h)]
        return poly

    def compute_risk_for_object(obj_label, obj_conf, high_risk_set, conf_threshold_map):
        """
        Unified risk computation.
        - High risk if class is in high_risk_set
        - Otherwise scale based on confidence thresholds
        """

        label_l = obj_label.lower()

        # Base thresholds
        med_conf = 0.5
        high_conf = 0.8

        # If user provided per-class thresholds, adapt
        if conf_threshold_map and label_l in conf_threshold_map:
            base = conf_threshold_map[label_l]
            med_conf = max(0.3, base)
            high_conf = min(0.99, base + 0.25)

        # Hard high-risk classes
        if label_l in high_risk_set:
            return "high"

        # Confidence-based risk
        if obj_conf >= high_conf:
            return "high"
        if obj_conf >= med_conf:
            return "medium"

        return "low"


    import time

    def write_state_file(state_obj, path=STATE_PATH):
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state_obj, f, indent=2, default=str)
        except Exception as e:
            print("[StateFile] temp write error:", e)
            return

        # try atomic replace with a few retries to avoid permission contention on Windows
        for _ in range(5):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.01)  # 10ms retry
        print("[StateFile] write error: access denied after retries")

    def read_state_file(path=STATE_PATH):
        try:
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print("[StateFile] read error:", e)
            return None

    # -----------------------------
    # Start external FastAPI server (separate process)
    # -----------------------------
    def start_fastapi_server(port=9000, state_path=STATE_PATH):
        module_dir = os.path.dirname(__file__) or "."
        module_name = os.path.splitext(os.path.basename(__file__))[0]

        print(f"[FastAPI] Launching API on port {port} using state-file: {state_path}")

        env = os.environ.copy()
        env["HAWKEYE_API_ONLY"] = "1"
        env["HAWKEYE_STATE_FILE"] = state_path
        env["POSE_WS_URL"] = env.get("POSE_WS_URL", "")

        cmd = [
            sys.executable, "-m", "uvicorn",
            f"{module_name}:app",
            "--host", "0.0.0.0",
            f"--port={port}",
            "--log-level", "info",
            "--reload"
        ]

        try:
            subprocess.Popen(cmd, cwd=module_dir, env=env)
            print(f"[FastAPI] Server ready at http://localhost:{port}/docs")
        except Exception as e:
            print("[FastAPI] Failed to start:", e)
            print("Try manually running:")
            print(f"  HAWKEYE_API_ONLY=1 HAWKEYE_STATE_FILE={state_path} python -m uvicorn {module_name}:app --host 0.0.0.0 --port {port}")

    # -----------------------------
    # Main YOLO engine run function
    # -----------------------------
    def run_yolo(
            source=0,
            model_path=DEFAULT_MODEL,
            device=DEFAULT_DEVICE,
            imgsz=DEFAULT_IMG_SIZE,
            fusion_url=None,
            skip=DEFAULT_SKIP,
            allowed_classes=None,
            class_conf_map=None,
            enable_auto_roi=False,
            roi_polygon=None,
            replay_frames=DEFAULT_REPLAY,
            log_json=False,
            state_path=STATE_PATH,
            pose_ws_url=None,
            pose_send_every=4
    ):
        print("[YOLO Engine] Loading model:", model_path)
        model = YOLO(model_path)

        cuda_available = torch.cuda.is_available()
        half = False
        if cuda_available:
            model.to(f"cuda:{device}")
            try:
                model.model.half()
                half = True
            except Exception:
                half = False
        else:
            print("[YOLO Engine] CUDA not available. Using CPU.")

        # Warmup
        try:
            model.predict(np.zeros((1, 3, imgsz, imgsz), dtype=np.uint8))
        except Exception:
            pass

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        # optional websocket sender
        ws_sender = None
        if fusion_url:
            print("[YOLO Engine] Starting websocket sender to", fusion_url)
            ws_sender = WebsocketSender(fusion_url)

        # pose ws client (sends frames, receives pose_result)
        pose_client = None
        if pose_ws_url:
            try:
                pose_client = PoseWSClient(pose_ws_url)
            except Exception as e:
                print("[YOLO Engine] Failed to start PoseWSClient:", e)
                pose_client = None

        replay_buf = deque(maxlen=replay_frames)
        active_roi = roi_polygon
        roi_computed = False

        fps_smooth = None
        last_time = time.time()
        frame_id = 0

        COLORS = {}

        def get_color_for_risk(risk_level):
            return RISK_COLORS.get(risk_level, (255, 255, 255))

        def random_color_for_class(cidx):
            if cidx not in COLORS:
                COLORS[cidx] = tuple(int(x) for x in np.random.randint(0, 255, size=3))
            return COLORS[cidx]

        # Initial state so API has something to read immediately
        initial_state = {
            "status": "initializing",
            "latest_packet": None,
            "risk_summary": {"low": 0, "medium": 0, "high": 0},
            "roi": active_roi,
            "fps": 0.0,
            "poses": [],
            "pose_events": [],
            "pose_fps": 0.0,
            "updated_at": datetime.utcnow().isoformat() + "Z"
        }
        write_state_file(initial_state, state_path)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[YOLO Engine] End of stream or cannot fetch frame.")
                    break

                frame_id += 1
                h, w = frame.shape[:2]
                do_process = (frame_id % skip == 0)

                packet = None
                if do_process:
                    # auto-ROI: compute once from first frame (if enabled)
                    if enable_auto_roi and not roi_computed:
                        active_roi = auto_generate_roi(frame)
                        roi_computed = True
                        print("[YOLO Engine] Auto-ROI computed:", active_roi)

                    results = model.predict(
                        source=frame,
                        stream=False,
                        device=f"cuda:{device}" if cuda_available else "cpu",
                        imgsz=imgsz,
                        conf=DEFAULT_CONF,
                        half=half
                    )
                    r = results[0]

                    # parse boxes safely
                    objects = []
                    try:
                        boxes = r.boxes
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        cls = boxes.cls.cpu().numpy()
                    except Exception:
                        xyxy, confs, cls = np.array([]), np.array([]), np.array([])

                    # build detections list with filtering
                    for i in range(len(confs)):
                        x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                        conf_val = float(confs[i])
                        class_id = int(cls[i]) if len(cls) > i else -1
                        label = model.names[class_id] if (class_id >= 0 and class_id < len(model.names)) else str(class_id)

                        # per-class global filtering
                        if allowed_classes is not None and len(allowed_classes) > 0:
                            if label.lower() not in allowed_classes:
                                continue

                        # per-class minimum confidence threshold (if provided)
                        if class_conf_map and label.lower() in class_conf_map:
                            if conf_val < class_conf_map[label.lower()]:
                                continue

                        # ROI check (if active)
                        if active_roi is not None:
                            cx, cy = bbox_center((x1, y1, x2, y2))
                            if not point_in_poly(cx, cy, active_roi):
                                continue

                        # compute risk
                        risk = compute_risk_for_object(label, conf_val, DEFAULT_HIGH_RISK_CLASSES, class_conf_map)

                        objects.append({
                            "label": label,
                            "class_id": class_id,
                            "confidence": round(conf_val, 3),
                            "bbox": [x1, y1, x2, y2],
                            "risk": risk
                        })

                    # FPS smoothing
                    now = time.time()
                    dt = now - last_time
                    fps = 1.0 / dt if dt > 0 else 0.0
                    last_time = now
                    fps_smooth = fps if fps_smooth is None else fps_smooth * 0.8 + fps * 0.2

                    # Build packet
                    packet = {
                        "source": str(source),
                        "frame_id": frame_id,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "fps": round(fps_smooth, 2),
                        "objects": objects,
                        "frame_shape": {"height": h, "width": w},
                        "roi": active_roi
                    }

                    # send full frame to pose server every N frames (non-blocking)
                    if pose_client and (frame_id % pose_send_every == 0):
                        try:
                            send_w = min(960, w)
                            send_h = int((send_w / w) * h)
                            send_img = cv2.resize(frame, (send_w, send_h))
                            pose_client.send_frame(frame_id, send_img)
                        except Exception as e:
                            print("[YOLO Engine] failed to send frame to pose:", e)

                    # merge latest pose if available (in-memory via WS consumer)
                    pose_state = _get_latest_pose_state()
                    poses = []
                    pose_events = []
                    pose_fps = 0.0
                    if pose_state:
                        poses = pose_state.get("poses", []) or []
                        pose_events = pose_state.get("pose_events", []) or []
                        pose_fps = pose_state.get("fps", pose_state.get("pose_fps", 0.0))

                    packet["poses"] = poses
                    packet["pose_events"] = pose_events
                    packet["pose_fps"] = pose_fps

                    # push to replay buffer (store small BGR frame + packet)
                    replay_buf.append({
                        "frame_id": frame_id,
                        "timestamp": packet["timestamp"],
                        "frame_bgr": frame.copy(),
                        "packet": packet
                    })

                    # Dump replay thumbnails for API (periodically)
                    if frame_id % 5 == 0:
                        dump_replay_to_disk(replay_buf)

                    # send to fusion if configured
                    if ws_sender:
                        ws_sender.send(packet)

                    # optional local logging
                    if log_json:
                        try:
                            os.makedirs("logs", exist_ok=True)
                            fname = os.path.join("logs", f"hawkeye_packet_{frame_id}.json")
                            with open(fname, "w", encoding="utf-8") as f:
                                json.dump(packet, f, indent=2)
                        except Exception as e:
                            print("[YOLO Engine] failed to write log:", e)

                    # Draw overlays
                    overlay = frame.copy()
                    # draw ROI polygon if present
                    if active_roi is not None:
                        pts = np.array(active_roi, np.int32)
                        cv2.polylines(overlay, [pts], isClosed=True, color=(200, 200, 200), thickness=2)
                        mask = overlay.copy()
                        cv2.fillPoly(mask, [pts], color=(40, 40, 40))
                        cv2.addWeighted(mask, 0.08, overlay, 0.92, 0, overlay)

                    # draw detections with risk coloring and labels
                    risk_counts = {"low": 0, "medium": 0, "high": 0}
                    for obj in objects:
                        x1, y1, x2, y2 = map(int, obj["bbox"])
                        risk = obj["risk"]
                        risk_counts[risk] += 1

                        color = get_color_for_risk(risk)
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

                        label_txt = f"{obj['label']} {obj['confidence']:.2f} [{risk[0].upper()}]"
                        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                        cv2.putText(overlay, label_txt, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # if pose info present, draw simple pose overlay (points)
                    if poses:
                        try:
                            for kps in poses:
                                for (x, y) in kps:
                                    cv2.circle(overlay, (int(x), int(y)), 3, (255, 255, 0), -1)
                        except Exception:
                            pass

                    # update unified state file (includes pose merge)
                    state_obj = {
                        "status": "running",
                        "latest_packet": packet,
                        "risk_summary": risk_counts,
                        "roi": active_roi,
                        "fps": packet["fps"],
                        "poses": poses,
                        "pose_events": pose_events,
                        "pose_fps": pose_fps,
                        "updated_at": datetime.utcnow().isoformat() + "Z"
                    }
                    write_state_file(state_obj, state_path)

                    # HUD
                    hud_text = f"YOLOv8 | FPS: {packet['fps']:.1f} | Detected: {len(objects)} | R:{risk_counts['high']} M:{risk_counts['medium']} L:{risk_counts['low']}"
                    cv2.putText(overlay, hud_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                    # show processed frame
                    cv2.imshow("Hawkeye - Vision (YOLO)", overlay)
                else:
                    # show raw frame while skipping processing
                    cv2.imshow("Hawkeye - Vision (YOLO)", frame)

                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[YOLO Engine] Quit received")
                    break
                elif key == ord("s"):
                    # save latest packet if present
                    if packet:
                        fname = f"capture_frame_{frame_id}.json"
                        with open(fname, "w", encoding="utf-8") as f:
                            json.dump(packet, f, indent=2)
                        print("[YOLO Engine] Saved packet to", fname)
                elif key == ord("r"):
                    # replay last N frames (non-blocking simple loop)
                    print("[YOLO Engine] Starting replay of buffer (frames:", len(replay_buf), ")")
                    for item in list(replay_buf):
                        cv2.imshow("Hawkeye - Replay", item["frame_bgr"])
                        if cv2.waitKey(50) & 0xFF == ord("q"):
                            break
                    cv2.destroyWindow("Hawkeye - Replay")
                elif key == ord("c"):
                    # print counts / quick debugging
                    print(f"[YOLO Engine] Replay buffer size: {len(replay_buf)}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            final_state = {
                "status": "stopped",
                "latest_packet": None,
                "risk_summary": {"low": 0, "medium": 0, "high": 0},
                "roi": active_roi,
                "fps": 0.0,
                "poses": [],
                "pose_events": [],
                "pose_fps": 0.0,
                "updated_at": datetime.utcnow().isoformat() + "Z"
            }
            write_state_file(final_state, state_path)
            if ws_sender:
                ws_sender.stop()
            if pose_client:
                pose_client.stop()
            print("[YOLO Engine] Shutdown complete.")

    # -----------------------------
    # CLI helpers
    # -----------------------------
    def parse_class_conf_map(s: str):
        if not s:
            return {}
        items = [x.strip() for x in s.split(",") if x.strip()]
        out = {}
        for it in items:
            if ":" in it:
                k, v = it.split(":", 1)
                try:
                    out[k.strip().lower()] = float(v.strip())
                except Exception:
                    pass
        return out

    def parse_roi_string(s: str):
        if not s:
            return None
        pts = []
        for part in s.split(";"):
            if not part.strip():
                continue
            try:
                x, y = part.strip().split(",")
                pts.append((int(float(x)), int(float(y))))
            except Exception:
                continue
        return pts if len(pts) >= 3 else None

    def parse_args():
        p = argparse.ArgumentParser(prog="yolo_engine")
        p.add_argument("--source", type=str, default="0", help="0 for webcam or path to video file.")
        p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="YOLO model path or name.")
        p.add_argument("--device", type=int, default=DEFAULT_DEVICE, help="CUDA device id.")
        p.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE, help="Inference image size (512 or 640).")
        p.add_argument("--skip", type=int, default=DEFAULT_SKIP, help="Process every Nth frame.")
        p.add_argument("--fusion", type=str, default=None, help="Optional websocket URL to send JSON packets.")
        p.add_argument("--classes", type=str, default=None, help="Comma-separated allowed classes (e.g., person,backpack).")
        p.add_argument("--class_confs", type=str, default=None,
                       help="Comma-separated per-class minimum confidences e.g. 'person:0.45,backpack:0.4'.")
        p.add_argument("--auto_roi", action="store_true", help="Attempt to auto-generate ROI polygon from frame edges.")
        p.add_argument("--roi", type=str, default=None,
                       help="Manual ROI polygon as semicolon-separated points 'x1,y1;x2,y2;...' (overrides auto).")
        p.add_argument("--replay", type=int, default=DEFAULT_REPLAY, help="Number of frames to keep in replay buffer.")
        p.add_argument("--log-json", action="store_true", help="Save JSON packets locally into logs/ directory.")
        p.add_argument("--state-file", type=str, default=STATE_PATH_DEFAULT, help="Path to unified hawkeye state JSON file.")
        p.add_argument("--api-port", type=int, default=9000, help="Port to launch API server on.")
        p.add_argument("--spawn-api", action="store_true", help="Spawn the external API subprocess automatically.")
        p.add_argument("--pose-ws-url", type=str, default=None, help="WebSocket URL of pose server (e.g. ws://localhost:8003).")
        p.add_argument("--pose-send-every", type=int, default=4, help="Send full frame to pose WS every N frames (default 4).")
        return p.parse_args()

    def main():
        args = parse_args()
        source = 0 if args.source == "0" else args.source

        allowed_classes = None
        if args.classes:
            allowed_classes = [c.strip().lower() for c in args.classes.split(",") if c.strip()]

        class_conf_map = parse_class_conf_map(args.class_confs)
        roi_poly = parse_roi_string(args.roi) if args.roi else None

        state_path = args.state_file or STATE_PATH_DEFAULT

        if args.log_json:
            os.makedirs("logs", exist_ok=True)

        if args.spawn_api:
            start_fastapi_server(port=args.api_port, state_path=state_path)

        run_yolo(
            source=source,
            model_path=args.model,
            device=args.device,
            imgsz=args.imgsz,
            fusion_url=args.fusion,
            skip=args.skip,
            allowed_classes=allowed_classes,
            class_conf_map=class_conf_map,
            enable_auto_roi=args.auto_roi,
            roi_polygon=roi_poly,
            replay_frames=args.replay,
            log_json=args.log_json,
            state_path=state_path,
            pose_ws_url=args.pose_ws_url,
            pose_send_every=args.pose_send_every
        )

    if __name__ == "__main__":
        main()
