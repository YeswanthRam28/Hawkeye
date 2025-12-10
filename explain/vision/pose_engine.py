# pose_engine.py
"""
Pose engine using YOLOv8-Pose.

Modes:
  1) WebSocket server mode (no camera): accept frames from YOLO, return pose JSON.
     python pose_engine.py --server-ws --ws-host 0.0.0.0 --ws-port 8003

  2) Standalone capture mode (default): open camera, process frames, write hawkeye_pose_state.json,
     optionally spawn API or forward pose packets to a fusion websocket.
     python pose_engine.py --source 0 --spawn-api --api-port 9100 --fusion ws://host:port
Outputs:
  - atomic JSON state file (default: hawkeye_pose_state.json)
  - optional FastAPI process when --spawn-api is set
"""

import os
import sys
import time
import json
import argparse
import subprocess
import threading
from datetime import datetime
from collections import deque
from queue import Queue

from typing import List, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import websockets
import base64
import asyncio

# -----------------------------
# Config & defaults
# -----------------------------
MODULE_DIR = os.path.dirname(__file__) or "."
STATE_PATH_DEFAULT = os.path.join(MODULE_DIR, "hawkeye_pose_state.json")
REPLAY_DIR_DEFAULT = os.path.join(MODULE_DIR, "pose_replay")
API_PORT_DEFAULT = 9100

POSE_MODEL_PATH = "yolov8s-pose.pt"
DEFAULT_DEVICE = 0
POSE_IMG_SIZE = 640
DEFAULT_CONF = 0.3

# Detection thresholds (tweakable)
FALL_ANGLE_THRESHOLD = 35       # degrees
RUNNING_STRIDE_THRESHOLD = 40   # pixels
HANDS_UP_Y_OFFSET = -20         # pixels
AGGRESSION_ARM_EXTENSION = 40   # pixels

# WebSocket server defaults for receiving frames from YOLO
POSE_WS_HOST_DEFAULT = "0.0.0.0"
POSE_WS_PORT_DEFAULT = 8003

# -----------------------------
# Geometry helpers
# -----------------------------
def angle(p1, p2, p3):
    a = np.array(p1, dtype=np.float32)
    b = np.array(p2, dtype=np.float32)
    c = np.array(p3, dtype=np.float32)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cosine = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))

def dist(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))

# -----------------------------
# Pose detectors
# -----------------------------
def detect_fall(kps: List[tuple]) -> bool:
    try:
        nose = kps[0]
        left_hip = kps[11]
        right_hip = kps[12]
        torso_angle = abs(angle(left_hip, nose, right_hip))
        return torso_angle < FALL_ANGLE_THRESHOLD
    except Exception:
        return False

def detect_running(kps: List[tuple]) -> bool:
    try:
        left_ankle = kps[15]
        right_ankle = kps[16]
        stride = dist(left_ankle, right_ankle)
        return stride > RUNNING_STRIDE_THRESHOLD
    except Exception:
        return False

def detect_hands_up(kps: List[tuple]) -> bool:
    try:
        left_shoulder = kps[5]
        right_shoulder = kps[6]
        shoulder_y = min(left_shoulder[1], right_shoulder[1])
        left_wrist = kps[9]
        right_wrist = kps[10]
        if left_wrist[1] < shoulder_y + HANDS_UP_Y_OFFSET:
            return True
        if right_wrist[1] < shoulder_y + HANDS_UP_Y_OFFSET:
            return True
    except Exception:
        pass
    return False

def detect_aggression(kps: List[tuple]) -> bool:
    try:
        left_elbow = kps[7]
        left_wrist = kps[9]
        right_elbow = kps[8]
        right_wrist = kps[10]
        left_ext = dist(left_elbow, left_wrist)
        right_ext = dist(right_elbow, right_wrist)
        return (left_ext > AGGRESSION_ARM_EXTENSION) or (right_ext > AGGRESSION_ARM_EXTENSION)
    except Exception:
        return False

# -----------------------------
# PoseEngine
# -----------------------------
class PoseEngine:
    def __init__(self, model_path=POSE_MODEL_PATH, device=DEFAULT_DEVICE, imgsz=POSE_IMG_SIZE, overlay=True):
        print("[PoseEngine] Loading model:", model_path)
        self.model = YOLO(model_path)
        self.overlay = overlay

        cuda_available = torch.cuda.is_available()
        self.device = f"cuda:{device}" if cuda_available else "cpu"
        if cuda_available:
            try:
                self.model.to(self.device)
            except Exception:
                pass

        self.imgsz = imgsz
        self.last_fps = 0.0
        self.last_frame_time = time.time()

    def process_frame(self, frame, conf=DEFAULT_CONF):
        """
        Run pose model on frame and return (poses, events).
        poses: list of lists of (x,y) tuples (17 keypoints, padded if necessary)
        events: list of event lists per detected human
        """
        results = self.model.predict(source=frame, stream=False, device=self.device, imgsz=self.imgsz, conf=conf)
        r = results[0]

        poses = []
        events = []

        if hasattr(r, "keypoints") and r.keypoints is not None:
            try:
                for p in r.keypoints:
                    try:
                        arr = p.xy.cpu().numpy()
                        # arr can be (1,17,2) or (17,2) depending on ultralytics return shape
                        if arr.ndim == 3:
                            kps_arr = arr[0]
                        elif arr.ndim == 2:
                            kps_arr = arr
                        else:
                            kps_arr = arr.reshape(-1, 2)
                        kps = [(float(x), float(y)) for x, y in kps_arr]
                    except Exception:
                        continue

                    if len(kps) < 17:
                        kps = kps + [(0.0, 0.0)] * (17 - len(kps))

                    ev = []
                    if detect_fall(kps):
                        ev.append("fall")
                    if detect_running(kps):
                        ev.append("running")
                    if detect_hands_up(kps):
                        ev.append("hands_up")
                    if detect_aggression(kps):
                        ev.append("aggression")

                    poses.append(kps)
                    events.append(ev)
            except Exception:
                pass

        now = time.time()
        dt = now - self.last_frame_time if now != self.last_frame_time else 1e-9
        self.last_frame_time = now
        self.last_fps = 1.0 / dt if dt > 0 else 0.0

        return poses, events

    def annotate(self, frame, poses, events):
        out = frame.copy()
        try:
            for kps, ev in zip(poses, events):
                for (x, y) in kps:
                    cv2.circle(out, (int(x), int(y)), 3, (255, 255, 0), -1)
                if ev:
                    label = ",".join(ev)
                    x, y = kps[0]
                    cv2.putText(out, label, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        except Exception:
            pass
        cv2.putText(out, f"Pose FPS: {self.last_fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        return out

# -----------------------------
# Atomic state helpers
# -----------------------------
def write_state_file(obj, path):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as e:
        print("[PoseState] write error:", e)

def read_state_file(path):
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("[PoseState] read error:", e)
        return None

# -----------------------------
# Replay dump helper
# -----------------------------
def dump_replay_to_disk(replay_buf, replay_dir=REPLAY_DIR_DEFAULT, max_images=120):
    try:
        os.makedirs(replay_dir, exist_ok=True)
        meta = []
        for item in list(replay_buf)[-max_images:]:
            fid = item["frame_id"]
            fname = os.path.join(replay_dir, f"pose_replay_{fid}.jpg")
            cv2.imwrite(fname, item["frame_bgr"])
            meta.append({"frame_id": fid, "file": fname, "timestamp": item["timestamp"]})
        return meta
    except Exception as e:
        print("[PoseReplay] dump error:", e)
        return []

def start_pose_api_server(port=9100, state_path=STATE_PATH_DEFAULT, replay_dir=REPLAY_DIR_DEFAULT):
    print(f"[PoseAPI] Launching external Pose API on port {port}")

    module_dir = os.path.dirname(__file__) or "."
    module_name = os.path.splitext(os.path.basename(__file__))[0]

    env = os.environ.copy()
    env["POSE_API_ONLY"] = "1"
    env["POSE_STATE_FILE"] = state_path
    env["POSE_REPLAY_DIR"] = replay_dir

    cmd = [
        sys.executable, "-m", "uvicorn",
        f"{module_name}:app",
        "--host", "0.0.0.0",
        f"--port={port}",
        "--log-level", "warning"
    ]

    try:
        subprocess.Popen(cmd, cwd=module_dir, env=env)
        print(f"[PoseAPI] Server started at http://localhost:{port}")
    except Exception as e:
        print("[PoseAPI] failed to start subprocess:", e)

# -----------------------------
# Optional outgoing WebSocket sender (for fusion forwarding)
# -----------------------------
class WebsocketSender:
    def __init__(self, url: str, max_queue=512):
        self.url = url
        self.queue = Queue(maxsize=max_queue)
        self._running = True
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()

    def _start_loop(self):
        asyncio.run(self._ws_loop())

    async def _ws_loop(self):
        reconnect_delay = 1.0
        while self._running:
            try:
                async with websockets.connect(self.url, max_size=2**24) as ws:
                    print(f"[PoseWS Sender] Connected to {self.url}")
                    reconnect_delay = 1.0
                    loop = asyncio.get_event_loop()
                    while self._running:
                        try:
                            # blocking Queue.get in executor
                            packet = await loop.run_in_executor(None, self.queue.get)
                            if packet is None:
                                return
                            await ws.send(json.dumps(packet))
                        except Exception as e:
                            print("[PoseWS Sender] send error:", e)
                            break
            except Exception as e:
                print(f"[PoseWS Sender] connection failed: {e}. Retrying in {reconnect_delay}s")
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
# WebSocket server handler (receives frames from YOLO, returns pose_result)
# -----------------------------
async def _pose_ws_handler(websocket, path=None):
    """
    Handler tolerant of different websockets versions:
      - older websockets: handler(websocket, path)
      - newer websockets: handler(websocket)
    """
    # remote addr: websocket.remote_address may be a tuple or None depending on client
    try:
        client_addr = websocket.remote_address if hasattr(websocket, "remote_address") else None
    except Exception:
        client_addr = None

    print("[PoseWS Server] client connected:", client_addr)
    pe = PoseEngine()
    try:
        async for msg in websocket:
            try:
                req = json.loads(msg)
            except Exception as e:
                print("[PoseWS] invalid json:", e)
                continue

            if req.get("type") != "frame":
                continue

            frame_id = req.get("frame_id")
            b64 = req.get("image_jpeg_b64")
            if not b64:
                continue

            # decode JPEG to BGR frame
            try:
                jpg = base64.b64decode(b64)
                arr = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("imdecode returned None")
            except Exception as e:
                print("[PoseWS] image decode error:", e)
                continue

            # run pose detection
            poses, events = pe.process_frame(frame)

            resp = {
                "type": "pose_result",
                "frame_id": frame_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "poses": poses,
                "pose_events": events,
                "fps": round(pe.last_fps, 2)
            }

            try:
                await websocket.send(json.dumps(resp))
            except Exception as e:
                print("[PoseWS] send error:", e)
    except websockets.exceptions.ConnectionClosed:
        print("[PoseWS Server] connection closed")
    except Exception as e:
        print("[PoseWS Server] handler error:", e)


import threading
import asyncio
import websockets

def start_pose_ws_server_in_bg(host: str = "0.0.0.0", port: int = 8003):
    """
    Start the pose WebSocket server in a background daemon thread.
    This creates and sets an asyncio event loop for the thread BEFORE awaiting websockets.serve,
    preventing the "no running event loop" RuntimeError.
    """
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _serve_forever():
            try:
                server = await websockets.serve(_pose_ws_handler, host, port, max_size=2**24)
                print(f"[PoseWS Server] listening on ws://{host}:{port}")
            except Exception as e:
                # propagate to outer try/except for consistent logging
                raise

            # keep the coroutine alive until loop is stopped
            await asyncio.Future()

        try:
            loop.run_until_complete(_serve_forever())
        except Exception as e:
            print("[PoseWS Server] failed to start or crashed:", e)
        finally:
            # Attempt graceful shutdown if possible
            try:
                pending = asyncio.all_tasks(loop=loop)
                for t in pending:
                    t.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            try:
                loop.stop()
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass
            print("[PoseWS Server] shutdown complete")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t

# -----------------------------
# API-only section (uvicorn imports with POSE_API_ONLY=1)
# -----------------------------
API_ONLY = os.environ.get("POSE_API_ONLY", "") == "1"
if API_ONLY:
    from fastapi import FastAPI, Response, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse

    app = FastAPI(title="Hawkeye Pose API (read-only)")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def _read_state():
        p = os.environ.get("POSE_STATE_FILE", STATE_PATH_DEFAULT)
        st = read_state_file(p)
        return st or {"status": "no_state", "updated_at": None}

    @app.get("/status")
    def status():
        st = _read_state()
        return {"status": st.get("status", "unknown"), "updated_at": st.get("updated_at")}

    @app.get("/latest")
    def latest():
        st = _read_state()
        return st.get("latest_packet", None)

    @app.get("/poses")
    def poses():
        st = _read_state()
        return st.get("poses", [])

    @app.get("/pose_events")
    def pose_events():
        st = _read_state()
        return st.get("pose_events", [])

    @app.get("/pose_fps")
    def pose_fps():
        st = _read_state()
        return {"pose_fps": st.get("pose_fps", 0.0)}

    @app.get("/replay")
    def replay_list():
        rd = os.environ.get("POSE_REPLAY_DIR", REPLAY_DIR_DEFAULT)
        files = []
        try:
            for fname in sorted(os.listdir(rd)):
                if fname.endswith(".jpg") and fname.startswith("pose_replay_"):
                    path = os.path.join(rd, fname)
                    try:
                        fid = int(fname.replace("pose_replay_", "").replace(".jpg", ""))
                    except:
                        fid = None
                    files.append({"frame_id": fid, "file": path})
        except Exception:
            pass
        return files

    @app.get("/replay/frame/{frame_id}")
    def replay_frame(frame_id: int):
        rd = os.environ.get("POSE_REPLAY_DIR", REPLAY_DIR_DEFAULT)
        fname = os.path.join(rd, f"pose_replay_{frame_id}.jpg")
        if not os.path.exists(fname):
            raise HTTPException(status_code=404, detail="frame not found")
        return FileResponse(fname, media_type="image/jpeg")

    @app.get("/health")
    def health():
        st = _read_state()
        return {"status": st.get("status", "unknown"), "pose_fps": st.get("pose_fps", 0.0)}

# -----------------------------
# Main run loop (capture mode)
# -----------------------------
def run_pose_loop(
    source=0,
    model_path=POSE_MODEL_PATH,
    device=DEFAULT_DEVICE,
    imgsz=POSE_IMG_SIZE,
    conf=DEFAULT_CONF,
    replay_frames=120,
    state_path=STATE_PATH_DEFAULT,
    replay_dir=REPLAY_DIR_DEFAULT,
    spawn_api=False,
    api_port=API_PORT_DEFAULT,
    overlay=True,
    fusion_url: Optional[str]=None,
    log_json=False
):
    pe = PoseEngine(model_path=model_path, device=device, imgsz=imgsz, overlay=overlay)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    # optional outgoing websocket for fusion
    ws_sender = None
    if fusion_url:
        print("[PoseEngine] Starting websocket sender to", fusion_url)
        ws_sender = WebsocketSender(fusion_url)

    # spawn API if requested
    if spawn_api:
        start_pose_api_server(port=api_port, state_path=state_path, replay_dir=replay_dir)

    replay_buf = deque(maxlen=replay_frames)
    frame_id = 0
    os.makedirs(replay_dir, exist_ok=True)

    initial_state = {
        "status": "initializing",
        "latest_packet": None,
        "poses": [],
        "pose_events": [],
        "pose_fps": 0.0,
        "replay_meta": [],
        "updated_at": datetime.utcnow().isoformat() + "Z"
    }
    write_state_file(initial_state, state_path)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[PoseEngine] End of stream.")
                break
            frame_id += 1

            poses, events = pe.process_frame(frame, conf=conf)

            packet = {
                "source": str(source),
                "frame_id": frame_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "poses": poses,
                "pose_events": events,
                "fps": round(pe.last_fps, 2)
            }

            # replay buffer store
            replay_buf.append({
                "frame_id": frame_id,
                "timestamp": packet["timestamp"],
                "frame_bgr": frame.copy(),
                "packet": packet
            })

            # dump replay thumbnails/meta for API
            replay_meta = dump_replay_to_disk(replay_buf, replay_dir=replay_dir, max_images=replay_frames)

            # forward to external websocket if configured (fusion)
            if ws_sender:
                ws_sender.send(packet)

            # write atomic pose-only state file
            state_obj = {
                "status": "running",
                "latest_packet": packet,
                "poses": poses,
                "pose_events": events,
                "pose_fps": packet["fps"],
                "replay_meta": replay_meta,
                "updated_at": datetime.utcnow().isoformat() + "Z"
            }
            write_state_file(state_obj, state_path)

            if log_json:
                try:
                    os.makedirs("logs", exist_ok=True)
                    fname = os.path.join("logs", f"pose_packet_{frame_id}.json")
                    with open(fname, "w", encoding="utf-8") as f:
                        json.dump(packet, f, indent=2)
                except Exception as e:
                    print("[PoseEngine] log write failed:", e)

            # overlay
            if overlay:
                overlay_frame = pe.annotate(frame, poses, events)
                cv2.imshow("Hawkeye Pose Engine", overlay_frame)
            else:
                cv2.imshow("Hawkeye Pose Engine (raw)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[PoseEngine] Quit requested.")
                break
            elif key == ord("r"):
                print("[PoseEngine] replaying buffer length:", len(replay_buf))
                for item in list(replay_buf):
                    cv2.imshow("Hawkeye Pose - Replay", item["frame_bgr"])
                    if cv2.waitKey(50) & 0xFF == ord("q"):
                        break
                cv2.destroyWindow("Hawkeye Pose - Replay")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        final_state = {
            "status": "stopped",
            "latest_packet": None,
            "poses": [],
            "pose_events": [],
            "pose_fps": 0.0,
            "replay_meta": [],
            "updated_at": datetime.utcnow().isoformat() + "Z"
        }
        write_state_file(final_state, state_path)
        if ws_sender:
            ws_sender.stop()
        print("[PoseEngine] Shutdown complete.")

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(prog="pose_engine")
    p.add_argument("--source", type=str, default="0", help="0 for webcam or path to video file.")
    p.add_argument("--model", type=str, default=POSE_MODEL_PATH, help="YOLOv8 pose model path.")
    p.add_argument("--device", type=int, default=DEFAULT_DEVICE, help="CUDA device id.")
    p.add_argument("--imgsz", type=int, default=POSE_IMG_SIZE, help="Inference image size.")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Min confidence for pose keypoints.")
    p.add_argument("--replay", type=int, default=120, help="Replay buffer size (frames).")
    p.add_argument("--state-file", type=str, default=STATE_PATH_DEFAULT, help="Path to write pose state JSON.")
    p.add_argument("--replay-dir", type=str, default=REPLAY_DIR_DEFAULT, help="Directory to store replay JPEGs.")
    p.add_argument("--spawn-api", action="store_true", help="Spawn external API subprocess for pose state.")
    p.add_argument("--api-port", type=int, default=API_PORT_DEFAULT, help="Port for spawned API server.")
    p.add_argument("--no-overlay", action="store_true", help="Disable drawing overlay window.")
    p.add_argument("--fusion", type=str, default=None, help="Optional websocket URL to send pose packets.")
    p.add_argument("--log-json", action="store_true", help="Save packet JSON locally.")
    # server mode
    p.add_argument("--server-ws", action="store_true", help="Run WebSocket server to accept frames from YOLO (no camera).")
    p.add_argument("--ws-host", type=str, default=POSE_WS_HOST_DEFAULT, help="Host for pose WS server.")
    p.add_argument("--ws-port", type=int, default=POSE_WS_PORT_DEFAULT, help="Port for pose WS server.")
    return p.parse_args()

def main():
    args = parse_args()
    source = 0 if args.source == "0" else args.source
    state_path = args.state_file
    replay_dir = args.replay_dir

    if args.server_ws:
        # run WS server mode; do not open camera
        start_pose_ws_server_in_bg(host=args.ws_host, port=args.ws_port)
        print("[PoseEngine] Running in WebSocket server mode (no camera). Press Ctrl+C to quit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("[PoseEngine] Keyboard interrupt - shutting down.")
            return


    # otherwise run capture mode
    run_pose_loop(
        source=source,
        model_path=args.model,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        replay_frames=args.replay,
        state_path=state_path,
        replay_dir=replay_dir,
        spawn_api=args.spawn_api,
        api_port=args.api_port,
        overlay=not args.no_overlay,
        fusion_url=args.fusion,
        log_json=args.log_json
    )

if __name__ == "__main__":
    main()
