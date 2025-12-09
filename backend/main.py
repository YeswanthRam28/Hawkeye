# backend/main.py
import base64
import io
import time
import os
import asyncio
import subprocess
import tempfile
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
import imageio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext

# Optional audio libs (may fail on some systems)
AUDIO_AVAILABLE = True
try:
    import sounddevice as sd
    import soundfile as sf
except Exception as e:
    AUDIO_AVAILABLE = False
    # print a short note; not fatal
    print("audio libs not available:", e)

# imageio-ffmpeg helper
try:
    from imageio_ffmpeg import get_ffmpeg_exe
except Exception:
    get_ffmpeg_exe = None

# ------------------------
# CONFIG
# ------------------------
FPS = 15
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
REPLAY_BUFFER_FRAMES = 120

JWT_SECRET = "replace_this_with_a_strong_secret_for_demo"
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Audio config
AUDIO_SR = 16000            # sampling rate for capture & export
AUDIO_CHANNELS = 1
AUDIO_CHUNK_FRAMES = int(AUDIO_SR / FPS)  # samples per chunk (~1/FPS seconds)

# ------------------------
# App + middleware
# ------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
app = FastAPI(title="Hawkeye Backend (A/V enabled)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Replay Buffer & Connections
# ------------------------
replay_buffer = deque(maxlen=REPLAY_BUFFER_FRAMES)  # stores dicts with image_bytes, audio_array, overlays...
connected_websockets: List[WebSocket] = []

# ------------------------
# Auth helpers
# ------------------------
fake_user_db = {
    "admin": {
        "username": "admin",
        "full_name": "Hawkeye Admin",
        "hashed_password": pwd_context.hash("password")
    }
}

class Token(BaseModel):
    access_token: str
    token_type: str

def verify_password(plain_password, hashed_password):
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        return False

def authenticate_user(username: str, password: str):
    user = fake_user_db.get(username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = fake_user_db.get(username)
    if user is None:
        raise credentials_exception
    return user

# ------------------------
# Token endpoint
# ------------------------
class LoginIn(BaseModel):
    username: str
    password: str

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: LoginIn):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user["username"]}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

# ------------------------
# Health
# ------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "clients": len(connected_websockets)}

# ------------------------
# Mock overlay generator (keeps demo simple)
# ------------------------
import random
def generate_mock_overlays(frame_id: int, w: int, h: int):
    boxes = []
    for i in range(random.randint(0, 3)):
        x = random.uniform(0.1, 0.7); y = random.uniform(0.1, 0.6)
        bw = random.uniform(0.05, 0.25); bh = random.uniform(0.08, 0.3)
        boxes.append({
            "id": f"obj_{i}",
            "x": int(x * w),
            "y": int(y * h),
            "w": int(bw * w),
            "h": int(bh * h),
            "label": random.choice(["person","bag","bicycle"]),
            "conf": round(random.uniform(0.5, 0.99), 2)
        })
    skeleton = []
    for j in range(random.randint(0,2)):
        pts = []
        for k in range(5):
            pts.append({"x": int(random.uniform(0.1,0.9)*w), "y": int(random.uniform(0.1,0.9)*h)})
        skeleton.append({"id": f"person_{j}", "keypoints": pts})
    heatmap = [[ round(random.uniform(0,1),2) for _ in range(8) ] for __ in range(4)]
    return {"boxes": boxes, "skeletons": skeleton, "heatmap": heatmap}

# ------------------------
# Internal: broadcast coroutine
# ------------------------
async def broadcast_frame(frame_packet: dict):
    dead = []
    for ws in list(connected_websockets):
        try:
            await ws.send_json(frame_packet)
        except Exception:
            dead.append(ws)
    for d in dead:
        try:
            connected_websockets.remove(d)
        except ValueError:
            pass

# ------------------------
# Camera capture & main loop (captures frames, attaches audio chunk if available)
# ------------------------
async def camera_capture_loop():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    frame_id = 0
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(1.0/FPS)
            continue
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # encode JPEG
        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        jpg_bytes = buf.tobytes()
        image_b64 = base64.b64encode(jpg_bytes).decode('utf-8')

        ts = time.time()

        overlays = generate_mock_overlays(frame_id, FRAME_WIDTH, FRAME_HEIGHT)
        risk_score = round(random.uniform(0,1), 3)

        # --- try to get a synced audio chunk if available ---
        audio_array = None
        if AUDIO_AVAILABLE and len(audio_buffer) > 0:
            try:
                audio_array = audio_buffer.popleft()
            except Exception:
                audio_array = None

        # store in replay buffer
        replay_buffer.append({
            "frame_id": frame_id,
            "ts": ts,
            "image_bytes": jpg_bytes,
            "overlays": overlays,
            "audio_array": audio_array
        })

        # prepare small audio_b64 for live broadcast if present
        audio_b64 = None
        if audio_array is not None:
            try:
                bio = io.BytesIO()
                sf.write(bio, audio_array, AUDIO_SR, format='WAV', subtype='PCM_16')
                audio_b64 = base64.b64encode(bio.getvalue()).decode('utf-8')
            except Exception:
                audio_b64 = None

        packet = {
            "type": "frame",
            "frame_id": frame_id,
            "ts": ts,
            "image_b64": image_b64,
            "overlays": overlays,
            "risk_score": risk_score,
            "audio_b64": audio_b64
        }

        # broadcast (non-blocking)
        asyncio.create_task(broadcast_frame(packet))

        frame_id += 1

        elapsed = time.time() - start
        await asyncio.sleep(max(0, (1.0/FPS) - elapsed))

    cap.release()

# ------------------------
# Audio capture loop (records small chunks ~1/FPS seconds)
# ------------------------
audio_buffer = deque(maxlen=REPLAY_BUFFER_FRAMES * 2)
async def audio_capture_loop():
    if not AUDIO_AVAILABLE:
        print("audio_capture_loop: audio libs unavailable, skipping.")
        return
    try:
        while True:
            # record in background thread to avoid blocking event loop
            chunk = await asyncio.to_thread(sd.rec, AUDIO_CHUNK_FRAMES, samplerate=AUDIO_SR, channels=AUDIO_CHANNELS, dtype='int16')
            await asyncio.to_thread(sd.wait)
            # sounddevice returns shape (N, channels); convert to 2D int16 numpy
            try:
                audio_buffer.append(chunk.copy())
            except Exception:
                pass
            # minimal yield
            await asyncio.sleep(0)
    except Exception as e:
        print("audio_capture_loop error:", e)

# ------------------------
# WebSocket endpoint (token read from query params)
# ------------------------
@app.websocket("/stream/video")
async def websocket_video_stream(websocket: WebSocket):
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008)
        return
    try:
        await get_current_user(token)
    except Exception:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    connected_websockets.append(websocket)

    try:
        while True:
            try:
                msg = await websocket.receive_text()
            except Exception:
                await asyncio.sleep(0.01)
                continue
            try:
                import json
                obj = json.loads(msg)
                if obj.get("type") == "replay":
                    speed = float(obj.get("speed", 1.0))
                    asyncio.create_task(send_replay_to_ws(websocket, speed))
                elif obj.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "ts": time.time()})
                else:
                    await websocket.send_json({"type": "info", "msg": "unknown command"})
            except Exception:
                continue
    except WebSocketDisconnect:
        pass
    finally:
        try:
            connected_websockets.remove(websocket)
        except ValueError:
            pass

# ------------------------
# Replay sender (to one websocket)
# ------------------------
async def send_replay_to_ws(ws: WebSocket, speed: float = 1.0):
    buffer_snapshot = list(replay_buffer)
    if not buffer_snapshot:
        try:
            await ws.send_json({"type":"replay_end", "reason":"empty_buffer"})
        except Exception:
            pass
        return

    for item in buffer_snapshot:
        # construct audio_b64 if available
        audio_b64 = None
        a = item.get("audio_array")
        if a is not None:
            try:
                bio = io.BytesIO()
                sf.write(bio, a, AUDIO_SR, format='WAV', subtype='PCM_16')
                audio_b64 = base64.b64encode(bio.getvalue()).decode('utf-8')
            except Exception:
                audio_b64 = None

        packet = {
            "type": "replay_frame",
            "frame_id": item["frame_id"],
            "ts": item["ts"],
            "image_b64": base64.b64encode(item["image_bytes"]).decode('utf-8'),
            "overlays": item.get("overlays", {}),
            "audio_b64": audio_b64
        }
        try:
            await ws.send_json(packet)
        except Exception:
            break
        await asyncio.sleep(max(0.001, (1.0 / FPS) / max(0.01, speed)))

    try:
        await ws.send_json({"type":"replay_end", "reason":"completed"})
    except Exception:
        pass

# ------------------------
# Export endpoint (write video, write stacked audio, mux via bundled ffmpeg)
# ------------------------
@app.post("/export")
async def export_clip(token: str = Body(..., embed=True)):
    await get_current_user(token)
    if len(replay_buffer) == 0:
        raise HTTPException(status_code=400, detail="No frames to export")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    raw_video = f"raw_{ts}.mp4"
    raw_audio = f"raw_{ts}.wav"
    out_file = f"evidence_{ts}.mp4"

    # 1) write raw video (frames only)
    writer = imageio.get_writer(raw_video, fps=FPS, codec="libx264")
    try:
        for item in replay_buffer:
            img = imageio.imread(item["image_bytes"], format='JPEG')
            img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
            writer.append_data(img)
    finally:
        writer.close()

    # 2) build audio track by stacking audio_array or inserting silence
    audio_parts = []
    for item in replay_buffer:
        a = item.get("audio_array")
        if a is not None:
            # ensure shape (N, channels) and dtype int16
            arr = np.asarray(a, dtype='int16')
            if arr.ndim == 1:
                arr = arr.reshape(-1, AUDIO_CHANNELS)
            audio_parts.append(arr)
        else:
            # silence chunk
            audio_parts.append(np.zeros((AUDIO_CHUNK_FRAMES, AUDIO_CHANNELS), dtype='int16'))

    if len(audio_parts) > 0:
        combined = np.vstack(audio_parts)
        try:
            sf.write(raw_audio, combined, AUDIO_SR, format='WAV', subtype='PCM_16')
        except Exception as e:
            print("audio write failed:", e)
            # write a tiny silent file to avoid ffmpeg errors
            sf.write(raw_audio, np.zeros((1, AUDIO_CHANNELS), dtype='int16'), AUDIO_SR, format='WAV', subtype='PCM_16')
    else:
        sf.write(raw_audio, np.zeros((1, AUDIO_CHANNELS), dtype='int16'), AUDIO_SR, format='WAV', subtype='PCM_16')

    # 3) mux using bundled ffmpeg if available
    ffmpeg_exe = None
    if get_ffmpeg_exe is not None:
        try:
            ffmpeg_exe = get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = None

    if ffmpeg_exe is None:
        # fallback: return video only
        return {"status": "saved", "file": raw_video}

    cmd = [
        ffmpeg_exe, "-y",
        "-i", raw_video,
        "-i", raw_audio,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        out_file
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("ffmpeg error:", e.stderr.decode('utf-8', errors='ignore'))
        out_file = raw_video

    return {"status": "saved", "file": out_file}

# ------------------------
# Replay metadata endpoint
# ------------------------
@app.get("/replay/meta")
async def replay_meta(token: str):
    await get_current_user(token)
    meta = [{"frame_id": item["frame_id"], "ts": item["ts"]} for item in replay_buffer]
    return {"count": len(meta), "meta": meta}

# ------------------------
# Serve exported files
# ------------------------
@app.get("/exports/{filename}")
async def get_export(filename: str, token: str):
    await get_current_user(token)
    path = os.path.abspath(filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)

# ------------------------
# Startup: spawn loops
# ------------------------
@app.on_event("startup")
async def startup_event():
    # start camera loop
    asyncio.create_task(camera_capture_loop())
    # start audio loop (if available)
    if AUDIO_AVAILABLE:
        try:
            asyncio.create_task(audio_capture_loop())
        except Exception as e:
            print("audio loop spawn error:", e)
