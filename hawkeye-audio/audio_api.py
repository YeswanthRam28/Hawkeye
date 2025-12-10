# audio_api.py
import os
import tempfile
import threading
import time
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Try import your existing detection module (yamnet_detect.py)
try:
    import yamnet_detect as yd
    has_detector_module = True
except Exception as e:
    has_detector_module = False
    import traceback
    traceback.print_exc()

app = FastAPI(title="Hawkeye Audio Detector API (FastAPI)")

# Allow CORS from anywhere for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monitor control globals
_monitor_thread: Optional[threading.Thread] = None
_monitor_lock = threading.Lock()
_monitor_stop_event = threading.Event()


# Helper: run file detection and return a simple JSON response
def _analyze_waveform_return_json(waveform, filename: str = "uploaded"):
    try:
        yd.run_file_detection(waveform, filename)
        return {"status": "ok", "message": f"Processed {filename}. Check evidence/ and terminal logs."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    """
    Upload a file (audio/video). The file is saved to a temp file and passed to yamnet_detect.load_audio_file().
    The function run_file_detection() is then invoked (it will log and save evidence as implemented).
    """
    if not has_detector_module:
        raise HTTPException(status_code=500, detail="Detector module not available on server.")

    # Save uploaded file to a temp path
    suffix = os.path.splitext(file.filename)[1] or ""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        tmp.close()

        waveform = yd.load_audio_file(tmp.name)
        if waveform is None:
            raise HTTPException(status_code=400, detail="Could not read audio from uploaded file.")

        result = _analyze_waveform_return_json(waveform, filename=file.filename)
        return JSONResponse(content=result)

    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


# Background monitor runner
def _monitor_runner(stop_event: threading.Event):
    """
    Calls your module's start_continuous_monitoring(stop_event).
    If your module does not accept a stop_event, a fallback will call it without args (blocking).
    """
    print("Monitor thread: starting.")
    try:
        # Preferred signature: start_continuous_monitoring(stop_event)
        try:
            yd.start_continuous_monitoring(stop_event)
        except TypeError:
            # Fallback if your function doesn't accept an event
            print("Monitor thread: calling start_continuous_monitoring() (no stop_event supported).")
            yd.start_continuous_monitoring()
    except Exception as e:
        print("Monitor thread exception:", e)
    finally:
        print("Monitor thread: exiting.")


@app.post("/start-monitor")
async def start_monitor():
    """
    Starts continuous monitoring in a background thread. The server process must have microphone access.
    """
    global _monitor_thread, _monitor_stop_event
    if not has_detector_module:
        raise HTTPException(status_code=500, detail="Detector module not available on server.")

    with _monitor_lock:
        if _monitor_thread is not None and _monitor_thread.is_alive():
            return JSONResponse(content={"status": "already_running"})

        # Reset/clear stop event and start monitor thread
        _monitor_stop_event.clear()
        _monitor_thread = threading.Thread(target=_monitor_runner, args=(_monitor_stop_event,), daemon=True)
        _monitor_thread.start()
        # small pause to let thread initialize
        time.sleep(0.3)
        return JSONResponse(content={"status": "started"})


@app.post("/stop-monitor")
async def stop_monitor():
    """
    Signals the monitor thread to stop and tries to join it.
    Your start_continuous_monitoring implementation in yamnet_detect.py should check the event.
    """
    global _monitor_thread, _monitor_stop_event
    with _monitor_lock:
        if _monitor_thread is None or not _monitor_thread.is_alive():
            return JSONResponse(content={"status": "not_running"})

        # Signal stop
        _monitor_stop_event.set()
        # Wait a short while for thread to finish
        _monitor_thread.join(timeout=5.0)
        alive = _monitor_thread.is_alive()
        if alive:
            return JSONResponse(content={"status": "stop_failed", "alive": True}, status_code=500)
        else:
            _monitor_thread = None
            return JSONResponse(content={"status": "stopped"})


@app.get("/status")
async def status():
    running = _monitor_thread is not None and _monitor_thread.is_alive()
    return JSONResponse(content={"monitor_running": running})
