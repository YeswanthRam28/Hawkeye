

import os
import tempfile
import threading
import time
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------------------------------------------
# IMPORT YOUR DETECTION MODULE (yamnet_detect.py)
# -------------------------------------------------------------------
try:
    import yamnet_detect as yd
    has_detector_module = True
except Exception as e:
    has_detector_module = False
    import traceback
    print("ERROR: Failed to import yamnet_detect.py")
    traceback.print_exc()


# -------------------------------------------------------------------
# FASTAPI INITIALIZATION
# -------------------------------------------------------------------
app = FastAPI(title="Hawkeye YAMNet Audio Detector API (FastAPI)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monitor thread globals
_monitor_thread: Optional[threading.Thread] = None
_monitor_stop_event = threading.Event()
_monitor_lock = threading.Lock()


# -------------------------------------------------------------------
# FASTAPI STARTUP → LOAD YAMNET MODEL HERE
# -------------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    """
    Ensures YAMNet loads before any API endpoint is called.
    Prevents 'NoneType object is not callable' errors.
    """
    if has_detector_module:
        print("\n=== FastAPI Startup: Loading YAMNet model ===")
        yd.load_model()
        print("=== YAMNet Model Loaded Successfully ===\n")
    else:
        print("ERROR: yamnet_detect module missing.")


# -------------------------------------------------------------------
# INTERNAL HELPER FUNCTION
# -------------------------------------------------------------------
def _analyze_waveform_return_json(waveform, filename="uploaded"):
    try:
        yd.run_file_detection(waveform, filename)
        return {"status": "ok", "message": f"Processed {filename}. Check evidence/ folder."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# -------------------------------------------------------------------
# ENDPOINT: ANALYZE UPLOADED AUDIO FILE
# -------------------------------------------------------------------
@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    """Accepts any audio file (wav, mp3, etc.) and runs YAMNet detection."""
    
    if not has_detector_module:
        raise HTTPException(status_code=500, detail="yamnet_detect module not found.")

    # Save file temporarily
    suffix = os.path.splitext(file.filename)[1] or ""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

    try:
        data = await file.read()
        tmp.write(data)
        tmp.flush()
        tmp.close()

        # IMPORTANT: Now call load_audio_file FROM yamnet_detect.py
        waveform = yd.load_audio_file(tmp.name)

        if waveform is None:
            raise HTTPException(status_code=400, detail="Audio decoding failed.")

        result = _analyze_waveform_return_json(waveform, filename=file.filename)
        return JSONResponse(result)

    finally:
        try:
            os.unlink(tmp.name)
        except:
            pass


# -------------------------------------------------------------------
# BACKGROUND THREAD FOR MICROPHONE MONITOR
# -------------------------------------------------------------------
def _monitor_runner(stop_event: threading.Event):
    print("Monitor Thread: STARTED")

    try:
        # If your function supports stop_event
        try:
            yd.start_continuous_monitoring(stop_event)
        except TypeError:
            print("WARNING: start_continuous_monitoring() does NOT accept stop_event.")
            yd.start_continuous_monitoring()
    except Exception as e:
        print("Monitor thread crashed:", e)

    print("Monitor Thread: EXITING")


# -------------------------------------------------------------------
# ENDPOINT: START MICROPHONE MONITOR
# -------------------------------------------------------------------
@app.post("/start-monitor")
async def start_monitor():
    """Starts the continuous microphone detection in background."""

    global _monitor_thread, _monitor_stop_event

    if not has_detector_module:
        raise HTTPException(status_code=500, detail="yamnet_detect module missing.")

    with _monitor_lock:
        if _monitor_thread is not None and _monitor_thread.is_alive():
            return {"status": "already_running"}

        _monitor_stop_event.clear()

        _monitor_thread = threading.Thread(
            target=_monitor_runner,
            args=(_monitor_stop_event,),
            daemon=True,
        )
        _monitor_thread.start()

        time.sleep(0.3)
        return {"status": "started"}


# -------------------------------------------------------------------
# ENDPOINT: STOP MICROPHONE MONITOR
# -------------------------------------------------------------------
@app.post("/stop-monitor")
async def stop_monitor():

    global _monitor_thread, _monitor_stop_event

    with _monitor_lock:
        if _monitor_thread is None or not _monitor_thread.is_alive():
            return {"status": "not_running"}

        _monitor_stop_event.set()
        _monitor_thread.join(timeout=5.0)

        if _monitor_thread.is_alive():
            return JSONResponse(
                {"status": "failed_to_stop", "alive": True},
                status_code=500
            )

        _monitor_thread = None
        return {"status": "stopped"}


# -------------------------------------------------------------------
# ENDPOINT: STATUS CHECK
# -------------------------------------------------------------------
@app.get("/status")
async def status():
    """Returns whether the microphone monitor is running."""
    running = _monitor_thread is not None and _monitor_thread.is_alive()
    return {"monitor_running": running}
