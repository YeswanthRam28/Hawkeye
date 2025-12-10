# server/routers/trigger_ws.py
import asyncio
import json
from typing import Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from replay.engine import ReplayEngine

router = APIRouter()
engine = ReplayEngine(pre_seconds=5, post_seconds=5)

@router.websocket("/ws/trigger")
async def trigger_replay_ws(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"status": "connected"})

    while True:
        data = await ws.receive_text()

        if data == "trigger":
            result = replay_engine.manual_trigger()
            await ws.send_json({"event": "triggered", "result": result})
        else:
            await ws.send_json({"error": "unknown command"})
            
# Async queue used to transport incident events into the asyncio world
_event_queue: asyncio.Queue = asyncio.Queue()
# Connected websockets
_clients: Set[WebSocket] = set()
# Background broadcaster task handle
_broadcaster_task: asyncio.Task | None = None


# Callback invoked by ReplayEngine when an incident is saved.
# It runs in the thread where engine finishes; push the info into the asyncio queue.
def _engine_callback(incident_info):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If no event loop in this thread, schedule on main loop by using call_soon_threadsafe
        loop = None

    if loop and loop.is_running():
        loop.call_soon_threadsafe(_event_queue.put_nowait, incident_info)
    else:
        # As a fallback, create a new loop to put the item (rare)
        try:
            asyncio.get_event_loop().call_soon_threadsafe(_event_queue.put_nowait, incident_info)
        except Exception:
            # last resort: ignore
            print("[trigger_ws] Warning: failed to schedule incident event")


# Register the callback with the engine
engine.register_callback(_engine_callback)


async def _broadcaster():
    """
    Background task: wait for incident events on the queue and broadcast
    to all connected websocket clients.
    """
    print("[trigger_ws] broadcaster started")
    while True:
        incident_info = await _event_queue.get()  # wait for an incident
        msg = {
            "event": "REPLAY_TRIGGERED",
            "incident": incident_info
        }
        text = json.dumps(msg)
        to_remove = []
        # broadcast to all clients
        for ws in list(_clients):
            try:
                await ws.send_text(text)
            except Exception:
                # mark broken websockets for removal
                to_remove.append(ws)
        for ws in to_remove:
            _clients.discard(ws)


@router.websocket("/ws/trigger")
async def websocket_trigger(ws: WebSocket):
    """
    WebSocket endpoint clients connect to for instant replay notifications.
    Server will broadcast {"event":"REPLAY_TRIGGERED", "incident": {...}} whenever
    the ReplayEngine saves an incident.
    """
    await ws.accept()
    _clients.add(ws)

    # ensure broadcaster started
    global _broadcaster_task
    if _broadcaster_task is None or _broadcaster_task.done():
        _broadcaster_task = asyncio.create_task(_broadcaster())

    try:
        # Keep the connection alive; optionally accept pings/messages from client
        while True:
            # We simply await any incoming text (so client can send heartbeats)
            try:
                await ws.receive_text()
            except WebSocketDisconnect:
                break
            except Exception:
                # ignore other receive errors and continue
                await asyncio.sleep(0.1)

    finally:
        _clients.discard(ws)
        try:
            await ws.close()
        except Exception:
            pass
