# explainability/server/server.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers using ABSOLUTE PATH
from explain.server.routers.explain import router as explain_router
from explain.server.routers.live import router as live_router
from explain.server.routers.incidents import router as incidents_router
from explain.server.routers.narrative import router as narrative_router
from explain.server.routers.trigger import router as trigger_router
from explain.server.routers.trigger_ws import router as trigger_ws_router
from explain.server.routers.live_yolo import router as live_yolo_router

app = FastAPI(title="Hawkeye Explainability API", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(explain_router, prefix="/frame", tags=["frame"])
app.include_router(live_router, prefix="/live", tags=["live"])
app.include_router(incidents_router, prefix="/incident", tags=["incident"])
app.include_router(narrative_router, prefix="/narrative", tags=["narrative"])
app.include_router(trigger_router, prefix="/trigger", tags=["trigger"])
app.include_router(trigger_ws_router, prefix="/ws", tags=["websocket"])
app.include_router(live_yolo_router)

@app.get("/")
def root():
    return {"status": "Hawkeye API running"}



