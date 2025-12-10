# server/server.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import all routers FIRST
from server.routers.explain import router as explain_router
from server.routers.live import router as live_router
from server.routers.incidents import router as incidents_router
from server.routers.narrative import router as narrative_router
from server.routers.trigger import router as trigger_router
from server.routers.trigger_ws import router as trigger_ws_router
from server.routers.live import router as live_ws_router


# Create FastAPI app correctly
app = FastAPI(
    title="Hawkeye Explainability API",
    version="1.0"
)

# CORS (allow frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # frontend can access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(live_router, prefix="/live", tags=["live"])
app.include_router(live_ws_router, tags=["websocket"])
app.include_router(explain_router, prefix="/frame", tags=["frame"])
app.include_router(incidents_router, prefix="/incident", tags=["incident"])
app.include_router(narrative_router, prefix="/narrative", tags=["narrative"])
app.include_router(trigger_router, prefix="/trigger", tags=["trigger"])
app.include_router(trigger_ws_router, tags=["ws"])   # WebSocket: /ws/trigger

@app.get("/")
def root():
    return {"status": "Hawkeye API running"}
