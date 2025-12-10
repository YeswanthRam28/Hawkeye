from fastapi import FastAPI
from explain.server.routers.explain import router as explain_router

app = FastAPI()

app.include_router(explain_router, prefix="/frame", tags=["Explain"])
