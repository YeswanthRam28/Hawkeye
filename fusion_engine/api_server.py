# fusion_engine/api_server.py
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fusion_engine.feature_schema import SensorPacket
from fusion_engine.fusion_core import FusionEngine
from fusion_engine.rule_engine import RuleEngine
from fusion_engine.ml_engine import MLEngine

app = FastAPI(title="Hawkeye Fusion API", version="1.0")

# Initialize engines
fusion = FusionEngine()
rules = RuleEngine()
ml = MLEngine()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def check_status():
    return {"status": "Fusion Engine Online"}


@app.post("/analyze")
def analyze_sensors(packet: SensorPacket):
    """
    Receives SensorPacket, runs fusion, rules, ML, and returns a structured result.
    Defensive: supports both fusion.process() and fusion.fuse() output shapes.
    """
    try:
        # --- Run fusion (support multiple API shapes) ---
        fused_out = None
        fused_threat = None

        if hasattr(fusion, "process"):
            fused_out = fusion.process(packet)
            # look for known numeric keys
            if isinstance(fused_out, dict):
                fused_threat = fused_out.get("final_score") or fused_out.get("fused_threat") or fused_out.get("fusion_score")

        if fused_threat is None and hasattr(fusion, "fuse"):
            fused_out = fusion.fuse(packet)
            if isinstance(fused_out, dict):
                fused_threat = fused_out.get("final_score") or fused_out.get("fused_threat") or fused_out.get("fusion_score")
            elif isinstance(fused_out, (float, int)):
                fused_threat = float(fused_out)

        if fused_threat is None:
            # last-ditch: check fused_out for common numeric names
            if isinstance(fused_out, dict):
                for key in ("final_score", "fused_threat", "fusion_score"):
                    if key in fused_out:
                        fused_threat = fused_out[key]
                        break

        if fused_threat is None:
            raise RuntimeError(f"Fusion engine did not return numeric fused threat. Got: {repr(fused_out)}")

        # --- Run rules (signature can be evaluate(fused_threat, packet) or evaluate(packet)) ---
        try:
            rules_out = rules.evaluate(fused_threat, packet)
        except TypeError:
            rules_out = rules.evaluate(packet)

        # --- Run ML scoring + future predictions ---
        try:
            ml_score = ml.score(packet)
            ml_pred = ml.predict_future([packet])
        except Exception:
            ml_score = None
            ml_pred = None

        # --- Build final response ---
        resp = {
            "fusion": fused_out,
            "fused_threat": fused_threat,
            "ml": {
                "anomaly_score": ml_score
            },
            "rules": rules_out,
            "timestamp": packet.timestamp,
        }

        return resp

    except Exception as e:
        tb = traceback.format_exc()
        # during development: return the trace so we can debug from Swagger
        raise HTTPException(status_code=500, detail={"error": str(e), "traceback": tb})
