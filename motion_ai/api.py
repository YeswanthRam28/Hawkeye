from fastapi import FastAPI
import json
import time

app = FastAPI()

def load_json(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except:
        return {"error": f"{filename} not ready", "timestamp": time.time()}

@app.get("/motion/crowd-analysis")
def get_crowd_analysis():
    return load_json("crowd_analysis.json")

@app.get("/motion/motion-vectors")
def get_motion_vectors():
    return load_json("motion_vectors.json")

@app.get("/motion/events")
def get_motion_events():
    return load_json("motion_events.json")
