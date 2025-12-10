# fusion_engine/sensor_stream_simulator.py

import time
import random

from fusion_engine.feature_schema import (
    VisionData,
    AudioData,
    MotionData,
    SensorPacket
)

from fusion_engine.fusion_core import FusionEngine
from fusion_engine.rule_engine import RuleEngine
from fusion_engine.ml_engine import MLEngine


fusion = FusionEngine()
rules = RuleEngine()
ml = MLEngine()


def generate_fake_vision():
    """Simulates what Vision AI team will send."""
    return VisionData(
        object_count=random.randint(0, 5),
        threat_score=round(random.uniform(0, 1), 2),
        bounding_boxes=[[10, 20, 40, 60]]
    )


def generate_fake_audio():
    """Simulates audio classifier output."""
    return AudioData(
        volume_db=random.uniform(20, 120),
        anomaly_score=round(random.uniform(0, 1), 2),
        keywords_detected=["help"] if random.random() > 0.85 else []
    )


def generate_fake_motion():
    """Simulates motion dynamics."""
    return MotionData(
        speed=round(random.uniform(0, 6), 2),
        acceleration=round(random.uniform(0, 3), 2),
        jerk=round(random.uniform(0, 2), 2),
    )


def run_stream():
    print("\n=== Hawkeye LIVE Sensor Fusion Stream ===\n")

    while True:
        packet = SensorPacket(
            vision=generate_fake_vision(),
            audio=generate_fake_audio(),
            motion=generate_fake_motion(),
            timestamp=time.time()
        )

        fused = fusion.process(packet)
        rule_output = rules.apply_rules(packet, fused)
        ml_score = ml.score(packet)

        print("\n-----------------------------------------")
        print("VISION:", packet.vision)
        print("AUDIO:", packet.audio)
        print("MOTION:", packet.motion)

        print("\nFUSED THREAT:", fused["fused_threat"])
        print("ML ANOMALY:", ml_score)
        print("RISK ENGINE:", rule_output["final_risk_score"], "-", rule_output["threat_level"])
        print("-----------------------------------------")

        time.sleep(1)


if __name__ == "__main__":
    run_stream()
