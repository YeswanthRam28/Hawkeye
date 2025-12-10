

import time
import random
import base64
import numpy as np
import cv2
import io

class MockState:
    def __init__(self):
        self.last_update = 0
        self.vision_data = {}
        self.audio_data = {}
        self.motion_data = {}
        self.risk_score = 0.0
        
        # Initialize default state
        self.update()

    def generate_random_base64_image(self, w=64, h=64):
        # Generate a small noisy image
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', img)
        return base64.b64encode(buf).decode('utf-8')

    def update(self):
        now = time.time()
        # Update at most every 100ms
        if now - self.last_update < 0.1:
            return

        self.last_update = now
        
        # --- Vision Mock ---
        self.vision_data = {
            "timestamp": now,
            "objects": [
                {"label": "person", "bbox": [random.randint(0,100), random.randint(0,100), random.randint(100,200), random.randint(100,200)], "confidence": round(random.uniform(0.8, 0.99), 2)},
                {"label": "backpack", "bbox": [random.randint(0,100), random.randint(0,100), random.randint(50,150), random.randint(50,150)], "confidence": round(random.uniform(0.7, 0.9), 2)}
            ],
            "poses": [
                {
                    "id": 1,
                    "keypoints": [{"x": random.randint(100, 500), "y": random.randint(100, 500)} for _ in range(17)],
                    "pose_confidence": round(random.uniform(0.8, 1.0), 2),
                    "action": random.choice(["walking", "standing", "running"]),
                    "action_confidence": round(random.uniform(0.6, 0.9), 2)
                }
            ],
            "vision_risk_factors": {
                "weapon_detected": random.choice([True, False]),
                "fall_detected": random.choice([True, False])
            },
            # Raw features
            "raw_heatmap": self.generate_random_base64_image(32, 32),
            "pose_map": self.generate_random_base64_image(32, 32),
            "object_scores_matrix": self.generate_random_base64_image(16, 16)
        }

        # --- Audio Mock ---
        self.audio_data = {
            "timestamp": now,
            "events": [
                {"label": random.choice(["speech", "noise", "scream"]), "confidence": round(random.uniform(0.7, 0.99), 2)}
            ],
            "audio_risk_score": round(random.uniform(0, 1), 2),
            # Raw features
            "spectrogram": self.generate_random_base64_image(64, 32),
            "mfcc": [round(random.uniform(-10, 10), 2) for _ in range(13)],
            "peaks": [round(random.uniform(0, 5000), 1) for _ in range(5)]
        }

        # --- Motion Mock ---
        self.motion_data = {
            "timestamp": now,
            "crowd_density": round(random.uniform(0, 1), 2),
            "surge_detected": random.choice([True, False]),
            "surge_direction": random.choice(["north", "south", "east", "west"]),
            "panic_score": round(random.uniform(0, 1), 2),
            # Raw features
            "optical_flow_vectors": self.generate_random_base64_image(32, 32),
            "velocity_map": self.generate_random_base64_image(32, 32),
            "avg_speed": round(random.uniform(0, 5), 1),
            "anomaly_regions": [
                {"region_id": i, "magnitude": round(random.uniform(0, 10), 1)} for i in range(random.randint(0, 3))
            ]
        }
        
        # --- Risk Fusion ---
        # Simple average of normalized scores
        v_score = 1.0 if self.vision_data["vision_risk_factors"]["weapon_detected"] else 0.1
        a_score = self.audio_data["audio_risk_score"]
        m_score = self.motion_data["panic_score"]
        
        self.risk_score = round((v_score + a_score + m_score) / 3.0, 3)

# Singleton instance
state = MockState()

def get_state():
    state.update()
    return state
