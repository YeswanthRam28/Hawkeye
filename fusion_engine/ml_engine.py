# fusion_engine/ml_engine.py
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
from typing import Optional, Dict
from fusion_engine.feature_schema import SensorPacket

# Best-effort model persistence path
MODEL_PATH = "henv/ml_isolation_forest.joblib"


class MLEngine:
    """
    Lightweight ML helper using IsolationForest for anomaly scoring.
    Trains on synthetic 'normal' data by default so it works offline.
    """

    def __init__(self, load_saved: bool = False):
        self.model: Optional[IsolationForest] = None
        if load_saved:
            try:
                self.model = joblib.load(MODEL_PATH)
            except Exception:
                self.model = None

        if self.model is None:
            self._train_synthetic()

    def _make_feature_vector(self, packet: SensorPacket):
        """
        Build a small feature vector:
        [vision_threat, audio_anomaly, motion_jerk, object_count]
        """
        v = packet.vision
        a = packet.audio
        m = packet.motion

        fv = [
            float(v.threat_score) if v is not None and getattr(v, "threat_score", None) is not None else 0.0,
            float(a.anomaly_score) if a is not None and getattr(a, "anomaly_score", None) is not None else 0.0,
            float(m.jerk) if m is not None and getattr(m, "jerk", None) is not None else 0.0,
            float(v.object_count) if v is not None and getattr(v, "object_count", None) is not None else 0.0,
        ]
        return np.array(fv).reshape(1, -1)

    def _train_synthetic(self, n_samples: int = 400):
        """Train IsolationForest on synthetic 'normal' data so the engine works offline."""
        rng = np.random.RandomState(42)
        vision = rng.uniform(0.0, 0.5, size=(n_samples, 1))      # mostly low vision threat
        audio = rng.uniform(0.0, 0.2, size=(n_samples, 1))       # low audio anomaly
        jerk = np.abs(rng.normal(0.0, 0.2, size=(n_samples, 1))) # small jerks
        obj = rng.poisson(1.0, size=(n_samples, 1))              # small object counts
        X = np.hstack([vision, audio, jerk, obj])

        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.model.fit(X)

        # best-effort persistence (ignore failures in restricted envs)
        try:
            joblib.dump(self.model, MODEL_PATH)
        except Exception:
            pass

    def score(self, packet: SensorPacket) -> float:
        """
        Returns anomaly score in 0..1 where 1 == most anomalous.
        Inverts decision_function and maps to 0..1.
        """
        fv = self._make_feature_vector(packet)
        raw = self.model.decision_function(fv)  # shape (1,)
        val = float(-raw[0])
        # make the sigmoid steeper and shift left so small negative vals produce higher anomaly
        score = 1.0 / (1.0 + np.exp(-2.5 * (val - 0.25)))
        return float(np.clip(score, 0.0, 1.0))


    def predict_future(self, history) -> Dict[str, float]:
        """
        Lightweight stub for future prediction. Expects list of recent packets.
        Returns dict with the same anomaly for 0.5s/1s/3s (hackathon stub).
        """
        if not history:
            return {"0.5s": 0.0, "1s": 0.0, "3s": 0.0}
        cur = self.score(history[-1])
        return {"0.5s": cur, "1s": cur, "3s": cur}
