# fusion_engine/rule_engine.py

from typing import Dict, Optional
from fusion_engine.feature_schema import SensorPacket
from fusion_engine.fusion_config import FusionConfig


class RuleEngine:
    """
    Applies deterministic rules on top of fused sensor data.
    Outputs a risk score + textual explanation.
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self.config.apply_mode()

    def evaluate(self, fused_score: float, packet: SensorPacket) -> Dict:
        """
        Evaluate the fused score + raw inputs to produce:
        - final_risk_score
        - threat_level (Low/Medium/High/Critical)
        - triggered_rules (list of strings)
        """

        triggered = []

        # ------------------------
        # Vision-based rules
        # ------------------------
        if packet.vision:
            if packet.vision.threat_score >= self.config.vision_threat_threshold:
                triggered.append(
                    f"Vision threat score exceeded threshold "
                    f"({packet.vision.threat_score:.2f} ≥ {self.config.vision_threat_threshold:.2f})"
                )

            if packet.vision.object_count >= 3:
                triggered.append("Multiple subjects detected (>=3)")

        # ------------------------
        # Audio-based rules
        # ------------------------
        if packet.audio:
            if packet.audio.anomaly_score >= self.config.audio_anomaly_threshold:
                triggered.append(
                    f"Audio anomaly exceeded threshold "
                    f"({packet.audio.anomaly_score:.2f} ≥ {self.config.audio_anomaly_threshold:.2f})"
                )

            if packet.audio.keywords_detected:
                if any(
                    k.lower() in ["help", "fire", "gun", "shout"]
                    for k in packet.audio.keywords_detected
                ):
                    triggered.append(
                        f"Critical keywords detected: {packet.audio.keywords_detected}"
                    )

        # ------------------------
        # Motion-based rules
        # ------------------------
        if packet.motion:
            if packet.motion.jerk >= self.config.motion_jerk_threshold:
                triggered.append(
                    f"Violent motion spike detected "
                    f"(jerk {packet.motion.jerk:.1f} ≥ {self.config.motion_jerk_threshold:.1f})"
                )

        # ------------------------
        # FINAL RISK SCORE
        # ------------------------
        rule_bonus = min(len(triggered) * 0.1, 0.3)  # cap boost at +0.3
        final_score = min(fused_score + rule_bonus, 1.0)

        # ------------------------
        # THREAT LEVELS
        # ------------------------
        if final_score < 0.3:
            threat_level = "Low"
        elif final_score < 0.6:
            threat_level = "Medium"
        elif final_score < 0.85:
            threat_level = "High"
        else:
            threat_level = "Critical"

        return {
        "base_fused_score": fused_score,
        "final_risk_score": round(final_score, 3),
        "final_risk": round(final_score, 3),  # <-- compatibility alias
        "threat_level": threat_level,
        "triggered_rules": triggered,
        }


    

    # ----------------------------------------------------------
    # COMPATIBILITY WRAPPER  (Required for simulator + API)
    # ----------------------------------------------------------
    def apply_rules(self, packet: SensorPacket, fused: Dict):
        """
        Wrapper used by FusionEngine, ML Engine, API server, and stream simulator.
        Ensures universal compatibility even if internal method names change.
        """
        return self.evaluate(fused["fused_threat"], packet)
