# fusion_engine/fusion_core.py

from fusion_engine.feature_schema import SensorPacket
from fusion_engine.rule_engine import RuleEngine
from fusion_engine.ml_engine import MLEngine
from typing import Dict


class FusionEngine:
    """
    Advanced Fusion Engine that orchestrates Rule-based logic and ML anomaly detection.
    Fixes the 'fake data' issue by performing actual multimodal correlation.
    """

    def __init__(self):
        self.rules = RuleEngine()
        self.ml = MLEngine()

    def process(self, packet: SensorPacket) -> Dict:
        """
        Combine all subsystem outputs into one intelligent fused threat level.
        Incorporates deterministic rules, ML isolation forest, and contradiction checks.
        """

        # 1. Base Score (Averaging the raw scores)
        threat_components = []
        if packet.vision:
            threat_components.append(packet.vision.threat_score)
        if packet.audio:
            threat_components.append(packet.audio.anomaly_score)
        if packet.motion:
            # Derived motion risk from intensity
            m_risk = min((abs(packet.motion.speed) + abs(packet.motion.jerk)) / 10.0, 1.0)
            threat_components.append(m_risk)

        base_fused = sum(threat_components) / len(threat_components) if threat_components else 0.0

        # 2. ML Anomaly Score
        ml_anomaly = self.ml.score(packet)

        # 3. Rule Evaluation
        rule_output = self.rules.evaluate(base_fused, packet)
        
        # 4. Contradiction Logic (Vision vs Audio)
        # Higher risk if vision is 'safe' but audio detects 'screams' or 'panic'
        contradiction_bonus = 0.0
        if packet.vision and packet.audio:
            if packet.vision.threat_score < 0.2 and packet.audio.anomaly_score > 0.7:
                contradiction_bonus = 0.2
                rule_output["triggered_rules"].append("CONTRADICTION: Low Vision Risk vs High Audio Anomaly")

        # 5. Final Unified Sentiment Risk
        # Blend Rule output with ML anomaly
        final_threat = (rule_output["final_risk_score"] * 0.7) + (ml_anomaly * 0.3) + contradiction_bonus
        final_threat = min(max(final_threat, 0.0), 1.0)

        # Update threat level based on the new final_threat
        threat_level = "Low"
        if final_threat >= 0.85: threat_level = "Critical"
        elif final_threat >= 0.6: threat_level = "High"
        elif final_threat >= 0.3: threat_level = "Medium"

        return {
            "fused_threat": round(final_threat, 3),
            "final_score": round(final_threat, 3), # Compatibility
            "ml_anomaly": round(ml_anomaly, 3),
            "threat_level": threat_level,
            "triggered_rules": rule_output["triggered_rules"],
            "timestamp": packet.timestamp
        }

