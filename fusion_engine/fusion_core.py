# fusion_engine/fusion_core.py

from fusion_engine.feature_schema import SensorPacket
from typing import Dict


class FusionEngine:
    """Simple Fusion Engine that combines vision, audio, and motion signals."""

    def process(self, packet: SensorPacket) -> Dict[str, float]:
        """
        Combine all subsystem outputs into one fused threat level.
        Returns dictionary of fused metrics.
        """

        threat_components = []

        # Vision component
        if packet.vision is not None:
            threat_components.append(packet.vision.threat_score)

        # Audio component
        if packet.audio is not None:
            threat_components.append(packet.audio.anomaly_score)

        # Motion component (derived risk: higher jerk or acceleration = higher threat)
        if packet.motion is not None:
            motion_risk = min(
                (abs(packet.motion.acceleration) + abs(packet.motion.jerk)) / 10.0,
                1.0
            )
            threat_components.append(motion_risk)

        # Fallback if no sensors present
        if not threat_components:
            fused_threat = 0.0
        else:
            fused_threat = sum(threat_components) / len(threat_components)

        return {
            "fused_threat": round(fused_threat, 3),
            "timestamp": packet.timestamp
        }
